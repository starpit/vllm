// SPDX-License-Identifier: Apache-2.0
//! `MetalWorkerPool`: growable, capped, semaphore-bounded checkout/checkin.
//!
//! Per `FERRITE_METAL_ARCHITECTURE.md` §2: the pool starts at size 1
//! and grows on demand up to `max_workers`. Each worker holds a private
//! arena, a private [`RuntimeBindings`], and one fully-baked ICB
//! per bucket — never shared across workers. `checkout()` blocks if
//! every worker is in use *and* the pool is at cap; otherwise it
//! grows by one (allocates arena + records ICBs) and hands the new
//! worker out.
//!
//! `max_workers` is supplied by the caller (typically derived as
//! `floor((device_total - weights - misc) / per_worker_arena)` by
//! the model loader / `#[forward]` macro). Keeping the budget
//! computation outside the pool avoids the pool needing to
//! introspect device or weight memory.
//!
//! Synchronization uses `std::sync::{Mutex, Condvar}` — the host
//! side is sync (one forward per checked-out worker), so async
//! primitives buy nothing.

#![cfg(feature = "metal")]

use std::ptr::copy_nonoverlapping;
use std::sync::{Arc, Condvar, Mutex};

use crate::interpreter::metal::__re::{
    Buffer, CommandQueue, Device, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCommandQueue,
};

use ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache;

use super::forward::{ForwardError, ForwardInputs};
use super::lowered::LoweredMetalTape;
use super::lowering::lower_pair;
use super::pipelines::SpecializedPipelines;
use super::runtime::RuntimeBindings;
use super::worker::{ArenaLayout, MetalWorker, WorkerError};
use crate::{CanonicalParams, Instruction};
use ferrite_cuda_core::MetalAllocator;

/// One bucket's compile-time data, ready to be lowered + handed to a
/// [`MetalWorkerPool`].
///
/// Emitted by the `#[forward]` macro (Phase 5.F.5) as one row per
/// solved bucket per canonical model. The macro materializes:
///  - `bucket_m` — the workload point this bucket was specialized for;
///  - `num_arena_slots` — colored slot count from `colored_slot_map()`;
///  - `backbone` / `lm_head` — the bucket's `Instruction<W>` static
///    slices, identical to the cuda-side `BACKBONE_M_<wp>` /
///    `LM_HEAD_M_<wp>` statics.
///
/// [`MetalWorkerPool::for_buckets`] calls [`lower`] on the concatenated
/// `(backbone ++ lm_head)` to produce a [`LoweredMetalTape`] per spec
/// at constructor time. Concatenation matches the cuda interpreter's
/// behavior — `forward()` runs backbone then lm_head as one logical
/// pass for a given bucket — and the metal worker bakes both halves
/// into the bucket's single ICB plan so `forward()` issues one
/// `executeCommandsInBuffer` per segment without an extra mid-bucket
/// boundary.
///
/// The slices are `&'static` because the macro emits them as static
/// items; the spec is `Copy` so callers can drop the bucket plan into
/// an `Arc<[MetalBucketSpec<W>]>` cheaply.
pub struct MetalBucketSpec<W: CanonicalParams + 'static> {
    pub bucket_m: u32,
    pub num_arena_slots: u32,
    /// Index in the colored arena where this bucket's terminal
    /// activation lands (lm_head output for decoder layouts; the
    /// backbone output for encoder layouts). The worker pool's
    /// `forward()` callback reads `worker.arena[terminal_slot]` to
    /// expose logits to the engine.
    pub terminal_slot: u32,
    /// Per-arena-slot byte sizes derived from the FUF tile shapes
    /// at macro-expansion time (post-coloring, with bucket-specific
    /// `num_tokens` baked in). `len() == num_arena_slots`. The
    /// pool takes the elementwise max across every bucket to size
    /// the single per-worker arena.
    pub arena_bytes: &'static [u64],
    pub backbone: &'static [Instruction<W>],
    pub lm_head: &'static [Instruction<W>],
}

impl<W: CanonicalParams + 'static> Clone for MetalBucketSpec<W> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<W: CanonicalParams + 'static> Copy for MetalBucketSpec<W> {}

/// Errors surfaced by [`MetalWorkerPool::for_buckets`] before the pool
/// reaches its first eager-spawn `WorkerError` path.
///
/// Covers (a) the `with_standard_shaders` shader-compile call —
/// distinct from per-worker `WorkerError::PipelineLookup` because the
/// failure is one-shot and pre-pool; (b) per-bucket `lower()` failures
/// — the macro should make these structurally impossible, but
/// surfacing them with `bucket_m` context beats panicking when a
/// future variant slips through; (c) any `WorkerError` from the
/// inner [`MetalWorkerPool::new`] call.
#[derive(Debug)]
pub enum PoolBuildError {
    /// `SpecializedPipelineCache::with_standard_shaders` failed —
    /// usually a missing or malformed MSL source. Message is the
    /// underlying [`MetalStreamError`](ferrite_metal_kernels::stream::MetalStreamError).
    PipelineCacheBuild(String),
    /// One bucket's [`lower`] failed. `bucket_m` identifies the row
    /// for the model author; `error` is the `Display` of the
    /// underlying [`LoweringError`].
    BucketLower { bucket_m: u32, error: String },
    /// The eager-spawn first worker (or any structural pool prereq)
    /// reported a [`WorkerError`].
    Worker(WorkerError),
    /// Caller-provided `bucket_specs` was empty. The pool always
    /// needs at least one bucket; degenerate models that emit none
    /// would fail at `pick_bucket` time anyway, so we fail early.
    NoBuckets,
}

impl std::fmt::Display for PoolBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PipelineCacheBuild(s) => write!(
                f,
                "MetalWorkerPool::for_buckets: pipeline cache build failed: {s}"
            ),
            Self::BucketLower { bucket_m, error } => write!(
                f,
                "MetalWorkerPool::for_buckets: lowering bucket M={bucket_m} failed: {error}"
            ),
            Self::Worker(e) => write!(f, "MetalWorkerPool::for_buckets: {e}"),
            Self::NoBuckets => write!(f, "MetalWorkerPool::for_buckets: bucket_specs is empty"),
        }
    }
}

impl std::error::Error for PoolBuildError {}

impl From<WorkerError> for PoolBuildError {
    fn from(e: WorkerError) -> Self {
        Self::Worker(e)
    }
}

/// Factory closure invoked once per worker creation to produce a
/// fresh [`RuntimeBindings`] sized for the worker's largest bucket.
///
/// The factory is `Arc<dyn Fn>` rather than a generic so the pool
/// can stay non-generic over the closure type — there's exactly one
/// runtime layout per (model, max bucket) and the factory captures
/// it once at pool construction.
// Note: the `+ Send + Sync` markers metal-rs's `Buffer`/`Device` carried
// implicitly aren't present on `objc2-metal`'s
// `Retained<ProtocolObject<dyn MTL*>>` (the protocol traits aren't
// `Send`/`Sync`). We invoke the factory on the same thread that owns
// the pool's device handle, so cross-thread bound is unnecessary —
// dropping it lets the macro-emitted closures capture `Vec<Buffer>`
// without manual newtype wrappers around every objc2 retained handle.
pub type RuntimeFactory = Arc<dyn Fn(&Device) -> RuntimeBindings>;

/// One unit the pool hands out: a worker plus its private
/// [`RuntimeBindings`].
///
/// The engine writes input/positions/etc. into `runtime.<field>.contents()`
/// before firing `worker.run_bucket(...)`. Both fields are owned by the
/// pool; `WorkerGuard` deref's to this struct.
pub struct PooledWorker<W: CanonicalParams> {
    pub worker: MetalWorker<W>,
    pub runtime: RuntimeBindings,
}

/// RAII guard returned by [`MetalWorkerPool::checkout`]. Returns the
/// underlying [`PooledWorker`] to the pool on drop.
///
/// `Deref`/`DerefMut` to `PooledWorker<W>` so the engine can call
/// `guard.worker.run_bucket(...)` and stage `guard.runtime.<field>`
/// writes through the guard.
pub struct WorkerGuard<'pool, W: CanonicalParams> {
    pool: &'pool MetalWorkerPool<W>,
    inner: Option<PooledWorker<W>>,
}

impl<W: CanonicalParams> std::ops::Deref for WorkerGuard<'_, W> {
    type Target = PooledWorker<W>;
    fn deref(&self) -> &PooledWorker<W> {
        self.inner.as_ref().expect("guard inner already taken")
    }
}

impl<W: CanonicalParams> std::ops::DerefMut for WorkerGuard<'_, W> {
    fn deref_mut(&mut self) -> &mut PooledWorker<W> {
        self.inner.as_mut().expect("guard inner already taken")
    }
}

impl<W: CanonicalParams> Drop for WorkerGuard<'_, W> {
    fn drop(&mut self) {
        if let Some(pooled) = self.inner.take() {
            self.pool.checkin(pooled);
        }
    }
}

/// Growable, capped worker pool. Generic over the model's
/// [`CanonicalParams`] — one pool per loaded model variant.
///
/// The pool stays parameterized over `W` for the per-bucket
/// `LoweredMetalTape<W>` (workers bake ICBs from these), but it
/// does *not* hold a back-reference to the loaded `Weights`
/// itself — callers pass `&W` into `forward`/`checkout` so the
/// pool can be stored as a field on the `Weights` struct without
/// an `Arc`-cycle.
pub struct MetalWorkerPool<W: CanonicalParams> {
    device: Arc<Device>,
    /// Allocator that owns the `MTLBuffer` arenas the loaded
    /// `GpuTensor`s point into. The worker uses
    /// [`MetalAllocator::buffer_for`] to map a tensor's raw pointer
    /// back to `(&MTLBuffer, offset)` for encoder bindings.
    ///
    /// The allocator also owns the shared `MetalResidencySet` that
    /// pins every weight / arena / KV-cache buffer as resident across
    /// cmdbufs (so large Llama-3.2-class working sets don't race
    /// against Apple's lazy paging and produce non-deterministic
    /// decode output). The pool reads the set off the allocator and
    /// (a) hands it to spawned workers so per-worker arena buffers
    /// also get pinned, and (b) attaches it to the dispatch queue on
    /// the first `forward()`.
    allocator: Arc<MetalAllocator>,
    pipelines: Arc<SpecializedPipelines>,
    bucket_tapes: Arc<[LoweredMetalTape<W>]>,
    arena_layout: Arc<ArenaLayout>,
    runtime_factory: RuntimeFactory,
    max_workers: usize,
    /// Tracks whether the allocator's residency set has been attached
    /// to a queue yet. Lazy-attached on the first `forward()` so the
    /// pool builder doesn't need a `CommandQueue` (the queue lives on
    /// `GpuDevice` and is passed in at forward time). `attach_to_queue`
    /// is idempotent per (queue, set) pair so a second attach from
    /// `ferrite_worker::initialize_cache` is harmless.
    residency_attached: std::sync::atomic::AtomicBool,
    inner: Mutex<PoolInner<W>>,
    cv: Condvar,
}

// `Retained<ProtocolObject<dyn MTL*>>` from objc2 isn't auto-Send/Sync
// because the protocol traits don't carry the markers; metal-rs bolted
// them on with `unsafe impl Send` on its own newtypes. The MTL retain/
// release / lifecycle ops are documented thread-safe and the worker
// pool spawns workers across threads, so we re-add the markers on the
// pool itself (and downstream worker types).
unsafe impl<W: CanonicalParams + Send + Sync> Send for MetalWorkerPool<W> {}
unsafe impl<W: CanonicalParams + Send + Sync> Sync for MetalWorkerPool<W> {}

struct PoolInner<W: CanonicalParams> {
    /// Workers ready to be handed out. `pop()` order is FIFO-ish
    /// (vec semantics: LIFO), which is fine — pool callers don't
    /// care about ordering.
    available: Vec<PooledWorker<W>>,
    /// Total workers ever created. `in_use = total_created - available.len()`.
    /// Capped at `max_workers`.
    total_created: usize,
}

impl<W: CanonicalParams> MetalWorkerPool<W> {
    /// Build the pool and eagerly create the first worker.
    ///
    /// Eager creation surfaces allocation/recording/pipeline-lookup
    /// failures at construction time and warms the first-forward
    /// path (no creation cost on the first checkout).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<Device>,
        weights: &W,
        allocator: Arc<MetalAllocator>,
        pipelines: Arc<SpecializedPipelines>,
        bucket_tapes: Arc<[LoweredMetalTape<W>]>,
        arena_layout: ArenaLayout,
        runtime_factory: RuntimeFactory,
        max_workers: usize,
    ) -> Result<Self, WorkerError> {
        assert!(max_workers >= 1, "max_workers must be >= 1");

        // The shared `MetalResidencySet` lives on the allocator now —
        // arena buffers are pinned automatically as `push_arena_locked`
        // runs (in `MetalAllocator`), so the pool no longer manages
        // residency creation or arena-hook wiring. Per-worker arena
        // buffers (allocated outside the allocator, in `MetalWorker::new`)
        // still need explicit insertion; that happens in `spawn_worker`
        // below by reading `allocator.residency()`.

        let pool = Self {
            device,
            allocator,
            pipelines,
            bucket_tapes,
            arena_layout: Arc::new(arena_layout),
            runtime_factory,
            max_workers,
            residency_attached: std::sync::atomic::AtomicBool::new(false),
            inner: Mutex::new(PoolInner {
                available: Vec::new(),
                total_created: 0,
            }),
            cv: Condvar::new(),
        };
        let first = pool.spawn_worker(weights)?;
        {
            let mut inner = pool.inner.lock().unwrap();
            inner.total_created = 1;
            inner.available.push(first);
        }
        Ok(pool)
    }

    /// Build the pool from a flat `&[MetalBucketSpec<W>]` plus the
    /// loaded model + its allocator + arena layout + runtime factory.
    ///
    /// This is the runtime-side prerequisite the `#[forward]` macro's
    /// emitted `metal_pool(...)` calls into. The macro materializes
    /// the static slices that back the `bucket_specs` and threads the
    /// loaded `Weights` + the `MetalAllocator` that owns its
    /// `MTLBuffer` arenas; the caller is responsible for the device,
    /// the arena byte-layout, the runtime factory, and the worker
    /// cap (typically derived from
    /// `floor((device_total - weights - misc) / per_worker_arena)`).
    pub fn for_buckets(
        device: Arc<Device>,
        weights: &W,
        allocator: Arc<MetalAllocator>,
        bucket_specs: &[MetalBucketSpec<W>],
        runtime_factory: RuntimeFactory,
        max_workers: usize,
    ) -> Result<Self, PoolBuildError> {
        if bucket_specs.is_empty() {
            return Err(PoolBuildError::NoBuckets);
        }

        // Worker arena is sized for the largest activation across
        // every bucket — every spec's `arena_bytes` has the same
        // length (the macro shares the colored slot map across
        // canonicals), so taking elementwise max is safe.
        let num_slots = bucket_specs[0].num_arena_slots as usize;
        let mut arena_layout: ArenaLayout = vec![0u64; num_slots];
        for spec in bucket_specs {
            debug_assert_eq!(
                spec.arena_bytes.len(),
                num_slots,
                "every bucket spec must share the same colored slot count"
            );
            for (slot, &bytes) in spec.arena_bytes.iter().enumerate() {
                if bytes > arena_layout[slot] {
                    arena_layout[slot] = bytes;
                }
            }
        }

        let cache = SpecializedPipelineCache::with_standard_shaders((*device).clone())
            .map_err(|e| PoolBuildError::PipelineCacheBuild(format!("{e:?}")))?;
        let pipelines = Arc::new(SpecializedPipelines::new(Arc::new(cache)));

        let mut tapes: Vec<LoweredMetalTape<W>> = Vec::with_capacity(bucket_specs.len());
        for spec in bucket_specs {
            let tape = lower_pair(
                spec.backbone,
                spec.lm_head,
                spec.bucket_m,
                spec.num_arena_slots,
            )
            .map_err(|e| PoolBuildError::BucketLower {
                bucket_m: spec.bucket_m,
                error: format!("{e}"),
            })?;
            tapes.push(tape);
        }
        let bucket_tapes: Arc<[LoweredMetalTape<W>]> = Arc::from(tapes);

        Self::new(
            device,
            weights,
            allocator,
            pipelines,
            bucket_tapes,
            arena_layout,
            runtime_factory,
            max_workers,
        )
        .map_err(PoolBuildError::Worker)
    }

    pub fn max_workers(&self) -> usize {
        self.max_workers
    }

    /// The Metal device the pool's workers were allocated on. Callers
    /// use it to spin up a `CommandQueue` for [`Self::forward`] (the
    /// pool intentionally doesn't own the queue — the engine may
    /// share one across multiple pools / streams).
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Total workers currently allocated by the pool (whether or
    /// not they're checked out). Monotonically grows up to
    /// `max_workers`.
    pub fn current_size(&self) -> usize {
        self.inner.lock().unwrap().total_created
    }

    /// Number of workers currently sitting in the available queue.
    /// Test-facing — production callers don't need this.
    pub fn available(&self) -> usize {
        self.inner.lock().unwrap().available.len()
    }

    /// Block until a worker is available, then return a RAII guard.
    ///
    /// Three paths:
    /// 1. A worker is already idle → pop and return.
    /// 2. The pool is below cap → reserve a slot, drop the lock,
    ///    spawn (allocates GPU memory + records ICBs), return.
    /// 3. The pool is at cap and all workers are busy → wait on the
    ///    condvar until a peer thread checks one back in.
    ///
    /// Spawn failures release the reserved slot and are propagated.
    pub fn checkout(&self, weights: &W) -> Result<WorkerGuard<'_, W>, WorkerError> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            if let Some(pooled) = inner.available.pop() {
                return Ok(WorkerGuard {
                    pool: self,
                    inner: Some(pooled),
                });
            }
            if inner.total_created < self.max_workers {
                inner.total_created += 1;
                drop(inner);
                match self.spawn_worker(weights) {
                    Ok(pooled) => {
                        return Ok(WorkerGuard {
                            pool: self,
                            inner: Some(pooled),
                        });
                    }
                    Err(e) => {
                        let mut inner = self.inner.lock().unwrap();
                        inner.total_created -= 1;
                        // A peer might be waiting on capacity that we
                        // just released by failing to spawn. Notify so
                        // they can re-check (they'll either find a
                        // freed worker or hit the same spawn path).
                        self.cv.notify_one();
                        return Err(e);
                    }
                }
            }
            inner = self.cv.wait(inner).unwrap();
        }
    }

    /// Non-blocking checkout. Returns `None` when every worker is
    /// busy *and* the pool is at cap.
    ///
    /// Spawn failures inside `try_checkout` produce `Some(Err(...))`
    /// so callers can distinguish "pool is full" (`None`) from
    /// "GPU allocation failed" (`Some(Err)`).
    pub fn try_checkout(&self, weights: &W) -> Option<Result<WorkerGuard<'_, W>, WorkerError>> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(pooled) = inner.available.pop() {
            return Some(Ok(WorkerGuard {
                pool: self,
                inner: Some(pooled),
            }));
        }
        if inner.total_created < self.max_workers {
            inner.total_created += 1;
            drop(inner);
            match self.spawn_worker(weights) {
                Ok(pooled) => Some(Ok(WorkerGuard {
                    pool: self,
                    inner: Some(pooled),
                })),
                Err(e) => {
                    let mut inner = self.inner.lock().unwrap();
                    inner.total_created -= 1;
                    self.cv.notify_one();
                    Some(Err(e))
                }
            }
        } else {
            None
        }
    }

    /// Map a real `num_tokens` to the bucket index that should run.
    ///
    /// Picks the smallest bucket whose `bucket_m >= num_tokens`. Tape
    /// ordering inside `bucket_tapes` is intentionally not assumed —
    /// the macro / model loader supplies whatever order it likes
    /// (typically ascending), and a linear scan over a handful of
    /// buckets is cheaper than maintaining a sorted invariant.
    pub fn pick_bucket(&self, num_tokens: u32) -> Result<usize, ForwardError> {
        if num_tokens == 0 {
            return Err(ForwardError::ZeroTokens);
        }
        let mut best: Option<(usize, u32)> = None;
        let mut max_bucket: u32 = 0;
        for (i, tape) in self.bucket_tapes.iter().enumerate() {
            if tape.bucket_m > max_bucket {
                max_bucket = tape.bucket_m;
            }
            if tape.bucket_m >= num_tokens {
                best = match best {
                    Some((_, bm)) if bm <= tape.bucket_m => best,
                    _ => Some((i, tape.bucket_m)),
                };
            }
        }
        best.map(|(i, _)| i).ok_or(ForwardError::NoBucketFits {
            num_tokens,
            max_bucket,
        })
    }

    /// Run one forward step.
    ///
    /// Pipeline:
    ///  1. Pick the bucket from `inputs.num_tokens`.
    ///  2. Check out a worker (eagerly grow the pool if below cap;
    ///     block if at cap).
    ///  3. Validate every present input slice against its runtime
    ///     buffer's capacity; copy bytes into the buffer's
    ///     `contents()`.
    ///  4. Allocate a fresh command buffer from `queue`, walk the
    ///     bucket's plan via `worker.run_bucket`, commit, and wait
    ///     until completed.
    ///  5. Run `with_output(&worker)` so the caller can read arena
    ///     buffers (e.g. logits in the final slot) before the worker
    ///     is checked back in.
    ///  6. Drop the guard — the worker returns to the pool.
    ///
    /// All validation runs *before* any GPU work is submitted, so a
    /// malformed [`ForwardInputs`] never partially executes.
    pub fn forward<R>(
        &self,
        weights: &W,
        queue: &CommandQueue,
        inputs: &ForwardInputs<'_>,
        with_output: impl FnOnce(&MetalWorker<W>, usize) -> R,
    ) -> Result<R, ForwardError> {
        let bucket_idx = self.pick_bucket(inputs.num_tokens)?;

        // Lazily commit + attach the allocator's residency set on the
        // first forward — both calls are idempotent per (queue, set),
        // but we gate with an atomic bool to avoid flooding the
        // driver with duplicate calls across thousands of forwards.
        // The pool only ever sees one queue (the one on `GpuDevice`),
        // so wire-once-and-cache is safe.
        //
        // commit() applies any inserts queued by `register_mmap` /
        // arena push / `ferrite_worker::initialize_cache`'s KV-cache
        // wiring (~125ms on Llama-3.2-3B's 4.7 GB KV pool). Done
        // here rather than in init_cache so it overlaps with the
        // pool build / first-dispatch encoding instead of blocking
        // the engine init path.
        if !self
            .residency_attached
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            self.allocator.residency().commit();
            self.allocator.residency().attach_to_queue(queue);
        }

        let guard = self.checkout(weights)?;
        write_runtime_inputs(&guard.runtime, inputs)?;

        // Default execution: per-step command buffer + direct dispatch
        // (ICB bypass). Two reasons we make this the default:
        //  1. The ICB execution path is currently broken for bf16 GEMM
        //     — the kernel runs but produces stale/wrong outputs for
        //     decode buckets. Symptom on Llama-3.2-3B: prefill emits a
        //     coherent first token then decode collapses to repeated
        //     special characters. Direct dispatch sidesteps the ICB
        //     write bug entirely (matches every passing kernel golden).
        //  2. Per-step cmdbufs are empirically ~10× faster than the
        //     batched single-cmdbuf `run_bucket` on Apple Silicon.
        //
        // The env vars below stay as overrides for debugging:
        //   `FERRITE_METAL_STEP_DEBUG=1` — per-step + eprintln dumps.
        //   `FERRITE_METAL_USE_BATCHED_CMDBUF=1` — old `run_bucket`
        //                                          batched path.
        //   `FERRITE_METAL_NO_DIRECT_DISPATCH=1` — keep per-step but
        //     drive ICB execution (broken for bf16; left for diagnosis).
        if std::env::var_os("FERRITE_METAL_STEP_DEBUG").is_some() {
            guard
                .worker
                .run_bucket_per_step_debug(bucket_idx, &self.device, queue)?;
        } else if std::env::var_os("FERRITE_METAL_USE_BATCHED_CMDBUF").is_some() {
            let trace = std::env::var_os("FERRITE_METAL_TRACE").is_some();
            let t_pre = std::time::Instant::now();
            let cb = queue.commandBuffer().expect("commandBuffer returned nil");
            guard.worker.run_bucket(bucket_idx, &self.device, &cb)?;
            let encoded = t_pre.elapsed();
            cb.commit();
            let committed = t_pre.elapsed();
            cb.waitUntilCompleted();
            let waited = t_pre.elapsed();
            let status = cb.status();
            if trace {
                eprintln!(
                    "[forward bucket={} num_tokens={}] encode={:?} commit={:?} wait={:?} status={:?}",
                    bucket_idx,
                    inputs.num_tokens,
                    encoded,
                    committed - encoded,
                    waited - committed,
                    status,
                );
            }
            if status != MTLCommandBufferStatus::Completed {
                return Err(ForwardError::ExecutionFailed(status));
            }
        } else {
            // Production default: per-step + direct-dispatch path. The
            // `run_bucket_per_step_silent` defaults `direct=true` and
            // reads `FERRITE_METAL_DIRECT_DISPATCH=0` as the off-switch.
            let trace = std::env::var_os("FERRITE_METAL_TRACE").is_some();
            let t0 = std::time::Instant::now();
            guard
                .worker
                .run_bucket_per_step_silent(bucket_idx, &self.device, queue)?;
            if trace {
                eprintln!(
                    "[forward bucket={} num_tokens={} per_step] total={:?}",
                    bucket_idx,
                    inputs.num_tokens,
                    t0.elapsed(),
                );
            }
        }

        // DIAGNOSTIC: dump non-zero counts for each arena slot. Tells
        // us where in the chain values transition from real to zero.
        if std::env::var_os("VLLM_DUMP_ARENA").is_some() {
            for slot in 0..guard.worker.arena.len() {
                let buf = &guard.worker.arena[slot];
                let len_bytes = buf.length();
                let row0 = unsafe {
                    std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len_bytes)
                };
                let nonzero_bytes = row0.iter().filter(|&&v| v != 0).count();
                eprintln!(
                    "[diag-arena] slot={:3} bytes={} nonzero_bytes={}/{}",
                    slot, len_bytes, nonzero_bytes, len_bytes,
                );
            }
        }

        // DIAGNOSTIC: dump first/last 4 bf16 values of every arena
        // slot. Used to bisect where forward N's outputs diverge from
        // forward N+1's across runs. Reads bytes directly as bf16.
        if std::env::var_os("VLLM_DUMP_ARENA_BF16").is_some() {
            fn bf16_bits_to_f32(bits: u16) -> f32 {
                f32::from_bits((bits as u32) << 16)
            }
            for slot in 0..guard.worker.arena.len() {
                let buf = &guard.worker.arena[slot];
                let len_bytes = buf.length();
                if len_bytes < 32 {
                    continue;
                }
                let bytes = unsafe {
                    std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len_bytes)
                };
                let head = (0..16)
                    .map(|j| {
                        let off = j * 2;
                        let bits = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
                        format!("{:.4}", bf16_bits_to_f32(bits))
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!("[diag-bf16] slot={:3} head=[{}]", slot, head);
            }
        }

        Ok(with_output(&guard.worker, bucket_idx))
    }

    fn spawn_worker(&self, weights: &W) -> Result<PooledWorker<W>, WorkerError> {
        let runtime = (self.runtime_factory)(&self.device);
        let worker = MetalWorker::<W>::new_with_residency(
            self.device.clone(),
            &self.arena_layout,
            &self.bucket_tapes,
            &self.pipelines,
            weights,
            &self.allocator,
            &runtime,
            Some(self.allocator.residency()),
        )?;
        Ok(PooledWorker { worker, runtime })
    }

    fn checkin(&self, pooled: PooledWorker<W>) {
        let mut inner = self.inner.lock().unwrap();
        inner.available.push(pooled);
        // Wake exactly one waiter — at most one of them can claim
        // this worker, the others stay blocked.
        self.cv.notify_one();
    }
}

/// Copy each present input slice into the matching runtime buffer's
/// host-visible `contents()`. Validates length first; on overflow
/// returns [`ForwardError::BufferTooSmall`] without touching any
/// buffer.
///
/// The runtime buffers are `MTLResourceOptions::StorageModeShared`
/// (per `RuntimeFactory` callers), so `contents()` is a host pointer
/// directly into GPU-visible memory — no staging copy needed.
fn write_runtime_inputs(
    runtime: &RuntimeBindings,
    inputs: &ForwardInputs<'_>,
) -> Result<(), ForwardError> {
    write_slice("input_ids", &runtime.input_ids, inputs.input_ids)?;
    write_slice("positions", &runtime.positions, inputs.positions)?;
    if let Some(s) = inputs.slot_mapping {
        // Padding lanes get sentinel `u32::MAX` so the rope_append
        // kernel can early-out before writing the paged K/V cache.
        // Zero-fill would otherwise route every padding token's K
        // projection into cache slot 0, overwriting the real K of
        // position 0 (each padding token has positions[t]=0 and
        // input_ids[t]=0, so they all write K_proj(token 0) to slot 0,
        // racing with — and winning against — the real position-0 write).
        write_slot_mapping(&runtime.slot_mapping, s)?;
    }
    if let Some(s) = inputs.cu_seqlens_q {
        write_slice("cu_seqlens_q", &runtime.cu_seqlens_q, s)?;
    }
    if let Some(s) = inputs.seq_used_k {
        write_slice("seq_used_k", &runtime.seq_used_k, s)?;
        if std::env::var("FERRITE_METAL_STEP_DEBUG").is_ok() {
            eprintln!("[runtime] seq_used_k = {:?} (len={})", s, s.len());
        }
    }
    if std::env::var("FERRITE_METAL_STEP_DEBUG").is_ok()
        || std::env::var("FERRITE_METAL_TRACE").is_ok()
    {
        if let Some(s) = inputs.block_table {
            eprintln!(
                "[runtime] block_table[0..min(8,len)] = {:?} (len={})",
                &s[..s.len().min(8)],
                s.len(),
            );
        }
        eprintln!(
            "[runtime] num_tokens = {} input_ids[0..min(8,len)] = {:?} positions[0..min(8,len)] = {:?}",
            inputs.num_tokens,
            &inputs.input_ids[..inputs.input_ids.len().min(8)],
            &inputs.positions[..inputs.positions.len().min(8)],
        );
    }
    if let Some(s) = inputs.block_table {
        write_slice("block_table", &runtime.block_table, s)?;
    }
    Ok(())
}

/// Write `src` into `buffer`, filling padding lanes with `u32::MAX`
/// (the rope_append sentinel that means "skip cache write"). Mirrors
/// `write_slice` except for the padding fill value.
fn write_slot_mapping(buffer: &Buffer, src: &[u32]) -> Result<(), ForwardError> {
    let bytes_needed = std::mem::size_of_val(src);
    let bytes_available = buffer.length();
    if bytes_needed > bytes_available {
        return Err(ForwardError::BufferTooSmall {
            kind: "slot_mapping",
            bytes_needed,
            bytes_available,
        });
    }
    unsafe {
        // 0xFF byte-fill = u32::MAX in every lane.
        std::ptr::write_bytes(
            buffer.contents().as_ptr() as *mut u8,
            0xFFu8,
            bytes_available,
        );
        if bytes_needed > 0 {
            copy_nonoverlapping(
                src.as_ptr() as *const u8,
                buffer.contents().as_ptr() as *mut u8,
                bytes_needed,
            );
        }
    }
    Ok(())
}

fn write_slice(kind: &'static str, buffer: &Buffer, src: &[u32]) -> Result<(), ForwardError> {
    let bytes_needed = std::mem::size_of_val(src);
    let bytes_available = buffer.length();
    if bytes_needed > bytes_available {
        return Err(ForwardError::BufferTooSmall {
            kind,
            bytes_needed,
            bytes_available,
        });
    }
    // Zero the WHOLE runtime buffer first, then overwrite the leading
    // `bytes_needed` from `src`. The kernels dispatch over the bucket's
    // padded M (= the buffer's full length), but only the first
    // `actual_num_tokens` of input data is meaningful — without this
    // zero-fill, padding lanes read whatever was left in the buffer
    // from a previous forward's RuntimeBindings allocation. RoPE is
    // the canary: `positions[t]` for `t >= actual_n` indexes the
    // cos_sin table, and a stale (uninitialized) value blows past the
    // table's bounds, faulting the GPU and hanging the command
    // buffer in `wait_until_completed`.
    //
    // Zero-pad is only correct for inputs whose `0`-th index is a valid
    // no-op for the consuming kernel: positions[t]=0 ↦ row-0 of the
    // cos_sin table; seq_used_k[s]=0 ↦ no kv tokens scanned;
    // block_table[s][b]=0 ↦ readable physical block with whatever was
    // already there. `slot_mapping` does NOT satisfy this — slot 0 is a
    // valid storage location, so a padding-lane write to it corrupts
    // real K/V data — and goes through `write_slot_mapping` (sentinel
    // u32::MAX + early-out in rope_append) instead.
    //
    // Safety: shared-storage buffers expose `contents()` as a
    // host-visible pointer; we've bounds-checked the byte count
    // against `length()` above; src and dst don't overlap (src is a
    // Rust slice in CPU memory).
    unsafe {
        std::ptr::write_bytes(buffer.contents().as_ptr() as *mut u8, 0u8, bytes_available);
        if bytes_needed > 0 {
            copy_nonoverlapping(
                src.as_ptr() as *const u8,
                buffer.contents().as_ptr() as *mut u8,
                bytes_needed,
            );
        }
    }
    Ok(())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::interpreter::metal::__re::{Buffer, MTLDevice, MTLResourceOptions};
    use crate::interpreter::metal::lowered::{
        Binding, DispatchShape, KernelId, LoweredCommand, LoweredMetalTape, WeightBundleKind,
        WeightTensor,
    };
    use ferrite_cuda_core::{DType, DeviceAllocator, GpuTensor};
    use ferrite_kernels::layers::RmsNorm;
    use ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Test fixture: holds `CanonicalParams` constants AND the layer
    /// instances the WtFn thunks below dereference. Pool tests only
    /// exercise RmsNorm bindings (the `synthetic_tape` builder
    /// below), so only the rmsnorm layer needs real backing.
    struct TestWeights {
        rmsnorm_layer: RmsNorm,
    }
    impl CanonicalParams for TestWeights {
        const HEAD_DIM: u32 = 64;
        const NUM_Q_HEADS: u32 = 32;
        const NUM_KV_HEADS: u32 = 4;
        const Q_SIZE: usize = 2048;
        const KV_SIZE: usize = 256;
        const INTERMEDIATE_SIZE: usize = 5632;
        const ATTN_SCALE: f32 = 0.125;
        const ATTN_SOFTCAP: f32 = 0.0;
        const SLIDING_WINDOW: i32 = -1;
        const KV_LORA_RANK: usize = 0;
        const QK_NOPE_HEAD_DIM: usize = 0;
        const QK_ROPE_HEAD_DIM: usize = 0;
        const V_HEAD_DIM: usize = 0;
        const FINAL_LOGIT_SOFTCAPPING: f32 = 0.0;
        const QK_HEAD_DIM: usize = 0;
        const MLA_ATTN_SCALE: f32 = 0.0;
    }

    fn rmsnorm_thunk(w: &TestWeights, _layer: u32) -> &RmsNorm {
        &w.rmsnorm_layer
    }

    /// Build a `TestWeights` + the allocator that owns its RmsNorm
    /// weight's backing MTLBuffer. The allocator is also threaded
    /// into the pool so the worker can map `weight.raw_ptr()` back
    /// to `(&MTLBuffer, offset)` at ICB-record time.
    fn build_test_weights(device: &Device) -> (Arc<TestWeights>, Arc<MetalAllocator>) {
        let mut allocator = MetalAllocator::new(device.clone());
        let bytes = vec![0u8; TestWeights::Q_SIZE * 2];
        let ptr = unsafe {
            allocator
                .alloc_and_copy_host(bytes.as_ptr(), bytes.len())
                .expect("rmsnorm weight alloc")
        };
        let tensor = unsafe { GpuTensor::new(ptr, &[TestWeights::Q_SIZE], DType::F16) };
        let weights = Arc::new(TestWeights {
            rmsnorm_layer: RmsNorm::new(tensor, 1e-5),
        });
        (weights, Arc::new(allocator))
    }

    fn alloc(device: &Device, bytes: u64) -> Buffer {
        device
            .newBufferWithLength_options(
                bytes.max(1) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBuffer")
    }

    fn empty_runtime(device: &Device, num_layers: usize) -> RuntimeBindings {
        RuntimeBindings {
            input_ids: alloc(device, 16),
            positions: alloc(device, 16),
            slot_mapping: alloc(device, 16),
            cu_seqlens_q: alloc(device, 16),
            seq_used_k: alloc(device, 16),
            block_table: alloc(device, 16),
            kv_cache_k: (0..num_layers).map(|_| alloc(device, 16)).collect(),
            kv_cache_v: (0..num_layers).map(|_| alloc(device, 16)).collect(),
        }
    }

    /// Single-bucket synthetic tape: one RmsNorm command. Enough to
    /// verify the worker bakes; the kernel itself isn't fired in
    /// 5.D tests (5.E hooks `run_bucket` to a real cmdbuf).
    fn synthetic_tape(bucket_m: u32) -> LoweredMetalTape<TestWeights> {
        let cmd = LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            function: "rmsnorm_f16_specialized",
            constants: vec![
                ferrite_metal_kernels::specialized_pipeline_cache::ConstantValue::uint(0, bucket_m),
                ferrite_metal_kernels::specialized_pipeline_cache::ConstantValue::uint(
                    1,
                    <TestWeights as crate::CanonicalParams>::Q_SIZE as u32,
                ),
                ferrite_metal_kernels::specialized_pipeline_cache::ConstantValue::float(
                    2,
                    <TestWeights as crate::CanonicalParams>::RMS_NORM_EPS,
                ),
            ],
            dispatch: DispatchShape {
                threadgroups: (bucket_m, 1, 1),
                threads_per_threadgroup: (256, 1, 1),
            },
            bindings: vec![
                Binding::ArenaSlot {
                    slot: 0,
                    binding_index: 0,
                },
                Binding::ArenaSlot {
                    slot: 1,
                    binding_index: 1,
                },
                Binding::Weight {
                    kind: WeightBundleKind::RmsNorm(rmsnorm_thunk),
                    which: WeightTensor::Weight,
                    layer: 0,
                    binding_index: 2,
                },
            ],
            gemm_dims: None,
        };
        LoweredMetalTape {
            bucket_m,
            num_arena_slots: 2,
            commands: vec![cmd],
        }
    }

    /// Build a pool with `max_workers = max` for a one-bucket
    /// TinyLlama-shaped synthetic tape. Returns `None` when no
    /// Metal device is present (lets each test silent-skip).
    fn build_pool(max: usize) -> Option<(Arc<TestWeights>, MetalWorkerPool<TestWeights>)> {
        let device = ferrite_metal_kernels::detect_device()?;
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = Arc::new(SpecializedPipelines::new(cache));

        let (weights, allocator) = build_test_weights(&device);

        let tapes: Arc<[_]> = Arc::from(vec![synthetic_tape(1)]);
        let arena_layout: ArenaLayout = vec![4096, 4096];
        let runtime_factory: RuntimeFactory = Arc::new(|d| empty_runtime(d, 1));

        let pool = MetalWorkerPool::<TestWeights>::new(
            device,
            &weights,
            allocator,
            pipelines,
            tapes,
            arena_layout,
            runtime_factory,
            max,
        )
        .expect("pool builds");
        Some((weights, pool))
    }

    #[test]
    fn pool_starts_with_one_worker() {
        let Some((_w, pool)) = build_pool(4) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        assert_eq!(pool.max_workers(), 4);
        assert_eq!(pool.current_size(), 1, "first worker eagerly created");
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn checkout_returns_eagerly_created_worker_first() {
        let Some((w, pool)) = build_pool(4) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let _g = pool.checkout(&w).expect("checkout 1");
        assert_eq!(pool.current_size(), 1, "first checkout reuses eager worker");
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn pool_grows_under_demand_up_to_cap() {
        let Some((w, pool)) = build_pool(3) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let g1 = pool.checkout(&w).expect("checkout 1");
        let g2 = pool.checkout(&w).expect("checkout 2");
        let g3 = pool.checkout(&w).expect("checkout 3");
        assert_eq!(pool.current_size(), 3, "grew to cap");
        assert_eq!(pool.available(), 0);
        // try_checkout at cap returns None, not Some(Err).
        assert!(pool.try_checkout(&w).is_none());
        drop(g1);
        drop(g2);
        drop(g3);
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.current_size(), 3, "cap unchanged after checkin");
    }

    #[test]
    fn guard_drop_returns_worker_to_pool() {
        let Some((w, pool)) = build_pool(2) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        {
            let _g = pool.checkout(&w).expect("checkout");
            assert_eq!(pool.available(), 0);
        }
        assert_eq!(pool.available(), 1, "drop returns worker");
        // Subsequent checkout reuses the existing worker rather than
        // growing the pool.
        let _g2 = pool.checkout(&w).expect("checkout 2");
        assert_eq!(pool.current_size(), 1, "no growth on reuse");
    }

    /// Concurrency test: hold the only worker on the main thread,
    /// spawn a second thread that calls `checkout()` (must block),
    /// release the worker, verify the spawned thread unblocks.
    ///
    /// Uses an `AtomicBool` + a small bounded sleep to detect "still
    /// blocked". Sleep is short to keep the test fast; flake risk is
    /// low because the spawned thread only flips the flag *after*
    /// `checkout()` returns.
    #[test]
    fn checkout_blocks_when_at_cap_unblocks_on_checkin() {
        let Some((w, pool)) = build_pool(1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = Arc::new(pool);
        let g1 = pool.checkout(&w).expect("checkout 1");
        assert_eq!(pool.available(), 0);

        let pool_c = pool.clone();
        let w_c = w.clone();
        let started = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(AtomicBool::new(false));
        let started_c = started.clone();
        let completed_c = completed.clone();

        let handle = std::thread::spawn(move || {
            started_c.store(true, Ordering::SeqCst);
            let _g = pool_c.checkout(&w_c).expect("blocking checkout");
            completed_c.store(true, Ordering::SeqCst);
        });

        // Wait for the spawned thread to enter checkout. SeqCst load
        // is sufficient — `started` is set before the blocking call.
        while !started.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        // Give the spawned thread a chance to enter the wait state.
        std::thread::sleep(Duration::from_millis(20));
        assert!(
            !completed.load(Ordering::SeqCst),
            "spawned thread should still be blocked while main holds the only worker"
        );

        drop(g1);
        handle.join().expect("spawned thread completed");
        assert!(
            completed.load(Ordering::SeqCst),
            "spawned thread unblocked once main checked the worker back in"
        );
        // Pool didn't grow — the spawned thread reused the existing one.
        assert_eq!(pool.current_size(), 1);
    }

    /// `try_checkout` returns `None` (not `Some(Err)`) when the pool
    /// is at cap and every worker is busy. Distinguishes "full" from
    /// "GPU OOM".
    #[test]
    fn try_checkout_at_cap_returns_none() {
        let Some((w, pool)) = build_pool(1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let _g = pool.checkout(&w).expect("checkout");
        match pool.try_checkout(&w) {
            None => {}
            Some(Ok(_)) => panic!("try_checkout should not have succeeded at cap"),
            Some(Err(e)) => panic!("try_checkout should be None at cap, got Err({e:?})"),
        }
    }

    /// Two concurrent threads each grow the pool by one and run to
    /// completion. Verifies the spawn-while-locking-released path
    /// doesn't double-allocate or deadlock.
    #[test]
    fn concurrent_growth_to_cap() {
        let Some((w, pool)) = build_pool(2) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = Arc::new(pool);
        let pool_a = pool.clone();
        let pool_b = pool.clone();
        let w_a = w.clone();
        let w_b = w.clone();

        let h_a = std::thread::spawn(move || {
            let _g = pool_a.checkout(&w_a).expect("checkout a");
            std::thread::sleep(Duration::from_millis(10));
        });
        let h_b = std::thread::spawn(move || {
            let _g = pool_b.checkout(&w_b).expect("checkout b");
            std::thread::sleep(Duration::from_millis(10));
        });

        h_a.join().unwrap();
        h_b.join().unwrap();
        assert!(
            pool.current_size() <= 2,
            "pool must not exceed max_workers (got {})",
            pool.current_size()
        );
        assert_eq!(pool.available(), pool.current_size());
    }

    // ---------------- Phase 5.E: forward() ----------------

    /// Build a pool whose tape carries one bucket per provided
    /// `bucket_m`. Arena slots are sized for the largest bucket so the
    /// runtime buffers / arena handle every entry. The runtime factory
    /// sizes per-token arrays for the largest bucket too — smaller
    /// `num_tokens` values fit trivially.
    fn build_multi_bucket_pool(
        bucket_ms: &[u32],
        max_workers: usize,
    ) -> Option<(Arc<TestWeights>, MetalWorkerPool<TestWeights>)> {
        let device = ferrite_metal_kernels::detect_device()?;
        let device = Arc::new(device.device.clone());
        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = Arc::new(SpecializedPipelines::new(cache));
        let (weights, allocator) = build_test_weights(&device);
        let tapes: Arc<[_]> = bucket_ms
            .iter()
            .copied()
            .map(synthetic_tape)
            .collect::<Vec<_>>()
            .into();
        // Arena slot for the synthetic RmsNorm: M × hidden_size f16 =
        // M × Q_SIZE × 2 bytes. Sized for the worst-case bucket.
        let max_m = bucket_ms.iter().copied().max().unwrap_or(1) as u64;
        let slot_bytes = max_m * (TestWeights::Q_SIZE as u64) * 2;
        let arena_layout: ArenaLayout = vec![slot_bytes, slot_bytes];
        // Per-token runtime arrays sized for max bucket.
        let max_m_bytes = (max_m * 4).max(16);
        let runtime_factory: RuntimeFactory = Arc::new(move |d| RuntimeBindings {
            input_ids: alloc(d, max_m_bytes),
            positions: alloc(d, max_m_bytes),
            slot_mapping: alloc(d, max_m_bytes),
            cu_seqlens_q: alloc(d, 16),
            seq_used_k: alloc(d, 16),
            block_table: alloc(d, 16),
            kv_cache_k: vec![alloc(d, 16)],
            kv_cache_v: vec![alloc(d, 16)],
        });
        let pool = MetalWorkerPool::<TestWeights>::new(
            device,
            &weights,
            allocator,
            pipelines,
            tapes,
            arena_layout,
            runtime_factory,
            max_workers,
        )
        .expect("pool builds");
        Some((weights, pool))
    }

    #[test]
    fn pick_bucket_returns_smallest_fit() {
        let Some((_w, pool)) = build_multi_bucket_pool(&[1, 8, 32], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        // Decode bucket.
        assert_eq!(pool.pick_bucket(1).unwrap(), 0);
        // Doesn't fit bucket 0; picks the next-smallest that fits.
        assert_eq!(pool.pick_bucket(2).unwrap(), 1);
        assert_eq!(pool.pick_bucket(8).unwrap(), 1);
        assert_eq!(pool.pick_bucket(9).unwrap(), 2);
        assert_eq!(pool.pick_bucket(32).unwrap(), 2);
    }

    #[test]
    fn pick_bucket_zero_tokens_errors() {
        let Some((_w, pool)) = build_multi_bucket_pool(&[1, 8], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        assert!(matches!(pool.pick_bucket(0), Err(ForwardError::ZeroTokens)));
    }

    #[test]
    fn pick_bucket_overflow_errors() {
        let Some((_w, pool)) = build_multi_bucket_pool(&[1, 8], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        match pool.pick_bucket(9) {
            Err(ForwardError::NoBucketFits {
                num_tokens: 9,
                max_bucket: 8,
            }) => {}
            other => panic!("expected NoBucketFits {{ 9, 8 }}, got {other:?}"),
        }
    }

    /// Tape order isn't required to be sorted — `pick_bucket` should
    /// still find the smallest fit when buckets come in arbitrary
    /// order.
    #[test]
    fn pick_bucket_handles_unsorted_tape_order() {
        let Some((_w, pool)) = build_multi_bucket_pool(&[32, 1, 8], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        assert_eq!(
            pool.pick_bucket(1).unwrap(),
            1,
            "smallest bucket at index 1"
        );
        assert_eq!(
            pool.pick_bucket(8).unwrap(),
            2,
            "8 fits index 2 (bucket_m=8)"
        );
        assert_eq!(pool.pick_bucket(9).unwrap(), 0, "9 only fits the 32 bucket");
    }

    #[test]
    fn forward_runs_one_decode_step() {
        let Some((w, pool)) = build_multi_bucket_pool(&[1], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let queue = pool.device().newCommandQueue().expect("newCommandQueue");
        let inputs = ForwardInputs {
            num_tokens: 1,
            input_ids: &[0u32],
            positions: &[0u32],
            slot_mapping: None,
            cu_seqlens_q: None,
            seq_used_k: None,
            block_table: None,
        };
        // Closure runs *while* the worker is checked out — assertion
        // is that we got it (not on numerical correctness; that's
        // 5.G's job via cpu_golden).
        let saw = pool
            .forward(&w, &queue, &inputs, |worker, _bucket_idx| {
                // Worker arena is alive in the closure; reading its
                // contents would inspect the rmsnorm output. We only
                // assert structural facts here.
                assert_eq!(worker.bucket_bakings.len(), 1);
                42u32
            })
            .expect("forward succeeds");
        assert_eq!(saw, 42);
        assert_eq!(pool.available(), 1, "worker returned to pool after forward");
    }

    #[test]
    fn forward_rejects_zero_tokens() {
        let Some((w, pool)) = build_multi_bucket_pool(&[1], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let queue = pool.device().newCommandQueue().expect("newCommandQueue");
        let inputs = ForwardInputs {
            num_tokens: 0,
            input_ids: &[],
            positions: &[],
            slot_mapping: None,
            cu_seqlens_q: None,
            seq_used_k: None,
            block_table: None,
        };
        let err = pool
            .forward(&w, &queue, &inputs, |_, _| ())
            .expect_err("zero-token forward rejected");
        assert!(matches!(err, ForwardError::ZeroTokens));
        // Worker was never checked out — pool stays at the eager 1.
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn forward_rejects_oversized_token_count() {
        let Some((w, pool)) = build_multi_bucket_pool(&[1, 8], 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let queue = pool.device().newCommandQueue().expect("newCommandQueue");
        let big = vec![0u32; 9];
        let inputs = ForwardInputs {
            num_tokens: 9,
            input_ids: &big,
            positions: &big,
            slot_mapping: None,
            cu_seqlens_q: None,
            seq_used_k: None,
            block_table: None,
        };
        match pool.forward(&w, &queue, &inputs, |_, _| ()) {
            Err(ForwardError::NoBucketFits {
                num_tokens: 9,
                max_bucket: 8,
            }) => {}
            other => panic!("expected NoBucketFits {{ 9, 8 }}, got {other:?}"),
        }
        assert_eq!(pool.available(), 1, "no checkout on bucket failure");
    }

    /// `BufferTooSmall` fires when the caller stages more bytes than
    /// the runtime buffer can hold. Crafted by sizing the runtime
    /// `input_ids` buffer to 16 bytes (default in this test setup) and
    /// passing a 5-element slice (20 bytes).
    #[test]
    fn forward_rejects_oversized_input_slice() {
        // Single-bucket pool with the *smaller* runtime layout — the
        // existing `build_pool` allocates 16-byte runtime buffers,
        // perfect for triggering the overflow path.
        let Some((w, pool)) = build_pool(1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let queue = pool.device().newCommandQueue().expect("newCommandQueue");
        // 5 × u32 = 20 bytes; runtime input_ids buffer is 16 bytes.
        let too_big = [0u32, 0u32, 0u32, 0u32, 0u32];
        let inputs = ForwardInputs {
            num_tokens: 1, // pick_bucket succeeds (bucket 0 = decode)
            input_ids: &too_big,
            positions: &[0u32],
            slot_mapping: None,
            cu_seqlens_q: None,
            seq_used_k: None,
            block_table: None,
        };
        let err = pool
            .forward(&w, &queue, &inputs, |_, _| ())
            .expect_err("oversized slice rejected");
        match err {
            ForwardError::BufferTooSmall {
                kind: "input_ids",
                bytes_needed: 20,
                bytes_available: 16,
            } => {}
            other => panic!("expected BufferTooSmall on input_ids, got {other:?}"),
        }
        // Checkout happened (validation runs after checkout) — verify
        // the worker came back via guard drop on the early Err return.
        assert_eq!(
            pool.available(),
            1,
            "worker returned to pool after validation failure"
        );
    }

    // ──────────────── Phase 5.F.4: for_buckets() ────────────────

    /// Empty backbone + lm_head per bucket — exercises the
    /// constructor's lower→pool-build path without requiring a real
    /// `Instruction<W>` to be constructible at the test site (the
    /// macro emits those at codegen time; pool tests stay structural).
    /// The resulting tape carries zero commands; the worker still
    /// bakes a (trivially empty) ICB per bucket and the pool still
    /// stands one worker up eagerly.
    const EMPTY_BACKBONE: &[Instruction<TestWeights>] = &[];
    const EMPTY_LM_HEAD: &[Instruction<TestWeights>] = &[];
    const TEST_ARENA_BYTES: &[u64] = &[4096, 4096];

    fn build_via_for_buckets(
        bucket_specs: &[MetalBucketSpec<TestWeights>],
        max_workers: usize,
    ) -> Option<Result<MetalWorkerPool<TestWeights>, PoolBuildError>> {
        let device = ferrite_metal_kernels::detect_device()?;
        let device = Arc::new(device.device.clone());
        let (weights, allocator) = build_test_weights(&device);
        let runtime_factory: RuntimeFactory = Arc::new(|d| empty_runtime(d, 1));
        Some(MetalWorkerPool::<TestWeights>::for_buckets(
            device,
            &weights,
            allocator,
            bucket_specs,
            runtime_factory,
            max_workers,
        ))
    }

    /// Empty bucket_specs surfaces `PoolBuildError::NoBuckets` —
    /// ahead of any Metal-device interaction so this test runs even
    /// on non-Apple hosts.
    #[test]
    fn for_buckets_rejects_empty_specs() {
        // Note: this path runs without a Metal device because the
        // `NoBuckets` check fires before any device call — no
        // silent-skip needed.
        let device = match ferrite_metal_kernels::detect_device() {
            Some(d) => Arc::new(d.device.clone()),
            None => {
                eprintln!("skipping: no Metal device");
                return;
            }
        };
        let (weights, allocator) = build_test_weights(&device);
        let runtime_factory: RuntimeFactory = Arc::new(|d| empty_runtime(d, 1));
        let res = MetalWorkerPool::<TestWeights>::for_buckets(
            device,
            &weights,
            allocator,
            &[],
            runtime_factory,
            1,
        );
        match res {
            Err(PoolBuildError::NoBuckets) => {}
            Err(other) => panic!("expected NoBuckets, got Err({other})"),
            Ok(_) => panic!("expected NoBuckets, got Ok(_)"),
        }
    }

    /// Single empty bucket builds: lower_pair on `(empty, empty)`
    /// produces an empty-command tape with the right `bucket_m` and
    /// `num_arena_slots`; the worker bakes one no-op ICB; the pool
    /// stands up a worker eagerly. Verifies the constructor wires
    /// pipeline cache / lowering / pool spawn end-to-end.
    #[test]
    fn for_buckets_builds_pool_for_single_empty_bucket() {
        let specs = [MetalBucketSpec {
            bucket_m: 1,
            num_arena_slots: 2,
            terminal_slot: 1,
            arena_bytes: TEST_ARENA_BYTES,
            backbone: EMPTY_BACKBONE,
            lm_head: EMPTY_LM_HEAD,
        }];
        let Some(res) = build_via_for_buckets(&specs, 2) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = res.expect("for_buckets constructs pool");
        assert_eq!(pool.max_workers(), 2);
        assert_eq!(pool.current_size(), 1, "first worker eagerly created");
        assert_eq!(pool.available(), 1);
    }

    /// Two empty buckets — verifies pick_bucket sees both
    /// `bucket_m`s and the constructor preserves spec order via
    /// `Arc<[…]>`.
    #[test]
    fn for_buckets_preserves_bucket_order() {
        let specs = [
            MetalBucketSpec {
                bucket_m: 1,
                num_arena_slots: 2,
                terminal_slot: 1,
                arena_bytes: TEST_ARENA_BYTES,
                backbone: EMPTY_BACKBONE,
                lm_head: EMPTY_LM_HEAD,
            },
            MetalBucketSpec {
                bucket_m: 8,
                num_arena_slots: 2,
                terminal_slot: 1,
                arena_bytes: TEST_ARENA_BYTES,
                backbone: EMPTY_BACKBONE,
                lm_head: EMPTY_LM_HEAD,
            },
        ];
        let Some(res) = build_via_for_buckets(&specs, 1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = res.expect("for_buckets constructs pool");
        assert_eq!(pool.pick_bucket(1).unwrap(), 0);
        assert_eq!(pool.pick_bucket(8).unwrap(), 1);
        assert!(matches!(
            pool.pick_bucket(9),
            Err(ForwardError::NoBucketFits { .. })
        ));
    }
}
