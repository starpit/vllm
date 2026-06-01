// SPDX-License-Identifier: Apache-2.0
//! `MetalWorkerPool`: growable, capped, semaphore-bounded checkout/checkin.
//!
//! The pool starts at size 1 and grows on demand up to `max_workers`.
//! Each worker holds a private arena, a private [`RuntimeBindings`],
//! and one baked execution plan per bucket — never shared across workers.
//! `checkout()` blocks if every worker is in use *and* the pool is at
//! cap; otherwise it grows by one and hands the new worker out.
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
    Buffer, CommandQueue, Device, MTLBuffer, MTLCommandBufferStatus,
};
use objc2_metal::{
    MTL4CommandAllocator, MTL4CommandBuffer, MTL4CommandEncoder, MTL4CommandQueue, MTLDevice,
    MTLSharedEvent,
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
///  - `backbone` / `lm_head` — the bucket's `Instruction` static
///    slices, identical to the cuda-side `BACKBONE_M_<wp>` /
///    `LM_HEAD_M_<wp>` statics.
///
/// [`MetalWorkerPool::for_buckets`] calls [`lower`] on the concatenated
/// `(backbone ++ lm_head)` to produce a [`LoweredMetalTape`] per spec
/// at constructor time. Concatenation matches the cuda interpreter's
/// behavior — `forward()` runs backbone then lm_head as one logical
/// pass for a given bucket — and the metal worker bakes both halves
/// into the bucket's single execution plan so `forward()` dispatches
/// both halves without an extra mid-bucket boundary.
///
/// The slices are `&'static` because the macro emits them as static
/// items; the spec is `Copy` so callers can drop the bucket plan into
/// an `Arc<[MetalBucketSpec]>` cheaply.
pub struct MetalBucketSpec {
    pub bucket_m: u32,
    /// Tape index passed to the per-arch [`crate::WeightAccessors`]
    /// impl when resolving weight bindings inside this bucket's
    /// backbone slice. The macro emits a unique id per
    /// `(canonical, backbone/lm_head)` pair so the trait's match
    /// arms can disambiguate same-op_idx-different-canonical cases.
    pub backbone_tape_index: u32,
    /// Tape index for the lm_head slice. See
    /// [`Self::backbone_tape_index`].
    pub lm_head_tape_index: u32,
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
    pub backbone: &'static [Instruction],
    pub lm_head: &'static [Instruction],
    /// MTL4 encoder barrier flags computed at macro time from the
    /// FUF dataflow graph (one bool per `Instruction` in
    /// `backbone`/`lm_head`). `true` means the MTL4 path must emit a
    /// `Dispatch→Dispatch` barrier before this instruction's first
    /// dispatched `LoweredCommand`. The runtime carries these
    /// straight to `Mtl4Step.barrier_before`; no runtime hazard
    /// walk. See `ferrite-forward-macro::interpreter_codegen::
    /// lower_bucket` for the analysis.
    pub backbone_barriers: &'static [bool],
    pub lm_head_barriers: &'static [bool],
}

impl Clone for MetalBucketSpec {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for MetalBucketSpec {}

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
pub struct PooledWorker<W: CanonicalParams> {
    pub worker: MetalWorker<W>,
    pub runtime: RuntimeBindings,
}

/// RAII guard returned by [`MetalWorkerPool::checkout`]. Returns the
/// underlying [`PooledWorker`] to the pool on drop.
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
/// `LoweredMetalTape` (workers bake ICBs from these), but it
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
    bucket_tapes: Arc<[LoweredMetalTape]>,
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
    /// Phase A.3 MTL4 surface. Lazily initialized on the first forward
    /// observing `FERRITE_METAL_MTL4=1`. Panics on init if MTL4 is
    /// unavailable — the env-var gate is treated as a hard assertion
    /// that the host supports MTL4 (macOS 15+ / Apple Family 7+).
    mtl4: Mutex<Option<Mtl4Pool>>,
    inner: Mutex<PoolInner<W>>,
    cv: Condvar,
}

/// Pool-owned MTL4 surface. The allocator is reset between forwards;
/// the command buffer is re-created per forward (cheap — Metal pools
/// internally). The shared event is monotonically signaled and
/// host-waited on each commit.
struct Mtl4Pool {
    queue: crate::interpreter::metal::__re::Mtl4Queue,
    allocator: crate::interpreter::metal::__re::Mtl4Allocator,
    shared_event: crate::interpreter::metal::__re::SharedEvent,
    signal_counter: u64,
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

/// GPU per-dispatch timing instrumentation. Enabled by setting
/// `FERRITE_METAL_DISPATCH_TIMING=1`. Allocates an MTL4CounterHeap
/// sized for `(N_dispatches + 1)` timestamp entries, threads it
/// through `run_bucket_mtl4_with_timing`, and resolves+prints the
/// per-dispatch GPU times after the forward completes. Per-pipeline
/// labels are derived from the pipeline pointer identity so the
/// caller can correlate slow dispatches back to specific kernels.
pub struct DispatchTimingState {
    pub heap: super::__re::Mtl4CounterHeap,
    pub heap_capacity: usize,
    /// Pipeline pointer per dispatch index — used purely as an
    /// opaque identifier so two dispatches sharing a pipeline get
    /// grouped together in the report.
    pipeline_ptr: std::cell::RefCell<Vec<usize>>,
    /// KernelId per dispatch index (human-readable label).
    pipeline_kernel: std::cell::RefCell<Vec<super::lowered::KernelId>>,
    /// (tg.x, tg.y, tg.z) per dispatch — captured post-`m_scaling`
    /// so it reflects the actual grid the GPU saw.
    dispatch_shape: std::cell::RefCell<Vec<(u32, u32, u32)>>,
    /// Total recorded dispatches (set on the closing timestamp).
    dispatch_count: std::cell::Cell<usize>,
    /// Nanoseconds per GPU timestamp tick — calibrated once at
    /// construction via two `sampleTimestamps:gpuTimestamp:` calls
    /// separated by a wall-clock interval. `resolve_and_print`
    /// multiplies raw counter-heap deltas by this to report real ns.
    /// Apple Silicon GPU timestamps tick at a device-specific rate
    /// (NOT 1 GHz), so treating the raw delta as ns gives results
    /// that are off by a constant factor (~24–40× on M-series).
    ns_per_gpu_tick: f64,
}

impl DispatchTimingState {
    fn new(device: &Device, count: usize) -> Option<Self> {
        use super::__re::{MTL4CounterHeapDescriptor, MTL4CounterHeapType};
        let desc = MTL4CounterHeapDescriptor::new();
        desc.setType(MTL4CounterHeapType::Timestamp);
        unsafe {
            desc.setCount(count);
        }
        let heap = device.newCounterHeapWithDescriptor_error(&desc).ok()?;

        // Calibrate GPU-tick → ns. `sampleTimestamps:gpuTimestamp:`
        // writes a synchronized pair: CPU timestamp in
        // mach_absolute_time ticks (= nanoseconds on Apple Silicon —
        // mach_timebase numer/denom is 1/1) and GPU timestamp in
        // GPU ticks. Two samples bracketed by a 10 ms sleep give
        // ns_per_gpu_tick = Δcpu_ns / Δgpu_ticks.
        let ns_per_gpu_tick = unsafe {
            use std::ptr::NonNull;
            let mut cpu1: u64 = 0;
            let mut gpu1: u64 = 0;
            device.sampleTimestamps_gpuTimestamp(
                NonNull::new_unchecked(&mut cpu1 as *mut u64),
                NonNull::new_unchecked(&mut gpu1 as *mut u64),
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
            let mut cpu2: u64 = 0;
            let mut gpu2: u64 = 0;
            device.sampleTimestamps_gpuTimestamp(
                NonNull::new_unchecked(&mut cpu2 as *mut u64),
                NonNull::new_unchecked(&mut gpu2 as *mut u64),
            );
            let cpu_dt = cpu2.wrapping_sub(cpu1) as f64;
            let gpu_dt = gpu2.wrapping_sub(gpu1) as f64;
            if gpu_dt > 0.0 { cpu_dt / gpu_dt } else { 1.0 }
        };
        eprintln!(
            "[dispatch-timing] GPU tick calibration: 1 GPU tick = {:.4} ns ({:.2} MHz)",
            ns_per_gpu_tick,
            1e3 / ns_per_gpu_tick,
        );

        Some(Self {
            heap,
            heap_capacity: count,
            pipeline_ptr: std::cell::RefCell::new(vec![0usize; count]),
            pipeline_kernel: std::cell::RefCell::new(vec![super::lowered::KernelId::Embed; count]),
            dispatch_shape: std::cell::RefCell::new(vec![(0u32, 0u32, 0u32); count]),
            dispatch_count: std::cell::Cell::new(0),
            ns_per_gpu_tick,
        })
    }

    pub fn record_label(
        &self,
        idx: usize,
        pipeline: &super::__re::ComputePipelineState,
        kernel: super::lowered::KernelId,
        tg: (u32, u32, u32),
    ) {
        let ptr = ::objc2::rc::Retained::as_ptr(pipeline) as *const () as usize;
        if let Some(slot) = self.pipeline_ptr.borrow_mut().get_mut(idx) {
            *slot = ptr;
        }
        if let Some(slot) = self.pipeline_kernel.borrow_mut().get_mut(idx) {
            *slot = kernel;
        }
        if let Some(slot) = self.dispatch_shape.borrow_mut().get_mut(idx) {
            *slot = tg;
        }
    }

    pub fn set_dispatch_count(&self, count: usize) {
        self.dispatch_count.set(count);
    }

    fn resolve_and_print(&self, bucket_idx: usize, num_tokens: usize, _device: &Device) {
        use objc2_foundation::{NSData, NSRange};
        use objc2_metal::MTL4CounterHeap as _;
        let n = self.dispatch_count.get();
        if n == 0 {
            return;
        }
        // Resolve [0, n+1) — n+1 timestamps for n dispatches.
        let data: Option<::objc2::rc::Retained<NSData>> = unsafe {
            self.heap.resolveCounterRange(NSRange {
                location: 0,
                length: n + 1,
            })
        };
        let Some(data) = data else {
            eprintln!("[dispatch-timing] resolveCounterRange returned nil");
            return;
        };
        let raw: &[u8] = unsafe { data.as_bytes_unchecked() };
        let n_u64 = raw.len() / 8;
        let bytes: &[u64] =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u64, n_u64) };
        eprintln!(
            "\n[dispatch-timing bucket={} num_tokens={}] {} dispatches",
            bucket_idx, num_tokens, n
        );
        // Aggregate by (pipeline_ptr, kernel) so we get per-kernel
        // totals + counts WITH human-readable labels.
        let mut by_pipe: std::collections::HashMap<usize, (u64, u32, super::lowered::KernelId)> =
            std::collections::HashMap::new();
        let mut total_ticks: u64 = 0;
        let labels = self.pipeline_ptr.borrow();
        let kernels = self.pipeline_kernel.borrow();
        for i in 0..n {
            let dt = bytes[i + 1].wrapping_sub(bytes[i]);
            total_ticks = total_ticks.wrapping_add(dt);
            let pid = labels[i];
            let kid = kernels[i];
            let entry = by_pipe.entry(pid).or_insert((0, 0, kid));
            entry.0 = entry.0.wrapping_add(dt);
            entry.1 += 1;
        }
        let mut sorted: Vec<(usize, u64, u32, super::lowered::KernelId)> = by_pipe
            .into_iter()
            .map(|(p, (sum, cnt, kid))| (p, sum, cnt, kid))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        let tick_ns = self.ns_per_gpu_tick;
        let total_ns_f = total_ticks as f64 * tick_ns;
        eprintln!(
            "    total GPU time (sum of dispatch deltas): {:>9.2} ms",
            total_ns_f / 1e6
        );
        eprintln!("    per-pipeline breakdown:");
        for (pid, sum, cnt, kid) in &sorted {
            let sum_ns = *sum as f64 * tick_ns;
            let avg_us = sum_ns / 1e3 / (*cnt as f64);
            let total_ms = sum_ns / 1e6;
            let pct = sum_ns / total_ns_f * 100.0;
            eprintln!(
                "      {:?}  pipe=0x{:016x}  count={:>3}  total={:>9.2} ms ({:>5.1}%)  avg/call={:>9.2} µs",
                kid, pid, cnt, total_ms, pct, avg_us
            );
        }
        // Per-dispatch ordered detail when FERRITE_METAL_DISPATCH_DETAIL=1.
        // Prints each dispatch's tick delta + kernel label + grid in
        // encoder order so we can see if specific kernel transitions
        // are slow and disambiguate same-kernel pipelines by shape.
        if std::env::var_os("FERRITE_METAL_DISPATCH_DETAIL").is_some() {
            let dispatch_shapes = self.dispatch_shape.borrow();
            eprintln!("    per-dispatch (in encoder order):");
            for i in 0..n {
                let dt_ticks = bytes[i + 1].wrapping_sub(bytes[i]);
                let dt_us = dt_ticks as f64 * tick_ns / 1e3;
                let pid = labels[i];
                let kid = kernels[i];
                let (tgx, tgy, tgz) = dispatch_shapes[i];
                let prev_pid = if i > 0 { labels[i - 1] } else { 0 };
                let switch = if i > 0 && pid != prev_pid {
                    "*SW*"
                } else {
                    "    "
                };
                eprintln!(
                    "      [{:>3}] {} {:?}  pipe=0x{:016x}  tg=({:>3},{:>3},{:>2})  dt={:>10.3} µs",
                    i, switch, kid, pid, tgx, tgy, tgz, dt_us
                );
            }
        }
    }
}

/// Probe MTL4 availability once at pool construction. Logs the
/// result at info level so cold-start traces show whether the
/// upcoming Phase A side-by-side path is reachable on this host.
///
/// `newMTL4CommandQueue()` returns `Some` iff the host runs macOS 15+
/// on Apple Family 7+ silicon. The queue is dropped immediately — this
/// is a one-shot capability probe; the production queue is built lazily
/// by `ensure_mtl4` on first use.
fn probe_mtl4_availability(device: &Device) {
    let available = device.newMTL4CommandQueue().is_some();
    tracing::info!(target: "ferrite-metal", available, "mtl4 capability probe");
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
        bucket_tapes: Arc<[LoweredMetalTape]>,
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

        // Phase A.1 MTL4 probe (see `FERRITE_METAL_MTL4_MIGRATION.md`).
        // Side-effect-free: tries `device.newMTL4CommandQueue()`, logs
        // availability, drops the queue. Result is recomputed cheaply
        // when the side-by-side path (A.2/A.3) consults
        // `FERRITE_METAL_MTL4` — kept out of the pool struct until the
        // hot path actually uses it.
        probe_mtl4_availability(&device);

        let pool = Self {
            device,
            allocator,
            pipelines,
            bucket_tapes,
            arena_layout: Arc::new(arena_layout),
            runtime_factory,
            max_workers,
            residency_attached: std::sync::atomic::AtomicBool::new(false),
            mtl4: Mutex::new(None),
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

    /// Build the pool from a flat `&[MetalBucketSpec]` plus the
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
        bucket_specs: &[MetalBucketSpec],
        runtime_factory: RuntimeFactory,
        max_workers: usize,
    ) -> Result<Self, PoolBuildError> {
        if bucket_specs.is_empty() {
            return Err(PoolBuildError::NoBuckets);
        }

        // Worker arena is sized for the largest activation across every
        // bucket, AND for the largest colored slot count across buckets.
        // Slot counts can differ per bucket: a bucket whose solver picked
        // a fusion that needs extra scratch — e.g. the synth pre-attn /
        // mlp-pre-down kernels, which write the updated residual to a
        // distinct `residual_out` slot instead of in place to avoid a
        // cross-threadgroup race — carries more colored slots than a
        // bucket that didn't. A bucket's tape only ever references slots
        // in `0..its own num_arena_slots`, so an arena sized to the max
        // serves every bucket; smaller buckets simply leave the tail
        // slots resident and idle. `arena_bytes` is elementwise-maxed
        // over whatever slots each spec defines.
        let num_slots = bucket_specs
            .iter()
            .map(|s| s.num_arena_slots as usize)
            .max()
            .unwrap_or(0);
        let mut arena_layout: ArenaLayout = vec![0u64; num_slots];
        for spec in bucket_specs {
            for (slot, &bytes) in spec.arena_bytes.iter().enumerate() {
                if bytes > arena_layout[slot] {
                    arena_layout[slot] = bytes;
                }
            }
        }

        let mut cache = SpecializedPipelineCache::with_standard_shaders((*device).clone())
            .map_err(|e| PoolBuildError::PipelineCacheBuild(format!("{e:?}")))?;
        // Compiler-driven synthesis (METAL_KITTENS_SYNTHESIS_PLAN.md):
        // each Metal arch exposes its macro-generated synthesized
        // kernel metallibs via `W::synthesized_kernel_metallibs()`,
        // loaded via `newLibraryWithData`.
        for (name, bytes) in W::synthesized_kernel_metallibs() {
            cache.register_metallib_library(name, bytes).map_err(|e| {
                PoolBuildError::PipelineCacheBuild(format!("synthesized kernel `{name}`: {e:?}"))
            })?;
        }
        let pipelines = Arc::new(SpecializedPipelines::new(Arc::new(cache)));

        // Detect this device's target profile so the lowering pass
        // can run cost-driven kernel-variant selection from the
        // sweep CSV. Falls through to `None` (heuristic fallback) if
        // we're on an uncalibrated chip — `detect_device` returns
        // the populated profile for M4 + M1 Max and an empty-cost
        // table for everything else.
        let target_profile = ferrite_metal_kernels::detect_device().map(|d| d.profile);
        let mut tapes: Vec<LoweredMetalTape> = Vec::with_capacity(bucket_specs.len());
        for spec in bucket_specs {
            let tape = lower_pair::<W>(
                spec.backbone,
                spec.lm_head,
                spec.backbone_barriers,
                spec.lm_head_barriers,
                spec.bucket_m,
                spec.num_arena_slots,
                spec.backbone_tape_index,
                spec.lm_head_tape_index,
                target_profile.as_ref(),
            )
            .map_err(|e| PoolBuildError::BucketLower {
                bucket_m: spec.bucket_m,
                error: format!("{e}"),
            })?;
            tapes.push(tape);
        }
        let bucket_tapes: Arc<[LoweredMetalTape]> = Arc::from(tapes);

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
    ///
    /// **Safe-bucket floor:** the fused MLP / MLX-steel kernels at
    /// `bucket_m >= 2` assume the steel BM=32 tile size; arenas sized
    /// for `bucket_m < SAFE_MULTI_ROW_BUCKET_M` can be overrun by the
    /// kernel writing beyond the slot. Until the small-bucket kernel
    /// path is hardened, we refuse to select buckets in the
    /// `(1, SAFE_MULTI_ROW_BUCKET_M)` range — `num_tokens=2..7` rounds
    /// up to the safe-floor bucket (typically 8), at the cost of
    /// padded compute for small batches.
    pub fn pick_bucket(&self, num_tokens: u32) -> Result<usize, ForwardError> {
        const SAFE_MULTI_ROW_BUCKET_M: u32 = 8;
        if num_tokens == 0 {
            return Err(ForwardError::ZeroTokens);
        }
        let effective_min = if num_tokens == 1 {
            1
        } else {
            num_tokens.max(SAFE_MULTI_ROW_BUCKET_M)
        };
        let mut best: Option<(usize, u32)> = None;
        let mut max_bucket: u32 = 0;
        for (i, tape) in self.bucket_tapes.iter().enumerate() {
            if tape.bucket_m > max_bucket {
                max_bucket = tape.bucket_m;
            }
            if tape.bucket_m >= effective_min {
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

    /// Lazily build the pool's MTL4 surface (queue + allocator +
    /// shared event). Panics if MTL4 is unavailable on this host
    /// (requires macOS 15+ on Apple Family 7+).
    fn ensure_mtl4(&self) {
        let mut slot = self.mtl4.lock().expect("mtl4 mutex");
        if slot.is_some() {
            return;
        }
        let queue = self.device.newMTL4CommandQueue().expect(
            "device.newMTL4CommandQueue() returned nil \
             — host does not support MTL4 (requires macOS 15+ on Apple Family 7+)",
        );
        let allocator = self
            .device
            .newCommandAllocator()
            .expect("device.newCommandAllocator() returned nil");
        let shared_event = self
            .device
            .newSharedEvent()
            .expect("device.newSharedEvent() returned nil");
        *slot = Some(Mtl4Pool {
            queue,
            allocator,
            shared_event,
            signal_counter: 0,
        });
    }

    /// MTL4 forward dispatch + optional encoder-tail hook.
    ///
    /// When `tail = Some(f)`, after the worker has encoded the bucket's
    /// dispatches and BEFORE the encoder is ended, the pool calls
    /// `f(&encoder)` so the caller can append additional dispatches
    /// (e.g. argmax) onto the same compute encoder. Forward + tail
    /// share one CB, one commit, and one host wait — no separate
    /// queue or shared event needed.
    fn run_bucket_mtl4_with_tail<F>(
        &self,
        worker: &MetalWorker<W>,
        bucket_idx: usize,
        num_tokens: usize,
        num_seqs: u32,
        has_spec_tokens: bool,
        tail: Option<F>,
    ) -> Result<(), ForwardError>
    where
        F: FnOnce(
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
            &MetalWorker<W>,
            usize,
        ) -> Result<(), ForwardError>,
    {
        use objc2::runtime::AnyObject;
        use std::ptr::NonNull;
        self.ensure_mtl4();
        let trace = std::env::var_os("FERRITE_METAL_TRACE").is_some();
        let timing_enabled = std::env::var_os("FERRITE_METAL_DISPATCH_TIMING").is_some();
        let timing_state = if timing_enabled {
            let n_dispatches = worker.count_dispatches(bucket_idx);
            DispatchTimingState::new(&self.device, n_dispatches + 1)
        } else {
            None
        };
        let t_pre = std::time::Instant::now();
        let cb = self
            .device
            .newCommandBuffer()
            .expect("newCommandBuffer returned nil");
        let (signal_value, queue_clone, event_clone) = {
            let mut slot = self.mtl4.lock().expect("mtl4 mutex");
            let mtl4 = slot.as_mut().expect("ensure_mtl4 succeeded");
            cb.beginCommandBufferWithAllocator(&mtl4.allocator);
            // Residency: MTL4 cmdbufs declare per-cmdbuf rather than
            // Residency: MTL4 cmdbufs declare per-cmdbuf.
            // Reuse the same set as the pool (weights + arenas + KV cache).
            let cb_ptr: *mut AnyObject =
                ::objc2::rc::Retained::as_ptr(&cb) as *const AnyObject as *mut AnyObject;
            unsafe {
                self.allocator
                    .residency()
                    .attach_to_mtl4_command_buffer(cb_ptr);
            }
            let enc = cb
                .computeCommandEncoder()
                .expect("MTL4 computeCommandEncoder returned nil");
            if let Some(ts) = timing_state.as_ref() {
                worker
                    .run_bucket_mtl4_with_timing(
                        bucket_idx,
                        num_tokens as u32,
                        num_seqs,
                        has_spec_tokens,
                        &enc,
                        ts,
                    )
                    .map_err(ForwardError::Worker)?;
            } else {
                worker
                    .run_bucket_mtl4(
                        bucket_idx,
                        num_tokens as u32,
                        num_seqs,
                        has_spec_tokens,
                        &enc,
                    )
                    .map_err(ForwardError::Worker)?;
            }
            // Caller-supplied encoder-tail hook (e.g. argmax dispatch)
            // runs on the SAME MTL4 compute encoder as the forward —
            // forward + tail share one CB, one commit, one host wait.
            if let Some(t) = tail {
                t(&enc, worker, bucket_idx)?;
            }
            enc.endEncoding();
            cb.endCommandBuffer();
            mtl4.signal_counter = mtl4.signal_counter.checked_add(1).expect("event overflow");
            let val = mtl4.signal_counter;
            let qc = mtl4.queue.clone();
            let ec = mtl4.shared_event.clone();
            // Drop the lock before the host-side wait so a concurrent
            // pool consumer can probe `ensure_mtl4` while we wait.
            (val, qc, ec)
        };
        let encoded = t_pre.elapsed();
        let cb_protocol: &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4CommandBuffer> =
            &cb;
        let cb_nn = NonNull::from(cb_protocol);
        let mut cb_array = [cb_nn];
        unsafe {
            queue_clone.commit_count(NonNull::from(&mut cb_array[0]), 1);
        }
        // Signal AFTER the cmdbuf so the wait fires only once GPU work
        // is fully drained.
        queue_clone.signalEvent_value(
            ::objc2::runtime::ProtocolObject::from_ref(&*event_clone),
            signal_value,
        );
        let committed = t_pre.elapsed();
        // 60s timeout — same order of magnitude as the longest single
        // bucket we'd ever expect; any wait approaching this is a
        // hang and we'd rather panic than spin forever.
        let ok = event_clone.waitUntilSignaledValue_timeoutMS(signal_value, 60_000);
        if !ok {
            return Err(ForwardError::ExecutionFailed(MTLCommandBufferStatus::Error));
        }
        // Reset the allocator now that the GPU is done. Holds the
        // mutex briefly.
        {
            let mut slot = self.mtl4.lock().expect("mtl4 mutex");
            if let Some(mtl4) = slot.as_mut() {
                mtl4.allocator.reset();
            }
        }
        let waited = t_pre.elapsed();
        if trace {
            eprintln!(
                "[forward bucket={} num_tokens={} mtl4] encode={:?} commit={:?} wait={:?}",
                bucket_idx,
                num_tokens,
                encoded,
                committed - encoded,
                waited - committed,
            );
        }
        if let Some(ts) = timing_state {
            ts.resolve_and_print(bucket_idx, num_tokens, &self.device);
        }
        Ok(())
    }

    /// Phase 6 chain-driver primitive. Opens ONE MTL4 command buffer
    /// on the pool's internal MTL4 queue and invokes `body` with the
    /// checked-out worker, its `RuntimeBindings`, and the live compute
    /// encoder. The body drives N forward dispatches (typically the
    /// K-step draft chain) plus any caller-supplied dispatches
    /// (e.g. `argmax_dual_write`, `chain_advance`) onto the same
    /// encoder.
    ///
    /// The pool owns:
    ///   * worker checkout / checkin (via the `WorkerGuard` drop),
    ///   * iter-0 input upload (via `write_runtime_inputs`),
    ///   * residency attach (idempotent),
    ///   * MTL4 CB begin/end,
    ///   * commit, signal, host wait,
    ///   * allocator reset.
    ///
    /// One CB ⇒ one commit ⇒ one host wait for the entire chain.
    /// `queue` here is the MTL3 `CommandQueue` used for residency
    /// attach (same lazy-attach pattern as `forward_with_tail`); the
    /// commit itself happens on the pool's internal MTL4 queue.
    pub fn with_chain_encoder<F, R>(
        &self,
        weights: &W,
        inputs: &ForwardInputs<'_>,
        queue: &CommandQueue,
        body: F,
    ) -> Result<R, ForwardError>
    where
        F: FnOnce(
            &MetalWorker<W>,
            &RuntimeBindings,
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
        ) -> Result<R, ForwardError>,
    {
        use objc2::runtime::AnyObject;
        use std::ptr::NonNull;

        self.ensure_mtl4();
        let trace = std::env::var_os("FERRITE_METAL_TRACE").is_some();

        if !self
            .residency_attached
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            self.allocator.residency().commit();
            self.allocator.residency().attach_to_queue(queue);
        }

        let guard = self.checkout(weights)?;
        write_runtime_inputs(&guard.runtime, inputs)?;

        let t_pre = std::time::Instant::now();
        let cb = self
            .device
            .newCommandBuffer()
            .expect("newCommandBuffer returned nil");
        let (signal_value, queue_clone, event_clone, body_result) = {
            let mut slot = self.mtl4.lock().expect("mtl4 mutex");
            let mtl4 = slot.as_mut().expect("ensure_mtl4 succeeded");
            cb.beginCommandBufferWithAllocator(&mtl4.allocator);
            let cb_ptr: *mut AnyObject =
                ::objc2::rc::Retained::as_ptr(&cb) as *const AnyObject as *mut AnyObject;
            unsafe {
                self.allocator
                    .residency()
                    .attach_to_mtl4_command_buffer(cb_ptr);
            }
            let enc = cb
                .computeCommandEncoder()
                .expect("MTL4 computeCommandEncoder returned nil");
            // Caller's body encodes the entire chain onto `enc`.
            let body_result = body(&guard.worker, &guard.runtime, &enc);
            enc.endEncoding();
            cb.endCommandBuffer();
            mtl4.signal_counter = mtl4.signal_counter.checked_add(1).expect("event overflow");
            let val = mtl4.signal_counter;
            let qc = mtl4.queue.clone();
            let ec = mtl4.shared_event.clone();
            (val, qc, ec, body_result)
        };
        // Propagate body errors AFTER the encoder/CB have been ended
        // (so allocator state stays consistent) and BEFORE committing
        // any half-encoded work to the GPU.
        let body_result = body_result?;

        let encoded = t_pre.elapsed();
        let cb_protocol: &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4CommandBuffer> =
            &cb;
        let cb_nn = NonNull::from(cb_protocol);
        let mut cb_array = [cb_nn];
        unsafe {
            queue_clone.commit_count(NonNull::from(&mut cb_array[0]), 1);
        }
        queue_clone.signalEvent_value(
            ::objc2::runtime::ProtocolObject::from_ref(&*event_clone),
            signal_value,
        );
        let committed = t_pre.elapsed();
        let ok = event_clone.waitUntilSignaledValue_timeoutMS(signal_value, 60_000);
        if !ok {
            return Err(ForwardError::ExecutionFailed(MTLCommandBufferStatus::Error));
        }
        {
            let mut slot = self.mtl4.lock().expect("mtl4 mutex");
            if let Some(mtl4) = slot.as_mut() {
                mtl4.allocator.reset();
            }
        }
        let waited = t_pre.elapsed();
        if trace {
            eprintln!(
                "[chain encoder mtl4] encode={:?} commit={:?} wait={:?}",
                encoded,
                committed - encoded,
                waited - committed,
            );
        }
        Ok(body_result)
    }

    /// Run one forward step via MTL4.
    ///
    /// Pipeline:
    ///  1. Pick the bucket from `inputs.num_tokens`.
    ///  2. Check out a worker (eagerly grow the pool if below cap;
    ///     block if at cap).
    ///  3. Validate every present input slice against its runtime
    ///     buffer's capacity; copy bytes into the buffer's `contents()`.
    ///  4. Encode the bucket's MTL4 steps, commit, and wait.
    ///  5. Run `with_output(&worker)` so the caller can read arena
    ///     buffers (e.g. logits) before the worker is checked back in.
    ///  6. Drop the guard — the worker returns to the pool.
    ///
    /// All validation runs before any GPU work is submitted.
    pub fn forward<R>(
        &self,
        weights: &W,
        queue: &CommandQueue,
        inputs: &ForwardInputs<'_>,
        with_output: impl FnOnce(&MetalWorker<W>, usize) -> R,
    ) -> Result<R, ForwardError> {
        self.forward_with_tail::<R, fn(
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
            &MetalWorker<W>,
            usize,
        ) -> Result<(), ForwardError>>(weights, queue, inputs, with_output, None)
    }

    /// Same as [`forward`], but takes an optional encoder-tail hook
    /// invoked on the same MTL4 compute encoder as the forward,
    /// AFTER the bucket's dispatches and BEFORE `endEncoding`. Lets
    /// callers append additional dispatches (e.g. argmax sampling)
    /// onto the same CB so the whole step lives in one command
    /// buffer with one commit and one host wait.
    pub fn forward_with_tail<R, F>(
        &self,
        weights: &W,
        queue: &CommandQueue,
        inputs: &ForwardInputs<'_>,
        with_output: impl FnOnce(&MetalWorker<W>, usize) -> R,
        tail: Option<F>,
    ) -> Result<R, ForwardError>
    where
        F: FnOnce(
            &::objc2::runtime::ProtocolObject<dyn ::objc2_metal::MTL4ComputeCommandEncoder>,
            &MetalWorker<W>,
            usize,
        ) -> Result<(), ForwardError>,
    {
        let bucket_idx = self.pick_bucket(inputs.num_tokens)?;
        // num_seqs = number of sequences packed into this forward.
        // Computed once here from the staged `cu_seqlens_q` slice
        // (`batch + 1` entries) and threaded through the dispatch
        // path so the per-sub-dispatch `RuntimeGate` checks can
        // pick the lm_head slice (single-seq) vs the full
        // M=bucket_m fallback (multi-seq). Defaults to 1 when
        // `cu_seqlens_q` is absent — those are the
        // `Instruction::AttentionViaCache` decode buckets that
        // always run a single-token forward, never the slice.
        let num_seqs: u32 = inputs
            .cu_seqlens_q
            .map(|cu| (cu.len().saturating_sub(1)).max(1) as u32)
            .unwrap_or(1);
        let has_spec_tokens = inputs.has_spec_tokens;

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

        // All execution goes through the MTL4 path. (The opt-in MTL3
        // dispatch path was removed — it only ever ran the all-ICB
        // buckets that MTL4 already handles, and its M4-era latency edge
        // no longer applies; see feedback_not_mtl3_vs_mtl4.) Buckets with
        // an MPS f16 GEMM step or a >31-binding kernel have no MTL4 plan
        // (`mtl4_steps == None`) and were never executable on the default
        // path anyway.
        assert!(
            guard.worker.bucket_bakings[bucket_idx].mtl4_steps.is_some(),
            "bucket {} is not MTL4-eligible (contains an MPS f16 GEMM step or \
             exceeds the 31-binding argument-table cap)",
            bucket_idx,
        );
        self.run_bucket_mtl4_with_tail(
            &guard.worker,
            bucket_idx,
            inputs.num_tokens as usize,
            num_seqs,
            has_spec_tokens,
            tail,
        )?;

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
    // Tell the gather kernel (used right before lm_head) what the
    // actual num_tokens of this forward is so it can pick the
    // last-token source row.
    unsafe {
        let ptr = runtime.num_tokens_u32.contents().as_ptr() as *mut u32;
        std::ptr::write(ptr, inputs.num_tokens);
    }
    // Plumb the lm_head sample-row index list. When the caller
    // supplies `last_token_indices` (the common case: every forward
    // that produces a sampled token) the slice trio uses these as
    // the gather sources and scatter destinations; otherwise the
    // count is 0 and the slice is a no-op for this step.
    let num_sample_rows = inputs
        .last_token_indices
        .map(|s| s.len() as u32)
        .unwrap_or(0);
    unsafe {
        let ptr = runtime.num_sample_rows_u32.contents().as_ptr() as *mut u32;
        std::ptr::write(ptr, num_sample_rows);
    }
    if let Some(s) = inputs.last_token_indices {
        write_slice("last_token_indices", &runtime.sample_indices, s)?;
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
            num_tokens_u32: alloc(device, 4),
            num_sample_rows_u32: alloc(device, 4),
            sample_indices: alloc(device, 16),
        }
    }

    /// Single-bucket synthetic tape: one RmsNorm command. Enough to
    /// verify the worker bakes; the kernel itself isn't fired in
    /// 5.D tests (5.E hooks `run_bucket` to a real cmdbuf).
    fn synthetic_tape(bucket_m: u32) -> LoweredMetalTape<TestWeights> {
        let cmd = LoweredCommand {
            kernel: KernelId::RmsNorm,
            library: "rmsnorm",
            function: "rmsnorm_f16_s_f16_specialized",
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
                m_scaling: None,
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
                    layer: crate::interpreter::metal::ids::LayerId(0),
                    binding_index: 2,
                },
            ],
            gemm_dims: None,
        };
        LoweredMetalTape {
            bucket_m,
            num_arena_slots: 2,
            commands: vec![cmd],
            splitk_scratch_bytes: 0,
            barrier_before: Vec::new(),
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
            num_tokens_u32: alloc(d, 4),
            num_sample_rows_u32: alloc(d, 4),
            sample_indices: alloc(d, max_m_bytes),
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
            has_spec_tokens: false,
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
            has_spec_tokens: false,
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
            has_spec_tokens: false,
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
            has_spec_tokens: false,
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
    /// `Instruction` to be constructible at the test site (the
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
