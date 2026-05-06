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

use std::sync::{Arc, Condvar, Mutex};

use ferrite_metal_kernels::metal::Device;

use super::lowered::LoweredMetalTape;
use super::model_meta::MetalModelMeta;
use super::pipelines::SpecializedPipelines;
use super::runtime::RuntimeBindings;
use super::worker::{ArenaLayout, MetalWorker, WorkerError};
use crate::CanonicalParams;

/// Factory closure invoked once per worker creation to produce a
/// fresh [`RuntimeBindings`] sized for the worker's largest bucket.
///
/// The factory is `Arc<dyn Fn>` rather than a generic so the pool
/// can stay non-generic over the closure type — there's exactly one
/// runtime layout per (model, max bucket) and the factory captures
/// it once at pool construction.
pub type RuntimeFactory =
    Arc<dyn Fn(&Device) -> RuntimeBindings + Send + Sync>;

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
pub struct MetalWorkerPool<W: CanonicalParams> {
    device: Arc<Device>,
    model_meta: Arc<dyn MetalModelMeta<W>>,
    pipelines: Arc<SpecializedPipelines>,
    bucket_tapes: Arc<[LoweredMetalTape<W>]>,
    arena_layout: Arc<ArenaLayout>,
    runtime_factory: RuntimeFactory,
    max_workers: usize,
    inner: Mutex<PoolInner<W>>,
    cv: Condvar,
}

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
    pub fn new(
        device: Arc<Device>,
        model_meta: Arc<dyn MetalModelMeta<W>>,
        pipelines: Arc<SpecializedPipelines>,
        bucket_tapes: Arc<[LoweredMetalTape<W>]>,
        arena_layout: ArenaLayout,
        runtime_factory: RuntimeFactory,
        max_workers: usize,
    ) -> Result<Self, WorkerError> {
        assert!(max_workers >= 1, "max_workers must be >= 1");
        let pool = Self {
            device,
            model_meta,
            pipelines,
            bucket_tapes,
            arena_layout: Arc::new(arena_layout),
            runtime_factory,
            max_workers,
            inner: Mutex::new(PoolInner {
                available: Vec::new(),
                total_created: 0,
            }),
            cv: Condvar::new(),
        };
        let first = pool.spawn_worker()?;
        {
            let mut inner = pool.inner.lock().unwrap();
            inner.total_created = 1;
            inner.available.push(first);
        }
        Ok(pool)
    }

    pub fn max_workers(&self) -> usize {
        self.max_workers
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
    pub fn checkout(&self) -> Result<WorkerGuard<'_, W>, WorkerError> {
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
                match self.spawn_worker() {
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
    pub fn try_checkout(&self) -> Option<Result<WorkerGuard<'_, W>, WorkerError>> {
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
            match self.spawn_worker() {
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

    fn spawn_worker(&self) -> Result<PooledWorker<W>, WorkerError> {
        let runtime = (self.runtime_factory)(&self.device);
        let worker = MetalWorker::<W>::new(
            self.device.clone(),
            &self.arena_layout,
            &self.bucket_tapes,
            &self.pipelines,
            &*self.model_meta,
            &runtime,
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::interpreter::metal::lowered::{
        Binding, DispatchShape, KernelId, LoweredCommand, LoweredMetalTape, WeightBundleKind,
        WeightTensor,
    };
    use crate::interpreter::metal::model_meta::BufferRef;
    use crate::interpreter::metal::pipelines::KernelExtras;
    use ferrite_kernels::layers::RmsNorm;
    use ferrite_metal_kernels::metal::{Buffer, MTLResourceOptions};
    use ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Mirror of the `TinyLlamaProbe` in `worker.rs::tests` — pool
    /// tests share the same model shape so the same compiled
    /// pipelines are reused across tests run in one process.
    struct TinyLlamaProbe;
    impl CanonicalParams for TinyLlamaProbe {
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

    fn stub_rmsnorm(_w: &TinyLlamaProbe, _layer: u32) -> &'static RmsNorm {
        unreachable!("test meta resolves by discriminant, not by invoking the WtFn")
    }

    /// Trivial meta — every weight ask returns one shared buffer.
    /// Sufficient for verifying pool flow; numerical correctness is
    /// 5.G's responsibility.
    struct StubMeta {
        rmsnorm_weight: Buffer,
    }

    impl MetalModelMeta<TinyLlamaProbe> for StubMeta {
        fn weight_buffer(
            &self,
            _kind: &WeightBundleKind<TinyLlamaProbe>,
            _layer: u32,
            _which: WeightTensor,
        ) -> BufferRef<'_> {
            BufferRef {
                buffer: &self.rmsnorm_weight,
                offset: 0,
            }
        }

        fn kernel_extras_for(&self, _cmd: &LoweredCommand<TinyLlamaProbe>) -> KernelExtras {
            KernelExtras {
                eps: 1e-5,
                ..KernelExtras::NONE
            }
        }
    }

    fn alloc(device: &Device, bytes: u64) -> Buffer {
        device.new_buffer(bytes.max(1), MTLResourceOptions::StorageModeShared)
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
    fn synthetic_tape(bucket_m: u32) -> LoweredMetalTape<TinyLlamaProbe> {
        let cmd = LoweredCommand {
            kernel: KernelId::RmsNorm,
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
                    kind: WeightBundleKind::RmsNorm(stub_rmsnorm),
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
    fn build_pool(max: usize) -> Option<MetalWorkerPool<TinyLlamaProbe>> {
        let device = ferrite_metal_kernels::detect_device()?;
        let device = Arc::new(device.device.clone());

        let cache = Arc::new(
            SpecializedPipelineCache::with_standard_shaders((*device).clone())
                .expect("compile standard shaders"),
        );
        let pipelines = Arc::new(SpecializedPipelines::new(cache));

        let meta: Arc<dyn MetalModelMeta<TinyLlamaProbe>> = Arc::new(StubMeta {
            rmsnorm_weight: alloc(&device, 4096),
        });

        let tapes: Arc<[_]> = Arc::from(vec![synthetic_tape(1)]);
        let arena_layout: ArenaLayout = vec![4096, 4096];
        let runtime_factory: RuntimeFactory =
            Arc::new(|d| empty_runtime(d, 1));

        Some(
            MetalWorkerPool::<TinyLlamaProbe>::new(
                device,
                meta,
                pipelines,
                tapes,
                arena_layout,
                runtime_factory,
                max,
            )
            .expect("pool builds"),
        )
    }

    #[test]
    fn pool_starts_with_one_worker() {
        let Some(pool) = build_pool(4) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        assert_eq!(pool.max_workers(), 4);
        assert_eq!(pool.current_size(), 1, "first worker eagerly created");
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn checkout_returns_eagerly_created_worker_first() {
        let Some(pool) = build_pool(4) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let _g = pool.checkout().expect("checkout 1");
        assert_eq!(pool.current_size(), 1, "first checkout reuses eager worker");
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn pool_grows_under_demand_up_to_cap() {
        let Some(pool) = build_pool(3) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let g1 = pool.checkout().expect("checkout 1");
        let g2 = pool.checkout().expect("checkout 2");
        let g3 = pool.checkout().expect("checkout 3");
        assert_eq!(pool.current_size(), 3, "grew to cap");
        assert_eq!(pool.available(), 0);
        // try_checkout at cap returns None, not Some(Err).
        assert!(matches!(pool.try_checkout(), None));
        drop(g1);
        drop(g2);
        drop(g3);
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.current_size(), 3, "cap unchanged after checkin");
    }

    #[test]
    fn guard_drop_returns_worker_to_pool() {
        let Some(pool) = build_pool(2) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        {
            let _g = pool.checkout().expect("checkout");
            assert_eq!(pool.available(), 0);
        }
        assert_eq!(pool.available(), 1, "drop returns worker");
        // Subsequent checkout reuses the existing worker rather than
        // growing the pool.
        let _g2 = pool.checkout().expect("checkout 2");
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
        let Some(pool) = build_pool(1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = Arc::new(pool);
        let g1 = pool.checkout().expect("checkout 1");
        assert_eq!(pool.available(), 0);

        let pool_c = pool.clone();
        let started = Arc::new(AtomicBool::new(false));
        let completed = Arc::new(AtomicBool::new(false));
        let started_c = started.clone();
        let completed_c = completed.clone();

        let handle = std::thread::spawn(move || {
            started_c.store(true, Ordering::SeqCst);
            let _g = pool_c.checkout().expect("blocking checkout");
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
        let Some(pool) = build_pool(1) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let _g = pool.checkout().expect("checkout");
        match pool.try_checkout() {
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
        let Some(pool) = build_pool(2) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let pool = Arc::new(pool);
        let pool_a = pool.clone();
        let pool_b = pool.clone();

        let h_a = std::thread::spawn(move || {
            let _g = pool_a.checkout().expect("checkout a");
            std::thread::sleep(Duration::from_millis(10));
        });
        let h_b = std::thread::spawn(move || {
            let _g = pool_b.checkout().expect("checkout b");
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
}
