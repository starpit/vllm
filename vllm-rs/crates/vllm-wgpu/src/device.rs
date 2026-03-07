// SPDX-License-Identifier: Apache-2.0
//! WebGPU device wrapper — initializes `wgpu::Device` + `Queue`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::WgpuError;

/// Try to resolve a future that should already be ready (e.g. after `device.poll(Wait)`).
/// Returns `None` if the future is not yet resolved.
fn resolve_now<F: std::future::Future<Output = Option<wgpu::Error>>>(
    fut: F,
) -> Option<wgpu::Error> {
    use std::pin::pin;
    use std::task::{Context, Poll, Wake, Waker};

    struct NoopWaker;
    impl Wake for NoopWaker {
        fn wake(self: std::sync::Arc<Self>) {}
    }
    let waker: Waker = std::sync::Arc::new(NoopWaker).into();
    let mut cx = Context::from_waker(&waker);
    let mut pinned = pin!(fut);
    match pinned.as_mut().poll(&mut cx) {
        Poll::Ready(result) => result,
        Poll::Pending => None,
    }
}

/// Validation level for GPU operations, inspired by ORT's ValidationMode.
/// Controls whether wgpu error scopes are pushed/popped around dispatches
/// to catch shader validation and out-of-memory errors early.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationMode {
    /// No extra validation (production default). Relies on wgpu's built-in
    /// validation which is always active.
    #[default]
    Disabled,
    /// Push/pop error scopes around each flush to catch validation errors.
    /// Adds ~1 async round-trip per flush. Suitable for development/debugging.
    Basic,
    /// Per-dispatch error scopes. Expensive but pinpoints the exact failing
    /// shader. Only useful for debugging specific kernel issues.
    Full,
}

/// Cached compute pipeline + bind group layout, keyed by shader source.
pub(crate) struct PipelineCacheInner {
    entries: HashMap<&'static str, (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>)>,
    /// Cache for dynamically-generated shaders (e.g. templated with runtime constants).
    dynamic_entries: HashMap<String, (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>)>,
}

/// A pending GPU operation accumulated by the command batcher.
pub(crate) enum PendingOp {
    /// A compute shader dispatch.
    Dispatch {
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: wgpu::BindGroup,
        workgroups: [u32; 3],
    },
    /// A buffer-to-buffer copy.
    Copy {
        src: Arc<wgpu::Buffer>,
        src_offset: u64,
        dst: Arc<wgpu::Buffer>,
        dst_offset: u64,
        size: u64,
    },
}

/// A command batcher that accumulates ops and replays them into minimal
/// compute passes on flush. Consecutive dispatches share a single compute
/// pass (= single MTLComputeCommandEncoder on Metal), eliminating the
/// per-dispatch encoder creation/destruction barrier overhead.
///
/// Auto-flushes after `MAX_PENDING` ops to keep the GPU busy while the
/// CPU encodes the next batch (inspired by ORT's `max_num_pending_dispatches`).
pub(crate) struct CommandBatcher {
    pending: Vec<PendingOp>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// When `Some`, ops are also recorded here for graph capture.
    capture_buf: Option<Vec<CapturedOp>>,
    validation_mode: ValidationMode,
}

/// Maximum number of pending ops before the batcher auto-flushes.
/// 32 is roughly one transformer layer worth of dispatches.
const MAX_PENDING: usize = 32;

impl CommandBatcher {
    fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        validation_mode: ValidationMode,
    ) -> Self {
        Self {
            pending: Vec::with_capacity(256),
            device,
            queue,
            capture_buf: None,
            validation_mode,
        }
    }

    /// Queue a compute dispatch. Auto-flushes if the pending count hits the threshold.
    /// During capture mode, the op goes to the capture buffer instead of pending.
    pub(crate) fn push_dispatch(
        &mut self,
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: wgpu::BindGroup,
        workgroups: [u32; 3],
    ) {
        if let Some(ref mut capture) = self.capture_buf {
            capture.push(CapturedOp::Dispatch {
                pipeline,
                bind_group,
                workgroups,
            });
        } else {
            self.pending.push(PendingOp::Dispatch {
                pipeline,
                bind_group,
                workgroups,
            });
            if self.pending.len() >= MAX_PENDING {
                self.flush_inner();
            }
        }
    }

    /// Queue a buffer-to-buffer copy. Auto-flushes if the pending count hits the threshold.
    /// During capture mode, the op goes to the capture buffer instead of pending.
    pub(crate) fn push_copy(
        &mut self,
        src: Arc<wgpu::Buffer>,
        src_offset: u64,
        dst: Arc<wgpu::Buffer>,
        dst_offset: u64,
        size: u64,
    ) {
        if let Some(ref mut capture) = self.capture_buf {
            capture.push(CapturedOp::Copy {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            });
        } else {
            self.pending.push(PendingOp::Copy {
                src,
                src_offset,
                dst,
                dst_offset,
                size,
            });
            if self.pending.len() >= MAX_PENDING {
                self.flush_inner();
            }
        }
    }

    /// Flush using externally-provided device/queue references.
    /// Kept for backward compatibility with `WgpuDevice::flush()`.
    pub(crate) fn flush(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        self.flush_inner();
    }

    /// Internal flush — submits all pending ops to the GPU.
    /// Wraps in error scopes when validation mode is enabled.
    fn flush_inner(&mut self) {
        if self.pending.is_empty() {
            return;
        }

        if self.validation_mode != ValidationMode::Disabled {
            self.device.push_error_scope(wgpu::ErrorFilter::Validation);
            self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batched_encoder"),
            });

        let mut i = 0;
        while i < self.pending.len() {
            match &self.pending[i] {
                PendingOp::Dispatch { .. } => {
                    // Open a single compute pass for all consecutive dispatches
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    while i < self.pending.len() {
                        match &self.pending[i] {
                            PendingOp::Dispatch {
                                pipeline,
                                bind_group,
                                workgroups,
                            } => {
                                pass.set_pipeline(pipeline);
                                pass.set_bind_group(0, bind_group, &[]);
                                pass.dispatch_workgroups(
                                    workgroups[0],
                                    workgroups[1],
                                    workgroups[2],
                                );
                                i += 1;
                            }
                            PendingOp::Copy { .. } => break,
                        }
                    }
                    // pass dropped here → ends the compute pass
                }
                PendingOp::Copy {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    encoder.copy_buffer_to_buffer(src, *src_offset, dst, *dst_offset, *size);
                    i += 1;
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        self.pending.clear();

        if self.validation_mode != ValidationMode::Disabled {
            // Pop error scopes (LIFO — OOM first, then validation).
            // After poll(Wait), the futures are immediately resolvable.
            let oom_future = self.device.pop_error_scope();
            let val_future = self.device.pop_error_scope();
            self.device.poll(wgpu::Maintain::Wait);

            if let Some(err) = resolve_now(oom_future) {
                tracing::error!("WebGPU OOM error during flush: {err}");
            }
            if let Some(err) = resolve_now(val_future) {
                tracing::error!("WebGPU validation error during flush: {err}");
            }
        }
    }

    /// Number of currently pending operations.
    #[allow(dead_code)]
    pub(crate) fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

/// Cache for small uniform (params) buffers, keyed by content.
/// Avoids re-creating identical 16/32-byte buffers every dispatch.
pub(crate) struct ParamsCache {
    pub(crate) cache4: HashMap<[u32; 4], Arc<wgpu::Buffer>>,
    pub(crate) cache8: HashMap<[u32; 8], Arc<wgpu::Buffer>>,
}

impl ParamsCache {
    fn new() -> Self {
        Self {
            cache4: HashMap::new(),
            cache8: HashMap::new(),
        }
    }
}

/// Bucket size classes for the buffer pool. Requests are rounded up to the
/// next bucket boundary, so a 260-byte request reuses a 512-byte buffer.
/// Mirrors the approach used by ONNX Runtime's `BucketCacheManager`.
const BUCKET_SIZES: &[u64] = &[
    64,
    128,
    256,
    512,
    2 * 1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
];

/// Maximum number of cached buffers per bucket. Small buckets can hold more
/// since the memory overhead is low; large buckets are capped tightly.
fn max_buffers_for_bucket(bucket_size: u64) -> usize {
    if bucket_size <= 4 * 1024 {
        200
    } else if bucket_size <= 256 * 1024 {
        50
    } else {
        10
    }
}

/// Round `size` up to the next bucket boundary. Returns `None` if `size`
/// exceeds the largest bucket (oversized buffers are not pooled).
fn bucket_for_size(size: u64) -> Option<u64> {
    BUCKET_SIZES.iter().copied().find(|&b| b >= size)
}

/// Bucket-based buffer pool. Buffers are grouped by size class (bucket) and
/// reused across requests of different exact sizes within the same bucket.
/// Avoids GPU driver round-trips from frequent buffer creation/destruction.
pub(crate) struct BufferPool {
    /// Per-bucket free lists. Key = bucket size, value = pooled buffers.
    pub(crate) buckets: HashMap<u64, Vec<Arc<wgpu::Buffer>>>,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
        }
    }

    /// Get a pooled buffer that can hold at least `size` bytes, or `None` if
    /// no suitable buffer is available (caller should allocate at bucket size).
    pub(crate) fn get(&mut self, size: u64) -> Option<Arc<wgpu::Buffer>> {
        let bucket = bucket_for_size(size)?;
        self.buckets.get_mut(&bucket).and_then(|v| v.pop())
    }

    /// Return a buffer to the pool. The buffer is placed in the bucket
    /// matching its actual size. If the bucket is full or the buffer is
    /// oversized, the buffer is silently dropped.
    pub(crate) fn put(&mut self, buf: Arc<wgpu::Buffer>) {
        let actual = buf.size();
        let Some(bucket) = bucket_for_size(actual) else {
            return; // oversized — don't pool
        };
        let list = self.buckets.entry(bucket).or_default();
        if list.len() < max_buffers_for_bucket(bucket) {
            list.push(buf);
        }
        // else: bucket full, drop the buffer
    }

    /// The bucket size that a given request would use, for callers that need
    /// to allocate a new buffer at the right size class.
    pub(crate) fn alloc_size(requested: u64) -> u64 {
        bucket_for_size(requested).unwrap_or(requested)
    }
}

// ---------------------------------------------------------------------------
// Command capture / replay (wgpu analog of CUDA graph capture)
// ---------------------------------------------------------------------------

/// A single recorded GPU operation for replay.
pub(crate) enum CapturedOp {
    /// A compute dispatch with a frozen bind group.
    Dispatch {
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: wgpu::BindGroup,
        workgroups: [u32; 3],
    },
    /// A buffer-to-buffer copy.
    Copy {
        src: Arc<wgpu::Buffer>,
        src_offset: u64,
        dst: Arc<wgpu::Buffer>,
        dst_offset: u64,
        size: u64,
    },
}

/// A captured command sequence that can be replayed without re-encoding.
/// Analogous to ORT's `CapturedCommandInfo` list or CUDA graph capture.
///
/// **Important**: The captured bind groups hold internal references to the
/// GPU buffers they were created with. As long as the `GraphCapture` is alive,
/// those buffers remain valid. On replay, the GPU executes the same dispatches
/// on the same buffers. Input buffers (e.g. token_id, position) should be
/// updated via `queue.write_buffer()` before calling `replay()`.
///
/// **Limitation**: Ops whose uniform params change per step (seq_len, position)
/// need dedicated mutable param buffers to support replay. Currently, step-varying
/// params are baked into cached ParamsCache entries, so full decode-loop replay
/// requires switching those ops to writable uniform buffers.
pub struct GraphCapture {
    ops: Vec<CapturedOp>,
}

impl GraphCapture {
    /// Replay the captured commands by re-encoding and submitting them.
    /// Caller should have updated any input buffers via `queue.write_buffer()`
    /// before calling this.
    pub fn replay(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.ops.is_empty() {
            return;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("graph_replay_encoder"),
        });

        let mut i = 0;
        while i < self.ops.len() {
            match &self.ops[i] {
                CapturedOp::Dispatch { .. } => {
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    while i < self.ops.len() {
                        match &self.ops[i] {
                            CapturedOp::Dispatch {
                                pipeline,
                                bind_group,
                                workgroups,
                            } => {
                                pass.set_pipeline(pipeline);
                                pass.set_bind_group(0, bind_group, &[]);
                                pass.dispatch_workgroups(
                                    workgroups[0],
                                    workgroups[1],
                                    workgroups[2],
                                );
                                i += 1;
                            }
                            CapturedOp::Copy { .. } => break,
                        }
                    }
                }
                CapturedOp::Copy {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    encoder.copy_buffer_to_buffer(src, *src_offset, dst, *dst_offset, *size);
                    i += 1;
                }
            }
        }

        queue.submit(Some(encoder.finish()));
    }

    /// Number of captured operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the capture is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl CommandBatcher {
    /// Begin capture mode. All subsequent `push_dispatch`/`push_copy` calls
    /// will be recorded into an internal capture buffer in addition to the
    /// normal pending list. Call `end_capture()` to finalize.
    pub(crate) fn begin_capture(&mut self) {
        self.capture_buf = Some(Vec::with_capacity(256));
    }

    /// End capture mode and return the recorded commands as a `GraphCapture`.
    /// Returns `None` if capture was not active.
    pub(crate) fn end_capture(&mut self) -> Option<GraphCapture> {
        self.capture_buf.take().map(|ops| GraphCapture { ops })
    }

    /// Whether capture mode is currently active.
    #[allow(dead_code)]
    pub(crate) fn is_capturing(&self) -> bool {
        self.capture_buf.is_some()
    }
}

/// A WebGPU device handle (shareable via `Arc`).
#[derive(Clone)]
pub struct WgpuDevice {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) pipeline_cache: Arc<Mutex<PipelineCacheInner>>,
    pub(crate) batcher: Arc<Mutex<CommandBatcher>>,
    pub(crate) buffer_pool: Arc<Mutex<BufferPool>>,
    pub(crate) params_cache: Arc<Mutex<ParamsCache>>,
    /// Maximum size (bytes) a single storage buffer binding can have.
    /// Typically 128 MB on desktop GPUs, lower on mobile.
    pub(crate) max_storage_buffer_binding_size: u64,
    /// Validation level for GPU error scope management.
    pub(crate) validation_mode: ValidationMode,
}

impl WgpuDevice {
    /// Create a new WebGPU device (async — works in both native and WASM).
    pub async fn new() -> Result<Self, WgpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(WgpuError::NoAdapter)?;

        let adapter_limits = adapter.limits();
        tracing::info!(
            "WebGPU adapter: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );
        tracing::info!(
            "  workgroup storage: {} bytes, max invocations: {}",
            adapter_limits.max_compute_workgroup_storage_size,
            adapter_limits.max_compute_invocations_per_workgroup,
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("vllm-wgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_buffer_size: adapter_limits.max_buffer_size,
                        max_storage_buffer_binding_size: adapter_limits
                            .max_storage_buffer_binding_size,
                        max_compute_workgroup_storage_size: adapter_limits
                            .max_compute_workgroup_storage_size,
                        max_compute_invocations_per_workgroup: adapter_limits
                            .max_compute_invocations_per_workgroup,
                        max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| WgpuError::DeviceCreation(e.to_string()))?;

        let max_storage = adapter_limits.max_storage_buffer_binding_size as u64;
        tracing::info!(
            "  max storage buffer binding: {} MB",
            max_storage / (1024 * 1024)
        );

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Default to Basic validation in debug builds for early error detection.
        let validation_mode = if cfg!(debug_assertions) {
            ValidationMode::Basic
        } else {
            ValidationMode::Disabled
        };

        Ok(Self {
            pipeline_cache: Arc::new(Mutex::new(PipelineCacheInner {
                entries: HashMap::new(),
                dynamic_entries: HashMap::new(),
            })),
            batcher: Arc::new(Mutex::new(CommandBatcher::new(
                device.clone(),
                queue.clone(),
                validation_mode,
            ))),
            buffer_pool: Arc::new(Mutex::new(BufferPool::new())),
            params_cache: Arc::new(Mutex::new(ParamsCache::new())),
            max_storage_buffer_binding_size: max_storage,
            validation_mode,
            device,
            queue,
        })
    }

    /// Get the underlying `wgpu::Device`.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get the underlying `wgpu::Queue`.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get or create a cached compute pipeline + bind group layout.
    /// `shader_src` must be a `&'static str` (e.g. from `include_str!`).
    pub(crate) fn get_pipeline(
        &self,
        shader_src: &'static str,
    ) -> (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>) {
        let mut cache = self.pipeline_cache.lock().unwrap();
        if let Some(entry) = cache.entries.get(shader_src) {
            return entry.clone();
        }

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let layout = Arc::new(pipeline.get_bind_group_layout(0));
        cache
            .entries
            .insert(shader_src, (pipeline.clone(), layout.clone()));
        (pipeline, layout)
    }

    /// Get or create a cached compute pipeline from a dynamically-generated shader string.
    pub(crate) fn get_pipeline_owned(
        &self,
        shader_src: &str,
    ) -> (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>) {
        let mut cache = self.pipeline_cache.lock().unwrap();
        if let Some(entry) = cache.dynamic_entries.get(shader_src) {
            return entry.clone();
        }

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let layout = Arc::new(pipeline.get_bind_group_layout(0));
        cache
            .dynamic_entries
            .insert(shader_src.to_string(), (pipeline.clone(), layout.clone()));
        (pipeline, layout)
    }

    /// Maximum size (bytes) a single storage buffer binding can have.
    pub fn max_binding_size(&self) -> u64 {
        self.max_storage_buffer_binding_size
    }

    /// Check whether a buffer size exceeds the storage binding limit.
    /// Returns `true` if the buffer would need segmentation.
    pub fn exceeds_binding_limit(&self, size_bytes: u64) -> bool {
        size_bytes > self.max_storage_buffer_binding_size
    }

    /// Set the validation mode. Also updates the batcher's mode so
    /// auto-flushes get the same validation behavior.
    pub fn set_validation_mode(&mut self, mode: ValidationMode) {
        self.validation_mode = mode;
        self.batcher.lock().unwrap().validation_mode = mode;
    }

    /// Flush all batched commands to the GPU.
    /// Validation error scopes are handled inside the batcher's `flush_inner`,
    /// so both explicit flushes and auto-flushes get the same coverage.
    pub fn flush(&self) {
        self.batcher
            .lock()
            .unwrap()
            .flush(&self.device, &self.queue);
    }
}
