// SPDX-License-Identifier: Apache-2.0
//! WebGPU device wrapper — initializes `wgpu::Device` + `Queue`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::WgpuError;

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
pub(crate) struct CommandBatcher {
    pending: Vec<PendingOp>,
}

impl CommandBatcher {
    fn new() -> Self {
        Self {
            pending: Vec::with_capacity(256),
        }
    }

    /// Queue a compute dispatch.
    pub(crate) fn push_dispatch(
        &mut self,
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: wgpu::BindGroup,
        workgroups: [u32; 3],
    ) {
        self.pending.push(PendingOp::Dispatch {
            pipeline,
            bind_group,
            workgroups,
        });
    }

    /// Queue a buffer-to-buffer copy.
    pub(crate) fn push_copy(
        &mut self,
        src: Arc<wgpu::Buffer>,
        src_offset: u64,
        dst: Arc<wgpu::Buffer>,
        dst_offset: u64,
        size: u64,
    ) {
        self.pending.push(PendingOp::Copy {
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        });
    }

    /// Flush all accumulated operations to the queue. No-op if empty.
    /// Groups consecutive dispatches into a single compute pass to minimize
    /// Metal encoder barriers.
    pub(crate) fn flush(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.pending.is_empty() {
            return;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

        queue.submit(Some(encoder.finish()));
        self.pending.clear();
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

/// Simple buffer pool keyed by size. Avoids GPU buffer allocation per op.
pub(crate) struct BufferPool {
    /// Available buffers grouped by size (bytes).
    pub(crate) pools: HashMap<u64, Vec<Arc<wgpu::Buffer>>>,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    /// Get a buffer of the given size, or return None to create a new one.
    pub(crate) fn get(&mut self, size: u64) -> Option<Arc<wgpu::Buffer>> {
        self.pools.get_mut(&size).and_then(|v| v.pop())
    }

    /// Return a buffer to the pool for reuse.
    pub(crate) fn put(&mut self, buf: Arc<wgpu::Buffer>) {
        self.pools.entry(buf.size()).or_default().push(buf);
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
            .map_err(|_| WgpuError::NoAdapter)?;

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
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("vllm-wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_buffer_size: adapter_limits.max_buffer_size,
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    max_compute_workgroup_storage_size: adapter_limits
                        .max_compute_workgroup_storage_size,
                    max_compute_invocations_per_workgroup: adapter_limits
                        .max_compute_invocations_per_workgroup,
                    max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|e| WgpuError::DeviceCreation(e.to_string()))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipeline_cache: Arc::new(Mutex::new(PipelineCacheInner {
                entries: HashMap::new(),
                dynamic_entries: HashMap::new(),
            })),
            batcher: Arc::new(Mutex::new(CommandBatcher::new())),
            buffer_pool: Arc::new(Mutex::new(BufferPool::new())),
            params_cache: Arc::new(Mutex::new(ParamsCache::new())),
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

    /// Flush all batched commands to the GPU.
    pub fn flush(&self) {
        self.batcher
            .lock()
            .unwrap()
            .flush(&self.device, &self.queue);
    }
}
