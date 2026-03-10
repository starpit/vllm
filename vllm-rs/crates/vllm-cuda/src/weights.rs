// SPDX-License-Identifier: Apache-2.0
//! Safetensors weight loading — streaming from CPU to GPU.
//!
//! Matches Python vLLM's approach: weights stay on CPU (mmap'd) and are copied
//! to GPU one at a time via `take()`. This ensures GPU memory usage during model
//! loading is minimal — only the final model weights live on GPU, with no
//! intermediate shard-level GPU buffers.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, bail};
use cudarc::driver::sys::CUstream;

use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;

// ---------------------------------------------------------------------------
// DType conversion
// ---------------------------------------------------------------------------

/// Map safetensors dtype string to our DType.
fn safetensors_dtype(dtype: safetensors::Dtype) -> Result<DType> {
    match dtype {
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::I64 => Ok(DType::I64),
        safetensors::Dtype::U32 => Ok(DType::U32),
        safetensors::Dtype::I32 => Ok(DType::I32),
        other => bail!("unsupported safetensors dtype: {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// CpuTensorRef — a reference to tensor data in a mmap'd safetensors file
// ---------------------------------------------------------------------------

/// A CPU-side reference to tensor data in a memory-mapped safetensors file.
struct CpuTensorRef {
    /// The mmap that backs this tensor.
    mmap: Arc<memmap2::Mmap>,
    /// Byte offset within the mmap where tensor data starts.
    data_offset: usize,
    /// Size of tensor data in bytes.
    size_bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
}

impl CpuTensorRef {
    fn data(&self) -> &[u8] {
        &self.mmap[self.data_offset..self.data_offset + self.size_bytes]
    }
}

// ---------------------------------------------------------------------------
// GpuWeights
// ---------------------------------------------------------------------------

/// Model weights loaded lazily from CPU (mmap) to GPU.
///
/// Matches Python vLLM's streaming weight loading: weights are memory-mapped
/// on CPU and copied to GPU one at a time when requested via `take()`.
/// No shard-level GPU buffers are allocated.
///
/// GPU memory allocated by `take()` is NOT freed on drop — ownership transfers
/// to the caller (model layers). This matches Python where `nn.Parameter` owns
/// the weight tensors, not the loader.
pub struct GpuWeights {
    /// Per-tensor CPU references, keyed by tensor name.
    tensors: HashMap<String, CpuTensorRef>,
    /// Stream used for H2D copies.
    stream: CUstream,
    /// Target dtype for floating-point weights. When set, F32 weights are cast
    /// to this dtype on CPU before H2D copy — matching Python vLLM where model
    /// parameters are initialized with torch_dtype and PyTorch auto-casts during
    /// `param.data.copy_(loaded_weight)`.
    target_dtype: Option<DType>,
    /// Reusable pinned host buffer for dtype casting. Using pinned memory
    /// enables async DMA transfers, matching PyTorch's copy_ behavior.
    /// (ptr, capacity_bytes). Grown as needed, never shrunk.
    cast_pinned: (*mut u8, usize),
}

// Safety: GPU device pointers accessible from any host thread.
unsafe impl Send for GpuWeights {}
unsafe impl Sync for GpuWeights {}

impl GpuWeights {
    /// Load all weights from a model directory (CPU-only — no GPU allocation).
    ///
    /// Handles both single-file (`model.safetensors`) and sharded
    /// (`model.safetensors.index.json`) models.
    pub fn from_dir(dir: impl AsRef<Path>, stream: CUstream) -> Result<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        let single_path = dir.join("model.safetensors");

        if index_path.exists() {
            Self::from_index(&index_path, stream)
        } else if single_path.exists() {
            Self::from_single_file(&single_path, stream)
        } else {
            bail!("No safetensors files found in {}", dir.display());
        }
    }

    /// Load from a single safetensors file (CPU-only).
    pub fn from_single_file(path: impl AsRef<Path>, stream: CUstream) -> Result<Self> {
        let path = path.as_ref();
        let mut gw = Self {
            tensors: HashMap::new(),
            stream,
            target_dtype: None,
            cast_pinned: (std::ptr::null_mut(), 0),
        };
        gw.load_shard(path)?;
        Ok(gw)
    }

    /// Load from a sharded model (index.json) (CPU-only).
    pub fn from_index(index_path: impl AsRef<Path>, stream: CUstream) -> Result<Self> {
        let index_path = index_path.as_ref();
        let dir = index_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("index file has no parent dir"))?;

        let index_data = std::fs::read_to_string(index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_data)?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("missing weight_map in index"))?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_files.sort();
        shard_files.dedup();

        let mut gw = Self {
            tensors: HashMap::new(),
            stream,
            target_dtype: None,
            cast_pinned: (std::ptr::null_mut(), 0),
        };

        for shard_name in &shard_files {
            let shard_path = dir.join(shard_name);
            gw.load_shard(&shard_path)?;
        }

        Ok(gw)
    }

    /// Parse a shard file and store CPU-side tensor references.
    fn load_shard(&mut self, path: &Path) -> Result<()> {
        let file = std::fs::File::open(path)?;
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file) }?);

        // Parse safetensors header to find tensor offsets.
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow::anyhow!("{}: {}", path.display(), e))?;

        let mut count = 0;
        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("{}: {}", name, e))?;
            let dtype = safetensors_dtype(view.dtype())?;
            let data = view.data();
            let size_bytes = data.len();
            let shape: Vec<usize> = view.shape().to_vec();

            // Compute the offset of this tensor's data within the mmap.
            let data_offset = data.as_ptr() as usize - mmap.as_ptr() as usize;

            self.tensors.insert(
                name.to_string(),
                CpuTensorRef {
                    mmap: Arc::clone(&mmap),
                    data_offset,
                    size_bytes,
                    shape,
                    dtype,
                },
            );
            count += 1;
        }

        tracing::info!(
            "Parsed shard {}: {} tensors (CPU mmap, no GPU allocation)",
            path.display(),
            count,
        );

        Ok(())
    }

    /// Ensure the pinned cast buffer has at least `needed` bytes.
    /// Grows by freeing + reallocating (pinned memory can't realloc).
    fn ensure_pinned_buf(&mut self, needed: usize) {
        if needed <= self.cast_pinned.1 {
            return;
        }
        // Free old buffer if any.
        if !self.cast_pinned.0.is_null() {
            unsafe { driver::mem_free_host(self.cast_pinned.0).ok() };
        }
        // Allocate new pinned buffer. Round up to 1MB alignment for reuse.
        let alloc_size = needed.next_power_of_two().max(1 << 20);
        let ptr = unsafe { driver::mem_alloc_host(alloc_size) }
            .expect("failed to allocate pinned host memory for dtype cast");
        self.cast_pinned = (ptr, alloc_size);
    }

    /// If target_dtype is set and the weight needs casting, cast on CPU into
    /// pinned host memory. Returns (data_ptr, size_bytes, effective_dtype).
    ///
    /// Only floating-point weights (F32, BF16, F16) are cast. Integer dtypes
    /// (I32, U32, I64) are left untouched — they're used for indices/metadata.
    fn maybe_cast_cpu(&mut self, cpu_ref: &CpuTensorRef) -> (*const u8, usize, DType) {
        let target = match self.target_dtype {
            Some(t) => t,
            None => return (cpu_ref.data().as_ptr(), cpu_ref.size_bytes, cpu_ref.dtype),
        };

        // Only cast floating-point types.
        let is_float = matches!(cpu_ref.dtype, DType::F32 | DType::F16 | DType::BF16);
        if !is_float || cpu_ref.dtype == target {
            return (cpu_ref.data().as_ptr(), cpu_ref.size_bytes, cpu_ref.dtype);
        }

        let numel = cpu_ref.size_bytes / cpu_ref.dtype.size_bytes();
        let cast_size = numel * target.size_bytes();
        self.ensure_pinned_buf(cast_size);

        let src = cpu_ref.data();
        let dst = self.cast_pinned.0;

        // Dispatch cast. The common case is F32 → BF16/F16.
        match (cpu_ref.dtype, target) {
            (DType::F32, DType::BF16) => {
                let src_f32 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, numel) };
                let dst_u16 = unsafe { std::slice::from_raw_parts_mut(dst as *mut u16, numel) };
                for (s, d) in src_f32.iter().zip(dst_u16.iter_mut()) {
                    *d = half::bf16::from_f32(*s).to_bits();
                }
            }
            (DType::F32, DType::F16) => {
                let src_f32 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const f32, numel) };
                let dst_u16 = unsafe { std::slice::from_raw_parts_mut(dst as *mut u16, numel) };
                for (s, d) in src_f32.iter().zip(dst_u16.iter_mut()) {
                    *d = half::f16::from_f32(*s).to_bits();
                }
            }
            (DType::F16, DType::BF16) => {
                let src_u16 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, numel) };
                let dst_u16 = unsafe { std::slice::from_raw_parts_mut(dst as *mut u16, numel) };
                for (s, d) in src_u16.iter().zip(dst_u16.iter_mut()) {
                    *d = half::bf16::from_f32(half::f16::from_bits(*s).to_f32()).to_bits();
                }
            }
            (DType::BF16, DType::F16) => {
                let src_u16 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, numel) };
                let dst_u16 = unsafe { std::slice::from_raw_parts_mut(dst as *mut u16, numel) };
                for (s, d) in src_u16.iter().zip(dst_u16.iter_mut()) {
                    *d = half::f16::from_f32(half::bf16::from_bits(*s).to_f32()).to_bits();
                }
            }
            (DType::BF16 | DType::F16, DType::F32) => {
                let src_u16 =
                    unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u16, numel) };
                let dst_f32 = unsafe { std::slice::from_raw_parts_mut(dst as *mut f32, numel) };
                if cpu_ref.dtype == DType::BF16 {
                    for (s, d) in src_u16.iter().zip(dst_f32.iter_mut()) {
                        *d = half::bf16::from_bits(*s).to_f32();
                    }
                } else {
                    for (s, d) in src_u16.iter().zip(dst_f32.iter_mut()) {
                        *d = half::f16::from_bits(*s).to_f32();
                    }
                }
            }
            _ => unreachable!("unhandled cast: {:?} → {:?}", cpu_ref.dtype, target),
        }

        tracing::debug!(
            "Cast weight: {:?} → {:?} ({} elements)",
            cpu_ref.dtype,
            target,
            numel,
        );

        (dst as *const u8, cast_size, target)
    }

    /// Set the target dtype for floating-point weight casting.
    ///
    /// When set, floating-point weights (F32, F16, BF16) are cast to the target
    /// dtype on CPU before H2D copy. Integer weights are never cast.
    /// This matches Python vLLM where model parameters are initialized with
    /// `torch_dtype` and PyTorch auto-casts during weight loading.
    pub fn set_target_dtype(&mut self, dtype: DType) {
        self.target_dtype = Some(dtype);
    }

    /// Remove a tensor by name and copy it to GPU. Returns a GPU tensor.
    ///
    /// This is the primary weight loading method — matches Python's streaming
    /// approach where each weight is copied to GPU on demand.
    pub fn take(&mut self, name: &str) -> Result<GpuTensor> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        let (data, size_bytes, dtype) = self.maybe_cast_cpu(&cpu_ref);

        let gpu_ptr = unsafe { driver::mem_alloc(size_bytes)? };

        unsafe {
            driver::memcpy_htod_async(gpu_ptr, data, size_bytes, self.stream)?;
        }

        Ok(unsafe { GpuTensor::new(gpu_ptr, &cpu_ref.shape, dtype) })
    }

    /// Copy a tensor's data directly to an offset within an existing GPU buffer.
    ///
    /// Used for fused weight loading (QKV, gate_up) — pre-allocate the fused
    /// tensor, then copy each component directly from CPU to the right offset.
    pub unsafe fn take_into(
        &mut self,
        name: &str,
        dst: *mut u8,
        stream: CUstream,
    ) -> Result<usize> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        let (data, size_bytes, _dtype) = self.maybe_cast_cpu(&cpu_ref);

        driver::memcpy_htod_async(dst, data, size_bytes, stream)?;

        Ok(size_bytes)
    }

    /// Get the shape and effective dtype of a tensor without loading it to GPU.
    ///
    /// If `target_dtype` is set and the tensor is a floating-point type, the
    /// returned dtype reflects the cast target (matching what `take`/`take_into`
    /// will produce). This ensures callers compute correct byte sizes for
    /// pre-allocated buffers.
    pub fn tensor_info(&self, name: &str) -> Option<(&[usize], DType)> {
        self.tensors.get(name).map(|r| {
            let effective_dtype = match self.target_dtype {
                Some(target)
                    if matches!(r.dtype, DType::F32 | DType::F16 | DType::BF16)
                        && r.dtype != target =>
                {
                    target
                }
                _ => r.dtype,
            };
            (r.shape.as_slice(), effective_dtype)
        })
    }

    /// Get a tensor by name (copies to GPU). For read-only access.
    ///
    /// WARNING: The returned GPU tensor is leaked — caller must arrange cleanup.
    /// Prefer `take()` which is more explicit about ownership transfer.
    pub fn get(&mut self, name: &str) -> Option<GpuTensor> {
        // Remove temporarily to satisfy borrow checker, then re-insert.
        let cpu_ref = self.tensors.remove(name)?;
        let (data, size_bytes, dtype) = self.maybe_cast_cpu(&cpu_ref);

        let gpu_ptr = unsafe { driver::mem_alloc(size_bytes).ok()? };

        unsafe {
            driver::memcpy_htod_async(gpu_ptr, data, size_bytes, self.stream).ok()?;
        }

        let shape = cpu_ref.shape.clone();
        self.tensors.insert(name.to_string(), cpu_ref);

        Some(unsafe { GpuTensor::new(gpu_ptr, &shape, dtype) })
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Number of loaded tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether no tensors are loaded.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Iterator over all tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    /// Strip a prefix from all tensor names (e.g. "model.").
    pub fn strip_prefix(&mut self, prefix: &str) {
        let stripped: HashMap<String, CpuTensorRef> = self
            .tensors
            .drain()
            .filter_map(|(name, tensor)| {
                name.strip_prefix(prefix)
                    .map(|rest| (rest.to_string(), tensor))
            })
            .collect();
        self.tensors = stripped;
    }

    /// Get the stream used for H2D copies.
    pub fn stream(&self) -> CUstream {
        self.stream
    }

    // -----------------------------------------------------------------------
    // Tensor-parallel sharding (CPU-side slice → GPU)
    // -----------------------------------------------------------------------

    /// Remove a tensor by name, slice it along `dim` for tensor parallelism,
    /// and copy only the shard to GPU. Returns a GPU tensor of the shard.
    ///
    /// For dim=0 sharding (column parallel): contiguous slice of rows.
    /// For dim=1 sharding (row parallel): strided extraction of columns,
    /// copied row-by-row into a contiguous pinned buffer before H2D.
    pub fn take_shard(
        &mut self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<GpuTensor> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        let (data, shard_shape, dtype) = self.shard_cpu_data(&cpu_ref, dim, rank, world_size);

        let size_bytes = shard_shape.iter().product::<usize>() * dtype.size_bytes();
        let gpu_ptr = unsafe { driver::mem_alloc(size_bytes)? };
        unsafe {
            driver::memcpy_htod_async(gpu_ptr, data, size_bytes, self.stream)?;
        }

        Ok(unsafe { GpuTensor::new(gpu_ptr, &shard_shape, dtype) })
    }

    /// Copy a shard of a tensor directly to an offset within an existing GPU buffer.
    ///
    /// Used for fused TP weight loading (e.g. QKV shards concatenated into one buffer).
    /// Returns the number of bytes written.
    pub unsafe fn take_shard_into(
        &mut self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
        dst: *mut u8,
        stream: CUstream,
    ) -> Result<usize> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        let (data, shard_shape, dtype) = self.shard_cpu_data(&cpu_ref, dim, rank, world_size);

        let size_bytes = shard_shape.iter().product::<usize>() * dtype.size_bytes();
        driver::memcpy_htod_async(dst, data, size_bytes, stream)?;

        Ok(size_bytes)
    }

    /// Internal: extract a shard from CPU tensor data. Returns (ptr, shard_shape, dtype).
    ///
    /// For dim=0: returns a pointer into the original data (contiguous slice).
    /// For dim=1: copies strided columns into the pinned cast buffer, returns pointer to that.
    fn shard_cpu_data(
        &mut self,
        cpu_ref: &CpuTensorRef,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> (*const u8, Vec<usize>, DType) {
        assert!(!cpu_ref.shape.is_empty(), "cannot shard scalar");
        assert!(dim < cpu_ref.shape.len(), "dim out of range");
        let full_size = cpu_ref.shape[dim];
        assert!(
            full_size.is_multiple_of(world_size),
            "dim {dim} size {full_size} not divisible by world_size {world_size}"
        );
        let shard_size = full_size / world_size;

        // Apply dtype casting first if needed.
        let (src_data, _src_bytes, dtype) = self.maybe_cast_cpu(cpu_ref);

        let elem_size = dtype.size_bytes();
        let mut shard_shape = cpu_ref.shape.clone();
        shard_shape[dim] = shard_size;

        if dim == 0 {
            // Contiguous slice: rows [rank*shard_size .. (rank+1)*shard_size].
            // Each row has product(shape[1:]) elements.
            let row_elems: usize = cpu_ref.shape[1..].iter().product();
            let row_bytes = row_elems * elem_size;
            let offset = rank * shard_size * row_bytes;
            let data = unsafe { src_data.add(offset) };
            (data, shard_shape, dtype)
        } else if dim == 1 && cpu_ref.shape.len() == 2 {
            // Strided column extraction for 2D tensor [rows, cols].
            // Extract columns [rank*shard_size .. (rank+1)*shard_size] from each row.
            let rows = cpu_ref.shape[0];
            let cols = cpu_ref.shape[1];
            let col_start = rank * shard_size;
            let shard_row_bytes = shard_size * elem_size;
            let needed = rows * shard_row_bytes;
            self.ensure_pinned_buf(needed);

            let dst = self.cast_pinned.0;
            for r in 0..rows {
                let src_offset = (r * cols + col_start) * elem_size;
                let dst_offset = r * shard_row_bytes;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_data.add(src_offset),
                        dst.add(dst_offset),
                        shard_row_bytes,
                    );
                }
            }
            (dst as *const u8, shard_shape, dtype)
        } else {
            panic!(
                "take_shard: unsupported dim={dim} for {}D tensor",
                cpu_ref.shape.len()
            );
        }
    }

    /// Take a tensor's raw CPU bytes without uploading to GPU.
    /// Returns (data_bytes, shape, dtype).
    pub fn take_cpu(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, DType)> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;
        let data = cpu_ref.data().to_vec();
        Ok((data, cpu_ref.shape, cpu_ref.dtype))
    }
}

impl Drop for GpuWeights {
    fn drop(&mut self) {
        // Free pinned cast buffer if allocated.
        if !self.cast_pinned.0.is_null() {
            unsafe { driver::mem_free_host(self.cast_pinned.0).ok() };
        }
        // GPU memory allocated by take()/take_into() is owned by model layers.
        // CPU mmaps are dropped automatically when Arc<Mmap> refcounts reach zero.
    }
}

// ---------------------------------------------------------------------------
// Quantized weight loading (AWQ/GPTQ → Marlin)
// ---------------------------------------------------------------------------

use crate::alloc::CachingAllocator;
use crate::layers::MarlinLinear;
use crate::quant::{self, QuantConfig};

/// Load a single quantized linear layer (AWQ or GPTQ) and repack to Marlin format.
///
/// Loads qweight, scales, qzeros from safetensors, uploads to GPU,
/// runs repack kernels, and applies scale/zero-point permutations.
///
/// # Arguments
/// * `weights` — mmap'd safetensors
/// * `prefix` — weight name prefix (e.g. "model.layers.0.self_attn.q_proj")
/// * `qconfig` — AWQ or GPTQ config
/// * `workspace` — shared Marlin workspace tensor `[num_sms]` i32
/// * `device_id` — CUDA device ordinal
/// * `alloc` — caching allocator for repack output
#[allow(clippy::too_many_arguments)]
pub fn load_marlin_linear(
    weights: &mut GpuWeights,
    prefix: &str,
    qconfig: &QuantConfig,
    workspace: GpuTensor,
    device_id: i32,
    alloc: &mut CachingAllocator,
) -> Result<MarlinLinear> {
    let stream = weights.stream();

    match qconfig {
        QuantConfig::Awq(cfg) => {
            load_awq_marlin_linear(weights, prefix, cfg, workspace, device_id, alloc, stream)
        }
        QuantConfig::Gptq(cfg) => {
            load_gptq_marlin_linear(weights, prefix, cfg, workspace, device_id, alloc, stream)
        }
        QuantConfig::None => bail!("load_marlin_linear called with QuantConfig::None"),
    }
}

/// Load AWQ quantized linear layer and repack to Marlin format.
#[allow(clippy::too_many_arguments)]
fn load_awq_marlin_linear(
    weights: &mut GpuWeights,
    prefix: &str,
    cfg: &quant::AwqConfig,
    workspace: GpuTensor,
    device_id: i32,
    _alloc: &mut CachingAllocator,
    stream: CUstream,
) -> Result<MarlinLinear> {
    // AWQ qweight: [K, N/8] i32 — packed along output dim
    let qw_name = format!("{prefix}.qweight");
    let scales_name = format!("{prefix}.scales");
    let qzeros_name = format!("{prefix}.qzeros");

    // Get dimensions from qweight shape
    let (qw_shape, _qw_dtype) = weights
        .tensor_info(&qw_name)
        .ok_or_else(|| anyhow::anyhow!("weight not found: {qw_name}"))?;
    let size_k = qw_shape[0];
    let size_n = qw_shape[1] * 8; // 4-bit: 8 values packed per i32
    let group_size = cfg.group_size;
    let num_groups = if group_size > 0 {
        size_k / group_size
    } else {
        1
    };

    // Upload qweight to GPU (raw alloc — will be freed after repack)
    let qweight_gpu = weights.take(&qw_name)?;

    // Repack AWQ → Marlin tiled layout on GPU.
    // Use driver::mem_alloc for the output (NOT caching allocator) because model
    // weights must survive free_leaked_blocks() during profiling.
    let num_u32 = size_k * size_n / 8;
    let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
    let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
    unsafe {
        crate::kernels::awq_repack_into(qweight_gpu, repack_ptr, size_k, size_n, device_id, stream);
        driver::stream_synchronize(stream)?;
        driver::mem_free(qweight_gpu.raw_ptr())?;
    }
    let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

    // Load and permute scales (CPU)
    let (scales_bytes, _scales_shape, scales_dtype) = weights.take_cpu(&scales_name)?;
    let mut scales_u16: Vec<u16> = scales_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    quant::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

    // Upload permuted scales to GPU
    let scales_bytes_permuted: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let scales_nbytes = scales_bytes_permuted.len();
    let scales_ptr = unsafe { driver::mem_alloc(scales_nbytes)? };
    unsafe {
        driver::memcpy_htod_async(
            scales_ptr,
            scales_bytes_permuted.as_ptr(),
            scales_nbytes,
            stream,
        )?;
    }
    let scales_gpu = unsafe { GpuTensor::new(scales_ptr, &[num_groups, size_n], scales_dtype) };

    // Load and convert zero points (CPU)
    let (qzeros_bytes, _qzeros_shape, _qzeros_dtype) = weights.take_cpu(&qzeros_name)?;
    let qzeros_u32: Vec<u32> = qzeros_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let marlin_zp = quant::awq_to_marlin_zero_points(&qzeros_u32, num_groups, size_n);

    // Upload zero points to GPU
    let zp_bytes: Vec<u8> = marlin_zp.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let zp_nbytes = zp_bytes.len();
    let zp_ptr = unsafe { driver::mem_alloc(zp_nbytes)? };
    unsafe { driver::memcpy_htod_async(zp_ptr, zp_bytes.as_ptr(), zp_nbytes, stream)? };
    let zeros_gpu = unsafe { GpuTensor::new(zp_ptr, &[num_groups, size_n / 8], DType::U32) };

    // Load bias if present
    let bias_name = format!("{prefix}.bias");
    let bias_gpu = if weights.contains(&bias_name) {
        Some(weights.take(&bias_name)?)
    } else {
        None
    };

    unsafe { driver::stream_synchronize(stream)? };

    Ok(MarlinLinear {
        qweight: qweight_marlin,
        scales: scales_gpu,
        zeros: Some(zeros_gpu),
        g_idx: None,
        g_idx_sort_indices: None,
        workspace,
        size_k,
        size_n,
        group_size,
        num_groups,
        has_zp: true,
        has_act_order: false,
        b_type_id: 1, // AWQ = uint4
        device_id,
        bias: bias_gpu,
    })
}

/// Load GPTQ quantized linear layer and repack to Marlin format.
#[allow(clippy::too_many_arguments)]
fn load_gptq_marlin_linear(
    weights: &mut GpuWeights,
    prefix: &str,
    cfg: &quant::GptqConfig,
    workspace: GpuTensor,
    device_id: i32,
    _alloc: &mut CachingAllocator,
    stream: CUstream,
) -> Result<MarlinLinear> {
    // GPTQ qweight: [K/8, N] i32 — packed along input dim
    let qw_name = format!("{prefix}.qweight");
    let scales_name = format!("{prefix}.scales");
    let qzeros_name = format!("{prefix}.qzeros");

    let (qw_shape, _qw_dtype) = weights
        .tensor_info(&qw_name)
        .ok_or_else(|| anyhow::anyhow!("weight not found: {qw_name}"))?;
    let qw_shape = qw_shape.to_vec();
    let size_k = qw_shape[0] * 8; // 4-bit: 8 values packed per i32
    let size_n = qw_shape[1];
    let group_size = cfg.group_size;
    let num_groups = if group_size > 0 {
        size_k / group_size
    } else {
        1
    };

    // Upload qweight to GPU (raw alloc — will be freed after repack)
    let qweight_gpu = weights.take(&qw_name)?;

    // Repack GPTQ → Marlin tiled layout on GPU (no act_order for now).
    // Use driver::mem_alloc for the output (NOT caching allocator) because model
    // weights must survive free_leaked_blocks() during profiling.
    let num_u32 = size_k * size_n / 8;
    let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
    let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
    unsafe {
        crate::kernels::gptq_repack_into(
            qweight_gpu,
            None,
            repack_ptr,
            size_k,
            size_n,
            device_id,
            stream,
        );
        // Sync so the repack kernel finishes before we free the source qweight
        driver::stream_synchronize(stream)?;
        // Free original qweight (it was raw-allocated by weights.take())
        driver::mem_free(qweight_gpu.raw_ptr())?;
    }
    let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

    // Load and permute scales (CPU)
    let (scales_bytes, _scales_shape, scales_dtype) = weights.take_cpu(&scales_name)?;
    let mut scales_u16: Vec<u16> = scales_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    quant::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

    let scales_bytes_permuted: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let scales_nbytes = scales_bytes_permuted.len();
    let scales_ptr = unsafe { driver::mem_alloc(scales_nbytes)? };
    unsafe {
        driver::memcpy_htod_async(
            scales_ptr,
            scales_bytes_permuted.as_ptr(),
            scales_nbytes,
            stream,
        )?;
    }
    let scales_gpu = unsafe { GpuTensor::new(scales_ptr, &[num_groups, size_n], scales_dtype) };

    // Handle zero points for asymmetric GPTQ
    let zeros_gpu = if !cfg.sym && weights.contains(&qzeros_name) {
        let qzeros = weights.take(&qzeros_name)?;
        Some(qzeros)
    } else {
        // Consume the tensor if it exists (so it doesn't cause "unused weight" warnings)
        if weights.contains(&qzeros_name) {
            let _ = weights.take(&qzeros_name);
        }
        None
    };

    // Consume g_idx if present (not used without desc_act)
    let g_idx_name = format!("{prefix}.g_idx");
    if weights.contains(&g_idx_name) {
        let _ = weights.take(&g_idx_name);
    }

    // Load bias if present
    let bias_name = format!("{prefix}.bias");
    let bias_gpu = if weights.contains(&bias_name) {
        Some(weights.take(&bias_name)?)
    } else {
        None
    };

    unsafe { driver::stream_synchronize(stream)? };

    Ok(MarlinLinear {
        qweight: qweight_marlin,
        scales: scales_gpu,
        zeros: zeros_gpu,
        g_idx: None,
        g_idx_sort_indices: None,
        workspace,
        size_k,
        size_n,
        group_size,
        num_groups,
        has_zp: !cfg.sym,
        has_act_order: false,
        b_type_id: 0, // GPTQ = uint4b8
        device_id,
        bias: bias_gpu,
    })
}

/// Concatenate multiple CPU tensors along dimension 1 (the N/output dimension).
///
/// All tensors must have the same dim-0 size, dtype, and be 2D.
/// Returns the concatenated bytes, new shape, and dtype.
fn concat_cpu_dim1(tensors: &[(&[u8], &[usize], DType)]) -> (Vec<u8>, Vec<usize>, DType) {
    assert!(!tensors.is_empty());
    let dtype = tensors[0].2;
    let dim0 = tensors[0].1[0];
    let elem_size = dtype.size_bytes();

    // Compute total dim1.
    let total_dim1: usize = tensors.iter().map(|(_, shape, _)| shape[1]).sum();

    // Row-interleaved concat: for each row, append columns from each tensor.
    let total_bytes = dim0 * total_dim1 * elem_size;
    let mut out = vec![0u8; total_bytes];

    for row in 0..dim0 {
        let mut col_offset = 0usize;
        for (data, shape, _) in tensors {
            let n = shape[1];
            let src_row_bytes = n * elem_size;
            let src_start = row * src_row_bytes;
            let dst_start = (row * total_dim1 + col_offset) * elem_size;
            out[dst_start..dst_start + src_row_bytes]
                .copy_from_slice(&data[src_start..src_start + src_row_bytes]);
            col_offset += n;
        }
    }

    (out, vec![dim0, total_dim1], dtype)
}

/// Load multiple quantized linear layers and fuse into a single Marlin GEMM.
///
/// This is the key optimization: instead of 3 separate Marlin GEMMs for QKV
/// (or 2 for gate_up), we concatenate the raw qweights/scales/qzeros along the
/// N dimension on CPU, repack once, and get a single wider Marlin GEMM.
/// This matches Python vLLM's `MergedColumnParallelLinear`.
///
/// 5→2 GEMMs per layer (QKV fused, gate_up fused).
#[allow(clippy::too_many_arguments)]
pub fn load_fused_marlin_linear(
    weights: &mut GpuWeights,
    prefixes: &[String],
    qconfig: &QuantConfig,
    workspace: GpuTensor,
    device_id: i32,
    alloc: &mut CachingAllocator,
) -> Result<MarlinLinear> {
    let stream = weights.stream();
    match qconfig {
        QuantConfig::Awq(cfg) => {
            load_fused_awq_marlin(weights, prefixes, cfg, workspace, device_id, alloc, stream)
        }
        QuantConfig::Gptq(cfg) => {
            load_fused_gptq_marlin(weights, prefixes, cfg, workspace, device_id, alloc, stream)
        }
        QuantConfig::None => bail!("load_fused_marlin_linear called with QuantConfig::None"),
    }
}

/// Load and fuse multiple GPTQ layers into a single Marlin layer.
///
/// GPTQ qweight: `[K/8, N]` i32 — concat along dim1 → `[K/8, N_total]`.
#[allow(clippy::too_many_arguments)]
fn load_fused_gptq_marlin(
    weights: &mut GpuWeights,
    prefixes: &[String],
    cfg: &quant::GptqConfig,
    workspace: GpuTensor,
    device_id: i32,
    _alloc: &mut CachingAllocator,
    stream: CUstream,
) -> Result<MarlinLinear> {
    // Gather raw CPU tensors for concat.
    let mut qw_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
    let mut sc_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
    let mut qz_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
    let mut has_qzeros = false;

    for prefix in prefixes {
        let qw_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        qw_parts.push(weights.take_cpu(&qw_name)?);
        sc_parts.push(weights.take_cpu(&scales_name)?);

        if !cfg.sym && weights.contains(&qzeros_name) {
            qz_parts.push(weights.take_cpu(&qzeros_name)?);
            has_qzeros = true;
        } else if weights.contains(&qzeros_name) {
            let _ = weights.take_cpu(&qzeros_name);
        }

        // Consume g_idx if present (not used without desc_act).
        let g_idx_name = format!("{prefix}.g_idx");
        if weights.contains(&g_idx_name) {
            let _ = weights.take_cpu(&g_idx_name);
        }
    }

    // Concat qweights along dim1: [K/8, N1] + [K/8, N2] + ... → [K/8, N_total]
    let qw_refs: Vec<_> = qw_parts
        .iter()
        .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
        .collect();
    let (qw_fused, qw_shape, _) = concat_cpu_dim1(&qw_refs);
    let size_k = qw_shape[0] * 8;
    let size_n = qw_shape[1];
    let group_size = cfg.group_size;
    let num_groups = if group_size > 0 {
        size_k / group_size
    } else {
        1
    };

    // Upload fused qweight to GPU and repack.
    let qw_nbytes = qw_fused.len();
    let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
    unsafe { driver::memcpy_htod_async(qw_gpu_ptr, qw_fused.as_ptr(), qw_nbytes, stream)? };
    let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &qw_shape, DType::I32) };

    let num_u32 = size_k * size_n / 8;
    let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
    let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
    unsafe {
        crate::kernels::gptq_repack_into(
            qw_gpu, None, repack_ptr, size_k, size_n, device_id, stream,
        );
        driver::stream_synchronize(stream)?;
        driver::mem_free(qw_gpu_ptr)?;
    }
    let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

    // Concat and permute scales: [num_groups, N1] + ... → [num_groups, N_total]
    let sc_refs: Vec<_> = sc_parts
        .iter()
        .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
        .collect();
    let (sc_fused, _sc_shape, scales_dtype) = concat_cpu_dim1(&sc_refs);
    let mut scales_u16: Vec<u16> = sc_fused
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    quant::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

    let scales_bytes_permuted: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let scales_nbytes = scales_bytes_permuted.len();
    let scales_ptr = unsafe { driver::mem_alloc(scales_nbytes)? };
    unsafe {
        driver::memcpy_htod_async(
            scales_ptr,
            scales_bytes_permuted.as_ptr(),
            scales_nbytes,
            stream,
        )?;
    }
    let scales_gpu = unsafe { GpuTensor::new(scales_ptr, &[num_groups, size_n], scales_dtype) };

    // Handle zero points.
    let zeros_gpu = if has_qzeros && !qz_parts.is_empty() {
        let qz_refs: Vec<_> = qz_parts
            .iter()
            .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
            .collect();
        // For GPTQ asymmetric zeros, just upload the fused tensor.
        let (qz_fused, qz_shape, qz_dtype) = concat_cpu_dim1(&qz_refs);
        let qz_nbytes = qz_fused.len();
        let qz_ptr = unsafe { driver::mem_alloc(qz_nbytes)? };
        unsafe { driver::memcpy_htod_async(qz_ptr, qz_fused.as_ptr(), qz_nbytes, stream)? };
        Some(unsafe { GpuTensor::new(qz_ptr, &qz_shape, qz_dtype) })
    } else {
        None
    };

    unsafe { driver::stream_synchronize(stream)? };

    Ok(MarlinLinear {
        qweight: qweight_marlin,
        scales: scales_gpu,
        zeros: zeros_gpu,
        g_idx: None,
        g_idx_sort_indices: None,
        workspace,
        size_k,
        size_n,
        group_size,
        num_groups,
        has_zp: !cfg.sym,
        has_act_order: false,
        b_type_id: 0, // GPTQ = uint4b8
        device_id,
        bias: None, // Fused layers don't have bias in GPTQ models
    })
}

/// Load and fuse multiple AWQ layers into a single Marlin layer.
///
/// AWQ qweight: `[K, N/8]` i32 — concat along dim1 → `[K, N_total/8]`.
#[allow(clippy::too_many_arguments)]
fn load_fused_awq_marlin(
    weights: &mut GpuWeights,
    prefixes: &[String],
    cfg: &quant::AwqConfig,
    workspace: GpuTensor,
    device_id: i32,
    _alloc: &mut CachingAllocator,
    stream: CUstream,
) -> Result<MarlinLinear> {
    // Gather raw CPU tensors.
    let mut qw_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
    let mut sc_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();
    let mut qz_parts: Vec<(Vec<u8>, Vec<usize>, DType)> = Vec::new();

    // Track per-part N sizes for zero-point handling.
    let mut part_n_sizes: Vec<usize> = Vec::new();

    for prefix in prefixes {
        let qw_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        let (qw_data, qw_shape, qw_dt) = weights.take_cpu(&qw_name)?;
        let part_n = qw_shape[1] * 8; // AWQ: N/8 packed
        part_n_sizes.push(part_n);
        qw_parts.push((qw_data, qw_shape, qw_dt));
        sc_parts.push(weights.take_cpu(&scales_name)?);
        qz_parts.push(weights.take_cpu(&qzeros_name)?);
    }

    // Concat qweights along dim1: [K, N1/8] + [K, N2/8] + ... → [K, N_total/8]
    let qw_refs: Vec<_> = qw_parts
        .iter()
        .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
        .collect();
    let (qw_fused, qw_shape, _) = concat_cpu_dim1(&qw_refs);
    let size_k = qw_shape[0];
    let size_n = qw_shape[1] * 8;
    let group_size = cfg.group_size;
    let num_groups = if group_size > 0 {
        size_k / group_size
    } else {
        1
    };

    // Upload fused qweight to GPU and repack.
    let qw_nbytes = qw_fused.len();
    let qw_gpu_ptr = unsafe { driver::mem_alloc(qw_nbytes)? };
    unsafe { driver::memcpy_htod_async(qw_gpu_ptr, qw_fused.as_ptr(), qw_nbytes, stream)? };
    let qw_gpu = unsafe { GpuTensor::new(qw_gpu_ptr, &qw_shape, DType::I32) };

    let num_u32 = size_k * size_n / 8;
    let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
    let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
    unsafe {
        crate::kernels::awq_repack_into(qw_gpu, repack_ptr, size_k, size_n, device_id, stream);
        driver::stream_synchronize(stream)?;
        driver::mem_free(qw_gpu_ptr)?;
    }
    let qweight_marlin = unsafe { GpuTensor::new(repack_ptr, &[num_u32], DType::U32) };

    // Concat and permute scales: [num_groups, N1] + ... → [num_groups, N_total]
    let sc_refs: Vec<_> = sc_parts
        .iter()
        .map(|(d, s, dt)| (d.as_slice(), s.as_slice(), *dt))
        .collect();
    let (sc_fused, _sc_shape, scales_dtype) = concat_cpu_dim1(&sc_refs);
    let mut scales_u16: Vec<u16> = sc_fused
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    quant::marlin_permute_scales(&mut scales_u16, size_k, size_n, group_size);

    let scales_bytes_permuted: Vec<u8> = scales_u16.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let scales_nbytes = scales_bytes_permuted.len();
    let scales_ptr = unsafe { driver::mem_alloc(scales_nbytes)? };
    unsafe {
        driver::memcpy_htod_async(
            scales_ptr,
            scales_bytes_permuted.as_ptr(),
            scales_nbytes,
            stream,
        )?;
    }
    let scales_gpu = unsafe { GpuTensor::new(scales_ptr, &[num_groups, size_n], scales_dtype) };

    // Concat zero points: each part's qzeros [num_groups, N_i/8] → convert to marlin format,
    // then row-interleave into the fused layout.
    // awq_to_marlin_zero_points returns Vec<u32> with shape [num_groups, N_i/8].
    let mut all_marlin_zp: Vec<Vec<u32>> = Vec::new();
    for (i, (qz_data, _qz_shape, _qz_dt)) in qz_parts.iter().enumerate() {
        let qzeros_u32: Vec<u32> = qz_data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let part_zp = quant::awq_to_marlin_zero_points(&qzeros_u32, num_groups, part_n_sizes[i]);
        all_marlin_zp.push(part_zp);
    }
    // Row-interleave: for each group row, append N_i/8 columns from each part.
    let total_n_div8 = size_n / 8;
    let mut fused_zp = vec![0u32; num_groups * total_n_div8];
    let mut col_offsets: Vec<usize> = Vec::new();
    let mut cumulative = 0usize;
    for &pn in &part_n_sizes {
        col_offsets.push(cumulative);
        cumulative += pn / 8;
    }
    for (i, &pn) in part_n_sizes.iter().enumerate() {
        let part_cols = pn / 8;
        for g in 0..num_groups {
            let dst_start = g * total_n_div8 + col_offsets[i];
            let src_start = g * part_cols;
            fused_zp[dst_start..dst_start + part_cols]
                .copy_from_slice(&all_marlin_zp[i][src_start..src_start + part_cols]);
        }
    }

    let zp_bytes: Vec<u8> = fused_zp.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let zp_nbytes = zp_bytes.len();
    let zp_ptr = unsafe { driver::mem_alloc(zp_nbytes)? };
    unsafe { driver::memcpy_htod_async(zp_ptr, zp_bytes.as_ptr(), zp_nbytes, stream)? };
    let zeros_gpu = unsafe { GpuTensor::new(zp_ptr, &[num_groups, total_n_div8], DType::U32) };

    unsafe { driver::stream_synchronize(stream)? };

    Ok(MarlinLinear {
        qweight: qweight_marlin,
        scales: scales_gpu,
        zeros: Some(zeros_gpu),
        g_idx: None,
        g_idx_sort_indices: None,
        workspace,
        size_k,
        size_n,
        group_size,
        num_groups,
        has_zp: true,
        has_act_order: false,
        b_type_id: 1, // AWQ = uint4
        device_id,
        bias: None,
    })
}

/// Allocate the shared Marlin workspace buffer `[num_sms]` i32.
///
/// This is shared across all MarlinLinear layers — only one allocation needed.
pub fn alloc_marlin_workspace(num_sm: i32, stream: CUstream) -> Result<GpuTensor> {
    // Match Python: max(2 * num_sm, 1024 * 1024) elements
    let num_elements = std::cmp::max(2 * num_sm as usize, 1024 * 1024);
    let nbytes = num_elements * std::mem::size_of::<i32>();
    let ptr = unsafe { driver::mem_alloc(nbytes)? };
    // Zero it — Marlin uses it as barrier locks
    unsafe { driver::memset_d8(ptr, 0, nbytes, stream)? };
    Ok(unsafe { GpuTensor::new(ptr, &[num_elements], DType::I32) })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_dtype_mapping() {
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::F16).unwrap(),
            DType::F16
        );
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::BF16).unwrap(),
            DType::BF16
        );
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::F32).unwrap(),
            DType::F32
        );
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::I64).unwrap(),
            DType::I64
        );
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::U32).unwrap(),
            DType::U32
        );
    }

    #[test]
    fn test_unsupported_dtype() {
        assert!(safetensors_dtype(safetensors::Dtype::BOOL).is_err());
        assert!(safetensors_dtype(safetensors::Dtype::U8).is_err());
    }

    #[test]
    fn test_concat_cpu_dim1() {
        // Two [2, 3] i32 tensors → [2, 6]
        let a: Vec<u8> = [1i32, 2, 3, 4, 5, 6]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b: Vec<u8> = [7i32, 8, 9, 10, 11, 12]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let (out, shape, _) =
            concat_cpu_dim1(&[(&a, &[2, 3], DType::I32), (&b, &[2, 3], DType::I32)]);

        assert_eq!(shape, vec![2, 6]);
        let vals: Vec<i32> = out
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Row 0: [1,2,3] ++ [7,8,9] = [1,2,3,7,8,9]
        // Row 1: [4,5,6] ++ [10,11,12] = [4,5,6,10,11,12]
        assert_eq!(vals, vec![1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]);
    }

    #[test]
    fn test_concat_cpu_dim1_three_tensors() {
        // Three [1, 2] tensors → [1, 6]
        let a: Vec<u8> = [1i32, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b: Vec<u8> = [3i32, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
        let c: Vec<u8> = [5i32, 6].iter().flat_map(|v| v.to_le_bytes()).collect();

        let (out, shape, _) = concat_cpu_dim1(&[
            (&a, &[1, 2], DType::I32),
            (&b, &[1, 2], DType::I32),
            (&c, &[1, 2], DType::I32),
        ]);

        assert_eq!(shape, vec![1, 6]);
        let vals: Vec<i32> = out
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals, vec![1, 2, 3, 4, 5, 6]);
    }

    // GPU tests for actual weight loading.
    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::*;

        fn init_cuda() -> CUstream {
            unsafe {
                driver::init().expect("CUDA init");
                let dev = driver::device_get(0).expect("device");
                let _ctx = driver::ctx_create(dev).expect("context");
                driver::stream_create().expect("stream")
            }
        }

        #[test]
        fn test_load_safetensors_file() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
            let data_b: Vec<f32> = vec![0.1, 0.2, 0.3]; // [3]

            let a_bytes: Vec<u8> = data_a.iter().flat_map(|f| f.to_le_bytes()).collect();
            let b_bytes: Vec<u8> = data_b.iter().flat_map(|f| f.to_le_bytes()).collect();

            let tensors = vec![
                (
                    "weight_a",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![2, 3],
                        &a_bytes,
                    )
                    .unwrap(),
                ),
                (
                    "weight_b",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![3],
                        &b_bytes,
                    )
                    .unwrap(),
                ),
            ];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();

            assert_eq!(gw.len(), 2);
            assert!(gw.contains("weight_a"));
            assert!(gw.contains("weight_b"));

            let a = gw.take("weight_a").unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };
            assert_eq!(a.ndim(), 2);
            assert_eq!(a.dim(0), 2);
            assert_eq!(a.dim(1), 3);
            assert_eq!(a.dtype(), DType::F32);

            let b = gw.take("weight_b").unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };
            assert_eq!(b.ndim(), 1);
            assert_eq!(b.dim(0), 3);

            // Verify data roundtrip: read back from GPU.
            let host = unsafe { driver::mem_alloc_host(a.size_bytes()).unwrap() };
            unsafe {
                driver::memcpy_dtoh_async(host, a.raw_ptr(), a.size_bytes(), stream).unwrap();
                driver::stream_synchronize(stream).unwrap();
            }
            let gpu_data = unsafe { std::slice::from_raw_parts(host as *const f32, 6) };
            for (i, (got, exp)) in gpu_data.iter().zip(data_a.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-6,
                    "weight_a mismatch at {i}: got {got}, expected {exp}"
                );
            }
            unsafe { driver::mem_free_host(host).unwrap() };
            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_load_from_dir() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data: Vec<f32> = vec![1.0; 16];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let tensors = vec![(
                "test.weight",
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4, 4], &bytes)
                    .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_dir(dir.path(), stream).unwrap();

            assert_eq!(gw.len(), 1);
            let t = gw.take("test.weight").unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };
            assert_eq!(t.dim(0), 4);
            assert_eq!(t.dim(1), 4);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_take_and_strip_prefix() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data: Vec<f32> = vec![0.0; 8];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let tensors = vec![(
                "model.layer.weight",
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 4], &bytes)
                    .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();

            // Strip prefix.
            gw.strip_prefix("model.");
            assert!(gw.contains("layer.weight"));
            assert!(!gw.contains("model.layer.weight"));

            // Take.
            let t = gw.take("layer.weight").unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };
            assert_eq!(t.dim(0), 2);
            assert_eq!(gw.len(), 0);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_nonexistent_dir() {
            let stream = init_cuda();
            let result = GpuWeights::from_dir("/nonexistent/path", stream);
            assert!(result.is_err());
            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_bf16_weights() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let bf16_data: Vec<u8> = vec![0x00, 0x3F, 0x00, 0x40]; // 0.5 and 2.0 in BF16
            let tensors = vec![(
                "w",
                safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2], &bf16_data)
                    .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();

            let w = gw.take("w").unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };
            assert_eq!(w.dtype(), DType::BF16);
            assert_eq!(w.numel(), 2);
            assert_eq!(w.size_bytes(), 4);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_names_iterator() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data: Vec<u8> = vec![0; 16];
            let tensors = vec![
                (
                    "a",
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], &data)
                        .unwrap(),
                ),
                (
                    "b",
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], &data)
                        .unwrap(),
                ),
            ];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let gw = GpuWeights::from_single_file(&path, stream).unwrap();

            let mut names: Vec<&str> = gw.names().collect();
            names.sort();
            assert_eq!(names, vec!["a", "b"]);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_take_into() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let data_b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
            let a_bytes: Vec<u8> = data_a.iter().flat_map(|f| f.to_le_bytes()).collect();
            let b_bytes: Vec<u8> = data_b.iter().flat_map(|f| f.to_le_bytes()).collect();

            let tensors = vec![
                (
                    "a",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![4],
                        &a_bytes,
                    )
                    .unwrap(),
                ),
                (
                    "b",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![4],
                        &b_bytes,
                    )
                    .unwrap(),
                ),
            ];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();

            // Pre-allocate a fused GPU buffer for both tensors.
            let total_bytes = 32; // 8 floats * 4 bytes
            let gpu_buf = unsafe { driver::mem_alloc(total_bytes).unwrap() };

            // Copy both tensors into the fused buffer.
            let size_a = unsafe { gw.take_into("a", gpu_buf, stream).unwrap() };
            assert_eq!(size_a, 16);
            let size_b = unsafe {
                gw.take_into("b", gpu_buf.wrapping_add(size_a), stream)
                    .unwrap()
            };
            assert_eq!(size_b, 16);

            // Verify roundtrip.
            let host = unsafe { driver::mem_alloc_host(total_bytes).unwrap() };
            unsafe {
                driver::memcpy_dtoh_async(host, gpu_buf as *mut u8, total_bytes, stream).unwrap();
                driver::stream_synchronize(stream).unwrap();
            }
            let gpu_data = unsafe { std::slice::from_raw_parts(host as *const f32, 8) };
            let expected: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            for (i, (got, exp)) in gpu_data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-6,
                    "mismatch at {i}: got {got}, expected {exp}"
                );
            }

            unsafe {
                driver::mem_free_host(host).unwrap();
                driver::mem_free(gpu_buf).unwrap();
                driver::stream_destroy(stream).unwrap();
            };
        }
    }
}
