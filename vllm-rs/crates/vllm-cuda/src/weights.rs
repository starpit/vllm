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
        safetensors::Dtype::U8 => Ok(DType::U8),
        other => bail!("unsupported safetensors dtype: {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// CPU dtype conversion helpers (for LoRA merging)
// ---------------------------------------------------------------------------

/// Read raw bytes in `dtype` into a pre-allocated f32 slice.
fn read_to_f32(data: &[u8], dtype: DType, out: &mut [f32]) {
    match dtype {
        DType::F32 => {
            let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, out.len()) };
            out.copy_from_slice(src);
        }
        DType::F16 => {
            let src =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const half::f16, out.len()) };
            for (s, d) in src.iter().zip(out.iter_mut()) {
                *d = s.to_f32();
            }
        }
        DType::BF16 => {
            let src = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, out.len())
            };
            for (s, d) in src.iter().zip(out.iter_mut()) {
                *d = s.to_f32();
            }
        }
        _ => panic!("read_to_f32: unsupported dtype {dtype}"),
    }
}

/// Write f32 values back to bytes in the given dtype.
fn write_from_f32(data: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => {
            let mut out = vec![0u8; data.len() * 4];
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, data.len()) };
            dst.copy_from_slice(data);
            out
        }
        DType::F16 => {
            let mut out = vec![0u8; data.len() * 2];
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, data.len()) };
            for (s, d) in data.iter().zip(dst.iter_mut()) {
                *d = half::f16::from_f32(*s).to_bits();
            }
            out
        }
        DType::BF16 => {
            let mut out = vec![0u8; data.len() * 2];
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, data.len()) };
            for (s, d) in data.iter().zip(dst.iter_mut()) {
                *d = half::bf16::from_f32(*s).to_bits();
            }
            out
        }
        _ => panic!("write_from_f32: unsupported dtype {dtype}"),
    }
}

// ---------------------------------------------------------------------------
// CpuTensorRef — a reference to tensor data in a mmap'd safetensors file
// ---------------------------------------------------------------------------

/// A CPU-side reference to tensor data — either mmap'd (read-only) or owned
/// (e.g. after LoRA merging).
struct CpuTensorRef {
    /// The mmap that backs this tensor (None for owned data).
    mmap: Option<Arc<memmap2::Mmap>>,
    /// Byte offset within the mmap where tensor data starts.
    data_offset: usize,
    /// Size of tensor data in bytes.
    size_bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
    /// Owned data buffer (used for merged weights). When set, `data()` returns
    /// this instead of the mmap slice.
    owned: Option<Arc<Vec<u8>>>,
}

impl CpuTensorRef {
    fn data(&self) -> &[u8] {
        if let Some(ref buf) = self.owned {
            buf.as_slice()
        } else {
            let mmap = self
                .mmap
                .as_ref()
                .expect("CpuTensorRef: no mmap or owned data");
            &mmap[self.data_offset..self.data_offset + self.size_bytes]
        }
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
                    mmap: Some(Arc::clone(&mmap)),
                    data_offset,
                    size_bytes,
                    shape,
                    dtype,
                    owned: None,
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

    /// Take a tensor and return its data as a CPU `Vec<f32>`.
    ///
    /// Useful for small per-head parameters (A_log, dt_bias, norm weights)
    /// that need to be kept on CPU or uploaded to GPU as f32.
    pub fn take_to_cpu_f32(&mut self, name: &str) -> Result<Vec<f32>> {
        let cpu_ref = self
            .tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))?;

        let data = cpu_ref.data();
        let num_elems: usize = cpu_ref.shape.iter().product();
        let mut result = Vec::with_capacity(num_elems);

        match cpu_ref.dtype {
            DType::F32 => {
                let src =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, num_elems) };
                result.extend_from_slice(src);
            }
            DType::F16 => {
                let src = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const half::f16, num_elems)
                };
                result.extend(src.iter().map(|v| v.to_f32()));
            }
            DType::BF16 => {
                let src = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const half::bf16, num_elems)
                };
                result.extend(src.iter().map(|v| v.to_f32()));
            }
            other => anyhow::bail!("take_to_cpu_f32: unsupported dtype {other}"),
        }

        Ok(result)
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
    // LoRA weight merging (CPU-side, before H2D copy)
    // -----------------------------------------------------------------------

    /// Merge a LoRA adapter's A/B weight pairs into base weights on CPU.
    ///
    /// For each LoRA target module, computes `W_merged = W + scaling * B @ A`
    /// in f32 intermediate precision and replaces the mmap'd `CpuTensorRef`
    /// with an owned buffer containing the merged result.
    ///
    /// Must be called BEFORE `take()`/`take_into()` so that model construction
    /// picks up already-merged weights (including fused QKV / gate_up).
    ///
    /// Returns the number of weight tensors merged.
    pub fn merge_lora(&mut self, adapter_dir: &Path) -> Result<usize> {
        use vllm_model::lora::LoraAdapterConfig;

        // 1. Parse adapter config.
        let config_path = adapter_dir.join("adapter_config.json");
        let config = LoraAdapterConfig::from_file(&config_path)
            .map_err(|e| anyhow::anyhow!("LoRA config: {e}"))?;
        let scaling = config.scaling();

        // 2. Load adapter weights (CPU mmap).
        let st_path = adapter_dir.join("adapter_model.safetensors");
        if !st_path.exists() {
            bail!(
                "adapter_model.safetensors not found in {}",
                adapter_dir.display()
            );
        }
        let file = std::fs::File::open(&st_path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow::anyhow!("LoRA safetensors: {e}"))?;

        // 3. Group A/B pairs by layer prefix.
        //    PEFT names: base_model.model.{prefix}.lora_A.weight
        #[allow(clippy::type_complexity)]
        let mut pairs: HashMap<
            String,
            (Option<&[u8]>, Vec<usize>, Option<&[u8]>, Vec<usize>),
        > = HashMap::new();

        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
            let (prefix, is_a) = if let Some(p) = name.strip_suffix(".lora_A.weight") {
                (p, true)
            } else if let Some(p) = name.strip_suffix(".lora_B.weight") {
                (p, false)
            } else {
                continue;
            };
            let clean = prefix.strip_prefix("base_model.model.").unwrap_or(prefix);
            let entry = pairs
                .entry(clean.to_string())
                .or_insert((None, vec![], None, vec![]));
            if is_a {
                entry.0 = Some(view.data());
                entry.1 = view.shape().to_vec();
            } else {
                entry.2 = Some(view.data());
                entry.3 = view.shape().to_vec();
            }
        }

        // 4. For each pair, merge into the base weight.
        let mut merged_count = 0usize;
        for (prefix, (a_data, a_shape, b_data, b_shape)) in &pairs {
            let a_data = match a_data {
                Some(d) => d,
                None => {
                    tracing::warn!("LoRA: missing lora_A for {prefix}, skipping");
                    continue;
                }
            };
            let b_data = match b_data {
                Some(d) => d,
                None => {
                    tracing::warn!("LoRA: missing lora_B for {prefix}, skipping");
                    continue;
                }
            };

            // Find matching base weight. The prefix should match a key in self.tensors
            // (after strip_prefix("model.") has been applied, or not).
            let base_name = if self.tensors.contains_key(&format!("{prefix}.weight")) {
                format!("{prefix}.weight")
            } else {
                tracing::debug!("LoRA: no base weight for {prefix}, skipping");
                continue;
            };

            let base = &self.tensors[&base_name];
            if base.shape.len() != 2 {
                tracing::warn!("LoRA: base weight {base_name} is not 2D, skipping");
                continue;
            }

            // A: [rank, in], B: [out, rank], W: [out, in]
            let rank = a_shape[0];
            let in_feat = a_shape[1];
            let out_feat = b_shape[0];

            if base.shape != [out_feat, in_feat] {
                tracing::warn!(
                    "LoRA: shape mismatch for {base_name}: base {:?} vs LoRA out={out_feat} in={in_feat}",
                    base.shape
                );
                continue;
            }

            // Read base weight to f32.
            let numel = out_feat * in_feat;
            let mut w_f32 = vec![0.0f32; numel];
            read_to_f32(base.data(), base.dtype, &mut w_f32);

            // Read A to f32 [rank, in_feat].
            let a_numel = rank * in_feat;
            let mut a_f32 = vec![0.0f32; a_numel];
            // LoRA weights are typically F32 in PEFT safetensors.
            read_to_f32(a_data, DType::F32, &mut a_f32);

            // Read B to f32 [out_feat, rank].
            let b_numel = out_feat * rank;
            let mut b_f32 = vec![0.0f32; b_numel];
            read_to_f32(b_data, DType::F32, &mut b_f32);

            // Compute delta = B @ A → [out_feat, in_feat], then W += scaling * delta.
            let scaling_f32 = scaling as f32;
            for i in 0..out_feat {
                for j in 0..in_feat {
                    let mut dot = 0.0f32;
                    for k in 0..rank {
                        dot += b_f32[i * rank + k] * a_f32[k * in_feat + j];
                    }
                    w_f32[i * in_feat + j] += scaling_f32 * dot;
                }
            }

            // Write merged weight back in base dtype.
            let merged_bytes = write_from_f32(&w_f32, base.dtype);
            let size_bytes = merged_bytes.len();
            let owned = Arc::new(merged_bytes);

            // Replace CpuTensorRef with one backed by owned data.
            self.tensors.insert(
                base_name,
                CpuTensorRef {
                    mmap: None,
                    data_offset: 0,
                    size_bytes,
                    shape: vec![out_feat, in_feat],
                    dtype: base.dtype,
                    owned: Some(owned),
                },
            );

            merged_count += 1;
            tracing::debug!("LoRA: merged {prefix} → [{out_feat}, {in_feat}]");
        }

        tracing::info!(
            "LoRA: merged {} weight tensors (rank={}, alpha={}, scaling={:.4})",
            merged_count,
            config.r,
            config.lora_alpha,
            scaling,
        );

        Ok(merged_count)
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
        QuantConfig::Bnb4bit(_) => bail!("load_marlin_linear called with Bnb4bit config"),
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

    // GPTQ uses uint4b8 scalar type which bakes in the zero-point (bias=8).
    // Python vLLM never passes zero-points for GPTQ — just consume and discard.
    if weights.contains(&qzeros_name) {
        let _ = weights.take(&qzeros_name);
    }

    // Handle g_idx for desc_act (activation ordering).
    // Must be done BEFORE repack because repack needs `perm` (sort_indices) on GPU.
    let g_idx_name = format!("{prefix}.g_idx");
    let (g_idx_gpu, sort_indices_gpu, has_act_order) =
        if cfg.desc_act && weights.contains(&g_idx_name) {
            // Load g_idx on CPU: Vec<i32> of shape [K]
            let (g_idx_bytes, _g_idx_shape, _g_idx_dtype) = weights.take_cpu(&g_idx_name)?;
            let g_idx_i32: Vec<i32> = g_idx_bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // Argsort: stable ascending sort by group ID
            let mut sort_indices: Vec<i32> = (0..g_idx_i32.len() as i32).collect();
            sort_indices.sort_by_key(|&i| g_idx_i32[i as usize]);

            // Compute sorted_g_idx
            let sorted_g_idx: Vec<i32> = sort_indices
                .iter()
                .map(|&i| g_idx_i32[i as usize])
                .collect();

            // Upload sorted_g_idx to GPU
            let g_idx_bytes: Vec<u8> = sorted_g_idx.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let g_idx_nbytes = g_idx_bytes.len();
            let g_idx_ptr = unsafe { driver::mem_alloc(g_idx_nbytes)? };
            unsafe {
                driver::memcpy_htod_async(g_idx_ptr, g_idx_bytes.as_ptr(), g_idx_nbytes, stream)?;
            }
            let g_idx_gpu = unsafe { GpuTensor::new(g_idx_ptr, &[size_k], DType::I32) };

            // Upload sort_indices to GPU (used as `perm` for repack and GEMM)
            let si_bytes: Vec<u8> = sort_indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
            let si_nbytes = si_bytes.len();
            let si_ptr = unsafe { driver::mem_alloc(si_nbytes)? };
            unsafe { driver::memcpy_htod_async(si_ptr, si_bytes.as_ptr(), si_nbytes, stream)? };
            let sort_indices_gpu = unsafe { GpuTensor::new(si_ptr, &[size_k], DType::I32) };

            (Some(g_idx_gpu), Some(sort_indices_gpu), true)
        } else {
            // Consume g_idx if present (not needed without desc_act)
            if weights.contains(&g_idx_name) {
                let _ = weights.take(&g_idx_name);
            }
            (None, None, false)
        };

    // Upload qweight to GPU (raw alloc — will be freed after repack)
    let qweight_gpu = weights.take(&qw_name)?;

    // Repack GPTQ → Marlin tiled layout on GPU.
    // When has_act_order, pass sort_indices as perm so the repack kernel
    // physically reorders weights so same-group channels are contiguous.
    // Use driver::mem_alloc for the output (NOT caching allocator) because model
    // weights must survive free_leaked_blocks() during profiling.
    let num_u32 = size_k * size_n / 8;
    let repack_nbytes = num_u32 * std::mem::size_of::<u32>();
    let repack_ptr = unsafe { driver::mem_alloc(repack_nbytes)? };
    unsafe {
        crate::kernels::gptq_repack_into(
            qweight_gpu,
            sort_indices_gpu,
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
        zeros: None,
        g_idx: g_idx_gpu,
        g_idx_sort_indices: sort_indices_gpu,
        workspace,
        size_k,
        size_n,
        group_size,
        num_groups,
        has_zp: false,
        has_act_order,
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
        QuantConfig::Bnb4bit(_) => bail!("load_fused_marlin_linear called with Bnb4bit config"),
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
    let mut g_idx_i32: Option<Vec<i32>> = None;

    for (i, prefix) in prefixes.iter().enumerate() {
        let qw_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        qw_parts.push(weights.take_cpu(&qw_name)?);
        sc_parts.push(weights.take_cpu(&scales_name)?);

        // GPTQ uses uint4b8 scalar type — never pass zero-points. Consume and discard.
        if weights.contains(&qzeros_name) {
            let _ = weights.take_cpu(&qzeros_name);
        }

        // For desc_act: all sub-layers share the same K dimension → same g_idx.
        // Take from the first prefix, consume and discard from the rest.
        let g_idx_name = format!("{prefix}.g_idx");
        if weights.contains(&g_idx_name) {
            if i == 0 && cfg.desc_act {
                let (g_bytes, _g_shape, _g_dtype) = weights.take_cpu(&g_idx_name)?;
                g_idx_i32 = Some(
                    g_bytes
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                );
            } else {
                let _ = weights.take_cpu(&g_idx_name);
            }
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

    // Handle g_idx for desc_act — must be done BEFORE repack.
    let (g_idx_gpu, sort_indices_gpu, has_act_order) = if let Some(g_idx) = g_idx_i32 {
        // Argsort: stable ascending sort by group ID
        let mut sort_indices: Vec<i32> = (0..g_idx.len() as i32).collect();
        sort_indices.sort_by_key(|&i| g_idx[i as usize]);

        // Compute sorted_g_idx
        let sorted_g_idx: Vec<i32> = sort_indices.iter().map(|&i| g_idx[i as usize]).collect();

        // Upload sorted_g_idx to GPU
        let g_idx_bytes: Vec<u8> = sorted_g_idx.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let g_idx_nbytes = g_idx_bytes.len();
        let g_idx_ptr = unsafe { driver::mem_alloc(g_idx_nbytes)? };
        unsafe {
            driver::memcpy_htod_async(g_idx_ptr, g_idx_bytes.as_ptr(), g_idx_nbytes, stream)?;
        }
        let g_idx_gpu = unsafe { GpuTensor::new(g_idx_ptr, &[size_k], DType::I32) };

        // Upload sort_indices to GPU (used as `perm` for repack and GEMM)
        let si_bytes: Vec<u8> = sort_indices.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let si_nbytes = si_bytes.len();
        let si_ptr = unsafe { driver::mem_alloc(si_nbytes)? };
        unsafe { driver::memcpy_htod_async(si_ptr, si_bytes.as_ptr(), si_nbytes, stream)? };
        let sort_indices_gpu = unsafe { GpuTensor::new(si_ptr, &[size_k], DType::I32) };

        (Some(g_idx_gpu), Some(sort_indices_gpu), true)
    } else {
        (None, None, false)
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
            qw_gpu,
            sort_indices_gpu,
            repack_ptr,
            size_k,
            size_n,
            device_id,
            stream,
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

    unsafe { driver::stream_synchronize(stream)? };

    Ok(MarlinLinear {
        qweight: qweight_marlin,
        scales: scales_gpu,
        zeros: None,
        g_idx: g_idx_gpu,
        g_idx_sort_indices: sort_indices_gpu,
        workspace,
        size_k,
        size_n,
        group_size,
        num_groups,
        has_zp: false,
        has_act_order,
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
// BitsAndBytes 4-bit weight loading
// ---------------------------------------------------------------------------

use crate::layers::Bnb4bitLinear;
use crate::quant::Bnb4bitConfig;

/// Parse the BNB quant_state JSON blob.
///
/// Returns `(nested_offset, blocksize, nested_blocksize)`.
fn parse_bnb_quant_state_json(data: &[u8]) -> Result<(f32, usize, usize)> {
    // Quant state blobs may have trailing null bytes after the JSON.
    let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
    let s = std::str::from_utf8(&data[..end])?;
    let v: serde_json::Value = serde_json::from_str(s)?;
    let nested_offset = v
        .get("nested_offset")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let blocksize = v.get("blocksize").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
    let nested_blocksize = v
        .get("nested_blocksize")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;
    Ok((nested_offset, blocksize, nested_blocksize))
}

/// Upload the NF4/FP4 lookup table to GPU (shared across all layers).
pub fn upload_bnb_code(code: &[f32; 16], stream: CUstream) -> Result<GpuTensor> {
    let nbytes = 16 * std::mem::size_of::<f32>();
    let ptr = unsafe { driver::mem_alloc(nbytes)? };
    unsafe {
        driver::memcpy_htod_async(ptr, code.as_ptr() as *const u8, nbytes, stream)?;
    }
    Ok(unsafe { GpuTensor::new(ptr, &[16], DType::F32) })
}

/// Allocate a shared dequantization scratch buffer for BNB 4-bit.
///
/// Size: max(out_features * in_features) across all linear layers × elem_size.
/// The caller should track the max dimensions during model loading.
pub fn alloc_bnb_dequant_scratch(
    max_elements: usize,
    dtype: DType,
    stream: CUstream,
) -> Result<GpuTensor> {
    let nbytes = max_elements * dtype.size_bytes();
    let ptr = unsafe { driver::mem_alloc(nbytes)? };
    // Zero-initialize (not strictly needed, but helps debugging).
    unsafe { driver::memset_d8(ptr, 0, nbytes, stream)? };
    Ok(unsafe { GpuTensor::new(ptr, &[max_elements], dtype) })
}

/// Dequantize double-quantized absmax on CPU.
///
/// BNB pre-quantized models store absmax as U8 (double-quantized).
/// This matches Python `_dequantize_dq`:
/// ```text
/// absmax_f32[i] = nested_quant_map[absmax_u8[i]] * nested_absmax[i / 256]
/// ```
pub fn dequantize_double_quant_absmax(
    absmax_u8: &[u8],
    nested_quant_map: &[f32], // [256]
    nested_absmax: &[f32],    // [num_blocks / nested_blocksize]
    nested_blocksize: usize,  // typically 256
    nested_offset: f32,       // from quant_state JSON
) -> Vec<f32> {
    absmax_u8
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let scale = nested_absmax[i / nested_blocksize];
            nested_quant_map[val as usize] * scale + nested_offset
        })
        .collect()
}

/// Load a single BNB 4-bit linear layer from safetensors.
///
/// Expects:
/// - `{prefix}.weight` — U8 packed nibbles
/// - `{prefix}.weight.absmax` — U8 double-quantized absmax
/// - `{prefix}.weight.nested_absmax` — F32 absmax of absmax
/// - `{prefix}.weight.nested_quant_map` — F32 [256] nested dequant table
/// - `{prefix}.weight.quant_map` — F32 [16] NF4/FP4 code (used to detect quant type)
/// - `{prefix}.weight.quant_state.bitsandbytes__nf4` — U8 metadata blob
///
/// `code_gpu` is the shared NF4/FP4 lookup table already on GPU.
/// `dequant_scratch` is the shared dequant scratch buffer.
/// `out_features`/`in_features` are the original weight dimensions.
#[allow(clippy::too_many_arguments)]
pub fn load_bnb4bit_linear(
    weights: &mut GpuWeights,
    prefix: &str,
    _qconfig: &Bnb4bitConfig,
    code_gpu: GpuTensor,
    dequant_scratch: GpuTensor,
    out_features: usize,
    in_features: usize,
    blocksize: usize,
    stream: CUstream,
) -> Result<Bnb4bitLinear> {
    let weight_name = format!("{prefix}.weight");
    let absmax_name = format!("{prefix}.weight.absmax");
    let nested_absmax_name = format!("{prefix}.weight.nested_absmax");
    let nested_quant_map_name = format!("{prefix}.weight.nested_quant_map");

    // Parse quant_state JSON to get nested_offset and actual blocksize.
    let quant_state_name = format!("{prefix}.weight.quant_state.bitsandbytes__nf4");
    let (nested_offset, actual_blocksize, nested_blocksize) = if weights.contains(&quant_state_name)
    {
        let (qs_bytes, _, _) = weights.take_cpu(&quant_state_name)?;
        parse_bnb_quant_state_json(&qs_bytes)?
    } else {
        (0.0, blocksize, 256)
    };
    let blocksize = actual_blocksize;

    // Load packed weight to GPU (U8).
    let packed_weight = weights.take(&weight_name)?;

    // Load and dequantize double-quantized absmax on CPU.
    let (absmax_bytes, _absmax_shape, absmax_dtype) = weights.take_cpu(&absmax_name)?;

    let absmax_f32 = if absmax_dtype == DType::U8 {
        // Double quantized — need nested_absmax and nested_quant_map.
        let (nqm_bytes, _nqm_shape, _) = weights.take_cpu(&nested_quant_map_name)?;
        let nested_quant_map: Vec<f32> = nqm_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let (na_bytes, _na_shape, _) = weights.take_cpu(&nested_absmax_name)?;
        let nested_absmax: Vec<f32> = na_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        dequantize_double_quant_absmax(
            &absmax_bytes,
            &nested_quant_map,
            &nested_absmax,
            nested_blocksize,
            nested_offset,
        )
    } else {
        // Already F32 (non-double-quantized).
        absmax_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    // Upload absmax F32 to GPU.
    let absmax_nbytes = absmax_f32.len() * 4;
    let absmax_ptr = unsafe { driver::mem_alloc(absmax_nbytes)? };
    unsafe {
        driver::memcpy_htod_async(
            absmax_ptr,
            absmax_f32.as_ptr() as *const u8,
            absmax_nbytes,
            stream,
        )?;
    }
    let absmax_gpu = unsafe { GpuTensor::new(absmax_ptr, &[absmax_f32.len()], DType::F32) };

    // Load bias if present.
    let bias_name = format!("{prefix}.bias");
    let bias = if weights.contains(&bias_name) {
        Some(weights.take(&bias_name)?)
    } else {
        None
    };

    // Consume remaining BNB metadata tensors so they don't cause "unused weight" warnings.
    let quant_map_name = format!("{prefix}.weight.quant_map");
    for name in &[
        &quant_map_name,
        &quant_state_name,
        &nested_absmax_name,
        &nested_quant_map_name,
    ] {
        if weights.contains(name) {
            let _ = weights.take_cpu(name);
        }
    }

    Ok(Bnb4bitLinear {
        packed_weight,
        absmax: absmax_gpu,
        code: code_gpu,
        dequant_scratch,
        out_features,
        in_features,
        blocksize,
        bias,
    })
}

/// Load fused BNB 4-bit linear (e.g., QKV or gate_up) from multiple prefixes.
///
/// Concatenates packed bytes + absmax from multiple shards on CPU, then uploads once.
/// Absmax blocks are independent per shard, so concat is straightforward.
#[allow(clippy::too_many_arguments)]
pub fn load_fused_bnb4bit_linear(
    weights: &mut GpuWeights,
    prefixes: &[String],
    _qconfig: &Bnb4bitConfig,
    code_gpu: GpuTensor,
    dequant_scratch: GpuTensor,
    out_features_per_shard: &[usize],
    in_features: usize,
    blocksize: usize,
    stream: CUstream,
) -> Result<Bnb4bitLinear> {
    let total_out_features: usize = out_features_per_shard.iter().sum();

    // Concatenate packed bytes from all shards on CPU.
    let mut all_packed: Vec<u8> = Vec::new();
    let mut all_absmax_f32: Vec<f32> = Vec::new();

    for (prefix, &out_feat) in prefixes.iter().zip(out_features_per_shard.iter()) {
        let weight_name = format!("{prefix}.weight");
        let absmax_name = format!("{prefix}.weight.absmax");
        let nested_absmax_name = format!("{prefix}.weight.nested_absmax");
        let nested_quant_map_name = format!("{prefix}.weight.nested_quant_map");
        let quant_state_name = format!("{prefix}.weight.quant_state.bitsandbytes__nf4");

        // Parse quant_state JSON for nested_offset.
        let (nested_offset, _actual_blocksize, nested_blocksize) =
            if weights.contains(&quant_state_name) {
                let (qs_bytes, _, _) = weights.take_cpu(&quant_state_name)?;
                parse_bnb_quant_state_json(&qs_bytes)?
            } else {
                (0.0, blocksize, 256)
            };

        // Load packed weight to CPU.
        let (packed_bytes, _packed_shape, _) = weights.take_cpu(&weight_name)?;
        all_packed.extend_from_slice(&packed_bytes);

        // Load and dequantize absmax.
        let (absmax_bytes, _absmax_shape, absmax_dtype) = weights.take_cpu(&absmax_name)?;
        let shard_absmax = if absmax_dtype == DType::U8 {
            let (nqm_bytes, _, _) = weights.take_cpu(&nested_quant_map_name)?;
            let nested_quant_map: Vec<f32> = nqm_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let (na_bytes, _, _) = weights.take_cpu(&nested_absmax_name)?;
            let nested_absmax: Vec<f32> = na_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            dequantize_double_quant_absmax(
                &absmax_bytes,
                &nested_quant_map,
                &nested_absmax,
                nested_blocksize,
                nested_offset,
            )
        } else {
            absmax_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        all_absmax_f32.extend_from_slice(&shard_absmax);

        // Consume remaining BNB metadata.
        let quant_map_name = format!("{prefix}.weight.quant_map");
        for name in &[
            &quant_map_name,
            &quant_state_name,
            &nested_absmax_name,
            &nested_quant_map_name,
        ] {
            if weights.contains(name) {
                let _ = weights.take_cpu(name);
            }
        }

        let _ = out_feat; // used for shape validation in debug
    }

    // Upload packed weight to GPU.
    let packed_nbytes = all_packed.len();
    let packed_ptr = unsafe { driver::mem_alloc(packed_nbytes)? };
    unsafe {
        driver::memcpy_htod_async(packed_ptr, all_packed.as_ptr(), packed_nbytes, stream)?;
    }
    let packed_gpu = unsafe { GpuTensor::new(packed_ptr, &[packed_nbytes], DType::U8) };

    // Upload absmax F32 to GPU.
    let absmax_nbytes = all_absmax_f32.len() * 4;
    let absmax_ptr = unsafe { driver::mem_alloc(absmax_nbytes)? };
    unsafe {
        driver::memcpy_htod_async(
            absmax_ptr,
            all_absmax_f32.as_ptr() as *const u8,
            absmax_nbytes,
            stream,
        )?;
    }
    let absmax_gpu = unsafe { GpuTensor::new(absmax_ptr, &[all_absmax_f32.len()], DType::F32) };

    Ok(Bnb4bitLinear {
        packed_weight: packed_gpu,
        absmax: absmax_gpu,
        code: code_gpu,
        dequant_scratch,
        out_features: total_out_features,
        in_features,
        blocksize,
        bias: None,
    })
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
    }

    #[test]
    fn test_u8_dtype() {
        assert_eq!(
            safetensors_dtype(safetensors::Dtype::U8).unwrap(),
            DType::U8
        );
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

    #[test]
    fn test_parse_bnb_quant_state_json() {
        let json = br#"{"quant_type":"nf4","blocksize":64,"dtype":"bfloat16","shape":[2048,1024],"nested_blocksize":256,"nested_dtype":"float32","nested_offset":0.07990148663520813}"#;
        let (offset, blocksize, nested_blocksize) = parse_bnb_quant_state_json(json).unwrap();
        assert!((offset - 0.0799).abs() < 1e-4);
        assert_eq!(blocksize, 64);
        assert_eq!(nested_blocksize, 256);
    }

    #[test]
    fn test_parse_bnb_quant_state_json_with_null_padding() {
        // Quant state blobs often have null bytes after the JSON.
        let mut json = br#"{"quant_type":"nf4","blocksize":64}"#.to_vec();
        json.extend_from_slice(&[0, 0, 0]);
        let (offset, blocksize, _) = parse_bnb_quant_state_json(&json).unwrap();
        assert_eq!(offset, 0.0); // no nested_offset key → default 0.0
        assert_eq!(blocksize, 64);
    }

    #[test]
    fn test_dequantize_double_quant_absmax_with_offset() {
        // Matches Python: nested_quant_map[val] * nested_absmax[i/bs] + offset
        let absmax_u8 = [3u8, 7, 1, 5];
        let nqm: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let na = [2.0f32, 3.0];
        let result = dequantize_double_quant_absmax(&absmax_u8, &nqm, &na, 2, 0.5);
        // [0]: 0.03*2.0+0.5=0.56, [1]: 0.07*2.0+0.5=0.64
        // [2]: 0.01*3.0+0.5=0.53, [3]: 0.05*3.0+0.5=0.65
        assert!((result[0] - 0.56).abs() < 1e-5);
        assert!((result[1] - 0.64).abs() < 1e-5);
        assert!((result[2] - 0.53).abs() < 1e-5);
        assert!((result[3] - 0.65).abs() < 1e-5);
    }

    #[test]
    fn test_dequantize_double_quant_absmax_zero_offset() {
        let absmax_u8 = [100u8, 200];
        let nqm: Vec<f32> = (0..256).map(|i| i as f32 * 0.001).collect();
        let na = [1.0f32];
        let result = dequantize_double_quant_absmax(&absmax_u8, &nqm, &na, 256, 0.0);
        // [0]: 0.100*1.0+0.0=0.1, [1]: 0.200*1.0+0.0=0.2
        assert!((result[0] - 0.1).abs() < 1e-5);
        assert!((result[1] - 0.2).abs() < 1e-5);
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

    #[test]
    fn test_merge_lora_f32() {
        use safetensors::tensor::TensorView;

        let dir = tempfile::tempdir().unwrap();

        // Create base model: single weight "model.layers.0.self_attn.q_proj.weight" [8, 4].
        let base_w: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let base_bytes: Vec<u8> = base_w.iter().flat_map(|f| f.to_le_bytes()).collect();

        let base_views = vec![(
            "model.layers.0.self_attn.q_proj.weight",
            TensorView::new(safetensors::Dtype::F32, vec![8, 4], &base_bytes).unwrap(),
        )];
        let base_st = safetensors::tensor::serialize(base_views, None).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), base_st).unwrap();

        // Create LoRA adapter: rank=2, alpha=4 → scaling=2.0.
        let adapter_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            adapter_dir.path().join("adapter_config.json"),
            r#"{"r": 2, "lora_alpha": 4.0, "target_modules": ["q_proj"]}"#,
        )
        .unwrap();

        // A: [2, 4], B: [8, 2]
        let a_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // identity-ish
        let b_data: Vec<f32> = vec![
            0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5,
        ];
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let adapter_views = vec![
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                TensorView::new(safetensors::Dtype::F32, vec![2, 4], &a_bytes).unwrap(),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                TensorView::new(safetensors::Dtype::F32, vec![8, 2], &b_bytes).unwrap(),
            ),
        ];
        let adapter_st = safetensors::tensor::serialize(adapter_views, None).unwrap();
        std::fs::write(
            adapter_dir.path().join("adapter_model.safetensors"),
            adapter_st,
        )
        .unwrap();

        // Load weights (no GPU needed — merge is CPU-only).
        let mut gw = GpuWeights {
            tensors: HashMap::new(),
            stream: std::ptr::null_mut(),
            target_dtype: None,
            cast_pinned: (std::ptr::null_mut(), 0),
        };
        gw.load_shard(&dir.path().join("model.safetensors"))
            .unwrap();

        // Strip "model." prefix to match what CudaWorker does.
        // Actually, merge_lora looks for "{prefix}.weight" keys, so let's check
        // what keys we have.
        let keys: Vec<String> = gw.tensors.keys().cloned().collect();
        assert!(keys.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));

        let merged = gw.merge_lora(adapter_dir.path()).unwrap();
        assert_eq!(merged, 1);

        // Verify merged values: W_merged = W + 2.0 * B @ A
        // B @ A: [8, 2] @ [2, 4] → [8, 4]
        // B has pattern: row i = [0.5, 0.0] or [0.0, 0.5] alternating
        // A = [[1,0,0,0],[0,1,0,0]]
        // B @ A row 0: 0.5*[1,0,0,0] + 0.0*[0,1,0,0] = [0.5,0,0,0]
        // B @ A row 1: 0.0*[1,0,0,0] + 0.5*[0,1,0,0] = [0,0.5,0,0]
        // etc.
        // delta = 2.0 * B@A
        let (data, shape, dtype) = gw
            .take_cpu("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(shape, vec![8, 4]);
        assert_eq!(dtype, DType::F32);
        let merged_w: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Row 0 of delta: [0.5, 0, 0, 0] * 2 = [1, 0, 0, 0]
        // Row 0 of W: [0, 0.1, 0.2, 0.3]
        // Merged: [1.0, 0.1, 0.2, 0.3]
        assert!((merged_w[0] - 1.0).abs() < 1e-5, "got {}", merged_w[0]);
        assert!((merged_w[1] - 0.1).abs() < 1e-5);
        assert!((merged_w[2] - 0.2).abs() < 1e-5);
        assert!((merged_w[3] - 0.3).abs() < 1e-5);

        // Row 1 of delta: [0, 0.5, 0, 0] * 2 = [0, 1, 0, 0]
        // Row 1 of W: [0.4, 0.5, 0.6, 0.7]
        // Merged: [0.4, 1.5, 0.6, 0.7]
        assert!((merged_w[4] - 0.4).abs() < 1e-5);
        assert!((merged_w[5] - 1.5).abs() < 1e-5, "got {}", merged_w[5]);
        assert!((merged_w[6] - 0.6).abs() < 1e-5);
        assert!((merged_w[7] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_merge_lora_rslora_scaling() {
        use safetensors::tensor::TensorView;

        let dir = tempfile::tempdir().unwrap();

        // Base weight: identity-like [2, 2].
        let base_w: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let base_bytes: Vec<u8> = base_w.iter().flat_map(|f| f.to_le_bytes()).collect();
        let base_views = vec![(
            "model.layers.0.self_attn.q_proj.weight",
            TensorView::new(safetensors::Dtype::F32, vec![2, 2], &base_bytes).unwrap(),
        )];
        std::fs::write(
            dir.path().join("model.safetensors"),
            safetensors::tensor::serialize(base_views, None).unwrap(),
        )
        .unwrap();

        // LoRA with rsLoRA: rank=4, alpha=8 → scaling = 8/sqrt(4) = 4.0
        // (normal would be 8/4 = 2.0)
        let adapter_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            adapter_dir.path().join("adapter_config.json"),
            r#"{"r": 4, "lora_alpha": 8.0, "target_modules": ["q_proj"], "use_rslora": true}"#,
        )
        .unwrap();

        // A: [4, 2], B: [2, 4] — simple so B@A = [[1,0],[0,1]] (identity)
        // A = [[1,0],[0,1],[0,0],[0,0]], B = [[1,0,0,0],[0,1,0,0]]
        let a_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let adapter_views = vec![
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                TensorView::new(safetensors::Dtype::F32, vec![4, 2], &a_bytes).unwrap(),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                TensorView::new(safetensors::Dtype::F32, vec![2, 4], &b_bytes).unwrap(),
            ),
        ];
        std::fs::write(
            adapter_dir.path().join("adapter_model.safetensors"),
            safetensors::tensor::serialize(adapter_views, None).unwrap(),
        )
        .unwrap();

        let mut gw = GpuWeights {
            tensors: HashMap::new(),
            stream: std::ptr::null_mut(),
            target_dtype: None,
            cast_pinned: (std::ptr::null_mut(), 0),
        };
        gw.load_shard(&dir.path().join("model.safetensors"))
            .unwrap();

        let merged = gw.merge_lora(adapter_dir.path()).unwrap();
        assert_eq!(merged, 1);

        let (data, shape, _) = gw
            .take_cpu("model.layers.0.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(shape, vec![2, 2]);
        let w: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // B@A = identity, scaling = 4.0 (rsLoRA), base = identity
        // merged = [[1,0],[0,1]] + 4.0 * [[1,0],[0,1]] = [[5,0],[0,5]]
        assert!((w[0] - 5.0).abs() < 1e-5, "got {}", w[0]);
        assert!((w[1] - 0.0).abs() < 1e-5, "got {}", w[1]);
        assert!((w[2] - 0.0).abs() < 1e-5, "got {}", w[2]);
        assert!((w[3] - 5.0).abs() < 1e-5, "got {}", w[3]);
    }

    #[test]
    fn test_read_write_f32_roundtrip() {
        let original = vec![1.0f32, -2.5, 3.14, 0.0];

        // F32 roundtrip.
        let bytes = write_from_f32(&original, DType::F32);
        let mut out = vec![0.0f32; 4];
        read_to_f32(&bytes, DType::F32, &mut out);
        assert_eq!(original, out);

        // BF16 roundtrip (lossy).
        let bytes = write_from_f32(&original, DType::BF16);
        let mut out = vec![0.0f32; 4];
        read_to_f32(&bytes, DType::BF16, &mut out);
        for (a, b) in original.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.1, "BF16 roundtrip: {a} vs {b}");
        }

        // F16 roundtrip (lossy).
        let bytes = write_from_f32(&original, DType::F16);
        let mut out = vec![0.0f32; 4];
        read_to_f32(&bytes, DType::F16, &mut out);
        for (a, b) in original.iter().zip(out.iter()) {
            assert!((a - b).abs() < 0.1, "F16 roundtrip: {a} vs {b}");
        }
    }
}
