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
