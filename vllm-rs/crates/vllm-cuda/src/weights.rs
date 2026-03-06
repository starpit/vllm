// SPDX-License-Identifier: Apache-2.0
//! Safetensors weight loading directly to GPU via cuMemcpy.
//!
//! Reads mmap'd safetensors files, copies raw tensor bytes to a contiguous
//! GPU buffer, and returns `HashMap<String, GpuTensor>` views into that buffer.
//!
//! Unlike the candle-based loader, this:
//! - Allocates ONE contiguous GPU buffer per shard file (fewer cuMemAlloc calls)
//! - Uses our `GpuTensor` type (no refcounting, no events, no strides)
//! - Supports async H2D copies on the transfer stream

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
        other => bail!("unsupported safetensors dtype: {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// GpuWeights
// ---------------------------------------------------------------------------

/// Metadata for a single tensor within a contiguous GPU buffer.
#[derive(Debug, Clone)]
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    dtype: DType,
    /// Byte offset within the contiguous GPU buffer.
    gpu_offset: usize,
    /// Size in bytes.
    size_bytes: usize,
}

/// Model weights loaded on GPU as `GpuTensor` views into contiguous buffers.
///
/// Each safetensors shard file is loaded into a single contiguous GPU allocation.
/// Individual tensors are zero-copy views (pointer + offset) into that buffer.
pub struct GpuWeights {
    /// Per-tensor views, keyed by tensor name.
    tensors: HashMap<String, GpuTensor>,
    /// Backing GPU buffers (one per shard file). Freed on drop.
    buffers: Vec<*mut u8>,
}

// Safety: GPU device pointers accessible from any host thread.
unsafe impl Send for GpuWeights {}
unsafe impl Sync for GpuWeights {}

impl GpuWeights {
    /// Load all weights from a model directory.
    ///
    /// Handles both single-file (`model.safetensors`) and sharded
    /// (`model.safetensors.index.json`) models.
    ///
    /// `stream` is used for async H2D copies (typically `GpuDevice.transfer_stream`).
    /// The caller must synchronize the stream before using the tensors.
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

    /// Load from a single safetensors file.
    pub fn from_single_file(path: impl AsRef<Path>, stream: CUstream) -> Result<Self> {
        let path = path.as_ref();
        let mut gw = Self {
            tensors: HashMap::new(),
            buffers: Vec::new(),
        };
        gw.load_shard(path, stream)?;
        Ok(gw)
    }

    /// Load from a sharded model (index.json).
    pub fn from_index(index_path: impl AsRef<Path>, stream: CUstream) -> Result<Self> {
        let index_path = index_path.as_ref();
        let dir = index_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("index file has no parent dir"))?;

        let index_data = std::fs::read_to_string(index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_data)?;

        // Extract unique shard filenames from the weight_map.
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
            buffers: Vec::new(),
        };

        for shard_name in &shard_files {
            let shard_path = dir.join(shard_name);
            gw.load_shard(&shard_path, stream)?;
        }

        Ok(gw)
    }

    /// Load a single shard file into a contiguous GPU buffer.
    fn load_shard(&mut self, path: &Path, stream: CUstream) -> Result<()> {
        // Memory-map the file.
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;

        // Parse safetensors header.
        let st = safetensors::SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow::anyhow!("{}: {}", path.display(), e))?;

        // First pass: compute total size and collect metadata.
        let mut metas = Vec::new();
        let mut total_bytes = 0usize;

        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("{}: {}", name, e))?;
            let dtype = safetensors_dtype(view.dtype())?;
            let size_bytes = view.data().len();
            let shape: Vec<usize> = view.shape().to_vec();

            metas.push(TensorMeta {
                name: name.to_string(),
                shape,
                dtype,
                gpu_offset: total_bytes,
                size_bytes,
            });
            // Align each tensor to 256 bytes for kernel compatibility.
            total_bytes += (size_bytes + 255) & !255;
        }

        if total_bytes == 0 {
            return Ok(());
        }

        // Allocate one contiguous GPU buffer for the entire shard.
        let gpu_buf = unsafe { driver::mem_alloc(total_bytes)? };
        self.buffers.push(gpu_buf);

        // Second pass: copy each tensor's data to its offset in the GPU buffer.
        for meta in &metas {
            let view = st.tensor(&meta.name).unwrap();
            let src_data = view.data();
            let dst = unsafe { gpu_buf.add(meta.gpu_offset) };

            unsafe {
                driver::memcpy_htod_async(dst, src_data.as_ptr(), meta.size_bytes, stream)?;
            }

            // Create GpuTensor view.
            let tensor = unsafe { GpuTensor::new(dst, &meta.shape, meta.dtype) };
            self.tensors.insert(meta.name.clone(), tensor);
        }

        tracing::info!(
            "Loaded shard {}: {} tensors, {:.1} MB GPU",
            path.display(),
            metas.len(),
            total_bytes as f64 / (1024.0 * 1024.0),
        );

        Ok(())
    }

    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<GpuTensor> {
        self.tensors.get(name).copied()
    }

    /// Remove and return a tensor by name.
    pub fn take(&mut self, name: &str) -> Result<GpuTensor> {
        self.tensors
            .remove(name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {name}"))
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
        let stripped: HashMap<String, GpuTensor> = self
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
        for buf in &self.buffers {
            unsafe {
                let _ = driver::mem_free(*buf);
            }
        }
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
            // Create a small safetensors file in a temp dir.
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            // Build a minimal safetensors file with two tensors.
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

            // Load via GpuWeights.
            let stream = init_cuda();
            let gw = GpuWeights::from_single_file(&path, stream).unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };

            assert_eq!(gw.len(), 2);
            assert!(gw.contains("weight_a"));
            assert!(gw.contains("weight_b"));

            let a = gw.get("weight_a").unwrap();
            assert_eq!(a.ndim(), 2);
            assert_eq!(a.dim(0), 2);
            assert_eq!(a.dim(1), 3);
            assert_eq!(a.dtype(), DType::F32);

            let b = gw.get("weight_b").unwrap();
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
            let gw = GpuWeights::from_dir(dir.path(), stream).unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };

            assert_eq!(gw.len(), 1);
            let t = gw.get("test.weight").unwrap();
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
            unsafe { driver::stream_synchronize(stream).unwrap() };

            // Strip prefix.
            gw.strip_prefix("model.");
            assert!(gw.contains("layer.weight"));
            assert!(!gw.contains("model.layer.weight"));

            // Take.
            let t = gw.take("layer.weight").unwrap();
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

            // BF16 data (raw bytes).
            let bf16_data: Vec<u8> = vec![0x00, 0x3F, 0x00, 0x40]; // 0.5 and 2.0 in BF16
            let tensors = vec![(
                "w",
                safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2], &bf16_data)
                    .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let gw = GpuWeights::from_single_file(&path, stream).unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };

            let w = gw.get("w").unwrap();
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
            unsafe { driver::stream_synchronize(stream).unwrap() };

            let mut names: Vec<&str> = gw.names().collect();
            names.sort();
            assert_eq!(names, vec!["a", "b"]);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }
    }
}
