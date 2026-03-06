// SPDX-License-Identifier: Apache-2.0
//! Model layers using `GpuTensor` — no candle dependency.
//!
//! These are minimal, inference-only layer types. Weights are stored as
//! `GpuTensor` (raw GPU pointers). Forward passes use cuBLAS GEMM from
//! the `GpuDevice` and fused CUDA kernels.

use anyhow::Result;

use crate::arena::ScratchArena;
use crate::cublas::CublasHandle;
use crate::tensor::GpuTensor;
use crate::weights::GpuWeights;

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Dense linear layer: y = x @ W^T + b
///
/// Weight is stored in `[out_features, in_features]` layout (NOT pre-transposed).
/// cuBLAS GEMM handles the transpose internally via `CUBLAS_OP_T`, which is
/// more efficient than a separate transpose copy.
pub struct Linear {
    pub weight: GpuTensor,       // [out_features, in_features]
    pub bias: Option<GpuTensor>, // [out_features]
}

impl Linear {
    /// Create from explicit weight and bias tensors.
    pub fn new(weight: GpuTensor, bias: Option<GpuTensor>) -> Self {
        debug_assert_eq!(weight.ndim(), 2);
        if let Some(ref b) = bias {
            debug_assert_eq!(b.ndim(), 1);
            debug_assert_eq!(b.dim(0), weight.dim(0));
        }
        Self { weight, bias }
    }

    /// Load from `GpuWeights` by prefix (e.g. "model.layers.0.self_attn.q_proj").
    pub fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");

        let weight = weights.take(&weight_name)?;
        let bias = if weights.contains(&bias_name) {
            Some(weights.take(&bias_name)?)
        } else {
            None
        };
        Ok(Self::new(weight, bias))
    }

    /// Forward: y = x @ W^T (+ bias)
    ///
    /// `x`: `[num_tokens, in_features]`
    /// Returns: `[num_tokens, out_features]` allocated from arena.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory. cuBLAS handle must be on the correct stream.
    pub unsafe fn forward(
        &self,
        x: GpuTensor,
        cublas: &CublasHandle,
        arena: &mut ScratchArena,
    ) -> GpuTensor {
        debug_assert_eq!(x.ndim(), 2);
        debug_assert_eq!(x.dim(1), self.weight.dim(1), "Linear: input dim mismatch");

        if let Some(bias) = self.bias {
            // Fused GEMM + bias via cublasLt epilogue (zero extra kernel launches).
            cublas.gemm_bias(x, self.weight, bias, arena)
        } else {
            cublas.gemm(x, self.weight, arena)
        }
    }

    pub fn out_features(&self) -> usize {
        self.weight.dim(0)
    }

    pub fn in_features(&self) -> usize {
        self.weight.dim(1)
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Token embedding lookup table.
///
/// Weight shape: `[vocab_size, hidden_size]`.
/// Forward gathers rows by token IDs.
pub struct Embedding {
    pub weight: GpuTensor, // [vocab_size, hidden_size]
}

impl Embedding {
    pub fn new(weight: GpuTensor) -> Self {
        debug_assert_eq!(weight.ndim(), 2);
        Self { weight }
    }

    /// Load from `GpuWeights` by prefix.
    pub fn load(weights: &mut GpuWeights, prefix: &str) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let weight = weights.take(&weight_name)?;
        Ok(Self::new(weight))
    }

    pub fn vocab_size(&self) -> usize {
        self.weight.dim(0)
    }

    pub fn hidden_size(&self) -> usize {
        self.weight.dim(1)
    }

    /// Forward: gather embedding rows by token IDs.
    ///
    /// `input_ids`: `[num_tokens]` (U32 on GPU)
    /// Returns: `[num_tokens, hidden_size]` allocated from arena.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory. Stream must be valid.
    /// This currently uses a simple gather kernel (TODO: implement via CUDA kernel).
    pub unsafe fn forward(
        &self,
        _input_ids: GpuTensor,
        _arena: &mut ScratchArena,
        _stream: cudarc::driver::sys::CUstream,
    ) -> GpuTensor {
        // TODO: implement embedding gather kernel.
        // For now, return a placeholder. The kernel is trivial:
        // one thread per (token, dim) reads weight[input_ids[token], dim].
        todo!("embedding gather kernel not yet implemented")
    }
}

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization.
///
/// `y = x / sqrt(mean(x^2) + eps) * weight`
///
/// Weight shape: `[hidden_size]`.
pub struct RmsNorm {
    pub weight: GpuTensor, // [hidden_size]
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: GpuTensor, eps: f32) -> Self {
        debug_assert_eq!(weight.ndim(), 1);
        Self { weight, eps }
    }

    /// Load from `GpuWeights` by prefix.
    pub fn load(weights: &mut GpuWeights, prefix: &str, eps: f32) -> Result<Self> {
        let weight_name = format!("{prefix}.weight");
        let weight = weights.take(&weight_name)?;
        Ok(Self::new(weight, eps))
    }

    pub fn hidden_size(&self) -> usize {
        self.weight.dim(0)
    }

    // forward() will be implemented when we wire the existing CUDA kernels
    // from vllm-kernels to accept GpuTensor raw pointers.
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_linear_dimensions() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[512, 4096], DType::BF16) };
        let linear = Linear::new(w, None);
        assert_eq!(linear.out_features(), 512);
        assert_eq!(linear.in_features(), 4096);
    }

    #[test]
    fn test_linear_with_bias() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[256, 128], DType::F16) };
        let b = unsafe { GpuTensor::new(0x2000 as *mut u8, &[256], DType::F16) };
        let linear = Linear::new(w, Some(b));
        assert!(linear.bias.is_some());
        assert_eq!(linear.out_features(), 256);
    }

    #[test]
    fn test_embedding_dimensions() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[32000, 4096], DType::BF16) };
        let emb = Embedding::new(w);
        assert_eq!(emb.vocab_size(), 32000);
        assert_eq!(emb.hidden_size(), 4096);
    }

    #[test]
    fn test_rms_norm_dimensions() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4096], DType::BF16) };
        let norm = RmsNorm::new(w, 1e-5);
        assert_eq!(norm.hidden_size(), 4096);
        assert_eq!(norm.eps, 1e-5);
    }

    #[test]
    fn test_rms_norm_eps_values() {
        let w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[128], DType::F16) };
        let norm = RmsNorm::new(w, 1e-6);
        assert!((norm.eps - 1e-6).abs() < 1e-10);
    }

    // GPU tests for layer loading and forward passes.
    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::*;
        use crate::DType;
        use crate::driver;

        fn init_cuda() -> cudarc::driver::sys::CUstream {
            unsafe {
                driver::init().expect("CUDA init");
                let dev = driver::device_get(0).expect("device");
                let _ctx = driver::ctx_create(dev).expect("context");
                driver::stream_create().expect("stream")
            }
        }

        #[test]
        fn test_linear_load_from_safetensors() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            // Create a safetensors file with linear weight and bias.
            let weight_data: Vec<f32> = vec![1.0; 8]; // [2, 4]
            let bias_data: Vec<f32> = vec![0.5; 2]; // [2]
            let w_bytes: Vec<u8> = weight_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let b_bytes: Vec<u8> = bias_data.iter().flat_map(|f| f.to_le_bytes()).collect();

            let tensors = vec![
                (
                    "proj.weight",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![2, 4],
                        &w_bytes,
                    )
                    .unwrap(),
                ),
                (
                    "proj.bias",
                    safetensors::tensor::TensorView::new(
                        safetensors::Dtype::F32,
                        vec![2],
                        &b_bytes,
                    )
                    .unwrap(),
                ),
            ];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };

            let linear = Linear::load(&mut gw, "proj").unwrap();
            assert_eq!(linear.out_features(), 2);
            assert_eq!(linear.in_features(), 4);
            assert!(linear.bias.is_some());

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_linear_forward_f32() {
            let stream = init_cuda();
            unsafe {
                let cublas = CublasHandle::new(stream).unwrap();
                let mut arena = ScratchArena::new(4 * 1024 * 1024).unwrap();

                // Weight [2, 3] = [[1,0,0],[0,1,0]] (identity-ish)
                let host_w = driver::mem_alloc_host(24).unwrap();
                std::slice::from_raw_parts_mut(host_w as *mut f32, 6)
                    .copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
                let gpu_w = driver::mem_alloc(24).unwrap();
                driver::memcpy_htod_async(gpu_w, host_w, 24, stream).unwrap();

                let w = GpuTensor::new(gpu_w, &[2, 3], DType::F32);
                let linear = Linear::new(w, None);

                // Input [4, 3] = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
                let host_x = driver::mem_alloc_host(48).unwrap();
                std::slice::from_raw_parts_mut(host_x as *mut f32, 12).copy_from_slice(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ]);
                let gpu_x = driver::mem_alloc(48).unwrap();
                driver::memcpy_htod_async(gpu_x, host_x, 48, stream).unwrap();
                let x = GpuTensor::new(gpu_x, &[4, 3], DType::F32);

                // Forward: x @ W^T = [4,3] @ [3,2] = [4,2]
                // Expected: [[1,2],[4,5],[7,8],[10,11]]
                let y = linear.forward(x, &cublas, &mut arena);
                assert_eq!(y.dim(0), 4);
                assert_eq!(y.dim(1), 2);

                let host_y = driver::mem_alloc_host(32).unwrap();
                driver::memcpy_dtoh_async(host_y, y.raw_ptr(), 32, stream).unwrap();
                driver::stream_synchronize(stream).unwrap();

                let result = std::slice::from_raw_parts(host_y as *const f32, 8);
                let expected = [1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0];
                for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-3,
                        "linear forward mismatch at {i}: got {got}, expected {exp}"
                    );
                }

                driver::mem_free_host(host_w).unwrap();
                driver::mem_free_host(host_x).unwrap();
                driver::mem_free_host(host_y).unwrap();
                driver::mem_free(gpu_w).unwrap();
                driver::mem_free(gpu_x).unwrap();
                driver::stream_destroy(stream).unwrap();
            }
        }

        #[test]
        fn test_rms_norm_load() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data: Vec<f32> = vec![1.0; 128];
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let tensors = vec![(
                "norm.weight",
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![128], &bytes)
                    .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };

            let norm = RmsNorm::load(&mut gw, "norm", 1e-5).unwrap();
            assert_eq!(norm.hidden_size(), 128);
            assert!((norm.eps - 1e-5).abs() < 1e-10);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }

        #[test]
        fn test_embedding_load() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("model.safetensors");

            let data: Vec<f32> = vec![0.0; 100 * 32]; // [100, 32]
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let tensors = vec![(
                "embed.weight",
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![100, 32],
                    &bytes,
                )
                .unwrap(),
            )];
            safetensors::serialize_to_file(tensors, None, &path).unwrap();

            let stream = init_cuda();
            let mut gw = GpuWeights::from_single_file(&path, stream).unwrap();
            unsafe { driver::stream_synchronize(stream).unwrap() };

            let emb = Embedding::load(&mut gw, "embed").unwrap();
            assert_eq!(emb.vocab_size(), 100);
            assert_eq!(emb.hidden_size(), 32);

            unsafe { driver::stream_destroy(stream).unwrap() };
        }
    }
}
