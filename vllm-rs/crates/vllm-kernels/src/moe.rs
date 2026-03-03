// SPDX-License-Identifier: Apache-2.0
//! MoE (Mixture of Experts) kernels.
//!
//! Trait abstraction for top-k softmax gating and weighted reduction.
//! Port of: `csrc/moe/topk_softmax_kernels.cu`, `csrc/moe/moe_align_sum_kernels.cu`

use candle_core::{DType, Tensor};

use crate::error::KernelResult;

/// MoE kernel interface.
///
/// Provides fused top-k softmax gating and weighted reduction across experts.
pub trait MoeKernels: Send + Sync {
    /// Top-k softmax gating: compute softmax over router logits, then select
    /// top-k experts per token.
    ///
    /// * `router_logits` — `[num_tokens, num_experts]` (any float dtype)
    /// * `top_k` — number of experts to select per token
    /// * `renormalize` — if true, renormalize weights to sum to 1
    ///
    /// Returns `(topk_weights, topk_ids)`:
    /// * `topk_weights` — `[num_tokens, top_k]` f32
    /// * `topk_ids` — `[num_tokens, top_k]` u32
    fn topk_softmax(
        &self,
        router_logits: &Tensor,
        top_k: usize,
        renormalize: bool,
    ) -> KernelResult<(Tensor, Tensor)>;

    /// Weighted sum across top-k expert outputs.
    ///
    /// * `input` — `[num_tokens, top_k, hidden_size]`
    /// * `top_k` — number of experts per token
    ///
    /// Returns `[num_tokens, hidden_size]`
    fn moe_sum(&self, input: &Tensor, top_k: usize) -> KernelResult<Tensor>;
}

// ---------------------------------------------------------------------------
// CPU implementation
// ---------------------------------------------------------------------------

/// CPU implementation of MoE kernels (for testing and CPU fallback).
pub struct CpuMoeKernels;

impl MoeKernels for CpuMoeKernels {
    fn topk_softmax(
        &self,
        router_logits: &Tensor,
        top_k: usize,
        renormalize: bool,
    ) -> KernelResult<(Tensor, Tensor)> {
        let (num_tokens, num_experts) = router_logits.dims2()?;

        // Softmax
        let logits_f32 = router_logits.to_dtype(DType::F32)?;
        let max_vals = logits_f32.max_keepdim(candle_core::D::Minus1)?;
        let shifted = logits_f32.broadcast_sub(&max_vals)?;
        let exp = shifted.exp()?;
        let sum = exp.sum_keepdim(candle_core::D::Minus1)?;
        let probs = exp.broadcast_div(&sum)?;
        let probs_vec = probs.to_vec2::<f32>()?;

        // Top-k selection per token
        let mut weights_data = vec![0.0f32; num_tokens * top_k];
        let mut ids_data = vec![0u32; num_tokens * top_k];

        for tok in 0..num_tokens {
            let token_probs = &probs_vec[tok];
            let mut indexed: Vec<(usize, f32)> = token_probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(top_k);

            let total: f32 = indexed.iter().map(|(_, p)| p).sum();
            let scale = if renormalize && total > 0.0 {
                1.0 / total
            } else {
                1.0
            };

            for (k_idx, &(expert_idx, prob)) in indexed.iter().enumerate() {
                weights_data[tok * top_k + k_idx] = prob * scale;
                ids_data[tok * top_k + k_idx] = expert_idx as u32;
            }
        }

        let device = router_logits.device();
        let _ = num_experts;
        let topk_weights = Tensor::from_slice(&weights_data, (num_tokens, top_k), device)?;
        let topk_ids = Tensor::from_slice(&ids_data, (num_tokens, top_k), device)?;

        Ok((topk_weights, topk_ids))
    }

    fn moe_sum(&self, input: &Tensor, _top_k: usize) -> KernelResult<Tensor> {
        // input: [num_tokens, top_k, hidden_size] -> sum over dim 1
        Ok(input.sum(1)?)
    }
}

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda_ffi {
    unsafe extern "C" {
        pub fn topk_softmax_f32(
            topk_weights: *mut f32,
            topk_ids: *mut i32,
            workspace: *mut f32,
            gating_output: *const f32,
            num_tokens: i32,
            num_experts: i32,
            topk: i32,
            renormalize: i32,
        );
        pub fn topk_softmax_bf16(
            topk_weights: *mut f32,
            topk_ids: *mut i32,
            workspace: *mut f32,
            gating_output: *const u16, // __nv_bfloat16
            num_tokens: i32,
            num_experts: i32,
            topk: i32,
            renormalize: i32,
        );
        pub fn topk_softmax_f16(
            topk_weights: *mut f32,
            topk_ids: *mut i32,
            workspace: *mut f32,
            gating_output: *const u16, // __half
            num_tokens: i32,
            num_experts: i32,
            topk: i32,
            renormalize: i32,
        );

        pub fn moe_sum_f32(
            out: *mut f32,
            input: *const f32,
            num_tokens: i32,
            hidden_size: i32,
            topk: i32,
        );
        pub fn moe_sum_f16(
            out: *mut u16,
            input: *const u16,
            num_tokens: i32,
            hidden_size: i32,
            topk: i32,
        );
        pub fn moe_sum_bf16(
            out: *mut u16,
            input: *const u16,
            num_tokens: i32,
            hidden_size: i32,
            topk: i32,
        );
    }
}

/// CUDA implementation of MoE kernels.
#[cfg(feature = "cuda")]
pub struct CudaMoeKernels;

#[cfg(feature = "cuda")]
impl CudaMoeKernels {
    fn device_ptr_of<T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
        tensor: &Tensor,
    ) -> KernelResult<usize> {
        use crate::error::KernelError;
        use cudarc::driver::DevicePtr;
        let cuda_dev = tensor
            .device()
            .as_cuda_device()
            .map_err(|e| KernelError::Other(format!("{e}")))?;
        let stream = cuda_dev.cuda_stream();
        let (storage, layout) = tensor.storage_and_layout();
        match &*storage {
            candle_core::Storage::Cuda(cuda_storage) => {
                let slice = cuda_storage.as_cuda_slice::<T>()?;
                let view = slice.slice(layout.start_offset()..);
                let (ptr, _sync_guard) = view.device_ptr(&stream);
                Ok(ptr as usize)
            }
            _ => Err(KernelError::Other("expected CUDA tensor".to_string())),
        }
    }
}

#[cfg(feature = "cuda")]
impl MoeKernels for CudaMoeKernels {
    fn topk_softmax(
        &self,
        router_logits: &Tensor,
        top_k: usize,
        renormalize: bool,
    ) -> KernelResult<(Tensor, Tensor)> {
        use crate::error::KernelError;
        let (num_tokens, num_experts) = router_logits.dims2()?;
        let router_logits = router_logits.contiguous()?;
        let device = router_logits.device();

        // Allocate output tensors
        let topk_weights = Tensor::zeros((num_tokens, top_k), DType::F32, device)?;
        let topk_ids = Tensor::zeros((num_tokens, top_k), DType::U32, device)?;

        // Workspace for non-power-of-2 fallback path
        let is_pow2 = num_experts > 0 && (num_experts & (num_experts - 1)) == 0;
        let needs_workspace = !is_pow2 || num_experts > 256;
        let workspace_size = if needs_workspace {
            num_tokens * num_experts
        } else {
            0
        };
        let workspace = Tensor::zeros(workspace_size, DType::F32, device)?;

        let w_ptr = Self::device_ptr_of::<f32>(&topk_weights)?;
        // topk_ids is U32 but the CUDA kernel writes int (i32). Both are 32-bit.
        let id_ptr = Self::device_ptr_of::<u32>(&topk_ids)?;
        let ws_ptr = Self::device_ptr_of::<f32>(&workspace)?;
        let renorm = if renormalize { 1i32 } else { 0i32 };

        match router_logits.dtype() {
            DType::F32 => {
                let g = Self::device_ptr_of::<f32>(&router_logits)?;
                unsafe {
                    cuda_ffi::topk_softmax_f32(
                        w_ptr as *mut f32,
                        id_ptr as *mut i32,
                        ws_ptr as *mut f32,
                        g as *const f32,
                        num_tokens as i32,
                        num_experts as i32,
                        top_k as i32,
                        renorm,
                    );
                }
            }
            DType::BF16 => {
                let g = Self::device_ptr_of::<half::bf16>(&router_logits)?;
                unsafe {
                    cuda_ffi::topk_softmax_bf16(
                        w_ptr as *mut f32,
                        id_ptr as *mut i32,
                        ws_ptr as *mut f32,
                        g as *const u16,
                        num_tokens as i32,
                        num_experts as i32,
                        top_k as i32,
                        renorm,
                    );
                }
            }
            DType::F16 => {
                let g = Self::device_ptr_of::<half::f16>(&router_logits)?;
                unsafe {
                    cuda_ffi::topk_softmax_f16(
                        w_ptr as *mut f32,
                        id_ptr as *mut i32,
                        ws_ptr as *mut f32,
                        g as *const u16,
                        num_tokens as i32,
                        num_experts as i32,
                        top_k as i32,
                        renorm,
                    );
                }
            }
            _ => {
                return Err(KernelError::Other(format!(
                    "topk_softmax: unsupported dtype {:?}",
                    router_logits.dtype()
                )));
            }
        }

        Ok((topk_weights, topk_ids))
    }

    fn moe_sum(&self, input: &Tensor, top_k: usize) -> KernelResult<Tensor> {
        use crate::error::KernelError;
        let dims = input.shape().dims();
        if dims.len() != 3 {
            return Err(KernelError::Other(format!(
                "moe_sum: expected 3D input [tokens, topk, hidden], got {:?}",
                input.shape()
            )));
        }
        let num_tokens = dims[0];
        let hidden_size = dims[2];
        let input = input.contiguous()?;
        let device = input.device();

        let out = Tensor::zeros((num_tokens, hidden_size), input.dtype(), device)?;

        match input.dtype() {
            DType::F32 => {
                let o = Self::device_ptr_of::<f32>(&out)?;
                let i = Self::device_ptr_of::<f32>(&input)?;
                unsafe {
                    cuda_ffi::moe_sum_f32(
                        o as *mut f32,
                        i as *const f32,
                        num_tokens as i32,
                        hidden_size as i32,
                        top_k as i32,
                    );
                }
            }
            DType::F16 => {
                let o = Self::device_ptr_of::<half::f16>(&out)?;
                let i = Self::device_ptr_of::<half::f16>(&input)?;
                unsafe {
                    cuda_ffi::moe_sum_f16(
                        o as *mut u16,
                        i as *const u16,
                        num_tokens as i32,
                        hidden_size as i32,
                        top_k as i32,
                    );
                }
            }
            DType::BF16 => {
                let o = Self::device_ptr_of::<half::bf16>(&out)?;
                let i = Self::device_ptr_of::<half::bf16>(&input)?;
                unsafe {
                    cuda_ffi::moe_sum_bf16(
                        o as *mut u16,
                        i as *const u16,
                        num_tokens as i32,
                        hidden_size as i32,
                        top_k as i32,
                    );
                }
            }
            _ => return CpuMoeKernels.moe_sum(&input, top_k),
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_cpu_topk_softmax_basic() {
        let kernels = CpuMoeKernels;
        let device = Device::Cpu;

        // 2 tokens, 4 experts, top_k=2
        let logits =
            Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], &device).unwrap();

        let (weights, ids) = kernels.topk_softmax(&logits, 2, true).unwrap();
        assert_eq!(weights.dims(), &[2, 2]);
        assert_eq!(ids.dims(), &[2, 2]);

        let ids_vec = ids.to_vec2::<u32>().unwrap();
        // Token 0: experts 3,2 (highest logits)
        assert_eq!(ids_vec[0][0], 3);
        assert_eq!(ids_vec[0][1], 2);
        // Token 1: experts 0,1 (highest logits)
        assert_eq!(ids_vec[1][0], 0);
        assert_eq!(ids_vec[1][1], 1);

        // Weights should sum to 1 (renormalized)
        let w_vec = weights.to_vec2::<f32>().unwrap();
        let sum0: f32 = w_vec[0].iter().sum();
        let sum1: f32 = w_vec[1].iter().sum();
        assert!((sum0 - 1.0).abs() < 1e-5);
        assert!((sum1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_topk_softmax_no_renormalize() {
        let kernels = CpuMoeKernels;
        let device = Device::Cpu;

        let logits = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device).unwrap();

        let (weights, _ids) = kernels.topk_softmax(&logits, 2, false).unwrap();
        let w_vec = weights.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Without renormalization, weights are raw softmax probabilities
        // sum should be less than 1 (only top-2 of 4)
        let sum: f32 = w_vec.iter().sum();
        assert!(sum < 1.0);
    }

    #[test]
    fn test_cpu_moe_sum() {
        let kernels = CpuMoeKernels;
        let device = Device::Cpu;

        // 2 tokens, top_k=2, hidden=3
        let input = Tensor::new(
            &[
                [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            &device,
        )
        .unwrap();

        let out = kernels.moe_sum(&input, 2).unwrap();
        assert_eq!(out.dims(), &[2, 3]);

        let vals = out.to_vec2::<f32>().unwrap();
        assert!((vals[0][0] - 5.0).abs() < 1e-5); // 1+4
        assert!((vals[0][1] - 7.0).abs() < 1e-5); // 2+5
        assert!((vals[0][2] - 9.0).abs() < 1e-5); // 3+6
        assert!((vals[1][0] - 17.0).abs() < 1e-5); // 7+10
    }

    // -----------------------------------------------------------------------
    // CUDA tests
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device required for this test")
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_topk_softmax_f32() {
        let dev = cuda_device();
        let cpu_kernels = CpuMoeKernels;
        let cuda_kernels = CudaMoeKernels;

        // Random router logits: 8 tokens, 64 experts, top_k=4
        let logits_cpu = Tensor::randn(0f32, 1.0, &[8, 64], &Device::Cpu).unwrap();
        let logits_gpu = logits_cpu.to_device(&dev).unwrap();

        let (cpu_w, cpu_ids) = cpu_kernels.topk_softmax(&logits_cpu, 4, true).unwrap();
        let (cuda_w, cuda_ids) = cuda_kernels.topk_softmax(&logits_gpu, 4, true).unwrap();

        let cuda_w_cpu = cuda_w.to_device(&Device::Cpu).unwrap();
        let cuda_ids_cpu = cuda_ids.to_device(&Device::Cpu).unwrap();

        // Compare expert IDs — should match
        let cpu_ids_vec = cpu_ids.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let cuda_ids_vec = cuda_ids_cpu
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        assert_eq!(cpu_ids_vec, cuda_ids_vec, "topk expert IDs mismatch");

        // Compare weights with tolerance
        let cpu_w_vec = cpu_w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cuda_w_vec = cuda_w_cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (c, g)) in cpu_w_vec.iter().zip(cuda_w_vec.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-4,
                "topk weight mismatch at {i}: cpu={c} cuda={g}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_topk_softmax_pow2_experts() {
        // Test various power-of-2 expert counts that hit the fast path
        let dev = cuda_device();
        let cuda_kernels = CudaMoeKernels;
        let cpu_kernels = CpuMoeKernels;

        for num_experts in [4, 8, 16, 32, 64, 128] {
            let logits_cpu = Tensor::randn(0f32, 1.0, &[4, num_experts], &Device::Cpu).unwrap();
            let logits_gpu = logits_cpu.to_device(&dev).unwrap();

            let (cpu_w, cpu_ids) = cpu_kernels.topk_softmax(&logits_cpu, 2, true).unwrap();
            let (cuda_w, cuda_ids) = cuda_kernels.topk_softmax(&logits_gpu, 2, true).unwrap();

            let cuda_ids_cpu = cuda_ids.to_device(&Device::Cpu).unwrap();
            let cuda_w_cpu = cuda_w.to_device(&Device::Cpu).unwrap();

            let cpu_ids_vec = cpu_ids.flatten_all().unwrap().to_vec1::<u32>().unwrap();
            let cuda_ids_vec = cuda_ids_cpu
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(
                cpu_ids_vec, cuda_ids_vec,
                "expert IDs mismatch for {num_experts} experts"
            );

            let cpu_w_vec = cpu_w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let cuda_w_vec = cuda_w_cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (i, (c, g)) in cpu_w_vec.iter().zip(cuda_w_vec.iter()).enumerate() {
                assert!(
                    (c - g).abs() < 1e-4,
                    "weight mismatch for {num_experts} experts at {i}: cpu={c} cuda={g}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_topk_softmax_non_pow2() {
        // Non-power-of-2: hits the moeSoftmax + moeTopK fallback
        let dev = cuda_device();
        let cuda_kernels = CudaMoeKernels;
        let cpu_kernels = CpuMoeKernels;

        let logits_cpu = Tensor::randn(0f32, 1.0, &[4, 13], &Device::Cpu).unwrap();
        let logits_gpu = logits_cpu.to_device(&dev).unwrap();

        let (cpu_w, cpu_ids) = cpu_kernels.topk_softmax(&logits_cpu, 2, true).unwrap();
        let (cuda_w, cuda_ids) = cuda_kernels.topk_softmax(&logits_gpu, 2, true).unwrap();

        let cuda_ids_cpu = cuda_ids.to_device(&Device::Cpu).unwrap();
        let cuda_w_cpu = cuda_w.to_device(&Device::Cpu).unwrap();

        let cpu_ids_vec = cpu_ids.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        let cuda_ids_vec = cuda_ids_cpu
            .flatten_all()
            .unwrap()
            .to_vec1::<u32>()
            .unwrap();
        assert_eq!(
            cpu_ids_vec, cuda_ids_vec,
            "expert IDs mismatch for non-pow2"
        );

        let cpu_w_vec = cpu_w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cuda_w_vec = cuda_w_cpu.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (c, g)) in cpu_w_vec.iter().zip(cuda_w_vec.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-4,
                "weight mismatch non-pow2 at {i}: cpu={c} cuda={g}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_moe_sum_f32() {
        let dev = cuda_device();
        let cpu_kernels = CpuMoeKernels;
        let cuda_kernels = CudaMoeKernels;

        // 4 tokens, top_k=2, hidden=128
        let input_cpu = Tensor::randn(0f32, 1.0, &[4, 2, 128], &Device::Cpu).unwrap();
        let input_gpu = input_cpu.to_device(&dev).unwrap();

        let cpu_out = cpu_kernels.moe_sum(&input_cpu, 2).unwrap();
        let cuda_out = cuda_kernels.moe_sum(&input_gpu, 2).unwrap();
        let cuda_out_cpu = cuda_out.to_device(&Device::Cpu).unwrap();

        let cpu_vals = cpu_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cuda_vals = cuda_out_cpu
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(cpu_vals.len(), cuda_vals.len());
        for (i, (c, g)) in cpu_vals.iter().zip(cuda_vals.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-4,
                "moe_sum mismatch at {i}: cpu={c} cuda={g}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_moe_sum_bf16() {
        let dev = cuda_device();
        let cpu_kernels = CpuMoeKernels;
        let cuda_kernels = CudaMoeKernels;

        let input_cpu = Tensor::randn(0f32, 1.0, &[4, 2, 128], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let input_gpu = input_cpu.to_device(&dev).unwrap();

        let cpu_out = cpu_kernels.moe_sum(&input_cpu, 2).unwrap();
        let cuda_out = cuda_kernels.moe_sum(&input_gpu, 2).unwrap();
        let cuda_out_cpu = cuda_out.to_device(&Device::Cpu).unwrap();

        let cpu_vals = cpu_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda_vals = cuda_out_cpu
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, (c, g)) in cpu_vals.iter().zip(cuda_vals.iter()).enumerate() {
            assert!(
                (c - g).abs() < 0.15,
                "moe_sum bf16 mismatch at {i}: cpu={c} cuda={g}"
            );
        }
    }
}
