// SPDX-License-Identifier: Apache-2.0
//! Activation kernels.
//!
//! Trait abstraction for fused activation kernels (SiLU+mul, GELU+mul).
//! Port of: `csrc/activation_kernels.cu`
//!
//! ## Simplifications vs Python vLLM
//!
//! The CUDA kernels here are simplified compared to Python vLLM's:
//! - Scalar loads instead of vectorized int4/u32x8 loads (~2-3x slower on large hidden)
//! - No packed half2/bfloat162 arithmetic path
//! - Separate gate/up pointers (matching Rust trait) instead of concatenated [2*d] input
//!
//! These will be upgraded to match Python vLLM's performance in a follow-up.

use candle_core::Tensor;

use crate::error::KernelResult;

/// Activation kernel interface.
///
/// Provides fused activation+multiply operations that are common in
/// transformer FFN blocks (gate projection * up projection).
pub trait ActivationKernels: Send + Sync {
    /// Fused SiLU and element-wise multiply.
    ///
    /// Computes `silu(gate) * up` where gate and up are the two halves
    /// of the input tensor split along the last dimension.
    ///
    /// Port of: `void silu_and_mul(out, input)` where input is [batch, 2*dim]
    fn silu_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor>;

    /// Fused GELU (tanh approx) and element-wise multiply.
    fn gelu_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor>;

    /// Fused GELU (new/exact) and element-wise multiply.
    fn gelu_new_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor>;
}

/// CPU implementation of activation kernels (for testing).
pub struct CpuActivationKernels;

impl ActivationKernels for CpuActivationKernels {
    fn silu_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor> {
        let activated = gate.silu()?;
        let out = activated.mul(up)?;
        Ok(out)
    }

    fn gelu_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor> {
        let activated = gate.gelu()?;
        let out = activated.mul(up)?;
        Ok(out)
    }

    fn gelu_new_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor> {
        let activated = gate.gelu_erf()?;
        let out = activated.mul(up)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda_ffi {
    unsafe extern "C" {
        // SiLU + mul
        pub fn silu_and_mul_f32(
            out: *mut f32,
            gate: *const f32,
            up: *const f32,
            num_tokens: i32,
            d: i32,
        );
        pub fn silu_and_mul_f16(
            out: *mut u16,
            gate: *const u16,
            up: *const u16,
            num_tokens: i32,
            d: i32,
        );
        pub fn silu_and_mul_bf16(
            out: *mut u16,
            gate: *const u16,
            up: *const u16,
            num_tokens: i32,
            d: i32,
        );
        // GELU (tanh approx) + mul
        pub fn gelu_and_mul_f32(
            out: *mut f32,
            gate: *const f32,
            up: *const f32,
            num_tokens: i32,
            d: i32,
        );
        pub fn gelu_and_mul_f16(
            out: *mut u16,
            gate: *const u16,
            up: *const u16,
            num_tokens: i32,
            d: i32,
        );
        pub fn gelu_and_mul_bf16(
            out: *mut u16,
            gate: *const u16,
            up: *const u16,
            num_tokens: i32,
            d: i32,
        );
        // GELU (exact/erf) + mul
        pub fn gelu_new_and_mul_f32(
            out: *mut f32,
            gate: *const f32,
            up: *const f32,
            num_tokens: i32,
            d: i32,
        );
        pub fn gelu_new_and_mul_f16(
            out: *mut u16,
            gate: *const u16,
            up: *const u16,
            num_tokens: i32,
            d: i32,
        );
        pub fn gelu_new_and_mul_bf16(
            out: *mut u16,
            gate: *const u16,
            up: *const u16,
            num_tokens: i32,
            d: i32,
        );
    }
}

/// CUDA implementation of activation kernels.
#[cfg(feature = "cuda")]
pub struct CudaActivationKernels;

#[cfg(feature = "cuda")]
impl CudaActivationKernels {
    /// Extract a raw device pointer (as usize) from a contiguous CUDA tensor.
    fn device_ptr_of<T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
        tensor: &Tensor,
    ) -> KernelResult<usize> {
        use cudarc::driver::DevicePtr;
        let cuda_dev = tensor
            .device()
            .as_cuda_device()
            .map_err(|e| crate::error::KernelError::Other(format!("{e}")))?;
        let stream = cuda_dev.cuda_stream();
        let (storage, layout) = tensor.storage_and_layout();
        match &*storage {
            candle_core::Storage::Cuda(cuda_storage) => {
                let slice = cuda_storage.as_cuda_slice::<T>()?;
                let view = slice.slice(layout.start_offset()..);
                let (ptr, _sync_guard) = view.device_ptr(&stream);
                Ok(ptr as usize)
            }
            _ => Err(crate::error::KernelError::Other(
                "expected CUDA tensor".to_string(),
            )),
        }
    }

    /// Shared helper: validate shapes, make contiguous, compute dimensions.
    fn prepare_tensors(
        gate: &Tensor,
        up: &Tensor,
    ) -> KernelResult<(Tensor, Tensor, Tensor, usize, usize)> {
        if gate.shape() != up.shape() {
            return Err(crate::error::KernelError::Other(format!(
                "gate shape {:?} != up shape {:?}",
                gate.shape(),
                up.shape()
            )));
        }
        let dims = gate.shape().dims();
        let d = *dims
            .last()
            .ok_or_else(|| crate::error::KernelError::Other("empty gate tensor".to_string()))?;
        let num_tokens: usize = dims[..dims.len() - 1].iter().product();

        let gate = gate.contiguous()?;
        let up = up.contiguous()?;
        let out = Tensor::zeros(gate.shape(), gate.dtype(), gate.device())?;
        Ok((gate, up, out, num_tokens, d))
    }
}

#[cfg(feature = "cuda")]
impl ActivationKernels for CudaActivationKernels {
    fn silu_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor> {
        use candle_core::DType;

        let (gate, up, out, num_tokens, d) = Self::prepare_tensors(gate, up)?;

        match gate.dtype() {
            DType::F32 => {
                let o = Self::device_ptr_of::<f32>(&out)?;
                let g = Self::device_ptr_of::<f32>(&gate)?;
                let u = Self::device_ptr_of::<f32>(&up)?;
                unsafe {
                    cuda_ffi::silu_and_mul_f32(
                        o as *mut f32,
                        g as *const f32,
                        u as *const f32,
                        num_tokens as i32,
                        d as i32,
                    );
                }
            }
            DType::F16 => {
                let o = Self::device_ptr_of::<half::f16>(&out)?;
                let g = Self::device_ptr_of::<half::f16>(&gate)?;
                let u = Self::device_ptr_of::<half::f16>(&up)?;
                unsafe {
                    cuda_ffi::silu_and_mul_f16(
                        o as *mut u16,
                        g as *const u16,
                        u as *const u16,
                        num_tokens as i32,
                        d as i32,
                    );
                }
            }
            DType::BF16 => {
                let o = Self::device_ptr_of::<half::bf16>(&out)?;
                let g = Self::device_ptr_of::<half::bf16>(&gate)?;
                let u = Self::device_ptr_of::<half::bf16>(&up)?;
                unsafe {
                    cuda_ffi::silu_and_mul_bf16(
                        o as *mut u16,
                        g as *const u16,
                        u as *const u16,
                        num_tokens as i32,
                        d as i32,
                    );
                }
            }
            _ => return CpuActivationKernels.silu_and_mul(&gate, &up),
        }
        Ok(out)
    }

    fn gelu_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor> {
        use candle_core::DType;

        let (gate, up, out, num_tokens, d) = Self::prepare_tensors(gate, up)?;

        match gate.dtype() {
            DType::F32 => {
                let o = Self::device_ptr_of::<f32>(&out)?;
                let g = Self::device_ptr_of::<f32>(&gate)?;
                let u = Self::device_ptr_of::<f32>(&up)?;
                unsafe {
                    cuda_ffi::gelu_and_mul_f32(
                        o as *mut f32,
                        g as *const f32,
                        u as *const f32,
                        num_tokens as i32,
                        d as i32,
                    );
                }
            }
            DType::F16 => {
                let o = Self::device_ptr_of::<half::f16>(&out)?;
                let g = Self::device_ptr_of::<half::f16>(&gate)?;
                let u = Self::device_ptr_of::<half::f16>(&up)?;
                unsafe {
                    cuda_ffi::gelu_and_mul_f16(
                        o as *mut u16,
                        g as *const u16,
                        u as *const u16,
                        num_tokens as i32,
                        d as i32,
                    );
                }
            }
            DType::BF16 => {
                let o = Self::device_ptr_of::<half::bf16>(&out)?;
                let g = Self::device_ptr_of::<half::bf16>(&gate)?;
                let u = Self::device_ptr_of::<half::bf16>(&up)?;
                unsafe {
                    cuda_ffi::gelu_and_mul_bf16(
                        o as *mut u16,
                        g as *const u16,
                        u as *const u16,
                        num_tokens as i32,
                        d as i32,
                    );
                }
            }
            _ => return CpuActivationKernels.gelu_and_mul(&gate, &up),
        }
        Ok(out)
    }

    fn gelu_new_and_mul(&self, gate: &Tensor, up: &Tensor) -> KernelResult<Tensor> {
        use candle_core::DType;

        let (gate, up, out, num_tokens, d) = Self::prepare_tensors(gate, up)?;

        match gate.dtype() {
            DType::F32 => {
                let o = Self::device_ptr_of::<f32>(&out)?;
                let g = Self::device_ptr_of::<f32>(&gate)?;
                let u = Self::device_ptr_of::<f32>(&up)?;
                unsafe {
                    cuda_ffi::gelu_new_and_mul_f32(
                        o as *mut f32,
                        g as *const f32,
                        u as *const f32,
                        num_tokens as i32,
                        d as i32,
                    );
                }
            }
            DType::F16 => {
                let o = Self::device_ptr_of::<half::f16>(&out)?;
                let g = Self::device_ptr_of::<half::f16>(&gate)?;
                let u = Self::device_ptr_of::<half::f16>(&up)?;
                unsafe {
                    cuda_ffi::gelu_new_and_mul_f16(
                        o as *mut u16,
                        g as *const u16,
                        u as *const u16,
                        num_tokens as i32,
                        d as i32,
                    );
                }
            }
            DType::BF16 => {
                let o = Self::device_ptr_of::<half::bf16>(&out)?;
                let g = Self::device_ptr_of::<half::bf16>(&gate)?;
                let u = Self::device_ptr_of::<half::bf16>(&up)?;
                unsafe {
                    cuda_ffi::gelu_new_and_mul_bf16(
                        o as *mut u16,
                        g as *const u16,
                        u as *const u16,
                        num_tokens as i32,
                        d as i32,
                    );
                }
            }
            _ => return CpuActivationKernels.gelu_new_and_mul(&gate, &up),
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
    fn test_cpu_silu_and_mul() {
        let kernels = CpuActivationKernels;

        let gate = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let up = Tensor::ones(&[2, 2], DType::F32, &Device::Cpu).unwrap();

        let out = kernels.silu_and_mul(&gate, &up).unwrap();
        assert_eq!(out.dims(), &[2, 2]);

        // silu(1) * 1 ~ 0.7311
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 0.7311).abs() < 0.01);
    }

    #[test]
    fn test_cpu_gelu_and_mul() {
        let kernels = CpuActivationKernels;

        let gate = Tensor::new(&[[1.0f32, -1.0]], &Device::Cpu).unwrap();
        let up = Tensor::new(&[[2.0f32, 2.0]], &Device::Cpu).unwrap();

        let out = kernels.gelu_and_mul(&gate, &up).unwrap();
        assert_eq!(out.dims(), &[1, 2]);

        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // gelu(1) * 2 ~ 0.841 * 2 = 1.682
        assert!(vals[0] > 1.0);
    }

    #[test]
    fn test_cpu_gelu_new_and_mul() {
        let kernels = CpuActivationKernels;

        let gate = Tensor::new(&[[0.0f32, 1.0]], &Device::Cpu).unwrap();
        let up = Tensor::ones(&[1, 2], DType::F32, &Device::Cpu).unwrap();

        let out = kernels.gelu_new_and_mul(&gate, &up).unwrap();
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(vals[0].abs() < 1e-6); // gelu(0) = 0
        assert!((vals[1] - 0.8413).abs() < 0.01); // gelu_erf(1) ~ 0.8413
    }

    // -----------------------------------------------------------------------
    // CUDA tests — compare CUDA kernel output against CPU reference
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device required for this test")
    }

    /// Helper: compare two f32 vecs with tolerance.
    #[cfg(feature = "cuda")]
    fn assert_close(label: &str, cpu: &[f32], cuda: &[f32], tol: f64) {
        assert_eq!(cpu.len(), cuda.len(), "{label}: length mismatch");
        for (i, (c, g)) in cpu.iter().zip(cuda.iter()).enumerate() {
            assert!(
                (c - g).abs() as f64 <= tol,
                "{label} mismatch at {i}: cpu={c} cuda={g}"
            );
        }
    }

    /// Helper: run an activation kernel on CPU and CUDA, compare.
    #[cfg(feature = "cuda")]
    fn assert_activation_cuda_matches_cpu(op: &str, shape: &[usize], dtype: DType, tol: f64) {
        use super::CudaActivationKernels;

        let gate_cpu = Tensor::randn(0f32, 1.0, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let up_cpu = Tensor::randn(0f32, 1.0, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        // CPU reference.
        let ref_out = match op {
            "silu" => CpuActivationKernels
                .silu_and_mul(&gate_cpu, &up_cpu)
                .unwrap(),
            "gelu" => CpuActivationKernels
                .gelu_and_mul(&gate_cpu, &up_cpu)
                .unwrap(),
            "gelu_new" => CpuActivationKernels
                .gelu_new_and_mul(&gate_cpu, &up_cpu)
                .unwrap(),
            _ => panic!("unknown op: {op}"),
        };

        // CUDA kernel.
        let dev = cuda_device();
        let gate_gpu = gate_cpu.to_device(&dev).unwrap();
        let up_gpu = up_cpu.to_device(&dev).unwrap();
        let cuda_out = match op {
            "silu" => CudaActivationKernels
                .silu_and_mul(&gate_gpu, &up_gpu)
                .unwrap(),
            "gelu" => CudaActivationKernels
                .gelu_and_mul(&gate_gpu, &up_gpu)
                .unwrap(),
            "gelu_new" => CudaActivationKernels
                .gelu_new_and_mul(&gate_gpu, &up_gpu)
                .unwrap(),
            _ => panic!("unknown op: {op}"),
        }
        .to_device(&Device::Cpu)
        .unwrap();

        let ref_vals = ref_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda_vals = cuda_out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_close(&format!("{op}_{dtype:?}"), &ref_vals, &cuda_vals, tol);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_silu_and_mul_f32() {
        assert_activation_cuda_matches_cpu("silu", &[8, 256], DType::F32, 1e-5);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_silu_and_mul_f16() {
        assert_activation_cuda_matches_cpu("silu", &[8, 256], DType::F16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_silu_and_mul_bf16() {
        // BF16 has 7-bit mantissa: at magnitude ~8, ULP = 0.0625. Allow 2 ULP.
        assert_activation_cuda_matches_cpu("silu", &[8, 256], DType::BF16, 0.15);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_gelu_and_mul_f32() {
        assert_activation_cuda_matches_cpu("gelu", &[8, 256], DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_gelu_and_mul_bf16() {
        assert_activation_cuda_matches_cpu("gelu", &[8, 256], DType::BF16, 5e-2);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_gelu_new_and_mul_f32() {
        assert_activation_cuda_matches_cpu("gelu_new", &[8, 256], DType::F32, 1e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_activation_large() {
        // Typical MLP intermediate size.
        assert_activation_cuda_matches_cpu("silu", &[16, 4864], DType::F32, 1e-5);
    }
}
