// SPDX-License-Identifier: Apache-2.0
//! GPU dequantization kernels for GPTQ and AWQ INT4 packed weights.
//!
//! These kernels dequantize packed INT4-in-INT32 weights to float on GPU,
//! replacing the CPU-side scalar unpacking that fails on CUDA.

use candle_core::{DType, Tensor};

use crate::error::{KernelError, KernelResult};

// ---------------------------------------------------------------------------
// CUDA FFI declarations
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda_ffi {
    unsafe extern "C" {
        // GPTQ dequantize
        pub fn gptq_dequantize_f32(
            out: *mut f32,
            qweight: *const i32,
            qzeros: *const i32,
            scales: *const f32,
            g_idx: *const i32,
            in_features: i32,
            out_features: i32,
            group_size: i32,
        );
        pub fn gptq_dequantize_f16(
            out: *mut u16,
            qweight: *const i32,
            qzeros: *const i32,
            scales: *const u16,
            g_idx: *const i32,
            in_features: i32,
            out_features: i32,
            group_size: i32,
        );
        pub fn gptq_dequantize_bf16(
            out: *mut u16,
            qweight: *const i32,
            qzeros: *const i32,
            scales: *const u16,
            g_idx: *const i32,
            in_features: i32,
            out_features: i32,
            group_size: i32,
        );

        // AWQ dequantize
        pub fn awq_dequantize_f32(
            out: *mut f32,
            qweight: *const i32,
            qzeros: *const i32,
            scales: *const f32,
            in_features: i32,
            out_features: i32,
            group_size: i32,
        );
        pub fn awq_dequantize_f16(
            out: *mut u16,
            qweight: *const i32,
            qzeros: *const i32,
            scales: *const u16,
            in_features: i32,
            out_features: i32,
            group_size: i32,
        );
        pub fn awq_dequantize_bf16(
            out: *mut u16,
            qweight: *const i32,
            qzeros: *const i32,
            scales: *const u16,
            in_features: i32,
            out_features: i32,
            group_size: i32,
        );
    }
}

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

/// Ensure a tensor is I32 and contiguous on CUDA.
/// Candle lacks I64→I32 cast on CUDA, so we round-trip through CPU if needed.
#[cfg(feature = "cuda")]
fn ensure_i32_cuda(t: &Tensor, device: &candle_core::Device) -> KernelResult<Tensor> {
    if t.dtype() == DType::I32 {
        Ok(t.contiguous()?)
    } else {
        // CPU round-trip for dtype cast (candle has no I64→I32 CUDA kernel)
        let cpu = t.to_device(&candle_core::Device::Cpu)?;
        let i32_cpu = cpu.to_dtype(DType::I32)?;
        Ok(i32_cpu.to_device(device)?)
    }
}

#[cfg(feature = "cuda")]
pub struct CudaQuantizeKernels;

#[cfg(feature = "cuda")]
impl CudaQuantizeKernels {
    fn device_ptr_of<T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
        tensor: &Tensor,
    ) -> KernelResult<usize> {
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

    /// Dequantize GPTQ INT4 packed weights on GPU.
    ///
    /// Returns `[in_features, out_features]` tensor in the same dtype as `scales`.
    pub fn gptq_dequantize(
        qweight: &Tensor,
        qzeros: &Tensor,
        scales: &Tensor,
        g_idx: Option<&Tensor>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> KernelResult<Tensor> {
        let dtype = scales.dtype();
        let device = scales.device();

        let qweight = ensure_i32_cuda(qweight, device)?;
        let qzeros = ensure_i32_cuda(qzeros, device)?;
        let scales = scales.contiguous()?;
        let g_idx_contig = g_idx.map(|t| ensure_i32_cuda(t, device)).transpose()?;

        let out = Tensor::zeros((in_features, out_features), dtype, device)?;

        let qw_ptr = Self::device_ptr_of::<i32>(&qweight)?;
        let qz_ptr = Self::device_ptr_of::<i32>(&qzeros)?;
        let g_idx_ptr = match &g_idx_contig {
            Some(t) => Self::device_ptr_of::<i32>(t)? as *const i32,
            None => std::ptr::null(),
        };

        match dtype {
            DType::F32 => {
                let o = Self::device_ptr_of::<f32>(&out)?;
                let s = Self::device_ptr_of::<f32>(&scales)?;
                unsafe {
                    cuda_ffi::gptq_dequantize_f32(
                        o as *mut f32,
                        qw_ptr as *const i32,
                        qz_ptr as *const i32,
                        s as *const f32,
                        g_idx_ptr,
                        in_features as i32,
                        out_features as i32,
                        group_size as i32,
                    );
                }
            }
            DType::F16 => {
                let o = Self::device_ptr_of::<half::f16>(&out)?;
                let s = Self::device_ptr_of::<half::f16>(&scales)?;
                unsafe {
                    cuda_ffi::gptq_dequantize_f16(
                        o as *mut u16,
                        qw_ptr as *const i32,
                        qz_ptr as *const i32,
                        s as *const u16,
                        g_idx_ptr,
                        in_features as i32,
                        out_features as i32,
                        group_size as i32,
                    );
                }
            }
            DType::BF16 => {
                let o = Self::device_ptr_of::<half::bf16>(&out)?;
                let s = Self::device_ptr_of::<half::bf16>(&scales)?;
                unsafe {
                    cuda_ffi::gptq_dequantize_bf16(
                        o as *mut u16,
                        qw_ptr as *const i32,
                        qz_ptr as *const i32,
                        s as *const u16,
                        g_idx_ptr,
                        in_features as i32,
                        out_features as i32,
                        group_size as i32,
                    );
                }
            }
            _ => {
                return Err(KernelError::Other(format!(
                    "gptq_dequantize: unsupported dtype {:?}",
                    dtype
                )));
            }
        }
        Ok(out)
    }

    /// Dequantize AWQ INT4 packed weights on GPU.
    ///
    /// Returns `[in_features, out_features]` tensor in the same dtype as `scales`.
    pub fn awq_dequantize(
        qweight: &Tensor,
        qzeros: &Tensor,
        scales: &Tensor,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> KernelResult<Tensor> {
        let dtype = scales.dtype();
        let device = scales.device();

        let qweight = ensure_i32_cuda(qweight, device)?;
        let qzeros = ensure_i32_cuda(qzeros, device)?;
        let scales = scales.contiguous()?;

        let out = Tensor::zeros((in_features, out_features), dtype, device)?;

        let qw_ptr = Self::device_ptr_of::<i32>(&qweight)?;
        let qz_ptr = Self::device_ptr_of::<i32>(&qzeros)?;

        match dtype {
            DType::F32 => {
                let o = Self::device_ptr_of::<f32>(&out)?;
                let s = Self::device_ptr_of::<f32>(&scales)?;
                unsafe {
                    cuda_ffi::awq_dequantize_f32(
                        o as *mut f32,
                        qw_ptr as *const i32,
                        qz_ptr as *const i32,
                        s as *const f32,
                        in_features as i32,
                        out_features as i32,
                        group_size as i32,
                    );
                }
            }
            DType::F16 => {
                let o = Self::device_ptr_of::<half::f16>(&out)?;
                let s = Self::device_ptr_of::<half::f16>(&scales)?;
                unsafe {
                    cuda_ffi::awq_dequantize_f16(
                        o as *mut u16,
                        qw_ptr as *const i32,
                        qz_ptr as *const i32,
                        s as *const u16,
                        in_features as i32,
                        out_features as i32,
                        group_size as i32,
                    );
                }
            }
            DType::BF16 => {
                let o = Self::device_ptr_of::<half::bf16>(&out)?;
                let s = Self::device_ptr_of::<half::bf16>(&scales)?;
                unsafe {
                    cuda_ffi::awq_dequantize_bf16(
                        o as *mut u16,
                        qw_ptr as *const i32,
                        qz_ptr as *const i32,
                        s as *const u16,
                        in_features as i32,
                        out_features as i32,
                        group_size as i32,
                    );
                }
            }
            _ => {
                return Err(KernelError::Other(format!(
                    "awq_dequantize: unsupported dtype {:?}",
                    dtype
                )));
            }
        }
        Ok(out)
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device required")
    }

    #[test]
    fn test_cuda_gptq_dequantize_f32() {
        let dev = cuda_device();

        // Pack 8 INT4 values (0..7) into one i32 (GPTQ row-packed)
        let packed: i32 =
            0 | (1 << 4) | (2 << 8) | (3 << 12) | (4 << 16) | (5 << 20) | (6 << 24) | (7 << 28);
        let qweight = Tensor::new(&[[packed]], &dev).unwrap(); // [1, 1] i32
        let qzeros = Tensor::new(&[[0i32]], &dev).unwrap(); // [1, 1] i32
        let scales = Tensor::new(&[[1.0f32]], &dev).unwrap(); // [1, 1] f32

        let out = CudaQuantizeKernels::gptq_dequantize(&qweight, &qzeros, &scales, None, 8, 1, 8)
            .unwrap();

        assert_eq!(out.dims(), &[8, 1]);
        let vals: Vec<f32> = out
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for i in 0..8 {
            assert!(
                (vals[i] - i as f32).abs() < 0.01,
                "expected {i}, got {} at position {i}",
                vals[i]
            );
        }
    }

    #[test]
    fn test_cuda_gptq_dequantize_f16() {
        let dev = cuda_device();

        let packed: i32 =
            0 | (1 << 4) | (2 << 8) | (3 << 12) | (4 << 16) | (5 << 20) | (6 << 24) | (7 << 28);
        let qweight = Tensor::new(&[[packed]], &dev).unwrap();
        let qzeros = Tensor::new(&[[0i32]], &dev).unwrap();
        let scales = Tensor::new(&[[1.0f32]], &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let out = CudaQuantizeKernels::gptq_dequantize(&qweight, &qzeros, &scales, None, 8, 1, 8)
            .unwrap();

        assert_eq!(out.dims(), &[8, 1]);
        assert_eq!(out.dtype(), DType::F16);
        let vals: Vec<f32> = out
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for i in 0..8 {
            assert!(
                (vals[i] - i as f32).abs() < 0.1,
                "expected {i}, got {} at position {i}",
                vals[i]
            );
        }
    }

    #[test]
    fn test_cuda_gptq_dequantize_realistic() {
        // Simulate realistic GPTQ: 128 in_features, 64 out_features, group_size=128, scales=F16
        let dev = cuda_device();
        let in_features = 128;
        let out_features = 64;
        let pack_factor = 8;
        let packed_rows = in_features / pack_factor; // 16
        let num_groups = 1;
        let group_size = 128;

        // All qweight = 0 (all values are 0)
        let qweight = Tensor::zeros((packed_rows, out_features), DType::I64, &dev).unwrap();
        let qzeros =
            Tensor::zeros((num_groups, out_features / pack_factor), DType::I64, &dev).unwrap();
        let scales = Tensor::ones((num_groups, out_features), DType::F16, &dev).unwrap();

        let out = CudaQuantizeKernels::gptq_dequantize(
            &qweight,
            &qzeros,
            &scales,
            None,
            in_features,
            out_features,
            group_size,
        )
        .unwrap();

        assert_eq!(out.dims(), &[in_features, out_features]);
        assert_eq!(out.dtype(), DType::F16);
    }

    #[test]
    fn test_cuda_gptq_dequantize_with_scales() {
        let dev = cuda_device();

        // All values = 8, zeros = 8, scale = 2.0 → output = 2*(8-8) = 0
        let val: i32 = 8;
        let packed: i32 = (0..8).fold(0i32, |acc, j: i32| acc | (val << (j * 4)));
        let zero_packed: i32 = (0..8).fold(0i32, |acc, j: i32| acc | (8i32 << (j * 4)));

        let qweight = Tensor::new(&[[packed]], &dev).unwrap();
        let qzeros = Tensor::new(&[[zero_packed]], &dev).unwrap();
        let scales = Tensor::new(&[[2.0f32]], &dev).unwrap();

        let out = CudaQuantizeKernels::gptq_dequantize(&qweight, &qzeros, &scales, None, 8, 1, 8)
            .unwrap();

        let vals: Vec<f32> = out
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for v in &vals {
            assert!(v.abs() < 0.01, "expected 0.0, got {v}");
        }
    }
}
