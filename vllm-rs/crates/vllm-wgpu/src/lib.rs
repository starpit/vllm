// SPDX-License-Identifier: Apache-2.0
//! WebGPU tensor backend for vLLM.
//!
//! Provides GPU-accelerated tensor operations using WGSL compute shaders,
//! targeting both native GPU (via wgpu/Vulkan/Metal/DX12) and in-browser
//! WebGPU via WASM.

pub mod device;
pub mod model;
pub mod ops;
pub mod tensor;

pub use device::WgpuDevice;
pub use tensor::{WgpuDType, WgpuTensor};

/// Errors from the WebGPU backend.
#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error("no suitable WebGPU adapter found")]
    NoAdapter,
    #[error("device creation failed: {0}")]
    DeviceCreation(String),
    #[error("shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    #[error("buffer map failed")]
    BufferMap,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocking_device() -> WgpuDevice {
        pollster::block_on(WgpuDevice::new()).expect("failed to create WebGPU device")
    }

    #[test]
    fn test_add() {
        let dev = blocking_device();
        let a = WgpuTensor::from_f32(&dev, &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = WgpuTensor::from_f32(&dev, &[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let c = ops::add(&a, &b).unwrap();
        let result = pollster::block_on(c.to_f32()).unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_matmul() {
        let dev = blocking_device();
        // [2,3] x [3,2] = [2,2]
        let a = WgpuTensor::from_f32(&dev, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = WgpuTensor::from_f32(&dev, &[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = ops::matmul(&a, &b).unwrap();
        let result = pollster::block_on(c.to_f32()).unwrap();
        // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_silu_mul() {
        let dev = blocking_device();
        let gate = WgpuTensor::from_f32(&dev, &[3], &[0.0, 1.0, -1.0]).unwrap();
        let up = WgpuTensor::from_f32(&dev, &[3], &[1.0, 1.0, 1.0]).unwrap();
        let out = ops::silu_mul(&gate, &up).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        // SiLU(0) = 0, SiLU(1) = 1/(1+e^-1) ≈ 0.7311, SiLU(-1) = -1/(1+e^1) ≈ -0.2689
        assert!((result[0]).abs() < 1e-5);
        assert!((result[1] - 0.7311).abs() < 1e-3);
        assert!((result[2] - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn test_softmax() {
        let dev = blocking_device();
        let input = WgpuTensor::from_f32(&dev, &[1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = ops::softmax(&input).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Values should be monotonically increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
        assert!(result[2] < result[3]);
    }

    #[test]
    fn test_embedding() {
        let dev = blocking_device();
        // vocab_size=3, dim=2
        let table = WgpuTensor::from_f32(&dev, &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices = WgpuTensor::from_u32(&dev, &[2], &[0, 2]).unwrap();
        let out = ops::embedding(&table, &indices).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_rms_norm() {
        let dev = blocking_device();
        let input = WgpuTensor::from_f32(&dev, &[1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let weight = WgpuTensor::from_f32(&dev, &[4], &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let out = ops::rms_norm(&input, &weight, 1e-6).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5_f32).sqrt();
        assert!((result[0] - 1.0 / rms).abs() < 1e-4);
        assert!((result[3] - 4.0 / rms).abs() < 1e-4);
    }
}
