use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLSize};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::shader_cache::ShaderCache;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type ComputeCommandEncoderRef = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    SiLU,
    GELU,
    GELUTanh,
    GELUQuick,
    FatReLU,
}

/// Data type for activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F16,
    BF16,
    F32,
}

impl DataType {
    pub fn element_size(&self) -> usize {
        match self {
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::F32 => 4,
        }
    }
}

/// Metal activation function executor
pub struct MetalActivation {
    shader_cache: Arc<ShaderCache>,
}

impl MetalActivation {
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        let shader_cache = Arc::new(ShaderCache::new(device)?);
        Ok(Self { shader_cache })
    }

    pub fn execute(
        &self,
        output: &Buffer,
        input: &Buffer,
        n: u32,
        activation: ActivationType,
        dtype: DataType,
        threshold: f32,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        let kernel_name = self.get_kernel_name(activation, dtype);
        let pipeline = self.shader_cache.get_pipeline(&kernel_name)?;

        encoder.setComputePipelineState(&pipeline);
        unsafe { encoder.setBuffer_offset_atIndex(Some(output), 0, 0); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(input), 0, 1); }

        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(&n as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                2,
            );
        }

        if activation == ActivationType::FatReLU {
            unsafe {
                encoder.setBytes_length_atIndex(
                    NonNull::new(&threshold as *const f32 as *mut c_void).unwrap(),
                    std::mem::size_of::<f32>(),
                    3,
                );
            }
        }

        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let num_threadgroups = MTLSize {
            width: (n as usize).div_ceil(256),
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(num_threadgroups, threadgroup_size);

        Ok(())
    }

    pub fn execute_silu_vec4(
        &self,
        output: &Buffer,
        input: &Buffer,
        n: u32,
        encoder: &ComputeCommandEncoderRef,
    ) -> Result<(), MetalStreamError> {
        let pipeline = self.shader_cache.get_pipeline("silu_vec4_f16")?;

        encoder.setComputePipelineState(&pipeline);
        unsafe { encoder.setBuffer_offset_atIndex(Some(output), 0, 0); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(input), 0, 1); }

        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(&n as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                2,
            );
        }

        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let num_threadgroups = MTLSize {
            width: (n as usize).div_ceil(256),
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(num_threadgroups, threadgroup_size);

        Ok(())
    }

    fn get_kernel_name(&self, activation: ActivationType, dtype: DataType) -> String {
        let dtype_suffix = match dtype {
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            DataType::F32 => "f32",
        };

        let activation_prefix = match activation {
            ActivationType::SiLU => "silu",
            ActivationType::GELU => "gelu",
            ActivationType::GELUTanh => "gelu_tanh",
            ActivationType::GELUQuick => "gelu_quick",
            ActivationType::FatReLU => "fatrelu",
        };

        format!("{}_{}", activation_prefix, dtype_suffix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::detect_device;
    use objc2_metal::MTLCommandQueue;

    #[test]
    fn test_silu_f16() {
        let device = detect_device().expect("Metal device required");
        let activation = MetalActivation::new(device.device.clone()).unwrap();

        let input_data: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let n = input_data.len() as u32;

        let input_f16: Vec<u16> = input_data
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();

        let input_buffer = unsafe {
            device
                .device
                .newBufferWithBytes_length_options(
                    NonNull::new(input_f16.as_ptr() as *mut c_void).unwrap(),
                    input_f16.len() * 2,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("buffer alloc")
        };

        let output_buffer = device
            .device
            .newBufferWithLength_options(
                (n as usize) * 2,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("buffer alloc");

        let command_buffer = device.queue.commandBuffer().expect("command buffer");
        let encoder = command_buffer
            .computeCommandEncoder()
            .expect("compute encoder");

        activation
            .execute(
                &output_buffer,
                &input_buffer,
                n,
                ActivationType::SiLU,
                DataType::F16,
                0.0,
                &encoder,
            )
            .unwrap();

        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        let output_ptr = output_buffer.contents().as_ptr() as *const u16;
        let output_f16: Vec<u16> =
            unsafe { std::slice::from_raw_parts(output_ptr, n as usize) }.to_vec();
        let output_data: Vec<f32> = output_f16
            .iter()
            .map(|&x| half::f16::from_bits(x).to_f32())
            .collect();

        for (i, &x) in input_data.iter().enumerate() {
            let expected = x / (1.0 + (-x).exp());
            let actual = output_data[i];
            let diff = (expected - actual).abs();
            assert!(
                diff < 0.01,
                "SiLU mismatch at index {}: expected {}, got {}, diff {}",
                i,
                expected,
                actual,
                diff
            );
        }
    }
}
