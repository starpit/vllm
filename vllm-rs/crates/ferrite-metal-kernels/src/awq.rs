//! AWQ (Activation-aware Weight Quantization) dequantization for Metal

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::device::MetalDevice;
use crate::gemm::MetalGemm;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

#[derive(Debug)]
pub enum AwqError {
    ShaderCompilationFailed(String),
    BufferCreationFailed(String),
    InvalidDimensions(String),
    GemmError(String),
}

impl std::fmt::Display for AwqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AwqError::ShaderCompilationFailed(msg) => {
                write!(f, "Failed to compile shader: {}", msg)
            }
            AwqError::BufferCreationFailed(msg) => write!(f, "Failed to create buffer: {}", msg),
            AwqError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            AwqError::GemmError(msg) => write!(f, "GEMM error: {}", msg),
        }
    }
}

impl std::error::Error for AwqError {}

pub struct MetalAwq {
    device: Arc<MetalDevice>,
    gemm: MetalGemm,
}

impl MetalAwq {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, AwqError> {
        let gemm =
            MetalGemm::new(device.clone()).map_err(|e| AwqError::GemmError(format!("{:?}", e)))?;

        Ok(Self { device, gemm })
    }

    fn get_pipeline(&self, kernel_name: &str) -> Result<ComputePipelineState, AwqError> {
        let opts = objc2_metal::MTLCompileOptions::new();
        let ns_source = NSString::from_str(include_str!("../shaders/awq_dequantize.metal"));
        let library = self
            .device
            .device
            .newLibraryWithSource_options_error(&ns_source, Some(&opts))
            .map_err(|e| AwqError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let ns_name = NSString::from_str(kernel_name);
        let function = library
            .newFunctionWithName(&ns_name)
            .ok_or_else(|| AwqError::ShaderCompilationFailed(format!("missing fn {kernel_name}")))?;

        self.device
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| AwqError::ShaderCompilationFailed(format!("{:?}", e)))
    }

    pub fn unpack_int4(
        &self,
        packed_weights: &Buffer,
        num_in_channels: usize,
        num_out_channels: usize,
    ) -> Result<Buffer, AwqError> {
        if num_out_channels % 8 != 0 {
            return Err(AwqError::InvalidDimensions(
                "num_out_channels must be multiple of 8".to_string(),
            ));
        }

        let num_packed = num_in_channels * (num_out_channels / 8);
        let num_unpacked = num_in_channels * num_out_channels;

        let output = self
            .device
            .device
            .newBufferWithLength_options(num_unpacked * 2, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| AwqError::BufferCreationFailed("output".into()))?;

        let pipeline = self.get_pipeline("awq_unpack_int4_to_fp16")?;

        let command_buffer = self
            .device
            .queue
            .commandBuffer()
            .ok_or_else(|| AwqError::BufferCreationFailed("commandBuffer".into()))?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| AwqError::BufferCreationFailed("computeCommandEncoder".into()))?;

        encoder.setComputePipelineState(&pipeline);
        unsafe { encoder.setBuffer_offset_atIndex(Some(packed_weights), 0, 0); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(&output), 0, 1); }

        let grid_size = MTLSize {
            width: num_packed,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(output)
    }

    pub fn dequantize(
        &self,
        packed_weights: &Buffer,
        scales: &Buffer,
        packed_zeros: &Buffer,
        group_size: u32,
        num_in_channels: usize,
        num_out_channels: usize,
    ) -> Result<Buffer, AwqError> {
        if num_out_channels % 8 != 0 {
            return Err(AwqError::InvalidDimensions(
                "num_out_channels must be multiple of 8".to_string(),
            ));
        }
        if num_in_channels % group_size as usize != 0 {
            return Err(AwqError::InvalidDimensions(
                "num_in_channels must be multiple of group_size".to_string(),
            ));
        }

        let num_packed = num_in_channels * (num_out_channels / 8);
        let num_output = num_in_channels * num_out_channels;

        let output = self
            .device
            .device
            .newBufferWithLength_options(num_output * 2, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| AwqError::BufferCreationFailed("output".into()))?;

        let pipeline = self.get_pipeline("awq_dequantize_weights")?;

        let command_buffer = self
            .device
            .queue
            .commandBuffer()
            .ok_or_else(|| AwqError::BufferCreationFailed("commandBuffer".into()))?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| AwqError::BufferCreationFailed("computeCommandEncoder".into()))?;

        encoder.setComputePipelineState(&pipeline);
        unsafe { encoder.setBuffer_offset_atIndex(Some(packed_weights), 0, 0); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(scales), 0, 1); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(packed_zeros), 0, 2); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(&output), 0, 3); }

        let num_out_channels_u32 = num_out_channels as u32;
        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(&group_size as *const u32 as *mut c_void).unwrap(),
                4,
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new(&num_out_channels_u32 as *const u32 as *mut c_void).unwrap(),
                4,
                5,
            );
        }

        let grid_size = MTLSize {
            width: num_packed,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(output)
    }

    pub fn dequantize_and_gemm(
        &self,
        activations: &Buffer,
        packed_weights: &Buffer,
        scales: &Buffer,
        packed_zeros: &Buffer,
        group_size: u32,
        m: usize,
        ic: usize,
        oc: usize,
    ) -> Result<Buffer, AwqError> {
        let dequantized_weights =
            self.dequantize(packed_weights, scales, packed_zeros, group_size, ic, oc)?;

        let output = self
            .device
            .device
            .newBufferWithLength_options(m * oc * 2, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| AwqError::BufferCreationFailed("output".into()))?;

        use crate::stream::MetalStream;
        let mut stream = MetalStream::new(&self.device.device);

        self.gemm
            .execute(
                &mut stream,
                activations,
                &dequantized_weights,
                &output,
                m as u32,
                oc as u32,
                ic as u32,
                1.0,
                0.0,
                false,
                false,
                true,
            )
            .map_err(|e| AwqError::GemmError(format!("{:?}", e)))?;

        stream
            .synchronize()
            .map_err(|e| AwqError::GemmError(format!("{:?}", e)))?;

        Ok(output)
    }

    pub fn dequantize_vec4(
        &self,
        packed_weights: &Buffer,
        scales: &Buffer,
        packed_zeros: &Buffer,
        group_size: u32,
        num_in_channels: usize,
        num_out_channels: usize,
    ) -> Result<Buffer, AwqError> {
        if num_out_channels % 8 != 0 {
            return Err(AwqError::InvalidDimensions(
                "num_out_channels must be multiple of 8".to_string(),
            ));
        }
        if num_in_channels % group_size as usize != 0 {
            return Err(AwqError::InvalidDimensions(
                "num_in_channels must be multiple of group_size".to_string(),
            ));
        }

        let num_packed = num_in_channels * (num_out_channels / 8);
        let num_output = num_in_channels * num_out_channels;

        let output = self
            .device
            .device
            .newBufferWithLength_options(num_output * 2, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| AwqError::BufferCreationFailed("output".into()))?;

        let pipeline = self.get_pipeline("awq_dequantize_weights_vec4")?;

        let command_buffer = self
            .device
            .queue
            .commandBuffer()
            .ok_or_else(|| AwqError::BufferCreationFailed("commandBuffer".into()))?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or_else(|| AwqError::BufferCreationFailed("computeCommandEncoder".into()))?;

        encoder.setComputePipelineState(&pipeline);
        unsafe { encoder.setBuffer_offset_atIndex(Some(packed_weights), 0, 0); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(scales), 0, 1); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(packed_zeros), 0, 2); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(&output), 0, 3); }

        let num_out_channels_u32 = num_out_channels as u32;
        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(&group_size as *const u32 as *mut c_void).unwrap(),
                4,
                4,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new(&num_out_channels_u32 as *const u32 as *mut c_void).unwrap(),
                4,
                5,
            );
        }

        let grid_size = MTLSize {
            width: num_packed,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect_device;

    #[test]
    fn test_awq_creation() {
        let device = detect_device().expect("Failed to create Metal device");
        let awq = MetalAwq::new(Arc::new(device));
        assert!(awq.is_ok());
    }
}
