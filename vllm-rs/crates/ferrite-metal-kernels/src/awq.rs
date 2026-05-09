//! AWQ (Activation-aware Weight Quantization) dequantization for Metal
//!
//! This module provides INT4 weight dequantization for AWQ-quantized models.
//! AWQ uses 4-bit integer quantization with group-wise scales and zero-points.
//!
//! # Format
//! - Weights: INT4 (0-15), packed 8 per uint32
//! - Scales: FP16, one per group
//! - Zeros: INT4 (0-15), packed 8 per uint32, one per group
//! - Group size: Typically 128
//!
//! # Dequantization Formula
//! ```text
//! dequantized = (int4_weight - zero) * scale
//! ```

use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::sync::Arc;

use crate::device::MetalDevice;
use crate::gemm::MetalGemm;

/// Errors that can occur during AWQ operations
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

/// AWQ dequantization and GEMM operations
pub struct MetalAwq {
    device: Arc<MetalDevice>,
    gemm: MetalGemm,
}

impl MetalAwq {
    /// Create a new AWQ dequantizer
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, AwqError> {
        let gemm =
            MetalGemm::new(device.clone()).map_err(|e| AwqError::GemmError(format!("{:?}", e)))?;

        Ok(Self { device, gemm })
    }

    /// Get or compile a shader pipeline
    fn get_pipeline(&self, kernel_name: &str) -> Result<ComputePipelineState, AwqError> {
        let library = self
            .device
            .device
            .new_library_with_source(
                include_str!("../shaders/awq_dequantize.metal"),
                &metal::CompileOptions::new(),
            )
            .map_err(|e| AwqError::ShaderCompilationFailed(e.to_string()))?;

        let function = library
            .get_function(kernel_name, None)
            .map_err(|e| AwqError::ShaderCompilationFailed(e.to_string()))?;

        self.device
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| AwqError::ShaderCompilationFailed(e.to_string()))
    }

    /// Unpack INT4 weights to FP16 (for testing/debugging)
    ///
    /// # Arguments
    /// * `packed_weights` - INT4 weights, 8 per uint32, shape [IC, OC/8]
    ///
    /// # Returns
    /// Unpacked FP16 weights, shape [IC, OC]
    pub fn unpack_int4(
        &self,
        packed_weights: &Buffer,
        num_in_channels: usize,
        num_out_channels: usize,
    ) -> Result<Buffer, AwqError> {
        // Validate dimensions
        if num_out_channels % 8 != 0 {
            return Err(AwqError::InvalidDimensions(
                "num_out_channels must be multiple of 8".to_string(),
            ));
        }

        let num_packed = num_in_channels * (num_out_channels / 8);
        let num_unpacked = num_in_channels * num_out_channels;

        // Create output buffer
        let output = self.device.device.new_buffer(
            (num_unpacked * 2) as u64, // 2 bytes per half
            MTLResourceOptions::StorageModeShared,
        );

        // Get pipeline
        let pipeline = self.get_pipeline("awq_unpack_int4_to_fp16")?;

        // Create command buffer and encoder
        let command_buffer = self.device.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(packed_weights), 0);
        encoder.set_buffer(1, Some(&output), 0);

        // Dispatch: one thread per packed uint32
        let grid_size = MTLSize::new(num_packed as u64, 1, 1);
        let threadgroup_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Dequantize INT4 weights to FP16
    ///
    /// # Arguments
    /// * `packed_weights` - INT4 weights, 8 per uint32, shape [IC, OC/8]
    /// * `scales` - FP16 scales, shape [IC/G, OC]
    /// * `packed_zeros` - INT4 zeros, 8 per uint32, shape [IC/G, OC/8]
    /// * `group_size` - Number of input channels per quantization group (e.g., 128)
    /// * `num_in_channels` - Total number of input channels (IC)
    /// * `num_out_channels` - Total number of output channels (OC)
    ///
    /// # Returns
    /// Dequantized FP16 weights, shape [IC, OC]
    pub fn dequantize(
        &self,
        packed_weights: &Buffer,
        scales: &Buffer,
        packed_zeros: &Buffer,
        group_size: u32,
        num_in_channels: usize,
        num_out_channels: usize,
    ) -> Result<Buffer, AwqError> {
        // Validate dimensions
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

        // Create output buffer
        let output = self.device.device.new_buffer(
            (num_output * 2) as u64, // 2 bytes per half
            MTLResourceOptions::StorageModeShared,
        );

        // Get pipeline
        let pipeline = self.get_pipeline("awq_dequantize_weights")?;

        // Create command buffer and encoder
        let command_buffer = self.device.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(packed_weights), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(packed_zeros), 0);
        encoder.set_buffer(3, Some(&output), 0);

        // Set scalar parameters
        let group_size_buffer = self.device.device.new_buffer_with_data(
            &group_size as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let num_out_channels_u32 = num_out_channels as u32;
        let num_out_channels_buffer = self.device.device.new_buffer_with_data(
            &num_out_channels_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(4, Some(&group_size_buffer), 0);
        encoder.set_buffer(5, Some(&num_out_channels_buffer), 0);

        // Dispatch: one thread per packed uint32
        let grid_size = MTLSize::new(num_packed as u64, 1, 1);
        let threadgroup_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output)
    }

    /// Dequantize INT4 weights and perform GEMM: C = A @ dequantize(B)
    ///
    /// # Arguments
    /// * `activations` - FP16 activations, shape [M, IC]
    /// * `packed_weights` - INT4 weights, 8 per uint32, shape [IC, OC/8]
    /// * `scales` - FP16 scales, shape [IC/G, OC]
    /// * `packed_zeros` - INT4 zeros, 8 per uint32, shape [IC/G, OC/8]
    /// * `group_size` - Number of input channels per quantization group
    /// * `m` - Batch size (number of rows in activations)
    /// * `ic` - Input channels (K dimension)
    /// * `oc` - Output channels (N dimension)
    ///
    /// # Returns
    /// Output activations, shape [M, OC]
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
        // Step 1: Dequantize weights
        let dequantized_weights =
            self.dequantize(packed_weights, scales, packed_zeros, group_size, ic, oc)?;

        // Step 2: Create output buffer
        let output = self.device.device.new_buffer(
            (m * oc * 2) as u64, // 2 bytes per half
            MTLResourceOptions::StorageModeShared,
        );

        // Step 3: Perform GEMM using MPS
        // C = A @ B where A is [M, IC] and B is [IC, OC]
        // Note: dequantize() already synchronized, so dequantized_weights is ready
        use crate::stream::MetalStream;
        let mut stream = MetalStream::new(&self.device.device);

        // Execute GEMM and wait for completion
        self.gemm
            .execute(
                &mut stream,
                activations,
                &dequantized_weights,
                &output,
                m as u32,
                oc as u32,
                ic as u32,
                1.0,   // alpha
                0.0,   // beta
                false, // transpose_a
                false, // transpose_b
                true,  // use_f16
            )
            .map_err(|e| AwqError::GemmError(format!("{:?}", e)))?;

        // Wait for GEMM to complete
        stream
            .synchronize()
            .map_err(|e| AwqError::GemmError(format!("{:?}", e)))?;

        Ok(output)
    }

    /// Vectorized dequantization using half4 for better memory bandwidth
    ///
    /// Same as `dequantize` but uses vectorized loads/stores
    pub fn dequantize_vec4(
        &self,
        packed_weights: &Buffer,
        scales: &Buffer,
        packed_zeros: &Buffer,
        group_size: u32,
        num_in_channels: usize,
        num_out_channels: usize,
    ) -> Result<Buffer, AwqError> {
        // Validate dimensions
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

        // Create output buffer
        let output = self.device.device.new_buffer(
            (num_output * 2) as u64, // 2 bytes per half
            MTLResourceOptions::StorageModeShared,
        );

        // Get pipeline
        let pipeline = self.get_pipeline("awq_dequantize_weights_vec4")?;

        // Create command buffer and encoder
        let command_buffer = self.device.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(packed_weights), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(packed_zeros), 0);
        encoder.set_buffer(3, Some(&output), 0);

        // Set scalar parameters
        let group_size_buffer = self.device.device.new_buffer_with_data(
            &group_size as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let num_out_channels_u32 = num_out_channels as u32;
        let num_out_channels_buffer = self.device.device.new_buffer_with_data(
            &num_out_channels_u32 as *const u32 as *const _,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(4, Some(&group_size_buffer), 0);
        encoder.set_buffer(5, Some(&num_out_channels_buffer), 0);

        // Dispatch: one thread per packed uint32
        let grid_size = MTLSize::new(num_packed as u64, 1, 1);
        let threadgroup_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

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
