// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Fused kernel implementations for memory-bandwidth optimization.
//!
//! These kernels combine multiple operations to eliminate intermediate
//! memory round-trips, which is critical on Apple Silicon's memory-bound
//! architecture.

use metal::{Buffer, Device, Library, MTLResourceOptions, MTLSize};
use std::sync::Arc;

use crate::{MetalDevice, MetalStream, MetalStreamError};

/// Fused Add + RMSNorm implementation
///
/// Computes: output = rmsnorm(input + residual, weight, eps)
///
/// This fusion eliminates one memory round-trip by computing the residual
/// add and normalization in a single pass.
pub struct FusedAddRmsNorm {
    device: Arc<MetalDevice>,
    pipeline_f16: metal::ComputePipelineState,
    pipeline_bf16: metal::ComputePipelineState,
    pipeline_f16_vec4: metal::ComputePipelineState,
}

impl FusedAddRmsNorm {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, MetalStreamError> {
        let library = device
            .device
            .new_library_with_source(
                include_str!("../shaders/fused_add_rmsnorm.metal"),
                &metal::CompileOptions::new(),
            )
            .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{:?}", e)))?;

        let pipeline_f16 =
            Self::create_pipeline(&device.device, &library, "fused_add_rmsnorm_f16")?;
        let pipeline_bf16 =
            Self::create_pipeline(&device.device, &library, "fused_add_rmsnorm_bf16")?;
        let pipeline_f16_vec4 =
            Self::create_pipeline(&device.device, &library, "fused_add_rmsnorm_f16_vec4")?;

        Ok(Self {
            device,
            pipeline_f16,
            pipeline_bf16,
            pipeline_f16_vec4,
        })
    }

    fn create_pipeline(
        device: &Device,
        library: &Library,
        name: &str,
    ) -> Result<metal::ComputePipelineState, MetalStreamError> {
        let function = library
            .get_function(name, None)
            .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{:?}", e)))?;

        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalStreamError::ShaderCompilationFailed(e))
    }

    /// Execute fused add + rmsnorm
    ///
    /// # Arguments
    /// * `stream` - Metal stream for command encoding
    /// * `input` - Input tensor [M, N]
    /// * `residual` - Residual tensor [M, N]
    /// * `weight` - RMSNorm weight [N]
    /// * `output` - Output tensor [M, N]
    /// * `residual_out` - Optional output for (input + residual) [M, N]
    /// * `m` - Batch size
    /// * `n` - Hidden size
    /// * `eps` - Epsilon for numerical stability
    /// * `use_f16` - Use FP16 (true) or BF16 (false)
    pub fn execute(
        &self,
        stream: &mut MetalStream,
        input: &Buffer,
        residual: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        residual_out: Option<&Buffer>,
        m: u32,
        n: u32,
        eps: f32,
        use_f16: bool,
    ) -> Result<(), MetalStreamError> {
        let encoder = stream.get_command_buffer()?.new_compute_command_encoder();

        // Choose pipeline based on dtype and vectorization
        let pipeline = if use_f16 && n % 4 == 0 {
            &self.pipeline_f16_vec4
        } else if use_f16 {
            &self.pipeline_f16
        } else {
            &self.pipeline_bf16
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(residual), 0);
        encoder.set_buffer(2, Some(weight), 0);
        encoder.set_buffer(3, Some(output), 0);

        if let Some(res_out) = residual_out {
            encoder.set_buffer(4, Some(res_out), 0);
        }

        // Create constant buffers for scalar parameters
        let n_param = if n % 4 == 0 && use_f16 { n / 4 } else { n };

        let m_buffer = self.device.device.new_buffer_with_data(
            &m as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let n_buffer = self.device.device.new_buffer_with_data(
            &n_param as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let eps_buffer = self.device.device.new_buffer_with_data(
            &eps as *const f32 as *const _,
            std::mem::size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(5, Some(&m_buffer), 0);
        encoder.set_buffer(6, Some(&n_buffer), 0);
        encoder.set_buffer(7, Some(&eps_buffer), 0);

        // Dispatch: M threadgroups, each with min(N, 1024) threads
        let threadgroup_size = n_param.min(1024);
        let grid_size = MTLSize::new(m as u64, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size as u64, 1, 1);

        encoder.dispatch_thread_groups(grid_size, threadgroup);
        encoder.end_encoding();

        stream.commit()?;
        Ok(())
    }
}

/// Fused Gate-Up-SiLU-Mul implementation (SwiGLU)
///
/// Computes: output = silu(gate) * up
/// Where: silu(x) = x * sigmoid(x)
///
/// Supports two input modes:
/// 1. Separate gate and up tensors
/// 2. Concatenated gate_up tensor [M, 2*N]
pub struct FusedGateUpSiluMul {
    device: Arc<MetalDevice>,
    pipeline_f16: metal::ComputePipelineState,
    pipeline_f16_concat: metal::ComputePipelineState,
    pipeline_bf16: metal::ComputePipelineState,
    pipeline_bf16_concat: metal::ComputePipelineState,
    pipeline_f16_vec4: metal::ComputePipelineState,
    pipeline_f16_concat_vec4: metal::ComputePipelineState,
    pipeline_gelu_f16: metal::ComputePipelineState,
    pipeline_gelu_exact_f16: metal::ComputePipelineState,
}

impl FusedGateUpSiluMul {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, MetalStreamError> {
        let library = device
            .device
            .new_library_with_source(
                include_str!("../shaders/fused_gate_up_silu_mul.metal"),
                &metal::CompileOptions::new(),
            )
            .map_err(|e| MetalStreamError::ShaderCompilationFailed(e))?;

        Ok(Self {
            device: device.clone(),
            pipeline_f16: Self::create_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_f16",
            )?,
            pipeline_f16_concat: Self::create_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_concat_f16",
            )?,
            pipeline_bf16: Self::create_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_bf16",
            )?,
            pipeline_bf16_concat: Self::create_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_concat_bf16",
            )?,
            pipeline_f16_vec4: Self::create_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_f16_vec4",
            )?,
            pipeline_f16_concat_vec4: Self::create_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_concat_f16_vec4",
            )?,
            pipeline_gelu_f16: Self::create_pipeline(
                &device.device,
                &library,
                "fused_gate_up_gelu_mul_f16",
            )?,
            pipeline_gelu_exact_f16: Self::create_pipeline(
                &device.device,
                &library,
                "fused_gate_up_gelu_mul_f16",
            )?, // Use approx for both
        })
    }

    fn create_pipeline(
        device: &Device,
        library: &Library,
        name: &str,
    ) -> Result<metal::ComputePipelineState, MetalStreamError> {
        let function = library
            .get_function(name, None)
            .map_err(|e| MetalStreamError::ShaderCompilationFailed(e))?;

        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalStreamError::ShaderCompilationFailed(e))
    }

    /// Execute with separate gate and up tensors
    pub fn execute_separate(
        &self,
        stream: &mut MetalStream,
        gate_out: &Buffer,
        up_out: &Buffer,
        output: &Buffer,
        m: u32,
        n: u32,
        use_f16: bool,
    ) -> Result<(), MetalStreamError> {
        let encoder = stream.get_command_buffer()?.new_compute_command_encoder();

        let pipeline = if use_f16 && n % 4 == 0 {
            &self.pipeline_f16_vec4
        } else if use_f16 {
            &self.pipeline_f16
        } else {
            &self.pipeline_bf16
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(gate_out), 0);
        encoder.set_buffer(1, Some(up_out), 0);
        encoder.set_buffer(2, Some(output), 0);

        let n_param = if n % 4 == 0 && use_f16 { n / 4 } else { n };

        let m_buffer = self.device.device.new_buffer_with_data(
            &m as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let n_buffer = self.device.device.new_buffer_with_data(
            &n_param as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(3, Some(&m_buffer), 0);
        encoder.set_buffer(4, Some(&n_buffer), 0);

        let threadgroup_size = n.min(1024);
        let grid_size = MTLSize::new(m as u64, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size as u64, 1, 1);

        encoder.dispatch_thread_groups(grid_size, threadgroup);
        encoder.end_encoding();

        stream.commit()?;
        Ok(())
    }

    /// Execute with concatenated gate_up tensor [M, 2*N]
    pub fn execute_concat(
        &self,
        stream: &mut MetalStream,
        gate_up: &Buffer,
        output: &Buffer,
        m: u32,
        n: u32,
        use_f16: bool,
    ) -> Result<(), MetalStreamError> {
        let encoder = stream.get_command_buffer()?.new_compute_command_encoder();

        let pipeline = if use_f16 && n % 4 == 0 {
            &self.pipeline_f16_concat_vec4
        } else if use_f16 {
            &self.pipeline_f16_concat
        } else {
            &self.pipeline_bf16_concat
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(gate_up), 0);
        encoder.set_buffer(1, Some(output), 0);

        let n_param = if n % 4 == 0 && use_f16 { n / 4 } else { n };

        let m_buffer = self.device.device.new_buffer_with_data(
            &m as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let n_buffer = self.device.device.new_buffer_with_data(
            &n_param as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(2, Some(&m_buffer), 0);
        encoder.set_buffer(3, Some(&n_buffer), 0);

        let threadgroup_size = n.min(1024);
        let grid_size = MTLSize::new(m as u64, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size as u64, 1, 1);

        encoder.dispatch_thread_groups(grid_size, threadgroup);
        encoder.end_encoding();

        stream.commit()?;
        Ok(())
    }

    /// Execute with GELU activation (for Gemma models)
    pub fn execute_gelu(
        &self,
        stream: &mut MetalStream,
        gate_out: &Buffer,
        up_out: &Buffer,
        output: &Buffer,
        m: u32,
        n: u32,
        exact: bool,
    ) -> Result<(), MetalStreamError> {
        let encoder = stream.get_command_buffer()?.new_compute_command_encoder();

        let pipeline = if exact {
            &self.pipeline_gelu_exact_f16
        } else {
            &self.pipeline_gelu_f16
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(gate_out), 0);
        encoder.set_buffer(1, Some(up_out), 0);
        encoder.set_buffer(2, Some(output), 0);

        let m_buffer = self.device.device.new_buffer_with_data(
            &m as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let n_buffer = self.device.device.new_buffer_with_data(
            &n as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        encoder.set_buffer(3, Some(&m_buffer), 0);
        encoder.set_buffer(4, Some(&n_buffer), 0);

        let threadgroup_size = n.min(1024);
        let grid_size = MTLSize::new(m as u64, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size as u64, 1, 1);

        encoder.dispatch_thread_groups(grid_size, threadgroup);
        encoder.end_encoding();

        stream.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect_device;

    #[test]
    fn test_fused_add_rmsnorm_creation() {
        let device = detect_device().expect("Metal device required");
        let fused_norm = FusedAddRmsNorm::new(std::sync::Arc::new(device));
        assert!(fused_norm.is_ok());
    }

    #[test]
    fn test_fused_gate_up_silu_mul_creation() {
        let device = detect_device().expect("Metal device required");
        let fused_silu = FusedGateUpSiluMul::new(std::sync::Arc::new(device));
        assert!(fused_silu.is_ok());
    }
}
