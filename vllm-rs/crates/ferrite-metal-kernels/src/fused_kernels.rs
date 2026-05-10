// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Fused kernel implementations for memory-bandwidth optimization.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::{MetalDevice, MetalStream, MetalStreamError};

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

pub struct FusedAddRmsNorm {
    pipeline_f16: ComputePipelineState,
    pipeline_bf16: ComputePipelineState,
    pipeline_f16_vec4: ComputePipelineState,
}

fn compile_library(device: &Device, source: &str) -> Result<Library, MetalStreamError> {
    let opts = objc2_metal::MTLCompileOptions::new();
    let ns_source = NSString::from_str(source);
    device
        .newLibraryWithSource_options_error(&ns_source, Some(&opts))
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{e:?}")))
}

fn compile_pipeline(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, MetalStreamError> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed(format!("missing fn {name}")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{e:?}")))
}

impl FusedAddRmsNorm {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, MetalStreamError> {
        let library = compile_library(
            &device.device,
            include_str!("../shaders/fused_add_rmsnorm.metal"),
        )?;
        let pipeline_f16 = compile_pipeline(&device.device, &library, "fused_add_rmsnorm_f16")?;
        let pipeline_bf16 = compile_pipeline(&device.device, &library, "fused_add_rmsnorm_bf16")?;
        let pipeline_f16_vec4 =
            compile_pipeline(&device.device, &library, "fused_add_rmsnorm_f16_vec4")?;

        Ok(Self {
            pipeline_f16,
            pipeline_bf16,
            pipeline_f16_vec4,
        })
    }

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
        let cmd_buf = stream.get_command_buffer()?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
        })?;

        let pipeline = if use_f16 && n % 4 == 0 {
            &self.pipeline_f16_vec4
        } else if use_f16 {
            &self.pipeline_f16
        } else {
            &self.pipeline_bf16
        };

        encoder.setComputePipelineState(pipeline);
        unsafe { encoder.setBuffer_offset_atIndex(Some(input), 0, 0); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(residual), 0, 1); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(weight), 0, 2); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(output), 0, 3); }

        if let Some(res_out) = residual_out {
            unsafe { encoder.setBuffer_offset_atIndex(Some(res_out), 0, 4); }
        }

        let n_param = if n % 4 == 0 && use_f16 { n / 4 } else { n };

        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(&m as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                5,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new(&n_param as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                6,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new(&eps as *const f32 as *mut c_void).unwrap(),
                std::mem::size_of::<f32>(),
                7,
            );
        }

        let threadgroup_size = (n_param as usize).min(1024);
        let grid_size = MTLSize {
            width: m as usize,
            height: 1,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: threadgroup_size,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup);
        encoder.endEncoding();

        stream.commit()?;
        Ok(())
    }
}

pub struct FusedGateUpSiluMul {
    pipeline_f16: ComputePipelineState,
    pipeline_f16_concat: ComputePipelineState,
    pipeline_bf16: ComputePipelineState,
    pipeline_bf16_concat: ComputePipelineState,
    pipeline_f16_vec4: ComputePipelineState,
    pipeline_f16_concat_vec4: ComputePipelineState,
    pipeline_gelu_f16: ComputePipelineState,
    pipeline_gelu_exact_f16: ComputePipelineState,
}

impl FusedGateUpSiluMul {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, MetalStreamError> {
        let library = compile_library(
            &device.device,
            include_str!("../shaders/fused_gate_up_silu_mul.metal"),
        )?;

        Ok(Self {
            pipeline_f16: compile_pipeline(&device.device, &library, "fused_gate_up_silu_mul_f16")?,
            pipeline_f16_concat: compile_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_concat_f16",
            )?,
            pipeline_bf16: compile_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_bf16",
            )?,
            pipeline_bf16_concat: compile_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_concat_bf16",
            )?,
            pipeline_f16_vec4: compile_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_f16_vec4",
            )?,
            pipeline_f16_concat_vec4: compile_pipeline(
                &device.device,
                &library,
                "fused_gate_up_silu_mul_concat_f16_vec4",
            )?,
            pipeline_gelu_f16: compile_pipeline(
                &device.device,
                &library,
                "fused_gate_up_gelu_mul_f16",
            )?,
            pipeline_gelu_exact_f16: compile_pipeline(
                &device.device,
                &library,
                "fused_gate_up_gelu_mul_f16",
            )?,
        })
    }

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
        let cmd_buf = stream.get_command_buffer()?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
        })?;

        let pipeline = if use_f16 && n % 4 == 0 {
            &self.pipeline_f16_vec4
        } else if use_f16 {
            &self.pipeline_f16
        } else {
            &self.pipeline_bf16
        };

        encoder.setComputePipelineState(pipeline);
        unsafe { encoder.setBuffer_offset_atIndex(Some(gate_out), 0, 0); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(up_out), 0, 1); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(output), 0, 2); }

        let n_param = if n % 4 == 0 && use_f16 { n / 4 } else { n };

        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(&m as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new(&n_param as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                4,
            );
        }

        let threadgroup_size = (n as usize).min(1024);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: m as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();

        stream.commit()?;
        Ok(())
    }

    pub fn execute_concat(
        &self,
        stream: &mut MetalStream,
        gate_up: &Buffer,
        output: &Buffer,
        m: u32,
        n: u32,
        use_f16: bool,
    ) -> Result<(), MetalStreamError> {
        let cmd_buf = stream.get_command_buffer()?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
        })?;

        let pipeline = if use_f16 && n % 4 == 0 {
            &self.pipeline_f16_concat_vec4
        } else if use_f16 {
            &self.pipeline_f16_concat
        } else {
            &self.pipeline_bf16_concat
        };

        encoder.setComputePipelineState(pipeline);
        unsafe { encoder.setBuffer_offset_atIndex(Some(gate_up), 0, 0); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(output), 0, 1); }

        let n_param = if n % 4 == 0 && use_f16 { n / 4 } else { n };

        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(&m as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                2,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new(&n_param as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                3,
            );
        }

        let threadgroup_size = (n as usize).min(1024);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: m as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();

        stream.commit()?;
        Ok(())
    }

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
        let cmd_buf = stream.get_command_buffer()?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
        })?;

        let pipeline = if exact {
            &self.pipeline_gelu_exact_f16
        } else {
            &self.pipeline_gelu_f16
        };

        encoder.setComputePipelineState(pipeline);
        unsafe { encoder.setBuffer_offset_atIndex(Some(gate_out), 0, 0); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(up_out), 0, 1); }
        unsafe { encoder.setBuffer_offset_atIndex(Some(output), 0, 2); }

        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::new(&m as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                3,
            );
            encoder.setBytes_length_atIndex(
                NonNull::new(&n as *const u32 as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                4,
            );
        }

        let threadgroup_size = (n as usize).min(1024);
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: m as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threadgroup_size,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();

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
