// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shader compilation and caching infrastructure.

use metal::{CompileOptions, ComputePipelineState, Device, Library};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::stream::MetalStreamError;

/// Cache for compiled Metal shaders
pub struct ShaderCache {
    device: Device,
    libraries: HashMap<String, Library>,
    pipelines: Mutex<HashMap<String, ComputePipelineState>>,
}

impl ShaderCache {
    /// Create a new shader cache with all shader libraries
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        let mut libraries = HashMap::new();

        // Compile activation.metal
        let activation_source = include_str!("../shaders/activation.metal");
        let activation_lib = device
            .new_library_with_source(activation_source, &CompileOptions::new())
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("activation.metal: {:?}", e))
            })?;
        libraries.insert("activation".to_string(), activation_lib);

        // Compile rope.metal
        let rope_source = include_str!("../shaders/rope.metal");
        let rope_lib = device
            .new_library_with_source(rope_source, &CompileOptions::new())
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("rope.metal: {:?}", e))
            })?;
        libraries.insert("rope".to_string(), rope_lib);

        // Compile rmsnorm.metal
        let rmsnorm_source = include_str!("../shaders/rmsnorm.metal");
        let rmsnorm_lib = device
            .new_library_with_source(rmsnorm_source, &CompileOptions::new())
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("rmsnorm.metal: {:?}", e))
            })?;
        libraries.insert("rmsnorm".to_string(), rmsnorm_lib);

        // Compile fused_add_rmsnorm.metal
        let fused_add_rmsnorm_source = include_str!("../shaders/fused_add_rmsnorm.metal");
        let fused_add_rmsnorm_lib = device
            .new_library_with_source(fused_add_rmsnorm_source, &CompileOptions::new())
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "fused_add_rmsnorm.metal: {:?}",
                    e
                ))
            })?;
        libraries.insert("fused_add_rmsnorm".to_string(), fused_add_rmsnorm_lib);

        // Compile fused_gate_up_silu_mul.metal
        let fused_swiglu_source = include_str!("../shaders/fused_gate_up_silu_mul.metal");
        let fused_swiglu_lib = device
            .new_library_with_source(fused_swiglu_source, &CompileOptions::new())
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "fused_gate_up_silu_mul.metal: {:?}",
                    e
                ))
            })?;
        libraries.insert("fused_gate_up_silu_mul".to_string(), fused_swiglu_lib);

        // Compile awq_dequantize.metal
        let awq_source = include_str!("../shaders/awq_dequantize.metal");
        let awq_lib = device
            .new_library_with_source(awq_source, &CompileOptions::new())
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("awq_dequantize.metal: {:?}", e))
            })?;
        libraries.insert("awq_dequantize".to_string(), awq_lib);

        // Compile elementwise.metal
        let elementwise_source = include_str!("../shaders/elementwise.metal");
        let elementwise_lib = device
            .new_library_with_source(elementwise_source, &CompileOptions::new())
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("elementwise.metal: {:?}", e))
            })?;
        libraries.insert("elementwise".to_string(), elementwise_lib);

        // Compile embed.metal
        let embed_source = include_str!("../shaders/embed.metal");
        let embed_lib = device
            .new_library_with_source(embed_source, &CompileOptions::new())
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("embed.metal: {:?}", e))
            })?;
        libraries.insert("embed".to_string(), embed_lib);

        Ok(Self {
            device,
            libraries,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    /// Get or compile a pipeline for a given kernel name
    pub fn get_pipeline(&self, name: &str) -> Result<ComputePipelineState, MetalStreamError> {
        // Check cache first
        {
            let pipelines = self.pipelines.lock().unwrap();
            if let Some(pipeline) = pipelines.get(name) {
                return Ok(pipeline.clone());
            }
        }

        // Determine which library contains this kernel
        let library = if name.starts_with("rope_") {
            self.libraries.get("rope")
        } else if name.starts_with("rmsnorm_") {
            self.libraries.get("rmsnorm")
        } else if name.starts_with("fused_add_rmsnorm_") {
            self.libraries.get("fused_add_rmsnorm")
        } else if name.starts_with("fused_gate_up_silu_mul_") {
            self.libraries.get("fused_gate_up_silu_mul")
        } else if name.starts_with("awq_") {
            self.libraries.get("awq_dequantize")
        } else if name.starts_with("add_")
            || name.starts_with("mul_")
            || name.starts_with("sub_")
            || name.starts_with("scalar_mul_")
            || name.starts_with("bias_add_")
            || name.starts_with("tanh_soft_cap_")
        {
            self.libraries.get("elementwise")
        } else if name.starts_with("embed_") {
            self.libraries.get("embed")
        } else {
            self.libraries.get("activation")
        }
        .ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "No library found for kernel '{}'",
                name
            ))
        })?;

        // Compile pipeline
        let function = library.get_function(name, None).map_err(|e| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "Failed to get function '{}': {:?}",
                name, e
            ))
        })?;

        // CRITICAL: Create pipeline with ICB support enabled
        // This is required for Indirect Command Buffers to work on Apple Silicon
        use metal::MTLPipelineOption;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "Failed to create pipeline for '{}': {}",
                    name, e
                ))
            })?;

        // Note: supportIndirectCommandBuffers must be set via MTLComputePipelineDescriptor
        // in Objective-C when creating the pipeline. The metal-rs crate doesn't expose
        // this API yet, so we'll need to use objc directly in the ICB recording context.

        // Cache pipeline
        {
            let mut pipelines = self.pipelines.lock().unwrap();
            pipelines.insert(name.to_string(), pipeline.clone());
        }

        Ok(pipeline)
    }
}
