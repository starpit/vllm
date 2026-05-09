// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shader compilation and caching infrastructure.

use metal::{ComputePipelineState, Device, Library};
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
    /// Create a new shader cache with all shader libraries.
    ///
    /// Loads precompiled `.metallib` blobs produced by `build.rs`
    /// (one per source `.metal` file under `shaders/`) via
    /// `new_library_with_data`. Replaces the previous
    /// `new_library_with_source` path that JIT-compiled MSL on every
    /// process start.
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        let mut libraries = HashMap::new();
        for (name, bytes) in [
            ("activation", &crate::embedded_metallib!("activation")[..]),
            ("rope", &crate::embedded_metallib!("rope")[..]),
            ("rmsnorm", &crate::embedded_metallib!("rmsnorm")[..]),
            (
                "fused_add_rmsnorm",
                &crate::embedded_metallib!("fused_add_rmsnorm")[..],
            ),
            (
                "fused_gate_up_silu_mul",
                &crate::embedded_metallib!("fused_gate_up_silu_mul")[..],
            ),
            (
                "awq_dequantize",
                &crate::embedded_metallib!("awq_dequantize")[..],
            ),
        ] {
            let lib = device.new_library_with_data(bytes).map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load `{name}.metallib`: {e:?}"))
            })?;
            libraries.insert(name.to_string(), lib);
        }
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

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "Failed to create pipeline for '{}': {}",
                    name, e
                ))
            })?;

        // Cache pipeline
        {
            let mut pipelines = self.pipelines.lock().unwrap();
            pipelines.insert(name.to_string(), pipeline.clone());
        }

        Ok(pipeline)
    }
}
