// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shader compilation and caching infrastructure.

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::stream::MetalStreamError;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Cache for compiled Metal shaders
pub struct ShaderCache {
    device: Device,
    libraries: HashMap<String, Library>,
    pipelines: Mutex<HashMap<String, ComputePipelineState>>,
}

impl ShaderCache {
    /// Create a new shader cache with all shader libraries.
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
            let lib = load_library_from_bytes(&device, bytes).map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load `{name}.metallib`: {e}"))
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
        {
            let pipelines = self.pipelines.lock().unwrap();
            if let Some(pipeline) = pipelines.get(name) {
                return Ok(pipeline.clone());
            }
        }

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

        let ns_name = NSString::from_str(name);
        let function = library.newFunctionWithName(&ns_name).ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "Failed to get function '{}'",
                name
            ))
        })?;

        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "Failed to create pipeline for '{}': {:?}",
                    name, e
                ))
            })?;

        {
            let mut pipelines = self.pipelines.lock().unwrap();
            pipelines.insert(name.to_string(), pipeline.clone());
        }

        Ok(pipeline)
    }
}

/// Wrap a static byte slice as a `dispatch_data_t` and load it as a Metal
/// library. The bytes typically come from `include_bytes!` so we keep the
/// destructor as no-op (default behavior of `DispatchData::from`'s
/// implementation copies into a managed buffer).
pub(crate) fn load_library_from_bytes(
    device: &Device,
    bytes: &'static [u8],
) -> Result<Library, String> {
    let data = DispatchData::from(bytes);
    device
        .newLibraryWithData_error(&data)
        .map_err(|e| format!("newLibraryWithData failed: {:?}", e))
}
