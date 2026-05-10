// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Specialized pipeline cache for ferrite-metal Phase 5.B.
//!
//! Builds (and memoizes) `MTLComputePipelineState` objects keyed on the
//! `(kernel name, function-constant bag)` pair.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary, MTLPipelineOption,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type FunctionConstantValues = Retained<MTLFunctionConstantValues>;

type MetallibBytes = &'static [u8];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstantType {
    UInt,
    Float,
}

impl ConstantType {
    fn metal_data_type(self) -> MTLDataType {
        match self {
            ConstantType::UInt => MTLDataType::UInt,
            ConstantType::Float => MTLDataType::Float,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstantValue {
    pub index: u16,
    pub bits: u32,
    pub ty: ConstantType,
}

impl ConstantValue {
    pub fn uint(index: u16, value: u32) -> Self {
        Self {
            index,
            bits: value,
            ty: ConstantType::UInt,
        }
    }

    pub fn float(index: u16, value: f32) -> Self {
        Self {
            index,
            bits: value.to_bits(),
            ty: ConstantType::Float,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub kernel_name: &'static str,
    pub library_name: &'static str,
    pub constants: Vec<ConstantValue>,
}

impl PipelineKey {
    pub fn new(
        library_name: &'static str,
        kernel_name: &'static str,
        mut constants: Vec<ConstantValue>,
    ) -> Self {
        constants.sort_by_key(|c| c.index);
        Self {
            kernel_name,
            library_name,
            constants,
        }
    }
}

pub struct SpecializedPipelineCache {
    device: Device,
    libraries: HashMap<&'static str, Library>,
    pipelines: Mutex<HashMap<PipelineKey, ComputePipelineState>>,
}

impl SpecializedPipelineCache {
    pub fn new(
        device: Device,
        libraries_in: &[(&'static str, MetallibBytes)],
    ) -> Result<Self, MetalStreamError> {
        let mut libraries = HashMap::with_capacity(libraries_in.len());
        for (name, bytes) in libraries_in {
            let lib = load_library_from_bytes(&device, bytes).map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load library `{name}`: {e}"))
            })?;
            libraries.insert(*name, lib);
        }
        Ok(Self {
            device,
            libraries,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    #[cfg(test)]
    pub(crate) fn from_sources(
        device: Device,
        sources: &[(&'static str, &str)],
    ) -> Result<Self, MetalStreamError> {
        let mut libraries = HashMap::with_capacity(sources.len());
        for (name, source) in sources {
            let opts = objc2_metal::MTLCompileOptions::new();
            let ns_source = NSString::from_str(source);
            let lib = device
                .newLibraryWithSource_options_error(&ns_source, Some(&opts))
                .map_err(|e| {
                    MetalStreamError::ShaderCompilationFailed(format!(
                        "compile library `{name}`: {e:?}"
                    ))
                })?;
            libraries.insert(*name, lib);
        }
        Ok(Self {
            device,
            libraries,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    pub fn with_standard_shaders(device: Device) -> Result<Self, MetalStreamError> {
        Self::new(
            device,
            &[
                ("rmsnorm", crate::embedded_metallib!("rmsnorm")),
                (
                    "fused_add_rmsnorm",
                    crate::embedded_metallib!("fused_add_rmsnorm"),
                ),
                (
                    "fused_gate_up_silu_mul",
                    crate::embedded_metallib!("fused_gate_up_silu_mul"),
                ),
                ("attention", crate::embedded_metallib!("attention")),
                ("rope", crate::embedded_metallib!("rope")),
                ("embed", crate::embedded_metallib!("embed")),
                ("activation", crate::embedded_metallib!("activation")),
                ("elementwise", crate::embedded_metallib!("elementwise")),
                (
                    "quantized_dequantize",
                    crate::embedded_metallib!("quantized_dequantize"),
                ),
                (
                    "quantized_qmv",
                    crate::embedded_metallib!("quantized_qmv"),
                ),
                (
                    "quantized_qmm",
                    crate::embedded_metallib!("quantized_qmm"),
                ),
                ("gemm", crate::embedded_metallib!("gemm")),
            ],
        )
    }

    pub fn len(&self) -> usize {
        self.pipelines.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_or_build(
        &self,
        key: &PipelineKey,
    ) -> Result<ComputePipelineState, MetalStreamError> {
        {
            let map = self.pipelines.lock().unwrap();
            if let Some(p) = map.get(key) {
                return Ok(p.clone());
            }
        }

        let library = self.libraries.get(key.library_name).ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "no library `{}` in SpecializedPipelineCache (call `new` with this library)",
                key.library_name
            ))
        })?;

        let constants = MTLFunctionConstantValues::new();
        for c in &key.constants {
            unsafe {
                constants.setConstantValue_type_atIndex(
                    NonNull::new(&c.bits as *const u32 as *mut c_void).unwrap(),
                    c.ty.metal_data_type(),
                    c.index as usize,
                );
            }
        }

        let ns_name = NSString::from_str(key.kernel_name);
        let function = library
            .newFunctionWithName_constantValues_error(&ns_name, &constants)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "newFunctionWithName(`{}`, {} constants) in library `{}`: {e:?}",
                    key.kernel_name,
                    key.constants.len(),
                    key.library_name,
                ))
            })?;

        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setSupportIndirectCommandBuffers(true);
        let pipeline = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "build pipeline `{}`: {e:?}",
                    key.kernel_name,
                ))
            })?;

        let mut map = self.pipelines.lock().unwrap();
        Ok(map.entry(key.clone()).or_insert(pipeline).clone())
    }
}
