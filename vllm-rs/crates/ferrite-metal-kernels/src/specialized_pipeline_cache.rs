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
    MTL4Compiler, MTL4CompilerDescriptor, MTL4ComputePipelineDescriptor,
    MTL4IndirectCommandBufferSupportState, MTL4LibraryFunctionDescriptor,
    MTL4SpecializedFunctionDescriptor, MTLComputePipelineState, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

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
    /// Signed 32-bit. Required for MLX-port kernels whose function
    /// constants are declared `constant int` — Metal validates the
    /// type byte-for-byte against the constant declaration, so a
    /// `uint` payload bound to an `int` slot fails pipeline build
    /// with `MTLLibraryErrorDomain` 3 ("Constant X is of type
    /// MTLDataTypeInt but value found has type MTLDataTypeUInt").
    Int,
    Float,
    /// MTLDataType::Bool. The bit payload is a single byte (0/1) but
    /// Metal validates the declared type per-constant — passing a UInt
    /// to a `constant bool [[function_constant(N)]]` slot fails with
    /// `MTLLibraryErrorDomain` 3. Required for MLX-port kernels whose
    /// function constants are `align_Q` / `align_K` / `has_mask` /
    /// `do_causal` / `has_sinks` (the steel_attention family).
    Bool,
}

impl ConstantType {
    pub(crate) fn metal_data_type(self) -> MTLDataType {
        match self {
            ConstantType::UInt => MTLDataType::UInt,
            ConstantType::Int => MTLDataType::Int,
            ConstantType::Float => MTLDataType::Float,
            ConstantType::Bool => MTLDataType::Bool,
        }
    }
}

/// `[[function_constant(N)]]` slot index newtype.
///
/// Distinct from a raw `u16` so that a `ConstantValue::uint(slot,
/// value)` call can't have its two arguments swapped — the value
/// (`u32`) doesn't satisfy `Into<ConstSlot>`. Phase 1 of the
/// type-safety plan; see `FERRITE_METAL_TYPE_SAFETY_PLAN.md`.
///
/// `From<u16>` is provided so existing literal call sites
/// `ConstantValue::uint(0u16, …)` keep working unchanged; Phase 2's
/// per-kernel constants struct migrates the literals to typed slot
/// references.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstSlot(pub u16);

impl ConstSlot {
    pub fn get(self) -> u16 {
        self.0
    }
}

impl From<u16> for ConstSlot {
    fn from(v: u16) -> Self {
        Self(v)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstantValue {
    pub index: u16,
    pub bits: u32,
    pub ty: ConstantType,
}

impl ConstantValue {
    pub fn uint(index: impl Into<ConstSlot>, value: u32) -> Self {
        Self {
            index: index.into().0,
            bits: value,
            ty: ConstantType::UInt,
        }
    }

    /// Signed 32-bit constant. Use for kernels whose function
    /// constants are declared `constant int` (the qmv / qmm_t MLX
    /// ports — see `quantized_qmv.metal::IN_VEC_SIZE` /
    /// `quantized_qmm.metal::QMM_K`).
    pub fn int(index: impl Into<ConstSlot>, value: i32) -> Self {
        Self {
            index: index.into().0,
            bits: value as u32,
            ty: ConstantType::Int,
        }
    }

    pub fn float(index: impl Into<ConstSlot>, value: f32) -> Self {
        Self {
            index: index.into().0,
            bits: value.to_bits(),
            ty: ConstantType::Float,
        }
    }

    pub fn boolean(index: impl Into<ConstSlot>, value: bool) -> Self {
        Self {
            index: index.into().0,
            bits: u32::from(value),
            ty: ConstantType::Bool,
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
    /// Lazy-built MTL4 compiler. Pipelines created through this
    /// compiler are MTL4 ICB-aware (see
    /// `MTL4IndirectCommandBufferSupportState::Enabled` below); the
    /// returned `MTLComputePipelineState` is the same legacy type
    /// that direct `setComputePipelineState` accepts AND that
    /// `executeCommandsInBuffer` plays back correctly inside an
    /// `MTL4ComputeCommandEncoder`. The legacy
    /// `MTLDevice.newComputePipelineStateWithDescriptor(.., MTL3
    /// `MTLComputePipelineDescriptor`, ..)` path produces
    /// MTL3-ICB-only pipelines whose ICB execution silently produces
    /// random outputs on MTL4 encoders.
    compiler: OnceLock<Retained<ProtocolObject<dyn MTL4Compiler>>>,
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
            compiler: OnceLock::new(),
        })
    }

    /// Register a synthesized kernel library from precompiled metallib
    /// bytes. The macro AOT-compiles synthesized `.metal` sources at
    /// proc-macro expansion time (shells out to `xcrun metal -c` +
    /// `xcrun metallib`) and embeds the resulting bytes as
    /// `&'static [u8]`. Same `newLibraryWithData` path used by all
    /// hand-written shaders — NOT the runtime MSL→AIR compile path
    /// (`newLibraryWithSource`), which produces different binaries
    /// across Apple GPU generations and was the source of a real M1
    /// runtime failure.
    ///
    /// Used by the compiler-driven megakernel synthesis pass
    /// (`ferrite-forward-macro/src/fuse_pass.rs`).
    pub fn register_metallib_library(
        &mut self,
        name: &'static str,
        bytes: &'static [u8],
    ) -> Result<(), MetalStreamError> {
        let lib = load_library_from_bytes(&self.device, bytes).map_err(|e| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "load synthesized metallib `{name}`: {e}"
            ))
        })?;
        self.libraries.insert(name, lib);
        Ok(())
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
            compiler: OnceLock::new(),
        })
    }

    pub fn with_standard_shaders(device: Device) -> Result<Self, MetalStreamError> {
        let mut cache = Self::new(
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
                (
                    "fused_qkv_rope_cache",
                    crate::embedded_metallib!("fused_qkv_rope_cache"),
                ),
                (
                    "fused_affine_qkv_rope_cache",
                    crate::embedded_metallib!("fused_affine_qkv_rope_cache"),
                ),
                ("attention", crate::embedded_metallib!("attention")),
                (
                    "attention_steel",
                    crate::embedded_metallib!("attention_steel"),
                ),
                (
                    "attention_steel_paged",
                    crate::embedded_metallib!("attention_steel_paged"),
                ),
                (
                    "gather_last_token",
                    crate::embedded_metallib!("gather_last_token"),
                ),
                ("rope", crate::embedded_metallib!("rope")),
                ("embed", crate::embedded_metallib!("embed")),
                ("activation", crate::embedded_metallib!("activation")),
                ("elementwise", crate::embedded_metallib!("elementwise")),
                (
                    "quantized_dequantize",
                    crate::embedded_metallib!("quantized_dequantize"),
                ),
                ("quantized_qmv", crate::embedded_metallib!("quantized_qmv")),
                ("quantized_qmm", crate::embedded_metallib!("quantized_qmm")),
                // `quantized_qmm_nax` is intentionally absent — runtime-
                // compiled below (offline metallib miscompiles MPP).
                ("quantized_qvm", crate::embedded_metallib!("quantized_qvm")),
                (
                    "quantized_splitk_reduce",
                    crate::embedded_metallib!("quantized_splitk_reduce"),
                ),
                ("silu_mul", crate::embedded_metallib!("silu_mul")),
                // PD-wavefront persistent decode megakernel / trivial tape
                // player (bindless operands via gpuAddress table).
                (
                    "wavefront_layer",
                    crate::embedded_metallib!("wavefront_layer"),
                ),
                ("gemm", crate::embedded_metallib!("gemm")),
                // MoE-on-Metal: router decomposition kernels.
                // `lower_metal_moe` (Phase A) emits commands that
                // reference these libraries by name; without them
                // the per-(library, function) pipeline lookup in
                // `get_or_build` panics at first MoE forward.
                ("softmax", crate::embedded_metallib!("softmax")),
                ("argpartition", crate::embedded_metallib!("argpartition")),
                (
                    "take_along_axis",
                    crate::embedded_metallib!("take_along_axis"),
                ),
                (
                    "slice_trailing_cols",
                    crate::embedded_metallib!("slice_trailing_cols"),
                ),
                (
                    "moe_weighted_sum",
                    crate::embedded_metallib!("moe_weighted_sum"),
                ),
            ],
        )?;
        // NAX qmm_t (`affine_qmm_t_nax_*`) MUST be compiled from source at
        // runtime via `newLibraryWithSource`: the offline `xcrun metal`
        // metallib toolchain miscompiles MetalPerformancePrimitives
        // `matmul2d` cooperative tensors (each MMA reduces only half its
        // K → ~95%-wrong qmm_t). The runtime compiler is correct; this is
        // also the path mlx uses. The cross-generation `newLibraryWithSource`
        // caveat noted on `register_metallib_library` does not apply here —
        // NAX only runs on M5+ (`is_nax_capable`), and there it's the only
        // correct option.
        let nax_lib = crate::shader_cache::compile_nax_library_from_source(&cache.device)
            .map_err(MetalStreamError::ShaderCompilationFailed)?;
        cache.libraries.insert("quantized_qmm_nax", nax_lib);
        Ok(cache)
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

        // MTL4 function-descriptor chain:
        //   MTL4LibraryFunctionDescriptor       (where to find the
        //                                         function: library +
        //                                         entry-point name)
        //   MTL4SpecializedFunctionDescriptor   (wraps it + bakes
        //                                         function-constant
        //                                         values for this
        //                                         specialization)
        //   MTL4ComputePipelineDescriptor       (the descriptor the
        //                                         compiler consumes;
        //                                         carries the MTL4
        //                                         ICB-support flag)
        let lib_fn_desc = MTL4LibraryFunctionDescriptor::new();
        lib_fn_desc.setName(Some(&ns_name));
        lib_fn_desc.setLibrary(Some(library));

        let spec_fn_desc = MTL4SpecializedFunctionDescriptor::new();
        // Upcast: spec descriptor's `setFunctionDescriptor` accepts
        // any `MTL4FunctionDescriptor` subclass.
        let lib_fn_super: &::objc2_metal::MTL4FunctionDescriptor = &lib_fn_desc;
        spec_fn_desc.setFunctionDescriptor(Some(lib_fn_super));
        spec_fn_desc.setConstantValues(Some(&constants));

        let pipe_desc = MTL4ComputePipelineDescriptor::new();
        let spec_fn_super: &::objc2_metal::MTL4FunctionDescriptor = &spec_fn_desc;
        pipe_desc.setComputeFunctionDescriptor(Some(spec_fn_super));
        // The whole point of this migration: MTL4 compute pipelines
        // need this enum-flavored support flag (NOT the bool one on
        // the legacy MTLComputePipelineDescriptor) for
        // `executeCommandsInBuffer` on an MTL4ComputeCommandEncoder
        // to fire the pipeline correctly. The MTL3 bool flag enables
        // MTL3-ICB compatibility only; pipelines created with it
        // appear to work in MTL4 direct dispatch but silently emit
        // wrong kernel state under MTL4 ICB execution.
        pipe_desc.setSupportIndirectCommandBuffers(MTL4IndirectCommandBufferSupportState::Enabled);

        let compiler = self.compiler.get_or_init(|| {
            let cdesc = MTL4CompilerDescriptor::new();
            self.device
                .newCompilerWithDescriptor_error(&cdesc)
                .expect("newCompilerWithDescriptor")
        });
        let pipeline = compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipe_desc, None)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "MTL4 build pipeline `{}` (lib `{}`, {} constants): {e:?}",
                    key.kernel_name,
                    key.library_name,
                    key.constants.len(),
                ))
            })?;

        let mut map = self.pipelines.lock().unwrap();
        Ok(map.entry(key.clone()).or_insert(pipeline).clone())
    }
}
