// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Specialized pipeline cache for ferrite-metal Phase 5.B.
//!
//! Builds (and memoizes) `MTLComputePipelineState` objects keyed on the
//! `(kernel name, function-constant bag)` pair. The Metal interpreter
//! (`ferrite-forward::interpreter::metal::pipelines`) populates the bag
//! from `CanonicalParams` + bucket M + per-kernel extras (eps, scale,
//! …) and asks the cache for the matching pipeline once per
//! `(model variant, bucket, kernel)` at worker init time.
//!
//! Distinct from [`crate::shader_cache::ShaderCache`]: that one keys on
//! kernel name only and never sets function constants. Both can coexist
//! during the Phase 5.B → 5.C transition (legacy ICB recorders still
//! reach the un-specialized cache; the new worker uses this one).

use metal::{
    ComputePipelineState, Device, FunctionConstantValues, Library, MTLDataType, NSUInteger,
};
use std::collections::HashMap;
use std::sync::Mutex;

use crate::stream::MetalStreamError;

/// Type tag for a function-constant value. Restricted to the two MSL
/// scalar types our shaders use today; expand as new shapes arrive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstantType {
    /// `uint` in MSL; `MTLDataType::UInt` at the API.
    UInt,
    /// `float` in MSL; `MTLDataType::Float` at the API.
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

/// One function-constant slot: index in the shader's
/// `[[function_constant(N)]]` numbering, the value bits (u32 for
/// `UInt`, `f32::to_bits()` for `Float`), and the type discriminator.
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

/// Hashable cache key. The `&'static str` kernel name is small and
/// hashed by pointer-equality on the interned literal in practice.
/// `constants` is kept sorted by `index` so the same logical bag
/// always hashes the same way regardless of insertion order.
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

/// Caches function-constant-specialized pipelines.
///
/// One instance per Metal `Device`. Holds a pre-compiled library per
/// shader file (same files as the legacy [`ShaderCache`]) plus a
/// pipeline map. `get_or_build()` is the only mutating entry point —
/// safe to share across threads via `Arc`.
pub struct SpecializedPipelineCache {
    device: Device,
    libraries: HashMap<&'static str, Library>,
    pipelines: Mutex<HashMap<PipelineKey, ComputePipelineState>>,
}

impl SpecializedPipelineCache {
    /// Compile every Metal shader source the cache knows about. The
    /// argument list is `(library_name, source)`; library names are
    /// the same `&'static str` callers pass in `PipelineKey` so the
    /// hashmap lookup is identity-cheap.
    pub fn new(device: Device, sources: &[(&'static str, &str)]) -> Result<Self, MetalStreamError> {
        let mut libraries = HashMap::with_capacity(sources.len());
        for (name, source) in sources {
            let opts = metal::CompileOptions::new();
            let lib = device.new_library_with_source(source, &opts).map_err(|e| {
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

    /// Construct with the standard ferrite-metal shader set.
    ///
    /// Kept here (rather than forcing every caller to pass the same
    /// bundle) so the worker can stand up the cache with one call
    /// once the Phase 5.B shader rewrites land.
    pub fn with_standard_shaders(device: Device) -> Result<Self, MetalStreamError> {
        Self::new(
            device,
            &[
                ("rmsnorm", include_str!("../shaders/rmsnorm.metal")),
                (
                    "fused_add_rmsnorm",
                    include_str!("../shaders/fused_add_rmsnorm.metal"),
                ),
                (
                    "fused_gate_up_silu_mul",
                    include_str!("../shaders/fused_gate_up_silu_mul.metal"),
                ),
                ("attention", include_str!("../shaders/attention.metal")),
                ("rope", include_str!("../shaders/rope.metal")),
                ("embed", include_str!("../shaders/embed.metal")),
                ("activation", include_str!("../shaders/activation.metal")),
                ("elementwise", include_str!("../shaders/elementwise.metal")),
                (
                    "awq_dequantize",
                    include_str!("../shaders/awq_dequantize.metal"),
                ),
            ],
        )
    }

    /// Number of distinct pipelines currently cached. Useful for
    /// asserting that subsequent `get_or_build` calls hit the cache
    /// rather than rebuilding.
    pub fn len(&self) -> usize {
        self.pipelines.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieve (or build) the pipeline state for `key`. Pipelines are
    /// cheap-clone — Metal returns a reference-counted handle.
    pub fn get_or_build(
        &self,
        key: &PipelineKey,
    ) -> Result<ComputePipelineState, MetalStreamError> {
        // Fast path: cache hit.
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

        // Populate function constants in the order the caller
        // specified — duplicates at the same index are an error
        // from the caller's perspective; metal silently keeps the
        // last write so we don't bother detecting it here.
        let constants = FunctionConstantValues::new();
        for c in &key.constants {
            let value_ptr = &c.bits as *const u32 as *const std::ffi::c_void;
            constants.set_constant_value_at_index(
                value_ptr,
                c.ty.metal_data_type(),
                c.index as NSUInteger,
            );
        }

        let function = library
            .get_function(key.kernel_name, Some(constants))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "get_function(`{}`, {} constants) in library `{}`: {e:?}",
                    key.kernel_name,
                    key.constants.len(),
                    key.library_name,
                ))
            })?;

        // Build via descriptor so we can flag
        // `supportIndirectCommandBuffers=true`. ICB execution under
        // `inheritPipelineState=true` requires every encoder-bound
        // pipeline to advertise ICB support; pipelines built via the
        // shorter `new_compute_pipeline_state_with_function` path
        // default to NO and trip the validation layer with
        // "compute pipeline set on this encoder does not support
        // indirect command buffers". This is the only deviation from
        // the legacy `ShaderCache` builder.
        let descriptor = metal::ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&function));
        descriptor.set_support_indirect_command_buffers(true);
        let pipeline = self
            .device
            .new_compute_pipeline_state(&descriptor)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "build pipeline `{}`: {e}",
                    key.kernel_name,
                ))
            })?;

        let mut map = self.pipelines.lock().unwrap();
        // Race window: another thread may have inserted the same
        // key between our miss-check and the build. Cloning is
        // cheap, so just keep the first winner.
        Ok(map.entry(key.clone()).or_insert(pipeline).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use foreign_types::ForeignType;

    /// Tiny inline shader with a single function constant. Lets the
    /// test exercise the `FunctionConstantValues` path without the
    /// kernels we haven't rewritten yet (Phase 5.B.3+).
    const PROBE_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint PROBE_M [[function_constant(0)]];

kernel void probe_const(
    device uint* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < PROBE_M) {
        out[tid] = PROBE_M;
    }
}
"#;

    fn skip_unless_metal_available() -> Option<Device> {
        crate::detect_device().map(|d| d.device.clone())
    }

    #[test]
    fn caches_pipelines_by_constant_bag() {
        let Some(device) = skip_unless_metal_available() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let cache = SpecializedPipelineCache::new(device, &[("probe", PROBE_SHADER)])
            .expect("compile probe library");

        // Two distinct constant bags → two distinct pipelines.
        let k1 = PipelineKey::new("probe", "probe_const", vec![ConstantValue::uint(0, 32)]);
        let k2 = PipelineKey::new("probe", "probe_const", vec![ConstantValue::uint(0, 64)]);

        let p1 = cache.get_or_build(&k1).unwrap();
        let p2 = cache.get_or_build(&k2).unwrap();
        assert_eq!(cache.len(), 2);

        // Same bag → cache hit (no growth).
        let p1_again = cache.get_or_build(&k1).unwrap();
        assert_eq!(cache.len(), 2);
        // The two distinct pipelines are different objects.
        assert!(!std::ptr::eq(
            p1.as_ptr() as *const _,
            p2.as_ptr() as *const _
        ));
        // Same bag returns same underlying object.
        assert!(std::ptr::eq(
            p1.as_ptr() as *const _,
            p1_again.as_ptr() as *const _
        ));
    }

    #[test]
    fn missing_library_errors_clearly() {
        let Some(device) = skip_unless_metal_available() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let cache = SpecializedPipelineCache::new(device, &[("probe", PROBE_SHADER)])
            .expect("compile probe library");
        let bogus = PipelineKey::new("not_compiled", "kernel", vec![]);
        let err = cache.get_or_build(&bogus).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("not_compiled"), "expected lib name in: {msg}");
    }

    #[test]
    fn unknown_function_errors_clearly() {
        let Some(device) = skip_unless_metal_available() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let cache = SpecializedPipelineCache::new(device, &[("probe", PROBE_SHADER)])
            .expect("compile probe library");
        let bogus = PipelineKey::new("probe", "nonexistent_kernel", vec![]);
        let err = cache.get_or_build(&bogus).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("nonexistent_kernel"),
            "expected fn name in: {msg}"
        );
    }
}
