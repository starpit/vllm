// SPDX-License-Identifier: Apache-2.0
//! IType trait — every per-canonical fused operation implements this.
//!
//! Phase 0: trait shape only; per-IType impls land in Phases 1-6 under
//! `mk/itypes/`.

use super::instruction::Instruction;
use crate::lower::OpDesc;

/// Static description of a single IType (kernel-fragment) family.
pub struct ITypeMeta {
    /// `#include "itypes/<file>.cuh"` filename without directory.
    pub cpp_include: &'static str,
    /// C++ struct name to instantiate. The macro emits
    /// `using <opcode_name> = <cpp_template>::tmpl<HIDDEN, INTER, ...>;`.
    pub cpp_template: &'static str,
    /// Stable opcode discriminant. Persistent multi-CTA scheduler
    /// dispatches on this value.
    pub opcode: u16,
    /// Number of in-kernel atomic-counted dataflow barriers this IType
    /// allocates per layer. Asserted byte-for-byte against the .cuh
    /// `static constexpr int NUM_BARRIERS` at canonical emit.
    pub num_barriers_per_layer: u32,
}

/// Trait every IType impl must satisfy. The macro walks the fusion
/// pass output and calls `lower(...)` per fused op to produce the
/// instruction stream.
pub trait IType {
    fn meta() -> ITypeMeta;

    /// Lower one fused op from the `LoweredOp` topo into a sequence of
    /// `Instruction` records (one per kernel invocation slot for this
    /// op — typically one per layer or one per (layer, kv_page) tuple).
    fn lower(op: &OpDesc) -> Vec<Instruction>;
}
