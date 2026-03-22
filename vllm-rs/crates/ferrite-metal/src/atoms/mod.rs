/// Metal atom traits — pluggable components of the kernel pipeline.
///
/// Each atom emits a chunk of MSL for one phase of the kernel.
/// The emitter controls the loop structure and composes atoms.
///
/// Inner loop structure (per K-step of 8):
/// ```text
///   1. FragmentLoadAtom::emit_load_a()  — threadgroup → A_mat register
///   2. TransformAtom::emit_transform()  — modify A_mat in-place
///   3. FragmentLoadAtom::emit_load_b()  — threadgroup → B_mat register
///   4. MmaAtom::emit_multiply()         — C_sram += A_mat × B_mat
/// ```
///
/// Atoms assume these variables exist in scope (emitted by the emitter):
/// - `A_block`, `B_block`: threadgroup pointers
/// - `A_mat`, `B_mat`: simdgroup_matrix locals
/// - `C_sram[tm][tn]`: accumulator array
/// - `kt`, `tm`, `tn`: tile loop indices
mod element_mul;
mod epilogue;
mod fragment_load;
mod mma;
mod residual_add;
pub mod rmsnorm;
pub mod rope;
mod tile_copy;
mod transform;

pub use element_mul::*;
pub use epilogue::*;
pub use fragment_load::*;
pub use mma::*;
pub use residual_add::*;
pub use rmsnorm::*;
pub use rope::*;
pub use tile_copy::*;
pub use transform::*;

use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

// ═══════════════════════════════════════════════════════════════════
// Trait definitions
// ═══════════════════════════════════════════════════════════════════

/// How tiles move from device to threadgroup memory.
pub trait TileCopyAtom {
    fn emit_tile_load(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
    fn emit_tile_sync(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

/// How 8×8 fragments are loaded from threadgroup into registers.
pub trait FragmentLoadAtom {
    fn emit_load_a(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
    fn emit_load_b(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

/// Transforms A_mat between load and multiply.
pub trait TransformAtom {
    fn emit_prologue(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}
    fn emit_k_setup(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}
    fn emit_transform(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}
    fn is_identity(&self) -> bool {
        false
    }
}

/// Emits simdgroup_multiply_accumulate.
pub trait MmaAtom {
    fn emit_multiply(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

/// Post-processes C_sram accumulators before store.
pub trait EpilogueAtom {
    fn emit_epilogue(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}
    fn is_identity(&self) -> bool {
        false
    }
}
