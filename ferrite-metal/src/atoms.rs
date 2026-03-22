/// Metal atom traits — pluggable components of the kernel pipeline.
///
/// Each atom emits a chunk of MSL for one phase of the GEMM kernel.
/// The emitter (gemm.rs) controls the loop structure and composes atoms.
///
/// Inner loop structure (per K-step of 8):
/// ```text
///   1. FragmentLoadAtom::emit_load_a()  — threadgroup → A_mat register
///   2. TransformAtom::emit_transform()  — modify A_mat in-place
///   3. FragmentLoadAtom::emit_load_b()  — threadgroup → B_mat register
///   4. MmaAtom::emit_multiply()         — C_sram += A_mat × B_mat
/// ```
///
/// Atoms assume these variables exist in scope (emitted by gemm.rs):
/// - `A_block`, `B_block`: threadgroup pointers
/// - `A_mat`, `B_mat`: simdgroup_matrix locals (declared by emitter)
/// - `C_sram[tm][tn]`: accumulator array
/// - `kt`, `tm`, `tn`: tile loop indices
/// - Config-derived constants via MslBuilder template variables
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

// ═══════════════════════════════════════════════════════════════════
// Tile Copy: device memory → threadgroup memory
// ═══════════════════════════════════════════════════════════════════

/// How tiles move from device to threadgroup memory.
///
/// Emits the cooperative copy at the start of each K-iteration.
/// Assumes `A_block`, `B_block`, `k`, `M_offset`, `N_offset`, etc. are in scope.
pub trait TileCopyAtom {
    fn emit_tile_load(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
    fn emit_tile_sync(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

// ═══════════════════════════════════════════════════════════════════
// Fragment Load: threadgroup → simdgroup registers
// ═══════════════════════════════════════════════════════════════════

/// How 8×8 fragments are loaded from threadgroup memory into registers.
///
/// Uses native `simdgroup_load` — the hardware handles per-lane distribution.
/// A and B are separate because the transform slot goes between them.
pub trait FragmentLoadAtom {
    /// Emit `simdgroup_load(A_mat, A_block, ...)` for one (kt, tm) tile.
    fn emit_load_a(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);

    /// Emit `simdgroup_load(B_mat, B_block, ...)` for one (kt, tn) tile.
    fn emit_load_b(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

// ═══════════════════════════════════════════════════════════════════
// Transform: in-place modification of A_mat before MMA
// ═══════════════════════════════════════════════════════════════════

/// Transforms A_mat between load and multiply.
///
/// Operates on `A_mat` (a `simdgroup_matrix<half, 8>`) already in registers.
/// The proc macro composes transforms to build fused kernels.
pub trait TransformAtom {
    /// One-time setup before the K-loop (e.g., load norm weights).
    fn emit_prologue(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}

    /// Per-K-step setup (e.g., index into gamma weights for this K chunk).
    fn emit_k_setup(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}

    /// Transform A_mat in-place. Called after load_a, before load_b.
    fn emit_transform(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}

    fn is_identity(&self) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════
// MMA: simdgroup matrix multiply-accumulate
// ═══════════════════════════════════════════════════════════════════

/// Emits `simdgroup_multiply_accumulate(C_sram[tm][tn], A_mat, B_mat, ...)`.
pub trait MmaAtom {
    fn emit_multiply(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

// ═══════════════════════════════════════════════════════════════════
// Epilogue: post-K-loop accumulator processing
// ═══════════════════════════════════════════════════════════════════

/// Post-processes C_sram accumulators before store.
/// Applied after the K-loop, before simdgroup_store.
pub trait EpilogueAtom {
    fn emit_epilogue(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}

    fn is_identity(&self) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════
// Implementations
// ═══════════════════════════════════════════════════════════════════

/// Identity transform — no modification to A_mat.
pub struct IdentityTransform;

impl TransformAtom for IdentityTransform {
    fn is_identity(&self) -> bool {
        true
    }
}

/// Identity epilogue — store accumulators directly.
pub struct IdentityEpilogue;

impl EpilogueAtom for IdentityEpilogue {
    fn is_identity(&self) -> bool {
        true
    }
}

/// Native simdgroup_load from threadgroup memory.
///
/// Emits `simdgroup_load(A_mat, A_block, lead, origin)` using
/// Metal 4's hardware intrinsic. No custom headers needed.
pub struct NativeFragmentLoad;

impl FragmentLoadAtom for NativeFragmentLoad {
    fn emit_load_a(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("A_LEAD", config.leading_block_dim('A').to_string());
        msl.block(
            r#"
simdgroup_load(A_mat, A_block, {{A_LEAD}}, ulong2(kt * 8, tm * 8));
"#,
        );
    }

    fn emit_load_b(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("B_LEAD", config.leading_block_dim('B').to_string());
        msl.block(
            r#"
simdgroup_load(B_mat, B_block, {{B_LEAD}}, ulong2(tn * 8, kt * 8));
"#,
        );
    }
}

/// Native simdgroup multiply-accumulate.
pub struct NativeMma;

impl MmaAtom for NativeMma {
    fn emit_multiply(&self, msl: &mut MslBuilder, _config: &MetalGemmConfig) {
        msl.raw("simdgroup_multiply_accumulate(C_sram[tm][tn], A_mat, B_mat, C_sram[tm][tn]);");
    }
}

/// Loop-based tile copy (universal, works on all Metal versions).
///
/// All threads cooperate to copy tiles from device to threadgroup memory
/// with zero-fill padding. No simdgroup_event dependency.
pub struct LoopTileCopy;

impl TileCopyAtom for LoopTileCopy {
    fn emit_tile_load(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        let a_lead = config.leading_block_dim('A');
        let b_lead = config.leading_block_dim('B');
        let a_rows = if config.transpose[0] {
            config.block_k
        } else {
            config.block_m
        };
        let b_rows = if config.transpose[1] {
            config.block_k
        } else {
            config.block_n
        };

        msl.set("THREADGROUP_SIZE", config.threadgroup_size().to_string());
        msl.set("A_TG_TOTAL", (a_rows as u32 * a_lead as u32).to_string());
        msl.set("A_TG_COLS", a_lead.to_string());
        msl.set("B_TG_TOTAL", (b_rows as u32 * b_lead as u32).to_string());
        msl.set("B_TG_COLS", b_lead.to_string());

        msl.block(
            r#"
{
    ushort tid = sidx * 32 + lane_id;
    uint A_lead_dim = A_trans ? M : K;
    uint B_lead_dim = B_trans ? K : N;
    ushort M_tile = min(uint(M_group), M - M_offset);
    ushort N_tile = min(uint(N_group), N - N_offset);
    ushort K_tile = min(uint(K_group), K - k);

    for (ushort i = tid; i < {{A_TG_TOTAL}}; i += {{THREADGROUP_SIZE}}) {
        ushort row = i / {{A_TG_COLS}};
        ushort col = i % {{A_TG_COLS}};
        bool valid = A_trans ? (row < K_tile && col < M_tile) : (row < M_tile && col < K_tile);
        if (valid) {
            uint idx = A_trans
                ? (M_offset + col) * A_lead_dim + (k + row)
                : (M_offset + row) * A_lead_dim + (k + col);
            A_block[i] = A[idx];
        } else {
            A_block[i] = 0;
        }
    }

    for (ushort i = tid; i < {{B_TG_TOTAL}}; i += {{THREADGROUP_SIZE}}) {
        ushort row = i / {{B_TG_COLS}};
        ushort col = i % {{B_TG_COLS}};
        bool valid = B_trans ? (row < K_tile && col < N_tile) : (row < N_tile && col < K_tile);
        if (valid) {
            uint idx = B_trans
                ? (N_offset + col) * B_lead_dim + (k + row)
                : (N_offset + row) * B_lead_dim + (k + col);
            B_block[i] = B[idx];
        } else {
            B_block[i] = 0;
        }
    }
}
"#,
        );
    }

    fn emit_tile_sync(&self, msl: &mut MslBuilder, _config: &MetalGemmConfig) {
        msl.raw("threadgroup_barrier(mem_flags::mem_threadgroup);");
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform_is_identity() {
        assert!(IdentityTransform.is_identity());
    }

    #[test]
    fn test_identity_epilogue_is_identity() {
        assert!(IdentityEpilogue.is_identity());
    }

    #[test]
    fn test_native_load_a_emits_simdgroup_load() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        NativeFragmentLoad.emit_load_a(&mut msl, &config);
        let s = msl.finish();
        assert!(
            s.contains("simdgroup_load(A_mat"),
            "Must use native simdgroup_load"
        );
        assert!(s.contains("A_block"), "Must load from A_block");
        assert!(s.contains("tm * 8"), "Must use tm tile index");
        assert!(s.contains("kt * 8"), "Must use kt K-tile index");
    }

    #[test]
    fn test_native_load_b_emits_simdgroup_load() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        NativeFragmentLoad.emit_load_b(&mut msl, &config);
        let s = msl.finish();
        assert!(
            s.contains("simdgroup_load(B_mat"),
            "Must use native simdgroup_load"
        );
        assert!(s.contains("B_block"), "Must load from B_block");
        assert!(s.contains("tn * 8"), "Must use tn tile index");
    }

    #[test]
    fn test_native_mma_emits_multiply_accumulate() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        NativeMma.emit_multiply(&mut msl, &config);
        let s = msl.finish();
        assert!(
            s.contains("simdgroup_multiply_accumulate"),
            "Must use native MMA"
        );
        assert!(s.contains("C_sram[tm][tn]"), "Must index accumulators");
        assert!(s.contains("A_mat"), "Must use A_mat");
        assert!(s.contains("B_mat"), "Must use B_mat");
    }

    #[test]
    fn test_loop_tile_copy_emits_cooperative_loop() {
        let config = MetalGemmConfig::default_apple8_f16();
        let mut msl = MslBuilder::new();
        LoopTileCopy.emit_tile_load(&mut msl, &config);
        let s = msl.finish();
        assert!(
            s.contains("for (ushort i = tid"),
            "Must have cooperative loop"
        );
        assert!(s.contains("A_block[i]"), "Must write to A_block");
        assert!(s.contains("B_block[i]"), "Must write to B_block");
        assert!(s.contains("= 0"), "Must zero-fill padding");
    }

    #[test]
    fn test_loop_tile_copy_sync_emits_barrier() {
        let config = MetalGemmConfig::default_apple8_f16();
        let mut msl = MslBuilder::new();
        LoopTileCopy.emit_tile_sync(&mut msl, &config);
        let s = msl.finish();
        assert!(s.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
    }

    #[test]
    fn test_atoms_compose_in_correct_order() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();

        NativeFragmentLoad.emit_load_a(&mut msl, &config);
        IdentityTransform.emit_transform(&mut msl, &config);
        NativeFragmentLoad.emit_load_b(&mut msl, &config);
        NativeMma.emit_multiply(&mut msl, &config);

        let s = msl.finish();
        let a_pos = s.find("simdgroup_load(A_mat").unwrap();
        let b_pos = s.find("simdgroup_load(B_mat").unwrap();
        let c_pos = s.find("simdgroup_multiply_accumulate").unwrap();
        assert!(a_pos < b_pos, "A load before B load");
        assert!(b_pos < c_pos, "B load before multiply");
    }
}
