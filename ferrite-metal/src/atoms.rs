/// Metal atom traits — the pluggable components of the kernel pipeline.
///
/// Decomposition of MFA's multiply_accumulate into composable phases:
///
/// ```text
/// Per K-step (8 elements):
///   1. FragmentLoadAtom::load_a()      — smem/device → A_sram registers
///   2. TransformAtom::transform(A_sram) — optional in-place transform (FUSION SLOT)
///   3. FragmentLoadAtom::load_b()      — smem/device → B_sram registers
///   4. MmaAtom::multiply()             — simdgroup_multiply_accumulate
/// ```
///
/// The transform between load_a and multiply is WHERE FUSION HAPPENS.
/// For standalone GEMM: IdentityTransform (no-op).
/// For fused norm→GEMM: RmsNormTransform (multiply A by norm_factor × gamma).
///
/// Used at COMPILE TIME by the proc macro. Emits MSL code strings.
use crate::config::MetalGemmConfig;
use crate::msl_builder::MslBuilder;

// ═══════════════════════════════════════════════════════════════════
// Tile Copy: device memory → threadgroup memory (the outer K-loop load)
// ═══════════════════════════════════════════════════════════════════

/// How tiles move from device to threadgroup memory.
///
/// This is the OUTER copy — bulk tile loading at the start of each K-iteration.
/// apple9: simdgroup_event async_copy (hardware DMA)
/// apple8: loop-based copy (polyfill)
pub trait TileCopyAtom {
    /// Emit code to load one A tile and one B tile from device to threadgroup memory.
    fn emit_tile_load(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);

    /// Emit synchronization after tile loads.
    fn emit_tile_sync(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

// ═══════════════════════════════════════════════════════════════════
// Fragment Load: threadgroup/device → simdgroup registers
// ═══════════════════════════════════════════════════════════════════

/// How fragments are loaded into simdgroup registers for MMA.
///
/// This is the INNER load — per-simdgroup fragment loading within a K-step.
/// Separate from TileCopyAtom because:
/// - TileCopy is per-threadgroup (all simdgroups cooperate)
/// - FragmentLoad is per-simdgroup (each loads its own tile portion)
///
/// A and B have separate load methods because:
/// - A load feeds into TransformAtom (fusion slot)
/// - B load feeds directly into MmaAtom
pub trait FragmentLoadAtom {
    /// Emit code to load A fragments into A_sram registers.
    /// After this, TransformAtom::transform() may modify A_sram in-place.
    fn emit_load_a(
        &self,
        msl: &mut MslBuilder,
        config: &MetalGemmConfig,
        k_var: &str,       // K-step variable name (e.g., "k")
        src_var: &str,     // Source pointer variable name
        leading_dim: &str, // Leading dimension expression
        trans: bool,       // Whether A is transposed
    );

    /// Emit code to load B fragments into B_sram registers.
    fn emit_load_b(
        &self,
        msl: &mut MslBuilder,
        config: &MetalGemmConfig,
        k_var: &str,
        src_var: &str,
        leading_dim: &str,
        trans: bool,
    );
}

// ═══════════════════════════════════════════════════════════════════
// Transform: in-place modification of A fragments before MMA
// ═══════════════════════════════════════════════════════════════════

/// How A fragments are transformed between load and MMA.
///
/// **This is WHERE FUSION HAPPENS for pre-GEMM operations.**
///
/// The transform operates on `simdgroup_matrix_storage<T>` elements
/// that are already in registers after FragmentLoadAtom::load_a().
///
/// Transforms must be:
/// - In-place (modify A_sram, don't create new arrays)
/// - Per-element (operate on the 8×8 matrix values)
/// - Independent across M tiles (each rm is separate)
///
/// Examples:
/// - Identity: no-op (standalone GEMM)
/// - RmsNorm: multiply each element by norm_factor × gamma[k_offset + lane]
/// - Dequantize: scale packed int4/int8 to f16
pub trait TransformAtom {
    /// One-time setup before the K-loop.
    /// Load norm factors, gamma weights, etc. into registers or threadgroup memory.
    fn emit_prologue(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}

    /// Per-K-iteration setup.
    /// Load the gamma slice for this K chunk from threadgroup memory.
    fn emit_k_setup(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig, _k_var: &str) {}

    /// Transform A_sram fragments in-place.
    /// Called once per K-step, AFTER load_a and BEFORE multiply.
    /// `register_m` iterations of 8×8 fragments are in A_sram.
    fn emit_transform(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}

    /// Whether this transform is a no-op.
    /// The pipeline can skip emitting the transform call if true.
    fn is_identity(&self) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════
// MMA: simdgroup matrix multiply-accumulate
// ═══════════════════════════════════════════════════════════════════

/// How tensor cores consume loaded fragments.
///
/// Reads from A_sram and B_sram, accumulates into C_sram.
/// This is the pure compute step — no memory access.
pub trait MmaAtom {
    /// Emit the multiply loop: for each (m, n) tile pair, C += A × B.
    fn emit_multiply(&self, msl: &mut MslBuilder, config: &MetalGemmConfig);
}

// ═══════════════════════════════════════════════════════════════════
// Epilogue: post-process accumulators before store
// ═══════════════════════════════════════════════════════════════════

/// How accumulators are post-processed before store.
///
/// **This is WHERE FUSION HAPPENS for post-GEMM operations.**
///
/// Operates on C_sram accumulators that contain the full matmul result.
/// Applied AFTER the K-loop completes, BEFORE store to device memory.
///
/// Examples:
/// - Identity: store raw accumulators
/// - SiLu: x * sigmoid(x) on each accumulator element
/// - ResidualAdd: load residual from device, add to accumulators
pub trait EpilogueAtom {
    /// Transform C_sram accumulators in-place before store.
    fn emit_epilogue(&self, _msl: &mut MslBuilder, _config: &MetalGemmConfig) {}

    /// Whether this epilogue is a no-op (just store).
    fn is_identity(&self) -> bool {
        false
    }
}

// ═══════════════════════════════════════════════════════════════════
// Default implementations
// ═══════════════════════════════════════════════════════════════════

/// Identity transform — no modification to A fragments.
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

/// Standard simdgroup fragment load from threadgroup memory.
///
/// Emits simdgroup_matrix_storage::load() calls for A and B.
/// This is MFA's default path when data is in threadgroup memory
/// (after TileCopyAtom has moved it from device).
pub struct ThreadgroupFragmentLoad;

impl FragmentLoadAtom for ThreadgroupFragmentLoad {
    fn emit_load_a(
        &self,
        msl: &mut MslBuilder,
        config: &MetalGemmConfig,
        k_var: &str,
        src_var: &str,
        leading_dim: &str,
        trans: bool,
    ) {
        let reg_m = config.register_m();
        let trans_str = if trans { "true" } else { "false" };

        msl.set("REGISTER_M", reg_m.to_string());
        msl.set("K_VAR", k_var);
        msl.set("A_SRC", src_var);
        msl.set("A_LEADING_DIM", leading_dim);
        msl.set("A_TRANS", trans_str);

        msl.block(
            r#"
#pragma clang loop unroll(full)
for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
    ushort2 origin(0, m);
    auto A = get_sram(A_sram, 8, origin);
    A->load({{A_SRC}}, {{A_LEADING_DIM}}, ushort2({{K_VAR}}, m), {{A_TRANS}});
}
"#,
        );
    }

    fn emit_load_b(
        &self,
        msl: &mut MslBuilder,
        config: &MetalGemmConfig,
        k_var: &str,
        src_var: &str,
        leading_dim: &str,
        trans: bool,
    ) {
        let reg_n = config.register_n();
        let trans_str = if trans { "true" } else { "false" };

        msl.set("REGISTER_N", reg_n.to_string());
        msl.set("K_VAR", k_var);
        msl.set("B_SRC", src_var);
        msl.set("B_LEADING_DIM", leading_dim);
        msl.set("B_TRANS", trans_str);

        msl.block(
            r#"
#pragma clang loop unroll(full)
for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
    ushort2 origin(n, 0);
    auto B = get_sram(B_sram, {{REGISTER_N}}, origin);
    B->load({{B_SRC}}, {{B_LEADING_DIM}}, ushort2(n, {{K_VAR}}), {{B_TRANS}});
}
"#,
        );
    }
}

/// Standard simdgroup multiply: C += A × B for all (m, n) tile pairs.
pub struct SimdgroupMma;

impl MmaAtom for SimdgroupMma {
    fn emit_multiply(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        let reg_m = config.register_m();
        let reg_n = config.register_n();

        msl.set("REGISTER_M", reg_m.to_string());
        msl.set("REGISTER_N", reg_n.to_string());

        msl.block(
            r#"
#pragma clang loop unroll(full)
for (ushort m = 0; m < {{REGISTER_M}}; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < {{REGISTER_N}}; n += 8) {
        auto A = get_sram(A_sram, 8, ushort2(0, m));
        auto B = get_sram(B_sram, {{REGISTER_N}}, ushort2(n, 0));
        auto C = get_sram(C_sram, {{REGISTER_N}}, ushort2(n, m));
        C->multiply(*A, *B);
    }
}
"#,
        );
    }
}

/// Async tile copy (apple9 / M3+).
///
/// Uses simdgroup_event for hardware-accelerated device→threadgroup DMA.
/// Only simdgroup 0 performs the copy; others skip and wait at the barrier.
pub struct AsyncTileCopy;

impl TileCopyAtom for AsyncTileCopy {
    fn emit_tile_load(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("BLOCK_BYTES_A", config.block_bytes('A').to_string());
        msl.set(
            "LEADING_BLOCK_DIM_A",
            config.leading_block_dim('A').to_string(),
        );
        msl.set(
            "LEADING_BLOCK_DIM_B",
            config.leading_block_dim('B').to_string(),
        );
        msl.set("MEMORY_NAME_A", config.memory_precisions.a.msl_name());
        msl.set("MEMORY_NAME_B", config.memory_precisions.b.msl_name());
        let a_trans = if config.transpose[0] {
            "A_trans"
        } else {
            "false"
        };
        let b_trans = if config.transpose[1] {
            "B_trans"
        } else {
            "false"
        };
        msl.set("A_TRANS_EXPR", a_trans);
        msl.set("B_TRANS_EXPR", b_trans);

        msl.block(
            r#"
uint A_leading_dimension = A_trans ? M : K;
uint B_leading_dimension = B_trans ? K : N;

if (sidx == 0) {
    uint2 A_offset(k, M_offset);
    uint2 B_offset(N_offset, k);
    auto A_src = simdgroup_matrix_storage<{{MEMORY_NAME_A}}>::apply_offset(
        A, A_leading_dimension, A_offset, {{A_TRANS_EXPR}});
    auto B_src = simdgroup_matrix_storage<{{MEMORY_NAME_B}}>::apply_offset(
        B, B_leading_dimension, B_offset, {{B_TRANS_EXPR}});

    ushort M_tile_dimension = min(uint(M_group), M - M_offset);
    ushort N_tile_dimension = min(uint(N_group), N - N_offset);
    ushort K_tile_dimension = min(uint(K_group), K - k);

    ushort2 A_tile_src(K_tile_dimension, M_tile_dimension);
    ushort2 B_tile_src(N_tile_dimension, K_tile_dimension);

    simdgroup_event events[2];
    events[0].async_copy<{{LEADING_BLOCK_DIM_A}}, 32>(
        A_block, A_tile_src, A_src, A_leading_dimension, A_tile_src, {{A_TRANS_EXPR}});
    events[1].async_copy<{{LEADING_BLOCK_DIM_B}}, 32>(
        B_block, B_tile_src, B_src, B_leading_dimension, B_tile_src, {{B_TRANS_EXPR}});
    simdgroup_event::wait(2, events);
}
"#,
        );
    }

    fn emit_tile_sync(&self, msl: &mut MslBuilder, _config: &MetalGemmConfig) {
        msl.raw("threadgroup_barrier(mem_flags::mem_threadgroup);");
    }
}

/// Loop-based tile copy (universal fallback).
///
/// All threads in the threadgroup cooperate to copy tiles from device
/// to threadgroup memory using a simple strided loop. No simdgroup_event,
/// no hardware DMA — works on all GPU generations and Metal versions.
///
/// This replaces DirectTileAccess (which was incorrect — it emitted no copy
/// but the K-loop still loaded from threadgroup memory).
pub struct LoopTileCopy;

impl TileCopyAtom for LoopTileCopy {
    fn emit_tile_load(&self, msl: &mut MslBuilder, config: &MetalGemmConfig) {
        msl.set("BLOCK_BYTES_A", config.block_bytes('A').to_string());
        msl.set(
            "LEADING_BLOCK_DIM_A",
            config.leading_block_dim('A').to_string(),
        );
        msl.set(
            "LEADING_BLOCK_DIM_B",
            config.leading_block_dim('B').to_string(),
        );
        msl.set("MEMORY_NAME_A", config.memory_precisions.a.msl_name());
        msl.set("MEMORY_NAME_B", config.memory_precisions.b.msl_name());
        msl.set("THREADGROUP_SIZE", config.threadgroup_size().to_string());

        // For A tile: the threadgroup tile has dimensions
        //   rows = M_group (or block_m), cols = LEADING_BLOCK_DIM_A
        // But only K_tile valid columns per row. Must zero-fill padding
        // because fragment loads via morton pattern read into padding columns.
        // (MFA's async_copy uses clamp_to_zero mode for this.)
        let a_rows = if config.transpose[0] {
            config.block_k
        } else {
            config.block_m
        };
        let a_cols = config.leading_block_dim('A');
        let b_rows = if config.transpose[1] {
            config.block_k
        } else {
            config.block_n
        };
        let b_cols = config.leading_block_dim('B');

        msl.set("A_TG_ROWS", a_rows.to_string());
        msl.set("A_TG_COLS", a_cols.to_string());
        msl.set("B_TG_ROWS", b_rows.to_string());
        msl.set("B_TG_COLS", b_cols.to_string());

        msl.block(
            r#"
{
    uint A_leading_dimension = A_trans ? M : K;
    uint B_leading_dimension = B_trans ? K : N;
    ushort tid = sidx * 32 + lane_id;

    // Tile dimensions (clamped at matrix edges)
    ushort M_tile = min(uint(M_group), M - M_offset);
    ushort N_tile = min(uint(N_group), N - N_offset);
    ushort K_tile = min(uint(K_group), K - k);

    // Copy A tile with zero-fill padding.
    // Full threadgroup tile = A_TG_ROWS × A_TG_COLS, valid region is smaller.
    ushort a_total = {{A_TG_ROWS}} * {{A_TG_COLS}};
    for (ushort i = tid; i < a_total; i += {{THREADGROUP_SIZE}}) {
        ushort row = i / {{A_TG_COLS}};
        ushort col = i % {{A_TG_COLS}};
        bool a_valid = A_trans
            ? (row < K_tile && col < M_tile)
            : (row < M_tile && col < K_tile);
        if (a_valid) {
            uint dev_idx = A_trans
                ? (k + row) * A_leading_dimension + (M_offset + col)
                : (M_offset + row) * A_leading_dimension + (k + col);
            A_block[i] = A[dev_idx];
        } else {
            A_block[i] = 0;
        }
    }

    // Copy B tile with zero-fill padding.
    ushort b_total = {{B_TG_ROWS}} * {{B_TG_COLS}};
    for (ushort i = tid; i < b_total; i += {{THREADGROUP_SIZE}}) {
        ushort row = i / {{B_TG_COLS}};
        ushort col = i % {{B_TG_COLS}};
        bool b_valid = B_trans
            ? (row < K_tile && col < N_tile)
            : (row < N_tile && col < K_tile);
        if (b_valid) {
            uint dev_idx = B_trans
                ? (k + row) * B_leading_dimension + (N_offset + col)
                : (N_offset + row) * B_leading_dimension + (k + col);
            B_block[i] = B[dev_idx];
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
    fn test_fragment_load_a_emits_unrolled_loop() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        ThreadgroupFragmentLoad.emit_load_a(&mut msl, &config, "k", "A_block_src", "32", false);
        let s = msl.finish();
        assert!(
            s.contains("#pragma clang loop unroll(full)"),
            "Must unroll A load loop"
        );
        assert!(s.contains("A->load(A_block_src"), "Must load from A source");
        assert!(s.contains("get_sram(A_sram"), "Must use get_sram helper");
    }

    #[test]
    fn test_fragment_load_b_emits_unrolled_loop() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        ThreadgroupFragmentLoad.emit_load_b(&mut msl, &config, "k", "B_block_src", "32", true);
        let s = msl.finish();
        assert!(
            s.contains("#pragma clang loop unroll(full)"),
            "Must unroll B load loop"
        );
        assert!(s.contains("B->load(B_block_src"), "Must load from B source");
        assert!(s.contains("true"), "Must pass transpose flag");
    }

    #[test]
    fn test_mma_emits_nested_multiply_loop() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        SimdgroupMma.emit_multiply(&mut msl, &config);
        let s = msl.finish();
        assert!(s.contains("C->multiply(*A, *B)"), "Must have multiply call");
        assert!(s.contains("get_sram(C_sram"), "Must access C accumulators");
        // Nested unrolled loops
        let unroll_count = s.matches("#pragma clang loop unroll(full)").count();
        assert_eq!(
            unroll_count, 2,
            "Must have 2 unroll pragmas (m and n loops)"
        );
    }

    #[test]
    fn test_async_tile_copy_emits_events() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        AsyncTileCopy.emit_tile_load(&mut msl, &config);
        let s = msl.finish();
        assert!(
            s.contains("simdgroup_event events[2]"),
            "Must declare 2 events"
        );
        assert!(s.contains("async_copy<"), "Must call async_copy");
        assert!(
            s.contains("simdgroup_event::wait(2, events)"),
            "Must wait for both events"
        );
        assert!(s.contains("if (sidx == 0)"), "Only simdgroup 0 copies");
    }

    #[test]
    fn test_async_tile_copy_sync_emits_barrier() {
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        AsyncTileCopy.emit_tile_sync(&mut msl, &config);
        let s = msl.finish();
        assert!(s.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
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
        assert!(s.contains("A_block["), "Must write to A_block");
        assert!(s.contains("B_block["), "Must write to B_block");
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
    fn test_fragment_load_respects_register_dimensions() {
        // apple9: register_m = 32, register_n = 32
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();
        ThreadgroupFragmentLoad.emit_load_a(&mut msl, &config, "k", "src", "32", false);
        let s = msl.finish();
        assert!(s.contains("m < 32"), "Should iterate to register_m=32");

        // With splits, register dimensions change
        let mut split_config = config.clone();
        split_config.splits = [2, 2]; // register_m = 16, register_n = 16
        let mut msl2 = MslBuilder::new();
        ThreadgroupFragmentLoad.emit_load_a(&mut msl2, &split_config, "k", "src", "16", false);
        let s2 = msl2.finish();
        assert!(
            s2.contains("m < 16"),
            "Should iterate to register_m=16 with splits"
        );
    }

    #[test]
    fn test_atoms_compose_in_correct_order() {
        // Verify that emitting load_a → transform → load_b → multiply
        // produces code in the right order.
        let config = MetalGemmConfig::default_apple9_f16();
        let mut msl = MslBuilder::new();

        ThreadgroupFragmentLoad.emit_load_a(&mut msl, &config, "0", "A_src", "32", false);
        // Transform slot (identity = nothing)
        IdentityTransform.emit_transform(&mut msl, &config);
        ThreadgroupFragmentLoad.emit_load_b(&mut msl, &config, "0", "B_src", "32", true);
        SimdgroupMma.emit_multiply(&mut msl, &config);

        let s = msl.finish();
        let a_pos = s.find("A->load").unwrap();
        let b_pos = s.find("B->load").unwrap();
        let c_pos = s.find("C->multiply").unwrap();
        assert!(a_pos < b_pos, "A load must come before B load");
        assert!(b_pos < c_pos, "B load must come before multiply");
    }
}
