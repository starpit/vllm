// SPDX-License-Identifier: Apache-2.0
//! `cuda_emit` — pure literal transcription of a typed [`MegaTape`]
//! into a complete `.cu` translation unit, calling **TK 2.0
//! primitives only** (`third_party/thunderkittens/include/`).
//!
//! See `MEGA_IR_PLAN.md` §0 / §8.0 / §8.0a + `CUDA_EMIT_TK20_AUDIT.md`
//! at the worktree root for the contract. The DOD for any variant
//! shipping is **the emitted `.cu` compiles against TK 2.0 on the
//! pod** — Rust unit tests are not the ground truth.
//!
//! Module layout:
//!
//! - [`cu`] — emitted-CUDA AST: [`CuExpr`] / [`CuStmt`] /
//!   [`CuBlock`] / [`CuVariant`].
//! - [`handles`] — typed handles ([`Sv`], [`Rv`], [`Semaphore`],
//!   [`GmemPtrRaw`], [`ScratchPtr`]) and ferrite substrate
//!   accessors (page_ready_sem, page_as_sv_bf, scratch_as,
//!   gmem_*_raw).
//! - [`tk20`] — Rust API surface mirroring TK 2.0's C++ primitive
//!   surface. Each fn cites its TK 2.0 source line.
//! - [`roles`] — per-`MegaNode` role-body emit.
//!
//! [`Sv`]: handles::Sv
//! [`Rv`]: handles::Rv
//! [`Semaphore`]: handles::Semaphore
//! [`GmemPtrRaw`]: handles::GmemPtrRaw
//! [`ScratchPtr`]: handles::ScratchPtr

pub mod cu;
pub mod handles;
pub mod roles;
pub mod tk20;

use crate::ir::nodes::MegaNode;
use crate::ir::tape::{MegaTape, TapeBudget};

pub use cu::{CuBlock, CuExpr, CuStmt, CuVariant};

/// Which host-side launch ABI tier the emitted megakernel exposes.
///
/// Mirror of `ferrite_forward::interpreter::mega::LaunchTier`. The
/// emit step picks the tier from the tape (which variants it
/// contains) and emits a kernel signature whose positional args
/// match the host-side `LaunchArgs*` struct of the same tier per
/// `MEGA_IR_PLAN.md` §8.0a item 2 ("the kernel signature you emit
/// must match what the host already constructs and passes").
///
/// - `Base` — neither paged-KV nor rotary metadata.
/// - `Qkv` — at least one `FusedQkvRopeCache`.
/// - `Attn` — additionally at least one `AttentionViaCache`.
///
/// Nested: Attn ⊃ Qkv ⊃ Base. Sets the kernel arg list. The host
/// pairs each variant with its matching `LaunchFn*` based on this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchTier {
    Base,
    Qkv,
    Attn,
}

fn launch_tier_for_tape(tape: &MegaTape) -> LaunchTier {
    let mut needs_attn = false;
    let mut needs_qkv = false;
    for node in tape.nodes() {
        match node {
            MegaNode::AttentionViaCache(_) => needs_attn = true,
            MegaNode::FusedQkvRopeCache(_) => needs_qkv = true,
            _ => {}
        }
    }
    if needs_attn {
        LaunchTier::Attn
    } else if needs_qkv {
        LaunchTier::Qkv
    } else {
        LaunchTier::Base
    }
}

/// Lower a typed [`MegaTape`] to a complete `.cu` translation unit
/// for the named canonical, against TK 2.0 primitives only.
pub fn lower_to_cuda(canonical: &str, tape: &MegaTape) -> CuVariant {
    let budget = tape.budget();
    let tier = launch_tier_for_tape(tape);

    let mut loader_block = CuBlock::new();
    let mut launcher_block = CuBlock::new();
    let mut consumer_block = CuBlock::new();
    let mut storer_block = CuBlock::new();
    let mut skipped: Vec<String> = Vec::new();

    for (idx, node) in tape.nodes().iter().enumerate() {
        let bodies = roles::emit_role_bodies(node, budget);
        if let Some(variant) = bodies.skipped {
            skipped.push(variant.to_string());
            let marker = CuStmt::new(format!("// SKIPPED node[{idx}]: {variant}"));
            loader_block.push(marker.clone());
            launcher_block.push(marker.clone());
            consumer_block.push(marker.clone());
            storer_block.push(marker);
            continue;
        }
        loader_block.extend(bodies.loader);
        launcher_block.extend(bodies.launcher);
        consumer_block.extend(bodies.consumer);
        storer_block.extend(bodies.storer);
    }

    let source = render_source(
        canonical,
        &budget,
        tier,
        &loader_block,
        &launcher_block,
        &consumer_block,
        &storer_block,
    );

    CuVariant {
        canonical: canonical.to_string(),
        source,
        skipped_variants: skipped,
    }
}

/// Render the full `.cu` text. Substrate scaffolding (includes,
/// per-canonical Config struct, role-body fns, kernel entry) wraps
/// the four role blocks emitted from per-MegaNode bodies.
fn render_source(
    canonical: &str,
    budget: &TapeBudget,
    tier: LaunchTier,
    loader: &CuBlock,
    launcher: &CuBlock,
    consumer: &CuBlock,
    storer: &CuBlock,
) -> String {
    // Register budgets — substrate-implementation values matching
    // the ferrite-side `set_consumer_registers<Config>` /
    // `set_non_consumer_registers<Config>` wrappers in
    // `ferrite_warp_roles.cuh`. Promoting these to per-canonical
    // values is a future sprint.
    const CONSUMER_REGISTERS: u32 = 240;
    const NON_CONSUMER_REGISTERS: u32 = 24;
    const INSTRUCTION_PIPE_STAGES: u32 = 4;

    let kernel_name = format!("ferrite_{canonical}_launch");
    let cfg_name = format!("Config_{canonical}");

    let mut s = String::new();
    s.push_str("// SPDX-License-Identifier: Apache-2.0\n");
    s.push_str(&format!(
        "// Auto-generated by ferrite_megakernel::cuda_emit for canonical \"{canonical}\".\n"
    ));
    s.push_str("// DO NOT EDIT — regenerate by re-running the proc-macro.\n\n");

    // Includes: ferrite substrate + ferrite barrier (cross-CTA gmem
    // counters). The substrate `.cuh`s pull in `kittens.cuh`
    // transitively (see ferrite_substrate.cuh:31).
    s.push_str("#include \"ferrite_substrate.cuh\"\n");
    s.push_str("#include \"ferrite_globals.cuh\"\n");
    s.push_str("#include \"ferrite_warp_roles.cuh\"\n");
    s.push_str("#include \"ferrite_barrier.cuh\"\n\n");

    s.push_str("namespace {\n\n");

    // Per-canonical Config struct.
    s.push_str(&format!("struct {cfg_name} {{\n"));
    s.push_str(&format!(
        "    static constexpr int NUM_CONSUMER_WARPS = {};\n",
        budget.num_consumer_warps
    ));
    s.push_str(&format!(
        "    static constexpr int NON_CONSUMER_REGISTERS = {NON_CONSUMER_REGISTERS};\n"
    ));
    s.push_str(&format!(
        "    static constexpr int CONSUMER_REGISTERS = {CONSUMER_REGISTERS};\n"
    ));
    s.push_str(&format!(
        "    static constexpr int NUM_PAGES = {};\n",
        budget.num_pages
    ));
    s.push_str(&format!(
        "    static constexpr int PAGE_SIZE = {};\n",
        budget.page_size
    ));
    s.push_str(&format!(
        "    static constexpr int SCRATCH_BYTES = {};\n",
        budget.scratch_bytes
    ));
    s.push_str(&format!(
        "    static constexpr int INSTRUCTION_PIPE_STAGES = {INSTRUCTION_PIPE_STAGES};\n"
    ));
    s.push_str("};\n");
    s.push_str(&format!("using ConfigT = {cfg_name};\n\n"));

    // Kernel entry. Args are referenced directly inside the role
    // blocks below (no aggregation struct). Signature matches the
    // host-side `LaunchFn*` ABI for `tier` in
    // `crates/ferrite-forward/src/interpreter/mega/mod.rs` per
    // `MEGA_IR_PLAN.md` §8.0a item 2.
    //
    // Nested ABI: Attn ⊃ Qkv ⊃ Base. Higher tiers append args to the
    // lower tier's prefix in declaration order. Variants whose tape
    // doesn't reach a tier still receive the args at higher tiers
    // (host passes null where unused) so the positional ABI stays
    // linear.
    s.push_str(&format!("extern \"C\" __global__ void {kernel_name}(\n"));
    s.push_str("    __nv_bfloat16* const*       act_ptrs,\n");
    s.push_str("    const __nv_bfloat16* const* weight_ptrs,\n");
    match tier {
        LaunchTier::Base => {
            s.push_str("    int32_t*                    barrier_slots,\n");
            s.push_str("    int32_t                     trace_level,\n");
            s.push_str("    const uint32_t*             input_ids\n");
        }
        LaunchTier::Qkv => {
            s.push_str("    const uint32_t*             input_ids,\n");
            s.push_str("    const uint32_t*             positions,\n");
            s.push_str("    const int64_t*              slot_mapping,\n");
            s.push_str("    __nv_bfloat16* const*       key_cache_ptrs,\n");
            s.push_str("    __nv_bfloat16* const*       value_cache_ptrs,\n");
            s.push_str("    int32_t*                    barrier_slots,\n");
            s.push_str("    int32_t                     trace_level\n");
        }
        LaunchTier::Attn => {
            s.push_str("    const uint32_t*             input_ids,\n");
            s.push_str("    const uint32_t*             positions,\n");
            s.push_str("    const int64_t*              slot_mapping,\n");
            s.push_str("    __nv_bfloat16* const*       key_cache_ptrs,\n");
            s.push_str("    __nv_bfloat16* const*       value_cache_ptrs,\n");
            s.push_str("    const int32_t*             seq_lens,\n");
            s.push_str("    const uint32_t*             block_table,\n");
            s.push_str("    uint32_t                    block_table_stride,\n");
            s.push_str("    int32_t*                    barrier_slots,\n");
            s.push_str("    int32_t                     trace_level\n");
        }
    }
    s.push_str(") {\n");
    s.push_str("    (void)trace_level;\n");
    if matches!(tier, LaunchTier::Qkv | LaunchTier::Attn) {
        // Suppress unused-arg warnings in the .cu when the tape's
        // FQRC ops don't use slot_mapping / KV cache pointers (cache
        // writes happen outside the megakernel as a follow-up D2D —
        // see `feedback_ff_mega_cuda_emit_s15a_handoff` design notes).
        s.push_str("    (void)slot_mapping;\n");
        s.push_str("    (void)key_cache_ptrs;\n");
        s.push_str("    (void)value_cache_ptrs;\n");
    }
    if matches!(tier, LaunchTier::Attn) {
        s.push_str("    (void)seq_lens;\n");
        s.push_str("    (void)block_table;\n");
        s.push_str("    (void)block_table_stride;\n");
    }
    s.push_str("    extern __shared__ uint8_t shmem_buf[];\n");
    s.push_str("    auto& ss = *reinterpret_cast<ferrite::SharedState<ConfigT>*>(shmem_buf);\n");
    s.push_str("    ferrite::init_shared_state<ConfigT>(ss);\n");
    s.push_str("    int wid = kittens::warpid();\n");
    s.push_str("    if (wid < ConfigT::NUM_CONSUMER_WARPS) {\n");
    s.push_str("        ferrite::set_consumer_registers<ConfigT>();\n");
    s.push_str("        // consumer\n");
    s.push_str(&consumer.render(8));
    s.push_str("    } else {\n");
    s.push_str("        ferrite::set_non_consumer_registers<ConfigT>();\n");
    s.push_str("        switch (wid - ConfigT::NUM_CONSUMER_WARPS) {\n");
    s.push_str("            case ferrite::kLoaderSlot: {\n");
    s.push_str(&loader.render(16));
    s.push_str("                break;\n");
    s.push_str("            }\n");
    s.push_str("            case ferrite::kLauncherSlot: {\n");
    s.push_str(&launcher.render(16));
    s.push_str("                break;\n");
    s.push_str("            }\n");
    s.push_str("            case ferrite::kStorerSlot: {\n");
    s.push_str(&storer.render(16));
    s.push_str("                break;\n");
    s.push_str("            }\n");
    s.push_str("        }\n");
    s.push_str("    }\n");
    s.push_str("}\n\n");

    s.push_str("} // namespace\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::lower::MegaTapeBuilder;
    use crate::ir::nodes::LayerIndex;
    use crate::ir::substrate::{
        ActSlotConst, ArrivesCount, BarSyncId, BarSyncPair, HiddenDim, MbarrierPhase,
        NumTokensConst, PageId, RmsNormScope, ScratchRegion, WeightAccessorConst,
    };

    type BuilderD = MegaTapeBuilder<8, 8, 32_768, 32_768, 4>;

    /// Sprint 10: Gemm — `D = A * B + C` (C zero) under AlongN
    /// warp split. ITERS=1 path: full b_tile staged in scratch.
    /// Smoke config: M=16, K=32, N=256, NCW=8. TILE_N=32,
    /// CHUNK_K=K=32. b_tile = 32 * 256 * 2 = 16384 bytes (fits
    /// in 32 KB scratch). All shapes multiples of 16 — required
    /// by `kittens::warp::mma_AB`.
    #[test]
    fn gemm_emits_tk20_calls() {
        use crate::ir::substrate::{
            BarSyncId, ChunkK, GemmScope, IterCount, MatmulK, MatmulM, MatmulN,
            ScratchRegion, TileN,
        };
        let mut b = BuilderD::new();
        b.push_gemm(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(), // in
            PageId::<1, 8>::new(), // weight
            PageId::<2, 8>::new(), // out
            ScratchRegion::<0, 16_384, 32_768, GemmScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            IterCount::<1>::new(),
            LayerIndex::<3, 16>::new(),
            MatmulN::<256>::new(),
            MatmulK::<32>::new(),
            MatmulM::<16>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<2, { u32::MAX }>::new(),
            WeightAccessorConst::<5, { u32::MAX }>::new(),
            TileN::<32>::new(),
            ChunkK::<32>::new(),
            BarSyncId::<7>::new(),
            "W::gemm".to_string(),
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_gemm", &tape);
        assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
        std::fs::write("/tmp/gemm_emit.cu", &cu.source).ok();

        for needle in [
            // Loader: tile-flavored TMA load of activation + b_tile.
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 1024);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 16384);",
            "kittens::group<1>::tma::load_async(",
            "(*reinterpret_cast<kittens::st_bf<16, 32>*>(ss.pages[0]))",
            "(*reinterpret_cast<kittens::st_bf<32, 256>*>(ss.scratch + 0))",
            // Consumer: rt decls + warp::load + warp::zero + mma_AB
            //          + warp::store + cross-warp sync + warp 0 publish.
            "kittens::rt_bf<16, 32> __gemm_a;",
            "kittens::rt_bf<32, 32, kittens::ducks::rt_layout::col> __gemm_b;",
            "kittens::rt_fl<16, 32> __gemm_acc;",
            "auto __gemm_b_sub = ",
            "auto __gemm_out_sub = ",
            ".template subtile<32, 32>(int2{0, static_cast<int>(kittens::warpid())})",
            ".template subtile<16, 32>(int2{0, static_cast<int>(kittens::warpid())})",
            "kittens::warp::load(__gemm_a, ",
            "kittens::warp::load(__gemm_b, __gemm_b_sub);",
            "kittens::warp::zero(__gemm_acc);",
            "kittens::warp::mma_AB(__gemm_acc, __gemm_a, __gemm_b, __gemm_acc);",
            "kittens::warp::store(__gemm_out_sub, __gemm_acc);",
            "kittens::group<8>::sync(7);",
            "kittens::group<1>::arrive(ss.page_done[2]);",
            "kittens::group<1>::arrive(ss.page_consumed[0]);",
            "kittens::group<1>::arrive(ss.page_consumed[1]);",
            // Storer: full out_smem TMA back to gmem.
            "kittens::group<1>::tma::store_async(",
            "act_ptrs[2]",
            "kittens::group<1>::tma::store_async_wait();",
            "kittens::group<1>::arrive(ss.page_consumed[2]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?} in source, got:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 11: TkFusedGemmAdd — `residual += A * B`. Mirror of
    /// the S10 Gemm smoke with the output writing back IN PLACE
    /// to the residual page instead of a separate out page (no
    /// `out_*` slot/page; residual_page is BOTH read and written).
    /// Smoke config: M=16, K=32, N=256, NCW=8. TILE_N=32,
    /// CHUNK_K=K=32. b_tile = 32 * 256 * 2 = 16384 bytes.
    #[test]
    fn tk_fused_gemm_add_emits_tk20_calls() {
        use crate::ir::substrate::{
            BarSyncId, ChunkK, GemmScope, IterCount, KFull, KOffset, MatmulK, MatmulN,
            ScratchRegion, TileN,
        };
        let mut b = BuilderD::new();
        b.push_tk_fused_gemm_add(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(), // in
            PageId::<1, 8>::new(), // weight
            PageId::<2, 8>::new(), // residual (read+write)
            ScratchRegion::<0, 16_384, 32_768, GemmScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            IterCount::<1>::new(),
            LayerIndex::<3, 16>::new(),
            MatmulN::<256>::new(),
            MatmulK::<32>::new(),
            NumTokensConst::<16>::new(),
            KOffset::<0>::new(),
            KFull::<32>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),       // in_act_slot
            ActSlotConst::<2, { u32::MAX }>::new(),       // residual_act_slot
            WeightAccessorConst::<5, { u32::MAX }>::new(),
            TileN::<32>::new(),
            ChunkK::<32>::new(),
            BarSyncId::<7>::new(),
            "W::tk_fused_gemm_add".to_string(),
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_tk_fused_gemm_add", &tape);
        assert!(
            cu.skipped_variants.is_empty(),
            "skipped: {:?}",
            cu.skipped_variants
        );
        std::fs::write("/tmp/tk_fused_gemm_add_emit.cu", &cu.source).ok();

        for needle in [
            // Loader: three TMA loads (act + b_tile + residual).
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 1024);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 16384);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[2], 8192);",
            "(*reinterpret_cast<kittens::st_bf<16, 32>*>(ss.pages[0]))",
            "(*reinterpret_cast<kittens::st_bf<32, 256>*>(ss.scratch + 0))",
            "(*reinterpret_cast<kittens::st_bf<16, 256>*>(ss.pages[2]))",
            // Consumer: rt decls + per-warp B + RESIDUAL subtiles.
            "kittens::rt_bf<16, 32> __gemm_a;",
            "kittens::rt_bf<32, 32, kittens::ducks::rt_layout::col> __gemm_b;",
            "kittens::rt_fl<16, 32> __gemm_acc;",
            "auto __gemm_b_sub = ",
            "auto __gemm_resid_sub = ",
            // The residual subtile is a [16, 32] slice of residual_smem.
            ".template subtile<16, 32>(int2{0, static_cast<int>(kittens::warpid())})",
            // bf16 -> bf16 loads for A and B; bf16 -> fp32 load for residual
            // (same `kittens::warp::load` call site — TK does the convert).
            "kittens::warp::load(__gemm_a, ",
            "kittens::warp::load(__gemm_b, __gemm_b_sub);",
            "kittens::warp::load(__gemm_acc, __gemm_resid_sub);",
            // Single fused mma: D = A*B + C, with C = residual (=acc).
            "kittens::warp::mma_AB(__gemm_acc, __gemm_a, __gemm_b, __gemm_acc);",
            // Store accumulator back IN PLACE to residual subtile.
            "kittens::warp::store(__gemm_resid_sub, __gemm_acc);",
            "kittens::group<8>::sync(7);",
            // Consumer publishes page_done[residual] + arrives on
            // both input pages' consumed sems. Residual_consumed is
            // the storer's job.
            "kittens::group<1>::arrive(ss.page_done[2]);",
            "kittens::group<1>::arrive(ss.page_consumed[0]);",
            "kittens::group<1>::arrive(ss.page_consumed[1]);",
            // Storer: TMA-store residual back to its own gmem slot
            // (act_ptrs[2]) and arrive on residual's consumed sem.
            "kittens::group<1>::tma::store_async(",
            "act_ptrs[2]",
            "kittens::group<1>::tma::store_async_wait();",
            "kittens::group<1>::arrive(ss.page_consumed[2]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?} in source, got:\n{}",
                cu.source
            );
        }

        // Crucially: NO `__gemm_out_sub` (TkFusedGemmAdd writes
        // back to residual, not a separate out page) and NO zero
        // of the accumulator (the residual fills that slot
        // instead).
        assert!(
            !cu.source.contains("__gemm_out_sub"),
            "TkFusedGemmAdd must not declare an out-subtile; got:\n{}",
            cu.source
        );
        assert!(
            !cu.source.contains("kittens::warp::zero(__gemm_acc)"),
            "TkFusedGemmAdd must not zero the accumulator (residual fills it); got:\n{}",
            cu.source
        );
    }

    /// Sprint 12: FusedGateUpActivateMul — `out = silu(A @ W_gate)
    /// * (A @ W_up)`. Mirror of the S10 Gemm smoke with two
    /// matmuls + activation + elementwise mul. Smoke config:
    /// M=16, HIDDEN=32, INTERMEDIATE=256, NCW=8. TILE_N=32.
    /// gate_bytes = up_bytes = 32 * 256 * 2 = 16_384 bytes;
    /// scratch budget 32_768 holds both halves.
    #[test]
    fn fused_gate_up_silu_emits_tk20_calls() {
        use crate::ir::nodes::GateUpActivation;
        use crate::ir::substrate::{
            BarSyncId, HiddenDim, IntermediateDim, IterCount, MlpScope, ScratchRegion, TileN,
        };
        let mut b = BuilderD::new();
        b.push_fused_gate_up_activate_mul(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(), // in
            PageId::<1, 8>::new(), // weight
            PageId::<2, 8>::new(), // out
            ScratchRegion::<0, 16_384, 32_768, MlpScope>::new(),
            ScratchRegion::<16_384, 16_384, 32_768, MlpScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            IterCount::<1>::new(),
            LayerIndex::<3, 16>::new(),
            HiddenDim::<32>::new(),
            IntermediateDim::<256>::new(),
            NumTokensConst::<16>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<2, { u32::MAX }>::new(),
            WeightAccessorConst::<5, { u32::MAX }>::new(),
            TileN::<32>::new(),
            BarSyncId::<7>::new(),
            "W::mlp".to_string(),
            GateUpActivation::Silu,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_fused_gate_up_silu", &tape);
        assert!(
            cu.skipped_variants.is_empty(),
            "skipped: {:?}",
            cu.skipped_variants
        );
        std::fs::write("/tmp/fused_gate_up_silu_emit.cu", &cu.source).ok();

        for needle in [
            // Loader: act + 2 weight TMAs (gate at offset 0, up at
            // gate_bytes). Single weight_ready expects total bytes.
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 1024);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 32768);",
            "(*reinterpret_cast<kittens::st_bf<32, 256>*>(ss.scratch + 0))",
            "(*reinterpret_cast<kittens::st_bf<32, 256>*>(ss.scratch + 16384))",
            // Up half pointer = base + gate_bytes/2 elements.
            // gate_bytes=16384 → element offset = 8192.
            "+ 8192)",
            // Consumer: 5 register tiles (A, gate_b, up_b, gate_acc,
            // up_acc) + 3 subtiles.
            "kittens::rt_bf<16, 32> __gu_a;",
            "kittens::rt_bf<32, 32, kittens::ducks::rt_layout::col> __gu_gate_b;",
            "kittens::rt_bf<32, 32, kittens::ducks::rt_layout::col> __gu_up_b;",
            "kittens::rt_fl<16, 32> __gu_gate_acc;",
            "kittens::rt_fl<16, 32> __gu_up_acc;",
            "auto __gu_gate_b_sub = ",
            "auto __gu_up_b_sub = ",
            "auto __gu_out_sub = ",
            // mma both halves; both accs zeroed; activation lambda;
            // tile-elementwise mul; store to out subtile.
            "kittens::warp::zero(__gu_gate_acc);",
            "kittens::warp::zero(__gu_up_acc);",
            "kittens::warp::mma_AB(__gu_gate_acc, __gu_a, __gu_gate_b, __gu_gate_acc);",
            "kittens::warp::mma_AB(__gu_up_acc, __gu_a, __gu_up_b, __gu_up_acc);",
            // silu lambda
            "x * (1.0f / (1.0f + __expf(-x)))",
            "kittens::warp::apply(__gu_gate_acc, __gu_gate_acc,",
            "kittens::warp::mul(__gu_gate_acc, __gu_gate_acc, __gu_up_acc);",
            "kittens::warp::store(__gu_out_sub, __gu_gate_acc);",
            "kittens::group<8>::sync(7);",
            "kittens::group<1>::arrive(ss.page_done[2]);",
            "kittens::group<1>::arrive(ss.page_consumed[0]);",
            "kittens::group<1>::arrive(ss.page_consumed[1]);",
            // Storer: TMA out_smem → act_ptrs[2].
            "kittens::group<1>::tma::store_async(",
            "act_ptrs[2]",
            "kittens::group<1>::arrive(ss.page_consumed[2]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?} in source, got:\n{}",
                cu.source
            );
        }

        // Negative: must NOT contain the gelu tanhf lambda when
        // activation is Silu.
        assert!(
            !cu.source.contains("tanhf(0.7978845608028654f"),
            "Silu kernel must not emit gelu tanhf; got:\n{}",
            cu.source
        );
    }

    /// Sprint 12: FusedGateUpActivateMul with Gelu activation.
    /// Same shape as the silu smoke; only the activation lambda
    /// differs. Verifies the activation enum drives the lambda
    /// selection in emit_fused_gate_up_activate_mul.
    #[test]
    fn fused_gate_up_gelu_emits_tk20_calls() {
        use crate::ir::nodes::GateUpActivation;
        use crate::ir::substrate::{
            BarSyncId, HiddenDim, IntermediateDim, IterCount, MlpScope, ScratchRegion, TileN,
        };
        let mut b = BuilderD::new();
        b.push_fused_gate_up_activate_mul(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(),
            PageId::<1, 8>::new(),
            PageId::<2, 8>::new(),
            ScratchRegion::<0, 16_384, 32_768, MlpScope>::new(),
            ScratchRegion::<16_384, 16_384, 32_768, MlpScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            IterCount::<1>::new(),
            LayerIndex::<3, 16>::new(),
            HiddenDim::<32>::new(),
            IntermediateDim::<256>::new(),
            NumTokensConst::<16>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<2, { u32::MAX }>::new(),
            WeightAccessorConst::<5, { u32::MAX }>::new(),
            TileN::<32>::new(),
            BarSyncId::<7>::new(),
            "W::mlp".to_string(),
            GateUpActivation::Gelu,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_fused_gate_up_gelu", &tape);
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/fused_gate_up_gelu_emit.cu", &cu.source).ok();
        // Gelu-tanh lambda present; silu lambda absent.
        assert!(cu.source.contains("tanhf(0.7978845608028654f"));
        assert!(!cu.source.contains("(1.0f / (1.0f + __expf(-x)))"));
    }

    /// Sprint 1: Rust-side smoke. Verifies the emit walks without
    /// panicking AND that key TK 2.0 primitive calls land in the
    /// emitted source. The REAL DOD is `nvcc` compiling this on
    /// the pod — see CUDA_EMIT_TK20_AUDIT.md §15.
    #[test]
    fn rms_norm_emits_tk20_calls() {
        let mut b = BuilderD::new();
        b.push_rms_norm(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(),
            PageId::<1, 8>::new(),
            ScratchRegion::<0, 32, 32_768, RmsNormScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            LayerIndex::<0, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<1>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            WeightAccessorConst::<7, { u32::MAX }>::new(),
            BarSyncId::<3>::new(),
            BarSyncId::<4>::new(),
            BarSyncPair::<3, 4>::new(),
            "W::rms".to_string(),
            1.0e-5_f32,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_rms", &tape);
        assert!(cu.skipped_variants.is_empty());

        // Dump for visual inspection / pod compile.
        std::fs::write("/tmp/rms_norm_emit.cu", &cu.source).ok();

        // Substrate scaffolding present.
        for needle in [
            "#include \"ferrite_substrate.cuh\"",
            "#include \"ferrite_warp_roles.cuh\"",
            "struct Config_test_rms",
            "static constexpr int NUM_CONSUMER_WARPS = 8;",
            "extern \"C\" __global__ void ferrite_test_rms_launch(",
            "ferrite::set_consumer_registers<ConfigT>();",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?} in source, got:\n{}",
                cu.source
            );
        }

        // TK 2.0 primitive calls (every `kittens::group<N>::*` call
        // from the per-variant emit must be a TK 2.0 surface
        // citation; see roles.rs::emit_rms_norm comment block).
        for needle in [
            "kittens::group<1>::wait(",
            "kittens::group<1>::tma::expect_bytes(",
            "kittens::group<1>::tma::load_async(",
            "kittens::group<1>::tma::store_async(",
            "kittens::group<1>::tma::store_async_wait();",
            "kittens::group<1>::arrive(",
            "kittens::group<8>::load(__rms_act_rv,",
            "kittens::group<8>::store(",
            "kittens::group<8>::sync(3);",
            "kittens::group<8>::sync(4);",
            "kittens::warp::copy(__rms_sq_rv, __rms_act_rv);",
            "kittens::warp::mul(__rms_sq_rv, __rms_sq_rv, __rms_sq_rv);",
            "kittens::warp::sum(__rms_partial_sum, __rms_sq_rv);",
            "kittens::warp::mul(__rms_act_rv, __rms_act_rv, __rms_scale);",
            "kittens::warp::mul(__rms_act_rv, __rms_act_rv, __rms_weight_rv);",
            "kittens::rv_fl<256> __rms_act_rv;",
            "rsqrtf(__rms_full_sum / 2048.0f + 1e-5f)",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected TK 2.0 call {needle:?} in source, got:\n{}",
                cu.source
            );
        }

        // (S2 negative shared with rms_norm's NEGATIVE block below.)

        // NEGATIVE: must NOT contain any TK 1.0 / VM-reference
        // patterns or pre-existing ferrite_tk_helpers.cuh wrappers.
        for forbidden in [
            // TK 1.0 free-function pattern.
            "kittens::wait(",
            "kittens::arrive(",
            "kittens::tma::load_async(",
            "kittens::tma::expect_bytes(",
            // Manual per-warp slice carving (the nuked emit's pattern).
            "kittens::warpid() * ",
            "* sizeof(__nv_bfloat16))",
            // ferrite::tk wrappers (Sprint 1 = TK 2.0 only).
            "ferrite::tk::rms_norm_vec",
            "ferrite::tk::rms_norm_scale_from_rv",
        ] {
            assert!(
                !cu.source.contains(forbidden),
                "FORBIDDEN pattern {forbidden:?} found in source — TK 1.0 pollution. Source:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 8: Embed — per-token TMA gather from vocab table.
    #[test]
    fn embed_emits_per_token_gather() {
        use crate::ir::substrate::VocabSize;
        let mut b = BuilderD::new();
        b.push_embed(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(), // out_page
            PageId::<1, 8>::new(), // embed_weight_page (unused by emit)
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<1>::new(),
            VocabSize::<128_256>::new(),
            ActSlotConst::<7, { u32::MAX }>::new(),
            WeightAccessorConst::<11, { u32::MAX }>::new(),
            "W::embed".to_string(),
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_embed", &tape);
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/embed_emit.cu", &cu.source).ok();
        for needle in [
            "const uint32_t*             input_ids\n",
            "for (int __embed_t = 0; __embed_t < 1; ++__embed_t)",
            "const uint32_t __embed_row = input_ids[__embed_t];",
            "weight_ptrs[11 * 16 + 0]",
            "kittens::group<1>::tma::load_async(__embed_dst, __embed_src, 4096,",
            "kittens::group<1>::tma::store_async(__pt_dst, __pt_src, 4096);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?}, got:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 7: BarrierSignal/Wait — cross-CTA gmem barriers via
    /// `ferrite::barrier_signal/wait`.
    #[test]
    fn barrier_signal_emits_ferrite_call() {
        use crate::ir::substrate::EdgeId;
        let mut b = BuilderD::new();
        b.push_barrier_signal::<2>(EdgeId::<2, 4>::new());
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_bsig", &tape);
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/barrier_signal_emit.cu", &cu.source).ok();
        for needle in [
            "ferrite::barrier_signal(&barrier_slots[2], 1);",
            "kittens::group<1>::sync();",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?}, got:\n{}",
                cu.source
            );
        }
    }

    #[test]
    fn barrier_wait_emits_ferrite_call() {
        use crate::ir::substrate::{EdgeId, ExpectedCount};
        let mut b = BuilderD::new();
        b.push_barrier_wait::<1, 16>(EdgeId::<1, 4>::new(), ExpectedCount::<16>::new());
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_bwait", &tape);
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/barrier_wait_emit.cu", &cu.source).ok();
        assert!(
            cu.source
                .contains("ferrite::barrier_wait(&barrier_slots[1], 16);"),
            "got:\n{}",
            cu.source
        );
    }

    /// Sprint 6: ScalarOffsetRmsNorm — out = (act * scale) *
    /// (weight + offset). gemma2's rms_norm_offset reformulation.
    #[test]
    fn scalar_offset_rms_norm_emits_tk20_calls() {
        use crate::ir::substrate::RmsNormScope;
        let mut b = BuilderD::new();
        b.push_scalar_offset_rms_norm(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(), // in
            PageId::<1, 8>::new(), // weight
            ScratchRegion::<0, 32, 32_768, RmsNormScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            LayerIndex::<5, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<1>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            WeightAccessorConst::<3, { u32::MAX }>::new(),
            BarSyncId::<1>::new(),
            BarSyncId::<2>::new(),
            BarSyncPair::<1, 2>::new(),
            "W::sors".to_string(),
            1.0_f32,
            1.0e-5_f32,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_sors", &tape);
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/scalar_offset_rms_norm_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::group<8>::load(__sors_act_rv,",
            "kittens::warp::sum(__sors_partial_sum, __sors_sq_rv);",
            "kittens::group<8>::sync(1);",
            "kittens::warp::mul(__sors_act_rv, __sors_act_rv, __sors_scale);",
            "kittens::warp::add(__sors_weight_rv, __sors_weight_rv, 1e0f);",
            "kittens::warp::mul(__sors_act_rv, __sors_act_rv, __sors_weight_rv);",
            "kittens::group<8>::sync(2);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?}, got:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 5: FusedAddRmsNorm — residual+=delta, then RmsNorm,
    /// in-place to residual page.
    #[test]
    fn fused_add_rms_norm_emits_tk20_calls() {
        use crate::ir::substrate::RmsNormScope;
        let mut b = BuilderD::new();
        b.push_fused_add_rms_norm(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(), // delta
            PageId::<1, 8>::new(), // residual
            PageId::<2, 8>::new(), // weight
            ScratchRegion::<0, 32, 32_768, RmsNormScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            LayerIndex::<3, 16>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<1>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            WeightAccessorConst::<5, { u32::MAX }>::new(),
            BarSyncId::<1>::new(),
            BarSyncId::<2>::new(),
            BarSyncPair::<1, 2>::new(),
            "W::farn".to_string(),
            1.0e-5_f32,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_farn", &tape);
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/fused_add_rms_norm_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::group<8>::load(__farn_delta_rv,",
            "kittens::group<8>::load(__farn_res_rv,",
            "kittens::warp::add(__farn_res_rv, __farn_res_rv, __farn_delta_rv);",
            "kittens::warp::sum(__farn_partial_sum, __farn_sq_rv);",
            "kittens::group<8>::sync(1);",
            "kittens::group<8>::sync(2);",
            "rsqrtf(__farn_full_sum / 2048.0f + 1e-5f)",
            "kittens::warp::mul(__farn_res_rv, __farn_res_rv, __farn_scale);",
            "kittens::warp::mul(__farn_res_rv, __farn_res_rv, __farn_weight_rv);",
            "kittens::group<8>::store(",
            "weight_ptrs[5 * 16 + 3]",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?}, got:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 4: TanhSoftCap via `kittens::warp::apply` lambda.
    #[test]
    fn tanh_soft_cap_emits_tk20_apply_lambda() {
        let mut b = BuilderD::new();
        b.push_tanh_soft_cap(
            ArrivesCount::<0>::new(),
            PageId::<6, 8>::new(),
            PageId::<7, 8>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<1>::new(),
            ActSlotConst::<2, { u32::MAX }>::new(),
            ActSlotConst::<3, { u32::MAX }>::new(),
            BarSyncId::<2>::new(),
            30.0_f32,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_softcap", &tape);
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/tanh_soft_cap_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::rv_fl<256> __tanh_rv;",
            "kittens::warp::apply(__tanh_rv, __tanh_rv,",
            "[] __device__ (int /*idx*/, float x) { return tanhf(x * (1.0f / 3e1f)) * 3e1f; }",
            "kittens::group<8>::sync(2);",
            "kittens::group<1>::tma::store_async(",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?}, got:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 3: ScalarMul (in→out path; in==out exercised at user
    /// build time when the proc-macro emits a tape with aliasing).
    #[test]
    fn scalar_mul_emits_tk20_calls() {
        let mut b = BuilderD::new();
        b.push_scalar_mul(
            ArrivesCount::<0>::new(),
            PageId::<4, 8>::new(),
            PageId::<5, 8>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<1>::new(),
            ActSlotConst::<2, { u32::MAX }>::new(),
            ActSlotConst::<3, { u32::MAX }>::new(),
            BarSyncId::<2>::new(),
            0.5_f32,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_smul", &tape);
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/scalar_mul_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::rv_fl<256> __smul_rv;",
            "kittens::group<8>::load(__smul_rv,",
            "kittens::warp::mul(__smul_rv, __smul_rv, 5e-1f);",
            "kittens::group<8>::sync(2);",
            "kittens::group<1>::arrive(ss.page_done[5]);",
            "kittens::group<1>::arrive(ss.page_consumed[4]);",
            "kittens::group<1>::tma::store_async(",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?}, got:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 2: Add. Same DOD pattern — Rust check + nvcc on pod.
    #[test]
    fn add_emits_tk20_calls() {
        let mut b = BuilderD::new();
        b.push_add(
            ArrivesCount::<0>::new(),
            PageId::<2, 8>::new(),
            PageId::<3, 8>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<1>::new(),
            ActSlotConst::<5, { u32::MAX }>::new(),
            ActSlotConst::<6, { u32::MAX }>::new(),
            BarSyncId::<2>::new(),
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_add", &tape);
        assert!(cu.skipped_variants.is_empty());

        std::fs::write("/tmp/add_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[2], 4096);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[3], 4096);",
            "kittens::rv_fl<256> __add_delta_rv;",
            "kittens::rv_fl<256> __add_res_rv;",
            "kittens::group<8>::load(__add_delta_rv,",
            "kittens::group<8>::load(__add_res_rv,",
            "kittens::warp::add(__add_res_rv, __add_res_rv, __add_delta_rv);",
            "kittens::group<8>::store(",
            "kittens::group<8>::sync(2);",
            "kittens::group<1>::arrive(ss.page_done[3]);",
            "kittens::group<1>::arrive(ss.page_consumed[2]);",
            "kittens::group<1>::tma::store_async(",
            "kittens::group<1>::tma::store_async_wait();",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected TK 2.0 call {needle:?}, got:\n{}",
                cu.source
            );
        }
    }

    /// Builder type for S13 norm-gemm smoke: same params as
    /// `BuilderD` but with double the scratch budget so the
    /// linear-projection b_tile fits alongside the per-warp partial
    /// sums.
    type BuilderLg = MegaTapeBuilder<8, 8, 32_768, 65_536, 4>;

    /// Sprint 13: TkFusedNormGemm — `out = (rms_norm(in) * norm_w)
    /// @ lin_w` (no-delta RmsNorm flavor). Smoke config: M=16,
    /// K=128, N=128, NCW=8. K_per_warp=16 (rv_fl<16>); TILE_N=16
    /// (mma_AB ok). b_tile = 128*128*2 = 32_768 bytes; partial =
    /// 8*4 = 32 bytes. Both fit in the 65_536-byte scratch budget.
    #[test]
    fn tk_fused_norm_gemm_no_delta_emits_tk20_calls() {
        use crate::ir::nodes::LmHeadNormKind;
        use crate::ir::substrate::{
            BarSyncId, BarSyncPair, ChunkK, GemmScope, IterCount, MatmulK, MatmulN,
            ScratchRegion, TileN,
        };
        let mut b = BuilderLg::new();
        b.push_tk_fused_norm_gemm_no_delta(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(), // in
            PageId::<1, 8>::new(), // norm_weight
            PageId::<2, 8>::new(), // linear_weight
            PageId::<3, 8>::new(), // out
            ScratchRegion::<0, 32, 65_536, GemmScope>::new(), // partial
            ScratchRegion::<32, 32_768, 65_536, GemmScope>::new(), // b_tile
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            IterCount::<1>::new(),
            LayerIndex::<3, 16>::new(),
            MatmulN::<128>::new(),
            MatmulK::<128>::new(),
            NumTokensConst::<16>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<3, { u32::MAX }>::new(),
            WeightAccessorConst::<5, { u32::MAX }>::new(),
            WeightAccessorConst::<6, { u32::MAX }>::new(),
            TileN::<16>::new(),
            ChunkK::<128>::new(),
            BarSyncId::<1>::new(),
            BarSyncId::<2>::new(),
            BarSyncPair::<1, 2>::new(),
            "W::lmh_norm".to_string(),
            "W::lmh_lin".to_string(),
            LmHeadNormKind::RmsNorm,
            1.0e-5_f32,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_tk_fused_norm_gemm_no_delta", &tape);
        assert!(
            cu.skipped_variants.is_empty(),
            "skipped: {:?}",
            cu.skipped_variants
        );
        std::fs::write("/tmp/tk_fused_norm_gemm_no_delta_emit.cu", &cu.source).ok();

        for needle in [
            // Loader: TMA-loads activation, norm_weight, linear_weight (-> b_tile).
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 4096);", // act
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 256);",  // norm_w
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[2], 32768);", // lin_w
            "(*reinterpret_cast<kittens::sv_bf<128>*>(ss.pages[0]))",
            "(*reinterpret_cast<kittens::sv_bf<128>*>(ss.pages[1]))",
            "(*reinterpret_cast<kittens::st_bf<128, 128>*>(ss.scratch + 32))",
            // Norm phase: per-warp register vecs + cross-warp sum.
            "kittens::rv_fl<16> __lmh_act_rv;",
            "kittens::rv_fl<16> __lmh_sq_rv;",
            "kittens::rv_fl<16> __lmh_norm_w_rv;",
            "kittens::group<8>::load(__lmh_act_rv,",
            "kittens::warp::sum(__lmh_partial_sum, __lmh_sq_rv);",
            "kittens::group<8>::sync(1);", // bar_reduce
            "rsqrtf(__lmh_full_sum / 128.0f + 1e-5f)",
            "kittens::warp::mul(__lmh_act_rv, __lmh_act_rv, __lmh_scale);",
            "kittens::warp::mul(__lmh_act_rv, __lmh_act_rv, __lmh_norm_w_rv);",
            // Norm writeback to in_smem (sv_bf view).
            "kittens::group<8>::store(",
            // Cross-warp sync after norm writeback (bar_publish reused).
            "kittens::group<8>::sync(2);",
            // GEMM phase: rt decls + per-warp B + OUT subtiles + mma.
            "kittens::rt_bf<16, 128> __lmh_a;",
            "kittens::rt_bf<128, 16, kittens::ducks::rt_layout::col> __lmh_b;",
            "kittens::rt_fl<16, 16> __lmh_acc;",
            "auto __lmh_b_sub = ",
            "auto __lmh_out_sub = ",
            "kittens::warp::zero(__lmh_acc);",
            "kittens::warp::mma_AB(__lmh_acc, __lmh_a, __lmh_b, __lmh_acc);",
            "kittens::warp::store(__lmh_out_sub, __lmh_acc);",
            // Per-warp arrives — out_done + 3 input page_consumeds.
            "kittens::group<1>::arrive(ss.page_done[3]);",
            "kittens::group<1>::arrive(ss.page_consumed[0]);",
            "kittens::group<1>::arrive(ss.page_consumed[1]);",
            "kittens::group<1>::arrive(ss.page_consumed[2]);",
            // Storer: TMA out + arrive on out_consumed.
            "kittens::group<1>::tma::store_async(",
            "act_ptrs[3]",
            "kittens::group<1>::tma::store_async_wait();",
            "kittens::group<1>::arrive(ss.page_consumed[3]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?} in source, got:\n{}",
                cu.source
            );
        }

        // No-delta: must NOT emit a delta load or delta_consumed
        // arrive. (delta_act_slot is None.)
        assert!(
            !cu.source.contains("__lmh_delta_rv"),
            "no-delta variant must not declare a delta rv; got:\n{}",
            cu.source
        );
        // RmsNorm flavor: must NOT emit a mean-subtract step.
        assert!(
            !cu.source.contains("__lmh_mean_partial"),
            "RmsNorm flavor must not emit a mean-subtract; got:\n{}",
            cu.source
        );
        // RmsNorm flavor (no offset): norm_weight must NOT have an
        // offset add. The only `warp::add(__lmh_*` should be absent
        // in the no-delta + no-offset case.
        assert!(
            !cu.source.contains("kittens::warp::add(__lmh_norm_w_rv,"),
            "RmsNorm with no offset must not add to norm_weight; got:\n{}",
            cu.source
        );
    }

    /// Sprint 13: TkFusedNormGemm — AddScalarOffsetRmsNorm flavor
    /// (delta fold + offset on norm_weight). Same shape config as
    /// the no-delta smoke, with the delta page wired to slot 1.
    #[test]
    fn tk_fused_norm_gemm_with_delta_emits_tk20_calls() {
        use crate::ir::nodes::LmHeadNormKind;
        use crate::ir::substrate::{
            BarSyncId, BarSyncPair, ChunkK, GemmScope, IterCount, MatmulK, MatmulN,
            ScratchRegion, TileN,
        };
        let mut b = BuilderLg::new();
        b.push_tk_fused_norm_gemm_with_delta(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(), // in (residual)
            PageId::<1, 8>::new(), // delta
            PageId::<2, 8>::new(), // norm_weight
            PageId::<3, 8>::new(), // linear_weight
            PageId::<4, 8>::new(), // out
            ScratchRegion::<0, 32, 65_536, GemmScope>::new(),
            ScratchRegion::<32, 32_768, 65_536, GemmScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            IterCount::<1>::new(),
            LayerIndex::<3, 16>::new(),
            MatmulN::<128>::new(),
            MatmulK::<128>::new(),
            NumTokensConst::<16>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(),
            ActSlotConst::<1, { u32::MAX }>::new(),
            ActSlotConst::<4, { u32::MAX }>::new(),
            WeightAccessorConst::<5, { u32::MAX }>::new(),
            WeightAccessorConst::<6, { u32::MAX }>::new(),
            TileN::<16>::new(),
            ChunkK::<128>::new(),
            BarSyncId::<1>::new(),
            BarSyncId::<2>::new(),
            BarSyncPair::<1, 2>::new(),
            "W::lmh_norm".to_string(),
            "W::lmh_lin".to_string(),
            LmHeadNormKind::AddScalarOffsetRmsNorm,
            Some(1.5_f32),
            1.0e-5_f32,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_tk_fused_norm_gemm_with_delta", &tape);
        assert!(
            cu.skipped_variants.is_empty(),
            "skipped: {:?}",
            cu.skipped_variants
        );
        std::fs::write("/tmp/tk_fused_norm_gemm_with_delta_emit.cu", &cu.source).ok();

        for needle in [
            // Loader: TMA-loads in + delta + norm_weight + linear_weight.
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 4096);", // in
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 4096);", // delta
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[2], 256);",  // norm_w
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[3], 32768);", // lin_w
            "(*reinterpret_cast<kittens::sv_bf<128>*>(ss.pages[1]))",
            // Residual fold: act += delta in registers.
            "kittens::rv_fl<16> __lmh_delta_rv;",
            "kittens::warp::add(__lmh_act_rv, __lmh_act_rv, __lmh_delta_rv);",
            // Offset added to norm_weight before mul.
            "kittens::warp::add(__lmh_norm_w_rv, __lmh_norm_w_rv, 1.5e0f);",
            // mma_AB still emits.
            "kittens::warp::mma_AB(__lmh_acc, __lmh_a, __lmh_b, __lmh_acc);",
            // Per-warp arrives now include delta_consumed.
            "kittens::group<1>::arrive(ss.page_consumed[1]);",
            "kittens::group<1>::arrive(ss.page_done[4]);",
            "kittens::group<1>::arrive(ss.page_consumed[4]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?} in source, got:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 14: SpliceMmEmbeds — passthrough emit. The actual
    /// D2D vision-embed copy happens outside the megakernel; the
    /// kernel only advances the page's barrier cycle (arrives bump
    /// in `push_splice_mm_embeds`) so the next op's phase parity
    /// stays correct. No `act_ptrs` / `weight_ptrs` / `input_ids`
    /// are touched.
    #[test]
    fn splice_mm_embeds_emits_passthrough() {
        let mut b = BuilderD::new();
        b.push_splice_mm_embeds(
            ArrivesCount::<0>::new(),
            PageId::<3, 8>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<7>::new(),
            ActSlotConst::<3, { u32::MAX }>::new(),
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_splice", &tape);
        assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
        std::fs::write("/tmp/splice_mm_embeds_emit.cu", &cu.source).ok();

        for needle in [
            // Loader: wait page_consumed, arrive page_ready.
            "kittens::group<1>::wait(ss.page_consumed[3], 1);",
            "kittens::group<1>::arrive(ss.page_ready[3]);",
            // Consumer: wait page_ready, warp 0 arrives page_done.
            "kittens::group<1>::wait(ss.page_ready[3], 0);",
            "if (kittens::warpid() == 0) {",
            "kittens::group<1>::arrive(ss.page_done[3]);",
            // Storer: wait page_done, arrive page_consumed.
            "kittens::group<1>::wait(ss.page_done[3], 1);",
            "kittens::group<1>::arrive(ss.page_consumed[3]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?} in source, got:\n{}",
                cu.source
            );
        }

        // Negative: no role-body reads of kernel-arg locals
        // (`act_ptrs[N]`, `weight_ptrs[...]`, `input_ids[t]`,
        // `barrier_slots[N]`) and no TMA. The kernel signature
        // still declares those args — these checks look for the
        // INDEXING form which only role bodies emit. Passthrough
        // touches only `ss.*` sems.
        for forbidden in [
            "act_ptrs[",
            "weight_ptrs[",
            "input_ids[",
            "barrier_slots[",
            "tma::load_async",
            "tma::store_async",
            "tma::expect_bytes",
        ] {
            assert!(
                !cu.source.contains(forbidden),
                "forbidden {forbidden:?} appeared in passthrough emit:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 15: FusedQkvRopeCache — fused QKV linear projection +
    /// RoPE rotation + (out-of-kernel) reshape_and_cache. Smoke
    /// asserts the LaunchTier::Qkv kernel signature, per-token cos/
    /// sin gather, per-head matmul + per-region routing, RoPE pair
    /// rotation lambda, and Q/K/V output stores.
    ///
    /// Smoke config: M=16, HIDDEN_DIM=64, HEAD_DIM=16, NUM_Q_HEADS=4,
    /// NUM_KV_HEADS=2, qkv_n=128, NCW=8 → tile_n=16 = head_dim →
    /// heads_per_warp=1. Each warp processes one head; warp 0..3 hit
    /// Q region, warp 4 hits K region (cols 64..80), warp 5 hits K
    /// (cols 80..96), warp 6 hits V (cols 96..112), warp 7 hits V
    /// (cols 112..128). Scratch: q_rope (0..512) + k_rope (512..1024)
    /// + qkv_b_tile (1024..17408). Pages sized to fit the tile.
    #[test]
    fn fused_qkv_rope_cache_emits_tk20_calls() {
        use crate::ir::substrate::{
            BarSyncId, ChunkK, GemmScope, HeadDim, HiddenDim, IterCount, NumKvHeads,
            NumQHeads, RopeScope, ScratchRegion, TileN,
        };
        let mut b = BuilderLg::new();
        b.push_fused_qkv_rope_cache(
            ArrivesCount::<0>::new(),
            PageId::<0, 8>::new(), // in
            PageId::<1, 8>::new(), // qkv_weight (page-ready barrier)
            PageId::<2, 8>::new(), // cos_sin
            PageId::<3, 8>::new(), // q_out
            PageId::<4, 8>::new(), // k_out
            PageId::<5, 8>::new(), // v_out
            ScratchRegion::<0, 512, 65_536, RopeScope>::new(),
            ScratchRegion::<512, 512, 65_536, RopeScope>::new(),
            ScratchRegion::<1024, 16_384, 65_536, GemmScope>::new(),
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            IterCount::<1>::new(),
            LayerIndex::<2, 16>::new(),
            HiddenDim::<64>::new(),
            HeadDim::<16>::new(),
            NumQHeads::<4>::new(),
            NumKvHeads::<2>::new(),
            NumTokensConst::<16>::new(),
            ActSlotConst::<0, { u32::MAX }>::new(), // in
            ActSlotConst::<3, { u32::MAX }>::new(), // q_out
            ActSlotConst::<4, { u32::MAX }>::new(), // k_out
            ActSlotConst::<5, { u32::MAX }>::new(), // v_out
            WeightAccessorConst::<7, { u32::MAX }>::new(), // qkv_weight
            WeightAccessorConst::<8, { u32::MAX }>::new(), // rotary
            TileN::<16>::new(),
            ChunkK::<64>::new(),
            BarSyncId::<1>::new(),
            "W::qkv".to_string(),
            "W::rot".to_string(),
            true,
            false,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_fqrc", &tape);
        assert!(
            cu.skipped_variants.is_empty(),
            "skipped: {:?}",
            cu.skipped_variants
        );
        std::fs::write("/tmp/fused_qkv_rope_cache_emit.cu", &cu.source).ok();

        for needle in [
            // QKV-tier kernel signature (LaunchTier::Qkv): the
            // emitted megakernel takes positions/slot_mapping/
            // key_cache_ptrs/value_cache_ptrs in addition to the
            // base-tier args. (The kernel doesn't dereference cache
            // ptrs today; the signature exists for ABI parity with
            // the host's `LaunchArgsQkv`.)
            "const uint32_t*             positions,",
            "const int64_t*              slot_mapping,",
            "__nv_bfloat16* const*       key_cache_ptrs,",
            "__nv_bfloat16* const*       value_cache_ptrs,",
            "(void)slot_mapping;",
            "(void)key_cache_ptrs;",
            "(void)value_cache_ptrs;",
            // Loader: TMAs activation, qkv weight (-> b_tile scratch),
            // cos_sin per-token gather indexed by positions[t].
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 2048);", // act
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 16384);", // qkv weight -> b_tile
            "(*reinterpret_cast<kittens::st_bf<16, 64>*>(ss.pages[0]))",
            "(*reinterpret_cast<kittens::st_bf<64, 128>*>(ss.scratch + 1024))",
            "for (int __cs_t = 0; __cs_t < 16; ++__cs_t)",
            "positions[__cs_t]",
            // Consumer: A reg tile, qkv_b_tile views, per-head loop,
            // per-region branching, RoPE apply lambda.
            "kittens::rt_bf<16, 64> __qkv_a;",
            "auto& __qkv_q_rope = *reinterpret_cast<kittens::st_bf<16, 64>*>(ss.scratch + 0);",
            "auto& __qkv_k_rope = *reinterpret_cast<kittens::st_bf<16, 32>*>(ss.scratch + 512);",
            "__nv_bfloat16* __qkv_cos_sin_ptr = reinterpret_cast<__nv_bfloat16*>(ss.pages[2]);",
            "for (int __qkv_h = 0; __qkv_h < 1; ++__qkv_h)",
            "kittens::rt_bf<64, 16, kittens::ducks::rt_layout::col> __qkv_b;",
            "kittens::rt_fl<16, 16> __qkv_acc;",
            "kittens::warp::mma_AB(__qkv_acc, __qkv_a, __qkv_b, __qkv_acc);",
            "if (__qkv_col < 64)",       // Q region
            "} else if (__qkv_col < 64 + 32)", // K region (q_dim + kv_dim)
            "} else {",                  // V region
            // RoPE pair rotation lambda.
            "kittens::warp::apply(__qkv_rot, __qkv_acc,",
            "constexpr int __half = 16 / 2;",
            "__bfloat162float(__qkv_stg_ptr[row * 16 + __pc])",
            "__bfloat162float(__qkv_cos_sin_ptr[row * 16 + __t])",
            "__bfloat162float(__qkv_cos_sin_ptr[row * 16 + __half + __t])",
            "return col < __half ? (x * __c - __paired * __s) : (x * __c + __paired * __s);",
            // Cross-warp publish bar + per-warp arrives.
            "kittens::group<8>::sync(1);",
            "kittens::group<1>::arrive(ss.page_done[3]);", // q_out
            "kittens::group<1>::arrive(ss.page_done[4]);", // k_out
            "kittens::group<1>::arrive(ss.page_done[5]);", // v_out
            // Storer: TMA Q/K/V outputs to act_ptrs[3..6].
            "act_ptrs[3]",
            "act_ptrs[4]",
            "act_ptrs[5]",
            "kittens::group<1>::tma::store_async_wait();",
            "kittens::group<1>::arrive(ss.page_consumed[3]);",
            "kittens::group<1>::arrive(ss.page_consumed[4]);",
            "kittens::group<1>::arrive(ss.page_consumed[5]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?} in source, got:\n{}",
                cu.source
            );
        }
    }
}
