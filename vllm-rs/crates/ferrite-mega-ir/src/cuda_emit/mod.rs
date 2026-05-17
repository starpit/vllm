// SPDX-License-Identifier: Apache-2.0
//! `cuda_emit` — pure literal transcription of a typed [`MegaTape`]
//! into a complete `.cu` translation unit.
//!
//! See `MEGA_IR_PLAN.md` §0 / §4a / §8.0 for the contract: every
//! value spliced into the emitted source comes from a typed getter
//! on a [`MegaNode`](crate::nodes::MegaNode). The emit step never
//! infers, never invents, never reaches outside the IR. If a
//! kernel-template / kernel-runtime arg has no IR getter, the IR is
//! incomplete and the variant's [`roles::emit_role_bodies`] returns
//! `RoleBodies::skipped` rather than fabricating a value.
//!
//! Architecturally:
//!
//! - [`tk`] is the Rust API surface for TK / ferrite::tk primitives.
//!   Each fn mirrors a CUDA entry point with typed handle args; the
//!   body emits the corresponding CUDA call as a [`CuStmt`] /
//!   [`CuExpr`]. Calling a TK primitive from emit code is a Rust
//!   function call, type-checked at codegen build time.
//!
//! - [`handles`] defines the typed handles ([`SmemColVec`],
//!   [`Semaphore`], [`ScratchPtr`], …) that flow through the [`tk`]
//!   API. Phantom dtype tags catch mismatches statically. Lengths
//!   are runtime `u32` values pulled from the IR.
//!
//! - [`roles`] holds the per-`MegaNode` role-body emit fns. Each
//!   returns four [`CuBlock`]s (loader / launcher / consumer /
//!   storer); [`lower_to_cuda`] concatenates them in tape order
//!   into the four role sections of the final kernel.
//!
//! Sprint 1 ships RmsNorm only. Every other variant returns
//! [`roles::RoleBodies::skipped`]; the walker collects the names
//! into [`CuVariant::skipped_variants`] and renders a
//! `// SKIPPED: <variant>` comment so a reader can see what's
//! missing in the kernel structure.

pub mod cu;
pub mod handles;
pub mod roles;
pub mod tk;

use crate::tape::{MegaTape, TapeBudget};

pub use cu::{CuBlock, CuExpr, CuStmt, CuVariant};

/// Lower a typed [`MegaTape`] to a complete `.cu` translation unit
/// for the named canonical.
///
/// The returned [`CuVariant::source`] is a single string holding the
/// full `.cu` text (includes preamble + per-canonical `Config` struct
/// + role-dispatch kernel entry point). Sprint 1 emits a real body
/// for `RmsNorm` only; every other variant lands as a `// SKIPPED`
/// comment in the appropriate role section, with the variant name
/// also appended to [`CuVariant::skipped_variants`].
pub fn lower_to_cuda(canonical: &str, tape: &MegaTape) -> CuVariant {
    let budget = tape.budget();

    let mut loader_block = CuBlock::new();
    let mut launcher_block = CuBlock::new();
    let mut consumer_block = CuBlock::new();
    let mut storer_block = CuBlock::new();
    let mut skipped: Vec<String> = Vec::new();

    for (idx, node) in tape.nodes().iter().enumerate() {
        let bodies = roles::emit_role_bodies(node, budget);
        if let Some(variant) = bodies.skipped {
            skipped.push(variant.to_string());
            // Render a SKIPPED marker into all four role sections so
            // the structural shape of the kernel reflects every node
            // in tape order.
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

    let source = render_source(canonical, &budget, &loader_block, &launcher_block, &consumer_block, &storer_block);

    CuVariant {
        canonical: canonical.to_string(),
        source,
        skipped_variants: skipped,
    }
}

/// Render the full `.cu` text. Pure literal-transcription of the
/// substrate scaffolding (substrate / globals / warp_roles /
/// tk_helpers includes; per-canonical `Config` struct; CTA-entry
/// init + role dispatch) wrapping the four role blocks.
fn render_source(
    canonical: &str,
    budget: &TapeBudget,
    loader: &CuBlock,
    launcher: &CuBlock,
    consumer: &CuBlock,
    storer: &CuBlock,
) -> String {
    // Register budgets are substrate-implementation values, not IR
    // values. They live in [`TapeBudget`] today only via the
    // num_consumer_warps field; consumer/non-consumer register
    // counts are fixed conventions of the substrate (matching
    // ferrite_warp_roles.cuh's expected Config interface). We
    // hardcode the values that ferrite kernels have always used;
    // promoting these to per-canonical values is a future sprint.
    const CONSUMER_REGISTERS: u32 = 240;
    const NON_CONSUMER_REGISTERS: u32 = 24;
    const INSTRUCTION_PIPE_STAGES: u32 = 4;

    let kernel_name = format!("ferrite_mega_{canonical}");
    let mut s = String::new();
    s.push_str("// SPDX-License-Identifier: Apache-2.0\n");
    s.push_str(&format!("// Auto-generated by ferrite-mega-ir::cuda_emit for canonical \"{canonical}\".\n"));
    s.push_str("// DO NOT EDIT — regenerate by re-running the proc-macro.\n\n");
    s.push_str("#include \"ferrite_substrate.cuh\"\n");
    s.push_str("#include \"ferrite_globals.cuh\"\n");
    s.push_str("#include \"ferrite_warp_roles.cuh\"\n");
    s.push_str("#include \"ferrite_tk_helpers.cuh\"\n\n");

    s.push_str("namespace {\n\n");

    // Per-canonical Config struct — the field set the substrate's
    // `SharedState<Config>` and `set_consumer_registers<Config>`
    // template arg requires.
    s.push_str(&format!("struct Config_{canonical} {{\n"));
    s.push_str(&format!("    static constexpr int NUM_CONSUMER_WARPS = {};\n", budget.num_consumer_warps));
    s.push_str(&format!("    static constexpr int NON_CONSUMER_REGISTERS = {NON_CONSUMER_REGISTERS};\n"));
    s.push_str(&format!("    static constexpr int CONSUMER_REGISTERS = {CONSUMER_REGISTERS};\n"));
    s.push_str(&format!("    static constexpr int NUM_PAGES = {};\n", budget.num_pages));
    s.push_str(&format!("    static constexpr int PAGE_SIZE = {};\n", budget.page_size));
    s.push_str(&format!("    static constexpr int SCRATCH_BYTES = {};\n", budget.scratch_bytes));
    s.push_str(&format!("    static constexpr int INSTRUCTION_PIPE_STAGES = {INSTRUCTION_PIPE_STAGES};\n"));
    s.push_str("};\n\n");

    s.push_str(&format!("using ConfigT = Config_{canonical};\n\n"));

    // Role bodies as inline functions. Each takes the per-CTA shared
    // state by reference plus a `Globals` struct holding act_ptrs /
    // weight_ptrs. The Globals shape is per-canonical; today we use
    // a minimal placeholder until lower_to_cuda is wired through to
    // the existing host launch ABI in `interpreter/mega/mod.rs`.
    s.push_str("struct Globals {\n");
    s.push_str("    __nv_bfloat16** act_ptrs;\n");
    s.push_str("    __nv_bfloat16** weight_ptrs;\n");
    s.push_str("};\n\n");

    s.push_str("__device__ __forceinline__ void loader_body(\n");
    s.push_str("    const Globals& g,\n");
    s.push_str("    ferrite::SharedState<ConfigT>& ss\n");
    s.push_str(") {\n");
    s.push_str(&loader.render(4));
    s.push_str("}\n\n");

    s.push_str("__device__ __forceinline__ void launcher_body(\n");
    s.push_str("    const Globals& g,\n");
    s.push_str("    ferrite::SharedState<ConfigT>& ss\n");
    s.push_str(") {\n");
    s.push_str(&launcher.render(4));
    s.push_str("}\n\n");

    s.push_str("__device__ __forceinline__ void consumer_body(\n");
    s.push_str("    const Globals& g,\n");
    s.push_str("    ferrite::SharedState<ConfigT>& ss\n");
    s.push_str(") {\n");
    s.push_str(&consumer.render(4));
    s.push_str("}\n\n");

    s.push_str("__device__ __forceinline__ void storer_body(\n");
    s.push_str("    const Globals& g,\n");
    s.push_str("    ferrite::SharedState<ConfigT>& ss\n");
    s.push_str(") {\n");
    s.push_str(&storer.render(4));
    s.push_str("}\n\n");

    // Kernel entry — fixed substrate scaffolding.
    s.push_str(&format!("extern \"C\" __global__ void {kernel_name}(\n"));
    s.push_str("    __nv_bfloat16** act_ptrs,\n");
    s.push_str("    __nv_bfloat16** weight_ptrs\n");
    s.push_str(") {\n");
    s.push_str("    extern __shared__ uint8_t shmem_buf[];\n");
    s.push_str("    auto& ss = *reinterpret_cast<ferrite::SharedState<ConfigT>*>(shmem_buf);\n");
    s.push_str("    ferrite::init_shared_state<ConfigT>(ss);\n");
    s.push_str("    Globals g{act_ptrs, weight_ptrs};\n");
    s.push_str("    int wid = kittens::warpid();\n");
    s.push_str("    if (wid < ConfigT::NUM_CONSUMER_WARPS) {\n");
    s.push_str("        ferrite::set_consumer_registers<ConfigT>();\n");
    s.push_str("        consumer_body(g, ss);\n");
    s.push_str("    } else {\n");
    s.push_str("        ferrite::set_non_consumer_registers<ConfigT>();\n");
    s.push_str("        switch (wid - ConfigT::NUM_CONSUMER_WARPS) {\n");
    s.push_str("            case ferrite::kLoaderSlot:   loader_body(g, ss); break;\n");
    s.push_str("            case ferrite::kLauncherSlot: launcher_body(g, ss); break;\n");
    s.push_str("            case ferrite::kStorerSlot:   storer_body(g, ss); break;\n");
    s.push_str("        }\n");
    s.push_str("    }\n");
    s.push_str("}\n\n");

    s.push_str("} // namespace\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::MegaTapeBuilder;
    use crate::nodes::LayerIndex;
    use crate::substrate::{
        ActSlotConst, ArrivesCount, BarSyncId, BarSyncPair, HiddenDim, MbarrierPhase,
        NumTokensConst, PageId, RmsNormScope, ScratchRegion, WeightAccessorConst,
    };

    type BuilderD = MegaTapeBuilder<8, 8, 32_768, 32_768, 4>;

    /// Sprint 1 sanity: a single RmsNorm tape produces a `.cu` that
    /// (a) contains the expected substrate includes, (b) calls
    /// `ferrite::tk::rms_norm_vec<NCW, HIDDEN_DIM, BAR>(...)` in the
    /// consumer body with values from the IR's typed getters,
    /// (c) carries a `Config` struct stamping the budget, and
    /// (d) reports zero skipped variants.
    #[test]
    fn lower_rms_norm_to_cuda_smoke() {
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

        let cu = lower_to_cuda("test_rms_only", &tape);

        assert_eq!(cu.canonical, "test_rms_only");
        assert!(cu.skipped_variants.is_empty(), "expected zero skipped variants, got {:?}", cu.skipped_variants);

        // Substrate scaffolding is present.
        for needle in [
            "#include \"ferrite_substrate.cuh\"",
            "#include \"ferrite_tk_helpers.cuh\"",
            "struct Config_test_rms_only",
            "static constexpr int NUM_CONSUMER_WARPS = 8;",
            "static constexpr int NUM_PAGES = 8;",
            "extern \"C\" __global__ void ferrite_mega_test_rms_only(",
            "ferrite::set_consumer_registers<ConfigT>();",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected source to contain {needle:?}, source was:\n{}",
                cu.source
            );
        }

        // Loader / consumer / storer call the right TK primitives
        // with values pulled from the IR's typed getters.
        for needle in [
            // Loader: tma::load_async to in_page (IN_ID=0) + weight_page (WEIGHT_ID=1).
            "kittens::tma::expect_bytes(ss.page_ready[0], 4096);", // 2048*1*2
            "kittens::tma::expect_bytes(ss.page_ready[1], 4096);", // 2048*1*2 (weight is per-row HIDDEN_DIM*sizeof(bf16))
            // Consumer: rms_norm_vec call with NCW=8, HIDDEN_DIM=2048, BAR=3.
            "ferrite::tk::rms_norm_vec<8, 2048, 3>(",
            // Cross-warp publish on BAR=4.
            "kittens::group<8>::sync(4);",
            // Storer: TMA store back to act_ptrs[OUT_ACT_SLOT=1].
            "kittens::tma::store_async(g.act_ptrs[1], ",
            "kittens::tma::store_async_wait();",
            "kittens::arrive(ss.page_consumed[0]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected source to contain {needle:?}, source was:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 2 — Add: in-place residual fold. Verifies the
    /// emitted .cu carries the load/add/store sequence with values
    /// pulled from the IR's typed getters.
    #[test]
    fn lower_add_to_cuda_smoke() {
        let mut b = BuilderD::new();
        b.push_add(
            ArrivesCount::<0>::new(),
            PageId::<2, 8>::new(), // delta page
            PageId::<3, 8>::new(), // residual page
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<1>::new(),
            ActSlotConst::<5, { u32::MAX }>::new(), // delta_act_slot
            ActSlotConst::<6, { u32::MAX }>::new(), // residual_act_slot
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_add_only", &tape);
        assert!(
            cu.skipped_variants.is_empty(),
            "expected zero skipped variants, got {:?}",
            cu.skipped_variants
        );
        for needle in [
            // Loader TMA-loads delta (page 2, slot 5) + residual (page 3, slot 6).
            "kittens::tma::expect_bytes(ss.page_ready[2], 4096);",
            "kittens::tma::expect_bytes(ss.page_ready[3], 4096);",
            "g.act_ptrs[5]",
            "g.act_ptrs[6]",
            // Consumer load/add/store with NCW=8, K_PER_WARP=256.
            "kittens::rv_fl<256> __add_delta_rv;",
            "kittens::rv_fl<256> __add_res_rv;",
            "kittens::warp::add(__add_res_rv, __add_res_rv, __add_delta_rv);",
            "kittens::warp::sync();",
            // Storer TMA-stores residual page (3) back to slot 6.
            "kittens::tma::store_async(g.act_ptrs[6], (*reinterpret_cast<kittens::sv_bf<2048>*>(ss.pages[3]))",
            "kittens::arrive(ss.page_consumed[3]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected source to contain {needle:?}, source was:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 3 — ScalarMul: in→out scale. Verifies the consumer
    /// body splices `kittens::warp::mul(rv, rv, <scale>)` with the
    /// IR's scale value and no cross-warp barrier (each warp scales
    /// its own slice independently).
    ///
    /// `in_page != out_page` here only because `MegaTapeBuilder`'s
    /// runtime [`PagePool`] doesn't currently model in-place ops
    /// (it rejects `take(IN)` immediately followed by `take(OUT)`
    /// when `IN == OUT`). The variant's IR const-asserts DO allow
    /// `IN_ID == OUT_ID` (gemma2's post-attn `* hidden` uses it),
    /// so the in-place path through `emit_scalar_mul` is exercised
    /// by the proc-macro at user-build time when the tape gets
    /// constructed — the unit test just covers the in→out path.
    #[test]
    fn lower_scalar_mul_to_cuda_smoke() {
        let mut b = BuilderD::new();
        b.push_scalar_mul(
            ArrivesCount::<0>::new(),
            PageId::<4, 8>::new(), // in
            PageId::<5, 8>::new(), // out (distinct)
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<1>::new(),
            ActSlotConst::<2, { u32::MAX }>::new(),
            ActSlotConst::<3, { u32::MAX }>::new(),
            0.5_f32,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_smul", &tape);
        assert!(
            cu.skipped_variants.is_empty(),
            "expected zero skipped variants, got {:?}",
            cu.skipped_variants
        );
        for needle in [
            "kittens::tma::expect_bytes(ss.page_ready[4], 4096);",
            "g.act_ptrs[2]",
            "kittens::rv_fl<256> __smul_rv;",
            "kittens::warp::mul(__smul_rv, __smul_rv, 5e-1f);",
            // Consumer: page_done on out_page (5), page_consumed
            // on in_page (4) since in != out.
            "kittens::arrive(ss.page_done[5]);",
            "kittens::arrive(ss.page_consumed[4]);",
            // Storer: TMA-store out_page (5) to out_act_slot (3).
            "kittens::tma::store_async(g.act_ptrs[3], (*reinterpret_cast<kittens::sv_bf<2048>*>(ss.pages[5]))",
            "kittens::arrive(ss.page_consumed[5]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected source to contain {needle:?}, source was:\n{}",
                cu.source
            );
        }
    }

    /// Sprint 4 — TanhSoftCap: per-row Gemma2 logit softcap.
    /// Verifies the consumer body splices
    /// `ferrite::tk::tanh_softcap_vec(rv, <cap>)` and matches the
    /// ScalarMul TMA-load/store skeleton.
    #[test]
    fn lower_tanh_soft_cap_to_cuda_smoke() {
        let mut b = BuilderD::new();
        b.push_tanh_soft_cap(
            ArrivesCount::<0>::new(),
            PageId::<6, 8>::new(), // in
            PageId::<7, 8>::new(), // out
            MbarrierPhase::<0>::new(),
            MbarrierPhase::<1>::new(),
            HiddenDim::<2048>::new(),
            NumTokensConst::<1>::new(),
            ActSlotConst::<2, { u32::MAX }>::new(),
            ActSlotConst::<3, { u32::MAX }>::new(),
            30.0_f32,
        );
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_softcap", &tape);
        assert!(
            cu.skipped_variants.is_empty(),
            "expected zero skipped variants, got {:?}",
            cu.skipped_variants
        );
        for needle in [
            "kittens::tma::expect_bytes(ss.page_ready[6], 4096);",
            "kittens::rv_fl<256> __tanh_rv;",
            "ferrite::tk::tanh_softcap_vec(__tanh_rv, 3e1f);",
            "kittens::tma::store_async(g.act_ptrs[3], (*reinterpret_cast<kittens::sv_bf<2048>*>(ss.pages[7]))",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected source to contain {needle:?}, source was:\n{}",
                cu.source
            );
        }
    }

    /// A tape with only a SKIPPED variant produces a `.cu` that
    /// still has the full substrate scaffold but reports the
    /// skipped variant in diagnostics + as a `// SKIPPED` comment.
    #[test]
    fn skipped_variant_renders_marker() {
        // Use a barrier op (Sprint 1: SKIPPED).
        use crate::substrate::EdgeId;
        let mut b = BuilderD::new();
        b.push_barrier_signal::<0>(EdgeId::<0, 4>::new());
        let tape = b.finish(16);
        let cu = lower_to_cuda("test_skip", &tape);
        assert_eq!(cu.skipped_variants, vec!["BarrierSignal".to_string()]);
        assert!(cu.source.contains("// SKIPPED node[0]: BarrierSignal"));
    }
}
