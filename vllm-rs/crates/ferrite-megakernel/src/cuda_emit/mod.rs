// SPDX-License-Identifier: Apache-2.0
//! `cuda_emit` — pure literal transcription of a typed [`MegaTape`]
//! into a complete `.cu` translation unit, calling **TK 2.0
//! primitives only** (`third_party/thunderkittens/include/`).
//!
//! See `MEGA_IR_PLAN.md` §0 / §8.0 / §8.0a / §8.0b +
//! `CUDA_EMIT_TK20_AUDIT.md` at the worktree root for the contract.
//! The DOD for any variant shipping is **the emitted `.cu` compiles
//! against TK 2.0 on the pod** — Rust unit tests are not the ground
//! truth.
//!
//! Module layout:
//!
//! - [`cu`] — emitted-CUDA AST: [`CuExpr`] / [`CuStmt`] /
//!   [`CuBlock`] / [`CuVariant`].
//! - [`handles`] — typed handles ([`Sv`], [`Rv`], [`Semaphore`],
//!   [`GmemPtrRaw`], [`ScratchPtr`]) and ferrite substrate
//!   accessors. `Sv<T, LEN>` / `Rv<T, LEN>` / `St<T, ROWS, COLS>` /
//!   `Rt<T, L, ROWS, COLS>` are const-generic in shape per
//!   [[feedback-end-to-end-compile-time-proofs]].
//! - [`tk20`] — Rust API surface mirroring TK 2.0's C++ primitive
//!   surface. Each fn cites its TK 2.0 source line.
//! - [`render`] — per-`MegaNode` const-generic render functions.
//!   Each `render_<variant><const ...>(...)` returns a [`RoleBodies`]
//!   (loader/launcher/consumer/storer); the proc-macro emits literal
//!   `render_*::<...>(...)` calls into `emit_for_canonical_<canonical>`.
//!
//! There is no runtime `lower_to_cuda(&MegaTape)` walker — per
//! `MEGA_IR_PLAN.md` §8.0b "Implementing end-to-end const generics:
//! proc-macro-time dispatch", emit dispatch happens at proc-macro
//! time so const generics flow through to `tk20::*` and `handles::*`
//! as Rust compile-time proofs. The runtime walker would erase those
//! proofs to bare `u32`.
//!
//! [`Sv`]: handles::Sv
//! [`Rv`]: handles::Rv
//! [`Semaphore`]: handles::Semaphore
//! [`GmemPtrRaw`]: handles::GmemPtrRaw
//! [`ScratchPtr`]: handles::ScratchPtr

pub mod cu;
pub mod handles;
pub mod render;
pub mod tk20;

use crate::ir::tape::TapeBudget;

pub use cu::{CuBlock, CuExpr, CuStmt, CuVariant};
pub use render::RoleBodies;

/// Registry entry for one canonical's `.cu` emit fn. The proc-macro
/// emits an `::ferrite_forward::inventory::submit!` block per
/// successfully-rendered canonical (the same set that gets a
/// `pub fn emit_for_canonical_<canonical>() -> CuVariant`),
/// pointing `emit_fn` at that fn.
///
/// `ferrite-cuda-builder/build.rs` walks
/// `::ferrite_forward::inventory::iter::<MegaCanonicalEmit>()`
/// before the existing megakernel `.cu` discovery pass, calls each
/// `emit_fn` to get the rendered source, and writes
/// `<cudaforge cache>/megakernels/ferrite_<canonical>.cu` so the
/// per-canonical `.cu` shows up alongside the existing per-op
/// `.cuh` files for nvcc to compile into `libmegakernels.a`.
///
/// Wires the proc-macro's compile-time-substrate-proof-bearing
/// `emit_for_canonical_*` fn into the production build pipeline
/// per `MEGA_IR_PLAN.md` "Phase C step 2 (emit `.cu` per
/// build_mega_tape fn)" — without forking the const-generic emit
/// chain into a runtime walker (which would break
/// [[feedback-end-to-end-compile-time-proofs]]).
///
/// Uses the `inventory` crate directly (not via the
/// `cuda`-feature-gated `::ferrite_forward::inventory` re-export)
/// so this registration is unconditionally present. Side-stepping
/// the cuda gate avoids dragging `ferrite-forward/cuda` (and its
/// transitive `ferrite-kernels/cuda` CUDA-link deps) into the
/// proc-macro host tree, which would break the `.so` link.
pub struct MegaCanonicalEmit {
    /// Canonical name slug, e.g. `"llama_3_2_1b_m_1_sk_128"`. Used
    /// as the `.cu` file stem (`ferrite_<canonical>.cu`).
    pub canonical: &'static str,
    /// Pointer to the proc-macro-emitted
    /// `pub fn emit_for_canonical_<canonical>() -> CuVariant` —
    /// see [`crate::codegen`] dispatch.
    pub emit_fn: fn() -> CuVariant,
}

::inventory::collect!(MegaCanonicalEmit);

/// Which host-side launch ABI tier the emitted megakernel exposes.
///
/// Mirror of `ferrite_forward::interpreter::mega::LaunchTier`. The
/// proc-macro picks the tier per canonical (which variants the tape
/// contains) and passes it to [`render_canonical`] so the kernel
/// signature matches the host-side `LaunchArgs*` struct of the same
/// tier per `MEGA_IR_PLAN.md` §8.0a item 2 ("the kernel signature
/// you emit must match what the host already constructs and passes").
///
/// - `Base` — neither paged-KV nor rotary metadata.
/// - `Qkv` — at least one `FusedQkvRopeCache`.
/// - `Attn` — additionally at least one `AttentionViaCache`.
///
/// Nested: Attn ⊃ Qkv ⊃ Base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchTier {
    Base,
    Qkv,
    Attn,
}

/// Aggregate per-MegaNode [`RoleBodies`] into the full `.cu` source.
/// Called from the proc-macro-emitted `emit_for_canonical_<canonical>`
/// after every `render_*::<...>(...)` call. Substrate scaffolding
/// (includes, per-canonical Config struct, kernel entry signature)
/// wraps the four concatenated role blocks.
pub fn render_canonical(
    canonical: &str,
    budget: &TapeBudget,
    tier: LaunchTier,
    role_bodies: &[RoleBodies],
) -> CuVariant {
    let mut loader_block = CuBlock::new();
    let mut launcher_block = CuBlock::new();
    let mut consumer_block = CuBlock::new();
    let mut storer_block = CuBlock::new();
    let mut skipped: Vec<String> = Vec::new();

    // Wrap each node's role bodies in their own `{ ... }` C++ block
    // so per-op local declarations (`__gemm_acc`, `__rms_act_rv`,
    // `__farn_delta_rv`, `__attn_q_rt`, ...) don't collide across
    // nodes when multiple instances of the same op type appear in
    // the canonical (e.g. one `Gemm` per of {q, k, v, o, gate, up,
    // down} × NUM_LAYERS = 7 × 16 in llama). Without scoping, nvcc
    // rejects with `"<name>" has already been declared in the
    // current scope`. Block scoping is the cheapest fix and matches
    // C++ idioms — a fresh `{}` per op makes its locals local.
    for (idx, bodies) in role_bodies.iter().enumerate() {
        if let Some(variant) = bodies.skipped {
            skipped.push(variant.to_string());
            let marker = CuStmt::new(format!("// SKIPPED node[{idx}]: {variant}"));
            loader_block.push(marker.clone());
            launcher_block.push(marker.clone());
            consumer_block.push(marker.clone());
            storer_block.push(marker);
            continue;
        }
        let marker_open = CuStmt::new(format!("// node[{idx}] {{"));
        let marker_close = CuStmt::new("// }".to_string());
        let open = CuStmt::new("{".to_string());
        let close = CuStmt::new("}".to_string());
        for (block, body) in [
            (&mut loader_block, &bodies.loader),
            (&mut launcher_block, &bodies.launcher),
            (&mut consumer_block, &bodies.consumer),
            (&mut storer_block, &bodies.storer),
        ] {
            if body.is_empty() {
                continue;
            }
            block.push(marker_open.clone());
            block.push(open.clone());
            block.extend(body.clone());
            block.push(close.clone());
            block.push(marker_close.clone());
        }
    }

    let source = render_source(
        canonical,
        budget,
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
    // TK 2.0 default_config (third_party/thunderkittens/prototype/vm/config.cuh).
    const CONSUMER_REGISTERS: u32 = 104;
    const NON_CONSUMER_REGISTERS: u32 = 64;
    const INSTRUCTION_PIPE_STAGES: u32 = 2;

    let kernel_name = format!("ferrite_{canonical}_launch");
    let cfg_name = format!("Config_{canonical}");

    let mut s = String::new();
    s.push_str("// SPDX-License-Identifier: Apache-2.0\n");
    s.push_str(&format!(
        "// Auto-generated by ferrite_megakernel::cuda_emit for canonical \"{canonical}\".\n"
    ));
    s.push_str("// DO NOT EDIT — regenerate by re-running the proc-macro.\n\n");

    s.push_str("#include \"ferrite_substrate.cuh\"\n");
    s.push_str("#include \"ferrite_globals.cuh\"\n");
    s.push_str("#include \"ferrite_warp_roles.cuh\"\n");
    s.push_str("#include \"ferrite_barrier.cuh\"\n\n");

    s.push_str("namespace {\n\n");

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

    // Host wrapper: triple-bracket launch with grid/block/shmem the
    // Rust dispatcher (interpreter::mega::launch{,_qkv,_attn}) can
    // call directly via `extern "C" fn`. The `__global__` symbol
    // alone is not host-callable from plain C, so emit a paired
    // `__host__ extern "C"` wrapper. Positional args mirror
    // `LaunchFn{,Qkv,Attn}` in
    // `ferrite-forward/src/interpreter/mega/mod.rs:370-424`
    // (stream is trailing). Returns the runtime error code as
    // `int` (0 == cudaSuccess) — caller treats nonzero as failure.
    let wrapper_name = format!("ferrite_{canonical}_launch_host");
    let block_threads = (budget.num_consumer_warps + 4) * 32;
    let shmem_bytes = budget.num_pages * budget.page_size + budget.scratch_bytes + 1024;
    s.push_str(&format!("extern \"C\" __host__ int {wrapper_name}(\n"));
    s.push_str("    __nv_bfloat16* const*       act_ptrs,\n");
    s.push_str("    const __nv_bfloat16* const* weight_ptrs,\n");
    let kernel_call_args = match tier {
        LaunchTier::Base => {
            s.push_str("    int32_t*                    barrier_slots,\n");
            s.push_str("    int32_t                     trace_level,\n");
            s.push_str("    const uint32_t*             input_ids,\n");
            s.push_str("    void*                       stream\n");
            "act_ptrs, weight_ptrs, barrier_slots, trace_level, input_ids"
        }
        LaunchTier::Qkv => {
            s.push_str("    const uint32_t*             input_ids,\n");
            s.push_str("    const uint32_t*             positions,\n");
            s.push_str("    const int64_t*              slot_mapping,\n");
            s.push_str("    __nv_bfloat16* const*       key_cache_ptrs,\n");
            s.push_str("    __nv_bfloat16* const*       value_cache_ptrs,\n");
            s.push_str("    int32_t*                    barrier_slots,\n");
            s.push_str("    int32_t                     trace_level,\n");
            s.push_str("    void*                       stream\n");
            "act_ptrs, weight_ptrs, input_ids, positions, slot_mapping, key_cache_ptrs, \
             value_cache_ptrs, barrier_slots, trace_level"
        }
        LaunchTier::Attn => {
            s.push_str("    const uint32_t*             input_ids,\n");
            s.push_str("    const uint32_t*             positions,\n");
            s.push_str("    const int64_t*              slot_mapping,\n");
            s.push_str("    __nv_bfloat16* const*       key_cache_ptrs,\n");
            s.push_str("    __nv_bfloat16* const*       value_cache_ptrs,\n");
            s.push_str("    const int32_t*              seq_lens,\n");
            s.push_str("    const uint32_t*             block_table,\n");
            s.push_str("    uint32_t                    block_table_stride,\n");
            s.push_str("    int32_t*                    barrier_slots,\n");
            s.push_str("    int32_t                     trace_level,\n");
            s.push_str("    void*                       stream\n");
            "act_ptrs, weight_ptrs, input_ids, positions, slot_mapping, key_cache_ptrs, \
             value_cache_ptrs, seq_lens, block_table, block_table_stride, barrier_slots, \
             trace_level"
        }
    };
    s.push_str(") {\n");
    s.push_str(&format!(
        "    cudaFuncSetAttribute((const void*){kernel_name}, \
         cudaFuncAttributeMaxDynamicSharedMemorySize, {shmem_bytes});\n"
    ));
    s.push_str(&format!(
        "    {kernel_name}<<<dim3(1,1,1), dim3({block_threads},1,1), {shmem_bytes}, \
         (cudaStream_t)stream>>>({kernel_call_args});\n"
    ));
    s.push_str("    return (int)cudaGetLastError();\n");
    s.push_str("}\n\n");

    s.push_str("} // namespace\n");
    s
}

#[cfg(test)]
mod tests {
    //! Per-variant render smoke tests. Each test calls
    //! `render_<variant>::<const-generic-args>(/* runtime args */)`
    //! directly (mirroring what the proc-macro emits at user-build
    //! time) and asserts on substring fragments of the resulting
    //! `.cu`. The DOD remains nvcc-on-pod compilation; these are
    //! Rust-side smoke tests.

    use super::render::*;
    use super::*;
    use crate::ir::nodes::{GateUpActivation, LmHeadNormKind};

    fn budget_d() -> TapeBudget {
        TapeBudget {
            num_pages: 8,
            num_consumer_warps: 8,
            page_size: 32_768,
            scratch_bytes: 32_768,
            num_edges: 4,
            num_layers: 16,
        }
    }

    fn budget_lg() -> TapeBudget {
        TapeBudget {
            num_pages: 8,
            num_consumer_warps: 8,
            page_size: 32_768,
            scratch_bytes: 65_536,
            num_edges: 4,
            num_layers: 16,
        }
    }

    /// Sprint 10: Gemm — `D = A * B + C` (C zero) under AlongN
    /// warp split. M=16, K=32, N=256, NCW=8, TILE_N=32.
    #[test]
    fn gemm_emits_tk20_calls() {
        let bodies = vec![render_gemm::<16, 32, 256, 32, 8, 16, 1>(
            /* in_page_id */ 0,
            /* weight_page_id */ 1,
            /* out_page_id */ 2,
            /* consumer_phase */ 0,
            /* storer_phase */ 1,
            /* layer */ 3,
            /* in_act_slot */ 0,
            /* out_act_slot */ 2,
            /* weight_accessor */ 5,
            /* bar_publish */ 7,
            /* b_tile_offset */ 0,
        )];
        let cu = render_canonical("test_gemm", &budget_d(), LaunchTier::Base, &bodies);
        assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
        std::fs::write("/tmp/gemm_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 1024);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 16384);",
            "kittens::group<1>::tma::load_async(",
            "(*reinterpret_cast<kittens::st_bf<16, 32>*>(ss.pages[0]))",
            "(*reinterpret_cast<kittens::st_bf<32, 256>*>(ss.scratch + 0))",
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

    /// Sprint 11: TkFusedGemmAdd — `residual += A * B`.
    #[test]
    fn tk_fused_gemm_add_emits_tk20_calls() {
        let bodies = vec![render_tk_fused_gemm_add::<16, 32, 256, 32, 8, 16, 1>(
            /* in_page */ 0,
            /* weight_page */ 1,
            /* residual_page */ 2,
            0,
            1,
            3,
            0,
            2,
            5,
            7,
            0,
        )];
        let cu = render_canonical("test_tk_fused_gemm_add", &budget_d(), LaunchTier::Base, &bodies);
        assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
        std::fs::write("/tmp/tk_fused_gemm_add_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 1024);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 16384);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[2], 8192);",
            "(*reinterpret_cast<kittens::st_bf<16, 32>*>(ss.pages[0]))",
            "(*reinterpret_cast<kittens::st_bf<32, 256>*>(ss.scratch + 0))",
            "(*reinterpret_cast<kittens::st_bf<16, 256>*>(ss.pages[2]))",
            "kittens::rt_bf<16, 32> __gemm_a;",
            "kittens::rt_bf<32, 32, kittens::ducks::rt_layout::col> __gemm_b;",
            "kittens::rt_fl<16, 32> __gemm_acc;",
            "auto __gemm_b_sub = ",
            "auto __gemm_resid_sub = ",
            ".template subtile<16, 32>(int2{0, static_cast<int>(kittens::warpid())})",
            "kittens::warp::load(__gemm_a, ",
            "kittens::warp::load(__gemm_b, __gemm_b_sub);",
            "kittens::warp::load(__gemm_acc, __gemm_resid_sub);",
            "kittens::warp::mma_AB(__gemm_acc, __gemm_a, __gemm_b, __gemm_acc);",
            "kittens::warp::store(__gemm_resid_sub, __gemm_acc);",
            "kittens::group<8>::sync(7);",
            "kittens::group<1>::arrive(ss.page_done[2]);",
            "kittens::group<1>::arrive(ss.page_consumed[0]);",
            "kittens::group<1>::arrive(ss.page_consumed[1]);",
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

        assert!(
            !cu.source.contains("__gemm_out_sub"),
            "TkFusedGemmAdd must not declare an out-subtile; got:\n{}",
            cu.source
        );
        assert!(
            !cu.source.contains("kittens::warp::zero(__gemm_acc)"),
            "TkFusedGemmAdd must not zero the accumulator; got:\n{}",
            cu.source
        );
    }

    /// Sprint 12: FusedGateUpActivateMul / Silu.
    #[test]
    fn fused_gate_up_silu_emits_tk20_calls() {
        let bodies = vec![render_fused_gate_up_activate_mul::<16, 32, 256, 32, 8, 16, 1>(
            0, 1, 2, 0, 1, 3, 0, 2, 5, 7,
            /* gate_offset */ 0,
            /* up_offset */ 16_384,
            /* gate_bytes */ 16_384,
            /* up_bytes */ 16_384,
            GateUpActivation::Silu,
        )];
        let cu = render_canonical(
            "test_fused_gate_up_silu",
            &budget_d(),
            LaunchTier::Base,
            &bodies,
        );
        assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
        std::fs::write("/tmp/fused_gate_up_silu_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 1024);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 32768);",
            "(*reinterpret_cast<kittens::st_bf<32, 256>*>(ss.scratch + 0))",
            "(*reinterpret_cast<kittens::st_bf<32, 256>*>(ss.scratch + 16384))",
            "+ 8192)",
            "kittens::rt_bf<16, 32> __gu_a;",
            "kittens::rt_bf<32, 32, kittens::ducks::rt_layout::col> __gu_gate_b;",
            "kittens::rt_bf<32, 32, kittens::ducks::rt_layout::col> __gu_up_b;",
            "kittens::rt_fl<16, 32> __gu_gate_acc;",
            "kittens::rt_fl<16, 32> __gu_up_acc;",
            "auto __gu_gate_b_sub = ",
            "auto __gu_up_b_sub = ",
            "auto __gu_out_sub = ",
            "kittens::warp::zero(__gu_gate_acc);",
            "kittens::warp::zero(__gu_up_acc);",
            "kittens::warp::mma_AB(__gu_gate_acc, __gu_a, __gu_gate_b, __gu_gate_acc);",
            "kittens::warp::mma_AB(__gu_up_acc, __gu_a, __gu_up_b, __gu_up_acc);",
            "x * (1.0f / (1.0f + __expf(-x)))",
            "kittens::warp::apply(__gu_gate_acc, __gu_gate_acc,",
            "kittens::warp::mul(__gu_gate_acc, __gu_gate_acc, __gu_up_acc);",
            "kittens::warp::store(__gu_out_sub, __gu_gate_acc);",
            "kittens::group<8>::sync(7);",
            "kittens::group<1>::arrive(ss.page_done[2]);",
            "kittens::group<1>::arrive(ss.page_consumed[0]);",
            "kittens::group<1>::arrive(ss.page_consumed[1]);",
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

        assert!(
            !cu.source.contains("tanhf(0.7978845608028654f"),
            "Silu kernel must not emit gelu tanhf; got:\n{}",
            cu.source
        );
    }

    /// Sprint 12: FusedGateUpActivateMul / Gelu.
    #[test]
    fn fused_gate_up_gelu_emits_tk20_calls() {
        let bodies = vec![render_fused_gate_up_activate_mul::<16, 32, 256, 32, 8, 16, 1>(
            0, 1, 2, 0, 1, 3, 0, 2, 5, 7, 0, 16_384, 16_384, 16_384,
            GateUpActivation::Gelu,
        )];
        let cu = render_canonical(
            "test_fused_gate_up_gelu",
            &budget_d(),
            LaunchTier::Base,
            &bodies,
        );
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/fused_gate_up_gelu_emit.cu", &cu.source).ok();
        assert!(cu.source.contains("tanhf(0.7978845608028654f"));
        assert!(!cu.source.contains("(1.0f / (1.0f + __expf(-x)))"));
    }

    /// Sprint 1: RmsNorm. HIDDEN_DIM=2048, NUM_TOKENS=1, NCW=8 →
    /// K_PER_WARP=256.
    #[test]
    fn rms_norm_emits_tk20_calls() {
        let bodies = vec![render_rms_norm::<2048, 1, 8, 256, 16>(
            /* in_page */ 0,
            /* weight_page */ 1,
            /* consumer_phase */ 0,
            /* storer_phase */ 1,
            /* layer */ 0,
            /* in_act_slot */ 0,
            /* out_act_slot */ 1,
            /* weight_accessor */ 7,
            /* bar_reduce */ 3,
            /* bar_publish */ 4,
            /* partial_offset */ 0,
            /* eps */ 1.0e-5_f32,
        )];
        let cu = render_canonical("test_rms", &budget_d(), LaunchTier::Base, &bodies);
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/rms_norm_emit.cu", &cu.source).ok();

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

        for forbidden in [
            "kittens::wait(",
            "kittens::arrive(",
            "kittens::tma::load_async(",
            "kittens::tma::expect_bytes(",
            "kittens::warpid() * ",
            "* sizeof(__nv_bfloat16))",
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

    /// Sprint 8: Embed — per-token TMA gather.
    #[test]
    fn embed_emits_per_token_gather() {
        let bodies = vec![render_embed::<2048, 1, 16>(
            /* out_page */ 0,
            /* consumer_phase */ 0,
            /* storer_phase */ 1,
            /* out_act_slot */ 7,
            /* weight_accessor */ 11,
        )];
        let cu = render_canonical("test_embed", &budget_d(), LaunchTier::Base, &bodies);
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

    /// Sprint 7: BarrierSignal.
    #[test]
    fn barrier_signal_emits_ferrite_call() {
        let bodies = vec![render_barrier_signal(2)];
        let cu = render_canonical("test_bsig", &budget_d(), LaunchTier::Base, &bodies);
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

    /// Sprint 7: BarrierWait.
    #[test]
    fn barrier_wait_emits_ferrite_call() {
        let bodies = vec![render_barrier_wait(1, 16)];
        let cu = render_canonical("test_bwait", &budget_d(), LaunchTier::Base, &bodies);
        assert!(cu.skipped_variants.is_empty());
        std::fs::write("/tmp/barrier_wait_emit.cu", &cu.source).ok();
        assert!(
            cu.source
                .contains("ferrite::barrier_wait(&barrier_slots[1], 16);"),
            "got:\n{}",
            cu.source
        );
    }

    /// Sprint 6: ScalarOffsetRmsNorm.
    #[test]
    fn scalar_offset_rms_norm_emits_tk20_calls() {
        let bodies = vec![render_scalar_offset_rms_norm::<2048, 1, 8, 256, 16>(
            /* in_page */ 0,
            /* weight_page */ 1,
            /* consumer_phase */ 0,
            /* storer_phase */ 1,
            /* layer */ 5,
            /* in_act_slot */ 0,
            /* out_act_slot */ 1,
            /* weight_accessor */ 3,
            /* bar_reduce */ 1,
            /* bar_publish */ 2,
            /* partial_offset */ 0,
            /* eps */ 1.0e-5_f32,
            /* offset */ 1.0_f32,
        )];
        let cu = render_canonical("test_sors", &budget_d(), LaunchTier::Base, &bodies);
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

    /// Sprint 5: FusedAddRmsNorm.
    #[test]
    fn fused_add_rms_norm_emits_tk20_calls() {
        let bodies = vec![render_fused_add_rms_norm::<2048, 1, 8, 256, 16>(
            /* delta_page */ 0,
            /* residual_page */ 1,
            /* weight_page */ 2,
            /* consumer_phase */ 0,
            /* storer_phase */ 1,
            /* layer */ 3,
            /* delta_act_slot */ 0,
            /* residual_act_slot */ 1,
            /* weight_accessor */ 5,
            /* bar_reduce */ 1,
            /* bar_publish */ 2,
            /* partial_offset */ 0,
            /* eps */ 1.0e-5_f32,
        )];
        let cu = render_canonical("test_farn", &budget_d(), LaunchTier::Base, &bodies);
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

    /// Sprint 4: TanhSoftCap via warp::apply lambda.
    #[test]
    fn tanh_soft_cap_emits_tk20_apply_lambda() {
        let bodies = vec![render_tanh_soft_cap::<2048, 1, 8, 256>(
            /* in_page */ 6,
            /* out_page */ 7,
            0, 1, 2, 3, 2, 30.0_f32,
        )];
        let cu = render_canonical("test_softcap", &budget_d(), LaunchTier::Base, &bodies);
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

    /// Sprint 3: ScalarMul.
    #[test]
    fn scalar_mul_emits_tk20_calls() {
        let bodies = vec![render_scalar_mul::<2048, 1, 8, 256>(
            /* in_page */ 4,
            /* out_page */ 5,
            0, 1, 2, 3, 2, 0.5_f32,
        )];
        let cu = render_canonical("test_smul", &budget_d(), LaunchTier::Base, &bodies);
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

    /// Sprint 2: Add.
    #[test]
    fn add_emits_tk20_calls() {
        let bodies = vec![render_add::<2048, 1, 8, 256>(
            /* delta_page */ 2,
            /* residual_page */ 3,
            0, 1, 5, 6, 2,
        )];
        let cu = render_canonical("test_add", &budget_d(), LaunchTier::Base, &bodies);
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

    /// Sprint 13 no-delta: TkFusedNormGemm RmsNorm flavor.
    #[test]
    fn tk_fused_norm_gemm_no_delta_emits_tk20_calls() {
        let bodies = vec![render_tk_fused_norm_gemm::<16, 128, 128, 16, 8, 16, 16, 1>(
            /* in_page */ 0,
            /* delta_page */ None,
            /* norm_weight_page */ 1,
            /* linear_weight_page */ 2,
            /* out_page */ 3,
            /* consumer_phase */ 0,
            /* storer_phase */ 1,
            /* layer */ 3,
            /* in_act_slot */ 0,
            /* delta_act_slot */ None,
            /* out_act_slot */ 3,
            /* norm_weight_accessor */ 5,
            /* linear_weight_accessor */ 6,
            /* bar_reduce */ 1,
            /* bar_publish */ 2,
            /* eps */ 1.0e-5_f32,
            LmHeadNormKind::RmsNorm,
            /* offset_opt */ None,
            /* b_tile_offset */ 32,
            /* partial_offset */ 0,
        )];
        let cu = render_canonical(
            "test_tk_fused_norm_gemm_no_delta",
            &budget_lg(),
            LaunchTier::Base,
            &bodies,
        );
        assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
        std::fs::write("/tmp/tk_fused_norm_gemm_no_delta_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 4096);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 256);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[2], 32768);",
            "(*reinterpret_cast<kittens::sv_bf<128>*>(ss.pages[0]))",
            "(*reinterpret_cast<kittens::sv_bf<128>*>(ss.pages[1]))",
            "(*reinterpret_cast<kittens::st_bf<128, 128>*>(ss.scratch + 32))",
            "kittens::rv_fl<16> __lmh_act_rv;",
            "kittens::rv_fl<16> __lmh_sq_rv;",
            "kittens::rv_fl<16> __lmh_norm_w_rv;",
            "kittens::group<8>::load(__lmh_act_rv,",
            "kittens::warp::sum(__lmh_partial_sum, __lmh_sq_rv);",
            "kittens::group<8>::sync(1);",
            "rsqrtf(__lmh_full_sum / 128.0f + 1e-5f)",
            "kittens::warp::mul(__lmh_act_rv, __lmh_act_rv, __lmh_scale);",
            "kittens::warp::mul(__lmh_act_rv, __lmh_act_rv, __lmh_norm_w_rv);",
            "kittens::group<8>::store(",
            "kittens::group<8>::sync(2);",
            "kittens::rt_bf<16, 128> __lmh_a;",
            "kittens::rt_bf<128, 16, kittens::ducks::rt_layout::col> __lmh_b;",
            "kittens::rt_fl<16, 16> __lmh_acc;",
            "auto __lmh_b_sub = ",
            "auto __lmh_out_sub = ",
            "kittens::warp::zero(__lmh_acc);",
            "kittens::warp::mma_AB(__lmh_acc, __lmh_a, __lmh_b, __lmh_acc);",
            "kittens::warp::store(__lmh_out_sub, __lmh_acc);",
            "kittens::group<1>::arrive(ss.page_done[3]);",
            "kittens::group<1>::arrive(ss.page_consumed[0]);",
            "kittens::group<1>::arrive(ss.page_consumed[1]);",
            "kittens::group<1>::arrive(ss.page_consumed[2]);",
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

        assert!(
            !cu.source.contains("__lmh_delta_rv"),
            "no-delta variant must not declare a delta rv; got:\n{}",
            cu.source
        );
        assert!(
            !cu.source.contains("__lmh_mean_partial"),
            "RmsNorm flavor must not emit a mean-subtract; got:\n{}",
            cu.source
        );
        assert!(
            !cu.source.contains("kittens::warp::add(__lmh_norm_w_rv,"),
            "RmsNorm with no offset must not add to norm_weight; got:\n{}",
            cu.source
        );
    }

    /// Sprint 13 with-delta: TkFusedNormGemm AddScalarOffsetRmsNorm.
    #[test]
    fn tk_fused_norm_gemm_with_delta_emits_tk20_calls() {
        let bodies = vec![render_tk_fused_norm_gemm::<16, 128, 128, 16, 8, 16, 16, 1>(
            /* in_page */ 0,
            /* delta_page */ Some(1),
            /* norm_weight_page */ 2,
            /* linear_weight_page */ 3,
            /* out_page */ 4,
            /* consumer_phase */ 0,
            /* storer_phase */ 1,
            /* layer */ 3,
            /* in_act_slot */ 0,
            /* delta_act_slot */ Some(1),
            /* out_act_slot */ 4,
            /* norm_weight_accessor */ 5,
            /* linear_weight_accessor */ 6,
            /* bar_reduce */ 1,
            /* bar_publish */ 2,
            /* eps */ 1.0e-5_f32,
            LmHeadNormKind::AddScalarOffsetRmsNorm,
            /* offset_opt */ Some(1.5_f32),
            /* b_tile_offset */ 32,
            /* partial_offset */ 0,
        )];
        let cu = render_canonical(
            "test_tk_fused_norm_gemm_with_delta",
            &budget_lg(),
            LaunchTier::Base,
            &bodies,
        );
        assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
        std::fs::write("/tmp/tk_fused_norm_gemm_with_delta_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 4096);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 4096);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[2], 256);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[3], 32768);",
            "(*reinterpret_cast<kittens::sv_bf<128>*>(ss.pages[1]))",
            "kittens::rv_fl<16> __lmh_delta_rv;",
            "kittens::warp::add(__lmh_act_rv, __lmh_act_rv, __lmh_delta_rv);",
            "kittens::warp::add(__lmh_norm_w_rv, __lmh_norm_w_rv, 1.5e0f);",
            "kittens::warp::mma_AB(__lmh_acc, __lmh_a, __lmh_b, __lmh_acc);",
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

    /// Sprint 14: SpliceMmEmbeds passthrough.
    #[test]
    fn splice_mm_embeds_emits_passthrough() {
        let bodies = vec![render_splice_mm_embeds(3, 0, 1)];
        let cu = render_canonical("test_splice", &budget_d(), LaunchTier::Base, &bodies);
        assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
        std::fs::write("/tmp/splice_mm_embeds_emit.cu", &cu.source).ok();

        for needle in [
            "kittens::group<1>::wait(ss.page_consumed[3], 1);",
            "kittens::group<1>::arrive(ss.page_ready[3]);",
            "kittens::group<1>::wait(ss.page_ready[3], 0);",
            "if (kittens::warpid() == 0) {",
            "kittens::group<1>::arrive(ss.page_done[3]);",
            "kittens::group<1>::wait(ss.page_done[3], 1);",
            "kittens::group<1>::arrive(ss.page_consumed[3]);",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?} in source, got:\n{}",
                cu.source
            );
        }

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

    /// Sprint 15: FusedQkvRopeCache. M=16, HIDDEN_DIM=64, HEAD_DIM=16,
    /// NUM_Q_HEADS=4, NUM_KV_HEADS=2, q_dim=64, kv_dim=32, qkv_n=128,
    /// NCW=8, TILE_N=16, heads_per_warp=1.
    #[test]
    fn fused_qkv_rope_cache_emits_tk20_calls() {
        let bodies = vec![render_fused_qkv_rope_cache::<
            16, 64, 16, 4, 2, 64, 32, 128, 16, 1, 8, 16, 1,
        >(
            /* in_page */ 0,
            /* qkv_weight_page */ 1,
            /* cos_sin_page */ 2,
            /* q_out_page */ 3,
            /* k_out_page */ 4,
            /* v_out_page */ 5,
            /* consumer_phase */ 0,
            /* storer_phase */ 1,
            /* layer */ 2,
            /* in_act_slot */ 0,
            /* q_out_act_slot */ 3,
            /* k_out_act_slot */ 4,
            /* v_out_act_slot */ 5,
            /* qkv_weight_accessor */ 7,
            /* rotary_accessor */ 8,
            /* bar_publish */ 1,
            /* q_rope_offset */ 0,
            /* k_rope_offset */ 512,
            /* qkv_b_tile_offset */ 1024,
        )];
        let cu = render_canonical("test_fqrc", &budget_lg(), LaunchTier::Qkv, &bodies);
        assert!(cu.skipped_variants.is_empty(), "skipped: {:?}", cu.skipped_variants);
        std::fs::write("/tmp/fused_qkv_rope_cache_emit.cu", &cu.source).ok();

        for needle in [
            "const uint32_t*             positions,",
            "const int64_t*              slot_mapping,",
            "__nv_bfloat16* const*       key_cache_ptrs,",
            "__nv_bfloat16* const*       value_cache_ptrs,",
            "(void)slot_mapping;",
            "(void)key_cache_ptrs;",
            "(void)value_cache_ptrs;",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[0], 2048);",
            "kittens::group<1>::tma::expect_bytes(ss.page_ready[1], 16384);",
            "(*reinterpret_cast<kittens::st_bf<16, 64>*>(ss.pages[0]))",
            "(*reinterpret_cast<kittens::st_bf<64, 128>*>(ss.scratch + 1024))",
            "for (int __cs_t = 0; __cs_t < 16; ++__cs_t)",
            "positions[__cs_t]",
            "kittens::rt_bf<16, 64> __qkv_a;",
            "auto& __qkv_q_rope = *reinterpret_cast<kittens::st_bf<16, 64>*>(ss.scratch + 0);",
            "auto& __qkv_k_rope = *reinterpret_cast<kittens::st_bf<16, 32>*>(ss.scratch + 512);",
            "__nv_bfloat16* __qkv_cos_sin_ptr = reinterpret_cast<__nv_bfloat16*>(ss.pages[2]);",
            "for (int __qkv_h = 0; __qkv_h < 1; ++__qkv_h)",
            "kittens::rt_bf<64, 16, kittens::ducks::rt_layout::col> __qkv_b;",
            "kittens::rt_fl<16, 16> __qkv_acc;",
            "kittens::warp::mma_AB(__qkv_acc, __qkv_a, __qkv_b, __qkv_acc);",
            "if (__qkv_col < 64)",
            "} else if (__qkv_col < 64 + 32)",
            "} else {",
            "kittens::warp::apply(__qkv_rot, __qkv_acc,",
            "constexpr int __half = 16 / 2;",
            "__bfloat162float(__qkv_stg_ptr[row * 16 + __pc])",
            "__bfloat162float(__qkv_cos_sin_ptr[row * 16 + __t])",
            "__bfloat162float(__qkv_cos_sin_ptr[row * 16 + __half + __t])",
            "return col < __half ? (x * __c - __paired * __s) : (x * __c + __paired * __s);",
            "kittens::group<8>::sync(1);",
            "kittens::group<1>::arrive(ss.page_done[3]);",
            "kittens::group<1>::arrive(ss.page_done[4]);",
            "kittens::group<1>::arrive(ss.page_done[5]);",
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

    /// AttentionViaCache — full FlashAttention-2 body. Const-generic
    /// shape: M=16 (Q rows, padded from NUM_TOKENS=1 decode case),
    /// HEAD_DIM=64, NUM_Q_HEADS=32, NUM_KV_HEADS=8 (GQA group of 4),
    /// BLOCK_SIZE=16, NCW=8 (4 Q-heads per warp), NUM_LAYERS=16.
    /// Validates the emitted role bodies contain the canonical
    /// FlashAttention-2 building blocks: Q@K^T (mma_ABt), online
    /// softmax (row_max + exp + row_sum), PV (mma_AB), and the
    /// finalizing div_row.
    #[test]
    fn render_attention_via_cache_basic() {
        let bodies = vec![render::render_attention_via_cache::<
            16, 64, 32, 8, 16, 256, 8, 16, 1,
        >(
            /*q_in_page_id=*/ 0,
            /*attn_out_page_id=*/ 1,
            /*consumer_phase=*/ 0,
            /*storer_phase=*/ 1,
            /*layer=*/ 5,
            /*q_in_act_slot=*/ 0,
            /*attn_out_act_slot=*/ 1,
            /*score_offset=*/ 0,
            /*pv_offset=*/ 4096,
            /*k_smem_page_id=*/ 6,
            /*v_smem_page_id=*/ 7,
            /*attn_scale=*/ 0.125_f32,
            /*attn_softcap=*/ 0.0_f32,
            /*interleaved=*/ false,
        )];
        let cu = render_canonical("test_attn", &budget_lg(), LaunchTier::Attn, &bodies);
        assert!(
            cu.skipped_variants.is_empty(),
            "skipped: {:?}",
            cu.skipped_variants
        );
        for needle in [
            // Loader role TMA-loads Q from gmem to q_in_page.
            "ss.pages[0]",
            "kittens::group<1>::tma::load_async",
            "act_ptrs[0]",
            // Consumer role declarations.
            "__shared__ kittens::semaphore __attn_k_arr;",
            "__shared__ kittens::semaphore __attn_v_arr;",
            "kittens::init_semaphore(__attn_k_arr, 0, 1);",
            "ss.pages[6]",         // K_smem
            "ss.pages[7]",         // V_smem
            "static_cast<int>(seq_lens[0])",
            "(__attn_seq_len + 16 - 1) / 16",
            // Per-Q-head outer loop.
            "__attn_qh < 4",
            // Q register tile + load.
            "kittens::rt_bf<16, 64> __attn_q_rt;",
            "__attn_q_tile.template subtile<16, 64>",
            "kittens::warp::load(__attn_q_rt",
            // Running state init — ortho layout so packed dtype
            // (float2) matches `rt<row>::col_vec_layout` for
            // row_max/row_sum/sub_row/mul_row/div_row.
            "kittens::rv_fl<16, kittens::ducks::rv_layout::ortho> __attn_max;",
            "kittens::rv_fl<16, kittens::ducks::rv_layout::ortho> __attn_sum;",
            "kittens::warp::zero(__attn_sum);",
            "kittens::rt_fl<16, 64> __attn_o;",
            "kittens::warp::zero(__attn_o);",
            // Per-block inner loop.
            "__attn_p < static_cast<uint32_t>(__attn_num_blocks)",
            // Paged-KV gather.
            "key_cache_ptrs[5] +",
            "block_table[__attn_p]",
            // QK^T pass.
            "kittens::warp::mma_ABt",
            // Softmax composition.
            "kittens::warp::row_max(__attn_new_max",
            "kittens::warp::sub_row(__attn_att",
            "kittens::warp::exp(__attn_att, __attn_att);",
            "kittens::warp::row_sum(__attn_sum",
            // PV pass.
            "kittens::warp::copy(__attn_att_bf, __attn_att);",
            "kittens::warp::mma_AB(__attn_o, __attn_att_bf",
            // Finalize.
            "kittens::warp::div_row(__attn_o, __attn_o, __attn_sum);",
            "kittens::warp::copy(__attn_o_bf, __attn_o);",
            "__attn_o_tile.template subtile<16, 64>",
            // Storer role TMA-stores attn_out_page to gmem.
            "kittens::group<1>::tma::store_async",
            "kittens::group<1>::arrive(ss.page_consumed[1])",
        ] {
            assert!(
                cu.source.contains(needle),
                "expected {needle:?} in source, got:\n{}",
                cu.source
            );
        }
    }

    /// Dump-only: prints the full attention .cu source. Marked
    /// ignored so it doesn't pollute regular runs; invoke via
    /// `cargo test -p ferrite-megakernel --lib dump_attention --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn dump_attention_via_cache_source() {
        let bodies = vec![render::render_attention_via_cache::<
            16, 64, 32, 8, 16, 256, 8, 16, 1,
        >(
            0, 1, 0, 1, 5, 0, 1,
            0, 4096, 6, 7,
            0.125_f32, 0.0_f32, false,
        )];
        let cu = render_canonical("test_attn", &budget_lg(), LaunchTier::Attn, &bodies);
        println!("{}", cu.source);
    }

    /// SlidingAttentionViaCache — same algorithm with the runtime
    /// sliding-window mask in the consumer body.
    #[test]
    fn render_sliding_attention_via_cache_basic() {
        let bodies = vec![render::render_sliding_attention_via_cache::<
            16, 64, 32, 8, 16, 256, 8, 16, 1,
        >(
            0, 1, 0, 1, 5, 0, 1,
            0, 4096, 6, 7,
            0.125_f32, 0.0_f32, false,
            /*sliding_window=*/ 4096,
        )];
        let cu = render_canonical(
            "test_sliding_attn",
            &budget_lg(),
            LaunchTier::Attn,
            &bodies,
        );
        assert!(cu.skipped_variants.is_empty());
        // Sliding-specific: the apply lambda referencing the window.
        assert!(
            cu.source
                .contains("(__attn_seq_len - 1 - 4096))"),
            "expected sliding-window mask referencing -4096, got:\n{}",
            cu.source
        );
    }
}
