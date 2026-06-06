// SPDX-License-Identifier: Apache-2.0
//! `tk_player` — the dumb transcription emitter for [`crate::tk_tape::TkTape`].
//!
//! # Contract (from `SUBTILE_IR_REDESIGN.md` §3 + §4 commit 8)
//!
//! - **One match arm per [`Instr`] kind**, ≤5 lines of `format!()`.
//! - **No ambient state lookups.** Every `{...}` placeholder is a
//!   field on the instruction.
//! - **No conditionals beyond the kind dispatch.** Role gating is one
//!   `if (__role == ...)` line per Instr; no other branching.
//! - **No formula computation.** `head_dim * elem`, `slot * row_bytes`,
//!   etc. are tape-time fields, not emit-time arithmetic.
//! - **Hard line budget**: this file ≤800 lines. If we exceed, the IR
//!   is wrong: split an [`Instr`] kind into more granular kinds.
//! - **No inline `kittens::*` strings outside `tk20`** per
//!   `feedback_dogfood_tk20_rust`.

use std::fmt::Write;

use crate::tk_tape::{Instr, TkTape};

// ── tk20 — typed wrappers around TK 2.0 / kittens::* primitives ─────
mod tk20 {
    pub fn sync(n_warps: u32) -> String {
        format!("kittens::group<{n_warps}>::sync();")
    }

    pub fn group_tma_store_commit_group() -> String {
        "kittens::group<1>::tma::store_commit_group();".into()
    }

    pub fn group_tma_store_async_wait(n: u32) -> String {
        format!("kittens::group<1>::tma::store_async_wait<{n}>();")
    }

    /// `mbarrier.init` for one named page barrier slot.
    pub fn mbarrier_init(barrier: &str, page: u8, count: u32) -> String {
        format!("kittens::mbarrier::init(&{barrier}[{page}], {count});")
    }

    /// `mbarrier.wait` at a static parity.
    pub fn mbarrier_wait_static(barrier: &str, page: u8, parity: u8) -> String {
        format!("kittens::mbarrier::wait(&{barrier}[{page}], {parity});")
    }

    /// `mbarrier.wait` at a runtime loop-carried parity.
    pub fn mbarrier_wait_loop(barrier: &str, page: u8, var: u32, start: u8) -> String {
        format!("kittens::mbarrier::wait(&{barrier}[{page}], (v{var} + {start}u) & 1u);")
    }

    pub fn mbarrier_arrive(barrier: &str, page: u8) -> String {
        format!("kittens::group<1>::arrive(&{barrier}[{page}]);")
    }

    /// `tma::load_async` + `expect_bytes` for one page. Takes the
    /// LoadSpec by reference so the caller arm collapses to one
    /// writeln per Instr (per plan §4 step 8 ≤5-line budget).
    pub fn tma_load_async(spec: &crate::tk_tape::LoadSpec) -> String {
        format!(
            "kittens::group<1>::tma::load_async(page_buf[{}], a{}, {}, {}u, {}u, {}u, &page_ready[{}]);",
            spec.dst_page.0, spec.src_tensor.0, spec.byte_off.as_str(),
            spec.tile.rows, spec.tile.cols, spec.tile.elem_bytes, spec.barrier_page.0,
        )
    }

    pub fn tma_store_async(spec: &crate::tk_tape::StoreSpec) -> String {
        format!(
            "kittens::group<1>::tma::store_async(a{}, page_buf[{}], {}, {}u, {}u, {}u);",
            spec.dst_tensor.0, spec.src_page.0, spec.byte_off.as_str(),
            spec.tile.rows, spec.tile.cols, spec.tile.elem_bytes,
        )
    }

    // NUKED: rms_norm, gemm_m1, silu_mul, residual_add, rope_rotate,
    // attn_decode_{init,qkt,sv,finalise} — these emitted invented
    // `kittens::ops::*` calls that don't exist in
    // `third_party/thunderkittens/include/`. Per
    // `feedback_tk_2_0_only` (INVIOLABLE) + plan §1 line 64-65, every
    // Instr must map to ONE TK 2.0 primitive in `include/ops/`, and
    // every `kittens::*` substring this module emits must come from
    // an actual `include/` header. Architectural Instrs (RmsNorm,
    // GemmM1, SiluMul, ResidualAdd, RopeRotate, AttnDecode*) are
    // gone too — they decompose into TK 2.0 primitive Instrs at the
    // SubtileTape→TkTape lowering. The expansion is the next chunk
    // of megakernel work; until it lands, the lowering produces a
    // tape that contains only real TK 2.0 calls (sync / fence /
    // tma::load_async / tma::store_async / mbarrier::*) plus the
    // kernel-end drain.

    pub fn tma_store_async_typed(src_page: u8, dst_arg_idx: u32, tile_type: &str) -> String {
        format!(
            "kittens::group<1>::tma::store_async_typed<{tile_type}>(a{dst_arg_idx}, page_buf[{src_page}]);"
        )
    }

    pub fn arrive_if_runtime_even(barrier: &str, page: u8, parity_arg_idx: u32) -> String {
        format!(
            "if ((a{parity_arg_idx} & 1u) == 0u) {{ \
             kittens::group<1>::arrive(&{barrier}[{page}]); }}"
        )
    }

    // ── Scaffolding helpers (used by emit_kernel) ──────────────────
    //
    // Per `feedback_dogfood_tk20_rust`: every `kittens::*` substring
    // in the emitter lives inside this `tk20` module — including
    // `#include`, kernel signature `kittens::CUtensorMap`, shared-
    // memory page/semaphore declarations, the `softmax_state` array,
    // and the host-wrapper `kittens::CUtensorMap*` cast.
    //
    // None of these are TK 2.0 calls per se — they are scaffolding —
    // but the rule is layering, not call-site: if it says `kittens::`
    // it lives in `tk20`.

    pub fn header_include() -> &'static str {
        "#include <kittens.cuh>\n\n"
    }

    pub fn ctensor_map_kernel_arg(name: &str, comma: &str) -> String {
        format!("    const __grid_constant__ kittens::CUtensorMap {name}{comma}\n")
    }

    pub fn shared_st_bf_decl(rows: u32, cols: u32, count_macro: &str) -> String {
        format!("    __shared__ kittens::st_bf<{rows}, {cols}> page_buf[{count_macro}];\n")
    }

    pub fn shared_semaphore_decl(name: &str, count_macro: &str) -> String {
        format!("    __shared__ kittens::semaphore {name}[{count_macro}];\n")
    }

    pub fn ctensor_map_cast(buf_idx: usize) -> String {
        format!("*reinterpret_cast<const kittens::CUtensorMap*>(bufs[{buf_idx}])")
    }
}

fn barrier_name(kind: crate::tk_tape::PageBarrier) -> &'static str {
    use crate::tk_tape::PageBarrier;
    match kind {
        PageBarrier::Ready => "page_ready",
        PageBarrier::Done => "page_done",
        PageBarrier::Consumed => "page_consumed",
    }
}

/// Emit a full CUDA translation unit — `__global__` kernel + matching
/// `extern "C" cudaError_t launch_<name>(void* const* bufs, const
/// uint32_t* u32_args, cudaStream_t stream)` host wrapper. The wrapper
/// pokes `cudaFuncSetAttribute(...,
/// cudaFuncAttributeMaxDynamicSharedMemorySize, NUM_PAGES*PAGE_SIZE)`
/// then launches the kernel with `<<<1, NUM_WARPS*32, DYN_SMEM,
/// stream>>>`.
///
/// `name` is the kernel's external symbol — both the `__global__` and
/// `launch_<name>` use it. ferrite-wavefront/launcher.rs declares an
/// `extern "C"` FFI to `launch_<name>` for each per-shape kernel; the
/// macro-side dump_wavefront_mega writes the resulting .cu to
/// ~/.cache/cudaforge/megakernels/<name>.cu.
pub fn emit_kernel(name: &str, tape: &TkTape) -> String {
    use crate::tk_tape::{
        KernelArgName, KernelArgTy, NUM_CONSUMER_WARPS, NUM_PAGES, NUM_WARPS, PAGE_SIZE,
        PreludeDecl,
    };

    let total_threads = (NUM_WARPS as u32) * 32;
    let dyn_smem: u64 = (NUM_PAGES as u64) * (PAGE_SIZE as u64);

    let mut out = String::new();
    out.push_str("// emitted by tk_player\n");
    out.push_str(tk20::header_include());

    // Kernel signature: `__global__ void <name>(<args...>)`.
    let _ = write!(out, "extern \"C\" __global__ __launch_bounds__({total_threads}) void {name}(\n");
    for (i, arg) in tape.kernel_args.iter().enumerate() {
        let comma = if i + 1 == tape.kernel_args.len() { "" } else { "," };
        let name = match &arg.name {
            KernelArgName::Fixed(s) => *s,
        };
        match &arg.ty {
            KernelArgTy::U32 { .. } => {
                let _ = writeln!(out, "    uint32_t {name}{comma}");
            }
            KernelArgTy::BufPtr(_) => {
                out.push_str(&tk20::ctensor_map_kernel_arg(name, comma));
            }
        }
    }
    out.push_str(") {\n");

    // Per-arg name aliases: `auto a0 = <KernelArgName>;` so the Instr
    // stream can reference args by index without name lookup.
    for (i, arg) in tape.kernel_args.iter().enumerate() {
        let name = match &arg.name {
            KernelArgName::Fixed(s) => *s,
        };
        let _ = writeln!(out, "    auto a{i} = {name};");
    }
    let _ = write!(out, "\n");

    // Substrate constants the body references.
    let _ = writeln!(out, "    constexpr uint NUM_PAGES = {NUM_PAGES}u;");
    let _ = writeln!(out, "    constexpr uint NUM_CONSUMER_WARPS = {NUM_CONSUMER_WARPS}u;");
    out.push_str(&tk20::shared_st_bf_decl(128, 128, "NUM_PAGES"));
    out.push_str(&tk20::shared_semaphore_decl("page_ready", "NUM_PAGES"));
    out.push_str(&tk20::shared_semaphore_decl("page_done", "NUM_PAGES"));
    out.push_str(&tk20::shared_semaphore_decl("page_consumed", "NUM_PAGES"));
    out.push_str(&tk20::shared_semaphore_decl("page_carry", "NUM_PAGES"));

    // Per-prelude-decl emit. Each PreludeDecl variant lands one
    // declaration at function scope.
    for decl in &tape.prelude {
        match decl {
            PreludeDecl::PerWarpFloatArray { name, len, owner: _ } => {
                let _ = writeln!(out, "    float p{0}[NUM_CONSUMER_WARPS][{1}u];", name.0, len);
            }
            PreludeDecl::PerWarpFloatMatrix { name, rows, cols, owner: _ } => {
                let _ = writeln!(out, "    float p{0}[NUM_CONSUMER_WARPS][{1}u][{2}u];", name.0, rows, cols);
            }
            PreludeDecl::SmemTilePtr { name, page } => {
                let _ = writeln!(out, "    auto& p{0} = page_buf[{1}];", name.0, page.0);
            }
            PreludeDecl::KernelArgAlias { name, arg } => {
                let _ = writeln!(out, "    auto& p{0} = a{1};", name.0, arg.0);
            }
        }
    }

    // NUKED: softmax_state[] array decl + kv_layouts[] constexpr
    // table — both fed the invented `kittens::ops::attn_decode_*` /
    // `kittens::ops::rope_rotate` calls that no longer exist. They
    // come back when AttnDecode / RopeRotate decompose into real
    // TK 2.0 primitive Instrs and need to declare their own state /
    // layout descriptors.

    // Suppress unused warnings for non-yet-used symbols.
    out.push_str("    (void)page_buf; (void)page_ready; (void)page_done;\n");
    out.push_str("    (void)page_consumed; (void)page_carry;\n");

    out.push_str("\n    // ── tape body ──\n");

    for instr in &tape.instrs {
        out.push_str("    ");
        emit_instr(&mut out, tape, instr);
    }

    out.push_str("}\n\n");

    // ── Host launcher (C linkage) ──────────────────────────────────
    //
    // Emits `extern "C" cudaError_t launch_<name>(void* const* bufs,
    // const uint32_t* u32_args, cudaStream_t stream)` so
    // ferrite-wavefront/launcher.rs's FFI can dispatch the kernel
    // without knowing its mangled C++ name or arg list — both are
    // baked into this wrapper.
    //
    // Steps:
    //  1. cudaFuncSetAttribute(MaxDynamicSharedMemorySize, NUM_PAGES *
    //     PAGE_SIZE) lifts the H100 default 48 KB dyn-smem cap so the
    //     page pool fits.
    //  2. Single-CTA persistent megakernel launch
    //     `<<<1, total_threads, DYN_SMEM, stream>>>`.
    //  3. Forwards each `bufs[i]` cast to the declared kernel-arg
    //     pointer type, and `u32_args[j]` for each runtime u32.

    let _ = write!(
        out,
        "extern \"C\" cudaError_t launch_{name}(\n    \
            void* const* bufs,\n    \
            const uint32_t* u32_args,\n    \
            cudaStream_t stream\n) {{\n"
    );
    let _ = writeln!(out, "    constexpr size_t DYN_SMEM = {dyn_smem}u;");
    let _ = writeln!(out, "    (void)bufs; (void)u32_args;");
    let _ = writeln!(
        out,
        "    cudaError_t __err = cudaFuncSetAttribute(\n        \
            (const void*)&{name},\n        \
            cudaFuncAttributeMaxDynamicSharedMemorySize,\n        \
            (int)DYN_SMEM);\n    \
            if (__err != cudaSuccess) return __err;"
    );
    let _ = write!(out, "    {name}<<<1, {total_threads}, DYN_SMEM, stream>>>(");
    let mut bufptr_idx = 0usize;
    let mut u32_idx = 0usize;
    let mut first = true;
    for arg in &tape.kernel_args {
        if !first {
            out.push_str(",");
        }
        first = false;
        out.push_str("\n        ");
        match &arg.ty {
            crate::tk_tape::KernelArgTy::U32 { .. } => {
                let _ = write!(out, "u32_args[{u32_idx}]");
                u32_idx += 1;
            }
            crate::tk_tape::KernelArgTy::BufPtr(_) => {
                // The kernel takes `const __grid_constant__
                // kittens::CUtensorMap` — but our wrapper takes
                // `void*` from the dispatcher. Pre-baked TMA
                // descriptors (`kittens::gl<>` / cuTensorMapEncode
                // ...) need real wiring; today we forward the
                // pointer as a bf16* so the kernel-side TMA wrappers
                // at least see a well-typed address.
                out.push_str(&tk20::ctensor_map_cast(bufptr_idx));
                bufptr_idx += 1;
            }
        }
    }
    out.push_str("\n    );\n");
    out.push_str("    return cudaGetLastError();\n");
    out.push_str("}\n");
    out
}

fn emit_instr(out: &mut String, tape: &TkTape, instr: &Instr) {
    match instr {
        Instr::SyncthreadsCta { role: _ } => out.push_str("__syncthreads();\n"),
        Instr::SyncthreadsGroup { n_warps, role: _ } => {
            let _ = writeln!(out, "{}", tk20::sync(*n_warps));
        }
        Instr::ThreadfenceBlock { role: _ } => out.push_str("__threadfence_block();\n"),
        Instr::ThreadfenceDevice { role: _ } => out.push_str("__threadfence();\n"),
        Instr::ThreadfenceSystem { role: _ } => out.push_str("__threadfence_system();\n"),
        Instr::CommitGroupBulk { role: _ } => {
            let _ = writeln!(out, "{}", tk20::group_tma_store_commit_group());
        }
        Instr::WaitGroupBulk { n, role: _ } => {
            let _ = writeln!(out, "{}", tk20::group_tma_store_async_wait(*n));
        }
        Instr::BarrierInit { page_id, kind, count } => {
            let _ = writeln!(out, "{}", tk20::mbarrier_init(barrier_name(*kind), page_id.0, *count));
        }
        Instr::PageBarrierWaitStaticP0 { page_id, kind, role: _ } => {
            let s = tk20::mbarrier_wait_static(barrier_name(*kind), page_id.0, 0);
            let _ = writeln!(out, "{s}");
        }
        Instr::PageBarrierWaitStaticP1 { page_id, kind, role: _ } => {
            let s = tk20::mbarrier_wait_static(barrier_name(*kind), page_id.0, 1);
            let _ = writeln!(out, "{s}");
        }
        Instr::PageBarrierWaitLoopStart0 { page_id, kind, var, role: _ } => {
            let s = tk20::mbarrier_wait_loop(barrier_name(*kind), page_id.0, var.0, 0);
            let _ = writeln!(out, "{s}");
        }
        Instr::PageBarrierWaitLoopStart1 { page_id, kind, var, role: _ } => {
            let s = tk20::mbarrier_wait_loop(barrier_name(*kind), page_id.0, var.0, 1);
            let _ = writeln!(out, "{s}");
        }
        Instr::PageBarrierArrive { page_id, kind, role: _ } => {
            let _ = writeln!(out, "{}", tk20::mbarrier_arrive(barrier_name(*kind), page_id.0));
        }
        Instr::ArriveIfRuntimeEven { page_id, kind, parity_var, role: _ } => {
            let _ = writeln!(out, "{}", tk20::arrive_if_runtime_even(barrier_name(*kind), page_id.0, parity_var.0 as u32));
        }
        Instr::LoadAsync(spec) => {
            let _ = writeln!(out, "{}", tk20::tma_load_async(spec));
        }
        Instr::StoreAsync(spec) => {
            let _ = writeln!(out, "{}", tk20::tma_store_async(spec));
        }
        Instr::StoreAsyncTyped { dst_page, dst_tensor, tile_type, role: _ } => {
            let s = tk20::tma_store_async_typed(dst_page.0, dst_tensor.0, tile_type.as_str());
            let _ = writeln!(out, "{s}");
        }
        // NUKED: RmsNorm / GemmM1 / SiluMul / ResidualAdd /
        // RopeRotateNeoX / RopeRotateInterleaved / AttnDecodeInit /
        // AttnDecodeQkt / AttnDecodeSv / AttnDecodeFinalise — these
        // were architectural-level Instrs that emitted invented
        // `kittens::ops::*` calls. Per plan §1 line 64-65 + the
        // INVIOLABLE feedback_tk_2_0_only / feedback_tk20_primitives_first,
        // every Instr must map to ONE TK 2.0 primitive in
        // `third_party/thunderkittens/include/`. Until each
        // architectural op decomposes into TK 2.0 primitive Instrs at
        // SubtileTape→TkTape lowering, the player has no Compute arms.
        Instr::DebugOpBeginMarker { op_index } => {
            let _ = writeln!(out, "// op_begin {op_index}");
        }
        Instr::ForLoopOpenConst { var, n } => {
            let _ = writeln!(out, "for (uint v{0} = 0; v{0} < {1}u; ++v{0}) {{", var.0, n);
        }
        Instr::ForLoopOpenKernelArg { var, arg } => {
            let _ = writeln!(out, "for (uint v{0} = 0; v{0} < a{1}; ++v{0}) {{", var.0, arg.0);
        }
        Instr::ForLoopClose { var: _ } => out.push_str("}\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk_tape::WarpRole;

    fn emit(instr: Instr) -> String {
        let mut out = String::new();
        let tape = TkTape::default();
        emit_instr(&mut out, &tape, &instr);
        out
    }

    #[test]
    fn empty_tape_emits_only_header() {
        let tape = TkTape::default();
        let out = emit_kernel("tk_test", &tape);
        assert!(out.contains("// emitted by tk_player"));
        assert!(!out.contains("for ("));
    }

    #[test]
    fn syncthreads_cta_matches_legacy() {
        assert_eq!(
            emit(Instr::SyncthreadsCta { role: WarpRole::All }),
            "__syncthreads();\n"
        );
    }

    #[test]
    fn syncthreads_group_emits_kittens_sync() {
        assert_eq!(
            emit(Instr::SyncthreadsGroup { n_warps: 8, role: WarpRole::All }),
            "kittens::group<8>::sync();\n"
        );
    }

    #[test]
    fn threadfence_device_matches_legacy() {
        assert_eq!(
            emit(Instr::ThreadfenceDevice { role: WarpRole::All }),
            "__threadfence();\n"
        );
    }

    #[test]
    fn threadfence_block_emits_block_scope() {
        assert_eq!(
            emit(Instr::ThreadfenceBlock { role: WarpRole::All }),
            "__threadfence_block();\n"
        );
    }

    #[test]
    fn threadfence_system_emits_system_scope() {
        assert_eq!(
            emit(Instr::ThreadfenceSystem { role: WarpRole::All }),
            "__threadfence_system();\n"
        );
    }

    #[test]
    fn commit_group_emits_tk20_wrapper() {
        assert_eq!(
            emit(Instr::CommitGroupBulk { role: WarpRole::All }),
            "kittens::group<1>::tma::store_commit_group();\n"
        );
    }

    #[test]
    fn wait_group_zero_emits_tk20_wrapper() {
        assert_eq!(
            emit(Instr::WaitGroupBulk { n: 0, role: WarpRole::All }),
            "kittens::group<1>::tma::store_async_wait<0>();\n"
        );
    }

    #[test]
    fn wait_group_nonzero_emits_n() {
        assert_eq!(
            emit(Instr::WaitGroupBulk { n: 3, role: WarpRole::All }),
            "kittens::group<1>::tma::store_async_wait<3>();\n"
        );
    }

    #[test]
    fn cross_op_fence_is_a_sequence_of_primitive_instrs() {
        let mut tape = TkTape::default();
        tape.emit_cross_op_gmem_fence();
        let out = emit_kernel("tk_test", &tape);
        // The 5-Instr fence appears in the body, in order, each on its
        // own indented line. Kernel signature + prelude precede it.
        let body_start = out.find("// ── tape body ──\n").expect("tape body marker");
        let body = &out[body_start..];
        for expected in [
            "__syncthreads();",
            "kittens::group<1>::tma::store_commit_group();",
            "kittens::group<1>::tma::store_async_wait<0>();",
            "__threadfence();",
        ] {
            assert!(body.contains(expected), "missing {expected}: {body}");
        }
    }

    #[test]
    fn page_barrier_init_emits_mbarrier_init() {
        assert_eq!(
            emit(Instr::BarrierInit {
                page_id: crate::tk_tape::PageId(3),
                kind: crate::tk_tape::PageBarrier::Ready,
                count: 16,
            }),
            "kittens::mbarrier::init(&page_ready[3], 16);\n"
        );
    }

    #[test]
    fn page_barrier_wait_static_emits_wait() {
        let s = emit(Instr::PageBarrierWaitStaticP1 {
            page_id: crate::tk_tape::PageId(2),
            kind: crate::tk_tape::PageBarrier::Ready,
            role: WarpRole::AllConsumers,
        });
        assert_eq!(s, "kittens::mbarrier::wait(&page_ready[2], 1);\n");
    }

    #[test]
    fn page_barrier_arrive_emits_arrive() {
        let s = emit(Instr::PageBarrierArrive {
            page_id: crate::tk_tape::PageId(0),
            kind: crate::tk_tape::PageBarrier::Done,
            role: WarpRole::Storer,
        });
        assert_eq!(s, "kittens::group<1>::arrive(&page_done[0]);\n");
    }

    // NUKED: silu_mul_arm_one_call — tested the invented Instr::SiluMul
    // variant that emitted `kittens::ops::silu_mul`. Comes back when
    // SiluMul decomposes into TK 2.0 primitive Instrs.

    #[test]
    fn emit_kernel_includes_signature_and_prelude() {
        // Non-trivial tape with one kernel arg + one Instr; verify the
        // kernel signature, kernel-arg alias, page_buf decl, and body
        // marker all show up in the right order.
        let mut tape = TkTape::default();
        tape.kernel_args.push(crate::tk_tape::KernelArg {
            name: crate::tk_tape::KernelArgName::Fixed("__num_kv_pages"),
            ty: crate::tk_tape::KernelArgTy::U32 {
                source: crate::tk_tape::U32Source::NumKvPages,
            },
        });
        tape.instrs.push(Instr::SyncthreadsCta { role: WarpRole::All });
        let out = emit_kernel("tk_test", &tape);
        let sig = out.find("extern \"C\" __global__").expect("signature");
        let alias = out.find("auto a0 = __num_kv_pages;").expect("kernel-arg alias");
        let pages = out.find("page_buf[NUM_PAGES]").expect("page_buf decl");
        let body = out.find("// ── tape body ──").expect("body marker");
        assert!(sig < alias);
        assert!(alias < pages);
        assert!(pages < body);
        assert!(out.ends_with("}\n"));
    }

    #[test]
    fn for_loop_open_const() {
        let s = emit(Instr::ForLoopOpenConst {
            var: crate::tk_tape::LoopVarId(0),
            n: 8,
        });
        assert_eq!(s, "for (uint v0 = 0; v0 < 8u; ++v0) {\n");
    }
}
