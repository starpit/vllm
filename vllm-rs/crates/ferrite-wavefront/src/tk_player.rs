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

    /// Format a [`ByteOffsetExpr`] as a CUDA arithmetic expression at
    /// emit time. Single-source format per arm — the IR carries
    /// structured data, not a pre-formatted string. Per
    /// `feedback_no_premature_string_encoding`.
    pub fn byte_offset_expr(off: &crate::tk_tape::ByteOffsetExpr) -> String {
        match off {
            crate::tk_tape::ByteOffsetExpr::Const(c) => format!("{c}u"),
            crate::tk_tape::ByteOffsetExpr::LinearLoop { var, stride, base } => {
                format!("({base}u + v{} * {stride}u)", var.0)
            }
        }
    }

    /// Format a [`TileTypeSpec`] as a `kittens::st_<suffix><R, C>`
    /// template alias at emit time. See
    /// `third_party/thunderkittens/include/types/shared/st.cuh:313`.
    pub fn tile_type_spec(spec: &crate::tk_tape::TileTypeSpec) -> String {
        format!(
            "kittens::st_{}<{}, {}>",
            spec.dtype.st_alias_suffix(),
            spec.rows,
            spec.cols,
        )
    }

    /// `tma::load_async` + `expect_bytes` for one page. Takes the
    /// LoadSpec by reference so the caller arm collapses to one
    /// writeln per Instr (per plan §4 step 8 ≤5-line budget).
    pub fn tma_load_async(spec: &crate::tk_tape::LoadSpec) -> String {
        format!(
            "kittens::group<1>::tma::load_async(page_buf[{}], a{}, {}, {}u, {}u, {}u, &page_ready[{}]);",
            spec.dst_page.0, spec.src_tensor.0, byte_offset_expr(&spec.byte_off),
            spec.tile.rows, spec.tile.cols, spec.tile.elem_bytes, spec.barrier_page.0,
        )
    }

    pub fn tma_store_async(spec: &crate::tk_tape::StoreSpec) -> String {
        format!(
            "kittens::group<1>::tma::store_async(a{}, page_buf[{}], {}, {}u, {}u, {}u);",
            spec.dst_tensor.0, spec.src_page.0, byte_offset_expr(&spec.byte_off),
            spec.tile.rows, spec.tile.cols, spec.tile.elem_bytes,
        )
    }

    // ── Compute primitives — one helper per TK 2.0 callable. ───────

    /// `kittens::group<N>::mul(dst, lhs, rhs)` —
    /// `ops/group/shared/tile/maps.cuh:306` (binary tile×tile mul,
    /// included into struct group<N> via `shared/shared.cuh` per
    /// `ops/group/group.cuh:45`). Used by ShTileMul.
    pub fn st_mul(group_n: u32, dst: u8, lhs: u8, rhs: u8) -> String {
        format!(
            "kittens::group<{group_n}>::mul(page_buf[{dst}], page_buf[{lhs}], page_buf[{rhs}]);"
        )
    }

    /// `kittens::group<N>::add(dst, lhs, rhs)` —
    /// `ops/group/shared/tile/maps.cuh:280` (binary tile+tile add,
    /// included into struct group<N> via `shared/shared.cuh` per
    /// `ops/group/group.cuh:45`). Used by ShTileAdd.
    pub fn st_add(group_n: u32, dst: u8, lhs: u8, rhs: u8) -> String {
        format!(
            "kittens::group<{group_n}>::add(page_buf[{dst}], page_buf[{lhs}], page_buf[{rhs}]);"
        )
    }

    /// `kittens::group<N>::div(dst, lhs, rhs)` —
    /// `ops/group/shared/tile/maps.cuh:319`. Used by ShTileDiv.
    pub fn st_div(group_n: u32, dst: u8, lhs: u8, rhs: u8) -> String {
        format!(
            "kittens::group<{group_n}>::div(page_buf[{dst}], page_buf[{lhs}], page_buf[{rhs}]);"
        )
    }

    /// `kittens::group<N>::exp(dst, src)` —
    /// `ops/group/shared/tile/maps.cuh:172` (unary, applies
    /// `base_ops::exp` element-wise). Used by ShTileExp.
    pub fn st_exp(group_n: u32, dst: u8, src: u8) -> String {
        format!("kittens::group<{group_n}>::exp(page_buf[{dst}], page_buf[{src}]);")
    }

    /// `kittens::group<N>::mul(dst, lhs, kittens::<dtype>(scalar))`
    /// — scalar overload of binary `mul` per
    /// `ops/group/shared/tile/maps.cuh:306` + `bin_map<op, T>(T &dst,
    /// const T &src, const typename T::dtype &param)` at line 38.
    /// Used by ShTileMulScalar.
    pub fn st_mul_scalar(
        group_n: u32,
        dst: u8,
        lhs: u8,
        scalar: f32,
        dtype: &crate::tk_tape::TileDtypeTag,
    ) -> String {
        let scalar_ty = dtype.scalar_name();
        format!(
            "kittens::group<{group_n}>::mul(page_buf[{dst}], page_buf[{lhs}], \
             kittens::{scalar_ty}({scalar}f));"
        )
    }

    /// `kittens::group<N>::add(dst, lhs, kittens::<dtype>(scalar))`.
    /// Used by ShTileAddScalar.
    pub fn st_add_scalar(
        group_n: u32,
        dst: u8,
        lhs: u8,
        scalar: f32,
        dtype: &crate::tk_tape::TileDtypeTag,
    ) -> String {
        let scalar_ty = dtype.scalar_name();
        format!(
            "kittens::group<{group_n}>::add(page_buf[{dst}], page_buf[{lhs}], \
             kittens::{scalar_ty}({scalar}f));"
        )
    }

    // ── Register-tile / register-vec emit helpers ─────────────────
    //
    // Each maps 1:1 to a TK 2.0 callable per its header citation.
    // The register handles emit as `rt_<slot>` / `rv_<slot>`; the
    // declarations live in the kernel preamble (see preamble walk
    // in emit_kernel below).

    /// `rt<kittens::<scalar>, R, C, layout> rt_<slot>;` — kernel
    /// preamble decl for a register tile. Per
    /// `feedback_dogfood_tk20_rust`, the layout token comes from
    /// `RegTileLayoutTag::layout_path()`.
    pub fn rt_decl(
        slot: u16,
        rows: u16,
        cols: u16,
        dtype: &crate::tk_tape::TileDtypeTag,
        layout: &crate::tk_tape::RegTileLayoutTag,
    ) -> String {
        let scalar = dtype.scalar_name();
        let lpath = layout.layout_path();
        format!("    kittens::rt<kittens::{scalar}, {rows}, {cols}, {lpath}> rt_{slot};\n")
    }

    /// `rv<kittens::<scalar>, LEN, layout> rv_<slot>;` — preamble decl.
    pub fn rv_decl(
        slot: u16,
        len: u16,
        dtype: &crate::tk_tape::TileDtypeTag,
        layout: &crate::tk_tape::RegVecLayoutTag,
    ) -> String {
        let scalar = dtype.scalar_name();
        let lpath = layout.layout_path();
        format!("    kittens::rv<kittens::{scalar}, {len}, {lpath}> rv_{slot};\n")
    }

    /// `kittens::group<N>::load(rt_dst, page_buf[src])` —
    /// `ops/group/memory/tile/shared_to_register.cuh:15`.
    pub fn load_shmem_to_reg_tile(group_n: u32, src_page: u8, dst_slot: u16) -> String {
        format!("kittens::group<{group_n}>::load(rt_{dst_slot}, page_buf[{src_page}]);")
    }

    /// `kittens::group<N>::store(page_buf[dst], rt_src)` —
    /// `ops/group/memory/tile/shared_to_register.cuh:139`.
    pub fn store_reg_tile_to_shmem(group_n: u32, src_slot: u16, dst_page: u8) -> String {
        format!("kittens::group<{group_n}>::store(page_buf[{dst_page}], rt_{src_slot});")
    }

    /// `kittens::group<N>::load(rv_dst, page_buf[src])` —
    /// `ops/group/memory/vec/shared_to_register.cuh:14`.
    pub fn load_smem_to_reg_vec(group_n: u32, src_page: u8, dst_slot: u16) -> String {
        format!("kittens::group<{group_n}>::load(rv_{dst_slot}, page_buf[{src_page}]);")
    }

    /// `kittens::group<N>::store(page_buf[dst], rv_src)` —
    /// `ops/group/memory/vec/shared_to_register.cuh:100`.
    pub fn store_reg_vec_to_shmem(group_n: u32, src_slot: u16, dst_page: u8) -> String {
        format!("kittens::group<{group_n}>::store(page_buf[{dst_page}], rv_{src_slot});")
    }

    /// `kittens::group<N>::neg(rt_dst, rt_src)` — `register/tile/maps.cuh:572`.
    pub fn rt_neg(group_n: u32, dst: u16, src: u16) -> String {
        format!("kittens::group<{group_n}>::neg(rt_{dst}, rt_{src});")
    }

    /// `kittens::group<N>::exp(rt_dst, rt_src)` — `register/tile/maps.cuh:464`.
    pub fn rt_exp(group_n: u32, dst: u16, src: u16) -> String {
        format!("kittens::group<{group_n}>::exp(rt_{dst}, rt_{src});")
    }

    /// `kittens::group<N>::add(rt_dst, rt_lhs, rt_rhs)` — `register/tile/maps.cuh:681`.
    pub fn rt_add(group_n: u32, dst: u16, lhs: u16, rhs: u16) -> String {
        format!("kittens::group<{group_n}>::add(rt_{dst}, rt_{lhs}, rt_{rhs});")
    }

    /// `kittens::group<N>::sub(rt_dst, rt_lhs, rt_rhs)` — `register/tile/maps.cuh:695`.
    pub fn rt_sub(group_n: u32, dst: u16, lhs: u16, rhs: u16) -> String {
        format!("kittens::group<{group_n}>::sub(rt_{dst}, rt_{lhs}, rt_{rhs});")
    }

    /// `kittens::group<N>::div(rt_dst, rt_lhs, rt_rhs)` — `register/tile/maps.cuh:722`.
    pub fn rt_div(group_n: u32, dst: u16, lhs: u16, rhs: u16) -> String {
        format!("kittens::group<{group_n}>::div(rt_{dst}, rt_{lhs}, rt_{rhs});")
    }

    /// `kittens::group<N>::mul_col(rt_dst, rt_src, rv_col)` — `register/tile/maps.cuh:841`.
    pub fn rt_mul_col(group_n: u32, dst: u16, src: u16, col_vec: u16) -> String {
        format!(
            "kittens::group<{group_n}>::mul_col(rt_{dst}, rt_{src}, rv_{col_vec});"
        )
    }

    /// `kittens::group<N>::add(rt_dst, rt_lhs, kittens::<dtype>(scalar))` —
    /// scalar overload of `register/tile/maps.cuh:681`.
    pub fn rt_add_scalar(
        group_n: u32,
        dst: u16,
        lhs: u16,
        scalar: f32,
        dtype: &crate::tk_tape::TileDtypeTag,
    ) -> String {
        let scalar_ty = dtype.scalar_name();
        format!(
            "kittens::group<{group_n}>::add(rt_{dst}, rt_{lhs}, kittens::{scalar_ty}({scalar}f));"
        )
    }

    // ── RmsNorm-unique TK 2.0 emit helpers (commit B) ──────────────

    /// `kittens::group<N>::row_sum(sv_dst, st_src)` —
    /// `ops/group/shared/tile/reductions.cuh:97`. The dst page is
    /// treated as a shared-vec view (until SmemVecId<R,T> lands).
    pub fn st_row_sum(group_n: u32, dst_page: u8, src_page: u8) -> String {
        format!(
            "kittens::group<{group_n}>::row_sum(page_buf[{dst_page}], page_buf[{src_page}]);"
        )
    }

    /// `kittens::group<N>::mul(sv_dst, sv_src, kittens::<dtype>(scalar))`
    /// — scalar overload, shared-vec.
    pub fn sv_mul_scalar(
        group_n: u32,
        dst_page: u8,
        src_page: u8,
        scalar: f32,
        dtype: &crate::tk_tape::TileDtypeTag,
    ) -> String {
        let scalar_ty = dtype.scalar_name();
        format!(
            "kittens::group<{group_n}>::mul(page_buf[{dst_page}], page_buf[{src_page}], \
             kittens::{scalar_ty}({scalar}f));"
        )
    }

    /// `kittens::group<N>::add(sv_dst, sv_src, kittens::<dtype>(scalar))`
    /// — scalar overload, shared-vec.
    pub fn sv_add_scalar(
        group_n: u32,
        dst_page: u8,
        src_page: u8,
        scalar: f32,
        dtype: &crate::tk_tape::TileDtypeTag,
    ) -> String {
        let scalar_ty = dtype.scalar_name();
        format!(
            "kittens::group<{group_n}>::add(page_buf[{dst_page}], page_buf[{src_page}], \
             kittens::{scalar_ty}({scalar}f));"
        )
    }

    /// `kittens::group<N>::unary_op<kittens::base_ops::rsqrt, RvT>(rv_dst, rv_src)` —
    /// `ops/group/register/vec/maps.cuh:17` + `common/base_ops.cuh:218`.
    /// The `RvT` template arg is the rv type — emitted via the kernel
    /// preamble `decltype` since rv_<id> already declares it.
    pub fn rv_unary_rsqrt(group_n: u32, dst: u16, src: u16) -> String {
        format!(
            "kittens::group<{group_n}>::unary_op<kittens::base_ops::rsqrt, decltype(rv_{dst})>(rv_{dst}, rv_{src});"
        )
    }

    /// `kittens::group<N>::mul_row(st_dst, st_src, sv_row_values)` —
    /// `ops/group/shared/tile/maps.cuh:361`.
    pub fn st_mul_row(group_n: u32, dst_page: u8, src_page: u8, row_vec_page: u8) -> String {
        format!(
            "kittens::group<{group_n}>::mul_row(page_buf[{dst_page}], page_buf[{src_page}], \
             page_buf[{row_vec_page}]);"
        )
    }

    /// `kittens::group<N>::mul_col(st_dst, st_src, sv_col_values)` —
    /// `ops/group/shared/tile/maps.cuh:428`.
    pub fn st_mul_col(group_n: u32, dst_page: u8, src_page: u8, col_vec_page: u8) -> String {
        format!(
            "kittens::group<{group_n}>::mul_col(page_buf[{dst_page}], page_buf[{src_page}], \
             page_buf[{col_vec_page}]);"
        )
    }

    pub fn tma_store_async_typed(
        src_page: u8,
        dst_arg_idx: u32,
        tile_type: &crate::tk_tape::TileTypeSpec,
    ) -> String {
        let tt = tile_type_spec(tile_type);
        format!(
            "kittens::group<1>::tma::store_async_typed<{tt}>(a{dst_arg_idx}, page_buf[{src_page}]);"
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

// NUKED: `group_n_for(role: WarpRole) -> u32` runtime mapping — it
// silently accepted `WarpRole::Loader` for compute Instrs and
// emitted `kittens::group<1>::mul` (a single TMA warp doing
// elementwise-mul, deadlock-adjacent). Replaced by sealed
// `GroupWidth<const N>` + `ComputeWidth` marker in tk_tape.rs:
// `Instr::sh_tile_mul(.., GroupWidth::<1>::PER_WARP)` is now a
// Rust compile error. Per `feedback_ff_subtile_compile_time_inviolable`
// + `feedback_end_to_end_compile_time_proofs`.

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

    // Register-tile / register-vec decls — walk the BTreeMap arenas
    // in id-order so preamble emit is deterministic (= stable goldens).
    // Per `feedback_dogfood_tk20_rust`: emit goes through tk20::*.
    if !tape.reg_tile_arena().is_empty() || !tape.reg_vec_arena().is_empty() {
        out.push_str("\n    // ── register-tile / register-vec decls ──\n");
    }
    for (slot, entry) in tape.reg_tile_arena() {
        out.push_str(&tk20::rt_decl(
            slot.0,
            entry.rows(),
            entry.cols(),
            &entry.dtype(),
            &entry.layout(),
        ));
    }
    for (slot, entry) in tape.reg_vec_arena() {
        out.push_str(&tk20::rv_decl(
            slot.0,
            entry.len(),
            &entry.dtype(),
            &entry.layout(),
        ));
    }

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
        Instr::SyncthreadsGroup { width, role: _ } => {
            let _ = writeln!(out, "{}", tk20::sync(width.n()));
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
            let s = tk20::tma_store_async_typed(dst_page.0, dst_tensor.0, tile_type);
            let _ = writeln!(out, "{s}");
        }
        Instr::ShTileMul { lhs, rhs, dst, width } => {
            let _ = writeln!(out, "{}", tk20::st_mul(width.n(), dst.0, lhs.0, rhs.0));
        }
        Instr::ShTileAdd { lhs, rhs, dst, width } => {
            let _ = writeln!(out, "{}", tk20::st_add(width.n(), dst.0, lhs.0, rhs.0));
        }
        Instr::ShTileDiv { lhs, rhs, dst, width } => {
            let _ = writeln!(out, "{}", tk20::st_div(width.n(), dst.0, lhs.0, rhs.0));
        }
        Instr::ShTileExp { src, dst, width } => {
            let _ = writeln!(out, "{}", tk20::st_exp(width.n(), dst.0, src.0));
        }
        Instr::ShTileMulScalar { lhs, dst, scalar, dtype, width } => {
            let _ = writeln!(out, "{}",
                tk20::st_mul_scalar(width.n(), dst.0, lhs.0, scalar.value(), dtype));
        }
        Instr::ShTileAddScalar { lhs, dst, scalar, dtype, width } => {
            let _ = writeln!(out, "{}",
                tk20::st_add_scalar(width.n(), dst.0, lhs.0, scalar.value(), dtype));
        }
        Instr::LoadShmemToReg { src, dst, width, role: _ } => {
            let _ = writeln!(out, "{}",
                tk20::load_shmem_to_reg_tile(width.n(), src.0, dst.0));
        }
        Instr::StoreRegTileToShmem { src, dst, width, role: _ } => {
            let _ = writeln!(out, "{}",
                tk20::store_reg_tile_to_shmem(width.n(), src.0, dst.0));
        }
        Instr::LoadVecSmemToReg { src, dst, width, role: _ } => {
            let _ = writeln!(out, "{}",
                tk20::load_smem_to_reg_vec(width.n(), src.0, dst.0));
        }
        Instr::StoreRegVecToShmem { src, dst, width, role: _ } => {
            let _ = writeln!(out, "{}",
                tk20::store_reg_vec_to_shmem(width.n(), src.0, dst.0));
        }
        Instr::RegTileNeg { src, dst, width, role: _ } => {
            let _ = writeln!(out, "{}", tk20::rt_neg(width.n(), dst.0, src.0));
        }
        Instr::RegTileExp { src, dst, width, role: _ } => {
            let _ = writeln!(out, "{}", tk20::rt_exp(width.n(), dst.0, src.0));
        }
        Instr::RegTileAdd { lhs, rhs, dst, width, role: _ } => {
            let _ = writeln!(out, "{}", tk20::rt_add(width.n(), dst.0, lhs.0, rhs.0));
        }
        Instr::RegTileSub { lhs, rhs, dst, width, role: _ } => {
            let _ = writeln!(out, "{}", tk20::rt_sub(width.n(), dst.0, lhs.0, rhs.0));
        }
        Instr::RegTileDiv { lhs, rhs, dst, width, role: _ } => {
            let _ = writeln!(out, "{}", tk20::rt_div(width.n(), dst.0, lhs.0, rhs.0));
        }
        Instr::RegTileMulCol { src, col_vec, dst, width, role: _ } => {
            let _ = writeln!(out, "{}",
                tk20::rt_mul_col(width.n(), dst.0, src.0, col_vec.0));
        }
        Instr::RegTileAddScalar { lhs, dst, scalar, dtype, width, role: _ } => {
            let _ = writeln!(out, "{}",
                tk20::rt_add_scalar(width.n(), dst.0, lhs.0, scalar.value(), dtype));
        }
        Instr::ShTileRowSum { src, dst, width, role: _ } => {
            let _ = writeln!(out, "{}", tk20::st_row_sum(width.n(), dst.0, src.0));
        }
        Instr::ShVecMulScalar { src, dst, scalar, dtype, width, role: _ } => {
            let _ = writeln!(out, "{}",
                tk20::sv_mul_scalar(width.n(), dst.0, src.0, scalar.value(), dtype));
        }
        Instr::ShVecAddScalar { src, dst, scalar, dtype, width, role: _ } => {
            let _ = writeln!(out, "{}",
                tk20::sv_add_scalar(width.n(), dst.0, src.0, scalar.value(), dtype));
        }
        Instr::RegVecUnaryRsqrt { src, dst, dtype: _, layout: _, width, role: _ } => {
            let _ = writeln!(out, "{}", tk20::rv_unary_rsqrt(width.n(), dst.0, src.0));
        }
        Instr::ShTileMulRow { src, row_vec, dst, width, role: _ } => {
            let _ = writeln!(out, "{}", tk20::st_mul_row(width.n(), dst.0, src.0, row_vec.0));
        }
        Instr::ShTileMulCol { src, col_vec, dst, width, role: _ } => {
            let _ = writeln!(out, "{}", tk20::st_mul_col(width.n(), dst.0, src.0, col_vec.0));
        }
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
        use crate::tk_tape::AllWarpsRole;
        assert_eq!(
            emit(Instr::syncthreads_cta(AllWarpsRole)),
            "__syncthreads();\n"
        );
    }

    /// `Instr::SyncthreadsGroup` is now constructed from a typed
    /// `GroupWidth<N>` witness (N ∈ sealed `{1, 4, 16, 20}`); the
    /// previous `n_warps: u32` runtime field would have accepted
    /// arbitrary template parameters like `<8>`.
    #[test]
    fn syncthreads_group_emits_kittens_sync() {
        use crate::tk_tape::GroupWidth;
        assert_eq!(
            emit(Instr::syncthreads_group(WarpRole::All, GroupWidth::<4>::WARPGROUP)),
            "kittens::group<4>::sync();\n",
        );
        assert_eq!(
            emit(Instr::syncthreads_group(
                WarpRole::AllConsumers,
                GroupWidth::<16>::ALL_CONSUMERS,
            )),
            "kittens::group<16>::sync();\n",
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

    /// `Instr::ShTileMul` emits one TK 2.0 call to
    /// `kittens::group<NUM_CONSUMER_WARPS>::mul(...)` from
    /// `ops/group/shared/tile/maps.cuh:306` — no invented helpers.
    /// Constructed via the typed [`GroupWidth<16>::ALL_CONSUMERS`]
    /// + [`SmemTileId<128, 128, Bf16>`] witnesses; const-generics
    /// propagate to emit as `<16>` and the shared shape proof rules
    /// out lhs/rhs/dst shape mismatch at rustc time.
    #[test]
    fn sh_tile_mul_emits_real_tk20_call() {
        use crate::tk_tape::{Bf16, GroupWidth, PageId, SmemTileId};
        let lhs = SmemTileId::<128, 128, Bf16>::from_page(PageId(1));
        let rhs = SmemTileId::<128, 128, Bf16>::from_page(PageId(2));
        let dst = SmemTileId::<128, 128, Bf16>::from_page(PageId(3));
        let s = emit(Instr::sh_tile_mul(
            lhs,
            rhs,
            dst,
            GroupWidth::<16>::ALL_CONSUMERS,
        ));
        assert_eq!(
            s,
            "kittens::group<16>::mul(page_buf[3], page_buf[1], page_buf[2]);\n",
        );
    }

    /// `Instr::store_async_typed` derives the
    /// `kittens::st_<NAME><ROWS, COLS>` template instantiation from
    /// the `SmemTileId<ROWS, COLS, T>` type-level shape — a stringly-
    /// typed mismatch (`"st_bf<128, 128>"` for a 64×128 tile) is
    /// no longer expressible. The constructor erases the const-
    /// generics into the runtime field; the player formats the call.
    #[test]
    fn store_async_typed_emits_typed_template_from_witness() {
        use crate::subtile_ir::TensorId;
        use crate::tk_tape::{Bf16, PageId, SmemTileId};
        let src = SmemTileId::<128, 128, Bf16>::from_page(PageId(5));
        let s = emit(Instr::store_async_typed(src, TensorId(7), crate::tk_tape::StorerRole));
        assert_eq!(
            s,
            "kittens::group<1>::tma::store_async_typed<\
             kittens::st_bf<128, 128>>(a7, page_buf[5]);\n",
        );
    }

    /// `Instr::ShTileAdd` mirrors ShTileMul, emits `kittens::group<N>::add`
    /// from `ops/group/shared/tile/maps.cuh:280`. Used by
    /// SubOp::Elementwise(Add) and SumReduce.
    #[test]
    fn sh_tile_add_emits_real_tk20_call() {
        use crate::tk_tape::{Bf16, GroupWidth, PageId, SmemTileId};
        let lhs = SmemTileId::<128, 128, Bf16>::from_page(PageId(4));
        let rhs = SmemTileId::<128, 128, Bf16>::from_page(PageId(5));
        let dst = SmemTileId::<128, 128, Bf16>::from_page(PageId(6));
        let s = emit(Instr::sh_tile_add(
            lhs,
            rhs,
            dst,
            GroupWidth::<16>::ALL_CONSUMERS,
        ));
        assert_eq!(
            s,
            "kittens::group<16>::add(page_buf[6], page_buf[4], page_buf[5]);\n",
        );
    }

    /// `Instr::ShTileDiv` emits `kittens::group<N>::div` from
    /// `ops/group/shared/tile/maps.cuh:319`. Used by SiluMul.
    #[test]
    fn sh_tile_div_emits_real_tk20_call() {
        use crate::tk_tape::{Bf16, GroupWidth, PageId, SmemTileId};
        let lhs = SmemTileId::<128, 128, Bf16>::from_page(PageId(7));
        let rhs = SmemTileId::<128, 128, Bf16>::from_page(PageId(8));
        let dst = SmemTileId::<128, 128, Bf16>::from_page(PageId(9));
        let s = emit(Instr::sh_tile_div(
            lhs,
            rhs,
            dst,
            GroupWidth::<16>::ALL_CONSUMERS,
        ));
        assert_eq!(
            s,
            "kittens::group<16>::div(page_buf[9], page_buf[7], page_buf[8]);\n",
        );
    }

    /// `Instr::ShTileExp` emits the unary exp.
    #[test]
    fn sh_tile_exp_emits_real_tk20_call() {
        use crate::tk_tape::{Bf16, GroupWidth, PageId, SmemTileId};
        let src = SmemTileId::<128, 128, Bf16>::from_page(PageId(2));
        let dst = SmemTileId::<128, 128, Bf16>::from_page(PageId(3));
        let s = emit(Instr::sh_tile_exp(src, dst, GroupWidth::<16>::ALL_CONSUMERS));
        assert_eq!(
            s,
            "kittens::group<16>::exp(page_buf[3], page_buf[2]);\n",
        );
    }

    /// `Instr::ShTileMulScalar` emits the scalar overload with the
    /// dtype-tagged literal `kittens::bf16(<value>f)` derived from
    /// `T::tag()`.
    #[test]
    fn sh_tile_mul_scalar_emits_typed_scalar_literal() {
        use crate::tk_tape::{Bf16, GroupWidth, PageId, ScalarF32, SmemTileId};
        let lhs = SmemTileId::<128, 128, Bf16>::from_page(PageId(0));
        let dst = SmemTileId::<128, 128, Bf16>::from_page(PageId(1));
        let s = emit(Instr::sh_tile_mul_scalar(
            lhs,
            dst,
            ScalarF32::new(-1.0),
            GroupWidth::<16>::ALL_CONSUMERS,
        ));
        assert_eq!(
            s,
            "kittens::group<16>::mul(page_buf[1], page_buf[0], kittens::bf16(-1f));\n",
        );
    }

    /// `Instr::ShTileAddScalar` mirrors mul_scalar with `add`.
    #[test]
    fn sh_tile_add_scalar_emits_typed_scalar_literal() {
        use crate::tk_tape::{Bf16, GroupWidth, PageId, ScalarF32, SmemTileId};
        let lhs = SmemTileId::<128, 128, Bf16>::from_page(PageId(4));
        let dst = SmemTileId::<128, 128, Bf16>::from_page(PageId(5));
        let s = emit(Instr::sh_tile_add_scalar(
            lhs,
            dst,
            ScalarF32::new(1.0),
            GroupWidth::<16>::ALL_CONSUMERS,
        ));
        assert_eq!(
            s,
            "kittens::group<16>::add(page_buf[5], page_buf[4], kittens::bf16(1f));\n",
        );
    }

    // ── Register-tile / register-vec emit tests (commit A) ────────

    /// Helper that builds a tape with a single register-tile slot
    /// minted in the arena, plus the given Instr. Used by the arm
    /// tests below so the emitted body has a consistent rt_<id> id.
    fn emit_with_rt_arena<F: FnOnce(&mut TkTape)>(f: F) -> String {
        use crate::tk_tape::{Bf16, RegTileId, RowLayout};
        let mut tape = TkTape::default();
        // Mint two register tiles + one register vec so slots are stable.
        let _: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let _: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        f(&mut tape);
        let mut out = String::new();
        // Emit just the body Instrs (skip kernel-arg / preamble).
        for instr in &tape.instrs {
            emit_instr(&mut out, &tape, instr);
        }
        out
    }

    #[test]
    fn load_shmem_to_reg_emits_real_tk20_call() {
        use crate::tk_tape::{AllConsumersRole, Bf16, GroupWidth, PageId, RegTileId, RowLayout, SmemTileId};
        let s = emit_with_rt_arena(|tape| {
            let dst: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let src = SmemTileId::<16, 128, Bf16>::from_page(PageId(3));
            tape.push(Instr::load_shmem_to_reg(
                src, dst, GroupWidth::<16>::ALL_CONSUMERS, AllConsumersRole,
            ));
        });
        assert_eq!(s, "kittens::group<16>::load(rt_2, page_buf[3]);\n");
    }

    #[test]
    fn store_reg_tile_to_shmem_emits_real_tk20_call() {
        use crate::tk_tape::{AllConsumersRole, Bf16, GroupWidth, PageId, RegTileId, RowLayout, SmemTileId};
        let s = emit_with_rt_arena(|tape| {
            let src: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let dst = SmemTileId::<16, 128, Bf16>::from_page(PageId(7));
            tape.push(Instr::store_reg_tile_to_shmem(
                src, dst, GroupWidth::<16>::ALL_CONSUMERS, AllConsumersRole,
            ));
        });
        assert_eq!(s, "kittens::group<16>::store(page_buf[7], rt_2);\n");
    }

    #[test]
    fn reg_tile_neg_emits_real_tk20_call() {
        use crate::tk_tape::{AllConsumersRole, Bf16, GroupWidth, RegTileId, RowLayout};
        let s = emit_with_rt_arena(|tape| {
            let src: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let dst: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            tape.push(Instr::reg_tile_neg(
                src, dst, GroupWidth::<16>::ALL_CONSUMERS, AllConsumersRole,
            ));
        });
        assert_eq!(s, "kittens::group<16>::neg(rt_3, rt_2);\n");
    }

    #[test]
    fn reg_tile_exp_emits_real_tk20_call() {
        use crate::tk_tape::{AllConsumersRole, Bf16, GroupWidth, RegTileId, RowLayout};
        let s = emit_with_rt_arena(|tape| {
            let src: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let dst: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            tape.push(Instr::reg_tile_exp(
                src, dst, GroupWidth::<16>::ALL_CONSUMERS, AllConsumersRole,
            ));
        });
        assert_eq!(s, "kittens::group<16>::exp(rt_3, rt_2);\n");
    }

    #[test]
    fn reg_tile_add_emits_real_tk20_call() {
        use crate::tk_tape::{AllConsumersRole, Bf16, GroupWidth, RegTileId, RowLayout};
        let s = emit_with_rt_arena(|tape| {
            let lhs: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let rhs: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let dst: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            tape.push(Instr::reg_tile_add(
                lhs, rhs, dst, GroupWidth::<16>::ALL_CONSUMERS, AllConsumersRole,
            ));
        });
        assert_eq!(s, "kittens::group<16>::add(rt_4, rt_2, rt_3);\n");
    }

    #[test]
    fn reg_tile_div_emits_real_tk20_call() {
        use crate::tk_tape::{AllConsumersRole, Bf16, GroupWidth, RegTileId, RowLayout};
        let s = emit_with_rt_arena(|tape| {
            let lhs: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let rhs: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let dst: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            tape.push(Instr::reg_tile_div(
                lhs, rhs, dst, GroupWidth::<16>::ALL_CONSUMERS, AllConsumersRole,
            ));
        });
        assert_eq!(s, "kittens::group<16>::div(rt_4, rt_2, rt_3);\n");
    }

    #[test]
    fn reg_tile_add_scalar_emits_typed_scalar_literal() {
        use crate::tk_tape::{AllConsumersRole, Bf16, GroupWidth, RegTileId, RowLayout, ScalarF32};
        let s = emit_with_rt_arena(|tape| {
            let src: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let dst: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            tape.push(Instr::reg_tile_add_scalar(
                src, dst, ScalarF32::new(1.0), GroupWidth::<16>::ALL_CONSUMERS, AllConsumersRole,
            ));
        });
        assert_eq!(s, "kittens::group<16>::add(rt_3, rt_2, kittens::bf16(1f));\n");
    }

    #[test]
    fn reg_tile_mul_col_emits_typed_call() {
        use crate::tk_tape::{AllConsumersRole, Bf16, GroupWidth, OrthoLayout, RegTileId, RegVecId, RowLayout};
        let s = emit_with_rt_arena(|tape| {
            let src: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let dst: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let col_vec: RegVecId<128, Bf16, OrthoLayout> = tape.mint_reg_vec();
            tape.push(Instr::reg_tile_mul_col(
                src, col_vec, dst, GroupWidth::<16>::ALL_CONSUMERS, AllConsumersRole,
            ));
        });
        assert_eq!(s, "kittens::group<16>::mul_col(rt_3, rt_2, rv_0);\n");
    }

    /// Verify the kernel preamble emits register-tile decls in
    /// deterministic id order (BTreeMap walk).
    #[test]
    fn emit_kernel_emits_register_tile_decls() {
        use crate::tk_tape::{Bf16, RegTileId, RegVecId, RowLayout, OrthoLayout};
        let mut tape = TkTape::default();
        tape.kernel_args.push(crate::tk_tape::KernelArg {
            name: crate::tk_tape::KernelArgName::Fixed("__num_kv_pages"),
            ty: crate::tk_tape::KernelArgTy::U32 {
                source: crate::tk_tape::U32Source::NumKvPages,
            },
        });
        let _: RegTileId<16, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let _: RegTileId<32, 64, Bf16, RowLayout> = tape.mint_reg_tile();
        let _: RegVecId<128, Bf16, OrthoLayout> = tape.mint_reg_vec();
        let out = emit_kernel("tk_test", &tape);
        // rt_0 first (id-order), rt_1 second
        let rt0 = out
            .find("kittens::rt<kittens::bf16, 16, 128, kittens::ducks::rt_layout::row> rt_0;")
            .expect("rt_0 decl");
        let rt1 = out
            .find("kittens::rt<kittens::bf16, 32, 64, kittens::ducks::rt_layout::row> rt_1;")
            .expect("rt_1 decl");
        let rv0 = out
            .find("kittens::rv<kittens::bf16, 128, kittens::ducks::rv_layout::ortho> rv_0;")
            .expect("rv_0 decl");
        assert!(rt0 < rt1);
        // rv decls follow rt decls
        assert!(rt1 < rv0);
    }

    /// Warpgroup-width construction (used later by mma_AB) also
    /// type-checks and emits `<4>`. Same code path, different N.
    #[test]
    fn sh_tile_mul_warpgroup_width_emits_group_4() {
        use crate::tk_tape::{Bf16, GroupWidth, PageId, SmemTileId};
        let lhs = SmemTileId::<128, 128, Bf16>::from_page(PageId(0));
        let rhs = SmemTileId::<128, 128, Bf16>::from_page(PageId(1));
        let dst = SmemTileId::<128, 128, Bf16>::from_page(PageId(2));
        let s = emit(Instr::sh_tile_mul(
            lhs,
            rhs,
            dst,
            GroupWidth::<4>::WARPGROUP,
        ));
        assert_eq!(
            s,
            "kittens::group<4>::mul(page_buf[2], page_buf[0], page_buf[1]);\n",
        );
    }

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
        tape.instrs.push(Instr::syncthreads_cta(crate::tk_tape::AllWarpsRole));
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
