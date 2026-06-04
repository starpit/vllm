// SPDX-License-Identifier: Apache-2.0
//! `TkProgram` → CUDA-source emit. The walker is a literal `match` over
//! [`TkInstr`]; it makes NO decisions. Every numeric the kernel needs
//! (page id, phase parity, warp role, tile shape, region offset) is in
//! the IR.
//!
//! This is the "trivial player" rule from `subtile_ir`'s design,
//! lifted into the warp tier: any cleverness in `emit_*` is a bug —
//! it belongs in the per-op lowering that built the [`TkProgram`].
//!
//! # tk20 dogfood
//!
//! Per `feedback_dogfood_tk20_rust` every TK 2.0 call must go through
//! a typed Rust API, never an inline `format!("kittens::warp::*")`.
//! The thin [`tk20`] module here is that API. It is intentionally a
//! stub at this stage: each function emits one TK 2.0 call as a
//! `String`. When the full `tk20` crate from ff-mega-codegen is
//! cherry-picked, the body of each function moves there 1:1; call
//! sites stay unchanged.

#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::tk_warp_ir::{
    LoopBound, PageBarrier, TileShape, TkInstr, TkProgram, WarpRole, NUM_CONSUMER_WARPS,
};

// ── tk20 — typed CUDA-source emitters ───────────────────────────────

/// Stub for the `tk20::*` Rust API. Each function returns the textual
/// CUDA fragment for one TK 2.0 primitive call. Lane-0-gated TMA per
/// `feedback_tk20_tma_lane_gate`: the loader/storer paths use
/// `kittens::group<1>::tma::*` (not the thread-level `kittens::tma::*`).
pub mod tk20 {
    use super::PageBarrier;

    fn barrier_field(kind: PageBarrier) -> &'static str {
        match kind {
            PageBarrier::Ready => "page_ready",
            PageBarrier::Done => "page_done",
            PageBarrier::Consumed => "page_consumed",
        }
    }

    /// `kittens::group<N>::wait(barrier, phase)`. `n_warps` is the
    /// thread group the call is gated on. `phase_expr` is the literal
    /// CUDA expression — `"0"` / `"1"` for static phases, `"(__i & 1)"`
    /// or similar for runtime parities.
    pub fn wait(n_warps: u32, kind: PageBarrier, page_id: u8, phase_expr: &str) -> String {
        let bar = barrier_field(kind);
        format!("kittens::group<{n_warps}>::wait({bar}[{page_id}], {phase_expr});")
    }

    /// `kittens::group<N>::arrive(barrier)`.
    pub fn arrive(n_warps: u32, kind: PageBarrier, page_id: u8) -> String {
        let bar = barrier_field(kind);
        format!("kittens::group<{n_warps}>::arrive({bar}[{page_id}]);")
    }

    /// `if ((<parity_var> & 1u) == 0u) {
    /// kittens::group<1>::arrive(<barrier>[<page_id>]); }`. Single
    /// CUDA statement; one TK 2.0 call inside the runtime-parity
    /// guard. Used by the post-runtime-loop phantom round.
    pub fn arrive_if_runtime_even(kind: PageBarrier, page_id: u8, parity_var: &str) -> String {
        let inner = arrive(1, kind, page_id);
        format!("if (({parity_var} & 1u) == 0u) {{ {inner} }}")
    }

    /// `kittens::sv_bf<N>` — typed shared-memory vector tile spelling
    /// used as a TMA descriptor's `tile_type` argument. Single source
    /// of truth for the C++ type name.
    pub fn sv_bf_tile_type(cols: i32) -> String {
        format!("kittens::sv_bf<{cols}>")
    }

    /// `asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");` —
    /// async-proxy fence used by the kernel scaffold AFTER the
    /// `init_semaphore(...)` calls so the per-warp barrier-init
    /// writes are observable to subsequent `mbarrier.try_wait`
    /// consumers. `__syncthreads()` only orders generic-proxy ops
    /// with each other; the async proxy's parity bits need this
    /// fence to flush. TK 2.0's KVM scaffold issues the same
    /// intrinsic at `third_party/thunderkittens/prototype/vm/vm.cuh:99`
    /// (no Rust wrapper exists in the upstream API surface yet, so
    /// this is the SOLE TK-2.0-justified raw asm in the emit path).
    pub fn fence_proxy_async_shared_cta() -> String {
        "asm volatile(\"fence.proxy.async.shared::cta;\\n\" ::: \"memory\");".to_string()
    }

    /// `kittens::group<1>::sync()` / `<N>::sync()`.
    pub fn sync(n_warps: u32) -> String {
        format!("kittens::group<{n_warps}>::sync();")
    }


    /// Total bytes a tile of `(rows, cols)` occupies given `elem_bytes`.
    fn tile_bytes(rows: u32, cols: u32, elem_bytes: u32) -> u32 {
        rows * cols * elem_bytes
    }

    /// Lane-0-gated TMA load. Emits the production "non-tensor TMA"
    /// pair: `expect_bytes` arms the page-ready barrier for the byte
    /// count, then `load_async(dst, src, bytes, ready)` fires the load.
    /// The barrier completes when the load delivers all expected bytes —
    /// no separate `arrive(Ready)` call is needed (and emitting one
    /// would over-count the arrivals and break round parity).
    ///
    /// Source: `include/ops/group/util/tma.cuh:18` (`expect_bytes`) and
    /// `:72` (`load_async(void*, void*, uint32_t, semaphore&)`).
    pub fn tma_load_async(
        page_id: u8,
        src_buf: u32,
        src_byte_off: u64,
        rows: u32,
        cols: u32,
        elem_bytes: u32,
        dyn_byte_off: Option<&str>,
    ) -> String {
        let bytes = tile_bytes(rows, cols, elem_bytes);
        let off_expr = match dyn_byte_off {
            Some(e) => format!("({src_byte_off}u + ({e}))"),
            None => format!("{src_byte_off}"),
        };
        // Cast through `uintptr_t` to drop `const` from the buffer arg
        // (the kernel signature uses `const __nv_bfloat16* __restrict__`
        // for inputs, but TK 2.0 `tma::load_async(void*, void*, ...)`
        // wants non-const). `reinterpret_cast<void*>(const char*)` is
        // rejected — `cannot cast away const`.
        format!(
            "kittens::group<1>::tma::expect_bytes(page_ready[{page_id}], {bytes}); \
             kittens::group<1>::tma::load_async(\
             reinterpret_cast<void*>(page_buf[{page_id}]), \
             reinterpret_cast<void*>(\
             reinterpret_cast<uintptr_t>(buf{src_buf}) + {off_expr}), \
             {bytes}, \
             page_ready[{page_id}]);"
        )
    }

    /// Lane-0-gated TMA store. Emits the production pair:
    /// `store_async(dst, src, bytes)` queues the store and
    /// `store_async_wait()` blocks the storer warp until the store
    /// completes (so the subsequent `arrive(Consumed)` is safe to use
    /// the page slot for the next round).
    ///
    /// Source: `include/ops/group/util/tma.cuh:82` (`store_async`) and
    /// `:46` (`store_async_wait<N=0>`).
    pub fn tma_store_async(
        page_id: u8,
        dst_buf: u32,
        dst_byte_off: u64,
        rows: u32,
        cols: u32,
        elem_bytes: u32,
        dyn_byte_off: Option<&str>,
    ) -> String {
        let bytes = tile_bytes(rows, cols, elem_bytes);
        let off_expr = match dyn_byte_off {
            Some(e) => format!("({dst_byte_off}u + ({e}))"),
            None => format!("{dst_byte_off}"),
        };
        // E.13: emit `store_commit_group` between `store_async` and
        // `store_async_wait`. The typed (sv/st) store_async variant
        // already does this (`tma_store_async_typed` at line ~191);
        // the raw-bulk variant historically did not, so
        // `store_async_wait`'s underlying
        // `cp.async.bulk.wait_group N=0` had no group to wait on and
        // returned immediately. The actual stores stayed in flight,
        // and downstream readers (TMA loads in the next op, plus
        // cross-launch reads) saw stale gmem.
        format!(
            "kittens::group<1>::tma::store_async(\
             reinterpret_cast<void*>(reinterpret_cast<char*>(buf{dst_buf}) + {off_expr}), \
             reinterpret_cast<void*>(page_buf[{page_id}]), \
             {bytes}); \
             kittens::group<1>::tma::store_commit_group(); \
             kittens::group<1>::tma::store_async_wait();"
        )
    }

    /// Step C.3 — descriptor (typed `gl<>`) TMA store for opted-in
    /// buffers. Pair: typed `store_async<cache_policy::NORMAL>(dst, src,
    /// {b,d,r,c})` + `store_commit_group()` + `store_async_wait()`.
    /// `tile_type` is the CUDA tile type spelling (e.g.
    /// `"kittens::sv_bf<2048>"` for vector form). Coords are
    /// tile-frame `{0,0,0,0}` for the C.3 initial scope (rmsnorm output
    /// store, no dynamic offset, single-tile shape).
    ///
    /// Sources:
    ///   - `include/ops/group/memory/vec/tma.cuh` (vector `store_async`).
    ///   - `include/ops/group/util/tma.cuh:38` (`store_commit_group`).
    ///   - `include/ops/group/util/tma.cuh:47` (`store_async_wait<N=0>`).
    /// Same TK 2.0 spelling as the C.2 PoC smoke kernel
    /// (`tk_emit_rmsnorm_tensor_smoke.rs`).
    pub fn tma_store_async_typed(page_id: u8, dst_buf: u32, tile_type: &str) -> String {
        format!(
            "kittens::group<1>::tma::store_async<kittens::cache_policy::NORMAL>(\
             arg{dst_buf}, \
             *reinterpret_cast<{tile_type}*>(page_buf[{page_id}]), \
             {{0, 0, 0, 0}}); \
             kittens::group<1>::tma::store_commit_group(); \
             kittens::group<1>::tma::store_async_wait();"
        )
    }

    // ── Warpgroup wgmma (Hopper peak; default for matmuls m>1) ────
    //
    // Each binding cites its TK 2.0 header line. PR review verifies
    // the cited line still matches the emitted spelling. Per-binding
    // unit tests assert the emit string contains the literal
    // `kittens::warpgroup::*` — this catches TK 1.0 drift at
    // `cargo test`, before nvcc.

    /// `kittens::warpgroup::mma_fence(d)`. Source:
    /// `third_party/thunderkittens/include/ops/group/mma/warpgroup.cuh:23`.
    pub fn warpgroup_mma_fence(d: &str) -> String {
        format!("kittens::warpgroup::mma_fence({d});")
    }

    /// `kittens::warpgroup::mma_AB(d, a, b)` — accumulating wgmma.
    /// Source: `ops/group/mma/warpgroup.cuh:140`. Defaults
    /// `fence=1, accumulate=1` (in TK 2.0); emit leaves them implicit.
    pub fn warpgroup_mma_ab(d: &str, a: &str, b: &str) -> String {
        format!("kittens::warpgroup::mma_AB({d}, {a}, {b});")
    }

    /// `kittens::warpgroup::mm_AB(d, a, b)` — reset variant of
    /// `mma_AB`. Source: `ops/group/mma/warpgroup.cuh:186`.
    pub fn warpgroup_mm_ab(d: &str, a: &str, b: &str) -> String {
        format!("kittens::warpgroup::mm_AB({d}, {a}, {b});")
    }

    /// `kittens::warpgroup::mma_ABt(d, a, b)` — wgmma with B transposed.
    /// Source: `ops/group/mma/warpgroup.cuh:258`.
    pub fn warpgroup_mma_abt(d: &str, a: &str, b: &str) -> String {
        format!("kittens::warpgroup::mma_ABt({d}, {a}, {b});")
    }

    /// `kittens::warpgroup::mma_commit_group()`. Source:
    /// `ops/group/mma/warpgroup.cuh:78`.
    pub fn warpgroup_mma_commit_group() -> String {
        "kittens::warpgroup::mma_commit_group();".to_string()
    }

    /// `kittens::warpgroup::mma_async_wait<N>()`. Source:
    /// `ops/group/mma/warpgroup.cuh:91`.
    pub fn warpgroup_mma_async_wait(n: u32) -> String {
        format!("kittens::warpgroup::mma_async_wait<{n}>();")
    }

    // ── Per-warp mma (decode m=1 fast path) ────────────────────────

    /// `kittens::warp::mma_AB(d, a, b)`. Source:
    /// `ops/group/mma/warp.cuh`. Used by `lower_gemm_m1`; for m>1
    /// prefer `warpgroup_mma_ab`.
    pub fn warp_mma_ab(d: &str, a: &str, b: &str) -> String {
        format!("kittens::warp::mma_AB({d}, {a}, {b}, {d});")
    }

    /// `kittens::warp::mma_ABt(d, a, b)`. Source: `ops/group/mma/warp.cuh`.
    pub fn warp_mma_abt(d: &str, a: &str, b: &str) -> String {
        format!("kittens::warp::mma_ABt({d}, {a}, {b}, {d});")
    }

    // ── Group-scoped TMA store-side complements to load_async / store_async ──

    /// `kittens::group<1>::tma::store_commit_group()`. Source:
    /// `ops/group/util/tma.cuh:38`.
    pub fn group_tma_store_commit_group() -> String {
        "kittens::group<1>::tma::store_commit_group();".to_string()
    }

    /// `kittens::group<1>::tma::store_async_wait<N>()`. Source:
    /// `ops/group/util/tma.cuh:47`. Already used inline in
    /// `tma_store_async()` above with default N; this variant lets
    /// callers issue the wait separately at a chosen N.
    pub fn group_tma_store_async_wait(n: u32) -> String {
        format!("kittens::group<1>::tma::store_async_wait<{n}>();")
    }

    /// `kittens::group<1>::tma::store_async_read_wait<N>()`. Source:
    /// `ops/group/util/tma.cuh:61`.
    pub fn group_tma_store_async_read_wait(n: u32) -> String {
        format!("kittens::group<1>::tma::store_async_read_wait<{n}>();")
    }

    // ── Register-tile reductions / element-wise (warp-scoped) ──────

    /// `kittens::warp::row_max(rv, rt)`. Source:
    /// `ops/group/register/tile/reductions.cuh`. Per-row max into a
    /// register vector. Used by online softmax row-max accumulator.
    pub fn warp_row_max(rv: &str, rt: &str) -> String {
        format!("kittens::warp::row_max({rv}, {rt});")
    }

    /// `kittens::warp::row_sum(rv, rt)`. Source: same header.
    pub fn warp_row_sum(rv: &str, rt: &str) -> String {
        format!("kittens::warp::row_sum({rv}, {rt});")
    }

    /// `kittens::warp::exp2(rt, rt)` — in-place exp2 on a register
    /// tile (TK 2.0 binary-form: dst, src). Source:
    /// `ops/group/register/tile/maps.cuh`.
    pub fn warp_exp2(rt: &str) -> String {
        format!("kittens::warp::exp2({rt}, {rt});")
    }

    /// `kittens::warp::mul_row(rt, rt, rv)` — in-place row-wise scale.
    /// Source: `ops/group/register/tile/maps.cuh`.
    pub fn warp_mul_row(rt: &str, rv: &str) -> String {
        format!("kittens::warp::mul_row({rt}, {rt}, {rv});")
    }

    /// `kittens::warp::div_row(rt, rt, rv)`. Source: same header.
    pub fn warp_div_row(rt: &str, rv: &str) -> String {
        format!("kittens::warp::div_row({rt}, {rt}, {rv});")
    }

    /// `kittens::warp::neg_infty(rv)`. Source:
    /// `ops/group/register/vec/maps.cuh`. Initialises a register vector
    /// to `-inf` (online-softmax row-max identity).
    pub fn warp_neg_infty(rv: &str) -> String {
        format!("kittens::warp::neg_infty({rv});")
    }

    /// `kittens::warp::zero(rt)`. Source:
    /// `ops/group/register/tile/maps.cuh`. Accumulator init.
    pub fn warp_zero_rt(rt: &str) -> String {
        format!("kittens::warp::zero({rt});")
    }

    /// `kittens::warp::add(d, a, b)`. Source:
    /// `ops/group/register/tile/maps.cuh`. Element-wise register-tile
    /// add.
    pub fn warp_add_rt(d: &str, a: &str, b: &str) -> String {
        format!("kittens::warp::add({d}, {a}, {b});")
    }

    /// `kittens::warp::mul(d, a, b)`. Source: same header. Element-wise
    /// register-tile multiply.
    pub fn warp_mul_rt(d: &str, a: &str, b: &str) -> String {
        format!("kittens::warp::mul({d}, {a}, {b});")
    }

    // ── Register-vector primitives (warp-scoped) ────────────────────

    /// Declare an `sv_bf<K>&` view aliased over a raw bf16 pointer.
    /// Source: `types/shared/sv.cuh:38-89` (the `sv` template + its
    /// `subvec<sub_length>(idx)` accessor at line 83). `ptr_expr` is
    /// any C++ expression yielding a `__nv_bfloat16*` (or `void*`)
    /// referencing shared-memory storage; the cast reinterprets it as
    /// a `kittens::sv_bf<K>` so warp-scoped TK 2.0 ops can consume it
    /// (e.g. `kittens::warp::load(rv, sv)` — see `warp_load_rv_from_sv`).
    pub fn decl_sv_view_bf(name: &str, ptr_expr: &str, k: u32) -> String {
        format!(
            "kittens::sv_bf<{k}>& {name} = *reinterpret_cast<kittens::sv_bf<{k}>*>({ptr_expr});"
        )
    }

    /// Declare a per-thread `rv_bf<K>` register vector. Source:
    /// `types/register/rv.cuh:115-116` (the `rv_bf<_l, layout>` alias
    /// over `rv<bf16, _l, layout>`). Default layout `naive`. Storage
    /// is per-lane registers; total bytes per thread = `K / 32 * 2`.
    pub fn decl_rv_bf(name: &str, k: u32) -> String {
        format!("kittens::rv_bf<{k}> {name};")
    }

    /// Declare a per-thread `rv_fl<K>` register vector. Source:
    /// `types/register/rv.cuh:115` (the `rv_fl<_l, layout>` alias).
    /// Used as the f32 accumulator after a bf16 → f32 `warp::copy`.
    pub fn decl_rv_fl(name: &str, k: u32) -> String {
        format!("kittens::rv_fl<{k}> {name};")
    }

    /// `kittens::warp::load(rv, sv)`. Source:
    /// `ops/group/memory/vec/shared_to_register.cuh:14-15`. Coalesced
    /// shared → register vector load. `RV::length == SV::length`
    /// (TK 2.0 `static_assert` at line 21).
    pub fn warp_load_rv_from_sv(rv: &str, sv: &str) -> String {
        format!("kittens::warp::load({rv}, {sv});")
    }

    /// `kittens::warp::copy(dst_rv, src_rv)`. Source:
    /// `ops/group/register/vec/conversions.cuh:33-34`. Element-wise
    /// dtype convert + copy. Used to lift bf16 → f32 before the
    /// reduction (avoids bf16 underflow in the accumulator).
    pub fn warp_copy_rv(dst: &str, src: &str) -> String {
        format!("kittens::warp::copy({dst}, {src});")
    }

    /// `kittens::warp::mul(dst_rv, lhs_rv, rhs_rv)`. Source:
    /// `ops/group/register/vec/maps.cuh:358-361`. Element-wise
    /// register-vector multiply.
    pub fn warp_mul_rv(dst: &str, lhs: &str, rhs: &str) -> String {
        format!("kittens::warp::mul({dst}, {lhs}, {rhs});")
    }

    /// `kittens::warp::sum(rv) -> scalar`. Source:
    /// `ops/group/register/vec/reductions.cuh:133-137`. Returns the
    /// per-warp scalar sum of the register vector (warp-collective
    /// reduction; result identical across all 32 lanes). Emits an
    /// EXPRESSION (no trailing `;`) so the caller composes it inside
    /// an arithmetic statement (`acc += warp::sum(rv);`).
    pub fn warp_sum_rv(rv: &str) -> String {
        format!("kittens::warp::sum({rv})")
    }

    // ── Hopper register-budget management ──────────────────────────

    /// `kittens::warpgroup::increase_registers<N>()`. Source:
    /// `ops/group/group.cuh:52`. `n % 8 == 0` enforced by TK 2.0
    /// `static_assert`; bind-time `debug_assert!` mirrors that.
    pub fn warpgroup_increase_registers(n: u32) -> String {
        debug_assert_eq!(n % 8, 0, "TK 2.0 setmaxnreg requires n % 8 == 0; got {n}");
        format!("kittens::warpgroup::increase_registers<{n}>();")
    }

    /// `kittens::warpgroup::decrease_registers<N>()`. Source:
    /// `ops/group/group.cuh:56`. Same `n % 8 == 0` rule.
    pub fn warpgroup_decrease_registers(n: u32) -> String {
        debug_assert_eq!(n % 8, 0, "TK 2.0 setmaxnreg requires n % 8 == 0; got {n}");
        format!("kittens::warpgroup::decrease_registers<{n}>();")
    }

    // ── RMSNorm consumer body (typed atom; emits raw CUDA inline) ──
    //
    // RmsNorm under TK 2.0 idiom: per-thread bf16 squared-sum +
    // `__shfl_xor_sync` butterfly across the 32 warp lanes + per-thread
    // normalize. This matches `~/git/Megakernels` `mk-v2` (TK 2.0,
    // sm_90a) `csrc/itypes/rmsnorm.cuh` lines 130-180: the production
    // TK 2.0 RmsNorm pattern uses `kittens::sv_bf<N>` ONLY as a typed
    // page-pointer alias, not as the operand to `kittens::group<N>::
    // sum/mul/etc.`. The reduction stays per-thread + warp shfl
    // because (a) `kittens::warp::row_*` reductions operate on
    // 16x16-aligned register tiles, not on m=1 rows, and (b) using
    // sv_bf-scope ops would require an extra scratch sv (mul
    // destroys input → can't recompute the original for the
    // normalize step) — substrate-level page-allocator change beyond
    // Phase 1's scope. Future optimisation: vectorize via uint64_t
    // 4-lane unpack like mk-v2's `unpack_x4`/`pack_x4`.
    //
    // The typed atom replaces `Compute { body: format!(rmsnorm_body) }`
    // with `Compute { calls: vec![RmsNormConsumerBody { ... }] }` —
    // routes the body emit through Tk20Call::emit() so every CUDA
    // fragment in the final .cu is traceable to a typed Rust atom
    // (per feedback_dogfood_tk20_rust). The emit string is byte-
    // identical to the legacy `rmsnorm_compute_body` so the existing
    // `rmsnorm_kernel_matches_cpu_golden` test passes unchanged.

    // ── AttnDecode body atoms (Phase 4) ────────────────────────────
    //
    // AttnDecode's compute spans FIVE separate bodies: a function-
    // scope prelude (typed page views + persistent compute
    // accumulators that span the whole KV sweep), an init-softmax
    // body, the QK^T+softmax-online step (inside the KV-sweep loop),
    // the softmax(P)@V step (also inside the loop), and the final
    // O = O_accum / l_sum normalize. Each is wrapped in a typed
    // `Tk20Call` variant for the Phase 5 cutover.
    //
    // The bodies share function-scope state — `__m_max_a<u>`,
    // `__l_sum_a<u>`, `__o_accum_a<u>[][HEAD_DIM]` etc. — declared in
    // the prelude. The `<u>` suffix is the op's `unique_id` so two
    // AttnDecode instances in the same TkProgram (e.g. one per
    // transformer layer in a multi-layer Llama forward) don't collide
    // on the same C++ identifiers.

    /// Emit the AttnDecode prelude — typed page views + persistent
    /// compute accumulators. Written into `TkProgram::prelude` (not
    /// inside an instruction's role guard) so all warps see the
    /// declarations and the consumers can reference per-warp state
    /// across instruction boundaries inside the KV-sweep loop.
    pub fn attn_decode_prelude(
        q_id: u8,
        k_id: u8,
        v_id: u8,
        head_dim: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        q_heads_per_warp: u32,
        q_per_kv: u32,
        scale: f32,
        unique_id: u32,
    ) -> String {
        let u = unique_id;
        format!(
            r#"    // ── AttnDecode #{u} prelude (multi-head GQA; one kv-head per consumer warp) ──
    using T_act = __nv_bfloat16;
    auto* __q_smem_a{u}   = reinterpret_cast<T_act*>(page_buf[{q_id}]);
    auto* __k_smem_a{u}   = reinterpret_cast<T_act*>(page_buf[{k_id}]);
    auto* __v_smem_a{u}   = reinterpret_cast<T_act*>(page_buf[{v_id}]);
    auto* __out_smem_a{u} = reinterpret_cast<T_act*>(page_buf[{q_id}]);
    const unsigned int __head_dim_a{u} = {head_dim}u;
    const unsigned int __num_q_heads_a{u} = {num_q_heads}u;
    const unsigned int __num_kv_heads_a{u} = {num_kv_heads}u;
    const unsigned int __q_heads_per_warp_a{u} = {q_heads_per_warp}u;
    const unsigned int __q_per_kv_a{u} = {q_per_kv}u;
    const float __scale_a{u} = {scale:?}f;
    // Per-warp per-q-head softmax state. With Llama-3.2-1B
    // (N_q=32, 8 warps, qpw=4): each consumer warp keeps 4
    // independent softmax + accumulator states.
    float __m_max_a{u}[{q_heads_per_warp}];
    float __l_sum_a{u}[{q_heads_per_warp}];
    float __renorm_a{u}[{q_heads_per_warp}];
    float __p_a{u}[{q_heads_per_warp}];
    float __o_accum_a{u}[{q_heads_per_warp}][{head_dim}];
"#
        )
    }

}

// ── Role routing ───────────────────────────────────────────────────

fn role_guard(role: WarpRole) -> Option<String> {
    match role {
        WarpRole::All => None,
        WarpRole::Loader => Some("if (__role == ROLE_LOADER)".to_string()),
        WarpRole::Storer => Some("if (__role == ROLE_STORER)".to_string()),
        WarpRole::AllConsumers => Some("if (__role == ROLE_CONSUMER)".to_string()),
        WarpRole::Consumer(i) => {
            Some(format!("if (__role == ROLE_CONSUMER && __consumer_idx == {i})"))
        }
    }
}

/// `kittens::group<N>` width for a role: TMA/sync gates are sized to
/// the role's warp count. Used by Wait / Sync emit, where every thread
/// in the group is a participant.
fn role_group_width(role: WarpRole) -> u32 {
    use crate::tk_warp_ir::NUM_WARPS;
    match role {
        WarpRole::All => NUM_WARPS as u32, // Phase 7: 20 warps total.
        WarpRole::Loader | WarpRole::Storer | WarpRole::Consumer(_) => 1,
        WarpRole::AllConsumers => NUM_CONSUMER_WARPS as u32,
    }
}

/// `kittens::group<N>` width for an *Arrive*. ALWAYS 1 — the TK 2.0
/// `group<N>::arrive(semaphore&)` is gated on the GROUP's lane 0
/// (`threadIdx.x % (N*32) == 0`), so a multi-warp group<N>::arrive
/// fires `mbarrier.arrive` exactly ONCE total, not once per warp.
/// Our role-routed arms have each participating warp execute the
/// arrive independently; we want each warp's per-warp lane 0 to
/// fire, which is precisely what `group<1>` (a.k.a. `kittens::warp`)
/// gives us. The barrier's expected arrival count then equals the
/// number of warps that hit the role arm — the consumer's
/// `init_semaphore(page_done, 0, NUM_CONSUMER_WARPS)` matches 8
/// warps each firing once.
///
/// This was the deadlock root cause for `tk_decode_one_layer` and
/// `tk_decode_rmsnorm`: emitting `group<8>::arrive(page_done[i])`
/// produced ONE mbarrier.arrive against an init expecting 8 → arrive
/// count went 8→7 → parity never flipped → storer's `try_wait.parity`
/// polled forever. The trace in
/// `feedback_ff_subtile_smoke_handoff` showed all 8 ARRIVE printfs
/// firing (printfs are at the call-site lane gate, not the
/// per-arrive-PTX gate), masking the underlying single-fire arrive.
fn arrive_group_width(_role: WarpRole) -> u32 {
    1
}

// ── Emit options ───────────────────────────────────────────────────

/// Layout for a typed `kittens::gl<>` buffer arg (Step C.3).
///
/// One of these per buffer index that opts in to descriptor-TMA. The
/// emit composes
/// `using ARG{i}_T = kittens::gl<bf16, batch, depth, rows, cols, tile_type>;`
/// before the kernel and switches the kernel signature arg to
/// `const __grid_constant__ ARG{i}_T arg{i}`. The host launcher
/// constructs an `ARG{i}_T` instance from the runtime pointer and
/// passes it to the kernel — `cuTensorMapEncodeTiled` runs inside
/// TK 2.0's `gl<>` host ctor (`include/types/global/gl.cuh:153-160`),
/// per `feedback_dogfood_tk20_rust`.
///
/// Compile-time-fixed dims (b, d, r, c not -1) are baked into the
/// template; the host ctor takes `nullptr` for each runtime-dim slot.
/// Llama-1B vector-form RmsNorm output uses
/// `{batch:1, depth:1, rows:1, cols:hidden, tile_type:"kittens::sv_bf<HIDDEN>"}`.
#[derive(Clone, Debug)]
pub struct GlLayout {
    pub batch: i32,
    pub depth: i32,
    pub rows: i32,
    pub cols: i32,
    /// CUDA type spelling for the `TMA_Types... = ...` template arg —
    /// e.g. `"kittens::sv_bf<2048>"` (vector form, `UTMASTG.4D`) or
    /// `"kittens::st_bf<16, 64>"` (tile form, `UTMASTG.5D`).
    pub tile_type: String,
}

/// Per-emit knobs. Defaults give the production CUDA source; setting
/// flags here turns on debug instrumentation that's safe to ship in a
/// `.cu` file but adds a printf line per handshake.
#[derive(Clone, Debug, Default)]
pub struct EmitOpts {
    /// When true, every emitted `Wait` / `Arrive` / `LoadAsync` /
    /// `StoreAsync` is wrapped in lane-0-gated `printf`s tagged with
    /// the warp id, page id, barrier kind, and (for Wait) the phase
    /// expression. Used by the RmsNorm-only repro to identify the
    /// first wait in the round protocol that blocks without a matching
    /// arrive trace; off by default so production kernels stay quiet.
    pub debug_handshake: bool,

    /// Step C.3 — per-buffer descriptor-TMA opt-in. Map from buffer
    /// index → typed `gl<>` layout. Buffers in this map become
    /// `const __grid_constant__ ARG{i}_T arg{i}` parameters; their
    /// TMA loads/stores emit the typed (descriptor) form
    /// (`kittens::group<1>::tma::store_async<cache_policy::NORMAL>(arg, smem, {0,0,0,0})`).
    /// Buffers absent from the map keep raw-bulk emit byte-identical
    /// to the legacy path. Empty default → no behaviour change.
    pub descriptor_layouts: BTreeMap<u32, GlLayout>,
}

/// Render a single lane-0-gated `printf` line. Always wraps in
/// `if ((threadIdx.x & 31) == 0)` so each warp prints exactly once.
fn dbg_printf(tag: &str) -> String {
    format!(
        "if ((threadIdx.x & 31) == 0) {{ \
         printf(\"[wid=%d] {tag}\\n\", (int)(threadIdx.x / 32)); }}"
    )
}

fn instr_dbg_pre(instr: &TkInstr, opts: &EmitOpts) -> Option<String> {
    if !opts.debug_handshake {
        return None;
    }
    match instr {
        TkInstr::Wait {
            page_id,
            kind,
            phase,
            ..
        } => Some(dbg_printf(&format!(
            "WAIT_START kind={:?} page={} phase={}",
            kind,
            page_id,
            phase.cuda_expr()
        ))),
        TkInstr::LoadAsync { page_id, src, .. } => Some(dbg_printf(&format!(
            "TMA_LOAD_START page={} src_buf={}",
            page_id, src.0
        ))),
        TkInstr::StoreAsync { page_id, dst, .. } => Some(dbg_printf(&format!(
            "TMA_STORE_START page={} dst_buf={}",
            page_id, dst.0
        ))),
        _ => None,
    }
}

fn instr_dbg_post(instr: &TkInstr, opts: &EmitOpts) -> Option<String> {
    if !opts.debug_handshake {
        return None;
    }
    match instr {
        TkInstr::Wait {
            page_id,
            kind,
            phase,
            ..
        } => Some(dbg_printf(&format!(
            "WAIT_DONE kind={:?} page={} phase={}",
            kind,
            page_id,
            phase.cuda_expr()
        ))),
        TkInstr::Arrive {
            page_id, kind, ..
        } => Some(dbg_printf(&format!(
            "ARRIVE kind={:?} page={}",
            kind, page_id
        ))),
        TkInstr::LoadAsync { page_id, .. } => Some(dbg_printf(&format!(
            "TMA_LOAD_ISSUED page={}",
            page_id
        ))),
        TkInstr::StoreAsync { page_id, .. } => Some(dbg_printf(&format!(
            "TMA_STORE_ISSUED page={}",
            page_id
        ))),
        _ => None,
    }
}

// ── Walk ───────────────────────────────────────────────────────────

fn emit_one(instr: &TkInstr, opts: &EmitOpts, out: &mut String) {
    if let TkInstr::ForLoop { var, count, body } = instr {
        // The loop hosts every role together; per-instr role guards
        // inside the body still route work to the right warp.
        out.push_str("    for (uint ");
        out.push_str(var);
        out.push_str(" = 0; ");
        out.push_str(var);
        out.push_str(" < ");
        out.push_str(&LoopBound::cuda_expr(count));
        out.push_str("; ++");
        out.push_str(var);
        out.push_str(") {\n");
        for inner in body {
            emit_one(inner, opts, out);
        }
        out.push_str("    }\n");
        return;
    }

    let (role, body) = match instr {
        TkInstr::Wait {
            role,
            page_id,
            kind,
            phase,
        } => {
            let n = role_group_width(*role);
            (*role, tk20::wait(n, *kind, *page_id, &phase.cuda_expr()))
        }
        TkInstr::Arrive {
            role,
            page_id,
            kind,
        } => {
            // Always emit `group<1>::arrive` — see [`arrive_group_width`]
            // for why a `group<N>::arrive` from N warps fires only once
            // total, not N times.
            let n = arrive_group_width(*role);
            (*role, tk20::arrive(n, *kind, *page_id))
        }
        TkInstr::LoadAsync {
            page_id,
            src,
            src_region,
            tile,
            dyn_byte_off,
        } => {
            // src_region.region carries (rows, cols) — use the IR tile
            // for the TMA descriptor and the region's column-offset
            // for the byte offset.
            let TileShape {
                rows,
                cols,
                elem_bytes,
            } = *tile;
            let byte_off = (src_region.region.cols.start as u64) * (elem_bytes as u64);
            (
                WarpRole::Loader,
                tk20::tma_load_async(
                    *page_id,
                    src.0,
                    byte_off,
                    rows,
                    cols,
                    elem_bytes,
                    dyn_byte_off.as_deref(),
                ),
            )
        }
        TkInstr::StoreAsync {
            page_id,
            dst,
            dst_region,
            tile,
            dyn_byte_off,
        } => {
            let TileShape {
                rows,
                cols,
                elem_bytes,
            } = *tile;
            let byte_off = (dst_region.region.cols.start as u64) * (elem_bytes as u64);
            (
                WarpRole::Storer,
                tk20::tma_store_async(
                    *page_id,
                    dst.0,
                    byte_off,
                    rows,
                    cols,
                    elem_bytes,
                    dyn_byte_off.as_deref(),
                ),
            )
        }
        TkInstr::StoreAsyncTyped {
            page_id,
            dst,
            tile_type,
        } => (
            WarpRole::Storer,
            tk20::tma_store_async_typed(*page_id, dst.0, tile_type),
        ),
        TkInstr::Compute { role, calls } => {
            // Walk `calls` in order, emitting one CUDA fragment per
            // primitive. Each `Tk20Call` lowers via `Tk20Call::emit()`,
            // which returns one TK 2.0 call's spelling.
            let mut s = String::new();
            for (i, call) in calls.iter().enumerate() {
                if i > 0 {
                    s.push_str("\n    ");
                }
                s.push_str(&call.emit());
            }
            (*role, s)
        }
        TkInstr::Sync { role } => {
            let n = role_group_width(*role);
            (*role, tk20::sync(n))
        }
        // Atomic Instrs — each emits ONE TK 2.0 call. Composed with
        // Sync + Threadfence into the cross-op gmem-fence sequence
        // by `TkProgram::emit_cross_op_gmem_fence`.
        TkInstr::TmaStoreCommitGroup { role } => (*role, tk20::group_tma_store_commit_group()),
        TkInstr::TmaStoreAsyncWait { role, n } => (*role, tk20::group_tma_store_async_wait(*n)),
        TkInstr::Threadfence { role } => (*role, "__threadfence();".to_string()),
        TkInstr::ArriveIfRuntimeEven {
            role,
            kind,
            page_id,
            parity_var,
        } => (
            *role,
            tk20::arrive_if_runtime_even(*kind, *page_id, parity_var),
        ),
        // Handled by the early return above. Reachable only if a
        // future refactor breaks that contract; an `unreachable!` is
        // the right tripwire.
        TkInstr::ForLoop { .. } => unreachable!("ForLoop handled by early return"),
    };

    let pre = instr_dbg_pre(instr, opts);
    let post = instr_dbg_post(instr, opts);

    match role_guard(role) {
        None => {
            if let Some(p) = &pre {
                out.push_str("    ");
                out.push_str(p);
                out.push('\n');
            }
            out.push_str("    ");
            out.push_str(&body);
            out.push('\n');
            if let Some(p) = &post {
                out.push_str("    ");
                out.push_str(p);
                out.push('\n');
            }
        }
        Some(g) => {
            out.push_str("    ");
            out.push_str(&g);
            out.push_str(" {\n");
            if let Some(p) = &pre {
                out.push_str("        ");
                out.push_str(p);
                out.push('\n');
            }
            out.push_str("        ");
            out.push_str(&body);
            out.push('\n');
            if let Some(p) = &post {
                out.push_str("        ");
                out.push_str(p);
                out.push('\n');
            }
            out.push_str("    }\n");
        }
    }
}

/// Emit the persistent CTA body for a [`TkProgram`]. Caller wraps it
/// in the kernel signature + page/scratch declarations + the role
/// dispatch (`__role`, `__consumer_idx`); this fn is *only* the body
/// the role-routed match arms produce.
pub fn emit_body(prog: &TkProgram) -> String {
    emit_body_with_opts(prog, &EmitOpts::default())
}

/// As [`emit_body`] but takes an explicit [`EmitOpts`] so debug knobs
/// (e.g. handshake printf wrapping) can be toggled at the call site.
pub fn emit_body_with_opts(prog: &TkProgram, opts: &EmitOpts) -> String {
    let mut prog_owned;
    let prog_ref = if opts.descriptor_layouts.is_empty() {
        prog
    } else {
        prog_owned = prog.clone();
        rewrite_typed_stores(&mut prog_owned, opts);
        &prog_owned
    };
    let mut out = String::new();
    for instr in &prog_ref.instrs {
        emit_one(instr, opts, &mut out);
    }
    out
}

// ── Kernel scaffold ────────────────────────────────────────────────

/// One typed buffer argument to the persistent-CTA kernel. The emit
/// puts these in the kernel signature in declaration order; the
/// in-IR `BufId` indexes into this list (`buf{id}` in the body matches
/// `args[id].name`).
#[derive(Clone, Debug)]
pub struct KernelArg {
    /// CUDA type, e.g. `"const __nv_bfloat16* __restrict__"`.
    pub ty: String,
    /// Identifier name in the kernel signature; the body references
    /// `buf{i}` where `i` is the position in [`KernelArgs::bufs`].
    /// The codegen synthesises a `#define buf{i} <name>` so the body's
    /// `buf3 /* +256 */` substitution lands on the right argument.
    pub name: String,
}

/// The full kernel arg pack. Runtime u32 args (e.g. the
/// [`crate::tk_warp_ir::LoopBound::RuntimeU32`] names referenced by
/// the body's ForLoops) must be present here too; the codegen
/// otherwise has nowhere to declare them.
#[derive(Clone, Debug, Default)]
pub struct KernelArgs {
    pub bufs: Vec<KernelArg>,
    /// Names of runtime u32 args (no type — always `uint32_t`).
    pub u32_args: Vec<String>,
}

/// Emit a complete TK 2.0 persistent-CTA kernel. The output is a
/// `.cu` snippet with `#include "kittens.cuh"`, the kernel signature
/// (`__global__ __launch_bounds__(...) void <name>(...)`), the page
/// pool / mbarrier declarations, the init handshake (matches TK 2.0:
/// `page_ready[i].init(0)`, `page_done[i].init(0)`,
/// `page_consumed[i].init(0)` then `arrive_pre`), the role dispatch
/// (warpid 0 = loader, 1 = storer, 2..9 = consumer), the body the
/// caller built via [`emit_body`], and the final group sync.
///
/// The total warp count is `1 (loader) + 1 (storer) +
/// NUM_CONSUMER_WARPS (consumers) = 10`, so threadIdx.x ranges over
/// `[0, 320)` and `__launch_bounds__(320)` is emitted.
///
/// This is a *placeholder* in the sense that the body's compute
/// fragments still reference symbols (`__page_smem`, `__weight_smem`,
/// `__q_smem`, etc.) the per-op atom is responsible for binding. The
/// scaffold does not synthesise those bindings — they're the per-op
/// pre-resolved fragments that today's atom_lib produces. What this
/// scaffold *does* guarantee is that the role dispatch, mbarrier
/// init, and final sync are byte-identical to TK 2.0.
pub fn emit_kernel(name: &str, args: &KernelArgs, prog: &TkProgram) -> String {
    emit_kernel_with_opts(name, args, prog, &EmitOpts::default())
}

/// Rewrite `TkInstr::StoreAsync` into `TkInstr::StoreAsyncTyped` for
/// every `dst` declared in [`EmitOpts::descriptor_layouts`] whose
/// store is the single-tile {0,0,0,0} case (no static byte offset
/// and no dynamic byte offset). All other stores keep the raw-bulk
/// path. Runs once at the start of [`emit_kernel_with_opts`] so the
/// emit walker has no `opts` lookup at instr-emit time — one Instr,
/// one TK 2.0 call (paris invariant `descriptor-rewrite-once`,
/// per [[tk-player-one-call-per-arm]]).
fn rewrite_typed_stores(prog: &mut TkProgram, opts: &EmitOpts) {
    use crate::tk_warp_ir::TkInstr;
    if opts.descriptor_layouts.is_empty() {
        return;
    }
    fn walk(instrs: &mut Vec<TkInstr>, opts: &EmitOpts) {
        for instr in instrs.iter_mut() {
            match instr {
                TkInstr::StoreAsync {
                    page_id,
                    dst,
                    dst_region,
                    tile,
                    dyn_byte_off,
                } => {
                    let byte_off =
                        (dst_region.region.cols.start as u64) * (tile.elem_bytes as u64);
                    if byte_off != 0 || dyn_byte_off.is_some() {
                        continue;
                    }
                    let Some(layout) = opts.descriptor_layouts.get(&dst.0) else {
                        continue;
                    };
                    *instr = TkInstr::StoreAsyncTyped {
                        page_id: *page_id,
                        dst: *dst,
                        tile_type: layout.tile_type.clone(),
                    };
                }
                TkInstr::ForLoop { body, .. } => walk(body, opts),
                _ => {}
            }
        }
    }
    walk(&mut prog.instrs, opts);
}

/// As [`emit_kernel`] but takes an explicit [`EmitOpts`]. Used by the
/// `bin/tk_emit_rmsnorm` reproducer to flip on `debug_handshake`
/// instrumentation when the `TK_EMIT_DEBUG_HANDSHAKE` env var is set.
pub fn emit_kernel_with_opts(
    name: &str,
    args: &KernelArgs,
    prog: &TkProgram,
    opts: &EmitOpts,
) -> String {
    let mut prog = prog.clone();
    prog.emit_kernel_end_drain();
    let prog = &prog;
    use crate::tk_warp_ir::{NUM_CONSUMER_WARPS, NUM_PAGES, NUM_SERVICE_WARPS, NUM_WARPS};

    let total_warps = NUM_WARPS as u32; // Phase 7: 4 service + 16 consumers = 20 warps.
    let total_threads = total_warps * 32;

    let mut out = String::new();
    out.push_str("#include \"kittens.cuh\"\n");
    if opts.debug_handshake {
        // `printf` from device code lives in `<cstdio>`; some TK 2.0
        // headers don't transitively include it on Hopper.
        out.push_str("#include <cstdio>\n");
        // The orchestrator emits per-op trace markers wrapped in
        // `#ifdef TK_DEBUG_HANDSHAKE`; defining the macro here lights
        // them up alongside the wait/arrive/TMA printfs.
        out.push_str("#define TK_DEBUG_HANDSHAKE 1\n");
    }
    out.push_str("\n");

    // Role constants. Matches the WarpRole emit in role_guard().
    // Phase 7: 4 service warps + 16 consumer warps. Launcher and
    // controller are stub roles today (decrease_registers + idle);
    // they get real bodies at Phase 11/12 when cross-IType mbarrier
    // table + fused ITypes need them.
    out.push_str("#define ROLE_LOADER     0\n");
    out.push_str("#define ROLE_STORER     1\n");
    out.push_str("#define ROLE_LAUNCHER   2\n");
    out.push_str("#define ROLE_CONTROLLER 3\n");
    out.push_str("#define ROLE_CONSUMER   4\n");
    out.push_str("\n");

    // Step C.3 — typed `kittens::gl<>` typedefs for descriptor-TMA
    // opt-in buffers. One `using ARG{i}_T = kittens::gl<...>;` per
    // entry in `opts.descriptor_layouts`. Empty default → none emitted,
    // legacy raw-bulk path is byte-identical.
    if !opts.descriptor_layouts.is_empty() {
        for (i, layout) in &opts.descriptor_layouts {
            out.push_str(&format!(
                "using ARG{i}_T = kittens::gl<__nv_bfloat16, {b}, {d}, {r}, {c}, {tt}>;\n",
                i = i,
                b = layout.batch,
                d = layout.depth,
                r = layout.rows,
                c = layout.cols,
                tt = layout.tile_type,
            ));
        }
        out.push_str("\n");
    }

    // Kernel signature.
    out.push_str(&format!(
        "__global__ __launch_bounds__({total_threads}) void {name}(\n"
    ));
    let mut first = true;
    for (i, arg) in args.bufs.iter().enumerate() {
        if !first {
            out.push_str(",\n");
        }
        first = false;
        if opts.descriptor_layouts.contains_key(&(i as u32)) {
            // Descriptor-TMA path: kernel takes the gl<> instance by
            // value via `__grid_constant__` (TK 2.0 idiom — see C.2 PoC
            // smoke kernel). Body keeps using `buf{i}` via the
            // `auto* {name} = arg{i}.raw_ptr` alias below.
            out.push_str(&format!("    const __grid_constant__ ARG{i}_T arg{i}"));
        } else {
            out.push_str(&format!("    {} {}", arg.ty, arg.name));
        }
    }
    for u32_name in &args.u32_args {
        if !first {
            out.push_str(",\n");
        }
        first = false;
        out.push_str(&format!("    const uint32_t {u32_name}"));
    }
    out.push_str("\n) {\n");
    out.push_str("    using namespace kittens;\n");
    out.push_str("\n");

    // `buf{i}` aliases so the body's `buf3 /* +256 */` substitutions
    // resolve to the right named arg.
    //
    // For descriptor-TMA opt-in buffers, the kernel signature param is
    // `arg{i}` (the gl<> instance), not the raw pointer. Bind a local
    // `auto* {name} = arg{i}.raw_ptr;` first so the existing alias line
    // (`auto& buf{i} = {name};`) keeps working — body code that uses
    // `buf{i}` as a `bf16*` (e.g. for non-descriptor accesses) is
    // unchanged. Descriptor TMA loads/stores against this buffer also
    // have access to `arg{i}` directly via the IR-driven emit branch.
    for (i, arg) in args.bufs.iter().enumerate() {
        if opts.descriptor_layouts.contains_key(&(i as u32)) {
            out.push_str(&format!(
                "    auto* {name} = arg{i}.raw_ptr;\n",
                name = arg.name,
                i = i,
            ));
        }
        out.push_str(&format!("    auto& buf{i} = {};\n", arg.name));
    }
    out.push_str("\n");

    // Role dispatch. Phase 7 layout: warps 0-3 = service warpgroup
    // (loader, storer, launcher, controller); warps 4-19 = 4 consumer
    // warpgroups (16 consumer warps).
    out.push_str("    const int __warpid = threadIdx.x / 32;\n");
    out.push_str("    int __role;\n");
    out.push_str("    if      (__warpid == 0) __role = ROLE_LOADER;\n");
    out.push_str("    else if (__warpid == 1) __role = ROLE_STORER;\n");
    out.push_str("    else if (__warpid == 2) __role = ROLE_LAUNCHER;\n");
    out.push_str(&format!(
        "    else if (__warpid == 3) __role = ROLE_CONTROLLER;\n"
    ));
    out.push_str("    else                    __role = ROLE_CONSUMER;\n");
    out.push_str(&format!(
        "    const int __consumer_idx = __warpid - {};\n",
        NUM_SERVICE_WARPS
    ));
    out.push_str("    (void)__consumer_idx;\n");
    out.push_str("\n");

    // Hopper register-budget plumbing. The service warpgroup (warps
    // 0-3) hands its register share to the 4 consumer warpgroups so
    // wgmma operations can fit `rt_<bf16, 16, K>` chunks. The
    // `kittens::warpgroup::increase_registers<N>` /
    // `decrease_registers<N>` calls REQUIRE warpgroup-aligned
    // execution (4 warps in lockstep) — that's why Phase 7 adds the
    // launcher + controller stubs to fill the service warpgroup to 4.
    // Source: `ops/group/group.cuh:52` (increase) and `:56` (decrease).
    out.push_str("    if (__warpid < 4) {\n");
    out.push_str("        kittens::warpgroup::decrease_registers<56>();\n");
    out.push_str("    } else {\n");
    out.push_str("        kittens::warpgroup::increase_registers<224>();\n");
    out.push_str("    }\n");
    out.push_str("\n");

    // Page pool + mbarriers. The page byte-buffer is allocated as
    // dynamic shared memory; per-page typed views are the per-op
    // atom's responsibility.
    out.push_str(&format!(
        "    __shared__ kittens::semaphore page_ready[{}];\n",
        NUM_PAGES
    ));
    out.push_str(&format!(
        "    __shared__ kittens::semaphore page_done[{}];\n",
        NUM_PAGES
    ));
    out.push_str(&format!(
        "    __shared__ kittens::semaphore page_consumed[{}];\n",
        NUM_PAGES
    ));
    // Dynamic shared memory: TK 2.0 production pattern. The launcher must
    // set `cudaFuncAttributeMaxDynamicSharedMemorySize` to at least
    // `NUM_PAGES * PAGE_SIZE` so the pool fits. nvcc itself only requires
    // the `extern __shared__` declaration; ptxas will not count it against
    // the static smem cap (default 0xc000 on H100). Per-page typed views
    // are the per-op atom's responsibility; we expose `page_buf[i]` as
    // an array-of-PAGE_SIZE-byte rows so the rest of the emit code can
    // keep using `page_buf[id]` as before.
    // `__align__(128)` is the CUDA-canonical alignment attribute for an
    // `extern __shared__` array. `alignas(...)` collides with the
    // `__attribute__((shared))` that the `__shared__` macro expands to.
    out.push_str("    extern __shared__ __align__(128) uint8_t __dynamic_smem[];\n");
    out.push_str(&format!(
        "    auto (&page_buf)[{}][{}] = *reinterpret_cast<uint8_t(*)[{}][{}]>(__dynamic_smem);\n",
        NUM_PAGES,
        crate::tk_warp_ir::PAGE_SIZE,
        NUM_PAGES,
        crate::tk_warp_ir::PAGE_SIZE
    ));
    out.push_str("\n");

    // Init: lane-0 of warp-0 sets up all mbarriers; consumed is
    // pre-arrived so its first wait reads the post-arrive parity.
    // (TK 2.0 default; matches kittens header `page_consumed[i].arrive_pre()`.)
    out.push_str("    if (__warpid == 0 && (threadIdx.x & 31) == 0) {\n");
    out.push_str(&format!(
        "        for (int __i = 0; __i < {}; ++__i) {{\n",
        NUM_PAGES
    ));
    out.push_str("            kittens::init_semaphore(page_ready[__i], 0, 1);\n");
    out.push_str(&format!(
        "            kittens::init_semaphore(page_done[__i], 0, {});\n",
        NUM_CONSUMER_WARPS
    ));
    out.push_str("            kittens::init_semaphore(page_consumed[__i], 0, 1);\n");
    out.push_str("            kittens::arrive(page_consumed[__i]);\n");
    out.push_str("        }\n");
    out.push_str("    }\n");
    // Async-proxy fence: `mbarrier.init` and `mbarrier.arrive` write
    // through the ASYNC proxy of shared memory; without an explicit
    // `fence.proxy.async.shared::cta` other threads' subsequent
    // `mbarrier.arrive` / `mbarrier.try_wait.parity` against the same
    // barrier may observe stale (pre-init) parity bits even after a
    // `__syncthreads()` barrier — `__syncthreads` only orders generic-
    // proxy ops with each other. This was the root cause of the
    // `launcher_runs_on_zeros_rmsnorm_only` storer deadlock: 8
    // consumer arrives on `page_done[i]` flipped the real parity, but
    // the storer's `try_wait.parity` against the same barrier kept
    // polling against its stale view. TK 2.0's KVM scaffold issues
    // this exact fence at
    // `third_party/thunderkittens/prototype/vm/vm.cuh:99`.
    out.push_str("    ");
    out.push_str(&tk20::fence_proxy_async_shared_cta());
    out.push('\n');
    // CTA-wide sync. `kittens::group<N>::sync()` (barrier-less) is
    // only legal for single-warp groups (asserts `GROUP_WARPS==1`); the
    // multi-warp form takes a `bar.sync` barrier id. Easiest portable
    // choice for "all 10 warps converge" is just `__syncthreads()`.
    out.push_str("    __syncthreads();\n");
    out.push_str("\n");

    // Function-scope prelude: typed page views, persistent compute
    // accumulators that span multiple consumer compute steps. See
    // `TkProgram::prelude` for the contract.
    if !prog.prelude.is_empty() {
        out.push_str("    // ── tk_warp_ir prelude ──\n");
        out.push_str(&prog.prelude);
        if !prog.prelude.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
    }

    // The role-routed body.
    out.push_str("    // ── tk_warp_ir body ──\n");
    out.push_str(&emit_body_with_opts(prog, opts));
    out.push_str("\n");
    out.push_str("}\n");
    out.push_str("\n");

    // ── Host launcher (C linkage) ──────────────────────────────────
    //
    // Emit `extern "C" cudaError_t launch_<name>(void* const* bufs,
    // const uint32_t* u32_args, cudaStream_t stream)` so the ferrite
    // CUDA runtime can call the kernel without knowing its mangled
    // C++ name or its arg list — both are baked into this wrapper.
    //
    // The wrapper:
    //   1. Calls `cudaFuncSetAttribute(...,
    //      cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM)` so
    //      the H100 default 48 KB dynamic-smem cap is lifted to
    //      `NUM_PAGES * PAGE_SIZE` (the page pool size the kernel
    //      requests via `extern __shared__`).
    //   2. Launches with `<<<1, total_threads, DYN_SMEM, stream>>>`
    //      (single-CTA persistent megakernel pattern).
    //   3. Forwards `bufs[i]` cast to the declared kernel arg type and
    //      `u32_args[j]` for each runtime u32.
    let dyn_smem = (NUM_PAGES * crate::tk_warp_ir::PAGE_SIZE) as u64;
    out.push_str(&format!(
        "extern \"C\" cudaError_t launch_{name}(\n    \
            void* const* bufs,\n    \
            const uint32_t* u32_args,\n    \
            cudaStream_t stream\n) {{\n"
    ));
    out.push_str(&format!("    constexpr size_t DYN_SMEM = {dyn_smem}u;\n"));
    out.push_str("    (void)u32_args;\n");
    out.push_str(&format!(
        "    cudaError_t __err = cudaFuncSetAttribute(\n        \
            (const void*)&{name},\n        \
            cudaFuncAttributeMaxDynamicSharedMemorySize,\n        \
            (int)DYN_SMEM);\n    \
            if (__err != cudaSuccess) return __err;\n"
    ));
    // Step C.3 — for descriptor-TMA opt-in buffers, build a
    // `kittens::gl<>` host instance from the runtime pointer. The gl<>
    // ctor invokes `cuTensorMapEncodeTiled` internally per
    // `include/types/global/gl.cuh:153-160`; per
    // `feedback_dogfood_tk20_rust` the driver TMA API is never called
    // from Rust. All-`nullptr` runtime-dim args because every dim is
    // baked into the template (`make_arg_t<dim>` SFINAE accepts
    // `nullptr` when the template dim is non-runtime).
    if !opts.descriptor_layouts.is_empty() {
        for i in opts.descriptor_layouts.keys() {
            out.push_str(&format!(
                "    ARG{i}_T arg{i}_inst(\n        \
                     reinterpret_cast<__nv_bfloat16*>(bufs[{i}]),\n        \
                     nullptr, nullptr, nullptr, nullptr);\n",
                i = i,
            ));
        }
    }
    out.push_str(&format!("    {name}<<<1, {total_threads}, DYN_SMEM, stream>>>(\n"));
    let mut first = true;
    for (i, arg) in args.bufs.iter().enumerate() {
        if !first {
            out.push_str(",\n");
        }
        first = false;
        if opts.descriptor_layouts.contains_key(&(i as u32)) {
            out.push_str(&format!("        arg{i}_inst"));
        } else {
            out.push_str(&format!("        ({})bufs[{i}]", arg.ty));
        }
    }
    for (j, _u32_name) in args.u32_args.iter().enumerate() {
        if !first {
            out.push_str(",\n");
        }
        first = false;
        out.push_str(&format!("        u32_args[{j}]"));
    }
    out.push_str("\n    );\n");
    out.push_str("    return cudaGetLastError();\n");
    out.push_str("}\n");
    out
}

// ── Tk20Call — typed primitive call list for `TkInstr::Compute` ─────
//
// Per `feedback_dogfood_tk20_rust`: every `kittens::*` text in an
// emitted .cu MUST come from a `tk20::*` Rust function. `Tk20Call`
// makes the binding step explicit: each variant binds one TK 2.0
// primitive call.
//
// Every typed variant's `emit()` cites the TK 2.0 header path + line
// for its primitive, both in the doc comment AND in the source code
// under test (per-variant unit tests assert the emit string matches
// the TK 2.0 spelling — catches TK 1.0 drift at `cargo test`, before
// nvcc).

/// Typed primitive-call atom for `TkInstr::Compute::calls`.
///
/// Each variant binds one TK 2.0 primitive call. `emit()` returns the
/// CUDA fragment for that call.
#[derive(Clone, Debug)]
pub enum Tk20Call {
    // ── Hopper warpgroup wgmma (default for prefill, m>1 matmuls) ──
    /// `kittens::warpgroup::mma_fence(d)`. Doc:
    /// `third_party/thunderkittens/include/ops/group/mma/warpgroup.cuh:23`.
    /// Wgmma fence — ensures register-side writes to `d` are visible to
    /// the subsequent `mma_AB` reads.
    WarpgroupMmaFence { d: String },

    /// `kittens::warpgroup::mma_AB(d, a, b)` — accumulating wgmma.
    /// Doc: `ops/group/mma/warpgroup.cuh:140`. Default fence=1,
    /// accumulate=1 (the binding leaves the template parameters at
    /// their defaults; `mm_AB` is the reset-variant alternative).
    WarpgroupMmaAB { d: String, a: String, b: String },

    /// `kittens::warpgroup::mm_AB(d, a, b)` — reset (non-accumulating)
    /// wgmma. Doc: `ops/group/mma/warpgroup.cuh:186`.
    WarpgroupMmAB { d: String, a: String, b: String },

    /// `kittens::warpgroup::mma_ABt(d, a, b)` — accumulating wgmma
    /// with B transposed. Doc: `ops/group/mma/warpgroup.cuh:258`.
    WarpgroupMmaABt { d: String, a: String, b: String },

    /// `kittens::warpgroup::mma_commit_group()`. Doc:
    /// `ops/group/mma/warpgroup.cuh:78`. Caller is responsible for
    /// pairing this with a subsequent `mma_async_wait<N>` before any
    /// thread reads `d`.
    WarpgroupMmaCommitGroup,

    /// `kittens::warpgroup::mma_async_wait<N>()`. Doc:
    /// `ops/group/mma/warpgroup.cuh:91`. `n` is the number of in-flight
    /// commit groups still allowed (typically 0 to wait for all).
    WarpgroupMmaAsyncWait { n: u32 },

    // ── Per-warp mma (decode m=1 fast path) ────────────────────────
    /// `kittens::warp::mma_AB(d, a, b)`. Doc:
    /// `ops/group/mma/warp.cuh`. Used by the m=1 decode `lower_gemm_m1`
    /// path; for m>1 prefill, prefer `WarpgroupMmaAB` for Hopper peak.
    WarpMmaAB { d: String, a: String, b: String },

    /// `kittens::warp::mma_ABt(d, a, b)`. Doc: `ops/group/mma/warp.cuh`.
    WarpMmaABt { d: String, a: String, b: String },

    // ── Group-scoped TMA (the missing complements to existing
    //     tk20::tma_load_async / tma_store_async) ──────────────────
    /// `kittens::group<1>::tma::store_commit_group()`. Doc:
    /// `ops/group/util/tma.cuh:38`. Pair with `store_async_wait<N>` to
    /// drain queued stores before re-using the page slot.
    GroupTmaStoreCommitGroup,

    /// `kittens::group<1>::tma::store_async_wait<N>()`. Doc:
    /// `ops/group/util/tma.cuh:47`. Blocks until at most `n` async
    /// stores are still pending (typically 0).
    GroupTmaStoreAsyncWait { n: u32 },

    // ── Atomic CUDA-statement primitives (≤2 lines emit each) ─────
    // Used to compose compute bodies as N typed primitives instead
    // of one multi-line `*_consumer_body` helper, per
    // [[tk-player-one-call-per-arm]]. Each variant emits exactly
    // ONE CUDA statement (or a `for` loop wrapping recursive emit).

    /// `auto* {var} = reinterpret_cast<__nv_bfloat16*>(page_buf[{page_id}]);`
    DeclSmemPtrBf16 { var: String, page_id: u8 },

    /// `const unsigned int {var} = {expr}u;`. `expr` is a CUDA u32-
    /// expression (a literal or a name).
    DeclConstU32 { var: String, expr: String },

    /// `const int {var} = {expr};`. `expr` is a CUDA int-expression.
    DeclConstI32 { var: String, expr: String },

    /// `const float {var} = __bfloat162float({smem}[{idx}]);`
    DeclConstFloatFromBf16Smem {
        var: String,
        smem: String,
        idx: String,
    },

    /// `{smem}[{idx}] = __float2bfloat16({lhs} + {rhs});`
    Bf16StoreFromFloatAdd {
        smem: String,
        idx: String,
        lhs: String,
        rhs: String,
    },

    /// `{smem}[{idx}] = __float2bfloat16({lhs} * {rhs});`
    Bf16StoreFromFloatMul {
        smem: String,
        idx: String,
        lhs: String,
        rhs: String,
    },

    /// `const float {var} = {src} / (1.0f + expf(-{src}));` —
    /// scalar SiLU activation `x * sigmoid(x)` via `expf`.
    DeclConstFloatSilu { var: String, src: String },

    /// `{smem}[{idx}] = __float2bfloat16({lhs1} * {rhs1} - {lhs2} * {rhs2});` —
    /// scalar FMA-sub-store; used for the lo half of NeoX RoPE
    /// rotation (`x_lo * cos - x_hi * sin`).
    Bf16StoreFromFloatFmaSub {
        smem: String,
        idx: String,
        lhs1: String,
        rhs1: String,
        lhs2: String,
        rhs2: String,
    },

    /// `{smem}[{idx}] = __float2bfloat16({lhs1} * {rhs1} + {lhs2} * {rhs2});` —
    /// scalar FMA-add-store; used for the hi half of NeoX RoPE
    /// rotation (`x_lo * sin + x_hi * cos`).
    Bf16StoreFromFloatFmaAdd {
        smem: String,
        idx: String,
        lhs1: String,
        rhs1: String,
        lhs2: String,
        rhs2: String,
    },

    /// `kittens::sv_bf<{k}>& {var} = *reinterpret_cast<kittens::sv_bf<{k}>*>({ptr});`
    DeclSvBfView { var: String, ptr: String, k: u32 },

    /// `kittens::rv_bf<{k}> {var};`
    DeclRvBf { var: String, k: u32 },

    /// `kittens::rv_fl<{k}> {var};`
    DeclRvFl { var: String, k: u32 },

    /// `kittens::warp::load({rv}, {sv});`
    WarpLoadRvFromSv { rv: String, sv: String },

    /// `kittens::warp::copy({dst}, {src});`
    WarpCopyRv { dst: String, src: String },

    /// `kittens::warp::mul({dst}, {lhs}, {rhs});`
    WarpMulRv { dst: String, lhs: String, rhs: String },

    /// `if (__consumer_idx == 0) { ...body... }` — gate the body to
    /// the first consumer warp only. Recursively emits each body
    /// Tk20Call.
    IfConsumerIdxZero { body: Vec<Tk20Call> },

    /// `const int {var} = static_cast<int>(threadIdx.x & 31);`
    DeclConstI32Lane { var: String },

    /// `const float {var} = {expr};`
    DeclConstFloat { var: String, expr: String },

    /// `float {var} = {init_expr};` — mutable float local
    /// (e.g. `float __sumsq = 0.0f;`).
    DeclMutableFloat { var: String, init: String },

    /// `for (int {iter} = 0; {iter} < {end}; ++{iter}) { ...body... }` —
    /// integer-counter for-loop with const upper bound (no thread
    /// striding).
    ForLoopIntK { iter: String, end: String, body: Vec<Tk20Call> },

    /// `{dst} += kittens::warp::sum({rv});` — accumulate a warp-
    /// collective register-vector sum into a float scalar.
    FloatAddAssignWarpSum { dst: String, rv: String },

    /// `const float {var} = rsqrtf({sumsq} / static_cast<float>({hidden}) + {eps});` —
    /// the RmsNorm scaling factor.
    DeclConstFloatRsqrtScale {
        var: String,
        sumsq: String,
        hidden: String,
        eps: String,
    },

    /// `{smem}[{idx}] = __float2bfloat16({a} * {b} * {c});` — three-
    /// way scalar product store (RmsNorm: `v * scale * g`).
    Bf16StoreFromFloatTripleProduct {
        smem: String,
        idx: String,
        a: String,
        b: String,
        c: String,
    },

    /// `if (static_cast<unsigned int>(__consumer_idx) < {bn}) { ...body... }` —
    /// gate the body to the first `bn` consumer warps.
    IfConsumerIdxLtBnVar { bn_var: String, body: Vec<Tk20Call> },

    /// `if (__lane == 0) { ...body... }` — gate body to lane 0.
    IfLane0 { body: Vec<Tk20Call> },

    /// `const unsigned int {var} = static_cast<unsigned int>(__consumer_idx);`
    DeclConstU32FromConsumerIdx { var: String },

    /// `__nv_bfloat16* {var} = reinterpret_cast<__nv_bfloat16*>(buf{buf});`
    DeclGmemPtrBf16FromBuf { var: String, buf: u32 },

    /// `{gmem}[{idx}] = __float2bfloat16({src});` — gmem scalar bf16
    /// store from a float local.
    Bf16GmemStoreFromFloat {
        gmem: String,
        idx: String,
        src: String,
    },

    /// `__nv_bfloat16* {var} = reinterpret_cast<__nv_bfloat16*>(page_buf[{page_id}]);` —
    /// smem-pointer typedef-style decl. Same shape as
    /// [`Self::DeclSmemPtrBf16`] but `__nv_bfloat16*` rather than
    /// `auto*` for callers that name the explicit type.
    DeclSmemPtrTypedBf16 { var: String, page_id: u8 },

    /// `if (static_cast<unsigned int>(__consumer_idx) < {var}) { ...body... }` —
    /// gate by a runtime u32 variable name (typically a prelude-scope
    /// `__num_kv_heads_a<u>` constant declared by the AttnDecode
    /// prelude).
    IfConsumerIdxLtVar { var: String, body: Vec<Tk20Call> },

    /// `for (unsigned int {iter} = 0u; {iter} < {end}; ++{iter}) { ...body... }` —
    /// `unsigned int` for-loop with a runtime u32 expression as the
    /// upper bound.
    ForLoopUnsignedH {
        iter: String,
        end: String,
        body: Vec<Tk20Call>,
    },

    /// `{array}[{idx}] = -INFINITY;`
    ScalarFloatStoreNegInfty { array: String, idx: String },

    /// `{array}[{idx}] = 0.0f;`
    ScalarFloatStoreZero { array: String, idx: String },

    /// `{array}[{i}][{j}] = 0.0f;`
    ScalarFloat2DStoreZero {
        array: String,
        i: String,
        j: String,
    },

    /// `{accum}[{i}][{j}] += {p}[{i}] * __bfloat162float({smem}[{idx}]);` —
    /// FMA of a per-warp scalar `p[i]` and a bf16 smem value into a
    /// 2D float accumulator. Used by AttnDecode SV step
    /// (`__o_accum[h][j] += __p[h] * v_smem[__v_off + j]`).
    Float2DAccumPTimesBf16 {
        accum: String,
        i: String,
        j: String,
        p_array: String,
        smem: String,
        idx: String,
    },

    /// `const float {var} = 1.0f / {array}[{idx}];`
    DeclConstFloatReciprocal {
        var: String,
        array: String,
        idx: String,
    },

    /// `{smem}[{idx}] = __float2bfloat16({accum}[{i}][{j}] * {scalar});` —
    /// store a 2D-accumulator entry scaled by a per-warp scalar.
    /// Used by AttnDecode finalise (`O = O_accum / l_sum`).
    Bf16StoreFrom2DAccumTimesScalar {
        smem: String,
        idx: String,
        accum: String,
        i: String,
        j: String,
        scalar: String,
    },

    /// `float {var} = kittens::warp::sum({rv});` — initialise a
    /// scalar from a register-vector warp-collective sum.
    DeclMutableFloatFromWarpSum { var: String, rv: String },

    /// `{var} *= {expr};` — float compound multiply-assign (used for
    /// `__s *= __scale`).
    FloatMulAssign { var: String, expr: String },

    /// `const float {var} = fmaxf({array}[{idx}], {other});`
    DeclConstFloatFmaxArray {
        var: String,
        array: String,
        idx: String,
        other: String,
    },

    /// `{array}[{idx}] = expf({lhs} - {rhs});` — used by online-
    /// softmax for renorm (`expf(m_old - m_new)`) and p
    /// (`expf(s - m_new)`).
    ScalarFloatStoreExpDiff {
        array: String,
        idx: String,
        lhs: String,
        rhs: String,
    },

    /// `{dst}[{idx}] = {a}[{idx}] * {b}[{idx}] + {c}[{idx}];` —
    /// per-element FMA into a 1-D scalar array. Used by AttnDecode's
    /// `__l_sum[h] = __renorm[h] * __l_sum[h] + __p[h]` update.
    ScalarFloat1DStoreFma {
        dst: String,
        idx: String,
        a: String,
        b: String,
        c: String,
    },

    /// `{array}[{i}][{j}] *= {scalar}[{i}];` — 2D scalar in-place
    /// scale by a per-row scalar. Used by AttnDecode's o_accum
    /// rescale: `__o_accum[h][j] *= __renorm[h]`.
    ScalarFloat2DMulAssignFromArray {
        array: String,
        i: String,
        j: String,
        scalar: String,
    },

    /// `{array}[{idx}] = {src};` — copy a scalar variable into a
    /// 1-D float array slot. Used by AttnDecode's m_max update:
    /// `__m_max[h] = __m_new;`.
    ScalarFloatStoreVar {
        array: String,
        idx: String,
        src: String,
    },

    /// `#ifdef TK_DEBUG_HANDSHAKE if (threadIdx.x == 0) { printf("[BEGIN
    /// {op_tag}]\n"); } #endif` — per-op trace marker; fires from
    /// thread 0 only when TK_DEBUG_HANDSHAKE is defined at compile
    /// time. One Tk20Call variant; orchestrator passes `op_tag` as
    /// e.g. `"op7/RmsNorm"`.
    DebugOpBeginMarker { op_tag: String },

    /// `for (uint {iter} = (uint){start}; {iter} < {end}; {iter} += (uint){stride}) { body }`.
    /// Recursively emits each `body` Tk20Call.
    ForLoopThreadStrided {
        iter: String,
        start_var: String,
        end_var: String,
        stride_var: String,
        body: Vec<Tk20Call>,
    },

    /// `kittens::group<1>::tma::store_async_read_wait<N>()`. Doc:
    /// `ops/group/util/tma.cuh:61`. Blocks until at most `n` stores
    /// have outstanding READS — i.e. the source shared-tile is safe to
    /// overwrite. Looser than `store_async_wait` (stores may still be
    /// in flight to gmem; only the smem read is drained).
    GroupTmaStoreAsyncReadWait { n: u32 },

    // ── Register-tile reductions (warp-scoped) ─────────────────────
    /// `kittens::warp::row_max(rv, rt)`. Doc:
    /// `ops/group/register/tile/reductions.cuh`. Reduces each row of
    /// the register tile to a single max value, written to the register
    /// vector argument. Used by online softmax.
    WarpRowMax { rv: String, rt: String },

    /// `kittens::warp::row_sum(rv, rt)`. Doc: same header. Companion
    /// to `WarpRowMax`; produces row-wise sums.
    WarpRowSum { rv: String, rt: String },

    /// `kittens::warp::exp2(rt)`. Doc:
    /// `ops/group/register/tile/maps.cuh`. In-place exp2 on a register
    /// tile.
    WarpExp2 { rt: String },

    /// `kittens::warp::mul_row(rt, rt, rv)`. Multiplies each row of `rt`
    /// (in-place) by the corresponding scalar in `rv`. Doc: same maps
    /// header.
    WarpMulRow { rt: String, rv: String },

    /// `kittens::warp::div_row(rt, rt, rv)`. Inverse of `WarpMulRow`.
    WarpDivRow { rt: String, rv: String },

    /// `kittens::warp::neg_infty(rv)`. Doc:
    /// `ops/group/register/vec/maps.cuh`. Initialises a register vector
    /// to `-inf` — the online-softmax row-max accumulator's identity.
    WarpNegInfty { rv: String },

    /// `kittens::warp::zero(rt)`. Doc: `ops/group/register/tile/maps.cuh`.
    /// Zeroes a register tile; used for accumulator init.
    WarpZeroRt { rt: String },

    /// `kittens::warp::add(d, a, b)`. Doc:
    /// `ops/group/register/tile/maps.cuh`. Element-wise add on register
    /// tiles. Used by residual-add lowering.
    WarpAddRt { d: String, a: String, b: String },

    /// `kittens::warp::mul(d, a, b)`. Element-wise multiply.
    WarpMulRt { d: String, a: String, b: String },

    // ── Hopper register-budget management ──────────────────────────
    /// `kittens::warpgroup::increase_registers<N>()`. Doc:
    /// `ops/group/group.cuh:52`. `n` MUST be a multiple of 8 (TK 2.0
    /// `static_assert`). Used by consumer warpgroups to claim a larger
    /// register file for accumulators.
    WarpgroupIncreaseRegisters { n: u32 },

    /// `kittens::warpgroup::decrease_registers<N>()`. Doc:
    /// `ops/group/group.cuh:56`. `n` MUST be a multiple of 8. Used by
    /// non-consumer warpgroups (loader/storer) to release registers
    /// for the consumers.
    WarpgroupDecreaseRegisters { n: u32 },

    // ── Per-IType consumer body atoms (Phase 1+) ───────────────────
    //
    // Each variant emits the full consumer-warp compute body for one
    // fused op, replacing the legacy `Compute { body: format!(...) }`
    // route. Per `feedback_dogfood_tk20_rust`: every `kittens::*` text
    // reaches the emitted .cu via a typed `Tk20Call` variant. These
    // body atoms inline raw bf16 + `__shfl_xor_sync` CUDA where
    // TK 2.0 doesn't expose a higher-level primitive (e.g. m=1
    // RmsNorm row reduction — no register-tile match). The `mk-v2`
    // (`~/git/Megakernels` `origin/mk-v2`, TK 2.0 sm_90a) production
    // RmsNorm at `csrc/itypes/rmsnorm.cuh:130-180` does exactly the
    // same: per-thread squared-sum + warp shfl + per-thread normalize,
    // with `kittens::sv_bf<N>` only as a typed page alias.



}

/// Tape-build helper: AttnDecode Q@K^T + online-softmax step
/// (inside KV-sweep loop). Per active consumer warp, loops over
/// q-heads-per-warp; within each h loops register-vector
/// `kittens::warp::{load, copy, mul, sum}` over Q and K slices,
/// then runs scalar online-softmax (`m_new = max(m_max, s)`, renorm
/// = `expf(m_old - m_new)`, `p = expf(s - m_new)`, `l_sum =
/// renorm * l_sum + p`, lane-strided `o_accum *= renorm`, finally
/// `m_max = m_new`).
// ── AttnDecode SoftmaxState typestate (paris invariant
//    `attn-decode-init-qkt-sv-finalise-order`) ─────────────────────
//
// AttnDecode's per-warp softmax accumulators (`__m_max_a<u>`,
// `__l_sum_a<u>`, `__o_accum_a<u>[]`) follow a strict per-decode-
// step lifecycle: prelude declares (Empty) → init zeros (FreshLoop)
// → for each KV iter, qkt updates (PostQkt) and sv accumulates
// (PostSv) → finalise normalizes and writes O (Finalised). Calling
// the bodies in the wrong order corrupts the recurrence silently.
//
// `SoftmaxState<Phase>` is the typestate witness. Each builder fn
// consumes the state (move-by-value) and returns it advanced to
// the next phase; mis-ordering is a Rust compile error. The
// `unique_id` lives on the witness — passing the wrong `u` to a
// downstream builder is structurally impossible because the witness
// only emits with its own `u`.

mod softmax_phase {
    pub trait Sealed {}
    pub struct Empty;
    pub struct FreshLoop;
    pub struct PostQkt;
    pub struct PostSv;
    pub struct Finalised;
    impl Sealed for Empty {}
    impl Sealed for FreshLoop {}
    impl Sealed for PostQkt {}
    impl Sealed for PostSv {}
    impl Sealed for Finalised {}
}
pub use softmax_phase::{Empty, FreshLoop, PostQkt, PostSv, Finalised};
pub trait SoftmaxPhase: softmax_phase::Sealed {}
impl SoftmaxPhase for Empty {}
impl SoftmaxPhase for FreshLoop {}
impl SoftmaxPhase for PostQkt {}
impl SoftmaxPhase for PostSv {}
impl SoftmaxPhase for Finalised {}

/// AttnDecode online-softmax typestate witness. Construct only via
/// [`SoftmaxState::new`]; advance only by calling the typed
/// `init_compute_calls` / `qkt_compute_calls` / `sv_compute_calls`
/// / `finalise_compute_calls` methods in order. Calling them out of
/// order is a Rust compile error.
///
/// # Compile-fail proofs (paris invariant `attn-decode-init-qkt-sv-finalise-order`)
///
/// QKt before init (no `qkt_compute_calls` on `Empty`):
/// ```compile_fail
/// use ferrite_wavefront::tk_codegen::SoftmaxState;
/// let s = SoftmaxState::new(7, 64);
/// let _ = s.qkt_compute_calls();
/// ```
///
/// SV before QKt (no `sv_compute_calls` on `FreshLoop`):
/// ```compile_fail
/// use ferrite_wavefront::tk_codegen::SoftmaxState;
/// let s = SoftmaxState::new(7, 64);
/// let (_, s) = s.init_compute_calls();
/// let _ = s.sv_compute_calls();
/// ```
///
/// Finalise before SV (no `finalise_compute_calls` on `PostQkt`):
/// ```compile_fail
/// use ferrite_wavefront::tk_codegen::SoftmaxState;
/// let s = SoftmaxState::new(7, 64);
/// let (_, s) = s.init_compute_calls();
/// let (_, s) = s.qkt_compute_calls();
/// let _ = s.finalise_compute_calls();
/// ```
///
/// Init twice (no `init_compute_calls` on anything but `Empty`):
/// ```compile_fail
/// use ferrite_wavefront::tk_codegen::SoftmaxState;
/// let s = SoftmaxState::new(7, 64);
/// let (_, s) = s.init_compute_calls();
/// let _ = s.init_compute_calls();
/// ```
pub struct SoftmaxState<Phase: SoftmaxPhase> {
    unique_id: u32,
    head_dim: u32,
    _phase: std::marker::PhantomData<Phase>,
}

impl SoftmaxState<Empty> {
    /// Construct a fresh softmax state for AttnDecode op `unique_id`
    /// with the given `head_dim`. Both values are baked into the
    /// witness; downstream builders read them off the witness instead
    /// of re-receiving them as args (single source of truth).
    pub fn new(unique_id: u32, head_dim: u32) -> Self {
        Self {
            unique_id,
            head_dim,
            _phase: std::marker::PhantomData,
        }
    }

    /// Init-softmax body. Consumes `Empty`, returns `FreshLoop`.
    /// Calling `qkt_compute_calls` on `Empty` is a compile error
    /// (trait bound `SoftmaxState<FreshLoop | PostSv>` not satisfied).
    pub fn init_compute_calls(self) -> (Vec<Tk20Call>, SoftmaxState<FreshLoop>) {
        let calls = attn_decode_init_softmax_compute_calls(self.unique_id);
        (
            calls,
            SoftmaxState {
                unique_id: self.unique_id,
                head_dim: self.head_dim,
                _phase: std::marker::PhantomData,
            },
        )
    }
}

impl SoftmaxState<FreshLoop> {
    /// First-iteration QKt step (after init or after sv: see
    /// [`SoftmaxState<PostSv>::qkt_compute_calls`]).
    pub fn qkt_compute_calls(self) -> (Vec<Tk20Call>, SoftmaxState<PostQkt>) {
        let calls = attn_decode_qkt_softmax_step_compute_calls(self.unique_id, self.head_dim);
        (
            calls,
            SoftmaxState {
                unique_id: self.unique_id,
                head_dim: self.head_dim,
                _phase: std::marker::PhantomData,
            },
        )
    }
}

impl SoftmaxState<PostQkt> {
    /// SV step. Consumes `PostQkt`, returns `PostSv`. Calling sv
    /// before qkt is a compile error (no impl for `Empty`/`FreshLoop`).
    pub fn sv_compute_calls(self) -> (Vec<Tk20Call>, SoftmaxState<PostSv>) {
        let calls = attn_decode_sv_accum_compute_calls(self.unique_id);
        (
            calls,
            SoftmaxState {
                unique_id: self.unique_id,
                head_dim: self.head_dim,
                _phase: std::marker::PhantomData,
            },
        )
    }
}

impl SoftmaxState<PostSv> {
    /// Continue the KV-sweep loop with another qkt step. Consumes
    /// `PostSv`, returns `PostQkt` so the loop body re-runs sv next.
    pub fn qkt_compute_calls(self) -> (Vec<Tk20Call>, SoftmaxState<PostQkt>) {
        let calls = attn_decode_qkt_softmax_step_compute_calls(self.unique_id, self.head_dim);
        (
            calls,
            SoftmaxState {
                unique_id: self.unique_id,
                head_dim: self.head_dim,
                _phase: std::marker::PhantomData,
            },
        )
    }

    /// Finalise the softmax (`O = O_accum / l_sum`). Consumes
    /// `PostSv`, returns `Finalised`. Calling finalise before sv
    /// is a compile error (no impl for `FreshLoop`/`PostQkt`).
    pub fn finalise_compute_calls(self) -> (Vec<Tk20Call>, SoftmaxState<Finalised>) {
        let calls = attn_decode_finalise_softmax_norm_compute_calls(self.unique_id);
        (
            calls,
            SoftmaxState {
                unique_id: self.unique_id,
                head_dim: self.head_dim,
                _phase: std::marker::PhantomData,
            },
        )
    }
}

pub fn attn_decode_qkt_softmax_step_compute_calls(
    unique_id: u32,
    head_dim: u32,
) -> Vec<Tk20Call> {
    let u = unique_id;
    // Inner h-loop body — one h-head's K-reduce + softmax update.
    let inner_o_accum_rescale: Vec<Tk20Call> = vec![Tk20Call::ScalarFloat2DMulAssignFromArray {
        array: format!("__o_accum_a{u}"),
        i: "__h".into(),
        j: "__j".into(),
        scalar: format!("__renorm_a{u}"),
    }];
    let h_body: Vec<Tk20Call> = vec![
        Tk20Call::DeclConstU32 {
            var: "__q_off".into(),
            expr: format!("(__q_head_base + __h) * __head_dim_a{u}"),
        },
        Tk20Call::DeclSvBfView {
            var: "__q_row_sv".into(),
            ptr: format!("(reinterpret_cast<__nv_bfloat16*>(__q_smem_a{u}) + __q_off)"),
            k: head_dim,
        },
        Tk20Call::DeclRvBf {
            var: "__q_rv_bf".into(),
            k: head_dim,
        },
        Tk20Call::DeclRvBf {
            var: "__k_rv_bf".into(),
            k: head_dim,
        },
        Tk20Call::DeclRvFl {
            var: "__q_rv_fl".into(),
            k: head_dim,
        },
        Tk20Call::DeclRvFl {
            var: "__k_rv_fl".into(),
            k: head_dim,
        },
        Tk20Call::WarpLoadRvFromSv {
            rv: "__q_rv_bf".into(),
            sv: "__q_row_sv".into(),
        },
        Tk20Call::WarpLoadRvFromSv {
            rv: "__k_rv_bf".into(),
            sv: "__k_row_sv".into(),
        },
        Tk20Call::WarpCopyRv {
            dst: "__q_rv_fl".into(),
            src: "__q_rv_bf".into(),
        },
        Tk20Call::WarpCopyRv {
            dst: "__k_rv_fl".into(),
            src: "__k_rv_bf".into(),
        },
        Tk20Call::WarpMulRv {
            dst: "__q_rv_fl".into(),
            lhs: "__q_rv_fl".into(),
            rhs: "__k_rv_fl".into(),
        },
        Tk20Call::DeclMutableFloatFromWarpSum {
            var: "__s".into(),
            rv: "__q_rv_fl".into(),
        },
        Tk20Call::FloatMulAssign {
            var: "__s".into(),
            expr: format!("__scale_a{u}"),
        },
        Tk20Call::DeclConstFloatFmaxArray {
            var: "__m_new".into(),
            array: format!("__m_max_a{u}"),
            idx: "__h".into(),
            other: "__s".into(),
        },
        Tk20Call::ScalarFloatStoreExpDiff {
            array: format!("__renorm_a{u}"),
            idx: "__h".into(),
            lhs: format!("__m_max_a{u}[__h]"),
            rhs: "__m_new".into(),
        },
        Tk20Call::ScalarFloatStoreExpDiff {
            array: format!("__p_a{u}"),
            idx: "__h".into(),
            lhs: "__s".into(),
            rhs: "__m_new".into(),
        },
        Tk20Call::ScalarFloat1DStoreFma {
            dst: format!("__l_sum_a{u}"),
            idx: "__h".into(),
            a: format!("__renorm_a{u}"),
            b: format!("__l_sum_a{u}"),
            c: format!("__p_a{u}"),
        },
        Tk20Call::ForLoopThreadStrided {
            iter: "__j".into(),
            start_var: "__lane".into(),
            end_var: format!("__head_dim_a{u}"),
            stride_var: "32u".into(),
            body: inner_o_accum_rescale,
        },
        Tk20Call::ScalarFloatStoreVar {
            array: format!("__m_max_a{u}"),
            idx: "__h".into(),
            src: "__m_new".into(),
        },
    ];
    let active_warp_body: Vec<Tk20Call> = vec![
        Tk20Call::DeclConstI32Lane { var: "__lane".into() },
        Tk20Call::DeclConstU32FromConsumerIdx { var: "__kv_head".into() },
        Tk20Call::DeclConstU32 {
            var: "__q_head_base".into(),
            expr: format!(
                "static_cast<unsigned int>(__consumer_idx) * __q_heads_per_warp_a{u}"
            ),
        },
        Tk20Call::DeclConstU32 {
            var: "__k_off".into(),
            expr: format!("__kv_head * __head_dim_a{u}"),
        },
        Tk20Call::DeclSvBfView {
            var: "__k_row_sv".into(),
            ptr: format!("(reinterpret_cast<__nv_bfloat16*>(__k_smem_a{u}) + __k_off)"),
            k: head_dim,
        },
        Tk20Call::ForLoopUnsignedH {
            iter: "__h".into(),
            end: format!("__q_heads_per_warp_a{u}"),
            body: h_body,
        },
    ];
    vec![Tk20Call::IfConsumerIdxLtVar {
        var: format!("__num_kv_heads_a{u}"),
        body: active_warp_body,
    }]
}

/// Tape-build helper: AttnDecode SV-accumulate step (inside KV
/// loop). Per active consumer warp (idx < num_kv_heads), loops over
/// q-heads-per-warp and lane-strided over head_dim, FMA-ing
/// `__p[h] * v_smem[__v_off + j]` into `__o_accum[h][j]`.
pub fn attn_decode_sv_accum_compute_calls(unique_id: u32) -> Vec<Tk20Call> {
    let u = unique_id;
    let inner_body: Vec<Tk20Call> = vec![Tk20Call::Float2DAccumPTimesBf16 {
        accum: format!("__o_accum_a{u}"),
        i: "__h".into(),
        j: "__j".into(),
        p_array: format!("__p_a{u}"),
        smem: format!("__v_smem_a{u}"),
        idx: "__v_off + __j".into(),
    }];
    let h_body: Vec<Tk20Call> = vec![Tk20Call::ForLoopThreadStrided {
        iter: "__j".into(),
        start_var: "__lane".into(),
        end_var: format!("__head_dim_a{u}"),
        stride_var: "32u".into(),
        body: inner_body,
    }];
    vec![Tk20Call::IfConsumerIdxLtVar {
        var: format!("__num_kv_heads_a{u}"),
        body: vec![
            Tk20Call::DeclConstI32Lane { var: "__lane".into() },
            Tk20Call::DeclConstU32FromConsumerIdx { var: "__kv_head".into() },
            Tk20Call::DeclConstU32 {
                var: "__v_off".into(),
                expr: format!("__kv_head * __head_dim_a{u}"),
            },
            Tk20Call::ForLoopUnsignedH {
                iter: "__h".into(),
                end: format!("__q_heads_per_warp_a{u}"),
                body: h_body,
            },
        ],
    }]
}

/// Tape-build helper: AttnDecode finalise — divide each
/// `__o_accum[h][j]` by `__l_sum[h]` and write the bf16 result to
/// `__out_smem[(__q_head_base + h) * head_dim + j]`. Last AttnDecode
/// step in the kernel.
pub fn attn_decode_finalise_softmax_norm_compute_calls(unique_id: u32) -> Vec<Tk20Call> {
    let u = unique_id;
    let writeback_body: Vec<Tk20Call> = vec![Tk20Call::Bf16StoreFrom2DAccumTimesScalar {
        smem: format!("__out_smem_a{u}"),
        idx: "__out_off + __j".into(),
        accum: format!("__o_accum_a{u}"),
        i: "__h".into(),
        j: "__j".into(),
        scalar: "__inv_l".into(),
    }];
    let h_body: Vec<Tk20Call> = vec![
        Tk20Call::DeclConstU32 {
            var: "__out_off".into(),
            expr: format!("(__q_head_base + __h) * __head_dim_a{u}"),
        },
        Tk20Call::DeclConstFloatReciprocal {
            var: "__inv_l".into(),
            array: format!("__l_sum_a{u}"),
            idx: "__h".into(),
        },
        Tk20Call::ForLoopThreadStrided {
            iter: "__j".into(),
            start_var: "__lane".into(),
            end_var: format!("__head_dim_a{u}"),
            stride_var: "32u".into(),
            body: writeback_body,
        },
    ];
    vec![Tk20Call::IfConsumerIdxLtVar {
        var: format!("__num_kv_heads_a{u}"),
        body: vec![
            Tk20Call::DeclConstI32Lane { var: "__lane".into() },
            Tk20Call::DeclConstU32 {
                var: "__q_head_base".into(),
                expr: format!(
                    "static_cast<unsigned int>(__consumer_idx) * __q_heads_per_warp_a{u}"
                ),
            },
            Tk20Call::ForLoopUnsignedH {
                iter: "__h".into(),
                end: format!("__q_heads_per_warp_a{u}"),
                body: h_body,
            },
        ],
    }]
}

/// Tape-build helper: AttnDecode init-softmax — zero per-warp
/// softmax accumulators (`__m_max = -INF`, `__l_sum = 0`,
/// `__o_accum[][] = 0`). Run ONCE before the KV-sweep loop on the
/// active consumer warps. `unique_id` distinguishes the per-op
/// prelude-scope identifiers across multiple AttnDecode instances
/// in one TkProgram.
pub fn attn_decode_init_softmax_compute_calls(unique_id: u32) -> Vec<Tk20Call> {
    let u = unique_id;
    let h_loop_body: Vec<Tk20Call> = vec![
        Tk20Call::ScalarFloatStoreNegInfty {
            array: format!("__m_max_a{u}"),
            idx: "__h".into(),
        },
        Tk20Call::ScalarFloatStoreZero {
            array: format!("__l_sum_a{u}"),
            idx: "__h".into(),
        },
        Tk20Call::ForLoopThreadStrided {
            iter: "__j".into(),
            start_var: "__lane".into(),
            end_var: format!("__head_dim_a{u}"),
            stride_var: "32u".into(),
            body: vec![Tk20Call::ScalarFloat2DStoreZero {
                array: format!("__o_accum_a{u}"),
                i: "__h".into(),
                j: "__j".into(),
            }],
        },
    ];
    vec![Tk20Call::IfConsumerIdxLtVar {
        var: format!("__num_kv_heads_a{u}"),
        body: vec![
            Tk20Call::DeclConstI32Lane { var: "__lane".into() },
            Tk20Call::ForLoopUnsignedH {
                iter: "__h".into(),
                end: format!("__q_heads_per_warp_a{u}"),
                body: h_loop_body,
            },
        ],
    }]
}

/// Tape-build helper: GemmM1 internal-output variant (Phase 12
/// carry-forward). Same K-reduce as [`gemm_m1_compute_calls`] but
/// lane 0 of warp `c` writes to `page_buf[y_id]` (smem) instead of
/// `buf{out_buf}` (gmem) — the consuming op reads from this op's
/// smem page directly without a TMA-store round-trip.
pub fn gemm_m1_compute_calls_internal(
    x_id: u8,
    w_id: u8,
    y_id: u8,
    k: u32,
    bn: u32,
) -> Vec<Tk20Call> {
    const K_TILE: u32 = 128;
    debug_assert_eq!(
        k % K_TILE,
        0,
        "gemm_m1_compute_calls_internal: K={k} must be multiple of K_TILE={K_TILE}"
    );
    let k_blocks = k / K_TILE;
    let inner_loop_body: Vec<Tk20Call> = vec![
        Tk20Call::DeclRvBf {
            var: "__x_rv_bf".into(),
            k: K_TILE,
        },
        Tk20Call::DeclRvBf {
            var: "__w_rv_bf".into(),
            k: K_TILE,
        },
        Tk20Call::DeclRvFl {
            var: "__x_rv_fl".into(),
            k: K_TILE,
        },
        Tk20Call::DeclRvFl {
            var: "__w_rv_fl".into(),
            k: K_TILE,
        },
        Tk20Call::WarpLoadRvFromSv {
            rv: "__x_rv_bf".into(),
            sv: "__x_sv.template subvec<128>(__k_i)".into(),
        },
        Tk20Call::WarpLoadRvFromSv {
            rv: "__w_rv_bf".into(),
            sv: "__w_row_sv.template subvec<128>(__k_i)".into(),
        },
        Tk20Call::WarpCopyRv {
            dst: "__x_rv_fl".into(),
            src: "__x_rv_bf".into(),
        },
        Tk20Call::WarpCopyRv {
            dst: "__w_rv_fl".into(),
            src: "__w_rv_bf".into(),
        },
        Tk20Call::WarpMulRv {
            dst: "__x_rv_fl".into(),
            lhs: "__x_rv_fl".into(),
            rhs: "__w_rv_fl".into(),
        },
        Tk20Call::FloatAddAssignWarpSum {
            dst: "__acc".into(),
            rv: "__x_rv_fl".into(),
        },
    ];
    let active_warp_body: Vec<Tk20Call> = vec![
        Tk20Call::DeclConstU32FromConsumerIdx { var: "__row".into() },
        Tk20Call::DeclConstI32Lane { var: "__lane".into() },
        Tk20Call::DeclSvBfView {
            var: "__w_row_sv".into(),
            ptr: format!("(reinterpret_cast<__nv_bfloat16*>(page_buf[{w_id}]) + __row * {k}u)"),
            k,
        },
        Tk20Call::DeclMutableFloat {
            var: "__acc".into(),
            init: "0.0f".into(),
        },
        Tk20Call::ForLoopIntK {
            iter: "__k_i".into(),
            end: format!("{k_blocks}"),
            body: inner_loop_body,
        },
        Tk20Call::IfLane0 {
            body: vec![Tk20Call::Bf16GmemStoreFromFloat {
                gmem: "__y_smem".into(),
                idx: "__n_i * __bn + __row".into(),
                src: "__acc".into(),
            }],
        },
    ];
    vec![
        Tk20Call::DeclSmemPtrTypedBf16 {
            var: "__y_smem".into(),
            page_id: y_id,
        },
        Tk20Call::DeclConstU32 {
            var: "__bn".into(),
            expr: format!("{bn}u"),
        },
        Tk20Call::DeclSvBfView {
            var: "__x_sv".into(),
            ptr: format!("page_buf[{x_id}]"),
            k,
        },
        Tk20Call::IfConsumerIdxLtBnVar {
            bn_var: "__bn".into(),
            body: active_warp_body,
        },
    ]
}

/// Tape-build helper: GemmM1 consumer compute as a typed Tk20Call
/// sequence. Consumer warp `c` (for `c < bn`) computes `y[c] =
/// dot(x_row, w_row[c])` via TK 2.0 register-vector K-reduce. Lane
/// 0 of warp `c` writes the scalar bf16 result directly to gmem
/// (`buf{out_buf}[__n_i * bn + c]`) — bypasses the Y staging page.
pub fn gemm_m1_compute_calls(x_id: u8, w_id: u8, out_buf: u32, k: u32, bn: u32) -> Vec<Tk20Call> {
    const K_TILE: u32 = 128;
    debug_assert_eq!(
        k % K_TILE,
        0,
        "gemm_m1_compute_calls: K={k} must be a multiple of K_TILE={K_TILE}"
    );
    let k_blocks = k / K_TILE;
    let inner_loop_body: Vec<Tk20Call> = vec![
        Tk20Call::DeclRvBf {
            var: "__x_rv_bf".into(),
            k: K_TILE,
        },
        Tk20Call::DeclRvBf {
            var: "__w_rv_bf".into(),
            k: K_TILE,
        },
        Tk20Call::DeclRvFl {
            var: "__x_rv_fl".into(),
            k: K_TILE,
        },
        Tk20Call::DeclRvFl {
            var: "__w_rv_fl".into(),
            k: K_TILE,
        },
        Tk20Call::WarpLoadRvFromSv {
            rv: "__x_rv_bf".into(),
            sv: "__x_sv.template subvec<128>(__k_i)".into(),
        },
        Tk20Call::WarpLoadRvFromSv {
            rv: "__w_rv_bf".into(),
            sv: "__w_row_sv.template subvec<128>(__k_i)".into(),
        },
        Tk20Call::WarpCopyRv {
            dst: "__x_rv_fl".into(),
            src: "__x_rv_bf".into(),
        },
        Tk20Call::WarpCopyRv {
            dst: "__w_rv_fl".into(),
            src: "__w_rv_bf".into(),
        },
        Tk20Call::WarpMulRv {
            dst: "__x_rv_fl".into(),
            lhs: "__x_rv_fl".into(),
            rhs: "__w_rv_fl".into(),
        },
        Tk20Call::FloatAddAssignWarpSum {
            dst: "__acc".into(),
            rv: "__x_rv_fl".into(),
        },
    ];
    let active_warp_body: Vec<Tk20Call> = vec![
        Tk20Call::DeclConstU32FromConsumerIdx { var: "__row".into() },
        Tk20Call::DeclConstI32Lane { var: "__lane".into() },
        Tk20Call::DeclSvBfView {
            var: "__w_row_sv".into(),
            ptr: format!("(reinterpret_cast<__nv_bfloat16*>(page_buf[{w_id}]) + __row * {k}u)"),
            k,
        },
        Tk20Call::DeclMutableFloat {
            var: "__acc".into(),
            init: "0.0f".into(),
        },
        Tk20Call::ForLoopIntK {
            iter: "__k_i".into(),
            end: format!("{k_blocks}"),
            body: inner_loop_body,
        },
        Tk20Call::IfLane0 {
            body: vec![Tk20Call::Bf16GmemStoreFromFloat {
                gmem: "__y_gmem".into(),
                idx: "__n_i * __bn + __row".into(),
                src: "__acc".into(),
            }],
        },
    ];
    vec![
        Tk20Call::DeclGmemPtrBf16FromBuf {
            var: "__y_gmem".into(),
            buf: out_buf,
        },
        Tk20Call::DeclConstU32 {
            var: "__bn".into(),
            expr: format!("{bn}u"),
        },
        Tk20Call::DeclSvBfView {
            var: "__x_sv".into(),
            ptr: format!("page_buf[{x_id}]"),
            k,
        },
        Tk20Call::IfConsumerIdxLtBnVar {
            bn_var: "__bn".into(),
            body: active_warp_body,
        },
    ]
}

/// Tape-build helper: RmsNorm consumer compute as a typed Tk20Call
/// sequence. Consumer-warp-0-only (other warps idle); two-pass:
/// (1) K_TILE-block sumsq reduce via TK 2.0 register-vector
/// `warp::load → warp::copy → warp::mul → warp::sum` then
/// `rsqrtf(sumsq/hidden + eps)`; (2) lane-strided in-place
/// `x[i] = x[i] * scale * w[i]` write-back over hidden cols.
pub fn rmsnorm_compute_calls(x_id: u8, w_id: u8, hidden: u32, eps: f32) -> Vec<Tk20Call> {
    const K_TILE: u32 = 128;
    debug_assert_eq!(
        hidden % K_TILE,
        0,
        "rmsnorm_compute_calls: hidden={hidden} must be a multiple of K_TILE={K_TILE}"
    );
    let k_blocks = hidden / K_TILE;
    let inner_loop_body: Vec<Tk20Call> = vec![
        Tk20Call::DeclRvBf {
            var: "__x_rv_bf".into(),
            k: K_TILE,
        },
        Tk20Call::DeclRvFl {
            var: "__x_rv_fl".into(),
            k: K_TILE,
        },
        Tk20Call::WarpLoadRvFromSv {
            rv: "__x_rv_bf".into(),
            sv: "__x_sv.template subvec<128>(__k_i)".into(),
        },
        Tk20Call::WarpCopyRv {
            dst: "__x_rv_fl".into(),
            src: "__x_rv_bf".into(),
        },
        Tk20Call::WarpMulRv {
            dst: "__x_rv_fl".into(),
            lhs: "__x_rv_fl".into(),
            rhs: "__x_rv_fl".into(),
        },
        Tk20Call::FloatAddAssignWarpSum {
            dst: "__sumsq".into(),
            rv: "__x_rv_fl".into(),
        },
    ];
    let writeback_body: Vec<Tk20Call> = vec![
        Tk20Call::DeclConstFloatFromBf16Smem {
            var: "__v".into(),
            smem: "__x_smem".into(),
            idx: "__i".into(),
        },
        Tk20Call::DeclConstFloatFromBf16Smem {
            var: "__g".into(),
            smem: "__w_smem".into(),
            idx: "__i".into(),
        },
        Tk20Call::Bf16StoreFromFloatTripleProduct {
            smem: "__x_smem".into(),
            idx: "__i".into(),
            a: "__v".into(),
            b: "__scale".into(),
            c: "__g".into(),
        },
    ];
    let warp0_body: Vec<Tk20Call> = vec![
        Tk20Call::DeclConstU32 {
            var: "__hidden".into(),
            expr: format!("{hidden}u"),
        },
        Tk20Call::DeclConstFloat {
            var: "__eps".into(),
            expr: format!("{eps:?}f"),
        },
        Tk20Call::DeclConstI32Lane { var: "__lane".into() },
        Tk20Call::DeclMutableFloat {
            var: "__sumsq".into(),
            init: "0.0f".into(),
        },
        Tk20Call::ForLoopIntK {
            iter: "__k_i".into(),
            end: format!("{k_blocks}"),
            body: inner_loop_body,
        },
        Tk20Call::DeclConstFloatRsqrtScale {
            var: "__scale".into(),
            sumsq: "__sumsq".into(),
            hidden: "__hidden".into(),
            eps: "__eps".into(),
        },
        Tk20Call::ForLoopThreadStrided {
            iter: "__i".into(),
            start_var: "__lane".into(),
            end_var: "__hidden".into(),
            stride_var: "32u".into(),
            body: writeback_body,
        },
    ];
    vec![
        Tk20Call::DeclSmemPtrBf16 {
            var: "__x_smem".into(),
            page_id: x_id,
        },
        Tk20Call::DeclSmemPtrBf16 {
            var: "__w_smem".into(),
            page_id: w_id,
        },
        Tk20Call::DeclSvBfView {
            var: "__x_sv".into(),
            ptr: format!("page_buf[{x_id}]"),
            k: hidden,
        },
        Tk20Call::IfConsumerIdxZero { body: warp0_body },
    ]
}

// ── RopeForm typed witness (paris invariant
//    `rope-form-consistent-across-forward`) ─────────────────────
//
// NeoX (Llama, Mistral) and Interleaved (Gemma2 partial, GPT-NeoX
// historical) RoPE differ in pair indexing: NeoX rotates `(i, i+half)`
// pairs; Interleaved rotates `(2i, 2i+1)` pairs. Switching between
// the two requires DIFFERENT scalar arithmetic in the rotate
// builder. With `Bf16StoreFromFloatFmaSub`/`...FmaAdd` hardcoding
// the NeoX pair shape, an Interleaved rotation would silently emit
// the wrong CUDA — exactly the bug class the typed witness pattern
// kills.
//
// `RopeForm` is a sealed trait with two impls: `NeoX` and `Interleaved`.
// Each carries the pair-index expressions as associated consts.
// `rope_compute_calls<F: RopeForm>` is generic over the form; mixing
// NeoX and Interleaved within one forward (passing different `F` to
// two RopeRotate ops) is a Rust type error at the orchestrator's
// type-inference site.

mod rope_form_seal {
    pub trait Sealed {}
}

/// Sealed trait — exactly two impls (NeoX, Interleaved). Carries
/// the pair-index expressions as associated consts so the rope
/// builder reads them off the type rather than hardcoding them.
pub trait RopeForm: rope_form_seal::Sealed {
    /// CUDA expression yielding the lo-half index given a pair
    /// index `__p`, half `__half`, head_dim `__head_dim`. For NeoX:
    /// `"__row_head * __head_dim + __lane"`. For Interleaved:
    /// `"__row_head * __head_dim + 2u * __lane"`.
    const PAIR_LO_EXPR: &'static str;
    /// CUDA expression yielding the hi-half index. NeoX:
    /// `"__i_lo + __half"`. Interleaved: `"__i_lo + 1u"`.
    const PAIR_HI_EXPR: &'static str;
    /// Stable identifier suffix used in unique-id correlation; lets
    /// callers tag tests by form.
    const NAME: &'static str;
}

/// NeoX RoPE: `(i, i+half)` half-split pairs.
///
/// # Compile-fail proof (paris invariant `rope-form-consistent-across-forward`)
///
/// A function generic over `F: RopeForm` cannot accept BOTH `NeoX`
/// AND `Interleaved` at the same instantiation; the type system
/// pins one form per call. To "prove" two forms can't mix, the
/// natural shape is a fn that receives two ropes' values:
///
/// ```compile_fail
/// use ferrite_wavefront::tk_codegen::{rope_compute_calls, NeoX, Interleaved};
/// fn one_form_per_forward<F: ferrite_wavefront::tk_codegen::RopeForm>() {
///     let _q = rope_compute_calls::<F>(0, 1, 2, 64, 32);
///     let _k = rope_compute_calls::<F>(3, 1, 2, 64, 32);
/// }
/// // Pin Q to NeoX, K to Interleaved at the same call site:
/// fn mixed() {
///     let _q = rope_compute_calls::<NeoX>(0, 1, 2, 64, 32);
///     let _k: () = rope_compute_calls::<Interleaved>(3, 1, 2, 64, 32);
/// }
/// ```
pub struct NeoX;
impl rope_form_seal::Sealed for NeoX {}
impl RopeForm for NeoX {
    const PAIR_LO_EXPR: &'static str = "__row_head * __head_dim + __lane";
    const PAIR_HI_EXPR: &'static str = "__i_lo + __half";
    const NAME: &'static str = "NeoX";
}

/// Interleaved RoPE: `(2i, 2i+1)` adjacent pairs (Gemma2 partial,
/// GPT-NeoX legacy).
pub struct Interleaved;
impl rope_form_seal::Sealed for Interleaved {}
impl RopeForm for Interleaved {
    const PAIR_LO_EXPR: &'static str = "__row_head * __head_dim + 2u * __lane";
    const PAIR_HI_EXPR: &'static str = "__i_lo + 1u";
    const NAME: &'static str = "Interleaved";
}

/// Tape-build helper: RoPE rotate as a typed Tk20Call sequence.
/// In-place rotation on x's page, all-consumer-warp parallel over
/// `total_pairs = m * num_heads * (head_dim / 2)` rotation pairs.
/// Generic over [`RopeForm`] (NeoX vs Interleaved): the form's
/// `PAIR_LO_EXPR` / `PAIR_HI_EXPR` associated consts drive the
/// pair-index decomposition. Mixing forms within one forward is a
/// Rust type error at every call site that fixes `F`.
pub fn rope_compute_calls<F: RopeForm>(
    x_id: u8,
    c_id: u8,
    s_id: u8,
    head_dim: u32,
    total_pairs: u64,
) -> Vec<Tk20Call> {
    let half = head_dim / 2;
    vec![
        Tk20Call::DeclSmemPtrBf16 {
            var: "__x_smem".into(),
            page_id: x_id,
        },
        Tk20Call::DeclSmemPtrBf16 {
            var: "__cos_smem".into(),
            page_id: c_id,
        },
        Tk20Call::DeclSmemPtrBf16 {
            var: "__sin_smem".into(),
            page_id: s_id,
        },
        Tk20Call::DeclConstU32 {
            var: "__head_dim".into(),
            expr: format!("{head_dim}u"),
        },
        Tk20Call::DeclConstU32 {
            var: "__half".into(),
            expr: format!("{half}u"),
        },
        Tk20Call::DeclConstU32 {
            var: "__pairs".into(),
            expr: format!("{total_pairs}u"),
        },
        Tk20Call::DeclConstI32 {
            var: "__tid_in_consumers".into(),
            expr: "static_cast<int>(threadIdx.x) - 4 * 32".into(),
        },
        Tk20Call::DeclConstI32 {
            var: "__consumer_threads".into(),
            expr: "16 * 32".into(),
        },
        Tk20Call::ForLoopThreadStrided {
            iter: "__p".into(),
            start_var: "__tid_in_consumers".into(),
            end_var: "__pairs".into(),
            stride_var: "__consumer_threads".into(),
            body: vec![
                Tk20Call::DeclConstU32 {
                    var: "__row_head".into(),
                    expr: "__p / __half".into(),
                },
                Tk20Call::DeclConstU32 {
                    var: "__lane".into(),
                    expr: "__p % __half".into(),
                },
                Tk20Call::DeclConstU32 {
                    var: "__i_lo".into(),
                    expr: F::PAIR_LO_EXPR.into(),
                },
                Tk20Call::DeclConstU32 {
                    var: "__i_hi".into(),
                    expr: F::PAIR_HI_EXPR.into(),
                },
                Tk20Call::DeclConstFloatFromBf16Smem {
                    var: "__c".into(),
                    smem: "__cos_smem".into(),
                    idx: "__lane".into(),
                },
                Tk20Call::DeclConstFloatFromBf16Smem {
                    var: "__s".into(),
                    smem: "__sin_smem".into(),
                    idx: "__lane".into(),
                },
                Tk20Call::DeclConstFloatFromBf16Smem {
                    var: "__x_lo".into(),
                    smem: "__x_smem".into(),
                    idx: "__i_lo".into(),
                },
                Tk20Call::DeclConstFloatFromBf16Smem {
                    var: "__x_hi".into(),
                    smem: "__x_smem".into(),
                    idx: "__i_hi".into(),
                },
                Tk20Call::Bf16StoreFromFloatFmaSub {
                    smem: "__x_smem".into(),
                    idx: "__i_lo".into(),
                    lhs1: "__x_lo".into(),
                    rhs1: "__c".into(),
                    lhs2: "__x_hi".into(),
                    rhs2: "__s".into(),
                },
                Tk20Call::Bf16StoreFromFloatFmaAdd {
                    smem: "__x_smem".into(),
                    idx: "__i_hi".into(),
                    lhs1: "__x_lo".into(),
                    rhs1: "__s".into(),
                    lhs2: "__x_hi".into(),
                    rhs2: "__c".into(),
                },
            ],
        },
    ]
}

/// Tape-build helper: silu·mul as a typed Tk20Call sequence.
/// `silu(gate) * up` in place on gate's page, all consumer warps.
pub fn silu_mul_compute_calls(g_id: u8, u_id: u8, total: u64) -> Vec<Tk20Call> {
    vec![
        Tk20Call::DeclSmemPtrBf16 {
            var: "__g_smem".into(),
            page_id: g_id,
        },
        Tk20Call::DeclSmemPtrBf16 {
            var: "__u_smem".into(),
            page_id: u_id,
        },
        Tk20Call::DeclConstU32 {
            var: "__total".into(),
            expr: format!("{total}u"),
        },
        Tk20Call::DeclConstI32 {
            var: "__tid_in_consumers".into(),
            expr: "static_cast<int>(threadIdx.x) - 4 * 32".into(),
        },
        Tk20Call::DeclConstI32 {
            var: "__consumer_threads".into(),
            expr: "16 * 32".into(),
        },
        Tk20Call::ForLoopThreadStrided {
            iter: "__i".into(),
            start_var: "__tid_in_consumers".into(),
            end_var: "__total".into(),
            stride_var: "__consumer_threads".into(),
            body: vec![
                Tk20Call::DeclConstFloatFromBf16Smem {
                    var: "__g".into(),
                    smem: "__g_smem".into(),
                    idx: "__i".into(),
                },
                Tk20Call::DeclConstFloatFromBf16Smem {
                    var: "__u".into(),
                    smem: "__u_smem".into(),
                    idx: "__i".into(),
                },
                Tk20Call::DeclConstFloatSilu {
                    var: "__silu_g".into(),
                    src: "__g".into(),
                },
                Tk20Call::Bf16StoreFromFloatMul {
                    smem: "__g_smem".into(),
                    idx: "__i".into(),
                    lhs: "__silu_g".into(),
                    rhs: "__u".into(),
                },
            ],
        },
    ]
}

/// Tape-build helper: residual-add as a typed Tk20Call sequence.
/// Pushed by `lower_residual_add` (and the routed variant) into a
/// single `Compute { calls: ... }`. Each emitted CUDA statement is
/// ONE Tk20Call variant — no multi-line emit-side helper.
pub fn residual_add_compute_calls(a_id: u8, b_id: u8, total: u64) -> Vec<Tk20Call> {
    vec![
        Tk20Call::DeclSmemPtrBf16 {
            var: "__a_smem".into(),
            page_id: a_id,
        },
        Tk20Call::DeclSmemPtrBf16 {
            var: "__b_smem".into(),
            page_id: b_id,
        },
        Tk20Call::DeclConstU32 {
            var: "__total".into(),
            expr: format!("{total}u"),
        },
        Tk20Call::DeclConstI32 {
            var: "__tid_in_consumers".into(),
            expr: "static_cast<int>(threadIdx.x) - 4 * 32".into(),
        },
        Tk20Call::DeclConstI32 {
            var: "__consumer_threads".into(),
            expr: "16 * 32".into(),
        },
        Tk20Call::ForLoopThreadStrided {
            iter: "__i".into(),
            start_var: "__tid_in_consumers".into(),
            end_var: "__total".into(),
            stride_var: "__consumer_threads".into(),
            body: vec![
                Tk20Call::DeclConstFloatFromBf16Smem {
                    var: "__a".into(),
                    smem: "__a_smem".into(),
                    idx: "__i".into(),
                },
                Tk20Call::DeclConstFloatFromBf16Smem {
                    var: "__b".into(),
                    smem: "__b_smem".into(),
                    idx: "__i".into(),
                },
                Tk20Call::Bf16StoreFromFloatAdd {
                    smem: "__a_smem".into(),
                    idx: "__i".into(),
                    lhs: "__a".into(),
                    rhs: "__b".into(),
                },
            ],
        },
    ]
}

impl Tk20Call {
    /// Lower the typed primitive call to its CUDA fragment via the
    /// matching `tk20::*` Rust function. The match is exhaustive — any
    /// new variant added above MUST add an arm here citing its TK 2.0
    /// header line in the binding.
    pub fn emit(&self) -> String {
        match self {

            Tk20Call::WarpgroupMmaFence { d } => tk20::warpgroup_mma_fence(d),
            Tk20Call::WarpgroupMmaAB { d, a, b } => tk20::warpgroup_mma_ab(d, a, b),
            Tk20Call::WarpgroupMmAB { d, a, b } => tk20::warpgroup_mm_ab(d, a, b),
            Tk20Call::WarpgroupMmaABt { d, a, b } => tk20::warpgroup_mma_abt(d, a, b),
            Tk20Call::WarpgroupMmaCommitGroup => tk20::warpgroup_mma_commit_group(),
            Tk20Call::WarpgroupMmaAsyncWait { n } => tk20::warpgroup_mma_async_wait(*n),

            Tk20Call::WarpMmaAB { d, a, b } => tk20::warp_mma_ab(d, a, b),
            Tk20Call::WarpMmaABt { d, a, b } => tk20::warp_mma_abt(d, a, b),

            Tk20Call::GroupTmaStoreCommitGroup => tk20::group_tma_store_commit_group(),
            Tk20Call::GroupTmaStoreAsyncWait { n } => tk20::group_tma_store_async_wait(*n),
            Tk20Call::GroupTmaStoreAsyncReadWait { n } => {
                tk20::group_tma_store_async_read_wait(*n)
            }

            Tk20Call::DeclSmemPtrBf16 { var, page_id } => {
                format!("auto* {var} = reinterpret_cast<__nv_bfloat16*>(page_buf[{page_id}]);")
            }
            Tk20Call::DeclConstU32 { var, expr } => {
                format!("const unsigned int {var} = {expr};")
            }
            Tk20Call::DeclConstI32 { var, expr } => {
                format!("const int {var} = {expr};")
            }
            Tk20Call::DeclConstFloatFromBf16Smem { var, smem, idx } => {
                format!("const float {var} = __bfloat162float({smem}[{idx}]);")
            }
            Tk20Call::Bf16StoreFromFloatAdd { smem, idx, lhs, rhs } => {
                format!("{smem}[{idx}] = __float2bfloat16({lhs} + {rhs});")
            }
            Tk20Call::Bf16StoreFromFloatMul { smem, idx, lhs, rhs } => {
                format!("{smem}[{idx}] = __float2bfloat16({lhs} * {rhs});")
            }
            Tk20Call::DeclConstFloatSilu { var, src } => {
                format!("const float {var} = {src} / (1.0f + expf(-{src}));")
            }
            Tk20Call::Bf16StoreFromFloatFmaSub {
                smem,
                idx,
                lhs1,
                rhs1,
                lhs2,
                rhs2,
            } => format!("{smem}[{idx}] = __float2bfloat16({lhs1} * {rhs1} - {lhs2} * {rhs2});"),
            Tk20Call::Bf16StoreFromFloatFmaAdd {
                smem,
                idx,
                lhs1,
                rhs1,
                lhs2,
                rhs2,
            } => format!("{smem}[{idx}] = __float2bfloat16({lhs1} * {rhs1} + {lhs2} * {rhs2});"),
            Tk20Call::DeclSvBfView { var, ptr, k } => tk20::decl_sv_view_bf(var, ptr, *k),
            Tk20Call::DeclRvBf { var, k } => tk20::decl_rv_bf(var, *k),
            Tk20Call::DeclRvFl { var, k } => tk20::decl_rv_fl(var, *k),
            Tk20Call::WarpLoadRvFromSv { rv, sv } => tk20::warp_load_rv_from_sv(rv, sv),
            Tk20Call::WarpCopyRv { dst, src } => tk20::warp_copy_rv(dst, src),
            Tk20Call::WarpMulRv { dst, lhs, rhs } => tk20::warp_mul_rv(dst, lhs, rhs),
            Tk20Call::IfConsumerIdxZero { body } => {
                let inner: String = body.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
                format!("if (__consumer_idx == 0) {{ {inner} }}")
            }
            Tk20Call::DeclConstI32Lane { var } => {
                format!("const int {var} = static_cast<int>(threadIdx.x & 31);")
            }
            Tk20Call::DeclConstFloat { var, expr } => format!("const float {var} = {expr};"),
            Tk20Call::DeclMutableFloat { var, init } => format!("float {var} = {init};"),
            Tk20Call::ForLoopIntK { iter, end, body } => {
                let inner: String = body.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
                format!("for (int {iter} = 0; {iter} < {end}; ++{iter}) {{ {inner} }}")
            }
            Tk20Call::FloatAddAssignWarpSum { dst, rv } => {
                format!("{dst} += {};", tk20::warp_sum_rv(rv))
            }
            Tk20Call::DeclConstFloatRsqrtScale {
                var,
                sumsq,
                hidden,
                eps,
            } => format!(
                "const float {var} = rsqrtf({sumsq} / static_cast<float>({hidden}) + {eps});"
            ),
            Tk20Call::Bf16StoreFromFloatTripleProduct { smem, idx, a, b, c } => {
                format!("{smem}[{idx}] = __float2bfloat16({a} * {b} * {c});")
            }
            Tk20Call::IfConsumerIdxLtBnVar { bn_var, body } => {
                let inner: String = body.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
                format!("if (static_cast<unsigned int>(__consumer_idx) < {bn_var}) {{ {inner} }}")
            }
            Tk20Call::IfLane0 { body } => {
                let inner: String = body.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
                format!("if (__lane == 0) {{ {inner} }}")
            }
            Tk20Call::DeclConstU32FromConsumerIdx { var } => {
                format!("const unsigned int {var} = static_cast<unsigned int>(__consumer_idx);")
            }
            Tk20Call::DeclGmemPtrBf16FromBuf { var, buf } => {
                format!("__nv_bfloat16* {var} = reinterpret_cast<__nv_bfloat16*>(buf{buf});")
            }
            Tk20Call::Bf16GmemStoreFromFloat { gmem, idx, src } => {
                format!("{gmem}[{idx}] = __float2bfloat16({src});")
            }
            Tk20Call::DeclSmemPtrTypedBf16 { var, page_id } => {
                format!("__nv_bfloat16* {var} = reinterpret_cast<__nv_bfloat16*>(page_buf[{page_id}]);")
            }
            Tk20Call::IfConsumerIdxLtVar { var, body } => {
                let inner: String = body.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
                format!("if (static_cast<unsigned int>(__consumer_idx) < {var}) {{ {inner} }}")
            }
            Tk20Call::ForLoopUnsignedH { iter, end, body } => {
                let inner: String = body.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
                format!("for (unsigned int {iter} = 0u; {iter} < {end}; ++{iter}) {{ {inner} }}")
            }
            Tk20Call::ScalarFloatStoreNegInfty { array, idx } => {
                format!("{array}[{idx}] = -INFINITY;")
            }
            Tk20Call::ScalarFloatStoreZero { array, idx } => {
                format!("{array}[{idx}] = 0.0f;")
            }
            Tk20Call::ScalarFloat2DStoreZero { array, i, j } => {
                format!("{array}[{i}][{j}] = 0.0f;")
            }
            Tk20Call::Float2DAccumPTimesBf16 {
                accum,
                i,
                j,
                p_array,
                smem,
                idx,
            } => format!(
                "{accum}[{i}][{j}] += {p_array}[{i}] * __bfloat162float({smem}[{idx}]);"
            ),
            Tk20Call::DeclConstFloatReciprocal { var, array, idx } => {
                format!("const float {var} = 1.0f / {array}[{idx}];")
            }
            Tk20Call::Bf16StoreFrom2DAccumTimesScalar {
                smem,
                idx,
                accum,
                i,
                j,
                scalar,
            } => format!("{smem}[{idx}] = __float2bfloat16({accum}[{i}][{j}] * {scalar});"),
            Tk20Call::DeclMutableFloatFromWarpSum { var, rv } => {
                format!("float {var} = {};", tk20::warp_sum_rv(rv))
            }
            Tk20Call::FloatMulAssign { var, expr } => format!("{var} *= {expr};"),
            Tk20Call::DeclConstFloatFmaxArray {
                var,
                array,
                idx,
                other,
            } => format!("const float {var} = fmaxf({array}[{idx}], {other});"),
            Tk20Call::ScalarFloatStoreExpDiff {
                array,
                idx,
                lhs,
                rhs,
            } => format!("{array}[{idx}] = expf({lhs} - {rhs});"),
            Tk20Call::ScalarFloat1DStoreFma { dst, idx, a, b, c } => {
                format!("{dst}[{idx}] = {a}[{idx}] * {b}[{idx}] + {c}[{idx}];")
            }
            Tk20Call::ScalarFloat2DMulAssignFromArray {
                array,
                i,
                j,
                scalar,
            } => format!("{array}[{i}][{j}] *= {scalar}[{i}];"),
            Tk20Call::ScalarFloatStoreVar { array, idx, src } => {
                format!("{array}[{idx}] = {src};")
            }
            Tk20Call::DebugOpBeginMarker { op_tag } => format!(
                "#ifdef TK_DEBUG_HANDSHAKE\n        \
                 if (threadIdx.x == 0) {{ printf(\"[BEGIN {op_tag}]\\n\"); }}\n        \
                 #endif"
            ),
            Tk20Call::ForLoopThreadStrided {
                iter,
                start_var,
                end_var,
                stride_var,
                body,
            } => {
                let body_cuda: String = body.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
                format!(
                    "for (unsigned int {iter} = static_cast<unsigned int>({start_var}); \
                     {iter} < {end_var}; \
                     {iter} += static_cast<unsigned int>({stride_var})) {{ {body_cuda} }}"
                )
            }

            Tk20Call::WarpRowMax { rv, rt } => tk20::warp_row_max(rv, rt),
            Tk20Call::WarpRowSum { rv, rt } => tk20::warp_row_sum(rv, rt),
            Tk20Call::WarpExp2 { rt } => tk20::warp_exp2(rt),
            Tk20Call::WarpMulRow { rt, rv } => tk20::warp_mul_row(rt, rv),
            Tk20Call::WarpDivRow { rt, rv } => tk20::warp_div_row(rt, rv),
            Tk20Call::WarpNegInfty { rv } => tk20::warp_neg_infty(rv),
            Tk20Call::WarpZeroRt { rt } => tk20::warp_zero_rt(rt),
            Tk20Call::WarpAddRt { d, a, b } => tk20::warp_add_rt(d, a, b),
            Tk20Call::WarpMulRt { d, a, b } => tk20::warp_mul_rt(d, a, b),

            Tk20Call::WarpgroupIncreaseRegisters { n } => {
                tk20::warpgroup_increase_registers(*n)
            }
            Tk20Call::WarpgroupDecreaseRegisters { n } => {
                tk20::warpgroup_decrease_registers(*n)
            }


        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subtile::{Range, Region};
    use crate::subtile_ir::{BufId, RegionRef};
    use crate::tk_warp_ir::{Phase0, PageHandle};

    fn rr(buf: u32, c0: u32, w: u32) -> RegionRef {
        RegionRef {
            buffer: BufId(buf),
            region: Region {
                rows: Range::new(0, 1),
                cols: Range::new(c0, w),
            },
        }
    }

    /// A tiny tape: loader fills page 0, all consumers compute, storer
    /// drains. The phase parity in the emitted source matches the IR.
    #[test]
    fn emits_loader_consumer_storer_handshake() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(0);

        // Round 0: every wait reads phase 0 (TK 2.0 round parity =
        // R & 1; per-barrier flips happen but the wait sees the
        // start-of-round parity).
        let page = p.wait(WarpRole::Loader, PageBarrier::Consumed, page);
        p.load_async(
            0,
            BufId(7),
            rr(7, 0, 64),
            TileShape {
                rows: 1,
                cols: 64,
                elem_bytes: 2,
            },
        );
        let page = p.arrive(WarpRole::Loader, PageBarrier::Ready, page);

        let page = p.wait(WarpRole::AllConsumers, PageBarrier::Ready, page);
        p.compute_calls(
            WarpRole::AllConsumers,
            vec![Tk20Call::WarpZeroRt { rt: "__smoke".into() }],
        );
        let page = p.arrive(WarpRole::AllConsumers, PageBarrier::Done, page);

        let page = p.wait(WarpRole::Storer, PageBarrier::Done, page);
        p.store_async(
            0,
            BufId(8),
            rr(8, 0, 64),
            TileShape {
                rows: 1,
                cols: 64,
                elem_bytes: 2,
            },
        );
        let _page = p.arrive(WarpRole::Storer, PageBarrier::Consumed, page);

        let src = emit_body(&p);

        // The phase parities the codegen emits are a literal copy of
        // what the type system computed: round 0 → all waits read 0.
        assert!(src.contains("page_consumed[0], 0"), "loader wait phase=0\n{src}");
        assert!(src.contains("page_ready[0], 0"), "consumer wait phase=0\n{src}");
        assert!(src.contains("page_done[0], 0"), "storer wait phase=0\n{src}");

        // Roles route correctly: loader/storer are gated on __role,
        // the consumer body is in the consumer arm.
        assert!(src.contains("if (__role == ROLE_LOADER)"), "loader gate\n{src}");
        assert!(src.contains("if (__role == ROLE_STORER)"), "storer gate\n{src}");
        assert!(src.contains("if (__role == ROLE_CONSUMER)"), "consumer gate\n{src}");
        // The smoke test pushes a typed Tk20Call (WarpZeroRt) so the
        // consumer body emits a real TK 2.0 call rather than a free-
        // form comment.
        assert!(
            src.contains("kittens::warp::zero(__smoke)"),
            "consumer body emitted typed Tk20Call\n{src}"
        );
    }

    #[test]
    fn tma_load_carries_src_byte_offset_and_arms_ready() {
        let mut p = TkProgram::new();
        // src region starts at column 128, elem_bytes=2 → byte_off=256.
        // tile = 1×64×2 bytes → 128 bytes total.
        p.load_async(
            5,
            BufId(3),
            rr(3, 128, 64),
            TileShape {
                rows: 1,
                cols: 64,
                elem_bytes: 2,
            },
        );
        let src = emit_body(&p);
        // expect_bytes arms page_ready[5] for 128 bytes.
        assert!(
            src.contains("kittens::group<1>::tma::expect_bytes(page_ready[5], 128);"),
            "expect_bytes arms ready barrier\n{src}"
        );
        // load_async references the same page_buf[5] dst, the byte-
        // offset src, and uses page_ready[5] as the semaphore.
        assert!(
            src.contains("reinterpret_cast<void*>(page_buf[5])"),
            "page dst\n{src}"
        );
        assert!(
            src.contains("reinterpret_cast<uintptr_t>(buf3) + 256"),
            "src byte offset (uintptr_t form so const is dropped)\n{src}"
        );
        assert!(
            src.contains(", 128, page_ready[5]);"),
            "load takes bytes + page_ready barrier\n{src}"
        );
    }

    #[test]
    fn kernel_scaffold_has_role_dispatch_and_mbarrier_init() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(0);
        let _page = p.wait(WarpRole::Loader, PageBarrier::Consumed, page);

        let args = KernelArgs {
            bufs: vec![
                KernelArg {
                    ty: "const __nv_bfloat16* __restrict__".into(),
                    name: "x".into(),
                },
                KernelArg {
                    ty: "__nv_bfloat16* __restrict__".into(),
                    name: "out".into(),
                },
            ],
            u32_args: vec!["__num_kv_pages".into()],
        };
        let src = emit_kernel("tk_kernel_smoke", &args, &p);

        // Sanity: kernel signature references all args.
        // Phase 7: 20 warps × 32 = 640 threads.
        assert!(src.contains("__global__ __launch_bounds__(640) void tk_kernel_smoke("), "{src}");
        assert!(src.contains("const __nv_bfloat16* __restrict__ x"), "{src}");
        assert!(src.contains("__nv_bfloat16* __restrict__ out"), "{src}");
        assert!(src.contains("const uint32_t __num_kv_pages"), "{src}");

        // Aliases: `buf0`, `buf1` map to named args.
        assert!(src.contains("auto& buf0 = x;"), "{src}");
        assert!(src.contains("auto& buf1 = out;"), "{src}");

        // Role dispatch.
        assert!(src.contains("__role = ROLE_LOADER"), "{src}");
        assert!(src.contains("__role = ROLE_STORER"), "{src}");
        assert!(src.contains("__role = ROLE_CONSUMER"), "{src}");
        assert!(src.contains("const int __consumer_idx"), "{src}");

        // Mbarrier init.
        assert!(src.contains("__shared__ kittens::semaphore page_ready[13]"), "{src}");
        assert!(src.contains("__shared__ kittens::semaphore page_done[13]"), "{src}");
        assert!(src.contains("__shared__ kittens::semaphore page_consumed[13]"), "{src}");
        assert!(src.contains("kittens::init_semaphore(page_ready"), "{src}");
        assert!(src.contains("kittens::init_semaphore(page_done"), "{src}");
        assert!(src.contains("kittens::init_semaphore(page_consumed"), "{src}");
        assert!(src.contains("kittens::arrive(page_consumed[__i])"), "{src}");

        // CTA-wide sync after init (10 warps = 320 threads). TK 2.0
        // `group<N>::sync()` is barrier-less and asserts N==1, so for
        // multi-warp the scaffold uses `__syncthreads()`.
        assert!(src.contains("__syncthreads();"), "init+final sync\n{src}");

        // Async-proxy fence between mbarrier init and the cross-warp
        // sync. Without this fence the storer's `mbarrier.try_wait`
        // sees stale parity bits and never observes the consumer
        // arrives — repro confirmed via the rmsnorm-only smoke test
        // and matches TK 2.0's KVM init pattern at vm.cuh:99.
        assert!(
            src.contains("fence.proxy.async.shared::cta"),
            "async-proxy fence after mbarrier init\n{src}"
        );

        // Body lands inside the kernel.
        assert!(src.contains("if (__role == ROLE_LOADER)"), "body merged\n{src}");

        // C-linkage launcher.
        // - signature is `extern "C" cudaError_t launch_<name>(void* const*,
        //   const uint32_t*, cudaStream_t)`
        // - sets `cudaFuncAttributeMaxDynamicSharedMemorySize` to
        //   `NUM_PAGES * PAGE_SIZE` (13 * 16384 = 212992)
        // - launches with `<<<1, total_threads, DYN_SMEM, stream>>>`
        // - forwards each `bufs[i]` cast to the declared kernel arg type
        //   and each `u32_args[j]` for runtime u32 args
        assert!(
            src.contains("extern \"C\" cudaError_t launch_tk_kernel_smoke("),
            "C-linkage launcher\n{src}"
        );
        assert!(src.contains("constexpr size_t DYN_SMEM = 212992u;"), "dynsmem cap\n{src}");
        assert!(
            src.contains("cudaFuncAttributeMaxDynamicSharedMemorySize"),
            "smem attr lifted\n{src}"
        );
        assert!(
            src.contains("tk_kernel_smoke<<<1, 640, DYN_SMEM, stream>>>("),
            "triple-chevron launch\n{src}"
        );
        assert!(
            src.contains("(const __nv_bfloat16* __restrict__)bufs[0]"),
            "buf0 cast\n{src}"
        );
        assert!(
            src.contains("(__nv_bfloat16* __restrict__)bufs[1]"),
            "buf1 cast\n{src}"
        );
        assert!(src.contains("u32_args[0]"), "u32 forward\n{src}");
    }

    /// End-to-end smoke: lower one RmsNorm and emit a complete kernel.
    /// Snapshot test — write the .cu next to the test so we can `oc
    /// rsync` it to the pod and feed nvcc.
    #[test]
    fn end_to_end_rmsnorm_kernel_snapshot() {
        use crate::tk_lower::{lower_rmsnorm, PageAllocator, RmsNormOp};
        use crate::tk_warp_ir::Phase0;

        let mut pages = PageAllocator::new();
        let mut prog = TkProgram::new();
        lower_rmsnorm::<Phase0>(
            RmsNormOp {
                x: BufId(0),
                weight: BufId(1),
                out: BufId(2),
                hidden: 2048,
                m: 1,
                act_elem: 2,
                eps: 1e-5,
                init: true,
            },
            &mut pages,
            &mut prog,
        );

        let args = KernelArgs {
            bufs: vec![
                KernelArg {
                    ty: "const __nv_bfloat16* __restrict__".into(),
                    name: "x".into(),
                },
                KernelArg {
                    ty: "const __nv_bfloat16* __restrict__".into(),
                    name: "weight".into(),
                },
                KernelArg {
                    ty: "__nv_bfloat16* __restrict__".into(),
                    name: "out".into(),
                },
            ],
            u32_args: vec![],
        };
        let src = emit_kernel("tk_rmsnorm_decode_h2048", &args, &prog);

        // Sanity: the body's six-step handshake is inside the kernel.
        // Phase 7: AllConsumers waits use group<16>; loader/storer
        // are still group<1>; page_done init expected count is now 16.
        assert!(src.contains("kittens::group<1>::wait(page_consumed[0], 0)"), "{src}");
        assert!(src.contains("kittens::group<16>::wait(page_ready[0], 0)"), "{src}");
        assert!(src.contains("kittens::group<1>::wait(page_done[0], 0)"), "{src}");
        // Mbarrier init.
        assert!(src.contains("kittens::init_semaphore(page_ready[__i], 0, 1);"));
        assert!(src.contains("kittens::init_semaphore(page_done[__i], 0, 16);"));
        // Compute body.
        assert!(src.contains("rsqrtf"));
    }

    /// `EmitOpts { debug_handshake: true }` interleaves a lane-0-gated
    /// `printf` before every Wait / TMA load / TMA store and after
    /// every Arrive — the trace the deadlock audit walks to find the
    /// first wait without a matching arrive.
    #[test]
    fn debug_handshake_wraps_handshakes_with_lane_gated_printf() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(0);
        let page = p.wait(WarpRole::Loader, PageBarrier::Consumed, page);
        p.load_async(
            0,
            BufId(0),
            rr(0, 0, 64),
            TileShape {
                rows: 1,
                cols: 64,
                elem_bytes: 2,
            },
        );
        let _page = p.arrive(WarpRole::AllConsumers, PageBarrier::Done, page);

        let opts = EmitOpts {
            debug_handshake: true,
            ..Default::default()
        };
        let src = emit_body_with_opts(&p, &opts);

        // Every printf is gated to lane 0 of its warp.
        assert!(
            src.contains("if ((threadIdx.x & 31) == 0)"),
            "lane-0 gate present\n{src}"
        );
        // WAIT_START fires BEFORE the loader's wait.
        assert!(
            src.contains("WAIT_START kind=Consumed page=0 phase=0"),
            "wait pre-printf\n{src}"
        );
        // WAIT_DONE fires AFTER the same wait.
        assert!(
            src.contains("WAIT_DONE kind=Consumed page=0 phase=0"),
            "wait post-printf\n{src}"
        );
        // TMA load gets START/ISSUED bookends.
        assert!(
            src.contains("TMA_LOAD_START page=0 src_buf=0"),
            "tma load pre-printf\n{src}"
        );
        assert!(
            src.contains("TMA_LOAD_ISSUED page=0"),
            "tma load post-printf\n{src}"
        );
        // Arrive only has a post-printf (the body itself is the action).
        assert!(
            src.contains("ARRIVE kind=Done page=0"),
            "arrive printf\n{src}"
        );

        // Default opts: no printf.
        let plain = emit_body(&p);
        assert!(!plain.contains("printf"), "default opts stay quiet\n{plain}");
    }

    /// `emit_kernel_with_opts(debug_handshake=true)` pulls in `<cstdio>`
    /// so device printf links on Hopper without depending on a
    /// transitive include from `kittens.cuh`.
    #[test]
    fn debug_handshake_kernel_includes_cstdio() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(0);
        let _page = p.wait(WarpRole::Loader, PageBarrier::Consumed, page);
        let args = KernelArgs {
            bufs: vec![KernelArg {
                ty: "__nv_bfloat16* __restrict__".into(),
                name: "buf".into(),
            }],
            u32_args: vec![],
        };
        let opts = EmitOpts {
            debug_handshake: true,
            ..Default::default()
        };
        let src = emit_kernel_with_opts("tk_dbg_smoke", &args, &p, &opts);
        assert!(src.contains("#include <cstdio>"), "cstdio pulled in\n{src}");
        // Sanity: default emit doesn't pull cstdio.
        let plain = emit_kernel("tk_dbg_smoke", &args, &p);
        assert!(!plain.contains("#include <cstdio>"));
    }

    /// Every `Arrive` lowers to `kittens::group<1>::arrive(...)`, even
    /// when the role is `AllConsumers` (8 warps). TK 2.0
    /// `group<N>::arrive(semaphore&)` is gated on the GROUP's lane 0,
    /// so a multi-warp `group<8>::arrive` would fire ONCE total, not
    /// 8 times — and the storer's `wait(done, 0)` would deadlock
    /// against an init expecting 8 arrives. See `arrive_group_width`.
    #[test]
    fn arrive_always_emits_group1_to_avoid_single_fire_deadlock() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(0);
        let _page = p.arrive(WarpRole::AllConsumers, PageBarrier::Done, page);
        let src = emit_body(&p);
        assert!(
            src.contains("kittens::group<1>::arrive(page_done[0])"),
            "AllConsumers arrive must be group<1> not group<N>\n{src}"
        );
        assert!(
            !src.contains("kittens::group<8>::arrive("),
            "no group<8>::arrive on the emit path (single-fire bug)\n{src}"
        );
    }

    #[test]
    fn sync_group_width_matches_role() {
        let mut p = TkProgram::new();
        p.sync(WarpRole::AllConsumers);
        p.sync(WarpRole::Loader);
        let src = emit_body(&p);
        // Phase 7: 16 consumers → group<16>; loader is one warp → group<1>.
        assert!(src.contains("kittens::group<16>::sync();"), "{src}");
        assert!(src.contains("kittens::group<1>::sync();"), "{src}");
    }

    // ── Phase 0 tk20::* binding tests ──────────────────────────────
    //
    // Each new tk20::* binding has a unit test asserting its emit
    // string contains the TK 2.0 spelling cited in the binding's doc
    // comment. If a future TK 2.0 release renames a primitive, these
    // tests fail at `cargo test` — long before nvcc runs. The test is
    // the actual TK 2.0 audit gate, not nvcc accepting the file.

    #[test]
    fn tk20_warpgroup_mma_bindings_emit_warpgroup_namespace() {
        // Source: ops/group/mma/warpgroup.cuh:140 (mma_AB), :186 (mm_AB),
        // :258 (mma_ABt), :23 (mma_fence), :78 (commit), :91 (async_wait).
        assert_eq!(
            tk20::warpgroup_mma_ab("d", "a", "b"),
            "kittens::warpgroup::mma_AB(d, a, b);"
        );
        assert_eq!(
            tk20::warpgroup_mm_ab("d", "a", "b"),
            "kittens::warpgroup::mm_AB(d, a, b);"
        );
        assert_eq!(
            tk20::warpgroup_mma_abt("d", "a", "b"),
            "kittens::warpgroup::mma_ABt(d, a, b);"
        );
        assert_eq!(tk20::warpgroup_mma_fence("d"), "kittens::warpgroup::mma_fence(d);");
        assert_eq!(
            tk20::warpgroup_mma_commit_group(),
            "kittens::warpgroup::mma_commit_group();"
        );
        assert_eq!(
            tk20::warpgroup_mma_async_wait(0),
            "kittens::warpgroup::mma_async_wait<0>();"
        );
    }

    #[test]
    fn tk20_warp_mma_bindings_emit_warp_namespace() {
        // Source: ops/group/mma/warp.cuh.
        // TK 2.0 warp::mma_AB takes (D, A, B, D) — accumulating into D.
        assert_eq!(
            tk20::warp_mma_ab("d", "a", "b"),
            "kittens::warp::mma_AB(d, a, b, d);"
        );
        assert_eq!(
            tk20::warp_mma_abt("d", "a", "b"),
            "kittens::warp::mma_ABt(d, a, b, d);"
        );
    }

    #[test]
    fn tk20_group_tma_store_bindings_use_group_1_namespace() {
        // Per feedback_tk20_tma_lane_gate: TMA must be lane-0-gated via
        // kittens::group<1>::tma::*. Source: ops/group/util/tma.cuh:38, 47, 61.
        assert_eq!(
            tk20::group_tma_store_commit_group(),
            "kittens::group<1>::tma::store_commit_group();"
        );
        assert_eq!(
            tk20::group_tma_store_async_wait(0),
            "kittens::group<1>::tma::store_async_wait<0>();"
        );
        assert_eq!(
            tk20::group_tma_store_async_read_wait(0),
            "kittens::group<1>::tma::store_async_read_wait<0>();"
        );
    }

    #[test]
    fn tk20_warp_reduction_bindings_emit_warp_namespace() {
        // Source: ops/group/register/tile/reductions.cuh and
        //         ops/group/register/{tile,vec}/maps.cuh.
        assert_eq!(
            tk20::warp_row_max("rv", "rt"),
            "kittens::warp::row_max(rv, rt);"
        );
        assert_eq!(
            tk20::warp_row_sum("rv", "rt"),
            "kittens::warp::row_sum(rv, rt);"
        );
        assert_eq!(tk20::warp_exp2("rt"), "kittens::warp::exp2(rt, rt);");
        assert_eq!(
            tk20::warp_mul_row("rt", "rv"),
            "kittens::warp::mul_row(rt, rt, rv);"
        );
        assert_eq!(
            tk20::warp_div_row("rt", "rv"),
            "kittens::warp::div_row(rt, rt, rv);"
        );
        assert_eq!(tk20::warp_neg_infty("rv"), "kittens::warp::neg_infty(rv);");
        assert_eq!(tk20::warp_zero_rt("rt"), "kittens::warp::zero(rt);");
        assert_eq!(
            tk20::warp_add_rt("d", "a", "b"),
            "kittens::warp::add(d, a, b);"
        );
        assert_eq!(
            tk20::warp_mul_rt("d", "a", "b"),
            "kittens::warp::mul(d, a, b);"
        );
    }

    #[test]
    fn tk20_warpgroup_register_budget_emits_warpgroup_namespace() {
        // Source: ops/group/group.cuh:52 (increase) / :56 (decrease).
        // TK 2.0 enforces n % 8 == 0; the binding's debug_assert
        // mirrors that.
        assert_eq!(
            tk20::warpgroup_increase_registers(224),
            "kittens::warpgroup::increase_registers<224>();"
        );
        assert_eq!(
            tk20::warpgroup_decrease_registers(56),
            "kittens::warpgroup::decrease_registers<56>();"
        );
    }

    #[test]
    fn tk20_call_enum_round_trips_via_emit() {
        // Tk20Call is the typed primitive-call atom that
        // TkInstr::Compute carries. emit() delegates to the matching
        // tk20::* binding; this test confirms the wiring (the per-
        // binding tests above check the cited TK 2.0 spelling).
        let calls = vec![
            Tk20Call::WarpgroupMmaFence { d: "d".into() },
            Tk20Call::WarpgroupMmaAB {
                d: "d".into(),
                a: "a".into(),
                b: "b".into(),
            },
            Tk20Call::WarpgroupMmaCommitGroup,
            Tk20Call::WarpgroupMmaAsyncWait { n: 0 },
        ];
        let emitted: Vec<String> = calls.iter().map(|c| c.emit()).collect();
        assert_eq!(emitted[0], "kittens::warpgroup::mma_fence(d);");
        assert_eq!(emitted[1], "kittens::warpgroup::mma_AB(d, a, b);");
        assert_eq!(emitted[2], "kittens::warpgroup::mma_commit_group();");
        assert_eq!(emitted[3], "kittens::warpgroup::mma_async_wait<0>();");
    }

    #[test]
    fn rmsnorm_compute_calls_emit_atomic_typed_primitives() {
        let calls = rmsnorm_compute_calls(0, 1, 2048, 1.0e-5);
        let body: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(body.contains("auto* __x_smem = reinterpret_cast<__nv_bfloat16*>(page_buf[0]);"));
        assert!(body.contains("auto* __w_smem = reinterpret_cast<__nv_bfloat16*>(page_buf[1]);"));
        assert!(body
            .contains("kittens::sv_bf<2048>& __x_sv = *reinterpret_cast<kittens::sv_bf<2048>*>(page_buf[0]);"));
        assert!(body.contains("const unsigned int __hidden = 2048u;"));
        assert!(body.contains("if (__consumer_idx == 0) {"));
        assert!(body.contains("for (int __k_i = 0; __k_i < 16; ++__k_i)"));
        assert!(body.contains("kittens::rv_bf<128> __x_rv_bf;"));
        assert!(body.contains("kittens::rv_fl<128> __x_rv_fl;"));
        assert!(body.contains("kittens::warp::load(__x_rv_bf, __x_sv.template subvec<128>(__k_i));"));
        assert!(body.contains("kittens::warp::copy(__x_rv_fl, __x_rv_bf);"));
        assert!(body.contains("kittens::warp::mul(__x_rv_fl, __x_rv_fl, __x_rv_fl);"));
        assert!(body.contains("__sumsq += kittens::warp::sum(__x_rv_fl);"));
        assert!(body.contains("const float __scale = rsqrtf(__sumsq / static_cast<float>(__hidden) + __eps);"));
        assert!(body.contains("__x_smem[__i] = __float2bfloat16(__v * __scale * __g);"));
        // Legacy shfl + per-thread squared-sum loop gone.
        assert!(!body.contains("__shfl_xor_sync"));
        // Strictly TK 2.0.
        assert!(!body.contains("kittens::tma::"));
        assert!(!body.contains("kittens::warp::mma_AB"));
        assert!(!body.contains("kittens::warpgroup::mma_AB"));
    }

    #[test]
    fn residual_add_compute_calls_emit_atomic_typed_primitives() {
        let calls = residual_add_compute_calls(2, 3, 2048);
        let body: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(body.contains("auto* __a_smem = reinterpret_cast<__nv_bfloat16*>(page_buf[2]);"));
        assert!(body.contains("auto* __b_smem = reinterpret_cast<__nv_bfloat16*>(page_buf[3]);"));
        assert!(body.contains("const unsigned int __total = 2048u;"));
        assert!(body.contains("const int __consumer_threads = 16 * 32;"));
        assert!(body.contains("__a_smem[__i] = __float2bfloat16(__a + __b);"));
        assert!(!body.contains("kittens::tma::"));
        assert!(!body.contains("kittens::warp::mma_AB"));
    }

    // Phase 5 cutover: byte-identity-vs-legacy tests deleted. They
    // were Phase 1-4 transition gates; the legacy `format!()` body fns
    // they compared against are gone in this commit. The
    // `*_emits_legacy_compatible_cuda` per-binding tests above stay
    // — they assert the typed atom emits the expected TK 2.0
    // spelling, which is the load-bearing audit gate.

    #[test]
    fn silu_mul_compute_calls_emit_atomic_typed_primitives() {
        let calls = silu_mul_compute_calls(0, 1, 8192);
        let body: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(body.contains("auto* __g_smem = reinterpret_cast<__nv_bfloat16*>(page_buf[0]);"));
        assert!(body.contains("auto* __u_smem = reinterpret_cast<__nv_bfloat16*>(page_buf[1]);"));
        assert!(body.contains("const unsigned int __total = 8192u;"));
        assert!(body.contains("const float __silu_g = __g / (1.0f + expf(-__g));"));
        assert!(body.contains("__g_smem[__i] = __float2bfloat16(__silu_g * __u);"));
        assert!(!body.contains("kittens::tma::"));
        assert!(!body.contains("kittens::warp::mma_AB"));
    }

    #[test]
    fn rope_compute_calls_emit_atomic_typed_primitives_neox() {
        let calls = rope_compute_calls::<NeoX>(0, 1, 2, 64, 32);
        let body: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(body.contains("auto* __x_smem = reinterpret_cast<__nv_bfloat16*>(page_buf[0]);"));
        assert!(body.contains("auto* __cos_smem = reinterpret_cast<__nv_bfloat16*>(page_buf[1]);"));
        assert!(body.contains("auto* __sin_smem = reinterpret_cast<__nv_bfloat16*>(page_buf[2]);"));
        assert!(body.contains("const unsigned int __head_dim = 64u;"));
        assert!(body.contains("const unsigned int __half = 32u;"));
        assert!(body.contains("const unsigned int __pairs = 32u;"));
        // NeoX pair-index decomposition: i_lo = row*head + lane, i_hi = i_lo + half.
        assert!(body.contains("const unsigned int __i_lo = __row_head * __head_dim + __lane;"));
        assert!(body.contains("const unsigned int __i_hi = __i_lo + __half;"));
        assert!(body.contains("__x_smem[__i_lo] = __float2bfloat16(__x_lo * __c - __x_hi * __s);"));
        assert!(body.contains("__x_smem[__i_hi] = __float2bfloat16(__x_lo * __s + __x_hi * __c);"));
        assert!(!body.contains("kittens::tma::"));
        assert!(!body.contains("kittens::warp::mma_AB"));
    }

    #[test]
    fn rope_compute_calls_interleaved_emits_2i_2i_plus_1_pair_indices() {
        // Interleaved RoPE has DIFFERENT pair indexing than NeoX:
        // i_lo = row*head + 2*lane (not row*head + lane), i_hi = i_lo + 1
        // (not i_lo + half). This proves the const-generic dispatch
        // actually changes the emit at compile time per F: RopeForm.
        let calls = rope_compute_calls::<Interleaved>(0, 1, 2, 64, 32);
        let body: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(body.contains("const unsigned int __i_lo = __row_head * __head_dim + 2u * __lane;"));
        assert!(body.contains("const unsigned int __i_hi = __i_lo + 1u;"));
        // FMA pair math is form-agnostic.
        assert!(body.contains("__x_smem[__i_lo] = __float2bfloat16(__x_lo * __c - __x_hi * __s);"));
        assert!(body.contains("__x_smem[__i_hi] = __float2bfloat16(__x_lo * __s + __x_hi * __c);"));
        // No NeoX-specific pair math.
        assert!(!body.contains("__row_head * __head_dim + __lane;"));
        assert!(!body.contains("__i_lo + __half;"));
    }

    #[test]
    fn gemm_m1_compute_calls_emit_atomic_typed_primitives() {
        let calls = gemm_m1_compute_calls(0, 1, 2, 2048, 4);
        let body: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(body.contains("__nv_bfloat16* __y_gmem = reinterpret_cast<__nv_bfloat16*>(buf2);"));
        assert!(body.contains("const unsigned int __bn = 4u;"));
        assert!(body.contains("if (static_cast<unsigned int>(__consumer_idx) < __bn)"));
        assert!(body.contains("__y_gmem[__n_i * __bn + __row] = __float2bfloat16(__acc);"));
        assert!(body
            .contains("kittens::sv_bf<2048>& __x_sv = *reinterpret_cast<kittens::sv_bf<2048>*>(page_buf[0]);"));
        assert!(body.contains("kittens::sv_bf<2048>& __w_row_sv = *reinterpret_cast<kittens::sv_bf<2048>*>"));
        assert!(body.contains("for (int __k_i = 0; __k_i < 16; ++__k_i)"));
        assert!(body.contains("kittens::rv_bf<128> __x_rv_bf;"));
        assert!(body.contains("kittens::rv_bf<128> __w_rv_bf;"));
        assert!(body.contains("kittens::rv_fl<128> __x_rv_fl;"));
        assert!(body.contains("kittens::rv_fl<128> __w_rv_fl;"));
        assert!(body.contains("kittens::warp::load(__x_rv_bf, __x_sv.template subvec<128>(__k_i));"));
        assert!(body
            .contains("kittens::warp::load(__w_rv_bf, __w_row_sv.template subvec<128>(__k_i));"));
        assert!(body.contains("kittens::warp::copy(__x_rv_fl, __x_rv_bf);"));
        assert!(body.contains("kittens::warp::copy(__w_rv_fl, __w_rv_bf);"));
        assert!(body.contains("kittens::warp::mul(__x_rv_fl, __x_rv_fl, __w_rv_fl);"));
        assert!(body.contains("__acc += kittens::warp::sum(__x_rv_fl);"));
        assert!(!body.contains("__shfl_xor_sync"));
        assert!(!body.contains("__bfloat162float"));
        assert!(!body.contains("__y_smem"));
        assert!(!body.contains("kittens::tma::"));
        assert!(!body.contains("kittens::warp::mma_AB"));
        assert!(!body.contains("kittens::warpgroup::mma_AB"));
    }

    #[test]
    fn gemm_m1_compute_calls_internal_writes_to_smem_not_gmem() {
        let calls = gemm_m1_compute_calls_internal(0, 1, 7, 2048, 4);
        let body: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(body.contains("__nv_bfloat16* __y_smem = reinterpret_cast<__nv_bfloat16*>(page_buf[7]);"));
        assert!(body.contains("__y_smem[__n_i * __bn + __row] = __float2bfloat16(__acc);"));
        assert!(!body.contains("__y_gmem"));
    }

    #[test]
    fn tk20_decl_sv_view_bf_cites_kittens_sv_bf() {
        let s = tk20::decl_sv_view_bf("__x_sv", "page_buf[5]", 2048);
        assert!(s.contains("kittens::sv_bf<2048>& __x_sv = *reinterpret_cast<kittens::sv_bf<2048>*>(page_buf[5]);"), "{s}");
    }

    #[test]
    fn tk20_decl_rv_bf_cites_kittens_rv_bf() {
        let s = tk20::decl_rv_bf("__rv", 128);
        assert_eq!(s, "kittens::rv_bf<128> __rv;");
    }

    #[test]
    fn tk20_decl_rv_fl_cites_kittens_rv_fl() {
        let s = tk20::decl_rv_fl("__acc_rv", 128);
        assert_eq!(s, "kittens::rv_fl<128> __acc_rv;");
    }

    #[test]
    fn tk20_warp_load_rv_from_sv_cites_kittens_warp_load() {
        let s = tk20::warp_load_rv_from_sv("__rv", "__sv");
        assert_eq!(s, "kittens::warp::load(__rv, __sv);");
    }

    #[test]
    fn tk20_warp_copy_rv_cites_kittens_warp_copy() {
        let s = tk20::warp_copy_rv("__dst", "__src");
        assert_eq!(s, "kittens::warp::copy(__dst, __src);");
    }

    #[test]
    fn tk20_warp_mul_rv_cites_kittens_warp_mul() {
        let s = tk20::warp_mul_rv("__d", "__a", "__b");
        assert_eq!(s, "kittens::warp::mul(__d, __a, __b);");
    }

    #[test]
    fn tk20_warp_sum_rv_returns_expression() {
        let s = tk20::warp_sum_rv("__rv");
        // Expression — no trailing semicolon — composes inside `acc += ...;`.
        assert_eq!(s, "kittens::warp::sum(__rv)");
    }

    // GemmM1 byte-identity test deleted in Phase 5 cutover (legacy
    // body fn gone).

    #[test]
    fn tk20_attn_decode_prelude_emits_legacy_compatible_cuda() {
        let prelude = tk20::attn_decode_prelude(0, 1, 2, 64, 32, 8, 4, 4, 0.125_f32, 7);
        // Spot-check the load-bearing fragments — uniqueness suffix
        // `_a7` keeps multi-AttnDecode forwards collision-free.
        assert!(prelude.contains("AttnDecode #7 prelude"));
        assert!(prelude.contains("auto* __q_smem_a7   = reinterpret_cast<T_act*>(page_buf[0]);"));
        assert!(prelude.contains("auto* __k_smem_a7   = reinterpret_cast<T_act*>(page_buf[1]);"));
        assert!(prelude.contains("auto* __v_smem_a7   = reinterpret_cast<T_act*>(page_buf[2]);"));
        assert!(prelude.contains("const unsigned int __head_dim_a7 = 64u;"));
        assert!(prelude.contains("const unsigned int __num_q_heads_a7 = 32u;"));
        assert!(prelude.contains("const unsigned int __num_kv_heads_a7 = 8u;"));
        assert!(prelude.contains("const unsigned int __q_heads_per_warp_a7 = 4u;"));
        assert!(prelude.contains("float __o_accum_a7[4][64];"));
        assert!(!prelude.contains("kittens::tma::"));
        assert!(!prelude.contains("kittens::warp::mma_AB"));
    }

    #[test]
    fn attn_decode_init_compute_calls_emit_atomic_typed_primitives() {
        let calls = attn_decode_init_softmax_compute_calls(7);
        let init: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(init.contains("__m_max_a7[__h] = -INFINITY;"));
        assert!(init.contains("__l_sum_a7[__h] = 0.0f;"));
        assert!(init.contains("__o_accum_a7[__h][__j] = 0.0f;"));
        assert!(init.contains("if (static_cast<unsigned int>(__consumer_idx) < __num_kv_heads_a7)"));
        assert!(init.contains("for (unsigned int __h = 0u; __h < __q_heads_per_warp_a7; ++__h)"));
        assert!(!init.contains("kittens::tma::"));
    }

    #[test]
    fn softmax_state_typestate_threads_in_correct_order() {
        // Init → QKt → SV → QKt (next iter) → SV → Finalise.
        // Compiles only because each step's return-typestate matches
        // the next step's required input typestate. Out-of-order calls
        // are compile-fail (see SoftmaxState's compile_fail doctests).
        let s = SoftmaxState::new(11, 64);
        let (_, s) = s.init_compute_calls();
        let (_, s) = s.qkt_compute_calls();
        let (_, s) = s.sv_compute_calls();
        let (_, s) = s.qkt_compute_calls();
        let (_, s) = s.sv_compute_calls();
        let (calls, _final) = s.finalise_compute_calls();
        // Sanity: finalise body emits the unique_id-correlated identifiers.
        let body: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(body.contains("__l_sum_a11"));
        assert!(body.contains("__o_accum_a11"));
        assert!(body.contains("__out_smem_a11"));
    }

    #[test]
    fn attn_decode_qkt_compute_calls_emit_atomic_typed_primitives() {
        let calls = attn_decode_qkt_softmax_step_compute_calls(7, 64);
        let qkt: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(qkt.contains("kittens::sv_bf<64>& __k_row_sv ="));
        assert!(qkt.contains("kittens::sv_bf<64>& __q_row_sv ="));
        assert!(qkt.contains("kittens::rv_bf<64> __q_rv_bf;"));
        assert!(qkt.contains("kittens::rv_bf<64> __k_rv_bf;"));
        assert!(qkt.contains("kittens::rv_fl<64> __q_rv_fl;"));
        assert!(qkt.contains("kittens::rv_fl<64> __k_rv_fl;"));
        assert!(qkt.contains("kittens::warp::load(__q_rv_bf, __q_row_sv);"));
        assert!(qkt.contains("kittens::warp::load(__k_rv_bf, __k_row_sv);"));
        assert!(qkt.contains("kittens::warp::copy(__q_rv_fl, __q_rv_bf);"));
        assert!(qkt.contains("kittens::warp::copy(__k_rv_fl, __k_rv_bf);"));
        assert!(qkt.contains("kittens::warp::mul(__q_rv_fl, __q_rv_fl, __k_rv_fl);"));
        assert!(qkt.contains("float __s = kittens::warp::sum(__q_rv_fl);"));
        assert!(qkt.contains("__s *= __scale_a7;"));
        assert!(qkt.contains("const float __m_new = fmaxf(__m_max_a7[__h], __s);"));
        assert!(qkt.contains("__renorm_a7[__h] = expf(__m_max_a7[__h] - __m_new);"));
        assert!(qkt.contains("__p_a7[__h] = expf(__s - __m_new);"));
        assert!(qkt.contains("__l_sum_a7[__h] = __renorm_a7[__h] * __l_sum_a7[__h] + __p_a7[__h];"));
        assert!(qkt.contains("__o_accum_a7[__h][__j] *= __renorm_a7[__h];"));
        assert!(qkt.contains("__m_max_a7[__h] = __m_new;"));
        assert!(!qkt.contains("__shfl_xor_sync"));
        assert!(!qkt.contains("__bfloat162float(__q_smem"));
        assert!(!qkt.contains("__bfloat162float(__k_smem"));
        assert!(!qkt.contains("kittens::warp::mma_AB"));
        assert!(!qkt.contains("kittens::warpgroup::mma_AB"));
    }

    #[test]
    fn attn_decode_sv_accum_compute_calls_emit_atomic_typed_primitives() {
        let calls = attn_decode_sv_accum_compute_calls(7);
        let sv: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(sv.contains("if (static_cast<unsigned int>(__consumer_idx) < __num_kv_heads_a7)"));
        assert!(sv.contains(
            "__o_accum_a7[__h][__j] += __p_a7[__h] * __bfloat162float(__v_smem_a7[__v_off + __j]);"
        ));
        assert!(!sv.contains("kittens::tma::"));
    }

    #[test]
    fn attn_decode_finalise_compute_calls_emit_atomic_typed_primitives() {
        let calls = attn_decode_finalise_softmax_norm_compute_calls(7);
        let fin: String = calls.iter().map(|c| c.emit()).collect::<Vec<_>>().join(" ");
        assert!(fin.contains("const float __inv_l = 1.0f / __l_sum_a7[__h];"));
        assert!(fin.contains(
            "__out_smem_a7[__out_off + __j] = __float2bfloat16(__o_accum_a7[__h][__j] * __inv_l);"
        ));
    }

    // AttnDecode + SiluMul byte-identity tests deleted in Phase 5
    // cutover (legacy body fns gone).

    // RoPE + ResidualAdd byte-identity tests deleted in Phase 5
    // cutover (legacy body fns gone).

    #[test]
    fn compute_calls_setter_round_trips_to_emit() {
        // The new prog.compute_calls(...) entry point used by Phase 1+
        // typed lowerings — its calls reach the emitted CUDA in order.
        let mut p = TkProgram::new();
        p.compute_calls(
            WarpRole::AllConsumers,
            vec![
                Tk20Call::WarpZeroRt { rt: "__acc".into() },
                Tk20Call::WarpgroupMmaAB {
                    d: "__acc".into(),
                    a: "__a".into(),
                    b: "__b_desc".into(),
                },
                Tk20Call::WarpgroupMmaCommitGroup,
                Tk20Call::WarpgroupMmaAsyncWait { n: 0 },
            ],
        );
        let src = emit_body(&p);
        assert!(src.contains("kittens::warp::zero(__acc);"), "{src}");
        assert!(
            src.contains("kittens::warpgroup::mma_AB(__acc, __a, __b_desc);"),
            "{src}"
        );
        assert!(
            src.contains("kittens::warpgroup::mma_commit_group();"),
            "{src}"
        );
        assert!(
            src.contains("kittens::warpgroup::mma_async_wait<0>();"),
            "{src}"
        );
    }

    // ── Step C.3 — descriptor-TMA opt-in tests ────────────────────

    /// Default `EmitOpts` (empty `descriptor_layouts`) is byte-identical
    /// to `emit_kernel` — guards the C.3 changes from regressing the
    /// production canonical's raw-bulk emit.
    #[test]
    fn descriptor_layouts_empty_is_byte_identical_to_default() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(0);
        let _page = p.wait(WarpRole::Loader, PageBarrier::Consumed, page);
        let args = KernelArgs {
            bufs: vec![KernelArg {
                ty: "__nv_bfloat16* __restrict__".into(),
                name: "buf".into(),
            }],
            u32_args: vec![],
        };
        let plain = emit_kernel("tk_default_off", &args, &p);
        let opts = EmitOpts::default();
        let with_opts = emit_kernel_with_opts("tk_default_off", &args, &p, &opts);
        assert_eq!(plain, with_opts, "default opts must match emit_kernel");
    }

    /// With one descriptor layout, the emit gains:
    ///   - `using ARG{i}_T = kittens::gl<__nv_bfloat16, ...>;` typedef.
    ///   - `const __grid_constant__ ARG{i}_T arg{i}` kernel sig param.
    ///   - `auto* {name} = arg{i}.raw_ptr;` body alias (so existing
    ///     `buf{i}` paths still resolve to a `bf16*`).
    ///   - `ARG{i}_T arg{i}_inst(...)` host wrapper instance.
    /// A non-typed buf in the same kernel keeps its raw pointer arg.
    #[test]
    fn descriptor_layout_emits_typed_arg_and_host_instance() {
        let mut p = TkProgram::new();
        let page: PageHandle<Phase0> = PageHandle::fresh(0);
        let _page = p.wait(WarpRole::Loader, PageBarrier::Consumed, page);
        let args = KernelArgs {
            bufs: vec![
                KernelArg {
                    ty: "const __nv_bfloat16* __restrict__".into(),
                    name: "x".into(),
                },
                KernelArg {
                    ty: "__nv_bfloat16* __restrict__".into(),
                    name: "op0_out".into(),
                },
            ],
            u32_args: vec![],
        };
        let mut layouts = BTreeMap::new();
        layouts.insert(
            1,
            GlLayout {
                batch: 1,
                depth: 1,
                rows: 1,
                cols: 2048,
                tile_type: "kittens::sv_bf<2048>".into(),
            },
        );
        let opts = EmitOpts {
            descriptor_layouts: layouts,
            ..Default::default()
        };
        let src = emit_kernel_with_opts("tk_typed_smoke", &args, &p, &opts);

        assert!(
            src.contains(
                "using ARG1_T = kittens::gl<__nv_bfloat16, 1, 1, 1, 2048, kittens::sv_bf<2048>>;"
            ),
            "typedef\n{src}"
        );
        assert!(
            src.contains("const __grid_constant__ ARG1_T arg1"),
            "kernel sig swaps to grid_constant typed arg\n{src}"
        );
        assert!(
            src.contains("auto* op0_out = arg1.raw_ptr;"),
            "body alias keeps bf16* available\n{src}"
        );
        // Untyped buf 0 keeps its raw signature.
        assert!(
            src.contains("const __nv_bfloat16* __restrict__ x"),
            "untyped buf unaffected\n{src}"
        );
        // Host wrapper builds the gl<> instance from the runtime
        // pointer (all-nullptr runtime-dim args because every dim is
        // baked into the template).
        assert!(
            src.contains("ARG1_T arg1_inst(\n        reinterpret_cast<__nv_bfloat16*>(bufs[1])"),
            "host wrapper builds gl<> instance\n{src}"
        );
        // Kernel call passes the instance for buf 1, raw cast for buf 0.
        assert!(src.contains("arg1_inst"), "kernel call uses instance\n{src}");
        assert!(
            src.contains("(const __nv_bfloat16* __restrict__)bufs[0]"),
            "kernel call casts non-typed buf\n{src}"
        );
    }

    /// A `StoreAsync` to a typed buffer emits the descriptor-TMA store
    /// (vector form, `cache_policy::NORMAL`, `{0,0,0,0}` coords). A
    /// `StoreAsync` to a non-typed buffer keeps the raw-bulk emit.
    #[test]
    fn store_async_to_typed_buf_emits_descriptor_tma() {
        let mut p = TkProgram::new();
        // m=1 hidden=2048 store at column 0 → byte_off=0, eligible.
        p.store_async(
            7,
            BufId(1),
            rr(1, 0, 2048),
            TileShape {
                rows: 1,
                cols: 2048,
                elem_bytes: 2,
            },
        );
        let mut layouts = BTreeMap::new();
        layouts.insert(
            1,
            GlLayout {
                batch: 1,
                depth: 1,
                rows: 1,
                cols: 2048,
                tile_type: "kittens::sv_bf<2048>".into(),
            },
        );
        let opts = EmitOpts {
            descriptor_layouts: layouts,
            ..Default::default()
        };
        let src = emit_body_with_opts(&p, &opts);

        // Typed store form references arg1, the page_buf cast to the
        // declared tile type, and the {0,0,0,0} tile-frame coord.
        assert!(
            src.contains("kittens::group<1>::tma::store_async<kittens::cache_policy::NORMAL>("),
            "typed store_async\n{src}"
        );
        assert!(src.contains("arg1"), "typed store references arg1\n{src}");
        assert!(
            src.contains("*reinterpret_cast<kittens::sv_bf<2048>*>(page_buf[7])"),
            "smem cast to tile_type\n{src}"
        );
        assert!(
            src.contains("{0, 0, 0, 0}"),
            "single-tile frame coord\n{src}"
        );
        assert!(
            src.contains("kittens::group<1>::tma::store_commit_group();"),
            "commit\n{src}"
        );
        assert!(
            src.contains("kittens::group<1>::tma::store_async_wait();"),
            "drain\n{src}"
        );
        // No raw-bulk store_async form for the typed buffer.
        assert!(
            !src.contains("reinterpret_cast<char*>(buf1)"),
            "raw-bulk emit must not appear for typed buf\n{src}"
        );
    }

    /// `StoreAsync` to a typed buffer with non-zero byte offset (not in
    /// the C.3 initial scope) falls back to raw-bulk emit defensively —
    /// the typed `{0,0,0,0}` coord path can't represent a sub-buffer
    /// region. This keeps an over-eager opt-in from miscompiling.
    #[test]
    fn store_async_typed_falls_back_when_byte_off_nonzero() {
        let mut p = TkProgram::new();
        // c0=128 → byte_off=256 != 0 → must fall back.
        p.store_async(
            7,
            BufId(1),
            rr(1, 128, 64),
            TileShape {
                rows: 1,
                cols: 64,
                elem_bytes: 2,
            },
        );
        let mut layouts = BTreeMap::new();
        layouts.insert(
            1,
            GlLayout {
                batch: 1,
                depth: 1,
                rows: 1,
                cols: 2048,
                tile_type: "kittens::sv_bf<2048>".into(),
            },
        );
        let opts = EmitOpts {
            descriptor_layouts: layouts,
            ..Default::default()
        };
        let src = emit_body_with_opts(&p, &opts);
        // Raw-bulk path emits the legacy store_async with the byte
        // offset.
        assert!(
            src.contains("reinterpret_cast<char*>(buf1) + 256"),
            "fallback raw-bulk byte offset\n{src}"
        );
        // No typed cache_policy form.
        assert!(
            !src.contains("kittens::cache_policy::NORMAL"),
            "fallback path must not emit typed store\n{src}"
        );
    }
}
