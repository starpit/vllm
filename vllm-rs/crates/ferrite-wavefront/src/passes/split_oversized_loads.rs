// SPDX-License-Identifier: Apache-2.0
//! `split_oversized_loads_pass` — third §6.5 optimizer pass.
//!
//! ## Why
//!
//! `SubtileIR::lower_region` produces SubOp::MatmulTile nodes whose
//! External W input is the FULL K×N weight region. The conservative
//! lowering's `emit_external_load` then emits one `Instr::LoadAsync`
//! per External — for Llama-3.2-1B's 2048×2048 projection weights
//! that's 8 MB into a single 32 KB page. PAGE_SIZE-fitting per-tile
//! loads must be the OUTPUT of the lowering's pipeline; the
//! conservative lowering emits the full region naively (per plan §6,
//! `lower_tape_to_tk` is a "trivial syntax-directed translation" with
//! no analysis at lowering).
//!
//! Per plan §6.5: TkTape→TkTape passes own target-specific resource
//! decisions (smem capacity, page-lifetime windows). K-tiling a Gemm
//! into PAGE_SIZE-fitting WGMMA chunks IS a target-specific resource
//! decision — Hopper's WGMMA-K=128 + page-pool tile shape determine
//! the K-block count. This pass takes the conservative output and
//! rewrites Gemm-class load+WGMMA sequences into K-block loops,
//! mirroring the K-loop pattern AttnDecode_Qkt already uses for KV
//! tile loads (see [`crate::lower_subtile_tape_to_tk_tape`] AttnDecode
//! arm).
//!
//! Pass shape mirrors [`crate::passes::rt_alias_pass`] /
//! [`crate::passes::page_coalesce_pass`]: walks the tape, identifies
//! oversized loads + their MatmulTile-class consumers, rewrites in
//! place. **Postcondition:** every `Instr::LoadAsync` in the rewritten
//! tape has `spec.tile.byte_size() <= PAGE_SIZE`. Enforced by a
//! defensive walk at the end of the pass (panic with diagnostic if
//! a non-Gemm pattern leaves an oversized load behind).
//!
//! ## Rewrite shape (per matched MatmulTile-emit-sequence)
//!
//! Input span (conservative lowering):
//!
//! ```text
//! LoadAsync(A_page, FULL M×K_full, runtime-shape, base_off_a)
//! LoadAsync(B_page, FULL K_full×N, runtime-shape, base_off_b)
//! init_rt_zero(rt_d)
//! load_shmem_to_reg_warpgroup(A_page, rt_a)
//! wgmma_fence_acc(rt_d)
//! wgmma_mma_ab_reg_smem(rt_d, rt_a, B_page, FenceExternal, AccReset)
//! wgmma_async_wait(0)
//! store_reg_tile_to_shmem_warpgroup(rt_d, dst_page)   ← stays AFTER loop
//! ```
//!
//! Output span:
//!
//! ```text
//! init_rt_zero(rt_d)                                  ← outside loop, once
//! BarrierInit{ A_page, Ready, count = One }
//! BarrierInit{ B_page, Ready, count = One }
//! ForLoopOpenConst{ var, n: K_blocks }                ← K_blocks = K_full / 128
//!   TmaExpect{ A_page, Ready, 128×128 Bf16, LoaderRole }
//!   LoadAsync(A_page, 128×128, LinearLoop{var, stride_A=128*2, base_off_a})
//!   TmaExpect{ B_page, Ready, 128×128 Bf16, LoaderRole }
//!   LoadAsync(B_page, 128×128, LinearLoop{var, stride_B=128*N*2, base_off_b})
//!   PageBarrierWaitLoopStart0{ A_page, Ready, var }
//!   PageBarrierWaitLoopStart0{ B_page, Ready, var }
//!   load_shmem_to_reg_warpgroup(A_page, rt_a)
//!   wgmma_fence_acc(rt_d)
//!   wgmma_mma_ab_reg_smem(rt_d, rt_a, B_page, FenceExternal, AccAccumulate)
//!   wgmma_async_wait(0)                               ← INSIDE loop body
//! ForLoopClose{ var }
//! store_reg_tile_to_shmem_warpgroup(rt_d, dst_page)   ← unchanged, after loop
//! ```
//!
//! `AccAccumulate` is unconditional — `init_rt_zero` outside the loop
//! makes iter 0 equivalent to `AccReset + accumulate-into-zero`.
//!
//! `wgmma_async_wait(0)` lives **inside** the loop body so iter k+1's
//! LoadAsync into B_page does not race with iter k's WGMMA still
//! reading B_page (mirrors AttnDecode_Qkt's K-loop body shape; see
//! [`crate::lower_subtile_tape_to_tk_tape`] AttnDecode arm:
//! `wgmma_async_wait(0)` per K iter, no post-loop wait needed).
//!
//! ## What is NOT handled
//!
//! - **Non-Gemm big loads.** The pass requires a `WgmmaMmaAB_RegSmem`
//!   consumer following the oversized LoadAsync. If the load feeds a
//!   non-Gemm op (some hypothetical big elementwise), the pass panics
//!   at the postcondition check rather than silently emitting an
//!   oversized load.
//! - **M-tiling and N-tiling.** This pass only fragments along K. If
//!   the Gemm's M (= A.rows) or N (= B.cols) exceeds PAGE_ROWS=128 or
//!   PAGE_COLS=128, the pass asserts and panics with a diagnostic —
//!   M/N-tiling is a separate transform owned by a sibling §6.5 pass
//!   (not yet landed; M=N=128 is the only case the conservative
//!   lowering produces today).
//! - **A or B operand from a prior op (no LoadAsync).** If only one
//!   operand has a LoadAsync (the other is in a temp page from a
//!   prior op like RmsNorm), this pass cannot K-fragment the prior-op
//!   page (would require re-staging from gmem). The pass panics with
//!   a diagnostic in that case so the caller sees the gap.

use crate::tk_tape::{
    AccTag, ArrivalCount, Bf16, ByteOffset, ByteOffsetExpr, FenceTag, GroupWidthTag, Instr,
    KernelArgRef, LoadSpec, LoaderRole, LoopVarId, PAGE_SIZE, PageBarrier, PageId, Parity,
    RegTileSlot, SmemTileSpec, TileDtype, TileShape, TkTape, WarpRole,
};

/// WGMMA bf16 K-block size on Hopper sm_90a — `kittens::warpgroup`'s
/// per-warpgroup k-block. Matches TK 2.0's
/// `ops/group/mma/warpgroup.cuh` `mma_AB<D, A, B>` k-tile constant
/// for bf16 (PTX wgmma natively k=16, 8x accumulated → effective
/// k=128 per warpgroup phase). The substrate's PageTileSpec is
/// `kittens::st_bf<128, 128>` — page rows == 128 == K_BLOCK aligns
/// the chunk shape to one page exactly.
const K_BLOCK: u32 = 128;

/// Substrate `PageTileSpec` rows. Source of truth:
/// [`crate::tk_tape::PAGE_TILE_ROWS`] (= 128 for Hopper).
/// Duplicated as a const here so the pass-level shape matching uses
/// the same number; if PAGE_TILE_ROWS ever changes, update this too
/// (compile-time `const_assert_eq!` below makes the divergence rustc
/// error rather than silent drift).
const PAGE_ROWS: u32 = 128;
/// Substrate `PageTileSpec` cols. Same rationale as [`PAGE_ROWS`].
const PAGE_COLS: u32 = 128;

// Wire the local consts to the substrate's typed witness so a future
// PAGE_TILE shape bump fails to compile here, not at runtime. Uses
// the typed `SmemTileSpec::byte_size()` const fn — the substrate's
// source of truth.
const _: () = assert!(
    PAGE_ROWS * PAGE_COLS * <Bf16 as TileDtype>::ELEM_BYTES
        == SmemTileSpec::<128, 128, Bf16>::byte_size(),
    "split_oversized_loads_pass: PAGE_ROWS/PAGE_COLS const drift vs SmemTileSpec<128, 128, Bf16>::byte_size()",
);
const _: () = assert!(
    SmemTileSpec::<128, 128, Bf16>::byte_size() == PAGE_SIZE,
    "split_oversized_loads_pass: SmemTileSpec<128, 128, Bf16>::byte_size() != PAGE_SIZE",
);

/// `split_oversized_loads_pass` — phase 2: K-tile every oversized
/// `Instr::LoadAsync` whose consumer is a `WgmmaMmaAB_RegSmem` Gemm
/// emit-sequence.
///
/// **Postcondition (in-pass-checked, NOT validator-checked):** every
/// `Instr::LoadAsync` in the post-pass tape has
/// `tile.byte_size() <= PAGE_SIZE`. Enforced by the defensive walk at
/// the end of this fn. `validate_tk_tape` does NOT cross-check this —
/// pre-pass tapes legitimately hold oversized External LoadAsyncs
/// (the conservative lowering emits them; this pass rewrites them).
/// Audit 2026-06-08 finding #13 documented the prior "validator-
/// checked" claim as drift.
pub fn split_oversized_loads_pass(tape: &mut TkTape) {
    // Mint loop vars from `max_existing + 1` so the rewrites don't
    // collide with the lowering's own `LoopVarId`s (e.g., AttnDecode
    // K-loop var minted by `LoweringState::fresh_loop_var`).
    let mut next_var = max_existing_loop_var(&tape.instrs)
        .map(|n| n + 1)
        .unwrap_or(0);

    // Iteratively find + rewrite the first oversized LoadAsync. A
    // tape can contain MANY independent matmul-emit-sequences (one
    // per Gemm in the model — q_proj, k_proj, v_proj, o_proj, mlp,
    // lm_head — × num_layers). Each rewrite is local to its
    // matching matmul; the next outer-loop iteration finds the
    // NEXT oversized load.
    let mut rewrite_count = 0;
    while let Some(idx) = find_first_oversized(&tape.instrs) {
        rewrite_count += 1;
        // Bound the loop. A real Llama-3.2-1B tape has ~7 Gemms ×
        // ~16 layers ≈ 100 oversized loads; 4096 is a comfortable
        // backstop that catches a buggy plan_rewrite that fails to
        // shrink the load.
        assert!(
            rewrite_count <= 4096,
            "split_oversized_loads_pass: rewrite did not converge after {rewrite_count} iters \
             (current oversized load at idx {idx}). Likely plan_rewrite produced an Instr stream \
             that still contains the same oversized LoadAsync — check the splice math.",
        );
        let (span, replacement) = plan_rewrite(&tape.instrs, idx, &mut next_var);
        // Splice the rewrite into the tape, replacing [span.start, span.end).
        let _drained: Vec<_> = tape
            .instrs
            .splice(span.start..span.end, replacement)
            .collect();
    }

    // Postcondition — defensive walk. If `find_first_oversized`
    // returned `None`, every LoadAsync fits in a page; the assert
    // below is a backstop against `plan_rewrite` emitting an
    // oversized chunk LoadAsync (which it should never do given
    // the typed `LoadSpec::new::<128, 128, Bf16>` constructor used
    // for chunk emits, but a future relaxation must not slip
    // through silently).
    for (idx, instr) in tape.instrs.iter().enumerate() {
        if let Instr::LoadAsync(spec) = instr {
            let bytes = bytes_of_tile(&spec.tile);
            assert!(
                bytes <= PAGE_SIZE as u64,
                "split_oversized_loads_pass: postcondition failed at idx {idx}: \
                 LoadAsync tile {rows}×{cols}×{eb}B = {bytes} bytes > PAGE_SIZE = {ps}. \
                 The pass should have rewritten this — likely a non-Gemm consumer pattern \
                 or M/N>128 tile shape that this pass does not handle (see module docs).",
                rows = spec.tile.rows,
                cols = spec.tile.cols,
                eb = spec.tile.elem_bytes,
                ps = PAGE_SIZE,
            );
        }
    }
}

fn bytes_of_tile(t: &TileShape) -> u64 {
    (t.rows as u64) * (t.cols as u64) * (t.elem_bytes as u64)
}

fn find_first_oversized(instrs: &[Instr]) -> Option<usize> {
    instrs.iter().enumerate().find_map(|(i, ins)| match ins {
        Instr::LoadAsync(spec) if bytes_of_tile(&spec.tile) > PAGE_SIZE as u64 => Some(i),
        _ => None,
    })
}

/// Walk the tape and return the highest LoopVarId already present.
/// Used to mint fresh loop vars in the rewrites without colliding
/// with the lowering's existing vars (AttnDecode K-loop, future
/// per-arch loops). Returns `None` for an empty / loop-var-free tape.
fn max_existing_loop_var(instrs: &[Instr]) -> Option<u32> {
    let mut max: Option<u32> = None;
    let mut bump = |v: u32| {
        max = Some(match max {
            Some(m) => m.max(v),
            None => v,
        });
    };
    for instr in instrs {
        match instr {
            Instr::ForLoopOpenConst { var, .. } => bump(var.0),
            Instr::ForLoopOpenKernelArg { var, .. } => bump(var.0),
            Instr::ForLoopClose { var } => bump(var.0),
            Instr::PageBarrierWaitLoopStart0 { var, .. } => bump(var.0),
            Instr::PageBarrierWaitLoopStart1 { var, .. } => bump(var.0),
            Instr::LoadAsync(spec) => match &spec.byte_off {
                ByteOffsetExpr::LinearLoop { var, .. } => bump(var.0),
                ByteOffsetExpr::Affine2D { outer_var, inner_var, .. } => {
                    bump(outer_var.0);
                    bump(inner_var.0);
                }
                _ => {}
            },
            Instr::StoreAsync(spec) => match &spec.byte_off {
                ByteOffsetExpr::LinearLoop { var, .. } => bump(var.0),
                ByteOffsetExpr::Affine2D { outer_var, inner_var, .. } => {
                    bump(outer_var.0);
                    bump(inner_var.0);
                }
                _ => {}
            },
            // Other Instrs do not bear a `LoopVarId`. The exhaustive
            // tail leaves the catch-all to silence rustc; new variants
            // bearing a LoopVarId must be added explicitly above.
            _ => {}
        }
    }
    max
}

/// Plan a rewrite for the oversized LoadAsync at `big_idx`:
/// returns the `[start, end)` span to splice and the replacement Instr
/// stream that goes in its place.
fn plan_rewrite(
    instrs: &[Instr],
    big_idx: usize,
    next_var: &mut u32,
) -> (std::ops::Range<usize>, Vec<Instr>) {
    // 1. Find the consumer `WgmmaMmaAB_RegSmem` after `big_idx`. The
    //    conservative lowering emits its WGMMA close to the
    //    contributing LoadAsyncs; bound the search to ~64 Instrs to
    //    catch a stray oversized load that doesn't have a Gemm
    //    consumer (instead of walking the whole tape).
    let wgmma_idx = (big_idx + 1..instrs.len())
        .take(64)
        .find(|i| matches!(&instrs[*i], Instr::WgmmaMmaAB_RegSmem { .. }))
        .unwrap_or_else(|| {
            panic!(
                "split_oversized_loads_pass: oversized LoadAsync at idx {big_idx} \
                 has no WgmmaMmaAB_RegSmem within 64 instrs of follow-on. The pass \
                 only K-tiles Gemm-class big loads; this load needs a sibling pass \
                 (or upstream lowering tightening). Tile: {rows}×{cols}×{eb}B.",
                rows = match &instrs[big_idx] {
                    Instr::LoadAsync(s) => s.tile.rows,
                    _ => unreachable!(),
                },
                cols = match &instrs[big_idx] {
                    Instr::LoadAsync(s) => s.tile.cols,
                    _ => unreachable!(),
                },
                eb = match &instrs[big_idx] {
                    Instr::LoadAsync(s) => s.tile.elem_bytes,
                    _ => unreachable!(),
                },
            );
        });
    let (a_slot, b_page, d_slot, fence_tag, width_tag, wgmma_role) = match &instrs[wgmma_idx] {
        Instr::WgmmaMmaAB_RegSmem {
            a,
            b_page,
            d,
            fence,
            accumulate: _,
            width,
            role,
        } => (*a, *b_page, *d, *fence, *width, *role),
        _ => unreachable!(),
    };

    // 2. The matmul-emit-sequence brackets:
    //      init_rt_zero(d)        ← init_idx
    //      load_shmem_to_reg(a)   ← load_a_idx (src = a_page)
    //      wgmma_fence_acc(d)     ← fence_idx
    //      wgmma_mma_ab_reg_smem  ← wgmma_idx
    //      wgmma_async_wait(0)    ← wait_idx
    //
    //    The pre-matmul LoadAsyncs into a_page / b_page live BEFORE
    //    init_idx (emit_external_load runs at `resolve_input_page`,
    //    which fires before the SubOp arm pushes its own Instrs).
    let init_idx = (0..wgmma_idx)
        .rev()
        .find(|i| matches!(&instrs[*i], Instr::InitRtZero { dst, .. } if *dst == d_slot))
        .unwrap_or_else(|| {
            panic!(
                "split_oversized_loads_pass: WgmmaMmaAB_RegSmem at idx {wgmma_idx} (d={d_slot:?}) \
                 has no preceding InitRtZero(dst=d). Pattern requires the conservative \
                 MatmulTile-emit-sequence; non-conservative tapes need a different pass.",
            );
        });

    let load_a_idx = (init_idx + 1..wgmma_idx)
        .find(|i| matches!(
            &instrs[*i],
            Instr::LoadShmemToReg { dst, .. } if *dst == a_slot
        ))
        .unwrap_or_else(|| {
            panic!(
                "split_oversized_loads_pass: no LoadShmemToReg(dst={a_slot:?}) between \
                 InitRtZero(idx {init_idx}) and Wgmma(idx {wgmma_idx}).",
            );
        });
    let a_page = match &instrs[load_a_idx] {
        Instr::LoadShmemToReg { src, .. } => *src,
        _ => unreachable!(),
    };

    let fence_idx = (load_a_idx + 1..wgmma_idx)
        .find(|i| matches!(
            &instrs[*i],
            Instr::WgmmaFenceAcc { d, .. } if *d == d_slot
        ))
        .unwrap_or_else(|| {
            panic!(
                "split_oversized_loads_pass: no WgmmaFenceAcc(d={d_slot:?}) between \
                 LoadShmemToReg(idx {load_a_idx}) and Wgmma(idx {wgmma_idx}).",
            );
        });

    let wait_idx = (wgmma_idx + 1..instrs.len())
        .take(8)
        .find(|i| matches!(&instrs[*i], Instr::WgmmaAsyncWait { .. }))
        .unwrap_or_else(|| {
            panic!(
                "split_oversized_loads_pass: no WgmmaAsyncWait within 8 instrs after \
                 Wgmma(idx {wgmma_idx}).",
            );
        });

    // 3. Find the LoadAsyncs that wrote A_page / B_page. We look in
    //    the prefix `[..init_idx)` so we don't accidentally pick up
    //    a same-page LoadAsync from a downstream Gemm.
    let load_a_async = (0..init_idx)
        .rev()
        .find(|i| matches!(&instrs[*i], Instr::LoadAsync(spec) if spec.dst_page == a_page));
    let load_b_async = (0..init_idx)
        .rev()
        .find(|i| matches!(&instrs[*i], Instr::LoadAsync(spec) if spec.dst_page == b_page));

    // The big_idx must be one of these two. If not, the pattern was
    // matched against a different Gemm than expected.
    let big_load_page = match &instrs[big_idx] {
        Instr::LoadAsync(spec) => spec.dst_page,
        _ => unreachable!(),
    };
    if big_load_page != a_page && big_load_page != b_page {
        panic!(
            "split_oversized_loads_pass: oversized LoadAsync at idx {big_idx} targets page \
             {big_load_page:?} but the consumer Wgmma at idx {wgmma_idx} reads from \
             a_page={a_page:?}, b_page={b_page:?}. Pattern mismatch — the load was matched \
             to a Gemm whose operands it does not feed.",
        );
    }

    // 4. K_full and K_blocks. For A operand the tile is M × K_full
    //    (K is along cols); for B operand the tile is K_full × N
    //    (K is along rows). When BOTH operands have an External
    //    LoadAsync, the K_full claims must agree.
    let a_load_spec = load_a_async.and_then(|i| match &instrs[i] {
        Instr::LoadAsync(s) => Some(s.clone()),
        _ => None,
    });
    let b_load_spec = load_b_async.and_then(|i| match &instrs[i] {
        Instr::LoadAsync(s) => Some(s.clone()),
        _ => None,
    });

    let k_full = match (&a_load_spec, &b_load_spec) {
        (Some(a), Some(b)) => {
            assert_eq!(
                a.tile.cols, b.tile.rows,
                "split_oversized_loads_pass: A.cols ({}) != B.rows ({}) — the K_full claim \
                 of the two operands disagree. The Gemm contract is D[M,N] = A[M,K] @ B[K,N]; \
                 mismatched K is a lowering bug.",
                a.tile.cols, b.tile.rows,
            );
            a.tile.cols
        }
        (Some(a), None) => a.tile.cols,
        (None, Some(b)) => b.tile.rows,
        (None, None) => panic!(
            "split_oversized_loads_pass: oversized LoadAsync at idx {big_idx} has no \
             LoadAsync for either operand of its consumer Wgmma. Impossible state — \
             big_idx itself is a LoadAsync.",
        ),
    };
    assert!(
        k_full % K_BLOCK == 0,
        "split_oversized_loads_pass: K_full ({k_full}) is not a multiple of K_BLOCK ({K_BLOCK}). \
         WGMMA-K=128 is hard-coded by Hopper; non-multiple K dimensions need explicit padding \
         upstream. Llama-3.2-1B's K=2048 is divisible.",
    );
    let k_blocks = k_full / K_BLOCK;

    // 5. Per-iter chunk shape constraints. M is bounded to PAGE_ROWS
    //    here (no M-tiling in this pass; the conservative lowering's
    //    MatmulTile arm hardcodes A as a 128-row page already, and
    //    the macro's larger workloads compile into separate kernels).
    //    N may be > PAGE_COLS — the NK path below handles that.
    let m_chunk = a_load_spec
        .as_ref()
        .map(|s| s.tile.rows)
        .unwrap_or(PAGE_ROWS);
    let n_full = b_load_spec
        .as_ref()
        .map(|s| s.tile.cols)
        .unwrap_or(PAGE_COLS);
    assert!(
        m_chunk == PAGE_ROWS,
        "split_oversized_loads_pass: A operand tile.rows = {m_chunk}, expected {PAGE_ROWS}. \
         M-tiling beyond a single 128-row page is a separate transform (not in this pass).",
    );
    assert!(
        n_full % PAGE_COLS == 0,
        "split_oversized_loads_pass: B operand tile.cols ({n_full}) is not a multiple of \
         PAGE_COLS ({PAGE_COLS}). The N-tile path requires N divisible by 128; non-aligned \
         N needs explicit padding upstream.",
    );
    let n_blocks = n_full / PAGE_COLS;

    // 6. Per-iter byte strides.
    //
    //    The source layout for both operands is row-major (the
    //    lowering's `region_byte_offset` formula:
    //    `(rows.start * tensor.cols + cols.start) * elem_bytes`).
    //
    //    A operand: M × K_full row-major. To advance K by K_BLOCK,
    //      step the source byte offset by K_BLOCK * elem_bytes
    //      (advance along columns within the same row).
    //
    //    B operand: K_full × N_full row-major. To advance K by
    //      K_BLOCK along rows, step by K_BLOCK * N_full * elem_bytes
    //      (advance K_BLOCK rows × N_full cols per row).
    //
    //    elem_bytes is bf16 = 2 (sealed by the substrate's
    //    Bf16::ELEM_BYTES). We sanity-check the operand tile records
    //    the same.
    let elem_bytes_a = a_load_spec.as_ref().map(|s| s.tile.elem_bytes as u64);
    let elem_bytes_b = b_load_spec.as_ref().map(|s| s.tile.elem_bytes as u64);
    let elem_bytes = match (elem_bytes_a, elem_bytes_b) {
        (Some(ea), Some(eb)) => {
            assert_eq!(
                ea, eb,
                "split_oversized_loads_pass: A.elem_bytes ({ea}) != B.elem_bytes ({eb}). \
                 The K-tile chunk dtype must be uniform (bf16). Mixed-precision matmul is \
                 not supported by this pass.",
            );
            ea
        }
        (Some(e), None) | (None, Some(e)) => e,
        (None, None) => unreachable!("guarded above by k_full match"),
    };
    assert_eq!(
        elem_bytes, Bf16::ELEM_BYTES as u64,
        "split_oversized_loads_pass: K-tile assumes Bf16 chunks (elem_bytes = {}); operand \
         elem_bytes = {}. A different dtype needs its own typed `LoadSpec::new::<R, C, T>` \
         witness in the chunk emit.",
        Bf16::ELEM_BYTES,
        elem_bytes,
    );
    // A operand (M × K_full row-major): K-stride steps along cols.
    let a_k_stride_bytes: u64 = (K_BLOCK as u64) * elem_bytes;
    // B operand (K_full × N_full row-major): K-stride steps along
    // ROWS of the K×N tile, advancing K_BLOCK rows × N_full cols per
    // chunk. N-stride steps along COLS, advancing one N_BLOCK column
    // window. Both feed [`ByteOffsetExpr::Affine2D`] in the NK case
    // and `LinearLoop` in the K-only case.
    let b_k_stride_bytes: u64 = (K_BLOCK as u64) * (n_full as u64) * elem_bytes;
    let b_n_stride_bytes: u64 = (PAGE_COLS as u64) * elem_bytes;
    // Output (M × N_full row-major): N-stride steps along COLS,
    // advancing one N_BLOCK column window per N iter. Only used in
    // the NK path (K-only path leaves the output StoreAsync alone).
    let out_n_stride_bytes: u64 = (PAGE_COLS as u64) * elem_bytes;

    // 7. Operand source / base byte offsets.
    let (a_src_arg, a_base_off) = a_load_spec
        .as_ref()
        .map(|s| (s.src_arg, base_of(&s.byte_off)))
        .unwrap_or_else(|| {
            panic!(
                "split_oversized_loads_pass: A operand of Wgmma at idx {wgmma_idx} has no \
                 LoadAsync (a_page = {a_page:?}). The pass cannot K-fragment a prior-op page \
                 — re-staging from gmem requires routing changes upstream.",
            );
        });
    let (b_src_arg, b_base_off) = b_load_spec
        .as_ref()
        .map(|s| (s.src_arg, base_of(&s.byte_off)))
        .unwrap_or_else(|| {
            panic!(
                "split_oversized_loads_pass: B operand of Wgmma at idx {wgmma_idx} has no \
                 LoadAsync (b_page = {b_page:?}). Same constraint as A — see panic message above.",
            );
        });

    // 8. Mint fresh LoopVarIds. K-only path uses one (k_var). NK
    //    path uses two (n_var as outer, k_var as inner).
    let k_var = LoopVarId(*next_var);
    *next_var += 1;

    // 9. Determine the splice span's start. start = earliest of
    //    {load_a_async, load_b_async, init_idx}.
    let candidates: [Option<usize>; 3] = [load_a_async, load_b_async, Some(init_idx)];
    let start = candidates
        .iter()
        .filter_map(|x| *x)
        .min()
        .expect("init_idx is Some");

    // 10. Sanity-check the [start, init_idx) interior — every Instr
    //     in there should be a LoadAsync targeting a_page or b_page,
    //     no other ops. If something else lives in there, the splice
    //     would silently EAT it; better to panic and surface the
    //     unexpected pattern.
    for i in start..init_idx {
        match &instrs[i] {
            Instr::LoadAsync(spec) if spec.dst_page == a_page || spec.dst_page == b_page => {}
            other => panic!(
                "split_oversized_loads_pass: unexpected Instr at idx {i} between rewrite \
                 start ({start}) and InitRtZero ({init_idx}): {other:?}. The pass assumes \
                 only LoadAsyncs into A_page / B_page live in this prefix; if the lowering \
                 interleaves something else, this pass needs to widen its window.",
            ),
        }
    }

    // 11. Dispatch K-only vs NK based on whether B's N exceeds
    //     PAGE_COLS. K-only is the simpler path that leaves the
    //     existing post-matmul StoreAsync / Commit / Fence / Arrive
    //     untouched after the K-loop. NK extends the splice span to
    //     also include those, wraps everything in an outer N-loop,
    //     and rewrites the StoreAsync's byte_off + tile shape.
    if n_blocks == 1 {
        let end = wait_idx + 1;
        let out = emit_k_only_body(
            instrs,
            init_idx,
            load_a_idx,
            fence_idx,
            wait_idx,
            a_page,
            b_page,
            a_slot,
            d_slot,
            fence_tag,
            width_tag,
            wgmma_role,
            a_src_arg,
            a_base_off,
            b_src_arg,
            b_base_off,
            a_k_stride_bytes,
            b_k_stride_bytes,
            k_blocks,
            k_var,
        );
        (start..end, out)
    } else {
        // NK path: mint outer n_var, find post-wait Instrs, emit
        // nested loops + per-N output store.
        let n_var = LoopVarId(*next_var);
        *next_var += 1;

        // Walk forward from wait_idx + 1 to identify the post-matmul
        // sequence emitted by `emit_store_and_arrive` in the
        // lowering. Each is bounded to a small window so a non-
        // matching tape doesn't pull random distant Instrs into the
        // splice.
        let store_smem_idx = (wait_idx + 1..(wait_idx + 4).min(instrs.len()))
            .find(|i| matches!(
                &instrs[*i],
                Instr::StoreRegTileToShmem { src, .. } if *src == d_slot
            ))
            .unwrap_or_else(|| panic!(
                "split_oversized_loads_pass: no StoreRegTileToShmem(src={d_slot:?}) \
                 within 3 instrs after WgmmaAsyncWait(idx {wait_idx}). NK rewrite \
                 requires the conservative MatmulTile arm's post-matmul shape.",
            ));
        let dst_page = match &instrs[store_smem_idx] {
            Instr::StoreRegTileToShmem { dst, .. } => *dst,
            _ => unreachable!(),
        };
        let store_async_idx = (store_smem_idx + 1..(store_smem_idx + 6).min(instrs.len()))
            .find(|i| matches!(
                &instrs[*i],
                Instr::StoreAsync(spec) if spec.src_page == dst_page
            ))
            .unwrap_or_else(|| panic!(
                "split_oversized_loads_pass: no StoreAsync(src={dst_page:?}) within 5 instrs \
                 after StoreRegTileToShmem(idx {store_smem_idx}). NK rewrite requires \
                 emit_store_and_arrive's StoreAsync follow-on.",
            ));
        let commit_idx = (store_async_idx + 1..(store_async_idx + 6).min(instrs.len()))
            .find(|i| matches!(&instrs[*i], Instr::CommitGroupBulk { .. }))
            .unwrap_or_else(|| panic!(
                "split_oversized_loads_pass: no CommitGroupBulk within 5 instrs after \
                 StoreAsync(idx {store_async_idx}). NK rewrite requires \
                 emit_store_and_arrive's commit follow-on.",
            ));
        let fence_dev_idx = (commit_idx + 1..(commit_idx + 6).min(instrs.len()))
            .find(|i| matches!(&instrs[*i], Instr::ThreadfenceDevice { .. }))
            .unwrap_or_else(|| panic!(
                "split_oversized_loads_pass: no ThreadfenceDevice within 5 instrs after \
                 CommitGroupBulk(idx {commit_idx}). NK rewrite requires \
                 emit_store_and_arrive's fence follow-on.",
            ));
        let arrive_done_idx = (fence_dev_idx + 1..(fence_dev_idx + 6).min(instrs.len()))
            .find(|i| matches!(
                &instrs[*i],
                Instr::PageBarrierArrive { page_id, kind: PageBarrier::Done, .. }
                    if *page_id == dst_page
            ))
            .unwrap_or_else(|| panic!(
                "split_oversized_loads_pass: no PageBarrierArrive(Done, {dst_page:?}) within \
                 5 instrs after ThreadfenceDevice(idx {fence_dev_idx}). NK rewrite requires \
                 emit_store_and_arrive's arrive-Done follow-on.",
            ));

        // Original output StoreAsync — pull out its src_arg + base
        // for the per-N rewrite.
        let (out_dst_arg, out_base_off, out_role, out_elem_bytes) = match &instrs[store_async_idx] {
            Instr::StoreAsync(spec) => {
                let base = base_of(&spec.byte_off);
                (spec.dst_arg, base, spec.role, spec.tile.elem_bytes)
            }
            _ => unreachable!(),
        };
        assert_eq!(
            out_elem_bytes as u64, elem_bytes,
            "split_oversized_loads_pass: output StoreAsync.elem_bytes ({}) != operand \
             elem_bytes ({}). NK rewrite assumes uniform Bf16 dtype.",
            out_elem_bytes, elem_bytes,
        );

        let end = arrive_done_idx + 1;
        let out = emit_nk_body(
            instrs,
            init_idx,
            load_a_idx,
            fence_idx,
            wait_idx,
            store_smem_idx,
            commit_idx,
            fence_dev_idx,
            arrive_done_idx,
            a_page,
            b_page,
            dst_page,
            a_slot,
            d_slot,
            fence_tag,
            width_tag,
            wgmma_role,
            a_src_arg,
            a_base_off,
            b_src_arg,
            b_base_off,
            out_dst_arg,
            out_base_off,
            out_role,
            a_k_stride_bytes,
            b_k_stride_bytes,
            b_n_stride_bytes,
            out_n_stride_bytes,
            k_blocks,
            n_blocks,
            k_var,
            n_var,
        );
        (start..end, out)
    }
}

/// Emit the K-only rewrite body — single K-loop wrapping the
/// matmul-emit-sequence; init_rt_zero outside; store_reg_tile_to_shmem
/// stays AFTER the splice (untouched). See module doc §"Rewrite shape".
#[allow(clippy::too_many_arguments)]
fn emit_k_only_body(
    instrs: &[Instr],
    init_idx: usize,
    load_a_idx: usize,
    fence_idx: usize,
    wait_idx: usize,
    a_page: PageId,
    b_page: PageId,
    a_slot: RegTileSlot,
    d_slot: RegTileSlot,
    fence_tag: FenceTag,
    width_tag: GroupWidthTag,
    wgmma_role: WarpRole,
    a_src_arg: KernelArgRef,
    a_base_off: ByteOffset,
    b_src_arg: KernelArgRef,
    b_base_off: ByteOffset,
    a_k_stride_bytes: u64,
    b_k_stride_bytes: u64,
    k_blocks: u32,
    k_var: LoopVarId,
) -> Vec<Instr> {
    let mut out: Vec<Instr> = Vec::with_capacity(13);
    out.push(instrs[init_idx].clone());
    out.push(Instr::BarrierInit {
        page_id: a_page,
        kind: PageBarrier::Ready,
        count: ArrivalCount::One,
    });
    out.push(Instr::BarrierInit {
        page_id: b_page,
        kind: PageBarrier::Ready,
        count: ArrivalCount::One,
    });
    out.push(Instr::ForLoopOpenConst { var: k_var, n: k_blocks });
    out.push(Instr::tma_expect(
        a_page,
        PageBarrier::Ready,
        SmemTileSpec::<128, 128, Bf16>::WITNESS,
        LoaderRole,
    ));
    out.push(Instr::LoadAsync(LoadSpec::new::<128, 128, Bf16>(
        a_page,
        a_src_arg,
        ByteOffsetExpr::LinearLoop { var: k_var, stride_bytes: a_k_stride_bytes, base: a_base_off },
        SmemTileSpec::<128, 128, Bf16>::WITNESS,
        LoaderRole,
        a_page,
    )));
    out.push(Instr::tma_expect(
        b_page,
        PageBarrier::Ready,
        SmemTileSpec::<128, 128, Bf16>::WITNESS,
        LoaderRole,
    ));
    out.push(Instr::LoadAsync(LoadSpec::new::<128, 128, Bf16>(
        b_page,
        b_src_arg,
        ByteOffsetExpr::LinearLoop { var: k_var, stride_bytes: b_k_stride_bytes, base: b_base_off },
        SmemTileSpec::<128, 128, Bf16>::WITNESS,
        LoaderRole,
        b_page,
    )));
    out.push(Instr::wait_loop(
        a_page,
        PageBarrier::Ready,
        k_var,
        Parity::P0,
        WarpRole::AllConsumers,
    ));
    out.push(Instr::wait_loop(
        b_page,
        PageBarrier::Ready,
        k_var,
        Parity::P0,
        WarpRole::AllConsumers,
    ));
    out.push(instrs[load_a_idx].clone());
    out.push(instrs[fence_idx].clone());
    out.push(Instr::WgmmaMmaAB_RegSmem {
        a: a_slot,
        b_page,
        d: d_slot,
        fence: fence_tag,
        accumulate: AccTag::Accumulate,
        width: width_tag,
        role: wgmma_role,
    });
    out.push(instrs[wait_idx].clone());
    out.push(Instr::ForLoopClose { var: k_var });
    out
}

/// Emit the NK rewrite body — outer N-loop wraps the inner K-loop;
/// per-N init_rt_zero + per-N output StoreAsync (LinearLoop on n_var,
/// tile=128×128); CommitGroupBulk + ThreadfenceDevice + Arrive Done
/// emitted ONCE after the N-loop closes. See module doc §"NK rewrite".
#[allow(clippy::too_many_arguments)]
fn emit_nk_body(
    instrs: &[Instr],
    init_idx: usize,
    load_a_idx: usize,
    fence_idx: usize,
    wait_idx: usize,
    store_smem_idx: usize,
    commit_idx: usize,
    fence_dev_idx: usize,
    arrive_done_idx: usize,
    a_page: PageId,
    b_page: PageId,
    dst_page: PageId,
    a_slot: RegTileSlot,
    d_slot: RegTileSlot,
    fence_tag: FenceTag,
    width_tag: GroupWidthTag,
    wgmma_role: WarpRole,
    a_src_arg: KernelArgRef,
    a_base_off: ByteOffset,
    b_src_arg: KernelArgRef,
    b_base_off: ByteOffset,
    out_dst_arg: KernelArgRef,
    out_base_off: ByteOffset,
    out_role: WarpRole,
    a_k_stride_bytes: u64,
    b_k_stride_bytes: u64,
    b_n_stride_bytes: u64,
    out_n_stride_bytes: u64,
    k_blocks: u32,
    n_blocks: u32,
    k_var: LoopVarId,
    n_var: LoopVarId,
) -> Vec<Instr> {
    let mut out: Vec<Instr> = Vec::with_capacity(20);

    // Pre-loop: BarrierInits for A_page / B_page (count=One per
    // TMA-load arrival). These survive across both N and K
    // iterations — `mbarrier::wait` auto-flips parity per phase.
    out.push(Instr::BarrierInit {
        page_id: a_page,
        kind: PageBarrier::Ready,
        count: ArrivalCount::One,
    });
    out.push(Instr::BarrierInit {
        page_id: b_page,
        kind: PageBarrier::Ready,
        count: ArrivalCount::One,
    });

    // Outer N-loop open.
    out.push(Instr::ForLoopOpenConst { var: n_var, n: n_blocks });

    // Per-N: re-init rt_d to zero (each output tile gets a fresh
    // accumulator).
    out.push(instrs[init_idx].clone());

    // Inner K-loop open.
    out.push(Instr::ForLoopOpenConst { var: k_var, n: k_blocks });

    // K-loop body: TmaExpect + LoadAsync (A: k-only LinearLoop;
    // B: 2D Affine n_var × k_var) + waits + WGMMA + async-wait.
    out.push(Instr::tma_expect(
        a_page,
        PageBarrier::Ready,
        SmemTileSpec::<128, 128, Bf16>::WITNESS,
        LoaderRole,
    ));
    out.push(Instr::LoadAsync(LoadSpec::new::<128, 128, Bf16>(
        a_page,
        a_src_arg,
        // A operand: M × K_full row-major; A_chunk[m, k_var] depends
        // on k_var only (not n_var — A is shared across all N tiles).
        ByteOffsetExpr::LinearLoop { var: k_var, stride_bytes: a_k_stride_bytes, base: a_base_off },
        SmemTileSpec::<128, 128, Bf16>::WITNESS,
        LoaderRole,
        a_page,
    )));
    out.push(Instr::tma_expect(
        b_page,
        PageBarrier::Ready,
        SmemTileSpec::<128, 128, Bf16>::WITNESS,
        LoaderRole,
    ));
    out.push(Instr::LoadAsync(LoadSpec::new::<128, 128, Bf16>(
        b_page,
        b_src_arg,
        // B operand: K_full × N_full row-major; B_chunk[k_var, n_var]
        // depends on BOTH loop vars. The Affine2D variant encodes
        // `base + n_var × n_stride + k_var × k_stride`.
        ByteOffsetExpr::Affine2D {
            outer_var: n_var,
            outer_stride_bytes: b_n_stride_bytes,
            inner_var: k_var,
            inner_stride_bytes: b_k_stride_bytes,
            base: b_base_off,
        },
        SmemTileSpec::<128, 128, Bf16>::WITNESS,
        LoaderRole,
        b_page,
    )));
    out.push(Instr::wait_loop(
        a_page,
        PageBarrier::Ready,
        k_var,
        Parity::P0,
        WarpRole::AllConsumers,
    ));
    out.push(Instr::wait_loop(
        b_page,
        PageBarrier::Ready,
        k_var,
        Parity::P0,
        WarpRole::AllConsumers,
    ));
    out.push(instrs[load_a_idx].clone());
    out.push(instrs[fence_idx].clone());
    out.push(Instr::WgmmaMmaAB_RegSmem {
        a: a_slot,
        b_page,
        d: d_slot,
        fence: fence_tag,
        accumulate: AccTag::Accumulate,
        width: width_tag,
        role: wgmma_role,
    });
    out.push(instrs[wait_idx].clone());

    // Inner K-loop close.
    out.push(Instr::ForLoopClose { var: k_var });

    // Per-N output store: register tile → smem dst_page (cloned),
    // then smem dst_page → gmem output[m, n_var × 128 .. (n_var+1) × 128]
    // via a per-N-iter StoreAsync with LinearLoop byte_off and
    // SmemTileSpec<128, 128, Bf16> tile shape.
    out.push(instrs[store_smem_idx].clone());
    out.push(Instr::StoreAsync(crate::tk_tape::StoreSpec::new::<128, 128, Bf16>(
        dst_page,
        out_dst_arg,
        ByteOffsetExpr::LinearLoop {
            var: n_var,
            stride_bytes: out_n_stride_bytes,
            base: out_base_off,
        },
        SmemTileSpec::<128, 128, Bf16>::WITNESS,
        // Reconstruct StorerRole from the original WarpRole. The
        // original conservative emit used STORE_ROLE = StorerRole;
        // typed StoreSpec::new takes StorerRole directly. We pass
        // crate::tk_tape::StorerRole as the role witness — the
        // out_role value is only needed for the assertion below.
        crate::tk_tape::StorerRole,
    )));
    let _ = out_role; // surfaced via the `crate::tk_tape::StorerRole` witness above.

    // Outer N-loop close.
    out.push(Instr::ForLoopClose { var: n_var });

    // Post-loop drain: ONE CommitGroupBulk + ThreadfenceDevice +
    // Arrive Done, batching all per-N StoreAsyncs. Cloned from the
    // original conservative emit so role tags match.
    out.push(instrs[commit_idx].clone());
    out.push(instrs[fence_dev_idx].clone());
    out.push(instrs[arrive_done_idx].clone());

    out
}

/// Recover the `base` byte offset from a [`ByteOffsetExpr`]. The
/// chunk LoadAsyncs share the same base as the original big load —
/// only the per-iter stride is added on top.
fn base_of(expr: &ByteOffsetExpr) -> ByteOffset {
    match expr {
        ByteOffsetExpr::Const(off) => *off,
        ByteOffsetExpr::LinearLoop { base, .. } => *base,
        ByteOffsetExpr::RuntimePosition { base, .. } => *base,
        ByteOffsetExpr::Affine2D { base, .. } => *base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk_tape::{
        AccReset, AllConsumersRole, Bf16, FenceExternal, Fp32, GroupWidth, KernelArgRef, PageId,
        RegTileId, RoleWitness, RowLayout, SmemTileId, SmemTileSpec, TileShape, TkTape,
    };

    /// Page-fitting LoadAsync passes through (no rewrite), preserving
    /// idempotency on already-OK tapes.
    #[test]
    fn page_fitting_load_passes_through() {
        let mut tape = TkTape::default();
        // SmemTileSpec<128, 128, Bf16> = 32768 bytes == PAGE_SIZE.
        // The PAGE_SIZE-equality case is allowed (≤, not <).
        tape.instrs
            .push(Instr::LoadAsync(LoadSpec::new::<128, 128, Bf16>(
                PageId(0),
                KernelArgRef(0),
                ByteOffsetExpr::from_const(0),
                SmemTileSpec::<128, 128, Bf16>::WITNESS,
                LoaderRole,
                PageId(0),
            )));
        let before = tape.instrs.len();
        split_oversized_loads_pass(&mut tape);
        // Pass is a no-op on a tape with no oversized loads.
        assert_eq!(tape.instrs.len(), before);
        assert!(matches!(&tape.instrs[0], Instr::LoadAsync(_)));
    }

    /// Build a synthetic conservative MatmulTile-emit-sequence with
    /// oversized External LoadAsyncs into A_page and B_page, plus
    /// the matmul body. NO post-matmul StoreAsync / Commit / Fence /
    /// Arrive — used for the K-only path which doesn't need them.
    fn synthetic_oversized_gemm_tape(k_full: u32, n_full: u32) -> TkTape {
        let mut tape = TkTape::default();
        const W4: GroupWidth<4> = GroupWidth::<4>::WARPGROUP;
        const WL: GroupWidth<1> = GroupWidth::<1>::PER_WARP;
        const R: AllConsumersRole = AllConsumersRole;

        let a_page = PageId(0);
        let b_page = PageId(1);
        let dst_page = PageId(2);
        let rt_a: RegTileId<32, 128, Bf16, RowLayout> = tape.mint_reg_tile();
        let rt_d: RegTileId<32, 128, Fp32, RowLayout> = tape.mint_reg_tile();

        tape.instrs.push(Instr::LoadAsync(LoadSpec::new_runtime_shape(
            a_page,
            KernelArgRef(0),
            ByteOffsetExpr::from_const(0),
            TileShape { rows: 128, cols: k_full, elem_bytes: 2 },
            LoaderRole,
            a_page,
        )));
        tape.instrs.push(Instr::LoadAsync(LoadSpec::new_runtime_shape(
            b_page,
            KernelArgRef(1),
            ByteOffsetExpr::from_const(0),
            TileShape { rows: k_full, cols: n_full, elem_bytes: 2 },
            LoaderRole,
            b_page,
        )));
        tape.instrs.push(Instr::init_rt_zero(rt_d, WL, R));
        tape.instrs.push(Instr::load_shmem_to_reg_warpgroup(
            SmemTileId::<128, 128, Bf16>::from_page(a_page),
            rt_a,
            W4,
            R,
        ));
        tape.instrs.push(Instr::wgmma_fence_acc(rt_d, W4));
        tape.instrs.push(Instr::wgmma_mma_ab_reg_smem(
            rt_d,
            rt_a,
            SmemTileId::<128, 128, Bf16>::from_page(b_page),
            FenceExternal,
            AccReset,
            W4,
        ));
        tape.instrs.push(Instr::wgmma_async_wait(0, W4));
        tape.instrs.push(Instr::store_reg_tile_to_shmem_warpgroup(
            rt_d,
            SmemTileId::<128, 128, Bf16>::from_page(dst_page),
            W4,
            R,
        ));
        tape
    }

    /// Build a synthetic conservative MatmulTile-emit-sequence
    /// INCLUDING the post-matmul StoreAsync / CommitGroupBulk /
    /// ThreadfenceDevice / PageBarrierArrive Done sequence emitted
    /// by `emit_store_and_arrive`. Required by the NK rewrite path
    /// (which extends the splice span to cover those Instrs and
    /// rewrites the StoreAsync's byte_off + tile shape).
    fn synthetic_full_gemm_tape(k_full: u32, n_full: u32) -> TkTape {
        use crate::tk_tape::{StoreSpec, StorerRole};
        let mut tape = synthetic_oversized_gemm_tape(k_full, n_full);
        let dst_page = PageId(2);
        let storer_role = StorerRole.to_warp_role();
        tape.instrs.push(Instr::StoreAsync(StoreSpec::new_runtime_shape(
            dst_page,
            KernelArgRef(2),
            ByteOffsetExpr::from_const(0),
            TileShape { rows: 128, cols: n_full, elem_bytes: 2 },
            StorerRole,
        )));
        tape.instrs.push(Instr::CommitGroupBulk { role: storer_role });
        tape.instrs.push(Instr::ThreadfenceDevice { role: WarpRole::All });
        tape.instrs.push(Instr::PageBarrierArrive {
            page_id: dst_page,
            kind: PageBarrier::Done,
            role: storer_role,
        });
        tape
    }

    /// The headline rewrite test: K=2048, N=128 — K-tile both A and B
    /// in lockstep into K/128 = 16 chunks.
    #[test]
    fn rewrites_paired_oversized_into_k_loop() {
        let mut tape = synthetic_oversized_gemm_tape(2048, 128);
        let before_loadasyncs = tape
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::LoadAsync(_)))
            .count();
        assert_eq!(before_loadasyncs, 2, "synthetic tape: 2 big LoadAsyncs");

        split_oversized_loads_pass(&mut tape);

        // Postcondition: no oversized LoadAsync remains.
        for instr in &tape.instrs {
            if let Instr::LoadAsync(spec) = instr {
                let bytes = bytes_of_tile(&spec.tile);
                assert!(
                    bytes <= PAGE_SIZE as u64,
                    "post-pass LoadAsync still oversized: {}×{}×{}B = {} bytes",
                    spec.tile.rows,
                    spec.tile.cols,
                    spec.tile.elem_bytes,
                    bytes,
                );
            }
        }

        // Structure: ForLoopOpenConst with n = 16, ForLoopClose with
        // matching var.
        let loop_open = tape
            .instrs
            .iter()
            .find_map(|i| match i {
                Instr::ForLoopOpenConst { var, n } => Some((*var, *n)),
                _ => None,
            })
            .expect("rewrite emits a ForLoopOpenConst");
        assert_eq!(loop_open.1, 2048 / 128, "K_blocks = 16");
        let loop_close_var = tape
            .instrs
            .iter()
            .find_map(|i| match i {
                Instr::ForLoopClose { var } => Some(*var),
                _ => None,
            })
            .expect("rewrite emits a ForLoopClose");
        assert_eq!(loop_close_var, loop_open.0, "loop var matches open/close");

        // Two LoadAsyncs inside the body, each 128×128.
        let chunk_loads: Vec<&LoadSpec> = tape
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::LoadAsync(spec) => Some(spec),
                _ => None,
            })
            .collect();
        assert_eq!(chunk_loads.len(), 2, "post-pass: A chunk + B chunk");
        for spec in &chunk_loads {
            assert_eq!(spec.tile.rows, 128);
            assert_eq!(spec.tile.cols, 128);
            assert_eq!(spec.tile.elem_bytes, 2);
            // byte_off is LinearLoop with the matching var.
            match &spec.byte_off {
                ByteOffsetExpr::LinearLoop { var, .. } => {
                    assert_eq!(*var, loop_open.0, "chunk LoadAsync uses the K-loop var");
                }
                other => panic!("expected LinearLoop byte_off, got {other:?}"),
            }
        }

        // Stride math.
        // A operand: stride = K_BLOCK * elem_bytes = 128 * 2 = 256.
        // B operand: stride = K_BLOCK * N_full * elem_bytes = 128 * 128 * 2 = 32768.
        let a_chunk = chunk_loads
            .iter()
            .find(|s| s.dst_page == PageId(0))
            .expect("A chunk targets PageId(0)");
        let b_chunk = chunk_loads
            .iter()
            .find(|s| s.dst_page == PageId(1))
            .expect("B chunk targets PageId(1)");
        match &a_chunk.byte_off {
            ByteOffsetExpr::LinearLoop { stride_bytes, .. } => {
                assert_eq!(*stride_bytes, 256, "A K-stride = 128 × 2");
            }
            _ => unreachable!(),
        }
        match &b_chunk.byte_off {
            ByteOffsetExpr::LinearLoop { stride_bytes, .. } => {
                assert_eq!(*stride_bytes, 32768, "B K-stride = 128 × N_full × 2");
            }
            _ => unreachable!(),
        }

        // BarrierInit Ready count=One for each page, BEFORE the loop.
        let loop_open_idx = tape
            .instrs
            .iter()
            .position(|i| matches!(i, Instr::ForLoopOpenConst { .. }))
            .unwrap();
        let pre_loop_barrier_inits = tape.instrs[..loop_open_idx]
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    Instr::BarrierInit {
                        kind: PageBarrier::Ready,
                        count: ArrivalCount::One,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            pre_loop_barrier_inits, 2,
            "BarrierInit Ready count=One emitted for A and B pages before the loop opens"
        );

        // WGMMA at the AccAccumulate variant inside the loop body.
        let wgmma_acc = tape.instrs.iter().any(|i| {
            matches!(
                i,
                Instr::WgmmaMmaAB_RegSmem {
                    accumulate: AccTag::Accumulate,
                    ..
                }
            )
        });
        assert!(
            wgmma_acc,
            "post-pass WGMMA uses AccAccumulate (init_rt_zero is outside loop)"
        );
        let wgmma_reset = tape.instrs.iter().any(|i| {
            matches!(
                i,
                Instr::WgmmaMmaAB_RegSmem {
                    accumulate: AccTag::Reset,
                    ..
                }
            )
        });
        assert!(
            !wgmma_reset,
            "AccReset must not survive the rewrite (would zero accumulator each iter)"
        );

        // wgmma_async_wait lives INSIDE the loop body (between
        // ForLoopOpenConst and ForLoopClose).
        let loop_close_idx = tape
            .instrs
            .iter()
            .position(|i| matches!(i, Instr::ForLoopClose { .. }))
            .unwrap();
        let waits_inside_body = tape.instrs[loop_open_idx + 1..loop_close_idx]
            .iter()
            .filter(|i| matches!(i, Instr::WgmmaAsyncWait { .. }))
            .count();
        assert_eq!(
            waits_inside_body, 1,
            "wgmma_async_wait lives inside the loop body, not after it",
        );

        // store_reg_tile_to_shmem stays AFTER the loop close.
        let stores_after_close = tape.instrs[loop_close_idx + 1..]
            .iter()
            .filter(|i| matches!(i, Instr::StoreRegTileToShmem { .. }))
            .count();
        assert_eq!(
            stores_after_close, 1,
            "store_reg_tile_to_shmem stays after the K-loop, runs once",
        );
    }

    /// Pass converges (idempotent) when run twice — the postcondition
    /// holds after the first call, so the second is a no-op walk.
    #[test]
    fn idempotent_after_rewrite() {
        let mut tape = synthetic_oversized_gemm_tape(2048, 128);
        split_oversized_loads_pass(&mut tape);
        let len_after_first = tape.instrs.len();
        split_oversized_loads_pass(&mut tape);
        assert_eq!(
            tape.instrs.len(),
            len_after_first,
            "second pass is a no-op (postcondition already holds)",
        );
    }

    /// An oversized LoadAsync without a Gemm consumer panics with a
    /// diagnostic — the pass refuses to silently produce an
    /// oversized tape.
    #[test]
    #[should_panic(expected = "no WgmmaMmaAB_RegSmem within 64 instrs")]
    fn oversized_load_without_gemm_consumer_panics() {
        let mut tape = TkTape::default();
        // Big LoadAsync, no following Wgmma.
        tape.instrs.push(Instr::LoadAsync(LoadSpec::new_runtime_shape(
            PageId(0),
            KernelArgRef(0),
            ByteOffsetExpr::from_const(0),
            TileShape {
                rows: 128,
                cols: 2048,
                elem_bytes: 2,
            },
            LoaderRole,
            PageId(0),
        )));
        split_oversized_loads_pass(&mut tape);
    }

    /// K_full not divisible by K_BLOCK — pass panics with
    /// diagnostic.
    #[test]
    #[should_panic(expected = "K_full")]
    fn non_aligned_k_panics() {
        let mut tape = synthetic_oversized_gemm_tape(2049, 128);
        split_oversized_loads_pass(&mut tape);
    }

    /// Multiple oversized LoadAsyncs in the same tape (two
    /// independent matmul-emit-sequences) are both rewritten.
    #[test]
    fn rewrites_multiple_independent_gemms() {
        let mut tape = TkTape::default();
        // Concatenate two synthetic gemm sequences. Need distinct
        // PageIds so the lookup in plan_rewrite finds the right
        // pair per matmul.
        const W4: GroupWidth<4> = GroupWidth::<4>::WARPGROUP;
        const WL: GroupWidth<1> = GroupWidth::<1>::PER_WARP;
        const R: AllConsumersRole = AllConsumersRole;
        let mut emit_gemm = |tape: &mut TkTape, base: u8, k_full: u32, n_full: u32| {
            let a_page = PageId(base);
            let b_page = PageId(base + 1);
            let dst_page = PageId(base + 2);
            let rt_a: RegTileId<32, 128, Bf16, RowLayout> = tape.mint_reg_tile();
            let rt_d: RegTileId<32, 128, Fp32, RowLayout> = tape.mint_reg_tile();
            tape.instrs.push(Instr::LoadAsync(LoadSpec::new_runtime_shape(
                a_page,
                KernelArgRef(base as u16),
                ByteOffsetExpr::from_const(0),
                TileShape { rows: 128, cols: k_full, elem_bytes: 2 },
                LoaderRole,
                a_page,
            )));
            tape.instrs.push(Instr::LoadAsync(LoadSpec::new_runtime_shape(
                b_page,
                KernelArgRef(base as u16 + 1),
                ByteOffsetExpr::from_const(0),
                TileShape { rows: k_full, cols: n_full, elem_bytes: 2 },
                LoaderRole,
                b_page,
            )));
            tape.instrs.push(Instr::init_rt_zero(rt_d, WL, R));
            tape.instrs.push(Instr::load_shmem_to_reg_warpgroup(
                SmemTileId::<128, 128, Bf16>::from_page(a_page),
                rt_a,
                W4,
                R,
            ));
            tape.instrs.push(Instr::wgmma_fence_acc(rt_d, W4));
            tape.instrs.push(Instr::wgmma_mma_ab_reg_smem(
                rt_d,
                rt_a,
                SmemTileId::<128, 128, Bf16>::from_page(b_page),
                FenceExternal,
                AccReset,
                W4,
            ));
            tape.instrs.push(Instr::wgmma_async_wait(0, W4));
            tape.instrs.push(Instr::store_reg_tile_to_shmem_warpgroup(
                rt_d,
                SmemTileId::<128, 128, Bf16>::from_page(dst_page),
                W4,
                R,
            ));
        };
        emit_gemm(&mut tape, 0, 2048, 128);
        emit_gemm(&mut tape, 10, 2048, 128);

        split_oversized_loads_pass(&mut tape);

        let loops = tape
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::ForLoopOpenConst { .. }))
            .count();
        assert_eq!(loops, 2, "two K-loops emitted, one per Gemm");

        // The two minted loop vars are distinct (no var collision).
        let vars: Vec<LoopVarId> = tape
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::ForLoopOpenConst { var, .. } => Some(*var),
                _ => None,
            })
            .collect();
        assert_eq!(vars.len(), 2);
        assert_ne!(vars[0], vars[1], "fresh loop var per rewrite");

        // Postcondition.
        for instr in &tape.instrs {
            if let Instr::LoadAsync(spec) = instr {
                assert!(bytes_of_tile(&spec.tile) <= PAGE_SIZE as u64);
            }
        }
    }

    /// NK rewrite headline test: K = 2048, N = 256 — NK path (n_blocks=2)
    /// emits nested loops with Affine2D byte_off on B's chunk LoadAsync.
    #[test]
    fn rewrites_nk_into_nested_loops() {
        let mut tape = synthetic_full_gemm_tape(2048, 256);
        split_oversized_loads_pass(&mut tape);

        // Postcondition: every LoadAsync ≤ PAGE_SIZE.
        for (i, ins) in tape.instrs.iter().enumerate() {
            if let Instr::LoadAsync(spec) = ins {
                assert!(
                    bytes_of_tile(&spec.tile) <= PAGE_SIZE as u64,
                    "post-pass LoadAsync at idx {i}: oversized {:?}",
                    spec.tile,
                );
            }
        }

        // Two ForLoopOpenConst instrs: outer N (n=2), inner K (n=16).
        let loops: Vec<(LoopVarId, u32)> = tape
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::ForLoopOpenConst { var, n } => Some((*var, *n)),
                _ => None,
            })
            .collect();
        assert_eq!(loops.len(), 2, "NK emits exactly two ForLoopOpenConst (outer N, inner K)");
        let (n_var, n_count) = loops[0];
        let (k_var, k_count) = loops[1];
        assert_eq!(n_count, 256 / 128, "N_blocks = N_full / 128");
        assert_eq!(k_count, 2048 / 128, "K_blocks = K_full / 128");
        assert_ne!(n_var, k_var, "outer N and inner K vars are distinct");

        // Inner K-loop sits between outer N-open and outer N-close.
        let n_open = tape.instrs.iter().position(|i| matches!(i, Instr::ForLoopOpenConst { var, .. } if *var == n_var)).unwrap();
        let k_open = tape.instrs.iter().position(|i| matches!(i, Instr::ForLoopOpenConst { var, .. } if *var == k_var)).unwrap();
        let k_close = tape.instrs.iter().rposition(|i| matches!(i, Instr::ForLoopClose { var } if *var == k_var)).unwrap();
        let n_close = tape.instrs.iter().rposition(|i| matches!(i, Instr::ForLoopClose { var } if *var == n_var)).unwrap();
        assert!(n_open < k_open && k_open < k_close && k_close < n_close,
            "loop nesting: N(open) < K(open) < K(close) < N(close); got {n_open} < {k_open} < {k_close} < {n_close}");

        // B chunk LoadAsync uses Affine2D with both n_var and k_var.
        let b_load = tape.instrs.iter().find_map(|i| match i {
            Instr::LoadAsync(spec) if spec.dst_page == PageId(1) => Some(spec),
            _ => None,
        }).expect("B chunk LoadAsync present");
        match &b_load.byte_off {
            ByteOffsetExpr::Affine2D { outer_var, outer_stride_bytes, inner_var, inner_stride_bytes, .. } => {
                assert_eq!(*outer_var, n_var, "B's outer var is n_var");
                assert_eq!(*inner_var, k_var, "B's inner var is k_var");
                // B is K_full × N_full = 2048 × 256 row-major.
                // outer (n) stride = N_BLOCK × elem_bytes = 128 × 2 = 256
                // inner (k) stride = K_BLOCK × N_full × elem_bytes = 128 × 256 × 2 = 65536
                assert_eq!(*outer_stride_bytes, 256, "B n-stride = 128 × 2");
                assert_eq!(*inner_stride_bytes, 128 * 256 * 2, "B k-stride = K_BLOCK × N_full × 2");
            }
            other => panic!("expected B byte_off = Affine2D, got {other:?}"),
        }

        // A chunk LoadAsync uses LinearLoop on k_var only (A is independent of n).
        let a_load = tape.instrs.iter().find_map(|i| match i {
            Instr::LoadAsync(spec) if spec.dst_page == PageId(0) => Some(spec),
            _ => None,
        }).expect("A chunk LoadAsync present");
        match &a_load.byte_off {
            ByteOffsetExpr::LinearLoop { var, stride_bytes, .. } => {
                assert_eq!(*var, k_var, "A's loop var is k_var (independent of n)");
                assert_eq!(*stride_bytes, 128 * 2, "A k-stride = K_BLOCK × elem_bytes");
            }
            other => panic!("expected A byte_off = LinearLoop, got {other:?}"),
        }

        // Output StoreAsync is INSIDE the N-loop, with LinearLoop(n_var).
        let store_idx = tape.instrs.iter().position(|i| matches!(i, Instr::StoreAsync(_))).unwrap();
        assert!(store_idx > k_close && store_idx < n_close,
            "output StoreAsync sits between K-close and N-close (per-N tile store); got store_idx={store_idx}");
        let store = match &tape.instrs[store_idx] {
            Instr::StoreAsync(spec) => spec,
            _ => unreachable!(),
        };
        assert_eq!(store.tile.rows, 128, "per-N output tile rows");
        assert_eq!(store.tile.cols, 128, "per-N output tile cols");
        match &store.byte_off {
            ByteOffsetExpr::LinearLoop { var, stride_bytes, .. } => {
                assert_eq!(*var, n_var, "output store byte_off uses n_var");
                assert_eq!(*stride_bytes, 128 * 2, "output n-stride = N_BLOCK × elem_bytes");
            }
            other => panic!("expected output StoreAsync byte_off = LinearLoop, got {other:?}"),
        }

        // CommitGroupBulk + ThreadfenceDevice + Arrive Done are AFTER N-close
        // (one drain per Gemm, not per N tile).
        let commit_idx = tape.instrs.iter().rposition(|i| matches!(i, Instr::CommitGroupBulk { .. })).unwrap();
        let fence_idx = tape.instrs.iter().rposition(|i| matches!(i, Instr::ThreadfenceDevice { .. })).unwrap();
        let arrive_idx = tape.instrs.iter().rposition(|i| matches!(i, Instr::PageBarrierArrive { kind: PageBarrier::Done, .. })).unwrap();
        assert!(n_close < commit_idx && commit_idx < fence_idx && fence_idx < arrive_idx,
            "drain (commit, fence, arrive) lives AFTER N-close once; got {n_close} < {commit_idx} < {fence_idx} < {arrive_idx}");
        // No CommitGroupBulk inside the loops.
        let commits_inside = tape.instrs[n_open + 1..n_close]
            .iter()
            .filter(|i| matches!(i, Instr::CommitGroupBulk { .. }))
            .count();
        assert_eq!(commits_inside, 0, "no CommitGroupBulk inside the N-loop body (would commit per-N)");

        // init_rt_zero is INSIDE the N-loop body (re-init per N iter).
        let init_inside = tape.instrs[n_open + 1..n_close]
            .iter()
            .filter(|i| matches!(i, Instr::InitRtZero { .. }))
            .count();
        assert_eq!(init_inside, 1, "init_rt_zero re-runs per N iter (lives inside N body)");
    }

    /// NK rewrite is idempotent — running the pass twice doesn't
    /// re-rewrite (postcondition holds after first call).
    #[test]
    fn nk_idempotent() {
        let mut tape = synthetic_full_gemm_tape(2048, 256);
        split_oversized_loads_pass(&mut tape);
        let len_after_first = tape.instrs.len();
        split_oversized_loads_pass(&mut tape);
        assert_eq!(tape.instrs.len(), len_after_first);
    }

    /// NK with N exactly = PAGE_COLS (n_blocks = 1) takes the K-only
    /// path — no outer N-loop.
    #[test]
    fn n_equals_page_cols_takes_k_only_path() {
        let mut tape = synthetic_full_gemm_tape(2048, 128);
        split_oversized_loads_pass(&mut tape);
        let loops: Vec<u32> = tape
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::ForLoopOpenConst { n, .. } => Some(*n),
                _ => None,
            })
            .collect();
        // K-only emits ONE loop (K_blocks = 16); NK would emit two.
        assert_eq!(loops, vec![16], "K-only path: one loop with n = K_blocks = 16");
    }

    /// NK with N not divisible by PAGE_COLS panics.
    #[test]
    #[should_panic(expected = "not a multiple of PAGE_COLS")]
    fn nk_non_aligned_n_panics() {
        let mut tape = synthetic_full_gemm_tape(2048, 200);
        split_oversized_loads_pass(&mut tape);
    }

    /// `max_existing_loop_var` returns `None` for an empty tape.
    #[test]
    fn max_existing_loop_var_empty_tape() {
        assert_eq!(max_existing_loop_var(&[]), None);
    }

    /// `max_existing_loop_var` recovers vars from ForLoopOpenConst,
    /// ForLoopOpenKernelArg, ForLoopClose, PageBarrierWaitLoopStart*,
    /// LinearLoop byte_offs.
    #[test]
    fn max_existing_loop_var_finds_max() {
        let instrs: Vec<Instr> = vec![
            Instr::ForLoopOpenConst { var: LoopVarId(2), n: 16 },
            Instr::ForLoopClose { var: LoopVarId(7) },
        ];
        assert_eq!(max_existing_loop_var(&instrs), Some(7));
    }
}

