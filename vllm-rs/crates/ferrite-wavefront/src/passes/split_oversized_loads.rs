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
//! loads need to be the OUTPUT of the lowering's pipeline, but the
//! conservative lowering emits the full region naively (per plan §6,
//! `lower_tape_to_tk` is "trivial syntax-directed translation"; no
//! analysis at lowering).
//!
//! Per plan §6.5: TkTape→TkTape passes own target-specific resource
//! decisions (smem capacity, page-lifetime windows). K-tiling a Gemm
//! into PAGE_SIZE-fitting WGMMA chunks IS a target-specific resource
//! decision — Hopper's WGMMA-K=128 + page-pool tile shape determine
//! the K-block count. This pass takes the conservative output and
//! rewrites Gemm-class load+WGMMA sequences into K-block loops.
//!
//! Pass shape mirrors [`crate::passes::rt_alias_pass`] /
//! [`crate::passes::page_coalesce_pass`]: walks the tape, identifies
//! oversized loads + their MatmulTile-class consumers, rewrites in
//! place. Postcondition: every `Instr::LoadAsync` in the rewritten
//! tape has `spec.tile.byte_size() <= PAGE_SIZE`.
//!
//! ## Status: SCAFFOLDING + DETECTION ONLY
//!
//! This commit lands the pass module shell + the detection walk that
//! panics with a diagnostic when a big LoadAsync is found, so the
//! gap is visible at codegen time (proc-macro expansion). The actual
//! rewrite — pattern-matching the (LoadAsync_A, LoadAsync_B,
//! MatmulTile-emit-sequence) triple and splicing in the K-loop
//! bracket — is the next commit.
//!
//! The detection panic surfaces the same information as
//! `LoadSpec::new_runtime_shape`'s in-construction assert (which
//! today fires inside the lowerer). Moving the gate to a tape-pass
//! lets future work do the actual rewrite without changing the
//! lowering — exactly the "TkTape→TkTape transformation" architecture.
//!
//! ## TODO (phase 2 — actual rewrite)
//!
//! For each oversized LoadAsync:
//!
//! 1. **Identify the consumer pattern.** Walk forward to find the
//!    MatmulTile-emit-sequence (`init_rt_zero` → `load_shmem_to_reg
//!    _warpgroup(a, rt_a)` → `wgmma_fence_acc` →
//!    `wgmma_mma_ab_reg_smem(d, rt_a, b, FenceExternal, AccReset)` →
//!    `wgmma_async_wait(0)` → `store_reg_tile_to_shmem_warpgroup`).
//!    The `b_page` field on the WGMMA identifies the B operand's
//!    page; the `load_shmem_to_reg_warpgroup`'s src page identifies
//!    the A operand's page.
//!
//! 2. **Walk back from the matmul to the LoadAsyncs** for each
//!    operand page.
//!
//! 3. **Compute K-block count.** For A (`M × K_full`), cols / 128.
//!    For B (`K_full × N`), rows / 128. Both must agree.
//!
//! 4. **Compute per-K-block strides** (in bytes) for both A and B,
//!    based on the original full-region's row-major layout.
//!
//! 5. **Replace the (loads + matmul-sequence) span** with the
//!    K-loop emit:
//!
//!    ```text
//!    init_rt_zero(rt_d)
//!    init_semaphore page_ready[A_page] (count=One)
//!    init_semaphore page_ready[B_page] (count=One)
//!    open_loop(K_blocks)
//!        if iter > 0: re-init the barriers (or use parity ring)
//!        expect_bytes(A_chunk)
//!        load_async A_chunk [byte_off = base_A + k * stride_A]
//!        expect_bytes(B_chunk)
//!        load_async B_chunk [byte_off = base_B + k * stride_B]
//!        wait page_ready[A_page]
//!        wait page_ready[B_page]
//!        load_shmem_to_reg_warpgroup(A_page → rt_a)
//!        wgmma_fence_acc(rt_d)
//!        wgmma_mma_ab_reg_smem(rt_d, rt_a, B_page, AccAccumulate)
//!    close_loop
//!    wgmma_async_wait(0)
//!    store_reg_tile_to_shmem_warpgroup(rt_d → dst)
//!    ```
//!
//!    `AccAccumulate` always (init_rt_zero outside the loop makes
//!    iter 0 equivalent to `AccReset + accumulate-into-zero`).
//!
//! 6. **Postcondition validator**: `validate_tk_tape` extension
//!    asserts every `Instr::LoadAsync.tile.byte_size() <= PAGE_SIZE`
//!    after the pass.
//!
//! Phase 2 needs ~400 LOC + tests; punted to a follow-up commit so
//! this pass-as-detector commit is small + reviewable.

use crate::tk_tape::{Instr, PAGE_SIZE, TkTape};

/// Phase-1 implementation: walk the tape, panic with diagnostic
/// when a `LoadAsync` exceeds `PAGE_SIZE`. The actual rewrite lands
/// in a follow-up commit; for now the panic exposes the gap at
/// codegen time so downstream callers see "this Gemm needs K-tiling"
/// rather than silent OOB at runtime.
///
/// **Postcondition (phase 2)**: every `Instr::LoadAsync.tile` will
/// have `byte_size() <= PAGE_SIZE`. Phase 1 detects only.
pub fn split_oversized_loads_pass(tape: &mut TkTape) {
    for (idx, instr) in tape.instrs.iter().enumerate() {
        let Instr::LoadAsync(spec) = instr else { continue };
        let bytes = (spec.tile.rows as u64)
            * (spec.tile.cols as u64)
            * (spec.tile.elem_bytes as u64);
        if bytes <= PAGE_SIZE as u64 {
            continue;
        }
        // Codegen-time panic — exposes the K-tiling gap so a future
        // phase-2 rewrite has concrete targets, and so the user
        // sees "this Gemm input is too big" instead of silent
        // page_buf[i] OOB at runtime.
        panic!(
            "split_oversized_loads_pass [phase 1 detect-only]: \
             Instr::LoadAsync at instr-index {idx} has \
             tile {rows}×{cols}×{elem_bytes}B = {bytes} bytes, \
             exceeds PAGE_SIZE = {page_size} bytes. \
             dst_page={dst_page}, barrier_page={barrier_page}, \
             src_arg={src_arg}. \
             This is a Gemm-class External load needing K-tile rewrite \
             (one big LoadAsync → K-loop of PAGE_SIZE-fitting LoadAsyncs \
             paired with WGMMA AccAccumulate). Phase 2 of this pass \
             will perform the rewrite; see split_oversized_loads.rs \
             module docs §TODO. For now, the lowering must be tightened \
             upstream to produce per-tile External regions, OR this \
             pass's rewrite must land.",
            rows = spec.tile.rows,
            cols = spec.tile.cols,
            elem_bytes = spec.tile.elem_bytes,
            page_size = PAGE_SIZE,
            dst_page = spec.dst_page.0,
            barrier_page = spec.barrier_page.0,
            src_arg = spec.src_arg.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tk_tape::{
        Bf16, KernelArgRef, LoadSpec, LoaderRole, PageId, SmemTileSpec, TkTape,
    };

    #[test]
    fn page_fitting_load_passes_through() {
        let mut tape = TkTape::default();
        // SmemTileSpec<128, 128, Bf16> → 32768 bytes == PAGE_SIZE.
        // The PAGE_SIZE-equality case is allowed (≤, not <).
        tape.instrs.push(Instr::LoadAsync(LoadSpec::new::<128, 128, Bf16>(
            PageId(0),
            KernelArgRef(0),
            crate::tk_tape::ByteOffsetExpr::from_const(0),
            SmemTileSpec::<128, 128, Bf16>::WITNESS,
            LoaderRole,
            PageId(0),
        )));
        // Should NOT panic.
        split_oversized_loads_pass(&mut tape);
    }

    #[test]
    #[should_panic(expected = "split_oversized_loads_pass [phase 1 detect-only]")]
    fn oversized_load_panics_with_diagnostic() {
        // Build a LoadSpec with a runtime tile that exceeds PAGE_SIZE.
        // Use `new_runtime_shape` directly... wait, that asserts on
        // construction. So we need to construct via the typed path
        // with a shape > 128×128. SmemTileSpec<256, 128, Bf16> is
        // 256*128*2 = 65536 bytes > 32768 = PAGE_SIZE.
        let mut tape = TkTape::default();
        // The compile-time witness path doesn't have a PAGE_SIZE
        // assert — only `new_runtime_shape` does. Use the typed
        // path here so the test focuses on this pass's detect
        // behavior.
        tape.instrs.push(Instr::LoadAsync(LoadSpec::new::<256, 128, Bf16>(
            PageId(0),
            KernelArgRef(0),
            crate::tk_tape::ByteOffsetExpr::from_const(0),
            SmemTileSpec::<256, 128, Bf16>::WITNESS,
            LoaderRole,
            PageId(0),
        )));
        split_oversized_loads_pass(&mut tape);
    }
}
