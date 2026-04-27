// SPDX-License-Identifier: Apache-2.0
// Device-callable wrapper for FlashInfer's persistent batched-paged
// attention.
//
// Mirrors the body of `flashinfer::PersistentKernelTemplate`
// (`flashinfer/include/flashinfer/attention/persistent_template.cuh`,
// lines 60–97). The vendor template is a `__global__` launched via
// `cudaLaunchCooperativeKernel` from
// `BatchPagedAttentionPersistent` (`persistent.cuh:599`); the DC
// version below is a `__device__` callable that runs the same
// `Runner1::Run + Runner2::Run + grid.sync + Reduction::Run`
// sequence inside another `__global__` (the megakernel persistent
// kernel).
//
// The two Runners
// (`flashinfer::BlockBatchPagedAttentionPersistent::Run`,
// `persistent.cuh:181`) and the reduction
// (`flashinfer::BlockBatchReductionPersistent::Run`,
// `persistent.cuh:488`) are both `static __device__ __forceinline__
// void` — directly callable. Same shape as the CUTLASS DC pattern
// (`dc_cutlass.cuh`); the only differences are (1) two Runners
// share `smem` sequentially (they reuse the buffer because Runner1
// finishes before Runner2 starts, with the CTA's __syncthreads
// providing the intra-CTA fence), and (2) a grid sync separates
// the Runners from the reduction phase because the reduction
// reads partial outputs the Runners wrote across CTAs.
//
// Caller contract:
//
//   - `params_1`, `params_2`: prepared by the host-side FlashInfer
//     plan path (`fi_plan_*` in `flashinfer_shim.cu.j2`). The
//     megakernel launcher copies them to device-resident storage
//     and passes pointers via the megakernel pointer table; the
//     calling arm in `prim_mega.cu` dereferences and forwards.
//   - `smem`: kernel-wide shared-memory base. Must be sized to
//     `max(KTraits1::SharedStorage, KTraits2::SharedStorage,
//     ReductionKTraits::SMEM_SIZE)` — same union-max
//     `BatchPagedAttentionPersistent` computes at line 628.
//
// Why path (1) and not "treat attention as a PrimMega boundary":
// FlashInfer's persistent attention is itself cooperatively
// launched (line 642 `cudaLaunchCooperativeKernel`) — its own
// internal `cg::this_grid().sync()` is the same primitive
// PrimMega already uses between phases. Nesting them is a no-op
// at runtime: the inner grid sync is the same hardware barrier
// as PrimMega's between-phase sync.

#pragma once

#include <cooperative_groups.h>

#include <flashinfer/attention/persistent.cuh>
#include <flashinfer/attention/scheduler.cuh>

namespace dc_flashinfer {

namespace cg = cooperative_groups;

// Mirrors the body of `flashinfer::PersistentKernelTemplate`
// (persistent_template.cuh:60–97). The template parameters are
// the three Runner classes the host-side
// `BatchPagedAttentionPersistent<...>` computes — pass them
// pre-instantiated so we don't reproduce its 10+ lines of KTraits
// derivation here.
template <typename Runner1, typename Runner2, typename Reduction, typename Params>
__device__ __forceinline__ void dc_persistent_attn(const Params& params_1,
                                                   const Params& params_2,
                                                   char* smem) {
    auto& smem_storage_1 =
        *reinterpret_cast<typename Runner1::KTraits::SharedStorage*>(smem);
    auto& smem_storage_2 =
        *reinterpret_cast<typename Runner2::KTraits::SharedStorage*>(smem);

    // Runner1 runs all CTAs' "long-Q" tasks (CTA_TILE_Q=128,
    // prefill-shaped). Runner2 runs the "short-Q" tasks
    // (CTA_TILE_Q=16, decode-shaped). They share `smem` because
    // each Runner's body has its own intra-CTA __syncthreads
    // fences, and Runner1 has fully completed when Runner2 starts
    // on the same CTA.
    Runner1::Run(params_1, &smem_storage_1);
    Runner2::Run(params_2, &smem_storage_2);

    // Grid-wide sync between the per-CTA partial outputs and the
    // reduction. Required because the reduction reads
    // `partial_o` / `partial_lse` slots populated by Runners on
    // OTHER CTAs.
    cg::this_grid().sync();

    Reduction::Run(params_1.partial_o,
                   params_1.final_o,
                   params_1.partial_lse,
                   params_1.final_lse,
                   *(params_1.num_packed_qo_len),
                   params_1.gqa_group_size,
                   params_1.num_kv_heads,
                   params_1.merge_indptr,
                   params_1.merge_o_indices,
                   smem);
}

}  // namespace dc_flashinfer
