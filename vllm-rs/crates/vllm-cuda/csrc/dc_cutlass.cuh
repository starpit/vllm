// SPDX-License-Identifier: Apache-2.0
// Device-callable wrappers for CUTLASS 2.x GEMM templates.
//
// Each `dc_cutlass_gemm<DeviceGemm>` constructs the kernel-namespace
// `Params` on-device (CUTLASS's `Params(...)` constructor and
// `ThreadblockSwizzle::get_tiled_shape(...)` are both
// `CUTLASS_HOST_DEVICE`) and then calls
// `GemmKernel::operator()(params, shared_storage)` — the same body
// that `cutlass::device_kernel<GemmKernel>` runs in the standalone
// host-launched grid (see `cutlass/device_kernel.h:118`).
//
// No host-side `prepare_*_params` shim is needed because Params
// construction has no host-side dependencies for non-splitK
// configs (split_k_slices = 1; the workspace branch in
// device::Gemm::initialize at gemm.h:413 is skipped). SplitK and
// sm90 GemmUniversal need their own DC wrappers (different Params
// shapes); land in follow-up commits.
//
// Caller contract:
//
//   - `M`, `N`, `K`: GEMM dims. Same as the host launcher.
//   - `A`, `B`, `C`: device pointers. A is RowMajor [M,K], B is
//     ColumnMajor [K,N] (= RowMajor [N,K] — what we store), C is
//     RowMajor [M,N]. Identical to the standalone CUTLASS GEMM.
//   - `alpha`, `beta`: epilogue scalars.
//   - `smem`: kernel-wide shared-memory base pointer; the DC fn
//     reinterprets it as `GemmKernel::SharedStorage*`. The
//     megakernel allocates the union-max across all phases'
//     SharedStorage sizes; this fn uses only its prefix.
//
// Block-coord handling: the GEMM template reads `blockIdx` to
// resolve its tile coord (gemm.h:202–216). The persistent kernel
// launches with grid_dim = max-needed across all phases; the
// GEMM's own bounds check at gemm.h:212 early-returns CTAs whose
// tile coord exceeds `grid_tiled_shape`, which is what we want.

#pragma once

#include <cooperative_groups.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/gemv.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/tensor_ref.h>

namespace dc_cutlass {

template <typename DeviceGemm>
__device__ __forceinline__ void dc_gemm(void* C,
                                        const void* A,
                                        const void* B,
                                        int M,
                                        int N,
                                        int K,
                                        float alpha,
                                        float beta,
                                        char* smem) {
    using GemmKernel = typename DeviceGemm::GemmKernel;
    using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
    using ThreadblockShape = typename DeviceGemm::ThreadblockShape;

    cutlass::gemm::GemmCoord problem_size{M, N, K};

    // Mirror device::Gemm::initialize() for split_k_slices = 1
    // (see cutlass/gemm/device/gemm.h:408). HOST_DEVICE callable —
    // no host-side dependencies on this branch.
    cutlass::gemm::GemmCoord grid_tiled_shape =
        ThreadblockSwizzle().get_tiled_shape(
            problem_size,
            {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
            /*split_k_slices=*/1);

    // Mma's iterators take *non-const* TensorRef (they touch
    // internal smem state). Host-side `device::Gemm::Arguments`
    // hides this via `args.ref_A.non_const_ref()` in
    // `device::Gemm::initialize`; on device we cast directly. The
    // kernel itself only reads A/B, so dropping const is safe at
    // the type-interface level — analogous to a `mut` borrow we
    // immediately project to read-only.
    typename DeviceGemm::ElementA* A_ptr = reinterpret_cast<
        typename DeviceGemm::ElementA*>(const_cast<void*>(A));
    typename DeviceGemm::ElementB* B_ptr = reinterpret_cast<
        typename DeviceGemm::ElementB*>(const_cast<void*>(B));
    typename DeviceGemm::ElementC* C_ptr =
        reinterpret_cast<typename DeviceGemm::ElementC*>(C);

    typename GemmKernel::Params params(
        problem_size,
        grid_tiled_shape,
        {A_ptr, K},
        {B_ptr, K},
        {C_ptr, N},
        {C_ptr, N},
        {alpha, beta});

    GemmKernel op;
    op(params, *reinterpret_cast<typename GemmKernel::SharedStorage*>(smem));
}

// ── dc_gemv — M=1 SIMT GEMV ──────────────────────────────────────
//
// Mirrors the standalone host launcher's argument layout
// (`cutlass_gemv_launch` in `cutlass_standalone_gemm.cu`):
//
//   y[N] = W[N,K] @ x[K]    where W (= our GEMM's B, ColumnMajor [K,N]
//                                    = RowMajor [N,K]) is GEMV's
//                                    weight matrix, x (= our GEMM's A
//                                    [1,K]) is the input vector,
//                                    y (= our GEMM's C [1,N]) is the
//                                    output vector.
//
// `GemvKernel`'s `Params` is its `Arguments` struct (`Params =
// Arguments`); construction has no host-side dependencies. The
// `SharedStorage` is an empty union — `smem` argument is unused but
// kept in the signature for symmetry with `dc_gemm`. The kernel
// itself reads `blockIdx` to dispatch row-tile work; the persistent
// kernel launches with grid_dim = max-needed across phases and the
// kernel's own bound check early-returns out-of-range CTAs.
//
// Caller contract (matches the host launcher's M=1 mapping):
//   - `M`: must be 1 (GEMV-only).
//   - `N`, `K`: same as GEMM dims.
//   - `A`: bf16 input vector (RowMajor [1,K]) — GEMV's `ptr_B`.
//   - `B`: bf16 weight (ColumnMajor [K,N] = RowMajor [N,K]) —
//     GEMV's `ref_A`.
//   - `C`: bf16 output vector (RowMajor [1,N]) — GEMV's
//     `ptr_C` and `ptr_D` (in-place for beta != 0).
//   - `alpha`, `beta`: epilogue scalars.

template <typename GemvKernel>
__device__ __forceinline__ void dc_gemv(void* C,
                                        const void* A,
                                        const void* B,
                                        int M,
                                        int N,
                                        int K,
                                        float alpha,
                                        float beta,
                                        char* /*smem*/) {
    // GEMV is M=1 by construction; out-of-contract caller silently
    // no-ops (matches host launcher's `if (M != 1) return -10`).
    if (M != 1) return;

    using ElementA = typename GemvKernel::ElementA;
    using ElementB = typename GemvKernel::ElementB;
    using ElementC = typename GemvKernel::ElementC;
    using TensorRefA = typename GemvKernel::TensorRefA;
    using EpilogueOutputOp = typename GemvKernel::EpilogueOutputOp;

    cutlass::layout::RowMajor layout_A(K);
    TensorRefA ref_A{
        reinterpret_cast<ElementA*>(const_cast<void*>(B)),
        layout_A};

    // `kernel::Gemv::Arguments` (alias `Params`) has a host-only
    // default + parameterized constructor — not `CUTLASS_HOST_DEVICE`
    // unlike `kernel::Gemm::Params`. Skip the constructor: place
    // raw bytes, cast to `Params*`, assign fields. Each field type
    // (MatrixCoord, int32_t, LinearCombination::Params, TensorRef,
    // pointers, int64_t) has a HOST_DEVICE constructor + copy-
    // assignment, so the field-by-field path avoids any host-only
    // codepath. Same trick CUTLASS itself uses internally to thread
    // arg values to running kernels via cudaLaunchKernel arg copy.
    alignas(16) char params_buf[sizeof(typename GemvKernel::Params)];
    typename GemvKernel::Params* params =
        reinterpret_cast<typename GemvKernel::Params*>(params_buf);
    params->problem_size = cutlass::MatrixCoord{N, K};
    params->batch_count = 1;
    params->output_op = typename EpilogueOutputOp::Params{alpha, beta};
    params->ref_A = ref_A;
    params->ptr_B = reinterpret_cast<ElementB const*>(A);
    params->ptr_C = reinterpret_cast<ElementC const*>(C);
    params->ptr_D = reinterpret_cast<ElementC*>(C);
    params->batch_stride_A = static_cast<int64_t>(N) * K;
    params->batch_stride_B = static_cast<int64_t>(K);
    params->batch_stride_C = static_cast<int64_t>(N);
    params->batch_stride_D = static_cast<int64_t>(N);

    typename GemvKernel::SharedStorage smem_storage{};
    GemvKernel op;
    op(*params, smem_storage);
}

// ── dc_gemm_splitk — GemmSplitKParallel (two-phase) ──────────────
//
// `device::GemmSplitKParallel`'s host launcher fires TWO kernels:
// (1) the GEMM grid writes per-K-slice partial sums to a workspace
// of shape [split_k, M, N] in `float` accumulator format; (2) a
// Reduction kernel reads the workspace and writes the final
// `[M, N]` bf16 output. Both kernels' Params and operator() are
// `CUTLASS_HOST_DEVICE` / `CUTLASS_DEVICE` callable, so we inline
// both phases here separated by `cg::this_grid().sync()`.
//
// Caller contract additions over `dc_gemm`:
//   - `workspace`: caller-supplied [split_k, M, N] float scratch.
//     Size formula: `split_k * M * N * sizeof(float)`. Same
//     contract as the host launcher's `cutlass_gemm_*_skN_launch`
//     workspace argument.
//   - `split_k_slices`: number of K partitions. Must match the
//     `DeviceGemmSplitK` template's calibration (the host's
//     `_skN` suffix in the CSV row — same N here).
//
// Persistent-kernel grid: blockIdx is shared across both phases.
// The launcher sizes grid_dim to the max-needed across all phases
// in the program. Each kernel's own bounds check early-returns
// out-of-range CTAs (GemmSplitKParallel: `params.grid_tiled_shape`
// guard at gemm_splitk_parallel.h:143; Reduction: thread_offset
// guard at reduce_split_k.h:159). The Reduction's logical grid is
// 2D (M-tiles × N-tiles); blockIdx.z is unused for it.

template <typename DeviceGemmSplitK>
__device__ __forceinline__ void dc_gemm_splitk(void* C,
                                               const void* A,
                                               const void* B,
                                               int M,
                                               int N,
                                               int K,
                                               float alpha,
                                               float beta,
                                               int split_k_slices,
                                               void* workspace,
                                               char* smem) {
    using GemmKernel = typename DeviceGemmSplitK::GemmKernel;
    using ReductionKernel = typename DeviceGemmSplitK::ReductionKernel;
    using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
    using ThreadblockShape = typename DeviceGemmSplitK::ThreadblockShape;
    using ElementAccumulator = typename DeviceGemmSplitK::ElementAccumulator;
    using OutputOp = typename GemmKernel::OutputOp;
    using ReductionOutputOp = typename ReductionKernel::OutputOp;
    using ReductionOp = typename ReductionKernel::ReductionOp;

    cutlass::gemm::GemmCoord problem_size{M, N, K};

    // Mirror device::GemmSplitKParallel::initialize() (gemm_splitk_parallel.h:271).
    cutlass::gemm::GemmCoord grid_tiled_shape =
        ThreadblockSwizzle().get_tiled_shape(
            problem_size,
            {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
            split_k_slices);

    // Workspace: [split_k, M, N] float, RowMajor stride N.
    cutlass::TensorRef<ElementAccumulator, cutlass::layout::RowMajor>
        ref_workspace(static_cast<ElementAccumulator*>(workspace), N);

    int64_t partition_stride = static_cast<int64_t>(M) * static_cast<int64_t>(N);

    // Phase 1: GEMM grid writes to workspace.
    typename DeviceGemmSplitK::ElementA* A_ptr = reinterpret_cast<
        typename DeviceGemmSplitK::ElementA*>(const_cast<void*>(A));
    typename DeviceGemmSplitK::ElementB* B_ptr = reinterpret_cast<
        typename DeviceGemmSplitK::ElementB*>(const_cast<void*>(B));
    typename DeviceGemmSplitK::ElementC* C_ptr =
        reinterpret_cast<typename DeviceGemmSplitK::ElementC*>(C);

    typename GemmKernel::Params gemm_params(
        problem_size,
        grid_tiled_shape,
        {A_ptr, K},
        {B_ptr, K},
        ref_workspace,
        typename OutputOp::Params{},  // GEMM-phase: identity convert (no alpha/beta — those land in reduction)
        partition_stride);

    GemmKernel gemm_op;
    gemm_op(gemm_params, *reinterpret_cast<typename GemmKernel::SharedStorage*>(smem));

    // Inter-phase barrier — the GEMM grid's writes to workspace must
    // be visible before the Reduction reads them.
    cooperative_groups::this_grid().sync();

    // Phase 2: Reduction reads workspace, writes final C.
    typename ReductionKernel::Params reduce_params(
        problem_size.mn(),
        grid_tiled_shape.k(),
        partition_stride,
        ref_workspace,
        {C_ptr, N},
        {C_ptr, N},
        typename ReductionOutputOp::Params{alpha, beta},
        typename ReductionOp::Params{});

    ReductionKernel reduce_op;
    typename ReductionKernel::SharedStorage reduce_smem{};
    reduce_op(reduce_params, reduce_smem);
}

}  // namespace dc_cutlass
