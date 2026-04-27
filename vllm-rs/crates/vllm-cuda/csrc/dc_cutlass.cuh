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

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/matrix_coord.h>

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

}  // namespace dc_cutlass
