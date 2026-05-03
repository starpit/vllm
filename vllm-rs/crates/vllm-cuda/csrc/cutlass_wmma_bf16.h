// SPDX-License-Identifier: Apache-2.0
// Adds a Wmma<bfloat16_t, ...> specialization to CUTLASS 2.x.
// Vendored CUTLASS only ships Wmma<half_t, ...> (sm70) and Wmma<int4b_t, ...> (sm75).
// cuBLAS uses bf16 wmma kernels heavily for small-M shapes on sm80+ (Ada/Ampere/Hopper).
//
// IMPLEMENTATION NOTE — bypassing CutlassToWmmaDataType:
// CUTLASS's existing CutlassToWmmaDataType<bfloat16_t> specialization is gated
// behind `#if __CUDA_ARCH__ >= 800` (cutlass/arch/wmma.h:86–91). That guard means
// the typedef is INVISIBLE during host-side template instantiation (cudafe1
// processes everything before per-arch device pass), so `Wmma<bf16,…>::FragmentA`
// would resolve to `nvcuda::wmma::fragment<…, cutlass::bfloat16_t, …>` — an
// incomplete type. We sidestep by hard-coding `__nv_bfloat16` directly here.
//
// This header MUST be included BEFORE any other CUTLASS header that consumes Wmma<>.

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/arch/wmma.h>

#if defined(CUTLASS_ARCH_WMMA_ENABLED)

#include <mma.h>
#include <cuda_bf16.h>

namespace cutlass {
namespace arch {

template <
    typename Shape_,
    typename LayoutA_,
    typename LayoutB_,
    typename ElementC_,
    typename LayoutC_>
struct Wmma<
    Shape_,
    cutlass::bfloat16_t,
    LayoutA_,
    cutlass::bfloat16_t,
    LayoutB_,
    ElementC_,
    LayoutC_,
    cutlass::arch::OpMultiplyAdd
> {
    using Shape    = Shape_;
    using ElementA = cutlass::bfloat16_t;
    using LayoutA  = LayoutA_;
    using ElementB = cutlass::bfloat16_t;
    using LayoutB  = LayoutB_;
    using ElementC = ElementC_;
    using LayoutC  = LayoutC_;
    using Operator = cutlass::arch::OpMultiplyAdd;
    using ArchTag  = arch::Sm80;

    static_assert(
        platform::is_same<cutlass::gemm::GemmShape<16, 16, 16>, Shape>::value ||
        platform::is_same<cutlass::gemm::GemmShape< 8, 32, 16>, Shape>::value ||
        platform::is_same<cutlass::gemm::GemmShape<32,  8, 16>, Shape>::value,
        "Supported wmma instruction shapes for bf16: 16x16x16, 8x32x16, 32x8x16");

    static_assert(
        platform::is_same<float, ElementC>::value ||
        platform::is_same<cutlass::bfloat16_t, ElementC>::value,
        "Supported wmma output types for bf16 multiplicands: f32, bf16");

    // Hard-coded `__nv_bfloat16` to bypass the guarded
    // `CutlassToWmmaDataType<bfloat16_t>` specialization (see header note).
    using FragmentA = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        Shape::kM, Shape::kN, Shape::kK,
        __nv_bfloat16,
        typename CutlassToWmmaLayout<LayoutA>::Layout>;

    using FragmentB = nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        Shape::kM, Shape::kN, Shape::kK,
        __nv_bfloat16,
        typename CutlassToWmmaLayout<LayoutB>::Layout>;

    using FragmentC = nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,
        Shape::kM, Shape::kN, Shape::kK,
        typename CutlassToWmmaDataType<ElementC>::Type>;

    CUTLASS_DEVICE
    void operator()(
        FragmentC&       D,
        FragmentA const& A,
        FragmentB const& B,
        FragmentC const& C) const {
        nvcuda::wmma::mma_sync(D, A, B, C);
    }
};

} // namespace arch
} // namespace cutlass

#endif // CUTLASS_ARCH_WMMA_ENABLED
