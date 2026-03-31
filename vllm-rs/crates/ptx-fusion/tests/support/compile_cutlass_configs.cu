// Compile multiple CUTLASS bf16 GEMM configurations to PTX.
//
// Each configuration is a separate .entry in the output PTX.
// The Rust runtime selects the best config per problem size.
//
// Compile to PTX (not binary):
//   nvcc -ptx -arch=sm_89 -O2 -std=c++17 \
//     -I$HOME/.cache/cutlass/include \
//     -o kernels/cutlass_bf16_configs_sm89.ptx \
//     compile_cutlass_configs.cu

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

using bf16 = cutlass::bfloat16_t;

// Config A: 64x128x32, 3 stages (36KB SMEM) — best for decode (M ≤ ~64)
// Wide-N tile maximizes parallelism for small batch sizes.
// Benchmarked: 3.74x faster than cuBLAS at bs=1, parity at bs=32.
using GemmA = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,
    bf16, cutlass::layout::ColumnMajor,
    bf16, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3, 8, 8>;

// Config B: 128x128x32, 3 stages (48KB SMEM) — good all-rounder
using GemmB = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,
    bf16, cutlass::layout::ColumnMajor,
    bf16, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3, 8, 8>;

// Config C: 128x128x64, 3 stages (96KB SMEM) — best for large prefill
using GemmC = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,
    bf16, cutlass::layout::ColumnMajor,
    bf16, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3, 8, 8>;

// Config D: 16x128x32, 3 stages (12KB SMEM) — best for decode (M ≤ 16)
// Minimal tile_m avoids wasted compute when M=1..16.
using GemmD = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor,
    bf16, cutlass::layout::ColumnMajor,
    bf16, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<16, 128, 32>,
    cutlass::gemm::GemmShape<16, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    3, 8, 8>;

// Force template instantiation — nvcc will emit .entry for each
template __global__ void cutlass::Kernel<GemmA::GemmKernel>(GemmA::GemmKernel::Params);
template __global__ void cutlass::Kernel<GemmB::GemmKernel>(GemmB::GemmKernel::Params);
template __global__ void cutlass::Kernel<GemmC::GemmKernel>(GemmC::GemmKernel::Params);
template __global__ void cutlass::Kernel<GemmD::GemmKernel>(GemmD::GemmKernel::Params);
