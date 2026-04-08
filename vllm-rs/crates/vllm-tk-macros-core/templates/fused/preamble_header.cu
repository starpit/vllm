// GENERATED: Fused prefill layer kernel ({{ mode_label }})

#define SM89_NUM_LAYERS             {{ nl }}
#define SM89_HIDDEN_DIM             {{ hd }}
#define SM89_INTERMEDIATE_DIM       {{ id }}
#define SM89_HEAD_DIM               {{ hdm }}
#define SM89_NUM_ATTENTION_HEADS    {{ nah }}
#define SM89_NUM_KV_HEADS           {{ nkh }}

#include "llama_sm89.cuh"

// CUTLASS device-side GEMM primitives (used by phases that opt in via the
// `cutlass_gemm` config flag). The headers are include-only; whether any
// CUTLASS code is actually emitted depends on the per-variant template.
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>

// ── CUTLASS type definitions for our GEMM phases ──
// These are SIZE-CHECKED at compile time; they don't emit any code unless a
// phase actually instantiates the Mma class. Used as the entry point for
// dropping cuBLAS-quality GEMM kernels into the megakernel.
//
// Picked for our 256-thread CTA (8 warps): ThreadblockShape 256x128 with
// WarpShape 64x64 = 4 (m) x 2 (n) = 8 warps. InstructionShape 16x8x16 is
// the bf16 tensor-core MMA on sm_80+.
//
// down_proj shape: M=seq, K=ID=8192, N=HD=2048. Tile-divisible by (256,128,32):
// M is padded to 256 (one row tile per CTA at seq=1024 → 4 m-tiles total),
// N=2048/128=16 n-tiles, K=8192/32=256 K-iters.
namespace pfl_cutlass {
    using ElementA       = cutlass::bfloat16_t;
    using ElementB       = cutlass::bfloat16_t;
    using ElementAccum   = float;
    using LayoutA        = cutlass::layout::RowMajor;     // input is [M,K] row-major
    using LayoutB        = cutlass::layout::ColumnMajor;  // weight is [N,K] row-major == [K,N] col-major
    using LayoutC        = cutlass::layout::RowMajor;     // output [M,N] row-major

    using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 32>;
    using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using DefaultMmaT = cutlass::gemm::threadblock::DefaultMma<
        ElementA, LayoutA, /*kAlignmentA=*/8,
        ElementB, LayoutB, /*kAlignmentB=*/8,
        ElementAccum, LayoutC,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape, WarpShape, InstructionShape,
        /*Stages=*/3,
        cutlass::arch::OpMultiplyAdd>;

    using ThreadblockMma = typename DefaultMmaT::ThreadblockMma;
    using IteratorA      = typename DefaultMmaT::IteratorA;
    using IteratorB      = typename DefaultMmaT::IteratorB;
    using MmaSharedStorage = typename ThreadblockMma::SharedStorage;

    // Epilogue (residual add via LinearCombination(alpha=1, beta=1)).
    using ElementOutput = cutlass::bfloat16_t;
    static constexpr int kEpilogueElementsPerAccess = 8;
    using OutputOpT = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        kEpilogueElementsPerAccess,
        ElementAccum,
        ElementAccum>;

    using DefaultEpilogueT = cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape,
        typename ThreadblockMma::Operator,
        /*PartitionsK=*/1,
        OutputOpT,
        kEpilogueElementsPerAccess>;

    using Epilogue              = typename DefaultEpilogueT::Epilogue;
    using OutputTileIterator    = typename DefaultEpilogueT::OutputTileIterator;
    using EpilogueSharedStorage = typename Epilogue::SharedStorage;

    union SharedStorage {
        MmaSharedStorage main_loop;
        EpilogueSharedStorage epilogue;
    };

    static_assert(sizeof(SharedStorage) <= 80 * 1024,
                  "CUTLASS SharedStorage exceeds 80KB budget");
    // Print the actual size at compile time via _Static_assert with the
    // value embedded in the message (commented out — uncomment to inspect):
    // static_assert(sizeof(SharedStorage) == 0, "see size");

    static constexpr int kCutlassThreads = ThreadblockMma::WarpCount::kCount * 32;
    static_assert(kCutlassThreads == 256,
                  "CUTLASS ThreadblockMma WarpCount must equal 8 (256 threads)");
}  // namespace pfl_cutlass
