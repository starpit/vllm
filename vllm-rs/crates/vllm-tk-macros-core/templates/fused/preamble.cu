// GENERATED: Fused prefill layer kernel ({{ mode_label }}, {{ cta_rows }}-row CTA)
// Grid: ceil(num_prefill_tokens/{{ cta_rows }}) CTAs

#define SM89_NUM_LAYERS             {{ nl }}
#define SM89_HIDDEN_DIM             {{ hd }}
#define SM89_INTERMEDIATE_DIM       {{ id }}
#define SM89_HEAD_DIM               {{ hdm }}
#define SM89_NUM_ATTENTION_HEADS    {{ nah }}
#define SM89_NUM_KV_HEADS           {{ nkh }}

#include "llama_sm89.cuh"

// CUTLASS device-side GEMM primitives. Include-only here; whether any
// CUTLASS code is actually emitted depends on per-variant phase templates.
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>

using namespace kittens;
using namespace kittens::prototype::vm;
using globals = llama_sm89_globals;

// ── CUTLASS type instantiation (mirrors preamble_header.cu) ──
namespace pfl_cutlass {
    using ElementA       = cutlass::bfloat16_t;
    using ElementB       = cutlass::bfloat16_t;
    using ElementAccum   = float;
    using LayoutA        = cutlass::layout::RowMajor;
    using LayoutB        = cutlass::layout::ColumnMajor;
    using LayoutC        = cutlass::layout::RowMajor;

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
        /*Stages=*/4,
        cutlass::arch::OpMultiplyAdd>;

    using ThreadblockMma = typename DefaultMmaT::ThreadblockMma;
    using IteratorA      = typename DefaultMmaT::IteratorA;
    using IteratorB      = typename DefaultMmaT::IteratorB;
    using MmaSharedStorage = typename ThreadblockMma::SharedStorage;

    // Epilogue: residual add via LinearCombination(alpha=1, beta=1).
    // ElementOutput = bf16, vec width = 8, accum = fp32, compute = fp32.
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

    using Epilogue            = typename DefaultEpilogueT::Epilogue;
    using OutputTileIterator  = typename DefaultEpilogueT::OutputTileIterator;
    using EpilogueSharedStorage = typename Epilogue::SharedStorage;

    // Total shmem for the cutlass GEMM phase: union of mainloop and epilogue.
    union SharedStorage {
        MmaSharedStorage main_loop;
        EpilogueSharedStorage epilogue;
    };

    // ── Custom output op: D = silu(alpha*acc) * source ──
    // For the gate_up fused phase: gate's epilogue reads the already-computed
    // up output from `source` (which lives in the same gmem buffer the gate
    // GEMM writes to) and emits silu(gate_acc) * up_existing.
    template <typename ElementOutput_, int Count_, typename ElementAccum_,
              typename ElementCompute_ = ElementAccum_>
    class LinearCombinationSiluMul {
    public:
        using ElementOutput      = ElementOutput_;
        using ElementAccumulator = ElementAccum_;
        using ElementCompute     = ElementCompute_;
        static int const kCount  = Count_;

        using FragmentOutput      = cutlass::Array<ElementOutput, kCount>;
        using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
        using FragmentSource      = cutlass::Array<ElementOutput, kCount>;
        using ComputeFragment     = cutlass::Array<ElementCompute, kCount>;

        struct Params {
            ElementCompute alpha = ElementCompute(1);
            ElementCompute beta  = ElementCompute(1);  // unused but kept for API parity
            ElementCompute const *alpha_ptr = nullptr;
            ElementCompute const *beta_ptr  = nullptr;
            ElementCompute const *const *alpha_ptr_array = nullptr;
            ElementCompute const *const *beta_ptr_array  = nullptr;
            CUTLASS_HOST_DEVICE
            Params() {}
            CUTLASS_HOST_DEVICE
            Params(ElementCompute a, ElementCompute b) : alpha(a), beta(b) {}
        };

    private:
        ElementCompute alpha_;

    public:
        CUTLASS_HOST_DEVICE
        explicit LinearCombinationSiluMul(Params const &params)
            : alpha_(params.alpha) {}

        CUTLASS_HOST_DEVICE
        bool is_source_needed() const { return true; }
        CUTLASS_HOST_DEVICE
        void set_k_partition(int k_partition, int k_partition_count) {}

        // residual fragment (acc only) — without source. Defines D = silu(α*acc).
        // CUTLASS LinearCombination provides this overload; we mirror it.
        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(FragmentAccumulator const &accum) const {
            FragmentOutput result;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                float a = float(accum[i]) * float(alpha_);
                float s = a / (1.0f + ::expf(-a));
                result[i] = ElementOutput(s);
            }
            return result;
        }

        // The fused op: D = silu(α*acc) * source
        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(FragmentAccumulator const &accum,
                                  FragmentSource const &source) const {
            FragmentOutput result;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                float a = float(accum[i]) * float(alpha_);
                float silu_a = a / (1.0f + ::expf(-a));
                float src = float(source[i]);
                result[i] = ElementOutput(silu_a * src);
            }
            return result;
        }
    };

    // Epilogue specialized to use the SiluMul output op.
    using OutputOpSiluMul = LinearCombinationSiluMul<
        ElementOutput, kEpilogueElementsPerAccess, ElementAccum, ElementAccum>;

    using DefaultEpilogueSiluMulT = cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape,
        typename ThreadblockMma::Operator,
        /*PartitionsK=*/1,
        OutputOpSiluMul,
        kEpilogueElementsPerAccess>;
    using EpilogueSiluMul = typename DefaultEpilogueSiluMulT::Epilogue;
}  // namespace pfl_cutlass

// ── Small CUTLASS shape: <128,128,32> with WarpShape <64,64,32> (4 warps).
// Tuned for QKV (M=q_size, K=2048, N=2304) and o_proj (M=q_size, K=2048, N=2048),
// where the default 256-row CTA M is mostly padding.
namespace pfl_cutlass_small {
    using ElementA       = cutlass::bfloat16_t;
    using ElementB       = cutlass::bfloat16_t;
    using ElementAccum   = float;
    using LayoutA        = cutlass::layout::RowMajor;
    using LayoutB        = cutlass::layout::ColumnMajor;
    using LayoutC        = cutlass::layout::RowMajor;

    // 8 warps (256 threads) to match the kernel-wide launch geometry.
    // 128x128 / (64x32) = 2 x 4 = 8 warps. K=32. Stages=3.
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape        = cutlass::gemm::GemmShape<64, 32, 32>;
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

    using ElementOutput = cutlass::bfloat16_t;
    static constexpr int kEpilogueElementsPerAccess = 8;
    using OutputOpT = cutlass::epilogue::thread::LinearCombination<
        ElementOutput, kEpilogueElementsPerAccess, ElementAccum, ElementAccum>;
    using DefaultEpilogueT = cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape, typename ThreadblockMma::Operator,
        /*PartitionsK=*/1, OutputOpT, kEpilogueElementsPerAccess>;
    using Epilogue            = typename DefaultEpilogueT::Epilogue;
    using OutputTileIterator  = typename DefaultEpilogueT::OutputTileIterator;
    using EpilogueSharedStorage = typename Epilogue::SharedStorage;

    static_assert(ThreadblockMma::WarpCount::kCount == 8,
                  "small CUTLASS must use 8 warps to match kernel launch");

    union SharedStorage {
        MmaSharedStorage main_loop;
        EpilogueSharedStorage epilogue;
    };

    // SiluMul not used by QKV/o, but include for template parity so the
    // ns-parametric template compiles unconditionally.
    template <typename ElementOutput_, int Count_, typename ElementAccum_,
              typename ElementCompute_ = ElementAccum_>
    class LinearCombinationSiluMul {
    public:
        using ElementOutput      = ElementOutput_;
        using ElementAccumulator = ElementAccum_;
        using ElementCompute     = ElementCompute_;
        static int const kCount  = Count_;
        using FragmentOutput      = cutlass::Array<ElementOutput, kCount>;
        using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
        using FragmentSource      = cutlass::Array<ElementOutput, kCount>;
        struct Params {
            ElementCompute alpha = ElementCompute(1);
            ElementCompute beta  = ElementCompute(1);
            ElementCompute const *alpha_ptr = nullptr;
            ElementCompute const *beta_ptr  = nullptr;
            ElementCompute const *const *alpha_ptr_array = nullptr;
            ElementCompute const *const *beta_ptr_array  = nullptr;
            CUTLASS_HOST_DEVICE Params() {}
            CUTLASS_HOST_DEVICE Params(ElementCompute a, ElementCompute b) : alpha(a), beta(b) {}
        };
    private:
        ElementCompute alpha_;
    public:
        CUTLASS_HOST_DEVICE explicit LinearCombinationSiluMul(Params const &p) : alpha_(p.alpha) {}
        CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
        CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(FragmentAccumulator const &accum) const {
            FragmentOutput r;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                float a = float(accum[i]) * float(alpha_);
                r[i] = ElementOutput(a / (1.0f + ::expf(-a)));
            }
            return r;
        }
        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(FragmentAccumulator const &accum,
                                  FragmentSource const &source) const {
            FragmentOutput r;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < kCount; ++i) {
                float a = float(accum[i]) * float(alpha_);
                float s = a / (1.0f + ::expf(-a));
                r[i] = ElementOutput(s * float(source[i]));
            }
            return r;
        }
    };
    using OutputOpSiluMul = LinearCombinationSiluMul<
        ElementOutput, kEpilogueElementsPerAccess, ElementAccum, ElementAccum>;
    using DefaultEpilogueSiluMulT = cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape, typename ThreadblockMma::Operator,
        /*PartitionsK=*/1, OutputOpSiluMul, kEpilogueElementsPerAccess>;
    using EpilogueSiluMul = typename DefaultEpilogueSiluMulT::Epilogue;
}  // namespace pfl_cutlass_small

constexpr int PFL_NUM_WARPS = {{ num_warps }};
constexpr int PFL_GQA_RATIO = {{ gqa_ratio }};
constexpr int PFL_KV_PAGE_SIZE = {{ kv_page_size }};
constexpr int PFL_ITERS_PER_PAGE = {{ iters_per_page }};
constexpr int PFL_HEAD_DIM = {{ hdm }};
constexpr int PFL_SHMEM = {{ total_shmem }};
constexpr int PFL_KV_TILE_BYTES = {{ kv_tile_bytes }};
// PFL_Q_ROWS: per-warp M for attention / rope / rmsnorm (fixed at 16).
constexpr int PFL_Q_ROWS = 16;
// PFL_GEMM_M: per-warp M for GEMM accumulator (parametric).
constexpr int PFL_GEMM_M = {{ gemm_warp_m }};
constexpr int PFL_GEMM_M_SUBS = {{ gemm_m_subs }};

constexpr int PFL_CTA_ROWS = {{ cta_rows }};
constexpr int PFL_K_DIM = {{ k_dim }};
constexpr int PFL_OUT_BLOCK = {{ out_block }};
constexpr int PFL_RDPW = {{ rdpw }};
constexpr int PFL_N_TILES = PFL_OUT_BLOCK / 16;

using pfl_a_st = st_bf<PFL_GEMM_M, PFL_K_DIM>;
using pfl_b_st = st_bf<PFL_OUT_BLOCK, PFL_K_DIM>;
using pfl_acc_rt = rt_fl<PFL_GEMM_M, PFL_OUT_BLOCK>;
using pfl_a_rt = rt_bf<PFL_GEMM_M, PFL_K_DIM>;
using pfl_b_slice_st = st_bf<16, PFL_K_DIM>;

using pfl_q_st  = st_bf<PFL_Q_ROWS, PFL_HEAD_DIM>;
using pfl_kv_st = st_bf<PFL_KV_PAGE_SIZE, PFL_HEAD_DIM>;
using pfl_q_rt  = rt_bf<PFL_Q_ROWS, PFL_HEAD_DIM>;
using pfl_k_rt  = rt_bf<PFL_KV_PAGE_SIZE, PFL_HEAD_DIM>;
using pfl_v_rt  = rt_bf<PFL_KV_PAGE_SIZE, PFL_HEAD_DIM, col_l>;
using pfl_score_fl = rt_fl<PFL_Q_ROWS, PFL_KV_PAGE_SIZE>;
using pfl_score_bf = rt_bf<PFL_Q_ROWS, PFL_KV_PAGE_SIZE>;
using pfl_o_rt  = rt_fl<PFL_Q_ROWS, PFL_HEAD_DIM>;
using pfl_o_bf  = rt_bf<PFL_Q_ROWS, PFL_HEAD_DIM>;
using pfl_max_rv = col_vec<rt_fl<PFL_Q_ROWS, PFL_HEAD_DIM>>;
using pfl_norm_rv = col_vec<rt_fl<PFL_Q_ROWS, PFL_HEAD_DIM>>;
using pfl_o_sv  = sv_bf<PFL_HEAD_DIM>;

__device__ static inline void pfl_cp_async_wait_all() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
    asm volatile("cp.async.wait_all;\n"     ::: "memory");
}

__device__ static inline void pfl_load_b_slice(
    rt_bf<16, PFL_K_DIM> &dst, const st_bf<16, PFL_K_DIM> &src) {
    uint32_t saddr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int lane = kittens::laneid();
    int row = lane % 16;
    bf16_2 tmp[4];
    #pragma unroll
    for (int j = 0; j < PFL_K_DIM / 16; j++) {
        int col = j * 16 + (lane / 16) * 8;
        move<bf16_2>::ldsm4(tmp[0], tmp[1], tmp[2], tmp[3], src.idx(saddr, {row, col}));
        dst.tiles[0][j].data[0] = tmp[0];
        dst.tiles[0][j].data[1] = tmp[1];
        dst.tiles[0][j].data[2] = tmp[2];
        dst.tiles[0][j].data[3] = tmp[3];
    }
}
