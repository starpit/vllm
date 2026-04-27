// SPDX-License-Identifier: Apache-2.0
// Primitive megakernel persistent kernel (Phase 1 of MEGA_HANDOFF.md).
//
// One persistent `__global__` per arch; this file currently carries
// the Llama-family kernel. The body is a switch over [i32; 32]
// opcodes; each arm calls a `__device__` wrapper around a vendor
// or in-tree DC kernel. CTAs cooperate per phase — the grid size is
// chosen by the launcher to be the max across all phases in the
// program, and per-phase early-exit skips the work for CTAs the op
// doesn't need.
//
// Phase barrier: `cooperative_groups::this_grid().sync()` between
// adjacent ops. We can't use plain `__syncthreads()` because the
// existing dc_* fns in megakernel_ops.cuh early-`return` on
// out-of-range CTAs (`if (row >= num_rows) return;`); a CTA that
// fully early-exits never reaches the post-arm barrier, so a
// __syncthreads on a CTA that DID participate would block forever.
// Grid sync recouples the whole grid regardless of which CTAs ran.
// This costs ~100us per phase boundary on L4 (handoff
// `Handoff::InKernelGridSync = 100us`), which is the price of the
// "primitive" tier — Phase 2 (KvmMega) drops it via per-SM tape +
// mbarrier choreography.
//
// Pointer table: 64-bit pointers don't fit in a single i32 row
// slot, and packing hi/lo halves makes the encoder ugly. Instead
// the launcher passes a side-channel `void**` table; rows index
// into it. Encoder-side this is a single u32 per pointer. CUTLASS
// Params blobs (Phase 1 follow-up, not yet wired) live in this
// table too.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "../megakernel_ops.cuh"

// CUTLASS DC infrastructure. Tile-config list lives in the shared
// `cutlass_gemm_configs.cuh`; we expand it three times here — once
// to declare typedefs, once to populate the `CutlassConfig` enum,
// and once to build the inner switch in `run_cutlass_gemm`. Single
// source of truth keeps the three in lockstep.
#include "../dc_cutlass.cuh"
#include "../cutlass_gemm_configs.cuh"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/gemm/kernel/gemv.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/conversion_op.h>
#include <cutlass/reduction/thread/reduction_operators.h>

// FlashInfer DC. Header carries the
// `dc_flashinfer::dc_persistent_attn<Runner1, Runner2, Reduction,
// Params>` template; no arm is wired up yet because the FI Params
// marshaling (host side: copy plan->params_1 / params_2 to device-
// resident storage; encoder: emit OP_FI_PAGED_ATTN with pt[]
// indices for the two Params blobs + smem offset) is the next
// commit. Including the header here gives the include path a
// compile-time check.
#include "../dc_flashinfer.cuh"

namespace prim_mega_cutlass_configs {

// Expansion #1: typedef per row.
//
// Mirrors the `using Gemm_*` aliases in cutlass_standalone_gemm.cu
// at line 49 (the CUTLASS_GEMM_CONFIG macro body) — same template
// shape with TB / Warp / instruction / epilogue / swizzle slots.
// Naming: `DcGemm_<TB_M>x<TB_N>x<TB_K>_s<STAGES>` (Dc prefix to
// avoid colliding with cutlass_standalone_gemm.cu's `Gemm_*`
// aliases when both are in scope).
#define DC_GEMM_TYPEDEF(TB_M, TB_N, TB_K, STAGES, WARP_M, WARP_N, WARP_K)        \
    using DcGemm_##TB_M##x##TB_N##x##TB_K##_s##STAGES = cutlass::gemm::device::Gemm< \
        cutlass::bfloat16_t, cutlass::layout::RowMajor,                          \
        cutlass::bfloat16_t, cutlass::layout::ColumnMajor,                       \
        cutlass::bfloat16_t, cutlass::layout::RowMajor,                          \
        float,                                                                   \
        cutlass::arch::OpClassTensorOp,                                          \
        cutlass::arch::Sm80,                                                     \
        cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,                              \
        cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>,                        \
        cutlass::gemm::GemmShape<16, 8, 16>,                                     \
        cutlass::epilogue::thread::LinearCombination<                            \
            cutlass::bfloat16_t, 8, float, float>,                               \
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,            \
        STAGES>;
CUTLASS_DC_GEMM_LIST(DC_GEMM_TYPEDEF)
#undef DC_GEMM_TYPEDEF

// SplitK typedef per row. Mirrors the standalone's
// `GemmSplitK_<TB_M>x<TB_N>_s<STAGES>` aliases — same template
// shape with the parallel-K reduction epilogue. SPLIT_K column is
// unused in the typedef itself (it's a runtime arg) but threaded
// through the X-macro for the per-(tile, split_k) switch arm
// below.
#define DC_SPLITK_TYPEDEF(TB_M, TB_N, TB_K, STAGES, WARP_M, WARP_N, WARP_K, SPLIT_K) \
    using DcGemmSplitK_##TB_M##x##TB_N##_s##STAGES##_sk##SPLIT_K =               \
        cutlass::gemm::device::GemmSplitKParallel<                               \
            cutlass::bfloat16_t, cutlass::layout::RowMajor,                      \
            cutlass::bfloat16_t, cutlass::layout::ColumnMajor,                   \
            cutlass::bfloat16_t, cutlass::layout::RowMajor,                      \
            float,                                                                \
            cutlass::arch::OpClassTensorOp,                                      \
            cutlass::arch::Sm80,                                                 \
            cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,                          \
            cutlass::gemm::GemmShape<WARP_M, WARP_N, WARP_K>,                    \
            cutlass::gemm::GemmShape<16, 8, 16>,                                 \
            cutlass::epilogue::thread::LinearCombination<                        \
                cutlass::bfloat16_t, 8, float, float>,                           \
            cutlass::epilogue::thread::Convert<float, 8, float>,                 \
            cutlass::reduction::thread::ReduceAdd<float, float, 8>,              \
            cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,  \
            STAGES>;
CUTLASS_DC_SPLITK_LIST(DC_SPLITK_TYPEDEF)
#undef DC_SPLITK_TYPEDEF

// CUTLASS GEMV kernel — single config, M=1 only. Mirrors
// `GemvKernel_8` in cutlass_standalone_gemm.cu so the CSV-calibrated
// "cutlass_gemv" cost row matches what the megakernel actually runs.
// Same kElementsPerAccess (8) — max bf16 vectorized load. Epilogue
// is `LinearCombination<bf16, 1, float, float>` (kElementsPerAccess=1
// matches the host launcher's epilogue config).
using DcGemvKernel_bf16_8 = cutlass::gemm::kernel::Gemv<
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::bfloat16_t,
    float,
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 1, float, float>,
    8>;

}  // namespace prim_mega_cutlass_configs

namespace cg = cooperative_groups;

namespace prim_mega {

constexpr int INSTRUCTION_WIDTH = 32;

// Opcode space (Phase 1). Extends as DC-sibling wrappers land for
// CUTLASS GEMM tile variants and FlashInfer attention. Negative
// opcodes are reserved for control flow (vendor convention; see
// `~/Megakernels/include/controller/instruction_fetch.cuh:32`
// where `instruction[0] == -1` is the SM-tape terminator).
enum Opcode : int {
    OP_END = -1,
    OP_NOP = 0,
    OP_RMS_NORM = 1,
    OP_FUSED_ADD_RMS_NORM = 2,
    OP_QKV_ROPE_CACHE = 3,
    OP_SILU_AND_MUL = 4,
    OP_GEMV = 5,
    // Generic CUTLASS GEMM dispatch. row[1] is the config id (one
    // of `CutlassConfig` below); the inner switch in
    // `run_cutlass_gemm` resolves to the right
    // `prim_mega_cutlass_configs::Gemm_*` template instantiation.
    // Phase-1 smoke-test carries one config (s4_64x128); wholesale
    // fan-out is a follow-up commit driven from
    // cutlass_standalone_gemm.cu's instantiation list.
    OP_CUTLASS_GEMM = 6,
    // CUTLASS SplitK GEMM (two-phase: GEMM + Reduction). row[1] is
    // the (tile, split_k) config id (one of `CutlassSplitKConfig`
    // below). Workspace for [split_k, M, N] float partial sums is
    // pre-allocated by the launcher and indexed into via the
    // pointer table.
    OP_CUTLASS_GEMM_SPLITK = 7,
    // Reserved for follow-up commits.
    // OP_FI_ATTN_DECODE   = 0x2000,
    // OP_FI_ATTN_PREFILL  = 0x2001,
};

// Expansion #2: per-config id. Stable ordering — encoder side
// (interpreters/prim_mega.rs) computes the same id from
// (TB_M, TB_N, TB_K, STAGES). Add new rows at the end of the list
// in cutlass_gemm_configs.cuh; never reorder existing rows or the
// encoder's emitted ids drift.
enum CutlassConfig : int {
#define DC_GEMM_ENUM_ENTRY(TB_M, TB_N, TB_K, STAGES, WARP_M, WARP_N, WARP_K) \
    CC_GEMM_##TB_M##x##TB_N##x##TB_K##_s##STAGES,
    CUTLASS_DC_GEMM_LIST(DC_GEMM_ENUM_ENTRY)
#undef DC_GEMM_ENUM_ENTRY
};

enum CutlassSplitKConfig : int {
#define DC_SPLITK_ENUM_ENTRY(TB_M, TB_N, TB_K, STAGES, WARP_M, WARP_N, WARP_K, SPLIT_K) \
    CC_SPLITK_##TB_M##x##TB_N##_s##STAGES##_sk##SPLIT_K,
    CUTLASS_DC_SPLITK_LIST(DC_SPLITK_ENUM_ENTRY)
#undef DC_SPLITK_ENUM_ENTRY
};

// Per-arm argument layouts. Each arm owns its slot interpretation;
// the encoder side (interpreters/prim_mega.rs) writes matching
// fields. Slot 0 is always opcode; slots 1..32 are op-specific.

// ── RMS_NORM ─────────────────────────────────────────────────────
// row[1]: ptr_idx out
// row[2]: ptr_idx input
// row[3]: ptr_idx weight
// row[4]: eps as float-bit pattern
// row[5]: hidden_size
// row[6]: num_rows (active CTA count for this phase)
// row[7]: smem byte offset within kernel-wide smem
template <typename T>
__device__ __forceinline__ void run_rms_norm(const int* row, void* const* pt) {
    extern __shared__ char smem[];
    T*       out    = reinterpret_cast<T*>(pt[row[1]]);
    const T* input  = reinterpret_cast<const T*>(pt[row[2]]);
    const T* weight = reinterpret_cast<const T*>(pt[row[3]]);
    float eps       = __int_as_float(row[4]);
    int   hidden    = row[5];
    int   num_rows  = row[6];
    char* op_smem   = smem + row[7];
    dc_rms_norm<T>(out, input, weight, eps, hidden, num_rows, op_smem);
}

// ── FUSED_ADD_RMS_NORM ──────────────────────────────────────────
// row[1]: ptr_idx input  (in/out — gets normalized output)
// row[2]: ptr_idx residual (in/out — accumulates input)
// row[3]: ptr_idx weight
// row[4]: eps as float-bit pattern
// row[5]: hidden_size
// row[6]: num_rows
// row[7]: smem byte offset
template <typename T>
__device__ __forceinline__ void run_fused_add_rms_norm(const int* row, void* const* pt) {
    extern __shared__ char smem[];
    T* input        = reinterpret_cast<T*>(pt[row[1]]);
    T* residual     = reinterpret_cast<T*>(pt[row[2]]);
    const T* weight = reinterpret_cast<const T*>(pt[row[3]]);
    float eps       = __int_as_float(row[4]);
    int   hidden    = row[5];
    int   num_rows  = row[6];
    char* op_smem   = smem + row[7];
    dc_fused_add_rms_norm<T>(input, residual, weight, eps, hidden, num_rows, op_smem);
}

// ── QKV_ROPE_CACHE ──────────────────────────────────────────────
// row[1]:  ptr_idx q_out
// row[2]:  ptr_idx key_cache
// row[3]:  ptr_idx value_cache
// row[4]:  ptr_idx qkv (input from QKV proj)
// row[5]:  ptr_idx positions (uint32_t*)
// row[6]:  ptr_idx cos_sin_cache
// row[7]:  ptr_idx slot_mapping (int64_t*)
// row[8]:  q_size
// row[9]:  kv_size
// row[10]: head_size
// row[11]: num_rows
// (smem unused but signature-uniform)
template <typename T>
__device__ __forceinline__ void run_qkv_rope_cache(const int* row, void* const* pt) {
    extern __shared__ char smem[];
    T* q_out                       = reinterpret_cast<T*>(pt[row[1]]);
    T* key_cache                   = reinterpret_cast<T*>(pt[row[2]]);
    T* value_cache                 = reinterpret_cast<T*>(pt[row[3]]);
    const T* qkv                   = reinterpret_cast<const T*>(pt[row[4]]);
    const uint32_t* positions      = reinterpret_cast<const uint32_t*>(pt[row[5]]);
    const T* cos_sin_cache         = reinterpret_cast<const T*>(pt[row[6]]);
    const int64_t* slot_mapping    = reinterpret_cast<const int64_t*>(pt[row[7]]);
    int q_size    = row[8];
    int kv_size   = row[9];
    int head_size = row[10];
    int num_rows  = row[11];
    dc_fused_qkv_rope_cache<T>(
        q_out, key_cache, value_cache, qkv,
        positions, cos_sin_cache, slot_mapping,
        q_size, kv_size, head_size, num_rows, smem);
}

// ── SILU_AND_MUL ────────────────────────────────────────────────
// row[1]: ptr_idx out
// row[2]: ptr_idx input ([num_rows, 2*d])
// row[3]: d (intermediate dim)
// row[4]: num_rows
template <typename T>
__device__ __forceinline__ void run_silu_and_mul(const int* row, void* const* pt) {
    T* out         = reinterpret_cast<T*>(pt[row[1]]);
    const T* input = reinterpret_cast<const T*>(pt[row[2]]);
    int d          = row[3];
    int num_rows   = row[4];
    dc_silu_and_mul<T>(out, input, d, num_rows);
}

// ── CUTLASS_GEMM ────────────────────────────────────────────────
// row[1]: ptr_idx C
// row[2]: ptr_idx A
// row[3]: ptr_idx B
// row[4]: M
// row[5]: N
// row[6]: K
// row[7]: alpha as float-bit pattern
// row[8]: beta  as float-bit pattern
// row[9]: smem byte offset within kernel-wide smem
// row[10]: CutlassConfig id (selects template instantiation)
__device__ __forceinline__ void run_cutlass_gemm(const int* row, void* const* pt) {
    extern __shared__ char smem[];
    void* C       = pt[row[1]];
    const void* A = pt[row[2]];
    const void* B = pt[row[3]];
    int M         = row[4];
    int N         = row[5];
    int K         = row[6];
    float alpha   = __int_as_float(row[7]);
    float beta    = __int_as_float(row[8]);
    char* op_smem = smem + row[9];
    int config_id = row[10];

    // Expansion #3: switch arm per config. Drives off the same
    // CUTLASS_DC_GEMM_LIST as the typedefs + enum above.
    switch (config_id) {
#define DC_GEMM_SWITCH_ARM(TB_M, TB_N, TB_K, STAGES, WARP_M, WARP_N, WARP_K) \
        case CC_GEMM_##TB_M##x##TB_N##x##TB_K##_s##STAGES:                   \
            dc_cutlass::dc_gemm<                                             \
                prim_mega_cutlass_configs::DcGemm_##TB_M##x##TB_N##x##TB_K##_s##STAGES>( \
                C, A, B, M, N, K, alpha, beta, op_smem);                     \
            break;
        CUTLASS_DC_GEMM_LIST(DC_GEMM_SWITCH_ARM)
#undef DC_GEMM_SWITCH_ARM
    }
}

// ── CUTLASS_GEMM_SPLITK ──────────────────────────────────────────
// row[1]:  ptr_idx C
// row[2]:  ptr_idx A
// row[3]:  ptr_idx B
// row[4]:  ptr_idx workspace ([split_k, M, N] float scratch)
// row[5]:  M
// row[6]:  N
// row[7]:  K
// row[8]:  alpha as float-bit pattern
// row[9]:  beta  as float-bit pattern
// row[10]: smem byte offset within kernel-wide smem
// row[11]: CutlassSplitKConfig id (selects template + split_k)
// row[12]: split_k_slices (runtime, must match config's SPLIT_K)
__device__ __forceinline__ void run_cutlass_gemm_splitk(const int* row, void* const* pt) {
    extern __shared__ char smem[];
    void* C        = pt[row[1]];
    const void* A  = pt[row[2]];
    const void* B  = pt[row[3]];
    void* workspace = pt[row[4]];
    int M          = row[5];
    int N          = row[6];
    int K          = row[7];
    float alpha    = __int_as_float(row[8]);
    float beta     = __int_as_float(row[9]);
    char* op_smem  = smem + row[10];
    int config_id  = row[11];
    int split_k    = row[12];

    switch (config_id) {
#define DC_SPLITK_SWITCH_ARM(TB_M, TB_N, TB_K, STAGES, WARP_M, WARP_N, WARP_K, SPLIT_K) \
        case CC_SPLITK_##TB_M##x##TB_N##_s##STAGES##_sk##SPLIT_K:                       \
            dc_cutlass::dc_gemm_splitk<                                                 \
                prim_mega_cutlass_configs::DcGemmSplitK_##TB_M##x##TB_N##_s##STAGES##_sk##SPLIT_K>( \
                C, A, B, M, N, K, alpha, beta, split_k, workspace, op_smem);            \
            break;
        CUTLASS_DC_SPLITK_LIST(DC_SPLITK_SWITCH_ARM)
#undef DC_SPLITK_SWITCH_ARM
    }
}

// ── GEMV (M=1) ──────────────────────────────────────────────────
// row[1]: ptr_idx out (= C, [1, N])
// row[2]: ptr_idx x (= A, [1, K] input vector)
// row[3]: ptr_idx W (= B, [N, K] row-major weight)
// row[4]: N
// row[5]: K
// row[6]: alpha as float-bit pattern
// row[7]: beta  as float-bit pattern
// row[8]: smem byte offset (reserved; CUTLASS GEMV's SharedStorage is empty)
//
// Dispatches to the CUTLASS-based `dc_cutlass::dc_gemv` (matches the
// host's `cutlass_gemv_launch` exactly so the CSV-calibrated
// "cutlass_gemv" cost row reflects megakernel runtime). M is implicit
// (=1) — out-of-contract M != 1 silently no-ops inside the template,
// mirroring the host launcher's `if (M != 1) return -10`.
__device__ __forceinline__ void run_gemv(const int* row, void* const* pt) {
    extern __shared__ char smem[];
    void* C       = pt[row[1]];
    const void* A = pt[row[2]];
    const void* B = pt[row[3]];
    int N         = row[4];
    int K         = row[5];
    float alpha   = __int_as_float(row[6]);
    float beta    = __int_as_float(row[7]);
    char* op_smem = smem + row[8];
    dc_cutlass::dc_gemv<prim_mega_cutlass_configs::DcGemvKernel_bf16_8>(
        C, A, B, /*M=*/1, N, K, alpha, beta, op_smem);
}

// Persistent kernel. Walks the tape sequentially; one row per op.
// Cooperative launch: the launcher uses cudaLaunchCooperativeKernel
// so `cg::this_grid().sync()` is a valid grid-wide barrier.
//
// The arg `pt` is the pointer table — an array of `void*` the host
// pre-fills with input/output/weight/cache pointers for the run.
__global__ void prim_mega_llama_kernel(const int* __restrict__ tape,
                                       int tape_len,
                                       void* const* __restrict__ pt) {
    cg::grid_group grid = cg::this_grid();
    for (int pc = 0; pc < tape_len; pc++) {
        const int* row = tape + pc * INSTRUCTION_WIDTH;
        const int opcode = row[0];
        if (opcode == OP_END) break;
        switch (opcode) {
            case OP_NOP:
                break;
            case OP_RMS_NORM:
                run_rms_norm<__nv_bfloat16>(row, pt);
                break;
            case OP_FUSED_ADD_RMS_NORM:
                run_fused_add_rms_norm<__nv_bfloat16>(row, pt);
                break;
            case OP_QKV_ROPE_CACHE:
                run_qkv_rope_cache<__nv_bfloat16>(row, pt);
                break;
            case OP_SILU_AND_MUL:
                run_silu_and_mul<__nv_bfloat16>(row, pt);
                break;
            case OP_GEMV:
                run_gemv(row, pt);
                break;
            case OP_CUTLASS_GEMM:
                run_cutlass_gemm(row, pt);
                break;
            case OP_CUTLASS_GEMM_SPLITK:
                run_cutlass_gemm_splitk(row, pt);
                break;
            // FlashInfer attention arms land in follow-up commits. No `default:` arm — an unknown
            // opcode here means the encoder emitted a row it shouldn't
            // have, and we want the kernel to silently skip rather
            // than corrupt memory. Once every variant has an arm we
            // switch to `__trap()` here.
        }
        grid.sync();
    }
}

extern "C" int prim_mega_llama_launch(const int* tape,
                                      int tape_len,
                                      void* const* ptr_table,
                                      int grid_x,
                                      int block_x,
                                      size_t smem_size,
                                      uint64_t stream) {
    void* args[] = {
        const_cast<void*>(reinterpret_cast<const void*>(&tape)),
        reinterpret_cast<void*>(&tape_len),
        const_cast<void*>(reinterpret_cast<const void*>(&ptr_table)),
    };
    cudaError_t err = cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(&prim_mega_llama_kernel),
        dim3(grid_x), dim3(block_x), args, smem_size,
        reinterpret_cast<cudaStream_t>(stream));
    return err == cudaSuccess ? 0 : -static_cast<int>(err);
}

}  // namespace prim_mega
