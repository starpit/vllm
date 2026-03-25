/*
 * Dense FP8 block-scaled GEMM kernel (SM80+).
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Computes C = A × B^T  where:
 *   A: BF16     [M, K] (activations)
 *   B: FP8 E4M3 [N, K] (weights, stored row-major as [N, K])
 *   C: BF16     [M, N] (output)
 *
 * Per-block weight scales are applied during the FP8→BF16 dequant step in
 * shared memory, so WMMA accumulates correctly-scaled BF16 values.
 * No activation scales are applied (caller handles that externally).
 *
 * Derived from fused_moe_gemm_fp8_block_dequant_wmma by removing all MoE
 * dispatch logic (expert_ids, sorted_token_ids, topk_weights).
 *
 * Tile sizes: BLOCK_M=variable (16/32/64/128), BLOCK_N=128, BLOCK_K=32
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

#define FP8BG_BLOCK_N 128
#define FP8BG_BLOCK_K 32
#define FP8BG_WMMA_M 16
#define FP8BG_WMMA_N 16
#define FP8BG_WMMA_K 16
#define FP8BG_THREADS 256
#define FP8BG_WARPS (FP8BG_THREADS / 32)
#define FP8BG_WARP_TILES_N (FP8BG_BLOCK_N / FP8BG_WMMA_N)  // 8
#define FP8BG_GROUP_SIZE_M 8

namespace vllm {

// Dense FP8 block-scaled GEMM: C[M,N] = A[M,K] * B[N,K]^T
// with per-block weight scales w_scales[ceil(N/bn), ceil(K/bk)].
template <int BLOCK_M_T>
__global__ void __launch_bounds__(FP8BG_THREADS)
fp8_block_scaled_gemm_kernel(
    __nv_bfloat16* __restrict__ output,    // [M, N]
    const __nv_bfloat16* __restrict__ input,  // BF16 [M, K]
    const uint8_t* __restrict__ weight,    // FP8 E4M3 [N, K]
    const float* __restrict__ w_scales,    // [ceil(N/bn), ceil(K/bk)]
    int32_t M,
    int32_t N,
    int32_t K,
    int32_t block_n,
    int32_t block_k,
    int32_t scale_stride_n)   // ceil(K/block_k)
{
    constexpr int WARPS_PER_M = BLOCK_M_T / FP8BG_WMMA_M;
    constexpr int WARPS_PER_N = FP8BG_WARPS / WARPS_PER_M;
    constexpr int N_TILES_PER_WARP = FP8BG_WARP_TILES_N / WARPS_PER_N;

    const int32_t num_n_blocks = (N + FP8BG_BLOCK_N - 1) / FP8BG_BLOCK_N;
    const int32_t num_m_blocks = (M + BLOCK_M_T - 1) / BLOCK_M_T;

    // GROUP_SIZE_M swizzle for L2 cache locality.
    const int32_t num_pid_in_group = FP8BG_GROUP_SIZE_M * num_n_blocks;
    const int32_t group_id = blockIdx.x / num_pid_in_group;
    const int32_t first_pid_m = group_id * FP8BG_GROUP_SIZE_M;
    const int32_t group_size_m = min(num_m_blocks - first_pid_m, FP8BG_GROUP_SIZE_M);
    if (group_size_m <= 0) return;
    const int32_t pid_m = first_pid_m + (blockIdx.x % group_size_m);
    const int32_t pid_n = (blockIdx.x % num_pid_in_group) / group_size_m;

    if (pid_m >= num_m_blocks || pid_n >= num_n_blocks) return;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int warp_m = warp_id / WARPS_PER_N;
    const int warp_n_start = (warp_id % WARPS_PER_N) * N_TILES_PER_WARP;

    __shared__ __nv_bfloat16 smem_a[BLOCK_M_T][FP8BG_BLOCK_K];
    __shared__ __nv_bfloat16 smem_b[FP8BG_BLOCK_K][FP8BG_BLOCK_N];

    wmma::fragment<wmma::accumulator, FP8BG_WMMA_M, FP8BG_WMMA_N, FP8BG_WMMA_K, float> acc[N_TILES_PER_WARP];
    #pragma unroll
    for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
        wmma::fill_fragment(acc[wn], 0.0f);
    }

    for (int32_t k_start = 0; k_start < K; k_start += FP8BG_BLOCK_K) {
        // Load A tile: BF16 → smem (activations are already BF16).
        for (int32_t idx = threadIdx.x; idx < BLOCK_M_T * FP8BG_BLOCK_K; idx += FP8BG_THREADS) {
            int32_t m_local = idx / FP8BG_BLOCK_K;
            int32_t k_local = idx % FP8BG_BLOCK_K;
            int32_t global_m = pid_m * BLOCK_M_T + m_local;
            int32_t global_k = k_start + k_local;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_m < M && global_k < K) {
                val = input[(int64_t)global_m * K + global_k];
            }
            smem_a[m_local][k_local] = val;
        }

        // Load B tile: FP8 → BF16 with per-block weight scale applied.
        for (int32_t idx = threadIdx.x; idx < FP8BG_BLOCK_K * FP8BG_BLOCK_N; idx += FP8BG_THREADS) {
            int32_t k_local = idx / FP8BG_BLOCK_N;
            int32_t n_local = idx % FP8BG_BLOCK_N;
            int32_t global_k = k_start + k_local;
            int32_t global_n = pid_n * FP8BG_BLOCK_N + n_local;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_k < K && global_n < N) {
                __nv_fp8_e4m3 fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(
                    &weight[(int64_t)global_n * K + global_k]);
                float raw = float(fp8);
                int32_t sn = global_n / block_n;
                int32_t sk = global_k / block_k;
                float scale = w_scales[sn * scale_stride_n + sk];
                val = __float2bfloat16(raw * scale);
            }
            smem_b[k_local][n_local] = val;
        }

        __syncthreads();

        // WMMA compute.
        #pragma unroll
        for (int32_t ki = 0; ki < FP8BG_BLOCK_K; ki += FP8BG_WMMA_K) {
            wmma::fragment<wmma::matrix_a, FP8BG_WMMA_M, FP8BG_WMMA_N, FP8BG_WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &smem_a[warp_m * FP8BG_WMMA_M][ki], FP8BG_BLOCK_K);

            #pragma unroll
            for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
                wmma::fragment<wmma::matrix_b, FP8BG_WMMA_M, FP8BG_WMMA_N, FP8BG_WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &smem_b[ki][(warp_n_start + wn) * FP8BG_WMMA_N], FP8BG_BLOCK_N);
                wmma::mma_sync(acc[wn], a_frag, b_frag, acc[wn]);
            }
        }

        __syncthreads();
    }

    // Epilogue: write f32 accumulators → BF16 output.
    __shared__ float warp_staging[FP8BG_WARPS][FP8BG_WMMA_M * FP8BG_WMMA_N];

    #pragma unroll
    for (int wn = 0; wn < N_TILES_PER_WARP; ++wn) {
        wmma::store_matrix_sync(
            &warp_staging[warp_id][0], acc[wn], FP8BG_WMMA_N, wmma::mem_row_major);
        __syncwarp();

        for (int i = lane_id; i < FP8BG_WMMA_M * FP8BG_WMMA_N; i += 32) {
            int32_t m_local = warp_m * FP8BG_WMMA_M + i / FP8BG_WMMA_N;
            int32_t n_local = (warp_n_start + wn) * FP8BG_WMMA_N + i % FP8BG_WMMA_N;
            int32_t global_m = pid_m * BLOCK_M_T + m_local;
            int32_t global_n = pid_n * FP8BG_BLOCK_N + n_local;

            if (global_m < M && global_n < N) {
                float val = warp_staging[warp_id][i];
                output[(int64_t)global_m * N + global_n] = __float2bfloat16(val);
            }
        }
    }
}

} // namespace vllm

// =====================================================================
// extern "C" launcher
// =====================================================================

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

extern "C" void fp8_block_scaled_gemm(
    void* output,           // BF16 [M, N]
    const void* input,      // BF16 [M, K]
    const void* weight,     // FP8  [N, K]
    const float* w_scales,  // f32  [ceil(N/bn), ceil(K/bk)]
    int M,
    int N,
    int K,
    int block_m,            // tile height (16/32/64/128)
    int block_n,            // quant block size along N
    int block_k,            // quant block size along K
    int scale_stride_n,     // ceil(K/block_k)
    cudaStream_t stream)
{
    int num_m_blocks = CEILDIV(M, block_m);
    int num_n_blocks = CEILDIV(N, FP8BG_BLOCK_N);
    int grid = num_m_blocks * num_n_blocks;
    if (grid == 0) return;

    #define LAUNCH_FP8BG(BM) \
        vllm::fp8_block_scaled_gemm_kernel<BM> \
            <<<grid, FP8BG_THREADS, 0, stream>>>( \
                reinterpret_cast<__nv_bfloat16*>(output), \
                reinterpret_cast<const __nv_bfloat16*>(input), \
                reinterpret_cast<const uint8_t*>(weight), \
                w_scales, M, N, K, block_n, block_k, scale_stride_n)

    switch (block_m) {
        case 16:  LAUNCH_FP8BG(16);  break;
        case 32:  LAUNCH_FP8BG(32);  break;
        case 64:  LAUNCH_FP8BG(64);  break;
        case 128: LAUNCH_FP8BG(128); break;
        default:  LAUNCH_FP8BG(128); break;
    }
    #undef LAUNCH_FP8BG
}
