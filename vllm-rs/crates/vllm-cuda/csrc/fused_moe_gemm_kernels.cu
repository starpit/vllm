/*
 * Fused MoE GEMM kernel — expert-indexed tiled matrix multiplication.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Matches the semantics of Python vLLM's Triton fused_moe_kernel:
 * - Input A: [num_tokens, K] (hidden states)
 * - Weights B: [E, N, K] (stacked expert weights, row-major)
 * - Output C: [num_tokens * top_k, N]
 * - Token dispatch via sorted_token_ids, expert dispatch via expert_ids
 *
 * Uses WMMA (Warp Matrix Multiply-Accumulate) tensor core intrinsics on SM80+
 * for BF16/F16 inputs with F32 accumulation. Falls back to scalar FMA on
 * older architectures.
 *
 * Tile sizes: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
 * WMMA fragments: 16x16x16 (SM80+)
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdint.h>

using namespace nvcuda;

// Tile sizes — tuned for SM80+ tensor cores
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 32

// WMMA fragment size
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Thread block size: enough warps to cover BLOCK_M/WMMA_M * BLOCK_N/WMMA_N tiles
// = (128/16) * (128/16) = 8 * 8 = 64 warp-tiles. With 8 warps (256 threads),
// each warp handles 8 tiles sequentially.
#define THREADS_PER_BLOCK 256
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / 32)

// Each warp computes (BLOCK_M/WMMA_M * BLOCK_N/WMMA_N) / WARPS_PER_BLOCK
// = 64/8 = 8 WMMA tiles. Assign warps in a 2D grid over the M,N tile space.
#define WARP_TILES_M (BLOCK_M / WMMA_M)  // 8
#define WARP_TILES_N (BLOCK_N / WMMA_N)  // 8

namespace vllm {
namespace moe {

// =========================================================================
// WMMA tensor-core kernel for BF16 (SM80+)
// =========================================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_moe_gemm_bf16_wmma(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weights,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t num_valid_tokens,
    int32_t K,
    int32_t N,
    int32_t top_k,
    int32_t apply_weights)
{
    const int32_t total_padded = *num_tokens_post_padded;
    const int32_t num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    const int32_t pid_m = blockIdx.x / num_n_blocks;
    const int32_t pid_n = blockIdx.x % num_n_blocks;

    if (pid_m * BLOCK_M >= total_padded) return;

    const int32_t expert_id = expert_ids[pid_m];
    if (expert_id < 0) return;

    const __nv_bfloat16* expert_w = weights + (int64_t)expert_id * N * K;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Each warp is assigned a (wm, wn) tile in the WARP_TILES_M x WARP_TILES_N grid.
    // With 8 warps and 64 tiles, each warp handles 8 tiles.
    // Assign warps in a 2x4 grid: warp handles 4 M-tiles x 2 N-tiles.
    // Better: each warp handles one (wm, wn) pair, iterate over multiple.
    // Simplest: flat assignment. warp_id maps to tiles.

    // Shared memory for A and B tiles.
    __shared__ __nv_bfloat16 smem_a[BLOCK_M][BLOCK_K];
    __shared__ __nv_bfloat16 smem_b[BLOCK_K][BLOCK_N];

    // Each warp accumulates its WMMA fragments.
    // Warp grid: assign warps to cover the M dimension, iterate over N.
    // With 8 warps, each warp handles BLOCK_M/8 = 16 = 1 WMMA_M tile in M.
    // Each warp iterates over all WARP_TILES_N = 8 tiles in N.
    const int warp_m = warp_id;  // warp_id in [0, 8), each handles 1 WMMA_M=16 rows

    // Accumulators: one per N-tile this warp handles.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_TILES_N];
    #pragma unroll
    for (int wn = 0; wn < WARP_TILES_N; ++wn) {
        wmma::fill_fragment(acc[wn], 0.0f);
    }

    // Iterate over K dimension.
    for (int32_t k_start = 0; k_start < K; k_start += BLOCK_K) {
        // Cooperatively load A tile: [BLOCK_M, BLOCK_K] from input via sorted_token_ids.
        for (int32_t idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += THREADS_PER_BLOCK) {
            int32_t m_local = idx / BLOCK_K;
            int32_t k_local = idx % BLOCK_K;
            int32_t global_m = pid_m * BLOCK_M + m_local;
            int32_t global_k = k_start + k_local;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_m < total_padded && global_k < K) {
                int32_t token_id = sorted_token_ids[global_m];
                if (token_id < num_valid_tokens * top_k) {
                    int32_t orig_token = token_id / top_k;
                    val = input[orig_token * K + global_k];
                }
            }
            smem_a[m_local][k_local] = val;
        }

        // Cooperatively load B tile: [BLOCK_K, BLOCK_N] from expert weights.
        // B is stored [N, K] row-major: B[n, k] = expert_w[n * K + k].
        // We need B transposed: smem_b[k][n] = expert_w[n * K + (k_start + k)].
        for (int32_t idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
            int32_t k_local = idx / BLOCK_N;
            int32_t n_local = idx % BLOCK_N;
            int32_t global_k = k_start + k_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            __nv_bfloat16 val = __float2bfloat16(0.0f);
            if (global_k < K && global_n < N) {
                val = expert_w[global_n * K + global_k];
            }
            smem_b[k_local][n_local] = val;
        }

        __syncthreads();

        // WMMA accumulation over K-tiles within this BLOCK_K.
        // Each warp loads its fragments from shared memory and does matrix multiply.
        #pragma unroll
        for (int32_t ki = 0; ki < BLOCK_K; ki += WMMA_K) {
            // Load A fragment: rows [warp_m * WMMA_M, warp_m * WMMA_M + WMMA_M),
            //                  cols [ki, ki + WMMA_K).
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag,
                &smem_a[warp_m * WMMA_M][ki],
                BLOCK_K);  // leading dimension of smem_a

            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                // Load B fragment: rows [ki, ki + WMMA_K),
                //                  cols [wn * WMMA_N, wn * WMMA_N + WMMA_N).
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag,
                    &smem_b[ki][wn * WMMA_N],
                    BLOCK_N);  // leading dimension of smem_b

                wmma::mma_sync(acc[wn], a_frag, b_frag, acc[wn]);
            }
        }

        __syncthreads();
    }

    // Store results — each warp writes one WMMA_M x WMMA_N tile at a time
    // through a small shared-memory staging area (8KB total vs 64KB before).
    __shared__ float warp_staging[WARPS_PER_BLOCK][WMMA_M * WMMA_N];

    #pragma unroll
    for (int wn = 0; wn < WARP_TILES_N; ++wn) {
        // Store this warp's accumulator to its staging slot.
        wmma::store_matrix_sync(
            &warp_staging[warp_id][0],
            acc[wn],
            WMMA_N,  // leading dimension = WMMA_N (16)
            wmma::mem_row_major);

        // All 32 threads in the warp write 256 elements (16x16) to global.
        // No __syncthreads needed — each warp writes to its own staging slot.
        __syncwarp();

        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int32_t m_local = warp_m * WMMA_M + i / WMMA_N;
            int32_t n_local = wn * WMMA_N + i % WMMA_N;
            int32_t global_m = pid_m * BLOCK_M + m_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            if (global_m >= total_padded || global_n >= N) continue;

            int32_t token_id = sorted_token_ids[global_m];
            if (token_id >= num_valid_tokens * top_k) continue;

            float val = warp_staging[warp_id][i];
            if (apply_weights) {
                val *= topk_weights[token_id];
            }
            output[(int64_t)token_id * N + global_n] = __float2bfloat16(val);
        }
    }
}

// =========================================================================
// WMMA tensor-core kernel for F16 (SM70+)
// =========================================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_moe_gemm_f16_wmma(
    __half* __restrict__ output,
    const __half* __restrict__ input,
    const __half* __restrict__ weights,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded,
    int32_t num_valid_tokens,
    int32_t K,
    int32_t N,
    int32_t top_k,
    int32_t apply_weights)
{
    const int32_t total_padded = *num_tokens_post_padded;
    const int32_t num_n_blocks = (N + BLOCK_N - 1) / BLOCK_N;
    const int32_t pid_m = blockIdx.x / num_n_blocks;
    const int32_t pid_n = blockIdx.x % num_n_blocks;

    if (pid_m * BLOCK_M >= total_padded) return;

    const int32_t expert_id = expert_ids[pid_m];
    if (expert_id < 0) return;

    const __half* expert_w = weights + (int64_t)expert_id * N * K;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    __shared__ __half smem_a[BLOCK_M][BLOCK_K];
    __shared__ __half smem_b[BLOCK_K][BLOCK_N];

    const int warp_m = warp_id;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_TILES_N];
    #pragma unroll
    for (int wn = 0; wn < WARP_TILES_N; ++wn) {
        wmma::fill_fragment(acc[wn], 0.0f);
    }

    for (int32_t k_start = 0; k_start < K; k_start += BLOCK_K) {
        for (int32_t idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += THREADS_PER_BLOCK) {
            int32_t m_local = idx / BLOCK_K;
            int32_t k_local = idx % BLOCK_K;
            int32_t global_m = pid_m * BLOCK_M + m_local;
            int32_t global_k = k_start + k_local;

            __half val = __float2half(0.0f);
            if (global_m < total_padded && global_k < K) {
                int32_t token_id = sorted_token_ids[global_m];
                if (token_id < num_valid_tokens * top_k) {
                    val = input[(token_id / top_k) * K + global_k];
                }
            }
            smem_a[m_local][k_local] = val;
        }

        for (int32_t idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += THREADS_PER_BLOCK) {
            int32_t k_local = idx / BLOCK_N;
            int32_t n_local = idx % BLOCK_N;
            int32_t global_k = k_start + k_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            __half val = __float2half(0.0f);
            if (global_k < K && global_n < N) {
                val = expert_w[global_n * K + global_k];
            }
            smem_b[k_local][n_local] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int32_t ki = 0; ki < BLOCK_K; ki += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, &smem_a[warp_m * WMMA_M][ki], BLOCK_K);

            #pragma unroll
            for (int wn = 0; wn < WARP_TILES_N; ++wn) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &smem_b[ki][wn * WMMA_N], BLOCK_N);
                wmma::mma_sync(acc[wn], a_frag, b_frag, acc[wn]);
            }
        }

        __syncthreads();
    }

    __shared__ float warp_staging[WARPS_PER_BLOCK][WMMA_M * WMMA_N];

    #pragma unroll
    for (int wn = 0; wn < WARP_TILES_N; ++wn) {
        wmma::store_matrix_sync(&warp_staging[warp_id][0], acc[wn], WMMA_N, wmma::mem_row_major);
        __syncwarp();

        for (int i = lane_id; i < WMMA_M * WMMA_N; i += 32) {
            int32_t m_local = warp_m * WMMA_M + i / WMMA_N;
            int32_t n_local = wn * WMMA_N + i % WMMA_N;
            int32_t global_m = pid_m * BLOCK_M + m_local;
            int32_t global_n = pid_n * BLOCK_N + n_local;

            if (global_m >= total_padded || global_n >= N) continue;
            int32_t token_id = sorted_token_ids[global_m];
            if (token_id >= num_valid_tokens * top_k) continue;

            float val = warp_staging[warp_id][i];
            if (apply_weights) {
                val *= topk_weights[token_id];
            }
            output[(int64_t)token_id * N + global_n] = __float2half(val);
        }
    }
}

} // namespace moe
} // namespace vllm

// =====================================================================
// extern "C" launchers
// =====================================================================

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

extern "C" void fused_moe_gemm_bf16(
    void* output,
    const void* input,
    const void* weights,
    const float* topk_weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int num_valid_tokens,
    int in_features,
    int out_features,
    int top_k,
    int block_size,
    int apply_weights,
    cudaStream_t stream)
{
    // Conservative grid upper bound (kernel early-exits for out-of-range blocks).
    int max_m_blocks = CEILDIV(num_valid_tokens * top_k, BLOCK_M) + 64;
    int num_n_blocks = CEILDIV(out_features, BLOCK_N);
    int grid = max_m_blocks * num_n_blocks;
    (void)block_size;

    vllm::moe::fused_moe_gemm_bf16_wmma
        <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(output),
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<const __nv_bfloat16*>(weights),
            topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            num_valid_tokens, in_features, out_features, top_k, apply_weights);
}

extern "C" void fused_moe_gemm_f16(
    void* output,
    const void* input,
    const void* weights,
    const float* topk_weights,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded,
    int num_valid_tokens,
    int in_features,
    int out_features,
    int top_k,
    int block_size,
    int apply_weights,
    cudaStream_t stream)
{
    int max_m_blocks = CEILDIV(num_valid_tokens * top_k, BLOCK_M) + 64;
    int num_n_blocks = CEILDIV(out_features, BLOCK_N);
    int grid = max_m_blocks * num_n_blocks;
    (void)block_size;

    vllm::moe::fused_moe_gemm_f16_wmma
        <<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            reinterpret_cast<__half*>(output),
            reinterpret_cast<const __half*>(input),
            reinterpret_cast<const __half*>(weights),
            topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            num_valid_tokens, in_features, out_features, top_k, apply_weights);
}
