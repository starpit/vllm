/*
 * MoE top-k softmax kernels, adapted from Python vLLM's
 * csrc/moe/topk_softmax_kernels.cu (originally from TensorRT-LLM).
 *
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stripped PyTorch/ATen dependencies, added extern "C" launchers for Rust FFI.
 * Uses CUB BlockReduce for the fallback path (non-power-of-2 expert counts).
 */

#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>
#include <type_traits>

// Inline cuda_compat.h macros (CUDA-only, no ROCm)
#define WARP_SIZE 32
#define VLLM_LDG(arg) __ldg(arg)
#define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)

// Inline cub_helpers.h
#if CUB_VERSION >= 200800
  #include <cuda/std/functional>
  using CubAddOp = cuda::std::plus<>;
  using CubMaxOp = cuda::maximum<>;
#else
  using CubAddOp = cub::Sum;
  using CubMaxOp = cub::Max;
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {
namespace moe {

// Aligned array type for vectorized loads
template <typename T, int N, int Alignment = sizeof(T) * N>
struct alignas(Alignment) AlignedArray {
    T data[N];
};

template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>) {
        return value;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(value);
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(value);
    }
}

// ====================== Softmax kernel ===============================
template <int TPB, typename InputType>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const InputType* input, float* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;
    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;
    float threadData(-FLT_MAX);

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        threadData = max(val, threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, CubMaxOp());
    if (threadIdx.x == 0) float_max = maxElem;
    __syncthreads();

    threadData = 0;
    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        threadData += expf(val - float_max);
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, CubAddOp());
    if (threadIdx.x == 0) normalizing_factor = 1.f / Z;
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        const int idx = thread_row_offset + ii;
        const float val = toFloat(input[idx]);
        output[idx] = expf(val - float_max) * normalizing_factor;
    }
}

// ====================== Top-K kernel (CUB BlockReduce) ===============
template <int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(
    const float* inputs_after_softmax,
    float* output,
    int* indices,
    const int num_experts,
    const int k,
    const bool renormalize)
{
    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int block_row = blockIdx.x;
    const int thread_read_offset = blockIdx.x * num_experts;

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f;

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            // Exclude previously selected experts
            for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
                const int prior_winning_expert = indices[k * block_row + prior_k];
                if (prior_winning_expert == expert) {
                    inp_kvp = thread_kvp;
                }
            }
            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0) {
            const int expert = result_kvp.key;
            const int idx = k * block_row + k_idx;
            output[idx] = inputs_after_softmax[thread_read_offset + expert];
            indices[idx] = expert;
            if (renormalize) {
                selected_sum += inputs_after_softmax[thread_read_offset + expert];
            }
        }
        __syncthreads();
    }

    // Renormalize
    if (renormalize && threadIdx.x == 0) {
        const float denom = selected_sum > 0.f ? selected_sum : 1.f;
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * block_row + k_idx;
            output[idx] = output[idx] / denom;
        }
    }
}

// ====================== Fused topkGating (warp-level) ===============
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,
          int WARP_SIZE_PARAM, typename InputType>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE_PARAM) __global__
    void topkGating(const InputType* input, float* output, const int num_rows,
        int* indices, const int k, const bool renormalize)
{
    static_assert(std::is_same_v<InputType, float> ||
                  std::is_same_v<InputType, __nv_bfloat16> ||
                  std::is_same_v<InputType, __half>,
                  "InputType must be float, __nv_bfloat16, or __half");

    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    if constexpr (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) {
        static_assert(ELTS_PER_LDG == 1 || ELTS_PER_LDG % 2 == 0,
            "ELTS_PER_LDG must be 1 or even for 16-bit conversion");
    }

    static_assert(VPT % ELTS_PER_LDG == 0);
    static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0);
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW));
    static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM);

    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0);

    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows) return;

    const InputType* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const InputType* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    float row_chunk[VPT];

    // Load and convert to float
    if constexpr (std::is_same_v<InputType, float>) {
        using VecType = AlignedArray<float, ELTS_PER_LDG>;
        VecType* row_chunk_vec_ptr = reinterpret_cast<VecType*>(&row_chunk);
        const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
        for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
            row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
        }
    } else if constexpr (std::is_same_v<InputType, __nv_bfloat16>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__nv_bfloat16, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __bfloat1622float2(
                        *reinterpret_cast<const __nv_bfloat162*>(vec.data + jj * 2));
                }
            }
        } else {
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __nv_bfloat16* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __bfloat162float(*scalar_ptr);
            }
        }
    } else if constexpr (std::is_same_v<InputType, __half>) {
        if constexpr (ELTS_PER_LDG >= 2) {
            using VecType = AlignedArray<__half, ELTS_PER_LDG>;
            float2* row_chunk_f2 = reinterpret_cast<float2*>(row_chunk);
            const VecType* vec_thread_read_ptr = reinterpret_cast<const VecType*>(thread_read_ptr);
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                VecType vec = vec_thread_read_ptr[ii * THREADS_PER_ROW];
                int base_idx_f2 = ii * ELTS_PER_LDG / 2;
#pragma unroll
                for (int jj = 0; jj < ELTS_PER_LDG / 2; ++jj) {
                    row_chunk_f2[base_idx_f2 + jj] = __half22float2(
                        *reinterpret_cast<const __half2*>(vec.data + jj * 2));
                }
            }
        } else {
#pragma unroll
            for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
                const __half* scalar_ptr = thread_read_ptr + ii * THREADS_PER_ROW;
                row_chunk[ii] = __half2float(*scalar_ptr);
            }
        }
    }

    // Softmax: max reduce
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
        thread_max = max(thread_max, row_chunk[ii]);
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        thread_max = max(thread_max, VLLM_SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }

    // Softmax: exp and sum
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        row_sum += VLLM_SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    const float reciprocal_row_sum = 1.f / row_sum;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;
    int start_col = first_elt_read_by_thread;

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];
                if (val > max_val) {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

        // Butterfly argmax reduce across threads in the row
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            float other_max = VLLM_SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = VLLM_SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);
            if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert = other_expert;
            }
        }

        if (thread_group_idx == 0) {
            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = expert;
            if (renormalize) selected_sum += max_val;
        }

        // Clear the winning value for next iteration
        if (k_idx + 1 < k) {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
            if (thread_group_idx == thread_to_clear_in_group) {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }

    // Renormalize
    if (renormalize && thread_group_idx == 0) {
        const float denom = selected_sum > 0.f ? selected_sum : 1.f;
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * thread_row + k_idx;
            output[idx] = output[idx] / denom;
        }
    }
}

// ====================== Launcher helpers =============================
namespace detail {
template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE_PARAM, typename InputType>
struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(InputType);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 ||
                  EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0, "");
    static constexpr int VECs_PER_THREAD = MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static const int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM, int MAX_BYTES_PER_LDG, typename InputType>
void topkGatingLauncherHelper(const InputType* input, float* output, int* indices,
    const int num_rows, const int k, const bool renormalize, cudaStream_t stream)
{
    static constexpr int BYTES_PER_LDG = MIN(MAX_BYTES_PER_LDG, sizeof(InputType) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE_PARAM, WARPS_PER_TB);
    topkGating<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG, WARP_SIZE_PARAM, InputType>
        <<<num_blocks, block_dim, 0, stream>>>(
            input, output, num_rows, indices, k, renormalize);
}

#define LAUNCH_TOPK(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)                     \
    topkGatingLauncherHelper<NUM_EXPERTS, WARPS_PER_TB, 32, MAX_BYTES,        \
                             InputType>(                                       \
        gating_output, topk_weights, topk_indices,                             \
        num_tokens, topk, renormalize, stream);

template <typename InputType>
void topkGatingKernelLauncher(
    const InputType* gating_output,
    float* topk_weights,
    int* topk_indices,
    float* workspace,
    const int num_tokens,
    const int num_experts,
    const int topk,
    const bool renormalize,
    cudaStream_t stream)
{
    static constexpr int WARPS_PER_TB = 4;
    static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
    static constexpr int BYTES_PER_LDG_MULTIPLE_64 =
        (std::is_same_v<InputType, __nv_bfloat16> || std::is_same_v<InputType, __half>) ? 4 : 8;

    switch (num_experts) {
        case 1:   LAUNCH_TOPK(1, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
        case 2:   LAUNCH_TOPK(2, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
        case 4:   LAUNCH_TOPK(4, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
        case 8:   LAUNCH_TOPK(8, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
        case 16:  LAUNCH_TOPK(16, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
        case 32:  LAUNCH_TOPK(32, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
        case 64:  LAUNCH_TOPK(64, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
        case 128: LAUNCH_TOPK(128, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
        case 256: LAUNCH_TOPK(256, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
        case 512: LAUNCH_TOPK(512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
        case 192: LAUNCH_TOPK(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
        case 320: LAUNCH_TOPK(320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
        case 384: LAUNCH_TOPK(384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
        case 448: LAUNCH_TOPK(448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
        case 576: LAUNCH_TOPK(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
        default: {
            // Fallback: separate softmax + topK using CUB BlockReduce
            static constexpr int TPB = 256;
            moeSoftmax<TPB, InputType><<<num_tokens, TPB, 0, stream>>>(
                gating_output, workspace, num_experts);
            moeTopK<TPB><<<num_tokens, TPB, 0, stream>>>(
                workspace, topk_weights, topk_indices,
                num_experts, topk, renormalize);
        }
    }
}

#undef LAUNCH_TOPK

} // namespace moe
} // namespace vllm

// =====================================================================
// extern "C" launchers for Rust FFI
// =====================================================================

extern "C" void topk_softmax_f32(
    float* topk_weights,
    int* topk_ids,
    float* workspace,
    const float* gating_output,
    int num_tokens,
    int num_experts,
    int topk,
    int renormalize)
{
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    vllm::moe::topkGatingKernelLauncher<float>(
        gating_output, topk_weights, topk_ids, workspace,
        num_tokens, num_experts, topk, renormalize != 0, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

extern "C" void topk_softmax_bf16(
    float* topk_weights,
    int* topk_ids,
    float* workspace,
    const void* gating_output,
    int num_tokens,
    int num_experts,
    int topk,
    int renormalize)
{
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    vllm::moe::topkGatingKernelLauncher<__nv_bfloat16>(
        reinterpret_cast<const __nv_bfloat16*>(gating_output),
        topk_weights, topk_ids, workspace,
        num_tokens, num_experts, topk, renormalize != 0, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

extern "C" void topk_softmax_f16(
    float* topk_weights,
    int* topk_ids,
    float* workspace,
    const void* gating_output,
    int num_tokens,
    int num_experts,
    int topk,
    int renormalize)
{
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    vllm::moe::topkGatingKernelLauncher<__half>(
        reinterpret_cast<const __half*>(gating_output),
        topk_weights, topk_ids, workspace,
        num_tokens, num_experts, topk, renormalize != 0, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}
