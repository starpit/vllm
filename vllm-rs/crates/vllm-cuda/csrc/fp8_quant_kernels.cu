// SPDX-License-Identifier: Apache-2.0
// FP8 E4M3 quantization kernels — matches Python vLLM's scaled_fp8_quant.
//
// Three modes:
// 1. Dynamic per-token: BF16 [num_tokens, hidden] → FP8 [num_tokens, hidden] + scales [num_tokens]
//    Computes absmax per row, derives scale = absmax / FP8_E4M3_MAX (448.0).
// 2. Static: BF16 [num_tokens, hidden] + input_scale (scalar) → FP8 [num_tokens, hidden]
//    Uses pre-calibrated scale directly.
// 3. Online weight quant: BF16 [N, K] → FP8 [N, K] + scale (scalar)
//    Per-tensor absmax across entire weight matrix.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cfloat>

// FP8 E4M3 max representable value
static constexpr float FP8_E4M3_MAX = 448.0f;

// Atomic max for floats using integer CAS trick.
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    if (val >= 0.0f) {
        atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(val));
    }
}

// ---------------------------------------------------------------------------
// Kernel 1: Dynamic per-token FP8 quantization (BF16 input)
//
// One block per token row. Block computes row absmax via shared-mem reduction,
// then quantizes the entire row with vectorized loads/stores.
// ---------------------------------------------------------------------------

__global__ void scaled_fp8_quant_dynamic_bf16_kernel(
    const uint16_t* __restrict__ input,   // [num_tokens, hidden_dim] BF16
    uint8_t* __restrict__ output,         // [num_tokens, hidden_dim] FP8 E4M3
    float* __restrict__ scales,           // [num_tokens] f32
    int hidden_dim)
{
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = token_idx * hidden_dim;

    extern __shared__ float smem[];

    // Phase 1: compute per-row absmax
    float local_max = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(&input[row_offset + i]));
        float abs_val = fabsf(val);
        if (abs_val > local_max) local_max = abs_val;
    }
    smem[tid] = local_max;
    __syncthreads();

    // Block reduction for absmax
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) {
            smem[tid] = smem[tid + s];
        }
        __syncthreads();
    }

    // Derive scale: scale = absmax / FP8_MAX. If absmax=0, scale=1.0 (safe).
    float row_absmax = smem[0];
    float scale = (row_absmax > 0.0f) ? (row_absmax / FP8_E4M3_MAX) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Store scale for this token
    if (tid == 0) {
        scales[token_idx] = scale;
    }

    // Phase 2: quantize the row with vectorized 8-wide stores
    // Process 8 elements at a time for coalesced memory access
    const int vec_elems = 8;
    const int vec_iters = hidden_dim / vec_elems;
    const int remainder_start = vec_iters * vec_elems;

    for (int vi = tid; vi < vec_iters; vi += blockDim.x) {
        int base = row_offset + vi * vec_elems;

        // Load 8 BF16 values (= 16 bytes = one uint4)
        uint4 in_vec = *reinterpret_cast<const uint4*>(&input[base]);
        const uint16_t* vals = reinterpret_cast<const uint16_t*>(&in_vec);

        // Quantize 8 values to FP8
        uint8_t fp8_bytes[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float fval = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(&vals[j]));
            fval *= inv_scale;
            __nv_fp8_e4m3 fp8(fval);
            fp8_bytes[j] = *reinterpret_cast<uint8_t*>(&fp8);
        }

        // Store 8 FP8 bytes (= 8 bytes = two uint32)
        uint2 out_vec;
        out_vec.x = *reinterpret_cast<uint32_t*>(&fp8_bytes[0]);
        out_vec.y = *reinterpret_cast<uint32_t*>(&fp8_bytes[4]);
        *reinterpret_cast<uint2*>(&output[row_offset + vi * vec_elems]) = out_vec;
    }

    // Handle remainder elements
    for (int i = remainder_start + tid; i < hidden_dim; i += blockDim.x) {
        float fval = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(&input[row_offset + i]));
        fval *= inv_scale;
        __nv_fp8_e4m3 fp8(fval);
        output[row_offset + i] = *reinterpret_cast<uint8_t*>(&fp8);
    }
}

// F16 variant
__global__ void scaled_fp8_quant_dynamic_f16_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int hidden_dim)
{
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int row_offset = token_idx * hidden_dim;

    extern __shared__ float smem[];

    float local_max = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __half2float(*reinterpret_cast<const __half*>(&input[row_offset + i]));
        float abs_val = fabsf(val);
        if (abs_val > local_max) local_max = abs_val;
    }
    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) {
            smem[tid] = smem[tid + s];
        }
        __syncthreads();
    }

    float row_absmax = smem[0];
    float scale = (row_absmax > 0.0f) ? (row_absmax / FP8_E4M3_MAX) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0) {
        scales[token_idx] = scale;
    }

    const int vec_elems = 8;
    const int vec_iters = hidden_dim / vec_elems;
    const int remainder_start = vec_iters * vec_elems;

    for (int vi = tid; vi < vec_iters; vi += blockDim.x) {
        int base = row_offset + vi * vec_elems;
        uint4 in_vec = *reinterpret_cast<const uint4*>(&input[base]);
        const uint16_t* vals = reinterpret_cast<const uint16_t*>(&in_vec);

        uint8_t fp8_bytes[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float fval = __half2float(*reinterpret_cast<const __half*>(&vals[j]));
            fval *= inv_scale;
            __nv_fp8_e4m3 fp8(fval);
            fp8_bytes[j] = *reinterpret_cast<uint8_t*>(&fp8);
        }

        uint2 out_vec;
        out_vec.x = *reinterpret_cast<uint32_t*>(&fp8_bytes[0]);
        out_vec.y = *reinterpret_cast<uint32_t*>(&fp8_bytes[4]);
        *reinterpret_cast<uint2*>(&output[row_offset + vi * vec_elems]) = out_vec;
    }

    for (int i = remainder_start + tid; i < hidden_dim; i += blockDim.x) {
        float fval = __half2float(*reinterpret_cast<const __half*>(&input[row_offset + i]));
        fval *= inv_scale;
        __nv_fp8_e4m3 fp8(fval);
        output[row_offset + i] = *reinterpret_cast<uint8_t*>(&fp8);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Static FP8 quantization (pre-calibrated scale)
// ---------------------------------------------------------------------------

__global__ void scaled_fp8_quant_static_bf16_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const float* __restrict__ scale_ptr,  // GPU scalar
    int num_elements)
{
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized: 8 BF16 at a time
    const int vec_elems = 8;
    const int total_vecs = num_elements / vec_elems;
    const int remainder_start = total_vecs * vec_elems;

    for (int vi = tid; vi < total_vecs; vi += blockDim.x * gridDim.x) {
        int base = vi * vec_elems;
        uint4 in_vec = *reinterpret_cast<const uint4*>(&input[base]);
        const uint16_t* vals = reinterpret_cast<const uint16_t*>(&in_vec);

        uint8_t fp8_bytes[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float fval = __bfloat162float(
                *reinterpret_cast<const __nv_bfloat16*>(&vals[j]));
            fval *= inv_scale;
            __nv_fp8_e4m3 fp8(fval);
            fp8_bytes[j] = *reinterpret_cast<uint8_t*>(&fp8);
        }

        uint2 out_vec;
        out_vec.x = *reinterpret_cast<uint32_t*>(&fp8_bytes[0]);
        out_vec.y = *reinterpret_cast<uint32_t*>(&fp8_bytes[4]);
        *reinterpret_cast<uint2*>(&output[base]) = out_vec;
    }

    // Remainder
    for (int i = remainder_start + tid; i < num_elements; i += blockDim.x * gridDim.x) {
        float fval = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(&input[i]));
        fval *= inv_scale;
        __nv_fp8_e4m3 fp8(fval);
        output[i] = *reinterpret_cast<uint8_t*>(&fp8);
    }
}

__global__ void scaled_fp8_quant_static_f16_kernel(
    const uint16_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const float* __restrict__ scale_ptr,
    int num_elements)
{
    float scale = *scale_ptr;
    float inv_scale = 1.0f / scale;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int vec_elems = 8;
    const int total_vecs = num_elements / vec_elems;
    const int remainder_start = total_vecs * vec_elems;

    for (int vi = tid; vi < total_vecs; vi += blockDim.x * gridDim.x) {
        int base = vi * vec_elems;
        uint4 in_vec = *reinterpret_cast<const uint4*>(&input[base]);
        const uint16_t* vals = reinterpret_cast<const uint16_t*>(&in_vec);

        uint8_t fp8_bytes[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            float fval = __half2float(*reinterpret_cast<const __half*>(&vals[j]));
            fval *= inv_scale;
            __nv_fp8_e4m3 fp8(fval);
            fp8_bytes[j] = *reinterpret_cast<uint8_t*>(&fp8);
        }

        uint2 out_vec;
        out_vec.x = *reinterpret_cast<uint32_t*>(&fp8_bytes[0]);
        out_vec.y = *reinterpret_cast<uint32_t*>(&fp8_bytes[4]);
        *reinterpret_cast<uint2*>(&output[base]) = out_vec;
    }

    for (int i = remainder_start + tid; i < num_elements; i += blockDim.x * gridDim.x) {
        float fval = __half2float(*reinterpret_cast<const __half*>(&input[i]));
        fval *= inv_scale;
        __nv_fp8_e4m3 fp8(fval);
        output[i] = *reinterpret_cast<uint8_t*>(&fp8);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: Online weight quantization (per-tensor absmax → scale → quantize)
//
// Two-phase: first compute global absmax via reduction, then quantize.
// ---------------------------------------------------------------------------

__global__ void weight_absmax_bf16_kernel(
    const uint16_t* __restrict__ tensor,
    int num_elements,
    float* __restrict__ abs_max_out)
{
    __shared__ float smem[256];
    const int tid = threadIdx.x;
    float local_max = 0.0f;

    for (int i = blockIdx.x * blockDim.x + tid; i < num_elements;
         i += blockDim.x * gridDim.x) {
        float val = __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16*>(&tensor[i]));
        float abs_val = fabsf(val);
        if (abs_val > local_max) local_max = abs_val;
    }

    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) {
            smem[tid] = smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(abs_max_out, smem[0]);
    }
}

__global__ void finalize_fp8_weight_scale_kernel(
    float* __restrict__ abs_max_and_scale)
{
    float abs_max = *abs_max_and_scale;
    float scale = (abs_max > 0.0f) ? (abs_max / FP8_E4M3_MAX) : 1.0f;
    *abs_max_and_scale = scale;
}

// ---------------------------------------------------------------------------
// C entry points
// ---------------------------------------------------------------------------

extern "C" {

// Dynamic per-token FP8 quantization: BF16 → FP8 + per-token scales
void scaled_fp8_quant_dynamic_bf16(
    const uint16_t* input,     // [num_tokens, hidden_dim] BF16
    uint8_t* output,           // [num_tokens, hidden_dim] FP8
    float* scales,             // [num_tokens] f32
    int num_tokens,
    int hidden_dim,
    cudaStream_t stream)
{
    const int threads = 256;
    const int smem_size = threads * sizeof(float);
    scaled_fp8_quant_dynamic_bf16_kernel<<<num_tokens, threads, smem_size, stream>>>(
        input, output, scales, hidden_dim);
}

// Dynamic per-token FP8 quantization: F16 → FP8 + per-token scales
void scaled_fp8_quant_dynamic_f16(
    const uint16_t* input,
    uint8_t* output,
    float* scales,
    int num_tokens,
    int hidden_dim,
    cudaStream_t stream)
{
    const int threads = 256;
    const int smem_size = threads * sizeof(float);
    scaled_fp8_quant_dynamic_f16_kernel<<<num_tokens, threads, smem_size, stream>>>(
        input, output, scales, hidden_dim);
}

// Static FP8 quantization: BF16 → FP8 with given scale
void scaled_fp8_quant_static_bf16(
    const uint16_t* input,
    uint8_t* output,
    const float* scale,       // GPU scalar
    int num_elements,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = min((num_elements / 8 + threads - 1) / threads, 1024);
    scaled_fp8_quant_static_bf16_kernel<<<blocks, threads, 0, stream>>>(
        input, output, scale, num_elements);
}

// Static FP8 quantization: F16 → FP8 with given scale
void scaled_fp8_quant_static_f16(
    const uint16_t* input,
    uint8_t* output,
    const float* scale,
    int num_elements,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = min((num_elements / 8 + threads - 1) / threads, 1024);
    scaled_fp8_quant_static_f16_kernel<<<blocks, threads, 0, stream>>>(
        input, output, scale, num_elements);
}

// Online weight quantization: compute per-tensor absmax then quantize.
// Phase 1: absmax reduction. Phase 2: finalize scale.
// Phase 3: call scaled_fp8_quant_static_bf16 with the computed scale.
void fp8_quantize_weight_bf16(
    const uint16_t* weight,    // [N, K] BF16
    uint8_t* output,           // [N, K] FP8
    float* scale_out,          // single f32 on GPU — will hold computed scale
    int num_elements,
    cudaStream_t stream)
{
    // Zero the scale output (used as absmax accumulator)
    cudaMemsetAsync(scale_out, 0, sizeof(float), stream);

    // Phase 1: absmax reduction
    const int threads = 256;
    const int blocks = min((num_elements + threads - 1) / threads, 1024);
    weight_absmax_bf16_kernel<<<blocks, threads, 0, stream>>>(
        weight, num_elements, scale_out);

    // Phase 2: finalize scale = absmax / FP8_MAX
    finalize_fp8_weight_scale_kernel<<<1, 1, 0, stream>>>(scale_out);

    // Phase 3: quantize with computed scale
    const int quant_blocks = min((num_elements / 8 + threads - 1) / threads, 1024);
    scaled_fp8_quant_static_bf16_kernel<<<quant_blocks, threads, 0, stream>>>(
        weight, output, scale_out, num_elements);
}

} // extern "C"
