// SPDX-License-Identifier: Apache-2.0
// Device-callable op wrappers for megakernel phases.
//
// Each function is a __device__ adaptation of the corresponding
// standalone __global__ kernel. The key differences:
//
// 1. __device__ instead of __global__ — called from within a
//    megakernel, not launched independently.
// 2. Explicit `num_rows` parameter — in a megakernel the grid size
//    is the max across all phases; CTAs beyond this phase's row
//    count early-exit.
// 3. Shared memory passed as a `char*` parameter — the megakernel
//    allocates the union-max and passes it to each phase.
//
// Vectorized using the same vec_utils.cuh patterns as standalone
// kernels.

#pragma once

#include <cuda_bf16.h>
#include "vec_utils.cuh"

// ── Warp/block reduction helpers ───────────────────────────────

__device__ __forceinline__ float dc_warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float dc_block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = dc_warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = dc_warp_reduce_sum(val);
    return val;
}

// ── RMS Norm (device-callable) ─────────────────────────────────
//
// out[row] = weight * input[row] * rsqrt(mean(input[row]^2) + eps)
// One CTA per row. CTAs beyond num_rows early-exit.

template <typename T>
__device__ void dc_rms_norm(
    T* __restrict__ out,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    float eps,
    int hidden_size,
    int num_rows,
    char* smem)
{
    const int row = blockIdx.x;
    if (row >= num_rows) return;

    constexpr int VEC_SIZE = VecType<T>::SIZE;
    using VT = typename VecType<T>::Type;

    const T* x = input + row * hidden_size;
    T* y = out + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: sum of squares.
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float buf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), buf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            ss += buf[j] * buf[j];
        }
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]);
        ss += v * v;
    }
    ss = dc_block_reduce_sum(ss);

    // Use smem for the shared inv_rms value.
    float* s_inv_rms = reinterpret_cast<float*>(smem);
    if (threadIdx.x == 0) {
        *s_inv_rms = rsqrtf(ss / hidden_size + eps);
    }
    __syncthreads();

    float inv_rms = *s_inv_rms;

    // Pass 2: normalize.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float xbuf[VEC_SIZE], wbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), xbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = xbuf[j] * inv_rms * wbuf[j];
        }
        vec_store(&y[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]) * inv_rms;
        y[i] = static_cast<T>(v * static_cast<float>(weight[i]));
    }
}

// ── Fused Add + RMS Norm (device-callable, in-place) ───────────
//
// residual += input; input = weight * residual * rsqrt(...)
// Same as fused_add_rms_norm_kernel but __device__.

template <typename T>
__device__ void dc_fused_add_rms_norm(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    float eps,
    int hidden_size,
    int num_rows,
    char* smem)
{
    const int row = blockIdx.x;
    if (row >= num_rows) return;

    constexpr int VEC_SIZE = VecType<T>::SIZE;

    T* x = input + row * hidden_size;
    T* r = residual + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: residual += input, accumulate sum of squares.
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float xbuf[VEC_SIZE], rbuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), xbuf);
        unpack_vec<T>(vec_load(&r[vi * VEC_SIZE]), rbuf);
        float sbuf[VEC_SIZE];
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            sbuf[j] = xbuf[j] + rbuf[j];
            ss += sbuf[j] * sbuf[j];
        }
        vec_store(&r[vi * VEC_SIZE], pack_vec<T>(sbuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float xv = static_cast<float>(x[i]);
        float rv = static_cast<float>(r[i]);
        float sv = xv + rv;
        r[i] = static_cast<T>(sv);
        ss += sv * sv;
    }
    ss = dc_block_reduce_sum(ss);

    float* s_inv_rms = reinterpret_cast<float*>(smem);
    if (threadIdx.x == 0) {
        *s_inv_rms = rsqrtf(ss / hidden_size + eps);
    }
    __syncthreads();

    float inv_rms = *s_inv_rms;

    // Pass 2: input = weight * residual * inv_rms.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float rbuf[VEC_SIZE], wbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&r[vi * VEC_SIZE]), rbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = rbuf[j] * inv_rms * wbuf[j];
        }
        vec_store(&x[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float rv = static_cast<float>(r[i]) * inv_rms;
        x[i] = static_cast<T>(rv * static_cast<float>(weight[i]));
    }
}

// ── GEMV (device-callable, bf16) ──────────────────────────────
//
// y[j] = alpha * W[j,K] @ x[K] + beta * C[j]
// One CTA per output element (row of W). Threads cooperatively
// reduce over K. W is [N,K] row-major (our GEMM's B transposed).

template <typename T>
__device__ void dc_gemv(
    T* __restrict__ out,         // [N]
    const T* __restrict__ x,     // [K]
    const T* __restrict__ W,     // [N, K] row-major
    int N, int K,
    float alpha, float beta,
    int num_rows)                // bounds check (= N in normal use)
{
    const int row = blockIdx.x;
    if (row >= num_rows || row >= N) return;

    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const T* w_row = W + row * K;
    const int num_vecs = K / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Accumulate dot product.
    float acc = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float wbuf[VEC_SIZE], xbuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&w_row[vi * VEC_SIZE]), wbuf);
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), xbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            acc += wbuf[j] * xbuf[j];
        }
    }
    // Scalar tail.
    for (int i = tail_start + threadIdx.x; i < K; i += blockDim.x) {
        acc += static_cast<float>(w_row[i]) * static_cast<float>(x[i]);
    }

    // Block-wide reduction.
    acc = dc_block_reduce_sum(acc);

    if (threadIdx.x == 0) {
        float result = alpha * acc;
        if (beta != 0.0f) {
            result += beta * static_cast<float>(out[row]);
        }
        out[row] = static_cast<T>(result);
    }
}

// ── Fused QKV + RoPE + KV cache write (device-callable, decode) ─
//
// One CTA per token. Reads from packed QKV output, applies rotary
// embeddings to Q and K heads, writes K/V directly to the paged
// cache, and outputs Q for attention.
//
// Identical logic to fused_qkv_rope_cache_kernel in
// pos_encoding_kernels.cu but as __device__ with num_rows guard.

template <typename T>
__device__ void dc_fused_qkv_rope_cache(
    T* __restrict__ q_out,                   // [num_tokens, q_size]
    T* __restrict__ key_cache,               // [num_blocks, block_size, kv_heads, head_dim]
    T* __restrict__ value_cache,             // same layout
    const T* __restrict__ qkv,               // [num_tokens, total_dim]
    const uint32_t* __restrict__ positions,  // [num_tokens]
    const T* __restrict__ cos_sin_cache,     // [max_pos, rotary_dim]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    int q_size,
    int kv_size,
    int head_size,
    int num_rows,
    char* /*smem — unused, included for uniform signature*/)
{
    constexpr int VEC = VecType<T>::SIZE;

    const int token_idx = blockIdx.x;
    if (token_idx >= num_rows) return;

    const int total_dim = q_size + 2 * kv_size;
    const int rotary_dim = head_size;  // Llama: rotary_dim == head_dim
    const int half_rot = rotary_dim / 2;
    const int half = (half_rot < head_size / 2) ? half_rot : (head_size / 2);
    const int half_vecs = half / VEC;
    const int half_tail = half_vecs * VEC;
    const int rot_dim_full = 2 * half;
    const int non_rot = head_size - rot_dim_full;

    const int64_t slot = slot_mapping[token_idx];
    const int pos = static_cast<int>(positions[token_idx]);
    const T* cos_ptr = cos_sin_cache + pos * rotary_dim;
    const T* sin_ptr = cos_ptr + half_rot;

    const T* row = qkv + token_idx * total_dim;
    T* q = q_out + token_idx * q_size;

    const int nqh = q_size / head_size;
    const int nkh = kv_size / head_size;

    // --- Q heads: read from QKV, apply RoPE, write contiguous ---
    for (int tid = threadIdx.x; tid < nqh * half_vecs; tid += blockDim.x) {
        const int h = tid / half_vecs;
        const int vi = tid % half_vecs;
        const int b = h * head_size + vi * VEC;

        float xbuf[VEC], ybuf[VEC], cbuf[VEC], sbuf[VEC];
        unpack_vec<T>(vec_load(&row[b]), xbuf);
        unpack_vec<T>(vec_load(&row[b + half]), ybuf);
        unpack_vec<T>(vec_load(&cos_ptr[vi * VEC]), cbuf);
        unpack_vec<T>(vec_load(&sin_ptr[vi * VEC]), sbuf);

        float ox[VEC], oy[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            ox[j] = xbuf[j] * cbuf[j] - ybuf[j] * sbuf[j];
            oy[j] = ybuf[j] * cbuf[j] + xbuf[j] * sbuf[j];
        }
        vec_store(&q[b], pack_vec<T>(ox));
        vec_store(&q[b + half], pack_vec<T>(oy));
    }
    // Scalar tail for Q rotary pairs.
    for (int tid = threadIdx.x; tid < nqh * (half - half_tail); tid += blockDim.x) {
        const int h = tid / (half - half_tail);
        const int r = half_tail + tid % (half - half_tail);
        const int b = h * head_size;
        float x = static_cast<float>(row[b + r]);
        float y = static_cast<float>(row[b + r + half]);
        float c = static_cast<float>(cos_ptr[r]);
        float s = static_cast<float>(sin_ptr[r]);
        q[b + r]        = static_cast<T>(x * c - y * s);
        q[b + r + half] = static_cast<T>(y * c + x * s);
    }
    // Copy non-rotary Q elements.
    for (int tid = threadIdx.x; tid < nqh * non_rot; tid += blockDim.x) {
        const int h = tid / non_rot;
        const int i = rot_dim_full + tid % non_rot;
        q[h * head_size + i] = row[h * head_size + i];
    }

    // --- K heads: read from QKV, apply RoPE, write to paged cache ---
    const T* k_row = row + q_size;
    T* k_dst = (slot >= 0) ? (key_cache + slot * kv_size) : nullptr;
    T* v_dst = (slot >= 0) ? (value_cache + slot * kv_size) : nullptr;

    if (k_dst) {
        for (int tid = threadIdx.x; tid < nkh * half_vecs; tid += blockDim.x) {
            const int h = tid / half_vecs;
            const int vi = tid % half_vecs;
            const int b = h * head_size + vi * VEC;

            float xbuf[VEC], ybuf[VEC], cbuf[VEC], sbuf[VEC];
            unpack_vec<T>(vec_load(&k_row[b]), xbuf);
            unpack_vec<T>(vec_load(&k_row[b + half]), ybuf);
            unpack_vec<T>(vec_load(&cos_ptr[vi * VEC]), cbuf);
            unpack_vec<T>(vec_load(&sin_ptr[vi * VEC]), sbuf);

            float ox[VEC], oy[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; j++) {
                ox[j] = xbuf[j] * cbuf[j] - ybuf[j] * sbuf[j];
                oy[j] = ybuf[j] * cbuf[j] + xbuf[j] * sbuf[j];
            }
            vec_store(&k_dst[b], pack_vec<T>(ox));
            vec_store(&k_dst[b + half], pack_vec<T>(oy));
        }
        for (int tid = threadIdx.x; tid < nkh * (half - half_tail); tid += blockDim.x) {
            const int h = tid / (half - half_tail);
            const int r = half_tail + tid % (half - half_tail);
            const int b = h * head_size;
            float x = static_cast<float>(k_row[b + r]);
            float y = static_cast<float>(k_row[b + r + half]);
            float c = static_cast<float>(cos_ptr[r]);
            float s = static_cast<float>(sin_ptr[r]);
            k_dst[b + r]        = static_cast<T>(x * c - y * s);
            k_dst[b + r + half] = static_cast<T>(y * c + x * s);
        }
        for (int tid = threadIdx.x; tid < nkh * non_rot; tid += blockDim.x) {
            const int h = tid / non_rot;
            const int i = rot_dim_full + tid % non_rot;
            k_dst[h * head_size + i] = k_row[h * head_size + i];
        }

        // --- V: straight copy to cache ---
        const T* v_row = row + q_size + kv_size;
        const int v_vecs = kv_size / VEC;
        for (int tid = threadIdx.x; tid < v_vecs; tid += blockDim.x) {
            vec_store(&v_dst[tid * VEC], vec_load(&v_row[tid * VEC]));
        }
        for (int tid = threadIdx.x; tid < kv_size - v_vecs * VEC; tid += blockDim.x) {
            v_dst[v_vecs * VEC + tid] = v_row[v_vecs * VEC + tid];
        }
    }
}

// ── SiLU-and-Mul (device-callable) ─────────────────────────────
//
// out[i] = silu(gate[i]) * up[i], where gate and up are
// packed as [gate|up] in a [num_rows, 2*d] tensor.
// One CTA per row.

__device__ __forceinline__ float dc_silu_scalar(float x) {
    return x / (1.0f + expf(-x));
}

template <typename T>
__device__ void dc_silu_and_mul(
    T* __restrict__ out,
    const T* __restrict__ input,  // [num_rows, 2*d]
    int d,
    int num_rows)
{
    const int row = blockIdx.x;
    if (row >= num_rows) return;

    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const T* gate = input + row * 2 * d;
    const T* up = gate + d;
    T* o = out + row * d;

    const int num_vecs = d / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float gbuf[VEC_SIZE], ubuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&gate[vi * VEC_SIZE]), gbuf);
        unpack_vec<T>(vec_load(&up[vi * VEC_SIZE]), ubuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = dc_silu_scalar(gbuf[j]) * ubuf[j];
        }
        vec_store(&o[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    for (int i = tail_start + threadIdx.x; i < d; i += blockDim.x) {
        float gv = dc_silu_scalar(static_cast<float>(gate[i]));
        float uv = static_cast<float>(up[i]);
        o[i] = static_cast<T>(gv * uv);
    }
}

// ── GELU-and-Mul (device-callable) ────────────────────────────────
//
// out[i] = gelu_tanh(gate[i]) * up[i], where gate and up are
// packed as [gate|up] in a [num_rows, 2*d] tensor.
// One CTA per row.

__device__ __forceinline__ float dc_gelu_tanh(float x) {
    constexpr float BETA = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float KAPPA = 0.044715f;
    float x3 = x * x * x;
    float inner = BETA * (x + KAPPA * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

template <typename T>
__device__ void dc_gelu_and_mul(
    T* __restrict__ out,
    const T* __restrict__ input,  // [num_rows, 2*d]
    int d,
    int num_rows)
{
    const int row = blockIdx.x;
    if (row >= num_rows) return;

    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const T* gate = input + row * 2 * d;
    const T* up = gate + d;
    T* o = out + row * d;

    const int num_vecs = d / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float gbuf[VEC_SIZE], ubuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&gate[vi * VEC_SIZE]), gbuf);
        unpack_vec<T>(vec_load(&up[vi * VEC_SIZE]), ubuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = dc_gelu_tanh(gbuf[j]) * ubuf[j];
        }
        vec_store(&o[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    for (int i = tail_start + threadIdx.x; i < d; i += blockDim.x) {
        float gv = dc_gelu_tanh(static_cast<float>(gate[i]));
        float uv = static_cast<float>(up[i]);
        o[i] = static_cast<T>(gv * uv);
    }
}

// ── Scalar multiply inplace (device-callable) ─────────────────────
//
// x[i] *= scalar.  One CTA processes the whole row.
// Used for embedding scaling (e.g. Gemma sqrt(hidden_size)).

template <typename T>
__device__ void dc_scalar_mul_inplace(
    T* __restrict__ x,
    float scalar,
    int n,
    int num_rows)
{
    // Grid-stride over total elements = num_rows * n.
    const int total = num_rows * n;
    constexpr int VEC_SIZE = VecType<T>::SIZE;
    const int num_vecs = total / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int vi = tid; vi < num_vecs; vi += stride) {
        float buf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), buf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            buf[j] *= scalar;
        }
        vec_store(&x[vi * VEC_SIZE], pack_vec<T>(buf));
    }
    for (int i = tail_start + tid; i < total; i += stride) {
        x[i] = static_cast<T>(static_cast<float>(x[i]) * scalar);
    }
}

// ── Tanh softcap inplace (device-callable) ────────────────────────
//
// x[i] = cap * tanh(x[i] / cap).  Grid-stride over all elements.
// Used for Gemma2 final logit softcapping.

template <typename T>
__device__ void dc_tanh_softcap_inplace(
    T* __restrict__ x,
    float inv_cap,
    float cap,
    int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float v = static_cast<float>(x[i]) * inv_cap;
        x[i] = static_cast<T>(cap * tanhf(v));
    }
}

// ── RMS Norm with weight offset (device-callable) ─────────────────
//
// out[row] = (1 + weight) * input[row] * rsqrt(mean(input[row]^2) + eps)
// The (1+w) convention is Gemma's norm weight format.
// Same as dc_rms_norm but adds 1.0 to the weight.

template <typename T>
__device__ void dc_rms_norm_with_offset(
    T* __restrict__ out,
    const T* __restrict__ input,
    const T* __restrict__ weight,
    float eps,
    float weight_offset,
    int hidden_size,
    int num_rows,
    char* smem)
{
    const int row = blockIdx.x;
    if (row >= num_rows) return;

    constexpr int VEC_SIZE = VecType<T>::SIZE;

    const T* x = input + row * hidden_size;
    T* y = out + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    // Pass 1: sum of squares.
    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float buf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), buf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            ss += buf[j] * buf[j];
        }
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]);
        ss += v * v;
    }
    ss = dc_block_reduce_sum(ss);

    float* s_inv_rms = reinterpret_cast<float*>(smem);
    if (threadIdx.x == 0) {
        *s_inv_rms = rsqrtf(ss / hidden_size + eps);
    }
    __syncthreads();

    float inv_rms = *s_inv_rms;

    // Pass 2: normalize with offset weight.
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float xbuf[VEC_SIZE], wbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), xbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = xbuf[j] * inv_rms * (wbuf[j] + weight_offset);
        }
        vec_store(&y[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float v = static_cast<float>(x[i]) * inv_rms;
        y[i] = static_cast<T>(v * (static_cast<float>(weight[i]) + weight_offset));
    }
}

// ── Fused Add + RMS Norm with weight offset (device-callable) ─────
//
// residual += input; input = (weight_offset + weight) * residual * rsqrt(...)
// Gemma2 uses weight_offset=1.0 for its (1+w) convention.

template <typename T>
__device__ void dc_fused_add_rms_norm_with_offset(
    T* __restrict__ input,
    T* __restrict__ residual,
    const T* __restrict__ weight,
    float eps,
    float weight_offset,
    int hidden_size,
    int num_rows,
    char* smem)
{
    const int row = blockIdx.x;
    if (row >= num_rows) return;

    constexpr int VEC_SIZE = VecType<T>::SIZE;

    T* x = input + row * hidden_size;
    T* r = residual + row * hidden_size;

    const int num_vecs = hidden_size / VEC_SIZE;
    const int tail_start = num_vecs * VEC_SIZE;

    float ss = 0.0f;
    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float xbuf[VEC_SIZE], rbuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&x[vi * VEC_SIZE]), xbuf);
        unpack_vec<T>(vec_load(&r[vi * VEC_SIZE]), rbuf);
        float sbuf[VEC_SIZE];
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            sbuf[j] = xbuf[j] + rbuf[j];
            ss += sbuf[j] * sbuf[j];
        }
        vec_store(&r[vi * VEC_SIZE], pack_vec<T>(sbuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float xv = static_cast<float>(x[i]);
        float rv = static_cast<float>(r[i]);
        float sv = xv + rv;
        r[i] = static_cast<T>(sv);
        ss += sv * sv;
    }
    ss = dc_block_reduce_sum(ss);

    float* s_inv_rms = reinterpret_cast<float*>(smem);
    if (threadIdx.x == 0) {
        *s_inv_rms = rsqrtf(ss / hidden_size + eps);
    }
    __syncthreads();

    float inv_rms = *s_inv_rms;

    for (int vi = threadIdx.x; vi < num_vecs; vi += blockDim.x) {
        float rbuf[VEC_SIZE], wbuf[VEC_SIZE], obuf[VEC_SIZE];
        unpack_vec<T>(vec_load(&r[vi * VEC_SIZE]), rbuf);
        unpack_vec<T>(vec_load(&weight[vi * VEC_SIZE]), wbuf);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            obuf[j] = rbuf[j] * inv_rms * (wbuf[j] + weight_offset);
        }
        vec_store(&x[vi * VEC_SIZE], pack_vec<T>(obuf));
    }
    for (int i = tail_start + threadIdx.x; i < hidden_size; i += blockDim.x) {
        float rv = static_cast<float>(r[i]) * inv_rms;
        x[i] = static_cast<T>(rv * (static_cast<float>(weight[i]) + weight_offset));
    }
}

// ── Fused QKV + RoPE + KV cache write (device-callable, prefill) ──
//
// Prefill variant: processes multiple tokens. Same logic as decode
// dc_fused_qkv_rope_cache but with num_rows > 1 and contiguous
// output rather than paged cache writes.
// Note: uses the same dc_fused_qkv_rope_cache function above —
// it already handles num_rows > 1 correctly.

// (No new function needed — dc_fused_qkv_rope_cache already works
// for both decode and prefill since it uses num_rows as the bound.
// The prefill DC impl just passes num_tokens > 1.)

// ── TK Attention Decode (device-callable stub) ────────────────────
//
// ThunderKittens wgmma-based attention for sm90+.  The actual
// persistent-kernel integration requires embedding the TK runner's
// device-side dispatch.  For now we provide a stub that the
// megakernel generator references — the real implementation lands
// with the TK runtime FFI.

// NOTE: TK attention is inherently a persistent kernel that manages
// its own warp scheduling.  It cannot be trivially wrapped in a
// megakernel phase like elementwise ops.  The megakernel codegen
// treats TK attention as a "boundary" phase that gets its own
// cooperative launch rather than being inlined as a __device__ call.
// This is NOT a limitation — it's by design: TK attention occupies
// all SMs and uses mbarrier-based warpgroup scheduling internally.
