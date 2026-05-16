// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK lm_head standalone smoke test.
//
// Drives the four walker roles of `lm_head.cuh` end-to-end for a
// single decode token against a CPU reference:
//   x_n[k]   = x[k] * rsqrt(mean(x^2) + eps) * norm_weight[k]
//   out[row] = sum_k W_gemm[row, k] * x_n[k]
//
// Grid `dim3(N)`; each CTA owns one output row. K = HIDDEN_DIM =
// 2048, N = 8192 — shrunk from the full Llama-3.2-1B vocab (128256)
// because the smoke just needs to exercise the reduction + dot
// pattern at one representative shape; 8192 rows take a few ms
// instead of ~3 seconds and hit every code path (K % (NCW*32) == 0
// math, cross-warp reduction, row-index bail gate).
//
// Build (pod nick, CUDA 12.9, H100 sm_90a):
//   nvcc -O3 -std=c++20 \
//     -gencode arch=compute_90a,code=sm_90a \
//     -DKITTENS_HOPPER --extended-lambda --expt-relaxed-constexpr \
//     -I /home/nickm/vllm-mega/vllm-rs/third_party/thunderkittens/include \
//     -I /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/tk \
//     /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/smoke/ferrite_lm_head_smoke.cu \
//     -o /tmp/ferrite_lm_head_smoke -lcuda
//
// Run: /tmp/ferrite_lm_head_smoke

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_warp_roles.cuh"
#include "ferrite_kernels/lm_head.cuh"

namespace {

static constexpr int K          = 2048;   // HIDDEN_DIM
static constexpr int N          = 8192;   // trimmed vocab (see header comment)
static constexpr int NUM_TOKENS = 1;
static constexpr float EPS      = 1e-5f;

struct SmokeConfig {
    static constexpr int NUM_CONSUMER_WARPS      = 4;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 192;
    static constexpr int NUM_PAGES               = 3;
    static constexpr int PAGE_SIZE               = 4096;   // K * 2
    static constexpr int SCRATCH_BYTES           = 256;
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

constexpr int TPB = (SmokeConfig::NUM_CONSUMER_WARPS + 3) * 32;

using SS = ferrite::SharedState<SmokeConfig>;

__global__ void lm_head_smoke_kernel(
    const __nv_bfloat16* __restrict__ x,            // [K]
    const __nv_bfloat16* __restrict__ norm_weight,  // [K]
    const __nv_bfloat16* __restrict__ W_gemm,       // [N, K]
    __nv_bfloat16*       __restrict__ out           // [N]
) {
    __shared__ SS ss;
    ferrite::init_shared_state<SmokeConfig>(ss);

    const int wid = kittens::warpid();
    if (wid < SmokeConfig::NUM_CONSUMER_WARPS) {
        ferrite::set_consumer_registers<SmokeConfig>();
        ferrite::ops::lm_head::consumer<
            SmokeConfig, K, N, NUM_TOKENS>(
                ss, /*base_stage=*/0, wid, EPS);
    } else {
        ferrite::set_non_consumer_registers<SmokeConfig>();
        switch (wid - SmokeConfig::NUM_CONSUMER_WARPS) {
            case ferrite::kLoaderSlot:
                ferrite::ops::lm_head::loader<
                    SmokeConfig, K, N, NUM_TOKENS>(
                        x, norm_weight, W_gemm, ss, /*base_stage=*/0);
                break;
            case ferrite::kLauncherSlot:
                ferrite::ops::lm_head::launcher<
                    SmokeConfig, K, N, NUM_TOKENS>(
                        ss, /*base_stage=*/0);
                break;
            case ferrite::kStorerSlot:
                ferrite::ops::lm_head::storer<
                    SmokeConfig, K, N, NUM_TOKENS>(
                        out, ss, /*base_stage=*/0);
                break;
            default: break;
        }
    }
}

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err__ = (call);                                       \
        if (err__ != cudaSuccess) {                                       \
            std::fprintf(stderr, "cuda error %s at %s:%d — %s\n",         \
                         #call, __FILE__, __LINE__,                       \
                         cudaGetErrorString(err__));                      \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}
float rand_u(uint32_t& s) {
    uint32_t v = xorshift32(s);
    return (float(v) / float(UINT32_MAX)) * 2.0f - 1.0f;
}

} // namespace

int main() {
    std::vector<__nv_bfloat16> h_x          (K);
    std::vector<__nv_bfloat16> h_norm_w     (K);
    std::vector<__nv_bfloat16> h_W_gemm     (size_t(N) * K);

    // Typical pre-lm_head activations are scale ~ 1-2 after the final
    // RmsNorm's residual accumulation. Use scale=1.0 for x so
    // sum_sq/K lands near 1/3 and rms_scale is bounded and realistic
    // (~1.7). norm_weight is ~1 in trained models; sample on [0.5, 1.5]
    // to exercise the multiply. W_gemm at 0.05 keeps the output in a
    // bounded band after the inv-rms rescale.
    uint32_t s = 0xABCDEF01u;
    for (auto& v : h_x)      v = __float2bfloat16_rn(1.0f  * rand_u(s));
    for (auto& v : h_norm_w) v = __float2bfloat16_rn(1.0f + 0.5f * rand_u(s));
    for (auto& v : h_W_gemm) v = __float2bfloat16_rn(0.05f * rand_u(s));

    // ---- CPU reference (fp32 math; bf16 round-trip on inputs and
    // at the final pack).
    float sum_sq = 0.0f;
    for (int k = 0; k < K; ++k) {
        float v = __bfloat162float(h_x[k]);
        sum_sq += v * v;
    }
    float rms_scale = 1.0f / std::sqrt(sum_sq / float(K) + EPS);

    std::vector<__nv_bfloat16> h_out_ref(N);
    for (int row = 0; row < N; ++row) {
        float dot = 0.0f;
        for (int k = 0; k < K; ++k) {
            float xv  = __bfloat162float(h_x[k]);
            float nwv = __bfloat162float(h_norm_w[k]);
            float gwv = __bfloat162float(h_W_gemm[size_t(row) * K + k]);
            dot += (xv * rms_scale * nwv) * gwv;
        }
        h_out_ref[row] = __float2bfloat16_rn(dot);
    }

    __nv_bfloat16 *d_x = nullptr, *d_nw = nullptr, *d_W = nullptr, *d_out = nullptr;
    const size_t x_bytes   = size_t(K) * sizeof(__nv_bfloat16);
    const size_t w_bytes   = size_t(N) * K * sizeof(__nv_bfloat16);
    const size_t out_bytes = size_t(N) * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_x,   x_bytes));
    CUDA_CHECK(cudaMalloc(&d_nw,  x_bytes));
    CUDA_CHECK(cudaMalloc(&d_W,   w_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_x,  h_x.data(),      x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nw, h_norm_w.data(), x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W,  h_W_gemm.data(), w_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, out_bytes));

    dim3 grid(N, 1, 1);
    dim3 block(TPB, 1, 1);
    lm_head_smoke_kernel<<<grid, block, 0, 0>>>(d_x, d_nw, d_W, d_out);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launch_err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out_got(N);
    CUDA_CHECK(cudaMemcpy(h_out_got.data(), d_out, out_bytes,
                          cudaMemcpyDeviceToHost));

    // Tolerance: output magnitudes ~0.5-2 (rms_scale ≈ 1.7 times
    // 0.05-scale weights times K=2048 dot product). bf16 ULP ~0.015,
    // fp32 reduction-order drift adds a few ULPs. 0.03 is ~2x that.
    const float TOL = 0.03f;
    float max_abs = 0.0f, sum_ref_sq = 0.0f, sum_diff_sq = 0.0f;
    int mismatches = 0;
    int first_mismatch = -1;
    float first_got = 0.0f, first_ref = 0.0f;
    for (int row = 0; row < N; ++row) {
        float got = __bfloat162float(h_out_got[row]);
        float ref = __bfloat162float(h_out_ref[row]);
        float diff = std::fabs(got - ref);
        if (diff > TOL) {
            ++mismatches;
            if (first_mismatch < 0) {
                first_mismatch = row;
                first_got = got;
                first_ref = ref;
            }
        }
        if (diff > max_abs) max_abs = diff;
        sum_ref_sq  += ref * ref;
        sum_diff_sq += diff * diff;
    }
    float rel_l2 = std::sqrt(sum_diff_sq / (sum_ref_sq > 0 ? sum_ref_sq : 1.0f));

    std::printf("K=%d N=%d ncw=%d rms_scale_ref=%.4f\n",
                K, N, SmokeConfig::NUM_CONSUMER_WARPS, rms_scale);
    std::printf("max_abs=%.4f rel_l2=%.4f mismatches(>%.3f)=%d\n",
                max_abs, rel_l2, TOL, mismatches);
    std::printf("sample rows (first 4):\n");
    for (int row = 0; row < 4; ++row) {
        std::printf("  row=%d got=%.4f ref=%.4f\n",
                    row,
                    __bfloat162float(h_out_got[row]),
                    __bfloat162float(h_out_ref[row]));
    }
    if (mismatches == 0) {
        std::printf("ok: lm_head matches CPU reference within tolerance\n");
    } else {
        std::printf("first mismatch at row=%d got=%.4f ref=%.4f diff=%.4f\n",
                    first_mismatch, first_got, first_ref,
                    std::fabs(first_got - first_ref));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_nw));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_out));
    return mismatches == 0 ? 0 : 3;
}
