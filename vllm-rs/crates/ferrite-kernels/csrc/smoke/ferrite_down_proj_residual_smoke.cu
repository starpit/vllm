// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK down_proj_residual standalone smoke test.
//
// Drives the four walker roles of `down_proj_residual.cuh` end-to-
// end for a single decode token against a CPU reference that does:
//   residual[N] += W[N, K] @ x[K]
//
// Grid `dim3(N)`; each CTA owns one output row. K = INTERMEDIATE_DIM
// = 8192, N = HIDDEN_DIM = 2048 — Llama-3.2-1B down_proj shape.
//
// Does not go through codegen; reproduces the minimum substrate a
// codegen'd .cu would emit.
//
// Build (on pod nick, CUDA 12.9, H100 sm_90a):
//   nvcc -O3 -std=c++20 \
//     -gencode arch=compute_90a,code=sm_90a \
//     -DKITTENS_HOPPER --extended-lambda --expt-relaxed-constexpr \
//     -I /home/nickm/vllm-mega/vllm-rs/third_party/thunderkittens/include \
//     -I /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/tk \
//     /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/smoke/ferrite_down_proj_residual_smoke.cu \
//     -o /tmp/ferrite_down_proj_residual_smoke -lcuda
//
// Run: /tmp/ferrite_down_proj_residual_smoke

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
#include "ferrite_kernels/down_proj_residual.cuh"

namespace {

// ---- Model dims (llama-3.2-1B decode down_proj).
static constexpr int K          = 8192;   // INTERMEDIATE_DIM
static constexpr int N          = 2048;   // HIDDEN_DIM
static constexpr int NUM_TOKENS = 1;

struct SmokeConfig {
    static constexpr int NUM_CONSUMER_WARPS      = 4;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 192;
    static constexpr int NUM_PAGES               = 2;
    static constexpr int PAGE_SIZE               = 16384;  // K * 2
    static constexpr int SCRATCH_BYTES           = 256;
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

constexpr int TPB = (SmokeConfig::NUM_CONSUMER_WARPS + 3) * 32;

using SS = ferrite::SharedState<SmokeConfig>;

__global__ void down_proj_residual_smoke_kernel(
    const __nv_bfloat16* __restrict__ x,         // [K]
    const __nv_bfloat16* __restrict__ W,         // [N, K]
    __nv_bfloat16*       __restrict__ residual   // [N] — in-place
) {
    __shared__ SS ss;
    ferrite::init_shared_state<SmokeConfig>(ss);

    const int wid = kittens::warpid();
    if (wid < SmokeConfig::NUM_CONSUMER_WARPS) {
        ferrite::set_consumer_registers<SmokeConfig>();
        ferrite::ops::down_proj_residual::consumer<
            SmokeConfig, K, N, NUM_TOKENS>(
                ss, /*base_stage=*/0, wid);
    } else {
        ferrite::set_non_consumer_registers<SmokeConfig>();
        switch (wid - SmokeConfig::NUM_CONSUMER_WARPS) {
            case ferrite::kLoaderSlot:
                ferrite::ops::down_proj_residual::loader<
                    SmokeConfig, K, N, NUM_TOKENS>(
                        x, W, ss, /*base_stage=*/0);
                break;
            case ferrite::kLauncherSlot:
                ferrite::ops::down_proj_residual::launcher<
                    SmokeConfig, K, N, NUM_TOKENS>(
                        ss, /*base_stage=*/0);
                break;
            case ferrite::kStorerSlot:
                ferrite::ops::down_proj_residual::storer<
                    SmokeConfig, K, N, NUM_TOKENS>(
                        residual, ss, /*base_stage=*/0);
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
    std::vector<__nv_bfloat16> h_x        (K);
    std::vector<__nv_bfloat16> h_W        (size_t(N) * K);
    std::vector<__nv_bfloat16> h_residual (N);
    std::vector<__nv_bfloat16> h_residual_orig(N);

    // Scale so fp32 dot product + residual starts in-range: with x,
    // W uniform on [-0.1, 0.1] and K=8192, stddev of the dot is
    // sqrt(K) * (0.1^2 / 3) ~= 1.2. Residual sampled on [-0.5, 0.5]
    // (roughly the scale of a post-silu-mul partial + pre-residual
    // layer stream) — the add stays bounded and the final bf16 pack
    // doesn't collapse rare large values into inf.
    uint32_t s = 0xD0FFA11Du;
    for (auto& v : h_x)        v = __float2bfloat16_rn(0.1f * rand_u(s));
    for (auto& v : h_W)        v = __float2bfloat16_rn(0.1f * rand_u(s));
    for (auto& v : h_residual) v = __float2bfloat16_rn(0.5f * rand_u(s));
    h_residual_orig = h_residual;

    // ---- CPU reference.
    std::vector<__nv_bfloat16> h_residual_ref(N);
    for (int row = 0; row < N; ++row) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float xv = __bfloat162float(h_x[k]);
            float wv = __bfloat162float(h_W[size_t(row) * K + k]);
            acc += xv * wv;
        }
        float prev = __bfloat162float(h_residual_orig[row]);
        h_residual_ref[row] = __float2bfloat16_rn(prev + acc);
    }

    // ---- Device buffers.
    __nv_bfloat16 *d_x = nullptr, *d_W = nullptr, *d_residual = nullptr;
    const size_t x_bytes   = size_t(K) * sizeof(__nv_bfloat16);
    const size_t w_bytes   = size_t(N) * K * sizeof(__nv_bfloat16);
    const size_t res_bytes = size_t(N) * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_x,        x_bytes));
    CUDA_CHECK(cudaMalloc(&d_W,        w_bytes));
    CUDA_CHECK(cudaMalloc(&d_residual, res_bytes));
    CUDA_CHECK(cudaMemcpy(d_x,        h_x.data(),             x_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W,        h_W.data(),             w_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual_orig.data(), res_bytes, cudaMemcpyHostToDevice));

    dim3 grid(N, 1, 1);
    dim3 block(TPB, 1, 1);
    down_proj_residual_smoke_kernel<<<grid, block, 0, 0>>>(d_x, d_W, d_residual);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launch_err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_residual_got(N);
    CUDA_CHECK(cudaMemcpy(h_residual_got.data(), d_residual, res_bytes,
                          cudaMemcpyDeviceToHost));

    // Tolerance: bf16 ULP at observed residual magnitudes (~0.5-2)
    // is ~1e-3 to 8e-3; fp32 reduction-order drift over K=8192 adds
    // a few ULPs. 0.02 is ~2x that — tight enough to catch a wrong-
    // sign residual or dropped reduction lane.
    const float TOL = 0.02f;
    float max_abs = 0.0f, sum_ref_sq = 0.0f, sum_diff_sq = 0.0f;
    int mismatches = 0;
    int first_mismatch = -1;
    float first_got = 0.0f, first_ref = 0.0f;
    for (int row = 0; row < N; ++row) {
        float got = __bfloat162float(h_residual_got[row]);
        float ref = __bfloat162float(h_residual_ref[row]);
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

    std::printf("N=%d K=%d ncw=%d\n", N, K, SmokeConfig::NUM_CONSUMER_WARPS);
    std::printf("max_abs=%.4f rel_l2=%.4f mismatches(>%.3f)=%d\n",
                max_abs, rel_l2, TOL, mismatches);
    std::printf("sample rows (first 4):\n");
    for (int row = 0; row < 4; ++row) {
        std::printf("  row=%d orig=%.4f got=%.4f ref=%.4f\n",
                    row,
                    __bfloat162float(h_residual_orig[row]),
                    __bfloat162float(h_residual_got[row]),
                    __bfloat162float(h_residual_ref[row]));
    }
    if (mismatches == 0) {
        std::printf("ok: down_proj_residual matches CPU reference within tolerance\n");
    } else {
        std::printf("first mismatch at row=%d got=%.4f ref=%.4f diff=%.4f\n",
                    first_mismatch, first_got, first_ref,
                    std::fabs(first_got - first_ref));
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_residual));
    return mismatches == 0 ? 0 : 3;
}
