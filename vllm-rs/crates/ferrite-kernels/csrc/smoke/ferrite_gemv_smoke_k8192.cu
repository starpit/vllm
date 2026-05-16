// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK gemv_bf16 smoke, K = 8192 (llama-3.2-1B
// intermediate_dim). Same structure as ferrite_gemv_smoke.cu;
// only the K and PAGE_SIZE differ.
//
// Build: see ferrite_gemv_smoke.cu header. PAGE_SIZE must be
// >= K*2 bytes; 16384 = 8192 * 2, already 128-aligned.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_warp_roles.cuh"
#include "ferrite_kernels/gemv_bf16.cuh"

namespace {

struct SmokeConfig {
    static constexpr int NUM_CONSUMER_WARPS      = 4;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 192;
    static constexpr int NUM_PAGES               = 2;
    static constexpr int PAGE_SIZE               = 16384;  // 8192 * 2 bytes
    static constexpr int SCRATCH_BYTES           = 256;
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

constexpr int K   = 8192;
constexpr int N   = 32;
constexpr int TPB = (SmokeConfig::NUM_CONSUMER_WARPS + 3) * 32;

using SS = ferrite::SharedState<SmokeConfig>;

__global__ void gemv_smoke_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16*       __restrict__ out
) {
    __shared__ SS ss;
    ferrite::init_shared_state<SmokeConfig>(ss);

    const int wid = kittens::warpid();
    if (wid < SmokeConfig::NUM_CONSUMER_WARPS) {
        ferrite::set_consumer_registers<SmokeConfig>();
        ferrite::ops::gemv_bf16::consumer<SmokeConfig, K, N>(ss, 0, wid);
    } else {
        ferrite::set_non_consumer_registers<SmokeConfig>();
        switch (wid - SmokeConfig::NUM_CONSUMER_WARPS) {
            case ferrite::kLoaderSlot:
                ferrite::ops::gemv_bf16::loader  <SmokeConfig, K, N>(x, W, ss, 0); break;
            case ferrite::kLauncherSlot:
                ferrite::ops::gemv_bf16::launcher<SmokeConfig, K, N>(ss, 0); break;
            case ferrite::kStorerSlot:
                ferrite::ops::gemv_bf16::storer  <SmokeConfig, K, N>(out, ss, 0); break;
        }
    }
}

#define CUDA_CHECK(call)                                                  \
    do { cudaError_t e__ = (call);                                        \
         if (e__ != cudaSuccess) {                                        \
             std::fprintf(stderr, "cuda error %s: %s\n", #call,           \
                          cudaGetErrorString(e__)); std::exit(1); } } while (0)

uint32_t xs(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
float    ru(uint32_t& s) { return (float(xs(s)) / float(UINT32_MAX)) * 2.0f - 1.0f; }

} // namespace

int main() {
    std::vector<__nv_bfloat16> x_bf16(K), W_bf16(N * K);
    uint32_t s = 0xBEEFu;
    for (int i = 0; i < K;     ++i) x_bf16[i] = __float2bfloat16(ru(s));
    for (int i = 0; i < N * K; ++i) W_bf16[i] = __float2bfloat16(ru(s));

    std::vector<float> out_ref(N);
    for (int row = 0; row < N; ++row) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += __bfloat162float(x_bf16[k]) * __bfloat162float(W_bf16[row * K + k]);
        }
        out_ref[row] = acc;
    }

    __nv_bfloat16 *d_x = nullptr, *d_W = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x,   K       * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_W,   N * K   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, N       * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_x, x_bf16.data(), K       * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W_bf16.data(), N * K   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0,           N       * sizeof(__nv_bfloat16)));

    gemv_smoke_kernel<<<dim3(N), dim3(TPB)>>>(d_x, d_W, d_out);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) { std::fprintf(stderr, "launch: %s\n", cudaGetErrorString(le)); return 2; }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out_bf16(N);
    CUDA_CHECK(cudaMemcpy(out_bf16.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f, sum_ref_sq = 0.0f, sum_diff_sq = 0.0f;
    int   mm = 0;
    // K=8192 has 4x more summation than K=2048, so one more ULP
    // of bf16 slack is expected. 0.5 accommodates output
    // magnitudes up to ~128.
    const float TOL = 0.5f;
    for (int row = 0; row < N; ++row) {
        float got = __bfloat162float(out_bf16[row]);
        float ref = out_ref[row];
        float d   = std::fabs(got - ref);
        if (d > TOL) ++mm;
        if (d > max_abs) max_abs = d;
        sum_ref_sq  += ref * ref;
        sum_diff_sq += d * d;
    }
    float rel = std::sqrt(sum_diff_sq / (sum_ref_sq > 0 ? sum_ref_sq : 1.0f));
    std::printf("N=%d K=%d max_abs=%.4f rel_l2=%.4f mismatches(>%.2f)=%d\n",
                N, K, max_abs, rel, TOL, mm);
    if (mm == 0) std::printf("ok: gemv_bf16 K=%d matches CPU reference\n", K);
    return mm == 0 ? 0 : 3;
}
