// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK gemv_bf16 standalone smoke test.
//
// Phase 3a exit gate: prove `ferrite::ops::gemv_bf16::{loader,
// consumer,storer}` produce a bf16 dot product that matches a CPU
// reference. Does not go through codegen — this file reproduces
// the minimum substrate a codegen'd .cu would emit so we isolate
// "did I write the op header correctly?" from "did I wire codegen
// correctly?".
//
// Variant: N = 64 rows, K = 2048 (llama-3.2-1B hidden_dim). bf16
// weights and activations, fp32 accumulator, random inputs.
//
// Build (on pod nick, CUDA 12.9, H100 sm_90a):
//   nvcc -O3 -std=c++20 -arch=sm_90a \
//     -DKITTENS_HOPPER \
//     -I /home/nickm/vllm-mega/vllm-rs/third_party/thunderkittens/include \
//     -I /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/tk \
//     /tmp/ferrite_gemv_smoke.cu -o /tmp/ferrite_gemv_smoke -lcuda
//
// Run: /tmp/ferrite_gemv_smoke

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
#include "ferrite_kernels/gemv_bf16.cuh"

namespace {

// Wave B defaults — mirrors FerriteConfig::phase3d(hidden_dim=2048,
// head_dim=64) for Llama-3.2-1B with matvec-floor ncw_ceiling.
//   - NUM_CONSUMER_WARPS = 4 so K_PER_WARP = K/NCW = 512, matching
//     the 16-row register-tile matvec budget.
//   - PAGE_SIZE = 16 * K_PER_WARP * 2 = 16384 so an `st_bf<16, 512>`
//     weight tile fits in one page.
//   - NUM_PAGES = 1 activation + NCW * STAGES weight = 9.
//   - SCRATCH_BYTES ≥ STAGES * 64 = 128 for the output ping-pong
//     (sv_fl<16> per stage); bump to 256 for headroom.
struct SmokeConfig {
    static constexpr int NUM_CONSUMER_WARPS      = 4;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 192;
    static constexpr int NUM_PAGES               = 9;
    static constexpr int PAGE_SIZE               = 16384;
    static constexpr int SCRATCH_BYTES           = 256;
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

constexpr int K    = 2048;
constexpr int N    = 64;
constexpr int TPB  = (SmokeConfig::NUM_CONSUMER_WARPS + 3) * 32;

using SS = ferrite::SharedState<SmokeConfig>;

// Wave B: NUM_PAGES * PAGE_SIZE = 9 * 16384 = 144 KB exceeds the 48 KB
// static-shmem cap on Hopper; need dynamic shmem with opt-in (same
// pattern the mega kernel uses). `extern __shared__` lets the launcher
// size the shmem via the third `<<<..., shmem_bytes, ...>>>` arg.
extern __shared__ __align__(128) uint8_t dynamic_smem[];

__global__ void gemv_smoke_kernel(
    const __nv_bfloat16* __restrict__ x,    // [K]
    const __nv_bfloat16* __restrict__ W,    // [N, K]
    __nv_bfloat16*       __restrict__ out   // [N]
) {
    SS& ss = *reinterpret_cast<SS*>(dynamic_smem);
    ferrite::init_shared_state<SmokeConfig>(ss);

    const int wid = kittens::warpid();
    if (wid < SmokeConfig::NUM_CONSUMER_WARPS) {
        ferrite::set_consumer_registers<SmokeConfig>();
        ferrite::ops::gemv_bf16::consumer<SmokeConfig, K, N>(
            ss, /*base_stage=*/0, wid);
    } else {
        ferrite::set_non_consumer_registers<SmokeConfig>();
        switch (wid - SmokeConfig::NUM_CONSUMER_WARPS) {
            case ferrite::kLoaderSlot:
                ferrite::ops::gemv_bf16::loader<SmokeConfig, K, N>(
                    x, W, ss, /*base_stage=*/0);
                break;
            case ferrite::kLauncherSlot:
                ferrite::ops::gemv_bf16::launcher<SmokeConfig, K, N>(
                    ss, /*base_stage=*/0);
                break;
            case ferrite::kStorerSlot:
                ferrite::ops::gemv_bf16::storer<SmokeConfig, K, N>(
                    out, ss, /*base_stage=*/0);
                break;
            default:
                break;
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

// Simple xorshift32 → uniform(-1, 1) as fp32, cast to bf16.
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
    std::vector<float>          x_f32(K);
    std::vector<float>          W_f32(N * K);
    std::vector<__nv_bfloat16>  x_bf16(K);
    std::vector<__nv_bfloat16>  W_bf16(N * K);

    uint32_t s = 0xF00Du;
    for (int i = 0; i < K;       ++i) { x_f32[i] = rand_u(s); x_bf16[i] = __float2bfloat16(x_f32[i]); }
    for (int i = 0; i < N * K;   ++i) { W_f32[i] = rand_u(s); W_bf16[i] = __float2bfloat16(W_f32[i]); }

    // CPU reference — quantize back through bf16 round-trip so
    // tolerance reflects the kernel's own bf16 loads.
    std::vector<float> out_ref(N, 0.0f);
    for (int row = 0; row < N; ++row) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float xv = __bfloat162float(x_bf16[k]);
            float wv = __bfloat162float(W_bf16[row * K + k]);
            acc += xv * wv;
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

    // Wave B: gemv produces 16 output rows per CTA via register-tile
    // matvec. Grid = ceil(N / 16) — N=64 gives 4 blocks. Persistent-
    // thread loop inside the kernel handles any grid ≤ N/16.
    constexpr size_t SMEM_BYTES = sizeof(SS);
    CUDA_CHECK(cudaFuncSetAttribute(
        gemv_smoke_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_BYTES));
    dim3 grid(N / 16, 1, 1);
    dim3 block(TPB, 1, 1);
    gemv_smoke_kernel<<<grid, block, SMEM_BYTES, 0>>>(d_x, d_W, d_out);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launch_err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out_bf16(N);
    CUDA_CHECK(cudaMemcpy(out_bf16.data(), d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

    float max_abs = 0.0f, sum_ref_sq = 0.0f, sum_diff_sq = 0.0f;
    int   mismatches = 0;
    // bf16 tolerance: output is cast to bf16 so 1 ULP at magnitude
    // ~20-40 is ~0.15. Observed max_abs from a correct run is
    // ~0.07; this tolerance absorbs 1-ULP bf16 packing plus the
    // per-thread vs sequential summation-order diff.
    const float TOL = 0.2f;
    for (int row = 0; row < N; ++row) {
        float got  = __bfloat162float(out_bf16[row]);
        float ref  = out_ref[row];
        float diff = std::fabs(got - ref);
        if (diff > TOL) ++mismatches;
        if (diff > max_abs) max_abs = diff;
        sum_ref_sq  += ref * ref;
        sum_diff_sq += diff * diff;
    }
    float rel_l2 = std::sqrt(sum_diff_sq / (sum_ref_sq > 0 ? sum_ref_sq : 1.0f));

    std::printf("N=%d K=%d max_abs=%.4f rel_l2=%.4f mismatches(>%.2f)=%d\n",
                N, K, max_abs, rel_l2, TOL, mismatches);
    if (mismatches == 0) {
        std::printf("ok: gemv_bf16 matches CPU reference within tolerance\n");
    } else {
        std::printf("sample rows:\n");
        for (int row = 0; row < std::min(8, N); ++row) {
            std::printf("  row=%d got=%.4f ref=%.4f diff=%.4f\n",
                        row,
                        __bfloat162float(out_bf16[row]),
                        out_ref[row],
                        std::fabs(__bfloat162float(out_bf16[row]) - out_ref[row]));
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_out));
    return mismatches == 0 ? 0 : 3;
}
