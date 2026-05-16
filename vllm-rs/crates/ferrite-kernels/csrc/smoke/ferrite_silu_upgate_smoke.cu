// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK silu_upgate standalone smoke test — Phase 3f part 2g-ii.
//
// Drives the four walker roles of `silu_upgate.cuh` end-to-end for a
// single decode token against a CPU reference that does:
//   gate = W[row,         :] @ x
//   up   = W[I + row,     :] @ x
//   out[row] = silu(gate) * up = (gate * sigmoid(gate)) * up
//
// Grid mirrors the header's layout: `dim3(INTERMEDIATE_DIM)`. Each CTA
// owns one output row. Weights live in one packed
// `[2*INTERMEDIATE_DIM, HIDDEN_DIM]` buffer (gate rows in the first
// half, up rows in the second), matching the host-side
// `FusedGateUpSiluMulImpl` `LinearLayer` claim.
//
// Does not go through codegen — this file reproduces the minimum
// substrate a codegen'd .cu would emit so we isolate "did I write the
// op header correctly?" from "did I wire codegen correctly?". Same
// policy as every other `-ii` pod smoke (gemv, fused_add_rms_norm,
// embed, fused_qkv_rope_cache, attention_partial).
//
// Variant: HIDDEN_DIM=2048, INTERMEDIATE_DIM=8192 (Llama-3.2-1B
// decode), NUM_CONSUMER_WARPS=4, NUM_PAGES=3 (act, gate_w, up_w).
//
// Build (on pod nick, CUDA 12.9, H100 sm_90a):
//   nvcc -O3 -std=c++20 \
//     -gencode arch=compute_90a,code=sm_90a \
//     -DKITTENS_HOPPER --extended-lambda --expt-relaxed-constexpr \
//     -I /home/nickm/vllm-mega/vllm-rs/third_party/thunderkittens/include \
//     -I /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/tk \
//     /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/smoke/ferrite_silu_upgate_smoke.cu \
//     -o /tmp/ferrite_silu_upgate_smoke -lcuda
//
// Run: /tmp/ferrite_silu_upgate_smoke

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
#include "ferrite_kernels/silu_upgate.cuh"

namespace {

// ---- Model dims (llama-3.2-1B decode shape).
static constexpr int HIDDEN_DIM       = 2048;
static constexpr int INTERMEDIATE_DIM = 8192;
static constexpr int NUM_TOKENS       = 1;

// ---- FerriteConfig — 3 pages each sized HIDDEN_DIM*2 bytes = 4096.
// Matches gemv_smoke's register allocation knobs; NUM_PAGES bumps to
// 3 because the op needs (act, gate_row, up_row) simultaneously.
struct SmokeConfig {
    static constexpr int NUM_CONSUMER_WARPS      = 4;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 192;
    static constexpr int NUM_PAGES               = 3;
    static constexpr int PAGE_SIZE               = 4096;  // HIDDEN_DIM * 2
    static constexpr int SCRATCH_BYTES           = 256;
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

constexpr int TPB = (SmokeConfig::NUM_CONSUMER_WARPS + 3) * 32;

using SS = ferrite::SharedState<SmokeConfig>;

__global__ void silu_upgate_smoke_kernel(
    const __nv_bfloat16* __restrict__ x,           // [HIDDEN_DIM]
    const __nv_bfloat16* __restrict__ W_gate_up,   // [2*INTERMEDIATE_DIM, HIDDEN_DIM]
    __nv_bfloat16*       __restrict__ out          // [INTERMEDIATE_DIM]
) {
    __shared__ SS ss;
    ferrite::init_shared_state<SmokeConfig>(ss);

    const int wid = kittens::warpid();
    if (wid < SmokeConfig::NUM_CONSUMER_WARPS) {
        ferrite::set_consumer_registers<SmokeConfig>();
        ferrite::ops::silu_upgate::consumer<
            SmokeConfig, HIDDEN_DIM, INTERMEDIATE_DIM, NUM_TOKENS>(
                ss, /*base_stage=*/0, wid);
    } else {
        ferrite::set_non_consumer_registers<SmokeConfig>();
        switch (wid - SmokeConfig::NUM_CONSUMER_WARPS) {
            case ferrite::kLoaderSlot:
                ferrite::ops::silu_upgate::loader<
                    SmokeConfig, HIDDEN_DIM, INTERMEDIATE_DIM, NUM_TOKENS>(
                        x, W_gate_up, ss, /*base_stage=*/0);
                break;
            case ferrite::kLauncherSlot:
                ferrite::ops::silu_upgate::launcher<
                    SmokeConfig, HIDDEN_DIM, INTERMEDIATE_DIM, NUM_TOKENS>(
                        ss, /*base_stage=*/0);
                break;
            case ferrite::kStorerSlot:
                ferrite::ops::silu_upgate::storer<
                    SmokeConfig, HIDDEN_DIM, INTERMEDIATE_DIM, NUM_TOKENS>(
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
    // ---- Host tensors.
    std::vector<__nv_bfloat16> h_x        (HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_W_gate_up(size_t(2) * INTERMEDIATE_DIM * HIDDEN_DIM);

    uint32_t s = 0xB16D00D5u;
    // Scale inputs so gate/up partial sums land in silu's nonlinear
    // band. With `x`, `W` uniform on [-scale, +scale], the variance
    // of each `x*w` term is `(scale^2/3)^2 = scale^4 / 9`; a K=2048
    // dot product has stddev `sqrt(2048 * scale^4 / 9) = scale^2 *
    // 15.1`. scale=0.2 → stddev ≈ 0.6 (values land in roughly [-2,
    // 2]), which crosses silu's inflection at |g|~1 and exercises
    // both the linear-near-zero and saturating-far-from-zero regimes.
    // At lower scales the answers fit in bf16 resolution exactly
    // (max_abs = 0) and the test can't see reduction-order drift;
    // at higher scales silu saturates on most rows and hides bugs
    // in the nonlinearity.
    for (auto& v : h_x)         v = __float2bfloat16_rn(0.2f * rand_u(s));
    for (auto& v : h_W_gate_up) v = __float2bfloat16_rn(0.2f * rand_u(s));

    // ---- CPU reference — quantize back through bf16 round-trip so
    // tolerance reflects the kernel's own bf16 loads.
    std::vector<__nv_bfloat16> h_out_ref(INTERMEDIATE_DIM,
                                          __float2bfloat16_rn(0.0f));
    for (int row = 0; row < INTERMEDIATE_DIM; ++row) {
        float acc_gate = 0.0f;
        float acc_up   = 0.0f;
        for (int k = 0; k < HIDDEN_DIM; ++k) {
            float xv  = __bfloat162float(h_x[k]);
            float wgv = __bfloat162float(h_W_gate_up[
                size_t(row) * HIDDEN_DIM + k]);
            float wuv = __bfloat162float(h_W_gate_up[
                size_t(INTERMEDIATE_DIM + row) * HIDDEN_DIM + k]);
            acc_gate += xv * wgv;
            acc_up   += xv * wuv;
        }
        float sigmoid_g = 1.0f / (1.0f + std::exp(-acc_gate));
        float out_f     = (acc_gate * sigmoid_g) * acc_up;
        h_out_ref[row]  = __float2bfloat16_rn(out_f);
    }

    // ---- Device buffers.
    __nv_bfloat16 *d_x = nullptr, *d_W = nullptr, *d_out = nullptr;
    const size_t x_bytes   = size_t(HIDDEN_DIM) * sizeof(__nv_bfloat16);
    const size_t w_bytes   = size_t(2) * INTERMEDIATE_DIM * HIDDEN_DIM *
                             sizeof(__nv_bfloat16);
    const size_t out_bytes = size_t(INTERMEDIATE_DIM) * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_x,   x_bytes));
    CUDA_CHECK(cudaMalloc(&d_W,   w_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(),         x_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W_gate_up.data(), w_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, out_bytes));

    // ---- Launch.
    dim3 grid(INTERMEDIATE_DIM, 1, 1);
    dim3 block(TPB, 1, 1);
    silu_upgate_smoke_kernel<<<grid, block, 0, 0>>>(d_x, d_W, d_out);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launch_err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_out_got(INTERMEDIATE_DIM);
    CUDA_CHECK(cudaMemcpy(h_out_got.data(), d_out, out_bytes,
                          cudaMemcpyDeviceToHost));

    // ---- Compare. Tolerance rationale at scale=0.2: typical gate/up
    // partial magnitudes are ~0.6, products land in [~0, 4]. bf16 ULP
    // at magnitude 4 is ~0.03; the kernel sums per-warp (32 threads →
    // shfl_xor tree → lane-0 publish → cross-warp sum in warp 0 lane
    // 0) vs the CPU's sequential fp32 sum, so reduction-order drift
    // adds up to a few ULPs on top. The `__expf` path in the kernel
    // vs `std::exp` in the CPU ref is the other drift source; the
    // absolute error is bounded by `|up| * |sigmoid_kernel -
    // sigmoid_ref|` which is a few times 1e-6. 0.05 catches honest
    // bugs while absorbing expected drift — a factor of ~10 over what
    // a passing run actually sees.
    const float TOL = 0.05f;
    float max_abs = 0.0f, sum_ref_sq = 0.0f, sum_diff_sq = 0.0f;
    int   mismatches = 0;
    int   first_mismatch = -1;
    float first_got = 0.0f, first_ref = 0.0f;
    for (int row = 0; row < INTERMEDIATE_DIM; ++row) {
        float got  = __bfloat162float(h_out_got[row]);
        float ref  = __bfloat162float(h_out_ref[row]);
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
    float rel_l2 = std::sqrt(sum_diff_sq /
                             (sum_ref_sq > 0 ? sum_ref_sq : 1.0f));

    std::printf("hidden_dim=%d intermediate_dim=%d ncw=%d\n",
                HIDDEN_DIM, INTERMEDIATE_DIM,
                SmokeConfig::NUM_CONSUMER_WARPS);
    std::printf("max_abs=%.4f rel_l2=%.4f mismatches(>%.2f)=%d\n",
                max_abs, rel_l2, TOL, mismatches);
    // Always print a few sample rows so a pass doesn't silently hide
    // an all-zero output regression. CPU ref magnitudes at this
    // input scale land in the ~[0, 50] band; a kernel that wrote
    // zeros would print `got=0.0000` here.
    std::printf("sample rows (first 4):\n");
    for (int row = 0; row < 4; ++row) {
        std::printf("  row=%d got=%.4f ref=%.4f\n",
                    row,
                    __bfloat162float(h_out_got[row]),
                    __bfloat162float(h_out_ref[row]));
    }
    if (mismatches == 0) {
        std::printf("ok: silu_upgate matches CPU reference within tolerance\n");
    } else {
        std::printf("first mismatch at row=%d got=%.4f ref=%.4f diff=%.4f\n",
                    first_mismatch, first_got, first_ref,
                    std::fabs(first_got - first_ref));
        std::printf("sample rows:\n");
        for (int row = 0; row < 8; ++row) {
            std::printf("  row=%d got=%.4f ref=%.4f diff=%.4f\n",
                        row,
                        __bfloat162float(h_out_got[row]),
                        __bfloat162float(h_out_ref[row]),
                        std::fabs(__bfloat162float(h_out_got[row]) -
                                  __bfloat162float(h_out_ref[row])));
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_out));
    return mismatches == 0 ? 0 : 3;
}
