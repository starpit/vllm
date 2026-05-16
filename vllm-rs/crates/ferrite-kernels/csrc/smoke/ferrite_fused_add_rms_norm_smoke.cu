// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK FusedAddRmsNorm standalone smoke test — Phase 3f part 2a.
//
// Exercises the ferrite-owned fused_add_rms_norm op against a CPU
// reference. The op mutates two slots in place:
//   residual_out = bf16(delta_in + residual_in)
//   delta_out    = bf16(rms_norm(residual_out) * weight)
//
// Why: the host-interpreter `FusedAddRmsNormImpl` claims both the
// RmsNorm and the preceding Add, and declares `output_alias` so
// `delta_slot` receives the rmsnorm output and `residual_slot`
// receives the fused-add sum. This `.cu` mirrors what the walker
// will emit for a single FusedAddRmsNorm op (base_stage=0,
// NUM_PAGES=3 = delta + residual + weight), against a minimal
// 3-page SharedState + 4-warp-role dispatch.
//
// Build (on pod, CUDA 12.9, H100 sm_90a):
//   nvcc -O3 -std=c++20 -arch=sm_90a \
//     -DKITTENS_HOPPER --extended-lambda --expt-relaxed-constexpr \
//     -I /home/nickm/vllm-mega/vllm-rs/third_party/thunderkittens/include \
//     -I /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/tk \
//     /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/smoke/ferrite_fused_add_rms_norm_smoke.cu \
//     -o /tmp/ferrite_fused_add_rms_norm_smoke -lcuda
//
// Run: /tmp/ferrite_fused_add_rms_norm_smoke

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
#include "ferrite_kernels/fused_add_rms_norm.cuh"

namespace {

// ---- FerriteConfig — mirrors mega.rs::FerriteConfig::phase3d for
// HIDDEN_DIM=2048. NUM_PAGES=3 (delta + residual + weight).
struct FerriteConfig {
    static constexpr int NUM_CONSUMER_WARPS      = 4;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 192;
    static constexpr int NUM_PAGES               = 3;
    static constexpr int PAGE_SIZE               = 4096;  // HIDDEN_DIM * 2 rounded to 128
    static constexpr int SCRATCH_BYTES           = 256;
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

static constexpr int NUM_LAYERS       = 1;
static constexpr int HIDDEN_DIM       = 2048;
static constexpr int NUM_TOKENS       = 8;
static constexpr int NUM_ACT_SLOTS    = 2;
static constexpr int NUM_WEIGHT_ACCESSORS = 1;
static constexpr float RMS_NORM_EPS   = 1.0e-5f;
static constexpr int TPB              = (FerriteConfig::NUM_CONSUMER_WARPS + 3) * 32;

using SS = ferrite::SharedState<FerriteConfig>;

// ---- Walker bodies — mirror emit_cu_variant's emit for a single
// FusedAddRmsNorm op at base_stage=0. Slot 0 = delta, slot 1 =
// residual. Both are in+out in place. Weight at accessor index 0,
// layer 0.

__device__ __forceinline__ void consumer_body(
    __nv_bfloat16* const*        act_ptrs,
    const __nv_bfloat16* const*  weight_ptrs,
    SS& ss,
    int  warp_in_role
) {
    (void)act_ptrs; (void)weight_ptrs;
    ferrite::ops::fused_add_rms_norm::consumer<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        ss, /*base_stage=*/0, warp_in_role, /*eps=*/RMS_NORM_EPS);
}

__device__ __forceinline__ void loader_body(
    __nv_bfloat16* const*        act_ptrs,
    const __nv_bfloat16* const*  weight_ptrs,
    SS& ss
) {
    ferrite::ops::fused_add_rms_norm::loader<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        /*delta_in=*/act_ptrs[0],
        /*residual_in=*/act_ptrs[1],
        /*rms_weight=*/weight_ptrs[0u * NUM_LAYERS + 0u],
        ss, /*base_stage=*/0);
}

__device__ __forceinline__ void launcher_body(
    __nv_bfloat16* const*        act_ptrs,
    const __nv_bfloat16* const*  weight_ptrs,
    SS& ss
) {
    (void)act_ptrs; (void)weight_ptrs;
    ferrite::ops::fused_add_rms_norm::launcher<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        ss, /*base_stage=*/0);
}

__device__ __forceinline__ void storer_body(
    __nv_bfloat16* const*        act_ptrs,
    const __nv_bfloat16* const*  weight_ptrs,
    SS& ss
) {
    (void)weight_ptrs;
    ferrite::ops::fused_add_rms_norm::storer<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        /*delta_out=*/act_ptrs[0],
        /*residual_out=*/act_ptrs[1],
        ss, /*base_stage=*/0);
}

__global__ void fused_add_rms_norm_smoke_kernel(
    __nv_bfloat16* const*        __restrict__ act_ptrs,
    const __nv_bfloat16* const*  __restrict__ weight_ptrs
) {
    __shared__ SS ss;
    ferrite::init_shared_state<FerriteConfig>(ss);

    const int wid = kittens::warpid();
    if (wid < FerriteConfig::NUM_CONSUMER_WARPS) {
        ferrite::set_consumer_registers<FerriteConfig>();
        consumer_body(act_ptrs, weight_ptrs, ss, wid);
    } else {
        ferrite::set_non_consumer_registers<FerriteConfig>();
        switch (wid - FerriteConfig::NUM_CONSUMER_WARPS) {
            case ferrite::kLoaderSlot:   loader_body  (act_ptrs, weight_ptrs, ss); break;
            case ferrite::kLauncherSlot: launcher_body(act_ptrs, weight_ptrs, ss); break;
            case ferrite::kStorerSlot:   storer_body  (act_ptrs, weight_ptrs, ss); break;
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

// CPU reference:
//   residual_out[i] = bf16(delta[i] + residual[i])
//   delta_out[i]    = bf16(rms_norm(residual_out) * weight[i])
// Rms-norm pass reads the bf16-rounded residual_out (matching the
// kernel, which quantizes the sum to bf16 in the residual page
// before reading it for the rms pass).
void fused_add_rms_norm_row_ref(
    const __nv_bfloat16* delta_in,
    const __nv_bfloat16* residual_in,
    const __nv_bfloat16* weight,
    __nv_bfloat16*       delta_out,
    __nv_bfloat16*       residual_out,
    int                  hidden_dim,
    float                eps
) {
    // Pass 1: sum → bf16 residual_out.
    for (int i = 0; i < hidden_dim; ++i) {
        float d = __bfloat162float(delta_in[i]);
        float r = __bfloat162float(residual_in[i]);
        residual_out[i] = __float2bfloat16_rn(d + r);
    }
    // Pass 2: rms of residual_out.
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        float v = __bfloat162float(residual_out[i]);
        sum_sq += v * v;
    }
    float rms = std::sqrt(sum_sq / float(hidden_dim) + eps);
    float inv = 1.0f / rms;
    for (int i = 0; i < hidden_dim; ++i) {
        float v = __bfloat162float(residual_out[i]);
        float w = __bfloat162float(weight[i]);
        delta_out[i] = __float2bfloat16_rn(v * inv * w);
    }
}

} // namespace

int main() {
    // ---- Allocate + populate host tensors.
    std::vector<__nv_bfloat16> h_delta   (NUM_TOKENS * HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_residual(NUM_TOKENS * HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_weight  (HIDDEN_DIM);

    uint32_t s = 0xD00FFEE7u;
    // Delta and residual both random in [-1, 1] — sum stays in a
    // range where bf16 quant is reasonably tight.
    for (auto& x : h_delta)    x = __float2bfloat16_rn(rand_u(s));
    for (auto& x : h_residual) x = __float2bfloat16_rn(rand_u(s));
    // Weight near 1.0 ± small.
    for (auto& x : h_weight)   x = __float2bfloat16_rn(1.0f + 0.1f * rand_u(s));

    // ---- CPU reference.
    std::vector<__nv_bfloat16> h_ref_delta   (NUM_TOKENS * HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_ref_residual(NUM_TOKENS * HIDDEN_DIM);
    for (int row = 0; row < NUM_TOKENS; ++row) {
        fused_add_rms_norm_row_ref(
            &h_delta   [row * HIDDEN_DIM],
            &h_residual[row * HIDDEN_DIM],
            h_weight.data(),
            &h_ref_delta   [row * HIDDEN_DIM],
            &h_ref_residual[row * HIDDEN_DIM],
            HIDDEN_DIM, RMS_NORM_EPS);
    }

    // ---- Device buffers.
    __nv_bfloat16 *d_delta = nullptr, *d_residual = nullptr, *d_weight = nullptr;
    const size_t act_bytes    = size_t(NUM_TOKENS) * HIDDEN_DIM * sizeof(__nv_bfloat16);
    const size_t weight_bytes = size_t(HIDDEN_DIM) * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_delta,    act_bytes));
    CUDA_CHECK(cudaMalloc(&d_residual, act_bytes));
    CUDA_CHECK(cudaMalloc(&d_weight,   weight_bytes));
    CUDA_CHECK(cudaMemcpy(d_delta,    h_delta.data(),    act_bytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), act_bytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight,   h_weight.data(),   weight_bytes, cudaMemcpyHostToDevice));

    // ---- Pool-ABI pointer arrays on device.
    __nv_bfloat16* h_act_ptrs[NUM_ACT_SLOTS] = { d_delta, d_residual };
    const __nv_bfloat16* h_w_ptrs[NUM_WEIGHT_ACCESSORS * NUM_LAYERS] = { d_weight };
    __nv_bfloat16**       d_act_ptrs = nullptr;
    const __nv_bfloat16** d_w_ptrs   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_act_ptrs, sizeof(h_act_ptrs)));
    CUDA_CHECK(cudaMalloc(&d_w_ptrs,   sizeof(h_w_ptrs)));
    CUDA_CHECK(cudaMemcpy(d_act_ptrs, h_act_ptrs, sizeof(h_act_ptrs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_ptrs,   h_w_ptrs,   sizeof(h_w_ptrs),   cudaMemcpyHostToDevice));

    // ---- Launch.
    dim3 grid(NUM_TOKENS, 1, 1);
    dim3 block(TPB, 1, 1);
    fused_add_rms_norm_smoke_kernel<<<grid, block, 0, 0>>>(d_act_ptrs, d_w_ptrs);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launch_err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Read back both in-place outputs.
    std::vector<__nv_bfloat16> h_got_delta   (NUM_TOKENS * HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_got_residual(NUM_TOKENS * HIDDEN_DIM);
    CUDA_CHECK(cudaMemcpy(h_got_delta.data(),    d_delta,    act_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_got_residual.data(), d_residual, act_bytes, cudaMemcpyDeviceToHost));

    auto compare = [&](const char* label, float tol,
                       const std::vector<__nv_bfloat16>& got,
                       const std::vector<__nv_bfloat16>& ref) -> int {
        float max_abs = 0.0f, sum_ref_sq = 0.0f, sum_diff_sq = 0.0f;
        int mismatches = 0;
        int per_row_errors[NUM_TOKENS] = {};
        for (int row = 0; row < NUM_TOKENS; ++row) {
            for (int i = 0; i < HIDDEN_DIM; ++i) {
                float gv = __bfloat162float(got[row * HIDDEN_DIM + i]);
                float rv = __bfloat162float(ref[row * HIDDEN_DIM + i]);
                float diff = std::fabs(gv - rv);
                if (diff > tol) { ++mismatches; ++per_row_errors[row]; }
                if (diff > max_abs) max_abs = diff;
                sum_ref_sq  += rv * rv;
                sum_diff_sq += diff * diff;
            }
        }
        float rel_l2 = std::sqrt(sum_diff_sq / (sum_ref_sq > 0 ? sum_ref_sq : 1.0f));
        std::printf("%s: max_abs=%.6f rel_l2=%.6f mismatches(>%.2f)=%d\n",
                    label, max_abs, rel_l2, tol, mismatches);
        std::printf("%s per_row_errors:", label);
        for (int r = 0; r < NUM_TOKENS; ++r) std::printf(" r%d=%d", r, per_row_errors[r]);
        std::printf("\n");
        if (mismatches != 0) {
            std::printf("%s sample (row 0, first 4 elements):\n", label);
            for (int i = 0; i < 4; ++i) {
                float gv = __bfloat162float(got[0 * HIDDEN_DIM + i]);
                float rv = __bfloat162float(ref[0 * HIDDEN_DIM + i]);
                std::printf("  i=%d got=%.6f ref=%.6f diff=%.6f\n",
                            i, gv, rv, std::fabs(gv - rv));
            }
        }
        return mismatches;
    };

    std::printf("num_tokens=%d hidden_dim=%d\n", NUM_TOKENS, HIDDEN_DIM);
    // Residual (the fused-add output) should be very close — just a
    // bf16-quantized sum. Delta (the rms-normed output) has the same
    // numerics tolerance as the standalone rms_norm smoke.
    int m_r = compare("residual (delta+residual)", 0.01f, h_got_residual, h_ref_residual);
    int m_d = compare("delta (rms_norm(sum)*W)",   0.03f, h_got_delta,    h_ref_delta);
    int mismatches = m_r + m_d;
    if (mismatches == 0) {
        std::printf("ok: fused_add_rms_norm matches CPU reference\n");
    }

    CUDA_CHECK(cudaFree(d_delta));
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_act_ptrs));
    CUDA_CHECK(cudaFree(d_w_ptrs));
    return mismatches == 0 ? 0 : 3;
}
