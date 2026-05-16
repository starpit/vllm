// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK pool-ABI standalone smoke test — Phase 3e exit gate.
//
// Phase 3d introduced the pool ABI: the codegen'd kernel takes two
// pointer-of-pointers kernel args —
//   __nv_bfloat16* const*        act_ptrs
//   const __nv_bfloat16* const*  weight_ptrs
// — and reads `act_ptrs[s]` / `weight_ptrs[w * NUM_LAYERS + l]` for
// each op's tile pointers. Phase 3e proves this shape runs clean on
// H100 end-to-end before the op set grows.
//
// This `.cu` mirrors `emit_cu_variant`'s output for a 2-op
// rms_norm→rms_norm variant (two distinct weight accessors, three
// distinct activation slots). Why not rms_norm+gemv, as Phase 3d's
// "next" bullet proposed? — rms_norm's natural grid is
// `dim3(NUM_TOKENS)` (CTA per token row) and gemv's is `dim3(N)`
// (CTA per output element). The walker currently emits
// `dim3(NUM_TOKENS)` unconditionally, which only composes cleanly
// for token-parallel ops. A mixed rms_norm+gemv variant would need
// the walker to unify its grid shape across ops first — a separate
// slice from the pool-ABI validation this file is scoped to. Two
// rms_norm ops back-to-back share grid requirements cleanly and
// exercise all the pool-ABI concerns:
//
//   - act_ptrs[0..2] indexed: input, out_a, out_b.
//   - weight_ptrs[0 * NUM_LAYERS + 0] and weight_ptrs[1 * NUM_LAYERS + 0]
//     for two distinct weight accessors.
//   - base_stage=0 for op 0, base_stage=2 for op 1 (each op claims
//     two pages: input + weight).
//   - NUM_PAGES=4, NUM_ACT_SLOTS=3, NUM_WEIGHT_ACCESSORS=2.
//   - Sequential walker-body composition: two rms_norm::{loader,
//     consumer,launcher,storer} calls per role.
//
// Schedule: both ops read the same input slot 0 with different
// weights and write disjoint output slots (1 and 2). They are
// intentionally data-independent — chained dependency between op
// N's storer and op N+1's loader would require a cross-op gmem
// barrier the walker doesn't emit yet (that's Phase 4 pipelining
// territory; an initial pool-ABI variant that tried op0→op1
// chaining stores "all zeros" into slot 2 because the loader warp
// races ahead of the storer warp on the shared gmem tensor, before
// we've even asked a consumer barrier to gate it). Two independent
// ops prove the pool ABI cleanly without conflating it with a
// synchronization hole that belongs in a later plan phase.
//
// Variant: NUM_TOKENS=8, HIDDEN_DIM=2048, NUM_LAYERS=1 (single
// "layer" — we only use l=0 for both accessors). bf16 in/out, fp32
// accumulator, random inputs. Tolerance mirrors Phase 2's rms_norm
// m=8 run (~0.01 observed; gate at 0.03).
//
// Build (on pod nick, CUDA 12.9, H100 sm_90a):
//   nvcc -O3 -std=c++20 -arch=sm_90a \
//     -DKITTENS_HOPPER --extended-lambda --expt-relaxed-constexpr \
//     -I /home/nickm/vllm-mega/vllm-rs/third_party/thunderkittens/include \
//     -I /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/tk \
//     /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/smoke/ferrite_pool_abi_smoke.cu \
//     -o /tmp/ferrite_pool_abi_smoke -lcuda
//
// Run: /tmp/ferrite_pool_abi_smoke

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
#include "ferrite_kernels/rms_norm.cuh"

namespace {

// ---- FerriteConfig — mirrors mega.rs::FerriteConfig::phase3d for
// HIDDEN_DIM=2048 (INTERMEDIATE_DIM irrelevant here; max-tile is
// HIDDEN_DIM). NUM_PAGES=4 = sum of both rms_norm ops' page counts
// (2 each).
struct FerriteConfig {
    static constexpr int NUM_CONSUMER_WARPS      = 4;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 192;
    static constexpr int NUM_PAGES               = 4;
    static constexpr int PAGE_SIZE               = 4096;  // HIDDEN_DIM * 2 rounded to 128
    static constexpr int SCRATCH_BYTES           = 256;
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

// ---- Variant-fixed constexprs — emitted verbatim by the walker.
static constexpr int NUM_LAYERS       = 1;
static constexpr int HIDDEN_DIM       = 2048;
static constexpr int NUM_TOKENS       = 8;
static constexpr int NUM_ACT_SLOTS    = 3;
static constexpr int NUM_WEIGHT_ACCESSORS = 2;
static constexpr float RMS_NORM_EPS   = 1.0e-5f;
static constexpr int TPB              = (FerriteConfig::NUM_CONSUMER_WARPS + 3) * 32;

using SS = ferrite::SharedState<FerriteConfig>;

// ---- Walker bodies — mirror emit_cu_variant's output. Each role
// body is the straight-line concatenation of two rms_norm op
// snippets (op #0 at base_stage=0, op #1 at base_stage=2). Same
// form that `variant_cpp::emit_rms_norm` produces, except the slot
// and weight indices are baked in for this smoke variant.

__device__ __forceinline__ void consumer_body(
    __nv_bfloat16* const*        act_ptrs,
    const __nv_bfloat16* const*  weight_ptrs,
    SS& ss,
    int  warp_in_role
) {
    (void)act_ptrs; (void)weight_ptrs;
    // ---- op: #0 RmsNorm  (base_stage=0, pages=2) ----
    ferrite::ops::rms_norm::consumer<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        ss, /*base_stage=*/0, warp_in_role, /*eps=*/RMS_NORM_EPS);
    // ---- op: #1 RmsNorm  (base_stage=2, pages=2) ----
    ferrite::ops::rms_norm::consumer<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        ss, /*base_stage=*/2, warp_in_role, /*eps=*/RMS_NORM_EPS);
}

__device__ __forceinline__ void loader_body(
    __nv_bfloat16* const*        act_ptrs,
    const __nv_bfloat16* const*  weight_ptrs,
    SS& ss
) {
    // ---- op: #0 RmsNorm  (base_stage=0, pages=2) ----
    // in_slot=0 (input), weight_accessor=0 layer=0, out_slot=1.
    ferrite::ops::rms_norm::loader<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        /*rms_input=*/act_ptrs[0],
        /*rms_weight=*/weight_ptrs[0u * NUM_LAYERS + 0u],
        ss, /*base_stage=*/0);
    // ---- op: #1 RmsNorm  (base_stage=2, pages=2) ----
    // in_slot=0 (same input — data-independent from op 0), weight
    // accessor=1 layer=0, out_slot=2.
    ferrite::ops::rms_norm::loader<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        /*rms_input=*/act_ptrs[0],
        /*rms_weight=*/weight_ptrs[1u * NUM_LAYERS + 0u],
        ss, /*base_stage=*/2);
}

__device__ __forceinline__ void launcher_body(
    __nv_bfloat16* const*        act_ptrs,
    const __nv_bfloat16* const*  weight_ptrs,
    SS& ss
) {
    (void)act_ptrs; (void)weight_ptrs;
    // ---- op: #0 RmsNorm ----
    ferrite::ops::rms_norm::launcher<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        ss, /*base_stage=*/0);
    // ---- op: #1 RmsNorm ----
    ferrite::ops::rms_norm::launcher<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        ss, /*base_stage=*/2);
}

__device__ __forceinline__ void storer_body(
    __nv_bfloat16* const*        act_ptrs,
    const __nv_bfloat16* const*  weight_ptrs,
    SS& ss
) {
    (void)weight_ptrs;
    // ---- op: #0 RmsNorm  (base_stage=0, pages=2) ----
    // out_slot=1.
    ferrite::ops::rms_norm::storer<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        /*rms_output=*/act_ptrs[1],
        ss, /*base_stage=*/0);
    // ---- op: #1 RmsNorm  (base_stage=2, pages=2) ----
    // out_slot=2.
    ferrite::ops::rms_norm::storer<FerriteConfig, HIDDEN_DIM, NUM_TOKENS>(
        /*rms_output=*/act_ptrs[2],
        ss, /*base_stage=*/2);
}

// ---- Kernel — identical shape to `emit_cu_variant`'s emit.
__global__ void pool_abi_smoke_kernel(
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

// CPU reference — rms_norm applied once per row. Quantizes through
// bf16 on both input reads and output packing to mirror the kernel's
// own numeric behaviour.
void rms_norm_row_ref(
    const __nv_bfloat16* in_row,
    const __nv_bfloat16* weight,
    __nv_bfloat16*       out_row,
    int                  hidden_dim,
    float                eps
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        float v = __bfloat162float(in_row[i]);
        sum_sq += v * v;
    }
    float rms = std::sqrt(sum_sq / float(hidden_dim) + eps);
    float inv = 1.0f / rms;
    for (int i = 0; i < hidden_dim; ++i) {
        float v = __bfloat162float(in_row[i]);
        float w = __bfloat162float(weight[i]);
        out_row[i] = __float2bfloat16_rn(v * inv * w);
    }
}

} // namespace

int main() {
    // ---- Allocate + populate host tensors. ----
    std::vector<__nv_bfloat16> h_input  (NUM_TOKENS * HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_weight0(HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_weight1(HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_inter  (NUM_TOKENS * HIDDEN_DIM, __float2bfloat16_rn(0.0f));
    std::vector<__nv_bfloat16> h_output (NUM_TOKENS * HIDDEN_DIM, __float2bfloat16_rn(0.0f));

    uint32_t s = 0xC0FFEEu;
    for (auto& x : h_input)   x = __float2bfloat16_rn(rand_u(s));
    // Weights near 1.0 ± small — matches a real Llama rms-norm
    // weight distribution more closely than uniform(-1, 1).
    for (auto& x : h_weight0) x = __float2bfloat16_rn(1.0f + 0.1f * rand_u(s));
    for (auto& x : h_weight1) x = __float2bfloat16_rn(1.0f + 0.1f * rand_u(s));

    // ---- CPU reference: both ops read the same input with
    // different weights. op 0 → slot 1 (h_ref_out_a), op 1 → slot 2
    // (h_ref_out_b).
    std::vector<__nv_bfloat16> h_ref_out_a(NUM_TOKENS * HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_ref_out_b(NUM_TOKENS * HIDDEN_DIM);
    for (int row = 0; row < NUM_TOKENS; ++row) {
        rms_norm_row_ref(&h_input[row * HIDDEN_DIM], h_weight0.data(),
                         &h_ref_out_a[row * HIDDEN_DIM], HIDDEN_DIM, RMS_NORM_EPS);
        rms_norm_row_ref(&h_input[row * HIDDEN_DIM], h_weight1.data(),
                         &h_ref_out_b[row * HIDDEN_DIM], HIDDEN_DIM, RMS_NORM_EPS);
    }

    // ---- Allocate + upload device buffers. ----
    __nv_bfloat16 *d_slot0 = nullptr, *d_slot1 = nullptr, *d_slot2 = nullptr;
    __nv_bfloat16 *d_w0 = nullptr, *d_w1 = nullptr;
    const size_t act_bytes    = size_t(NUM_TOKENS) * HIDDEN_DIM * sizeof(__nv_bfloat16);
    const size_t weight_bytes = size_t(HIDDEN_DIM) * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_slot0, act_bytes));
    CUDA_CHECK(cudaMalloc(&d_slot1, act_bytes));
    CUDA_CHECK(cudaMalloc(&d_slot2, act_bytes));
    CUDA_CHECK(cudaMalloc(&d_w0,    weight_bytes));
    CUDA_CHECK(cudaMalloc(&d_w1,    weight_bytes));
    CUDA_CHECK(cudaMemcpy(d_slot0, h_input.data(),   act_bytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot1, h_inter.data(),   act_bytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot2, h_output.data(),  act_bytes,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w0,    h_weight0.data(), weight_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1,    h_weight1.data(), weight_bytes, cudaMemcpyHostToDevice));

    // ---- Stage pool-ABI pointer arrays on device. Walker expects
    // a device-resident act_ptrs[NUM_ACT_SLOTS] and
    // weight_ptrs[NUM_WEIGHT_ACCESSORS * NUM_LAYERS] before launch.
    __nv_bfloat16* h_act_ptrs[NUM_ACT_SLOTS] = { d_slot0, d_slot1, d_slot2 };
    const __nv_bfloat16* h_w_ptrs[NUM_WEIGHT_ACCESSORS * NUM_LAYERS] = { d_w0, d_w1 };
    __nv_bfloat16**       d_act_ptrs = nullptr;
    const __nv_bfloat16** d_w_ptrs   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_act_ptrs, sizeof(h_act_ptrs)));
    CUDA_CHECK(cudaMalloc(&d_w_ptrs,   sizeof(h_w_ptrs)));
    CUDA_CHECK(cudaMemcpy(d_act_ptrs, h_act_ptrs, sizeof(h_act_ptrs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_ptrs,   h_w_ptrs,   sizeof(h_w_ptrs),   cudaMemcpyHostToDevice));

    // ---- Launch. ----
    dim3 grid(NUM_TOKENS, 1, 1);
    dim3 block(TPB, 1, 1);
    pool_abi_smoke_kernel<<<grid, block, 0, 0>>>(d_act_ptrs, d_w_ptrs);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launch_err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Read back slot 1 (op 0's output) and slot 2 (op 1's output). ----
    std::vector<__nv_bfloat16> h_got_a(NUM_TOKENS * HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_got_b(NUM_TOKENS * HIDDEN_DIM);
    CUDA_CHECK(cudaMemcpy(h_got_a.data(), d_slot1, act_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_got_b.data(), d_slot2, act_bytes, cudaMemcpyDeviceToHost));

    auto compare = [&](const char* label,
                       const std::vector<__nv_bfloat16>& got,
                       const std::vector<__nv_bfloat16>& ref) -> int {
        float max_abs = 0.0f, sum_ref_sq = 0.0f, sum_diff_sq = 0.0f;
        int mismatches = 0;
        const float TOL = 0.03f;
        int per_row_errors[NUM_TOKENS] = {};
        for (int row = 0; row < NUM_TOKENS; ++row) {
            for (int i = 0; i < HIDDEN_DIM; ++i) {
                float gv = __bfloat162float(got[row * HIDDEN_DIM + i]);
                float rv = __bfloat162float(ref[row * HIDDEN_DIM + i]);
                float diff = std::fabs(gv - rv);
                if (diff > TOL) { ++mismatches; ++per_row_errors[row]; }
                if (diff > max_abs) max_abs = diff;
                sum_ref_sq  += rv * rv;
                sum_diff_sq += diff * diff;
            }
        }
        float rel_l2 = std::sqrt(sum_diff_sq / (sum_ref_sq > 0 ? sum_ref_sq : 1.0f));
        std::printf("%s: max_abs=%.6f rel_l2=%.6f mismatches(>%.2f)=%d\n",
                    label, max_abs, rel_l2, TOL, mismatches);
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
    int m_a = compare("op0 slot1", h_got_a, h_ref_out_a);
    int m_b = compare("op1 slot2", h_got_b, h_ref_out_b);
    int mismatches = m_a + m_b;
    if (mismatches == 0) {
        std::printf("ok: 2-op pool-ABI variant matches CPU reference\n");
    }

    CUDA_CHECK(cudaFree(d_slot0));
    CUDA_CHECK(cudaFree(d_slot1));
    CUDA_CHECK(cudaFree(d_slot2));
    CUDA_CHECK(cudaFree(d_w0));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_act_ptrs));
    CUDA_CHECK(cudaFree(d_w_ptrs));
    return mismatches == 0 ? 0 : 3;
}
