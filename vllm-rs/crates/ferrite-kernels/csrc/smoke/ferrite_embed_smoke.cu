// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK embed standalone smoke test.
//
// Phase 3f-2e-ii exit gate: prove `ferrite::ops::embed::{loader,
// consumer,launcher,storer}` produce a per-row gather that matches
// a CPU reference. Does not go through codegen — this file
// reproduces the minimum substrate a codegen'd .cu would emit so we
// isolate "did I write the op header correctly?" from "did I wire
// codegen correctly?".
//
// Variant: HIDDEN_DIM = 2048 (llama-3.2-1B hidden_dim), VOCAB_SIZE
// = 128 (small enough that a few random token IDs touch a good
// spread of rows), NUM_TOKENS = 4. bf16 embed table and outputs,
// u32 input_ids.
//
// Build (on pod nick, CUDA 12.9, H100 sm_90a):
//   nvcc -O3 -std=c++20 -arch=sm_90a \
//     -DKITTENS_HOPPER \
//     -I /home/nickm/vllm-mega/vllm-rs/third_party/thunderkittens/include \
//     -I /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/tk \
//     /tmp/ferrite_embed_smoke.cu -o /tmp/ferrite_embed_smoke -lcuda
//
// Run: /tmp/ferrite_embed_smoke

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
#include "ferrite_kernels/embed.cuh"

namespace {

// Phase 2/3 defaults — mirrors FerriteConfig::phase2(hidden_dim)
// in the Rust codegen. PAGE_SIZE must be >= HIDDEN_DIM * 2 bytes,
// rounded up to 128. For HIDDEN_DIM=2048 that's 4096.
//
// NUM_PAGES: embed claims 1 page per op (the output row), vs
// gemv's 2 (activation + weight row). 1 would work in principle,
// but the ferrite_substrate.cuh init path sizes its semaphore
// arrays by NUM_PAGES and there's no win to trimming below 2 for
// a smoke — keep at 2 to match the gemv smoke's layout and make
// any substrate-size drift across ops visible as a kernel
// mis-launch rather than a silent config-drift.
struct SmokeConfig {
    static constexpr int NUM_CONSUMER_WARPS      = 4;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 192;
    static constexpr int NUM_PAGES               = 2;
    static constexpr int PAGE_SIZE               = 4096;
    static constexpr int SCRATCH_BYTES           = 256;
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

constexpr int HIDDEN_DIM = 2048;
constexpr int VOCAB_SIZE = 128;
constexpr int NUM_TOKENS = 4;
constexpr int TPB        = (SmokeConfig::NUM_CONSUMER_WARPS + 3) * 32;

using SS = ferrite::SharedState<SmokeConfig>;

__global__ void embed_smoke_kernel(
    const uint32_t*      __restrict__ input_ids,    // [NUM_TOKENS]
    const __nv_bfloat16* __restrict__ embed_tokens, // [VOCAB_SIZE, HIDDEN_DIM]
    __nv_bfloat16*       __restrict__ out           // [NUM_TOKENS, HIDDEN_DIM]
) {
    __shared__ SS ss;
    ferrite::init_shared_state<SmokeConfig>(ss);

    const int wid = kittens::warpid();
    if (wid < SmokeConfig::NUM_CONSUMER_WARPS) {
        ferrite::set_consumer_registers<SmokeConfig>();
        ferrite::ops::embed::consumer<SmokeConfig, HIDDEN_DIM, NUM_TOKENS>(
            ss, /*base_stage=*/0, wid);
    } else {
        ferrite::set_non_consumer_registers<SmokeConfig>();
        switch (wid - SmokeConfig::NUM_CONSUMER_WARPS) {
            case ferrite::kLoaderSlot:
                ferrite::ops::embed::loader<SmokeConfig, HIDDEN_DIM, NUM_TOKENS>(
                    input_ids, embed_tokens, ss, /*base_stage=*/0);
                break;
            case ferrite::kLauncherSlot:
                ferrite::ops::embed::launcher<SmokeConfig, HIDDEN_DIM, NUM_TOKENS>(
                    ss, /*base_stage=*/0);
                break;
            case ferrite::kStorerSlot:
                ferrite::ops::embed::storer<SmokeConfig, HIDDEN_DIM, NUM_TOKENS>(
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
    std::vector<uint32_t>        ids(NUM_TOKENS);
    std::vector<float>           table_f32(VOCAB_SIZE * HIDDEN_DIM);
    std::vector<__nv_bfloat16>   table_bf16(VOCAB_SIZE * HIDDEN_DIM);

    uint32_t s = 0xC0DEu;
    for (int i = 0; i < VOCAB_SIZE * HIDDEN_DIM; ++i) {
        table_f32[i]  = rand_u(s);
        table_bf16[i] = __float2bfloat16(table_f32[i]);
    }
    // Pick a deterministic but non-monotonic set of token IDs so
    // we exercise different regions of the table. mod VOCAB_SIZE
    // keeps everything in range.
    const uint32_t id_seeds[] = { 7u, 42u, 91u, 3u };
    for (int t = 0; t < NUM_TOKENS; ++t) {
        ids[t] = id_seeds[t] % static_cast<uint32_t>(VOCAB_SIZE);
    }

    // CPU reference — pure gather; quantize through bf16 round-trip
    // so tolerance reflects the kernel's own bf16 loads.
    std::vector<float> out_ref(NUM_TOKENS * HIDDEN_DIM);
    for (int t = 0; t < NUM_TOKENS; ++t) {
        const uint32_t tok = ids[t];
        for (int i = 0; i < HIDDEN_DIM; ++i) {
            out_ref[t * HIDDEN_DIM + i] =
                __bfloat162float(table_bf16[tok * HIDDEN_DIM + i]);
        }
    }

    uint32_t*       d_ids   = nullptr;
    __nv_bfloat16*  d_table = nullptr;
    __nv_bfloat16*  d_out   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ids,
                          NUM_TOKENS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_table,
                          VOCAB_SIZE * HIDDEN_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out,
                          NUM_TOKENS * HIDDEN_DIM * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(),
                          NUM_TOKENS * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_table, table_bf16.data(),
                          VOCAB_SIZE * HIDDEN_DIM * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0,
                          NUM_TOKENS * HIDDEN_DIM * sizeof(__nv_bfloat16)));

    dim3 grid(NUM_TOKENS, 1, 1);
    dim3 block(TPB, 1, 1);
    embed_smoke_kernel<<<grid, block, 0, 0>>>(d_ids, d_table, d_out);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n",
                     cudaGetErrorString(launch_err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> out_bf16(NUM_TOKENS * HIDDEN_DIM);
    CUDA_CHECK(cudaMemcpy(out_bf16.data(), d_out,
                          NUM_TOKENS * HIDDEN_DIM * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    float max_abs = 0.0f, sum_ref_sq = 0.0f, sum_diff_sq = 0.0f;
    int   mismatches = 0;
    // Pure gather — no math, so the only source of error is the
    // bf16 round-trip on both the reference and the device read.
    // Both sides hold the same bf16 bit pattern, so the expected
    // diff is zero. Give an epsilon-slop in case a future substrate
    // tweak introduces a non-round-trip-stable codepath (e.g.
    // fp32-accumulated scaling).
    const float TOL = 1e-3f;
    for (int i = 0; i < NUM_TOKENS * HIDDEN_DIM; ++i) {
        float got  = __bfloat162float(out_bf16[i]);
        float ref  = out_ref[i];
        float diff = std::fabs(got - ref);
        if (diff > TOL) ++mismatches;
        if (diff > max_abs) max_abs = diff;
        sum_ref_sq  += ref * ref;
        sum_diff_sq += diff * diff;
    }
    float rel_l2 = std::sqrt(sum_diff_sq / (sum_ref_sq > 0 ? sum_ref_sq : 1.0f));

    std::printf("NUM_TOKENS=%d HIDDEN_DIM=%d VOCAB_SIZE=%d "
                "max_abs=%.6f rel_l2=%.6f mismatches(>%.0e)=%d\n",
                NUM_TOKENS, HIDDEN_DIM, VOCAB_SIZE,
                max_abs, rel_l2, TOL, mismatches);
    if (mismatches == 0) {
        std::printf("ok: embed matches CPU reference exactly\n");
    } else {
        std::printf("sample rows:\n");
        for (int t = 0; t < NUM_TOKENS; ++t) {
            // Show the first 4 elements of each token row.
            std::printf("  token=%d id=%u ", t, ids[t]);
            int first_bad = -1;
            for (int i = 0; i < HIDDEN_DIM; ++i) {
                float got = __bfloat162float(out_bf16[t * HIDDEN_DIM + i]);
                float ref = out_ref[t * HIDDEN_DIM + i];
                if (std::fabs(got - ref) > TOL) { first_bad = i; break; }
            }
            if (first_bad >= 0) {
                std::printf("first_bad_i=%d got=%.6f ref=%.6f",
                            first_bad,
                            __bfloat162float(out_bf16[t * HIDDEN_DIM + first_bad]),
                            out_ref[t * HIDDEN_DIM + first_bad]);
            } else {
                std::printf("ok");
            }
            std::printf("\n");
        }
    }

    CUDA_CHECK(cudaFree(d_ids));
    CUDA_CHECK(cudaFree(d_table));
    CUDA_CHECK(cudaFree(d_out));
    return mismatches == 0 ? 0 : 3;
}
