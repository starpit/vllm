// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK gemm_bf16 standalone smoke test — Wave E wgmma.
//
// Tests both execution paths:
//
// NCW=2 (head_dim=64 models, e.g. Llama-3.2-1B) — warp-level mma_ABt,
//   unswizzled tiles, kMTile=32:
//     M=8   (partial M-tile: 8 of 32 valid rows)
//     M=32  (exactly one M-tile)
//     M=64  (two M-tiles — exercises Phase 5 loop)
//
// NCW=4 (head_dim=128+ models) — wgmma::mma_ABt, swizzled tiles, kMTile=64:
//     M=8   (partial M-tile: 8 of 64 valid rows)
//     M=64  (exactly one M-tile)
//     M=128 (two M-tiles — exercises Phase 5 loop)
//
// For all cases: K=2048, N=64 (one N-tile).
//
// Build (on pod nick, CUDA 12.9, H100 sm_90a):
//   nvcc -O3 -std=c++20 -gencode=arch=compute_90a,code=sm_90a \
//     -DKITTENS_HOPPER --extended-lambda --expt-relaxed-constexpr \
//     -I /home/nickm/vllm-mega-codegen/vllm-rs/third_party/thunderkittens/include \
//     -I /home/nickm/vllm-mega-codegen/vllm-rs/crates/ferrite-kernels/csrc/tk \
//     /home/nickm/vllm-mega-codegen/vllm-rs/crates/ferrite-kernels/csrc/smoke/ferrite_gemm_smoke.cu \
//     -o /tmp/ferrite_gemm_smoke -lcuda
//
// Expected output: all cases mismatches=0

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kittens.cuh"
#include "ferrite_globals.cuh"
#include "ferrite_substrate.cuh"
#include "ferrite_warp_roles.cuh"
#include "ferrite_kernels/gemm_bf16.cuh"

namespace {

constexpr int K = 2048;
constexpr int N = 64;
const float TOL = 0.2f;

// ───── NCW=2 config (Llama-3.2-1B style, head_dim=64) ─────────────────────
// PAGE_SIZE  = max(2*max(hidden,inter), matvec_chunk) ≥ kMTile*kKChunk*2
//           = max(32*64*2, 64*512*2) = max(4096, 65536) = 65536 bytes
// (use 16384 to match the actual ferrite config floor of 16KB)
// For the smoke test, PAGE_SIZE = kMTile*kKChunk*2 = 32*64*2 = 4096 is enough.
// SCRATCH_BYTES = kMTile*kNTile*2 = 32*64*2 = 4096 bytes
struct Config2 {
    static constexpr int NUM_CONSUMER_WARPS      = 2;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 232;
    static constexpr int NUM_PAGES               = 2;
    static constexpr int PAGE_SIZE               = 16384;  // safe upper bound
    static constexpr int SCRATCH_BYTES           = 8192;   // 64*64*2 (kMTile=64)
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

// ───── NCW=4 config (head_dim=128+) ─────────────────────────────────────
// PAGE_SIZE = kMTile*kKChunk*2 = 64*64*2 = 8192 bytes
// SCRATCH_BYTES = kMTile*kNTile*2 = 64*64*2 = 8192 bytes
struct Config4 {
    static constexpr int NUM_CONSUMER_WARPS      = 4;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 232;
    static constexpr int NUM_PAGES               = 2;
    static constexpr int PAGE_SIZE               = 8192;   // 64*64*2
    static constexpr int SCRATCH_BYTES           = 8192;   // 64*64*2
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

template<typename Config>
constexpr int tpb() { return (Config::NUM_CONSUMER_WARPS + 3) * 32; }

template <typename Config, int M_VAL>
__global__ void gemm_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16*       __restrict__ out
) {
    __shared__ ferrite::SharedState<Config> ss;
    ferrite::init_shared_state<Config>(ss);

    const int wid = kittens::warpid();
    if (wid < Config::NUM_CONSUMER_WARPS) {
        ferrite::set_consumer_registers<Config>();
        ferrite::ops::gemm_bf16::consumer<Config, K, N, M_VAL>(
            out, ss, /*base_stage=*/0, /*warp_in_role=*/wid);
    } else {
        ferrite::set_non_consumer_registers<Config>();
        switch (wid - Config::NUM_CONSUMER_WARPS) {
            case ferrite::kLoaderSlot:
                ferrite::ops::gemm_bf16::loader<Config, K, N, M_VAL>(
                    x, W, ss, /*base_stage=*/0);
                break;
            case ferrite::kLauncherSlot:
                ferrite::ops::gemm_bf16::launcher<Config, K, N, M_VAL>(
                    ss, /*base_stage=*/0);
                break;
            case ferrite::kStorerSlot:
                ferrite::ops::gemm_bf16::storer<Config, K, N, M_VAL>(
                    out, ss, /*base_stage=*/0);
                break;
            default: break;
        }
    }
}

#define CUDA_CHECK(call) do {                                              \
    cudaError_t e__ = (call);                                              \
    if (e__ != cudaSuccess) {                                              \
        std::fprintf(stderr, "cuda error %s at %s:%d: %s\n",               \
                     #call, __FILE__, __LINE__, cudaGetErrorString(e__));  \
        std::exit(1);                                                      \
    } } while(0)

uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
float rand_u(uint32_t& s) {
    return (float(xorshift32(s)) / float(UINT32_MAX)) * 2.f - 1.f;
}

template <typename Config, int M_VAL>
int run_case(const std::string& label) {
    constexpr int NCW = Config::NUM_CONSUMER_WARPS;
    constexpr int kMT = NCW * ferrite::ops::gemm_bf16::kMChunk;

    std::vector<__nv_bfloat16> x_h(M_VAL * K), W_h(N * K);
    std::vector<float>         x_f(M_VAL * K), W_f(N * K);
    uint32_t seed = 0xF00Du + M_VAL + NCW;
    for (auto& v : x_f) { v = rand_u(seed); }
    for (auto& v : W_f) { v = rand_u(seed); }
    for (int i = 0; i < M_VAL*K; ++i) x_h[i] = __float2bfloat16(x_f[i]);
    for (int i = 0; i < N*K;     ++i) W_h[i] = __float2bfloat16(W_f[i]);

    // CPU reference (bf16 round-trip for precision match).
    std::vector<float> ref(M_VAL * N, 0.f);
    for (int t = 0; t < M_VAL; ++t)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += __bfloat162float(x_h[t*K+k]) * __bfloat162float(W_h[n*K+k]);
            ref[t*N+n] = acc;
        }

    __nv_bfloat16 *d_x, *d_W, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x,   M_VAL * K * sizeof(*d_x)));
    CUDA_CHECK(cudaMalloc(&d_W,   N * K     * sizeof(*d_W)));
    CUDA_CHECK(cudaMalloc(&d_out, M_VAL * N * sizeof(*d_out)));
    CUDA_CHECK(cudaMemcpy(d_x, x_h.data(), M_VAL*K*sizeof(*d_x), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W_h.data(), N*K*sizeof(*d_W),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, M_VAL*N*sizeof(*d_out)));

    constexpr int N_TILES   = N / ferrite::ops::gemm_bf16::kNTile;
    constexpr int M_TILES   = (M_VAL + kMT - 1) / kMT;
    constexpr int tot_tiles = N_TILES * M_TILES;

    std::printf("%-10s NCW=%d grid=%d block=%d M=%d N=%d K=%d (M_TILES=%d)\n",
                label.c_str(), NCW, tot_tiles, tpb<Config>(),
                M_VAL, N, K, M_TILES);

    gemm_kernel<Config, M_VAL><<<tot_tiles, tpb<Config>()>>>(d_x, d_W, d_out);
    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> got_h(M_VAL * N);
    CUDA_CHECK(cudaMemcpy(got_h.data(), d_out, M_VAL*N*sizeof(*d_out), cudaMemcpyDeviceToHost));

    float max_abs = 0.f, ssq = 0.f, sdq = 0.f;
    int mis = 0;
    for (int i = 0; i < M_VAL*N; ++i) {
        float g = __bfloat162float(got_h[i]), r = ref[i], d = std::fabs(g-r);
        if (d > TOL) ++mis;
        max_abs = std::max(max_abs, d);
        ssq += r*r; sdq += d*d;
    }
    float rl2 = std::sqrt(sdq / (ssq > 0 ? ssq : 1.f));
    std::printf("           M=%d N=%d K=%d max_abs=%.4f rel_l2=%.6f mis=%d\n",
                M_VAL, N, K, max_abs, rl2, mis);
    if (mis > 0) std::printf("  FAIL\n"); else std::printf("  ok\n");

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_out));
    return mis;
}

} // namespace

int main() {
    int fail = 0;

    std::printf("=== NCW=2 (warp-level mma_ABt, head_dim=64) ===\n");
    fail += (run_case<Config2,   8>("M=8/NCW2")   != 0) ? 1 : 0;
    fail += (run_case<Config2,  64>("M=64/NCW2")  != 0) ? 1 : 0;
    fail += (run_case<Config2, 128>("M=128/NCW2") != 0) ? 1 : 0;

    std::printf("\n=== NCW=4 (wgmma::mma_ABt, head_dim=128+) ===\n");
    fail += (run_case<Config4,   8>("M=8/NCW4")   != 0) ? 1 : 0;
    fail += (run_case<Config4,  64>("M=64/NCW4")  != 0) ? 1 : 0;
    fail += (run_case<Config4, 128>("M=128/NCW4") != 0) ? 1 : 0;

    if (fail == 0)
        std::printf("\nok: gemm_bf16 all 6 cases pass\n");
    else
        std::printf("\nFAIL: %d case(s) had mismatches\n", fail);
    return fail == 0 ? 0 : 3;
}
