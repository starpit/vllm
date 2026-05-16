// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK AttentionPartial standalone smoke test — Wave F/1.5.
//
// Exercises `attention_partial.cuh` at NUM_TOKENS ∈ {1, 2, 4, 8} against
// a CPU reference. Verifies Wave F/1's batched-decode kernel + ABI
// before the solver-side work in Wave F/2 wires it into mega dispatch.
//
// Per-token inputs:
//   q_in        [NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM]
//   seq_lens    [NUM_TOKENS]              per-token KV-cache length
//   block_table [NUM_TOKENS, block_table_stride]  per-token physical pages
// Per-token output:
//   o_out       [NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM]
//
// Math per (token, q_head):
//   kv_head = q_head / (NUM_Q_HEADS / NUM_KV_HEADS)
//   Q  = q_in[token, q_head, :]
//   for tok in [0, seq_lens[token]):
//     blk = block_table[token, tok / BLOCK_SIZE]
//     K_j = key_cache  [blk, tok % BLOCK_SIZE, kv_head, :]
//     V_j = value_cache[blk, tok % BLOCK_SIZE, kv_head, :]
//     S[tok] = scale * (Q · K_j)
//   P = softmax(S)
//   O[token, q_head, :] = sum_j P[j] * V_j
//
// Grid: dim3(NUM_Q_HEADS, 1, 1) — same as the NUM_TOKENS==1 test; each
// CTA loops over a strided slice of the NUM_KV_HEADS * NUM_TOKENS tile
// set. NUM_TOKENS=8 sets T_TOTAL=64 > gridDim=32 so every CTA runs two
// tiles and exercises the cross-tile Q/O handshakes added in F/1.
//
// Scope caps (match attention_partial.cuh):
//   SPLITS         == 1  — single CTA per (kv_head, token).
//   SLIDING_WINDOW == 0  — no windowed attention.
//   HAS_SOFTCAP    == 0  — no Gemma3 softcap.
//
// Dimensions: Llama-3.2-1B decode shape — HEAD_DIM=64, NUM_Q_HEADS=32,
// NUM_KV_HEADS=8 (GQA_RATIO=4), BLOCK_SIZE=16.
//
// Build (on pod, CUDA 12.9, H100 sm_90a):
//   nvcc -O3 -std=c++20 \
//     -gencode=arch=compute_90a,code=sm_90a \
//     -DKITTENS_HOPPER --extended-lambda --expt-relaxed-constexpr \
//     -I /home/nickm/vllm-mega-codegen/vllm-rs/third_party/thunderkittens/include \
//     -I /home/nickm/vllm-mega-codegen/vllm-rs/crates/ferrite-kernels/csrc/tk \
//     /home/nickm/vllm-mega-codegen/vllm-rs/crates/ferrite-kernels/csrc/smoke/ferrite_attention_partial_smoke.cu \
//     -o /tmp/ferrite_attention_partial_smoke -lcuda
//
// (`-arch=sm_90a` gets lowered to plain `sm_90` by some ptxas
// toolchains, which rejects `setmaxnreg`; the explicit `-gencode`
// form pins the target to sm_90a.)
//
// Run: /tmp/ferrite_attention_partial_smoke

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
#include "ferrite_kernels/attention_partial.cuh"

namespace {

// ---- Model dims (shared across all NUM_TOKENS cases).
static constexpr int HEAD_DIM         = 64;
static constexpr int NUM_Q_HEADS      = 32;
static constexpr int NUM_KV_HEADS     = 8;
static constexpr int GROUP_SIZE       = NUM_Q_HEADS / NUM_KV_HEADS;   // 4
static constexpr int BLOCK_SIZE       = 16;
// 16 physical blocks easily covers the 8-token case's 13 used pages
// without sharing (deterministic block assignment below).
static constexpr int NUM_KV_BLOCKS    = 16;
static constexpr int MAX_PAGES_PER_SEQ = 2;
static constexpr int SPLITS           = 1;
static constexpr int SLIDING_WINDOW   = 0;
static constexpr int HAS_SOFTCAP      = 0;

// ---- FerriteConfig.
//
// Same shape as the original NUM_TOKENS==1 test — header
// static_asserts INSTRUCTION_PIPE_STAGES==2 so bumping it here would
// have to move in lockstep with variant_cpp.rs::op_page_count.
struct FerriteConfig {
    static constexpr int NUM_CONSUMER_WARPS      = 2;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 192;
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
    static constexpr int NUM_PAGES               =
        2 + 2 * INSTRUCTION_PIPE_STAGES;
    static constexpr int PAGE_SIZE               = 2048;
    static constexpr int SCRATCH_BYTES           = 512;
};

static constexpr int NUM_ACT_SLOTS       = 2;   // [q_in, o_out]
static constexpr int TPB                 = (FerriteConfig::NUM_CONSUMER_WARPS + 3) * 32;

using SS = ferrite::SharedState<FerriteConfig>;

// ---- Walker bodies — templated on NUM_TOKENS so one set of stubs
// serves every case the smoke test drives.

template <int NUM_TOKENS>
__device__ __forceinline__ void consumer_body(
    const int32_t* seq_lens,
    SS& ss,
    int warp_in_role,
    float softmax_scale)
{
    ferrite::ops::attention_partial::consumer<
        FerriteConfig, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS,
        BLOCK_SIZE, NUM_TOKENS, SPLITS, SLIDING_WINDOW, HAS_SOFTCAP>(
            seq_lens, ss, /*base_stage=*/0, warp_in_role, softmax_scale);
}

template <int NUM_TOKENS>
__device__ __forceinline__ void loader_body(
    const __nv_bfloat16* q_in,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache,
    const uint32_t*      block_table,
    uint32_t             block_table_stride,
    const int32_t*       seq_lens,
    SS& ss)
{
    ferrite::ops::attention_partial::loader<
        FerriteConfig, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS,
        BLOCK_SIZE, NUM_TOKENS, SPLITS, SLIDING_WINDOW, HAS_SOFTCAP>(
            q_in, key_cache, value_cache,
            block_table, block_table_stride,
            seq_lens, ss, /*base_stage=*/0);
}

template <int NUM_TOKENS>
__device__ __forceinline__ void launcher_body(SS& ss)
{
    ferrite::ops::attention_partial::launcher<
        FerriteConfig, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS,
        BLOCK_SIZE, NUM_TOKENS, SPLITS, SLIDING_WINDOW, HAS_SOFTCAP>(
            ss, /*base_stage=*/0);
}

template <int NUM_TOKENS>
__device__ __forceinline__ void storer_body(
    __nv_bfloat16* o_out,
    SS& ss)
{
    ferrite::ops::attention_partial::storer<
        FerriteConfig, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS,
        BLOCK_SIZE, NUM_TOKENS, SPLITS, SLIDING_WINDOW, HAS_SOFTCAP>(
            o_out, ss, /*base_stage=*/0);
}

template <int NUM_TOKENS>
__global__ void attention_partial_smoke_kernel(
    __nv_bfloat16* const*        __restrict__ act_ptrs,    // [q_in, o_out]
    const __nv_bfloat16*         __restrict__ key_cache,
    const __nv_bfloat16*         __restrict__ value_cache,
    const uint32_t*              __restrict__ block_table,
    uint32_t                                  block_table_stride,
    const int32_t*               __restrict__ seq_lens,
    float                                     softmax_scale)
{
    __shared__ SS ss;
    ferrite::init_shared_state<FerriteConfig>(ss);

    const __nv_bfloat16* q_in  = act_ptrs[0];
    __nv_bfloat16*       o_out = act_ptrs[1];

    const int wid = kittens::warpid();
    if (wid < FerriteConfig::NUM_CONSUMER_WARPS) {
        ferrite::set_consumer_registers<FerriteConfig>();
        consumer_body<NUM_TOKENS>(seq_lens, ss, wid, softmax_scale);
    } else {
        ferrite::set_non_consumer_registers<FerriteConfig>();
        switch (wid - FerriteConfig::NUM_CONSUMER_WARPS) {
            case ferrite::kLoaderSlot:
                loader_body<NUM_TOKENS>(
                    q_in, key_cache, value_cache,
                    block_table, block_table_stride,
                    seq_lens, ss);
                break;
            case ferrite::kLauncherSlot:
                launcher_body<NUM_TOKENS>(ss);
                break;
            case ferrite::kStorerSlot:
                storer_body<NUM_TOKENS>(o_out, ss);
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

// ---- CPU reference — per-token attention against the paged KV cache.
//
// For each token, walks block_table[token] to gather K and V rows at
// valid positions [0, seq_lens[token]), then runs softmax and weighted
// sum. All arithmetic in fp32 for a tight golden. Mirrors the kernel's
// math op-for-op: scale * Q·K, subtract max, exp, sum, divide.
void cpu_reference(
    int num_tokens,
    const std::vector<__nv_bfloat16>& q_in,          // [NT, NUM_Q_HEADS, HEAD_DIM]
    const std::vector<__nv_bfloat16>& key_cache,     // [NUM_KV_BLOCKS, BS, NUM_KV_HEADS, HEAD_DIM]
    const std::vector<__nv_bfloat16>& value_cache,
    const std::vector<uint32_t>&      block_table,   // [NT, block_table_stride]
    uint32_t                          block_table_stride,
    const std::vector<int32_t>&       seq_lens,      // [NT]
    float                             softmax_scale,
    std::vector<__nv_bfloat16>&       o_out)         // [NT, NUM_Q_HEADS, HEAD_DIM]
{
    const size_t block_stride = size_t(BLOCK_SIZE) * NUM_KV_HEADS * HEAD_DIM;
    const size_t row_stride   = size_t(NUM_KV_HEADS) * HEAD_DIM;

    for (int t = 0; t < num_tokens; ++t) {
        const int seq_len = seq_lens[t];
        for (int q_head = 0; q_head < NUM_Q_HEADS; ++q_head) {
            const int kv_head = q_head / GROUP_SIZE;
            float q_f[HEAD_DIM];
            for (int i = 0; i < HEAD_DIM; ++i) {
                q_f[i] = __bfloat162float(
                    q_in[(size_t(t) * NUM_Q_HEADS + q_head) * HEAD_DIM + i]);
            }
            std::vector<float> s(seq_len, 0.0f);
            for (int tok = 0; tok < seq_len; ++tok) {
                int p = tok / BLOCK_SIZE;
                int j = tok % BLOCK_SIZE;
                uint32_t blk =
                    block_table[size_t(t) * block_table_stride + p];
                const __nv_bfloat16* k_row =
                    key_cache.data()
                    + size_t(blk) * block_stride
                    + size_t(j)   * row_stride
                    + size_t(kv_head) * HEAD_DIM;
                float acc = 0.0f;
                for (int i = 0; i < HEAD_DIM; ++i) {
                    acc += q_f[i] * __bfloat162float(k_row[i]);
                }
                s[tok] = acc * softmax_scale;
            }
            float m = -INFINITY;
            for (float v : s) if (v > m) m = v;
            float l = 0.0f;
            for (int tok = 0; tok < seq_len; ++tok) {
                s[tok] = std::exp(s[tok] - m);
                l += s[tok];
            }
            float inv_l = 1.0f / l;
            float o_f[HEAD_DIM];
            for (int i = 0; i < HEAD_DIM; ++i) o_f[i] = 0.0f;
            for (int tok = 0; tok < seq_len; ++tok) {
                int p = tok / BLOCK_SIZE;
                int j = tok % BLOCK_SIZE;
                uint32_t blk =
                    block_table[size_t(t) * block_table_stride + p];
                const __nv_bfloat16* v_row =
                    value_cache.data()
                    + size_t(blk) * block_stride
                    + size_t(j)   * row_stride
                    + size_t(kv_head) * HEAD_DIM;
                float p_tok = s[tok] * inv_l;
                for (int i = 0; i < HEAD_DIM; ++i) {
                    o_f[i] += p_tok * __bfloat162float(v_row[i]);
                }
            }
            for (int i = 0; i < HEAD_DIM; ++i) {
                o_out[(size_t(t) * NUM_Q_HEADS + q_head) * HEAD_DIM + i] =
                    __float2bfloat16_rn(o_f[i]);
            }
        }
    }
}

int compare_tolerance(const char* label, float tol,
                      const std::vector<__nv_bfloat16>& got,
                      const std::vector<__nv_bfloat16>& ref)
{
    float max_abs = 0.0f, sum_ref_sq = 0.0f, sum_diff_sq = 0.0f;
    int mismatches = 0;
    int first_mismatch_idx = -1;
    float first_got = 0.0f, first_ref = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        float gv = __bfloat162float(got[i]);
        float rv = __bfloat162float(ref[i]);
        float diff = std::fabs(gv - rv);
        if (diff > tol) {
            ++mismatches;
            if (first_mismatch_idx < 0) {
                first_mismatch_idx = int(i);
                first_got = gv;
                first_ref = rv;
            }
        }
        if (diff > max_abs) max_abs = diff;
        sum_ref_sq  += rv * rv;
        sum_diff_sq += diff * diff;
    }
    float rel_l2 = std::sqrt(sum_diff_sq / (sum_ref_sq > 0 ? sum_ref_sq : 1.0f));
    std::printf("  %s: max_abs=%.6f rel_l2=%.6f mismatches(>%.3f)=%d\n",
                label, max_abs, rel_l2, tol, mismatches);
    if (mismatches > 0) {
        std::printf("    first mismatch at idx=%d got=%.6f ref=%.6f diff=%.6f\n",
                    first_mismatch_idx, first_got, first_ref,
                    std::fabs(first_got - first_ref));
    }
    return mismatches;
}

// ---- Per-token scenario builder.
//
// Each token gets its own seq_len and its own block_table row.
// Lengths are picked to mix tail-masked pages (seq_len % BLOCK_SIZE
// != 0) with fully-used pages (seq_len % BLOCK_SIZE == 0), and to mix
// single-page tokens (seq_len <= BLOCK_SIZE) with multi-page tokens
// so the kernel's per-token num_pages branch is exercised.
//
// Physical blocks are handed out deterministically from a pool so no
// two (token, page) pairs share a physical block; that lets the CPU
// reference and the kernel agree even if a block-id typo slipped in.
//
// At NUM_TOKENS=8 the total pages (13) fits inside NUM_KV_BLOCKS=16.
template <int NUM_TOKENS>
struct Scenario {
    std::vector<int32_t>  seq_lens;
    std::vector<uint32_t> block_table;
    uint32_t              block_table_stride;
};

template <int NUM_TOKENS>
Scenario<NUM_TOKENS> build_scenario() {
    // Deterministic per-token seq_lens — vary tail/no-tail + page counts.
    // Suffix of this array defines shorter-NUM_TOKENS cases so truncation
    // to NT<8 stays valid.
    static constexpr int32_t kAllSeqLens[8] = { 23, 17, 11, 29, 7, 31, 19, 5 };

    Scenario<NUM_TOKENS> sc;
    sc.seq_lens.resize(NUM_TOKENS);
    sc.block_table_stride = MAX_PAGES_PER_SEQ;
    sc.block_table.assign(size_t(NUM_TOKENS) * MAX_PAGES_PER_SEQ, 0u);

    uint32_t next_block = 2;  // skip 0 and 1 so block-table is non-trivial
    for (int t = 0; t < NUM_TOKENS; ++t) {
        sc.seq_lens[t] = kAllSeqLens[t];
        int num_pages = (sc.seq_lens[t] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int p = 0; p < num_pages; ++p) {
            sc.block_table[size_t(t) * MAX_PAGES_PER_SEQ + p] = next_block++;
        }
    }
    return sc;
}

template <int NUM_TOKENS>
int run_case(uint32_t seed) {
    std::printf("\n=== NUM_TOKENS=%d ===\n", NUM_TOKENS);

    const float kScale = 1.0f / std::sqrt(float(HEAD_DIM));
    Scenario<NUM_TOKENS> sc = build_scenario<NUM_TOKENS>();

    // Host buffers.
    const size_t kv_cache_count =
        size_t(NUM_KV_BLOCKS) * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    std::vector<__nv_bfloat16> h_q_in       (size_t(NUM_TOKENS) * NUM_Q_HEADS * HEAD_DIM);
    std::vector<__nv_bfloat16> h_key_cache  (kv_cache_count,
                                              __float2bfloat16_rn(0.0f));
    std::vector<__nv_bfloat16> h_value_cache(kv_cache_count,
                                              __float2bfloat16_rn(0.0f));

    uint32_t s = seed;
    for (auto& v : h_q_in) v = __float2bfloat16_rn(0.25f * rand_u(s));

    // Fill only the physical blocks actually referenced by the block
    // table, and only up to the per-token seq_len. Tail rows of the
    // last page are left at zero to exercise the consumer's -inf tail
    // mask without perturbing the golden.
    const size_t block_stride = size_t(BLOCK_SIZE) * NUM_KV_HEADS * HEAD_DIM;
    const size_t row_stride   = size_t(NUM_KV_HEADS) * HEAD_DIM;
    for (int t = 0; t < NUM_TOKENS; ++t) {
        int seq_len   = sc.seq_lens[t];
        int num_pages = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int p = 0; p < num_pages; ++p) {
            uint32_t blk =
                sc.block_table[size_t(t) * MAX_PAGES_PER_SEQ + p];
            int valid = (p == num_pages - 1)
                            ? (seq_len - (num_pages - 1) * BLOCK_SIZE)
                            : BLOCK_SIZE;
            for (int j = 0; j < valid; ++j) {
                for (int kv = 0; kv < NUM_KV_HEADS; ++kv) {
                    for (int i = 0; i < HEAD_DIM; ++i) {
                        size_t idx = size_t(blk) * block_stride
                                   + size_t(j)   * row_stride
                                   + size_t(kv)  * HEAD_DIM
                                   + i;
                        h_key_cache[idx]   = __float2bfloat16_rn(0.20f * rand_u(s));
                        h_value_cache[idx] = __float2bfloat16_rn(0.30f * rand_u(s));
                    }
                }
            }
        }
    }

    // CPU reference.
    std::vector<__nv_bfloat16> h_o_ref(h_q_in.size(),
                                        __float2bfloat16_rn(0.0f));
    cpu_reference(NUM_TOKENS,
                  h_q_in, h_key_cache, h_value_cache,
                  sc.block_table, sc.block_table_stride,
                  sc.seq_lens, kScale, h_o_ref);

    // Device buffers.
    __nv_bfloat16* d_q_in        = nullptr;
    __nv_bfloat16* d_key_cache   = nullptr;
    __nv_bfloat16* d_value_cache = nullptr;
    __nv_bfloat16* d_o_out       = nullptr;
    uint32_t*      d_block_table = nullptr;
    int32_t*       d_seq_lens    = nullptr;

    const size_t q_bytes  = h_q_in.size()          * sizeof(__nv_bfloat16);
    const size_t kv_bytes = kv_cache_count          * sizeof(__nv_bfloat16);
    const size_t o_bytes  = h_o_ref.size()          * sizeof(__nv_bfloat16);
    const size_t bt_bytes = sc.block_table.size()   * sizeof(uint32_t);
    const size_t sl_bytes = sc.seq_lens.size()      * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_q_in,        q_bytes));
    CUDA_CHECK(cudaMalloc(&d_key_cache,   kv_bytes));
    CUDA_CHECK(cudaMalloc(&d_value_cache, kv_bytes));
    CUDA_CHECK(cudaMalloc(&d_o_out,       o_bytes));
    CUDA_CHECK(cudaMalloc(&d_block_table, bt_bytes));
    CUDA_CHECK(cudaMalloc(&d_seq_lens,    sl_bytes));

    CUDA_CHECK(cudaMemcpy(d_q_in,        h_q_in.data(),            q_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_key_cache,   h_key_cache.data(),       kv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value_cache, h_value_cache.data(),     kv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_table, sc.block_table.data(),    bt_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens,    sc.seq_lens.data(),       sl_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_o_out, 0, o_bytes));

    __nv_bfloat16* h_act_ptrs[NUM_ACT_SLOTS] = { d_q_in, d_o_out };
    __nv_bfloat16** d_act_ptrs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_act_ptrs, sizeof(h_act_ptrs)));
    CUDA_CHECK(cudaMemcpy(d_act_ptrs, h_act_ptrs, sizeof(h_act_ptrs), cudaMemcpyHostToDevice));

    dim3 grid(NUM_Q_HEADS, 1, 1);
    dim3 block(TPB, 1, 1);
    attention_partial_smoke_kernel<NUM_TOKENS><<<grid, block, 0, 0>>>(
        d_act_ptrs,
        d_key_cache, d_value_cache,
        d_block_table, sc.block_table_stride,
        d_seq_lens,
        kScale);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "launch error at NUM_TOKENS=%d: %s\n",
                     NUM_TOKENS, cudaGetErrorString(launch_err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> h_o_got(h_o_ref.size());
    CUDA_CHECK(cudaMemcpy(h_o_got.data(), d_o_out, o_bytes, cudaMemcpyDeviceToHost));

    std::printf("  grid=%d block=%d T_TOTAL=%d tiles_per_cta=%d seq_lens=[",
                NUM_Q_HEADS, TPB,
                NUM_KV_HEADS * NUM_TOKENS,
                (NUM_KV_HEADS * NUM_TOKENS + NUM_Q_HEADS - 1) / NUM_Q_HEADS);
    for (int t = 0; t < NUM_TOKENS; ++t) {
        std::printf("%s%d", (t ? "," : ""), int(sc.seq_lens[t]));
    }
    std::printf("]\n");

    // Compare whole-tensor first, then per-token slices so a failure
    // narrows to which token's output diverged.
    int total_mismatches = compare_tolerance("all-tokens", 0.02f,
                                             h_o_got, h_o_ref);
    if (total_mismatches > 0) {
        for (int t = 0; t < NUM_TOKENS; ++t) {
            size_t off = size_t(t) * NUM_Q_HEADS * HEAD_DIM;
            size_t len = size_t(NUM_Q_HEADS) * HEAD_DIM;
            std::vector<__nv_bfloat16> got_t(h_o_got.begin() + off,
                                             h_o_got.begin() + off + len);
            std::vector<__nv_bfloat16> ref_t(h_o_ref.begin() + off,
                                             h_o_ref.begin() + off + len);
            char label[32];
            std::snprintf(label, sizeof(label), "token-%d", t);
            compare_tolerance(label, 0.02f, got_t, ref_t);
        }
    }

    CUDA_CHECK(cudaFree(d_q_in));
    CUDA_CHECK(cudaFree(d_key_cache));
    CUDA_CHECK(cudaFree(d_value_cache));
    CUDA_CHECK(cudaFree(d_o_out));
    CUDA_CHECK(cudaFree(d_block_table));
    CUDA_CHECK(cudaFree(d_seq_lens));
    CUDA_CHECK(cudaFree(d_act_ptrs));
    return total_mismatches;
}

} // namespace

int main() {
    std::printf("head_dim=%d q_heads=%d kv_heads=%d group_size=%d block_size=%d num_blocks=%d\n",
                HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, GROUP_SIZE,
                BLOCK_SIZE, NUM_KV_BLOCKS);
    std::printf("max_pages_per_seq=%d consumer_warps=%d pipe_stages=%d\n",
                MAX_PAGES_PER_SEQ,
                FerriteConfig::NUM_CONSUMER_WARPS,
                FerriteConfig::INSTRUCTION_PIPE_STAGES);

    int total = 0;
    total += run_case<1>(0xA77E1710u);
    total += run_case<2>(0xA77E1720u);
    total += run_case<4>(0xA77E1740u);
    total += run_case<8>(0xA77E1780u);

    if (total == 0) {
        std::printf("\nok: attention_partial matches CPU reference at NUM_TOKENS ∈ {1,2,4,8}\n");
        return 0;
    }
    std::printf("\nFAIL: %d total mismatches across cases\n", total);
    return 3;
}
