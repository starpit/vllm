// SPDX-License-Identifier: Apache-2.0
// Ferrite-TK FusedQkvRopeCache standalone smoke test — Phase 3f part 2b-ii.
//
// Drives the four walker roles of `fused_qkv_rope_cache.cuh` end-to-end
// for a single decode token against a CPU reference that does:
//   - packed QKV GEMM   qkv = W_packed @ x   (no bias; BIASED=false)
//   - NeoX RoPE on Q/K  Q' = rope(Q), K' = rope(K)   (INTERLEAVED=false)
//   - paged KV-cache write for K, V at slot_mapping[0]
//   - Q' returned to q_out
//
// Grid mirrors the header's layout: dim3(HEAD_DIM/2, NUM_HEADS_TOTAL).
// Each CTA produces exactly one rope pair inside one head.
//
// Pool-ABI note: `emit_cu_variant`'s current pool shape is
// (act_ptrs, weight_ptrs). fused_qkv_rope_cache needs additional
// pointer families — cos_sin_cache, positions, slot_mapping,
// key_cache, value_cache. Phase 3f part 2b-iii (codegen dispatch)
// will extend EmitCtx with those families; this smoke passes them
// as extra positional kernel args so we can validate the op numerics
// ahead of the ABI extension. act_ptrs carries (x_in, q_out);
// weight_ptrs carries (w_packed) — those two pass through the
// existing pool shape unchanged.
//
// Dimensions: Llama-3.2-1B decode — HIDDEN_DIM=2048, HEAD_DIM=64,
// NUM_Q_HEADS=32, NUM_KV_HEADS=8. NUM_HEADS_TOTAL=48, PACKED_N=3072.
//
// Build (on pod, CUDA 12.9, H100 sm_90a):
//   nvcc -O3 -std=c++20 -arch=sm_90a \
//     -DKITTENS_HOPPER --extended-lambda --expt-relaxed-constexpr \
//     -I /home/nickm/vllm-mega/vllm-rs/third_party/thunderkittens/include \
//     -I /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/tk \
//     /home/nickm/vllm-mega/vllm-rs/crates/ferrite-kernels/csrc/smoke/ferrite_fused_qkv_rope_cache_smoke.cu \
//     -o /tmp/ferrite_fused_qkv_rope_cache_smoke -lcuda
//
// Run: /tmp/ferrite_fused_qkv_rope_cache_smoke

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
#include "ferrite_kernels/fused_qkv_rope_cache.cuh"

namespace {

// ---- Model dims.
static constexpr int HIDDEN_DIM       = 2048;
static constexpr int HEAD_DIM         = 64;
static constexpr int NUM_Q_HEADS      = 32;
static constexpr int NUM_KV_HEADS     = 8;
static constexpr int NUM_HEADS_TOTAL  = NUM_Q_HEADS + 2 * NUM_KV_HEADS;  // 48
static constexpr int PACKED_N         = NUM_HEADS_TOTAL * HEAD_DIM;      // 3072
static constexpr int BLOCK_SIZE       = 16;
static constexpr int NUM_KV_BLOCKS    = 4;
static constexpr int MAX_POS          = 128;
static constexpr int NUM_TOKENS       = 4;     // batched decode smoke
static constexpr bool BIASED          = false;
static constexpr bool INTERLEAVED     = false;

// ---- FerriteConfig — 4 pages (act + wx + wy + cos_sin), each sized
// HIDDEN_DIM*2 bytes = 4096 (cos_sin only needs HEAD_DIM*2 but shares
// the page-size knob to keep the loader's TMA pattern uniform).
struct FerriteConfig {
    static constexpr int NUM_CONSUMER_WARPS      = 4;
    static constexpr int NON_CONSUMER_REGISTERS  = 64;
    static constexpr int CONSUMER_REGISTERS      = 192;
    static constexpr int NUM_PAGES               = 4;
    static constexpr int PAGE_SIZE               = 4096;  // HIDDEN_DIM * 2
    static constexpr int SCRATCH_BYTES           = 256;
    static constexpr int INSTRUCTION_PIPE_STAGES = 2;
};

static constexpr int NUM_LAYERS          = 1;
static constexpr int NUM_ACT_SLOTS       = 2;   // [x_in, q_out]
static constexpr int NUM_WEIGHT_ACCESSORS = 1;  // [w_packed]
static constexpr int TPB                 = (FerriteConfig::NUM_CONSUMER_WARPS + 3) * 32;

using SS = ferrite::SharedState<FerriteConfig>;

// ---- Walker bodies — mirror what `emit_cu_variant` will emit for a
// single FusedQkvRopeCache op at base_stage=0 once 2b-iii lands. Slot 0
// = x_in (the only slot the loader reads); slot 1 = q_out (the storer
// target for Q heads). The four extra pointer families come in as
// separate positional args — 2b-iii will thread them through extended
// EmitCtx closures.

__device__ __forceinline__ void consumer_body(
    SS& ss, int warp_in_role, int tok)
{
    ferrite::ops::fused_qkv_rope_cache::consumer<
        FerriteConfig, HIDDEN_DIM, HEAD_DIM,
        NUM_Q_HEADS, NUM_KV_HEADS, BIASED, INTERLEAVED>(
            ss, /*base_stage=*/0, warp_in_role, tok);
}

__device__ __forceinline__ void loader_body(
    const __nv_bfloat16* x,
    const __nv_bfloat16* w_packed,
    const __nv_bfloat16* cos_sin_cache,
    const uint32_t*      positions,
    SS& ss, int tok)
{
    ferrite::ops::fused_qkv_rope_cache::loader<
        FerriteConfig, HIDDEN_DIM, HEAD_DIM,
        NUM_Q_HEADS, NUM_KV_HEADS, BIASED, INTERLEAVED>(
            x, w_packed, cos_sin_cache, positions, ss, /*base_stage=*/0, tok);
}

__device__ __forceinline__ void launcher_body(SS& ss)
{
    ferrite::ops::fused_qkv_rope_cache::launcher<
        FerriteConfig, HIDDEN_DIM, HEAD_DIM,
        NUM_Q_HEADS, NUM_KV_HEADS, BIASED, INTERLEAVED>(
            ss, /*base_stage=*/0);
}

__device__ __forceinline__ void storer_body(
    __nv_bfloat16*       q_out,
    __nv_bfloat16*       key_cache,
    __nv_bfloat16*       value_cache,
    const int64_t*       slot_mapping,
    SS& ss, int tok)
{
    ferrite::ops::fused_qkv_rope_cache::storer<
        FerriteConfig, HIDDEN_DIM, HEAD_DIM,
        NUM_Q_HEADS, NUM_KV_HEADS, BLOCK_SIZE, BIASED, INTERLEAVED>(
            q_out, key_cache, value_cache, slot_mapping, ss, /*base_stage=*/0, tok);
}

__global__ void fused_qkv_rope_cache_smoke_kernel(
    __nv_bfloat16* const*        __restrict__ act_ptrs,       // [x_in, q_out]
    const __nv_bfloat16* const*  __restrict__ weight_ptrs,    // [w_packed]
    const __nv_bfloat16*         __restrict__ cos_sin_cache,  // [MAX_POS, HEAD_DIM]
    const uint32_t*              __restrict__ positions,      // [NUM_TOKENS]
    __nv_bfloat16*               __restrict__ key_cache,      // [num_blocks, BS, KV_H, HEAD]
    __nv_bfloat16*               __restrict__ value_cache,    // same shape
    const int64_t*               __restrict__ slot_mapping    // [NUM_TOKENS]
) {
    __shared__ SS ss;
    ferrite::init_shared_state<FerriteConfig>(ss);

    const __nv_bfloat16*  x_in     = act_ptrs[0];
    __nv_bfloat16*        q_out    = act_ptrs[1];
    const __nv_bfloat16*  w_packed = weight_ptrs[0u * NUM_LAYERS + 0u];

    const int wid = kittens::warpid();
    if (wid < FerriteConfig::NUM_CONSUMER_WARPS) {
        ferrite::set_consumer_registers<FerriteConfig>();
        for (int __tok = 0; __tok < NUM_TOKENS; ++__tok) {
            if (__tok > 0) __syncthreads();
            consumer_body(ss, wid, __tok);
        }
    } else {
        ferrite::set_non_consumer_registers<FerriteConfig>();
        switch (wid - FerriteConfig::NUM_CONSUMER_WARPS) {
            case ferrite::kLoaderSlot:
                for (int __tok = 0; __tok < NUM_TOKENS; ++__tok) {
                    if (__tok > 0) __syncthreads();
                    loader_body(x_in, w_packed, cos_sin_cache, positions, ss, __tok);
                }
                break;
            case ferrite::kLauncherSlot:
                for (int __tok = 0; __tok < NUM_TOKENS; ++__tok) {
                    if (__tok > 0) __syncthreads();
                    launcher_body(ss);
                }
                break;
            case ferrite::kStorerSlot:
                for (int __tok = 0; __tok < NUM_TOKENS; ++__tok) {
                    if (__tok > 0) __syncthreads();
                    storer_body(q_out, key_cache, value_cache, slot_mapping, ss, __tok);
                }
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

// ---- CPU reference.
//
// classify_head mirrors the header: Q heads occupy [0, NUM_Q_HEADS),
// K heads [NUM_Q_HEADS, NUM_Q_HEADS+NUM_KV_HEADS), V heads next.
// packed_row_start locates the per-head first row in the packed
// [PACKED_N, HIDDEN_DIM] weight: Q at h * HEAD_DIM; K at Q_SIZE +
// kv_h * HEAD_DIM; V at Q_SIZE + KV_SIZE + kv_h * HEAD_DIM.

enum class HeadFamily { Q, K, V };

struct HeadInfo {
    HeadFamily family;
    int intra;
    int packed_row_start;
};

HeadInfo classify_head_ref(int head_idx) {
    constexpr int Q_SIZE  = NUM_Q_HEADS  * HEAD_DIM;
    constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM;
    HeadInfo info;
    if (head_idx < NUM_Q_HEADS) {
        info.family = HeadFamily::Q;
        info.intra  = head_idx;
        info.packed_row_start = head_idx * HEAD_DIM;
    } else if (head_idx < NUM_Q_HEADS + NUM_KV_HEADS) {
        info.family = HeadFamily::K;
        info.intra  = head_idx - NUM_Q_HEADS;
        info.packed_row_start = Q_SIZE + info.intra * HEAD_DIM;
    } else {
        info.family = HeadFamily::V;
        info.intra  = head_idx - NUM_Q_HEADS - NUM_KV_HEADS;
        info.packed_row_start = Q_SIZE + KV_SIZE + info.intra * HEAD_DIM;
    }
    return info;
}

// Process one token. `x_tok` points to the [HIDDEN_DIM] activation for
// this token. q_out_ref is the full [NUM_TOKENS, NUM_Q_HEADS, HEAD_DIM]
// array; tok_offset selects this token's slice.
void cpu_reference_one_token(
    const __nv_bfloat16*              x_tok,     // [HIDDEN_DIM]
    const std::vector<__nv_bfloat16>& w_packed,
    const std::vector<__nv_bfloat16>& cos_sin_cache,
    uint32_t pos,
    int64_t  slot,
    int      tok_q_offset,                       // = tok * NUM_Q_HEADS * HEAD_DIM
    std::vector<__nv_bfloat16>&       q_out_ref,
    std::vector<__nv_bfloat16>&       key_cache_ref,
    std::vector<__nv_bfloat16>&       value_cache_ref)
{
    constexpr int embed_dim = HEAD_DIM / 2;

    for (int h = 0; h < NUM_HEADS_TOTAL; ++h) {
        HeadInfo info = classify_head_ref(h);
        for (int p = 0; p < embed_dim; ++p) {
            int x_row = info.packed_row_start + p;
            int y_row = info.packed_row_start + p + embed_dim;
            float acc_x = 0.0f, acc_y = 0.0f;
            for (int k = 0; k < HIDDEN_DIM; ++k) {
                float xv = __bfloat162float(x_tok[k]);
                float wx = __bfloat162float(w_packed[x_row * HIDDEN_DIM + k]);
                float wy = __bfloat162float(w_packed[y_row * HIDDEN_DIM + k]);
                acc_x += xv * wx;
                acc_y += xv * wy;
            }
            float out_x, out_y;
            if (info.family != HeadFamily::V) {
                // cos_sin row layout per header: [cos_0..cos_{E-1}, sin_0..sin_{E-1}].
                float cos_v = __bfloat162float(cos_sin_cache[pos * HEAD_DIM + p]);
                float sin_v = __bfloat162float(cos_sin_cache[pos * HEAD_DIM + p + embed_dim]);
                out_x = acc_x * cos_v - acc_y * sin_v;
                out_y = acc_y * cos_v + acc_x * sin_v;
            } else {
                out_x = acc_x;
                out_y = acc_y;
            }

            if (info.family == HeadFamily::Q) {
                size_t base = size_t(tok_q_offset) + size_t(info.intra) * HEAD_DIM;
                q_out_ref[base + p]             = __float2bfloat16_rn(out_x);
                q_out_ref[base + p + embed_dim] = __float2bfloat16_rn(out_y);
            } else {
                if (slot < 0) continue;  // padded-token short-circuit.
                int64_t block_idx = slot / BLOCK_SIZE;
                int64_t block_off = slot % BLOCK_SIZE;
                size_t block_stride = size_t(BLOCK_SIZE) * NUM_KV_HEADS * HEAD_DIM;
                size_t page_stride  = size_t(NUM_KV_HEADS) * HEAD_DIM;
                size_t head_stride  = HEAD_DIM;
                size_t base =
                    size_t(block_idx) * block_stride
                  + size_t(block_off) * page_stride
                  + size_t(info.intra) * head_stride;
                auto& dst = (info.family == HeadFamily::K) ? key_cache_ref : value_cache_ref;
                dst[base + p]             = __float2bfloat16_rn(out_x);
                dst[base + p + embed_dim] = __float2bfloat16_rn(out_y);
            }
        }
    }
}

} // namespace

int main() {
    // ---- Host tensors.
    std::vector<__nv_bfloat16> h_x             (NUM_TOKENS * HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_w_packed      (size_t(PACKED_N) * HIDDEN_DIM);
    std::vector<__nv_bfloat16> h_cos_sin_cache (size_t(MAX_POS)  * HEAD_DIM);
    std::vector<uint32_t>      h_positions     (NUM_TOKENS);
    std::vector<int64_t>       h_slot_mapping  (NUM_TOKENS);

    uint32_t s = 0xC0FFEE17u;
    // Scale x and weights to keep fp32 accumulation in a narrow range
    // (K=2048 dot products of bf16s with |.|<=1 can reach ~45; rope
    // rotation preserves magnitude).
    for (auto& v : h_x)        v = __float2bfloat16_rn(0.5f * rand_u(s));
    for (auto& v : h_w_packed) v = __float2bfloat16_rn(0.1f * rand_u(s));
    // cos/sin cache layout [MAX_POS, HEAD_DIM] with [cos_0..cos_{E-1},
    // sin_0..sin_{E-1}] per row — matches the header's pointer math.
    // Use a synthetic RoPE frequency schedule (theta base = 10000) so
    // the values are stable and representative.
    constexpr int embed_dim = HEAD_DIM / 2;
    for (int pos = 0; pos < MAX_POS; ++pos) {
        for (int i = 0; i < embed_dim; ++i) {
            float inv_freq = std::pow(10000.0f, -float(2 * i) / float(HEAD_DIM));
            float angle    = float(pos) * inv_freq;
            h_cos_sin_cache[pos * HEAD_DIM + i]             = __float2bfloat16_rn(std::cos(angle));
            h_cos_sin_cache[pos * HEAD_DIM + i + embed_dim] = __float2bfloat16_rn(std::sin(angle));
        }
    }
    // NUM_TOKENS distinct positions and slots — spread across different
    // blocks so they don't alias in the KV cache.
    for (int t = 0; t < NUM_TOKENS; ++t) {
        h_positions[t]    = uint32_t(7 + t * 3);   // 7, 10, 13, 16, ...
        h_slot_mapping[t] = int64_t(t * 4 + 3);    // slots 3, 7, 11, 15 → different blocks
    }

    // ---- CPU reference. Caches init'd to zero; the op only writes to
    // the slots in slot_mapping, so the rest of the buffer should
    // match zero on both sides.
    const size_t kv_cache_count =
        size_t(NUM_KV_BLOCKS) * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
    std::vector<__nv_bfloat16> h_q_ref        (NUM_TOKENS * NUM_Q_HEADS * HEAD_DIM,
                                                __float2bfloat16_rn(0.0f));
    std::vector<__nv_bfloat16> h_k_cache_ref  (kv_cache_count, __float2bfloat16_rn(0.0f));
    std::vector<__nv_bfloat16> h_v_cache_ref  (kv_cache_count, __float2bfloat16_rn(0.0f));
    for (int t = 0; t < NUM_TOKENS; ++t) {
        cpu_reference_one_token(
            h_x.data() + t * HIDDEN_DIM,
            h_w_packed, h_cos_sin_cache,
            h_positions[t], h_slot_mapping[t],
            t * NUM_Q_HEADS * HEAD_DIM,
            h_q_ref, h_k_cache_ref, h_v_cache_ref);
    }

    // ---- Device buffers.
    __nv_bfloat16 *d_x = nullptr, *d_w_packed = nullptr, *d_cos_sin_cache = nullptr;
    __nv_bfloat16 *d_q_out = nullptr, *d_k_cache = nullptr, *d_v_cache = nullptr;
    uint32_t      *d_positions = nullptr;
    int64_t       *d_slot_mapping = nullptr;

    const size_t x_bytes        = NUM_TOKENS * HIDDEN_DIM * sizeof(__nv_bfloat16);
    const size_t w_bytes        = size_t(PACKED_N) * HIDDEN_DIM * sizeof(__nv_bfloat16);
    const size_t cos_sin_bytes  = size_t(MAX_POS) * HEAD_DIM * sizeof(__nv_bfloat16);
    const size_t q_out_bytes    = NUM_TOKENS * NUM_Q_HEADS * HEAD_DIM * sizeof(__nv_bfloat16);
    const size_t kv_cache_bytes = kv_cache_count * sizeof(__nv_bfloat16);
    const size_t pos_bytes      = NUM_TOKENS * sizeof(uint32_t);
    const size_t slot_bytes     = NUM_TOKENS * sizeof(int64_t);

    CUDA_CHECK(cudaMalloc(&d_x,             x_bytes));
    CUDA_CHECK(cudaMalloc(&d_w_packed,      w_bytes));
    CUDA_CHECK(cudaMalloc(&d_cos_sin_cache, cos_sin_bytes));
    CUDA_CHECK(cudaMalloc(&d_q_out,         q_out_bytes));
    CUDA_CHECK(cudaMalloc(&d_k_cache,       kv_cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_cache,       kv_cache_bytes));
    CUDA_CHECK(cudaMalloc(&d_positions,     pos_bytes));
    CUDA_CHECK(cudaMalloc(&d_slot_mapping,  slot_bytes));

    CUDA_CHECK(cudaMemcpy(d_x,             h_x.data(),             x_bytes,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_packed,      h_w_packed.data(),      w_bytes,        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos_sin_cache, h_cos_sin_cache.data(), cos_sin_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positions,     h_positions.data(),     pos_bytes,      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot_mapping,  h_slot_mapping.data(),  slot_bytes,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_q_out,   0, q_out_bytes));
    CUDA_CHECK(cudaMemset(d_k_cache, 0, kv_cache_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache, 0, kv_cache_bytes));

    // ---- Pool-ABI pointer arrays on device.
    __nv_bfloat16* h_act_ptrs[NUM_ACT_SLOTS] = { d_x, d_q_out };
    const __nv_bfloat16* h_w_ptrs[NUM_WEIGHT_ACCESSORS * NUM_LAYERS] = { d_w_packed };
    __nv_bfloat16**       d_act_ptrs = nullptr;
    const __nv_bfloat16** d_w_ptrs   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_act_ptrs, sizeof(h_act_ptrs)));
    CUDA_CHECK(cudaMalloc(&d_w_ptrs,   sizeof(h_w_ptrs)));
    CUDA_CHECK(cudaMemcpy(d_act_ptrs, h_act_ptrs, sizeof(h_act_ptrs), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w_ptrs,   h_w_ptrs,   sizeof(h_w_ptrs),   cudaMemcpyHostToDevice));

    // ---- Launch.
    dim3 grid(HEAD_DIM / 2, NUM_HEADS_TOTAL, 1);  // (32, 48)
    dim3 block(TPB, 1, 1);
    fused_qkv_rope_cache_smoke_kernel<<<grid, block, 0, 0>>>(
        d_act_ptrs, d_w_ptrs,
        d_cos_sin_cache, d_positions,
        d_k_cache, d_v_cache, d_slot_mapping);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        std::fprintf(stderr, "launch error: %s\n", cudaGetErrorString(launch_err));
        return 2;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Read back.
    std::vector<__nv_bfloat16> h_q_got       (NUM_TOKENS * NUM_Q_HEADS * HEAD_DIM);
    std::vector<__nv_bfloat16> h_k_cache_got (kv_cache_count);
    std::vector<__nv_bfloat16> h_v_cache_got (kv_cache_count);
    CUDA_CHECK(cudaMemcpy(h_q_got.data(),       d_q_out,   q_out_bytes,    cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_k_cache_got.data(), d_k_cache, kv_cache_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v_cache_got.data(), d_v_cache, kv_cache_bytes, cudaMemcpyDeviceToHost));

    auto compare = [&](const char* label, float tol,
                       const std::vector<__nv_bfloat16>& got,
                       const std::vector<__nv_bfloat16>& ref) -> int {
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
        std::printf("%s: max_abs=%.6f rel_l2=%.6f mismatches(>%.3f)=%d\n",
                    label, max_abs, rel_l2, tol, mismatches);
        if (mismatches > 0) {
            std::printf("  first mismatch at idx=%d got=%.6f ref=%.6f diff=%.6f\n",
                        first_mismatch_idx, first_got, first_ref,
                        std::fabs(first_got - first_ref));
        }
        return mismatches;
    };

    std::printf("num_tokens=%d hidden_dim=%d head_dim=%d q_heads=%d kv_heads=%d\n",
                NUM_TOKENS, HIDDEN_DIM, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS);
    std::printf("pos=%ld slot=%ld block_size=%d num_blocks=%d\n",
                long(kPos), long(kSlot), BLOCK_SIZE, NUM_KV_BLOCKS);

    // Tolerances: K=2048 bf16 dot product max_abs in the gemv smoke was
    // ~0.066 at |x|<=1, |w|<=1. Here we use 0.5*|x|, 0.1*|w|, so acc ~
    // 0.05 * (K/3) on average; the RoPE fold is a rotation of two such
    // terms so max_abs scales similarly. 0.05 is comfortable for Q/K,
    // V is unfolded so can be a touch tighter.
    int m_q = compare("q_out   (Q family)",       0.05f, h_q_got,       h_q_ref);
    int m_k = compare("key_cache (K at slot)",    0.05f, h_k_cache_got, h_k_cache_ref);
    int m_v = compare("value_cache (V at slot)",  0.03f, h_v_cache_got, h_v_cache_ref);
    int mismatches = m_q + m_k + m_v;
    if (mismatches == 0) {
        std::printf("ok: fused_qkv_rope_cache matches CPU reference\n");
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w_packed));
    CUDA_CHECK(cudaFree(d_cos_sin_cache));
    CUDA_CHECK(cudaFree(d_q_out));
    CUDA_CHECK(cudaFree(d_k_cache));
    CUDA_CHECK(cudaFree(d_v_cache));
    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaFree(d_slot_mapping));
    CUDA_CHECK(cudaFree(d_act_ptrs));
    CUDA_CHECK(cudaFree(d_w_ptrs));
    return mismatches == 0 ? 0 : 3;
}
