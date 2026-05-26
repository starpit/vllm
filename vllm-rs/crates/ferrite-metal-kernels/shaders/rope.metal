// SPDX-License-Identifier: Apache-2.0
//! Rotary Position Embedding (RoPE) Metal shaders
//!
//! Implements both NeoX-style and GPT-J-style rotary embeddings:
//! - NeoX: pairs element i with i + half_dim
//! - GPT-J (interleaved): pairs element 2i with 2i+1
//!
//! Algorithm:
//! For each pair (x, y):
//!   x' = x * cos(θ) - y * sin(θ)
//!   y' = y * cos(θ) + x * sin(θ)
//!
//! Where θ is position-dependent and pre-computed in cos_sin_cache.

#include <metal_stdlib>
using namespace metal;

// ThunderMittens — the NeoX RoPE rotation atom the rope_append_*_specialized
// wrappers and the wavefront megakernel below compose.
#include "mittens/rope.h"

// ---------------------------------------------------------------------------
// NeoX-style RoPE (standard Llama, GPT-NeoX)
// ---------------------------------------------------------------------------

/// Apply rotary embedding to a single token's query/key vectors.
/// NeoX style: pairs element i with i + half_dim.
///
/// @param query: [num_heads, head_size] - query vector for this token
/// @param key: [num_kv_heads, head_size] - key vector for this token (nullable)
/// @param cos_sin_cache: [rot_dim] - concatenated [cos; sin] for this position
/// @param num_heads: number of query heads
/// @param num_kv_heads: number of key heads
/// @param rot_dim: rotary dimension (typically head_size or head_size/2)
/// @param head_size: size of each head
kernel void rope_neox_f16(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    constant half* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant half* cos_ptr = cos_sin_cache;
    constant half* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = rot_offset;
        const uint y_index = embed_dim + rot_offset;
        
        const half cos_val = cos_ptr[x_index];
        const half sin_val = sin_ptr[x_index];
        
        device half* head_ptr = query + head_idx * head_size;
        const half x = head_ptr[x_index];
        const half y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = rot_offset;
            const uint y_index = embed_dim + rot_offset;
            
            const half cos_val = cos_ptr[x_index];
            const half sin_val = sin_ptr[x_index];
            
            device half* head_ptr = key + head_idx * head_size;
            const half x = head_ptr[x_index];
            const half y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}

/// BFloat16 variant of NeoX-style RoPE
kernel void rope_neox_bf16(
    device bfloat* query [[buffer(0)]],
    device bfloat* key [[buffer(1)]],
    constant bfloat* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant bfloat* cos_ptr = cos_sin_cache;
    constant bfloat* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = rot_offset;
        const uint y_index = embed_dim + rot_offset;
        
        const bfloat cos_val = cos_ptr[x_index];
        const bfloat sin_val = sin_ptr[x_index];
        
        device bfloat* head_ptr = query + head_idx * head_size;
        const bfloat x = head_ptr[x_index];
        const bfloat y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = rot_offset;
            const uint y_index = embed_dim + rot_offset;
            
            const bfloat cos_val = cos_ptr[x_index];
            const bfloat sin_val = sin_ptr[x_index];
            
            device bfloat* head_ptr = key + head_idx * head_size;
            const bfloat x = head_ptr[x_index];
            const bfloat y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Interleaved RoPE (GPT-J style, Cohere CommandR)
// ---------------------------------------------------------------------------

/// Apply rotary embedding with interleaved pairing.
/// GPT-J style: pairs element 2i with 2i+1.
///
/// Used by Cohere's CommandR family.
kernel void rope_interleaved_f16(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    constant half* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant half* cos_ptr = cos_sin_cache;
    constant half* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = 2 * rot_offset;
        const uint y_index = 2 * rot_offset + 1;
        
        const half cos_val = cos_ptr[rot_offset];
        const half sin_val = sin_ptr[rot_offset];
        
        device half* head_ptr = query + head_idx * head_size;
        const half x = head_ptr[x_index];
        const half y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = 2 * rot_offset;
            const uint y_index = 2 * rot_offset + 1;
            
            const half cos_val = cos_ptr[rot_offset];
            const half sin_val = sin_ptr[rot_offset];
            
            device half* head_ptr = key + head_idx * head_size;
            const half x = head_ptr[x_index];
            const half y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 5.G.3: rope_append_f16_specialized — paged-cache RoPE writer
//
// In-place NeoX-style RoPE on Q and K, plus paged write of (rotated K,
// un-rotated V) into the per-layer KV cache. Mirrors
// `Instruction::RopeAppend` (CUDA) / `KernelId::RopeAppend` (Metal).
//
// Function constants (must match
// `ferrite_forward::interpreter::metal::pipelines::constants_for`):
//   0 = HEAD_DIM
//   1 = NUM_Q_HEADS
//   2 = NUM_KV_HEADS
//   3 = ROT_DIM     (typically == HEAD_DIM; partial-rope models pass < HEAD_DIM)
//   4 = BLOCK_SIZE  (paged KV cache page size)
//
// Bindings (must match `interpreter::metal::lowering::lower_one` for
// `Instruction::RopeAppend`):
//   buffer(0) = q_inout       [bucket_m, NUM_Q_HEADS  * HEAD_DIM]   in/out
//   buffer(1) = k_inout       [bucket_m, NUM_KV_HEADS * HEAD_DIM]   in/out
//   buffer(2) = v_inout       [bucket_m, NUM_KV_HEADS * HEAD_DIM]   in/out (un-rotated; cache copy only)
//   buffer(3) = cos_sin       [max_pos, ROT_DIM] — row = [cos[half] | sin[half]]
//   buffer(4) = positions     [bucket_m]
//   buffer(5) = slot_mapping  [bucket_m] — global cache slot per token
//   buffer(6) = kv_cache_k    [num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
//   buffer(7) = kv_cache_v    same shape as kv_cache_k
//
// Dispatch (set by `interpreter::metal::lowering::lower_one`):
//   threadgroups: (bucket_m, NUM_Q_HEADS, 1)
//   threads_per_threadgroup: (HEAD_DIM, 1, 1)
//
// Per (token, q_head) threadgroup:
//   - Threads with `d < ROT_DIM/2` rotate the (d, d+half) pair of
//     `q_inout[token, q_head, :]`.
//   - Threadgroups whose q_head owns a kv_head (i.e. `q_head %
//     group_ratio == 0` where `group_ratio = NUM_Q_HEADS / NUM_KV_HEADS`)
//     additionally:
//       a. Rotate `k_inout[token, kv_head, :]` (same pair shape).
//       b. Copy the rotated K and un-rotated V into the cache slot.
//   - Other q_heads do Q only.
//
// No threadgroup_barrier needed: each thread reads-then-writes its own
// (d, d+half) pair before any other thread touches the same indices,
// and the K/V paged write happens after the K rotation in the same
// thread (sequential dependency).
// ---------------------------------------------------------------------------

constant uint ROPE_HEAD_DIM     [[function_constant(0)]];
constant uint ROPE_NUM_Q_HEADS  [[function_constant(1)]];
constant uint ROPE_NUM_KV_HEADS [[function_constant(2)]];
constant uint ROPE_ROT_DIM      [[function_constant(3)]];
constant uint ROPE_BLOCK_SIZE   [[function_constant(4)]];

kernel void rope_append_f16_specialized(
    device       half* q_inout      [[buffer(0)]],
    device       half* k_inout      [[buffer(1)]],
    device       half* v_inout      [[buffer(2)]],
    device const half* cos_sin      [[buffer(3)]],
    device const uint* positions    [[buffer(4)]],
    device const uint* slot_mapping [[buffer(5)]],
    device       half* kv_cache_k   [[buffer(6)]],
    device       half* kv_cache_v   [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid    [[thread_position_in_threadgroup]])
{
    const uint t        = tg_pos.x;
    const uint q_head   = tg_pos.y;
    const uint d        = tid.x;
    const uint head_dim = ROPE_HEAD_DIM;
    const uint rot_dim  = ROPE_ROT_DIM;
    const uint half_dim = rot_dim / 2;
    const uint num_q    = ROPE_NUM_Q_HEADS;
    const uint num_kv   = ROPE_NUM_KV_HEADS;
    const uint block_sz = ROPE_BLOCK_SIZE;
    const uint group_r  = num_q / num_kv;

    if (q_head >= num_q || d >= head_dim) return;

    const uint pos = positions[t];
    device const half* cos_row = cos_sin + pos * rot_dim;
    device const half* sin_row = cos_sin + pos * rot_dim + half_dim;

    // ── Q rotation (in-place) ────────────────────────────────────────
    const uint q_dim = num_q * head_dim;
    device half* q_row = q_inout + t * q_dim + q_head * head_dim;
    mittens::rope_rotate_pair<half>(q_row, cos_row, sin_row, d, half_dim);

    // ── K/V rotation + paged write (only owning q_head per kv_head) ─
    if (q_head % group_r != 0) return;
    const uint kv_head = q_head / group_r;
    const uint kv_dim  = num_kv * head_dim;
    device half* k_row = k_inout + t * kv_dim + kv_head * head_dim;
    device half* v_row = v_inout + t * kv_dim + kv_head * head_dim;

    // K rotation (in-place).
    mittens::rope_rotate_pair<half>(k_row, cos_row, sin_row, d, half_dim);
    // Fence the K writes — the paged write below has thread `d` read
    // `k_row[d]`, which (for d ≥ half_dim) was written by thread
    // `d - half_dim`. Without the barrier the paged write may see the
    // pre-rotation half.
    threadgroup_barrier(mem_flags::mem_device);

    // Paged write: kv_cache layout [num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM].
    // Sentinel `0xFFFFFFFF` marks padding lanes (write_slot_mapping in
    // pool.rs fills padding with u32::MAX) — skip the cache write so
    // padding's K_proj(token 0) does not corrupt slot 0.
    const uint slot         = slot_mapping[t];
    if (slot == 0xFFFFFFFFu) return;
    const uint block_id     = slot / block_sz;
    const uint block_offset = slot % block_sz;
    const uint kv_blk_stride  = num_kv * block_sz * head_dim;
    const uint kv_head_stride = block_sz * head_dim;
    const uint kv_tok_stride  = head_dim;
    device half* k_dst = kv_cache_k
        + block_id     * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;
    device half* v_dst = kv_cache_v
        + block_id     * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;

    // Each thread copies one element of K (rotated, post-write above)
    // and V (un-rotated). For partial-rope models (rot_dim < head_dim),
    // the tail [rot_dim, head_dim) of k_row is unrotated and copied
    // through unchanged.
    k_dst[d] = k_row[d];
    v_dst[d] = v_row[d];
}

/// BF16 specialized variant — same dispatch shape, function constants,
/// rotation math, and paged-cache layout as the f16 variant. Bindings
/// switch to `device bfloat*`; the cos_sin cache must be uploaded as
/// bf16 too (`upload_via_gpuweights` honors the dtype passed in).
kernel void rope_append_bf16_specialized(
    device       bfloat* q_inout      [[buffer(0)]],
    device       bfloat* k_inout      [[buffer(1)]],
    device       bfloat* v_inout      [[buffer(2)]],
    device const bfloat* cos_sin      [[buffer(3)]],
    device const uint*   positions    [[buffer(4)]],
    device const uint*   slot_mapping [[buffer(5)]],
    device       bfloat* kv_cache_k   [[buffer(6)]],
    device       bfloat* kv_cache_v   [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid    [[thread_position_in_threadgroup]])
{
    const uint t        = tg_pos.x;
    const uint q_head   = tg_pos.y;
    const uint d        = tid.x;
    const uint head_dim = ROPE_HEAD_DIM;
    const uint rot_dim  = ROPE_ROT_DIM;
    const uint half_dim = rot_dim / 2;
    const uint num_q    = ROPE_NUM_Q_HEADS;
    const uint num_kv   = ROPE_NUM_KV_HEADS;
    const uint block_sz = ROPE_BLOCK_SIZE;
    const uint group_r  = num_q / num_kv;

    if (q_head >= num_q || d >= head_dim) return;

    const uint pos = positions[t];
    device const bfloat* cos_row = cos_sin + pos * rot_dim;
    device const bfloat* sin_row = cos_sin + pos * rot_dim + half_dim;

    const uint q_dim = num_q * head_dim;
    device bfloat* q_row = q_inout + t * q_dim + q_head * head_dim;
    mittens::rope_rotate_pair<bfloat>(q_row, cos_row, sin_row, d, half_dim);

    if (q_head % group_r != 0) return;
    const uint kv_head = q_head / group_r;
    const uint kv_dim  = num_kv * head_dim;
    device bfloat* k_row = k_inout + t * kv_dim + kv_head * head_dim;
    device bfloat* v_row = v_inout + t * kv_dim + kv_head * head_dim;

    mittens::rope_rotate_pair<bfloat>(k_row, cos_row, sin_row, d, half_dim);
    threadgroup_barrier(mem_flags::mem_device);

    // Sentinel `0xFFFFFFFF` marks padding lanes (write_slot_mapping in
    // pool.rs fills padding with u32::MAX) — skip the cache write so
    // padding's K_proj(token 0) does not corrupt slot 0.
    const uint slot         = slot_mapping[t];
    if (slot == 0xFFFFFFFFu) return;
    const uint block_id     = slot / block_sz;
    const uint block_offset = slot % block_sz;
    const uint kv_blk_stride  = num_kv * block_sz * head_dim;
    const uint kv_head_stride = block_sz * head_dim;
    const uint kv_tok_stride  = head_dim;
    device bfloat* k_dst = kv_cache_k
        + block_id     * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;
    device bfloat* v_dst = kv_cache_v
        + block_id     * kv_blk_stride
        + kv_head      * kv_head_stride
        + block_offset * kv_tok_stride;

    k_dst[d] = k_row[d];
    v_dst[d] = v_row[d];
}

// ─────────────────────────────────────────────────────────────────
// wavefront_rope_mega — PD-wavefront RoPE rotation composition proof.
// P co-resident threadgroups (one per GPU core), each with HEAD_DIM threads,
// loop the flattened (token, q_head) pairs they own (g = tgpos, +grid_tg, …)
// composing mittens::rope_rotate_pair for Q (+ K on the owning q_head per
// kv_head). ROTATION ONLY — no paged-cache write (design #4: the megakernel
// keeps rotated K as a dataflow edge into attention), so q_inout/k_inout must
// be BIT-EXACT vs the whole rope_append_*_specialized (whose cache write +
// barrier never touch q_inout/k_inout). Each (t, qh) is rotated once (disjoint
// striping) and the rotation is per-element ⇒ no reduction order to preserve.
// Constant 5 = NUM_TOKENS (mega-only; the whole kernel gets it from the grid).
// ─────────────────────────────────────────────────────────────────
constant uint ROPE_NUM_TOKENS [[function_constant(5)]];

template <typename T>
[[kernel]] void wavefront_rope_mega(
    device       T*    q_inout   [[buffer(0)]],
    device       T*    k_inout   [[buffer(1)]],
    device const T*    cos_sin   [[buffer(2)]],
    device const uint* positions [[buffer(3)]],
    uint tgpos   [[threadgroup_position_in_grid]],
    uint grid_tg [[threadgroups_per_grid]],
    uint d       [[thread_position_in_threadgroup]]) {
  const uint head_dim = ROPE_HEAD_DIM;
  const uint rot_dim  = ROPE_ROT_DIM;
  const uint half_dim = rot_dim / 2;
  const uint num_q    = ROPE_NUM_Q_HEADS;
  const uint num_kv   = ROPE_NUM_KV_HEADS;
  const uint group_r  = num_q / num_kv;
  const uint total    = ROPE_NUM_TOKENS * num_q; // flattened (token, q_head)

  for (uint g = tgpos; g < total; g += grid_tg) {
    const uint t  = g / num_q;
    const uint qh = g % num_q;
    const uint pos = positions[t];
    device const T* cos_row = cos_sin + pos * rot_dim;
    device const T* sin_row = cos_sin + pos * rot_dim + half_dim;

    device T* q_row = q_inout + t * (num_q * head_dim) + qh * head_dim;
    mittens::rope_rotate_pair<T>(q_row, cos_row, sin_row, d, half_dim);

    if (qh % group_r == 0) {
      const uint kv_head = qh / group_r;
      device T* k_row = k_inout + t * (num_kv * head_dim) + kv_head * head_dim;
      mittens::rope_rotate_pair<T>(k_row, cos_row, sin_row, d, half_dim);
    }
  }
}

#define INST_WF_ROPE_MEGA(tag, type)                                         \
  template [[host_name("wavefront_rope_mega_" #tag)]]                        \
  [[kernel]] decltype(wavefront_rope_mega<type>) wavefront_rope_mega<type>;
INST_WF_ROPE_MEGA(f16, half)
INST_WF_ROPE_MEGA(bf16, bfloat)

/// BFloat16 variant of interleaved RoPE
kernel void rope_interleaved_bf16(
    device bfloat* query [[buffer(0)]],
    device bfloat* key [[buffer(1)]],
    constant bfloat* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant bfloat* cos_ptr = cos_sin_cache;
    constant bfloat* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = 2 * rot_offset;
        const uint y_index = 2 * rot_offset + 1;
        
        const bfloat cos_val = cos_ptr[rot_offset];
        const bfloat sin_val = sin_ptr[rot_offset];
        
        device bfloat* head_ptr = query + head_idx * head_size;
        const bfloat x = head_ptr[x_index];
        const bfloat y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = 2 * rot_offset;
            const uint y_index = 2 * rot_offset + 1;
            
            const bfloat cos_val = cos_ptr[rot_offset];
            const bfloat sin_val = sin_ptr[rot_offset];
            
            device bfloat* head_ptr = key + head_idx * head_size;
            const bfloat x = head_ptr[x_index];
            const bfloat y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}
