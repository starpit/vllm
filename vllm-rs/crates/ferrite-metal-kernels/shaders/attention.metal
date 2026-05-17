// SPDX-License-Identifier: Apache-2.0
//
// Ferrite metal attention kernels — direct ports of MLX attention.
//
//   `attention_via_cache_v2_f16/bf16_specialized` —
//       paged-cache adaptation of MLX `sdpa_vector` from
//       `mlx/backend/metal/kernels/sdpa_vector.h`. Decode path.
//   `attention_prefill_sdpa_v2_paged_f16/bf16_specialized` —
//       Paged-cache prefill, shares the decode kernel's outer
//       structure (1 Q per TG, online softmax + per-simdgroup
//       K-axis split). K/V read through `block_table` indirection;
//       K-axis covers the FULL `seqused_k[seq]` (prefix + new), and
//       the per-Q causal mask shifts by `(seqused_k[seq] -
//       new_q_for_seq)` to account for prior cached prefix.
//       Required for chunked prefill, prefix caching, mixed
//       prefill/decode batches, and multi-turn chat continuation.
//       The only prefill kernel emitted on metal post-Phase B
//       (Phase B macro adapter at metal/attention.rs always emits
//       `Instruction::AttentionPrefillPaged`).
//
// Function constant indices (must match `pipelines::constants_for`):
//   0  ATTN_HEAD_DIM           uint
//   1  ATTN_NUM_Q_HEADS        uint
//   2  ATTN_NUM_KV_HEADS       uint
//   3  ATTN_SCALE_FC           float
//   4  ATTN_BLOCK_SIZE         uint
//   5  ATTN_MAX_BLOCKS_PER_SEQ uint

#include <metal_stdlib>
using namespace metal;



constant uint  ATTN_HEAD_DIM           [[function_constant(0)]];
constant uint  ATTN_NUM_Q_HEADS        [[function_constant(1)]];
constant uint  ATTN_NUM_KV_HEADS       [[function_constant(2)]];
constant float ATTN_SCALE_FC           [[function_constant(3)]];
constant uint  ATTN_BLOCK_SIZE         [[function_constant(4)]];
constant uint  ATTN_MAX_BLOCKS_PER_SEQ [[function_constant(5)]];

// Cap on `seq_used_k[seq]` the shared-logits buffer can hold.
// Each token uses 4 bytes; this cap × 4 == threadgroup memory bytes
// dedicated to the partial-logits scratch. Smaller is better for
// concurrent-threadgroup occupancy on Apple GPU clusters (each
// cluster reserves the per-threadgroup TGSM budget per resident
// group). 256 * 4 = 1KB still covers TinyLlama-class decode lengths
// without spilling logits to device memory; sequences longer than
// this cap fall back to a larger-cap pipeline (TODO).
#define ATTN_MAX_SHARED_LOGITS 256u

// ============================================================================
// attention_via_cache_v2_f16_specialized — paged-cache decode attention
// ============================================================================
//
// Paged-cache adaptation of MLX's `sdpa_vector` (from
// `mlx/backend/metal/kernels/sdpa_vector.h`). Key structural moves vs
// the original v1 kernel above:
//
//   1. Online softmax: max + sum_exp accumulated per-step in
//      registers; output accumulator is rescaled when a new max is
//      seen. No `shared_logits[]` threadgroup buffer, no two-pass
//      structure, no per-token `threadgroup_barrier` in the K loop.
//
//   2. BN simdgroups split the K-axis. Each simdgroup processes
//      keys at index `simd_gid, simd_gid + BN, simd_gid + 2*BN, ...`
//      so the work fans out across simdgroups without cross-simd
//      reductions in the inner loop.
//
//   3. Each lane handles `qk_per_thread = HEAD_DIM / 32 = 2`
//      elements of K and V. The dot product reduces inside one
//      simdgroup via `simd_sum`.
//
// The combine step at the end (after the K loop) collects per-
// simdgroup partials, reconciles their max+sum_exp via simd_max +
// simd_sum on threadgroup-mem-staged values, then produces the
// final output.
//
// Bindings (must match `interpreter::metal::lowering::lower_one`'s
// `Instruction::AttentionViaCache` arm):
//   buffer(0) = output      [batch, num_q_heads, head_dim]
//   buffer(1) = q           [batch, num_q_heads, head_dim]
//   buffer(2) = seq_used_k  [batch]
//   buffer(3) = block_table [batch, MAX_BLOCKS_PER_SEQ]
//   buffer(4) = k_cache     [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
//   buffer(5) = v_cache     [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
//
// Function constants 0..5: same as v1 (HEAD_DIM, NUM_Q_HEADS,
// NUM_KV_HEADS, ATTN_SCALE, BLOCK_SIZE, MAX_BLOCKS_PER_SEQ).
//
// Dispatch: threadgroups (batch, num_q_heads, 1), threads (1024, 1, 1)
// = 32 simdgroups × 32 lanes. The lowering pass picks this shape only
// for the v2 symbol; v1 kept around as a fallback for now.
//
// Constraint: HEAD_DIM must equal 32 * qk_per_thread (i.e. evenly
// divisible by 32). For TinyLlama HEAD_DIM=64 this means
// qk_per_thread = 2.

kernel void attention_via_cache_v2_f16_specialized(
    device       half* output      [[buffer(0)]],   // [batch, num_q_heads, head_dim]
    device const half* q           [[buffer(1)]],   // [batch, num_q_heads, head_dim]
    device const uint* seq_used_k  [[buffer(2)]],   // [batch]
    device const uint* block_table [[buffer(3)]],   // [batch, MAX_BLOCKS_PER_SEQ]
    device const half* k_cache     [[buffer(4)]],
    device const half* v_cache     [[buffer(5)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint3  tid       [[thread_position_in_threadgroup]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr int BN = 32; // simdgroups per threadgroup
    constexpr int BD = 32; // lanes per simdgroup
    typedef float U;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;

    // qk_per_thread = HEAD_DIM / 32. For TinyLlama 64/32 = 2.
    const uint qk_per_thread = head_dim / uint(BD);

    const uint seq_idx     = tg_pos.x;            // batch index
    const uint q_head_idx  = tg_pos.y;            // 0..NUM_Q_HEADS
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;
    const uint kv_len      = seq_used_k[seq_idx];

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    // Per-thread Q + accumulators (qk_per_thread should be a
    // compile-time constant; runtime division of head_dim/BD makes
    // this a runtime sized loop).
    thread U q_reg[8];                  // qk_per_thread <= 8 (head_dim<=256)
    thread U o_reg[8];

    // Threadgroup scratch for per-simdgroup max + sum_exp combine.
    threadgroup U tg_outputs[BN * BD];
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];

    // Q row pointer + scaled load. Each lane owns qk_per_thread
    // contiguous elements at offset simd_lid * qk_per_thread.
    device const half* q_row = q + (seq_idx * num_q + q_head_idx) * head_dim;
    device       half* o_row = output + (seq_idx * num_q + q_head_idx) * head_dim;
    device const uint* row_block_table = block_table + seq_idx * max_blocks;

    // Pre-multiply Q by scale (MLX `sdpa_vector`: `q[i] = scale * queries[i]`).
    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    // Initialize per-thread max with finite minimum (MLX uses
    // `Limits<U>::finite_min`; -FLT_MAX is the f32 equivalent).
    // fast::exp doesn't handle -INFINITY safely so we avoid it.
    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    // For each key, simdgroup `simd_gid` handles tokens at indices
    // simd_gid, simd_gid+BN, simd_gid+2*BN, ... The simdgroup that
    // overshoots `kv_len` skips its iteration and contributes 0.
    for (uint i = simd_gid; i < kv_len; i += uint(BN)) {
        // Resolve paged cache pointer for token i in this simdgroup.
        const uint logical_block = i / block_size;
        const uint physical_block = row_block_table[logical_block];
        const uint token_in_block = i - logical_block * block_size;
        device const half* k_ptr =
            k_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;
        device const half* v_ptr =
            v_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;

        // Dot product of q · k for this lane's slice; simd_sum
        // reduces within the simdgroup.
        U score = 0;
        for (uint j = 0; j < qk_per_thread; ++j) {
            score += q_reg[j] * U(k_ptr[j]);
        }
        score = simd_sum(score);

        // Online softmax update. Match MLX `sdpa_vector`: fast::exp
        // for both factor + exp_score.
        U new_max = max(max_score, score);
        U factor = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        // Accumulate weighted V; rescale prior accumulator with factor.
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_reg[j] = o_reg[j] * factor + exp_score * U(v_ptr[j]);
        }
    }

    // ── Combine per-simdgroup partials ───────────────────────────
    //
    // Each simdgroup's lane 0 publishes its max + sum_exp; all
    // simdgroups then read all values via lane id and reduce.
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane (within simdgroup_id 0..BN-1) reads tg_max[simd_lid]
    // / tg_sum[simd_lid]; simd_max + simd_sum produce the global max
    // and (factor-rescaled) global sum_exp.
    U other_max = tg_max[simd_lid];
    U global_max = simd_max(other_max);
    U factor = metal::fast::exp(other_max - global_max);
    U global_sum = simd_sum(tg_sum[simd_lid] * factor);

    // Combine output partials. Each simdgroup wrote o_reg[j] for
    // its slice; we need to weight each simdgroup's contribution by
    // its `factor` (the rescaling for the global max), then sum
    // across simdgroups, then divide by global_sum.
    for (uint j = 0; j < qk_per_thread; ++j) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Each simdgroup reads its column from tg_outputs and sums
        // across the BD partials, weighted by per-simdgroup factor.
        U val = tg_outputs[simd_gid * BD + simd_lid] * factor;
        U combined = simd_sum(val);
        if (global_sum != 0) {
            combined = combined / global_sum;
        }
        o_reg[j] = combined;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane 0 of each simdgroup writes its qk_per_thread output slice.
    if (simd_lid == 0) {
        device half* o_ptr = o_row + simd_gid * qk_per_thread;
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_ptr[j] = half(o_reg[j]);
        }
    }
}

/// BF16 sibling of `attention_via_cache_v2_f16_specialized`. Same
/// algorithm: paged-cache adaptation of MLX's `sdpa_vector` (online
/// softmax + per-simdgroup K-axis split). Reads pre-rotated K from
/// the cache (rope-on-write — `rope_append_bf16` rotates K and writes
/// rotated K to the cache).
///
/// Constraint: HEAD_DIM must be a multiple of 32. Llama-3.2-1B
/// (HEAD_DIM=64), Llama-3.2-3B (HEAD_DIM=128), and Qwen-class
/// (HEAD_DIM=128) all satisfy.
kernel void attention_via_cache_v2_bf16_specialized(
    device       bfloat* output      [[buffer(0)]],   // [batch, num_q_heads, head_dim]
    device const bfloat* q           [[buffer(1)]],   // [batch, num_q_heads, head_dim]
    device const uint*   seq_used_k  [[buffer(2)]],   // [batch]
    device const uint*   block_table [[buffer(3)]],   // [batch, MAX_BLOCKS_PER_SEQ]
    device const bfloat* k_cache     [[buffer(4)]],
    device const bfloat* v_cache     [[buffer(5)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint3  tid       [[thread_position_in_threadgroup]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr int BN = 32; // simdgroups per threadgroup
    constexpr int BD = 32; // lanes per simdgroup
    typedef float U;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;

    // Each lane handles `qk_per_thread` contiguous elements of head_dim.
    const uint qk_per_thread = head_dim / uint(BD);

    const uint seq_idx     = tg_pos.x;
    const uint q_head_idx  = tg_pos.y;
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;
    const uint kv_len      = seq_used_k[seq_idx];

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    thread U q_reg[8];
    thread U o_reg[8];

    threadgroup U tg_outputs[BN * BD];
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];

    device const bfloat* q_row = q + (seq_idx * num_q + q_head_idx) * head_dim;
    device       bfloat* o_row = output + (seq_idx * num_q + q_head_idx) * head_dim;
    device const uint*   row_block_table = block_table + seq_idx * max_blocks;

    // Pre-multiply Q by scale (MLX `sdpa_vector`: `q[i] = scale * queries[i]`).
    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    // Initialize per-thread max with finite minimum (MLX uses
    // `Limits<U>::finite_min`; -FLT_MAX is the f32 equivalent).
    // fast::exp doesn't handle -INFINITY safely so we avoid it.
    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    // Online softmax over K axis. Each simdgroup `simd_gid` covers
    // tokens at indices simd_gid, simd_gid+BN, simd_gid+2*BN, ...
    for (uint i = simd_gid; i < kv_len; i += uint(BN)) {
        const uint logical_block = i / block_size;
        const uint physical_block = row_block_table[logical_block];
        const uint token_in_block = i - logical_block * block_size;
        device const bfloat* k_ptr =
            k_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;
        device const bfloat* v_ptr =
            v_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;

        U score = 0;
        for (uint j = 0; j < qk_per_thread; ++j) {
            score += q_reg[j] * U(k_ptr[j]);
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        // Match MLX `sdpa_vector`: fast::exp for both factor + exp_score.
        U factor = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < qk_per_thread; ++j) {
            o_reg[j] = o_reg[j] * factor + exp_score * U(v_ptr[j]);
        }
    }

    // Combine per-simdgroup partials (online-softmax merge).
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U other_max = tg_max[simd_lid];
    U global_max = simd_max(other_max);
    U factor = metal::fast::exp(other_max - global_max);
    U global_sum = simd_sum(tg_sum[simd_lid] * factor);

    for (uint j = 0; j < qk_per_thread; ++j) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        U val = tg_outputs[simd_gid * BD + simd_lid] * factor;
        U combined = simd_sum(val);
        if (global_sum != 0) {
            combined = combined / global_sum;
        }
        o_reg[j] = combined;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device bfloat* o_ptr = o_row + simd_gid * qk_per_thread;
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_ptr[j] = bfloat(o_reg[j]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// attention_prefill_sdpa_v2_paged — paged-cache variant of the prefill
// sdpa_vector port. Same outer structure as
// `attention_prefill_sdpa_v2_*` (1 Q per TG, dispatch on
// `(num_q_heads, total_q, 1)`, online softmax + per-simdgroup K-axis
// split) but K/V are read from the paged cache via `block_table`
// (mirroring the decode kernel `attention_via_cache_v2_*`).
//
// Use cases (where contiguous prefill cannot be used):
//   - Chunked prefill: prompts > max_num_batched_tokens get split
//     across steps; chunks 2+ have prior K already in the paged cache.
//   - Prefix caching: requests sharing a prefix start with
//     num_computed_tokens > 0; the new tokens must attend over the
//     prior cached K.
//   - Mixed prefill/decode batches: V1 scheduler interleaves a
//     prefilling sequence (with prior K) alongside decoding sequences.
//   - Multi-turn chat continuation: each new turn extends a sequence
//     whose prior turns are already cached.
//
// Differences vs the contiguous prefill kernel:
//   1. K/V read from `k_cache`/`v_cache` via `block_table[seq_idx]`
//      indirection (paged-cache layout) — same access shape as the
//      decode kernel.
//   2. K-axis loop runs over the FULL cached length `seqused_k[seq]`,
//      not just `seq_end - seq_start` — the prior cached K is in
//      cache slots `[0, seqused_k[seq] - new_q_for_seq)` and the new
//      tokens just appended (by the upstream `RopeAppend`) are at
//      `[seqused_k[seq] - new_q_for_seq, seqused_k[seq])`.
//   3. Causal mask compares K position `i` against the absolute Q
//      position `q_abs_pos = (seqused_k[seq] - new_q_for_seq) +
//      q_pos_in_new`, where `new_q_for_seq = cu_seqlens_q[seq+1] -
//      cu_seqlens_q[seq]` and `q_pos_in_new = global_q -
//      cu_seqlens_q[seq]`. The `(seqused_k - new_q_for_seq)` shift
//      is the prefix-length offset that contiguous prefill doesn't
//      need (it has no prior cached K).
//   4. Function constants extend to BLOCK_SIZE + MAX_BLOCKS_PER_SEQ
//      (paging) — same set as the decode kernel.
//
// Buffer bindings:
//   buffer(0) = output       [total_q, num_q_heads, head_dim]
//   buffer(1) = q            [total_q, num_q_heads, head_dim]
//   buffer(2) = cu_seqlens_q [batch+1]
//   buffer(3) = seqused_k    [batch]   (total cached K including new)
//   buffer(4) = block_table  [batch, MAX_BLOCKS_PER_SEQ]
//   buffer(5) = k_cache      [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
//   buffer(6) = v_cache      [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
//
// Function constants 0..5: HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS,
// ATTN_SCALE_FC, BLOCK_SIZE, MAX_BLOCKS_PER_SEQ. Same indices as
// `attention_via_cache_v2_*`.
//
// Dispatch: threadgroups `(num_q_heads, total_q, 1)`, threads
// `(1024, 1, 1)` = 32 simdgroups × 32 lanes (same as the contiguous
// prefill kernel and decode kernel).
//
// Constraint: HEAD_DIM must be a multiple of 32 (qk_per_thread =
// HEAD_DIM / 32). Llama-3.2 / Qwen / Mistral / Phi all satisfy.

kernel void attention_prefill_sdpa_v2_paged_f16_specialized(
    device       half* output       [[buffer(0)]],   // [total_q, num_q_heads, head_dim]
    device const half* q            [[buffer(1)]],   // [total_q, num_q_heads, head_dim]
    device const uint* cu_seqlens_q [[buffer(2)]],   // [batch+1]
    device const uint* seq_used_k   [[buffer(3)]],   // [batch]
    device const uint* block_table  [[buffer(4)]],   // [batch, MAX_BLOCKS_PER_SEQ]
    device const half* k_cache      [[buffer(5)]],
    device const half* v_cache      [[buffer(6)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint3  tid       [[thread_position_in_threadgroup]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr int BN = 32;
    constexpr int BD = 32;
    typedef float U;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;
    const uint qk_per_thread = head_dim / uint(BD);

    const uint q_head_idx  = tg_pos.x;            // 0..NUM_Q_HEADS
    const uint global_q    = tg_pos.y;            // 0..total_q
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;

    // Locate this Q token's sequence via cu_seqlens_q. Same linear
    // scan as the contiguous prefill kernel; sentinel exit on the
    // first `hi <= lo` (production buffer is `(max_m+1)*4` bytes,
    // OOB reads land in zero-init memory).
    uint seq_idx   = 0;
    uint seq_start = 0;
    uint seq_end   = 0;
    bool in_range  = false;
    for (uint b = 0; b < 1024u; ++b) {
        const uint lo = cu_seqlens_q[b];
        const uint hi = cu_seqlens_q[b + 1];
        if (global_q >= lo && global_q < hi) {
            seq_idx   = b;
            seq_start = lo;
            seq_end   = hi;
            in_range  = true;
            break;
        }
        if (hi <= lo) break;       // sentinel: end of batch
    }
    if (!in_range) {
        // Padding lane (global_q past the last sequence). Match
        // contiguous-prefill: lane 0 of each simdgroup writes zero.
        if (simd_lid == 0) {
            device half* o_ptr =
                output + (global_q * num_q + q_head_idx) * head_dim
                       + simd_gid * qk_per_thread;
            for (uint j = 0; j < qk_per_thread; ++j) o_ptr[j] = half(0);
        }
        return;
    }

    const uint new_q_for_seq = seq_end - seq_start;
    const uint q_pos_in_new  = global_q - seq_start;
    const uint kv_len        = seq_used_k[seq_idx];
    // Absolute position of this Q in the K axis. Prefix length is
    // `kv_len - new_q_for_seq` (caller guarantees seq_used_k already
    // includes the just-appended new tokens; upstream RopeAppend ran
    // before this attention dispatch).
    const uint q_abs_pos     = (kv_len - new_q_for_seq) + q_pos_in_new;

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    thread U q_reg[8];                  // qk_per_thread <= 8 (head_dim<=256)
    thread U o_reg[8];

    threadgroup U tg_outputs[BN * BD];
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];

    device const half* q_row = q + (global_q * num_q + q_head_idx) * head_dim;
    device       half* o_row = output + (global_q * num_q + q_head_idx) * head_dim;
    device const uint* row_block_table = block_table + seq_idx * max_blocks;

    // Pre-multiply Q by scale (MLX `sdpa_vector`: `q[i] = scale * queries[i]`).
    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    // Online softmax over the FULL cached K. Each simdgroup `simd_gid`
    // covers tokens at indices simd_gid, simd_gid + BN, simd_gid +
    // 2*BN, … Causal: skip K positions > q_abs_pos. The branch is
    // simdgroup-uniform (`i` derives from simd_gid; `q_abs_pos` is
    // threadgroup-uniform).
    for (uint i = simd_gid; i < kv_len; i += uint(BN)) {
        if (i > q_abs_pos) continue;

        const uint logical_block = i / block_size;
        const uint physical_block = row_block_table[logical_block];
        const uint token_in_block = i - logical_block * block_size;
        device const half* k_ptr =
            k_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;
        device const half* v_ptr =
            v_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;

        U score = 0;
        for (uint j = 0; j < qk_per_thread; ++j) {
            score += q_reg[j] * U(k_ptr[j]);
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < qk_per_thread; ++j) {
            o_reg[j] = o_reg[j] * factor + exp_score * U(v_ptr[j]);
        }
    }

    // Combine per-simdgroup partials (identical to decode + contiguous
    // prefill kernels).
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U other_max = tg_max[simd_lid];
    U global_max = simd_max(other_max);
    U factor = metal::fast::exp(other_max - global_max);
    U global_sum = simd_sum(tg_sum[simd_lid] * factor);

    for (uint j = 0; j < qk_per_thread; ++j) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        U val = tg_outputs[simd_gid * BD + simd_lid] * factor;
        U combined = simd_sum(val);
        if (global_sum != 0) {
            combined = combined / global_sum;
        }
        o_reg[j] = combined;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device half* o_ptr = o_row + simd_gid * qk_per_thread;
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_ptr[j] = half(o_reg[j]);
        }
    }
}

/// BF16 sibling of `attention_prefill_sdpa_v2_paged_f16_specialized`.
/// Same algorithm; sole difference is the `bfloat`/`half` element
/// type on the device pointers. f32 accumulator preserved.
kernel void attention_prefill_sdpa_v2_paged_bf16_specialized(
    device       bfloat* output       [[buffer(0)]],   // [total_q, num_q_heads, head_dim]
    device const bfloat* q            [[buffer(1)]],   // [total_q, num_q_heads, head_dim]
    device const uint*   cu_seqlens_q [[buffer(2)]],   // [batch+1]
    device const uint*   seq_used_k   [[buffer(3)]],   // [batch]
    device const uint*   block_table  [[buffer(4)]],   // [batch, MAX_BLOCKS_PER_SEQ]
    device const bfloat* k_cache      [[buffer(5)]],
    device const bfloat* v_cache      [[buffer(6)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint3  tid       [[thread_position_in_threadgroup]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr int BN = 32;
    constexpr int BD = 32;
    typedef float U;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;
    const uint qk_per_thread = head_dim / uint(BD);

    const uint q_head_idx  = tg_pos.x;
    const uint global_q    = tg_pos.y;
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;

    uint seq_idx   = 0;
    uint seq_start = 0;
    uint seq_end   = 0;
    bool in_range  = false;
    for (uint b = 0; b < 1024u; ++b) {
        const uint lo = cu_seqlens_q[b];
        const uint hi = cu_seqlens_q[b + 1];
        if (global_q >= lo && global_q < hi) {
            seq_idx   = b;
            seq_start = lo;
            seq_end   = hi;
            in_range  = true;
            break;
        }
        if (hi <= lo) break;
    }
    if (!in_range) {
        if (simd_lid == 0) {
            device bfloat* o_ptr =
                output + (global_q * num_q + q_head_idx) * head_dim
                       + simd_gid * qk_per_thread;
            for (uint j = 0; j < qk_per_thread; ++j) o_ptr[j] = bfloat(0);
        }
        return;
    }

    const uint new_q_for_seq = seq_end - seq_start;
    const uint q_pos_in_new  = global_q - seq_start;
    const uint kv_len        = seq_used_k[seq_idx];
    const uint q_abs_pos     = (kv_len - new_q_for_seq) + q_pos_in_new;

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    thread U q_reg[8];
    thread U o_reg[8];

    threadgroup U tg_outputs[BN * BD];
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];

    device const bfloat* q_row = q + (global_q * num_q + q_head_idx) * head_dim;
    device       bfloat* o_row = output + (global_q * num_q + q_head_idx) * head_dim;
    device const uint*   row_block_table = block_table + seq_idx * max_blocks;

    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    for (uint i = simd_gid; i < kv_len; i += uint(BN)) {
        if (i > q_abs_pos) continue;

        const uint logical_block = i / block_size;
        const uint physical_block = row_block_table[logical_block];
        const uint token_in_block = i - logical_block * block_size;
        device const bfloat* k_ptr =
            k_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;
        device const bfloat* v_ptr =
            v_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;

        U score = 0;
        for (uint j = 0; j < qk_per_thread; ++j) {
            score += q_reg[j] * U(k_ptr[j]);
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < qk_per_thread; ++j) {
            o_reg[j] = o_reg[j] * factor + exp_score * U(v_ptr[j]);
        }
    }

    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U other_max = tg_max[simd_lid];
    U global_max = simd_max(other_max);
    U factor = metal::fast::exp(other_max - global_max);
    U global_sum = simd_sum(tg_sum[simd_lid] * factor);

    for (uint j = 0; j < qk_per_thread; ++j) {
        tg_outputs[simd_lid * BD + simd_gid] = o_reg[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        U val = tg_outputs[simd_gid * BD + simd_lid] * factor;
        U combined = simd_sum(val);
        if (global_sum != 0) {
            combined = combined / global_sum;
        }
        o_reg[j] = combined;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        device bfloat* o_ptr = o_row + simd_gid * qk_per_thread;
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_ptr[j] = bfloat(o_reg[j]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// attention_via_cache_v2_<dtype>_bn8_specialized — BN=8 decode variant
// ─────────────────────────────────────────────────────────────────────
//
// Same algorithm as `attention_via_cache_v2_<dtype>_specialized` above
// (paged-cache port of MLX `sdpa_vector`, online softmax + per-
// simdgroup K-axis split), but dispatched at threads/TG=256 (8
// simdgroups × 32 lanes) instead of 1024 (32×32). Built for use inside
// a persistent decode megakernel where the residency cap forbids
// 1024-thread TGs: see `project_metal_persistent_megakernel_feasibility`
// — on M4-base, the safe concurrent-TG count at 1024 threads/TG is 30,
// below the 32 q-heads Llama-3.2-1B dispatches per layer. At 256
// threads/TG the safe cap rises to 118, which fits the whole-forward
// grid.
//
// Combine step (BN-agnostic; replaces the BN==BD transpose used by the
// 1024-thread kernel):
//   1. After the K loop, each lane writes its `o_reg[0..qk_per_thread)`
//      slice into `tg_partials[simd_gid * MAX_HEAD_DIM + simd_lid *
//      qk_per_thread + j]`. Simdgroup `simd_gid` owns K-axis partition
//      `simd_gid`; lane `simd_lid` owns hidden-dim slice
//      `[simd_lid*qk_per_thread, (simd_lid+1)*qk_per_thread)`.
//   2. Lane 0 of each simdgroup publishes (max_score, sum_exp_score)
//      into `tg_max` / `tg_sum`.
//   3. All threads compute the global max + per-thread weighted sum
//      across the BN partials; BN=8 so the scan is cheap and the
//      result is threadgroup-uniform.
//   4. Simdgroup 0 writes the final output. Every simdgroup computed
//      the same `final_o` from the same tg buffers, so any simdgroup
//      could write; pick 0 for clarity.
//
// Threadgroup memory: `BN * MAX_HEAD_DIM * 4` + BN*4*2 = 8*256*4 + 64
// = ~8.3 KiB; well under the 32 KiB Apple-GPU TG limit.

#define ATTN_BN8_MAX_HEAD_DIM 256u

kernel void attention_via_cache_v2_f16_bn8_specialized(
    device       half* output      [[buffer(0)]],
    device const half* q           [[buffer(1)]],
    device const uint* seq_used_k  [[buffer(2)]],
    device const uint* block_table [[buffer(3)]],
    device const half* k_cache     [[buffer(4)]],
    device const half* v_cache     [[buffer(5)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint3  tid       [[thread_position_in_threadgroup]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr int BN = 8;
    constexpr int BD = 32;
    typedef float U;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;
    const uint qk_per_thread = head_dim / uint(BD);

    const uint seq_idx     = tg_pos.x;
    const uint q_head_idx  = tg_pos.y;
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;
    const uint kv_len      = seq_used_k[seq_idx];

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    thread U q_reg[8];
    thread U o_reg[8];

    threadgroup U tg_partials[BN * ATTN_BN8_MAX_HEAD_DIM];
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];

    device const half* q_row = q + (seq_idx * num_q + q_head_idx) * head_dim;
    device       half* o_row = output + (seq_idx * num_q + q_head_idx) * head_dim;
    device const uint* row_block_table = block_table + seq_idx * max_blocks;

    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    for (uint i = simd_gid; i < kv_len; i += uint(BN)) {
        const uint logical_block = i / block_size;
        const uint physical_block = row_block_table[logical_block];
        const uint token_in_block = i - logical_block * block_size;
        device const half* k_ptr =
            k_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;
        device const half* v_ptr =
            v_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;

        U score = 0;
        for (uint j = 0; j < qk_per_thread; ++j) {
            score += q_reg[j] * U(k_ptr[j]);
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < qk_per_thread; ++j) {
            o_reg[j] = o_reg[j] * factor + exp_score * U(v_ptr[j]);
        }
    }

    for (uint j = 0; j < qk_per_thread; ++j) {
        tg_partials[simd_gid * ATTN_BN8_MAX_HEAD_DIM
                    + simd_lid * qk_per_thread + j] = o_reg[j];
    }
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U global_max = -FLT_MAX;
    for (uint g = 0; g < uint(BN); ++g) {
        global_max = max(global_max, tg_max[g]);
    }
    U global_sum = 0;
    U factors[BN];
    for (uint g = 0; g < uint(BN); ++g) {
        factors[g] = metal::fast::exp(tg_max[g] - global_max);
        global_sum += tg_sum[g] * factors[g];
    }

    thread U final_o[8];
    for (uint j = 0; j < qk_per_thread; ++j) final_o[j] = 0;
    for (uint g = 0; g < uint(BN); ++g) {
        const U f = factors[g];
        for (uint j = 0; j < qk_per_thread; ++j) {
            final_o[j] += tg_partials[g * ATTN_BN8_MAX_HEAD_DIM
                                       + simd_lid * qk_per_thread + j] * f;
        }
    }
    if (global_sum != 0) {
        for (uint j = 0; j < qk_per_thread; ++j) final_o[j] /= global_sum;
    }

    if (simd_gid == 0) {
        device half* o_ptr = o_row + simd_lid * qk_per_thread;
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_ptr[j] = half(final_o[j]);
        }
    }
}

kernel void attention_via_cache_v2_bf16_bn8_specialized(
    device       bfloat* output      [[buffer(0)]],
    device const bfloat* q           [[buffer(1)]],
    device const uint*   seq_used_k  [[buffer(2)]],
    device const uint*   block_table [[buffer(3)]],
    device const bfloat* k_cache     [[buffer(4)]],
    device const bfloat* v_cache     [[buffer(5)]],
    uint3  tg_pos    [[threadgroup_position_in_grid]],
    uint3  tid       [[thread_position_in_threadgroup]],
    uint   simd_gid  [[simdgroup_index_in_threadgroup]],
    uint   simd_lid  [[thread_index_in_simdgroup]])
{
    constexpr int BN = 8;
    constexpr int BD = 32;
    typedef float U;

    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;
    const uint qk_per_thread = head_dim / uint(BD);

    const uint seq_idx     = tg_pos.x;
    const uint q_head_idx  = tg_pos.y;
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;
    const uint kv_len      = seq_used_k[seq_idx];

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    thread U q_reg[8];
    thread U o_reg[8];

    threadgroup U tg_partials[BN * ATTN_BN8_MAX_HEAD_DIM];
    threadgroup U tg_max[BN];
    threadgroup U tg_sum[BN];

    device const bfloat* q_row = q + (seq_idx * num_q + q_head_idx) * head_dim;
    device       bfloat* o_row = output + (seq_idx * num_q + q_head_idx) * head_dim;
    device const uint*   row_block_table = block_table + seq_idx * max_blocks;

    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    U max_score = -FLT_MAX;
    U sum_exp_score = 0;

    for (uint i = simd_gid; i < kv_len; i += uint(BN)) {
        const uint logical_block = i / block_size;
        const uint physical_block = row_block_table[logical_block];
        const uint token_in_block = i - logical_block * block_size;
        device const bfloat* k_ptr =
            k_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;
        device const bfloat* v_ptr =
            v_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride
            + token_in_block * kv_tok_stride
            + simd_lid * qk_per_thread;

        U score = 0;
        for (uint j = 0; j < qk_per_thread; ++j) {
            score += q_reg[j] * U(k_ptr[j]);
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = metal::fast::exp(max_score - new_max);
        U exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (uint j = 0; j < qk_per_thread; ++j) {
            o_reg[j] = o_reg[j] * factor + exp_score * U(v_ptr[j]);
        }
    }

    for (uint j = 0; j < qk_per_thread; ++j) {
        tg_partials[simd_gid * ATTN_BN8_MAX_HEAD_DIM
                    + simd_lid * qk_per_thread + j] = o_reg[j];
    }
    if (simd_lid == 0) {
        tg_max[simd_gid] = max_score;
        tg_sum[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U global_max = -FLT_MAX;
    for (uint g = 0; g < uint(BN); ++g) {
        global_max = max(global_max, tg_max[g]);
    }
    U global_sum = 0;
    U factors[BN];
    for (uint g = 0; g < uint(BN); ++g) {
        factors[g] = metal::fast::exp(tg_max[g] - global_max);
        global_sum += tg_sum[g] * factors[g];
    }

    thread U final_o[8];
    for (uint j = 0; j < qk_per_thread; ++j) final_o[j] = 0;
    for (uint g = 0; g < uint(BN); ++g) {
        const U f = factors[g];
        for (uint j = 0; j < qk_per_thread; ++j) {
            final_o[j] += tg_partials[g * ATTN_BN8_MAX_HEAD_DIM
                                       + simd_lid * qk_per_thread + j] * f;
        }
    }
    if (global_sum != 0) {
        for (uint j = 0; j < qk_per_thread; ++j) final_o[j] /= global_sum;
    }

    if (simd_gid == 0) {
        device bfloat* o_ptr = o_row + simd_lid * qk_per_thread;
        for (uint j = 0; j < qk_per_thread; ++j) {
            o_ptr[j] = bfloat(final_o[j]);
        }
    }
}
