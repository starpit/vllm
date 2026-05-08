#include <metal_stdlib>
using namespace metal;

// Basic single-head attention kernel (Phase 4.1.1)
// Simplified version without paging, quantization, or advanced features
// Goal: Prove numerical correctness of Metal attention implementation

struct AttentionParams {
    uint seq_len;        // Length of key/value sequence
    uint head_size;      // Dimension of each attention head (e.g., 64, 128)
    float scale;         // Attention scale factor (1/sqrt(head_size))
    uint block_size;     // KV cache block size (e.g., 16)
};

// Phase 4.1.1: Basic attention without paging
// Input:  Q [head_size], K [seq_len, head_size], V [seq_len, head_size]
// Output: O [head_size]
// Algorithm:
//   1. Compute attention scores: scores[i] = Q · K[i] * scale
//   2. Softmax: weights[i] = exp(scores[i] - max) / sum(exp(scores - max))
//   3. Weighted sum: O = sum(weights[i] * V[i])

kernel void attention_single_head(
    device const half* q [[buffer(0)]],           // [head_size]
    device const half* k [[buffer(1)]],           // [seq_len, head_size]
    device const half* v [[buffer(2)]],           // [seq_len, head_size]
    device half* output [[buffer(3)]],            // [head_size]
    constant AttentionParams& params [[buffer(4)]],
    threadgroup float* shared_logits [[threadgroup(0)]],  // [seq_len]
    uint tid [[thread_position_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    const uint seq_len = params.seq_len;
    const uint head_size = params.head_size;
    const float scale = params.scale;
    
    // Step 1: Compute Q·K attention scores
    // Each thread computes scores for multiple tokens
    float max_logit = -INFINITY;
    
    for (uint token_idx = tid; token_idx < seq_len; token_idx += threadgroup_size) {
        // Compute dot product: Q · K[token_idx]
        float qk_dot = 0.0f;
        for (uint i = 0; i < head_size; i++) {
            qk_dot += float(q[i]) * float(k[token_idx * head_size + i]);
        }
        
        float logit = qk_dot * scale;
        shared_logits[token_idx] = logit;
        max_logit = max(max_logit, logit);
    }
    
    // Synchronize to ensure all logits are computed
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 2: Find global max logit (for numerical stability)
    // Reduce within simdgroup (32 threads)
    max_logit = simd_max(max_logit);
    
    // Reduce across simdgroups using threadgroup memory
    threadgroup float simdgroup_maxes[32];  // Max 32 simdgroups per threadgroup
    if (lane_id == 0) {
        simdgroup_maxes[simdgroup_id] = max_logit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction (only first simdgroup participates)
    if (simdgroup_id == 0) {
        float global_max = lane_id < (threadgroup_size / 32) ? simdgroup_maxes[lane_id] : -INFINITY;
        global_max = simd_max(global_max);
        if (lane_id == 0) {
            simdgroup_maxes[0] = global_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_max_logit = simdgroup_maxes[0];
    
    // Step 3: Compute exp(logit - max) and sum
    float exp_sum = 0.0f;
    for (uint token_idx = tid; token_idx < seq_len; token_idx += threadgroup_size) {
        float logit = shared_logits[token_idx];
        float exp_val = exp(logit - global_max_logit);
        shared_logits[token_idx] = exp_val;
        exp_sum += exp_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 4: Reduce exp_sum across threads
    exp_sum = simd_sum(exp_sum);
    
    threadgroup float simdgroup_sums[32];
    if (lane_id == 0) {
        simdgroup_sums[simdgroup_id] = exp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simdgroup_id == 0) {
        float global_sum = lane_id < (threadgroup_size / 32) ? simdgroup_sums[lane_id] : 0.0f;
        global_sum = simd_sum(global_sum);
        if (lane_id == 0) {
            simdgroup_sums[0] = global_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float global_exp_sum = simdgroup_sums[0];
    
    // Step 5: Normalize to get softmax weights
    float inv_sum = 1.0f / (global_exp_sum + 1e-6f);
    for (uint token_idx = tid; token_idx < seq_len; token_idx += threadgroup_size) {
        shared_logits[token_idx] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Step 6: Compute weighted sum: output = sum(softmax[i] * V[i])
    // Each thread computes partial sums for different output dimensions
    for (uint dim_idx = tid; dim_idx < head_size; dim_idx += threadgroup_size) {
        float acc = 0.0f;
        for (uint token_idx = 0; token_idx < seq_len; token_idx++) {
            float weight = shared_logits[token_idx];
            float value = float(v[token_idx * head_size + dim_idx]);
            acc += weight * value;
        }
        output[dim_idx] = half(acc);
    }
}

// ---------------------------------------------------------------------------
// Phase 5.C.4: Specialized multi-Q-token attention kernels
//
// Function-constant bindings (must match `pipelines.rs`):
//   0  HEAD_DIM            uint
//   1  NUM_Q_HEADS         uint
//   2  NUM_KV_HEADS        uint
//   3  ATTN_SCALE          float
//   4  BLOCK_SIZE          uint   (AttentionViaCache only)
//   5  MAX_BLOCKS_PER_SEQ  uint   (AttentionViaCache only)
//   6  PREFILL_TILE_Q      uint   (AttentionPrefillContiguous only)
//
// Per the MSL spec, a function-constant index must have consistent
// type+name across a single compilation unit. The two kernels share
// indices 0..3 (HEAD_DIM/NUM_Q_HEADS/NUM_KV_HEADS/ATTN_SCALE) and own
// disjoint slots above that.
//
// Numerically-correct reference implementations: one threadgroup per
// (token, head) for decode and per (Q-tile, head) for prefill, with
// `HEAD_DIM` threads per group cooperating on dot products and the
// output spread. Performance-tuning (FlashAttention-style blocking
// over K, vectorized loads) is deferred to Phase 5.6 — these are
// here to unblock end-to-end forward bring-up.
//
// Shared-logits buffer is fixed at 4096 floats (16KB) — the largest
// `seq_used_k[seq]` we can handle without spilling. Production paths
// will switch to incremental softmax; tracked in PHASE5_PLAN.md.
// ---------------------------------------------------------------------------

constant uint  ATTN_HEAD_DIM           [[function_constant(0)]];
constant uint  ATTN_NUM_Q_HEADS        [[function_constant(1)]];
constant uint  ATTN_NUM_KV_HEADS       [[function_constant(2)]];
constant float ATTN_SCALE_FC           [[function_constant(3)]];
constant uint  ATTN_BLOCK_SIZE         [[function_constant(4)]];
constant uint  ATTN_MAX_BLOCKS_PER_SEQ [[function_constant(5)]];

constant uint  ATTN_PREFILL_TILE_Q     [[function_constant(6)]];

// Cap on `seq_used_k[seq]` the shared-logits buffer can hold.
// Each token uses 4 bytes; this cap × 4 == threadgroup memory bytes
// dedicated to the partial-logits scratch. Smaller is better for
// concurrent-threadgroup occupancy on Apple GPU clusters (each
// cluster reserves the per-threadgroup TGSM budget per resident
// group). 256 * 4 = 1KB still covers TinyLlama-class decode lengths
// without spilling logits to device memory; sequences longer than
// this cap fall back to a larger-cap pipeline (TODO).
#define ATTN_MAX_SHARED_LOGITS 256u

/// Decode-bucket attention reading from a paged KV cache.
///
/// Layout assumed for K/V cache: `[num_blocks, num_kv_heads,
/// BLOCK_SIZE, HEAD_DIM]` for both K and V. Produced by the
/// `RopeAppend` writer; the worker binds one buffer per (layer, K|V).
///
/// Q is `[batch, num_q_heads, head_dim]`, output mirrors. `bucket_m
/// == batch` for decode.
///
/// Dispatch: threadgroups (batch, num_q_heads, 1), threads
/// (HEAD_DIM, 1, 1). One threadgroup per (seq, q_head). Threads
/// cooperate over `head_dim` loads/stores.
kernel void attention_via_cache_f16_specialized(
    device       half* output      [[buffer(0)]],   // [batch, num_q_heads, head_dim]
    device const half* q           [[buffer(1)]],   // [batch, num_q_heads, head_dim]
    device const uint* seq_used_k  [[buffer(2)]],   // [batch]
    device const uint* block_table [[buffer(3)]],   // [batch, MAX_BLOCKS_PER_SEQ]
    device const half* k_cache     [[buffer(4)]],   // [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
    device const half* v_cache     [[buffer(5)]],   // same layout as k_cache
    uint3  tg_pos  [[threadgroup_position_in_grid]],
    uint3  tid     [[thread_position_in_threadgroup]],
    uint   simd_id [[simdgroup_index_in_threadgroup]],
    uint   lane_id [[thread_index_in_simdgroup]])
{
    const uint seq_idx     = tg_pos.x;            // batch index
    const uint q_head_idx  = tg_pos.y;            // 0..NUM_Q_HEADS
    const uint d           = tid.x;               // 0..HEAD_DIM
    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const uint block_size  = ATTN_BLOCK_SIZE;
    const uint max_blocks  = ATTN_MAX_BLOCKS_PER_SEQ;
    const float scale      = ATTN_SCALE_FC;
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;

    const uint kv_blk_stride  = num_kv * block_size * head_dim;
    const uint kv_head_stride = block_size * head_dim;
    const uint kv_tok_stride  = head_dim;

    // Q row for this threadgroup (stride: num_q_heads * head_dim per token).
    device const half* q_row =
        q + (seq_idx * num_q + q_head_idx) * head_dim;
    device       half* o_row =
        output + (seq_idx * num_q + q_head_idx) * head_dim;
    device const uint* row_block_table =
        block_table + seq_idx * max_blocks;

    threadgroup float shared_logits[ATTN_MAX_SHARED_LOGITS];
    threadgroup float simd_scratch[32];

    // Load Q vector into a register array via threadgroup memory so
    // every thread can read every element in the dot-product loop.
    threadgroup half q_local[1024];     // covers head_dim ≤ 1024
    if (d < head_dim) {
        q_local[d] = q_row[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kv_len = seq_used_k[seq_idx];
    const uint num_logical_blocks = (kv_len + block_size - 1) / block_size;

    // Step 1: scores[token] = (Q · K[token]) * scale.
    float max_logit = -INFINITY;
    for (uint logical_block = 0; logical_block < num_logical_blocks; ++logical_block) {
        const uint physical_block = row_block_table[logical_block];
        device const half* k_block =
            k_cache
            + physical_block * kv_blk_stride
            + kv_head_idx    * kv_head_stride;

        // Each thread takes a stride of HEAD_DIM tokens through this
        // block (one per `d`), but for simplicity we serialize the
        // outer block_offset and parallelize the dot product across
        // threads via simd_sum below.
        for (uint block_offset = 0; block_offset < block_size; ++block_offset) {
            const uint token_idx = logical_block * block_size + block_offset;
            if (token_idx >= kv_len) break;
            if (token_idx >= ATTN_MAX_SHARED_LOGITS) break;

            device const half* k_vec = k_block + block_offset * kv_tok_stride;
            // Each thread contributes its `d`-th product, then we
            // simd-reduce across the head_dim threads.
            float partial = 0.0f;
            if (d < head_dim) {
                partial = float(q_local[d]) * float(k_vec[d]);
            }
            float dot = simd_sum(partial);
            // Lane 0 of each simdgroup writes its partial; cross-simd
            // reduce afterwards. For HEAD_DIM ≤ 32 (one simdgroup)
            // the simd_sum already produced the full dot.
            if (head_dim > 32) {
                if (lane_id == 0) simd_scratch[simd_id] = dot;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_id == 0) {
                    const uint num_simds = (head_dim + 31) / 32;
                    float v = (lane_id < num_simds) ? simd_scratch[lane_id] : 0.0f;
                    v = simd_sum(v);
                    if (lane_id == 0) simd_scratch[0] = v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                dot = simd_scratch[0];
            }
            const float logit = dot * scale;
            if (d == 0) shared_logits[token_idx] = logit;
            max_logit = max(max_logit, logit);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: max-reduce across all threads in the threadgroup.
    max_logit = simd_max(max_logit);
    if (lane_id == 0) simd_scratch[simd_id] = max_logit;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        const uint num_simds = (head_dim + 31) / 32;
        float v = (lane_id < num_simds) ? simd_scratch[lane_id] : -INFINITY;
        v = simd_max(v);
        if (lane_id == 0) simd_scratch[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float global_max = simd_scratch[0];

    // Step 3: exp(logit - max) + sum.
    float exp_sum = 0.0f;
    const uint clamped_kv = min(kv_len, ATTN_MAX_SHARED_LOGITS);
    for (uint t = d; t < clamped_kv; t += head_dim) {
        const float val = exp(shared_logits[t] - global_max);
        shared_logits[t] = val;
        exp_sum += val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    exp_sum = simd_sum(exp_sum);
    if (lane_id == 0) simd_scratch[simd_id] = exp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        const uint num_simds = (head_dim + 31) / 32;
        float v = (lane_id < num_simds) ? simd_scratch[lane_id] : 0.0f;
        v = simd_sum(v);
        if (lane_id == 0) simd_scratch[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_sum = 1.0f / (simd_scratch[0] + 1e-6f);

    // Step 4: weighted sum over V — each thread owns one `d` of the
    // output, accumulating across all KV tokens.
    if (d < head_dim) {
        float acc = 0.0f;
        for (uint logical_block = 0; logical_block < num_logical_blocks; ++logical_block) {
            const uint physical_block = row_block_table[logical_block];
            device const half* v_block =
                v_cache
                + physical_block * kv_blk_stride
                + kv_head_idx    * kv_head_stride;
            for (uint block_offset = 0; block_offset < block_size; ++block_offset) {
                const uint token_idx = logical_block * block_size + block_offset;
                if (token_idx >= kv_len) break;
                if (token_idx >= ATTN_MAX_SHARED_LOGITS) break;
                const float w = shared_logits[token_idx] * inv_sum;
                const float v = float(v_block[block_offset * kv_tok_stride + d]);
                acc += w * v;
            }
        }
        o_row[d] = half(acc);
    }
}

/// Prefill-bucket attention over contiguous Q/K/V tiles.
///
/// Q: `[total_tokens, num_q_heads, head_dim]`,
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

    for (uint i = 0; i < qk_per_thread; ++i) {
        q_reg[i] = U(scale) * U(q_row[simd_lid * qk_per_thread + i]);
        o_reg[i] = 0;
    }

    U max_score = -INFINITY;
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

        // Online softmax update.
        U new_max = max(max_score, score);
        U factor = exp(max_score - new_max);
        U exp_score = exp(score - new_max);

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
    U factor = exp(other_max - global_max);
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

/// K/V: `[total_tokens, num_kv_heads, head_dim]`,
/// O: `[total_tokens, num_q_heads, head_dim]`.
/// `cu_seqlens_q[batch+1]` gives per-sequence boundaries; causal
/// masking reads `q_pos < k_pos` as -inf.
///
/// Dispatch: threadgroups (ceil(total_tokens / PREFILL_TILE_Q),
/// num_q_heads, 1), threads (HEAD_DIM, 1, 1). Each threadgroup
/// processes `PREFILL_TILE_Q` Q tokens for one head; tile sequence
/// boundaries are looked up via `cu_seqlens_q`.
kernel void attention_prefill_contiguous_f16_specialized(
    device       half* output       [[buffer(0)]],   // [total, num_q_heads, head_dim]
    device const half* q            [[buffer(1)]],   // [total, num_q_heads, head_dim]
    device const half* k            [[buffer(2)]],   // [total, num_kv_heads, head_dim]
    device const half* v            [[buffer(3)]],   // [total, num_kv_heads, head_dim]
    device const uint* cu_seqlens_q [[buffer(4)]],   // [batch+1]
    uint3  tg_pos  [[threadgroup_position_in_grid]],
    uint3  tid     [[thread_position_in_threadgroup]],
    uint   simd_id [[simdgroup_index_in_threadgroup]],
    uint   lane_id [[thread_index_in_simdgroup]])
{
    const uint q_tile_idx  = tg_pos.x;            // 0..ceil(total/PREFILL_TILE_Q)
    const uint q_head_idx  = tg_pos.y;            // 0..NUM_Q_HEADS
    const uint d           = tid.x;               // 0..HEAD_DIM
    const uint head_dim    = ATTN_HEAD_DIM;
    const uint num_q       = ATTN_NUM_Q_HEADS;
    const uint num_kv      = ATTN_NUM_KV_HEADS;
    const float scale      = ATTN_SCALE_FC;
    const uint tile_q      = ATTN_PREFILL_TILE_Q;
    const uint group_ratio = num_q / num_kv;
    const uint kv_head_idx = q_head_idx / group_ratio;

    threadgroup float simd_scratch[32];
    threadgroup half  q_local[1024];                            // hoisted; HEAD_DIM ≤ 1024
    threadgroup float shared_logits[ATTN_MAX_SHARED_LOGITS];    // hoisted

    // For each Q token in this tile, find its sequence and the
    // sequence's contiguous K/V range, then run a per-Q-row
    // softmax-attention against those K/V tokens. This is the
    // O(tile_q × seqlen) reference path; the FlashAttention-style
    // tiling lives in 5.6.
    for (uint local_q = 0; local_q < tile_q; ++local_q) {
        const uint global_q = q_tile_idx * tile_q + local_q;

        // Linear scan over cu_seqlens_q to find this q's sequence.
        // Batches are typically small (<= 32) so linear is fine; a
        // binary search is a 5.6 optimization.
        uint seq_start = 0;
        uint seq_end   = 0;
        bool in_range  = false;
        for (uint b = 0;; ++b) {
            const uint lo = cu_seqlens_q[b];
            const uint hi = cu_seqlens_q[b + 1];
            if (global_q >= lo && global_q < hi) {
                seq_start = lo;
                seq_end   = hi;
                in_range  = true;
                break;
            }
            if (hi <= lo) break;        // sentinel for end of batch
            if (b > 1024u) break;       // safety cap
        }
        if (!in_range) {
            if (d < head_dim) {
                output[(global_q * num_q + q_head_idx) * head_dim + d] = half(0.0);
            }
            continue;
        }

        const uint q_pos_in_seq = global_q - seq_start;

        // Load this Q row into threadgroup memory.
        device const half* q_row =
            q + (global_q * num_q + q_head_idx) * head_dim;
        if (d < head_dim) q_local[d] = q_row[d];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pass 1: max-logit + write logits to threadgroup memory.
        const uint kv_len = seq_end - seq_start;
        const uint clamped_kv = min(kv_len, ATTN_MAX_SHARED_LOGITS);

        float local_max = -INFINITY;
        for (uint k_idx = 0; k_idx < clamped_kv; ++k_idx) {
            const uint k_global = seq_start + k_idx;
            float partial = 0.0f;
            if (d < head_dim) {
                device const half* k_row =
                    k + (k_global * num_kv + kv_head_idx) * head_dim;
                partial = float(q_local[d]) * float(k_row[d]);
            }
            float dot = simd_sum(partial);
            if (head_dim > 32) {
                if (lane_id == 0) simd_scratch[simd_id] = dot;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_id == 0) {
                    const uint num_simds = (head_dim + 31) / 32;
                    float vred = (lane_id < num_simds) ? simd_scratch[lane_id] : 0.0f;
                    vred = simd_sum(vred);
                    if (lane_id == 0) simd_scratch[0] = vred;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                dot = simd_scratch[0];
            }
            // Causal mask: k_pos > q_pos → -inf.
            float logit = dot * scale;
            if (k_idx > q_pos_in_seq) logit = -INFINITY;
            if (d == 0) shared_logits[k_idx] = logit;
            local_max = max(local_max, logit);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Reduce max across threadgroup.
        local_max = simd_max(local_max);
        if (lane_id == 0) simd_scratch[simd_id] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_id == 0) {
            const uint num_simds = (head_dim + 31) / 32;
            float vmax = (lane_id < num_simds) ? simd_scratch[lane_id] : -INFINITY;
            vmax = simd_max(vmax);
            if (lane_id == 0) simd_scratch[0] = vmax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float row_max = simd_scratch[0];

        // Pass 2: exp + sum.
        float local_sum = 0.0f;
        for (uint k_idx = d; k_idx < clamped_kv; k_idx += head_dim) {
            const float e = exp(shared_logits[k_idx] - row_max);
            shared_logits[k_idx] = e;
            local_sum += e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        local_sum = simd_sum(local_sum);
        if (lane_id == 0) simd_scratch[simd_id] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_id == 0) {
            const uint num_simds = (head_dim + 31) / 32;
            float vs = (lane_id < num_simds) ? simd_scratch[lane_id] : 0.0f;
            vs = simd_sum(vs);
            if (lane_id == 0) simd_scratch[0] = vs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_sum = 1.0f / (simd_scratch[0] + 1e-6f);

        // Pass 3: weighted sum over V.
        if (d < head_dim) {
            float acc = 0.0f;
            for (uint k_idx = 0; k_idx < clamped_kv; ++k_idx) {
                const uint k_global = seq_start + k_idx;
                device const half* v_row =
                    v + (k_global * num_kv + kv_head_idx) * head_dim;
                acc += shared_logits[k_idx] * inv_sum * float(v_row[d]);
            }
            output[(global_q * num_q + q_head_idx) * head_dim + d] = half(acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
