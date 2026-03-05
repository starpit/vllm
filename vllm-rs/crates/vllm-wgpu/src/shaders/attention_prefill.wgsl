// Causal self-attention for prefill (M>1 query tokens).
// Each workgroup handles one Q-head. For each query position q (0..M),
// computes attention over valid K positions (0..cache_len+q+1) with causal mask.
//
// Q: [M, num_q_heads * head_dim]  (query tokens)
// K_cache: [max_seq, num_kv_heads * head_dim]  (already contains cache_len tokens)
// V_cache: [max_seq, num_kv_heads * head_dim]
// K_new: [M, num_kv_heads * head_dim]  (new K values to write to cache)
// V_new: [M, num_kv_heads * head_dim]  (new V values to write to cache)
// Output: [M, num_q_heads * head_dim]
//
// The shader first writes K_new/V_new into the cache at positions [cache_len..cache_len+M),
// then computes causal attention for each of the M query positions.

struct Params {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_m: u32,        // number of new query tokens (M)
    scale_bits: u32,
    cache_len: u32,    // existing tokens in cache before prefill
    _pad6: u32,
    _pad7: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read> k_new: array<f32>;
@group(0) @binding(4) var<storage, read> v_new: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

const WG: u32 = 64u;
const MAX_HD: u32 = 128u;

// Shared memory for cooperative reduction
var<workgroup> s_max: array<f32, 64>;
var<workgroup> s_sum: array<f32, 64>;
var<workgroup> s_v: array<f32, 64>;
// Shared memory for new KV values to avoid re-reading from global memory
var<workgroup> s_kv_written: u32;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let q_head = wid.x;
    let tid = lid.x;

    if q_head >= params.num_q_heads {
        return;
    }

    let scale = bitcast<f32>(params.scale_bits);
    let gqa_ratio = params.num_q_heads / params.num_kv_heads;
    let kv_head = q_head / gqa_ratio;
    let kv_stride = params.num_kv_heads * params.head_dim;
    let q_stride = params.num_q_heads * params.head_dim;
    let q_head_off = q_head * params.head_dim;
    let kv_head_off = kv_head * params.head_dim;

    // Step 1: Write K_new and V_new into the cache.
    // Each thread writes part of the M new KV rows cooperatively.
    let total_kv_elems = params.seq_m * kv_stride;
    var elem = tid;
    while elem < total_kv_elems {
        let m_idx = elem / kv_stride;
        let d_idx = elem % kv_stride;
        let cache_row = params.cache_len + m_idx;
        k_cache[cache_row * kv_stride + d_idx] = k_new[m_idx * kv_stride + d_idx];
        v_cache[cache_row * kv_stride + d_idx] = v_new[m_idx * kv_stride + d_idx];
        elem = elem + WG;
    }
    workgroupBarrier();

    // Step 2: For each query position q_pos, compute causal attention.
    // Total sequence length visible to q_pos: cache_len + q_pos + 1
    for (var q_pos: u32 = 0u; q_pos < params.seq_m; q_pos = q_pos + 1u) {
        let total_seq = params.cache_len + q_pos + 1u;
        let q_base = q_pos * q_stride + q_head_off;

        // Online softmax: single pass over all valid KV positions
        var my_max: f32 = -3.402823e+38;
        var my_sum: f32 = 0.0;
        var my_v: array<f32, 128>;
        for (var i: u32 = 0u; i < MAX_HD; i = i + 1u) {
            my_v[i] = 0.0;
        }

        var t = tid;
        while t < total_seq {
            // QK dot product
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
                dot = dot + q[q_base + d] * k_cache[t * kv_stride + kv_head_off + d];
            }
            let s = dot * scale;

            // Online softmax update
            if s > my_max {
                let correction = exp(my_max - s);
                my_sum = my_sum * correction;
                for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
                    my_v[d] = my_v[d] * correction;
                }
                my_max = s;
            }
            let w = exp(s - my_max);
            my_sum = my_sum + w;
            for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
                my_v[d] = my_v[d] + w * v_cache[t * kv_stride + kv_head_off + d];
            }

            t = t + WG;
        }

        // Cooperative reduction across threads
        s_max[tid] = my_max;
        s_sum[tid] = my_sum;
        workgroupBarrier();

        // Global max reduction
        var stride: u32 = WG / 2u;
        while stride > 0u {
            if tid < stride {
                if s_max[tid + stride] > s_max[tid] {
                    s_max[tid] = s_max[tid + stride];
                }
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let global_max = s_max[0];
        workgroupBarrier();

        // Rescale to global max
        let correction = exp(my_max - global_max);
        my_sum = my_sum * correction;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            my_v[d] = my_v[d] * correction;
        }

        // Sum reduction
        s_sum[tid] = my_sum;
        workgroupBarrier();
        stride = WG / 2u;
        while stride > 0u {
            if tid < stride {
                s_sum[tid] = s_sum[tid] + s_sum[tid + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        let inv_total = 1.0 / s_sum[0];
        workgroupBarrier();

        // V reduction: one dimension at a time
        let out_base = q_pos * q_stride + q_head_off;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            s_v[tid] = my_v[d];
            workgroupBarrier();
            stride = WG / 2u;
            while stride > 0u {
                if tid < stride {
                    s_v[tid] = s_v[tid] + s_v[tid + stride];
                }
                workgroupBarrier();
                stride = stride / 2u;
            }
            if tid == 0u {
                output[out_base + d] = s_v[0] * inv_total;
            }
            workgroupBarrier();
        }
    }
}
