// Parallelized single-query decode attention with online softmax.
// Each workgroup = one Q head, 64 threads partition seq_len.
// Uses online softmax (single pass over KV) to avoid recomputing scores.
// Each thread maintains running (max, sum_exp, weighted_v[head_dim]).
// Then we do a cooperative reduction to merge partial results.

struct Params {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    scale_bits: u32,
    _pad5: u32,
    _pad6: u32,
    _pad7: u32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 64u;
// Max head_dim we support (128 for most models). We store partial V per thread.
const MAX_HD: u32 = 128u;

var<workgroup> s_max: array<f32, 64>;
var<workgroup> s_sum: array<f32, 64>;
var<workgroup> s_v: array<f32, 64>; // reused per dim during reduction

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
    let q_off = q_head * params.head_dim;
    let kv_off = kv_head * params.head_dim;

    // Online softmax: single pass over assigned timesteps.
    // Each thread tracks (running_max, running_sum_exp, running_v[head_dim]).
    var my_max: f32 = -3.402823e+38;
    var my_sum: f32 = 0.0;
    // Private V accumulator — WGSL doesn't allow variable-length private arrays,
    // so we use the max supported head_dim.
    var my_v: array<f32, 128>;
    for (var i: u32 = 0u; i < MAX_HD; i = i + 1u) {
        my_v[i] = 0.0;
    }

    var t = tid;
    while t < params.seq_len {
        // Compute QK dot product
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
            dot = dot + q[q_off + d] * k_cache[t * kv_stride + kv_off + d];
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
            my_v[d] = my_v[d] + w * v_cache[t * kv_stride + kv_off + d];
        }

        t = t + WG;
    }

    // Store partial max/sum for reduction
    s_max[tid] = my_max;
    s_sum[tid] = my_sum;
    workgroupBarrier();

    // Tree reduction to merge online softmax states across threads.
    // When merging (max_a, sum_a, v_a) with (max_b, sum_b, v_b):
    //   new_max = max(max_a, max_b)
    //   sum_a' = sum_a * exp(max_a - new_max), sum_b' = sum_b * exp(max_b - new_max)
    //   new_sum = sum_a' + sum_b'
    //   new_v = v_a * exp(max_a - new_max) + v_b * exp(max_b - new_max)
    // We do the V merge per-dimension to avoid needing WG×head_dim shared memory.
    // First merge max/sum, tracking correction factors.

    // Compute global max via reduction
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

    // Each thread rescales its sum and V to the global max
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

    // V reduction: one dimension at a time through shared memory
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
            output[q_off + d] = s_v[0] * inv_total;
        }
        workgroupBarrier();
    }
}
