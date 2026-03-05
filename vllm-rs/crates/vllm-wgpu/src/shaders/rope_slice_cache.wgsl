// Fused QKV slice + RoPE + KV cache write.
// Takes fused QKV output [1, q_size + 2*kv_size], applies RoPE to Q and K,
// writes K and V to cache, outputs Q_rope [1, q_size].
//
// Each thread handles one element of the output.

struct Params {
    q_size: u32,        // num_q_heads * head_dim
    kv_size: u32,       // num_kv_heads * head_dim
    head_dim: u32,
    position: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    max_seq_len: u32,
    cache_stride: u32,  // num_kv_heads * head_dim (= kv_size)
}

@group(0) @binding(0) var<storage, read> qkv: array<f32>;       // [q_size + 2*kv_size]
@group(0) @binding(1) var<storage, read> cos_cache: array<f32>;  // [max_seq, head_dim/2]
@group(0) @binding(2) var<storage, read> sin_cache: array<f32>;  // [max_seq, head_dim/2]
@group(0) @binding(3) var<storage, read_write> q_out: array<f32>;   // [q_size]
@group(0) @binding(4) var<storage, read_write> k_cache: array<f32>; // [max_seq, kv_size]
@group(0) @binding(5) var<storage, read_write> v_cache: array<f32>; // [max_seq, kv_size]
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.q_size + params.kv_size; // total elements to process (Q rope + K rope)

    let half_dim = params.head_dim / 2u;
    let pos = params.position;
    let cos_base = pos * half_dim;

    if idx < params.q_size {
        // Q path: apply RoPE and write to q_out
        let head = idx / params.head_dim;
        let d = idx % params.head_dim;
        let src = qkv[idx];

        if d < half_dim {
            let cos_val = cos_cache[cos_base + d];
            let sin_val = sin_cache[cos_base + d];
            let pair_val = qkv[head * params.head_dim + d + half_dim];
            q_out[idx] = src * cos_val - pair_val * sin_val;
        } else {
            let d2 = d - half_dim;
            let cos_val = cos_cache[cos_base + d2];
            let sin_val = sin_cache[cos_base + d2];
            let pair_val = qkv[head * params.head_dim + d2];
            q_out[idx] = pair_val * sin_val + src * cos_val;
        }
    } else if idx < total {
        // K path: apply RoPE and write to k_cache at position
        let k_idx = idx - params.q_size;
        let head = k_idx / params.head_dim;
        let d = k_idx % params.head_dim;
        let src = qkv[params.q_size + k_idx];

        var roped: f32;
        if d < half_dim {
            let cos_val = cos_cache[cos_base + d];
            let sin_val = sin_cache[cos_base + d];
            let pair_val = qkv[params.q_size + head * params.head_dim + d + half_dim];
            roped = src * cos_val - pair_val * sin_val;
        } else {
            let d2 = d - half_dim;
            let cos_val = cos_cache[cos_base + d2];
            let sin_val = sin_cache[cos_base + d2];
            let pair_val = qkv[params.q_size + head * params.head_dim + d2];
            roped = pair_val * sin_val + src * cos_val;
        }
        k_cache[pos * params.cache_stride + k_idx] = roped;
    }

    // V path: copy V to cache (no RoPE). Separate range.
    let v_start = params.q_size + params.kv_size;
    let v_idx = idx; // reuse threads for V copy
    if v_idx < params.kv_size {
        v_cache[pos * params.cache_stride + v_idx] = qkv[v_start + v_idx];
    }
}
