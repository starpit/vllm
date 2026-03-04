// Rotary Position Embedding (RoPE)
// Applies rotation to pairs of elements in Q and K tensors.
// Input Q/K: [N, num_heads, head_dim], positions: [N]
// cos_cache/sin_cache: [max_seq_len, head_dim/2]

struct Params {
    N: u32,
    num_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
}

@group(0) @binding(0) var<storage, read_write> qk: array<f32>;
@group(0) @binding(1) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(2) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(3) var<storage, read> positions: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let half_dim = params.head_dim / 2u;
    let total_pairs = params.N * params.num_heads * half_dim;

    if idx >= total_pairs {
        return;
    }

    // Decompose flat index
    let pair_in_head = idx % half_dim;
    let remaining = idx / half_dim;
    let head = remaining % params.num_heads;
    let token = remaining / params.num_heads;

    let pos = positions[token];
    let cos_val = cos_cache[pos * half_dim + pair_in_head];
    let sin_val = sin_cache[pos * half_dim + pair_in_head];

    let base = token * params.num_heads * params.head_dim + head * params.head_dim;
    let i0 = base + pair_in_head;
    let i1 = base + pair_in_head + half_dim;

    let x0 = qk[i0];
    let x1 = qk[i1];

    qk[i0] = x0 * cos_val - x1 * sin_val;
    qk[i1] = x0 * sin_val + x1 * cos_val;
}
