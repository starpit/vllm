// Embedding lookup: output[i] = table[indices[i]]
// table: [vocab_size, dim], indices: [N], output: [N, dim]

struct Params {
    N: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.N * params.dim;

    if idx >= total {
        return;
    }

    let token = idx / params.dim;
    let d = idx % params.dim;
    let vocab_idx = indices[token];

    output[idx] = table[vocab_idx * params.dim + d];
}
