// Row-wise softmax: y[i] = exp(x[i] - max) / sum(exp(x - max))
// Input: [N, D], Output: [N, D]

struct Params {
    N: u32,
    D: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.N {
        return;
    }

    let base = row * params.D;

    // Find max for numerical stability
    var max_val: f32 = -1e30;
    for (var i: u32 = 0u; i < params.D; i = i + 1u) {
        max_val = max(max_val, input[base + i]);
    }

    // Compute exp and sum
    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < params.D; i = i + 1u) {
        let e = exp(input[base + i] - max_val);
        output[base + i] = e;
        sum_exp = sum_exp + e;
    }

    // Normalize
    for (var i: u32 = 0u; i < params.D; i = i + 1u) {
        output[base + i] = output[base + i] / sum_exp;
    }
}
