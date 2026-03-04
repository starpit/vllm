// RMS Normalization: y[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
// Input: [N, D], Weight: [D], Output: [N, D]

struct Params {
    N: u32,
    D: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.N {
        return;
    }

    let base = row * params.D;

    // Compute mean of squares
    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.D; i = i + 1u) {
        let val = input[base + i];
        sum_sq = sum_sq + val * val;
    }
    let rms = sqrt(sum_sq / f32(params.D) + params.eps);

    // Normalize and scale
    for (var i: u32 = 0u; i < params.D; i = i + 1u) {
        output[base + i] = input[base + i] * weight[i] / rms;
    }
}
