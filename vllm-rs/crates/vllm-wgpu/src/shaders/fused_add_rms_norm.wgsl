// Fused residual add + RMS normalization.
// residual: [N, D], input: [N, D], weight: [D] → output: [N, D], residual_out: [N, D]
// residual_out = residual + input
// output = rms_norm(residual_out) * weight

struct Params {
    N: u32,
    D: u32,
    eps_bits: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> residual_out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.N {
        return;
    }

    let eps = bitcast<f32>(params.eps_bits);
    let base = row * params.D;

    // Compute residual + input, and sum of squares
    var ss: f32 = 0.0;
    for (var d: u32 = 0u; d < params.D; d = d + 1u) {
        let val = residual[base + d] + input[base + d];
        residual_out[base + d] = val;
        ss = ss + val * val;
    }

    let rms = 1.0 / sqrt(ss / f32(params.D) + eps);
    for (var d: u32 = 0u; d < params.D; d = d + 1u) {
        output[base + d] = residual_out[base + d] * rms * weight[d];
    }
}
