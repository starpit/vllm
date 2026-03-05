// Fused SiLU(gate) * up from a concatenated [gate | up] buffer.
// input: [N, 2*half], output: [N, half]
// gate = input[..half], up = input[half..2*half] per row.

struct Params {
    total: u32,   // N * half
    half: u32,    // intermediate_size
    stride: u32,  // 2 * half (input last dim)
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.total {
        return;
    }
    let row = idx / params.half;
    let col = idx % params.half;
    let base = row * params.stride;
    let gate_val = input[base + col];
    let up_val = input[base + params.half + col];
    let silu = gate_val / (1.0 + exp(-gate_val));
    output[idx] = silu * up_val;
}
