// Fused SiLU(gate) * up activation
// gate: [N], up: [N], output: [N]
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    total: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.total {
        return;
    }
    let x = gate[idx];
    let silu = x / (1.0 + exp(-x));
    output[idx] = silu * up[idx];
}
