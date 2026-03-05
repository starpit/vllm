// Strided slice along last dimension.
// input: [rows, src_dim], output: [rows, length]
// Each thread copies one element.

struct Params {
    rows: u32,
    src_dim: u32,
    offset: u32,
    length: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
    _pad7: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.rows * params.length;
    if idx >= total {
        return;
    }
    let row = idx / params.length;
    let col = idx % params.length;
    output[idx] = input[row * params.src_dim + params.offset + col];
}
