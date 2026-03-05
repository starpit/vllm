// Argmax: find the index of the maximum value in an array.
// input: [N], output: [1] (u32 index)
// Uses parallel reduction with (value, index) pairs.

struct Params {
    N: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;

var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;

    // Each thread finds max in its stride
    var best_val: f32 = -3.402823e+38;
    var best_idx: u32 = 0u;
    var i = tid;
    while i < params.N {
        let v = input[i];
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
        i = i + WG;
    }

    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    workgroupBarrier();

    // Tree reduction
    var stride: u32 = WG / 2u;
    while stride > 0u {
        if tid < stride {
            if s_val[tid + stride] > s_val[tid] {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        output[0] = s_idx[0];
    }
}
