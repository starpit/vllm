// Final argmax reduction over partial results from workgroups.
// vals: [N], idxs: [N] → output: [1] (u32 index from idxs)

struct Params {
    N: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> vals: array<f32>;
@group(0) @binding(1) var<storage, read> idxs: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;

var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    var best_val: f32 = -3.402823e+38;
    var best_idx: u32 = 0u;
    var i = tid;
    while i < params.N {
        if vals[i] > best_val {
            best_val = vals[i];
            best_idx = idxs[i];
        }
        i = i + WG;
    }

    s_val[tid] = best_val;
    s_idx[tid] = best_idx;
    workgroupBarrier();

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
