// Fused matvec + argmax with row-major W: [N, K].
// Each thread computes dot(x, W[col, :]) and participates in workgroup argmax.

struct Params {
    K: u32,
    N: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> partial_vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial_idxs: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;

var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let col = gid.x;
    let tid = lid.x;

    var dot: f32 = -3.402823e+38;
    var my_col: u32 = 0u;

    if col < params.N {
        var sum: f32 = 0.0;
        let w_base = col * params.K;
        let k_end4 = (params.K / 4u) * 4u;
        for (var k: u32 = 0u; k < k_end4; k = k + 4u) {
            sum = sum + x[k] * w[w_base + k]
                      + x[k + 1u] * w[w_base + k + 1u]
                      + x[k + 2u] * w[w_base + k + 2u]
                      + x[k + 3u] * w[w_base + k + 3u];
        }
        for (var k: u32 = k_end4; k < params.K; k = k + 1u) {
            sum = sum + x[k] * w[w_base + k];
        }
        dot = sum;
        my_col = col;
    }

    s_val[tid] = dot;
    s_idx[tid] = my_col;
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
        partial_vals[wid.x] = s_val[0];
        partial_idxs[wid.x] = s_idx[0];
    }
}
