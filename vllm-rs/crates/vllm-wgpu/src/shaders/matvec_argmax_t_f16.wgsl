// Fused matvec + argmax with TRANSPOSED f16-packed weights.
// W_t: [K, N] with f16 packed along K dimension.
// packed[kp * N + col] = pack(w[2*kp, col], w[2*kp+1, col])
// Adjacent threads read adjacent cols → coalesced memory access.

struct Params {
    K: u32,     // full K (must be even)
    N: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_t: array<u32>;  // packed f16×2
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
        let k_pairs = params.K / 2u;

        let kp_end4 = (k_pairs / 4u) * 4u;
        for (var kp: u32 = 0u; kp < kp_end4; kp = kp + 4u) {
            let v0 = unpack2x16float(w_t[(kp) * params.N + col]);
            let v1 = unpack2x16float(w_t[(kp + 1u) * params.N + col]);
            let v2 = unpack2x16float(w_t[(kp + 2u) * params.N + col]);
            let v3 = unpack2x16float(w_t[(kp + 3u) * params.N + col]);
            let k0 = kp * 2u;
            sum = sum + x[k0] * v0.x + x[k0 + 1u] * v0.y
                      + x[k0 + 2u] * v1.x + x[k0 + 3u] * v1.y
                      + x[k0 + 4u] * v2.x + x[k0 + 5u] * v2.y
                      + x[k0 + 6u] * v3.x + x[k0 + 7u] * v3.y;
        }
        for (var kp: u32 = kp_end4; kp < k_pairs; kp = kp + 1u) {
            let v = unpack2x16float(w_t[kp * params.N + col]);
            let k0 = kp * 2u;
            sum = sum + x[k0] * v.x + x[k0 + 1u] * v.y;
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
