// Fused matvec + argmax with TRANSPOSED Q4_0-packed weights.
// W_t stored as Q4_0 blocks in transposed [K/32, N] order, 5 u32s per block.
// Adjacent threads read adjacent cols → coalesced memory access.

struct Params {
    K: u32,     // full K (must be multiple of 32)
    N: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_q4: array<u32>;
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
        let k_groups = params.K / 32u;

        for (var kg: u32 = 0u; kg < k_groups; kg = kg + 1u) {
            let base = (kg * params.N + col) * 5u;
            let scale = unpack2x16float(w_q4[base]).x;
            let k_off = kg * 32u;

            for (var w: u32 = 0u; w < 4u; w = w + 1u) {
                let packed = w_q4[base + 1u + w];
                let elem_off = k_off + w * 8u;

                let b0 = packed & 0xFFu;
                let b1 = (packed >> 8u) & 0xFFu;
                let b2 = (packed >> 16u) & 0xFFu;
                let b3 = (packed >> 24u) & 0xFFu;

                let e0 = f32(i32(b0 & 0xFu) - 8) * scale;
                let e1 = f32(i32((b0 >> 4u) & 0xFu) - 8) * scale;
                let e2 = f32(i32(b1 & 0xFu) - 8) * scale;
                let e3 = f32(i32((b1 >> 4u) & 0xFu) - 8) * scale;
                let e4 = f32(i32(b2 & 0xFu) - 8) * scale;
                let e5 = f32(i32((b2 >> 4u) & 0xFu) - 8) * scale;
                let e6 = f32(i32(b3 & 0xFu) - 8) * scale;
                let e7 = f32(i32((b3 >> 4u) & 0xFu) - 8) * scale;

                sum = sum + x[elem_off] * e0
                          + x[elem_off + 1u] * e1
                          + x[elem_off + 2u] * e2
                          + x[elem_off + 3u] * e3
                          + x[elem_off + 4u] * e4
                          + x[elem_off + 5u] * e5
                          + x[elem_off + 6u] * e6
                          + x[elem_off + 7u] * e7;
            }
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
