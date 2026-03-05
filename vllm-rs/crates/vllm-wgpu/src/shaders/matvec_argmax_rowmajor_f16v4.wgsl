// Fused matvec + argmax with f16-packed row-major W: [N, K].
// Uses vec4<u32> for 16-byte vectorized loads — 8 f16 values per load instruction.
// Buffer layout: each row of K/2 u32s is read as K/8 vec4<u32>s.
// K must be divisible by 8 (K/2 divisible by 4).

struct Params {
    K: u32,     // full K (must be divisible by 8)
    N: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<vec4<u32>>;  // packed f16×2, vectorized
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
        let k_vec4 = params.K / 8u;  // number of vec4<u32> per row
        let w_base = col * k_vec4;

        // Main loop: 8 f16 values per iteration via one vec4<u32> load
        for (var i: u32 = 0u; i < k_vec4; i = i + 1u) {
            let v = w[w_base + i];
            let p0 = unpack2x16float(v.x);
            let p1 = unpack2x16float(v.y);
            let p2 = unpack2x16float(v.z);
            let p3 = unpack2x16float(v.w);
            let k0 = i * 8u;
            sum = sum + x[k0] * p0.x + x[k0 + 1u] * p0.y
                      + x[k0 + 2u] * p1.x + x[k0 + 3u] * p1.y
                      + x[k0 + 4u] * p2.x + x[k0 + 5u] * p2.y
                      + x[k0 + 6u] * p3.x + x[k0 + 7u] * p3.y;
        }

        // Handle remainder if K not divisible by 8 (K/2 not divisible by 4)
        let k_half = params.K / 2u;
        let kp_done = k_vec4 * 4u;
        for (var kp: u32 = kp_done; kp < k_half; kp = kp + 1u) {
            let v = unpack2x16float(w[col * k_half + kp].x);  // scalar fallback
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
