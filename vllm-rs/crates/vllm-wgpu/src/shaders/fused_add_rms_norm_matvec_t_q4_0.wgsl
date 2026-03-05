// Fused: residual_add + RMS norm + matvec with transposed Q4_0 weights.
// Replaces 2 dispatches (fused_add_rms_norm + matvec_t_q4_0) with 1.
//
// Each workgroup:
//   1. Cooperatively computes hidden_new = residual + addition (in shared mem)
//   2. Cooperatively computes sum-of-squares reduction for RMS norm
//   3. Workgroup 0 writes hidden_new to output buffer
//   4. In-place normalize shared mem: s_data[k] *= norm_weight[k] * rms_inv
//   5. Each thread computes one Q4_0 matvec output column from normalized input

struct Params {
    K: u32,        // hidden size (must be multiple of 32)
    N: u32,        // output dimension (number of output columns)
    eps_bits: u32, // f32 epsilon as bits (bitcast in shader)
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> addition: array<f32>;
@group(0) @binding(2) var<storage, read> norm_weight: array<f32>;
@group(0) @binding(3) var<storage, read> w_q4: array<u32>;  // Q4_0 packed, [K/32, N, 5]
@group(0) @binding(4) var<storage, read_write> matvec_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> hidden_out: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

const WG: u32 = 256u;

var<workgroup> s_data: array<f32, __MAX_K__>;
var<workgroup> s_reduce: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let col = gid.x;
    let K = params.K;
    let eps = bitcast<f32>(params.eps_bits);

    // Step 1: Cooperatively compute hidden_new = residual + addition
    var local_sq_sum: f32 = 0.0;
    var k: u32 = tid;
    while k < K {
        let h = residual[k] + addition[k];
        s_data[k] = h;
        local_sq_sum = local_sq_sum + h * h;
        k = k + WG;
    }
    s_reduce[tid] = local_sq_sum;
    workgroupBarrier();

    // Step 2: Tree reduction for sum of squares
    var stride: u32 = WG / 2u;
    while stride > 0u {
        if tid < stride {
            s_reduce[tid] = s_reduce[tid] + s_reduce[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let rms_inv = inverseSqrt(s_reduce[0] / f32(K) + eps);

    // Step 3: Workgroup 0 writes hidden_new to output
    if wid.x == 0u {
        k = tid;
        while k < K {
            hidden_out[k] = s_data[k];
            k = k + WG;
        }
    }
    workgroupBarrier();

    // Step 4: In-place normalize s_data
    k = tid;
    while k < K {
        s_data[k] = s_data[k] * norm_weight[k] * rms_inv;
        k = k + WG;
    }
    workgroupBarrier();

    // Step 5: Each thread computes one Q4_0 matvec output column
    if col >= params.N {
        return;
    }

    var sum: f32 = 0.0;
    let k_groups = K / 32u;

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

            sum = sum + s_data[elem_off] * e0
                      + s_data[elem_off + 1u] * e1
                      + s_data[elem_off + 2u] * e2
                      + s_data[elem_off + 3u] * e3
                      + s_data[elem_off + 4u] * e4
                      + s_data[elem_off + 5u] * e5
                      + s_data[elem_off + 6u] * e6
                      + s_data[elem_off + 7u] * e7;
        }
    }

    matvec_out[col] = sum;
}
