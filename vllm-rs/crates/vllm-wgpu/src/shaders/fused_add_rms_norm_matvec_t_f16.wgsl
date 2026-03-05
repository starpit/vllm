// Fused: residual_add + RMS norm + matvec with transposed f16 weights.
// Replaces 2 dispatches (fused_add_rms_norm + matvec_t_f16) with 1.
//
// Each workgroup:
//   1. Cooperatively computes hidden_new = residual + addition (in shared mem)
//   2. Cooperatively computes sum-of-squares reduction for RMS norm
//   3. Workgroup 0 writes hidden_new to output buffer
//   4. In-place normalize shared mem: s_data[k] *= norm_weight[k] * rms_inv
//   5. Each thread computes one matvec output column from normalized input
//
// Uses a single shared array to support K up to 4096 within 32KB workgroup storage.
// K must be even.

struct Params {
    K: u32,        // hidden size
    N: u32,        // output dimension (number of output columns)
    eps_bits: u32, // f32 epsilon as bits (bitcast in shader)
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> residual: array<f32>;
@group(0) @binding(1) var<storage, read> addition: array<f32>;
@group(0) @binding(2) var<storage, read> norm_weight: array<f32>;
@group(0) @binding(3) var<storage, read> w_t: array<u32>;  // f16 packed, [K/2, N]
@group(0) @binding(4) var<storage, read_write> matvec_out: array<f32>;
@group(0) @binding(5) var<storage, read_write> hidden_out: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

const WG: u32 = 256u;

var<workgroup> s_data: array<f32, __MAX_K__>;   // set at runtime to actual hidden_size
var<workgroup> s_reduce: array<f32, 256>;  // for RMS reduction — 1KB

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
    // and sum of squares for RMS norm.
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

    // Compute RMS scale
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

    // Step 5: Each thread computes one matvec output column
    if col >= params.N {
        return;
    }

    var sum: f32 = 0.0;
    let k_pairs = K / 2u;
    let kp_end4 = (k_pairs / 4u) * 4u;

    for (var kp: u32 = 0u; kp < kp_end4; kp = kp + 4u) {
        let v0 = unpack2x16float(w_t[(kp) * params.N + col]);
        let v1 = unpack2x16float(w_t[(kp + 1u) * params.N + col]);
        let v2 = unpack2x16float(w_t[(kp + 2u) * params.N + col]);
        let v3 = unpack2x16float(w_t[(kp + 3u) * params.N + col]);
        let k0 = kp * 2u;
        sum = sum + s_data[k0] * v0.x + s_data[k0 + 1u] * v0.y
                  + s_data[k0 + 2u] * v1.x + s_data[k0 + 3u] * v1.y
                  + s_data[k0 + 4u] * v2.x + s_data[k0 + 5u] * v2.y
                  + s_data[k0 + 6u] * v3.x + s_data[k0 + 7u] * v3.y;
    }
    for (var kp: u32 = kp_end4; kp < k_pairs; kp = kp + 1u) {
        let v = unpack2x16float(w_t[kp * params.N + col]);
        let k0 = kp * 2u;
        sum = sum + s_data[k0] * v.x + s_data[k0 + 1u] * v.y;
    }

    matvec_out[col] = sum;
}
