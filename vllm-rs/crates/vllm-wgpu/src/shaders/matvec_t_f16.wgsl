// Matrix-vector multiply with f16-packed weights: y = x * W_t
// W_t is [K, N] stored as packed f16 pairs along K dimension.
// Buffer is array<u32> where each u32 holds 2 consecutive K-values:
//   w_t_packed[k_pair * N + col] = pack(w_t[2*k_pair, col], w_t[2*k_pair+1, col])
// x is f32, output is f32. Adjacent threads read adjacent u32s → coalesced.

struct Params {
    K: u32,     // full K (must be even)
    N: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_t: array<u32>;  // packed f16×2
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= params.N {
        return;
    }

    var sum: f32 = 0.0;
    let k_pairs = params.K / 2u;

    // Each iteration processes 2 K-values from one u32
    let k_pairs_end4 = (k_pairs / 4u) * 4u;
    for (var kp: u32 = 0u; kp < k_pairs_end4; kp = kp + 4u) {
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
    for (var kp: u32 = k_pairs_end4; kp < k_pairs; kp = kp + 1u) {
        let v = unpack2x16float(w_t[kp * params.N + col]);
        let k0 = kp * 2u;
        sum = sum + x[k0] * v.x + x[k0 + 1u] * v.y;
    }

    y[col] = sum;
}
