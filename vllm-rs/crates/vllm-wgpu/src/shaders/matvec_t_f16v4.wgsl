// Matrix-vector multiply with f16-packed transposed weights: y = x * W_t
// W_t is [K, N] with f16 packed along K: packed[kp * N + col].
// Uses vec4<u32> to read 4 adjacent columns' K-pair values in one load.
// Each thread handles 4 output columns. N must be divisible by 4.

struct Params {
    K: u32,     // full K (must be even)
    N: u32,     // must be divisible by 4
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_t: array<vec4<u32>>;  // packed f16×2, vectorized
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col4 = gid.x;  // each thread handles 4 columns
    let col_base = col4 * 4u;
    if col_base >= params.N {
        return;
    }

    var sum0: f32 = 0.0;
    var sum1: f32 = 0.0;
    var sum2: f32 = 0.0;
    var sum3: f32 = 0.0;

    let k_pairs = params.K / 2u;
    let n4 = params.N / 4u;

    for (var kp: u32 = 0u; kp < k_pairs; kp = kp + 1u) {
        let v = w_t[kp * n4 + col4];
        let p0 = unpack2x16float(v.x);
        let p1 = unpack2x16float(v.y);
        let p2 = unpack2x16float(v.z);
        let p3 = unpack2x16float(v.w);
        let k0 = kp * 2u;
        let x0 = x[k0];
        let x1 = x[k0 + 1u];
        sum0 = sum0 + x0 * p0.x + x1 * p0.y;
        sum1 = sum1 + x0 * p1.x + x1 * p1.y;
        sum2 = sum2 + x0 * p2.x + x1 * p2.y;
        sum3 = sum3 + x0 * p3.x + x1 * p3.y;
    }

    y[col_base] = sum0;
    if col_base + 1u < params.N { y[col_base + 1u] = sum1; }
    if col_base + 2u < params.N { y[col_base + 2u] = sum2; }
    if col_base + 3u < params.N { y[col_base + 3u] = sum3; }
}
