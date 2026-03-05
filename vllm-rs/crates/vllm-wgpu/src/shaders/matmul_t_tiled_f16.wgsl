// Register-tiled matrix multiplication with B transposed and f16-packed: C = A * B^T
// A: [M, K] f32, B: [N, K] stored as packed f16x2 (row-major, K packed), C: [M, N] f32
// Each thread computes a 4x4 tile of output. 16x16 workgroup = 64x64 output tile.

struct Params {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;  // packed f16x2
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TM: u32 = 4u;
const TN: u32 = 4u;
const WG: u32 = 16u;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row_base = wid.x * (WG * TM) + lid.x * TM;
    let col_base = wid.y * (WG * TN) + lid.y * TN;

    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    let r0_valid = row_base < params.M;
    let r1_valid = (row_base + 1u) < params.M;
    let r2_valid = (row_base + 2u) < params.M;
    let r3_valid = (row_base + 3u) < params.M;
    let c0_valid = col_base < params.N;
    let c1_valid = (col_base + 1u) < params.N;
    let c2_valid = (col_base + 2u) < params.N;
    let c3_valid = (col_base + 3u) < params.N;

    let a_r0 = row_base * params.K;
    let a_r1 = (row_base + 1u) * params.K;
    let a_r2 = (row_base + 2u) * params.K;
    let a_r3 = (row_base + 3u) * params.K;

    let k_half = params.K / 2u;
    let b_c0 = col_base * k_half;
    let b_c1 = (col_base + 1u) * k_half;
    let b_c2 = (col_base + 2u) * k_half;
    let b_c3 = (col_base + 3u) * k_half;

    // Process 2 K elements per iteration (f16x2 packed)
    for (var kh: u32 = 0u; kh < k_half; kh = kh + 1u) {
        let k = kh * 2u;

        // Load A values (2 per row for this k pair)
        let a0_0 = select(0.0, a[a_r0 + k], r0_valid);
        let a0_1 = select(0.0, a[a_r0 + k + 1u], r0_valid && (k + 1u) < params.K);
        let a1_0 = select(0.0, a[a_r1 + k], r1_valid);
        let a1_1 = select(0.0, a[a_r1 + k + 1u], r1_valid && (k + 1u) < params.K);
        let a2_0 = select(0.0, a[a_r2 + k], r2_valid);
        let a2_1 = select(0.0, a[a_r2 + k + 1u], r2_valid && (k + 1u) < params.K);
        let a3_0 = select(0.0, a[a_r3 + k], r3_valid);
        let a3_1 = select(0.0, a[a_r3 + k + 1u], r3_valid && (k + 1u) < params.K);

        // Load B values (packed f16x2)
        var b0: vec2<f32> = vec2<f32>(0.0, 0.0);
        var b1: vec2<f32> = vec2<f32>(0.0, 0.0);
        var b2: vec2<f32> = vec2<f32>(0.0, 0.0);
        var b3: vec2<f32> = vec2<f32>(0.0, 0.0);
        if c0_valid { b0 = unpack2x16float(b[b_c0 + kh]); }
        if c1_valid { b1 = unpack2x16float(b[b_c1 + kh]); }
        if c2_valid { b2 = unpack2x16float(b[b_c2 + kh]); }
        if c3_valid { b3 = unpack2x16float(b[b_c3 + kh]); }

        // Accumulate: 2 k values per iteration
        acc00 = acc00 + a0_0 * b0.x + a0_1 * b0.y;
        acc01 = acc01 + a0_0 * b1.x + a0_1 * b1.y;
        acc02 = acc02 + a0_0 * b2.x + a0_1 * b2.y;
        acc03 = acc03 + a0_0 * b3.x + a0_1 * b3.y;

        acc10 = acc10 + a1_0 * b0.x + a1_1 * b0.y;
        acc11 = acc11 + a1_0 * b1.x + a1_1 * b1.y;
        acc12 = acc12 + a1_0 * b2.x + a1_1 * b2.y;
        acc13 = acc13 + a1_0 * b3.x + a1_1 * b3.y;

        acc20 = acc20 + a2_0 * b0.x + a2_1 * b0.y;
        acc21 = acc21 + a2_0 * b1.x + a2_1 * b1.y;
        acc22 = acc22 + a2_0 * b2.x + a2_1 * b2.y;
        acc23 = acc23 + a2_0 * b3.x + a2_1 * b3.y;

        acc30 = acc30 + a3_0 * b0.x + a3_1 * b0.y;
        acc31 = acc31 + a3_0 * b1.x + a3_1 * b1.y;
        acc32 = acc32 + a3_0 * b2.x + a3_1 * b2.y;
        acc33 = acc33 + a3_0 * b3.x + a3_1 * b3.y;
    }

    // Write results
    if r0_valid && c0_valid { c[row_base * params.N + col_base] = acc00; }
    if r0_valid && c1_valid { c[row_base * params.N + col_base + 1u] = acc01; }
    if r0_valid && c2_valid { c[row_base * params.N + col_base + 2u] = acc02; }
    if r0_valid && c3_valid { c[row_base * params.N + col_base + 3u] = acc03; }

    if r1_valid && c0_valid { c[(row_base + 1u) * params.N + col_base] = acc10; }
    if r1_valid && c1_valid { c[(row_base + 1u) * params.N + col_base + 1u] = acc11; }
    if r1_valid && c2_valid { c[(row_base + 1u) * params.N + col_base + 2u] = acc12; }
    if r1_valid && c3_valid { c[(row_base + 1u) * params.N + col_base + 3u] = acc13; }

    if r2_valid && c0_valid { c[(row_base + 2u) * params.N + col_base] = acc20; }
    if r2_valid && c1_valid { c[(row_base + 2u) * params.N + col_base + 1u] = acc21; }
    if r2_valid && c2_valid { c[(row_base + 2u) * params.N + col_base + 2u] = acc22; }
    if r2_valid && c3_valid { c[(row_base + 2u) * params.N + col_base + 3u] = acc23; }

    if r3_valid && c0_valid { c[(row_base + 3u) * params.N + col_base] = acc30; }
    if r3_valid && c1_valid { c[(row_base + 3u) * params.N + col_base + 1u] = acc31; }
    if r3_valid && c2_valid { c[(row_base + 3u) * params.N + col_base + 2u] = acc32; }
    if r3_valid && c3_valid { c[(row_base + 3u) * params.N + col_base + 3u] = acc33; }
}
