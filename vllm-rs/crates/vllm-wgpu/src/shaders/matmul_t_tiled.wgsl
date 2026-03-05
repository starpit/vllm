// Register-tiled matrix multiplication with B transposed: C = A * B^T
// A: [M, K], B: [N, K] (row-major), C: [M, N]
// Each thread computes a 4x4 tile of output. 16x16 workgroup = 64x64 output tile.
// No shared memory — register tiling with manually unrolled K loop.

struct Params {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TM: u32 = 4u;
const TN: u32 = 4u;
const WG: u32 = 16u;  // workgroup size per dimension
// Output tile per workgroup: WG*TM x WG*TN = 64x64

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    // This thread is responsible for output rows [row_base..row_base+TM) and cols [col_base..col_base+TN)
    let row_base = wid.x * (WG * TM) + lid.x * TM;
    let col_base = wid.y * (WG * TN) + lid.y * TN;

    // Accumulator registers: 4x4 = 16 values
    var acc00: f32 = 0.0; var acc01: f32 = 0.0; var acc02: f32 = 0.0; var acc03: f32 = 0.0;
    var acc10: f32 = 0.0; var acc11: f32 = 0.0; var acc12: f32 = 0.0; var acc13: f32 = 0.0;
    var acc20: f32 = 0.0; var acc21: f32 = 0.0; var acc22: f32 = 0.0; var acc23: f32 = 0.0;
    var acc30: f32 = 0.0; var acc31: f32 = 0.0; var acc32: f32 = 0.0; var acc33: f32 = 0.0;

    // Check bounds for the full 4x4 tile
    let r0_valid = row_base < params.M;
    let r1_valid = (row_base + 1u) < params.M;
    let r2_valid = (row_base + 2u) < params.M;
    let r3_valid = (row_base + 3u) < params.M;
    let c0_valid = col_base < params.N;
    let c1_valid = (col_base + 1u) < params.N;
    let c2_valid = (col_base + 2u) < params.N;
    let c3_valid = (col_base + 3u) < params.N;

    // Pre-compute row offsets into A and B (B is transposed, so cols index B rows)
    let a_r0 = row_base * params.K;
    let a_r1 = (row_base + 1u) * params.K;
    let a_r2 = (row_base + 2u) * params.K;
    let a_r3 = (row_base + 3u) * params.K;
    let b_c0 = col_base * params.K;
    let b_c1 = (col_base + 1u) * params.K;
    let b_c2 = (col_base + 2u) * params.K;
    let b_c3 = (col_base + 3u) * params.K;

    // Loop over K dimension
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        // Load A values for this k (4 rows)
        let a0 = select(0.0, a[a_r0 + k], r0_valid);
        let a1 = select(0.0, a[a_r1 + k], r1_valid);
        let a2 = select(0.0, a[a_r2 + k], r2_valid);
        let a3 = select(0.0, a[a_r3 + k], r3_valid);

        // Load B values for this k (4 cols = 4 B rows since transposed)
        let b0 = select(0.0, b[b_c0 + k], c0_valid);
        let b1 = select(0.0, b[b_c1 + k], c1_valid);
        let b2 = select(0.0, b[b_c2 + k], c2_valid);
        let b3 = select(0.0, b[b_c3 + k], c3_valid);

        // Accumulate outer product
        acc00 = acc00 + a0 * b0; acc01 = acc01 + a0 * b1; acc02 = acc02 + a0 * b2; acc03 = acc03 + a0 * b3;
        acc10 = acc10 + a1 * b0; acc11 = acc11 + a1 * b1; acc12 = acc12 + a1 * b2; acc13 = acc13 + a1 * b3;
        acc20 = acc20 + a2 * b0; acc21 = acc21 + a2 * b1; acc22 = acc22 + a2 * b2; acc23 = acc23 + a2 * b3;
        acc30 = acc30 + a3 * b0; acc31 = acc31 + a3 * b1; acc32 = acc32 + a3 * b2; acc33 = acc33 + a3 * b3;
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
