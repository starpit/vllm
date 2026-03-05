// Tiled matrix multiplication: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
// Uses 16×16 tiles in workgroup shared memory for better GPU utilization.

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

const TILE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>; // 16×16
var<workgroup> tile_b: array<f32, 256>; // 16×16

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = gid.x;
    let col = gid.y;
    let lr = lid.x;
    let lc = lid.y;

    var sum: f32 = 0.0;
    let num_tiles = (params.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A into shared memory
        let a_col = t * TILE + lc;
        if row < params.M && a_col < params.K {
            tile_a[lr * TILE + lc] = a[row * params.K + a_col];
        } else {
            tile_a[lr * TILE + lc] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = t * TILE + lr;
        if b_row < params.K && col < params.N {
            tile_b[lr * TILE + lc] = b[b_row * params.N + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
        }

        workgroupBarrier();

        // Accumulate from shared memory
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            sum = sum + tile_a[lr * TILE + k] * tile_b[k * TILE + lc];
        }

        workgroupBarrier();
    }

    if row < params.M && col < params.N {
        c[row * params.N + col] = sum;
    }
}
