// Matrix-vector multiply: y = x * W_t where W_t is [K, N] (pre-transposed).
// Each thread computes one output element. Adjacent threads read adjacent
// W_t elements within each K iteration → coalesced memory access.

struct Params {
    K: u32,
    N: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_t: array<f32>;  // [K, N] layout
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= params.N {
        return;
    }

    var sum: f32 = 0.0;

    let k_end4 = (params.K / 4u) * 4u;
    for (var k: u32 = 0u; k < k_end4; k = k + 4u) {
        sum = sum + x[k] * w_t[k * params.N + col]
                  + x[k + 1u] * w_t[(k + 1u) * params.N + col]
                  + x[k + 2u] * w_t[(k + 2u) * params.N + col]
                  + x[k + 3u] * w_t[(k + 3u) * params.N + col];
    }
    for (var k: u32 = k_end4; k < params.K; k = k + 1u) {
        sum = sum + x[k] * w_t[k * params.N + col];
    }

    y[col] = sum;
}
