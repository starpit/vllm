// Matrix-vector multiply: y = x * W^T where W is [N, K] (row-major).
// Each thread computes one output element y[col] = dot(x, W[col, :]).

struct Params {
    K: u32,
    N: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= params.N {
        return;
    }

    let w_base = col * params.K;
    var sum: f32 = 0.0;

    let k_end4 = (params.K / 4u) * 4u;
    for (var k: u32 = 0u; k < k_end4; k = k + 4u) {
        sum = sum + x[k] * w[w_base + k]
                  + x[k + 1u] * w[w_base + k + 1u]
                  + x[k + 2u] * w[w_base + k + 2u]
                  + x[k + 3u] * w[w_base + k + 3u];
    }
    for (var k: u32 = k_end4; k < params.K; k = k + 1u) {
        sum = sum + x[k] * w[w_base + k];
    }

    y[col] = sum;
}
