// Matrix multiplication with B transposed: C = A * B^T
// A: [M, K], B: [N, K] (stored row-major), C: [M, N]
// Each thread computes one output element: C[row,col] = dot(A[row,:], B[col,:])

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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;

    if row >= params.M || col >= params.N {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k = k + 1u) {
        sum = sum + a[row * params.K + k] * b[col * params.K + k];
    }
    c[row * params.N + col] = sum;
}
