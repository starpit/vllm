// Matrix-vector multiply with Q4_0-packed weights: y = x * W_t
// W_t is stored as Q4_0 blocks in transposed [K/32, N] order.
// Each block is 5 u32s (20 bytes):
//   u32[0]: f16 scale in low 16 bits
//   u32[1..5]: 32 nibble values packed sequentially
//     Each u32 has 8 elements: byte b has elem 2b (low nibble), elem 2b+1 (high nibble)
// Buffer indexed as: w_q4[(kg * N + col) * 5 + word]
// x is f32, output is f32.

struct Params {
    K: u32,     // full K (must be multiple of 32)
    N: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w_q4: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= params.N {
        return;
    }

    var sum: f32 = 0.0;
    let k_groups = params.K / 32u;

    for (var kg: u32 = 0u; kg < k_groups; kg = kg + 1u) {
        let base = (kg * params.N + col) * 5u;
        let scale = unpack2x16float(w_q4[base]).x;
        let k_off = kg * 32u;

        // 4 u32s of nibble data, 8 elements each
        for (var w: u32 = 0u; w < 4u; w = w + 1u) {
            let packed = w_q4[base + 1u + w];
            let elem_off = k_off + w * 8u;

            // Unroll 4 bytes (8 elements) per u32
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

            sum = sum + x[elem_off] * e0
                      + x[elem_off + 1u] * e1
                      + x[elem_off + 2u] * e2
                      + x[elem_off + 3u] * e3
                      + x[elem_off + 4u] * e4
                      + x[elem_off + 5u] * e5
                      + x[elem_off + 6u] * e6
                      + x[elem_off + 7u] * e7;
        }
    }

    y[col] = sum;
}
