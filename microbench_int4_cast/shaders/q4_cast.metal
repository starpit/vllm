#include <metal_stdlib>
using namespace metal;

// Constants are templated via function constants so we can reuse the same
// shader source for multiple shapes.
constant uint K          [[function_constant(0)]];
constant uint GS         [[function_constant(1)]];
constant uint K_OVER_8   [[function_constant(2)]]; // K / 8
constant uint K_OVER_GS  [[function_constant(3)]]; // K / GS

// One threadgroup per output element. 32 lanes = 1 simdgroup.
// Each lane handles one full group of GS=64 weights (so 32 lanes × 64 = 2048 K)
// or two full groups when K=4096, etc. We compute groups_per_lane = K_OVER_GS / 32.
//
// Inner FMA loop dequants packed 4-bit weights using a per-group scale + bias,
// then accumulates into a thread-local scalar. simd_sum reduces across lanes.
// Lane 0 writes the final output.

// ============================================================================
// Variant A: cast-at-load — scales/biases are pre-cast to bf16 in the buffer.
// Kernel just loads bfloat directly.
// ============================================================================
[[kernel]] void q4_qmv_cast_at_load(
    const device uint*   w  [[buffer(0)]],   // [N, K/8] packed
    const device bfloat* s  [[buffer(1)]],   // [N, K/GS] bf16
    const device bfloat* b  [[buffer(2)]],   // [N, K/GS] bf16
    const device bfloat* x  [[buffer(3)]],   // [K]
    device       bfloat* y  [[buffer(4)]],   // [N]
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint groups_per_lane = K_OVER_GS / 32u;
    const uint k_per_lane      = K / 32u;

    float acc = 0.0f;

    for (uint g = 0; g < groups_per_lane; ++g) {
        // Group index for this lane on this iteration:
        // lane handles groups [lane * groups_per_lane + g] across the row.
        uint group_idx = lane * groups_per_lane + g;

        bfloat scale = s[tg * K_OVER_GS + group_idx];
        bfloat bias  = b[tg * K_OVER_GS + group_idx];

        // GS weights packed into GS/8 uint32 values. With GS=64, that's 8 packed
        // u32s per group. Iterate through them.
        const uint k_base_in_row    = group_idx * GS;     // K position
        const uint w_idx_base_row   = tg * K_OVER_8 + k_base_in_row / 8u;

        for (uint pi = 0; pi < GS / 8u; ++pi) {
            uint packed = w[w_idx_base_row + pi];
            uint k0 = k_base_in_row + pi * 8u;

            #pragma unroll
            for (uint q = 0; q < 8; ++q) {
                uint nib  = (packed >> (q * 4u)) & 0xFu;
                bfloat wd = bfloat(nib) * scale + bias;
                acc += float(x[k0 + q]) * float(wd);
            }
        }
    }

    // Reduce across the simdgroup
    acc = simd_sum(acc);
    if (lane == 0) {
        y[tg] = bfloat(acc);
    }
}

// ============================================================================
// Variant B: cast-in-register — scales/biases stored as half (f16) in memory,
// kernel casts to bfloat at the load point each group iteration.
// ============================================================================
[[kernel]] void q4_qmv_cast_in_register(
    const device uint*   w  [[buffer(0)]],   // [N, K/8] packed
    const device half*   s  [[buffer(1)]],   // [N, K/GS] f16  <-- different
    const device half*   b  [[buffer(2)]],   // [N, K/GS] f16  <-- different
    const device bfloat* x  [[buffer(3)]],   // [K]
    device       bfloat* y  [[buffer(4)]],   // [N]
    uint tg   [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint groups_per_lane = K_OVER_GS / 32u;

    float acc = 0.0f;

    for (uint g = 0; g < groups_per_lane; ++g) {
        uint group_idx = lane * groups_per_lane + g;

        // Same load address, but f16 storage → cast to bfloat in register.
        bfloat scale = bfloat(s[tg * K_OVER_GS + group_idx]);
        bfloat bias  = bfloat(b[tg * K_OVER_GS + group_idx]);

        const uint k_base_in_row    = group_idx * GS;
        const uint w_idx_base_row   = tg * K_OVER_8 + k_base_in_row / 8u;

        for (uint pi = 0; pi < GS / 8u; ++pi) {
            uint packed = w[w_idx_base_row + pi];
            uint k0 = k_base_in_row + pi * 8u;

            #pragma unroll
            for (uint q = 0; q < 8; ++q) {
                uint nib  = (packed >> (q * 4u)) & 0xFu;
                bfloat wd = bfloat(nib) * scale + bias;
                acc += float(x[k0 + q]) * float(wd);
            }
        }
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        y[tg] = bfloat(acc);
    }
}
