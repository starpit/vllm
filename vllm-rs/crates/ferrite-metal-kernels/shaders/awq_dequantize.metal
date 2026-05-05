#include <metal_stdlib>
using namespace metal;

// AWQ (Activation-aware Weight Quantization) Dequantization Kernels
// 
// AWQ uses 4-bit integer quantization with group-wise scales and zero-points.
// Format:
// - Weights: INT4 (0-15), packed 8 per uint32
// - Scales: FP16, one per group
// - Zeros: INT4 (0-15), packed 8 per uint32, one per group
// - Group size: Typically 128
//
// Dequantization formula: dequantized = (int4_weight - zero) * scale

// ============================================================================
// INT4 Unpacking Utilities
// ============================================================================

/// Unpack 8 INT4 values from a uint32
/// @param packed: uint32 containing 8 packed INT4 values (4 bits each)
/// @param output: Array of 8 half values to write unpacked results
inline void unpack_int4_to_half(uint packed, thread half* output) {
    // Extract each 4-bit value using shift and mask
    for (int i = 0; i < 8; i++) {
        uint shift = i * 4;
        uint mask = 0xF;
        uint int4_val = (packed >> shift) & mask;
        output[i] = half(int4_val);
    }
}

/// Unpack 8 INT4 values from a uint32 (vectorized version)
/// @param packed: uint32 containing 8 packed INT4 values
/// @return: Array of 8 half values
inline void unpack_int4_to_half_vec(uint packed, thread half* output) {
    // Unroll loop for better performance
    output[0] = half((packed >> 0) & 0xF);
    output[1] = half((packed >> 4) & 0xF);
    output[2] = half((packed >> 8) & 0xF);
    output[3] = half((packed >> 12) & 0xF);
    output[4] = half((packed >> 16) & 0xF);
    output[5] = half((packed >> 20) & 0xF);
    output[6] = half((packed >> 24) & 0xF);
    output[7] = half((packed >> 28) & 0xF);
}

// ============================================================================
// Kernel 1: Simple INT4 Unpacking (for testing)
// ============================================================================

/// Unpack INT4 weights to FP16 without dequantization
/// Used for testing and debugging
kernel void awq_unpack_int4_to_fp16(
    device const uint* packed_weights [[buffer(0)]],
    device half* unpacked_weights [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint packed = packed_weights[gid];
    half output[8];
    unpack_int4_to_half_vec(packed, output);
    
    // Write 8 unpacked values
    for (int i = 0; i < 8; i++) {
        unpacked_weights[gid * 8 + i] = output[i];
    }
}

// ============================================================================
// Kernel 2: Full Dequantization (INT4 -> FP16 with scales and zeros)
// ============================================================================

/// Dequantize INT4 weights to FP16 using group-wise scales and zeros
/// Formula: dequantized = (int4_weight - zero) * scale
///
/// @param packed_weights: INT4 weights, 8 per uint32, shape [IC, OC/8]
/// @param scales: FP16 scales, shape [IC/G, OC]
/// @param packed_zeros: INT4 zeros, 8 per uint32, shape [IC/G, OC/8]
/// @param dequantized_weights: Output FP16 weights, shape [IC, OC]
/// @param group_size: Number of input channels per quantization group (e.g., 128)
/// @param num_out_channels: Total number of output channels (OC)
kernel void awq_dequantize_weights(
    device const uint* packed_weights [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const uint* packed_zeros [[buffer(2)]],
    device half* dequantized_weights [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    constant uint& num_out_channels [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread processes one uint32 (8 packed INT4 weights)
    // gid represents the index in the packed array
    
    // Calculate position in weight matrix
    // packed_weights has shape [IC, OC/8]
    // We need to map gid -> (ic, oc_block) where oc_block = oc / 8
    uint oc_blocks = num_out_channels / 8;
    uint ic = gid / oc_blocks;
    uint oc_block = gid % oc_blocks;
    uint oc_base = oc_block * 8;
    
    // 1. Unpack 8 INT4 weights
    uint packed = packed_weights[gid];
    half weights[8];
    unpack_int4_to_half_vec(packed, weights);
    
    // 2. Determine group index for this input channel
    uint group_idx = ic / group_size;
    
    // 3. Unpack 8 INT4 zeros for this group and output channel block
    uint zero_idx = group_idx * oc_blocks + oc_block;
    uint packed_zero = packed_zeros[zero_idx];
    half zeros[8];
    unpack_int4_to_half_vec(packed_zero, zeros);
    
    // 4. Load 8 scales for this group and output channel block
    // scales has shape [IC/G, OC], so index is [group_idx, oc_base:oc_base+8]
    uint scale_base = group_idx * num_out_channels + oc_base;
    half scale_vals[8];
    for (int i = 0; i < 8; i++) {
        scale_vals[i] = scales[scale_base + i];
    }
    
    // 5. Dequantize: (weight - zero) * scale
    uint output_base = ic * num_out_channels + oc_base;
    for (int i = 0; i < 8; i++) {
        dequantized_weights[output_base + i] = (weights[i] - zeros[i]) * scale_vals[i];
    }
}

// ============================================================================
// Kernel 3: Vectorized Dequantization (optimized for memory bandwidth)
// ============================================================================

/// Vectorized version of dequantization using half4 for better memory bandwidth
/// Same algorithm as awq_dequantize_weights but with vectorized loads/stores
kernel void awq_dequantize_weights_vec4(
    device const uint* packed_weights [[buffer(0)]],
    device const half4* scales [[buffer(1)]],
    device const uint* packed_zeros [[buffer(2)]],
    device half4* dequantized_weights [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    constant uint& num_out_channels [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread processes one uint32 (8 packed INT4 weights)
    // Output as 2x half4 for vectorized writes
    
    uint oc_blocks = num_out_channels / 8;
    uint ic = gid / oc_blocks;
    uint oc_block = gid % oc_blocks;
    uint oc_base = oc_block * 8;
    
    // 1. Unpack 8 INT4 weights
    uint packed = packed_weights[gid];
    half weights[8];
    unpack_int4_to_half_vec(packed, weights);
    
    // 2. Determine group index
    uint group_idx = ic / group_size;
    
    // 3. Unpack 8 INT4 zeros
    uint zero_idx = group_idx * oc_blocks + oc_block;
    uint packed_zero = packed_zeros[zero_idx];
    half zeros[8];
    unpack_int4_to_half_vec(packed_zero, zeros);
    
    // 4. Load 8 scales as 2x half4
    uint scale_base = group_idx * (num_out_channels / 4) + (oc_base / 4);
    half4 scale_vals0 = scales[scale_base];
    half4 scale_vals1 = scales[scale_base + 1];
    
    // 5. Dequantize and write as 2x half4
    uint output_base = ic * (num_out_channels / 4) + (oc_base / 4);
    
    half4 result0 = half4(
        (weights[0] - zeros[0]) * scale_vals0.x,
        (weights[1] - zeros[1]) * scale_vals0.y,
        (weights[2] - zeros[2]) * scale_vals0.z,
        (weights[3] - zeros[3]) * scale_vals0.w
    );
    
    half4 result1 = half4(
        (weights[4] - zeros[4]) * scale_vals1.x,
        (weights[5] - zeros[5]) * scale_vals1.y,
        (weights[6] - zeros[6]) * scale_vals1.z,
        (weights[7] - zeros[7]) * scale_vals1.w
    );
    
    dequantized_weights[output_base] = result0;
    dequantized_weights[output_base + 1] = result1;
}

// ============================================================================
// Kernel 4: Dequantization with Transpose (for row-major -> column-major)
// ============================================================================

/// Dequantize and transpose weights in one pass
/// Useful when MPS GEMM expects column-major layout
///
/// Input: packed_weights [IC, OC/8] row-major
/// Output: dequantized_weights [OC, IC] column-major (transposed)
kernel void awq_dequantize_weights_transpose(
    device const uint* packed_weights [[buffer(0)]],
    device const half* scales [[buffer(1)]],
    device const uint* packed_zeros [[buffer(2)]],
    device half* dequantized_weights [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    constant uint& num_in_channels [[buffer(5)]],
    constant uint& num_out_channels [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint oc_blocks = num_out_channels / 8;
    uint ic = gid / oc_blocks;
    uint oc_block = gid % oc_blocks;
    uint oc_base = oc_block * 8;
    
    // 1. Unpack 8 INT4 weights
    uint packed = packed_weights[gid];
    half weights[8];
    unpack_int4_to_half_vec(packed, weights);
    
    // 2. Determine group index
    uint group_idx = ic / group_size;
    
    // 3. Unpack 8 INT4 zeros
    uint zero_idx = group_idx * oc_blocks + oc_block;
    uint packed_zero = packed_zeros[zero_idx];
    half zeros[8];
    unpack_int4_to_half_vec(packed_zero, zeros);
    
    // 4. Load 8 scales
    uint scale_base = group_idx * num_out_channels + oc_base;
    half scale_vals[8];
    for (int i = 0; i < 8; i++) {
        scale_vals[i] = scales[scale_base + i];
    }
    
    // 5. Dequantize and write transposed: [OC, IC] instead of [IC, OC]
    for (int i = 0; i < 8; i++) {
        uint oc = oc_base + i;
        uint output_idx = oc * num_in_channels + ic;
        dequantized_weights[output_idx] = (weights[i] - zeros[i]) * scale_vals[i];
    }
}
