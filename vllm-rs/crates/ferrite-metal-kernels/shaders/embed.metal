// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Embed: Embedding lookup operation
// out[i, :] = table[indices[i], :]
// ============================================================================

kernel void embed_f16(
    device const half* table [[buffer(0)]],      // [vocab_size, hidden_size]
    device const int* indices [[buffer(1)]],     // [num_tokens]
    device half* out [[buffer(2)]],              // [num_tokens, hidden_size]
    constant uint& hidden_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread processes one token (copies one row)
    int idx = indices[tid];
    device const half* src = table + idx * hidden_size;
    device half* dst = out + tid * hidden_size;
    
    // Copy the entire row
    for (uint i = 0; i < hidden_size; i++) {
        dst[i] = src[i];
    }
}

kernel void embed_bf16(
    device const bfloat* table [[buffer(0)]],    // [vocab_size, hidden_size]
    device const int* indices [[buffer(1)]],     // [num_tokens]
    device bfloat* out [[buffer(2)]],            // [num_tokens, hidden_size]
    constant uint& hidden_size [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread processes one token (copies one row)
    int idx = indices[tid];
    device const bfloat* src = table + idx * hidden_size;
    device bfloat* dst = out + tid * hidden_size;
    
    // Copy the entire row
    for (uint i = 0; i < hidden_size; i++) {
        dst[i] = src[i];
    }
}
