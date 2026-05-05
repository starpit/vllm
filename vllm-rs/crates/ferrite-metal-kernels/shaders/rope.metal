// SPDX-License-Identifier: Apache-2.0
//! Rotary Position Embedding (RoPE) Metal shaders
//!
//! Implements both NeoX-style and GPT-J-style rotary embeddings:
//! - NeoX: pairs element i with i + half_dim
//! - GPT-J (interleaved): pairs element 2i with 2i+1
//!
//! Algorithm:
//! For each pair (x, y):
//!   x' = x * cos(θ) - y * sin(θ)
//!   y' = y * cos(θ) + x * sin(θ)
//!
//! Where θ is position-dependent and pre-computed in cos_sin_cache.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// NeoX-style RoPE (standard Llama, GPT-NeoX)
// ---------------------------------------------------------------------------

/// Apply rotary embedding to a single token's query/key vectors.
/// NeoX style: pairs element i with i + half_dim.
///
/// @param query: [num_heads, head_size] - query vector for this token
/// @param key: [num_kv_heads, head_size] - key vector for this token (nullable)
/// @param cos_sin_cache: [rot_dim] - concatenated [cos; sin] for this position
/// @param num_heads: number of query heads
/// @param num_kv_heads: number of key heads
/// @param rot_dim: rotary dimension (typically head_size or head_size/2)
/// @param head_size: size of each head
kernel void rope_neox_f16(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    constant half* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant half* cos_ptr = cos_sin_cache;
    constant half* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = rot_offset;
        const uint y_index = embed_dim + rot_offset;
        
        const half cos_val = cos_ptr[x_index];
        const half sin_val = sin_ptr[x_index];
        
        device half* head_ptr = query + head_idx * head_size;
        const half x = head_ptr[x_index];
        const half y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = rot_offset;
            const uint y_index = embed_dim + rot_offset;
            
            const half cos_val = cos_ptr[x_index];
            const half sin_val = sin_ptr[x_index];
            
            device half* head_ptr = key + head_idx * head_size;
            const half x = head_ptr[x_index];
            const half y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}

/// BFloat16 variant of NeoX-style RoPE
kernel void rope_neox_bf16(
    device bfloat* query [[buffer(0)]],
    device bfloat* key [[buffer(1)]],
    constant bfloat* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant bfloat* cos_ptr = cos_sin_cache;
    constant bfloat* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = rot_offset;
        const uint y_index = embed_dim + rot_offset;
        
        const bfloat cos_val = cos_ptr[x_index];
        const bfloat sin_val = sin_ptr[x_index];
        
        device bfloat* head_ptr = query + head_idx * head_size;
        const bfloat x = head_ptr[x_index];
        const bfloat y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = rot_offset;
            const uint y_index = embed_dim + rot_offset;
            
            const bfloat cos_val = cos_ptr[x_index];
            const bfloat sin_val = sin_ptr[x_index];
            
            device bfloat* head_ptr = key + head_idx * head_size;
            const bfloat x = head_ptr[x_index];
            const bfloat y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}

// ---------------------------------------------------------------------------
// Interleaved RoPE (GPT-J style, Cohere CommandR)
// ---------------------------------------------------------------------------

/// Apply rotary embedding with interleaved pairing.
/// GPT-J style: pairs element 2i with 2i+1.
///
/// Used by Cohere's CommandR family.
kernel void rope_interleaved_f16(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    constant half* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant half* cos_ptr = cos_sin_cache;
    constant half* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = 2 * rot_offset;
        const uint y_index = 2 * rot_offset + 1;
        
        const half cos_val = cos_ptr[rot_offset];
        const half sin_val = sin_ptr[rot_offset];
        
        device half* head_ptr = query + head_idx * head_size;
        const half x = head_ptr[x_index];
        const half y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = 2 * rot_offset;
            const uint y_index = 2 * rot_offset + 1;
            
            const half cos_val = cos_ptr[rot_offset];
            const half sin_val = sin_ptr[rot_offset];
            
            device half* head_ptr = key + head_idx * head_size;
            const half x = head_ptr[x_index];
            const half y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}

/// BFloat16 variant of interleaved RoPE
kernel void rope_interleaved_bf16(
    device bfloat* query [[buffer(0)]],
    device bfloat* key [[buffer(1)]],
    constant bfloat* cos_sin_cache [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& num_kv_heads [[buffer(4)]],
    constant uint& rot_dim [[buffer(5)]],
    constant uint& head_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint embed_dim = rot_dim / 2;
    constant bfloat* cos_ptr = cos_sin_cache;
    constant bfloat* sin_ptr = cos_sin_cache + embed_dim;
    
    // Apply to query heads
    const uint nq = num_heads * embed_dim;
    if (tid < nq) {
        const uint head_idx = tid / embed_dim;
        const uint rot_offset = tid % embed_dim;
        
        const uint x_index = 2 * rot_offset;
        const uint y_index = 2 * rot_offset + 1;
        
        const bfloat cos_val = cos_ptr[rot_offset];
        const bfloat sin_val = sin_ptr[rot_offset];
        
        device bfloat* head_ptr = query + head_idx * head_size;
        const bfloat x = head_ptr[x_index];
        const bfloat y = head_ptr[y_index];
        
        head_ptr[x_index] = x * cos_val - y * sin_val;
        head_ptr[y_index] = y * cos_val + x * sin_val;
    }
    
    // Apply to key heads (if present)
    if (key != nullptr) {
        const uint nk = num_kv_heads * embed_dim;
        if (tid < nk) {
            const uint head_idx = tid / embed_dim;
            const uint rot_offset = tid % embed_dim;
            
            const uint x_index = 2 * rot_offset;
            const uint y_index = 2 * rot_offset + 1;
            
            const bfloat cos_val = cos_ptr[rot_offset];
            const bfloat sin_val = sin_ptr[rot_offset];
            
            device bfloat* head_ptr = key + head_idx * head_size;
            const bfloat x = head_ptr[x_index];
            const bfloat y = head_ptr[y_index];
            
            head_ptr[x_index] = x * cos_val - y * sin_val;
            head_ptr[y_index] = y * cos_val + x * sin_val;
        }
    }
}
