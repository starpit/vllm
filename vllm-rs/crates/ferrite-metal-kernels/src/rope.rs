// SPDX-License-Identifier: Apache-2.0
//! Rotary Position Embedding (RoPE) Metal implementation

use metal::{Buffer, CommandBufferRef, MTLSize};
use std::sync::Arc;

use crate::device::MetalDevice;
use crate::shader_cache::ShaderCache;
use crate::stream::MetalStreamError;

/// RoPE style: NeoX (standard) or Interleaved (GPT-J/CommandR)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// NeoX style: pairs element i with i + half_dim
    /// Used by Llama, GPT-NeoX, most models
    NeoX,
    /// Interleaved style: pairs element 2i with 2i+1
    /// Used by Cohere CommandR family
    Interleaved,
}

/// Data type for RoPE computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeDataType {
    F16,
    BF16,
}

/// Metal RoPE kernel wrapper
pub struct MetalRope {
    device: Arc<MetalDevice>,
    shader_cache: Arc<ShaderCache>,
}

impl MetalRope {
    /// Create a new MetalRope instance
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, MetalStreamError> {
        let shader_cache = Arc::new(ShaderCache::new(device.device.clone())?);
        Ok(Self {
            device,
            shader_cache,
        })
    }

    /// Apply rotary position embedding to query and optionally key tensors
    ///
    /// # Arguments
    /// * `query` - Query tensor [num_tokens, num_heads, head_size]
    /// * `key` - Optional key tensor [num_tokens, num_kv_heads, head_size]
    /// * `cos_sin_cache` - Pre-computed cos/sin cache [max_pos, rot_dim]
    /// * `positions` - Token positions [num_tokens]
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key heads
    /// * `rot_dim` - Rotary dimension (typically head_size)
    /// * `head_size` - Size of each head
    /// * `style` - RoPE style (NeoX or Interleaved)
    /// * `dtype` - Data type (F16 or BF16)
    pub fn execute(
        &self,
        command_buffer: &CommandBufferRef,
        query: &Buffer,
        key: Option<&Buffer>,
        cos_sin_cache: &Buffer,
        positions: &Buffer,
        num_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rot_dim: usize,
        head_size: usize,
        style: RopeStyle,
        dtype: RopeDataType,
    ) -> Result<(), MetalStreamError> {
        // Select kernel based on style and dtype
        let kernel_name = match (style, dtype) {
            (RopeStyle::NeoX, RopeDataType::F16) => "rope_neox_f16",
            (RopeStyle::NeoX, RopeDataType::BF16) => "rope_neox_bf16",
            (RopeStyle::Interleaved, RopeDataType::F16) => "rope_interleaved_f16",
            (RopeStyle::Interleaved, RopeDataType::BF16) => "rope_interleaved_bf16",
        };

        let pipeline = self.shader_cache.get_pipeline(kernel_name)?;

        let encoder = command_buffer.new_compute_command_encoder().to_owned();

        encoder.set_compute_pipeline_state(&pipeline);

        // Process each token separately (each token gets its own position-specific cos/sin)
        for token_idx in 0..num_tokens {
            // Read position for this token
            let position = unsafe {
                let pos_ptr = positions.contents() as *const u32;
                *pos_ptr.add(token_idx) as usize
            };

            // Calculate buffer offsets for this token
            let query_offset = token_idx * num_heads * head_size;
            let key_offset = token_idx * num_kv_heads * head_size;
            let cache_offset = position * rot_dim;

            // Set buffers with offsets
            encoder.set_buffer(0, Some(query), (query_offset * 2) as u64); // *2 for fp16
            if let Some(key_buf) = key {
                encoder.set_buffer(1, Some(key_buf), (key_offset * 2) as u64);
            } else {
                encoder.set_buffer(1, None, 0);
            }
            encoder.set_buffer(2, Some(cos_sin_cache), (cache_offset * 2) as u64);

            // Set scalar parameters
            encoder.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &num_heads as *const usize as *const _,
            );
            encoder.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                &num_kv_heads as *const usize as *const _,
            );
            encoder.set_bytes(
                5,
                std::mem::size_of::<u32>() as u64,
                &rot_dim as *const usize as *const _,
            );
            encoder.set_bytes(
                6,
                std::mem::size_of::<u32>() as u64,
                &head_size as *const usize as *const _,
            );

            // Dispatch threads
            let embed_dim = rot_dim / 2;
            let total_threads = num_heads.max(num_kv_heads) * embed_dim;
            let threadgroup_size = 256.min(total_threads);
            let threadgroups = (total_threads + threadgroup_size - 1) / threadgroup_size;

            encoder.dispatch_thread_groups(
                MTLSize::new(threadgroups as u64, 1, 1),
                MTLSize::new(threadgroup_size as u64, 1, 1),
            );
        }

        encoder.end_encoding();
        Ok(())
    }

    /// Execute RoPE with NeoX style (standard Llama/GPT-NeoX)
    pub fn execute_neox(
        &self,
        command_buffer: &CommandBufferRef,
        query: &Buffer,
        key: Option<&Buffer>,
        cos_sin_cache: &Buffer,
        positions: &Buffer,
        num_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rot_dim: usize,
        head_size: usize,
        dtype: RopeDataType,
    ) -> Result<(), MetalStreamError> {
        self.execute(
            command_buffer,
            query,
            key,
            cos_sin_cache,
            positions,
            num_tokens,
            num_heads,
            num_kv_heads,
            rot_dim,
            head_size,
            RopeStyle::NeoX,
            dtype,
        )
    }

    /// Execute RoPE with interleaved style (Cohere CommandR)
    pub fn execute_interleaved(
        &self,
        command_buffer: &CommandBufferRef,
        query: &Buffer,
        key: Option<&Buffer>,
        cos_sin_cache: &Buffer,
        positions: &Buffer,
        num_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rot_dim: usize,
        head_size: usize,
        dtype: RopeDataType,
    ) -> Result<(), MetalStreamError> {
        self.execute(
            command_buffer,
            query,
            key,
            cos_sin_cache,
            positions,
            num_tokens,
            num_heads,
            num_kv_heads,
            rot_dim,
            head_size,
            RopeStyle::Interleaved,
            dtype,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    fn create_test_device() -> Arc<MetalDevice> {
        Arc::new(crate::device::detect_device().expect("Failed to detect Metal device"))
    }

    #[test]
    fn test_rope_neox_basic() {
        let device = create_test_device();
        let rope = MetalRope::new(device.clone()).expect("Failed to create MetalRope");

        // Test parameters
        let num_tokens = 2;
        let num_heads = 4;
        let num_kv_heads = 4;
        let head_size = 64;
        let rot_dim = 64;

        // Create test buffers
        let query_size = num_tokens * num_heads * head_size;
        let key_size = num_tokens * num_kv_heads * head_size;
        let cache_size = 2048 * rot_dim; // max_pos = 2048
        let pos_size = num_tokens;

        let query = device.device.new_buffer(
            (query_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let key = device.device.new_buffer(
            (key_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let cos_sin_cache = device.device.new_buffer(
            (cache_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let positions = device.device.new_buffer(
            (pos_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Initialize positions [0, 1]
        unsafe {
            let pos_ptr = positions.contents() as *mut u32;
            *pos_ptr = 0;
            *pos_ptr.add(1) = 1;
        }

        // Initialize cos_sin_cache with zeros (fp16 format)
        // In a real test, we'd initialize with proper cos/sin values
        unsafe {
            let cache_ptr = cos_sin_cache.contents() as *mut u8;
            std::ptr::write_bytes(cache_ptr, 0, cache_size * 2);
        }

        // Execute
        let command_buffer = device.queue.new_command_buffer();
        rope.execute_neox(
            command_buffer,
            &query,
            Some(&key),
            &cos_sin_cache,
            &positions,
            num_tokens,
            num_heads,
            num_kv_heads,
            rot_dim,
            head_size,
            RopeDataType::F16,
        )
        .expect("RoPE execution failed");

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Basic smoke test - just verify it doesn't crash
        assert!(true);
    }

    #[test]
    fn test_rope_interleaved_basic() {
        let device = create_test_device();
        let rope = MetalRope::new(device.clone()).expect("Failed to create MetalRope");

        let num_tokens = 1;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_size = 32;
        let rot_dim = 32;

        let query_size = num_tokens * num_heads * head_size;
        let cache_size = 1024 * rot_dim;
        let pos_size = num_tokens;

        let query = device.device.new_buffer(
            (query_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let cos_sin_cache = device.device.new_buffer(
            (cache_size * 2) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let positions = device.device.new_buffer(
            (pos_size * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            let pos_ptr = positions.contents() as *mut u32;
            *pos_ptr = 0;
        }

        let command_buffer = device.queue.new_command_buffer();
        rope.execute_interleaved(
            command_buffer,
            &query,
            None, // No key
            &cos_sin_cache,
            &positions,
            num_tokens,
            num_heads,
            num_kv_heads,
            rot_dim,
            head_size,
            RopeDataType::F16,
        )
        .expect("RoPE execution failed");

        command_buffer.commit();
        command_buffer.wait_until_completed();

        assert!(true);
    }
}
