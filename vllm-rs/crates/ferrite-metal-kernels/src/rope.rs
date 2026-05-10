// SPDX-License-Identifier: Apache-2.0
//! Rotary Position Embedding (RoPE) Metal implementation

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::device::MetalDevice;
use crate::shader_cache::ShaderCache;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandBufferRef = ProtocolObject<dyn MTLCommandBuffer>;

/// RoPE style: NeoX (standard) or Interleaved (GPT-J/CommandR)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    NeoX,
    Interleaved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeDataType {
    F16,
    BF16,
}

/// Metal RoPE kernel wrapper
pub struct MetalRope {
    shader_cache: Arc<ShaderCache>,
}

impl MetalRope {
    pub fn new(device: Arc<MetalDevice>) -> Result<Self, MetalStreamError> {
        let shader_cache = Arc::new(ShaderCache::new(device.device.clone())?);
        Ok(Self { shader_cache })
    }

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
        let kernel_name = match (style, dtype) {
            (RopeStyle::NeoX, RopeDataType::F16) => "rope_neox_f16",
            (RopeStyle::NeoX, RopeDataType::BF16) => "rope_neox_bf16",
            (RopeStyle::Interleaved, RopeDataType::F16) => "rope_interleaved_f16",
            (RopeStyle::Interleaved, RopeDataType::BF16) => "rope_interleaved_bf16",
        };

        let pipeline = self.shader_cache.get_pipeline(kernel_name)?;

        let encoder = command_buffer.computeCommandEncoder().ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
        })?;

        encoder.setComputePipelineState(&pipeline);

        for token_idx in 0..num_tokens {
            let position = unsafe {
                let pos_ptr = positions.contents().as_ptr() as *const u32;
                *pos_ptr.add(token_idx) as usize
            };

            let query_offset = token_idx * num_heads * head_size;
            let key_offset = token_idx * num_kv_heads * head_size;
            let cache_offset = position * rot_dim;

            unsafe {
                encoder.setBuffer_offset_atIndex(Some(query), query_offset * 2, 0);
            }
            if let Some(key_buf) = key {
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(key_buf), key_offset * 2, 1);
                }
            } else {
                unsafe {
                    encoder.setBuffer_offset_atIndex(None, 0, 1);
                }
            }
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(cos_sin_cache), cache_offset * 2, 2);
            }

            unsafe {
                encoder.setBytes_length_atIndex(
                    NonNull::new(&num_heads as *const usize as *mut c_void).unwrap(),
                    std::mem::size_of::<u32>(),
                    3,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&num_kv_heads as *const usize as *mut c_void).unwrap(),
                    std::mem::size_of::<u32>(),
                    4,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&rot_dim as *const usize as *mut c_void).unwrap(),
                    std::mem::size_of::<u32>(),
                    5,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&head_size as *const usize as *mut c_void).unwrap(),
                    std::mem::size_of::<u32>(),
                    6,
                );
            }

            let embed_dim = rot_dim / 2;
            let total_threads = num_heads.max(num_kv_heads) * embed_dim;
            let threadgroup_size = 256.min(total_threads);
            let threadgroups = total_threads.div_ceil(threadgroup_size);

            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: threadgroups,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: threadgroup_size,
                    height: 1,
                    depth: 1,
                },
            );
        }

        encoder.endEncoding();
        Ok(())
    }

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
