// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal weight loading from safetensors files.
//!
//! This module provides the `MetalWeights` struct for loading model weights
//! from safetensors format into Metal buffers. It handles:
//! - Memory-mapped file I/O for efficient loading
//! - Safetensors header parsing
//! - Metal buffer allocation and data transfer
//! - fp16/bf16 dtype support

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use metal::{Buffer, Device, MTLResourceOptions};

/// Metal weights loaded from safetensors.
///
/// Holds memory-mapped safetensors file and Metal buffers for each tensor.
/// Tensors are indexed by their safetensors key (e.g., "model.layers.0.self_attn.q_proj.weight").
pub struct MetalWeights {
    /// Metal device for buffer allocation
    device: Device,
    /// Memory-mapped safetensors file
    _mmap: Mmap,
    /// Parsed tensor metadata from safetensors header
    tensors: HashMap<String, TensorInfo>,
    /// Metal buffers for each loaded tensor
    buffers: HashMap<String, Buffer>,
}

/// Metadata for a single tensor from safetensors header
#[derive(Debug, Clone)]
struct TensorInfo {
    /// Tensor shape (e.g., [4096, 4096] for a weight matrix)
    shape: Vec<usize>,
    /// Data type ("F16", "BF16", "F32", etc.)
    dtype: String,
    /// Byte offset in the safetensors file (after 8-byte header)
    data_offset: usize,
    /// Total size in bytes
    byte_size: usize,
}

impl MetalWeights {
    /// Load safetensors file into Metal buffers.
    ///
    /// # Arguments
    /// * `device` - Metal device for buffer allocation
    /// * `path` - Path to safetensors file (e.g., "model.safetensors")
    ///
    /// # Returns
    /// `MetalWeights` with all tensors loaded into Metal buffers
    pub fn load_safetensors(device: Device, path: &Path) -> Result<Self, String> {
        // Open and memory-map the safetensors file
        let file = File::open(path)
            .map_err(|e| format!("Failed to open safetensors file: {}", e))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap safetensors file: {}", e))?;

        // Parse safetensors header
        let tensors = Self::parse_header(&mmap)?;

        // Allocate Metal buffers and copy data
        let mut buffers = HashMap::new();
        for (name, info) in &tensors {
            let buffer = Self::create_buffer(&device, &mmap, info)?;
            buffers.insert(name.clone(), buffer);
        }

        Ok(Self {
            device,
            _mmap: mmap,
            tensors,
            buffers,
        })
    }

    /// Parse safetensors header to extract tensor metadata.
    ///
    /// Safetensors format:
    /// - First 8 bytes: header length (little-endian u64)
    /// - Next N bytes: JSON header with tensor metadata
    /// - Remaining bytes: tensor data
    fn parse_header(mmap: &Mmap) -> Result<HashMap<String, TensorInfo>, String> {
        if mmap.len() < 8 {
            return Err("Safetensors file too small (< 8 bytes)".to_string());
        }

        // Read header length (first 8 bytes, little-endian)
        let header_len = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3],
            mmap[4], mmap[5], mmap[6], mmap[7],
        ]) as usize;

        if mmap.len() < 8 + header_len {
            return Err(format!(
                "Safetensors file too small for header (need {}, got {})",
                8 + header_len,
                mmap.len()
            ));
        }

        // Parse JSON header
        let header_bytes = &mmap[8..8 + header_len];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| format!("Invalid UTF-8 in safetensors header: {}", e))?;
        let header: serde_json::Value = serde_json::from_str(header_str)
            .map_err(|e| format!("Invalid JSON in safetensors header: {}", e))?;

        // Extract tensor metadata
        let mut tensors = HashMap::new();
        if let Some(obj) = header.as_object() {
            for (name, value) in obj {
                // Skip metadata entry (not a tensor)
                if name == "__metadata__" {
                    continue;
                }

                let tensor_obj = value.as_object()
                    .ok_or_else(|| format!("Tensor '{}' is not an object", name))?;

                // Parse shape
                let shape_arr = tensor_obj.get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("Tensor '{}' missing 'shape' array", name))?;
                let shape: Vec<usize> = shape_arr
                    .iter()
                    .map(|v| v.as_u64().map(|n| n as usize))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| format!("Tensor '{}' has invalid shape", name))?;

                // Parse dtype
                let dtype = tensor_obj.get("dtype")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("Tensor '{}' missing 'dtype' string", name))?
                    .to_string();

                // Parse data_offsets [start, end]
                let offsets = tensor_obj.get("data_offsets")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("Tensor '{}' missing 'data_offsets' array", name))?;
                if offsets.len() != 2 {
                    return Err(format!("Tensor '{}' data_offsets must have 2 elements", name));
                }
                let start = offsets[0].as_u64()
                    .ok_or_else(|| format!("Tensor '{}' data_offsets[0] not a number", name))? as usize;
                let end = offsets[1].as_u64()
                    .ok_or_else(|| format!("Tensor '{}' data_offsets[1] not a number", name))? as usize;

                let byte_size = end - start;
                let data_offset = 8 + header_len + start;

                tensors.insert(name.clone(), TensorInfo {
                    shape,
                    dtype,
                    data_offset,
                    byte_size,
                });
            }
        }

        Ok(tensors)
    }

    /// Create Metal buffer and copy tensor data from mmap.
    fn create_buffer(
        device: &Device,
        mmap: &Mmap,
        info: &TensorInfo,
    ) -> Result<Buffer, String> {
        // Validate data bounds
        if info.data_offset + info.byte_size > mmap.len() {
            return Err(format!(
                "Tensor data out of bounds (offset={}, size={}, file_len={})",
                info.data_offset, info.byte_size, mmap.len()
            ));
        }

        // Get tensor data slice
        let data = &mmap[info.data_offset..info.data_offset + info.byte_size];

        // Allocate Metal buffer (StorageModeShared for CPU-GPU shared memory)
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            info.byte_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(buffer)
    }

    /// Get Metal buffer for a tensor by name.
    ///
    /// # Arguments
    /// * `name` - Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight")
    ///
    /// # Returns
    /// Reference to Metal buffer, or None if tensor not found
    pub fn get_tensor(&self, name: &str) -> Option<&Buffer> {
        self.buffers.get(name)
    }

    /// Get tensor metadata by name.
    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Take ownership of a tensor buffer (removes from internal map).
    ///
    /// Useful when transferring buffers to executor.
    pub fn take_tensor(&mut self, name: &str) -> Option<Buffer> {
        self.buffers.remove(name)
    }

    /// List all tensor names in the safetensors file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get the Metal device used for buffer allocation.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weights_struct_compiles() {
        // Placeholder test - actual loading requires a real safetensors file
        assert!(true, "MetalWeights struct compiles");
    }
}
