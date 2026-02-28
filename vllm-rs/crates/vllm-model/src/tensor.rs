// SPDX-License-Identifier: Apache-2.0
//! Tensor and device abstractions wrapping `candle-core`.
//!
//! Provides a thin vLLM-specific API over `candle_core::Tensor` that:
//! - Re-exports `candle_core::DType` and `candle_core::Device`
//! - Adds convenience constructors and conversion utilities
//! - Provides tensor sharding helpers for tensor parallelism
//!
//! We wrap rather than re-invent so that users can drop down to the
//! underlying `candle_core::Tensor` when needed.

use std::fmt;

use candle_core::{DType, Device, Shape, Tensor, WithDType};

use crate::error::{ModelError, ModelResult};

// Re-export candle-core types for convenience.
pub use candle_core::{DType as TensorDType, Device as TensorDevice};

// ---------------------------------------------------------------------------
// DType helpers
// ---------------------------------------------------------------------------

/// Number of bytes per element for a given dtype.
pub fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::U8 => 1,
        DType::U32 | DType::I32 | DType::F32 => 4,
        DType::I16 | DType::BF16 | DType::F16 => 2,
        DType::I64 | DType::F64 => 8,
        DType::F8E4M3 | DType::F8E8M0 => 1,
        DType::F6E2M3 | DType::F6E3M2 => 1,
        DType::F4 => 1,
    }
}

/// Convert a safetensors dtype string to candle DType.
pub fn dtype_from_str(s: &str) -> ModelResult<DType> {
    match s {
        "F16" | "float16" => Ok(DType::F16),
        "BF16" | "bfloat16" => Ok(DType::BF16),
        "F32" | "float32" => Ok(DType::F32),
        "F64" | "float64" => Ok(DType::F64),
        "U8" | "uint8" => Ok(DType::U8),
        "U32" | "uint32" => Ok(DType::U32),
        "I32" | "int32" => Ok(DType::I32),
        "I64" | "int64" => Ok(DType::I64),
        _ => Err(ModelError::UnsupportedDType(s.to_string())),
    }
}

/// Convert a candle DType to a human-readable string.
pub fn dtype_to_str(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "uint8",
        DType::U32 => "uint32",
        DType::I16 => "int16",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::BF16 => "bfloat16",
        DType::F16 => "float16",
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::F8E4M3 => "float8_e4m3",
        DType::F6E2M3 => "float6_e2m3",
        DType::F6E3M2 => "float6_e3m2",
        DType::F4 => "float4",
        DType::F8E8M0 => "float8_e8m0",
    }
}

// ---------------------------------------------------------------------------
// Tensor creation helpers
// ---------------------------------------------------------------------------

/// Create a tensor filled with zeros.
pub fn zeros(shape: &[usize], dtype: DType, device: &Device) -> ModelResult<Tensor> {
    Tensor::zeros(shape, dtype, device).map_err(ModelError::Candle)
}

/// Create a tensor filled with ones.
pub fn ones(shape: &[usize], dtype: DType, device: &Device) -> ModelResult<Tensor> {
    Tensor::ones(shape, dtype, device).map_err(ModelError::Candle)
}

/// Create a 1-D tensor from a slice.
pub fn from_slice<T: WithDType>(data: &[T], device: &Device) -> ModelResult<Tensor> {
    Tensor::from_slice(data, data.len(), device).map_err(ModelError::Candle)
}

/// Create a tensor from raw bytes with the given shape and dtype.
///
/// The bytes must be in the native byte order for the dtype.
pub fn from_raw_bytes(
    data: &[u8],
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> ModelResult<Tensor> {
    let num_elements: usize = shape.iter().product();
    let expected_bytes = num_elements * dtype_size(dtype);
    if data.len() != expected_bytes {
        return Err(ModelError::ShapeMismatch {
            expected: expected_bytes,
            got: data.len(),
        });
    }

    // Helper macro: try zero-copy cast first, fall back to aligned copy.
    macro_rules! cast_or_copy {
        ($ty:ty, $label:expr) => {{
            if let Some(slice) = bytemuck_cast_slice::<$ty>(data) {
                Tensor::from_slice(slice, Shape::from_dims(shape), &Device::Cpu)
                    .map_err(ModelError::Candle)?
            } else {
                let aligned =
                    copy_to_aligned::<$ty>(data).ok_or(ModelError::ByteCastError($label))?;
                Tensor::from_slice(&aligned, Shape::from_dims(shape), &Device::Cpu)
                    .map_err(ModelError::Candle)?
            }
        }};
    }

    // Build on CPU from the raw bytes then move to target device.
    let cpu_tensor = match dtype {
        DType::F32 => cast_or_copy!(f32, "f32"),
        DType::F16 => cast_or_copy!(half::f16, "f16"),
        DType::BF16 => cast_or_copy!(half::bf16, "bf16"),
        DType::F64 => cast_or_copy!(f64, "f64"),
        DType::U8 => Tensor::from_slice(data, Shape::from_dims(shape), &Device::Cpu)
            .map_err(ModelError::Candle)?,
        DType::U32 => cast_or_copy!(u32, "u32"),
        DType::I64 => cast_or_copy!(i64, "i64"),
        _ => {
            return Err(ModelError::UnsupportedDType(format!(
                "from_raw_bytes does not support {:?}",
                dtype
            )));
        }
    };

    if device.is_cpu() {
        Ok(cpu_tensor)
    } else {
        cpu_tensor.to_device(device).map_err(ModelError::Candle)
    }
}

/// Safe byte-slice casting (like bytemuck but without the dependency).
fn bytemuck_cast_slice<T: Copy>(data: &[u8]) -> Option<&[T]> {
    let elem_size = std::mem::size_of::<T>();
    if elem_size == 0 || !data.len().is_multiple_of(elem_size) {
        return None;
    }
    if !(data.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
        return None;
    }
    let len = data.len() / elem_size;
    // SAFETY: we checked alignment and length.
    Some(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, len) })
}

/// Copy bytes into an aligned Vec<T> when the source data isn't properly aligned.
/// This is needed because safetensors data within a file may not be aligned to the
/// element type's alignment (e.g. u32 tensors at odd offsets).
fn copy_to_aligned<T: Copy>(data: &[u8]) -> Option<Vec<T>> {
    let elem_size = std::mem::size_of::<T>();
    if elem_size == 0 || !data.len().is_multiple_of(elem_size) {
        return None;
    }
    let len = data.len() / elem_size;
    let mut aligned = vec![unsafe { std::mem::zeroed::<T>() }; len];
    // SAFETY: we copy exactly `data.len()` bytes into a properly-aligned buffer.
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), aligned.as_mut_ptr() as *mut u8, data.len());
    }
    Some(aligned)
}

// ---------------------------------------------------------------------------
// Tensor sharding utilities (for tensor parallelism)
// ---------------------------------------------------------------------------

/// Shard a tensor along a given dimension for tensor parallelism.
///
/// Returns the `rank`-th shard when splitting `tensor` into `world_size`
/// equal pieces along `dim`.
pub fn shard_tensor(
    tensor: &Tensor,
    dim: usize,
    rank: usize,
    world_size: usize,
) -> ModelResult<Tensor> {
    let dim_size = tensor.dim(dim).map_err(ModelError::Candle)?;
    if dim_size % world_size != 0 {
        return Err(ModelError::ShardingError {
            dim,
            dim_size,
            world_size,
        });
    }
    let shard_size = dim_size / world_size;
    let start = rank * shard_size;
    tensor
        .narrow(dim, start, shard_size)
        .map_err(ModelError::Candle)
}

/// Information about a tensor's shape and dtype (without data).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorInfo {
    /// Number of elements in this tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.num_elements() * dtype_size(self.dtype)
    }
}

impl fmt::Display for TensorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:?} ({})",
            self.name,
            self.shape,
            dtype_to_str(self.dtype)
        )
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

pub mod error {
    use thiserror::Error;

    pub type ModelResult<T> = Result<T, ModelError>;

    #[derive(Debug, Error)]
    pub enum ModelError {
        #[error("candle error: {0}")]
        Candle(#[from] candle_core::Error),

        #[error("unsupported dtype: {0}")]
        UnsupportedDType(String),

        #[error("shape mismatch: expected {expected} bytes, got {got}")]
        ShapeMismatch { expected: usize, got: usize },

        #[error("byte cast error for type {0}: alignment or size mismatch")]
        ByteCastError(&'static str),

        #[error(
            "sharding error: dim {dim} of size {dim_size} not divisible by world_size {world_size}"
        )]
        ShardingError {
            dim: usize,
            dim_size: usize,
            world_size: usize,
        },

        #[error("weight not found: {0}")]
        WeightNotFound(String),

        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),

        #[error("JSON parse error: {0}")]
        Json(#[from] serde_json::Error),

        #[error("safetensors error: {0}")]
        SafeTensors(String),

        #[error("{0}")]
        Other(String),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(dtype_size(DType::F16), 2);
        assert_eq!(dtype_size(DType::BF16), 2);
        assert_eq!(dtype_size(DType::F32), 4);
        assert_eq!(dtype_size(DType::F64), 8);
        assert_eq!(dtype_size(DType::U8), 1);
        assert_eq!(dtype_size(DType::U32), 4);
        assert_eq!(dtype_size(DType::I64), 8);
    }

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(dtype_from_str("F16").unwrap(), DType::F16);
        assert_eq!(dtype_from_str("float16").unwrap(), DType::F16);
        assert_eq!(dtype_from_str("BF16").unwrap(), DType::BF16);
        assert_eq!(dtype_from_str("bfloat16").unwrap(), DType::BF16);
        assert_eq!(dtype_from_str("F32").unwrap(), DType::F32);
        assert_eq!(dtype_from_str("float32").unwrap(), DType::F32);
        assert!(dtype_from_str("invalid").is_err());
    }

    #[test]
    fn test_dtype_to_str() {
        assert_eq!(dtype_to_str(DType::F16), "float16");
        assert_eq!(dtype_to_str(DType::BF16), "bfloat16");
        assert_eq!(dtype_to_str(DType::F32), "float32");
    }

    #[test]
    fn test_zeros() {
        let t = zeros(&[2, 3], DType::F32, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = ones(&[4], DType::F32, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[4]);
        let data = t.to_vec1::<f32>().unwrap();
        assert!(data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = from_slice(&data, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[4]);
        assert_eq!(t.to_vec1::<f32>().unwrap(), data);
    }

    #[test]
    fn test_from_raw_bytes_f32() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let t = from_raw_bytes(&bytes, &[2, 3], DType::F32, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[2, 3]);
        let flat = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat, data);
    }

    #[test]
    fn test_from_raw_bytes_wrong_size() {
        let bytes = vec![0u8; 10]; // not a valid f32 buffer for shape [2, 3]
        let result = from_raw_bytes(&bytes, &[2, 3], DType::F32, &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_raw_bytes_unaligned_u32() {
        // Simulate unaligned data by embedding u32 bytes at an odd offset in a buffer.
        let values: Vec<u32> = vec![42, 100, 200];
        let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        // Create a buffer with 1 byte padding to force misalignment.
        let mut padded = vec![0u8; raw_bytes.len() + 1];
        padded[1..].copy_from_slice(&raw_bytes);
        let unaligned = &padded[1..]; // data pointer is now misaligned for u32

        let t = from_raw_bytes(unaligned, &[3], DType::U32, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[3]);
        let result = t.to_vec1::<u32>().unwrap();
        assert_eq!(result, values);
    }

    #[test]
    fn test_from_raw_bytes_unaligned_f32() {
        let values: Vec<f32> = vec![1.0, 2.5, 3.75];
        let raw_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut padded = vec![0u8; raw_bytes.len() + 1];
        padded[1..].copy_from_slice(&raw_bytes);
        let unaligned = &padded[1..];

        let t = from_raw_bytes(unaligned, &[3], DType::F32, &Device::Cpu).unwrap();
        let result = t.to_vec1::<f32>().unwrap();
        assert_eq!(result, values);
    }

    #[test]
    fn test_shard_tensor() {
        let t = Tensor::arange(0f32, 12.0, &Device::Cpu)
            .unwrap()
            .reshape(&[3, 4])
            .unwrap();

        // Shard along dim 1 into 2 shards.
        let shard0 = shard_tensor(&t, 1, 0, 2).unwrap();
        let shard1 = shard_tensor(&t, 1, 1, 2).unwrap();
        assert_eq!(shard0.dims(), &[3, 2]);
        assert_eq!(shard1.dims(), &[3, 2]);

        // First shard should be columns 0,1; second should be 2,3.
        let s0 = shard0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(s0, vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0]);
        let s1 = shard1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(s1, vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0]);
    }

    #[test]
    fn test_shard_tensor_not_divisible() {
        let t = Tensor::arange(0f32, 6.0, &Device::Cpu)
            .unwrap()
            .reshape(&[2, 3])
            .unwrap();
        let result = shard_tensor(&t, 1, 0, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_shard_tensor_dim0() {
        let t = Tensor::arange(0f32, 12.0, &Device::Cpu)
            .unwrap()
            .reshape(&[4, 3])
            .unwrap();

        let shard0 = shard_tensor(&t, 0, 0, 2).unwrap();
        let shard1 = shard_tensor(&t, 0, 1, 2).unwrap();
        assert_eq!(shard0.dims(), &[2, 3]);
        assert_eq!(shard1.dims(), &[2, 3]);

        let s0 = shard0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(s0, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_tensor_info() {
        let info = TensorInfo {
            name: "model.layers.0.self_attn.q_proj.weight".to_string(),
            shape: vec![4096, 4096],
            dtype: DType::F16,
        };
        assert_eq!(info.num_elements(), 4096 * 4096);
        assert_eq!(info.size_bytes(), 4096 * 4096 * 2);
        assert!(info.to_string().contains("q_proj.weight"));
    }
}
