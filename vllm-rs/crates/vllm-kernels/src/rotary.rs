// SPDX-License-Identifier: Apache-2.0
//! Rotary embedding kernels.
//!
//! Trait abstraction for rotary position embedding (RoPE) kernels.
//! Port of: `csrc/pos_encoding_kernels.cu`

use candle_core::Tensor;

use crate::error::KernelResult;

/// Rotary embedding kernel interface.
pub trait RotaryKernels: Send + Sync {
    /// Apply rotary embedding to query and key tensors.
    ///
    /// * `positions` — position indices [batch] or [seq_len]
    /// * `query` — query tensor [num_tokens, num_heads * head_dim]
    /// * `key` — key tensor [num_tokens, num_kv_heads * head_dim]
    /// * `cos_sin_cache` — precomputed [max_pos, rotary_dim]
    /// * `is_neox` — whether to use NeoX-style rotation (split in half)
    ///
    /// Returns (rotated_query, rotated_key).
    ///
    /// Port of: `void rotary_embedding(positions, query, key, head_size,
    ///           cos_sin_cache, is_neox)`
    fn rotary_embedding(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
        cos_sin_cache: &Tensor,
        is_neox: bool,
    ) -> KernelResult<(Tensor, Tensor)>;
}

/// CPU implementation of rotary kernels (for testing).
pub struct CpuRotaryKernels;

impl RotaryKernels for CpuRotaryKernels {
    fn rotary_embedding(
        &self,
        positions: &Tensor,
        query: &Tensor,
        key: &Tensor,
        cos_sin_cache: &Tensor,
        _is_neox: bool,
    ) -> KernelResult<(Tensor, Tensor)> {
        // Gather cos/sin for the given positions.
        // cos_sin_cache shape: [max_pos, rotary_dim] where first half is cos, second half is sin.
        let rotary_dim = cos_sin_cache.dim(1)?;
        let half = rotary_dim / 2;

        let gathered = cos_sin_cache.index_select(positions, 0)?; // [num_tokens, rotary_dim]
        let cos = gathered.narrow(1, 0, half)?; // [num_tokens, half]
        let sin = gathered.narrow(1, half, half)?; // [num_tokens, half]

        let q_rot = apply_rotary_1d(query, &cos, &sin, half)?;
        let k_rot = apply_rotary_1d(key, &cos, &sin, half)?;

        Ok((q_rot, k_rot))
    }
}

/// Apply rotary to a flat [num_tokens, dim] tensor.
/// Only rotates the first `2 * half` dimensions, leaving the rest unchanged.
fn apply_rotary_1d(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    half: usize,
) -> KernelResult<Tensor> {
    let dim = x.dim(1)?;
    let rot_dim = 2 * half;

    if rot_dim > dim {
        return Err(crate::error::KernelError::Shape(format!(
            "rotary dim {} > tensor dim {}",
            rot_dim, dim
        )));
    }

    let x_rot = x.narrow(1, 0, rot_dim)?;
    let x1 = x_rot.narrow(1, 0, half)?;
    let x2 = x_rot.narrow(1, half, half)?;

    // Rotate: [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
    let r1 = (x1.broadcast_mul(cos)? - x2.broadcast_mul(sin)?)?;
    let r2 = (x1.broadcast_mul(sin)? + x2.broadcast_mul(cos)?)?;
    let rotated = Tensor::cat(&[&r1, &r2], 1)?;

    if rot_dim < dim {
        let pass_through = x.narrow(1, rot_dim, dim - rot_dim)?;
        let result = Tensor::cat(&[&rotated, &pass_through], 1)?;
        Ok(result)
    } else {
        Ok(rotated)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn make_cos_sin_cache(max_pos: usize, half_dim: usize) -> Tensor {
        // Simple cache: cos and sin for frequencies
        let rotary_dim = 2 * half_dim;
        let mut data = vec![0.0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half_dim {
                let freq = 1.0 / 10000f64.powf(2.0 * i as f64 / (2 * half_dim) as f64);
                let angle = pos as f64 * freq;
                data[pos * rotary_dim + i] = angle.cos() as f32;
                data[pos * rotary_dim + half_dim + i] = angle.sin() as f32;
            }
        }
        Tensor::from_slice(&data, (max_pos, rotary_dim), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_cpu_rotary_position_zero() {
        let kernels = CpuRotaryKernels;
        let cache = make_cos_sin_cache(10, 4); // rotary_dim = 8
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();
        let q = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], &Device::Cpu).unwrap();
        let k = q.clone();

        let (q_rot, _k_rot) = kernels.rotary_embedding(&positions, &q, &k, &cache, true).unwrap();
        assert_eq!(q_rot.dims(), &[1, 8]);

        // At position 0, cos=1, sin=0 -> output = input
        let vals = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, v) in vals.iter().enumerate() {
            assert!(
                (v - (i as f32 + 1.0)).abs() < 1e-4,
                "pos 0 should be identity, got {} at idx {}",
                v,
                i
            );
        }
    }

    #[test]
    fn test_cpu_rotary_shape_preservation() {
        let kernels = CpuRotaryKernels;
        let cache = make_cos_sin_cache(100, 4);
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();
        let q = Tensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[4, 8], DType::F32, &Device::Cpu).unwrap();

        let (q_rot, k_rot) = kernels.rotary_embedding(&positions, &q, &k, &cache, true).unwrap();
        assert_eq!(q_rot.dims(), &[4, 8]);
        assert_eq!(k_rot.dims(), &[4, 8]);
    }

    #[test]
    fn test_cpu_rotary_partial_dim() {
        // Test when rotary_dim < total dim (pass-through for remaining dims)
        let kernels = CpuRotaryKernels;
        let cache = make_cos_sin_cache(10, 2); // rotary_dim = 4, but tensor dim = 8
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();
        let q = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], &Device::Cpu).unwrap();
        let k = q.clone();

        let (q_rot, _) = kernels.rotary_embedding(&positions, &q, &k, &cache, true).unwrap();
        assert_eq!(q_rot.dims(), &[1, 8]);

        // Last 4 dims should be unchanged (pass-through).
        let vals = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[4] - 5.0).abs() < 1e-4);
        assert!((vals[5] - 6.0).abs() < 1e-4);
        assert!((vals[6] - 7.0).abs() < 1e-4);
        assert!((vals[7] - 8.0).abs() < 1e-4);
    }
}
