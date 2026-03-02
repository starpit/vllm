// SPDX-License-Identifier: Apache-2.0
//! Rotary Position Embedding (RoPE).
//!
//! Port of: `vllm/model_executor/layers/rotary_embedding/`
//!
//! RoPE applies a rotation to query and key vectors based on their position
//! in the sequence. This enables the model to learn relative position
//! relationships.
//!
//! Formula for each pair of dimensions (2i, 2i+1):
//!   q_rot[2i]   = q[2i] * cos(pos * theta_i) - q[2i+1] * sin(pos * theta_i)
//!   q_rot[2i+1] = q[2i] * sin(pos * theta_i) + q[2i+1] * cos(pos * theta_i)
//! where theta_i = 1 / base^(2i/dim)

use candle_core::{DType, Device, Tensor};

use crate::error::{ModelError, ModelResult};

// ---------------------------------------------------------------------------
// RotaryEmbedding
// ---------------------------------------------------------------------------

/// Rotary Position Embedding (RoPE).
///
/// Precomputes cos/sin tables up to `max_position` and applies rotation
/// to query/key tensors.
pub struct RotaryEmbedding {
    /// Precomputed cosine values: [max_position, head_dim]
    cos_cache: Tensor,
    /// Precomputed sine values: [max_position, head_dim]
    sin_cache: Tensor,
    /// Head dimension.
    head_dim: usize,
    /// Maximum cached position.
    max_position: usize,
}

impl RotaryEmbedding {
    /// Create a new RoPE with the given parameters.
    ///
    /// * `head_dim` — dimension per attention head (must be even)
    /// * `max_position` — max sequence length to cache cos/sin for
    /// * `base` — base for the frequency computation (typically 10000.0)
    /// * `dtype` — compute dtype for cos/sin
    /// * `device` — target device
    pub fn new(
        head_dim: usize,
        max_position: usize,
        base: f64,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        if !head_dim.is_multiple_of(2) {
            return Err(ModelError::Other(format!(
                "RoPE head_dim must be even, got {}",
                head_dim
            )));
        }

        // Compute inverse frequency: theta_i = 1 / base^(2i/head_dim)
        // Use f32 on-device (Metal doesn't support f64 matmul). The f64
        // powf is done on the CPU scalar side for precision, then stored as f32.
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (1.0 / base.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();

        let inv_freq_tensor =
            Tensor::from_slice(&inv_freq, half_dim, device).map_err(ModelError::Candle)?;

        // Position indices: [0, 1, 2, ..., max_position-1]
        let positions: Vec<f32> = (0..max_position).map(|p| p as f32).collect();
        let pos_tensor =
            Tensor::from_slice(&positions, max_position, device).map_err(ModelError::Candle)?;

        // Outer product: freqs[pos, i] = pos * inv_freq[i]
        // pos_tensor: [max_position] -> [max_position, 1]
        // inv_freq:   [half_dim]     -> [1, half_dim]
        let pos_2d = pos_tensor
            .reshape((max_position, 1))
            .map_err(ModelError::Candle)?;
        let inv_freq_2d = inv_freq_tensor
            .reshape((1, half_dim))
            .map_err(ModelError::Candle)?;
        let freqs = pos_2d.matmul(&inv_freq_2d).map_err(ModelError::Candle)?; // [max_position, half_dim]

        // Duplicate freqs for full head_dim: [max_position, head_dim]
        let freqs_full = Tensor::cat(&[&freqs, &freqs], 1).map_err(ModelError::Candle)?;

        // Compute cos and sin, then cast to target dtype.
        let cos_cache = freqs_full
            .cos()
            .map_err(ModelError::Candle)?
            .to_dtype(dtype)
            .map_err(ModelError::Candle)?;
        let sin_cache = freqs_full
            .sin()
            .map_err(ModelError::Candle)?
            .to_dtype(dtype)
            .map_err(ModelError::Candle)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
            max_position,
        })
    }

    /// Apply RoPE to query and key tensors.
    ///
    /// * `q` — query tensor of shape `[batch, seq_len, num_heads, head_dim]`
    ///   or `[seq_len, num_heads, head_dim]`
    /// * `k` — key tensor of same shape as q (possibly different num_heads for GQA)
    /// * `positions` — position indices tensor of shape `[batch, seq_len]` or `[seq_len]`
    ///
    /// Returns (q_rotated, k_rotated).
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions: &Tensor,
    ) -> ModelResult<(Tensor, Tensor)> {
        // Gather cos/sin for the given positions.
        let cos = self
            .cos_cache
            .index_select(positions, 0)
            .map_err(ModelError::Candle)?; // [seq_len, head_dim]
        let sin = self
            .sin_cache
            .index_select(positions, 0)
            .map_err(ModelError::Candle)?;

        let q_rot = apply_rotary_to_tensor(q, &cos, &sin)?;
        let k_rot = apply_rotary_to_tensor(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Maximum cached position.
    pub fn max_position(&self) -> usize {
        self.max_position
    }

    /// Access the cos cache.
    pub fn cos_cache(&self) -> &Tensor {
        &self.cos_cache
    }

    /// Create a YaRN-scaled RoPE (used by DeepSeek V2/V3).
    ///
    /// YaRN applies frequency-dependent corrections: low-frequency dimensions
    /// are scaled by the factor, high-frequency dimensions are kept, and
    /// middle-range dimensions are interpolated.
    ///
    /// * `head_dim` — dimension per attention head (must be even)
    /// * `max_position` — max sequence length to cache cos/sin for
    /// * `base` — base for the frequency computation (typically 10000.0)
    /// * `scaling_factor` — NTK scaling factor
    /// * `beta_fast` — boundary for high-frequency (unscaled) dimensions
    /// * `beta_slow` — boundary for low-frequency (fully scaled) dimensions
    /// * `mscale` — magnitude scaling (mscale_all_dim param from config)
    /// * `original_max_pos` — original max_position_embeddings before scaling
    /// * `dtype` — compute dtype for cos/sin
    /// * `device` — target device
    #[allow(clippy::too_many_arguments)]
    pub fn new_yarn(
        head_dim: usize,
        max_position: usize,
        base: f64,
        scaling_factor: f64,
        beta_fast: f64,
        beta_slow: f64,
        mscale_all_dim: f64,
        original_max_pos: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        if !head_dim.is_multiple_of(2) {
            return Err(ModelError::Other(format!(
                "RoPE head_dim must be even, got {}",
                head_dim
            )));
        }

        let half_dim = head_dim / 2;

        // Standard inverse frequencies.
        let inv_freq_base: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f64 / head_dim as f64))
            .collect();

        // Compute YaRN frequency corrections.
        // low/high dimension boundaries for interpolation.
        let low = (original_max_pos as f64 / (beta_fast / 2.0 * std::f64::consts::PI))
            .floor()
            .max(1.0);
        let high = (original_max_pos as f64 / (beta_slow / 2.0 * std::f64::consts::PI))
            .ceil()
            .max(1.0);

        let inv_freq: Vec<f32> = inv_freq_base
            .iter()
            .enumerate()
            .map(|(i, &freq)| {
                // Wavelength of this frequency dimension.
                let wavelength = 2.0 * std::f64::consts::PI / freq;
                // Normalized position in the dim range [0, 1].
                let dim_pos = 2.0 * i as f64 / head_dim as f64;
                let _ = dim_pos;
                // Ramp function: 0 for low dims (high freq), 1 for high dims (low freq).
                let ramp = if high == low {
                    0.0
                } else {
                    let r = (wavelength - low) / (high - low);
                    r.clamp(0.0, 1.0)
                };
                // Interpolate: ramp=0 → keep original, ramp=1 → scale by factor.
                let scaled_freq = freq / scaling_factor;
                let corrected = (1.0 - ramp) * freq + ramp * scaled_freq;
                corrected as f32
            })
            .collect();

        let inv_freq_tensor =
            Tensor::from_slice(&inv_freq, half_dim, device).map_err(ModelError::Candle)?;

        // Position indices.
        let positions: Vec<f32> = (0..max_position).map(|p| p as f32).collect();
        let pos_tensor =
            Tensor::from_slice(&positions, max_position, device).map_err(ModelError::Candle)?;

        let pos_2d = pos_tensor
            .reshape((max_position, 1))
            .map_err(ModelError::Candle)?;
        let inv_freq_2d = inv_freq_tensor
            .reshape((1, half_dim))
            .map_err(ModelError::Candle)?;
        let freqs = pos_2d.matmul(&inv_freq_2d).map_err(ModelError::Candle)?;

        let freqs_full = Tensor::cat(&[&freqs, &freqs], 1).map_err(ModelError::Candle)?;

        // YaRN mscale: magnitude correction for attention scaling.
        let mscale = if mscale_all_dim > 0.0 {
            let m = yarn_get_mscale(scaling_factor, mscale_all_dim);
            m as f32
        } else {
            1.0f32
        };

        let cos_cache = (freqs_full.cos().map_err(ModelError::Candle)? * mscale as f64)
            .map_err(ModelError::Candle)?
            .to_dtype(dtype)
            .map_err(ModelError::Candle)?;
        let sin_cache = (freqs_full.sin().map_err(ModelError::Candle)? * mscale as f64)
            .map_err(ModelError::Candle)?
            .to_dtype(dtype)
            .map_err(ModelError::Candle)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
            max_position,
        })
    }

    /// Apply RoPE to a single tensor (just Q or just K).
    ///
    /// Useful for MLA where Q and K have different dimensions.
    pub fn apply_one(&self, x: &Tensor, positions: &Tensor) -> ModelResult<Tensor> {
        let cos = self
            .cos_cache
            .index_select(positions, 0)
            .map_err(ModelError::Candle)?;
        let sin = self
            .sin_cache
            .index_select(positions, 0)
            .map_err(ModelError::Candle)?;

        apply_rotary_to_tensor(x, &cos, &sin)
    }

    /// Access the sin cache.
    pub fn sin_cache(&self) -> &Tensor {
        &self.sin_cache
    }
}

/// YaRN magnitude scaling function.
///
/// From DeepSeek V2: `mscale = 0.1 * ln(factor) + 1.0` when `mscale_all_dim > 0`.
fn yarn_get_mscale(scaling_factor: f64, mscale_all_dim: f64) -> f64 {
    if scaling_factor <= 1.0 {
        return 1.0;
    }
    0.1 * mscale_all_dim * scaling_factor.ln() + 1.0
}

/// Apply rotary embedding to a single tensor.
///
/// `x` shape: `[seq_len, num_heads, head_dim]`
/// `cos`/`sin` shape: `[seq_len, head_dim]`
pub fn apply_rotary_to_tensor(x: &Tensor, cos: &Tensor, sin: &Tensor) -> ModelResult<Tensor> {
    let half_dim = x.dim(candle_core::D::Minus1).map_err(ModelError::Candle)? / 2;

    // Split x into first half and second half along last dim.
    let x1 = x
        .narrow(candle_core::D::Minus1, 0, half_dim)
        .map_err(ModelError::Candle)?;
    let x2 = x
        .narrow(candle_core::D::Minus1, half_dim, half_dim)
        .map_err(ModelError::Candle)?;

    // Rotate: [-x2, x1] (the "rotated" version)
    let neg_x2 = x2.neg().map_err(ModelError::Candle)?;
    let x_rotated =
        Tensor::cat(&[&neg_x2, &x1], candle_core::D::Minus1).map_err(ModelError::Candle)?;

    // Apply: x * cos + x_rotated * sin
    // We need cos/sin to broadcast over the num_heads dimension.
    // cos shape: [seq_len, head_dim] -> need [seq_len, 1, head_dim] for broadcasting
    let ndim = x.dims().len();
    let cos_b = if ndim == 3 {
        cos.unsqueeze(1).map_err(ModelError::Candle)?
    } else {
        cos.clone()
    };
    let sin_b = if ndim == 3 {
        sin.unsqueeze(1).map_err(ModelError::Candle)?
    } else {
        sin.clone()
    };

    let result = (x.broadcast_mul(&cos_b).map_err(ModelError::Candle)?)
        .add(
            &x_rotated
                .broadcast_mul(&sin_b)
                .map_err(ModelError::Candle)?,
        )
        .map_err(ModelError::Candle)?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotary_embedding_creation() {
        let rope = RotaryEmbedding::new(64, 2048, 10000.0, DType::F32, &Device::Cpu).unwrap();
        assert_eq!(rope.head_dim(), 64);
        assert_eq!(rope.max_position(), 2048);
        assert_eq!(rope.cos_cache().dims(), &[2048, 64]);
        assert_eq!(rope.sin_cache().dims(), &[2048, 64]);
    }

    #[test]
    fn test_rotary_embedding_odd_dim_rejected() {
        let result = RotaryEmbedding::new(63, 100, 10000.0, DType::F32, &Device::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_rotary_embedding_position_zero() {
        // At position 0, cos should be 1 and sin should be 0.
        // So RoPE at position 0 should be identity.
        let rope = RotaryEmbedding::new(4, 10, 10000.0, DType::F32, &Device::Cpu).unwrap();

        // q = [1, 2, 3, 4] at position 0
        let q = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0]]], &Device::Cpu).unwrap();
        // shape: [1, 1, 4] = [seq_len, num_heads, head_dim]
        let k = q.clone();
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, &positions).unwrap();
        assert_eq!(q_rot.dims(), &[1, 1, 4]);

        // At position 0, all freqs are 0, so cos=1, sin=0 -> output = input
        let q_vals = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((q_vals[0] - 1.0).abs() < 1e-5);
        assert!((q_vals[1] - 2.0).abs() < 1e-5);
        assert!((q_vals[2] - 3.0).abs() < 1e-5);
        assert!((q_vals[3] - 4.0).abs() < 1e-5);

        let k_vals = k_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((k_vals[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotary_embedding_shape_preservation() {
        let rope = RotaryEmbedding::new(8, 100, 10000.0, DType::F32, &Device::Cpu).unwrap();

        // [seq_len=4, num_heads=2, head_dim=8]
        let q = Tensor::ones(&[4, 2, 8], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[4, 2, 8], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, &positions).unwrap();
        assert_eq!(q_rot.dims(), &[4, 2, 8]);
        assert_eq!(k_rot.dims(), &[4, 2, 8]);
    }

    #[test]
    fn test_yarn_rope_creation() {
        let rope = RotaryEmbedding::new_yarn(
            64,   // head_dim
            4096, // max_position
            10000.0,
            40.0, // scaling_factor
            32.0, // beta_fast
            1.0,  // beta_slow
            0.1,  // mscale_all_dim
            4096, // original_max_pos
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        assert_eq!(rope.head_dim(), 64);
        assert_eq!(rope.max_position(), 4096);
        assert_eq!(rope.cos_cache().dims(), &[4096, 64]);
        assert_eq!(rope.sin_cache().dims(), &[4096, 64]);
    }

    #[test]
    fn test_yarn_rope_differs_from_standard() {
        let standard = RotaryEmbedding::new(64, 1024, 10000.0, DType::F32, &Device::Cpu).unwrap();
        let yarn = RotaryEmbedding::new_yarn(
            64,
            1024,
            10000.0,
            40.0,
            32.0,
            1.0,
            0.1,
            4096,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        // At position 1, the cos values should differ due to different frequencies.
        let std_cos = standard
            .cos_cache()
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let yarn_cos = yarn
            .cos_cache()
            .narrow(0, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // They should not be identical (YaRN modifies frequencies + applies mscale).
        let mut any_different = false;
        for (a, b) in std_cos.iter().zip(yarn_cos.iter()) {
            if (a - b).abs() > 1e-6 {
                any_different = true;
                break;
            }
        }
        assert!(
            any_different,
            "YaRN RoPE should produce different cos values than standard RoPE"
        );
    }

    #[test]
    fn test_yarn_rope_apply() {
        let rope = RotaryEmbedding::new_yarn(
            8,
            128,
            10000.0,
            4.0,
            32.0,
            1.0,
            0.1,
            128,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();

        let q = Tensor::ones(&[4, 2, 8], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[4, 2, 8], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, &positions).unwrap();
        assert_eq!(q_rot.dims(), &[4, 2, 8]);
        assert_eq!(k_rot.dims(), &[4, 2, 8]);
    }

    #[test]
    fn test_apply_one() {
        let rope = RotaryEmbedding::new(8, 128, 10000.0, DType::F32, &Device::Cpu).unwrap();

        let x = Tensor::ones(&[3, 2, 8], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2], &Device::Cpu).unwrap();

        let x_rot = rope.apply_one(&x, &positions).unwrap();
        assert_eq!(x_rot.dims(), &[3, 2, 8]);
    }

    #[test]
    fn test_rotary_embedding_cos_sin_at_zero() {
        let rope = RotaryEmbedding::new(4, 10, 10000.0, DType::F32, &Device::Cpu).unwrap();

        // cos(0) should be 1 for all frequencies
        let cos_row0 = rope
            .cos_cache()
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for v in &cos_row0 {
            assert!((v - 1.0).abs() < 1e-5, "cos(0) should be 1, got {}", v);
        }

        // sin(0) should be 0 for all frequencies
        let sin_row0 = rope
            .sin_cache()
            .narrow(0, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for v in &sin_row0 {
            assert!(v.abs() < 1e-5, "sin(0) should be 0, got {}", v);
        }
    }
}
