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
    /// Combined cos|sin cache for CUDA fused kernel: [max_position, head_dim]
    /// Layout per row: [cos(f0)..cos(f_{half-1}), sin(f0)..sin(f_{half-1})]
    cos_sin_cache: Tensor,
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

        // Build combined cos|sin cache for CUDA fused kernel:
        // [max_pos, head_dim] where first half cols = cos, second half = sin.
        // Force contiguous to ensure the CUDA kernel can access it with simple
        // pointer arithmetic (pos * rotary_dim).
        let cos_sin_cache = Tensor::cat(
            &[
                &cos_cache
                    .narrow(1, 0, half_dim)
                    .map_err(ModelError::Candle)?,
                &sin_cache
                    .narrow(1, 0, half_dim)
                    .map_err(ModelError::Candle)?,
            ],
            1,
        )
        .map_err(ModelError::Candle)?
        .contiguous()
        .map_err(ModelError::Candle)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            cos_sin_cache,
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

    /// Access the cos cache: `[max_position, head_dim]`.
    pub fn cos_cache(&self) -> &Tensor {
        &self.cos_cache
    }

    /// Combined cos|sin cache for CUDA fused kernel: `[max_position, head_dim]`.
    ///
    /// Layout per row: `[cos(f0)..cos(f_{half-1}), sin(f0)..sin(f_{half-1})]`.
    pub fn cos_sin_cache(&self) -> &Tensor {
        &self.cos_sin_cache
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

        let cos_sin_cache = Tensor::cat(
            &[
                &cos_cache
                    .narrow(1, 0, half_dim)
                    .map_err(ModelError::Candle)?,
                &sin_cache
                    .narrow(1, 0, half_dim)
                    .map_err(ModelError::Candle)?,
            ],
            1,
        )
        .map_err(ModelError::Candle)?
        .contiguous()
        .map_err(ModelError::Candle)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            cos_sin_cache,
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

    /// Apply M-RoPE (Multi-dimensional Rotary Position Embedding) used by Qwen2-VL.
    ///
    /// M-RoPE splits `head_dim` into 3 sections (temporal, height, width) and
    /// applies independent RoPE per section using separate position sequences.
    ///
    /// * `q` — query tensor `[seq_len, num_heads, head_dim]`
    /// * `k` — key tensor `[seq_len, num_kv_heads, head_dim]`
    /// * `positions_3d` — position indices `[3, seq_len]` (time, height, width)
    /// * `sections` — dimension sections `[s0, s1, s2]` where `s0+s1+s2 = head_dim/2`
    pub fn apply_mrope(
        &self,
        q: &Tensor,
        k: &Tensor,
        positions_3d: &Tensor,
        sections: &[usize; 3],
    ) -> ModelResult<(Tensor, Tensor)> {
        let half_dim = self.head_dim / 2;
        debug_assert_eq!(
            sections[0] + sections[1] + sections[2],
            half_dim,
            "M-RoPE sections must sum to head_dim/2"
        );

        // Gather cos/sin for each of the 3 position dimensions.
        // positions_3d[i] → [seq_len], gather → [seq_len, head_dim]
        let pos_t = positions_3d
            .narrow(0, 0, 1)
            .map_err(ModelError::Candle)?
            .squeeze(0)
            .map_err(ModelError::Candle)?;
        let pos_h = positions_3d
            .narrow(0, 1, 1)
            .map_err(ModelError::Candle)?
            .squeeze(0)
            .map_err(ModelError::Candle)?;
        let pos_w = positions_3d
            .narrow(0, 2, 1)
            .map_err(ModelError::Candle)?
            .squeeze(0)
            .map_err(ModelError::Candle)?;

        let cos_t = self
            .cos_cache
            .index_select(&pos_t, 0)
            .map_err(ModelError::Candle)?;
        let sin_t = self
            .sin_cache
            .index_select(&pos_t, 0)
            .map_err(ModelError::Candle)?;
        let cos_h = self
            .cos_cache
            .index_select(&pos_h, 0)
            .map_err(ModelError::Candle)?;
        let sin_h = self
            .sin_cache
            .index_select(&pos_h, 0)
            .map_err(ModelError::Candle)?;
        let cos_w = self
            .cos_cache
            .index_select(&pos_w, 0)
            .map_err(ModelError::Candle)?;
        let sin_w = self
            .sin_cache
            .index_select(&pos_w, 0)
            .map_err(ModelError::Candle)?;

        // Build per-section cos/sin by narrowing the head_dim dimension.
        // cos_cache has shape [seq_len, head_dim] where head_dim = 2*half_dim,
        // laid out as [freq0, freq1, ..., freq_{hd/2-1}, freq0, freq1, ...].
        // Section i uses dimensions [offset..offset+sec_i] and [half_dim+offset..half_dim+offset+sec_i].
        let s0 = sections[0];
        let s1 = sections[1];
        let s2 = sections[2];

        // Narrow each cos/sin to its section, then concatenate to form full head_dim cos/sin.
        // First half: [s0 from t, s1 from h, s2 from w]
        // Second half: [s0 from t, s1 from h, s2 from w] (same pattern, offset by half_dim)
        let cos_first = Tensor::cat(
            &[
                &cos_t.narrow(1, 0, s0).map_err(ModelError::Candle)?,
                &cos_h.narrow(1, s0, s1).map_err(ModelError::Candle)?,
                &cos_w.narrow(1, s0 + s1, s2).map_err(ModelError::Candle)?,
            ],
            1,
        )
        .map_err(ModelError::Candle)?;
        let cos_second = Tensor::cat(
            &[
                &cos_t.narrow(1, half_dim, s0).map_err(ModelError::Candle)?,
                &cos_h
                    .narrow(1, half_dim + s0, s1)
                    .map_err(ModelError::Candle)?,
                &cos_w
                    .narrow(1, half_dim + s0 + s1, s2)
                    .map_err(ModelError::Candle)?,
            ],
            1,
        )
        .map_err(ModelError::Candle)?;
        let cos = Tensor::cat(&[&cos_first, &cos_second], 1).map_err(ModelError::Candle)?;

        let sin_first = Tensor::cat(
            &[
                &sin_t.narrow(1, 0, s0).map_err(ModelError::Candle)?,
                &sin_h.narrow(1, s0, s1).map_err(ModelError::Candle)?,
                &sin_w.narrow(1, s0 + s1, s2).map_err(ModelError::Candle)?,
            ],
            1,
        )
        .map_err(ModelError::Candle)?;
        let sin_second = Tensor::cat(
            &[
                &sin_t.narrow(1, half_dim, s0).map_err(ModelError::Candle)?,
                &sin_h
                    .narrow(1, half_dim + s0, s1)
                    .map_err(ModelError::Candle)?,
                &sin_w
                    .narrow(1, half_dim + s0 + s1, s2)
                    .map_err(ModelError::Candle)?,
            ],
            1,
        )
        .map_err(ModelError::Candle)?;
        let sin = Tensor::cat(&[&sin_first, &sin_second], 1).map_err(ModelError::Candle)?;

        let q_rot = apply_rotary_to_tensor(q, &cos, &sin)?;
        let k_rot = apply_rotary_to_tensor(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
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
    fn test_mrope_shape_preservation() {
        let rope = RotaryEmbedding::new(8, 100, 10000.0, DType::F32, &Device::Cpu).unwrap();

        let q = Tensor::ones(&[4, 2, 8], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[4, 2, 8], DType::F32, &Device::Cpu).unwrap();
        // 3 position dims, each with 4 positions
        let positions_3d =
            Tensor::new(&[[0u32, 1, 2, 3], [0, 0, 1, 1], [0, 1, 0, 1]], &Device::Cpu).unwrap();
        let sections = [2, 1, 1]; // sum = 4 = head_dim/2

        let (q_rot, k_rot) = rope.apply_mrope(&q, &k, &positions_3d, &sections).unwrap();
        assert_eq!(q_rot.dims(), &[4, 2, 8]);
        assert_eq!(k_rot.dims(), &[4, 2, 8]);
    }

    #[test]
    fn test_mrope_position_zero_is_identity() {
        // When all 3 position dimensions are 0, M-RoPE should be identity (like standard RoPE).
        let rope = RotaryEmbedding::new(8, 100, 10000.0, DType::F32, &Device::Cpu).unwrap();

        let q = Tensor::new(
            &[[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]],
            &Device::Cpu,
        )
        .unwrap();
        let k = q.clone();
        let positions_3d = Tensor::new(&[[0u32], [0], [0]], &Device::Cpu).unwrap();
        let sections = [2, 1, 1];

        let (q_rot, _) = rope.apply_mrope(&q, &k, &positions_3d, &sections).unwrap();
        let vals = q_rot.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            let expected = (i + 1) as f32;
            assert!(
                (v - expected).abs() < 1e-5,
                "M-RoPE at pos 0 should be identity, dim {i}: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn test_mrope_matches_standard_when_positions_equal() {
        // When all 3 position dims have the same values, M-RoPE should match standard RoPE.
        let rope = RotaryEmbedding::new(8, 100, 10000.0, DType::F32, &Device::Cpu).unwrap();

        let q = Tensor::ones(&[3, 2, 8], DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones(&[3, 2, 8], DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[5u32, 10, 15], &Device::Cpu).unwrap();
        let positions_3d =
            Tensor::new(&[[5u32, 10, 15], [5, 10, 15], [5, 10, 15]], &Device::Cpu).unwrap();
        let sections = [2, 1, 1];

        let (q_standard, _) = rope.apply(&q, &k, &positions).unwrap();
        let (q_mrope, _) = rope.apply_mrope(&q, &k, &positions_3d, &sections).unwrap();

        let std_vals = q_standard.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mrope_vals = q_mrope.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (s, m)) in std_vals.iter().zip(mrope_vals.iter()).enumerate() {
            assert!(
                (s - m).abs() < 1e-4,
                "dim {i}: standard={s}, mrope={m} should match when positions are equal"
            );
        }
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
