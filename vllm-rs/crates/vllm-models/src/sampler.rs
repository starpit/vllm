// SPDX-License-Identifier: Apache-2.0
//! Sampler for converting logits to token IDs.
//!
//! Supports greedy decoding (argmax) and random sampling with temperature,
//! top-k, and top-p.
//!
//! Port of: `vllm/v1/sample/sampler.py`

use candle_core::{DType, Tensor};
use rand::Rng;

use vllm_model::ModelResult;
use vllm_model::error::ModelError;

/// Sampler output for a single request.
#[derive(Debug, Clone)]
pub struct SamplerOutput {
    /// Sampled token IDs, one per request.
    pub token_ids: Vec<u32>,
}

/// Sampler that converts logits to sampled token IDs.
pub struct Sampler {
    /// Random number generator for non-greedy sampling.
    rng: rand::rngs::ThreadRng,
}

impl Sampler {
    /// Create a new sampler.
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    /// Greedy decode: take the argmax of logits for each row.
    ///
    /// `logits` shape: `[num_tokens, vocab_size]`
    ///
    /// Returns token IDs for each row.
    pub fn greedy(&self, logits: &Tensor) -> ModelResult<Vec<u32>> {
        let indices = logits
            .argmax(candle_core::D::Minus1)
            .map_err(ModelError::Candle)?;
        let ids = indices.to_vec1::<u32>().map_err(ModelError::Candle)?;
        Ok(ids)
    }

    /// Sample with temperature.
    ///
    /// `logits` shape: `[num_tokens, vocab_size]`
    /// `temperature` — temperature for softmax (0 = greedy, >0 = random)
    ///
    /// Returns one sampled token ID per row.
    pub fn sample(&mut self, logits: &Tensor, temperature: f32) -> ModelResult<Vec<u32>> {
        if temperature < 1e-5 {
            return self.greedy(logits);
        }

        let logits_f32 = logits.to_dtype(DType::F32).map_err(ModelError::Candle)?;
        let scaled = (&logits_f32 / temperature as f64).map_err(ModelError::Candle)?;

        // Softmax to get probabilities.
        let max_vals = scaled
            .max_keepdim(candle_core::D::Minus1)
            .map_err(ModelError::Candle)?;
        let shifted = scaled
            .broadcast_sub(&max_vals)
            .map_err(ModelError::Candle)?;
        let exp = shifted.exp().map_err(ModelError::Candle)?;
        let sum = exp
            .sum_keepdim(candle_core::D::Minus1)
            .map_err(ModelError::Candle)?;
        let probs = exp.broadcast_div(&sum).map_err(ModelError::Candle)?;

        let probs_2d = probs.to_vec2::<f32>().map_err(ModelError::Candle)?;

        let mut token_ids = Vec::with_capacity(probs_2d.len());
        for row in &probs_2d {
            let token_id = self.sample_from_probs(row);
            token_ids.push(token_id);
        }

        Ok(token_ids)
    }

    /// Sample with temperature, top-k, and top-p.
    pub fn sample_top_k_top_p(
        &mut self,
        logits: &Tensor,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> ModelResult<Vec<u32>> {
        if temperature < 1e-5 {
            return self.greedy(logits);
        }

        let logits_f32 = logits.to_dtype(DType::F32).map_err(ModelError::Candle)?;
        let scaled = (&logits_f32 / temperature as f64).map_err(ModelError::Candle)?;
        let logits_2d = scaled.to_vec2::<f32>().map_err(ModelError::Candle)?;

        let mut token_ids = Vec::with_capacity(logits_2d.len());
        for row in &logits_2d {
            let token_id = self.sample_with_filters(row, top_k, top_p);
            token_ids.push(token_id);
        }

        Ok(token_ids)
    }

    /// Sample a single token from a probability distribution.
    fn sample_from_probs(&mut self, probs: &[f32]) -> u32 {
        let r: f32 = self.rng.r#gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum > r {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }

    /// Sample with top-k and top-p filtering applied to logits.
    fn sample_with_filters(&mut self, logits: &[f32], top_k: usize, top_p: f32) -> u32 {
        // Create (index, logit) pairs and sort by logit descending.
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply top-k: keep only the top-k entries.
        let k = if top_k > 0 && top_k < indexed.len() {
            top_k
        } else {
            indexed.len()
        };
        indexed.truncate(k);

        // Softmax on the remaining entries (in-place).
        let max_logit = indexed[0].1;
        for entry in &mut indexed {
            entry.1 = (entry.1 - max_logit).exp();
        }
        let sum: f32 = indexed.iter().map(|(_, p)| p).sum();
        for entry in &mut indexed {
            entry.1 /= sum;
        }

        // Apply top-p: keep tokens until cumulative probability exceeds top_p.
        if top_p > 0.0 && top_p < 1.0 {
            let mut cumsum = 0.0;
            let mut cutoff = indexed.len();
            for (i, &(_, p)) in indexed.iter().enumerate() {
                cumsum += p;
                if cumsum > top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);

            // Re-normalize.
            let new_sum: f32 = indexed.iter().map(|(_, p)| p).sum();
            for entry in &mut indexed {
                entry.1 /= new_sum;
            }
        }

        // Sample from the filtered distribution.
        let r: f32 = self.rng.r#gen();
        let mut cumsum = 0.0;
        for &(idx, p) in &indexed {
            cumsum += p;
            if cumsum > r {
                return idx as u32;
            }
        }
        indexed.last().map(|&(idx, _)| idx as u32).unwrap_or(0)
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_greedy_sampling() {
        let sampler = Sampler::new();

        // Logits where the max is at different positions for each row.
        let logits = Tensor::new(
            &[[0.1f32, 0.9, 0.0], [0.0, 0.1, 0.9], [0.9, 0.1, 0.0]],
            &Device::Cpu,
        )
        .unwrap();

        let ids = sampler.greedy(&logits).unwrap();
        assert_eq!(ids, vec![1, 2, 0]);
    }

    #[test]
    fn test_greedy_single_row() {
        let sampler = Sampler::new();
        let logits = Tensor::new(&[[1.0f32, 5.0, 3.0, 2.0]], &Device::Cpu).unwrap();
        let ids = sampler.greedy(&logits).unwrap();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn test_temperature_zero_is_greedy() {
        let mut sampler = Sampler::new();
        let logits = Tensor::new(&[[0.1f32, 0.9, 0.0], [0.0, 0.1, 0.9]], &Device::Cpu).unwrap();

        let ids = sampler.sample(&logits, 0.0).unwrap();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_temperature_sampling() {
        let mut sampler = Sampler::new();

        // High-confidence logits: one value much larger than others.
        let logits = Tensor::new(&[[0.0f32, 100.0, 0.0, 0.0]], &Device::Cpu).unwrap();

        // Even with temperature=1.0, the dominant logit should almost always win.
        let ids = sampler.sample(&logits, 1.0).unwrap();
        assert_eq!(ids[0], 1);
    }

    #[test]
    fn test_top_k_sampling() {
        let mut sampler = Sampler::new();

        // Logits with a clear winner.
        let logits = Tensor::new(&[[0.0f32, 100.0, 0.0, 0.0]], &Device::Cpu).unwrap();

        let ids = sampler.sample_top_k_top_p(&logits, 1.0, 2, 1.0).unwrap();
        assert_eq!(ids[0], 1);
    }

    #[test]
    fn test_top_p_sampling() {
        let mut sampler = Sampler::new();

        // Logits with a clear winner.
        let logits = Tensor::new(&[[0.0f32, 100.0, 0.0, 0.0]], &Device::Cpu).unwrap();

        let ids = sampler.sample_top_k_top_p(&logits, 1.0, 0, 0.9).unwrap();
        assert_eq!(ids[0], 1);
    }

    #[test]
    fn test_sampler_output_shape() {
        let mut sampler = Sampler::new();
        let logits = Tensor::ones(&[5, 100], DType::F32, &Device::Cpu).unwrap();

        let ids = sampler.sample(&logits, 1.0).unwrap();
        assert_eq!(ids.len(), 5);
        // All token IDs should be valid vocab indices.
        for &id in &ids {
            assert!((id as usize) < 100);
        }
    }
}
