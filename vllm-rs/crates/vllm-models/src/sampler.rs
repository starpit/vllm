// SPDX-License-Identifier: Apache-2.0
//! Sampler for converting logits to token IDs.
//!
//! Supports greedy decoding (argmax) and random sampling with temperature,
//! top-k, and top-p.
//!
//! Port of: `vllm/v1/sample/sampler.py`

use std::collections::HashMap;

use candle_core::{DType, Tensor};
use rand::Rng;

use vllm_common::SamplingParams;
use vllm_common::sampling::{LogprobsOutput, TokenLogprob};
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

    // -------------------------------------------------------------------
    // Unified sample_one entry point
    // -------------------------------------------------------------------

    /// Sample a single token from raw logits, applying all configured
    /// transformations: logit_bias → grammar mask → penalties → temperature →
    /// top-k → top-p → min_p → sample → optional logprobs.
    ///
    /// `logits` — vocab-sized raw logits for one request.
    /// `params` — full sampling parameters.
    /// `prev_tokens` — all previously generated tokens for this request.
    /// `grammar_allowed` — if set, only these token IDs are allowed (constrained decoding).
    ///
    /// Returns `(sampled_token_id, optional_logprobs)`.
    pub fn sample_one(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        prev_tokens: &[u32],
        grammar_allowed: Option<&[u32]>,
    ) -> (u32, Option<LogprobsOutput>) {
        let mut logits_buf: Vec<f32> = logits.to_vec();

        // 1. Apply logit bias.
        if let Some(bias) = &params.logit_bias {
            apply_logit_bias(&mut logits_buf, bias);
        }

        // 1.5. Apply grammar mask (constrained decoding).
        if let Some(allowed) = grammar_allowed {
            crate::grammar::apply_grammar_mask(&mut logits_buf, allowed);
        }

        // 2. Apply repetition/frequency/presence penalties.
        let rep = params.repetition_penalty as f32;
        let freq = params.frequency_penalty as f32;
        let pres = params.presence_penalty as f32;
        if rep != 1.0 || freq != 0.0 || pres != 0.0 {
            apply_penalties(&mut logits_buf, prev_tokens, rep, freq, pres);
        }

        let temperature = params.temperature as f32;
        let top_k = params.top_k.max(0) as usize;
        let top_p = params.top_p as f32;
        let min_p = params.min_p as f32;

        // 3. Greedy path.
        if temperature < 1e-5 {
            let token_id = argmax(&logits_buf);
            let logprobs = params
                .logprobs
                .map(|n| compute_logprobs(&logits_buf, token_id, n as usize));
            return (token_id, logprobs);
        }

        // 4. Temperature scaling.
        for l in &mut logits_buf {
            *l /= temperature;
        }

        // Save a copy for logprobs computation (after temperature scaling).
        let logprobs_requested = params.logprobs;
        let logits_for_logprobs: Option<Vec<f32>> = logprobs_requested.map(|_| logits_buf.clone());

        // 5. Build sorted (index, logit) pairs for top-k/top-p/min_p filtering.
        let mut indexed: Vec<(usize, f32)> = logits_buf.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 6. Top-k.
        if top_k > 0 && top_k < indexed.len() {
            indexed.truncate(top_k);
        }

        // 7. Softmax on remaining entries.
        let max_logit = indexed[0].1;
        for entry in &mut indexed {
            entry.1 = (entry.1 - max_logit).exp();
        }
        let sum: f32 = indexed.iter().map(|(_, p)| p).sum();
        for entry in &mut indexed {
            entry.1 /= sum;
        }

        // 8. Top-p.
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
            let new_sum: f32 = indexed.iter().map(|(_, p)| p).sum();
            for entry in &mut indexed {
                entry.1 /= new_sum;
            }
        }

        // 9. Min-p: remove tokens with prob < min_p * max_prob.
        if min_p > 0.0 {
            apply_min_p(&mut indexed, min_p);
        }

        // 10. Sample from filtered distribution.
        let r: f32 = self.rng.r#gen();
        let mut cumsum = 0.0;
        let mut token_id = indexed.last().map(|&(idx, _)| idx as u32).unwrap_or(0);
        for &(idx, p) in &indexed {
            cumsum += p;
            if cumsum > r {
                token_id = idx as u32;
                break;
            }
        }

        // 11. Compute logprobs if requested.
        let logprobs = logprobs_requested
            .map(|n| compute_logprobs(logits_for_logprobs.as_ref().unwrap(), token_id, n as usize));

        (token_id, logprobs)
    }
}

// ---------------------------------------------------------------------------
// Standalone helper functions
// ---------------------------------------------------------------------------

/// Apply repetition, frequency, and presence penalties to logits.
///
/// Matches Python vLLM `_apply_penalties`:
/// - For each previously seen token, if logit > 0 divide by `repetition_penalty`,
///   else multiply by `repetition_penalty`.
/// - Subtract `frequency_penalty * count(token)` from the logit.
/// - Subtract `presence_penalty` from the logit (if count > 0).
fn apply_penalties(
    logits: &mut [f32],
    token_ids: &[u32],
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
) {
    if token_ids.is_empty() {
        return;
    }

    // Count frequency of each token.
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &tid in token_ids {
        *counts.entry(tid).or_insert(0) += 1;
    }

    for (&tid, &count) in &counts {
        let idx = tid as usize;
        if idx >= logits.len() {
            continue;
        }
        // Repetition penalty: multiplicative.
        if repetition_penalty != 1.0 {
            if logits[idx] > 0.0 {
                logits[idx] /= repetition_penalty;
            } else {
                logits[idx] *= repetition_penalty;
            }
        }
        // Frequency penalty: proportional to count.
        if frequency_penalty != 0.0 {
            logits[idx] -= frequency_penalty * count as f32;
        }
        // Presence penalty: flat penalty for any seen token.
        if presence_penalty != 0.0 {
            logits[idx] -= presence_penalty;
        }
    }
}

/// Apply min_p filtering: remove tokens with prob < min_p * max_prob.
///
/// `indexed` is sorted descending by probability and already softmaxed.
fn apply_min_p(indexed: &mut Vec<(usize, f32)>, min_p: f32) {
    if indexed.is_empty() || min_p <= 0.0 {
        return;
    }
    let max_prob = indexed[0].1;
    let threshold = min_p * max_prob;
    indexed.retain(|&(_, p)| p >= threshold);

    // Re-normalize.
    if indexed.is_empty() {
        return;
    }
    let new_sum: f32 = indexed.iter().map(|(_, p)| p).sum();
    if new_sum > 0.0 {
        for entry in indexed.iter_mut() {
            entry.1 /= new_sum;
        }
    }
}

/// Apply logit bias: add per-token bias to logits.
fn apply_logit_bias(logits: &mut [f32], logit_bias: &HashMap<u32, f32>) {
    for (&tid, &bias) in logit_bias {
        let idx = tid as usize;
        if idx < logits.len() {
            logits[idx] += bias;
        }
    }
}

/// Compute log-probabilities: return the sampled token's logprob and
/// the top-N tokens with logprobs.
///
/// `logits` — temperature-scaled logits (pre-softmax, full vocab).
/// `sampled_token_id` — the token that was sampled.
/// `top_n` — number of top alternatives to return.
fn compute_logprobs(logits: &[f32], sampled_token_id: u32, top_n: usize) -> LogprobsOutput {
    // Compute log-softmax.
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let log_sum_exp = max_logit + log_sum_exp.ln();

    // Build (index, logprob) pairs and sort descending.
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, l - log_sum_exp))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Collect top-N.
    let n = if top_n == 0 {
        0
    } else {
        top_n.min(indexed.len())
    };
    let top_logprobs: Vec<TokenLogprob> = indexed[..n]
        .iter()
        .enumerate()
        .map(|(rank, &(token_id, logprob))| TokenLogprob {
            token_id,
            logprob,
            rank: rank as u32 + 1,
        })
        .collect();

    // Find the sampled token's rank and logprob.
    let sampled_idx = sampled_token_id as usize;
    let sampled_logprob = if sampled_idx < logits.len() {
        logits[sampled_idx] - log_sum_exp
    } else {
        f32::NEG_INFINITY
    };
    let sampled_rank = indexed
        .iter()
        .position(|&(tid, _)| tid == sampled_token_id)
        .map(|p| p as u32 + 1)
        .unwrap_or(0);

    LogprobsOutput {
        sampled: TokenLogprob {
            token_id: sampled_token_id,
            logprob: sampled_logprob,
            rank: sampled_rank,
        },
        top_logprobs,
    }
}

/// Argmax over a slice, returning the index of the maximum value.
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
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

    // ---------------------------------------------------------------
    // Tests for penalty / min_p / logit_bias / logprobs / sample_one
    // ---------------------------------------------------------------

    #[test]
    fn test_repetition_penalty() {
        // Token 1 was seen before — its logit should be penalized.
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let prev_tokens = vec![1u32];
        apply_penalties(&mut logits, &prev_tokens, 1.5, 0.0, 0.0);

        // Token 1 (logit 2.0 > 0) → divided by 1.5.
        assert!((logits[1] - 2.0 / 1.5).abs() < 1e-6);
        // Other tokens unchanged.
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty_negative_logit() {
        // Negative logit → multiplied by penalty (makes it more negative).
        let mut logits = vec![-2.0f32, 1.0];
        let prev_tokens = vec![0u32];
        apply_penalties(&mut logits, &prev_tokens, 2.0, 0.0, 0.0);

        assert!((logits[0] - (-2.0 * 2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_frequency_penalty() {
        // Token 2 appears 3 times → penalized by freq_penalty * 3.
        let mut logits = vec![1.0, 2.0, 5.0];
        let prev_tokens = vec![2, 2, 2];
        apply_penalties(&mut logits, &prev_tokens, 1.0, 0.5, 0.0);

        assert!((logits[2] - (5.0 - 0.5 * 3.0)).abs() < 1e-6);
        // Token 0 and 1 unchanged.
        assert!((logits[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_presence_penalty() {
        // Any seen token gets flat presence penalty.
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        let prev_tokens = vec![1, 3];
        apply_penalties(&mut logits, &prev_tokens, 1.0, 0.0, 1.0);

        assert!((logits[1] - (2.0 - 1.0)).abs() < 1e-6);
        assert!((logits[3] - (4.0 - 1.0)).abs() < 1e-6);
        // Unseen tokens unchanged.
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_p_filtering() {
        // Probabilities: [0.5, 0.3, 0.1, 0.1] (already softmaxed).
        let mut indexed = vec![(0, 0.5f32), (1, 0.3), (2, 0.1), (3, 0.1)];
        // min_p = 0.5: threshold = 0.5 * 0.5 = 0.25.
        // Keep tokens with prob >= 0.25: indices 0 (0.5) and 1 (0.3).
        apply_min_p(&mut indexed, 0.5);
        assert_eq!(indexed.len(), 2);
        assert_eq!(indexed[0].0, 0);
        assert_eq!(indexed[1].0, 1);
        // Should be re-normalized.
        let sum: f32 = indexed.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_logit_bias() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let bias: HashMap<u32, f32> = [(0, 10.0), (2, -5.0)].into_iter().collect();
        apply_logit_bias(&mut logits, &bias);

        assert!((logits[0] - 11.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);
        assert!((logits[2] - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_logprobs() {
        // Logits: token 2 has the highest logit.
        let logits = vec![1.0f32, 2.0, 10.0, 0.5];
        let result = compute_logprobs(&logits, 2, 3);

        // Sampled token should be token 2.
        assert_eq!(result.sampled.token_id, 2);
        assert_eq!(result.sampled.rank, 1); // highest logit
        // logprob should be close to 0 (dominant).
        assert!(result.sampled.logprob > -1.0);

        // Top 3 should be ordered by logprob descending.
        assert_eq!(result.top_logprobs.len(), 3);
        assert_eq!(result.top_logprobs[0].token_id, 2); // highest
        assert_eq!(result.top_logprobs[0].rank, 1);
        assert_eq!(result.top_logprobs[1].token_id, 1); // second
        assert_eq!(result.top_logprobs[1].rank, 2);
        // Each successive logprob should be less than or equal.
        assert!(result.top_logprobs[0].logprob >= result.top_logprobs[1].logprob);
        assert!(result.top_logprobs[1].logprob >= result.top_logprobs[2].logprob);
    }

    #[test]
    fn test_compute_logprobs_zero_n() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let result = compute_logprobs(&logits, 2, 0);
        assert!(result.top_logprobs.is_empty());
        // Sampled token info still returned.
        assert_eq!(result.sampled.token_id, 2);
    }

    #[test]
    fn test_sample_one_greedy() {
        let mut sampler = Sampler::new();
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let (token, lp) = sampler.sample_one(&logits, &params, &[], None);
        assert_eq!(token, 1); // argmax
        assert!(lp.is_none()); // no logprobs requested
    }

    #[test]
    fn test_sample_one_greedy_with_logprobs() {
        let mut sampler = Sampler::new();
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            logprobs: Some(2),
            ..Default::default()
        };
        let (token, lp) = sampler.sample_one(&logits, &params, &[], None);
        assert_eq!(token, 1);
        let lp = lp.unwrap();
        assert_eq!(lp.sampled.token_id, 1);
        assert_eq!(lp.top_logprobs.len(), 2);
    }

    #[test]
    fn test_sample_one_with_penalties() {
        let mut sampler = Sampler::new();
        // Token 1 has the highest logit, but it was seen before with rep_penalty.
        let logits = vec![1.0f32, 5.0, 4.9, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        // Token 1 seen: logit 5.0 / 2.0 = 2.5, so token 2 (4.9) wins.
        let (token, _) = sampler.sample_one(&logits, &params, &[1], None);
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_one_with_logit_bias() {
        let mut sampler = Sampler::new();
        let logits = vec![1.0f32, 2.0, 3.0];
        let bias: HashMap<u32, f32> = [(0, 100.0)].into_iter().collect();
        let params = SamplingParams {
            temperature: 0.0,
            logit_bias: Some(bias),
            ..Default::default()
        };
        // Token 0 boosted by 100 → should be selected.
        let (token, _) = sampler.sample_one(&logits, &params, &[], None);
        assert_eq!(token, 0);
    }

    #[test]
    fn test_sample_one_with_grammar_mask() {
        let mut sampler = Sampler::new();
        // Token 1 has highest logit, but grammar only allows tokens 0 and 2.
        let logits = vec![1.0f32, 5.0, 3.0, 2.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let allowed = vec![0u32, 2];
        let (token, _) = sampler.sample_one(&logits, &params, &[], Some(&allowed));
        // Token 2 (logit 3.0) should win since token 1 is masked out.
        assert_eq!(token, 2);
    }

    #[test]
    fn test_sample_one_grammar_mask_with_temperature() {
        let mut sampler = Sampler::new();
        // Token 1 dominates, but grammar masks it out.
        let logits = vec![1.0f32, 100.0, 50.0, 2.0];
        let params = SamplingParams {
            temperature: 1.0,
            ..Default::default()
        };
        let allowed = vec![2u32];
        let (token, _) = sampler.sample_one(&logits, &params, &[], Some(&allowed));
        // Only token 2 is allowed.
        assert_eq!(token, 2);
    }
}
