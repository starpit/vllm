// SPDX-License-Identifier: Apache-2.0
//! Sampler for converting logits to token IDs.
//!
//! Supports greedy decoding (argmax) and random sampling with temperature,
//! top-k, and top-p.
//!
//! Port of: `vllm/v1/sample/sampler.py`

use std::collections::HashMap;

use rand::Rng;

use vllm_common::SamplingParams;
use vllm_common::sampling::{LogprobsOutput, TokenLogprob};

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
            apply_allow_mask(&mut logits_buf, allowed);
        }

        // 1.6. Apply allowed_token_ids whitelist.
        if let Some(ref allowed) = params.allowed_token_ids {
            apply_allow_mask(&mut logits_buf, allowed);
        }

        // 1.7. Suppress bad words: if output ends with a bad word prefix, mask the completing token.
        if let Some(ref bad_words) = params.bad_words_token_ids {
            for bad_word in bad_words {
                if let Some(suppress_token) = bad_word_suffix_match(prev_tokens, bad_word)
                    && (suppress_token as usize) < logits_buf.len()
                {
                    logits_buf[suppress_token as usize] = f32::NEG_INFINITY;
                }
            }
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
/// Mask all token IDs not in `allowed` to `-inf`.
/// Check if `output_tokens` ends with the prefix of `bad_word` (all but the last token).
/// If so, return the completing token (the last token of the bad word).
fn bad_word_suffix_match(output_tokens: &[u32], bad_word: &[u32]) -> Option<u32> {
    if bad_word.is_empty() {
        return None;
    }
    if bad_word.len() == 1 {
        return Some(bad_word[0]);
    }
    let prefix = &bad_word[..bad_word.len() - 1];
    if output_tokens.len() >= prefix.len()
        && output_tokens[output_tokens.len() - prefix.len()..] == *prefix
    {
        Some(bad_word[bad_word.len() - 1])
    } else {
        None
    }
}

fn apply_allow_mask(logits: &mut [f32], allowed: &[u32]) {
    let mut mask = vec![false; logits.len()];
    for &tid in allowed {
        let idx = tid as usize;
        if idx < mask.len() {
            mask[idx] = true;
        }
    }
    for (i, l) in logits.iter_mut().enumerate() {
        if !mask[i] {
            *l = f32::NEG_INFINITY;
        }
    }
}

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
pub fn compute_logprobs(logits: &[f32], sampled_token_id: u32, top_n: usize) -> LogprobsOutput {
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

    // ---------------------------------------------------------------
    // Tests for prompt logprobs (compute_logprobs used for known tokens)
    // ---------------------------------------------------------------

    #[test]
    fn test_compute_logprobs_for_prompt_token() {
        // Simulate prompt logprobs: given logits at position i,
        // the "sampled" token is the actual prompt token at position i+1.
        let logits = vec![1.0f32, 2.0, 10.0, 0.5];
        // If the actual prompt token is 1 (not the highest-logit token):
        let result = compute_logprobs(&logits, 1, 3);

        // Sampled token should be 1 (the prompt token, not the argmax).
        assert_eq!(result.sampled.token_id, 1);
        // Rank should be 2 (token 2 has higher logit).
        assert_eq!(result.sampled.rank, 2);
        // Logprob should be negative (not the most likely token).
        assert!(result.sampled.logprob < 0.0);
        // Top 3 should still be ordered by probability descending.
        assert_eq!(result.top_logprobs.len(), 3);
        assert_eq!(result.top_logprobs[0].token_id, 2); // highest logit
    }

    #[test]
    fn test_prompt_logprobs_sequence() {
        // Simulate a 4-token prompt: [A, B, C, D]
        // logits[0] predicts B, logits[1] predicts C, logits[2] predicts D.
        let prompt_tokens = vec![10u32, 20, 30, 40];
        let logits_per_position = vec![
            vec![0.0f32; 50], // position 0 → predicts token 20
            vec![0.0f32; 50], // position 1 → predicts token 30
            vec![0.0f32; 50], // position 2 → predicts token 40
        ];

        // Set high logits for the actual next tokens to verify correctness.
        let mut logits0 = logits_per_position[0].clone();
        logits0[20] = 10.0;
        let mut logits1 = logits_per_position[1].clone();
        logits1[30] = 10.0;
        let mut logits2 = logits_per_position[2].clone();
        logits2[40] = 10.0;

        // Build prompt logprobs the same way the worker does.
        let mut plps = Vec::new();
        for (i, logits) in [logits0, logits1, logits2].iter().enumerate() {
            let actual_token = prompt_tokens[i + 1];
            plps.push(compute_logprobs(logits, actual_token, 2));
        }

        assert_eq!(plps.len(), 3);
        assert_eq!(plps[0].sampled.token_id, 20);
        assert_eq!(plps[0].sampled.rank, 1); // highest logit
        assert_eq!(plps[1].sampled.token_id, 30);
        assert_eq!(plps[1].sampled.rank, 1);
        assert_eq!(plps[2].sampled.token_id, 40);
        assert_eq!(plps[2].sampled.rank, 1);
    }

    #[test]
    fn test_prompt_logprobs_with_none_for_first_position() {
        // Simulate the full prompt_logprobs output format:
        // Position 0: None (no prior context)
        // Position 1+: Some(LogprobsOutput)
        let logits = vec![1.0f32, 5.0, 3.0];
        let prompt_tokens = vec![0u32, 1, 2]; // 3-token prompt

        let mut result: Vec<Option<LogprobsOutput>> = Vec::new();
        result.push(None); // position 0
        for i in 0..prompt_tokens.len() - 1 {
            result.push(Some(compute_logprobs(&logits, prompt_tokens[i + 1], 2)));
        }

        assert_eq!(result.len(), 3);
        assert!(result[0].is_none());
        assert!(result[1].is_some());
        assert_eq!(result[1].as_ref().unwrap().sampled.token_id, 1);
        assert!(result[2].is_some());
        assert_eq!(result[2].as_ref().unwrap().sampled.token_id, 2);
    }

    #[test]
    fn test_allowed_token_ids_restricts_sampling() {
        let mut sampler = Sampler::new();
        // Logits: token 0 has highest logit, but only token 2 is allowed.
        let logits = vec![10.0f32, 5.0, 1.0, 0.0];
        let params = SamplingParams {
            temperature: 0.0, // greedy
            allowed_token_ids: Some(vec![2]),
            ..Default::default()
        };
        let (token_id, _) = sampler.sample_one(&logits, &params, &[], None);
        assert_eq!(token_id, 2);
    }

    #[test]
    fn test_allowed_token_ids_with_grammar_mask() {
        let mut sampler = Sampler::new();
        // Grammar allows tokens 1 and 2; allowed_token_ids allows 0 and 2.
        // Intersection should be token 2.
        let logits = vec![10.0f32, 8.0, 1.0];
        let params = SamplingParams {
            temperature: 0.0,
            allowed_token_ids: Some(vec![0, 2]),
            ..Default::default()
        };
        let grammar_allowed = vec![1u32, 2];
        let (token_id, _) = sampler.sample_one(&logits, &params, &[], Some(&grammar_allowed));
        assert_eq!(token_id, 2);
    }

    #[test]
    fn test_bad_words_single_token_suppressed() {
        let mut sampler = Sampler::new();
        // Token 0 has highest logit, but it's a single-token bad word.
        let logits = vec![10.0f32, 5.0, 1.0, 0.0];
        let params = SamplingParams {
            temperature: 0.0,
            bad_words_token_ids: Some(vec![vec![0]]),
            ..Default::default()
        };
        let (token_id, _) = sampler.sample_one(&logits, &params, &[], None);
        assert_eq!(token_id, 1, "single-token bad word should be suppressed");
    }

    #[test]
    fn test_bad_words_multi_token_prefix_match() {
        let mut sampler = Sampler::new();
        // Bad word is [10, 20, 0]. Output ends with [10, 20].
        // Token 0 (the completing token) should be suppressed.
        let logits = vec![10.0f32, 5.0, 1.0];
        let prev_tokens = vec![10, 20];
        let params = SamplingParams {
            temperature: 0.0,
            bad_words_token_ids: Some(vec![vec![10, 20, 0]]),
            ..Default::default()
        };
        let (token_id, _) = sampler.sample_one(&logits, &params, &prev_tokens, None);
        assert_eq!(
            token_id, 1,
            "completing token of bad word should be suppressed"
        );
    }

    #[test]
    fn test_bad_words_no_match_no_suppression() {
        let mut sampler = Sampler::new();
        // Bad word is [10, 20, 0]. Output is [5, 6] — no prefix match.
        let logits = vec![10.0f32, 5.0, 1.0];
        let prev_tokens = vec![5, 6];
        let params = SamplingParams {
            temperature: 0.0,
            bad_words_token_ids: Some(vec![vec![10, 20, 0]]),
            ..Default::default()
        };
        let (token_id, _) = sampler.sample_one(&logits, &params, &prev_tokens, None);
        assert_eq!(token_id, 0, "no prefix match means no suppression");
    }

    #[test]
    fn test_bad_words_multiple_words() {
        let mut sampler = Sampler::new();
        // Two bad words: [0] (single token) and [99, 1] (multi-token, output ends with [99]).
        // Tokens 0 and 1 should both be suppressed, leaving token 2.
        let logits = vec![10.0f32, 8.0, 1.0, 0.0];
        let prev_tokens = vec![99];
        let params = SamplingParams {
            temperature: 0.0,
            bad_words_token_ids: Some(vec![vec![0], vec![99, 1]]),
            ..Default::default()
        };
        let (token_id, _) = sampler.sample_one(&logits, &params, &prev_tokens, None);
        assert_eq!(token_id, 2, "both bad words should be suppressed");
    }
}
