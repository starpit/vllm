// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Sampling parameters for text generation, ported from `vllm/sampling_params.py`.

use serde::{Deserialize, Serialize};

/// Epsilon used to distinguish greedy from random sampling.
const SAMPLING_EPS: f64 = 1e-5;

// ---------------------------------------------------------------------------
// SamplingType
// ---------------------------------------------------------------------------

/// The type of sampling to use for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SamplingType {
    /// Temperature is effectively zero -- pick the argmax token.
    Greedy = 0,
    /// Sample from the distribution (no fixed seed).
    Random = 1,
    /// Sample from the distribution with a fixed seed for reproducibility.
    RandomSeed = 2,
}

// ---------------------------------------------------------------------------
// RequestOutputKind
// ---------------------------------------------------------------------------

/// Controls how incremental output is returned to the caller.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum RequestOutputKind {
    /// Return the entire output so far in every `RequestOutput`.
    #[default]
    Cumulative = 0,
    /// Return only deltas in each `RequestOutput`.
    Delta = 1,
    /// Do not return intermediate `RequestOutput`; only the final one.
    FinalOnly = 2,
}

// ---------------------------------------------------------------------------
// SamplingParams
// ---------------------------------------------------------------------------

/// Parameters that control token sampling during generation.
///
/// Mirrors the Python `SamplingParams` class from `vllm/sampling_params.py`.
/// Only the core fields used by the scheduler and sampler are included here;
/// fields that require a tokenizer or model config (e.g. `bad_words_token_ids`,
/// structured-output backends) are handled at a higher layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Number of output sequences to generate per prompt.
    pub n: u32,

    /// Penalizes tokens that have already appeared, regardless of frequency.
    /// Values in \[-2, 2\].
    pub presence_penalty: f64,

    /// Penalizes tokens proportionally to how often they have appeared.
    /// Values in \[-2, 2\].
    pub frequency_penalty: f64,

    /// Multiplicative penalty for repeated tokens. Values > 1 discourage
    /// repetition; values < 1 encourage it.
    pub repetition_penalty: f64,

    /// Softmax temperature. 0 means greedy.
    pub temperature: f64,

    /// Nucleus-sampling cumulative probability threshold. Must be in (0, 1].
    pub top_p: f64,

    /// Top-k filtering. 0 (or -1) disables.
    pub top_k: i32,

    /// Minimum probability relative to the most-likely token. Must be in [0, 1].
    pub min_p: f64,

    /// Optional random seed for reproducible sampling.
    pub seed: Option<u64>,

    /// Stop strings. Generation stops when any of these is emitted.
    pub stop: Vec<String>,

    /// Token IDs that trigger a stop.
    pub stop_token_ids: Vec<u32>,

    /// Whether to ignore the EOS token and keep generating.
    pub ignore_eos: bool,

    /// Maximum number of tokens to generate. `None` means unlimited
    /// (bounded only by the model's context length).
    pub max_tokens: Option<u32>,

    /// Minimum number of tokens to generate before allowing EOS / stop tokens.
    pub min_tokens: u32,

    /// Number of per-token log-probabilities to return.
    /// `None` means do not return logprobs. `-1` means return all.
    pub logprobs: Option<i32>,

    /// Number of per-prompt-token log-probabilities to return.
    pub prompt_logprobs: Option<i32>,

    /// Whether to detokenize the output.
    pub detokenize: bool,

    /// Whether to skip special tokens in the detokenized output.
    pub skip_special_tokens: bool,

    /// Whether to include the stop string in the generated output text.
    pub include_stop_str_in_output: bool,

    /// How incremental output is delivered to the caller.
    pub output_kind: RequestOutputKind,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            n: 1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: None,
            stop: Vec::new(),
            stop_token_ids: Vec::new(),
            ignore_eos: false,
            max_tokens: Some(16),
            min_tokens: 0,
            logprobs: None,
            prompt_logprobs: None,
            detokenize: true,
            skip_special_tokens: true,
            include_stop_str_in_output: false,
            output_kind: RequestOutputKind::default(),
        }
    }
}

impl SamplingParams {
    /// Determine the [`SamplingType`] implied by these parameters.
    ///
    /// * `temperature < EPS` => Greedy
    /// * `seed.is_some()`    => RandomSeed
    /// * otherwise           => Random
    pub fn sampling_type(&self) -> SamplingType {
        if self.temperature < SAMPLING_EPS {
            SamplingType::Greedy
        } else if self.seed.is_some() {
            SamplingType::RandomSeed
        } else {
            SamplingType::Random
        }
    }

    /// Validate the sampling parameters, returning an error message on failure.
    ///
    /// This mirrors the `_verify_args` logic from the Python implementation.
    pub fn validate(&self) -> Result<(), String> {
        if self.n < 1 {
            return Err(format!("n must be at least 1, got {}", self.n));
        }
        if !(-2.0..=2.0).contains(&self.presence_penalty) {
            return Err(format!(
                "presence_penalty must be in [-2, 2], got {}",
                self.presence_penalty
            ));
        }
        if !(-2.0..=2.0).contains(&self.frequency_penalty) {
            return Err(format!(
                "frequency_penalty must be in [-2, 2], got {}",
                self.frequency_penalty
            ));
        }
        if self.repetition_penalty <= 0.0 {
            return Err(format!(
                "repetition_penalty must be > 0, got {}",
                self.repetition_penalty
            ));
        }
        if self.temperature < 0.0 {
            return Err(format!(
                "temperature must be non-negative, got {}",
                self.temperature
            ));
        }
        if !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(format!("top_p must be in (0, 1], got {}", self.top_p));
        }
        if self.top_k < -1 {
            return Err(format!(
                "top_k must be 0 (disable), or at least 1, got {}",
                self.top_k
            ));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(format!("min_p must be in [0, 1], got {}", self.min_p));
        }
        if let Some(max) = self.max_tokens
            && max < 1
        {
            return Err(format!("max_tokens must be at least 1, got {max}"));
        }
        if self.min_tokens > 0
            && let Some(max) = self.max_tokens
            && self.min_tokens > max
        {
            return Err(format!(
                "min_tokens ({}) must be <= max_tokens ({max})",
                self.min_tokens
            ));
        }
        if let Some(lp) = self.logprobs
            && lp != -1
            && lp < 0
        {
            return Err(format!("logprobs must be non-negative or -1, got {lp}"));
        }
        if let Some(plp) = self.prompt_logprobs
            && plp != -1
            && plp < 0
        {
            return Err(format!(
                "prompt_logprobs must be non-negative or -1, got {plp}"
            ));
        }
        if !self.stop.is_empty() && !self.detokenize {
            return Err("stop strings are only supported when detokenize is true".into());
        }
        if self.stop.iter().any(|s| s.is_empty()) {
            return Err("stop cannot contain an empty string".into());
        }
        // Greedy-specific check.
        if self.temperature < SAMPLING_EPS && self.n > 1 {
            return Err(format!(
                "n must be 1 when using greedy sampling, got {}",
                self.n
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_sampling_params() {
        let p = SamplingParams::default();
        assert_eq!(p.n, 1);
        assert_eq!(p.temperature, 1.0);
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.top_k, 0);
        assert_eq!(p.max_tokens, Some(16));
        assert!(p.stop.is_empty());
        assert!(p.stop_token_ids.is_empty());
        assert!(p.detokenize);
        assert!(p.skip_special_tokens);
    }

    #[test]
    fn test_sampling_type_greedy() {
        let p = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(p.sampling_type(), SamplingType::Greedy);
    }

    #[test]
    fn test_sampling_type_random() {
        let p = SamplingParams {
            temperature: 0.8,
            seed: None,
            ..Default::default()
        };
        assert_eq!(p.sampling_type(), SamplingType::Random);
    }

    #[test]
    fn test_sampling_type_random_seed() {
        let p = SamplingParams {
            temperature: 0.8,
            seed: Some(42),
            ..Default::default()
        };
        assert_eq!(p.sampling_type(), SamplingType::RandomSeed);
    }

    #[test]
    fn test_validate_ok() {
        let p = SamplingParams::default();
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_validate_presence_penalty_out_of_range() {
        let p = SamplingParams {
            presence_penalty: 3.0,
            ..Default::default()
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("presence_penalty"));
    }

    #[test]
    fn test_validate_frequency_penalty_out_of_range() {
        let p = SamplingParams {
            frequency_penalty: -3.0,
            ..Default::default()
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("frequency_penalty"));
    }

    #[test]
    fn test_validate_repetition_penalty_zero() {
        let p = SamplingParams {
            repetition_penalty: 0.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_negative_temperature() {
        let p = SamplingParams {
            temperature: -1.0,
            ..Default::default()
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("temperature"));
    }

    #[test]
    fn test_validate_top_p_zero() {
        let p = SamplingParams {
            top_p: 0.0,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_top_p_above_one() {
        let p = SamplingParams {
            top_p: 1.5,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_top_k_invalid() {
        let p = SamplingParams {
            top_k: -2,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_min_p_out_of_range() {
        let p = SamplingParams {
            min_p: 1.5,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_max_tokens_zero() {
        let p = SamplingParams {
            max_tokens: Some(0),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_min_tokens_exceeds_max() {
        let p = SamplingParams {
            min_tokens: 100,
            max_tokens: Some(10),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_greedy_n_greater_than_one() {
        let p = SamplingParams {
            temperature: 0.0,
            n: 2,
            ..Default::default()
        };
        let err = p.validate().unwrap_err();
        assert!(err.contains("greedy"));
    }

    #[test]
    fn test_validate_empty_stop_string() {
        let p = SamplingParams {
            stop: vec!["".into()],
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_stop_without_detokenize() {
        let p = SamplingParams {
            stop: vec!["</s>".into()],
            detokenize: false,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_logprobs_invalid() {
        let p = SamplingParams {
            logprobs: Some(-2),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_logprobs_minus_one_ok() {
        let p = SamplingParams {
            logprobs: Some(-1),
            ..Default::default()
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn test_validate_prompt_logprobs_invalid() {
        let p = SamplingParams {
            prompt_logprobs: Some(-5),
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_serde_roundtrip() {
        let p = SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            seed: Some(123),
            stop: vec!["STOP".into()],
            max_tokens: Some(256),
            ..Default::default()
        };
        let json = serde_json::to_string(&p).unwrap();
        let p2: SamplingParams = serde_json::from_str(&json).unwrap();
        assert_eq!(p.temperature, p2.temperature);
        assert_eq!(p.top_p, p2.top_p);
        assert_eq!(p.seed, p2.seed);
        assert_eq!(p.stop, p2.stop);
        assert_eq!(p.max_tokens, p2.max_tokens);
    }

    #[test]
    fn test_request_output_kind_default() {
        assert_eq!(RequestOutputKind::default(), RequestOutputKind::Cumulative);
    }

    #[test]
    fn test_sampling_type_repr() {
        assert_eq!(SamplingType::Greedy as u8, 0);
        assert_eq!(SamplingType::Random as u8, 1);
        assert_eq!(SamplingType::RandomSeed as u8, 2);
    }

    #[test]
    fn test_request_output_kind_repr() {
        assert_eq!(RequestOutputKind::Cumulative as u8, 0);
        assert_eq!(RequestOutputKind::Delta as u8, 1);
        assert_eq!(RequestOutputKind::FinalOnly as u8, 2);
    }

    #[test]
    fn test_max_tokens_none_is_unlimited() {
        let p = SamplingParams {
            max_tokens: None,
            ..Default::default()
        };
        assert!(p.validate().is_ok());
    }
}
