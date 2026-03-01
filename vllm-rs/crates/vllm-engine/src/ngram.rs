// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! N-gram speculative decoding proposer.
//!
//! Scans a request's token history (prompt + output) for matching n-grams
//! and proposes continuation tokens as speculative drafts. The target model
//! verifies these drafts in a single multi-token forward pass, accepting
//! all tokens up to the first mismatch.
//!
//! This is the simplest form of speculative decoding — no draft model needed,
//! just a lookup table over the request's own token history. Works well for
//! repetitive text, code, and reformatting tasks.
//!
//! Port of: `vllm/v1/spec_decode/ngram_proposer.py`

/// Configuration for the N-gram proposer.
#[derive(Debug, Clone)]
pub struct NgramProposerConfig {
    /// Number of draft tokens to propose per step.
    pub num_speculative_tokens: usize,
    /// Maximum n-gram size to search for (tries this first, then smaller).
    /// Minimum is 1 (unigram — matches any single token).
    pub max_ngram_size: usize,
    /// Minimum n-gram size. Stops searching below this.
    pub min_ngram_size: usize,
}

impl Default for NgramProposerConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 5,
            max_ngram_size: 4,
            min_ngram_size: 1,
        }
    }
}

/// Proposes draft tokens by matching n-grams in the token history.
#[derive(Debug)]
pub struct NgramProposer {
    config: NgramProposerConfig,
}

impl NgramProposer {
    /// Create a new proposer with the given configuration.
    pub fn new(config: NgramProposerConfig) -> Self {
        Self { config }
    }

    /// Propose draft tokens for speculative decoding.
    ///
    /// Scans `all_token_ids` (prompt + output so far) for an n-gram at the
    /// tail that also appears earlier in the sequence. When found, proposes
    /// the tokens that followed the earlier occurrence.
    ///
    /// Tries the largest n-gram size first (most specific match), falling
    /// back to smaller sizes. Returns up to `num_speculative_tokens` drafts.
    ///
    /// Returns an empty vec if no matching n-gram is found.
    pub fn propose(&self, all_token_ids: &[u32]) -> Vec<u32> {
        let num_tokens = all_token_ids.len();
        if num_tokens < 2 {
            return Vec::new();
        }

        // Try n-gram sizes from largest to smallest (longest match first).
        let max_n = self.config.max_ngram_size.min(num_tokens);
        let min_n = self.config.min_ngram_size.max(1);

        for n in (min_n..=max_n).rev() {
            // The n-gram at the tail of the sequence.
            let tail = &all_token_ids[num_tokens - n..];

            // Search for this n-gram earlier in the sequence.
            // We need at least n tokens for the match plus at least 1 token
            // after it for the proposal.
            let search_end = num_tokens - n;
            if search_end == 0 {
                continue;
            }

            // Scan backwards from most recent occurrence (more likely to
            // match current context).
            for start in (0..search_end).rev() {
                if start + n > search_end {
                    continue;
                }
                let candidate = &all_token_ids[start..start + n];
                if candidate == tail {
                    // Found a match! Propose the tokens that follow.
                    let follow_start = start + n;
                    let follow_end =
                        (follow_start + self.config.num_speculative_tokens).min(num_tokens);
                    if follow_start >= follow_end {
                        continue;
                    }
                    return all_token_ids[follow_start..follow_end].to_vec();
                }
            }
        }

        Vec::new()
    }

    /// Get the configuration.
    pub fn config(&self) -> &NgramProposerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_proposer() -> NgramProposer {
        NgramProposer::new(NgramProposerConfig::default())
    }

    fn proposer(num_spec: usize, max_n: usize, min_n: usize) -> NgramProposer {
        NgramProposer::new(NgramProposerConfig {
            num_speculative_tokens: num_spec,
            max_ngram_size: max_n,
            min_ngram_size: min_n,
        })
    }

    #[test]
    fn test_empty_input() {
        let p = default_proposer();
        assert!(p.propose(&[]).is_empty());
    }

    #[test]
    fn test_single_token() {
        let p = default_proposer();
        assert!(p.propose(&[42]).is_empty());
    }

    #[test]
    fn test_no_match() {
        let p = default_proposer();
        // All unique tokens — no n-gram repeats.
        let tokens: Vec<u32> = (0..20).collect();
        assert!(p.propose(&tokens).is_empty());
    }

    #[test]
    fn test_simple_bigram_match() {
        // Tokens: [A, B, C, D, A, B]
        // Tail bigram [A, B] matches at position 0.
        // Continuation after position 0+2 = [C, D, A, B] → propose [C, D, A, B].
        let p = proposer(5, 4, 1);
        let tokens = vec![10, 20, 30, 40, 10, 20];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![30, 40, 10, 20]);
    }

    #[test]
    fn test_trigram_preferred_over_bigram() {
        // Tokens: [A, B, C, D, E, A, B, C]
        // Tail trigram [A, B, C] matches at position 0.
        // Continuation: [D, E, A, B, C] → capped at 5 → [D, E, A, B, C].
        let p = proposer(5, 4, 1);
        let tokens = vec![10, 20, 30, 40, 50, 10, 20, 30];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![40, 50, 10, 20, 30]);
    }

    #[test]
    fn test_max_speculative_tokens_limit() {
        // Many tokens after the match, but limited to 3.
        let p = proposer(3, 2, 1);
        let tokens = vec![10, 20, 30, 40, 50, 60, 10, 20];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![30, 40, 50]);
    }

    #[test]
    fn test_most_recent_match_preferred() {
        // Tokens: [A, B, X, A, B, Y, A, B]
        // Bigram [A, B] at tail matches at positions 0 and 3.
        // Most recent match (position 3) is preferred → propose [Y, A, B].
        let p = proposer(5, 2, 1);
        let tokens = vec![10, 20, 100, 10, 20, 200, 10, 20];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![200, 10, 20]);
    }

    #[test]
    fn test_unigram_fallback() {
        // Tokens: [A, B, C, A] — tail unigram [A] matches at position 0.
        // Larger n-grams don't match. Continuation: [B, C, A].
        let p = proposer(5, 3, 1);
        let tokens = vec![10, 20, 30, 10];
        let drafts = p.propose(&tokens);
        // Trigram [30, 10] at tail? No, tail trigram would be [20, 30, 10] — need 3 tokens.
        // Let's trace: n=3: tail=[20,30,10], search in [10], no match.
        // n=2: tail=[30,10], search in [10,20], no match for [30,10].
        // n=1: tail=[10], search in [10,20,30]. Position 0: [10] matches!
        // Continuation: tokens[1..min(1+5, 4)] = [20, 30, 10].
        assert_eq!(drafts, vec![20, 30, 10]);
    }

    #[test]
    fn test_min_ngram_size_respected() {
        // With min_ngram_size=2, unigram fallback should be disabled.
        let p = proposer(5, 3, 2);
        let tokens = vec![10, 20, 30, 10];
        // n=3: tail=[20,30,10], no match in prefix.
        // n=2: tail=[30,10], no match in prefix [10,20].
        // n=1 is below min, so no fallback.
        let drafts = p.propose(&tokens);
        assert!(drafts.is_empty());
    }

    #[test]
    fn test_repetitive_sequence() {
        // Highly repetitive: [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]
        // Tail bigram [1, 2] at position 9,10.
        // Most recent earlier match at position 6,7 → continuation [3, 1, 2].
        let p = proposer(5, 3, 1);
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2];
        let drafts = p.propose(&tokens);
        // Tail trigram [3, 1, 2]: positions 8,9,10. Search earlier:
        // Position 5: [3, 1, 2] matches! Continuation: tokens[8..min(8+5, 11)] = [3, 1, 2].
        assert_eq!(drafts, vec![3, 1, 2]);
    }

    #[test]
    fn test_match_at_boundary() {
        // Match right before the tail — continuation has only 1 token.
        let p = proposer(5, 2, 1);
        let tokens = vec![10, 20, 30, 10, 20];
        // Bigram [10, 20] at tail matches position 0.
        // Continuation: tokens[2..min(2+5, 5)] = [30, 10, 20].
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![30, 10, 20]);
    }

    #[test]
    fn test_code_like_pattern() {
        // Simulate code with repeated patterns:
        // "def foo():\n    return 0\ndef bar():\n    return"
        // Using token IDs as placeholders.
        let def_tok = 100;
        let colon = 101;
        let newline = 102;
        let indent = 103;
        let return_tok = 104;
        let zero = 105;

        let tokens = vec![
            def_tok, 200, colon, newline, indent, return_tok, zero, newline, def_tok, 201, colon,
            newline, indent, return_tok,
        ];

        let p = proposer(3, 4, 1);
        let drafts = p.propose(&tokens);
        // Tail 4-gram [newline, indent, return_tok] (n=3 since 4-gram may not fit).
        // Actually n=4: tail=[colon, newline, indent, return_tok].
        // Search: position 2: [colon, newline, indent, return_tok] matches!
        // Continuation: tokens[6..min(6+3, 14)] = [zero, newline, def_tok].
        assert_eq!(drafts, vec![zero, newline, def_tok]);
    }

    #[test]
    fn test_default_config() {
        let config = NgramProposerConfig::default();
        assert_eq!(config.num_speculative_tokens, 5);
        assert_eq!(config.max_ngram_size, 4);
        assert_eq!(config.min_ngram_size, 1);
    }

    #[test]
    fn test_config_accessor() {
        let p = proposer(3, 5, 2);
        assert_eq!(p.config().num_speculative_tokens, 3);
        assert_eq!(p.config().max_ngram_size, 5);
        assert_eq!(p.config().min_ngram_size, 2);
    }

    #[test]
    fn test_two_tokens_input() {
        // Minimum viable: [A, A] → unigram match, propose nothing (no tokens after match).
        // Actually: tail=[A] (n=1), search in [A]. Position 0 matches.
        // Continuation: tokens[1..min(1+5, 2)] = [A]. Propose [A]!
        let p = proposer(5, 1, 1);
        let tokens = vec![42, 42];
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![42]);
    }

    #[test]
    fn test_long_match_but_short_continuation() {
        // 4-gram match but only 1 token of continuation available.
        let p = proposer(5, 4, 1);
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3, 4];
        // Tail 4-gram [2,3,4]: n=4 → tail=[1,2,3,4] at pos 5..9.
        // Search: pos 0: [1,2,3,4] matches! Continuation: tokens[4..min(4+5,9)] = [5,1,2,3,4].
        let drafts = p.propose(&tokens);
        assert_eq!(drafts, vec![5, 1, 2, 3, 4]);
    }
}
