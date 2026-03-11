// SPDX-License-Identifier: Apache-2.0
//! Grammar-guided constrained decoding.
//!
//! Wraps `llguidance` to compile JSON schemas, regex patterns, or Lark/EBNF
//! grammars into token masks for constrained decoding. Each request with a
//! `GuidedGrammar` gets its own `GrammarGuide` that tracks parser state and
//! provides the set of allowed tokens at each step.

use std::sync::Arc;

use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use toktrie::TokenId;
use toktrie_hf_tokenizers::ByteTokenizer;
use vllm_common::sampling::GuidedGrammar;

// Re-export for consumers (cuda_worker, mlx_worker).
pub use llguidance::ParserFactory as LlgParserFactory;

/// A compiled grammar guide that tracks parser state for constrained decoding.
///
/// Created once per request (when `guided_grammar` is set), then queried
/// and advanced at each decode step.
pub struct GrammarGuide {
    matcher: Matcher,
}

/// A generic JSON schema that matches any valid JSON object.
const JSON_OBJECT_SCHEMA: &str = r#"{"type": "object"}"#;

impl GrammarGuide {
    /// Build a grammar guide from a JSON schema.
    pub fn from_json_schema(
        schema: &serde_json::Value,
        factory: &Arc<ParserFactory>,
    ) -> Result<Self, String> {
        let grammar = TopLevelGrammar::from_json_schema(schema.clone());
        let parser = factory.create_parser(grammar).map_err(|e| e.to_string())?;
        Ok(Self {
            matcher: Matcher::new(Ok(parser)),
        })
    }

    /// Build a grammar guide from an arbitrary regex pattern.
    pub fn from_regex(pattern: &str, factory: &Arc<ParserFactory>) -> Result<Self, String> {
        let grammar = TopLevelGrammar::from_regex(pattern);
        let parser = factory.create_parser(grammar).map_err(|e| e.to_string())?;
        Ok(Self {
            matcher: Matcher::new(Ok(parser)),
        })
    }

    /// Build a grammar guide for generic JSON object output.
    pub fn from_json_object(factory: &Arc<ParserFactory>) -> Result<Self, String> {
        let schema: serde_json::Value =
            serde_json::from_str(JSON_OBJECT_SCHEMA).map_err(|e| e.to_string())?;
        Self::from_json_schema(&schema, factory)
    }

    /// Build a grammar guide from a Lark/EBNF grammar string.
    pub fn from_ebnf(grammar: &str, factory: &Arc<ParserFactory>) -> Result<Self, String> {
        let tlg = TopLevelGrammar::from_lark(grammar.to_string());
        let parser = factory.create_parser(tlg).map_err(|e| e.to_string())?;
        Ok(Self {
            matcher: Matcher::new(Ok(parser)),
        })
    }

    /// Create a `GrammarGuide` from a `GuidedGrammar` specification.
    pub fn from_guided_grammar(
        grammar: &GuidedGrammar,
        factory: &Arc<ParserFactory>,
    ) -> Result<Self, String> {
        match grammar {
            GuidedGrammar::Json => Self::from_json_object(factory),
            GuidedGrammar::JsonSchema { schema } => Self::from_json_schema(schema, factory),
            GuidedGrammar::Regex { pattern } => Self::from_regex(pattern, factory),
            GuidedGrammar::Ebnf { grammar } => Self::from_ebnf(grammar, factory),
        }
    }

    /// Get the set of token IDs allowed at the current state.
    ///
    /// Returns `None` if the parser is in an error/stopped state.
    pub fn allowed_tokens(&mut self) -> Option<Vec<TokenId>> {
        if self.matcher.is_stopped() {
            return None;
        }
        match self.matcher.compute_mask() {
            Ok(mask) => {
                let mut tokens = Vec::new();
                mask.iter_set_entries(|idx| tokens.push(idx as TokenId));
                Some(tokens)
            }
            Err(_) => None,
        }
    }

    /// Advance the parser to the next state given a sampled token.
    ///
    /// Returns `true` if the transition was valid, `false` on error.
    pub fn advance(&mut self, token_id: TokenId) -> bool {
        self.matcher.consume_token(token_id).is_ok()
    }

    /// Check if the parser is in a finished/stopped state.
    pub fn is_finished(&self) -> bool {
        self.matcher.is_stopped()
    }
}

/// Mask logits so that only `allowed` token IDs survive.
///
/// Sets all logits for tokens NOT in `allowed` to `-inf`.
/// This is applied early in the sampling pipeline (before penalties/temperature)
/// so that hard grammar constraints are not softened by later steps.
pub fn apply_grammar_mask(logits: &mut [f32], allowed: &[TokenId]) {
    // Mark allowed positions, then sweep all logits.
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

/// Build a `TokEnv` and `ParserFactory` from raw tokenizer.json bytes.
///
/// Workers call this once at model-load time and cache the result.
/// The `ParserFactory` is expensive to create (builds sliced bias computer)
/// but is reused across all requests.
pub fn build_parser_factory(tokenizer_json: &[u8]) -> Result<Arc<ParserFactory>, String> {
    let byte_tok = ByteTokenizer::from_json_bytes(tokenizer_json).map_err(|e| e.to_string())?;
    let tok_env = byte_tok.into_tok_env(None).map_err(|e| e.to_string())?;
    let mut factory = ParserFactory::new_simple(&tok_env).map_err(|e| e.to_string())?;
    factory.set_stderr_log_level(0);
    Ok(Arc::new(factory))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_factory() -> Arc<ParserFactory> {
        // Use a minimal BPE tokenizer JSON for testing.
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 99, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": "<unk>",
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "vocab": {
                    "{": 1,
                    "}": 2,
                    "\"": 3,
                    ":": 4,
                    ",": 5,
                    " ": 6,
                    "0": 7,
                    "1": 8,
                    "2": 9,
                    "3": 10,
                    "a": 11,
                    "b": 12,
                    "n": 13,
                    "u": 14,
                    "l": 15,
                    "t": 16,
                    "r": 17,
                    "e": 18,
                    "f": 19,
                    "s": 20,
                    "[": 21,
                    "]": 22,
                    ".": 23,
                    "-": 24,
                    "4": 25,
                    "5": 26,
                    "6": 27,
                    "7": 28,
                    "8": 29,
                    "9": 30
                },
                "merges": []
            }
        }"#;
        build_parser_factory(tokenizer_json.as_bytes()).unwrap()
    }

    #[test]
    fn test_grammar_mask_basic() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let allowed = vec![1u32, 3];
        apply_grammar_mask(&mut logits, &allowed);

        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], 4.0);
        assert_eq!(logits[4], f32::NEG_INFINITY);
    }

    #[test]
    fn test_grammar_mask_empty_allowed() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let allowed: Vec<u32> = vec![];
        apply_grammar_mask(&mut logits, &allowed);

        assert!(logits.iter().all(|&l| l == f32::NEG_INFINITY));
    }

    #[test]
    fn test_grammar_mask_all_allowed() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let allowed = vec![0u32, 1, 2];
        apply_grammar_mask(&mut logits, &allowed);

        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_build_parser_factory() {
        let factory = make_test_factory();
        // Factory should be usable for creating parsers.
        let grammar = TopLevelGrammar::from_regex("[0-9]+");
        assert!(factory.create_parser(grammar).is_ok());
    }

    #[test]
    fn test_from_regex_digit_pattern() {
        let factory = make_test_factory();
        let mut guide = GrammarGuide::from_regex("[0-9]+", &factory).unwrap();
        let allowed = guide.allowed_tokens();
        assert!(allowed.is_some());
        assert!(!allowed.unwrap().is_empty());
    }

    #[test]
    fn test_from_guided_grammar_regex_variant() {
        let factory = make_test_factory();
        let grammar = GuidedGrammar::Regex {
            pattern: "[0-9]+".to_string(),
        };
        let mut guide = GrammarGuide::from_guided_grammar(&grammar, &factory).unwrap();
        let allowed = guide.allowed_tokens();
        assert!(allowed.is_some());
        assert!(!allowed.unwrap().is_empty());
    }

    #[test]
    fn test_advance_and_finish() {
        let factory = make_test_factory();
        // Use a simple regex: exactly one digit.
        let mut guide = GrammarGuide::from_regex("[0-9]", &factory).unwrap();

        // At initial state, we should have allowed tokens.
        let allowed = guide.allowed_tokens().unwrap();
        assert!(!allowed.is_empty());

        // Advance with "1" (token_id=8 in our vocab).
        assert!(guide.advance(8));

        // After one digit, the guide should be finished (regex fully matched).
        // Need to call allowed_tokens to trigger stop check.
        let _ = guide.allowed_tokens();
        assert!(guide.is_finished());
    }

    #[test]
    fn test_from_ebnf_grammar() {
        let factory = make_test_factory();
        // Simple Lark grammar for digits.
        let grammar = r#"start: /[0-9]+/"#;
        let mut guide = GrammarGuide::from_ebnf(grammar, &factory).unwrap();
        let allowed = guide.allowed_tokens();
        assert!(allowed.is_some());
        assert!(!allowed.unwrap().is_empty());
    }

    #[test]
    fn test_from_guided_grammar_ebnf_variant() {
        let factory = make_test_factory();
        let grammar = GuidedGrammar::Ebnf {
            grammar: r#"start: /[0-9]+/"#.to_string(),
        };
        let mut guide = GrammarGuide::from_guided_grammar(&grammar, &factory).unwrap();
        let allowed = guide.allowed_tokens();
        assert!(allowed.is_some());
        assert!(!allowed.unwrap().is_empty());
    }
}
