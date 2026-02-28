// SPDX-License-Identifier: Apache-2.0
//! Grammar-guided constrained decoding.
//!
//! Wraps `outlines-core` to compile JSON schemas (or generic JSON) into
//! finite-state automata that mask logits during sampling. Each request
//! with a `GuidedGrammar` gets its own `GrammarGuide` that tracks the
//! current FSM state and provides the set of allowed tokens at each step.

use outlines_core::index::Index;
use outlines_core::json_schema;
use outlines_core::prelude::{StateId, TokenId, Vocabulary};
use vllm_common::sampling::GuidedGrammar;

/// A compiled grammar guide that tracks FSM state for constrained decoding.
///
/// Created once per request (when `guided_grammar` is set), then queried
/// and advanced at each decode step.
pub struct GrammarGuide {
    index: Index,
    current_state: StateId,
}

/// A generic JSON regex that matches any valid JSON value.
///
/// This is used for `response_format: { type: "json_object" }` which
/// requires valid JSON output without a specific schema.
const JSON_OBJECT_SCHEMA: &str = r#"{"type": "object"}"#;

impl GrammarGuide {
    /// Build a grammar guide from a JSON schema.
    ///
    /// Compiles `schema` → regex → FSM index using the provided vocabulary.
    pub fn from_json_schema(
        schema: &serde_json::Value,
        vocabulary: &Vocabulary,
    ) -> Result<Self, String> {
        let schema_str = serde_json::to_string(schema).map_err(|e| e.to_string())?;
        let regex =
            json_schema::regex_from_str(&schema_str, None, None).map_err(|e| e.to_string())?;
        let index = Index::new(&regex, vocabulary).map_err(|e| e.to_string())?;
        let initial_state = index.initial_state();
        Ok(Self {
            index,
            current_state: initial_state,
        })
    }

    /// Build a grammar guide from an arbitrary regex pattern.
    ///
    /// Compiles the regex directly into an FSM index (no JSON schema step).
    pub fn from_regex(pattern: &str, vocabulary: &Vocabulary) -> Result<Self, String> {
        let index = Index::new(pattern, vocabulary).map_err(|e| e.to_string())?;
        let initial_state = index.initial_state();
        Ok(Self {
            index,
            current_state: initial_state,
        })
    }

    /// Build a grammar guide for generic JSON object output.
    ///
    /// Uses a built-in `{"type": "object"}` schema that accepts any JSON object.
    pub fn from_json_object(vocabulary: &Vocabulary) -> Result<Self, String> {
        let schema: serde_json::Value =
            serde_json::from_str(JSON_OBJECT_SCHEMA).map_err(|e| e.to_string())?;
        Self::from_json_schema(&schema, vocabulary)
    }

    /// Create a `GrammarGuide` from a `GuidedGrammar` specification.
    pub fn from_guided_grammar(
        grammar: &GuidedGrammar,
        vocabulary: &Vocabulary,
    ) -> Result<Self, String> {
        match grammar {
            GuidedGrammar::Json => Self::from_json_object(vocabulary),
            GuidedGrammar::JsonSchema { schema } => Self::from_json_schema(schema, vocabulary),
            GuidedGrammar::Regex { pattern } => Self::from_regex(pattern, vocabulary),
        }
    }

    /// Get the set of token IDs allowed at the current state.
    ///
    /// Returns `None` if the current state has no transitions (should not
    /// happen during normal generation — it means the FSM is stuck).
    pub fn allowed_tokens(&self) -> Option<Vec<TokenId>> {
        self.index.allowed_tokens(&self.current_state)
    }

    /// Advance the FSM to the next state given a sampled token.
    ///
    /// Returns `true` if the transition was valid, `false` if the token
    /// had no valid transition (the state is left unchanged).
    pub fn advance(&mut self, token_id: TokenId) -> bool {
        if let Some(next) = self.index.next_state(&self.current_state, &token_id) {
            self.current_state = next;
            true
        } else {
            false
        }
    }

    /// Check if the current state is a final (accepting) state.
    pub fn is_finished(&self) -> bool {
        self.index.is_final_state(&self.current_state)
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

/// Build an `outlines-core` `Vocabulary` from a list of `(token_id, token_string)` pairs.
///
/// Workers call this once at model-load time and cache the result.
pub fn build_vocabulary(tokens: &[(u32, String)], eos_token_id: u32) -> Result<Vocabulary, String> {
    let mut vocab = Vocabulary::new(eos_token_id);
    for (token_id, token_str) in tokens {
        if *token_id == eos_token_id {
            continue;
        }
        vocab
            .try_insert(token_str.as_str(), *token_id)
            .map_err(|e| format!("failed to insert token {token_id} ({token_str:?}): {e}"))?;
    }
    Ok(vocab)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vocab() -> Vocabulary {
        // Minimal vocabulary for testing: digits, braces, quotes, colon, comma.
        let eos = 99;
        let mut vocab = Vocabulary::new(eos);
        let tokens = [
            (0, "{"),
            (1, "}"),
            (2, "\""),
            (3, ":"),
            (4, ","),
            (5, " "),
            (6, "0"),
            (7, "1"),
            (8, "2"),
            (9, "a"),
            (10, "b"),
            (11, "n"),
            (12, "u"),
            (13, "l"),
            (14, "t"),
            (15, "r"),
            (16, "e"),
            (17, "f"),
            (18, "s"),
            (19, "["),
            (20, "]"),
            (21, "."),
            (22, "-"),
            (23, "3"),
            (24, "4"),
            (25, "5"),
            (26, "6"),
            (27, "7"),
            (28, "8"),
            (29, "9"),
        ];
        for (id, tok) in tokens {
            vocab.try_insert(tok, id).unwrap();
        }
        vocab
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
    fn test_build_vocabulary() {
        let tokens = vec![
            (0, "hello".to_string()),
            (1, "world".to_string()),
            (2, "eos".to_string()),
        ];
        let vocab = build_vocabulary(&tokens, 2).unwrap();
        assert!(vocab.token_ids("hello").is_some());
        assert!(vocab.token_ids("world").is_some());
        // EOS token should not be inserted.
        assert!(vocab.token_ids("eos").is_none());
    }

    #[test]
    fn test_from_json_schema_compiles() {
        let vocab = make_test_vocab();
        let schema: serde_json::Value = serde_json::from_str(
            r#"{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}"#,
        )
        .unwrap();
        let guide = GrammarGuide::from_json_schema(&schema, &vocab);
        // May fail with IncompatibleVocabulary for minimal vocab, that's OK.
        // We're testing that the compilation path doesn't panic.
        match guide {
            Ok(g) => {
                // Should have allowed tokens at initial state.
                let allowed = g.allowed_tokens();
                assert!(allowed.is_some());
                assert!(!allowed.unwrap().is_empty());
            }
            Err(e) => {
                // IncompatibleVocabulary is expected with a tiny vocab.
                assert!(
                    e.contains("incompatible")
                        || e.contains("Incompatible")
                        || e.contains("vocabulary"),
                    "Unexpected error: {e}"
                );
            }
        }
    }

    #[test]
    fn test_from_regex_digit_pattern() {
        let eos = 99;
        let mut vocab = Vocabulary::new(eos);
        for (id, tok) in [(0, "0"), (1, "1"), (2, "2"), (3, "3")] {
            vocab.try_insert(tok, id).unwrap();
        }
        let guide = GrammarGuide::from_regex("[0-3]+", &vocab).unwrap();
        let allowed = guide.allowed_tokens().unwrap();
        assert!(!allowed.is_empty());
    }

    #[test]
    fn test_from_guided_grammar_regex_variant() {
        let eos = 99;
        let mut vocab = Vocabulary::new(eos);
        for (id, tok) in [(0, "0"), (1, "1"), (2, "2"), (3, "3")] {
            vocab.try_insert(tok, id).unwrap();
        }
        let grammar = GuidedGrammar::Regex {
            pattern: "[0-3]+".to_string(),
        };
        let guide = GrammarGuide::from_guided_grammar(&grammar, &vocab).unwrap();
        let allowed = guide.allowed_tokens().unwrap();
        assert!(!allowed.is_empty());
    }

    #[test]
    fn test_advance_and_finish() {
        // Use a simple integer regex to test state machine transitions.
        let eos = 99;
        let mut vocab = Vocabulary::new(eos);
        for (id, tok) in [(0, "0"), (1, "1"), (2, "2"), (3, "3")] {
            vocab.try_insert(tok, id).unwrap();
        }

        let regex = "0|[1-3][0-3]*";
        let index = Index::new(regex, &vocab).unwrap();
        let mut guide = GrammarGuide {
            current_state: index.initial_state(),
            index,
        };

        // At initial state, we should have allowed tokens.
        let allowed = guide.allowed_tokens().unwrap();
        assert!(!allowed.is_empty());

        // Advance with "1" (token_id=1).
        assert!(guide.advance(1));
        // After "1", the guide should be in a final state (valid integer).
        assert!(guide.is_finished());

        // Can still advance with more digits.
        let allowed2 = guide.allowed_tokens();
        assert!(allowed2.is_some());
    }
}
