// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Tokenizer wrapper around the HuggingFace `tokenizers` crate.
//!
//! Provides encoding (text → token IDs) and decoding (token IDs → text)
//! for use in input processing and incremental detokenization.
//!
//! Port of: `vllm/tokenizers/` (subset)

use std::collections::HashSet;
use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;

use crate::error::{ServeError, ServeResult};

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// A wrapper around the HuggingFace `tokenizers::Tokenizer` that provides
/// the core encode/decode interface needed by the serving layer.
pub struct Tokenizer {
    inner: HfTokenizer,
    /// Cached set of special token IDs for fast lookup.
    special_ids: HashSet<u32>,
    /// The EOS token ID, if present.
    eos_token_id: Option<u32>,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> ServeResult<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| ServeError::Engine(format!("Failed to load tokenizer: {e}")))?;
        Ok(Self::from_hf_tokenizer(inner))
    }

    /// Create a `Tokenizer` from a raw HuggingFace `Tokenizer` instance.
    pub fn from_hf_tokenizer(inner: HfTokenizer) -> Self {
        let special_ids = Self::collect_special_ids(&inner);
        let eos_token_id = inner
            .token_to_id("</s>")
            .or_else(|| inner.token_to_id("<|endoftext|>"))
            .or_else(|| inner.token_to_id("<|end_of_text|>"));
        Self {
            inner,
            special_ids,
            eos_token_id,
        }
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> ServeResult<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| ServeError::Engine(format!("Tokenization failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> ServeResult<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| ServeError::Engine(format!("Detokenization failed: {e}")))
    }

    /// Get the string representation of a single token ID.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Get the token ID for a string token.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the EOS token ID, if known.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Check if a token ID is a special token.
    pub fn is_special_token(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    /// Get the full vocabulary as a token-to-id mapping.
    pub fn get_vocab(&self) -> std::collections::HashMap<String, u32> {
        self.inner.get_vocab(true)
    }

    /// Get a reference to the underlying HuggingFace tokenizer.
    pub fn inner(&self) -> &HfTokenizer {
        &self.inner
    }

    /// Collect all special token IDs from the tokenizer.
    fn collect_special_ids(tokenizer: &HfTokenizer) -> HashSet<u32> {
        let mut ids = HashSet::new();
        let added = tokenizer.get_added_tokens_decoder();
        for (&id, token) in &added {
            if token.special {
                ids.insert(id);
            }
        }
        ids
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("vocab_size", &self.vocab_size())
            .field("eos_token_id", &self.eos_token_id)
            .field("num_special_tokens", &self.special_ids.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Create a simple BPE tokenizer for testing.
///
/// Uses a byte-level pre-tokenizer (GPT-2 style) with the full 256 byte-token
/// vocabulary. No merges — each byte is its own token. Good enough for testing
/// encode/decode roundtrips.
#[cfg(test)]
pub(crate) fn make_test_tokenizer() -> Tokenizer {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    // Build vocabulary JSON from the byte-level alphabet.
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort();

    let mut vocab_entries = Vec::new();
    for (i, &c) in alphabet.iter().enumerate() {
        // Escape the character for JSON.
        let escaped = serde_json::to_string(&c.to_string()).unwrap();
        vocab_entries.push(format!("{escaped}:{i}"));
    }
    let vocab_json = format!("{{{}}}", vocab_entries.join(","));

    // Build a complete tokenizer JSON with byte-level BPE.
    let tokenizer_json = format!(
        r#"{{
            "version": "1.0",
            "model": {{
                "type": "BPE",
                "vocab": {vocab_json},
                "merges": []
            }},
            "pre_tokenizer": {{
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            }},
            "decoder": {{
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": true,
                "use_regex": true
            }}
        }}"#
    );

    let hf_tok: HfTokenizer = tokenizer_json.parse().unwrap();
    Tokenizer::from_hf_tokenizer(hf_tok)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode_decode_roundtrip() {
        let tok = make_test_tokenizer();
        let text = "Hello, world!";
        let ids = tok.encode(text, false).unwrap();
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_tokenizer_encode_empty() {
        let tok = make_test_tokenizer();
        let ids = tok.encode("", false).unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_tokenizer_decode_empty() {
        let tok = make_test_tokenizer();
        let text = tok.decode(&[], false).unwrap();
        assert_eq!(text, "");
    }

    #[test]
    fn test_tokenizer_vocab_size() {
        let tok = make_test_tokenizer();
        // Byte-level BPE has at least 256 byte tokens.
        assert!(tok.vocab_size() >= 256);
    }

    #[test]
    fn test_tokenizer_debug() {
        let tok = make_test_tokenizer();
        let debug_str = format!("{tok:?}");
        assert!(debug_str.contains("Tokenizer"));
        assert!(debug_str.contains("vocab_size"));
    }

    #[test]
    fn test_tokenizer_multibyte_text() {
        let tok = make_test_tokenizer();
        let text = "こんにちは世界";
        let ids = tok.encode(text, false).unwrap();
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids, false).unwrap();
        assert_eq!(decoded, text);
    }
}
