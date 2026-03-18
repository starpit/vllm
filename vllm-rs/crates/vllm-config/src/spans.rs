// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Configuration for relocatable KV cache blocks ("spans").
//!
//! Spans enable position-independent KV cache block reuse, primarily for RAG
//! workloads where document blocks can be preloaded and shared across requests
//! regardless of their position in the sequence.
//!
//! All settings are read from environment variables to match the Python vLLM
//! convention (`envs.VLLM_V1_SPANS_*`).

use serde::{Deserialize, Serialize};

/// Configuration for relocatable KV cache blocks (spans).
///
/// When enabled, special tokens control span behavior:
/// - **`token_plus`** (`SPAN_TOK_PLUS`): Marks the start of a fan-in span.
///   Blocks beginning with this token get `parent_block_hash = NONE_HASH`,
///   allowing the block to be cached and reused independently of preceding
///   tokens.
/// - **`token_cross`** (`SPAN_TOK_CROSS`): Marks a span that depends on all
///   previous tokens. Forces recomputation if any prior context differs.
///
/// Keys are stored in the KV cache *without* rotary position embedding (RoPE).
/// Position-specific rotation is applied at attention time, making blocks
/// relocatable to any sequence position.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpansConfig {
    /// Master switch: enables all span functionality.
    /// Env: `VLLM_V1_SPANS_ENABLED` (default: `false`).
    pub enabled: bool,

    /// Emit debug logging for span operations (hash resets, recomputation, etc.).
    /// Env: `VLLM_V1_SPANS_DEBUG` (default: `false`).
    pub debug: bool,

    /// Token ID that marks the start of a fan-in span.
    /// Blocks whose first token matches this value have their parent hash
    /// reset to `NONE_HASH`, enabling position-independent caching.
    /// Env: `VLLM_V1_SPANS_TOKEN_PLUS` (default: `None`).
    pub token_plus: Option<u32>,

    /// Token ID that marks a cross-context span.
    /// Blocks whose first token matches this value include all preceding
    /// tokens in their hash, forcing recomputation when context differs.
    /// Env: `VLLM_V1_SPANS_TOKEN_CROSS` (default: `None`).
    pub token_cross: Option<u32>,

    /// Disable position adjustment (RoPE fusion) in the attention kernel.
    /// When `true`, keys are stored and read with their original position
    /// encoding (i.e., spans are hashed but not relocatable).
    /// Env: `VLLM_V1_SPANS_DISABLE_REPOSITION` (default: `false`).
    pub disable_reposition: bool,
}

impl SpansConfig {
    /// Build a `SpansConfig` from environment variables.
    ///
    /// | Variable | Type | Default |
    /// |---|---|---|
    /// | `VLLM_V1_SPANS_ENABLED` | bool | `false` |
    /// | `VLLM_V1_SPANS_DEBUG` | bool | `false` |
    /// | `VLLM_V1_SPANS_TOKEN_PLUS` | u32 | None |
    /// | `VLLM_V1_SPANS_TOKEN_CROSS` | u32 | None |
    /// | `VLLM_V1_SPANS_DISABLE_REPOSITION` | bool | `false` |
    pub fn from_env() -> Self {
        Self {
            enabled: parse_bool_env("VLLM_V1_SPANS_ENABLED"),
            debug: parse_bool_env("VLLM_V1_SPANS_DEBUG"),
            token_plus: parse_u32_env("VLLM_V1_SPANS_TOKEN_PLUS"),
            token_cross: parse_u32_env("VLLM_V1_SPANS_TOKEN_CROSS"),
            disable_reposition: parse_bool_env("VLLM_V1_SPANS_DISABLE_REPOSITION"),
        }
    }

    /// Returns `true` if fan-in is configured (enabled + token_plus set).
    #[inline]
    pub fn has_fan_in(&self) -> bool {
        self.enabled && self.token_plus.is_some()
    }

    /// Returns `true` if cross-context recomputation is configured.
    #[inline]
    pub fn has_cross(&self) -> bool {
        self.enabled && self.token_cross.is_some()
    }

    /// Returns `true` if RoPE should be fused into attention (not applied
    /// during prefill key computation).
    #[inline]
    pub fn fuse_rope(&self) -> bool {
        self.enabled && !self.disable_reposition
    }
}

/// Parse a boolean from an environment variable.
/// Accepts "true", "True", "1" as truthy; everything else (including unset) is false.
fn parse_bool_env(key: &str) -> bool {
    std::env::var(key)
        .map(|v| matches!(v.as_str(), "true" | "True" | "1"))
        .unwrap_or(false)
}

/// Parse an optional u32 from an environment variable.
/// Returns `None` if unset or unparseable.
fn parse_u32_env(key: &str) -> Option<u32> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_spans_config() {
        let cfg = SpansConfig::default();
        assert!(!cfg.enabled);
        assert!(!cfg.debug);
        assert!(cfg.token_plus.is_none());
        assert!(cfg.token_cross.is_none());
        assert!(!cfg.disable_reposition);
        assert!(!cfg.has_fan_in());
        assert!(!cfg.has_cross());
        assert!(!cfg.fuse_rope());
    }

    #[test]
    fn test_spans_config_helpers() {
        let cfg = SpansConfig {
            enabled: true,
            debug: false,
            token_plus: Some(10),
            token_cross: Some(31),
            disable_reposition: false,
        };
        assert!(cfg.has_fan_in());
        assert!(cfg.has_cross());
        assert!(cfg.fuse_rope());

        let cfg_no_reposition = SpansConfig {
            disable_reposition: true,
            ..cfg.clone()
        };
        assert!(!cfg_no_reposition.fuse_rope());
    }

    #[test]
    fn test_spans_config_roundtrip() {
        let cfg = SpansConfig {
            enabled: true,
            debug: true,
            token_plus: Some(10),
            token_cross: Some(31),
            disable_reposition: false,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: SpansConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.enabled, cfg2.enabled);
        assert_eq!(cfg.token_plus, cfg2.token_plus);
        assert_eq!(cfg.token_cross, cfg2.token_cross);
    }

    #[test]
    fn test_from_env() {
        // Save and restore env state
        let keys = [
            "VLLM_V1_SPANS_ENABLED",
            "VLLM_V1_SPANS_DEBUG",
            "VLLM_V1_SPANS_TOKEN_PLUS",
            "VLLM_V1_SPANS_TOKEN_CROSS",
            "VLLM_V1_SPANS_DISABLE_REPOSITION",
        ];
        let saved: Vec<_> = keys.iter().map(|k| std::env::var(k).ok()).collect();

        // SAFETY: test is single-threaded; we save and restore all modified vars.
        unsafe {
            std::env::set_var("VLLM_V1_SPANS_ENABLED", "true");
            std::env::set_var("VLLM_V1_SPANS_DEBUG", "1");
            std::env::set_var("VLLM_V1_SPANS_TOKEN_PLUS", "10");
            std::env::set_var("VLLM_V1_SPANS_TOKEN_CROSS", "31");
            std::env::set_var("VLLM_V1_SPANS_DISABLE_REPOSITION", "false");
        }

        let cfg = SpansConfig::from_env();
        assert!(cfg.enabled);
        assert!(cfg.debug);
        assert_eq!(cfg.token_plus, Some(10));
        assert_eq!(cfg.token_cross, Some(31));
        assert!(!cfg.disable_reposition);

        // Restore env
        unsafe {
            for (key, val) in keys.iter().zip(saved.iter()) {
                match val {
                    Some(v) => std::env::set_var(key, v),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}
