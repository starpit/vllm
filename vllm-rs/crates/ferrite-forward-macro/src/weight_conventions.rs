// SPDX-License-Identifier: Apache-2.0
//! Standard HuggingFace transformer weight-name conventions.
//!
//! Every decoder-only LLM in the HF ecosystem (Llama, Qwen2,
//! Mistral, Phi, DeepSeek, Gemma, …) uses the same set of weight
//! names for the same roles with shapes expressible in the same
//! config.json bound vocabulary. This module encodes those
//! conventions as a table so shape inference has an anchor for
//! every standard weight — including MLP weights whose dims
//! ("intermediate_size") aren't pinned by any op signature.
//!
//! Weights not in this table are left unanchored by this module;
//! a later phase can supply a per-arch `weights.json` override
//! when an arch has non-standard weights (MoE routers, vision
//! patch embeddings, etc.). For every arch currently in scope
//! (Llama, Qwen2, Gemma2) the table alone is sufficient.

use crate::shape::{Dim, Shape};

/// Look up the conventional shape for a dotted weight path.
///
/// Returns `None` for paths not covered by the standard HF
/// convention — callers should leave those shapes as-inferred.
pub fn standard_shape(path: &[String]) -> Option<Shape> {
    let dotted = path.join(".");
    Some(match dotted.as_str() {
        // Token embeddings and output projection.
        "embed_tokens" => vec![bound("vocab_size"), bound("hidden_size")],
        "lm_head" => vec![bound("hidden_size"), bound("vocab_size")],

        // Attention block.
        "self_attn.q_proj" => {
            vec![bound("hidden_size"), heads("num_attention_heads")]
        }
        "self_attn.k_proj" => {
            vec![bound("hidden_size"), heads("num_key_value_heads")]
        }
        "self_attn.v_proj" => {
            vec![bound("hidden_size"), heads("num_key_value_heads")]
        }
        "self_attn.o_proj" => {
            vec![heads("num_attention_heads"), bound("hidden_size")]
        }

        // Norms — both in-attention (input) and in-MLP (post_attention)
        // plus the final pre-lm_head norm.
        "input_layernorm" => vec![bound("hidden_size")],
        "post_attention_layernorm" => vec![bound("hidden_size")],
        "norm" => vec![bound("hidden_size")],

        // MLP block.
        "mlp.gate_proj" => vec![bound("hidden_size"), bound("intermediate_size")],
        "mlp.up_proj" => vec![bound("hidden_size"), bound("intermediate_size")],
        "mlp.down_proj" => vec![bound("intermediate_size"), bound("hidden_size")],

        _ => return None,
    })
}

fn bound(name: &str) -> Dim {
    Dim::Bound(name.into())
}

/// `num_heads * head_dim` as a canonical Mul.
fn heads(num_heads_bound: &str) -> Dim {
    // Order must match canonical_mul's sort (Bound variants sort
    // lexicographically). head_dim < num_*_heads alphabetically.
    Dim::Mul(vec![bound("head_dim"), bound(num_heads_bound)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn attention_weights_covered() {
        assert!(standard_shape(&path(&["self_attn", "q_proj"])).is_some());
        assert!(standard_shape(&path(&["self_attn", "k_proj"])).is_some());
        assert!(standard_shape(&path(&["self_attn", "v_proj"])).is_some());
        assert!(standard_shape(&path(&["self_attn", "o_proj"])).is_some());
    }

    #[test]
    fn mlp_weights_covered() {
        let gate = standard_shape(&path(&["mlp", "gate_proj"])).unwrap();
        assert_eq!(gate, vec![bound("hidden_size"), bound("intermediate_size")]);
        let down = standard_shape(&path(&["mlp", "down_proj"])).unwrap();
        assert_eq!(down, vec![bound("intermediate_size"), bound("hidden_size")]);
    }

    #[test]
    fn norm_weights_covered() {
        for name in ["input_layernorm", "post_attention_layernorm", "norm"] {
            let s = standard_shape(&path(&[name])).unwrap();
            assert_eq!(s, vec![bound("hidden_size")], "{name}");
        }
    }

    #[test]
    fn embed_and_lm_head_covered() {
        assert_eq!(
            standard_shape(&path(&["embed_tokens"])).unwrap(),
            vec![bound("vocab_size"), bound("hidden_size")]
        );
        assert_eq!(
            standard_shape(&path(&["lm_head"])).unwrap(),
            vec![bound("hidden_size"), bound("vocab_size")]
        );
    }

    #[test]
    fn nonstandard_path_returns_none() {
        assert!(standard_shape(&path(&["mystery_weight"])).is_none());
        assert!(standard_shape(&path(&["self_attn", "mystery"])).is_none());
        assert!(standard_shape(&path(&["expert", "router"])).is_none());
    }
}
