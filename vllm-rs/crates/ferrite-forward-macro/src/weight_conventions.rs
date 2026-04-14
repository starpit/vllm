// SPDX-License-Identifier: Apache-2.0
//! Standard HuggingFace transformer conventions.
//!
//! Every decoder-only LLM in the HF ecosystem (Llama, Qwen2,
//! Mistral, Phi, DeepSeek, Gemma, …) agrees on two sets of rules
//! that aren't written down in any individual model file but are
//! understood by every HF loader (including `transformers`
//! itself). This module encodes both:
//!
//! 1. **Weight-name → shape conventions** — `self_attn.q_proj`
//!    has shape `[hidden_size, num_attention_heads * head_dim]`,
//!    `mlp.down_proj` has shape `[intermediate_size, hidden_size]`,
//!    and so on. [`standard_shape`] exposes the lookup.
//!
//! 2. **Config.json field defaults** — when a config.json omits a
//!    field, HF's loaders substitute a derived value. E.g. if
//!    `head_dim` is missing, it's `hidden_size / num_attention_heads`.
//!    If `num_key_value_heads` is missing, it defaults to
//!    `num_attention_heads` (multi-head attention, i.e. not GQA).
//!    [`derive_implicit_bounds`] applies these defaults to the
//!    bounds map loaded from a config.json.
//!
//! Both sets live together because they're aspects of the same
//! protocol and they reference the same vocabulary of field names.
//! Keeping them in one file means when a new arch needs an
//! extension (e.g. Gemma2's `query_pre_attn_scalar`), there's one
//! obvious place to add it.

use std::collections::BTreeMap;

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

/// Runtime weight path for a DSL weight reference.
///
/// HF safetensors files name the backbone weights under `model.`
/// and place `lm_head` at the root. Per-layer weights get a
/// `model.layers.<i>.` prefix. This fn encodes that convention so
/// the compiler doesn't carry a model.rs-specific layout rule in
/// its code.
///
/// Examples:
///
/// - `["self_attn", "q_proj"]`, layer=Some(3)  → `model.layers.3.self_attn.q_proj`
/// - `["embed_tokens"]`, layer=None            → `model.embed_tokens`
/// - `["norm"]`, layer=None                    → `model.norm`
/// - `["lm_head"]`, layer=None                 → `lm_head`
///
/// The returned string is the *prefix* passed to
/// `Linear::load` / `Embedding::load` / `RmsNorm::load`; the
/// runtime loaders append `.weight` / `.bias` as appropriate.
pub fn hf_weight_prefix(path: &[String], layer: Option<usize>) -> String {
    let dotted = path.join(".");
    if let Some(i) = layer {
        return format!("model.layers.{i}.{dotted}");
    }
    // Global: lm_head is at the root; everything else lives under
    // `model.`. If more HF-convention exceptions surface (e.g.
    // embeddings in some MoE configs), add them here.
    if dotted == "lm_head" {
        dotted
    } else {
        format!("model.{dotted}")
    }
}

/// Apply HF's implicit config.json defaults to the bounds map.
///
/// HF configs are allowed to omit certain fields that have
/// well-defined defaults. A loader that didn't apply these would
/// look at `llama-2-13b/config.json` (which lists `hidden_size`
/// and `num_attention_heads` but NOT `head_dim`) and correctly
/// conclude the shape inference can't close `num_heads * head_dim`.
/// Applying the default unblocks shape inference without any
/// per-arch special case — the defaults are the same across every
/// HF transformer.
///
/// Defaults applied here:
///
/// - **`head_dim`** → `hidden_size / num_attention_heads`. Llama-2
///   and Llama-3 (not 3.2+) omit the field; Llama-3.2+, Qwen2.5+,
///   and Gemma2 list it explicitly. Both forms are valid HF JSON.
///
/// - **`num_key_value_heads`** → `num_attention_heads`. Older
///   configs written before grouped-query attention existed assume
///   multi-head attention (kv_heads == heads) and don't list the
///   field.
///
/// Defaults are only written if the field is missing — an explicit
/// value in the JSON always wins.
pub fn derive_implicit_bounds(bounds: &mut BTreeMap<String, u64>) {
    // head_dim: hidden_size / num_attention_heads. Only derive if
    // the division is exact; a non-divisible config is malformed and
    // we'd rather surface it downstream than silently round.
    if !bounds.contains_key("head_dim")
        && let (Some(&hidden), Some(&heads)) =
            (bounds.get("hidden_size"), bounds.get("num_attention_heads"))
        && heads != 0
        && hidden.is_multiple_of(heads)
    {
        bounds.insert("head_dim".to_string(), hidden / heads);
    }

    // num_key_value_heads: num_attention_heads (MHA default, no GQA).
    if !bounds.contains_key("num_key_value_heads")
        && let Some(&heads) = bounds.get("num_attention_heads")
    {
        bounds.insert("num_key_value_heads".to_string(), heads);
    }
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

    #[test]
    fn head_dim_derived_from_hidden_over_heads() {
        // Matches llama-2-13b: hidden=5120, heads=40 → head_dim=128.
        let mut b = BTreeMap::new();
        b.insert("hidden_size".into(), 5120);
        b.insert("num_attention_heads".into(), 40);
        derive_implicit_bounds(&mut b);
        assert_eq!(b.get("head_dim"), Some(&128));
    }

    #[test]
    fn explicit_head_dim_wins_over_default() {
        // Matches llama-3.2-1b: explicit head_dim that's NOT
        // hidden/heads (2048/32 == 64, matches here, but the rule
        // is that explicit wins regardless).
        let mut b = BTreeMap::new();
        b.insert("hidden_size".into(), 2048);
        b.insert("num_attention_heads".into(), 32);
        b.insert("head_dim".into(), 999);
        derive_implicit_bounds(&mut b);
        assert_eq!(b.get("head_dim"), Some(&999));
    }

    #[test]
    fn num_kv_heads_defaults_to_num_heads_when_missing() {
        let mut b = BTreeMap::new();
        b.insert("num_attention_heads".into(), 40);
        derive_implicit_bounds(&mut b);
        assert_eq!(b.get("num_key_value_heads"), Some(&40));
    }

    #[test]
    fn hf_weight_prefix_per_layer() {
        let p = hf_weight_prefix(&["self_attn".into(), "q_proj".into()], Some(3));
        assert_eq!(p, "model.layers.3.self_attn.q_proj");
    }

    #[test]
    fn hf_weight_prefix_global_under_model() {
        assert_eq!(
            hf_weight_prefix(&["embed_tokens".into()], None),
            "model.embed_tokens"
        );
        assert_eq!(hf_weight_prefix(&["norm".into()], None), "model.norm");
    }

    #[test]
    fn hf_weight_prefix_lm_head_at_root() {
        assert_eq!(hf_weight_prefix(&["lm_head".into()], None), "lm_head");
    }

    #[test]
    fn explicit_kv_heads_wins() {
        // GQA config: 32 heads but only 8 KV heads.
        let mut b = BTreeMap::new();
        b.insert("num_attention_heads".into(), 32);
        b.insert("num_key_value_heads".into(), 8);
        derive_implicit_bounds(&mut b);
        assert_eq!(b.get("num_key_value_heads"), Some(&8));
    }
}
