// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! ModernBERT — encoder-only, bidirectional MaskedLM transformer.
//!
//! Differences from the decoder fleet:
//!
//! 1. **Encoder attention.** Bidirectional self-attention with no
//!    causal mask and no KV cache reuse. Expressed as the 3-arg
//!    `attention(q, k, v)` form claimed by `EncoderAttentionImpl`
//!    (1b). The DSL still threads the rotated Q/K/V through
//!    `rope_append` because rotary application + per-token K/V
//!    storage share that op's signature; the per-step KV writes
//!    on the encoder side land in scratch storage and are never
//!    consumed by a downstream `attention(..., kv_cache, ...)` tile.
//!
//! 2. **Dual rotary, no sliding mask.** Every Nth layer
//!    (`layer % global_attn_every_n_layers == 0`) uses the global
//!    rope theta; the rest use the local theta. The encoder body
//!    runs full bidirectional attention on both — the local
//!    layers' window-size constant is not yet enforced kernel-side
//!    (deferred until a windowed encoder Impl lands).
//!
//! 3. **CohereLayerNorm with no bias.** ModernBERT-base ships
//!    `norm_bias: false`, so each LayerNorm is the math primitive
//!    trio `(mean, sub, rmsnorm)` claimed by `MeanSubRmsNormImpl`
//!    (Phase 0). The 4-tile `MeanSubRmsNormBiasAdd` fusion (1c) is
//!    available but unused on this checkpoint.
//!
//! 4. **Layer 0 identity attn pre-norm.** The HF checkpoint omits
//!    `model.layers.0.attn_norm` — the embeddings layer-norm
//!    already normalized the residual stream, so layer 0 feeds
//!    `hidden_states` directly into Q/K/V. Expressed as
//!    `if layer < 1 { normed = hidden_states * 1.0; } else { ... }`:
//!    both branches bind `normed`, the `* 1.0` is a structural
//!    passthrough (claimed by `ScalarMulImpl`) so the merge-carry
//!    typecheck is satisfied without forcing layer 0 to load an
//!    `attn_norm` weight that doesn't exist.
//!
//! 5. **GeGLU MLP.** `down(gelu(gate) * up)` — same fusion shape as
//!    Gemma3, with `gelu` instead of `silu`. The on-disk
//!    `mlp.Wi.weight` is fused `[2*intermediate, hidden]`; the
//!    manifest's `__packed_splits__` carves it into virtual
//!    `mlp.gate_proj` / `mlp.up_proj` row-slices at load time.
//!
//! 6. **Packed `attn.Wqkv`.** ModernBERT ships the QKV projection
//!    as one fused matrix (`[3*hidden, hidden]`). The manifest's
//!    `__packed_splits__` splits it into virtual `attn.q_proj` /
//!    `attn.k_proj` / `attn.v_proj` so the DSL reads three
//!    separate refs (matching every other arch's Q/K/V shape).
//!
//! 7. **Encoder backbone terminator.** The forward returns
//!    hidden states `[num_tokens, hidden_size]` after the final
//!    `final_norm`. No `lm_head` tile — `backbone_layout` (1d)
//!    classifies this as Encoder because the last tile is RmsNorm,
//!    not a `gemm(_, lm_head)`. The MaskedLM head (`head.dense`,
//!    `head.norm`, tied `decoder.weight + decoder.bias`) is wired
//!    by 1f's dispatch layer outside the `#[forward]` body.
//!
//! Reference: hand-written `vllm-cuda/src/model/modernbert.rs`.
//! Probe target: `answerdotai/ModernBERT-base`.

use ferrite_forward_macro::forward;

#[forward()]
mod modernbert {
    // ModernBERT spells its dual rotary bases `global_rope_theta` /
    // `local_rope_theta`; the per-field readers (and the emitted
    // `rotary` / `rotary_local` cache ctors) consume the
    // gemma-convention `rope_theta` / `rope_local_base_freq` names.
    // Alias, don't rename — the config stays verbatim and an explicit
    // standard-name field still wins.
    const CONFIG_ALIASES: &[(&str, &str)] = &[
        ("rope_local_base_freq", "local_rope_theta"),
        ("rope_theta", "global_rope_theta"),
    ];

    fn forward() {
        // Embedding lookup + initial CohereLayerNorm.
        hidden_states = embed(input_ids, embeddings.tok_embeddings);
        hidden_states = rmsnorm(sub(hidden_states, mean(hidden_states)), embeddings.norm);

        for layer in 0..num_hidden_layers {
            // Attn pre-norm. Layer 0 is identity (HF checkpoint has no
            // `attn_norm.weight` for layer 0; the embeddings LN already
            // did the work). The `* 1.0` passthrough satisfies the
            // if/else merge-carry rule (both branches must bind
            // `normed`) without needing a layer-0 weight load.
            if layer < 1 {
                normed = hidden_states * 1.0;
            } else {
                normed = rmsnorm(sub(hidden_states, mean(hidden_states)), attn_norm[layer]);
            }

            q = gemm(normed, attn.q_proj[layer]);
            k = gemm(normed, attn.k_proj[layer]);
            v = gemm(normed, attn.v_proj[layer]);

            // Dual rotary: every Nth layer rotates with the global
            // theta cache, the rest with the local-theta cache. The
            // 3-arg `attention(q, k, v)` form picks
            // `EncoderAttentionImpl` (bidirectional FA2, no causal mask,
            // no KV cache read) — the kv_cache extern threaded into
            // `rope_append` is the per-step write path; the encoder
            // never reads those stored K/V back.
            if layer % global_attn_every_n_layers == 0 {
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                attn_out = attention(q, k, v);
            } else {
                (q, k, v) = rope_append(q, k, v, positions, rotary_local, kv_cache[layer]);
                attn_out = attention(q, k, v);
            }
            oproj = gemm(attn_out, attn.Wo[layer]);
            hidden_states = add(oproj, hidden_states);

            // MLP pre-norm + GeGLU + residual.
            normed = rmsnorm(sub(hidden_states, mean(hidden_states)), mlp_norm[layer]);
            gate = gelu(gemm(normed, mlp.gate_proj[layer]));
            up = gemm(normed, mlp.up_proj[layer]);
            down = gemm(gate * up, mlp.Wo[layer]);
            hidden_states = add(down, hidden_states);
        }

        // Final encoder norm — backbone terminator (1d). The output of
        // this RmsNorm is `[num_tokens, hidden_size]`; the dispatch
        // layer reads it via `forward_backbone` (1f).
        hidden_states = rmsnorm(sub(hidden_states, mean(hidden_states)), final_norm);
    }
}
