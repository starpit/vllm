// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Gemma 4 (`Gemma4UnifiedForConditionalGeneration` text decoder) — a
//! hybrid of sliding-window and global attention layers whose two classes
//! differ in dims (the first supported arch to do so):
//!
//!   * **Sliding layers** (5 of every 6; `layer % 6 != 5`): head_dim 256,
//!     8 KV heads, window 1024, rope theta 1e4 full-rotary
//!     (`rotary_local`), weighted per-head q/k norms + UNIT-gain v norm
//!     (`rmsnorm_unit`, mlx `RMSNormNoScale`) before the cache write.
//!   * **Global layers** (`layer % 6 == 5`, idx 5,11,…,47): head_dim
//!     **512**, **1** KV head, `attention_k_eq_v` — ONE `k_proj`
//!     projection feeds both K and V: K = proportional-rope(k_norm(kv))
//!     rotating only the first 128 of 512 dims with the freq exponent
//!     denominator = the FULL head_dim (mlx `ProportionalRoPE`,
//!     theta 1e6, `rotary` extern); V = `rmsnorm_unit(kv)`, NOT roped.
//!     The `*_global` DSL names map back to the shared on-disk leaves
//!     (`self_attn.q_proj` etc.) via `weight_leaf_renames` — same disk
//!     name, different shape per class.
//!
//! Attention softmax scale is **1.0 for BOTH classes** (mlx sets
//! `self.scale = 1.0` unconditionally) — expressed via
//! `attention_multiplier: 1.0` in the config (the Granite path).
//!
//! Other Gemma-isms: embed scaled by sqrt(hidden); FOUR sandwich norms
//! per layer (post-attention/post-feedforward norms apply BEFORE the
//! residual adds); GeGLU MLP (`gelu_pytorch_tanh`); a per-layer
//! `layer_scalar` ([1]-shaped weight) multiplying the hidden state at
//! the end of every layer; tied quantized embeddings as lm_head; final
//! logit softcapping `tanh(x/30)*30`. Norm weights are stored FULL
//! (mlx-vlm `RMSNormZeroShift`) — no Gemma-2/3 `(1+w)` offset.
//!
//! On-disk text weights live under `language_model.model.*`
//! (`decoder_safetensors_prefix: "language_model"`); the checkpoint's
//! `vision_embedder.*` / `embed_audio.*` / `embed_vision.*` towers are
//! never referenced (text-only path). Quantization is MLX-affine 4-bit
//! g64 with every `mlp.{gate,up,down}_proj` at 8-bit g64
//! (`mlx-affine-b4-g64-mlp8` preset); scales/biases/norm gains are BF16
//! (`is_bf16_scale_arch` covers `Gemma4*`).
//!
//! Oracle: `~/git/mlx-vlm/mlx_vlm/models/gemma4/` (the converter);
//! `mlx-lm gemma4_text.py` is the equivalent text-only reference.
//! Verification target: `mlx-community/gemma-4-12B-it-4bit`.

use ferrite_forward_macro::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
)]
fn gemma4() {
    hidden_states = embed(input_ids, embed_tokens) * sqrt(hidden_size);
    for layer in 0..num_hidden_layers {
        pre_attn_normed = rmsnorm(hidden_states, input_layernorm[layer]);

        // Branch-local variable names are deliberately DISTINCT per
        // class — the if/else merge unifies same-named assignments
        // across branches, and the two classes' q/k/v widths differ
        // (sliding 16×256 vs global 16×512). The branches converge at
        // `oproj`, which is [.., hidden_size] in both.
        if layer % sliding_window_pattern == sliding_window_global_remainder {
            // ── GLOBAL class: head_dim 512, 1 kv head, k_eq_v ──
            qg = gemm(pre_attn_normed, self_attn.q_proj_global[layer]);
            qg = rmsnorm(qg, self_attn.q_norm_global[layer]);
            kvg = gemm(pre_attn_normed, self_attn.k_proj_global[layer]);
            kg = rmsnorm(kvg, self_attn.k_norm_global[layer]);
            vg = rmsnorm_unit(kvg);
            (qg, kg, vg) = rope_append(qg, kg, vg, positions, rotary, kv_cache[layer]);
            attng = attention(qg, kg, vg, kv_cache[layer], block_table);
            oproj = gemm(attng, self_attn.o_proj_global[layer]);
        } else {
            // ── SLIDING class: head_dim 256, 8 kv heads, window 1024 ──
            qs = gemm(pre_attn_normed, self_attn.q_proj[layer]);
            qs = rmsnorm(qs, self_attn.q_norm[layer]);
            ks = gemm(pre_attn_normed, self_attn.k_proj[layer]);
            ks = rmsnorm(ks, self_attn.k_norm[layer]);
            vs = rmsnorm_unit(gemm(pre_attn_normed, self_attn.v_proj[layer]));
            (qs, ks, vs) = rope_append(qs, ks, vs, positions, rotary_local, kv_cache[layer]);
            attns = sliding_attention(qs, ks, vs, kv_cache[layer], block_table);
            oproj = gemm(attns, self_attn.o_proj[layer]);
        }

        post_attn_normed = rmsnorm(oproj, post_attention_layernorm[layer]);
        hidden_states = add(post_attn_normed, hidden_states);

        pre_ffwd_normed = rmsnorm(hidden_states, pre_feedforward_layernorm[layer]);
        gate = gelu(gemm(pre_ffwd_normed, mlp.gate_proj[layer]));
        up = gemm(pre_ffwd_normed, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);
        post_ffwd_normed = rmsnorm(down, post_feedforward_layernorm[layer]);
        hidden_states = add(post_ffwd_normed, hidden_states);

        hidden_states = scalar_weight_mul(hidden_states, layer_scalar[layer]);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = tanh_softcap(gemm(normed, lm_head));
}
