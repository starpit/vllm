// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen3.5-MoE (`Qwen3_5MoeForConditionalGeneration` text decoder) — the
//! Gated-DeltaNet + periodic full-attention hybrid of `ferrite-model-qwen3-5`
//! (`full_attention_interval = 4`; full layers at index `l % 4 == 3`;
//! `attn_output_gate=True` doubled q_proj) with the dense SwiGLU MLP replaced
//! on EVERY layer (`mlp_only_layers = []`) by a sparse MoE block:
//!   * **Routed experts** via `moe_block` — softmax router → top-8 of 256
//!     experts → `norm_topk_prob` renorm → SwitchGLU (`moe_intermediate_size
//!     = 512`) → weighted sum. This is the exact routed-only configuration
//!     proven on Metal by Qwen3-MoE-30B-A3B: the verbatim HF config DOES
//!     carry `shared_expert_intermediate_size: 512`, but the DSL body below
//!     owns the shared expert (`mlp.shared_expert.*` subtree), so the
//!     single-owner rule (`has_subtree(["mlp", "shared_expert"])` in
//!     `metal/moe.rs` + codegen's `plan_field_load`) zeroes the MoE
//!     payload's shared width — the `lower_metal_moe` routed-only arm
//!     applies and `AffineSharedFusedMoELayer::load` skips shared-weight
//!     loading (no double-load).
//!   * **Shared expert** (inter = 512, `mlp.shared_expert.*`) expressed as
//!     ordinary DSL: a quantized SwiGLU MLP plus a `[T,1]` sigmoid gate
//!     (`mlp.shared_expert_gate`), combined via `gate_scale(routed, shared_y,
//!     g) = routed + shared_y * sigmoid(g)` (row-broadcast).
//!
//! On-disk text weights live under `language_model.model.*` (set via
//! `decoder_safetensors_prefix`); `tie_word_embeddings=False` with a
//! quantized `embed_tokens` → the `qembed` preset, like the dense 9B. Vision
//! (`model.visual.*`) and MTP (`mtp.*`) weights are ignored — text-only path.
//! Routed experts ship pre-stacked (`mlp.switch_mlp.{gate,up,down}_proj`,
//! `[256, out, in/8]`), the modern `AffineFusedMoELayer` naming path.
//!
//! Reference: mlx-lm `qwen3_5.py` + `qwen3_next.py` `Qwen3NextSparseMoeBlock`;
//! Python vLLM `qwen3_5.py` (`model_type == "qwen3_5_moe_text"`).

use ferrite_forward_macro::forward;

// No `workloads`: uses the global default ladder (`DEFAULT_DECODER_WORKLOADS`
// in ferrite-forward-macro), compiled in full for every device and pruned at
// load time by `select_prefill_bucket` to the largest bucket that fits while
// leaving a KV floor. This GDN+256-expert MoE prunes to 512 on a 32 GiB box
// (KV preserved) and keeps the ladder up to 4096 on a large one — no
// per-device magic constant. The dense GDN sibling (`ferrite-model-qwen3-5`)
// and the MoE sibling (`ferrite-model-qwen2-moe`) run the same ladder with the
// same gated_delta_net / gather-GEMM bodies, so neither is the limit.
#[forward]
mod qwen3_5_moe {
    /// Checkpoints ship BF16 RMSNorm gains (Qwen3-family
    /// convention) — the metal gain-reader symbols must match.
    const SCALE_DTYPE: ScaleDtype = ScaleDtype::Bf16;
    // Qwen3.5 `*RMSNorm` stores zero-centered gains (`x * (1 + w)`);
    // ferrite keeps the on-disk form and offsets in-kernel. The
    // sparse-MoE router renormalizes top-k weights — newer configs
    // omit `norm_topk_prob` and the modeling code defaults it TRUE.
    const BOUND_DEFAULTS: &[(&str, u64)] = &[("rms_norm_zero_centered", 1), ("norm_topk_prob", 1)];
    // `Qwen3_5MoeForConditionalGeneration` nests the text decoder
    // under `language_model.*` on disk.
    const DECODER_PREFIX: &str = "language_model";

    fn forward() {
        hidden_states = embed(input_ids, embed_tokens);
        for layer in 0..num_hidden_layers {
            normed = rmsnorm(hidden_states, input_layernorm[layer]);

            // Token mixer: GDN linear attention for `l % interval != 3`,
            // full attention (with output gate) for the periodic `l % interval == 3`.
            if layer % full_attention_interval != 3 {
                // ── Gated-DeltaNet linear attention ─────────────────────
                qkv = gemm(normed, linear_attn.in_proj_qkv[layer]);
                z = gemm(normed, linear_attn.in_proj_z[layer]);
                a = gemm(normed, linear_attn.in_proj_a[layer]);
                b = gemm(normed, linear_attn.in_proj_b[layer]);
                core = gated_delta_net(qkv, z, a, b, linear_attn[layer]);
                mixer_out = gemm(core, linear_attn.out_proj[layer]);
            } else {
                // ── Full attention with sigmoid output gate ─────────────
                // q_proj is doubled; gate_split deinterleaves per-head
                // [query | gate]. q/k get per-head RMSNorm; partial RoPE.
                qg = gemm(normed, self_attn.q_proj[layer]);
                (q, gate) = gate_split(qg);
                q = rmsnorm(q, self_attn.q_norm[layer]);
                k = gemm(normed, self_attn.k_proj[layer]);
                k = rmsnorm(k, self_attn.k_norm[layer]);
                v = gemm(normed, self_attn.v_proj[layer]);
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
                attn = attention(q, k, v, kv_cache[layer], block_table);
                // Sigmoid output gate: attn * sigmoid(gate), fused (bare `*` /
                // `sigmoid` aren't DSL-callable — Silu/Mul are synthesis-only).
                attn = gate_apply(attn, gate);
                mixer_out = gemm(attn, self_attn.o_proj[layer]);
            }
            hidden_states = add(mixer_out, hidden_states);

            // Sparse MoE MLP: routed experts (softmax → top-8 → renorm →
            // SwitchGLU → weighted sum, all inside moe_block) + the always-on
            // shared expert — a quantized SwiGLU MLP scaled by the per-token
            // sigmoid gate ([T, 1], row-broadcast inside gate_scale).
            normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
            routed = moe_block(normed2, mlp[layer]);
            shared_y = gemm(
                silu(gemm(normed2, mlp.shared_expert.gate_proj[layer]))
                    * gemm(normed2, mlp.shared_expert.up_proj[layer]),
                mlp.shared_expert.down_proj[layer],
            );
            g = gemm(normed2, mlp.shared_expert_gate[layer]);
            mlp_out = gate_scale(routed, shared_y, g);
            hidden_states = add(mlp_out, hidden_states);
        }
        normed = rmsnorm(hidden_states, norm);
        logits = gemm(normed, lm_head);
    }
}
