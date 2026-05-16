// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen2-MoE / Qwen1.5-MoE (`Qwen2MoeForCausalLM`) — Mistral-style
//! attention math (no per-head q/k norm, unlike Qwen3-MoE) with the
//! dense SwiGLU MLP replaced by a fused MoE block + shared expert.
//!
//! On-disk MoE prefix: `model.layers.{l}.mlp` (matches HF's
//! safetensors layout — the default `safetensors_prefix` mapping
//! handles it). Routed experts: `experts.{e}.{gate,up,down}_proj.weight`;
//! shared expert: `mlp.shared_expert.{gate,up,down}_proj.weight`
//! with sigmoid gate at `mlp.shared_expert_gate.weight`.
//!
//! `Qwen2MoeForCausalLM` produces both biased QKV (`bias=True` on
//! q/k/v_proj) and shared-expert layers — distinct from Qwen3-MoE's
//! per-head norm path. The single L4-fitting coherent fixture is
//! `JacobAndersson/slimed-qwen-2`, a 2-layer trim of
//! `Qwen/Qwen1.5-MoE-A2.7B-Chat` that preserves the full vocab,
//! hidden, and 60-expert + shared-expert MoE block.
//!
//! Reference: `vllm-cuda/src/model/qwen2_moe.rs`. Ferrite ships the
//! BF16 path here; FP8 / quantized variants land in a follow-up
//! alongside their dedicated Impls (same shape as the Mixtral and
//! Qwen3-MoE quant follow-ups).
//!
//! Scope: this crate handles the all-MoE-layers configuration only.
//! Configs whose `mlp_only_layers` lists a non-empty prefix of dense
//! layers (or `decoder_sparse_step > 1`) aren't covered yet — same
//! restriction as `ferrite-model-qwen3-moe`. Qwen1.5-MoE-A2.7B and
//! its slimmed variants ship `decoder_sparse_step=1` and empty
//! `mlp_only_layers`, so the all-MoE assumption holds.

use ferrite_forward_macro::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
)]
fn qwen2_moe() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        normed = rmsnorm(hidden_states, input_layernorm[layer]);
        q = gemm(normed, self_attn.q_proj[layer]);
        q = bias_add(q, self_attn.q_proj.bias[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        k = bias_add(k, self_attn.k_proj.bias[layer]);
        v = gemm(normed, self_attn.v_proj[layer]);
        v = bias_add(v, self_attn.v_proj.bias[layer]);
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        attn = attention(q, k, v, kv_cache[layer], block_table);
        oproj = gemm(attn, self_attn.o_proj[layer]);
        hidden_states = add(oproj, hidden_states);

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        mlp_out = moe_block(normed2, mlp[layer]);
        hidden_states = add(mlp_out, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = gemm(normed, lm_head);
}
