// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen3-MoE (`Qwen3MoeForCausalLM`) — Qwen3 attention math (per-head
//! q/k RMSNorm) with the dense SwiGLU MLP replaced by a fused MoE
//! block + optional shared expert.
//!
//! On-disk MoE prefix: `model.layers.{l}.mlp` (the dotted name `mlp`
//! matches HF's safetensors layout exactly — the default
//! `safetensors_prefix` mapping handles it). Routed experts use the
//! standard `experts.{e}.{gate,up,down}_proj.weight` naming; the
//! shared expert lives at `mlp.shared_expert.{gate,up,down}_proj.weight`
//! with the sigmoid gate at `mlp.shared_expert_gate.weight`.
//!
//! Reference: `vllm-cuda/src/model/qwen3_moe.rs`. Ferrite ships the
//! BF16 path here; FP8 / quantized variants land alongside their
//! dedicated Impls in a follow-up.
//!
//! Scope: this crate handles the all-MoE-layers configuration only.
//! Configs whose `mlp_only_layers` lists a non-empty prefix of dense
//! layers (Qwen3-Next-style hybrid) are deliberately not covered yet
//! — they would need either a layer-index-conditioned DSL split or a
//! per-layer DSL like DeepSeek-V2's `first_k_dense_replace`. None of
//! the official Qwen3-MoE-Instruct checkpoints ship a non-empty
//! `mlp_only_layers`, so the all-MoE assumption holds for the
//! current consumers.

use ferrite_forward::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
)]
fn qwen3_moe() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        normed = rmsnorm(hidden_states, input_layernorm[layer]);
        q = gemm(normed, self_attn.q_proj[layer]);
        q = rmsnorm(q, self_attn.q_norm[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        k = rmsnorm(k, self_attn.k_norm[layer]);
        v = gemm(normed, self_attn.v_proj[layer]);
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
