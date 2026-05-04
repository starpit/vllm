// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen3-Next (`Qwen3NextForCausalLM`) — hybrid GDN + full-attention decoder.
//!
//! Layer types alternate by transformer-layer index. The default
//! pattern is `linear_attention` for layers where `(i + 1) % 4 != 0`
//! and `full_attention` otherwise — i.e. three GDN layers followed by
//! one full-attention layer, repeating. Both `full_attn_period` and
//! `full_attn_remainder` are advertised on the per-arch `model.json`
//! so the DSL `if`-predicate can name them.
//!
//! Architecture highlights:
//! 1. **Gated Delta Net** linear attention on most layers — recurrent
//!    state pool (conv1d + SSM) lives on `ForwardCtx::gdn_state`.
//! 2. **Full attention with output gate** on every fourth layer —
//!    doubled-Q projection (Q + sigmoid gate), per-head Gemma-style
//!    Q/K RMSNorm, partial RoPE on Q (FA2 rotates K on read), and
//!    a sigmoid output gate before `o_proj`.
//! 3. **Qwen-MoE** routed + shared-expert MLP on every layer (per
//!    the official 80B-A3B-Instruct / 80B-A3B-Thinking checkpoints
//!    `decoder_sparse_step == 1`, `mlp_only_layers == []`).
//!
//! Reference: `vllm-cuda/src/model/qwen3_next.rs`. This crate
//! ships the BF16 path; FP8 / GGUF / quantized variants land
//! alongside their dedicated Impls in a follow-up.

use ferrite_forward::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
)]
fn qwen3_next() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        normed = rmsnorm(hidden_states, input_layernorm[layer]);
        if layer % full_attn_period == full_attn_remainder {
            attn_out = gated_attention(normed, self_attn[layer], positions, rotary, kv_cache[layer], block_table);
        } else {
            attn_out = gdn_attention(normed, linear_attn[layer]);
        }
        hidden_states = add(attn_out, hidden_states);

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        mlp_out = moe_block(normed2, mlp[layer]);
        hidden_states = add(mlp_out, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = gemm(normed, lm_head);
}
