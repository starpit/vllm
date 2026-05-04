// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen2 / Qwen2.5 — same math as Llama except the QKV projections
//! carry a learned bias term. The bias is explicit in the DSL as a
//! `bias_add` tile; the solver's `FusedQkvRopeCacheImpl` /
//! `FusedQkvRopePrefillImpl` claim the `(Gemm, BiasAdd) × 3 + RopeAppend`
//! pattern and emit cuBLAS `gemm_bias` on the packed weight — so the
//! bias rides through one fused kernel launch, not a separate add.
//!
//! One `#[forward]` body per architecture; per-model configs fan out
//! via `crates/ferrite-model-qwen2/configs/*.json`. The `qwen2-vl-2b.json`
//! config registers this text-decoder forward for arch
//! `Qwen2VLForConditionalGeneration`; the vision tower for that arch
//! lives in the sibling `ferrite-model-qwen2-vl` crate, which contributes
//! its own `MultimodalForward` registration via `inventory::submit!`.
//! Qwen2.5-VL is `ferrite-model-qwen2-5-vl` (different vision math).

use ferrite_forward::forward;

#[forward(
    workloads = [1, 8, 64, 512, 4096],
    sk_buckets = [128, 512, 2048, 8192],
)]
fn qwen2() {
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
        gate = silu(gemm(normed2, mlp.gate_proj[layer]));
        up = gemm(normed2, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);
        hidden_states = add(down, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = gemm(normed, lm_head);
}
