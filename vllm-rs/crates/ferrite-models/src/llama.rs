// SPDX-License-Identifier: Apache-2.0
//! LLaMA model architecture via the `#[forward]` attribute macro.
//!
//! The macro compiles the DSL body below against every JSON in
//! `model_architectures/llama/` at each workload bucket, emitting
//! `pub mod llama::<model_ident>` with:
//!   - a `WeightBundle` trait the caller implements to expose weights,
//!   - `forward_m_<N>` fns — one per workload bucket,
//!   - a dispatching `forward(wm, ctx, device, num_tokens)` fn.

use ferrite_forward::forward;

#[forward(
    models_dir = "../../../model_architectures/llama",
    target = "../../../target_profiles/l4_sm89.json",
    workloads = [1, 8, 64, 512, 4096],
)]
fn llama() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        normed = rmsnorm(hidden_states, input_layernorm[layer]);
        q = gemm(normed, self_attn.q_proj[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        v = gemm(normed, self_attn.v_proj[layer]);
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
