// SPDX-License-Identifier: Apache-2.0
//! Qwen2-VL vision encoder body. The DSL describes the GPU graph;
//! the macro emits per-variant `Weights` + `forward` + the
//! `VisionArchWeights` impl + `try_load_mm` + inventory rows. No
//! hand-written `MultimodalForward` / `try_load_mm` / inventory
//! submits — host-side glue is generic in `ferrite-forward`.

use ferrite_forward::vision_forward;

#[vision_forward(
    workloads = [256, 1024, 4096, 16384],
    pixel_pack = ferrite_vision::pack_qwen2_vl,
)]
fn qwen2_vl() {
    hidden_states = gemm(pixels, patch_embed.proj);

    for layer in 0..vision_depth {
        normed = layer_norm_bias(hidden_states, norm1[layer]);
        q = gemm(normed, attn.q[layer]);
        q = bias_add(q, attn.q.bias[layer]);
        k = gemm(normed, attn.k[layer]);
        k = bias_add(k, attn.k.bias[layer]);
        v = gemm(normed, attn.v[layer]);
        v = bias_add(v, attn.v.bias[layer]);
        (q, k) = vision_rope(q, k, cos, sin);
        attn_out = varlen_attention(q, k, v, cu_seqlens, max_seqlen);
        oproj = gemm(attn_out, attn.proj[layer]);
        oproj = bias_add(oproj, attn.proj.bias[layer]);
        hidden_states = add(oproj, hidden_states);

        normed2 = layer_norm_bias(hidden_states, norm2[layer]);
        fc1 = gemm(normed2, mlp.fc1[layer]);
        fc1 = bias_add(fc1, mlp.fc1.bias[layer]);
        fc1 = quick_gelu(fc1);
        fc2 = gemm(fc1, mlp.fc2[layer]);
        fc2 = bias_add(fc2, mlp.fc2.bias[layer]);
        hidden_states = add(fc2, hidden_states);
    }

    merged = layer_norm_bias(hidden_states, merger.ln_q);
    merged = reshape(
        merged,
        [num_tokens / vision_merge_factor, vision_merge_hidden],
    );
    mlp0 = gemm(merged, merger.mlp_0);
    mlp0 = bias_add(mlp0, merger.mlp_0.bias);
    mlp0 = gelu_erf(mlp0);
    out = gemm(mlp0, merger.mlp_2);
    out = bias_add(out, merger.mlp_2.bias);
}
