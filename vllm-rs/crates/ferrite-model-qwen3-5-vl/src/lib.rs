// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen3.5-VL vision tower. The text decoder is the Gated-DeltaNet hybrid in
//! `ferrite-model-qwen3-5` (arch `Qwen3_5ForConditionalGeneration`); this crate
//! contributes only the vision-side `MultimodalForward` registration via the
//! `#[vision_forward]` macro — no hand-written code beyond the DSL body.
//!
//! Vision arch (verified vs mlx-vlm qwen3_vl/vision.py + the HF checkpoint):
//! - 27 blocks, embed 1152, 16 heads → head_dim 72, half_rot 36.
//! - norms = LayerNorm WITH BIAS (norm1/norm2/merger.norm) — the
//!   `(mean, sub, rmsnorm, bias_add)` 4-tile chain (see qwen2-vl).
//! - fused `attn.qkv` [3*1152, 1152] → packed-split to q/k/v at load.
//! - block MLP: `linear_fc1 → gelu (gelu_pytorch_tanh) → linear_fc2` (biased).
//! - single bidirectional varlen attention (NO window, NO deepstack).
//! - merger: LayerNorm → linear_fc1 → gelu → linear_fc2 (→ d_model 4096).
//! - block arrangement validated == mlx golden (cosine ~1.0):
//!   vllm-rs/tools/vision_parity/validate_block.py.
//!
//! TODO (P4 wiring): the learned `pos_embed` (`fast_pos_embed_interpolate`,
//! 4-corner bilinear over a 48×48 grid) is computed host-side and added after
//! patch_embed — not yet in the DSL (needs a `pos_embeds` runtime extern).

#[cfg(any(feature = "cuda", feature = "metal"))]
use ferrite_forward_macro::vision_forward;

/// Qwen3.5-VL CPU preprocessing. patch 16 × spatial_merge 2 → smart-resize
/// factor 32, CLIP normalization, `<|image_pad|>` placeholder, per-image grid.
pub const PROCESSOR: ferrite_vision::MmMetadata = ferrite_vision::MmMetadata {
    hf_token_id_key: "image_token_id",
    hf_token_id_default: 151655,
    size_policy: ferrite_vision::SizePolicy::SmartResize {
        factor: 32,
        default_min_pixels: 3136,
        default_max_pixels: 12_845_056,
    },
    tokens_per_image: ferrite_vision::TokensPerImage::PerImageGrid {
        spatial_merge_default: 2,
    },
    preprocess: ferrite_vision::preprocess::preprocess_clip_normalized,
    default_image_size: 384,
    chat_template_image_part_type: "image",
    placeholder_policy: ferrite_vision::PlaceholderPolicy::RepeatMarker,
    mrope_positions: true,
};

#[cfg(any(feature = "cuda", feature = "metal"))]
#[vision_forward(workloads = [256, 1024, 4096, 16384], processor = crate::PROCESSOR)]
fn qwen3_5_vl() {
    hidden_states = gemm(pixels, patch_embed.proj);
    hidden_states = bias_add(hidden_states, patch_embed.proj.bias);
    // TODO(P4): hidden_states = add(hidden_states, pos_embeds);  // host-interp

    for layer in 0..vision_depth {
        // LayerNorm-with-bias as the (mean, sub, rmsnorm, bias_add) 4-tile chain.
        m1 = mean(hidden_states);
        c1 = sub(hidden_states, m1);
        n1 = rmsnorm(c1, norm1[layer]);
        normed = bias_add(n1, norm1.bias[layer]);
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

        m2 = mean(hidden_states);
        c2 = sub(hidden_states, m2);
        n2 = rmsnorm(c2, norm2[layer]);
        normed2 = bias_add(n2, norm2.bias[layer]);
        fc1 = gemm(normed2, mlp.linear_fc1[layer]);
        fc1 = bias_add(fc1, mlp.linear_fc1.bias[layer]);
        fc1 = gelu(fc1);
        fc2 = gemm(fc1, mlp.linear_fc2[layer]);
        fc2 = bias_add(fc2, mlp.linear_fc2.bias[layer]);
        hidden_states = add(fc2, hidden_states);
    }

    mq = mean(hidden_states);
    cq = sub(hidden_states, mq);
    nq = rmsnorm(cq, merger.norm);
    merged = bias_add(nq, merger.norm.bias);
    merged = reshape(
        merged,
        [num_tokens / vision_merge_factor, vision_merge_hidden],
    );
    mlp0 = gemm(merged, merger.linear_fc1);
    mlp0 = bias_add(mlp0, merger.linear_fc1.bias);
    mlp0 = gelu(mlp0);
    out = gemm(mlp0, merger.linear_fc2);
    out = bias_add(out, merger.linear_fc2.bias);
}
