// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! LLaVA-1.5 multimodal vision tower (CLIP-ViT-L/14 @ 336² + 2-layer
//! MLP projector). The text decoder is Vicuna (Llama-class) and lives
//! in `ferrite-model-llama` (registered for arch
//! `LlavaForConditionalGeneration` via this crate's
//! `decoder_safetensors_prefix = "language_model"`).
//!
//! Compared to Gemma3-MM the SigLIP→AvgPool projector is replaced
//! with a CLS-strip + 2-layer MLP, and the encoder MLP uses
//! QuickGELU instead of GELU-tanh. Otherwise the encoder block math
//! is structurally identical: full LayerNorm γ+β, separate q/k/v/out
//! projections all with biases, full per-image attention via
//! `varlen_attention(q, k, v, cu_seqlens, max_seqlen)`.
//!
//! Two CLIP-isms handled host-side (in the wrapper, NOT in the DSL
//! body) so the body stays a clean ~50 lines:
//!
//! 1. **Class token (CLS).** CLIP prepends a learnable `[1,
//!    embed_dim]` `class_embedding` to the patch sequence pre-encoder.
//!    Rather than introducing a `cls_prepend` op, we fold
//!    `class_embedding` into `position_embedding[0]` at load time
//!    (CPU-side add of the rank-1 class_embedding into row 0 of the
//!    rank-2 position_embedding) and pad pixels with a leading zero
//!    row in `pixel_pack`. With `bias=False` on
//!    `embeddings.patch_embedding` (CLIP convention), the gemm
//!    output's row 0 is zero; the subsequent `add(pos_emb)` injects
//!    `class_embedding + position_embedding[0]` at row 0 — the same
//!    value HF computes via concat-then-add. The math is exact.
//!
//! 2. **Penultimate-layer tap.** HF tags `vision_feature_layer = -2`,
//!    which means the LM consumes the output of block 22 (out of 24).
//!    The variant config sets `vision_depth = 23`, so the loader
//!    pulls weights for blocks 0..22 only and the encoder loop runs
//!    those 23 blocks. Block 23's weights stay on disk un-consumed,
//!    and `post_layernorm` is never applied (matching HF behavior
//!    for `vision_feature_select_strategy = "default"`).
//!
//! The remaining HF-ism is post-encoder: HF strips the CLS row
//! (`hidden_states[:, 1:]`) before the projector. That's the one new
//! DSL primitive Phase H adds: `strip_cls(x: [vision_num_positions, e])
//! -> [vision_in_seq_len, e]`. The matched `Instruction::StripCls`
//! lowers to a single D2D copy of rows `1..L`.

#[cfg(feature = "cuda")]
use ferrite_forward::vision_forward;

/// LLaVA-1.5 CPU preprocessing. Combines the CLIP image processor
/// (shortest-edge resize to 336, center-crop to 336², CLIP mean/std
/// normalization) with the `RepeatMarker` placeholder policy. The HF
/// chat template emits the image-marker token (`image_token_index` =
/// 32000) per image, and the processor expands it to 576 copies of
/// token 32000 (= patch grid `(image_size / patch_size)² = (336 /
/// 14)² = 576`, matching `vision_in_seq_len`). Tokens-per-image is
/// fixed; the `FromConfig` policy reads it via `_get_image_seq_length`
/// math against `hf_config.vision_config`.
pub const PROCESSOR: ferrite_vision::MmMetadata = ferrite_vision::MmMetadata {
    hf_token_id_key: "image_token_index",
    hf_token_id_default: 32000,
    size_policy: ferrite_vision::SizePolicy::FixedSquare,
    tokens_per_image: ferrite_vision::TokensPerImage::FromConfig,
    preprocess: ferrite_vision::preprocess::preprocess_clip_normalized,
    default_image_size: 336,
    chat_template_image_part_type: "image",
    placeholder_policy: ferrite_vision::PlaceholderPolicy::RepeatMarker,
    // LLaVA-1.5 / Vicuna text decoder uses standard 1D RoPE.
    mrope_positions: false,
};

/// CLIP-class pixel packer with CLS row prepended. Calls into
/// [`ferrite_vision::VisionConfig::pack_pixels_clip_with_cls`] which
/// produces `[1 + L, feat]` bf16 patches with row 0 zero (CLS slot)
/// and rows `1..1+L` the natural row-major patches. Required for
/// LLaVA-1.5 because the default `pixel_pack` returns `[L, feat]`
/// without the CLS row; without this override the wrapper would
/// upload `vision_in_seq_len` rows to the encoder, which expects
/// `vision_num_positions`.
#[cfg(feature = "cuda")]
pub fn pack_pixels_with_cls(
    cfg: &ferrite_vision::VisionConfig,
    pixels: &[f32],
    height: u32,
    width: u32,
) -> (Vec<u16>, (u32, u32, u32)) {
    cfg.pack_pixels_clip_with_cls(pixels, height, width)
}

#[cfg(feature = "cuda")]
#[vision_forward(
    workloads = [256, 1024, 4096, 16384],
    processor = crate::PROCESSOR,
    pixel_pack = crate::pack_pixels_with_cls,
)]
fn llava_1_5() {
    // Patch embed: Conv2d(in=3, out=1024, k=14, s=14), `bias=False`.
    // The patch_embedding.weight ships as 4D `[1024, 3, 14, 14]` and
    // is flattened to 2D `[1024, 588]` at load time via the
    // `vision_patch_embed_flatten` config. Pixels are uploaded as
    // `[vision_num_positions=577, vision_in_features=588]` with row 0
    // zero-padded (host-side in pixel_pack); rows 1..577 carry the
    // 24×24 = 576 real patches in row-major order.
    hidden_states = gemm(pixels, embeddings.patch_embedding);

    // Learned positional embedding lookup. `position_ids` is a
    // vision-prelude extern of `[num_tokens=577]` u32 the wrapper
    // builds as `[0, 1, ..., 576]` per image. The table itself is the
    // `[577, 1024]` learned weight, with row 0 pre-folded host-side
    // to be `position_embedding[0] + class_embedding` (CLS injection).
    pos_emb = pos_embed(position_ids, embeddings.position_embedding);
    hidden_states = add(hidden_states, pos_emb);

    // Pre-encoder LayerNorm γ+β. PyTorch `nn.LayerNorm` decomposes
    // as `(mean, sub, rmsnorm, bias_add)`; the
    // `MeanSubRmsNormBiasAddImpl` matcher claims the 4-tile chain
    // and lowers to one `kernels::layer_norm_bias` call. Note the
    // HF typo `pre_layrnorm` (sic — preserved for safetensors-key
    // compatibility).
    m0 = mean(hidden_states);
    c0 = sub(hidden_states, m0);
    n0 = rmsnorm(c0, pre_layrnorm);
    hidden_states = bias_add(n0, pre_layrnorm.bias);

    for layer in 0..vision_depth {
        // Pre-LN attention. CLIP uses full per-image attention with
        // separate q/k/v/out_proj projections, all biased.
        m1 = mean(hidden_states);
        c1 = sub(hidden_states, m1);
        n1 = rmsnorm(c1, layer_norm1[layer]);
        normed = bias_add(n1, layer_norm1.bias[layer]);

        q = gemm(normed, self_attn.q_proj[layer]);
        q = bias_add(q, self_attn.q_proj.bias[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        k = bias_add(k, self_attn.k_proj.bias[layer]);
        v = gemm(normed, self_attn.v_proj[layer]);
        v = bias_add(v, self_attn.v_proj.bias[layer]);

        // Full attention per-image. Single `cu_seqlens` segments the
        // batched-flat tensor at image boundaries (one segment per
        // image, max_seqlen = vision_num_positions = 577).
        attn_out = varlen_attention(q, k, v, cu_seqlens, max_seqlen);

        oproj = gemm(attn_out, self_attn.out_proj[layer]);
        oproj = bias_add(oproj, self_attn.out_proj.bias[layer]);
        hidden_states = add(oproj, hidden_states);

        // Pre-LN MLP (QuickGELU, biases on both fc1 / fc2). CLIP
        // historically uses QuickGELU (= x * sigmoid(1.702 * x))
        // rather than the erf-based GELU.
        m2 = mean(hidden_states);
        c2 = sub(hidden_states, m2);
        n2 = rmsnorm(c2, layer_norm2[layer]);
        normed2 = bias_add(n2, layer_norm2.bias[layer]);

        fc1 = gemm(normed2, mlp.fc1[layer]);
        fc1 = bias_add(fc1, mlp.fc1.bias[layer]);
        fc1 = quick_gelu(fc1);
        fc2 = gemm(fc1, mlp.fc2[layer]);
        fc2 = bias_add(fc2, mlp.fc2.bias[layer]);
        hidden_states = add(fc2, hidden_states);
    }

    // CLS strip: drop row 0 (the encoder ran 577 tokens, but the LM
    // consumes 576 image tokens to match `_get_image_seq_length`).
    // No `post_layernorm` — `vision_feature_layer = -2` taps the
    // encoder output before HF's terminal LN.
    hidden_states = strip_cls(hidden_states);

    // Multimodal projector: Linear (1024 → 4096) + GELU (erf) + Linear
    // (4096 → 4096). HF default `projector_hidden_act = "gelu"` maps
    // to `nn.GELU()` with no `approximate=` arg → erf-based GELU.
    proj1 = gemm(hidden_states, proj.linear_1);
    proj1 = bias_add(proj1, proj.linear_1.bias);
    proj1 = gelu_erf(proj1);
    proj2 = gemm(proj1, proj.linear_2);
    out = bias_add(proj2, proj.linear_2.bias);
}
