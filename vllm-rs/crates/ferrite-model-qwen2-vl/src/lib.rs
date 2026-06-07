// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen2-VL vision tower. The text decoder is shared with plain Qwen2
//! and lives in `ferrite-model-qwen2` (registered for arch
//! `Qwen2VLForConditionalGeneration` via that crate's
//! `configs/qwen2-vl-2b.json`). This crate contributes only the
//! vision-side `MultimodalForward` registration, which is emitted by
//! the `#[vision_forward]` macro — no hand-written code beyond the
//! DSL body itself.
//!
//! Qwen2.5-VL has materially different vision math (window attention,
//! RMSNorm in vision blocks, revised 2D RoPE) and lives in its own
//! crate (`ferrite-model-qwen2-5-vl`). Both VL crates rely on
//! `ferrite-model-qwen2` for the shared text decoder.

#[cfg(feature = "cuda")]
use ferrite_forward_macro::vision_forward;

/// Per-arch CPU preprocessing declaration baked into every emitted
/// `FerriteMmRegistration` row by the `#[vision_forward(processor = ...)]`
/// arg. Qwen2-VL uses smart-resize (post-resize dims vary per image)
/// with CLIP-mean/std normalization; placeholder token id is the
/// `<|image_pad|>` token id from `hf_config.image_token_id` (defaults
/// to 151655). Tokens-per-image is per-image grid divided by spatial
/// merge size (= 2 for every Qwen2-VL variant).
pub const PROCESSOR: ferrite_vision::MmMetadata = ferrite_vision::MmMetadata {
    hf_token_id_key: "image_token_id",
    hf_token_id_default: 151655,
    size_policy: ferrite_vision::SizePolicy::SmartResize {
        // patch_size(14) · spatial_merge_size(2)
        factor: 28,
        // HF Qwen2VLImageProcessor defaults; per-checkpoint
        // `preprocessor_config.json` overrides at init time.
        default_min_pixels: 3136,
        default_max_pixels: 12_845_056,
    },
    tokens_per_image: ferrite_vision::TokensPerImage::PerImageGrid {
        spatial_merge_default: 2,
    },
    preprocess: ferrite_vision::preprocess::preprocess_clip_normalized,
    default_image_size: 392,
    chat_template_image_part_type: "image",
    placeholder_policy: ferrite_vision::PlaceholderPolicy::RepeatMarker,
    mrope_positions: true,
    numbered_image_tag_marker: None,
};

#[cfg(feature = "cuda")]
#[vision_forward(workloads = [256, 1024, 4096, 16384], processor = crate::PROCESSOR)]
mod qwen2_vl {
    /// Qwen2-VL vision tower params — field name = the bound name the
    /// DSL / weights.json reference; `#[from]` paths index the VERBATIM
    /// HF config.json's nested `vision_config` block (configs/ carry it
    /// byte-for-byte). Tower geometry uses the Qwen2-VL key spellings
    /// (`embed_dim` / `depth` / `num_heads` / `in_chans`); MLP width is
    /// a ratio (integer in every shipped Qwen2-VL config), so
    /// `vision_mlp_hidden = embed · ratio`.
    struct Params {
        #[from = "vision_config.embed_dim"]
        vision_embed_dim: u64,
        #[from = "vision_config.depth"]
        vision_depth: u64,
        #[from = "vision_config.num_heads"]
        vision_num_heads: u64,
        #[from = "vision_config.in_chans"]
        vision_in_chans: u64,
        #[from = "vision_config.patch_size"]
        vision_patch_size: u64,
        #[from("vision_config.temporal_patch_size", default = 1)]
        vision_temporal_patch_size: u64,
        #[from("vision_config.spatial_merge_size", default = 1)]
        vision_spatial_merge_size: u64,
        #[expr = "vision_embed_dim / vision_num_heads"]
        vision_head_dim: u64,
        #[expr = "vision_in_chans * vision_temporal_patch_size * vision_patch_size * vision_patch_size"]
        vision_in_features: u64,
        #[expr = "vision_spatial_merge_size * vision_spatial_merge_size"]
        vision_merge_factor: u64,
        #[expr = "vision_embed_dim * vision_merge_factor"]
        vision_merge_hidden: u64,
        #[expr = "vision_head_dim / 2"]
        vision_rope_half_dim: u64,
        #[from = "vision_config.mlp_ratio"]
        vision_mlp_ratio: u64,
        #[expr = "vision_embed_dim * vision_mlp_ratio"]
        vision_mlp_hidden: u64,
        #[from = "vision_config.hidden_size"]
        d_model: u64,
    }

    /// Qwen2-VL vision blocks hardcode 1e-6 in the modeling code
    /// (transformers / mlx-vlm); the flat `vision_norm_eps` override
    /// wins when present. (Macro default is 1e-6 — declared here for
    /// the record.)
    const NORM_EPS: f64 = 1e-6;
    const SAFETENSORS: Layout = Layout {
        root: "visual",
        blocks: "blocks",
        subtrees: &[],
    };
    const FINGERPRINT: Fingerprint = Fingerprint {
        key: "visual.merger.mlp.2.weight",
        dim: 0,
    };
    const PATCH_EMBED_FLATTEN: Flatten = Flatten {
        key: "visual.patch_embed.proj.weight",
        leading_dim: 0,
        channels_last: false,
    };

    fn forward() {
    hidden_states = gemm(pixels, patch_embed.proj);

    for layer in 0..vision_depth {
        // PyTorch `nn.LayerNorm` decomposes as
        // `(mean, sub, rmsnorm, bias_add)`. The
        // `MeanSubRmsNormBiasAddImpl` matcher claims the 4-tile chain
        // and lowers to one `kernels::layer_norm_bias` call; the
        // bias-side weight ref is structural (the loader pulls both
        // `<prefix>.weight` and `<prefix>.bias` from the rmsnorm's
        // weight ref via the `LayerNorm` wrapper).
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
        fc1 = gemm(normed2, mlp.fc1[layer]);
        fc1 = bias_add(fc1, mlp.fc1.bias[layer]);
        fc1 = quick_gelu(fc1);
        fc2 = gemm(fc1, mlp.fc2[layer]);
        fc2 = bias_add(fc2, mlp.fc2.bias[layer]);
        hidden_states = add(fc2, hidden_states);
    }

    mq = mean(hidden_states);
    cq = sub(hidden_states, mq);
    nq = rmsnorm(cq, merger.ln_q);
    merged = bias_add(nq, merger.ln_q.bias);
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
}
