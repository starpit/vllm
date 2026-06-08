// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Qwen2.5-VL vision tower. The text decoder is shared with plain Qwen2 /
//! Qwen2.5 / Qwen2-VL and lives in `ferrite-model-qwen2` (registered for
//! arch `Qwen2_5_VLForConditionalGeneration` via that crate's
//! `configs/qwen2.5-vl-3b.json`). This crate contributes only the
//! vision-side `MultimodalForward` registration, emitted by the
//! `#[vision_forward]` macro — no hand-written code beyond the DSL body.
//!
//! Vision math deltas vs Qwen2-VL:
//! - block norms + merger.ln_q: LayerNorm → RMSNorm.
//! - block MLP: SwiGLU (`gate_proj`, `up_proj`, `down_proj` all biased)
//!   instead of `fc1 → QuickGELU → fc2`. `FusedGateUpSiluMul` claims
//!   `(gate_gemm, up_gemm, silu, mul)` rooted at the gate gemm and
//!   emits one packed `LinearLayer::forward` (with the loader-
//!   concatenated `[gate.bias | up.bias]`) followed by
//!   `silu_and_mul_fused`. `down` then uses an explicit `bias_add`
//!   tile so it routes through `FusedGemmBias` — the unfused
//!   `Instruction::Gemm` calls `cublas.gemm(weight, ...)` and SKIPS
//!   bias, which would silently drop `down_proj.bias` on the singleton
//!   path. `vision_intermediate_size` (3420 on the 3B) is padded to
//!   next mult of 8 by `__pad_to_mult8__` so cuBLAS bf16 GEMM accepts
//!   K=3424. The codegen path that bakes `W::INTERMEDIATE_SIZE` (the
//!   split point `silu_and_mul_fused` reads) falls back to
//!   `vision_intermediate_size_padded` when `intermediate_size` is
//!   absent — vision-only crates' bounds don't carry the latter.
//! - per-layer attention dispatch: 4-of-32 layers
//!   (`fullatt_block_indexes = [7, 15, 23, 31]`) read `cu_seqlens_full`
//!   / `max_seqlen_full`; the other 28 read `cu_seqlens_window` /
//!   `max_seqlen_window`.
//! - tokens are gather-permuted into window order on entry
//!   (`embedding_gather(_, window_index)`) and unpermuted at the merger
//!   output (`embedding_gather(_, reverse_indices)`). Cos/sin are
//!   pre-permuted host-side by the vision wrapper before upload.

#[cfg(any(feature = "cuda", feature = "metal"))]
use ferrite_forward_macro::vision_forward;

/// Qwen2.5-VL CPU preprocessing: same family conventions as Qwen2-VL —
/// `<|image_pad|>` placeholder, smart-resize at factor 28, CLIP
/// normalization. Per-image grid token count.
pub const PROCESSOR: ferrite_vision::MmMetadata = ferrite_vision::MmMetadata {
    hf_token_id_key: "image_token_id",
    hf_token_id_default: 151655,
    size_policy: ferrite_vision::SizePolicy::SmartResize {
        factor: 28,
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

#[cfg(any(feature = "cuda", feature = "metal"))]
#[vision_forward(workloads = [256, 1024, 4096, 16384], processor = crate::PROCESSOR)]
mod qwen2_5_vl {
    /// Qwen2.5-VL vision tower params — field name = the bound name the
    /// DSL / weights.json reference; `#[from]` paths index the VERBATIM
    /// HF config.json's nested `vision_config` block (configs/ carry it
    /// byte-for-byte). Embed dim is spelled `hidden_size` (not Qwen2-VL's
    /// `embed_dim`); the SwiGLU MLP width is `intermediate_size`, zero-
    /// padded to the next multiple of 8 for the cuBLAS bf16 GEMM K
    /// (`__pad_to_mult8__` in weights.json). The block MLP weights key
    /// off `vision_intermediate_size_padded`, so — matching the legacy
    /// derivation — no `vision_mlp_hidden` bound is declared.
    struct Params {
        #[from = "vision_config.hidden_size"]
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
        #[from = "vision_config.intermediate_size"]
        vision_intermediate_size: u64,
        #[expr = "(vision_intermediate_size + 7) / 8 * 8"]
        vision_intermediate_size_padded: u64,
        #[from = "vision_config.window_size"]
        vision_window_size: u64,
        #[from = "vision_config.out_hidden_size"]
        d_model: u64,
    }

    /// Qwen2.5-VL vision blocks hardcode 1e-6 (transformers / mlx-vlm);
    /// the flat `vision_norm_eps` override wins when present. (Macro
    /// default is 1e-6 — declared here for the record.)
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

        // Window-permute hidden_states at S² (= vision_merge_factor) row
        // granularity so each window contains a contiguous run.
        hidden_states = reshape(
            hidden_states,
            [
                num_tokens / vision_merge_factor,
                vision_merge_factor * vision_embed_dim,
            ],
        );
        hidden_states = embedding_gather(hidden_states, window_index);
        hidden_states = reshape(hidden_states, [num_tokens, vision_embed_dim]);

        for layer in 0..vision_depth {
            normed = rmsnorm(hidden_states, norm1[layer]);
            q = gemm(normed, attn.q[layer]);
            q = bias_add(q, attn.q.bias[layer]);
            k = gemm(normed, attn.k[layer]);
            k = bias_add(k, attn.k.bias[layer]);
            v = gemm(normed, attn.v[layer]);
            v = bias_add(v, attn.v.bias[layer]);
            (q, k) = vision_rope(q, k, cos, sin);
            if [7, 15, 23, 31].contains(&layer) {
                attn_out = varlen_attention(q, k, v, cu_seqlens_full, max_seqlen_full);
            } else {
                attn_out = varlen_attention(q, k, v, cu_seqlens_window, max_seqlen_window);
            }
            oproj = gemm(attn_out, attn.proj[layer]);
            oproj = bias_add(oproj, attn.proj.bias[layer]);
            hidden_states = add(oproj, hidden_states);

            normed2 = rmsnorm(hidden_states, norm2[layer]);
            // SwiGLU MLP. Explicit bias_add tiles after each gemm so the
            // (Gemm, BiasAdd) matcher routes through FusedGemmBias —
            // singleton Instruction::Gemm skips bias entirely. See lib
            // docstring for the full rationale.
            gate = silu(gemm(normed2, mlp.gate_proj[layer]));
            up = gemm(normed2, mlp.up_proj[layer]);
            down = gemm(gate * up, mlp.down_proj[layer]);
            down = bias_add(down, mlp.down_proj.bias[layer]);
            hidden_states = add(down, hidden_states);
        }

        merged = rmsnorm(hidden_states, merger.ln_q);
        merged = reshape(
            merged,
            [num_tokens / vision_merge_factor, vision_merge_hidden],
        );
        mlp0 = gemm(merged, merger.mlp_0);
        mlp0 = bias_add(mlp0, merger.mlp_0.bias);
        mlp0 = gelu_erf(mlp0);
        projected = gemm(mlp0, merger.mlp_2);
        projected = bias_add(projected, merger.mlp_2.bias);
        out = embedding_gather(projected, reverse_indices);
    }
}
