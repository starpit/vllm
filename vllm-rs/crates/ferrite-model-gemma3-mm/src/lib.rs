// SPDX-License-Identifier: Apache-2.0
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::too_many_arguments)]
//! Gemma3 multimodal vision tower (SigLIP encoder + Gemma3 MM
//! projector). The text decoder is shared with text-only Gemma3 and
//! lives in `ferrite-model-gemma3` (registered for arch
//! `Gemma3ForCausalLM`). This crate registers
//! `Gemma3ForConditionalGeneration` (the MM-bearing class) via the
//! `#[vision_forward]`-emitted `MultimodalForward` glue.
//!
//! Compared to Qwen2-VL / Qwen2.5-VL the SigLIP encoder is materially
//! simpler — full attention (no windowing), full LayerNorm (γ+β) per
//! pre-norm block, separate q/k/v projections (all with bias), GELU-
//! tanh MLP, post-LN. The non-trivial bit is the MM projector, which
//! reshapes `[L=ph², e]` into a 2D spatial grid, applies AvgPool2d
//! (k×k spatial averaging) to land on `[256, e]`, then RMSNorms and
//! projects to the text d_model.
//!
//! Positional embedding (G.7(c.1)) lands as `pos_embed(position_ids,
//! embeddings.position_embedding)` — same DSL surface as the decoder's
//! `embed(input_ids, embed_tokens)` and the same kernel
//! (`embedding_gather_masked`), differing only in which extern feeds
//! the indices and which bound names the table dims unify against.
//! The host wrapper (`VisionWrapper`) builds `[0..vision_num_positions,
//! 0..vision_num_positions, ...]` u32 per image and uploads as
//! `ForwardCtx::vision_position_ids`.

#[cfg(feature = "cuda")]
use ferrite_forward::vision_forward;

/// Gemma3-MM CPU preprocessing: SigLIP encoder + 4×4 avg-pool projector.
/// HF chat template emits `<start_of_image>` (= `boi_token_index`,
/// 255999) per image — NOT the soft-token id (262144) carried under
/// `image_token_index`. Tokens-per-image is fixed at 256 (post-pool);
/// `hf_config.mm_tokens_per_image` ships this value, so the
/// `FromConfig` policy reads it directly. Pixels are resized to a
/// fixed 896² square and normalized to symmetric ±1.
pub const PROCESSOR: ferrite_vision::MmMetadata = ferrite_vision::MmMetadata {
    hf_token_id_key: "boi_token_index",
    hf_token_id_default: 255999,
    size_policy: ferrite_vision::SizePolicy::FixedSquare,
    tokens_per_image: ferrite_vision::TokensPerImage::FromConfig,
    preprocess: ferrite_vision::preprocess::preprocess_symmetric_unit,
    default_image_size: 896,
    chat_template_image_part_type: "image",
    // HF Gemma3 processor expands `<start_of_image>` to
    // `\n\n<start_of_image><image_soft_token>×256<end_of_image>\n\n`.
    // The model is trained on this exact bracketed structure — a flat
    // RepeatMarker of boi gives garbled output.
    // Token IDs: boi=255999 (set as hf_token_id_default above),
    // soft=262144, eoi=256000, wrap=108 (Gemma tokenizer's `\n\n`).
    placeholder_policy: ferrite_vision::PlaceholderPolicy::BoiSoftEoiWrap {
        soft_token_id: 262144,
        eoi_token_id: 256000,
        wrap_token_id: 108,
    },
    // Gemma3 text decoder uses standard 1D RoPE — no MRoPE override.
    mrope_positions: false,
};

#[cfg(feature = "cuda")]
#[vision_forward(workloads = [256, 1024, 4096, 16384], processor = crate::PROCESSOR)]
fn gemma3_mm() {
    // Patch embed: Conv2d(in=3, out=1152, k=14, s=14) flattened at
    // load time to a [1152, 588] linear. With bias.
    hidden_states = gemm(pixels, embeddings.patch_embedding);
    hidden_states = bias_add(hidden_states, embeddings.patch_embedding.bias);

    // Learned positional embedding lookup. `position_ids` is a vision-
    // prelude extern of `[num_tokens]` u32 the wrapper builds as
    // `[0..vision_num_positions, ...]` per image; the table is the
    // 2D `[vision_num_positions, vision_embed_dim]` learned weight.
    pos_emb = pos_embed(position_ids, embeddings.position_embedding);
    hidden_states = add(hidden_states, pos_emb);

    for layer in 0..vision_depth {
        // Pre-norm self-attention. PyTorch `nn.LayerNorm` (γ+β)
        // decomposes as `(mean, sub, rmsnorm, bias_add)`; the
        // `MeanSubRmsNormBiasAddImpl` matcher claims the 4-tile chain
        // and lowers to one `kernels::layer_norm_bias` call.
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

        // SigLIP runs full attention per-image. Single cu_seqlens
        // segments the batched-flat tensor at image boundaries (one
        // segment per image, max_seqlen = vision_num_positions).
        attn_out = varlen_attention(q, k, v, cu_seqlens, max_seqlen);

        oproj = gemm(attn_out, self_attn.out_proj[layer]);
        oproj = bias_add(oproj, self_attn.out_proj.bias[layer]);
        hidden_states = add(oproj, hidden_states);

        // Pre-norm MLP (GELU-tanh, biases on both fc1 / fc2).
        m2 = mean(hidden_states);
        c2 = sub(hidden_states, m2);
        n2 = rmsnorm(c2, layer_norm2[layer]);
        normed2 = bias_add(n2, layer_norm2.bias[layer]);

        fc1 = gemm(normed2, mlp.fc1[layer]);
        fc1 = bias_add(fc1, mlp.fc1.bias[layer]);
        fc1 = gelu(fc1);
        fc2 = gemm(fc1, mlp.fc2[layer]);
        fc2 = bias_add(fc2, mlp.fc2.bias[layer]);
        hidden_states = add(fc2, hidden_states);
    }

    // Post-encoder LayerNorm (γ+β) — same 4-tile decomposition.
    mp = mean(hidden_states);
    cp = sub(hidden_states, mp);
    np = rmsnorm(cp, post_layernorm);
    hidden_states = bias_add(np, post_layernorm.bias);

    // MM projector: AvgPool2d(k=4) collapses the 64×64 patch grid to
    // 16×16 = 256 tokens, RMSNorm (no bias) on the pooled features,
    // matmul-only `nn.Parameter` projection to text d_model.
    //
    // Gemma3RMSNorm uses `output * (1.0 + weight)`, NOT `output * weight`
    // (so zero-init weights are identity-equivalent). Same `+ 1.0` as
    // every RMSNorm site in the text-decoder body (`ferrite-model-gemma3`).
    // Without this, garbled output: the projector RMSNorm scales the
    // post-pool features by ~zero (initial weights are tiny floats), the
    // projection collapses, and the decoder receives near-noise embeds.
    pooled = avg_pool_2d(hidden_states);
    normed_pool = rmsnorm(pooled, mm.mm_soft_emb_norm + 1.0);
    out = gemm(normed_pool, mm.mm_input_projection_weight);
}
