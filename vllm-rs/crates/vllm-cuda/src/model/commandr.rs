// SPDX-License-Identifier: Apache-2.0
//! Command R (CohereForCausalLM) model using `GpuTensor` — zero-allocation forward pass.
//!
//! Key differences from LLaMA:
//! - CohereLayerNorm (full LayerNorm with mean subtraction, weight only, no bias)
//! - Parallel attention + MLP: one norm per layer, both branches read same normed input
//! - Interleaved RoPE: adjacent pairs (2i, 2i+1) rotated together (Cohere convention)
//! - Logit scaling: `logits *= logit_scale` (e.g. 0.0625 = 1/16)
//! - Optional QK norm (CohereLayerNorm on Q/K after projection, before RoPE)
//! - Tied word embeddings (lm_head shares embed_tokens weight)
//!
//! Port of: `vllm/model_executor/models/commandr.py`

use anyhow::Result;

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::layers::{CohereLayerNorm, Embedding, Linear};
use crate::model::llama::{LlamaAttention, LlamaMLP, RotaryCache};
use crate::tensor::TensorView;
use crate::weights::GpuWeights;
use crate::weights::{self as gpu_weights};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Parsed configuration for a Command R model.
#[derive(Debug, Clone)]
pub struct CommandRConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub logit_scale: f32,
    pub use_qk_norm: bool,
    pub tie_word_embeddings: bool,
}

// ---------------------------------------------------------------------------
// CommandRAttention
// ---------------------------------------------------------------------------

/// Command R attention — reuses LlamaAttention but with interleaved RoPE.
///
/// The structural layout (fused QKV, O proj, optional QK norm) is identical
/// to LlamaAttention. The only difference is the RoPE convention: interleaved
/// pairs (2i, 2i+1) instead of NeoX (i, i+half).
pub struct CommandRAttention {
    pub inner: LlamaAttention,
}

impl CommandRAttention {
    /// Forward pass with interleaved RoPE.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        rotary: &RotaryCache,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let attn = &self.inner;
        let num_tokens = hidden_states.dim(0);

        // QKV projection — handle both fused (dense) and separate (BNB/quantized) paths.
        let qkv = if let (Some(k_proj), Some(v_proj)) = (attn.k_proj.as_ref(), attn.v_proj.as_ref())
        {
            // Quantized: separate Q, K, V GEMMs → concat
            let q_out = attn.qkv_proj.forward(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            let k_out = k_proj.forward(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            let v_out = v_proj.forward(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            let qk = kernels::concat_dim1(
                *q_out.view(),
                *k_out.view(),
                &mut device.caching,
                device.compute_stream,
            );
            drop(q_out);
            drop(k_out);
            let qkv = kernels::concat_dim1(
                *qk.view(),
                *v_out.view(),
                &mut device.caching,
                device.compute_stream,
            );
            drop(qk);
            drop(v_out);
            qkv
        } else {
            // Dense: single fused QKV GEMM
            attn.qkv_proj.forward(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            )
        };

        // Split QKV and apply interleaved RoPE.
        // Command R uses is_neox_style=False (interleaved pairs at 2i, 2i+1).
        let (q, k, v) =
            if let (Some(q_norm_w), Some(k_norm_w)) = (attn.q_norm_weight, attn.k_norm_weight) {
                // QK-norm path: split first, then per-head norm, then Q-only RoPE.
                let (q, k, v) = kernels::split_qkv(
                    *qkv.view(),
                    attn.q_size,
                    attn.kv_size,
                    attn.num_q_heads,
                    attn.num_kv_heads,
                    attn.head_dim,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(qkv);

                // Apply per-head CohereLayerNorm on Q and K, then Q-only RoPE.
                kernels::qk_norm_inplace(
                    *q.view(),
                    *k.view(),
                    q_norm_w,
                    k_norm_w,
                    attn.num_q_heads,
                    attn.num_kv_heads,
                    attn.head_dim,
                    attn.qk_norm_eps,
                    0.0,
                    0.0,
                    device.compute_stream,
                );
                kernels::rotary_embedding_q_only(
                    *q.view(),
                    *positions,
                    rotary.cos_sin_cache,
                    attn.num_q_heads,
                    attn.head_dim,
                    device.compute_stream,
                );
                (q, k, v)
            } else if max_seqlen_q == 1 {
                // Decode: rotate BOTH Q AND K (interleaved layout), then write
                // ROTATED K to cache. Mirrors Llama's path; the previous
                // Q-only-rotate + FA2-in-kernel-K-rotate approach was broken
                // because FA2's in-kernel rotation only fires for span blocks
                // (when `block_unrotated_gpu` flags are set). With no spans
                // active, K stayed unrotated when read by attention.
                let (q, k, v) = kernels::split_qkv(
                    *qkv.view(),
                    attn.q_size,
                    attn.kv_size,
                    attn.num_q_heads,
                    attn.num_kv_heads,
                    attn.head_dim,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(qkv);

                // Rotate Q and K in-place with interleaved (2i, 2i+1) pairing.
                // 2D layout `[T, heads*head_dim]` — kernel reads dim(0)/dim(1).
                kernels::rotary_embedding_interleaved_inplace(
                    q.as_gpu_tensor(),
                    k.as_gpu_tensor(),
                    *positions,
                    rotary.cos_sin_cache,
                    attn.head_dim,
                    device.compute_stream,
                );

                // Write ROTATED K and V to cache.
                crate::attention_helpers::write_kv_cache(
                    k.view(),
                    v.view(),
                    slot_mapping,
                    kv_cache,
                    attn.layer_idx,
                    device.compute_stream,
                );
                drop(k);
                drop(v);

                // K is pre-rotated → null cos_sin_cache (no in-kernel rotation
                // needed). Spans path stays handled by passing the real ptr
                // when spans are active, identical to Llama's pattern.
                let has_spans = !kv_cache.block_unrotated_gpu().is_null();
                let (cos_sin_ptr, rotary_dim) = if has_spans {
                    (
                        rotary.cos_sin_cache.raw_ptr() as *const u8,
                        rotary.cos_sin_cache.dim(1),
                    )
                } else {
                    (std::ptr::null::<u8>(), 0)
                };
                let attn_output = crate::attention_helpers::attention_decode_from_cache(
                    q.view(),
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    attn.scale,
                    0.0,
                    -1,
                    kv_cache,
                    attn.layer_idx,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                    cos_sin_ptr,
                    rotary_dim,
                    true, // interleaved RoPE flavor (only consumed by spans path)
                );
                drop(q);

                let attn_flat = attn_output.view().reshape(&[num_tokens, attn.q_size]);
                let result = attn.o_proj.forward(
                    attn_flat,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(attn_output);
                return result;
            } else {
                // Prefill: rotate BOTH Q AND K (interleaved), then write
                // ROTATED K to cache, then run attention with null cos_sin_ptr.
                // Same fix as the decode path above.
                let (q, k, v) = kernels::split_qkv(
                    *qkv.view(),
                    attn.q_size,
                    attn.kv_size,
                    attn.num_q_heads,
                    attn.num_kv_heads,
                    attn.head_dim,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(qkv);

                // Rotate Q and K in-place with interleaved layout BEFORE caching.
                kernels::rotary_embedding_interleaved_inplace(
                    q.as_gpu_tensor(),
                    k.as_gpu_tensor(),
                    *positions,
                    rotary.cos_sin_cache,
                    attn.head_dim,
                    device.compute_stream,
                );

                // Write ROTATED K/V to cache.
                crate::attention_helpers::write_kv_cache(
                    k.view(),
                    v.view(),
                    slot_mapping,
                    kv_cache,
                    attn.layer_idx,
                    device.compute_stream,
                );

                // K pre-rotated → null cos_sin_cache for the contiguous prefill
                // path. Spans only matter for paged decode, so `has_spans`
                // doesn't need consulting here.
                let attn_output = crate::attention_helpers::attention_standard(
                    q.view(),
                    k.view(),
                    v.view(),
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    attn.scale,
                    kv_cache,
                    attn.layer_idx,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                    std::ptr::null::<u8>(),
                    0,
                    true, // interleaved RoPE flavor (unused when rotary_dim=0)
                );
                drop(q);
                drop(k);
                drop(v);

                let attn_flat = attn_output.view().reshape(&[num_tokens, attn.q_size]);
                let result = attn.o_proj.forward(
                    attn_flat,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(attn_output);
                return result;
            };

        // QK-norm fallthrough: write K/V then run attention.
        crate::attention_helpers::write_kv_cache(
            k.view(),
            v.view(),
            slot_mapping,
            kv_cache,
            attn.layer_idx,
            device.compute_stream,
        );

        // QK-norm uses NeoX Q-only RoPE (rotary_embedding_q_only), K unrotated.
        // Note: if CommandR QK-norm models use interleaved RoPE, this needs
        // rotary_embedding_interleaved_q_only instead.
        let attn_output = crate::attention_helpers::attention_standard(
            q.view(),
            k.view(),
            v.view(),
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            attn.scale,
            kv_cache,
            attn.layer_idx,
            device.num_sm,
            &mut device.caching,
            device.compute_stream,
            rotary.cos_sin_cache.raw_ptr() as *const u8,
            rotary.cos_sin_cache.dim(1),
            false, // QK-norm path uses NeoX RoPE
        );
        drop(q);
        drop(k);
        drop(v);

        let attn_flat = attn_output.view().reshape(&[num_tokens, attn.q_size]);
        let result = attn.o_proj.forward(
            attn_flat,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );
        drop(attn_output);
        result
    }
}

// ---------------------------------------------------------------------------
// CommandRDecoderLayer
// ---------------------------------------------------------------------------

/// A single Command R decoder layer with parallel attention + MLP.
///
/// Uses one CohereLayerNorm per layer. Both attention and MLP read from the
/// same normed input and their outputs are summed with the residual:
///   `out = residual + attn(norm(x)) + mlp(norm(x))`
///
/// This matches Python vLLM's CohereDecoderLayer.forward():
///   residual = hidden_states
///   hidden_states = layer_norm(hidden_states)
///   attn_out = self_attn(hidden_states)
///   mlp_out = mlp(hidden_states)
///   hidden_states = residual + attn_out + mlp_out
pub struct CommandRDecoderLayer {
    pub self_attn: CommandRAttention,
    pub mlp: LlamaMLP,
    pub input_layernorm: CohereLayerNorm,
}

impl CommandRDecoderLayer {
    /// Forward pass with caching allocator.
    ///
    /// * `hidden_states` — input to this layer
    ///
    /// Returns output hidden_states.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        hidden_states: OwnedTensor,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        rotary: &RotaryCache,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        // CohereLayerNorm (standalone, no fused residual add).
        // normed is a new buffer; hidden_states preserved as residual.
        let normed = kernels::cohere_layer_norm(
            *hidden_states,
            self.input_layernorm.weight,
            self.input_layernorm.eps,
            &mut device.caching,
            device.compute_stream,
        );

        // Parallel attention and MLP, both reading from normed.
        // normed (GpuTensor) is a view into normed (OwnedTensor) — both
        // attn and mlp will read from it before either writes.
        let normed_view = normed.view();

        let attn_output = self.self_attn.forward(
            normed_view,
            positions,
            slot_mapping,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            kv_cache,
            rotary,
            device,
        );

        let mlp_output = self.mlp.forward(normed_view, device);
        drop(normed); // free normed buffer

        // hidden_states = residual + attn_output + mlp_output
        // Add attn_output into hidden_states in-place.
        kernels::add_inplace(*hidden_states, *attn_output.view(), device.compute_stream);
        drop(attn_output);

        // Add mlp_output into hidden_states in-place.
        kernels::add_inplace(*hidden_states, *mlp_output.view(), device.compute_stream);
        drop(mlp_output);

        hidden_states
    }
}

// ---------------------------------------------------------------------------
// CommandRModel
// ---------------------------------------------------------------------------

/// Command R transformer backbone.
pub struct CommandRModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<CommandRDecoderLayer>,
    pub norm: CohereLayerNorm,
    pub rotary: RotaryCache,
}

impl CommandRModel {
    /// Forward pass with paged attention.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        input_ids: TensorView<'_>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        // Embedding lookup.
        let hidden_states = kernels::embedding_gather(
            self.embed_tokens.weight,
            *input_ids,
            &mut device.caching,
            device.compute_stream,
        );

        let mut hidden_states: OwnedTensor = hidden_states;

        for layer in self.layers.iter() {
            hidden_states = layer.forward(
                hidden_states,
                positions,
                slot_mapping,
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_seqlen_q,
                max_seqlen_k,
                kv_cache,
                &self.rotary,
                device,
            );
        }

        // Final CohereLayerNorm.
        let normed = kernels::cohere_layer_norm(
            *hidden_states,
            self.norm.weight,
            self.norm.eps,
            &mut device.caching,
            device.compute_stream,
        );
        drop(hidden_states);
        normed
    }
}

// ---------------------------------------------------------------------------
// CommandRForCausalLM
// ---------------------------------------------------------------------------

/// Command R for causal language modeling.
///
/// Wraps `CommandRModel` with a language model head (tied to embed_tokens)
/// and logit scaling.
pub struct CommandRForCausalLM {
    pub model: CommandRModel,
    pub lm_head: Linear,
    pub logit_scale: f32,
}

impl CommandRForCausalLM {
    /// Forward pass: input_ids → logits.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        input_ids: TensorView<'_>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<TensorView<'_>>,
    ) -> OwnedTensor {
        let hidden_states = self.model.forward(
            input_ids,
            positions,
            slot_mapping,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            kv_cache,
            device,
        );

        // Gather last-token hidden states.
        let hidden_states = if let Some(indices) = last_token_indices {
            kernels::embedding_gather(
                *hidden_states,
                *indices,
                &mut device.caching,
                device.compute_stream,
            )
        } else {
            hidden_states
        };

        // lm_head: logits = hidden_states @ lm_head_weight^T
        let logits = self.lm_head.forward(
            hidden_states.view(),
            &mut device.cublas,
            &mut device.caching,
        );

        // Apply logit scaling.
        if self.logit_scale != 1.0 {
            kernels::scale_inplace(*logits.view(), self.logit_scale, &device.cublas);
        }

        logits
    }

    /// Load the full model from safetensors weights.
    pub fn load(
        weights: &mut GpuWeights,
        config: &CommandRConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let stream = device.compute_stream;

        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // Build a LlamaConfig-like struct for loading attention weights.
            let llama_config = crate::model::llama::LlamaConfig {
                hidden_size: config.hidden_size,
                num_attention_heads: config.num_attention_heads,
                num_kv_heads: config.num_kv_heads,
                num_hidden_layers: config.num_hidden_layers,
                intermediate_size: config.intermediate_size,
                vocab_size: config.vocab_size,
                max_position_embeddings: config.max_position_embeddings,
                rms_norm_eps: config.layer_norm_eps,
                rope_theta: config.rope_theta,
                head_dim: config.head_dim,
                tie_word_embeddings: config.tie_word_embeddings,
                llama3_rope_scaling: None,
            };

            // Load attention — uses the same fused QKV loading as LLaMA.
            let inner_attn = if config.use_qk_norm {
                LlamaAttention::load_fused_with_qk_norm(
                    weights,
                    &format!("{prefix}.self_attn"),
                    &llama_config,
                    i,
                    config.layer_norm_eps,
                    stream,
                )?
            } else {
                LlamaAttention::load_fused(
                    weights,
                    &format!("{prefix}.self_attn"),
                    &llama_config,
                    i,
                    stream,
                )?
            };
            let self_attn = CommandRAttention { inner: inner_attn };

            // Load MLP — same as LLaMA (SiLU activation, fused gate+up).
            let mlp = LlamaMLP::load_fused(
                weights,
                &format!("{prefix}.mlp"),
                config.intermediate_size,
                stream,
            )?;

            // CohereLayerNorm (one per layer — Command R has no post_attention_layernorm).
            let input_layernorm = CohereLayerNorm::load(
                weights,
                &format!("{prefix}.input_layernorm"),
                config.layer_norm_eps,
            )?;

            layers.push(CommandRDecoderLayer {
                self_attn,
                mlp,
                input_layernorm,
            });
        }

        // Final CohereLayerNorm.
        let norm = CohereLayerNorm::load(weights, "model.norm", config.layer_norm_eps)?;

        // Rotary cache — same as LLaMA but cos/sin will be indexed differently
        // at runtime by the interleaved RoPE kernel.
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None, // no llama3 rope scaling
                dtype,
                device,
            )?
        };

        // Tied word embeddings: lm_head shares embed_tokens weight.
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };

        Ok(Self {
            model: CommandRModel {
                embed_tokens,
                layers,
                norm,
                rotary,
            },
            lm_head,
            logit_scale: config.logit_scale,
        })
    }

    /// Load an FP8 quantized Command R model.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        config: &CommandRConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let stream = device.compute_stream;

        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let llama_config = crate::model::llama::LlamaConfig {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            num_hidden_layers: config.num_hidden_layers,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.layer_norm_eps,
            rope_theta: config.rope_theta,
            head_dim: config.head_dim,
            tie_word_embeddings: config.tie_word_embeddings,
            llama3_rope_scaling: None,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // FP8 attention — LlamaAttention already supports load_fp8.
            let inner_attn = LlamaAttention::load_fp8(
                weights,
                &format!("{prefix}.self_attn"),
                &llama_config,
                i,
                dtype,
                0.0, // CommandR uses separate QK-norm loading
                stream,
            )?;
            let self_attn = CommandRAttention { inner: inner_attn };

            // FP8 MLP — LlamaMLP already supports load_fp8.
            let mlp = LlamaMLP::load_fp8(
                weights,
                &format!("{prefix}.mlp"),
                config.intermediate_size,
                dtype,
                stream,
            )?;

            // CohereLayerNorm — always dense (not quantized).
            let input_layernorm = CohereLayerNorm::load(
                weights,
                &format!("{prefix}.input_layernorm"),
                config.layer_norm_eps,
            )?;

            layers.push(CommandRDecoderLayer {
                self_attn,
                mlp,
                input_layernorm,
            });
        }

        let norm = CohereLayerNorm::load(weights, "model.norm", config.layer_norm_eps)?;
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
                device,
            )?
        };

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };

        Ok(Self {
            model: CommandRModel {
                embed_tokens,
                layers,
                norm,
                rotary,
            },
            lm_head,
            logit_scale: config.logit_scale,
        })
    }

    /// Load a BitsAndBytes 4-bit (NF4/FP4) quantized Command R model.
    pub fn load_bnb4bit(
        weights: &mut GpuWeights,
        config: &CommandRConfig,
        dtype: DType,
        qconfig: &crate::quant::Bnb4bitConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let stream = device.compute_stream;

        // Upload shared NF4/FP4 code table to GPU.
        let code_table = match qconfig.quant_type {
            crate::quant::BnbQuantType::NF4 => &crate::quant::NF4_CODE,
            crate::quant::BnbQuantType::FP4 => &crate::quant::FP4_CODE,
        };
        let code_gpu = gpu_weights::upload_bnb_code(code_table, stream)?;
        let blocksize = qconfig.blocksize;

        // Compute max dequant buffer size.
        let hidden = config.hidden_size;
        let q_size = config.num_attention_heads * config.head_dim;
        let inter = config.intermediate_size;
        let max_elements = [
            q_size * hidden,
            hidden * q_size,
            inter * hidden,
            hidden * inter,
        ]
        .into_iter()
        .max()
        .unwrap();

        let dequant_scratch = gpu_weights::alloc_bnb_dequant_scratch(max_elements, dtype, stream)?;

        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let llama_config = crate::model::llama::LlamaConfig {
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            num_hidden_layers: config.num_hidden_layers,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            max_position_embeddings: config.max_position_embeddings,
            rms_norm_eps: config.layer_norm_eps,
            rope_theta: config.rope_theta,
            head_dim: config.head_dim,
            tie_word_embeddings: config.tie_word_embeddings,
            llama3_rope_scaling: None,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let inner_attn = LlamaAttention::load_bnb4bit(
                weights,
                &format!("{prefix}.self_attn"),
                &llama_config,
                i,
                qconfig,
                code_gpu,
                dequant_scratch,
                blocksize,
                stream,
            )?;
            let self_attn = CommandRAttention { inner: inner_attn };

            let mlp = LlamaMLP::load_bnb4bit(
                weights,
                &format!("{prefix}.mlp"),
                &llama_config,
                qconfig,
                code_gpu,
                dequant_scratch,
                blocksize,
                stream,
            )?;

            let input_layernorm = CohereLayerNorm::load(
                weights,
                &format!("{prefix}.input_layernorm"),
                config.layer_norm_eps,
            )?;

            layers.push(CommandRDecoderLayer {
                self_attn,
                mlp,
                input_layernorm,
            });
        }

        let norm = CohereLayerNorm::load(weights, "model.norm", config.layer_norm_eps)?;

        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
                device,
            )?
        };

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };

        Ok(Self {
            model: CommandRModel {
                embed_tokens,
                layers,
                norm,
                rotary,
            },
            lm_head,
            logit_scale: config.logit_scale,
        })
    }
}
