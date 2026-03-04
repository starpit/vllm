// SPDX-License-Identifier: Apache-2.0
//! Quantized Gemma 3 model architecture using GGUF weights.
//!
//! Parallel to `quantized_llama.rs` but uses Gemma3 architecture:
//! - GELU activation (instead of SiLU)
//! - GemmaRmsNorm with +1 weight offset (instead of standard RmsNorm)
//! - 4 norms per decoder layer
//! - Per-head QK norms before RoPE
//! - Per-layer RoPE theta (global vs local/sliding)
//! - Embedding normalized by sqrt(hidden_size)
//! - Always tied embeddings (embed_tokens as lm_head)
//! - query_pre_attn_scalar for attention scale
//!
//! GGUF tensor names follow the gguf-py gemma3 mapping.

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::gguf::GgufFile;
use vllm_model::layers::{Embedding, GemmaRmsNorm, QuantizedLinear};
use vllm_model::weight::HfModelConfig;

use vllm_model::layers::RotaryEmbedding;

use crate::attention::attention_with_cache;
use crate::gemma3::Gemma3Config;

// ---------------------------------------------------------------------------
// Helper: load a dequantized GemmaRmsNorm from GGUF
// ---------------------------------------------------------------------------

/// Dequantize a GGUF tensor to f32 and construct a GemmaRmsNorm.
fn load_gemma_rms_norm(
    gguf: &mut GgufFile,
    name: &str,
    eps: f64,
    device: &Device,
) -> ModelResult<GemmaRmsNorm> {
    let weight = gguf
        .tensor(name, device)?
        .dequantize(device)
        .map_err(|e| ModelError::Other(format!("dequantize norm: {e}")))?;
    GemmaRmsNorm::new(weight, eps).map_err(ModelError::Candle)
}

// ---------------------------------------------------------------------------
// QuantizedGemma3MLP
// ---------------------------------------------------------------------------

/// Gemma3 MLP with quantized linear projections and GELU activation.
struct QuantizedGemma3MLP {
    gate_proj: QuantizedLinear,
    up_proj: QuantizedLinear,
    down_proj: QuantizedLinear,
}

impl QuantizedGemma3MLP {
    fn load(gguf: &mut GgufFile, prefix: &str, device: &Device) -> ModelResult<Self> {
        Ok(Self {
            gate_proj: QuantizedLinear::from_gguf(
                gguf,
                &format!("{prefix}.ffn_gate.weight"),
                device,
            )?,
            up_proj: QuantizedLinear::from_gguf(gguf, &format!("{prefix}.ffn_up.weight"), device)?,
            down_proj: QuantizedLinear::from_gguf(
                gguf,
                &format!("{prefix}.ffn_down.weight"),
                device,
            )?,
        })
    }
}

impl Module for QuantizedGemma3MLP {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = crate::ops::gelu_and_mul(&gate, &up)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// QuantizedGemma3Attention
// ---------------------------------------------------------------------------

/// Gemma3 attention with quantized Q/K/V/O projections, per-head QK norms,
/// per-layer RoPE theta, and interleaved RoPE (GGML convention).
struct QuantizedGemma3Attention {
    q_proj: QuantizedLinear,
    k_proj: QuantizedLinear,
    v_proj: QuantizedLinear,
    o_proj: QuantizedLinear,
    /// Per-head Q norm (GemmaRmsNorm, weight is [head_dim]).
    q_norm: GemmaRmsNorm,
    /// Per-head K norm (GemmaRmsNorm, weight is [head_dim]).
    k_norm: GemmaRmsNorm,
    /// NeoX-style RoPE (Gemma3 GGUF stores weights in HF convention, not GGML interleaved).
    rotary_emb: RotaryEmbedding,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
    sliding_window: Option<usize>,
}

impl QuantizedGemma3Attention {
    fn load(
        gguf: &mut GgufFile,
        prefix: &str,
        config: &Gemma3Config,
        rotary_emb: RotaryEmbedding,
        device: &Device,
    ) -> ModelResult<Self> {
        let q_proj = QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_q.weight"), device)?;
        let k_proj = QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_k.weight"), device)?;
        let v_proj = QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_v.weight"), device)?;
        let o_proj =
            QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_output.weight"), device)?;

        // Per-head QK norms — dequantize to f32, wrap as GemmaRmsNorm (+1 offset).
        let q_norm = load_gemma_rms_norm(
            gguf,
            &format!("{prefix}.attn_q_norm.weight"),
            config.rms_norm_eps,
            device,
        )?;
        let k_norm = load_gemma_rms_norm(
            gguf,
            &format!("{prefix}.attn_k_norm.weight"),
            config.rms_norm_eps,
            device,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_q_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5),
            sliding_window: None,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        let num_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;

        // Q/K/V projections (QMatMul output is f32 on CPU).
        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;

        // Reshape to [num_tokens, num_heads, head_dim].
        let q = q
            .reshape((num_tokens, self.num_q_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let k = k
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;
        let v = v
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;

        // Per-head QK norms (GemmaRmsNorm normalizes last dim, broadcasts over heads).
        let q = crate::ops::gemma_rms_norm(&q, &self.q_norm).map_err(ModelError::Candle)?;
        let k = crate::ops::gemma_rms_norm(&k, &self.k_norm).map_err(ModelError::Candle)?;

        // Apply NeoX (half-split) RoPE — Gemma3 GGUF stores weights in HF convention.
        let cos = self
            .rotary_emb
            .cos_cache()
            .index_select(positions, 0)
            .map_err(ModelError::Candle)?;
        let sin = self
            .rotary_emb
            .sin_cache()
            .index_select(positions, 0)
            .map_err(ModelError::Candle)?;
        let q = vllm_model::layers::apply_rotary_to_tensor(&q, &cos, &sin)
            .map_err(|e| ModelError::Other(format!("RoPE: {e}")))?;
        let k = vllm_model::layers::apply_rotary_to_tensor(&k, &cos, &sin)
            .map_err(|e| ModelError::Other(format!("RoPE: {e}")))?;

        // Cache-merge + attention.
        let attn_output =
            attention_with_cache(&q, &k, &v, self.scale, kv_cache, self.sliding_window)?;

        // Reshape and output projection.
        let attn_output = attn_output
            .reshape((num_tokens, self.num_q_heads * self.head_dim))
            .map_err(ModelError::Candle)?;

        self.o_proj
            .forward(&attn_output)
            .map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// QuantizedGemma3DecoderLayer
// ---------------------------------------------------------------------------

/// A single quantized Gemma3 decoder layer with 4 GemmaRmsNorm norms.
///
/// Forward:
///   input_layernorm → attention → post_attention_layernorm → residual_add
///   → pre_feedforward_layernorm → MLP → post_feedforward_layernorm → residual_add
struct QuantizedGemma3DecoderLayer {
    self_attn: QuantizedGemma3Attention,
    mlp: QuantizedGemma3MLP,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
}

impl QuantizedGemma3DecoderLayer {
    fn load(
        gguf: &mut GgufFile,
        prefix: &str,
        config: &Gemma3Config,
        rotary_emb: RotaryEmbedding,
        device: &Device,
    ) -> ModelResult<Self> {
        let self_attn = QuantizedGemma3Attention::load(gguf, prefix, config, rotary_emb, device)?;
        let mlp = QuantizedGemma3MLP::load(gguf, prefix, device)?;

        // 4 norms per layer (GGUF names → Gemma3 roles):
        // attn_norm         → input_layernorm
        // post_attention_norm → post_attention_layernorm
        // ffn_norm           → pre_feedforward_layernorm
        // post_ffw_norm      → post_feedforward_layernorm
        let input_layernorm = load_gemma_rms_norm(
            gguf,
            &format!("{prefix}.attn_norm.weight"),
            config.rms_norm_eps,
            device,
        )?;
        let post_attention_layernorm = load_gemma_rms_norm(
            gguf,
            &format!("{prefix}.post_attention_norm.weight"),
            config.rms_norm_eps,
            device,
        )?;
        let pre_feedforward_layernorm = load_gemma_rms_norm(
            gguf,
            &format!("{prefix}.ffn_norm.weight"),
            config.rms_norm_eps,
            device,
        )?;
        let post_feedforward_layernorm = load_gemma_rms_norm(
            gguf,
            &format!("{prefix}.post_ffw_norm.weight"),
            config.rms_norm_eps,
            device,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<(Tensor, Tensor)> {
        // Pre-attention norm: fuse previous MLP residual add when available.
        let (normed, residual) = if let Some(residual) = residual {
            crate::ops::fused_add_gemma_rms_norm(hidden_states, residual, &self.input_layernorm)
                .map_err(ModelError::Candle)?
        } else {
            let normed = crate::ops::gemma_rms_norm(hidden_states, &self.input_layernorm)
                .map_err(ModelError::Candle)?;
            (normed, hidden_states.clone())
        };

        let attn_output = self.self_attn.forward(&normed, positions, kv_cache)?;
        // Post-attention norm.
        let attn_output = crate::ops::gemma_rms_norm(&attn_output, &self.post_attention_layernorm)
            .map_err(ModelError::Candle)?;

        // Fused residual add + pre-feedforward norm.
        let (normed, residual) = crate::ops::fused_add_gemma_rms_norm(
            &attn_output,
            &residual,
            &self.pre_feedforward_layernorm,
        )
        .map_err(ModelError::Candle)?;

        // MLP + post-feedforward norm (residual add deferred to next layer or final norm).
        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;
        let mlp_output = crate::ops::gemma_rms_norm(&mlp_output, &self.post_feedforward_layernorm)
            .map_err(ModelError::Candle)?;

        Ok((mlp_output, residual))
    }
}

// ---------------------------------------------------------------------------
// QuantizedGemma3Model
// ---------------------------------------------------------------------------

/// Quantized Gemma3 transformer backbone.
///
/// Embedding (* sqrt(hidden_size)) -> N decoder layers -> final GemmaRMS norm.
struct QuantizedGemma3Model {
    embed_tokens: Embedding,
    layers: Vec<QuantizedGemma3DecoderLayer>,
    norm: GemmaRmsNorm,
    /// Embedding normalizer: sqrt(hidden_size).
    normalizer: f64,
}

impl QuantizedGemma3Model {
    fn load(gguf: &mut GgufFile, config: &Gemma3Config, device: &Device) -> ModelResult<Self> {
        // Embedding: dequantize to f32 (relatively small).
        let embed_weight = gguf
            .tensor("token_embd.weight", device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize embedding: {e}")))?;
        let embed_tokens = Embedding::new(embed_weight);

        // Decoder layers — each layer gets its own RotaryEmbedding based on RoPE theta.
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let rope_theta = config.rope_theta_for_layer(i);
            let rotary_emb = RotaryEmbedding::new(
                config.head_dim,
                config.max_position_embeddings,
                rope_theta,
                DType::F32,
                device,
            )?;

            let mut layer = QuantizedGemma3DecoderLayer::load(
                gguf,
                &format!("blk.{i}"),
                config,
                rotary_emb,
                device,
            )?;
            // Apply per-layer sliding window.
            if i < config.layer_is_sliding.len() && config.layer_is_sliding[i] {
                layer.self_attn.sliding_window = config.sliding_window;
            }
            layers.push(layer);
        }

        // Final norm: dequantize.
        let norm = load_gemma_rms_norm(gguf, "output_norm.weight", config.rms_norm_eps, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            normalizer: (config.hidden_size as f64).sqrt(),
        })
    }

    /// Embed token IDs and scale by sqrt(hidden_size).
    fn embed(&self, input_ids: &Tensor) -> ModelResult<Tensor> {
        let hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(ModelError::Candle)?;
        (hidden_states * self.normalizer).map_err(ModelError::Candle)
    }

    /// Run the transformer backbone on pre-computed embeddings.
    fn backbone(
        &self,
        mut hidden_states: Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let mut residual: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_handle = kv_cache.as_mut().map(|s| s.layer_handle(i));
            let (hs, res) =
                layer.forward(&hidden_states, residual.as_ref(), positions, layer_handle)?;
            hidden_states = hs;
            residual = Some(res);
        }

        // Final norm: fuse last MLP's residual add into the norm.
        let (normed, _) = crate::ops::fused_add_gemma_rms_norm(
            &hidden_states,
            residual.as_ref().unwrap(),
            &self.norm,
        )
        .map_err(ModelError::Candle)?;
        Ok(normed)
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self.embed(input_ids)?;
        self.backbone(hidden_states, positions, kv_cache)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// QuantizedGemma3ForCausalLM
// ---------------------------------------------------------------------------

/// Quantized Gemma3 for causal language modeling.
///
/// Always uses tied embeddings (embed_tokens weight as lm_head).
pub struct QuantizedGemma3ForCausalLM {
    model: QuantizedGemma3Model,
    lm_head: QuantizedLinear,
}

impl QuantizedGemma3ForCausalLM {
    /// Load the full quantized model from a GGUF file.
    pub fn load(gguf: &mut GgufFile, config: &Gemma3Config, device: &Device) -> ModelResult<Self> {
        let model = QuantizedGemma3Model::load(gguf, config, device)?;

        // lm_head: Gemma3 always uses tied embeddings.
        // Some GGUF models may have "output.weight", but typically it's tied.
        let has_output = gguf.tensor_names().contains(&"output.weight");
        let lm_head = if has_output {
            QuantizedLinear::from_gguf(gguf, "output.weight", device)?
        } else {
            // Tie with embedding: re-read the original quantized tensor.
            let qt = gguf.tensor("token_embd.weight", device)?;
            QuantizedLinear::from_qtensor(qt)?
        };

        Ok(Self { model, lm_head })
    }
}

impl crate::Model for QuantizedGemma3ForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self.model.forward(input_ids, positions, kv_cache)?;
        let logits = self
            .lm_head
            .forward(&hidden_states)
            .map_err(ModelError::Candle)?;
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn forward_embeds(
        &self,
        inputs_embeds: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self
            .model
            .backbone(inputs_embeds.clone(), positions, kv_cache)?;
        let logits = self
            .lm_head
            .forward(&hidden_states)
            .map_err(ModelError::Candle)?;
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn hidden_states(&self, input_ids: &Tensor, positions: &Tensor) -> ModelResult<Tensor> {
        self.model.forward(input_ids, positions, None)
    }
}

// ---------------------------------------------------------------------------
// QuantizedGemma3ForConditionalGeneration (multimodal GGUF)
// ---------------------------------------------------------------------------

/// Quantized Gemma3 multimodal model: vision encoder + projector + quantized text LM.
///
/// The text backbone is loaded from the main GGUF (quantized), while the vision
/// encoder and projector are loaded from a sibling mmproj.gguf (float).
pub struct QuantizedGemma3ForConditionalGeneration {
    language_model: QuantizedGemma3ForCausalLM,
    vision_tower: crate::siglip::SiglipVisionModel,
    multi_modal_projector: Gemma3MultiModalProjector,
    #[allow(dead_code)]
    image_token_index: u32,
    #[allow(dead_code)]
    mm_tokens_per_image: usize,
    /// Stashed multimodal data for the next forward pass.
    stashed_mm_data: Option<vllm_common::multimodal::MultimodalData>,
    /// Embedding normalizer: sqrt(hidden_size).
    #[allow(dead_code)]
    normalizer: f64,
    dtype: DType,
}

/// Projector for quantized multimodal Gemma3.
///
/// Replicates Gemma3MultiModalProjector from gemma3_mm.rs but can be loaded
/// from mmproj GGUF-dequantized weights (ModelWeights).
struct Gemma3MultiModalProjector {
    mm_input_projection_weight: Tensor,
    mm_soft_emb_norm: GemmaRmsNorm,
    patches_per_image: usize,
    kernel_size: usize,
}

impl Gemma3MultiModalProjector {
    fn load(
        weights: &vllm_model::weight::ModelWeights,
        prefix: &str,
        patches_per_image: usize,
        kernel_size: usize,
        norm_eps: f64,
        dtype: DType,
    ) -> ModelResult<Self> {
        let mm_input_projection_weight =
            weights.get_cast(&format!("{prefix}.mm_input_projection_weight"), dtype)?;
        let mm_soft_emb_norm = GemmaRmsNorm::load(
            weights,
            &format!("{prefix}.mm_soft_emb_norm"),
            norm_eps,
            dtype,
        )?;

        Ok(Self {
            mm_input_projection_weight,
            mm_soft_emb_norm,
            patches_per_image,
            kernel_size,
        })
    }

    /// Forward: vision_outputs `[B, num_patches, vision_hidden]` → `[B, pooled_tokens, text_hidden]`
    fn forward(&self, vision_outputs: &Tensor) -> ModelResult<Tensor> {
        let (batch, _num_patches, seq_length) =
            vision_outputs.dims3().map_err(ModelError::Candle)?;

        let x = vision_outputs.transpose(1, 2).map_err(ModelError::Candle)?;
        let grid = self.patches_per_image;
        let x = x
            .reshape((batch, seq_length, grid, grid))
            .map_err(ModelError::Candle)?;

        // AvgPool2d with kernel_size stride.
        let k = self.kernel_size;
        let out_grid = grid / k;
        let x = x
            .reshape((batch, seq_length, out_grid, k, out_grid, k))
            .map_err(ModelError::Candle)?;
        let x = x
            .mean_keepdim(5)
            .map_err(ModelError::Candle)?
            .mean_keepdim(3)
            .map_err(ModelError::Candle)?;
        let x = x
            .reshape((batch, seq_length, out_grid, out_grid))
            .map_err(ModelError::Candle)?;

        let pooled_tokens = out_grid * out_grid;
        let x = x
            .reshape((batch, seq_length, pooled_tokens))
            .map_err(ModelError::Candle)?;
        let x = x.transpose(1, 2).map_err(ModelError::Candle)?;

        let x = self
            .mm_soft_emb_norm
            .forward(&x)
            .map_err(ModelError::Candle)?;

        let (b, t, _c) = x.dims3().map_err(ModelError::Candle)?;
        let x = x.reshape((b * t, ())).map_err(ModelError::Candle)?;
        let x = x
            .matmul(&self.mm_input_projection_weight)
            .map_err(ModelError::Candle)?;
        let text_hidden = x.dim(1).map_err(ModelError::Candle)?;
        x.reshape((b, t, text_hidden)).map_err(ModelError::Candle)
    }
}

impl QuantizedGemma3ForConditionalGeneration {
    /// Create from a pre-loaded text model and mmproj weights.
    pub fn new(
        language_model: QuantizedGemma3ForCausalLM,
        mmproj_weights: &vllm_model::weight::ModelWeights,
        config: &HfModelConfig,
        dtype: DType,
    ) -> ModelResult<Self> {
        use crate::siglip::{SiglipVisionConfig, SiglipVisionModel};

        // Parse vision config from the HfModelConfig extra fields.
        let vision_json = config
            .extra
            .get("vision_config")
            .ok_or_else(|| ModelError::Other("missing vision_config for mmproj".into()))?;
        let vision_config = SiglipVisionConfig::from_json(vision_json)?;

        let mm_tokens_per_image = config
            .extra
            .get("mm_tokens_per_image")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(vision_config.num_patches());

        let image_token_index = config
            .extra
            .get("image_token_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(255999) as u32;

        let patches_per_image = vision_config.image_size / vision_config.patch_size;
        let tokens_per_side = (mm_tokens_per_image as f64).sqrt() as usize;
        let pool_kernel_size = patches_per_image / tokens_per_side;

        let vision_tower = SiglipVisionModel::load(
            mmproj_weights,
            "vision_tower.vision_model",
            &vision_config,
            dtype,
        )?;

        let multi_modal_projector = Gemma3MultiModalProjector::load(
            mmproj_weights,
            "multi_modal_projector",
            patches_per_image,
            pool_kernel_size,
            vision_config.layer_norm_eps,
            dtype,
        )?;

        let hidden_size = config.hidden_size.unwrap_or(2560);
        let normalizer = (hidden_size as f64).sqrt();

        Ok(Self {
            language_model,
            vision_tower,
            multi_modal_projector,
            image_token_index,
            mm_tokens_per_image,
            stashed_mm_data: None,
            normalizer,
            dtype,
        })
    }

    /// Encode images and merge with text embeddings.
    fn merge_vision_embeddings(
        &self,
        input_ids: &Tensor,
        mm_data: &vllm_common::multimodal::MultimodalData,
    ) -> ModelResult<Tensor> {
        // Get text embeddings (already scaled by sqrt(hidden_size)).
        let text_embeds = self.language_model.model.embed(input_ids)?;
        let device = text_embeds.device().clone();

        if mm_data.images.is_empty() {
            return Ok(text_embeds);
        }

        // Build pixel tensor [N_images, 3, H, W].
        let pixel_tensors: Vec<Tensor> = mm_data
            .images
            .iter()
            .map(|img| {
                Tensor::from_vec(img.pixels.clone(), (1, 3, img.height, img.width), &device)
                    .map_err(ModelError::Candle)?
                    .to_dtype(self.dtype)
                    .map_err(ModelError::Candle)
            })
            .collect::<ModelResult<Vec<_>>>()?;
        let pixel_values = Tensor::cat(&pixel_tensors, 0).map_err(ModelError::Candle)?;

        // Vision encode + project.
        let vision_outputs = self.vision_tower.forward(&pixel_values)?;
        let projected = self.multi_modal_projector.forward(&vision_outputs)?;
        let n_images = projected.dim(0).map_err(ModelError::Candle)?;

        // Scatter image embeddings into text embeddings at placeholder positions.
        let mut merged = text_embeds;
        for (img_idx, placeholder) in mm_data.image_placeholders.iter().enumerate() {
            if img_idx >= n_images {
                break;
            }
            let image_embeds = projected.get(img_idx).map_err(ModelError::Candle)?;
            let n_embed_tokens = image_embeds.dim(0).map_err(ModelError::Candle)?;

            let num_tokens = merged.dim(0).map_err(ModelError::Candle)?;
            let end = (placeholder.offset + placeholder.length).min(num_tokens);
            let actual_len = end.saturating_sub(placeholder.offset);
            if actual_len == 0 {
                continue;
            }

            let use_len = actual_len.min(n_embed_tokens);
            let image_embeds = image_embeds
                .narrow(0, 0, use_len)
                .map_err(ModelError::Candle)?;

            let mut parts = Vec::new();
            if placeholder.offset > 0 {
                parts.push(
                    merged
                        .narrow(0, 0, placeholder.offset)
                        .map_err(ModelError::Candle)?,
                );
            }
            parts.push(image_embeds);
            if end < num_tokens {
                parts.push(
                    merged
                        .narrow(0, end, num_tokens - end)
                        .map_err(ModelError::Candle)?,
                );
            }
            merged = Tensor::cat(&parts, 0).map_err(ModelError::Candle)?;
        }

        Ok(merged)
    }
}

impl crate::Model for QuantizedGemma3ForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        if let Some(mm_data) = &self.stashed_mm_data {
            let merged_embeds = self.merge_vision_embeddings(input_ids, mm_data)?;
            self.language_model
                .forward_embeds(&merged_embeds, positions, kv_cache)
        } else {
            self.language_model.forward(input_ids, positions, kv_cache)
        }
    }

    fn forward_embeds(
        &self,
        inputs_embeds: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        self.language_model
            .forward_embeds(inputs_embeds, positions, kv_cache)
    }

    fn set_mm_data(&mut self, mm_data: Option<vllm_common::multimodal::MultimodalData>) {
        self.stashed_mm_data = mm_data;
    }

    fn num_layers(&self) -> usize {
        self.language_model.num_layers()
    }
}

// ---------------------------------------------------------------------------
// Factory function for the GGUF registry
// ---------------------------------------------------------------------------

/// Create a quantized Gemma3 model from a GGUF file.
pub fn create_gemma3_gguf(
    gguf: &mut GgufFile,
    config: &HfModelConfig,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let gemma3_config = Gemma3Config::from_hf_config(config)?;
    let model = QuantizedGemma3ForCausalLM::load(gguf, &gemma3_config, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemma3_config_from_gguf_metadata() {
        // Verify config parsing works for the quantized path.
        let hf_config = HfModelConfig {
            hidden_size: Some(2560),
            num_attention_heads: Some(8),
            num_key_value_heads: Some(4),
            num_hidden_layers: Some(34),
            intermediate_size: Some(10240),
            vocab_size: Some(262144),
            max_position_embeddings: Some(131072),
            rms_norm_eps: Some(1e-6),
            rope_theta: Some(1_000_000.0),
            head_dim: Some(256),
            extra: std::collections::HashMap::from([
                ("query_pre_attn_scalar".to_string(), serde_json::json!(256)),
                ("sliding_window".to_string(), serde_json::json!(1024)),
                ("sliding_window_pattern".to_string(), serde_json::json!(6)),
                (
                    "rope_local_base_freq".to_string(),
                    serde_json::json!(10000.0),
                ),
            ]),
            ..Default::default()
        };
        let config = Gemma3Config::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 256);
        assert!((config.query_pre_attn_scalar - 256.0).abs() < 1.0);
        assert_eq!(config.sliding_window, Some(1024));
        assert_eq!(config.layer_is_sliding.len(), 34);
        // Layer 0: sliding (pattern 6), Layer 5: global
        assert!(config.layer_is_sliding[0]);
        assert!(!config.layer_is_sliding[5]);
    }

    #[test]
    fn test_gemma3_gguf_registry() {
        let registry = crate::ModelRegistry::default_registry();
        assert!(registry.contains_gguf("gemma3"));
    }
}
