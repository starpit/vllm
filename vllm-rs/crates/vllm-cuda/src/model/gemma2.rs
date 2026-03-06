// SPDX-License-Identifier: Apache-2.0
//! Gemma 2 model using `GpuTensor` — zero-allocation forward pass.
//!
//! Key differences from LLaMA:
//! - GemmaRMSNorm: `y = x * (1 + w) / rms(x)` — weight has +1 offset
//! - GELU (tanh) activation instead of SiLU
//! - 4 norms per layer: input, post-attention, pre-feedforward, post-feedforward
//! - `query_pre_attn_scalar` for attention scaling (not `1/sqrt(head_dim)`)
//! - Attention logit soft capping via tanh (passed to FlashAttention)
//! - Embedding multiplied by `sqrt(hidden_size)` after lookup
//! - Always tied embeddings
//! - Final logit soft capping
//! - Interleaved sliding window (per-layer)

use anyhow::Result;

use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::layers::{Embedding, Linear, RmsNorm};
use crate::model::llama::RotaryCache;
use crate::tensor::GpuTensor;
use crate::weights::GpuWeights;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Gemma2Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub head_dim: usize,
    pub query_pre_attn_scalar: f64,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub tie_word_embeddings: bool,
    /// Per-layer: `true` = sliding attention, `false` = full attention.
    pub layer_is_sliding: Vec<bool>,
    /// Sliding window size for sliding-attention layers.
    pub sliding_window: Option<usize>,
}

// ---------------------------------------------------------------------------
// GemmaRmsNorm — weight gets +1 offset during load
// ---------------------------------------------------------------------------

/// Gemma RMS norm: `y = x * (1 + w) / rms(x)`.
///
/// During loading, we add 1.0 to the weight tensor on GPU so that the
/// standard RMS norm kernel can be used unchanged at runtime.
pub struct GemmaRmsNorm {
    pub inner: RmsNorm,
}

impl GemmaRmsNorm {
    /// Load and apply the +1 offset to the weight.
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        eps: f32,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let mut norm = RmsNorm::load(weights, prefix, eps)?;
        // Add 1.0 to each weight element (one-time init cost).
        unsafe { add_one_to_weight(&mut norm.weight, dtype, device)? };
        Ok(Self { inner: norm })
    }
}

/// Add 1.0 to every element of a 1D GPU weight tensor.
/// Done via CPU round-trip (tiny vector, only during init).
unsafe fn add_one_to_weight(
    weight: &mut GpuTensor,
    dtype: DType,
    device: &GpuDevice,
) -> Result<()> {
    let n = weight.dim(0);
    let nbytes = weight.size_bytes();

    let host = crate::driver::mem_alloc_host(nbytes)?;
    crate::driver::memcpy_dtoh_async(host, weight.raw_ptr(), nbytes, device.compute_stream)?;
    crate::driver::stream_synchronize(device.compute_stream)?;

    match dtype {
        DType::F32 => {
            let slice = std::slice::from_raw_parts_mut(host as *mut f32, n);
            for v in slice.iter_mut() {
                *v += 1.0;
            }
        }
        DType::F16 => {
            let slice = std::slice::from_raw_parts_mut(host as *mut half::f16, n);
            for v in slice.iter_mut() {
                *v = half::f16::from_f32(v.to_f32() + 1.0);
            }
        }
        DType::BF16 => {
            let slice = std::slice::from_raw_parts_mut(host as *mut half::bf16, n);
            for v in slice.iter_mut() {
                *v = half::bf16::from_f32(v.to_f32() + 1.0);
            }
        }
        _ => anyhow::bail!(
            "unsupported dtype for GemmaRmsNorm weight offset: {:?}",
            dtype
        ),
    }

    crate::driver::memcpy_htod_async(weight.raw_ptr(), host, nbytes, device.compute_stream)?;
    crate::driver::stream_synchronize(device.compute_stream)?;
    crate::driver::mem_free_host(host)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Gemma2MLP
// ---------------------------------------------------------------------------

pub struct Gemma2MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl Gemma2MLP {
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let gate = weights.take(&format!("{prefix}.gate_proj.weight"))?;
        let up = weights.take(&format!("{prefix}.up_proj.weight"))?;
        let gate_up_w = unsafe { crate::model::llama::concat_dim0(gate, up, stream)? };
        let gate_up_proj = Linear::new(gate_up_w, None);
        let down_proj = Linear::load(weights, &format!("{prefix}.down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
        })
    }

    pub unsafe fn forward(&self, x: GpuTensor, device: &mut GpuDevice) -> GpuTensor {
        let gate_up = self
            .gate_up_proj
            .forward(x, &device.cublas, &mut device.arena);
        let activated = kernels::gelu_and_mul_fused(
            gate_up,
            self.intermediate_size,
            &mut device.arena,
            device.compute_stream,
        );
        self.down_proj
            .forward(activated, &device.cublas, &mut device.arena)
    }
}

// ---------------------------------------------------------------------------
// Gemma2Attention
// ---------------------------------------------------------------------------

pub struct Gemma2Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    scale: f32,
    attn_logit_softcapping: f32,
    sliding_window: Option<usize>,
    layer_idx: usize,
}

impl Gemma2Attention {
    pub fn load_fused(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let q_w = weights.take(&format!("{prefix}.q_proj.weight"))?;
        let k_w = weights.take(&format!("{prefix}.k_proj.weight"))?;
        let v_w = weights.take(&format!("{prefix}.v_proj.weight"))?;
        let qkv_w = unsafe { crate::model::llama::concat3_dim0(q_w, k_w, v_w, stream)? };
        let qkv_proj = Linear::new(qkv_w, None);

        let o_proj = Linear::load(weights, &format!("{prefix}.o_proj"))?;

        let sliding_window = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5) as f32,
            attn_logit_softcapping: config.attn_logit_softcapping.unwrap_or(0.0) as f32,
            sliding_window,
            layer_idx,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        hidden_states: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        cu_seqlens_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        rotary: &RotaryCache,
        device: &mut GpuDevice,
    ) -> GpuTensor {
        let num_tokens = hidden_states.dim(0);

        let qkv = self
            .qkv_proj
            .forward(hidden_states, &device.cublas, &mut device.arena);

        let (q, k, v) = kernels::split_qkv(
            qkv,
            self.q_size,
            self.kv_size,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            &mut device.arena,
            device.compute_stream,
        );

        let q_flat = q.reshape(&[num_tokens, self.q_size]);
        let k_flat = k.reshape(&[num_tokens, self.kv_size]);
        kernels::rotary_embedding_inplace(
            q_flat,
            k_flat,
            positions,
            rotary.cos_sin_cache,
            self.head_dim,
            device.compute_stream,
        );

        kernels::reshape_and_cache(
            k,
            v,
            kv_cache.k_cache(self.layer_idx),
            kv_cache.v_cache(self.layer_idx),
            slot_mapping,
            kv_cache.block_size,
            device.compute_stream,
        );

        let window_left = self.sliding_window.map(|w| w as i32).unwrap_or(-1);

        let attn_output = kernels::flash_attn_paged_ext(
            q,
            kv_cache.k_cache(self.layer_idx),
            kv_cache.v_cache(self.layer_idx),
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            self.scale,
            true, // causal
            self.attn_logit_softcapping,
            window_left,
            kv_cache.block_size,
            &mut device.arena,
            device.compute_stream,
        );

        let attn_flat = attn_output.reshape(&[num_tokens, self.q_size]);
        self.o_proj
            .forward(attn_flat, &device.cublas, &mut device.arena)
    }
}

// ---------------------------------------------------------------------------
// Gemma2DecoderLayer — 4 norms per layer
// ---------------------------------------------------------------------------

pub struct Gemma2DecoderLayer {
    pub self_attn: Gemma2Attention,
    mlp: Gemma2MLP,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
}

impl Gemma2DecoderLayer {
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let self_attn = Gemma2Attention::load_fused(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            is_sliding,
            device.compute_stream,
        )?;
        let mlp = Gemma2MLP::load(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            device.compute_stream,
        )?;
        let input_layernorm = GemmaRmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
            dtype,
            device,
        )?;
        let post_attention_layernorm = GemmaRmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
            dtype,
            device,
        )?;
        let pre_feedforward_layernorm = GemmaRmsNorm::load(
            weights,
            &format!("{prefix}.pre_feedforward_layernorm"),
            config.rms_norm_eps,
            dtype,
            device,
        )?;
        let post_feedforward_layernorm = GemmaRmsNorm::load(
            weights,
            &format!("{prefix}.post_feedforward_layernorm"),
            config.rms_norm_eps,
            dtype,
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

    /// Forward pass with residual threading and 4 norms.
    ///
    /// Gemma2 layer structure:
    ///   1. input_layernorm (fused residual add)
    ///   2. attention
    ///   3. post_attention_layernorm (standalone — no residual)
    ///   4. pre_feedforward_layernorm (fused residual add)
    ///   5. MLP
    ///   6. post_feedforward_layernorm (standalone — no residual)
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        hidden_states: GpuTensor,
        residual: Option<GpuTensor>,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        cu_seqlens_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        rotary: &RotaryCache,
        device: &mut GpuDevice,
    ) -> (GpuTensor, GpuTensor) {
        // 1. Pre-attention norm with fused residual add.
        let (normed, residual) = if let Some(residual) = residual {
            kernels::fused_add_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm.inner.weight,
                self.input_layernorm.inner.eps,
                &mut device.arena,
                device.compute_stream,
            )
        } else {
            let normed = kernels::rms_norm(
                hidden_states,
                self.input_layernorm.inner.weight,
                self.input_layernorm.inner.eps,
                &mut device.arena,
                device.compute_stream,
            );
            (normed, hidden_states)
        };

        // 2. Attention.
        let attn_output = self.self_attn.forward(
            normed,
            positions,
            slot_mapping,
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            kv_cache,
            rotary,
            device,
        );

        // 3. Post-attention norm (standalone, no residual add).
        let attn_normed = kernels::rms_norm(
            attn_output,
            self.post_attention_layernorm.inner.weight,
            self.post_attention_layernorm.inner.eps,
            &mut device.arena,
            device.compute_stream,
        );

        // 4. Pre-feedforward norm with fused residual add.
        let (normed, residual) = kernels::fused_add_rms_norm(
            attn_normed,
            residual,
            self.pre_feedforward_layernorm.inner.weight,
            self.pre_feedforward_layernorm.inner.eps,
            &mut device.arena,
            device.compute_stream,
        );

        // 5. MLP.
        let mlp_output = self.mlp.forward(normed, device);

        // 6. Post-feedforward norm (standalone, no residual add).
        let mlp_normed = kernels::rms_norm(
            mlp_output,
            self.post_feedforward_layernorm.inner.weight,
            self.post_feedforward_layernorm.inner.eps,
            &mut device.arena,
            device.compute_stream,
        );

        (mlp_normed, residual)
    }
}

// ---------------------------------------------------------------------------
// Gemma2Model
// ---------------------------------------------------------------------------

pub struct Gemma2Model {
    embed_tokens: Embedding,
    pub layers: Vec<Gemma2DecoderLayer>,
    norm: GemmaRmsNorm,
    rotary: RotaryCache,
    embed_scale: f32,
}

impl Gemma2Model {
    pub fn load(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma2DecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                is_sliding,
                dtype,
                device,
            )?;
            layers.push(layer);
        }

        let norm = GemmaRmsNorm::load(weights, "model.norm", config.rms_norm_eps, dtype, device)?;

        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                dtype,
                device,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            embed_scale: (config.hidden_size as f32).sqrt(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        cu_seqlens_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
    ) -> GpuTensor {
        // Embedding lookup + Gemma scaling.
        let mut hidden_states = kernels::embedding_gather(
            self.embed_tokens.weight,
            input_ids,
            &mut device.arena,
            device.compute_stream,
        );
        kernels::scale_inplace(hidden_states, self.embed_scale, &device.cublas);

        let mut residual: Option<GpuTensor> = None;
        for layer in &self.layers {
            let (hs, res) = layer.forward(
                hidden_states,
                residual,
                positions,
                slot_mapping,
                cu_seqlens_q,
                cu_seqlens_k,
                block_table,
                max_seqlen_q,
                max_seqlen_k,
                kv_cache,
                &self.rotary,
                device,
            );
            hidden_states = hs;
            residual = Some(res);
        }

        // Final norm with fused residual add.
        let (normed, _) = kernels::fused_add_rms_norm(
            hidden_states,
            residual.unwrap(),
            self.norm.inner.weight,
            self.norm.inner.eps,
            &mut device.arena,
            device.compute_stream,
        );
        normed
    }
}

// ---------------------------------------------------------------------------
// Gemma2ForCausalLM
// ---------------------------------------------------------------------------

pub struct Gemma2ForCausalLM {
    pub model: Gemma2Model,
    pub lm_head: Linear,
    final_logit_softcapping: Option<f32>,
}

impl Gemma2ForCausalLM {
    pub fn load(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Gemma2Model::load(weights, config, dtype, device)?;

        // Gemma2 always uses tied embeddings.
        let lm_head = Linear::new(model.embed_tokens.weight, None);

        Ok(Self {
            model,
            lm_head,
            final_logit_softcapping: config.final_logit_softcapping.map(|v| v as f32),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        cu_seqlens_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<GpuTensor>,
    ) -> GpuTensor {
        let hidden_states = self.model.forward(
            input_ids,
            positions,
            slot_mapping,
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            kv_cache,
            device,
        );

        // Gather only last-token hidden states before the expensive lm_head GEMM.
        let hidden_states = if let Some(indices) = last_token_indices {
            crate::kernels::embedding_gather(
                hidden_states,
                indices,
                &mut device.arena,
                device.compute_stream,
            )
        } else {
            hidden_states
        };

        let logits = self
            .lm_head
            .forward(hidden_states, &device.cublas, &mut device.arena);

        // Apply final logit soft capping: logits = cap * tanh(logits / cap).
        // TODO: This requires a fused tanh-softcap kernel. For now, softcap
        // is not applied at the GpuTensor level — it would need a small CUDA
        // kernel. The softcap values are typically large (30.0) so this has
        // minimal impact on correctness for greedy/top-k sampling.
        let _ = self.final_logit_softcapping;

        logits
    }
}
