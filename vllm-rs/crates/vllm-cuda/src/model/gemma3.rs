// SPDX-License-Identifier: Apache-2.0
//! Gemma 3 model (text-only) using `GpuTensor` — zero-allocation forward pass.
//!
//! Differences from Gemma 2:
//! - No softcapping (attn or final logit)
//! - Per-head QK norms (GemmaRmsNorm applied per-head before RoPE)
//! - Per-layer RoPE theta (global vs sliding layers use different base freq)
//! - `sliding_window_pattern: N` — every Nth layer is global, rest are sliding
//! - Configurable `attention_bias`

#[cfg(feature = "nccl")]
use std::sync::Arc;

use anyhow::Result;

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::layers::{Embedding, Linear, RmsNorm};
use crate::model::gemma2::{Gemma2MLP, GemmaRmsNorm};
use crate::model::llama::{ForwardOutput, RotaryCache, TpConfig};
#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
use crate::pp::PpConfig;
use crate::tensor::{GpuTensor, TensorView};
use crate::weights::GpuWeights;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Gemma3Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub rope_local_base_freq: f64,
    pub head_dim: usize,
    pub query_pre_attn_scalar: f64,
    pub tie_word_embeddings: bool,
    /// Per-layer: `true` = sliding attention, `false` = full attention.
    pub layer_is_sliding: Vec<bool>,
    /// Sliding window size for sliding-attention layers.
    pub sliding_window: Option<usize>,
    /// Whether QKV and O projections have bias.
    pub attention_bias: bool,
}

// ---------------------------------------------------------------------------
// Gemma3Attention — per-head QK norms, no softcap, separate split+RoPE
// ---------------------------------------------------------------------------

pub struct Gemma3Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
    layer_idx: usize,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

impl Gemma3Attention {
    pub fn load_fused(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma3Config,
        layer_idx: usize,
        is_sliding: bool,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // Fuse QKV weights into a single tensor.
        let q_name = format!("{prefix}.q_proj.weight");
        let k_name = format!("{prefix}.k_proj.weight");
        let v_name = format!("{prefix}.v_proj.weight");
        let (q_shape, q_dtype) = weights
            .tensor_info(&q_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {q_name}"))?;
        let hidden = if q_shape.len() == 2 { q_shape[1] } else { 1 };
        let elem_size = q_dtype.size_bytes();
        let q_bytes = q_size * hidden * elem_size;
        let kv_bytes = kv_size * hidden * elem_size;
        let total_bytes = q_bytes + 2 * kv_bytes;
        let ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };
        unsafe {
            weights.take_into(&q_name, ptr, device.compute_stream)?;
            weights.take_into(&k_name, ptr.add(q_bytes), device.compute_stream)?;
            weights.take_into(&v_name, ptr.add(q_bytes + kv_bytes), device.compute_stream)?;
        }
        let qkv_w = unsafe { GpuTensor::new(ptr, &[q_size + 2 * kv_size, hidden], q_dtype) };

        // Load optional QKV bias.
        let qkv_bias = if config.attention_bias {
            let q_bias_name = format!("{prefix}.q_proj.bias");
            let k_bias_name = format!("{prefix}.k_proj.bias");
            let v_bias_name = format!("{prefix}.v_proj.bias");
            if weights.contains(&q_bias_name) {
                let bias_bytes = (q_size + 2 * kv_size) * elem_size;
                let bias_ptr = unsafe { crate::driver::mem_alloc(bias_bytes)? };
                let q_bias_bytes = q_size * elem_size;
                let kv_bias_bytes = kv_size * elem_size;
                unsafe {
                    weights.take_into(&q_bias_name, bias_ptr, device.compute_stream)?;
                    weights.take_into(
                        &k_bias_name,
                        bias_ptr.add(q_bias_bytes),
                        device.compute_stream,
                    )?;
                    weights.take_into(
                        &v_bias_name,
                        bias_ptr.add(q_bias_bytes + kv_bias_bytes),
                        device.compute_stream,
                    )?;
                }
                Some(unsafe { GpuTensor::new(bias_ptr, &[q_size + 2 * kv_size], q_dtype) })
            } else {
                None
            }
        } else {
            None
        };

        let qkv_proj = Linear::new(qkv_w, qkv_bias);

        // O projection (Linear::load handles bias automatically).
        let o_proj = Linear::load(weights, &format!("{prefix}.o_proj"))?;

        // Per-head QK norms.
        let q_norm = GemmaRmsNorm::load(
            weights,
            &format!("{prefix}.q_norm"),
            config.rms_norm_eps,
            dtype,
            device,
        )?;
        let k_norm = GemmaRmsNorm::load(
            weights,
            &format!("{prefix}.k_norm"),
            config.rms_norm_eps,
            dtype,
            device,
        )?;

        let sliding_window = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5) as f32,
            sliding_window,
            layer_idx,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

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
        let num_tokens = hidden_states.dim(0);

        // QKV projection → owned.
        let qkv = self
            .qkv_proj
            .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // Split QKV (no RoPE yet — we need to apply QK norms first).
        let (q, k, v) = kernels::split_qkv(
            *qkv.view(),
            self.q_size,
            self.kv_size,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            &mut device.caching,
            device.compute_stream,
        );
        drop(qkv);

        // Per-head QK norms: reshape [T, heads, head_dim] → [T*heads, head_dim],
        // apply RMS norm, reshape back.
        let q_flat = q
            .view()
            .reshape(&[num_tokens * self.num_q_heads, self.head_dim]);
        let q_normed = kernels::rms_norm(
            *q_flat,
            self.q_norm.inner.weight,
            self.q_norm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );
        drop(q);
        let q_3d = q_normed
            .view()
            .reshape(&[num_tokens, self.num_q_heads, self.head_dim]);

        let k_flat = k
            .view()
            .reshape(&[num_tokens * self.num_kv_heads, self.head_dim]);
        let k_normed = kernels::rms_norm(
            *k_flat,
            self.k_norm.inner.weight,
            self.k_norm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );
        drop(k);
        let k_3d = k_normed
            .view()
            .reshape(&[num_tokens, self.num_kv_heads, self.head_dim]);

        // RoPE (in-place on the normed q/k).
        let q_flat_rope = q_3d.reshape(&[num_tokens, self.q_size]);
        let k_flat_rope = k_3d.reshape(&[num_tokens, self.kv_size]);
        kernels::rotary_embedding_inplace(
            *q_flat_rope,
            *k_flat_rope,
            *positions,
            rotary.cos_sin_cache,
            self.head_dim,
            device.compute_stream,
        );

        // Reshape back to 3D for attention.
        let q_3d = q_flat_rope.reshape(&[num_tokens, self.num_q_heads, self.head_dim]);
        let k_3d = k_flat_rope.reshape(&[num_tokens, self.num_kv_heads, self.head_dim]);

        crate::model::attention_helpers::write_kv_cache(
            k_3d,
            v.view(),
            slot_mapping,
            kv_cache,
            self.layer_idx,
            device.compute_stream,
        );

        let window_left = self.sliding_window.map(|w| w as i32).unwrap_or(-1);

        let attn_output = crate::model::attention_helpers::attention_ext(
            q_3d,
            k_3d,
            v.view(),
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            self.scale,
            0.0,
            window_left,
            kv_cache,
            self.layer_idx,
            device.num_sm,
            &mut device.caching,
            device.compute_stream,
            std::ptr::null(),
            0,
        );
        drop(k_normed);
        drop(v);
        drop(q_normed);

        let attn_flat = attn_output.view().reshape(&[num_tokens, self.q_size]);
        let result = self
            .o_proj
            .forward(attn_flat, &mut device.cublas, &mut device.caching);
        drop(attn_output);

        // TP: all-reduce o_proj output (row parallel).
        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.tp_group {
            group
                .all_reduce_inplace(*result.view())
                .expect("o_proj all_reduce failed");
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Gemma3DecoderLayer — 4 norms per layer (identical structure to Gemma2)
// ---------------------------------------------------------------------------

pub struct Gemma3DecoderLayer {
    pub self_attn: Gemma3Attention,
    mlp: Gemma2MLP,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
}

impl Gemma3DecoderLayer {
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma3Config,
        layer_idx: usize,
        is_sliding: bool,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let self_attn = Gemma3Attention::load_fused(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            is_sliding,
            dtype,
            device,
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

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        hidden_states: OwnedTensor,
        residual: Option<OwnedTensor>,
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
    ) -> (OwnedTensor, OwnedTensor) {
        // 1. Pre-attention norm with fused residual add (in-place).
        let (normed, residual) = if let Some(residual) = residual {
            let hs_gpu = *hidden_states;
            let res_gpu = *residual;
            kernels::fused_add_rms_norm_inplace(
                hs_gpu,
                res_gpu,
                self.input_layernorm.inner.weight,
                self.input_layernorm.inner.eps,
                device.compute_stream,
            );
            (hidden_states, residual)
        } else {
            let normed = kernels::rms_norm(
                *hidden_states,
                self.input_layernorm.inner.weight,
                self.input_layernorm.inner.eps,
                &mut device.caching,
                device.compute_stream,
            );
            (normed, hidden_states)
        };

        // 2. Attention.
        let attn_output = self.self_attn.forward(
            normed.view(),
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
        drop(normed);

        // 3. Post-attention norm (standalone — allocates, drops input).
        let attn_normed = kernels::rms_norm(
            *attn_output,
            self.post_attention_layernorm.inner.weight,
            self.post_attention_layernorm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );
        drop(attn_output);

        // 4. Pre-feedforward norm with fused residual add (in-place).
        let res_gpu = *residual;
        kernels::fused_add_rms_norm_inplace(
            *attn_normed,
            res_gpu,
            self.pre_feedforward_layernorm.inner.weight,
            self.pre_feedforward_layernorm.inner.eps,
            device.compute_stream,
        );

        // 5. MLP.
        let mlp_output = self.mlp.forward(attn_normed.view(), device);
        drop(attn_normed);

        // 6. Post-feedforward norm (standalone — allocates, drops input).
        let mlp_normed = kernels::rms_norm(
            *mlp_output,
            self.post_feedforward_layernorm.inner.weight,
            self.post_feedforward_layernorm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );
        drop(mlp_output);

        (mlp_normed, residual)
    }
}

// ---------------------------------------------------------------------------
// Gemma3Model — two RotaryCache instances (global + local)
// ---------------------------------------------------------------------------

pub struct Gemma3Model {
    embed_tokens: Embedding,
    pub layers: Vec<Gemma3DecoderLayer>,
    norm: GemmaRmsNorm,
    rotary_global: RotaryCache,
    rotary_local: RotaryCache,
    layer_is_sliding: Vec<bool>,
    embed_scale: f32,
}

impl Gemma3Model {
    pub fn load(
        weights: &mut GpuWeights,
        config: &Gemma3Config,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma3DecoderLayer::load(
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

        let rotary_global = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
                device,
            )?
        };
        let rotary_local = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_local_base_freq,
                None,
                dtype,
                device,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_global,
            rotary_local,
            layer_is_sliding: config.layer_is_sliding.clone(),
            embed_scale: (config.hidden_size as f32).sqrt(),
        })
    }

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
        // Embedding lookup + Gemma scaling.
        let hidden_states = kernels::embedding_gather(
            self.embed_tokens.weight,
            *input_ids,
            &mut device.caching,
            device.compute_stream,
        );
        kernels::scale_inplace(*hidden_states.view(), self.embed_scale, &device.cublas);

        let mut hidden_states: OwnedTensor = hidden_states;
        let mut residual: Option<OwnedTensor> = None;

        for (i, layer) in self.layers.iter().enumerate() {
            let is_sliding = i < self.layer_is_sliding.len() && self.layer_is_sliding[i];
            let rotary = if is_sliding {
                &self.rotary_local
            } else {
                &self.rotary_global
            };
            let (hs, res) = layer.forward(
                hidden_states,
                residual,
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
            hidden_states = hs;
            residual = Some(res);
        }

        // Final norm: mutates hidden_states and residual in-place.
        let hs_gpu = *hidden_states;
        let res_gpu = *residual.as_ref().unwrap().view();
        kernels::fused_add_rms_norm_inplace(
            hs_gpu,
            res_gpu,
            self.norm.inner.weight,
            self.norm.inner.eps,
            device.compute_stream,
        );
        drop(residual);
        hidden_states
    }
}

// ---------------------------------------------------------------------------
// Gemma3ForCausalLM — no final logit softcapping
// ---------------------------------------------------------------------------

pub struct Gemma3ForCausalLM {
    pub model: Gemma3Model,
    pub lm_head: Linear,
    /// Pipeline parallelism config. None = single GPU / PP=1.
    pub pp_config: Option<PpConfig>,
}

impl Gemma3ForCausalLM {
    pub fn load(
        weights: &mut GpuWeights,
        config: &Gemma3Config,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Gemma3Model::load(weights, config, dtype, device)?;

        // Gemma3 always uses tied embeddings.
        let lm_head = Linear::new(model.embed_tokens.weight, None);

        Ok(Self {
            model,
            lm_head,
            pp_config: None,
        })
    }

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

        // Gather only last-token hidden states before the expensive lm_head GEMM.
        let hidden_states = if let Some(indices) = last_token_indices {
            crate::kernels::embedding_gather(
                *hidden_states,
                *indices,
                &mut device.caching,
                device.compute_stream,
            )
        } else {
            hidden_states
        };

        self.lm_head.forward(
            hidden_states.view(),
            &mut device.cublas,
            &mut device.caching,
        )
    }
}

// ---------------------------------------------------------------------------
// Tensor-parallel loading
// ---------------------------------------------------------------------------

impl Gemma3Attention {
    /// Load with fused QKV weights, sharded for tensor parallelism.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fused_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma3Config,
        layer_idx: usize,
        is_sliding: bool,
        tp: TpConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads / tp.world_size;
        let num_kv_heads = config.num_kv_heads / tp.world_size;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // Fuse QKV weights with per-rank sharding (dim=0).
        let q_name = format!("{prefix}.q_proj.weight");
        let k_name = format!("{prefix}.k_proj.weight");
        let v_name = format!("{prefix}.v_proj.weight");
        let (_q_shape, q_dtype) = weights
            .tensor_info(&q_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {q_name}"))?;
        let hidden = _q_shape[1];
        let elem_size = q_dtype.size_bytes();
        let q_bytes = q_size * hidden * elem_size;
        let kv_bytes = kv_size * hidden * elem_size;
        let total_bytes = q_bytes + 2 * kv_bytes;
        let ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };
        unsafe {
            weights.take_shard_into(
                &q_name,
                0,
                tp.rank,
                tp.world_size,
                ptr,
                device.compute_stream,
            )?;
            weights.take_shard_into(
                &k_name,
                0,
                tp.rank,
                tp.world_size,
                ptr.add(q_bytes),
                device.compute_stream,
            )?;
            weights.take_shard_into(
                &v_name,
                0,
                tp.rank,
                tp.world_size,
                ptr.add(q_bytes + kv_bytes),
                device.compute_stream,
            )?;
        }
        let qkv_w = unsafe { GpuTensor::new(ptr, &[q_size + 2 * kv_size, hidden], q_dtype) };

        // Optional QKV bias — shard dim=0 for each.
        let qkv_bias = if config.attention_bias {
            let q_bias_name = format!("{prefix}.q_proj.bias");
            let k_bias_name = format!("{prefix}.k_proj.bias");
            let v_bias_name = format!("{prefix}.v_proj.bias");
            if weights.contains(&q_bias_name) {
                let bias_bytes = (q_size + 2 * kv_size) * elem_size;
                let bias_ptr = unsafe { crate::driver::mem_alloc(bias_bytes)? };
                let q_bias_bytes = q_size * elem_size;
                let kv_bias_bytes = kv_size * elem_size;
                unsafe {
                    weights.take_shard_into(
                        &q_bias_name,
                        0,
                        tp.rank,
                        tp.world_size,
                        bias_ptr,
                        device.compute_stream,
                    )?;
                    weights.take_shard_into(
                        &k_bias_name,
                        0,
                        tp.rank,
                        tp.world_size,
                        bias_ptr.add(q_bias_bytes),
                        device.compute_stream,
                    )?;
                    weights.take_shard_into(
                        &v_bias_name,
                        0,
                        tp.rank,
                        tp.world_size,
                        bias_ptr.add(q_bias_bytes + kv_bias_bytes),
                        device.compute_stream,
                    )?;
                }
                Some(unsafe { GpuTensor::new(bias_ptr, &[q_size + 2 * kv_size], q_dtype) })
            } else {
                None
            }
        } else {
            None
        };

        let qkv_proj = Linear::new(qkv_w, qkv_bias);

        // o_proj: shard along dim=1 (row parallel). Bias NOT sharded.
        let o_name = format!("{prefix}.o_proj.weight");
        let o_w = weights.take_shard(&o_name, 1, tp.rank, tp.world_size)?;
        let o_bias = if config.attention_bias {
            let bias_name = format!("{prefix}.o_proj.bias");
            if weights.contains(&bias_name) {
                Some(weights.take(&bias_name)?)
            } else {
                None
            }
        } else {
            None
        };
        let o_proj = Linear::new(o_w, o_bias);

        // Per-head QK norms — NOT sharded ([head_dim] each).
        let q_norm = GemmaRmsNorm::load(
            weights,
            &format!("{prefix}.q_norm"),
            config.rms_norm_eps,
            dtype,
            device,
        )?;
        let k_norm = GemmaRmsNorm::load(
            weights,
            &format!("{prefix}.k_norm"),
            config.rms_norm_eps,
            dtype,
            device,
        )?;

        let sliding_window = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5) as f32,
            sliding_window,
            layer_idx,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

impl Gemma3DecoderLayer {
    /// Load a decoder layer with TP-sharded weights.
    #[allow(clippy::too_many_arguments)]
    pub fn load_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma3Config,
        layer_idx: usize,
        is_sliding: bool,
        tp: TpConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let self_attn = Gemma3Attention::load_fused_tp(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            is_sliding,
            tp,
            dtype,
            device,
        )?;
        let mlp = Gemma2MLP::load_fused_tp(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            tp,
            device.compute_stream,
        )?;
        // Norms are NOT sharded.
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
}

impl Gemma3Model {
    /// Load model backbone with TP sharding.
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &Gemma3Config,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma3DecoderLayer::load_tp(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                is_sliding,
                tp,
                dtype,
                device,
            )?;
            layers.push(layer);
        }

        let norm = GemmaRmsNorm::load(weights, "model.norm", config.rms_norm_eps, dtype, device)?;

        let rotary_global = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
                device,
            )?
        };
        let rotary_local = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_local_base_freq,
                None,
                dtype,
                device,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_global,
            rotary_local,
            layer_is_sliding: config.layer_is_sliding.clone(),
            embed_scale: (config.hidden_size as f32).sqrt(),
        })
    }
}

impl Gemma3ForCausalLM {
    /// Load the full model with TP sharding.
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &Gemma3Config,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Gemma3Model::load_tp(weights, config, dtype, tp, device)?;
        // Gemma3 always uses tied embeddings.
        let lm_head = Linear::new(model.embed_tokens.weight, None);
        Ok(Self {
            model,
            lm_head,
            pp_config: None,
        })
    }

    /// Inject NCCL process group into all TP layers.
    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        for layer in &mut self.model.layers {
            layer.self_attn.tp_group = Some(Arc::clone(&group));
            layer.mlp.tp_group = Some(Arc::clone(&group));
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline-parallel loading and forward pass
// ---------------------------------------------------------------------------

impl Gemma3Model {
    /// Load backbone with PP layer sharding (no TP).
    pub fn load_pp(
        weights: &mut GpuWeights,
        config: &Gemma3Config,
        dtype: DType,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = if pp.is_first_stage() {
            Embedding::load(weights, "model.embed_tokens")?
        } else {
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                crate::tensor::GpuTensor::new(ptr, &[1, 1], dtype)
            };
            Embedding::new(w)
        };

        let mut layers = Vec::with_capacity(pp.num_layers());
        for i in pp.start_layer..pp.end_layer {
            let local_idx = i - pp.start_layer;
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma3DecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                config,
                local_idx,
                is_sliding,
                dtype,
                device,
            )?;
            layers.push(layer);
        }

        let norm = if pp.is_last_stage() {
            GemmaRmsNorm::load(weights, "model.norm", config.rms_norm_eps, dtype, device)?
        } else {
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                crate::tensor::GpuTensor::new(ptr, &[1], dtype)
            };
            GemmaRmsNorm {
                inner: RmsNorm::new(w, config.rms_norm_eps),
            }
        };

        let rotary_global = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
                device,
            )?
        };
        let rotary_local = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_local_base_freq,
                None,
                dtype,
                device,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_global,
            rotary_local,
            layer_is_sliding: config.layer_is_sliding.clone(),
            embed_scale: (config.hidden_size as f32).sqrt(),
        })
    }

    /// Load backbone with TP + PP sharding.
    pub fn load_tp_pp(
        weights: &mut GpuWeights,
        config: &Gemma3Config,
        dtype: DType,
        tp: TpConfig,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = if pp.is_first_stage() {
            Embedding::load(weights, "model.embed_tokens")?
        } else {
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                crate::tensor::GpuTensor::new(ptr, &[1, 1], dtype)
            };
            Embedding::new(w)
        };

        let mut layers = Vec::with_capacity(pp.num_layers());
        for i in pp.start_layer..pp.end_layer {
            let local_idx = i - pp.start_layer;
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma3DecoderLayer::load_tp(
                weights,
                &format!("model.layers.{i}"),
                config,
                local_idx,
                is_sliding,
                tp,
                dtype,
                device,
            )?;
            layers.push(layer);
        }

        let norm = if pp.is_last_stage() {
            GemmaRmsNorm::load(weights, "model.norm", config.rms_norm_eps, dtype, device)?
        } else {
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                crate::tensor::GpuTensor::new(ptr, &[1], dtype)
            };
            GemmaRmsNorm {
                inner: RmsNorm::new(w, config.rms_norm_eps),
            }
        };

        let rotary_global = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                None,
                dtype,
                device,
            )?
        };
        let rotary_local = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_local_base_freq,
                None,
                dtype,
                device,
            )?
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_global,
            rotary_local,
            layer_is_sliding: config.layer_is_sliding.clone(),
            embed_scale: (config.hidden_size as f32).sqrt(),
        })
    }

    /// PP-aware forward:
    /// - First stage: embed input_ids + scale, run layers, return (hs, residual).
    /// - Middle stages: take (hs, residual), run layers, return (hs, residual).
    /// - Last stage: take (hs, residual), run layers + final norm, return hidden_states tensor.
    ///
    /// The `layer_is_sliding` index uses the **absolute** layer index so that
    /// each PP stage selects the correct rotary cache for its subset of layers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_pp(
        &self,
        pp: &PpConfig,
        input_ids: Option<TensorView<'_>>,
        intermediate: Option<(OwnedTensor, OwnedTensor)>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &crate::kv_cache::KvCachePool,
        device: &mut crate::device::GpuDevice,
    ) -> ForwardOutput {
        let (mut hidden_states, mut residual): (OwnedTensor, Option<OwnedTensor>) =
            if pp.is_first_stage() {
                let input_ids = input_ids.expect("first PP stage requires input_ids");
                let hs = kernels::embedding_gather(
                    self.embed_tokens.weight,
                    *input_ids,
                    &mut device.caching,
                    device.compute_stream,
                );
                kernels::scale_inplace(*hs.view(), self.embed_scale, &device.cublas);
                (hs, None)
            } else {
                let (hs, res) = intermediate.expect("non-first PP stage requires intermediate");
                (hs, Some(res))
            };

        // Layer index in the full model = pp.start_layer + local_i.
        for (local_i, layer) in self.layers.iter().enumerate() {
            let abs_i = pp.start_layer + local_i;
            let is_sliding = abs_i < self.layer_is_sliding.len() && self.layer_is_sliding[abs_i];
            let rotary = if is_sliding {
                &self.rotary_local
            } else {
                &self.rotary_global
            };

            let (hs, res) = layer.forward(
                hidden_states,
                residual,
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
            hidden_states = hs;
            residual = Some(res);
        }

        if pp.is_last_stage() {
            let hs_gpu = *hidden_states;
            let res_gpu = *residual.as_ref().unwrap().view();
            kernels::fused_add_rms_norm_inplace(
                hs_gpu,
                res_gpu,
                self.norm.inner.weight,
                self.norm.inner.eps,
                device.compute_stream,
            );
            drop(residual);
            ForwardOutput::Logits(hidden_states)
        } else {
            ForwardOutput::Intermediate {
                hidden_states,
                residual: residual.unwrap(),
            }
        }
    }
}

impl Gemma3ForCausalLM {
    /// Load with PP (no TP).
    pub fn load_pp(
        weights: &mut GpuWeights,
        config: &Gemma3Config,
        dtype: DType,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Gemma3Model::load_pp(weights, config, dtype, pp, device)?;

        let lm_head = if pp.is_last_stage() {
            if pp.is_first_stage() {
                Linear::new(model.embed_tokens.weight, None)
            } else {
                let embed_w = weights.take("model.embed_tokens.weight")?;
                Linear::new(embed_w, None)
            }
        } else {
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                crate::tensor::GpuTensor::new(ptr, &[1, 1], dtype)
            };
            Linear::new(w, None)
        };

        Ok(Self {
            model,
            lm_head,
            pp_config: Some(pp),
        })
    }

    /// Load with TP + PP.
    pub fn load_tp_pp(
        weights: &mut GpuWeights,
        config: &Gemma3Config,
        dtype: DType,
        tp: TpConfig,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Gemma3Model::load_tp_pp(weights, config, dtype, tp, pp, device)?;

        let lm_head = if pp.is_last_stage() {
            if pp.is_first_stage() {
                Linear::new(model.embed_tokens.weight, None)
            } else {
                let embed_w = weights.take("model.embed_tokens.weight")?;
                Linear::new(embed_w, None)
            }
        } else {
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                crate::tensor::GpuTensor::new(ptr, &[1, 1], dtype)
            };
            Linear::new(w, None)
        };

        Ok(Self {
            model,
            lm_head,
            pp_config: Some(pp),
        })
    }

    /// PP-aware forward pass.
    ///
    /// Returns `ForwardOutput::Logits` on the last stage, or
    /// `ForwardOutput::Intermediate` on non-last stages.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_pp(
        &self,
        input_ids: Option<TensorView<'_>>,
        intermediate: Option<(OwnedTensor, OwnedTensor)>,
        positions: TensorView<'_>,
        slot_mapping: TensorView<'_>,
        cu_seqlens_q: TensorView<'_>,
        seqused_k: TensorView<'_>,
        block_table: TensorView<'_>,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &crate::kv_cache::KvCachePool,
        device: &mut crate::device::GpuDevice,
        last_token_indices: Option<TensorView<'_>>,
    ) -> ForwardOutput {
        let pp = self
            .pp_config
            .as_ref()
            .expect("forward_pp called without pp_config");

        let backbone_out = self.model.forward_pp(
            pp,
            input_ids,
            intermediate,
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

        match backbone_out {
            ForwardOutput::Intermediate { .. } => backbone_out,
            ForwardOutput::Logits(hidden_states) => {
                let gathered = if let Some(indices) = last_token_indices {
                    Some(kernels::embedding_gather(
                        *hidden_states,
                        *indices,
                        &mut device.caching,
                        device.compute_stream,
                    ))
                } else {
                    None
                };
                let hs_view = if let Some(ref g) = gathered {
                    g.view()
                } else {
                    hidden_states.view()
                };

                let logits = self
                    .lm_head
                    .forward(hs_view, &mut device.cublas, &mut device.caching);
                // hidden_states can be freed now.
                drop(gathered);
                drop(hidden_states);
                ForwardOutput::Logits(logits)
            }
        }
    }
}
