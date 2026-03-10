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
use crate::layers::{Embedding, Linear};
use crate::model::gemma2::{Gemma2MLP, GemmaRmsNorm};
use crate::model::llama::RotaryCache;
#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
use crate::tensor::GpuTensor;
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
        hidden_states: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        rotary: &RotaryCache,
        device: &mut GpuDevice,
    ) -> GpuTensor {
        let num_tokens = hidden_states.dim(0);

        // QKV projection.
        let qkv = self
            .qkv_proj
            .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // Split QKV (no RoPE yet — we need to apply QK norms first).
        let (q, k, v) = kernels::split_qkv(
            qkv,
            self.q_size,
            self.kv_size,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            &mut device.caching,
            device.compute_stream,
        );

        // Per-head QK norms: reshape [T, heads, head_dim] → [T*heads, head_dim],
        // apply RMS norm, reshape back.
        let q_flat = q
            .as_gpu_tensor()
            .reshape(&[num_tokens * self.num_q_heads, self.head_dim]);
        let q_normed = kernels::rms_norm(
            q_flat,
            self.q_norm.inner.weight,
            self.q_norm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );
        let q_3d =
            q_normed
                .into_gpu_tensor()
                .reshape(&[num_tokens, self.num_q_heads, self.head_dim]);

        let k_flat = k
            .as_gpu_tensor()
            .reshape(&[num_tokens * self.num_kv_heads, self.head_dim]);
        let k_normed = kernels::rms_norm(
            k_flat,
            self.k_norm.inner.weight,
            self.k_norm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );
        let k_3d =
            k_normed
                .into_gpu_tensor()
                .reshape(&[num_tokens, self.num_kv_heads, self.head_dim]);

        // RoPE (in-place on the normed q/k).
        // rotary_embedding_inplace expects [T, total_dim] flat layout.
        let q_flat_rope = q_3d.reshape(&[num_tokens, self.q_size]);
        let k_flat_rope = k_3d.reshape(&[num_tokens, self.kv_size]);
        kernels::rotary_embedding_inplace(
            q_flat_rope,
            k_flat_rope,
            positions,
            rotary.cos_sin_cache,
            self.head_dim,
            device.compute_stream,
        );

        // Reshape back to 3D for attention.
        let q_3d = q_flat_rope.reshape(&[num_tokens, self.num_q_heads, self.head_dim]);
        let k_3d = k_flat_rope.reshape(&[num_tokens, self.num_kv_heads, self.head_dim]);

        kernels::reshape_and_cache(
            k_3d,
            *v,
            kv_cache.k_cache(self.layer_idx),
            kv_cache.v_cache(self.layer_idx),
            slot_mapping,
            kv_cache.block_size,
            device.compute_stream,
        );

        let window_left = self.sliding_window.map(|w| w as i32).unwrap_or(-1);

        let fresh_prefill = max_seqlen_q > 1 && max_seqlen_q == max_seqlen_k;
        let attn_output = if fresh_prefill {
            kernels::flash_attn_contiguous(
                q_3d,
                k_3d,
                *v,
                cu_seqlens_q,
                cu_seqlens_q,
                max_seqlen_q,
                max_seqlen_k,
                self.scale,
                true,
                0.0, // no softcap
                window_left,
                &mut device.caching,
                device.compute_stream,
            )
        } else {
            kernels::flash_attn_paged_ext(
                q_3d,
                kv_cache.k_cache(self.layer_idx),
                kv_cache.v_cache(self.layer_idx),
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_seqlen_q,
                max_seqlen_k,
                self.scale,
                true,
                0.0, // no softcap
                window_left,
                kv_cache.block_size,
                device.num_sm,
                &mut device.caching,
                device.compute_stream,
            )
        };

        let attn_flat = attn_output
            .into_gpu_tensor()
            .reshape(&[num_tokens, self.q_size]);
        let out = self
            .o_proj
            .forward(attn_flat, &mut device.cublas, &mut device.caching);

        // TP: all-reduce o_proj output (row parallel).
        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.tp_group {
            group
                .all_reduce_inplace(out)
                .expect("o_proj all_reduce failed");
        }

        out
    }

    /// Forward pass returning `OwnedTensor` (caching-allocator path).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_owned(
        &self,
        hidden_states: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        rotary: &RotaryCache,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);

        // QKV projection → owned.
        let qkv =
            self.qkv_proj
                .forward_owned(hidden_states, &mut device.cublas, &mut device.caching);

        // Split QKV (no RoPE yet — we need to apply QK norms first).
        let (q, k, v) = kernels::split_qkv(
            qkv.as_gpu_tensor(),
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
            .as_gpu_tensor()
            .reshape(&[num_tokens * self.num_q_heads, self.head_dim]);
        let q_normed = kernels::rms_norm(
            q_flat,
            self.q_norm.inner.weight,
            self.q_norm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );
        drop(q);
        let q_3d = q_normed
            .as_gpu_tensor()
            .reshape(&[num_tokens, self.num_q_heads, self.head_dim]);

        let k_flat = k
            .as_gpu_tensor()
            .reshape(&[num_tokens * self.num_kv_heads, self.head_dim]);
        let k_normed = kernels::rms_norm(
            k_flat,
            self.k_norm.inner.weight,
            self.k_norm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );
        drop(k);
        let k_3d =
            k_normed
                .as_gpu_tensor()
                .reshape(&[num_tokens, self.num_kv_heads, self.head_dim]);

        // RoPE (in-place on the normed q/k).
        let q_flat_rope = q_3d.reshape(&[num_tokens, self.q_size]);
        let k_flat_rope = k_3d.reshape(&[num_tokens, self.kv_size]);
        kernels::rotary_embedding_inplace(
            q_flat_rope,
            k_flat_rope,
            positions,
            rotary.cos_sin_cache,
            self.head_dim,
            device.compute_stream,
        );

        // Reshape back to 3D for attention.
        let q_3d = q_flat_rope.reshape(&[num_tokens, self.num_q_heads, self.head_dim]);
        let k_3d = k_flat_rope.reshape(&[num_tokens, self.num_kv_heads, self.head_dim]);

        kernels::reshape_and_cache(
            k_3d,
            *v,
            kv_cache.k_cache(self.layer_idx),
            kv_cache.v_cache(self.layer_idx),
            slot_mapping,
            kv_cache.block_size,
            device.compute_stream,
        );

        let window_left = self.sliding_window.map(|w| w as i32).unwrap_or(-1);

        let fresh_prefill = max_seqlen_q > 1 && max_seqlen_q == max_seqlen_k;
        let attn_output = if fresh_prefill {
            kernels::flash_attn_contiguous(
                q_3d,
                k_3d,
                *v,
                cu_seqlens_q,
                cu_seqlens_q,
                max_seqlen_q,
                max_seqlen_k,
                self.scale,
                true,
                0.0,
                window_left,
                &mut device.caching,
                device.compute_stream,
            )
        } else {
            drop(k_normed);
            drop(v);
            kernels::flash_attn_paged_ext(
                q_3d,
                kv_cache.k_cache(self.layer_idx),
                kv_cache.v_cache(self.layer_idx),
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_seqlen_q,
                max_seqlen_k,
                self.scale,
                true,
                0.0,
                window_left,
                kv_cache.block_size,
                device.num_sm,
                &mut device.caching,
                device.compute_stream,
            )
        };
        drop(q_normed);

        let attn_flat = attn_output
            .as_gpu_tensor()
            .reshape(&[num_tokens, self.q_size]);
        let result = self
            .o_proj
            .forward_owned(attn_flat, &mut device.cublas, &mut device.caching);
        drop(attn_output);

        // TP: all-reduce o_proj output (row parallel).
        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.tp_group {
            group
                .all_reduce_inplace(result.as_gpu_tensor())
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
        hidden_states: GpuTensor,
        residual: Option<GpuTensor>,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
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
                &mut device.caching,
                device.compute_stream,
            )
        } else {
            let normed = kernels::rms_norm(
                hidden_states,
                self.input_layernorm.inner.weight,
                self.input_layernorm.inner.eps,
                &mut device.caching,
                device.compute_stream,
            );
            (normed.into_gpu_tensor(), hidden_states)
        };

        // 2. Attention.
        let attn_output = self.self_attn.forward(
            normed,
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

        // 3. Post-attention norm (standalone, no residual add).
        let attn_normed = kernels::rms_norm(
            attn_output,
            self.post_attention_layernorm.inner.weight,
            self.post_attention_layernorm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );

        // 4. Pre-feedforward norm with fused residual add.
        let (normed, residual) = kernels::fused_add_rms_norm(
            *attn_normed,
            residual,
            self.pre_feedforward_layernorm.inner.weight,
            self.pre_feedforward_layernorm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );

        // 5. MLP.
        let mlp_output = self.mlp.forward(normed, device);

        // 6. Post-feedforward norm (standalone, no residual add).
        let mlp_normed = kernels::rms_norm(
            mlp_output,
            self.post_feedforward_layernorm.inner.weight,
            self.post_feedforward_layernorm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );

        (mlp_normed.into_gpu_tensor(), residual)
    }

    /// Forward using caching allocator with proper Rust ownership.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_owned(
        &self,
        hidden_states: OwnedTensor,
        residual: Option<OwnedTensor>,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
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
        let attn_output = self.self_attn.forward_owned(
            *normed,
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
        let mlp_output = self.mlp.forward_owned(*attn_normed, device);
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
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
    ) -> GpuTensor {
        // Embedding lookup + Gemma scaling.
        let hidden_states = kernels::embedding_gather(
            self.embed_tokens.weight,
            input_ids,
            &mut device.caching,
            device.compute_stream,
        );
        let hidden_states = hidden_states.into_gpu_tensor();
        kernels::scale_inplace(hidden_states, self.embed_scale, &device.cublas);

        // Per-layer arena scoping.
        let num_tokens = hidden_states.dim(0);
        let hidden_size = hidden_states.dim(1);
        let dtype = hidden_states.dtype();
        let hs_buf = device
            .caching
            .alloc_gpu_tensor(&[num_tokens, hidden_size], dtype);
        let res_buf = device
            .caching
            .alloc_gpu_tensor(&[num_tokens, hidden_size], dtype);
        crate::driver::memcpy_dtod_async(
            hs_buf.raw_ptr() as *mut u8,
            hidden_states.raw_ptr() as *const u8,
            hidden_states.size_bytes(),
            device.compute_stream,
        )
        .expect("dtod copy initial hidden_states");

        let mut residual: Option<GpuTensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let is_sliding = i < self.layer_is_sliding.len() && self.layer_is_sliding[i];
            let rotary = if is_sliding {
                &self.rotary_local
            } else {
                &self.rotary_global
            };
            let (hs, res) = layer.forward(
                hs_buf,
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
            if res.raw_ptr() != res_buf.raw_ptr() {
                crate::driver::memcpy_dtod_async(
                    res_buf.raw_ptr() as *mut u8,
                    res.raw_ptr() as *const u8,
                    res.size_bytes(),
                    device.compute_stream,
                )
                .expect("dtod copy residual");
            }
            crate::driver::memcpy_dtod_async(
                hs_buf.raw_ptr() as *mut u8,
                hs.raw_ptr() as *const u8,
                hs.size_bytes(),
                device.compute_stream,
            )
            .expect("dtod copy hidden_states");
            residual = Some(res_buf);
        }
        let hidden_states = hs_buf;

        // Final norm with fused residual add.
        let (normed, _) = kernels::fused_add_rms_norm(
            hidden_states,
            residual.unwrap(),
            self.norm.inner.weight,
            self.norm.inner.eps,
            &mut device.caching,
            device.compute_stream,
        );
        normed
    }

    /// Forward using caching allocator — zero D2D copies between layers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_owned(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
    ) -> GpuTensor {
        // Embedding lookup + Gemma scaling.
        let hidden_states = kernels::embedding_gather(
            self.embed_tokens.weight,
            input_ids,
            &mut device.caching,
            device.compute_stream,
        );
        kernels::scale_inplace(
            hidden_states.as_gpu_tensor(),
            self.embed_scale,
            &device.cublas,
        );

        let mut hidden_states: OwnedTensor = hidden_states;
        let mut residual: Option<OwnedTensor> = None;

        for (i, layer) in self.layers.iter().enumerate() {
            let is_sliding = i < self.layer_is_sliding.len() && self.layer_is_sliding[i];
            let rotary = if is_sliding {
                &self.rotary_local
            } else {
                &self.rotary_global
            };
            let (hs, res) = layer.forward_owned(
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
        let res_gpu = residual.as_ref().unwrap().as_gpu_tensor();
        kernels::fused_add_rms_norm_inplace(
            hs_gpu,
            res_gpu,
            self.norm.inner.weight,
            self.norm.inner.eps,
            device.compute_stream,
        );
        drop(residual);
        hidden_states.into_gpu_tensor()
    }
}

// ---------------------------------------------------------------------------
// Gemma3ForCausalLM — no final logit softcapping
// ---------------------------------------------------------------------------

pub struct Gemma3ForCausalLM {
    pub model: Gemma3Model,
    pub lm_head: Linear,
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

        Ok(Self { model, lm_head })
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
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
                hidden_states,
                indices,
                &mut device.caching,
                device.compute_stream,
            )
            .into_gpu_tensor()
        } else {
            hidden_states
        };

        self.lm_head
            .forward(hidden_states, &mut device.cublas, &mut device.caching)
    }

    /// Forward using caching allocator (zero D2D copies between layers).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_owned(
        &self,
        input_ids: GpuTensor,
        positions: GpuTensor,
        slot_mapping: GpuTensor,
        cu_seqlens_q: GpuTensor,
        seqused_k: GpuTensor,
        block_table: GpuTensor,
        max_seqlen_q: usize,
        max_seqlen_k: usize,
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
        last_token_indices: Option<GpuTensor>,
    ) -> GpuTensor {
        let hidden_states = self.model.forward_owned(
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
                hidden_states,
                indices,
                &mut device.caching,
                device.compute_stream,
            )
            .into_gpu_tensor()
        } else {
            hidden_states
        };

        let logits =
            self.lm_head
                .forward_owned(hidden_states, &mut device.cublas, &mut device.caching);
        logits.into_gpu_tensor()
    }
}

// ---------------------------------------------------------------------------
// Tensor-parallel loading
// ---------------------------------------------------------------------------

use crate::model::llama::TpConfig;

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
        Ok(Self { model, lm_head })
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
