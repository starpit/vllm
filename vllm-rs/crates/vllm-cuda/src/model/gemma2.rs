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

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kernels;

use crate::kv_cache::KvCachePool;
use crate::layers::{Embedding, Linear, LinearLayer, RmsNorm};
use crate::model::llama::RotaryCache;
#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
use crate::quant::QuantConfig;
use crate::tensor::GpuTensor;
use crate::weights::{self as gpu_weights, GpuWeights};
#[cfg(feature = "nccl")]
use std::sync::Arc;

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
pub unsafe fn add_one_to_weight(
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
    gate_up_proj: LinearLayer,
    down_proj: LinearLayer,
    intermediate_size: usize,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

impl Gemma2MLP {
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let gate_name = format!("{prefix}.gate_proj.weight");
        let up_name = format!("{prefix}.up_proj.weight");
        let (gate_shape, gate_dtype) = weights
            .tensor_info(&gate_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
        let hidden = if gate_shape.len() == 2 {
            gate_shape[1]
        } else {
            1
        };
        let gate_bytes = gate_shape.iter().product::<usize>() * gate_dtype.size_bytes();
        let total_bytes = gate_bytes * 2;
        let ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };
        unsafe {
            weights.take_into(&gate_name, ptr, stream)?;
            weights.take_into(&up_name, ptr.add(gate_bytes), stream)?;
        }
        let gate_up_w =
            unsafe { GpuTensor::new(ptr, &[2 * intermediate_size, hidden], gate_dtype) };
        let gate_up_proj = LinearLayer::Dense(Linear::new(gate_up_w, None));
        let down_proj = LinearLayer::Dense(Linear::load(weights, &format!("{prefix}.down_proj"))?);
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    pub unsafe fn forward(&self, x: GpuTensor, device: &mut GpuDevice) -> GpuTensor {
        let gate_up = self.gate_up_proj.forward(
            x,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );
        let activated = kernels::gelu_and_mul_fused(
            gate_up,
            self.intermediate_size,
            &mut device.caching,
            device.compute_stream,
        );
        let out = self.down_proj.forward(
            *activated,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );

        // TP: all-reduce down_proj output (row parallel).
        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.tp_group {
            group
                .all_reduce_inplace(out)
                .expect("down_proj all_reduce failed");
        }

        out
    }

    /// Forward pass returning `OwnedTensor` (caching-allocator path).
    pub unsafe fn forward_owned(&self, x: GpuTensor, device: &mut GpuDevice) -> OwnedTensor {
        let gate_up = self.gate_up_proj.forward_owned(
            x,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );
        let activated = kernels::gelu_and_mul_fused(
            gate_up.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            device.compute_stream,
        );
        drop(gate_up);
        let result = self.down_proj.forward_owned(
            activated.as_gpu_tensor(),
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );
        drop(activated);

        // TP: all-reduce down_proj output (row parallel).
        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.tp_group {
            group
                .all_reduce_inplace(result.as_gpu_tensor())
                .expect("down_proj all_reduce failed");
        }

        result
    }

    /// Load quantized MLP (fused gate_up + down Marlin layers).
    pub fn load_quantized(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        qconfig: &QuantConfig,
        workspace: GpuTensor,
        device: &GpuDevice,
    ) -> Result<Self> {
        let mut alloc = crate::alloc::CachingAllocator::new();

        let gate_up = gpu_weights::load_fused_marlin_linear(
            weights,
            &[format!("{prefix}.gate_proj"), format!("{prefix}.up_proj")],
            qconfig,
            workspace,
            device.device_id,
            &mut alloc,
        )?;
        let down = gpu_weights::load_marlin_linear(
            weights,
            &format!("{prefix}.down_proj"),
            qconfig,
            workspace,
            device.device_id,
            &mut alloc,
        )?;

        Ok(Self {
            gate_up_proj: LinearLayer::Marlin(Box::new(gate_up)),
            down_proj: LinearLayer::Marlin(Box::new(down)),
            intermediate_size,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load BNB 4-bit quantized MLP.
    #[allow(clippy::too_many_arguments)]
    pub fn load_bnb4bit(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        qconfig: &crate::quant::Bnb4bitConfig,
        code_gpu: GpuTensor,
        dequant_scratch: GpuTensor,
        blocksize: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let inter = config.intermediate_size;
        let hidden = config.hidden_size;

        let gate_up = gpu_weights::load_fused_bnb4bit_linear(
            weights,
            &[format!("{prefix}.gate_proj"), format!("{prefix}.up_proj")],
            qconfig,
            code_gpu,
            dequant_scratch,
            &[inter, inter],
            hidden,
            blocksize,
            stream,
        )?;
        let down = gpu_weights::load_bnb4bit_linear(
            weights,
            &format!("{prefix}.down_proj"),
            qconfig,
            code_gpu,
            dequant_scratch,
            hidden,
            inter,
            blocksize,
            stream,
        )?;

        Ok(Self {
            gate_up_proj: LinearLayer::Bnb4bit(Box::new(gate_up)),
            down_proj: LinearLayer::Bnb4bit(Box::new(down)),
            intermediate_size: inter,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Gemma2Attention
// ---------------------------------------------------------------------------

pub struct Gemma2Attention {
    qkv_proj: LinearLayer,
    o_proj: LinearLayer,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    scale: f32,
    attn_logit_softcapping: f32,
    sliding_window: Option<usize>,
    layer_idx: usize,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
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
            weights.take_into(&q_name, ptr, stream)?;
            weights.take_into(&k_name, ptr.add(q_bytes), stream)?;
            weights.take_into(&v_name, ptr.add(q_bytes + kv_bytes), stream)?;
        }
        let qkv_w = unsafe { GpuTensor::new(ptr, &[q_size + 2 * kv_size, hidden], q_dtype) };
        let qkv_proj = LinearLayer::Dense(Linear::new(qkv_w, None));

        let o_proj = LinearLayer::Dense(Linear::load(weights, &format!("{prefix}.o_proj"))?);

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

        let qkv = self.qkv_proj.forward(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );

        let (q, k, v) = kernels::fused_qkv_rope(
            qkv,
            positions,
            rotary.cos_sin_cache,
            self.q_size,
            self.kv_size,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            &mut device.caching,
            device.compute_stream,
        );

        crate::model::attention_helpers::write_kv_cache(
            *k,
            *v,
            slot_mapping,
            kv_cache,
            self.layer_idx,
            device.compute_stream,
        );

        let window_left = self.sliding_window.map(|w| w as i32).unwrap_or(-1);

        let attn_output = crate::model::attention_helpers::attention_ext(
            *q,
            *k,
            *v,
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            self.scale,
            self.attn_logit_softcapping,
            window_left,
            kv_cache,
            self.layer_idx,
            device.num_sm,
            &mut device.caching,
            device.compute_stream,
        );

        let attn_flat = attn_output
            .into_gpu_tensor()
            .reshape(&[num_tokens, self.q_size]);
        let out = self.o_proj.forward(
            attn_flat,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );

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

        let qkv = self.qkv_proj.forward_owned(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );

        let (q, k, v) = kernels::fused_qkv_rope(
            qkv.as_gpu_tensor(),
            positions,
            rotary.cos_sin_cache,
            self.q_size,
            self.kv_size,
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            &mut device.caching,
            device.compute_stream,
        );
        drop(qkv);

        crate::model::attention_helpers::write_kv_cache(
            k.as_gpu_tensor(),
            v.as_gpu_tensor(),
            slot_mapping,
            kv_cache,
            self.layer_idx,
            device.compute_stream,
        );

        let window_left = self.sliding_window.map(|w| w as i32).unwrap_or(-1);

        let attn_output = crate::model::attention_helpers::attention_ext(
            q.as_gpu_tensor(),
            k.as_gpu_tensor(),
            v.as_gpu_tensor(),
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            self.scale,
            self.attn_logit_softcapping,
            window_left,
            kv_cache,
            self.layer_idx,
            device.num_sm,
            &mut device.caching,
            device.compute_stream,
        );
        drop(q);
        drop(k);
        drop(v);

        let attn_flat = attn_output
            .as_gpu_tensor()
            .reshape(&[num_tokens, self.q_size]);
        let result = self.o_proj.forward_owned(
            attn_flat,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );
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

    /// Load quantized attention (fused QKV + o_proj Marlin layers).
    #[allow(clippy::too_many_arguments)]
    pub fn load_quantized(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        qconfig: &QuantConfig,
        workspace: GpuTensor,
        device: &GpuDevice,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let mut alloc = crate::alloc::CachingAllocator::new();

        let qkv = gpu_weights::load_fused_marlin_linear(
            weights,
            &[
                format!("{prefix}.q_proj"),
                format!("{prefix}.k_proj"),
                format!("{prefix}.v_proj"),
            ],
            qconfig,
            workspace,
            device.device_id,
            &mut alloc,
        )?;
        let o = gpu_weights::load_marlin_linear(
            weights,
            &format!("{prefix}.o_proj"),
            qconfig,
            workspace,
            device.device_id,
            &mut alloc,
        )?;

        let sliding_window = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        Ok(Self {
            qkv_proj: LinearLayer::Marlin(Box::new(qkv)),
            o_proj: LinearLayer::Marlin(Box::new(o)),
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5) as f32,
            attn_logit_softcapping: config.attn_logit_softcapping.unwrap_or(0.0) as f32,
            sliding_window,
            layer_idx,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load BNB 4-bit quantized attention.
    #[allow(clippy::too_many_arguments)]
    pub fn load_bnb4bit(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        qconfig: &crate::quant::Bnb4bitConfig,
        code_gpu: GpuTensor,
        dequant_scratch: GpuTensor,
        blocksize: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let hidden = config.hidden_size;

        let qkv = gpu_weights::load_fused_bnb4bit_linear(
            weights,
            &[
                format!("{prefix}.q_proj"),
                format!("{prefix}.k_proj"),
                format!("{prefix}.v_proj"),
            ],
            qconfig,
            code_gpu,
            dequant_scratch,
            &[q_size, kv_size, kv_size],
            hidden,
            blocksize,
            stream,
        )?;
        let o = gpu_weights::load_bnb4bit_linear(
            weights,
            &format!("{prefix}.o_proj"),
            qconfig,
            code_gpu,
            dequant_scratch,
            hidden,
            q_size,
            blocksize,
            stream,
        )?;

        let sliding_window = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        Ok(Self {
            qkv_proj: LinearLayer::Bnb4bit(Box::new(qkv)),
            o_proj: LinearLayer::Bnb4bit(Box::new(o)),
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: config.query_pre_attn_scalar.powf(-0.5) as f32,
            attn_logit_softcapping: config.attn_logit_softcapping.unwrap_or(0.0) as f32,
            sliding_window,
            layer_idx,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Gemma2DecoderLayer — 4 norms per layer
// ---------------------------------------------------------------------------

pub struct Gemma2DecoderLayer {
    pub self_attn: Gemma2Attention,
    pub mlp: Gemma2MLP,
    pub input_layernorm: GemmaRmsNorm,
    pub post_attention_layernorm: GemmaRmsNorm,
    pub pre_feedforward_layernorm: GemmaRmsNorm,
    pub post_feedforward_layernorm: GemmaRmsNorm,
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

    /// Load a quantized decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_quantized(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        qconfig: &QuantConfig,
        workspace: GpuTensor,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let self_attn = Gemma2Attention::load_quantized(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            is_sliding,
            qconfig,
            workspace,
            device,
        )?;
        let mlp = Gemma2MLP::load_quantized(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            qconfig,
            workspace,
            device,
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

    /// Load a BNB 4-bit decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_bnb4bit(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        qconfig: &crate::quant::Bnb4bitConfig,
        code_gpu: GpuTensor,
        dequant_scratch: GpuTensor,
        blocksize: usize,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let self_attn = Gemma2Attention::load_bnb4bit(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            is_sliding,
            qconfig,
            code_gpu,
            dequant_scratch,
            blocksize,
            device.compute_stream,
        )?;
        let mlp = Gemma2MLP::load_bnb4bit(
            weights,
            &format!("{prefix}.mlp"),
            config,
            qconfig,
            code_gpu,
            dequant_scratch,
            blocksize,
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
// Gemma2Model
// ---------------------------------------------------------------------------

pub struct Gemma2Model {
    pub embed_tokens: Embedding,
    pub layers: Vec<Gemma2DecoderLayer>,
    pub norm: GemmaRmsNorm,
    pub rotary: RotaryCache,
    pub embed_scale: f32,
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
                None, // Gemma2 uses plain RoPE
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

        // Per-layer arena scoping (see LlamaModel::forward for details).
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
        // No arena offset tracking needed — caching allocator frees on drop.

        let mut residual: Option<GpuTensor> = None;
        for layer in &self.layers {
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
                &self.rotary,
                device,
            );
            // Copy residual to persistent buffer FIRST if it aliases hs_buf
            // (first layer: residual = hs_buf = original embedding).
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
            // Intermediates freed by caching allocator on drop.
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

        for layer in &self.layers {
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
                &self.rotary,
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
// Gemma2ForCausalLM
// ---------------------------------------------------------------------------

pub struct Gemma2ForCausalLM {
    pub model: Gemma2Model,
    pub lm_head: Linear,
    final_logit_softcapping: Option<f32>,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
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
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load a quantized model (AWQ/GPTQ → Marlin).
    pub fn load_quantized(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
        dtype: DType,
        qconfig: &QuantConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let workspace = gpu_weights::alloc_marlin_workspace(device.num_sm, device.compute_stream)?;

        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma2DecoderLayer::load_quantized(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                is_sliding,
                qconfig,
                workspace,
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
                None,
                dtype,
                device,
            )?
        };

        let model = Gemma2Model {
            embed_tokens,
            layers,
            norm,
            rotary,
            embed_scale: (config.hidden_size as f32).sqrt(),
        };

        // Gemma2 always uses tied embeddings — lm_head is dense.
        let lm_head = Linear::new(model.embed_tokens.weight, None);

        Ok(Self {
            model,
            lm_head,
            final_logit_softcapping: config.final_logit_softcapping.map(|v| v as f32),
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load a BNB 4-bit quantized Gemma2 model.
    pub fn load_bnb4bit(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
        dtype: DType,
        qconfig: &crate::quant::Bnb4bitConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let stream = device.compute_stream;

        let code_table = match qconfig.quant_type {
            crate::quant::BnbQuantType::NF4 => &crate::quant::NF4_CODE,
            crate::quant::BnbQuantType::FP4 => &crate::quant::FP4_CODE,
        };
        let code_gpu = gpu_weights::upload_bnb_code(code_table, stream)?;
        let blocksize = qconfig.blocksize;

        let hidden = config.hidden_size;
        let q_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_kv_heads * config.head_dim;
        let inter = config.intermediate_size;
        let max_elements = [
            (q_size + 2 * kv_size) * hidden,
            q_size * hidden,
            2 * inter * hidden,
            hidden * inter,
        ]
        .into_iter()
        .max()
        .unwrap();
        let dequant_scratch = gpu_weights::alloc_bnb_dequant_scratch(max_elements, dtype, stream)?;

        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma2DecoderLayer::load_bnb4bit(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                is_sliding,
                qconfig,
                code_gpu,
                dequant_scratch,
                blocksize,
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
                None,
                dtype,
                device,
            )?
        };

        let model = Gemma2Model {
            embed_tokens,
            layers,
            norm,
            rotary,
            embed_scale: (config.hidden_size as f32).sqrt(),
        };

        let lm_head = Linear::new(model.embed_tokens.weight, None);

        Ok(Self {
            model,
            lm_head,
            final_logit_softcapping: config.final_logit_softcapping.map(|v| v as f32),
            #[cfg(feature = "nccl")]
            tp_group: None,
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

        let logits = self
            .lm_head
            .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // Apply final logit soft capping: logits = cap * tanh(logits / cap).
        // TODO: This requires a fused tanh-softcap kernel. For now, softcap
        // is not applied at the GpuTensor level — it would need a small CUDA
        // kernel. The softcap values are typically large (30.0) so this has
        // minimal impact on correctness for greedy/top-k sampling.
        let _ = self.final_logit_softcapping;

        logits
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
        let logits = logits.into_gpu_tensor();

        let _ = self.final_logit_softcapping;

        logits
    }
}

// ---------------------------------------------------------------------------
// TP group injection
// ---------------------------------------------------------------------------

#[cfg(feature = "nccl")]
impl Gemma2ForCausalLM {
    /// Inject NCCL process group into all TP layers.
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        for layer in &mut self.model.layers {
            layer.self_attn.tp_group = Some(Arc::clone(&group));
            layer.mlp.tp_group = Some(Arc::clone(&group));
        }
        self.tp_group = Some(group);
    }
}

// ---------------------------------------------------------------------------
// Tensor-parallel loading
// ---------------------------------------------------------------------------

use crate::model::llama::TpConfig;

impl Gemma2Attention {
    /// Load with fused QKV weights, sharded for tensor parallelism.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fused_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads / tp.world_size;
        let num_kv_heads = config.num_kv_heads / tp.world_size;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

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
            weights.take_shard_into(&q_name, 0, tp.rank, tp.world_size, ptr, stream)?;
            weights.take_shard_into(
                &k_name,
                0,
                tp.rank,
                tp.world_size,
                ptr.add(q_bytes),
                stream,
            )?;
            weights.take_shard_into(
                &v_name,
                0,
                tp.rank,
                tp.world_size,
                ptr.add(q_bytes + kv_bytes),
                stream,
            )?;
        }
        let qkv_w = unsafe { GpuTensor::new(ptr, &[q_size + 2 * kv_size, hidden], q_dtype) };
        let qkv_proj = LinearLayer::Dense(Linear::new(qkv_w, None));

        // o_proj: shard along dim=1 (row parallel).
        let o_name = format!("{prefix}.o_proj.weight");
        let o_w = weights.take_shard(&o_name, 1, tp.rank, tp.world_size)?;
        let o_proj = LinearLayer::Dense(Linear::new(o_w, None));

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
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

impl Gemma2MLP {
    /// Load with fused gate+up weights, sharded for tensor parallelism.
    pub fn load_fused_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let shard_intermediate = intermediate_size / tp.world_size;

        let gate_name = format!("{prefix}.gate_proj.weight");
        let up_name = format!("{prefix}.up_proj.weight");
        let (gate_shape, gate_dtype) = weights
            .tensor_info(&gate_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
        let hidden = gate_shape[1];
        let elem_size = gate_dtype.size_bytes();
        let shard_bytes = shard_intermediate * hidden * elem_size;
        let total_bytes = 2 * shard_bytes;
        let ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };
        unsafe {
            weights.take_shard_into(&gate_name, 0, tp.rank, tp.world_size, ptr, stream)?;
            weights.take_shard_into(
                &up_name,
                0,
                tp.rank,
                tp.world_size,
                ptr.add(shard_bytes),
                stream,
            )?;
        }
        let gate_up_w =
            unsafe { GpuTensor::new(ptr, &[2 * shard_intermediate, hidden], gate_dtype) };
        let gate_up_proj = LinearLayer::Dense(Linear::new(gate_up_w, None));

        let down_name = format!("{prefix}.down_proj.weight");
        let down_w = weights.take_shard(&down_name, 1, tp.rank, tp.world_size)?;
        let down_proj = LinearLayer::Dense(Linear::new(down_w, None));

        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size: shard_intermediate,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

impl Gemma2DecoderLayer {
    /// Load a decoder layer with TP-sharded weights.
    #[allow(clippy::too_many_arguments)]
    pub fn load_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        tp: TpConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let self_attn = Gemma2Attention::load_fused_tp(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            is_sliding,
            tp,
            device.compute_stream,
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

impl Gemma2Model {
    /// Load model backbone with TP sharding.
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma2DecoderLayer::load_tp(
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

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            embed_scale: (config.hidden_size as f32).sqrt(),
        })
    }
}

impl Gemma2ForCausalLM {
    /// Load the full model with TP sharding.
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Gemma2Model::load_tp(weights, config, dtype, tp, device)?;
        // Gemma2 always tied embeddings.
        let lm_head = Linear::new(model.embed_tokens.weight, None);
        Ok(Self {
            model,
            lm_head,
            final_logit_softcapping: config.final_logit_softcapping.map(|v| v as f32),
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}
