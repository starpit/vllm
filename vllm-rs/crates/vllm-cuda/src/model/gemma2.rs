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
use crate::model::llama::{ForwardOutput, RotaryCache};
#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
use crate::pp::PpConfig;
use crate::quant::QuantConfig;
use crate::tensor::{GpuTensor, TensorView};
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
        weights.record_alloc(ptr, total_bytes);
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

    pub unsafe fn forward(&self, x: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let gate_up = self.gate_up_proj.forward(
            x,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );
        let activated = kernels::gelu_and_mul_fused(
            *gate_up.view(),
            self.intermediate_size,
            &mut device.caching,
            device.compute_stream,
        );
        drop(gate_up);
        let result = self.down_proj.forward(
            activated.view(),
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

    /// Load FP8 quantized MLP.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        output_dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let gate_up = gpu_weights::load_fused_fp8_linear(
            weights,
            &[format!("{prefix}.gate_proj"), format!("{prefix}.up_proj")],
            output_dtype,
            stream,
        )?;
        let down =
            gpu_weights::load_fp8_linear(weights, &format!("{prefix}.down_proj"), output_dtype)?;

        Ok(Self {
            gate_up_proj: LinearLayer::Fp8(Box::new(gate_up)),
            down_proj: LinearLayer::Fp8(Box::new(down)),
            intermediate_size,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load per-tensor FP8 MLP with TP (column-parallel gate+up, row-parallel down).
    pub fn load_fp8_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        output_dtype: DType,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let ipp = intermediate_size / tp.world_size;
        let gate_up = gpu_weights::load_fused_fp8_linear_tp(
            weights,
            &[format!("{prefix}.gate_proj"), format!("{prefix}.up_proj")],
            output_dtype,
            tp.rank,
            tp.world_size,
            stream,
        )?;
        let down = gpu_weights::load_fp8_linear_tp(
            weights,
            &format!("{prefix}.down_proj"),
            output_dtype,
            tp.rank,
            tp.world_size,
        )?;

        Ok(Self {
            gate_up_proj: LinearLayer::Fp8(Box::new(gate_up)),
            down_proj: LinearLayer::Fp8(Box::new(down)),
            intermediate_size: ipp,
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
        weights.record_alloc(ptr, total_bytes);
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

        let qkv = self.qkv_proj.forward(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );

        if max_seqlen_q == 1 {
            // Decode path: fused QKV split + RoPE + cache write.
            let q = if kv_cache.is_fp8() {
                kernels::fused_qkv_rope_cache_fp8(
                    *qkv.view(),
                    *positions,
                    rotary.cos_sin_cache,
                    *slot_mapping,
                    *kv_cache.k_cache(self.layer_idx),
                    *kv_cache.v_cache(self.layer_idx),
                    kv_cache.k_scale_ptr(self.layer_idx),
                    kv_cache.v_scale_ptr(self.layer_idx),
                    self.q_size,
                    self.kv_size,
                    self.num_q_heads,
                    self.head_dim,
                    &mut device.caching,
                    device.compute_stream,
                )
            } else {
                kernels::fused_qkv_rope_cache(
                    *qkv.view(),
                    *positions,
                    rotary.cos_sin_cache,
                    *slot_mapping,
                    *kv_cache.k_cache(self.layer_idx),
                    *kv_cache.v_cache(self.layer_idx),
                    self.q_size,
                    self.kv_size,
                    self.num_q_heads,
                    self.head_dim,
                    &mut device.caching,
                    device.compute_stream,
                )
            };
            drop(qkv);

            let window_left = self.sliding_window.map(|w| w as i32).unwrap_or(-1);

            let attn_output = crate::model::attention_helpers::attention_decode_from_cache(
                q.view(),
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
                rotary.cos_sin_cache.raw_ptr() as *const u8,
                rotary.cos_sin_cache.dim(1),
            );
            drop(q);

            let attn_flat = attn_output.view().reshape(&[num_tokens, self.q_size]);
            let result = self.o_proj.forward(
                attn_flat,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            drop(attn_output);

            #[cfg(feature = "nccl")]
            if let Some(ref group) = self.tp_group {
                group
                    .all_reduce_inplace(result.as_gpu_tensor())
                    .expect("o_proj all_reduce failed");
            }

            return result;
        }

        let (q, k, v) = kernels::fused_qkv_rope(
            *qkv.view(),
            *positions,
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
            k.view(),
            v.view(),
            slot_mapping,
            kv_cache,
            self.layer_idx,
            device.compute_stream,
        );

        let window_left = self.sliding_window.map(|w| w as i32).unwrap_or(-1);

        let attn_output = crate::model::attention_helpers::attention_ext(
            q.view(),
            k.view(),
            v.view(),
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
            rotary.cos_sin_cache.raw_ptr() as *const u8,
            rotary.cos_sin_cache.dim(1),
        );
        drop(q);
        drop(k);
        drop(v);

        let attn_flat = attn_output.view().reshape(&[num_tokens, self.q_size]);
        let result = self.o_proj.forward(
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

    /// Load FP8 quantized attention.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        output_dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let qkv = gpu_weights::load_fused_fp8_linear(
            weights,
            &[
                format!("{prefix}.q_proj"),
                format!("{prefix}.k_proj"),
                format!("{prefix}.v_proj"),
            ],
            output_dtype,
            stream,
        )?;
        let o = gpu_weights::load_fp8_linear(weights, &format!("{prefix}.o_proj"), output_dtype)?;

        let sliding_window = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        Ok(Self {
            qkv_proj: LinearLayer::Fp8(Box::new(qkv)),
            o_proj: LinearLayer::Fp8(Box::new(o)),
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

    /// Load per-tensor FP8 attention with TP (column-parallel QKV, row-parallel O).
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        output_dtype: DType,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads / tp.world_size;
        let num_kv_heads = config.num_kv_heads / tp.world_size;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let qkv = gpu_weights::load_fused_fp8_linear_tp(
            weights,
            &[
                format!("{prefix}.q_proj"),
                format!("{prefix}.k_proj"),
                format!("{prefix}.v_proj"),
            ],
            output_dtype,
            tp.rank,
            tp.world_size,
            stream,
        )?;
        let o = gpu_weights::load_fp8_linear_tp(
            weights,
            &format!("{prefix}.o_proj"),
            output_dtype,
            tp.rank,
            tp.world_size,
        )?;

        let sliding_window = if is_sliding {
            config.sliding_window
        } else {
            None
        };

        Ok(Self {
            qkv_proj: LinearLayer::Fp8(Box::new(qkv)),
            o_proj: LinearLayer::Fp8(Box::new(o)),
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

    /// Load an FP8 quantized decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        output_dtype: DType,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let self_attn = Gemma2Attention::load_fp8(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            is_sliding,
            output_dtype,
            device.compute_stream,
        )?;
        let mlp = Gemma2MLP::load_fp8(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            output_dtype,
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

    /// Load a per-tensor FP8 decoder layer with TP.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Gemma2Config,
        layer_idx: usize,
        is_sliding: bool,
        output_dtype: DType,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let self_attn = Gemma2Attention::load_fp8_tp(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            is_sliding,
            output_dtype,
            tp,
            device.compute_stream,
        )?;
        let mlp = Gemma2MLP::load_fp8_tp(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            output_dtype,
            tp,
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
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

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

        for layer in &self.layers {
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
        hidden_states
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
    /// Pipeline parallelism config. None = single GPU / PP=1.
    pub pp_config: Option<PpConfig>,
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
            pp_config: None,
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
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

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
            pp_config: None,
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
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

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
            pp_config: None,
        })
    }

    /// Load an FP8 quantized Gemma2 model.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma2DecoderLayer::load_fp8(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                is_sliding,
                dtype,
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
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

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
            pp_config: None,
        })
    }

    /// Load a per-tensor FP8 Gemma2 model with TP sharding.
    pub fn load_fp8_tp(
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
            let layer = Gemma2DecoderLayer::load_fp8_tp(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                is_sliding,
                dtype,
                dtype,
                tp,
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
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

        let model = Gemma2Model {
            embed_tokens,
            layers,
            norm,
            rotary,
            embed_scale: (config.hidden_size as f32).sqrt(),
        };

        // Gemma2 always uses tied embeddings.
        let lm_head = Linear::new(model.embed_tokens.weight, None);

        Ok(Self {
            model,
            lm_head,
            final_logit_softcapping: config.final_logit_softcapping.map(|v| v as f32),
            #[cfg(feature = "nccl")]
            tp_group: None,
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

        let logits = self.lm_head.forward(
            hidden_states.view(),
            &mut device.cublas,
            &mut device.caching,
        );

        // Apply final logit soft capping: logits = cap * tanh(logits / cap).
        if let Some(cap) = self.final_logit_softcapping {
            kernels::tanh_softcap_inplace(logits.as_gpu_tensor(), cap, device.compute_stream);
        }

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
        weights.record_alloc(ptr, total_bytes);
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
        weights.record_alloc(ptr, total_bytes);
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
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

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
            pp_config: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Pipeline-parallel loading and forward pass
// ---------------------------------------------------------------------------

impl Gemma2Model {
    /// Load backbone with PP layer sharding (no TP).
    pub fn load_pp(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
        dtype: DType,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = if pp.is_first_stage() {
            Embedding::load(weights, "model.embed_tokens")?
        } else {
            // Dummy 1-element embedding. Never used — non-first stages receive
            // intermediate hidden states instead of input_ids.
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                weights.record_alloc(ptr, dtype.size_bytes());
                GpuTensor::new(ptr, &[1, 1], dtype)
            };
            Embedding::new(w)
        };

        let mut layers = Vec::with_capacity(pp.num_layers());
        for i in pp.start_layer..pp.end_layer {
            let local_idx = i - pp.start_layer;
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma2DecoderLayer::load(
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
            // Dummy norm — never used on non-last stages.
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                weights.record_alloc(ptr, dtype.size_bytes());
                GpuTensor::new(ptr, &[1], dtype)
            };
            GemmaRmsNorm {
                inner: RmsNorm::new(w, config.rms_norm_eps),
            }
        };

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
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            embed_scale: (config.hidden_size as f32).sqrt(),
        })
    }

    /// Load backbone with TP + PP sharding.
    pub fn load_tp_pp(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
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
                weights.record_alloc(ptr, dtype.size_bytes());
                GpuTensor::new(ptr, &[1, 1], dtype)
            };
            Embedding::new(w)
        };

        let mut layers = Vec::with_capacity(pp.num_layers());
        for i in pp.start_layer..pp.end_layer {
            let local_idx = i - pp.start_layer;
            let is_sliding = i < config.layer_is_sliding.len() && config.layer_is_sliding[i];
            let layer = Gemma2DecoderLayer::load_tp(
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
                weights.record_alloc(ptr, dtype.size_bytes());
                GpuTensor::new(ptr, &[1], dtype)
            };
            GemmaRmsNorm {
                inner: RmsNorm::new(w, config.rms_norm_eps),
            }
        };

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
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            embed_scale: (config.hidden_size as f32).sqrt(),
        })
    }

    /// PP-aware forward:
    /// - First stage: embed input_ids + scale, run layers, return (hs, residual).
    /// - Middle stages: take (hs, residual), run layers, return (hs, residual).
    /// - Last stage: take (hs, residual), run layers + final norm, return logits tensor.
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
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
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

        for layer in &self.layers {
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
                &self.rotary,
                device,
            );
            hidden_states = hs;
            residual = Some(res);
        }

        if pp.is_last_stage() {
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
            ForwardOutput::Logits(hidden_states)
        } else {
            ForwardOutput::Intermediate {
                hidden_states,
                residual: residual.unwrap(),
            }
        }
    }
}

impl Gemma2ForCausalLM {
    /// Load with PP (no TP).
    pub fn load_pp(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
        dtype: DType,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Gemma2Model::load_pp(weights, config, dtype, pp, device)?;

        // Gemma2 always uses tied embeddings — embed_tokens weight IS lm_head.
        // On the last stage, if it is also the first stage, model.embed_tokens was
        // loaded already. If it's NOT the first stage (PP=2+), embed_tokens is a
        // dummy [1,1] tensor — load the weight separately from disk.
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
                weights.record_alloc(ptr, dtype.size_bytes());
                GpuTensor::new(ptr, &[1, 1], dtype)
            };
            Linear::new(w, None)
        };

        Ok(Self {
            model,
            lm_head,
            final_logit_softcapping: config.final_logit_softcapping.map(|v| v as f32),
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: Some(pp),
        })
    }

    /// Load with TP + PP.
    pub fn load_tp_pp(
        weights: &mut GpuWeights,
        config: &Gemma2Config,
        dtype: DType,
        tp: TpConfig,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Gemma2Model::load_tp_pp(weights, config, dtype, tp, pp, device)?;

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
                weights.record_alloc(ptr, dtype.size_bytes());
                GpuTensor::new(ptr, &[1, 1], dtype)
            };
            Linear::new(w, None)
        };

        Ok(Self {
            model,
            lm_head,
            final_logit_softcapping: config.final_logit_softcapping.map(|v| v as f32),
            #[cfg(feature = "nccl")]
            tp_group: None,
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
        kv_cache: &KvCachePool,
        device: &mut GpuDevice,
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
                // Last stage: gather last tokens then lm_head.
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

                #[allow(unused_mut)]
                let mut logits =
                    self.lm_head
                        .forward(hs_view, &mut device.cublas, &mut device.caching);
                // hidden_states can be freed now.
                drop(gathered);
                drop(hidden_states);

                if let Some(cap) = self.final_logit_softcapping {
                    kernels::tanh_softcap_inplace(
                        logits.as_gpu_tensor(),
                        cap,
                        device.compute_stream,
                    );
                }

                #[cfg(feature = "nccl")]
                if let Some(ref group) = self.tp_group {
                    let gathered = group.all_gather(logits.as_gpu_tensor(), &mut device.caching);
                    drop(logits);
                    logits = gathered;
                }

                ForwardOutput::Logits(logits)
            }
        }
    }
}
