// SPDX-License-Identifier: Apache-2.0
//! LLaMA model using `GpuTensor` — zero-allocation forward pass.
//!
//! All intermediate tensors are arena-allocated.
//! The same CUDA kernels from vllm-kernels are called via raw FFI with
//! `GpuTensor::as_ptr()` — one line per pointer extraction instead of ten.
//!
//! Port of `LlamaForCausalLM`.

use anyhow::Result;

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::layers::{Embedding, Linear, LinearLayer, RmsNorm};
use crate::pp::PpConfig;
use crate::quant::QuantConfig;
use crate::tensor::{GpuTensor, TensorView};
use crate::weights::GpuWeights;
use crate::weights::{self as gpu_weights};

#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
#[cfg(feature = "nccl")]
use std::sync::Arc;

// ── Fused norm+GEMM kernels (compile-time PTX fusion) ──
#[cfg(feature = "ferrite")]
const FUSED_NORM_GEMM: ptx_fusion::FeriteKernel = ptx_fusion::compile!(
    a = rms_norm,
    b = gemm_64x128x32,
    bind = { a.output => b.param_0 },
    name = "ferrite_fused_norm_gemm",
);

// ---------------------------------------------------------------------------
// ForwardOutput — PP-aware return type
// ---------------------------------------------------------------------------

/// Output of a model forward pass. For single-GPU or the last PP stage,
/// this is `Logits`. For non-last PP stages, it's `Intermediate` containing
/// the hidden states and residual to pass to the next stage.
pub enum ForwardOutput {
    /// Final logits `[num_reqs, vocab_size]` — only from the last PP stage.
    /// Wrapped in `OwnedTensor` so GPU memory is freed on drop (RAII).
    Logits(OwnedTensor),
    /// Intermediate hidden states + residual to send to next PP stage.
    /// Both are `[num_tokens, hidden_size]` in the model's compute dtype.
    Intermediate {
        hidden_states: OwnedTensor,
        residual: OwnedTensor,
    },
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Parsed LLaMA config (mirrors the HF config).
/// Llama 3.x rope_scaling parameters.
#[derive(Debug, Clone)]
pub struct Llama3RopeScaling {
    pub factor: f64,
    pub low_freq_factor: f64,
    pub high_freq_factor: f64,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
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
    pub tie_word_embeddings: bool,
    pub llama3_rope_scaling: Option<Llama3RopeScaling>,
}

// ---------------------------------------------------------------------------
// RoPE cache
// ---------------------------------------------------------------------------

/// Pre-computed rotary embedding cos/sin cache on GPU.
pub struct RotaryCache {
    /// `[max_pos, rotary_dim]` combined cos|sin cache (used by RoPE kernels).
    pub cos_sin_cache: GpuTensor,
    /// `[max_pos, rotary_dim/2]` separate cos cache (used by FA2 fused RoPE).
    pub cos_cache: GpuTensor,
    /// `[max_pos, rotary_dim/2]` separate sin cache (used by FA2 fused RoPE).
    pub sin_cache: GpuTensor,
    pub head_dim: usize,
}

impl RotaryCache {
    /// Build the cos/sin cache on GPU.
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    pub unsafe fn new(
        head_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let rotary_dim = head_dim; // full rotary for LLaMA
        let half = rotary_dim / 2;

        // Compute inverse frequencies, optionally with llama3 scaling.
        let inv_freqs: Vec<f64> = (0..half)
            .map(|i| {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / rotary_dim as f64);
                if let Some(scaling) = llama3_scaling {
                    let old_context_len = scaling.original_max_position_embeddings as f64;
                    let low_freq_wavelen = old_context_len / scaling.low_freq_factor;
                    let high_freq_wavelen = old_context_len / scaling.high_freq_factor;
                    let wavelen = 2.0 * std::f64::consts::PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq // high frequency: keep as-is
                    } else if wavelen > low_freq_wavelen {
                        freq / scaling.factor // low frequency: scale down
                    } else {
                        // smooth interpolation
                        let smooth = (old_context_len / wavelen - scaling.low_freq_factor)
                            / (scaling.high_freq_factor - scaling.low_freq_factor);
                        (1.0 - smooth) * freq / scaling.factor + smooth * freq
                    }
                } else {
                    freq
                }
            })
            .collect();

        // Build on CPU, then copy to GPU.
        let mut cache = vec![0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let angle = pos as f64 * inv_freqs[i];
                cache[pos * rotary_dim + i] = angle.cos() as f32;
                cache[pos * rotary_dim + half + i] = angle.sin() as f32;
            }
        }

        let nbytes = max_pos * rotary_dim * dtype.size_bytes();
        let gpu_ptr = crate::driver::mem_alloc(nbytes)?;

        // Convert to target dtype and upload.
        match dtype {
            DType::F32 => {
                let host = crate::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(cache.as_ptr() as *const u8, host, nbytes);
                crate::driver::memcpy_htod_async(gpu_ptr, host, nbytes, device.compute_stream)?;
                crate::driver::stream_synchronize(device.compute_stream)?;
                crate::driver::mem_free_host(host)?;
            }
            DType::F16 => {
                let f16_data: Vec<half::f16> =
                    cache.iter().map(|&v| half::f16::from_f32(v)).collect();
                let host = crate::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(f16_data.as_ptr() as *const u8, host, nbytes);
                crate::driver::memcpy_htod_async(gpu_ptr, host, nbytes, device.compute_stream)?;
                crate::driver::stream_synchronize(device.compute_stream)?;
                crate::driver::mem_free_host(host)?;
            }
            DType::BF16 => {
                let bf16_data: Vec<half::bf16> =
                    cache.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let host = crate::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(bf16_data.as_ptr() as *const u8, host, nbytes);
                crate::driver::memcpy_htod_async(gpu_ptr, host, nbytes, device.compute_stream)?;
                crate::driver::stream_synchronize(device.compute_stream)?;
                crate::driver::mem_free_host(host)?;
            }
            _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
        }

        let cos_sin_cache = GpuTensor::new(gpu_ptr, &[max_pos, rotary_dim], dtype);

        // Build separate cos/sin caches for FA2 fused RoPE (needs contiguous buffers).
        let (cos_cache, sin_cache) = Self::build_separate_cos_sin(
            &cache,
            max_pos,
            rotary_dim,
            dtype,
            device.compute_stream,
        )?;

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
        })
    }

    /// Build cos/sin cache with partial rotary dimension (rotary_dim < head_dim).
    ///
    /// # Safety
    /// Requires valid CUDA context and stream.
    pub unsafe fn new_partial(
        head_dim: usize,
        rotary_dim: usize,
        max_pos: usize,
        rope_theta: f64,
        llama3_scaling: Option<&Llama3RopeScaling>,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        // The pos_encoding_kernels.cu already handles head_size > rotary_dim
        // by copying non-rotary elements through. The cos_sin_cache just needs
        // to have shape [max_pos, rotary_dim] where rotary_dim <= head_dim.
        let half = rotary_dim / 2;

        let inv_freqs: Vec<f64> = (0..half)
            .map(|i| {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / rotary_dim as f64);
                if let Some(scaling) = llama3_scaling {
                    let old_context_len = scaling.original_max_position_embeddings as f64;
                    let low_freq_wavelen = old_context_len / scaling.low_freq_factor;
                    let high_freq_wavelen = old_context_len / scaling.high_freq_factor;
                    let wavelen = 2.0 * std::f64::consts::PI / freq;
                    if wavelen < high_freq_wavelen {
                        freq
                    } else if wavelen > low_freq_wavelen {
                        freq / scaling.factor
                    } else {
                        let smooth = (old_context_len / wavelen - scaling.low_freq_factor)
                            / (scaling.high_freq_factor - scaling.low_freq_factor);
                        (1.0 - smooth) * freq / scaling.factor + smooth * freq
                    }
                } else {
                    freq
                }
            })
            .collect();

        let mut cache = vec![0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let angle = pos as f64 * inv_freqs[i];
                cache[pos * rotary_dim + i] = angle.cos() as f32;
                cache[pos * rotary_dim + half + i] = angle.sin() as f32;
            }
        }

        let nbytes = max_pos * rotary_dim * dtype.size_bytes();
        let gpu_ptr = crate::driver::mem_alloc(nbytes)?;

        match dtype {
            DType::F32 => {
                let host = crate::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(cache.as_ptr() as *const u8, host, nbytes);
                crate::driver::memcpy_htod_async(gpu_ptr, host, nbytes, device.compute_stream)?;
                crate::driver::stream_synchronize(device.compute_stream)?;
                crate::driver::mem_free_host(host)?;
            }
            DType::F16 => {
                let f16_data: Vec<half::f16> =
                    cache.iter().map(|&v| half::f16::from_f32(v)).collect();
                let host = crate::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(f16_data.as_ptr() as *const u8, host, nbytes);
                crate::driver::memcpy_htod_async(gpu_ptr, host, nbytes, device.compute_stream)?;
                crate::driver::stream_synchronize(device.compute_stream)?;
                crate::driver::mem_free_host(host)?;
            }
            DType::BF16 => {
                let bf16_data: Vec<half::bf16> =
                    cache.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let host = crate::driver::mem_alloc_host(nbytes)?;
                std::ptr::copy_nonoverlapping(bf16_data.as_ptr() as *const u8, host, nbytes);
                crate::driver::memcpy_htod_async(gpu_ptr, host, nbytes, device.compute_stream)?;
                crate::driver::stream_synchronize(device.compute_stream)?;
                crate::driver::mem_free_host(host)?;
            }
            _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
        }

        let cos_sin_cache = GpuTensor::new(gpu_ptr, &[max_pos, rotary_dim], dtype);

        // Build separate cos/sin caches for FA2 fused RoPE (needs contiguous buffers).
        let (cos_cache, sin_cache) = Self::build_separate_cos_sin(
            &cache,
            max_pos,
            rotary_dim,
            dtype,
            device.compute_stream,
        )?;

        Ok(Self {
            cos_sin_cache,
            cos_cache,
            sin_cache,
            head_dim,
        })
    }

    unsafe fn build_separate_cos_sin(
        cache: &[f32],
        max_pos: usize,
        rotary_dim: usize,
        dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(GpuTensor, GpuTensor)> {
        let half = rotary_dim / 2;
        let half_nbytes = max_pos * half * dtype.size_bytes();
        let cos_data: Vec<f32> = (0..max_pos)
            .flat_map(|p| (0..half).map(move |i| cache[p * rotary_dim + i]))
            .collect();
        let sin_data: Vec<f32> = (0..max_pos)
            .flat_map(|p| (0..half).map(move |i| cache[p * rotary_dim + half + i]))
            .collect();

        let cos_gpu = crate::driver::mem_alloc(half_nbytes)?;
        let sin_gpu = crate::driver::mem_alloc(half_nbytes)?;
        let host = crate::driver::mem_alloc_host(half_nbytes)?;

        macro_rules! upload {
            ($data:expr, $gpu:expr, $T:ty) => {{
                let converted: Vec<$T> = $data.iter().map(|&v| <$T>::from_f32(v)).collect();
                std::ptr::copy_nonoverlapping(converted.as_ptr() as *const u8, host, half_nbytes);
                crate::driver::memcpy_htod_async($gpu, host, half_nbytes, stream)?;
            }};
        }

        match dtype {
            DType::F16 => {
                upload!(cos_data, cos_gpu, half::f16);
                upload!(sin_data, sin_gpu, half::f16);
            }
            DType::BF16 => {
                upload!(cos_data, cos_gpu, half::bf16);
                upload!(sin_data, sin_gpu, half::bf16);
            }
            DType::F32 => {
                std::ptr::copy_nonoverlapping(cos_data.as_ptr() as *const u8, host, half_nbytes);
                crate::driver::memcpy_htod_async(cos_gpu, host, half_nbytes, stream)?;
                std::ptr::copy_nonoverlapping(sin_data.as_ptr() as *const u8, host, half_nbytes);
                crate::driver::memcpy_htod_async(sin_gpu, host, half_nbytes, stream)?;
            }
            _ => anyhow::bail!("unsupported dtype for RoPE cache: {:?}", dtype),
        }
        crate::driver::stream_synchronize(stream)?;
        crate::driver::mem_free_host(host)?;

        Ok((
            GpuTensor::new(cos_gpu, &[max_pos, half], dtype),
            GpuTensor::new(sin_gpu, &[max_pos, half], dtype),
        ))
    }
}

// ---------------------------------------------------------------------------
// LlamaMLP
// ---------------------------------------------------------------------------

/// LLaMA MLP with fused or separate gate+up projection.
///
/// Dense: gate_up_proj(x) → SiLU(gate) * up → down_proj  (single fused GEMM)
/// Quantized: gate_proj(x), up_proj(x) → SiLU(gate) * up → down_proj  (separate GEMMs)
pub struct LlamaMLP {
    /// Fused gate+up for dense, or gate-only for quantized.
    pub(crate) gate_up_proj: LinearLayer,
    /// Separate up projection — only used for quantized (None for dense).
    up_proj: Option<LinearLayer>,
    pub(crate) down_proj: LinearLayer,
    pub(crate) intermediate_size: usize,
    /// NCCL group for TP all-reduce after down_proj (row parallel).
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

impl LlamaMLP {
    /// Forward pass returning `OwnedTensor` (caching-allocator path).
    pub unsafe fn forward(&self, x: TensorView<'_>, device: &mut GpuDevice) -> OwnedTensor {
        let gate_up = if let Some(ref up_proj) = self.up_proj {
            // Quantized: separate gate + up GEMMs, then concat
            let gate_out = self.gate_up_proj.forward(
                x,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            let up_out = up_proj.forward(
                x,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            // Concat gate_out and up_out along dim 1 → [num_tokens, 2*intermediate]
            let concat = kernels::concat_dim1(
                gate_out.as_gpu_tensor(),
                up_out.as_gpu_tensor(),
                &mut device.caching,
                device.compute_stream,
            );
            drop(gate_out);
            drop(up_out);
            concat
        } else {
            // Dense: single fused gate+up GEMM
            #[cfg(feature = "ferrite")]
            {
                self.gate_up_proj.forward_ferrite(
                    x,
                    &mut device.cublas,
                    &device.ferrite,
                    &mut device.caching,
                    device.compute_stream,
                )
            }
            #[cfg(not(feature = "ferrite"))]
            self.gate_up_proj.forward(
                x,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            )
        };

        let activated = kernels::silu_and_mul_fused(
            gate_up.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            device.compute_stream,
        );
        drop(gate_up);

        #[cfg(feature = "ferrite")]
        let result = self.down_proj.forward_ferrite(
            activated.view(),
            &mut device.cublas,
            &device.ferrite,
            &mut device.caching,
            device.compute_stream,
        );
        #[cfg(not(feature = "ferrite"))]
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
}

// ---------------------------------------------------------------------------
// LlamaAttention
// ---------------------------------------------------------------------------

/// LLaMA multi-head attention with RoPE, GQA, fused QKV.
///
/// Dense: single fused QKV GEMM. Quantized: separate Q, K, V GEMMs.
pub struct LlamaAttention {
    /// Fused QKV for dense, or Q-only for quantized.
    pub(crate) qkv_proj: LinearLayer,
    /// Separate K projection — only used for quantized.
    pub(crate) k_proj: Option<LinearLayer>,
    /// Separate V projection — only used for quantized.
    pub(crate) v_proj: Option<LinearLayer>,
    pub(crate) o_proj: LinearLayer,
    pub(crate) q_size: usize,
    pub(crate) kv_size: usize,
    pub(crate) num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub(crate) layer_idx: usize,
    /// Optional per-head QK-norm weights (Qwen3, Gemma3).
    /// When present, forward uses `qk_norm_rope` instead of `fused_qkv_rope`.
    pub q_norm_weight: Option<GpuTensor>,
    pub k_norm_weight: Option<GpuTensor>,
    pub qk_norm_eps: f32,
    /// NCCL group for TP all-reduce after o_proj (row parallel).
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

impl LlamaAttention {
    /// Load attention weights with fused QKV projection.
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        _layer_idx: usize,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let _q_size = num_q_heads * head_dim;
        let _kv_size = num_kv_heads * head_dim;

        // Load and fuse Q/K/V into single [q_size + 2*kv_size, hidden] weight.
        let _q_w = weights.take(&format!("{prefix}.q_proj.weight"))?;
        let k_w = weights.take(&format!("{prefix}.k_proj.weight"))?;
        let v_w = weights.take(&format!("{prefix}.v_proj.weight"))?;

        // TODO: fuse into contiguous buffer. For now, this needs D2D concat.
        // Use `load_fused` instead, which does the concat.
        let _ = (_q_w, k_w, v_w, _q_size, _kv_size, _layer_idx);
        anyhow::bail!("QKV fusion not yet implemented — use load_fused instead")
    }

    /// Forward pass with paged KV cache and FlashAttention-2, returning OwnedTensor.
    ///
    /// * `hidden_states`: `[num_tokens, hidden_size]`
    /// * `positions`: `[num_tokens]` (U32)
    /// * `slot_mapping`: `[num_tokens]` (I64) — absolute slot index for each new token
    /// * `cu_seqlens_q`: `[batch_size + 1]` (U32) — cumulative Q lengths
    /// * `seqused_k`: `[batch_size]` (U32) — per-sequence K lengths (including cached)
    /// * `block_table`: `[batch_size, max_blocks_per_seq]` (U32) — page table
    /// * `max_seqlen_q` / `max_seqlen_k`: max sequence lengths in batch
    /// * `kv_cache`: the paged KV cache pool
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
        // QKV projection → owned.
        let qkv = if let (Some(k_proj), Some(v_proj)) = (self.k_proj.as_ref(), self.v_proj.as_ref())
        {
            // Quantized: separate Q, K, V GEMMs → concat
            let q_out = self.qkv_proj.forward(
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
            // Concat Q, K, V along dim 1 → [num_tokens, q_size + 2*kv_size]
            let qk = kernels::concat_dim1(
                q_out.as_gpu_tensor(),
                k_out.as_gpu_tensor(),
                &mut device.caching,
                device.compute_stream,
            );
            drop(q_out);
            drop(k_out);
            let qkv = kernels::concat_dim1(
                qk.as_gpu_tensor(),
                v_out.as_gpu_tensor(),
                &mut device.caching,
                device.compute_stream,
            );
            drop(qk);
            drop(v_out);
            qkv
        } else {
            // Dense: single fused QKV GEMM
            #[cfg(feature = "ferrite")]
            {
                self.qkv_proj.forward_ferrite(
                    hidden_states,
                    &mut device.cublas,
                    &device.ferrite,
                    &mut device.caching,
                    device.compute_stream,
                )
            }
            #[cfg(not(feature = "ferrite"))]
            self.qkv_proj.forward(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            )
        };

        self.forward_from_qkv(
            qkv,
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
        )
    }

    /// Forward pass starting from pre-projected QKV tensor.
    /// Used by the fused norm+QKV path in the decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_from_qkv(
        &self,
        qkv: OwnedTensor,
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
        let num_tokens = qkv.as_gpu_tensor().dim(0);

        // Split QKV and apply RoPE (with optional per-head QK-norm for Qwen3/Gemma3).
        let (q, k, v) =
            if let (Some(q_norm_w), Some(k_norm_w)) = (self.q_norm_weight, self.k_norm_weight) {
                // QK-norm path: split QKV first, then fused QK-norm + RoPE in-place.
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
                // QK-norm (no RoPE on K — FA2 handles it on read).
                kernels::qk_norm_inplace(
                    q.as_gpu_tensor(),
                    k.as_gpu_tensor(),
                    q_norm_w,
                    k_norm_w,
                    self.num_q_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.qk_norm_eps,
                    device.compute_stream,
                );
                kernels::rotary_embedding_q_only(
                    q.as_gpu_tensor(),
                    *positions,
                    rotary.cos_sin_cache,
                    self.num_q_heads,
                    self.head_dim,
                    device.compute_stream,
                );
                (q, k, v)
            } else if max_seqlen_q == 1 {
                // Decode: fused QKV split + Q RoPE + cache write (K unrotated).
                // FA2 rotates cached K in shared memory during attention.
                let q = if kv_cache.is_fp8() {
                    kernels::fused_qkv_rope_cache_fp8(
                        qkv.as_gpu_tensor(),
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
                        qkv.as_gpu_tensor(),
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

                // K is stored rotated — FA2 only needs cos_sin_cache for
                // span blocks (identified by per-block flags). When no spans
                // are active (flags ptr is null), pass null cos_sin_cache so
                // rotate_cached_k stays false.
                let has_spans = !kv_cache.block_unrotated_gpu().is_null();
                let (cos_sin_ptr, rotary_dim) = if has_spans {
                    (
                        rotary.cos_sin_cache.raw_ptr() as *const u8,
                        rotary.cos_sin_cache.dim(1),
                    )
                } else {
                    (std::ptr::null(), 0)
                };
                let attn_output = crate::model::attention_helpers::attention_decode_from_cache(
                    q.view(),
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.scale,
                    0.0,
                    -1,
                    kv_cache,
                    self.layer_idx,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                    cos_sin_ptr,
                    rotary_dim,
                    false,
                );
                drop(q);

                // Reshape to [num_tokens, q_size] and output projection.
                let attn_flat = attn_output.view().reshape(&[num_tokens, self.q_size]);
                #[cfg(feature = "ferrite")]
                let result = self.o_proj.forward_ferrite(
                    attn_flat,
                    &mut device.cublas,
                    &device.ferrite,
                    &mut device.caching,
                    device.compute_stream,
                );
                #[cfg(not(feature = "ferrite"))]
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
            } else {
                // Prefill: split QKV, then rotate both Q and K in-place.
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

                // rotary_embedding_inplace expects 2D [num_tokens, total_dim].
                let num_tokens = q.as_gpu_tensor().dim(0);
                kernels::rotary_embedding_inplace(
                    q.as_gpu_tensor().reshape(&[num_tokens, self.q_size]),
                    k.as_gpu_tensor().reshape(&[num_tokens, self.kv_size]),
                    *positions,
                    rotary.cos_sin_cache,
                    self.head_dim,
                    device.compute_stream,
                );

                // Write rotated K and V to cache.
                crate::model::attention_helpers::write_kv_cache(
                    k.view(),
                    v.view(),
                    slot_mapping,
                    kv_cache,
                    self.layer_idx,
                    device.compute_stream,
                );

                // K is already rotated. Only pass cos_sin_cache when spans
                // are active so FA2 can rotate flagged blocks.
                let has_spans = !kv_cache.block_unrotated_gpu().is_null();
                let (cos_sin_ptr, rot_dim) = if has_spans {
                    (
                        rotary.cos_sin_cache.raw_ptr() as *const u8,
                        rotary.cos_sin_cache.dim(1),
                    )
                } else {
                    (std::ptr::null(), 0)
                };
                let attn_output = crate::model::attention_helpers::attention_standard(
                    q.view(),
                    k.view(),
                    v.view(),
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.scale,
                    kv_cache,
                    self.layer_idx,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                    cos_sin_ptr,
                    rot_dim,
                    false,
                );
                drop(q);
                drop(k);
                drop(v);

                let attn_flat = attn_output.view().reshape(&[num_tokens, self.q_size]);
                #[cfg(feature = "ferrite")]
                let result = self.o_proj.forward_ferrite(
                    attn_flat,
                    &mut device.cublas,
                    &device.ferrite,
                    &mut device.caching,
                    device.compute_stream,
                );
                #[cfg(not(feature = "ferrite"))]
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
            };

        // QK-norm fallthrough: write K/V then run attention.
        crate::model::attention_helpers::write_kv_cache(
            k.view(),
            v.view(),
            slot_mapping,
            kv_cache,
            self.layer_idx,
            device.compute_stream,
        );

        // QK-norm attention with span rotation (external kernel pre/post).
        let attn_output = crate::model::attention_helpers::with_span_rotation(
            kv_cache,
            self.layer_idx,
            TensorView::from_raw(rotary.cos_sin_cache),
            cu_seqlens_q,
            seqused_k,
            block_table,
            max_seqlen_k,
            device.compute_stream,
            || {
                crate::model::attention_helpers::attention_standard(
                    q.view(),
                    k.view(),
                    v.view(),
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    self.scale,
                    kv_cache,
                    self.layer_idx,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                    rotary.cos_sin_cache.raw_ptr() as *const u8,
                    rotary.cos_sin_cache.dim(1),
                    false,
                )
            },
        );
        drop(q);
        drop(k);
        drop(v);

        // Reshape to [num_tokens, q_size] and output projection.
        let attn_flat = attn_output.view().reshape(&[num_tokens, self.q_size]);
        #[cfg(feature = "ferrite")]
        let result = self.o_proj.forward_ferrite(
            attn_flat,
            &mut device.cublas,
            &device.ferrite,
            &mut device.caching,
            device.compute_stream,
        );
        #[cfg(not(feature = "ferrite"))]
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
}

// ---------------------------------------------------------------------------
// LlamaDecoderLayer
// ---------------------------------------------------------------------------

/// A single LLaMA decoder layer.
pub struct LlamaDecoderLayer {
    pub self_attn: LlamaAttention,
    pub mlp: LlamaMLP,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
    /// Granite residual multiplier (1.0 = no-op for LLaMA).
    pub residual_multiplier: f32,
}

impl LlamaDecoderLayer {
    /// Forward pass with residual threading and paged attention.
    ///
    /// * `hidden_states` — MLP output from previous layer (or embedding for first layer)
    /// * `residual` — residual stream (`None` for first layer)
    ///
    /// Returns `(mlp_output, residual)`.
    /// Forward using caching allocator with proper Rust ownership.
    ///
    /// Uses `fused_add_rms_norm_inplace` (zero copies) and caching allocator
    /// for all intermediate tensors.
    #[allow(clippy::too_many_arguments)]
    /// Forward using caching allocator with proper Rust ownership.
    ///
    /// `hidden_states`: OwnedTensor (MLP output from prev layer, or embedding).
    ///   Consumed by this function — memory freed after attention reads it.
    /// `residual`: Option<OwnedTensor>. None for first layer. Persists across layers.
    ///
    /// Returns `(mlp_output: OwnedTensor, residual: OwnedTensor)`.
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
        // ── Fused norm+QKV path (ferrite, Dense layers only, not first layer) ──
        // First layer falls through to standard path (no residual to add into).
        // Fused norm+GEMM path (ferrite, Dense layers only, not first layer)
        #[cfg(feature = "ferrite")]
        if residual.is_some()
            && matches!(
                self.self_attn.qkv_proj,
                crate::layers::LinearLayer::Dense(_)
            )
        {
            let residual = residual.unwrap();
            let qkv_linear = match &self.self_attn.qkv_proj {
                crate::layers::LinearLayer::Dense(l) => l,
                _ => unreachable!(),
            };

            // Ferrite: add_inplace + fused_norm_gemm for QKV
            kernels::add_inplace(*residual, *hidden_states, device.compute_stream);
            drop(hidden_states);

            let hidden = residual.as_gpu_tensor().dim(1) as u32;
            let qkv = crate::ferrite::launch_fused_norm_gemm(
                &FUSED_NORM_GEMM,
                *residual,
                qkv_linear.weight,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
                hidden,
                None,
                1.0,
                0.0,
                &mut device.caching,
                device.compute_stream,
            );

            // Attention from pre-projected QKV
            let attn_output = self.self_attn.forward_from_qkv(
                qkv,
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

            // forward_from_qkv already includes o_proj — just add to residual
            if self.residual_multiplier != 1.0 {
                kernels::scale_inplace(*attn_output, self.residual_multiplier, &device.cublas);
            }

            // Use standard fused_add_rms_norm for MLP norm too
            let res_gpu2 = *residual;
            kernels::fused_add_rms_norm_inplace(
                *attn_output,
                res_gpu2,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
                device.compute_stream,
            );
            // attn_output now contains normed data for MLP

            if let crate::layers::LinearLayer::Dense(ref gate_up_linear) = self.mlp.gate_up_proj {
                let gate_up = gate_up_linear.forward_ferrite(
                    attn_output.view(),
                    &device.ferrite,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(attn_output);

                let activated = kernels::silu_and_mul_fused(
                    gate_up.as_gpu_tensor(),
                    self.mlp.intermediate_size,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(gate_up);

                let mlp_output = self.mlp.down_proj.forward_ferrite(
                    activated.view(),
                    &mut device.cublas,
                    &device.ferrite,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(activated);

                if self.residual_multiplier != 1.0 {
                    kernels::scale_inplace(*mlp_output, self.residual_multiplier, &device.cublas);
                }

                return (mlp_output, residual);
            }

            // Fallback: if gate_up or down isn't Dense, use standard MLP path
            let mlp_output = self.mlp.forward(residual.view(), device);
            if self.residual_multiplier != 1.0 {
                kernels::scale_inplace(*mlp_output, self.residual_multiplier, &device.cublas);
            }
            return (mlp_output, residual);
        }

        // ── Standard path (non-ferrite or quantized) ──
        let (normed, residual) = if let Some(residual) = residual {
            let hs_gpu = *hidden_states;
            let res_gpu = *residual;
            kernels::fused_add_rms_norm_inplace(
                hs_gpu,
                res_gpu,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
                device.compute_stream,
            );
            (hidden_states, residual)
        } else {
            let normed = kernels::rms_norm(
                *hidden_states,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
                &mut device.caching,
                device.compute_stream,
            );
            (normed, hidden_states)
        };

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

        if self.residual_multiplier != 1.0 {
            kernels::scale_inplace(*attn_output, self.residual_multiplier, &device.cublas);
        }

        let res_gpu = *residual;
        kernels::fused_add_rms_norm_inplace(
            *attn_output,
            res_gpu,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
            device.compute_stream,
        );

        let mlp_output = self.mlp.forward(attn_output.view(), device);
        drop(attn_output);

        if self.residual_multiplier != 1.0 {
            kernels::scale_inplace(*mlp_output, self.residual_multiplier, &device.cublas);
        }

        (mlp_output, residual)
    }
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

/// LLaMA transformer backbone.
pub struct LlamaModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<LlamaDecoderLayer>,
    pub norm: RmsNorm,
    pub rotary: RotaryCache,
    /// Granite embedding multiplier (1.0 = no-op for LLaMA).
    pub embedding_multiplier: f32,
}

impl LlamaModel {
    /// Forward pass using caching allocator — zero D2D copies between layers.
    ///
    /// This matches Python's PyTorch flow: intermediates are freed on drop,
    /// `fused_add_rms_norm` mutates in-place, and only hidden_states + residual
    /// survive between layers.
    ///
    /// Returns hidden states `[num_tokens, hidden_size]` as OwnedTensor.
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
        // Embedding lookup — owned, survives into first layer.
        let hidden_states = kernels::embedding_gather(
            self.embed_tokens.weight,
            *input_ids,
            &mut device.caching,
            device.compute_stream,
        );

        // Granite: scale embeddings.
        if self.embedding_multiplier != 1.0 {
            kernels::scale_inplace(
                hidden_states.as_gpu_tensor(),
                self.embedding_multiplier,
                &device.cublas,
            );
        }

        // Like Python: OwnedTensor handles memory lifetime via Rust ownership.
        // When hidden_states is replaced, the old OwnedTensor is dropped and
        // its memory returns to the caching allocator's free list.
        let mut hidden_states: OwnedTensor = hidden_states;
        let mut residual: Option<OwnedTensor> = None;

        for layer in self.layers.iter() {
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
            // Old hidden_states was consumed by the layer (dropped inside).
            // Old residual was passed through (or created from hidden_states).
            hidden_states = hs;
            residual = Some(res);
        }

        // Final norm: mutates hidden_states and residual in-place.
        let hs_gpu = *hidden_states;
        let res_gpu = residual.as_ref().unwrap().as_gpu_tensor();
        kernels::fused_add_rms_norm_inplace(
            hs_gpu,
            res_gpu,
            self.norm.weight,
            self.norm.eps,
            device.compute_stream,
        );
        // hidden_states buffer now contains normed values.
        // Drop residual (frees the embedding/residual buffer).
        drop(residual);
        // Return hidden_states — caller owns this memory.
        hidden_states
    }
}

// ---------------------------------------------------------------------------
// LlamaForCausalLM
// ---------------------------------------------------------------------------

/// LLaMA for causal language modeling.
pub struct LlamaForCausalLM {
    pub model: LlamaModel,
    pub lm_head: LinearLayer,
    /// Granite logits scaling (1.0 = no-op for LLaMA).
    pub logits_scaling: f32,
    /// NCCL group for TP all-gather after lm_head (column parallel).
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
    /// Pipeline parallelism config. None = single GPU / PP=1.
    pub pp_config: Option<PpConfig>,
}

impl LlamaForCausalLM {
    /// Forward pass: input_ids → logits.
    ///
    /// If `last_token_indices` is provided, gathers only those rows from hidden
    /// states before the lm_head projection, returning `[num_reqs, vocab_size]`
    /// instead of `[num_tokens, vocab_size]`. This avoids computing the expensive
    /// vocab projection for tokens whose logits are never used.
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
            kernels::embedding_gather(
                hidden_states.as_gpu_tensor(),
                *indices,
                &mut device.caching,
                device.compute_stream,
            )
        } else {
            hidden_states
        };
        #[allow(unused_mut)]
        let mut logits = self.lm_head.forward(
            hidden_states.view(),
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );
        drop(hidden_states);

        // TP: all-gather logits (column parallel lm_head).
        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.tp_group {
            let gathered = group.all_gather(logits.as_gpu_tensor(), &mut device.caching);
            drop(logits);
            logits = gathered;
        }

        // Granite: scale logits by 1/logits_scaling.
        if self.logits_scaling != 1.0 {
            kernels::scale_inplace(
                logits.as_gpu_tensor(),
                self.logits_scaling.recip(),
                &device.cublas,
            );
        }

        logits
    }
}

// ---------------------------------------------------------------------------
// TP group injection
// ---------------------------------------------------------------------------

#[cfg(feature = "nccl")]
impl LlamaForCausalLM {
    /// Inject NCCL process group into all TP layers.
    /// Must be called after loading with `load_tp()`.
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        for layer in &mut self.model.layers {
            layer.self_attn.tp_group = Some(Arc::clone(&group));
            layer.mlp.tp_group = Some(Arc::clone(&group));
        }
        self.tp_group = Some(group);
    }
}

// ---------------------------------------------------------------------------
// Fused weight loading (CPU → GPU direct)
// ---------------------------------------------------------------------------

impl LlamaAttention {
    /// Load with fused QKV weights (dense) — streams directly from CPU to GPU.
    pub fn load_fused(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // Try pre-fused qkv_proj first (Phi-4 / Phi-3 models), then fall back
        // to separate q/k/v (standard Llama/Qwen2/Mistral).
        let fused_name = format!("{prefix}.qkv_proj.weight");
        let qkv_proj = if weights.contains(&fused_name) {
            let w = weights.take(&fused_name)?;
            let bias_name = format!("{prefix}.qkv_proj.bias");
            let bias = if weights.contains(&bias_name) {
                Some(weights.take(&bias_name)?)
            } else {
                None
            };
            LinearLayer::Dense(Linear::new(w, bias))
        } else {
            // Get shapes/dtypes from CPU metadata to pre-allocate fused tensor.
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

            // Pre-allocate fused QKV tensor on GPU.
            let ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };
            weights.record_alloc(ptr, total_bytes);

            // Stream each component directly from CPU → GPU offset.
            unsafe {
                weights.take_into(&q_name, ptr, stream)?;
                weights.take_into(&k_name, ptr.add(q_bytes), stream)?;
                weights.take_into(&v_name, ptr.add(q_bytes + kv_bytes), stream)?;
            }

            let qkv_w = unsafe { GpuTensor::new(ptr, &[q_size + 2 * kv_size, hidden], q_dtype) };

            // Fuse QKV bias if present (Qwen2 has QKV bias, LLaMA doesn't).
            let q_bias_name = format!("{prefix}.q_proj.bias");
            let k_bias_name = format!("{prefix}.k_proj.bias");
            let v_bias_name = format!("{prefix}.v_proj.bias");
            let qkv_bias = if weights.contains(&q_bias_name) {
                let (q_b_shape, q_b_dtype) = weights
                    .tensor_info(&q_bias_name)
                    .ok_or_else(|| anyhow::anyhow!("weight not found: {q_bias_name}"))?;
                let q_b_bytes = q_b_shape.iter().product::<usize>() * q_b_dtype.size_bytes();
                let (k_b_shape, _) = weights
                    .tensor_info(&k_bias_name)
                    .ok_or_else(|| anyhow::anyhow!("weight not found: {k_bias_name}"))?;
                let k_b_bytes = k_b_shape.iter().product::<usize>() * q_b_dtype.size_bytes();
                let (v_b_shape, _) = weights
                    .tensor_info(&v_bias_name)
                    .ok_or_else(|| anyhow::anyhow!("weight not found: {v_bias_name}"))?;
                let v_b_bytes = v_b_shape.iter().product::<usize>() * q_b_dtype.size_bytes();
                let total_bias_bytes = q_b_bytes + k_b_bytes + v_b_bytes;
                let total_elems = total_bias_bytes / q_b_dtype.size_bytes();

                let bias_ptr = unsafe { crate::driver::mem_alloc(total_bias_bytes)? };
                weights.record_alloc(bias_ptr, total_bias_bytes);
                unsafe {
                    weights.take_into(&q_bias_name, bias_ptr, stream)?;
                    weights.take_into(&k_bias_name, bias_ptr.add(q_b_bytes), stream)?;
                    weights.take_into(&v_bias_name, bias_ptr.add(q_b_bytes + k_b_bytes), stream)?;
                }
                Some(unsafe { GpuTensor::new(bias_ptr, &[total_elems], q_b_dtype) })
            } else {
                None
            };
            LinearLayer::Dense(Linear::new(qkv_w, qkv_bias))
        };

        let o_proj = LinearLayer::Dense(Linear::load(weights, &format!("{prefix}.o_proj"))?);

        // Auto-detect QK-norm weights (Qwen3, Gemma3, etc.).
        let q_norm_name = format!("{prefix}.q_norm.weight");
        let k_norm_name = format!("{prefix}.k_norm.weight");
        let q_norm_weight = if weights.contains(&q_norm_name) {
            Some(weights.take(&q_norm_name)?)
        } else {
            None
        };
        let k_norm_weight = if weights.contains(&k_norm_name) {
            Some(weights.take(&k_norm_name)?)
        } else {
            None
        };
        let qk_norm_eps = if q_norm_weight.is_some() {
            config.rms_norm_eps
        } else {
            0.0
        };

        Ok(Self {
            qkv_proj,
            k_proj: None,
            v_proj: None,
            o_proj,
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            layer_idx,
            q_norm_weight,
            k_norm_weight,
            qk_norm_eps,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load with fused QKV weights + QK-norm weights (Qwen3 MoE, Gemma3).
    ///
    /// Same as `load_fused` but also loads `{prefix}.q_norm.weight` and
    /// `{prefix}.k_norm.weight` for per-head RMS normalization before RoPE.
    pub fn load_fused_with_qk_norm(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        qk_norm_eps: f32,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let mut attn = Self::load_fused(weights, prefix, config, layer_idx, stream)?;

        let q_norm_name = format!("{prefix}.q_norm.weight");
        let k_norm_name = format!("{prefix}.k_norm.weight");

        if weights.contains(&q_norm_name) {
            attn.q_norm_weight = Some(weights.take(&q_norm_name)?);
        }
        if weights.contains(&k_norm_name) {
            attn.k_norm_weight = Some(weights.take(&k_norm_name)?);
        }
        attn.qk_norm_eps = qk_norm_eps;

        Ok(attn)
    }

    /// Load with QK-norm and TP-sharded weights.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fused_with_qk_norm_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        qk_norm_eps: f32,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let mut attn = Self::load_fused_tp(weights, prefix, config, layer_idx, tp, stream)?;

        let q_norm_name = format!("{prefix}.q_norm.weight");
        let k_norm_name = format!("{prefix}.k_norm.weight");

        if weights.contains(&q_norm_name) {
            attn.q_norm_weight = Some(weights.take(&q_norm_name)?);
        }
        if weights.contains(&k_norm_name) {
            attn.k_norm_weight = Some(weights.take(&k_norm_name)?);
        }
        attn.qk_norm_eps = qk_norm_eps;

        Ok(attn)
    }
}

impl LlamaMLP {
    /// Load with fused gate+up weights — streams directly from CPU to GPU.
    ///
    /// Matches Python vLLM's MergedColumnParallelLinear: pre-allocates the
    /// fused [2*intermediate, hidden] tensor, then copies gate and up from CPU.
    pub fn load_fused(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        // Try pre-fused gate_up_proj first (Phi-4 / Phi-3 models), then fall
        // back to separate gate/up (standard Llama/Mistral).
        let fused_name = format!("{prefix}.gate_up_proj.weight");
        let gate_up_proj = if weights.contains(&fused_name) {
            let w = weights.take(&fused_name)?;
            LinearLayer::Dense(Linear::new(w, None))
        } else {
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
            let elem_size = gate_dtype.size_bytes();
            let gate_bytes = gate_shape.iter().product::<usize>() * elem_size;
            let up_bytes = gate_bytes; // same shape

            let total_bytes = gate_bytes + up_bytes;
            let ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };
            weights.record_alloc(ptr, total_bytes);

            unsafe {
                weights.take_into(&gate_name, ptr, stream)?;
                weights.take_into(&up_name, ptr.add(gate_bytes), stream)?;
            }

            let gate_up_w =
                unsafe { GpuTensor::new(ptr, &[2 * intermediate_size, hidden], gate_dtype) };
            LinearLayer::Dense(Linear::new(gate_up_w, None))
        };

        let down_proj = LinearLayer::Dense(Linear::load(weights, &format!("{prefix}.down_proj"))?);
        Ok(Self {
            gate_up_proj,
            up_proj: None,
            down_proj,
            intermediate_size,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Construct from pre-built linear layers (used by GGUF DeepSeek dense MLP).
    pub fn from_parts(
        gate_up_proj: LinearLayer,
        up_proj: Option<LinearLayer>,
        down_proj: LinearLayer,
        intermediate_size: usize,
    ) -> Self {
        Self {
            gate_up_proj,
            up_proj,
            down_proj,
            intermediate_size,
            #[cfg(feature = "nccl")]
            tp_group: None,
        }
    }
}

impl LlamaAttention {
    /// Load quantized attention (separate Q, K, V, O Marlin layers).
    #[allow(clippy::too_many_arguments)]
    pub fn load_quantized(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
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

        // Fused QKV: concat q/k/v on CPU, single repack + single Marlin GEMM.
        // Matches Python vLLM's QKVParallelLinear.
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

        Ok(Self {
            qkv_proj: LinearLayer::Marlin(Box::new(qkv)),
            k_proj: None,
            v_proj: None,
            o_proj: LinearLayer::Marlin(Box::new(o)),
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            layer_idx,
            q_norm_weight: None,
            k_norm_weight: None,
            qk_norm_eps: 0.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load FP8 quantized attention with fused QKV.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        output_dtype: DType,
        qk_norm_eps: f32,
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

        let q_norm_name = format!("{prefix}.q_norm.weight");
        let k_norm_name = format!("{prefix}.k_norm.weight");
        let q_norm_weight = if weights.contains(&q_norm_name) {
            Some(weights.take(&q_norm_name)?)
        } else {
            None
        };
        let k_norm_weight = if weights.contains(&k_norm_name) {
            Some(weights.take(&k_norm_name)?)
        } else {
            None
        };

        Ok(Self {
            qkv_proj: LinearLayer::Fp8(Box::new(qkv)),
            k_proj: None,
            v_proj: None,
            o_proj: LinearLayer::Fp8(Box::new(o)),
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            layer_idx,
            q_norm_weight,
            k_norm_weight,
            qk_norm_eps,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load FP8 block-quantized attention with fused QKV, TP sharded.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8_block_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        output_dtype: DType,
        qk_norm_eps: f32,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads / tp.world_size;
        let num_kv_heads = config.num_kv_heads / tp.world_size;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // QKV: column parallel (shard dim=0).
        let qkv = gpu_weights::load_fused_fp8_block_linear_tp(
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
        // O: row parallel (shard dim=1).
        let o = gpu_weights::load_fp8_block_linear_tp(
            weights,
            &format!("{prefix}.o_proj"),
            output_dtype,
            tp.rank,
            tp.world_size,
        )?;

        let q_norm_name = format!("{prefix}.q_norm.weight");
        let k_norm_name = format!("{prefix}.k_norm.weight");
        let q_norm_weight = if weights.contains(&q_norm_name) {
            Some(weights.take(&q_norm_name)?)
        } else {
            None
        };
        let k_norm_weight = if weights.contains(&k_norm_name) {
            Some(weights.take(&k_norm_name)?)
        } else {
            None
        };

        Ok(Self {
            qkv_proj: LinearLayer::Fp8Block(Box::new(qkv)),
            k_proj: None,
            v_proj: None,
            o_proj: LinearLayer::Fp8Block(Box::new(o)),
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            layer_idx,
            q_norm_weight,
            k_norm_weight,
            qk_norm_eps,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load per-tensor FP8 attention with TP (column-parallel QKV, row-parallel O).
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        output_dtype: DType,
        qk_norm_eps: f32,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads / tp.world_size;
        let num_kv_heads = config.num_kv_heads / tp.world_size;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // QKV: column parallel (shard dim=0).
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
        // O: row parallel (shard dim=1).
        let o = gpu_weights::load_fp8_linear_tp(
            weights,
            &format!("{prefix}.o_proj"),
            output_dtype,
            tp.rank,
            tp.world_size,
        )?;

        let q_norm_name = format!("{prefix}.q_norm.weight");
        let k_norm_name = format!("{prefix}.k_norm.weight");
        let q_norm_weight = if weights.contains(&q_norm_name) {
            Some(weights.take(&q_norm_name)?)
        } else {
            None
        };
        let k_norm_weight = if weights.contains(&k_norm_name) {
            Some(weights.take(&k_norm_name)?)
        } else {
            None
        };

        Ok(Self {
            qkv_proj: LinearLayer::Fp8(Box::new(qkv)),
            k_proj: None,
            v_proj: None,
            o_proj: LinearLayer::Fp8(Box::new(o)),
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            layer_idx,
            q_norm_weight,
            k_norm_weight,
            qk_norm_eps,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load FP8 block-quantized attention with fused QKV.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8_block(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        output_dtype: DType,
        qk_norm_eps: f32,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        let qkv = gpu_weights::load_fused_fp8_block_linear(
            weights,
            &[
                format!("{prefix}.q_proj"),
                format!("{prefix}.k_proj"),
                format!("{prefix}.v_proj"),
            ],
            output_dtype,
            stream,
        )?;
        let o =
            gpu_weights::load_fp8_block_linear(weights, &format!("{prefix}.o_proj"), output_dtype)?;

        let q_norm_name = format!("{prefix}.q_norm.weight");
        let k_norm_name = format!("{prefix}.k_norm.weight");
        let q_norm_weight = if weights.contains(&q_norm_name) {
            Some(weights.take(&q_norm_name)?)
        } else {
            None
        };
        let k_norm_weight = if weights.contains(&k_norm_name) {
            Some(weights.take(&k_norm_name)?)
        } else {
            None
        };

        Ok(Self {
            qkv_proj: LinearLayer::Fp8Block(Box::new(qkv)),
            k_proj: None,
            v_proj: None,
            o_proj: LinearLayer::Fp8Block(Box::new(o)),
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            layer_idx,
            q_norm_weight,
            k_norm_weight,
            qk_norm_eps,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

impl LlamaMLP {
    /// Load quantized MLP (separate gate, up, down Marlin layers).
    #[allow(clippy::too_many_arguments)]
    pub fn load_quantized(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        qconfig: &QuantConfig,
        workspace: GpuTensor,
        device: &GpuDevice,
    ) -> Result<Self> {
        let mut alloc = crate::alloc::CachingAllocator::new();

        // Fused gate_up: concat gate/up on CPU, single repack + single Marlin GEMM.
        // Matches Python vLLM's MergedColumnParallelLinear.
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
            up_proj: None,
            down_proj: LinearLayer::Marlin(Box::new(down)),
            intermediate_size,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load FP8 quantized MLP with fused gate+up.
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
            up_proj: None,
            down_proj: LinearLayer::Fp8(Box::new(down)),
            intermediate_size,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load FP8 block-quantized MLP with fused gate+up, TP sharded.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8_block_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        output_dtype: DType,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let ipp = intermediate_size / tp.world_size;
        let gate_up = gpu_weights::load_fused_fp8_block_linear_tp(
            weights,
            &[format!("{prefix}.gate_proj"), format!("{prefix}.up_proj")],
            output_dtype,
            tp.rank,
            tp.world_size,
            stream,
        )?;
        let down = gpu_weights::load_fp8_block_linear_tp(
            weights,
            &format!("{prefix}.down_proj"),
            output_dtype,
            tp.rank,
            tp.world_size,
        )?;

        Ok(Self {
            gate_up_proj: LinearLayer::Fp8Block(Box::new(gate_up)),
            up_proj: None,
            down_proj: LinearLayer::Fp8Block(Box::new(down)),
            intermediate_size: ipp,
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
            up_proj: None,
            down_proj: LinearLayer::Fp8(Box::new(down)),
            intermediate_size: ipp,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load FP8 block-quantized MLP with fused gate+up.
    pub fn load_fp8_block(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        output_dtype: DType,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let gate_up = gpu_weights::load_fused_fp8_block_linear(
            weights,
            &[format!("{prefix}.gate_proj"), format!("{prefix}.up_proj")],
            output_dtype,
            stream,
        )?;
        let down = gpu_weights::load_fp8_block_linear(
            weights,
            &format!("{prefix}.down_proj"),
            output_dtype,
        )?;

        Ok(Self {
            gate_up_proj: LinearLayer::Fp8Block(Box::new(gate_up)),
            up_proj: None,
            down_proj: LinearLayer::Fp8Block(Box::new(down)),
            intermediate_size,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

impl LlamaDecoderLayer {
    /// Load a quantized decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_quantized(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        qconfig: &QuantConfig,
        workspace: GpuTensor,
        device: &GpuDevice,
    ) -> Result<Self> {
        let self_attn = LlamaAttention::load_quantized(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            qconfig,
            workspace,
            device,
        )?;
        let mlp = LlamaMLP::load_quantized(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            qconfig,
            workspace,
            device,
        )?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: 1.0,
        })
    }

    /// Load an FP8 quantized decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        output_dtype: DType,
        qk_norm_eps: f32,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let self_attn = LlamaAttention::load_fp8(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            output_dtype,
            qk_norm_eps,
            stream,
        )?;
        let mlp = LlamaMLP::load_fp8(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            output_dtype,
            stream,
        )?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: 1.0,
        })
    }

    /// Load an FP8 block-quantized decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8_block(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        output_dtype: DType,
        qk_norm_eps: f32,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let self_attn = LlamaAttention::load_fp8_block(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            output_dtype,
            qk_norm_eps,
            stream,
        )?;
        let mlp = LlamaMLP::load_fp8_block(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            output_dtype,
            stream,
        )?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: 1.0,
        })
    }

    /// Load a per-tensor FP8 decoder layer with TP.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        output_dtype: DType,
        qk_norm_eps: f32,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let self_attn = LlamaAttention::load_fp8_tp(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            output_dtype,
            qk_norm_eps,
            tp,
            stream,
        )?;
        let mlp = LlamaMLP::load_fp8_tp(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            output_dtype,
            tp,
            stream,
        )?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: 1.0,
        })
    }

    /// Load a block FP8 decoder layer with TP.
    #[allow(clippy::too_many_arguments)]
    pub fn load_fp8_block_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        output_dtype: DType,
        qk_norm_eps: f32,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let self_attn = LlamaAttention::load_fp8_block_tp(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            output_dtype,
            qk_norm_eps,
            tp,
            stream,
        )?;
        let mlp = LlamaMLP::load_fp8_block_tp(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            output_dtype,
            tp,
            stream,
        )?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: 1.0,
        })
    }

    /// Load a decoder layer with fused weights (dense).
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let self_attn = LlamaAttention::load_fused(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            stream,
        )?;
        let mlp = LlamaMLP::load_fused(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            stream,
        )?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: 1.0,
        })
    }
}

impl LlamaModel {
    /// Load the model backbone.
    pub fn load(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = LlamaDecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                device.compute_stream,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;

        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
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
            embedding_multiplier: 1.0,
        })
    }
}

impl LlamaForCausalLM {
    /// Load the full model.
    pub fn load(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaModel::load(weights, config, dtype, device)?;

        let lm_head = LinearLayer::Dense(if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        });

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: None,
        })
    }

    /// Load a quantized model (AWQ/GPTQ → Marlin).
    pub fn load_quantized(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        qconfig: &QuantConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let workspace = gpu_weights::alloc_marlin_workspace(device.num_sm, device.compute_stream)?;

        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let layer = LlamaDecoderLayer::load_quantized(
                weights, &prefix, config, i, qconfig, workspace, device,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
                dtype,
                device,
            )?
        };
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

        let model = LlamaModel {
            embed_tokens,
            layers,
            norm,
            rotary,
            embedding_multiplier: 1.0,
        };

        // lm_head is always dense (not quantized)
        let lm_head = LinearLayer::Dense(if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        });

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: None,
        })
    }

    /// Load an FP8 quantized model.
    pub fn load_fp8(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        qk_norm_eps: f32,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let layer = LlamaDecoderLayer::load_fp8(
                weights,
                &prefix,
                config,
                i,
                dtype,
                qk_norm_eps,
                device.compute_stream,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
                dtype,
                device,
            )?
        };
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

        let model = LlamaModel {
            embed_tokens,
            layers,
            norm,
            rotary,
            embedding_multiplier: 1.0,
        };

        // lm_head is always dense (not quantized) — matches Python.
        let lm_head = LinearLayer::Dense(if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        });

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: None,
        })
    }

    /// Load an FP8 block-quantized model (e.g. Qwen3-8B-FP8).
    pub fn load_fp8_block(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        qk_norm_eps: f32,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let layer = LlamaDecoderLayer::load_fp8_block(
                weights,
                &prefix,
                config,
                i,
                dtype,
                qk_norm_eps,
                device.compute_stream,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
                dtype,
                device,
            )?
        };
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

        let model = LlamaModel {
            embed_tokens,
            layers,
            norm,
            rotary,
            embedding_multiplier: 1.0,
        };

        // lm_head is always dense (not quantized) — matches Python.
        let lm_head = LinearLayer::Dense(if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        });

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: None,
        })
    }

    /// Load a per-tensor FP8 model with TP sharding.
    pub fn load_fp8_tp(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        qk_norm_eps: f32,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let layer = LlamaDecoderLayer::load_fp8_tp(
                weights,
                &prefix,
                config,
                i,
                dtype,
                qk_norm_eps,
                tp,
                device.compute_stream,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
                dtype,
                device,
            )?
        };
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

        let model = LlamaModel {
            embed_tokens,
            layers,
            norm,
            rotary,
            embedding_multiplier: 1.0,
        };

        let lm_head = if config.tie_word_embeddings {
            LinearLayer::Dense(Linear::new(model.embed_tokens.weight, None))
        } else {
            let lm_w = weights.take_shard("lm_head.weight", 0, tp.rank, tp.world_size)?;
            LinearLayer::Dense(Linear::new(lm_w, None))
        };

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: None,
        })
    }

    /// Load an FP8 block-quantized model with TP sharding.
    pub fn load_fp8_block_tp(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        qk_norm_eps: f32,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let layer = LlamaDecoderLayer::load_fp8_block_tp(
                weights,
                &prefix,
                config,
                i,
                dtype,
                qk_norm_eps,
                tp,
                device.compute_stream,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
                dtype,
                device,
            )?
        };
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

        let model = LlamaModel {
            embed_tokens,
            layers,
            norm,
            rotary,
            embedding_multiplier: 1.0,
        };

        let lm_head = if config.tie_word_embeddings {
            LinearLayer::Dense(Linear::new(model.embed_tokens.weight, None))
        } else {
            let lm_w = weights.take_shard("lm_head.weight", 0, tp.rank, tp.world_size)?;
            LinearLayer::Dense(Linear::new(lm_w, None))
        };

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Tensor-parallel loading (Phase 4)
// ---------------------------------------------------------------------------

// ColumnParallelLinear/RowParallelLinear/VocabParallelEmbedding available in layers
// but TP loading constructs plain LinearLayer + injects NcclGroup on the model structs.

/// TP sharding config passed through the load call chain.
#[derive(Debug, Clone, Copy)]
pub struct TpConfig {
    pub rank: usize,
    pub world_size: usize,
}

impl LlamaAttention {
    /// Load with fused QKV weights, sharded for tensor parallelism.
    ///
    /// QKV: shard Q by num_q_heads/world_size, K/V by num_kv_heads/world_size (dim=0).
    /// o_proj: shard along dim=1 (row parallel).
    #[allow(clippy::too_many_arguments)]
    pub fn load_fused_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_q_heads = config.num_attention_heads / tp.world_size;
        let num_kv_heads = config.num_kv_heads / tp.world_size;
        let head_dim = config.head_dim;
        let q_size = num_q_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // Try pre-fused qkv_proj first (Phi-4 / Phi-3 models), then fall back
        // to separate q/k/v.
        let fused_name = format!("{prefix}.qkv_proj.weight");
        let qkv_proj = if weights.contains(&fused_name) {
            // Pre-fused: shard the single qkv_proj along dim=0.
            let w = weights.take_shard(&fused_name, 0, tp.rank, tp.world_size)?;
            let bias_name = format!("{prefix}.qkv_proj.bias");
            let bias = if weights.contains(&bias_name) {
                Some(weights.take_shard(&bias_name, 0, tp.rank, tp.world_size)?)
            } else {
                None
            };
            LinearLayer::Dense(Linear::new(w, bias))
        } else {
            // Get dtype from weight metadata.
            let q_name = format!("{prefix}.q_proj.weight");
            let k_name = format!("{prefix}.k_proj.weight");
            let v_name = format!("{prefix}.v_proj.weight");

            let (_q_shape, q_dtype) = weights
                .tensor_info(&q_name)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {q_name}"))?;
            let hidden = _q_shape[1];
            let elem_size = q_dtype.size_bytes();

            // Pre-allocate fused QKV tensor for this rank's shard.
            let q_bytes = q_size * hidden * elem_size;
            let kv_bytes = kv_size * hidden * elem_size;
            let total_bytes = q_bytes + 2 * kv_bytes;
            let ptr = unsafe { crate::driver::mem_alloc(total_bytes)? };
            weights.record_alloc(ptr, total_bytes);

            // Shard each component along dim=0 and stream into fused buffer.
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

            // Fuse QKV bias if present (Qwen2), also sharded.
            let q_bias_name = format!("{prefix}.q_proj.bias");
            let k_bias_name = format!("{prefix}.k_proj.bias");
            let v_bias_name = format!("{prefix}.v_proj.bias");
            let qkv_bias = if weights.contains(&q_bias_name) {
                let (_, q_b_dtype) = weights
                    .tensor_info(&q_bias_name)
                    .ok_or_else(|| anyhow::anyhow!("weight not found: {q_bias_name}"))?;
                let q_b_bytes = q_size * q_b_dtype.size_bytes();
                let k_b_bytes = kv_size * q_b_dtype.size_bytes();
                let v_b_bytes = kv_size * q_b_dtype.size_bytes();
                let total_bias_bytes = q_b_bytes + k_b_bytes + v_b_bytes;
                let total_elems = total_bias_bytes / q_b_dtype.size_bytes();

                let bias_ptr = unsafe { crate::driver::mem_alloc(total_bias_bytes)? };
                weights.record_alloc(bias_ptr, total_bias_bytes);
                unsafe {
                    weights.take_shard_into(
                        &q_bias_name,
                        0,
                        tp.rank,
                        tp.world_size,
                        bias_ptr,
                        stream,
                    )?;
                    weights.take_shard_into(
                        &k_bias_name,
                        0,
                        tp.rank,
                        tp.world_size,
                        bias_ptr.add(q_b_bytes),
                        stream,
                    )?;
                    weights.take_shard_into(
                        &v_bias_name,
                        0,
                        tp.rank,
                        tp.world_size,
                        bias_ptr.add(q_b_bytes + k_b_bytes),
                        stream,
                    )?;
                }
                Some(unsafe { GpuTensor::new(bias_ptr, &[total_elems], q_b_dtype) })
            } else {
                None
            };
            LinearLayer::Dense(Linear::new(qkv_w, qkv_bias))
        };

        // o_proj: shard along dim=1 (row parallel — input is split across ranks).
        let o_name = format!("{prefix}.o_proj.weight");
        let o_w = weights.take_shard(&o_name, 1, tp.rank, tp.world_size)?;
        // o_proj bias (if any) is added AFTER all-reduce, so load full bias.
        let o_bias_name = format!("{prefix}.o_proj.bias");
        let o_bias = if weights.contains(&o_bias_name) {
            Some(weights.take(&o_bias_name)?)
        } else {
            None
        };
        let o_proj = LinearLayer::Dense(Linear::new(o_w, o_bias));

        Ok(Self {
            qkv_proj,
            k_proj: None,
            v_proj: None,
            o_proj,
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            layer_idx,
            q_norm_weight: None,
            k_norm_weight: None,
            qk_norm_eps: 0.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

impl LlamaMLP {
    /// Load with fused gate+up weights, sharded for tensor parallelism.
    ///
    /// gate_up_proj: shard along dim=0 (column parallel).
    /// down_proj: shard along dim=1 (row parallel).
    pub fn load_fused_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let shard_intermediate = intermediate_size / tp.world_size;

        // Try pre-fused gate_up_proj first (Phi-4 / Phi-3 models), then fall
        // back to separate gate/up.
        let fused_name = format!("{prefix}.gate_up_proj.weight");
        let gate_up_proj = if weights.contains(&fused_name) {
            let w = weights.take_shard(&fused_name, 0, tp.rank, tp.world_size)?;
            LinearLayer::Dense(Linear::new(w, None))
        } else {
            let gate_name = format!("{prefix}.gate_proj.weight");
            let up_name = format!("{prefix}.up_proj.weight");

            let (gate_shape, gate_dtype) = weights
                .tensor_info(&gate_name)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
            let hidden = gate_shape[1];
            let elem_size = gate_dtype.size_bytes();

            // Each shard: [shard_intermediate, hidden].
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
            LinearLayer::Dense(Linear::new(gate_up_w, None))
        };

        // down_proj: shard along dim=1 (row parallel).
        let down_name = format!("{prefix}.down_proj.weight");
        let down_w = weights.take_shard(&down_name, 1, tp.rank, tp.world_size)?;
        let down_proj = LinearLayer::Dense(Linear::new(down_w, None));

        Ok(Self {
            gate_up_proj,
            up_proj: None,
            down_proj,
            intermediate_size: shard_intermediate,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

impl LlamaDecoderLayer {
    /// Load a decoder layer with TP-sharded weights.
    #[allow(clippy::too_many_arguments)]
    pub fn load_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let self_attn = LlamaAttention::load_fused_tp(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            tp,
            stream,
        )?;
        let mlp = LlamaMLP::load_fused_tp(
            weights,
            &format!("{prefix}.mlp"),
            config.intermediate_size,
            tp,
            stream,
        )?;
        // Norms are NOT sharded — identical on all ranks.
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: 1.0,
        })
    }
}

impl LlamaModel {
    /// Load the model backbone with TP sharding.
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        // Embedding: load full (all ranks need all vocab for now).
        // VocabParallelEmbedding sharding is handled at the CausalLM level.
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = LlamaDecoderLayer::load_tp(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                tp,
                device.compute_stream,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;

        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
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
            embedding_multiplier: 1.0,
        })
    }
}

impl LlamaForCausalLM {
    /// Load the full model with TP sharding.
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaModel::load_tp(weights, config, dtype, tp, device)?;

        // lm_head: column-parallel (shard output dim).
        let lm_head = if config.tie_word_embeddings {
            // Tied embeddings — use full weight (all-gather at inference time).
            LinearLayer::Dense(Linear::new(model.embed_tokens.weight, None))
        } else {
            let lm_w = weights.take_shard("lm_head.weight", 0, tp.rank, tp.world_size)?;
            LinearLayer::Dense(Linear::new(lm_w, None))
        };

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: None,
        })
    }

    /// Load from GGUF file (GGML quantized weights).
    ///
    /// Linear layers stay quantized on GPU (GgmlLinear), norms are dequantized to f32,
    /// embeddings are dequantized to f32 (for the embedding gather kernel).
    pub fn load_gguf(
        gguf_weights: &mut crate::ggml::GgufGpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        use crate::ggml::GgufWeight;
        use crate::layers::{GgmlLinear, LinearLayer};

        // Helper: create a LinearLayer from a GGUF weight (quantized or dense).
        let make_linear = |w: GgufWeight| -> LinearLayer {
            match w {
                GgufWeight::Quantized(s) => LinearLayer::Ggml(Box::new(GgmlLinear {
                    storage: s,
                    bias: None,
                })),
                GgufWeight::Dense(t) => LinearLayer::Dense(Linear::new(t, None)),
            }
        };

        // Embedding: dequantized at load time.
        let embed_w = gguf_weights.take_dense("model.embed_tokens.weight")?;
        let embed_tokens = Embedding::new(embed_w);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // Attention: separate Q, K, V, O projections (no fused QKV for GGUF).
            let q_w = gguf_weights
                .take(&format!("{prefix}.self_attn.q_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.self_attn.q_proj.weight"))?;
            let k_w = gguf_weights
                .take(&format!("{prefix}.self_attn.k_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.self_attn.k_proj.weight"))?;
            let v_w = gguf_weights
                .take(&format!("{prefix}.self_attn.v_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.self_attn.v_proj.weight"))?;
            let o_w = gguf_weights
                .take(&format!("{prefix}.self_attn.o_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.self_attn.o_proj.weight"))?;

            let num_q_heads = config.num_attention_heads;
            let num_kv_heads = config.num_kv_heads;
            let head_dim = config.head_dim;
            let q_size = num_q_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;

            // For GGUF, we use separate Q/K/V projections (not fused QKV).
            // The Q proj goes in qkv_proj, K and V in k_proj/v_proj.
            // QK norms (Qwen3, etc.) — optional.
            let q_norm_name = format!("{prefix}.self_attn.q_norm.weight");
            let k_norm_name = format!("{prefix}.self_attn.k_norm.weight");
            let q_norm_weight = gguf_weights.take(&q_norm_name).map(|w| match w {
                GgufWeight::Dense(t) => t,
                GgufWeight::Quantized(_) => panic!("q_norm should be dense, not quantized"),
            });
            let k_norm_weight = gguf_weights.take(&k_norm_name).map(|w| match w {
                GgufWeight::Dense(t) => t,
                GgufWeight::Quantized(_) => panic!("k_norm should be dense, not quantized"),
            });
            let qk_norm_eps = if q_norm_weight.is_some() {
                config.rms_norm_eps
            } else {
                0.0
            };

            let self_attn = LlamaAttention {
                qkv_proj: make_linear(q_w),
                k_proj: Some(make_linear(k_w)),
                v_proj: Some(make_linear(v_w)),
                o_proj: make_linear(o_w),
                q_size,
                kv_size,
                num_q_heads,
                num_kv_heads,
                head_dim,
                scale: 1.0 / (head_dim as f32).sqrt(),
                layer_idx: i,
                q_norm_weight,
                k_norm_weight,
                qk_norm_eps,
                #[cfg(feature = "nccl")]
                tp_group: None,
            };

            // MLP: separate gate, up, down.
            let gate_w = gguf_weights
                .take(&format!("{prefix}.mlp.gate_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.mlp.gate_proj.weight"))?;
            let up_w = gguf_weights
                .take(&format!("{prefix}.mlp.up_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.mlp.up_proj.weight"))?;
            let down_w = gguf_weights
                .take(&format!("{prefix}.mlp.down_proj.weight"))
                .ok_or_else(|| anyhow::anyhow!("missing {prefix}.mlp.down_proj.weight"))?;

            let mlp = LlamaMLP {
                gate_up_proj: make_linear(gate_w),
                up_proj: Some(make_linear(up_w)),
                down_proj: make_linear(down_w),
                intermediate_size: config.intermediate_size,
                #[cfg(feature = "nccl")]
                tp_group: None,
            };

            // Norms: dequantized to f32.
            let input_ln_w =
                gguf_weights.take_dense(&format!("{prefix}.input_layernorm.weight"))?;
            let post_ln_w =
                gguf_weights.take_dense(&format!("{prefix}.post_attention_layernorm.weight"))?;

            layers.push(LlamaDecoderLayer {
                self_attn,
                mlp,
                input_layernorm: RmsNorm::new(input_ln_w, config.rms_norm_eps),
                post_attention_layernorm: RmsNorm::new(post_ln_w, config.rms_norm_eps),
                residual_multiplier: 1.0,
            });
        }

        let norm_w = gguf_weights.take_dense("model.norm.weight")?;
        let norm = RmsNorm::new(norm_w, config.rms_norm_eps);

        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
                dtype,
                device,
            )?
        };

        let model = LlamaModel {
            embed_tokens,
            layers,
            norm,
            rotary,
            embedding_multiplier: 1.0,
        };

        // lm_head: check if GGUF has output.weight (→ lm_head.weight), else tie embeddings.
        let lm_head = if gguf_weights.contains("lm_head.weight") {
            let lm_w = gguf_weights.take_dense("lm_head.weight")?;
            LinearLayer::Dense(Linear::new(lm_w, None))
        } else {
            // Tied embeddings.
            LinearLayer::Dense(Linear::new(model.embed_tokens.weight, None))
        };

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: None,
        })
    }

    /// Load a BitsAndBytes 4-bit (NF4/FP4) quantized model.
    pub fn load_bnb4bit(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
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

        // Compute blocksize from the model config: BNB default is 64,
        // but the actual blocksize is stored in the quant_state metadata.
        // Use the config blocksize as default.
        let blocksize = qconfig.blocksize;

        // Compute max dequant buffer size across all linear layers (per-shard, no fusing).
        let hidden = config.hidden_size;
        let q_size = config.num_attention_heads * config.head_dim;
        let inter = config.intermediate_size;
        let max_elements = [
            q_size * hidden, // q_proj (largest attention proj)
            hidden * q_size, // o_proj
            inter * hidden,  // gate_proj or up_proj
            hidden * inter,  // down_proj
        ]
        .into_iter()
        .max()
        .unwrap();

        let dequant_scratch = gpu_weights::alloc_bnb_dequant_scratch(max_elements, dtype, stream)?;

        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let layer = LlamaDecoderLayer::load_bnb4bit(
                weights,
                &prefix,
                config,
                i,
                qconfig,
                code_gpu,
                dequant_scratch,
                blocksize,
                stream,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;
        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
                dtype,
                device,
            )?
        };
        // Track rotary cache GPU allocation for sleep/wake.
        weights.record_alloc(
            rotary.cos_sin_cache.raw_ptr(),
            rotary.cos_sin_cache.size_bytes(),
        );

        let model = LlamaModel {
            embed_tokens,
            layers,
            norm,
            rotary,
            embedding_multiplier: 1.0,
        };

        let lm_head = LinearLayer::Dense(if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        });

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: None,
        })
    }
}

impl LlamaDecoderLayer {
    /// Load a BNB 4-bit decoder layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_bnb4bit(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
        qconfig: &crate::quant::Bnb4bitConfig,
        code_gpu: GpuTensor,
        dequant_scratch: GpuTensor,
        blocksize: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let self_attn = LlamaAttention::load_bnb4bit(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            qconfig,
            code_gpu,
            dequant_scratch,
            blocksize,
            stream,
        )?;
        let mlp = LlamaMLP::load_bnb4bit(
            weights,
            &format!("{prefix}.mlp"),
            config,
            qconfig,
            code_gpu,
            dequant_scratch,
            blocksize,
            stream,
        )?;
        let input_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.input_layernorm"),
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = RmsNorm::load(
            weights,
            &format!("{prefix}.post_attention_layernorm"),
            config.rms_norm_eps,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            residual_multiplier: 1.0,
        })
    }
}

impl LlamaAttention {
    /// Load BNB 4-bit quantized attention — per-shard matmuls (matches Python vLLM exactly).
    #[allow(clippy::too_many_arguments)]
    pub fn load_bnb4bit(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        layer_idx: usize,
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

        // Per-shard: separate q, k, v projections (exactly what Python vLLM does).
        let q = gpu_weights::load_bnb4bit_linear(
            weights,
            &format!("{prefix}.q_proj"),
            qconfig,
            code_gpu,
            dequant_scratch,
            q_size,
            hidden,
            blocksize,
            stream,
        )?;
        let k = gpu_weights::load_bnb4bit_linear(
            weights,
            &format!("{prefix}.k_proj"),
            qconfig,
            code_gpu,
            dequant_scratch,
            kv_size,
            hidden,
            blocksize,
            stream,
        )?;
        let v = gpu_weights::load_bnb4bit_linear(
            weights,
            &format!("{prefix}.v_proj"),
            qconfig,
            code_gpu,
            dequant_scratch,
            kv_size,
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

        // Load QK-norm weights if present (Qwen3).
        let q_norm_name = format!("{prefix}.q_norm.weight");
        let k_norm_name = format!("{prefix}.k_norm.weight");
        let q_norm_weight = if weights.contains(&q_norm_name) {
            Some(weights.take(&q_norm_name)?)
        } else {
            None
        };
        let k_norm_weight = if weights.contains(&k_norm_name) {
            Some(weights.take(&k_norm_name)?)
        } else {
            None
        };

        Ok(Self {
            qkv_proj: LinearLayer::Bnb4bit(Box::new(q)),
            k_proj: Some(LinearLayer::Bnb4bit(Box::new(k))),
            v_proj: Some(LinearLayer::Bnb4bit(Box::new(v))),
            o_proj: LinearLayer::Bnb4bit(Box::new(o)),
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            layer_idx,
            q_norm_weight,
            k_norm_weight,
            qk_norm_eps: 1e-6,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

impl LlamaMLP {
    /// Load BNB 4-bit quantized MLP — per-shard matmuls (matches Python vLLM exactly).
    #[allow(clippy::too_many_arguments)]
    pub fn load_bnb4bit(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &LlamaConfig,
        qconfig: &crate::quant::Bnb4bitConfig,
        code_gpu: GpuTensor,
        dequant_scratch: GpuTensor,
        blocksize: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let inter = config.intermediate_size;
        let hidden = config.hidden_size;

        // Per-shard: separate gate and up projections (exactly what Python vLLM does).
        let gate = gpu_weights::load_bnb4bit_linear(
            weights,
            &format!("{prefix}.gate_proj"),
            qconfig,
            code_gpu,
            dequant_scratch,
            inter,
            hidden,
            blocksize,
            stream,
        )?;
        let up = gpu_weights::load_bnb4bit_linear(
            weights,
            &format!("{prefix}.up_proj"),
            qconfig,
            code_gpu,
            dequant_scratch,
            inter,
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
            gate_up_proj: LinearLayer::Bnb4bit(Box::new(gate)),
            up_proj: Some(LinearLayer::Bnb4bit(Box::new(up))),
            down_proj: LinearLayer::Bnb4bit(Box::new(down)),
            intermediate_size: inter,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }
}

// ---------------------------------------------------------------------------
// Pipeline Parallelism — PP-aware load + forward
// ---------------------------------------------------------------------------

impl LlamaModel {
    /// Load backbone with PP: only loads this stage's layers, embedding
    /// (first stage only), and norm (last stage only). Skips all other weights.
    pub fn load_pp(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        // Embedding: only first stage.
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

        // Only load layers in [start_layer, end_layer).
        // Use local index (0-based) for KV cache access, but absolute index for weight names.
        let mut layers = Vec::with_capacity(pp.num_layers());
        for i in pp.start_layer..pp.end_layer {
            let local_idx = i - pp.start_layer;
            let layer = LlamaDecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                config,
                local_idx,
                device.compute_stream,
            )?;
            layers.push(layer);
        }

        // Final norm: only last stage.
        let norm = if pp.is_last_stage() {
            RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?
        } else {
            // Dummy norm — never used on non-last stages.
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                weights.record_alloc(ptr, dtype.size_bytes());
                GpuTensor::new(ptr, &[1], dtype)
            };
            RmsNorm::new(w, config.rms_norm_eps)
        };

        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
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
            embedding_multiplier: 1.0,
        })
    }

    /// Load backbone with TP + PP sharding.
    pub fn load_tp_pp(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
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
            let layer = LlamaDecoderLayer::load_tp(
                weights,
                &format!("model.layers.{i}"),
                config,
                local_idx,
                tp,
                device.compute_stream,
            )?;
            layers.push(layer);
        }

        let norm = if pp.is_last_stage() {
            RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?
        } else {
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                weights.record_alloc(ptr, dtype.size_bytes());
                GpuTensor::new(ptr, &[1], dtype)
            };
            RmsNorm::new(w, config.rms_norm_eps)
        };

        let rotary = unsafe {
            RotaryCache::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.llama3_rope_scaling.as_ref(),
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
            embedding_multiplier: 1.0,
        })
    }

    /// PP-aware forward: handles embedding (first stage), layer subset,
    /// and final norm (last stage).
    ///
    /// - First stage: embeds `input_ids`, runs layers, returns (hs, residual)
    /// - Middle stages: takes (hs, residual), runs layers, returns (hs, residual)
    /// - Last stage: takes (hs, residual), runs layers + final norm, returns hidden_states
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_pp(
        &self,
        pp: &PpConfig,
        // First stage only — input token IDs.
        input_ids: Option<TensorView<'_>>,
        // Non-first stages — received from previous stage.
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
                // First stage: embedding lookup.
                let input_ids = input_ids.expect("first PP stage requires input_ids");
                let hs = kernels::embedding_gather(
                    self.embed_tokens.weight,
                    *input_ids,
                    &mut device.caching,
                    device.compute_stream,
                );
                if self.embedding_multiplier != 1.0 {
                    kernels::scale_inplace(
                        hs.as_gpu_tensor(),
                        self.embedding_multiplier,
                        &device.cublas,
                    );
                }
                (hs, None)
            } else {
                // Non-first stage: use received intermediate tensors.
                let (hs, res) = intermediate.expect("non-first PP stage requires intermediate");
                (hs, Some(res))
            };

        // Run this stage's layers.
        for layer in self.layers.iter() {
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
            // Last stage: final norm → return hidden_states for lm_head.
            let hs_gpu = *hidden_states;
            let res_gpu = residual.as_ref().unwrap().as_gpu_tensor();
            kernels::fused_add_rms_norm_inplace(
                hs_gpu,
                res_gpu,
                self.norm.weight,
                self.norm.eps,
                device.compute_stream,
            );
            drop(residual);
            ForwardOutput::Logits(hidden_states)
        } else {
            // Non-last stage: pass hidden_states + residual to next stage.
            ForwardOutput::Intermediate {
                hidden_states,
                residual: residual.unwrap(),
            }
        }
    }
}

impl LlamaForCausalLM {
    /// Load with PP (no TP).
    pub fn load_pp(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaModel::load_pp(weights, config, dtype, pp, device)?;

        let lm_head = if pp.is_last_stage() {
            if config.tie_word_embeddings {
                // For PP last stage with tied embeddings, we need to load the
                // embedding weight separately for lm_head since embed_tokens
                // might be a dummy on this stage.
                let embed_w = weights.take("model.embed_tokens.weight")?;
                LinearLayer::Dense(Linear::new(embed_w, None))
            } else {
                LinearLayer::Dense(Linear::load(weights, "lm_head")?)
            }
        } else {
            // Dummy lm_head — never used on non-last stages.
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                weights.record_alloc(ptr, dtype.size_bytes());
                GpuTensor::new(ptr, &[1, 1], dtype)
            };
            LinearLayer::Dense(Linear::new(w, None))
        };

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
            pp_config: Some(pp),
        })
    }

    /// Load with TP + PP.
    pub fn load_tp_pp(
        weights: &mut GpuWeights,
        config: &LlamaConfig,
        dtype: DType,
        tp: TpConfig,
        pp: PpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = LlamaModel::load_tp_pp(weights, config, dtype, tp, pp, device)?;

        let lm_head = if pp.is_last_stage() {
            if config.tie_word_embeddings {
                LinearLayer::Dense(Linear::new(model.embed_tokens.weight, None))
            } else {
                let lm_w = weights.take_shard("lm_head.weight", 0, tp.rank, tp.world_size)?;
                LinearLayer::Dense(Linear::new(lm_w, None))
            }
        } else {
            let w = unsafe {
                let ptr = crate::driver::mem_alloc(dtype.size_bytes())?;
                weights.record_alloc(ptr, dtype.size_bytes());
                GpuTensor::new(ptr, &[1, 1], dtype)
            };
            LinearLayer::Dense(Linear::new(w, None))
        };

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
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

        // Run backbone with PP routing.
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
                // Last stage: lm_head projection.
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
                let mut logits = self.lm_head.forward(
                    hs_view,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                // hidden_states can be freed now.
                drop(gathered);
                drop(hidden_states);

                // TP: all-gather logits.
                #[cfg(feature = "nccl")]
                if let Some(ref group) = self.tp_group {
                    let gathered = group.all_gather(logits.as_gpu_tensor(), &mut device.caching);
                    drop(logits);
                    logits = gathered;
                }

                if self.logits_scaling != 1.0 {
                    kernels::scale_inplace(
                        logits.as_gpu_tensor(),
                        self.logits_scaling.recip(),
                        &device.cublas,
                    );
                }

                ForwardOutput::Logits(logits)
            }
        }
    }
}
