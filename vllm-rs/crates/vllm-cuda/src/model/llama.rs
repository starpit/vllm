// SPDX-License-Identifier: Apache-2.0
//! LLaMA model using `GpuTensor` — zero-allocation forward pass.
//!
//! All intermediate tensors are arena-allocated. No candle dependency.
//! The same CUDA kernels from vllm-kernels are called via raw FFI with
//! `GpuTensor::as_ptr()` — one line per pointer extraction instead of ten.
//!
//! Port of the candle-based `LlamaForCausalLM` in `vllm-models/src/llama.rs`.

use anyhow::Result;

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::layers::{Embedding, Linear, LinearLayer, RmsNorm};
use crate::quant::QuantConfig;
use crate::tensor::GpuTensor;
use crate::weights::GpuWeights;
use crate::weights::{self as gpu_weights};

#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Parsed LLaMA config (mirrors the candle version but no candle types).
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
    /// `[max_pos, rotary_dim]` combined cos|sin cache.
    pub cos_sin_cache: GpuTensor,
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
        Ok(Self {
            cos_sin_cache,
            head_dim,
        })
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
    gate_up_proj: LinearLayer,
    /// Separate up projection — only used for quantized (None for dense).
    up_proj: Option<LinearLayer>,
    down_proj: LinearLayer,
    intermediate_size: usize,
    /// NCCL group for TP all-reduce after down_proj (row parallel).
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

impl LlamaMLP {
    /// Forward pass.
    pub unsafe fn forward(&self, x: GpuTensor, device: &mut GpuDevice) -> GpuTensor {
        self.forward_owned(x, device).into_gpu_tensor()
    }

    /// Forward pass returning `OwnedTensor` (caching-allocator path).
    pub unsafe fn forward_owned(&self, x: GpuTensor, device: &mut GpuDevice) -> OwnedTensor {
        let gate_up = if let Some(ref up_proj) = self.up_proj {
            // Quantized: separate gate + up GEMMs, then concat
            let gate_out = self.gate_up_proj.forward_owned(
                x,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            let up_out = up_proj.forward_owned(
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
            self.gate_up_proj.forward_owned(
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

    /// Forward pass with paged KV cache and FlashAttention-2.
    ///
    /// * `hidden_states`: `[num_tokens, hidden_size]`
    /// * `positions`: `[num_tokens]` (U32)
    /// * `slot_mapping`: `[num_tokens]` (I64) — absolute slot index for each new token
    /// * `cu_seqlens_q`: `[batch_size + 1]` (U32) — cumulative Q lengths
    /// * `seqused_k`: `[batch_size]` (U32) — per-sequence K lengths (including cached)
    /// * `block_table`: `[batch_size, max_blocks_per_seq]` (U32) — page table
    /// * `max_seqlen_q` / `max_seqlen_k`: max sequence lengths in batch
    /// * `kv_cache`: the paged KV cache pool
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
        self.forward_owned(
            hidden_states,
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
        .into_gpu_tensor()
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
        let qkv = if let (Some(k_proj), Some(v_proj)) = (self.k_proj.as_ref(), self.v_proj.as_ref())
        {
            // Quantized: separate Q, K, V GEMMs → concat
            let q_out = self.qkv_proj.forward_owned(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            let k_out = k_proj.forward_owned(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            let v_out = v_proj.forward_owned(
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
            self.qkv_proj.forward_owned(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            )
        };

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
                kernels::qk_norm_rope_inplace(
                    q.as_gpu_tensor(),
                    k.as_gpu_tensor(),
                    q_norm_w,
                    k_norm_w,
                    rotary.cos_sin_cache,
                    positions,
                    self.num_q_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.qk_norm_eps,
                    device.compute_stream,
                );
                (q, k, v)
            } else {
                // Standard path: fused QKV split + RoPE.
                let result = kernels::fused_qkv_rope(
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
                result
            };

        // Write new K/V into paged cache (reads from k, v).
        kernels::reshape_and_cache(
            k.as_gpu_tensor(),
            v.as_gpu_tensor(),
            kv_cache.k_cache(self.layer_idx),
            kv_cache.v_cache(self.layer_idx),
            slot_mapping,
            kv_cache.block_size,
            device.compute_stream,
        );

        let fresh_prefill = max_seqlen_q > 1 && max_seqlen_q == max_seqlen_k;
        let attn_output = if fresh_prefill {
            kernels::flash_attn_contiguous(
                q.as_gpu_tensor(),
                k.as_gpu_tensor(),
                v.as_gpu_tensor(),
                cu_seqlens_q,
                cu_seqlens_q,
                max_seqlen_q,
                max_seqlen_k,
                self.scale,
                true,
                0.0,
                -1,
                &mut device.caching,
                device.compute_stream,
            )
        } else {
            // K, V no longer needed after reshape_and_cache (data is in the cache).
            drop(k);
            drop(v);
            kernels::flash_attn_paged(
                q.as_gpu_tensor(),
                kv_cache.k_cache(self.layer_idx),
                kv_cache.v_cache(self.layer_idx),
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_seqlen_q,
                max_seqlen_k,
                self.scale,
                true,
                kv_cache.block_size,
                device.num_sm,
                &mut device.caching,
                device.compute_stream,
            )
        };
        drop(q); // free Q

        // Reshape to [num_tokens, q_size] and output projection.
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
}

// ---------------------------------------------------------------------------
// LlamaDecoderLayer
// ---------------------------------------------------------------------------

/// A single LLaMA decoder layer.
pub struct LlamaDecoderLayer {
    pub self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
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
        // Pre-attention norm with fused residual add (in-place).
        let (normed, residual) = if let Some(residual) = residual {
            // fused_add_rms_norm_inplace mutates both in-place:
            //   hidden_states buffer → normed values
            //   residual buffer → residual += old_hidden_states
            // After this, hidden_states OwnedTensor still owns its buffer (now normed).
            // residual OwnedTensor still owns its buffer (updated).
            let hs_gpu = *hidden_states; // GpuTensor copy via deref
            let res_gpu = *residual; // GpuTensor copy via deref
            kernels::fused_add_rms_norm_inplace(
                hs_gpu,
                res_gpu,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
                device.compute_stream,
            );
            // hidden_states buffer now contains normed values.
            // residual buffer is updated. Both OwnedTensors keep ownership.
            (hidden_states, residual)
        } else {
            // First layer: allocate NEW buffer for normed. hidden_states becomes residual.
            let normed = kernels::rms_norm(
                *hidden_states,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
                &mut device.caching,
                device.compute_stream,
            );
            (normed, hidden_states) // hidden_states ownership transfers to residual
        };

        // Attention reads normed values from hidden_states buffer.
        let attn_output = self.self_attn.forward_owned(
            *normed, // GpuTensor copy — kernel reads from this buffer
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
        // normed (= hidden_states buffer) consumed by QKV projection — free it.
        drop(normed);

        // Granite: scale attention output.
        if self.residual_multiplier != 1.0 {
            kernels::scale_inplace(*attn_output, self.residual_multiplier, &device.cublas);
        }

        // Post-attention norm: mutates attn_output buffer → post-normed,
        // updates residual buffer.
        let res_gpu = *residual;
        kernels::fused_add_rms_norm_inplace(
            *attn_output,
            res_gpu,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
            device.compute_stream,
        );
        // attn_output buffer now contains post-normed values.

        // MLP reads post-normed from attn_output buffer.
        let mlp_output = self.mlp.forward_owned(*attn_output, device);
        // attn_output consumed by MLP — free it.
        drop(attn_output);

        // Granite: scale MLP output.
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
    /// Forward pass with paged attention.
    ///
    /// * `input_ids`: `[num_tokens]` (U32 on GPU)
    /// * `positions`: `[num_tokens]` (U32 on GPU)
    /// * `slot_mapping`: `[num_tokens]` (I64) — slot indices for new tokens
    /// * `cu_seqlens_q`: `[batch_size + 1]` (U32) — cumulative Q lengths
    /// * `seqused_k`: `[batch_size]` (U32) — per-sequence K lengths
    /// * `block_table`: `[batch_size, max_blocks_per_seq]` (U32)
    /// * `kv_cache`: paged KV cache pool
    ///
    /// Returns hidden states `[num_tokens, hidden_size]`.
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
        // Delegates to forward_owned (caching allocator, zero D2D copies).
        self.forward_owned(
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
        )
    }

    /// Forward pass using caching allocator — zero D2D copies between layers.
    ///
    /// This matches Python's PyTorch flow: intermediates are freed on drop,
    /// `fused_add_rms_norm` mutates in-place, and only hidden_states + residual
    /// survive between layers.
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
        // Embedding lookup — owned, survives into first layer.
        let hidden_states = kernels::embedding_gather(
            self.embed_tokens.weight,
            input_ids,
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
        hidden_states.into_gpu_tensor()
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
            kernels::embedding_gather(
                hidden_states,
                indices,
                &mut device.caching,
                device.compute_stream,
            )
            .into_gpu_tensor()
        } else {
            hidden_states
        };
        #[allow(unused_mut)]
        let mut logits = self.lm_head.forward(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );

        // TP: all-gather logits (column parallel lm_head).
        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.tp_group {
            logits = group.all_gather(logits, &mut device.caching);
        }

        // Granite: scale logits by 1/logits_scaling.
        if self.logits_scaling != 1.0 {
            kernels::scale_inplace(logits, 1.0 / self.logits_scaling, &device.cublas);
        }

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

        // Gather last-token hidden states.
        let hidden_states = if let Some(indices) = last_token_indices {
            let gathered = kernels::embedding_gather(
                hidden_states,
                indices,
                &mut device.caching,
                device.compute_stream,
            );
            gathered.into_gpu_tensor()
        } else {
            hidden_states
        };

        // lm_head: logits = hidden_states @ lm_head_weight^T
        let logits = self.lm_head.forward_owned(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );
        #[allow(unused_mut)]
        let mut logits = logits.into_gpu_tensor();

        // TP: all-gather logits (column parallel lm_head — each rank has vocab shard).
        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.tp_group {
            logits = group.all_gather(logits, &mut device.caching);
        }

        if self.logits_scaling != 1.0 {
            kernels::scale_inplace(logits, self.logits_scaling.recip(), &device.cublas);
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
            unsafe {
                weights.take_into(&q_bias_name, bias_ptr, stream)?;
                weights.take_into(&k_bias_name, bias_ptr.add(q_b_bytes), stream)?;
                weights.take_into(&v_bias_name, bias_ptr.add(q_b_bytes + k_b_bytes), stream)?;
            }
            Some(unsafe { GpuTensor::new(bias_ptr, &[total_elems], q_b_dtype) })
        } else {
            None
        };
        let qkv_proj = LinearLayer::Dense(Linear::new(qkv_w, qkv_bias));

        let o_proj = LinearLayer::Dense(Linear::load(weights, &format!("{prefix}.o_proj"))?);

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
            up_proj: None,
            down_proj,
            intermediate_size,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
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
        let qkv_proj = LinearLayer::Dense(Linear::new(qkv_w, qkv_bias));

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
        })
    }
}
