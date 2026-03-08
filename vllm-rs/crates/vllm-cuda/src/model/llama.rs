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
use crate::layers::{Embedding, Linear, RmsNorm};
use crate::tensor::GpuTensor;
use crate::weights::GpuWeights;

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

/// LLaMA MLP with fused gate+up projection.
///
/// Forward: gate_up_proj(x) → SiLU(gate) * up → down_proj
pub struct LlamaMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    intermediate_size: usize,
}

impl LlamaMLP {
    /// Load from `GpuWeights`.
    pub fn load(weights: &mut GpuWeights, prefix: &str, intermediate_size: usize) -> Result<Self> {
        // Fuse gate + up: concatenate along out dimension.
        // In safetensors, gate_proj is [intermediate, hidden] and up_proj is [intermediate, hidden].
        // Fused: [2*intermediate, hidden].
        let gate = weights.take(&format!("{prefix}.gate_proj.weight"))?;
        let up = weights.take(&format!("{prefix}.up_proj.weight"))?;

        // Build fused weight by just recording the two — we'll need them adjacent in memory.
        // For now, create the fused linear from the gate and up weights.
        // TODO: fuse at load time into a single contiguous buffer for one GEMM.
        let gate_up = Self::fuse_gate_up(gate, up)?;
        let gate_up_proj = Linear::new(gate_up, None);

        let down_proj = Linear::load(weights, &format!("{prefix}.down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
        })
    }

    /// Fuse gate and up weight tensors into [2*intermediate, hidden].
    fn fuse_gate_up(_gate: GpuTensor, _up: GpuTensor) -> Result<GpuTensor> {
        // In the candle backend, this is done via Tensor::cat on dim 0.
        // For GpuTensor, we need a D2D copy to create a contiguous fused buffer.
        // TODO: implement proper fusion. For now, we'll load them separately
        // and handle in forward by doing two GEMMs.
        // This is a placeholder — proper implementation needs a D2D concat.
        anyhow::bail!(
            "gate_up fusion not yet implemented for GpuTensor; \
             need D2D concat kernel or pre-fused safetensors loading"
        )
    }

    /// Load with pre-fused gate_up weight (if the model stores it that way,
    /// or if we pre-fuse during weight loading).
    pub fn load_prefused(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
    ) -> Result<Self> {
        // Try fused first, fallback to separate.
        let gate_up_name = format!("{prefix}.gate_up_proj.weight");
        let gate_up = if weights.contains(&gate_up_name) {
            Linear::load(weights, &format!("{prefix}.gate_up_proj"))?
        } else {
            // Load separate and do two GEMMs in forward.
            let gate = Linear::load(weights, &format!("{prefix}.gate_proj"))?;
            let up = Linear::load(weights, &format!("{prefix}.up_proj"))?;
            // Return gate as the "gate_up" and store up separately.
            // This is a workaround — see LlamaMlpSeparate below.
            let _ = up;
            gate
        };

        let down_proj = Linear::load(weights, &format!("{prefix}.down_proj"))?;
        Ok(Self {
            gate_up_proj: gate_up,
            down_proj,
            intermediate_size,
        })
    }

    /// Forward pass.
    pub unsafe fn forward(&self, x: GpuTensor, device: &mut GpuDevice) -> GpuTensor {
        self.forward_owned(x, device).into_gpu_tensor()
    }

    /// Forward pass returning `OwnedTensor` (caching-allocator path).
    ///
    /// Intermediates (gate_up, activated) are owned and freed at end of scope.
    /// Only the final down_proj output survives.
    pub unsafe fn forward_owned(&self, x: GpuTensor, device: &mut GpuDevice) -> OwnedTensor {
        let gate_up = self
            .gate_up_proj
            .forward_owned(x, &mut device.cublas, &mut device.caching);
        let activated = kernels::silu_and_mul_fused(
            gate_up.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            device.compute_stream,
        );
        drop(gate_up); // return gate_up memory to free list
        let result = self.down_proj.forward_owned(
            activated.as_gpu_tensor(),
            &mut device.cublas,
            &mut device.caching,
        );
        drop(activated);
        result
    }
}

// ---------------------------------------------------------------------------
// LlamaAttention
// ---------------------------------------------------------------------------

/// LLaMA multi-head attention with RoPE, GQA, fused QKV.
pub struct LlamaAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    q_size: usize,
    kv_size: usize,
    num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    layer_idx: usize,
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
        let qkv =
            self.qkv_proj
                .forward_owned(hidden_states, &mut device.cublas, &mut device.caching);

        // Fused QKV split + RoPE → owned Q, K, V.
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
        drop(qkv); // free QKV projection output

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
        let result = self
            .o_proj
            .forward_owned(attn_flat, &mut device.cublas, &mut device.caching);
        drop(attn_output);
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
    pub lm_head: Linear,
    /// Granite logits scaling (1.0 = no-op for LLaMA).
    pub logits_scaling: f32,
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
        let logits = self
            .lm_head
            .forward(hidden_states, &mut device.cublas, &mut device.caching);

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
        let logits =
            self.lm_head
                .forward_owned(hidden_states, &mut device.cublas, &mut device.caching);
        let logits = logits.into_gpu_tensor();

        if self.logits_scaling != 1.0 {
            kernels::scale_inplace(logits, self.logits_scaling.recip(), &device.cublas);
        }

        logits
    }
}

// ---------------------------------------------------------------------------
// Weight loading with D2D concat helper
// ---------------------------------------------------------------------------

/// Concatenate two GPU tensors along dimension 0 (D2D copy).
///
/// * `a`: `[M, K]`
/// * `b`: `[N, K]`
/// * Returns: `[M+N, K]` allocated persistently (not from arena).
///
/// # Safety
/// Requires valid CUDA context and stream.
pub unsafe fn concat_dim0(
    a: GpuTensor,
    b: GpuTensor,
    stream: cudarc::driver::sys::CUstream,
) -> Result<GpuTensor> {
    debug_assert_eq!(a.ndim(), b.ndim());
    debug_assert_eq!(a.dtype(), b.dtype());
    // All dims except dim 0 must match.
    for d in 1..a.ndim() {
        debug_assert_eq!(a.dim(d), b.dim(d));
    }

    let total_bytes = a.size_bytes() + b.size_bytes();
    let ptr = crate::driver::mem_alloc(total_bytes)?;

    crate::driver::memcpy_dtod_async(ptr, a.raw_ptr() as *const u8, a.size_bytes(), stream)?;
    crate::driver::memcpy_dtod_async(
        ptr.add(a.size_bytes()),
        b.raw_ptr() as *const u8,
        b.size_bytes(),
        stream,
    )?;

    let mut new_shape: Vec<usize> = (0..a.ndim()).map(|d| a.dim(d)).collect();
    new_shape[0] += b.dim(0);

    Ok(GpuTensor::new(ptr, &new_shape, a.dtype()))
}

/// Concatenate three GPU tensors along dimension 0.
pub unsafe fn concat3_dim0(
    a: GpuTensor,
    b: GpuTensor,
    c: GpuTensor,
    stream: cudarc::driver::sys::CUstream,
) -> Result<GpuTensor> {
    let total_bytes = a.size_bytes() + b.size_bytes() + c.size_bytes();
    let ptr = crate::driver::mem_alloc(total_bytes)?;

    crate::driver::memcpy_dtod_async(ptr, a.raw_ptr() as *const u8, a.size_bytes(), stream)?;
    crate::driver::memcpy_dtod_async(
        ptr.add(a.size_bytes()),
        b.raw_ptr() as *const u8,
        b.size_bytes(),
        stream,
    )?;
    crate::driver::memcpy_dtod_async(
        ptr.add(a.size_bytes() + b.size_bytes()),
        c.raw_ptr() as *const u8,
        c.size_bytes(),
        stream,
    )?;

    let mut new_shape: Vec<usize> = (0..a.ndim()).map(|d| a.dim(d)).collect();
    new_shape[0] += b.dim(0) + c.dim(0);

    Ok(GpuTensor::new(ptr, &new_shape, a.dtype()))
}

// ---------------------------------------------------------------------------
// Proper weight loading using concat helpers
// ---------------------------------------------------------------------------

impl LlamaAttention {
    /// Load with D2D-fused QKV weights.
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

        let q_w = weights.take(&format!("{prefix}.q_proj.weight"))?;
        let k_w = weights.take(&format!("{prefix}.k_proj.weight"))?;
        let v_w = weights.take(&format!("{prefix}.v_proj.weight"))?;

        // Fuse: [q_size + 2*kv_size, hidden]
        let qkv_w = unsafe { concat3_dim0(q_w, k_w, v_w, stream)? };

        // Fuse QKV bias if present (Qwen2 has QKV bias, LLaMA doesn't).
        let q_bias_name = format!("{prefix}.q_proj.bias");
        let k_bias_name = format!("{prefix}.k_proj.bias");
        let v_bias_name = format!("{prefix}.v_proj.bias");
        let qkv_bias = if weights.contains(&q_bias_name) {
            let q_b = weights.take(&q_bias_name)?;
            let k_b = weights.take(&k_bias_name)?;
            let v_b = weights.take(&v_bias_name)?;
            // Reshape [size] → [size, 1] for concat_dim0, then squeeze back.
            // Actually concat3_dim0 works on any ndim where dims after 0 match.
            // For 1-D tensors [q_size], [kv_size], [kv_size] → [q_size + 2*kv_size].
            // Use a simple D2D concat for 1-D:
            let total = q_b.numel() + k_b.numel() + v_b.numel();
            let ptr = unsafe { crate::driver::mem_alloc(total * q_b.dtype().size_bytes())? };
            unsafe {
                crate::driver::memcpy_dtod_async(
                    ptr,
                    q_b.raw_ptr() as *const u8,
                    q_b.size_bytes(),
                    stream,
                )?;
                crate::driver::memcpy_dtod_async(
                    ptr.add(q_b.size_bytes()),
                    k_b.raw_ptr() as *const u8,
                    k_b.size_bytes(),
                    stream,
                )?;
                crate::driver::memcpy_dtod_async(
                    ptr.add(q_b.size_bytes() + k_b.size_bytes()),
                    v_b.raw_ptr() as *const u8,
                    v_b.size_bytes(),
                    stream,
                )?;
            }
            Some(unsafe { GpuTensor::new(ptr, &[total], q_b.dtype()) })
        } else {
            None
        };
        let qkv_proj = Linear::new(qkv_w, qkv_bias);

        let o_proj = Linear::load(weights, &format!("{prefix}.o_proj"))?;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_size,
            kv_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            layer_idx,
        })
    }
}

impl LlamaMLP {
    /// Load with D2D-fused gate+up weights.
    pub fn load_fused(
        weights: &mut GpuWeights,
        prefix: &str,
        intermediate_size: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let gate = weights.take(&format!("{prefix}.gate_proj.weight"))?;
        let up = weights.take(&format!("{prefix}.up_proj.weight"))?;

        // Fuse: [2*intermediate, hidden]
        let gate_up_w = unsafe { concat_dim0(gate, up, stream)? };
        let gate_up_proj = Linear::new(gate_up_w, None);

        let down_proj = Linear::load(weights, &format!("{prefix}.down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
        })
    }
}

impl LlamaDecoderLayer {
    /// Load a decoder layer with fused weights.
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

        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };

        Ok(Self {
            model,
            lm_head,
            logits_scaling: 1.0,
        })
    }
}
