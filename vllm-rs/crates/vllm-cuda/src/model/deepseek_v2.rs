// SPDX-License-Identifier: Apache-2.0
//! DeepSeek V2/V3 model using `GpuTensor` — zero-allocation forward pass.
//!
//! Key architectural differences from LLaMA:
//! - **MLA (Multi-Latent Attention)**: Low-rank Q/KV projections with partial RoPE.
//!   Q: hidden → q_lora_rank → q_a_layernorm → q_b_proj → split(nope|pe) → RoPE on pe
//!   KV: hidden → kv_a_proj_with_mqa → split(latent|k_pe) → layernorm → kv_b_proj → split(k_nope|v)
//!   Cache full K=[k_nope|k_pe_broadcast] and V (zero-padded to qk_head_dim).
//! - **MoE with shared experts**: Top-k routed experts + unconditional shared expert.
//!   No sigmoid gate on shared expert (unlike Qwen2/3 MoE). Uses `routed_scaling_factor`.
//! - **YaRN RoPE**: Frequency-dependent interpolation with mscale correction.
//! - **Interleaved RoPE**: is_neox_style=False (pairs at [2i, 2i+1]).
//!
//! Non-absorbed MLA path (matches Python `DeepseekV2Attention.forward`).

use anyhow::Result;

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::driver;
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::layers::{Embedding, Linear, RmsNorm};
use crate::layers_moe::{Fp8FusedMoELayer, FusedMoELayer};
use crate::model::llama::{LlamaMLP, RotaryCache, TpConfig};
use crate::tensor::GpuTensor;
use crate::weights as gpu_weights;
use crate::weights::GpuWeights;

#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
#[cfg(feature = "nccl")]
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct YarnRopeScaling {
    pub factor: f64,
    pub mscale: f64,
    pub mscale_all_dim: f64,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

#[derive(Debug, Clone)]
pub struct DeepSeekV2Config {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    // MLA fields
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    // MoE fields
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub first_k_dense_replace: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    // YaRN RoPE
    pub yarn_rope_scaling: Option<YarnRopeScaling>,
}

impl DeepSeekV2Config {
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

// ---------------------------------------------------------------------------
// YaRN RotaryCache
// ---------------------------------------------------------------------------

fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// Find the YaRN correction range frequencies.
fn yarn_find_correction_range(
    beta_fast: f64,
    beta_slow: f64,
    dim: usize,
    base: f64,
    original_max_pos: usize,
) -> (f64, f64) {
    let n_orig = original_max_pos as f64;
    let low =
        (n_orig / (beta_fast * 2.0 * std::f64::consts::PI)).ln() / (2.0 / dim as f64 * base.ln());
    let high =
        (n_orig / (beta_slow * 2.0 * std::f64::consts::PI)).ln() / (2.0 / dim as f64 * base.ln());
    (
        low.floor().max(0.0),
        high.ceil().min(dim as f64 / 2.0 - 1.0),
    )
}

/// Linear ramp mask for YaRN correction.
fn yarn_linear_ramp_mask(low: f64, high: f64, dim: usize) -> Vec<f64> {
    let len = dim / 2;
    (0..len)
        .map(|i| {
            let t = i as f64;
            if t < low {
                0.0 // interpolated
            } else if t > high {
                1.0 // original
            } else {
                (t - low) / (high - low) // blend
            }
        })
        .collect()
}

/// Build a YaRN RoPE cos/sin cache on GPU.
///
/// Matches Python vLLM's `DeepseekScalingRotaryEmbedding` / `yarn_get_mscale`.
pub unsafe fn yarn_rotary_cache(
    rope_dim: usize, // qk_rope_head_dim
    max_pos: usize,
    rope_theta: f64,
    scaling: &YarnRopeScaling,
    dtype: DType,
    device: &GpuDevice,
) -> Result<RotaryCache> {
    let half = rope_dim / 2;
    let factor = scaling.factor;
    let (low, high) = yarn_find_correction_range(
        scaling.beta_fast,
        scaling.beta_slow,
        rope_dim,
        rope_theta,
        scaling.original_max_position_embeddings,
    );
    let ramp = yarn_linear_ramp_mask(low, high, rope_dim);

    // Compute inv frequencies with YaRN interpolation.
    let inv_freqs: Vec<f64> = (0..half)
        .map(|i| {
            let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / rope_dim as f64);
            let freq_inter = freq / factor;
            // Blend: ramp[i]=0 → interpolated, ramp[i]=1 → original
            freq_inter * (1.0 - ramp[i]) + freq * ramp[i]
        })
        .collect();

    // Build cos/sin cache on CPU.
    let mut cache = vec![0f32; max_pos * rope_dim];
    for pos in 0..max_pos {
        for i in 0..half {
            let angle = pos as f64 * inv_freqs[i];
            cache[pos * rope_dim + i] = angle.cos() as f32;
            cache[pos * rope_dim + half + i] = angle.sin() as f32;
        }
    }

    let nbytes = max_pos * rope_dim * dtype.size_bytes();
    let gpu_ptr = driver::mem_alloc(nbytes)?;

    // Convert to target dtype and upload.
    match dtype {
        DType::F32 => {
            let host = driver::mem_alloc_host(nbytes)?;
            std::ptr::copy_nonoverlapping(cache.as_ptr() as *const u8, host, nbytes);
            driver::memcpy_htod_async(gpu_ptr, host, nbytes, device.compute_stream)?;
            driver::stream_synchronize(device.compute_stream)?;
            driver::mem_free_host(host)?;
        }
        DType::F16 => {
            let f16_data: Vec<half::f16> = cache.iter().map(|&v| half::f16::from_f32(v)).collect();
            let host = driver::mem_alloc_host(nbytes)?;
            std::ptr::copy_nonoverlapping(f16_data.as_ptr() as *const u8, host, nbytes);
            driver::memcpy_htod_async(gpu_ptr, host, nbytes, device.compute_stream)?;
            driver::stream_synchronize(device.compute_stream)?;
            driver::mem_free_host(host)?;
        }
        DType::BF16 => {
            let bf16_data: Vec<half::bf16> =
                cache.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let host = driver::mem_alloc_host(nbytes)?;
            std::ptr::copy_nonoverlapping(bf16_data.as_ptr() as *const u8, host, nbytes);
            driver::memcpy_htod_async(gpu_ptr, host, nbytes, device.compute_stream)?;
            driver::stream_synchronize(device.compute_stream)?;
            driver::mem_free_host(host)?;
        }
        _ => anyhow::bail!("unsupported dtype for YaRN RoPE cache: {:?}", dtype),
    }

    let cos_sin_cache = GpuTensor::new(gpu_ptr, &[max_pos, rope_dim], dtype);
    Ok(RotaryCache {
        cos_sin_cache,
        head_dim: rope_dim,
    })
}

// ---------------------------------------------------------------------------
// DeepSeekV2Attention — MLA
// ---------------------------------------------------------------------------

/// MLA attention: low-rank Q/KV projections with partial RoPE.
pub struct DeepSeekV2Attention {
    // Q path
    q_a_proj: Option<Linear>,       // hidden → q_lora_rank (Replicated)
    q_a_layernorm: Option<RmsNorm>, // RmsNorm on q_lora_rank
    q_b_proj: Linear,               // q_lora_rank → num_heads * qk_head_dim (ColumnParallel)
    // or direct q_proj: hidden → num_heads * qk_head_dim (if no q_lora_rank)

    // KV path
    kv_a_proj_with_mqa: Linear, // hidden → kv_lora_rank + rope_dim (Replicated)
    kv_a_layernorm: RmsNorm,    // RmsNorm on kv_lora_rank
    kv_b_proj: Linear, // kv_lora_rank → num_heads * (nope_dim + v_head_dim) (ColumnParallel)

    // Output
    o_proj: Linear, // num_heads * v_head_dim → hidden (RowParallel)

    // Dims
    pub num_heads: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_lora_rank: usize,
    pub scale: f32,
    pub layer_idx: usize,

    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

impl DeepSeekV2Attention {
    /// Load MLA attention weights (single GPU).
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &DeepSeekV2Config,
        layer_idx: usize,
    ) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let qk_nope_head_dim = config.qk_nope_head_dim;
        let qk_rope_head_dim = config.qk_rope_head_dim;
        let qk_head_dim = config.qk_head_dim();
        let v_head_dim = config.v_head_dim;
        let kv_lora_rank = config.kv_lora_rank;

        // Q path
        let (q_a_proj, q_a_layernorm, q_b_proj) = if let Some(q_lora_rank) = config.q_lora_rank {
            let q_a = Linear::load(weights, &format!("{prefix}.q_a_proj"))?;
            let q_a_ln = RmsNorm::load(
                weights,
                &format!("{prefix}.q_a_layernorm"),
                config.rms_norm_eps,
            )?;
            let q_b = Linear::load(weights, &format!("{prefix}.q_b_proj"))?;
            let _ = q_lora_rank; // used for config validation
            (Some(q_a), Some(q_a_ln), q_b)
        } else {
            let q_proj = Linear::load(weights, &format!("{prefix}.q_proj"))?;
            (None, None, q_proj)
        };

        // KV path
        let kv_a = Linear::load(weights, &format!("{prefix}.kv_a_proj_with_mqa"))?;
        let kv_a_ln = RmsNorm::load(
            weights,
            &format!("{prefix}.kv_a_layernorm"),
            config.rms_norm_eps,
        )?;
        let kv_b = Linear::load(weights, &format!("{prefix}.kv_b_proj"))?;

        // Output
        let o_proj = Linear::load(weights, &format!("{prefix}.o_proj"))?;

        // Attention scaling
        let mut scale = 1.0 / (qk_head_dim as f32).sqrt();
        if let Some(ref yarn) = config.yarn_rope_scaling {
            let mscale = yarn_get_mscale(yarn.factor, yarn.mscale_all_dim);
            scale *= (mscale * mscale) as f32;
        }

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa: kv_a,
            kv_a_layernorm: kv_a_ln,
            kv_b_proj: kv_b,
            o_proj,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_lora_rank,
            scale,
            layer_idx,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Load with TP sharding.
    pub fn load_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &DeepSeekV2Config,
        layer_idx: usize,
        tp: TpConfig,
        _stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let total_heads = config.num_attention_heads;
        let num_heads = total_heads / tp.world_size;
        let qk_nope_head_dim = config.qk_nope_head_dim;
        let qk_rope_head_dim = config.qk_rope_head_dim;
        let qk_head_dim = config.qk_head_dim();
        let v_head_dim = config.v_head_dim;
        let kv_lora_rank = config.kv_lora_rank;

        // Q path: q_a_proj is Replicated, q_b_proj is ColumnParallel (shard dim 0)
        let (q_a_proj, q_a_layernorm, q_b_proj) = if config.q_lora_rank.is_some() {
            let q_a = Linear::load(weights, &format!("{prefix}.q_a_proj"))?; // replicated
            let q_a_ln = RmsNorm::load(
                weights,
                &format!("{prefix}.q_a_layernorm"),
                config.rms_norm_eps,
            )?;
            let q_b_w = weights.take_shard(
                &format!("{prefix}.q_b_proj.weight"),
                0,
                tp.rank,
                tp.world_size,
            )?;
            let q_b = Linear::new(q_b_w, None);
            (Some(q_a), Some(q_a_ln), q_b)
        } else {
            let q_w = weights.take_shard(
                &format!("{prefix}.q_proj.weight"),
                0,
                tp.rank,
                tp.world_size,
            )?;
            let q_proj = Linear::new(q_w, None);
            (None, None, q_proj)
        };

        // KV path: kv_a_proj_with_mqa is Replicated
        let kv_a = Linear::load(weights, &format!("{prefix}.kv_a_proj_with_mqa"))?;
        let kv_a_ln = RmsNorm::load(
            weights,
            &format!("{prefix}.kv_a_layernorm"),
            config.rms_norm_eps,
        )?;
        // kv_b_proj is ColumnParallel (shard dim 0)
        let kv_b_w = weights.take_shard(
            &format!("{prefix}.kv_b_proj.weight"),
            0,
            tp.rank,
            tp.world_size,
        )?;
        let kv_b = Linear::new(kv_b_w, None);

        // o_proj is RowParallel (shard dim 1)
        let o_w = weights.take_shard(
            &format!("{prefix}.o_proj.weight"),
            1,
            tp.rank,
            tp.world_size,
        )?;
        let o_proj = Linear::new(o_w, None);

        let mut scale = 1.0 / (qk_head_dim as f32).sqrt();
        if let Some(ref yarn) = config.yarn_rope_scaling {
            let mscale = yarn_get_mscale(yarn.factor, yarn.mscale_all_dim);
            scale *= (mscale * mscale) as f32;
        }

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa: kv_a,
            kv_a_layernorm: kv_a_ln,
            kv_b_proj: kv_b,
            o_proj,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_lora_rank,
            scale,
            layer_idx,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// MLA forward pass.
    ///
    /// Matches Python `DeepseekV2Attention.forward` (non-absorbed path).
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
        let stream = device.compute_stream;

        // ===== Q path =====
        // q_a_proj → q_a_layernorm → q_b_proj (or direct q_proj)
        let q_proj_out =
            if let (Some(q_a_proj), Some(q_a_ln)) = (&self.q_a_proj, &self.q_a_layernorm) {
                // q = q_a_proj(hidden_states)  → [num_tokens, q_lora_rank]
                let q_a =
                    q_a_proj.forward_owned(hidden_states, &mut device.cublas, &mut device.caching);
                // q = q_a_layernorm(q_a) → [num_tokens, q_lora_rank]
                let q_normed = kernels::rms_norm(
                    q_a.as_gpu_tensor(),
                    q_a_ln.weight,
                    q_a_ln.eps,
                    &mut device.caching,
                    stream,
                );
                drop(q_a);
                // q = q_b_proj(q_normed) → [num_tokens, num_heads * qk_head_dim]
                let q = self.q_b_proj.forward_owned(
                    q_normed.as_gpu_tensor(),
                    &mut device.cublas,
                    &mut device.caching,
                );
                drop(q_normed);
                q
            } else {
                // Direct: q_proj(hidden_states) → [num_tokens, num_heads * qk_head_dim]
                self.q_b_proj
                    .forward_owned(hidden_states, &mut device.cublas, &mut device.caching)
            };

        // Reshape Q to [num_tokens, num_heads, qk_head_dim]
        // Then split into q_nope [num_tokens, num_heads, nope_dim] and q_pe [num_tokens, num_heads, rope_dim]
        // For RoPE, we need q_pe as a contiguous [num_tokens, num_heads * rope_dim] 2D tensor.
        // We'll work with the flat layout and use pointer arithmetic.

        // q_proj_out is [num_tokens, num_heads * qk_head_dim]
        // We need to extract q_pe (last rope_dim of each head) and apply interleaved RoPE.

        // ===== KV path =====
        // kv_a_proj_with_mqa(hidden_states) → [num_tokens, kv_lora_rank + rope_dim]
        let kv_a_out = self.kv_a_proj_with_mqa.forward_owned(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
        );

        // Split into latent [kv_lora_rank] and k_pe_compressed [rope_dim]
        // latent_cache = kv_a_out (we keep both parts, split is virtual)
        // kv_a = first kv_lora_rank elements per token → layernorm
        // k_pe = last rope_dim elements per token → RoPE

        // For RMS norm on the latent part, we need a contiguous [num_tokens, kv_lora_rank] tensor.
        // Split kv_a_out into two parts.
        let kv_a_latent = device.caching.alloc_tensor(
            &[num_tokens, self.kv_lora_rank],
            kv_a_out.as_gpu_tensor().dtype(),
        );
        let k_pe_compressed = device.caching.alloc_tensor(
            &[num_tokens, self.qk_rope_head_dim],
            kv_a_out.as_gpu_tensor().dtype(),
        );

        // Split kv_a_out[:, :kv_lora_rank] → kv_a_latent, kv_a_out[:, kv_lora_rank:] → k_pe
        kernels::mla_split_kv_a(
            kv_a_out.as_gpu_tensor(),
            kv_a_latent.as_gpu_tensor(),
            k_pe_compressed.as_gpu_tensor(),
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            stream,
        );
        drop(kv_a_out);

        // kv_a_layernorm on latent → [num_tokens, kv_lora_rank]
        let kv_a_normed = kernels::rms_norm(
            kv_a_latent.as_gpu_tensor(),
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.eps,
            &mut device.caching,
            stream,
        );
        drop(kv_a_latent);

        // kv_b_proj(kv_a_normed) → [num_tokens, num_heads * (nope_dim + v_head_dim)]
        let kv_b_out = self.kv_b_proj.forward_owned(
            kv_a_normed.as_gpu_tensor(),
            &mut device.cublas,
            &mut device.caching,
        );
        drop(kv_a_normed);

        // kv_b_out is [num_tokens, num_heads * (nope_dim + v_head_dim)]
        // View as [num_tokens, num_heads, nope_dim + v_head_dim]
        // Split into k_nope [num_tokens, num_heads, nope_dim] and v [num_tokens, num_heads, v_head_dim]

        // ===== Assemble Q, K, V for attention =====
        // We need to:
        // 1. Extract q_pe from q_proj_out, apply interleaved RoPE
        // 2. Apply interleaved RoPE to k_pe_compressed (1 head)
        // 3. Assemble full Q = [q_nope | q_pe] (already in q_proj_out after RoPE)
        // 4. Assemble full K = [k_nope | k_pe_broadcast] (k_pe broadcast from 1 head to num_heads)
        // 5. Zero-pad V from v_head_dim to qk_head_dim

        // First, apply RoPE. q_pe is the last rope_dim elements of each qk_head_dim-sized head.
        // We need to call rotary_embedding_interleaved_inplace on q_pe and k_pe.
        // But q_pe is NOT contiguous (it's interleaved with q_nope in q_proj_out).
        // We need to extract q_pe, apply RoPE, write back.

        // Allocate q_pe contiguous: [num_tokens, num_heads, rope_dim]
        let dtype = hidden_states.dtype();
        let q_pe = device
            .caching
            .alloc_tensor(&[num_tokens, self.num_heads, self.qk_rope_head_dim], dtype);

        // Extract q_pe from q_proj_out: for each head h, the last rope_dim elements
        kernels::mla_extract_q_pe(
            q_proj_out.as_gpu_tensor(),
            q_pe.as_gpu_tensor()
                .reshape(&[num_tokens, self.num_heads * self.qk_rope_head_dim]),
            self.num_heads,
            self.qk_head_dim,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            stream,
        );

        // Apply interleaved RoPE to q_pe [num_tokens, num_heads * rope_dim]
        // and k_pe [num_tokens, 1 * rope_dim]
        let q_pe_2d = q_pe
            .as_gpu_tensor()
            .reshape(&[num_tokens, self.num_heads * self.qk_rope_head_dim]);
        let k_pe_2d = k_pe_compressed
            .as_gpu_tensor()
            .reshape(&[num_tokens, self.qk_rope_head_dim]);
        kernels::rotary_embedding_interleaved_inplace(
            q_pe_2d,
            k_pe_2d,
            positions,
            rotary.cos_sin_cache,
            self.qk_rope_head_dim,
            stream,
        );

        // Write q_pe back into q_proj_out (the rope portion of each head)
        kernels::mla_write_q_pe(
            q_pe.as_gpu_tensor()
                .reshape(&[num_tokens, self.num_heads * self.qk_rope_head_dim]),
            q_proj_out.as_gpu_tensor(),
            self.num_heads,
            self.qk_head_dim,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            stream,
        );
        drop(q_pe);

        // Now q_proj_out has the full Q with RoPE applied to the pe portion.
        // Reshape to [num_tokens, num_heads, qk_head_dim]
        let q = q_proj_out
            .as_gpu_tensor()
            .reshape(&[num_tokens, self.num_heads, self.qk_head_dim]);

        // Assemble K: [k_nope | k_pe_broadcast] → [num_tokens, num_heads, qk_head_dim]
        let k = device
            .caching
            .alloc_tensor(&[num_tokens, self.num_heads, self.qk_head_dim], dtype);
        kernels::mla_assemble_k(
            kv_b_out.as_gpu_tensor(),
            k_pe_compressed.as_gpu_tensor(),
            k.as_gpu_tensor()
                .reshape(&[num_tokens, self.num_heads * self.qk_head_dim]),
            self.num_heads,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.qk_head_dim,
            stream,
        );
        drop(k_pe_compressed);

        // Assemble V: zero-pad from v_head_dim to qk_head_dim → [num_tokens, num_heads, qk_head_dim]
        let v = device
            .caching
            .alloc_tensor(&[num_tokens, self.num_heads, self.qk_head_dim], dtype);
        // Zero the entire V buffer first
        driver::memset_d8(
            v.as_gpu_tensor().raw_ptr(),
            0,
            v.as_gpu_tensor().size_bytes(),
            stream,
        )
        .expect("memset V");
        // Copy v_head_dim portion from kv_b_out
        kernels::mla_assemble_v(
            kv_b_out.as_gpu_tensor(),
            v.as_gpu_tensor()
                .reshape(&[num_tokens, self.num_heads * self.qk_head_dim]),
            self.num_heads,
            self.qk_nope_head_dim,
            self.v_head_dim,
            self.qk_head_dim,
            stream,
        );
        drop(kv_b_out);

        // Write K, V into paged cache (BF16→FP8 when FP8 cache).
        crate::model::attention_helpers::write_kv_cache(
            k.as_gpu_tensor(),
            v.as_gpu_tensor(),
            slot_mapping,
            kv_cache,
            self.layer_idx,
            stream,
        );

        // FlashAttention
        let attn_output = crate::model::attention_helpers::attention_standard(
            q,
            k.as_gpu_tensor(),
            v.as_gpu_tensor(),
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
            stream,
        );
        drop(k);
        drop(v);
        drop(q_proj_out); // free Q buffer

        // attn_output is [num_tokens, num_heads, qk_head_dim]
        // Slice to [num_tokens, num_heads, v_head_dim] then reshape to [num_tokens, num_heads * v_head_dim]
        // We need to extract just the first v_head_dim from each head's output.
        let o_input = if self.v_head_dim < self.qk_head_dim {
            let sliced = device
                .caching
                .alloc_tensor(&[num_tokens, self.num_heads * self.v_head_dim], dtype);
            kernels::mla_slice_attn_output(
                attn_output
                    .as_gpu_tensor()
                    .reshape(&[num_tokens, self.num_heads * self.qk_head_dim]),
                sliced.as_gpu_tensor(),
                self.num_heads,
                self.qk_head_dim,
                self.v_head_dim,
                stream,
            );
            drop(attn_output);
            sliced
        } else {
            // v_head_dim == qk_head_dim: no slicing needed (not expected for DeepSeek MLA)
            unreachable!("v_head_dim should be < qk_head_dim for DeepSeek MLA");
        };

        // o_proj: [num_tokens, num_heads * v_head_dim] → [num_tokens, hidden_size]
        let result = self.o_proj.forward_owned(
            o_input.as_gpu_tensor(),
            &mut device.cublas,
            &mut device.caching,
        );
        drop(o_input);

        // TP all-reduce
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
// DeepSeekV2MoE
// ---------------------------------------------------------------------------

/// MoE with shared expert (no sigmoid gate, just add).
/// output = routed_scaling_factor * moe(x) + shared_expert(x)
pub struct DeepSeekV2MoE {
    pub moe: FusedMoELayer,
    /// Shared expert: fused gate+up → [2*intermediate, hidden]
    pub shared_gate_up: Linear,
    /// Shared expert: down → [hidden, intermediate]
    pub shared_down: Linear,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f64,
}

impl DeepSeekV2MoE {
    pub unsafe fn forward_owned(
        &self,
        hidden_states: GpuTensor,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let stream = device.compute_stream;

        // MoE path
        let moe_out = self.moe.forward_owned(hidden_states, device);

        // Scale routed output by routed_scaling_factor (Python: final_hidden_states *= routed_scaling_factor)
        if self.routed_scaling_factor != 1.0 {
            kernels::scale_inplace(
                moe_out.as_gpu_tensor(),
                self.routed_scaling_factor as f32,
                &device.cublas,
            );
        }

        // Shared expert path (unconditional, no sigmoid gate)
        let shared_gu = self.shared_gate_up.forward_owned(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
        );
        let shared_activated = kernels::silu_and_mul_fused(
            shared_gu.as_gpu_tensor(),
            self.shared_intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(shared_gu);

        let shared_out = self.shared_down.forward_owned(
            shared_activated.as_gpu_tensor(),
            &mut device.cublas,
            &mut device.caching,
        );
        drop(shared_activated);

        // output = moe_out + shared_out (simple add, no sigmoid gate)
        kernels::add_inplace(moe_out.as_gpu_tensor(), shared_out.as_gpu_tensor(), stream);
        drop(shared_out);

        moe_out
    }
}

pub struct DeepSeekV2Fp8MoE {
    pub moe: Fp8FusedMoELayer,
    pub shared_gate_up: Linear,
    pub shared_down: Linear,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f64,
}

impl DeepSeekV2Fp8MoE {
    pub unsafe fn forward_owned(
        &self,
        hidden_states: GpuTensor,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let stream = device.compute_stream;

        let moe_out = self.moe.forward_owned(hidden_states, device);

        if self.routed_scaling_factor != 1.0 {
            kernels::scale_inplace(
                moe_out.as_gpu_tensor(),
                self.routed_scaling_factor as f32,
                &device.cublas,
            );
        }

        let shared_gu = self.shared_gate_up.forward_owned(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
        );
        let shared_activated = kernels::silu_and_mul_fused(
            shared_gu.as_gpu_tensor(),
            self.shared_intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(shared_gu);

        let shared_out = self.shared_down.forward_owned(
            shared_activated.as_gpu_tensor(),
            &mut device.cublas,
            &mut device.caching,
        );
        drop(shared_activated);

        kernels::add_inplace(moe_out.as_gpu_tensor(), shared_out.as_gpu_tensor(), stream);
        drop(shared_out);

        moe_out
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

enum DeepSeekV2Mlp {
    Dense(LlamaMLP),
    MoE(DeepSeekV2MoE),
    Fp8MoE(DeepSeekV2Fp8MoE),
}

impl DeepSeekV2Mlp {
    unsafe fn forward_owned(
        &self,
        hidden_states: GpuTensor,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        match self {
            Self::Dense(mlp) => mlp.forward_owned(hidden_states, device),
            Self::MoE(moe) => moe.forward_owned(hidden_states, device),
            Self::Fp8MoE(moe) => moe.forward_owned(hidden_states, device),
        }
    }
}

pub struct DeepSeekV2DecoderLayer {
    pub self_attn: DeepSeekV2Attention,
    mlp: DeepSeekV2Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DeepSeekV2DecoderLayer {
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &DeepSeekV2Config,
        layer_idx: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let self_attn =
            DeepSeekV2Attention::load(weights, &format!("{prefix}.self_attn"), config, layer_idx)?;

        let is_dense = layer_idx < config.first_k_dense_replace;
        let mlp = if is_dense {
            let dense = LlamaMLP::load_fused(
                weights,
                &format!("{prefix}.mlp"),
                config.intermediate_size,
                stream,
            )?;
            DeepSeekV2Mlp::Dense(dense)
        } else {
            let moe_prefix = format!("{prefix}.mlp");
            let first_gate_name = format!("{moe_prefix}.experts.0.gate_proj.weight");
            let is_fp8 = weights
                .tensor_info(&first_gate_name)
                .map(|(_, dt)| dt == DType::Fp8E4m3)
                .unwrap_or(false);

            if is_fp8 {
                let fp8_moe = Self::load_fp8_moe(weights, &moe_prefix, config, stream)?;
                DeepSeekV2Mlp::Fp8MoE(fp8_moe)
            } else {
                let moe = Self::load_moe(weights, &moe_prefix, config, None, stream)?;
                DeepSeekV2Mlp::MoE(moe)
            }
        };

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
        })
    }

    fn load_moe(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &DeepSeekV2Config,
        tp: Option<TpConfig>,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<DeepSeekV2MoE> {
        let num_experts = config.n_routed_experts;
        let inter = config.moe_intermediate_size;
        let hidden = config.hidden_size;
        let (rank, world_size) = tp.map_or((0, 1), |t| (t.rank, t.world_size));
        let ipp = inter / world_size;

        let gate = Linear::load(weights, &format!("{prefix}.gate"))?;

        let first_gate = format!("{prefix}.experts.0.gate_proj.weight");
        let (_, dtype) = weights
            .tensor_info(&first_gate)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_gate}"))?;
        let elem = dtype.size_bytes();

        // Stack expert weights
        let w1_bytes = num_experts * 2 * ipp * hidden * elem;
        let w2_bytes = num_experts * hidden * ipp * elem;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        for e in 0..num_experts {
            let gate_name = format!("{prefix}.experts.{e}.gate_proj.weight");
            let up_name = format!("{prefix}.experts.{e}.up_proj.weight");
            let down_name = format!("{prefix}.experts.{e}.down_proj.weight");

            let expert_w1_offset = e * 2 * ipp * hidden * elem;
            let gate_proj_bytes = ipp * hidden * elem;

            unsafe {
                if world_size > 1 {
                    weights.take_shard_into(
                        &gate_name,
                        0,
                        rank,
                        world_size,
                        w1_ptr.add(expert_w1_offset),
                        stream,
                    )?;
                    weights.take_shard_into(
                        &up_name,
                        0,
                        rank,
                        world_size,
                        w1_ptr.add(expert_w1_offset + gate_proj_bytes),
                        stream,
                    )?;
                    let expert_w2_offset = e * hidden * ipp * elem;
                    weights.take_shard_into(
                        &down_name,
                        1,
                        rank,
                        world_size,
                        w2_ptr.add(expert_w2_offset),
                        stream,
                    )?;
                } else {
                    weights.take_into(&gate_name, w1_ptr.add(expert_w1_offset), stream)?;
                    weights.take_into(
                        &up_name,
                        w1_ptr.add(expert_w1_offset + gate_proj_bytes),
                        stream,
                    )?;
                    let expert_w2_offset = e * hidden * inter * elem;
                    weights.take_into(&down_name, w2_ptr.add(expert_w2_offset), stream)?;
                }
            }
        }

        let w1 = unsafe { GpuTensor::new(w1_ptr, &[num_experts, 2 * ipp, hidden], dtype) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[num_experts, hidden, ipp], dtype) };

        let moe = FusedMoELayer {
            gate,
            w1,
            w2,
            num_experts,
            top_k: config.num_experts_per_tok,
            intermediate_size: ipp,
            hidden_size: hidden,
            renormalize: config.norm_topk_prob,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        // Shared expert: n_shared_experts * moe_intermediate_size
        let shared_inter = config.n_shared_experts * config.moe_intermediate_size;
        let sipp = shared_inter / world_size;

        let shared_gate_up = if world_size > 1 {
            let gate_name = format!("{prefix}.shared_experts.gate_proj.weight");
            let up_name = format!("{prefix}.shared_experts.up_proj.weight");
            let gate_proj_bytes = sipp * hidden * elem;
            let total = 2 * gate_proj_bytes;
            let ptr = unsafe { driver::mem_alloc(total)? };
            unsafe {
                weights.take_shard_into(&gate_name, 0, rank, world_size, ptr, stream)?;
                weights.take_shard_into(
                    &up_name,
                    0,
                    rank,
                    world_size,
                    ptr.add(gate_proj_bytes),
                    stream,
                )?;
            }
            let w = unsafe { GpuTensor::new(ptr, &[2 * sipp, hidden], dtype) };
            Linear::new(w, None)
        } else {
            let gate_name = format!("{prefix}.shared_experts.gate_proj.weight");
            let up_name = format!("{prefix}.shared_experts.up_proj.weight");
            let gate_proj_bytes = shared_inter * hidden * elem;
            let total = 2 * gate_proj_bytes;
            let ptr = unsafe { driver::mem_alloc(total)? };
            unsafe {
                weights.take_into(&gate_name, ptr, stream)?;
                weights.take_into(&up_name, ptr.add(gate_proj_bytes), stream)?;
            }
            let w = unsafe { GpuTensor::new(ptr, &[2 * shared_inter, hidden], dtype) };
            Linear::new(w, None)
        };

        let shared_down = if world_size > 1 {
            let name = format!("{prefix}.shared_experts.down_proj.weight");
            let w = weights.take_shard(&name, 1, rank, world_size)?;
            Linear::new(w, None)
        } else {
            Linear::load(weights, &format!("{prefix}.shared_experts.down_proj"))?
        };

        Ok(DeepSeekV2MoE {
            moe,
            shared_gate_up,
            shared_down,
            shared_intermediate_size: sipp,
            routed_scaling_factor: config.routed_scaling_factor,
        })
    }

    fn load_fp8_moe(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &DeepSeekV2Config,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<DeepSeekV2Fp8MoE> {
        let fp8_moe = gpu_weights::load_fp8_moe_experts(
            weights,
            prefix,
            config.n_routed_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            config.num_experts_per_tok,
            config.norm_topk_prob,
            "gate_proj",
            "up_proj",
            "down_proj",
        )?;

        // Shared expert (dense BF16, unconditional — no sigmoid gate)
        let shared_inter = config.n_shared_experts * config.moe_intermediate_size;
        let hidden = config.hidden_size;

        let shared_gate_up = {
            let gate_name = format!("{prefix}.shared_experts.gate_proj.weight");
            let up_name = format!("{prefix}.shared_experts.up_proj.weight");
            let (_, dtype) = weights
                .tensor_info(&gate_name)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
            let elem = dtype.size_bytes();
            let gate_proj_bytes = shared_inter * hidden * elem;
            let total = 2 * gate_proj_bytes;
            let ptr = unsafe { driver::mem_alloc(total)? };
            unsafe {
                weights.take_into(&gate_name, ptr, stream)?;
                weights.take_into(&up_name, ptr.add(gate_proj_bytes), stream)?;
            }
            let w = unsafe { GpuTensor::new(ptr, &[2 * shared_inter, hidden], dtype) };
            Linear::new(w, None)
        };

        let shared_down = Linear::load(weights, &format!("{prefix}.shared_experts.down_proj"))?;

        Ok(DeepSeekV2Fp8MoE {
            moe: fp8_moe,
            shared_gate_up,
            shared_down,
            shared_intermediate_size: shared_inter,
            routed_scaling_factor: config.routed_scaling_factor,
        })
    }

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

        let res_gpu = *residual;
        kernels::fused_add_rms_norm_inplace(
            *attn_output,
            res_gpu,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
            device.compute_stream,
        );

        let mlp_output = self.mlp.forward_owned(*attn_output, device);
        drop(attn_output);

        (mlp_output, residual)
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2Model + ForCausalLM
// ---------------------------------------------------------------------------

pub struct DeepSeekV2Model {
    pub embed_tokens: Embedding,
    pub layers: Vec<DeepSeekV2DecoderLayer>,
    pub norm: RmsNorm,
    pub rotary: RotaryCache,
}

impl DeepSeekV2Model {
    pub fn load(
        weights: &mut GpuWeights,
        config: &DeepSeekV2Config,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DeepSeekV2DecoderLayer::load(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                device.compute_stream,
            )?);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;

        let rotary = if let Some(ref yarn) = config.yarn_rope_scaling {
            unsafe {
                yarn_rotary_cache(
                    config.qk_rope_head_dim,
                    config.max_position_embeddings,
                    config.rope_theta,
                    yarn,
                    dtype,
                    device,
                )?
            }
        } else {
            unsafe {
                RotaryCache::new(
                    config.qk_rope_head_dim,
                    config.max_position_embeddings,
                    config.rope_theta,
                    None,
                    dtype,
                    device,
                )?
            }
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
        })
    }

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
        let hidden_states = kernels::embedding_gather(
            self.embed_tokens.weight,
            input_ids,
            &mut device.caching,
            device.compute_stream,
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
        hidden_states.into_gpu_tensor()
    }
}

pub struct DeepSeekV2ForCausalLM {
    pub model: DeepSeekV2Model,
    pub lm_head: Linear,
}

impl DeepSeekV2ForCausalLM {
    pub fn load(
        weights: &mut GpuWeights,
        config: &DeepSeekV2Config,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = DeepSeekV2Model::load(weights, config, dtype, device)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };
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

        self.lm_head
            .forward(hidden_states, &mut device.cublas, &mut device.caching)
    }

    /// Set TP group on all layers.
    #[cfg(feature = "nccl")]
    pub fn set_tp_group(&mut self, group: Arc<NcclGroup>) {
        for layer in &mut self.model.layers {
            layer.self_attn.tp_group = Some(Arc::clone(&group));
            match &mut layer.mlp {
                DeepSeekV2Mlp::MoE(ref mut moe) => {
                    moe.moe.tp_group = Some(Arc::clone(&group));
                }
                DeepSeekV2Mlp::Dense(ref mut mlp) => {
                    mlp.tp_group = Some(Arc::clone(&group));
                }
                DeepSeekV2Mlp::Fp8MoE(ref mut moe) => {
                    moe.moe.tp_group = Some(Arc::clone(&group));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TP loading
// ---------------------------------------------------------------------------

impl DeepSeekV2DecoderLayer {
    pub fn load_tp(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &DeepSeekV2Config,
        layer_idx: usize,
        tp: TpConfig,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let self_attn = DeepSeekV2Attention::load_tp(
            weights,
            &format!("{prefix}.self_attn"),
            config,
            layer_idx,
            tp,
            stream,
        )?;

        let is_dense = layer_idx < config.first_k_dense_replace;
        let mlp = if is_dense {
            let dense = LlamaMLP::load_fused_tp(
                weights,
                &format!("{prefix}.mlp"),
                config.intermediate_size,
                tp,
                stream,
            )?;
            DeepSeekV2Mlp::Dense(dense)
        } else {
            let moe = Self::load_moe(weights, &format!("{prefix}.mlp"), config, Some(tp), stream)?;
            DeepSeekV2Mlp::MoE(moe)
        };

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
        })
    }
}

impl DeepSeekV2Model {
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &DeepSeekV2Config,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DeepSeekV2DecoderLayer::load_tp(
                weights,
                &format!("model.layers.{i}"),
                config,
                i,
                tp,
                device.compute_stream,
            )?);
        }

        let norm = RmsNorm::load(weights, "model.norm", config.rms_norm_eps)?;

        let rotary = if let Some(ref yarn) = config.yarn_rope_scaling {
            unsafe {
                yarn_rotary_cache(
                    config.qk_rope_head_dim,
                    config.max_position_embeddings,
                    config.rope_theta,
                    yarn,
                    dtype,
                    device,
                )?
            }
        } else {
            unsafe {
                RotaryCache::new(
                    config.qk_rope_head_dim,
                    config.max_position_embeddings,
                    config.rope_theta,
                    None,
                    dtype,
                    device,
                )?
            }
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
        })
    }
}

impl DeepSeekV2ForCausalLM {
    pub fn load_tp(
        weights: &mut GpuWeights,
        config: &DeepSeekV2Config,
        dtype: DType,
        tp: TpConfig,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = DeepSeekV2Model::load_tp(weights, config, dtype, tp, device)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(model.embed_tokens.weight, None)
        } else {
            Linear::load(weights, "lm_head")?
        };
        Ok(Self { model, lm_head })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yarn_get_mscale() {
        assert_eq!(yarn_get_mscale(1.0, 1.0), 1.0);
        assert_eq!(yarn_get_mscale(0.5, 1.0), 1.0);

        // factor=40, mscale_all_dim=0.1 → 0.1 * 0.1 * ln(40) + 1.0
        let result = yarn_get_mscale(40.0, 0.1);
        let expected = 0.1 * 0.1 * 40.0_f64.ln() + 1.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_deepseek_config_qk_head_dim() {
        let config = DeepSeekV2Config {
            hidden_size: 2048,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            intermediate_size: 10944,
            vocab_size: 102400,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            q_lora_rank: Some(1536),
            kv_lora_rank: 512,
            n_routed_experts: 64,
            n_shared_experts: 2,
            num_experts_per_tok: 6,
            first_k_dense_replace: 1,
            moe_intermediate_size: 1408,
            norm_topk_prob: false,
            routed_scaling_factor: 1.0,
            yarn_rope_scaling: None,
        };

        assert_eq!(config.qk_head_dim(), 192);
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests_cuda {
    use crate::driver;
    use crate::dtype::DType;
    use crate::kernels;
    use crate::tensor::GpuTensor;

    unsafe fn test_init() -> cudarc::driver::sys::CUstream {
        driver::init().expect("CUDA init");
        let dev = driver::device_get(0).expect("device 0");
        let _ctx = driver::ctx_create(dev).expect("ctx_create");
        driver::stream_create().expect("stream_create")
    }

    unsafe fn upload(data: &[f32], stream: cudarc::driver::sys::CUstream) -> *mut u8 {
        let bytes = data.len() * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memcpy_htod_async(ptr, data.as_ptr() as *const u8, bytes, stream).expect("htod");
        driver::stream_synchronize(stream).expect("sync");
        ptr
    }

    unsafe fn download(
        ptr: *mut u8,
        count: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Vec<f32> {
        let bytes = count * 4;
        let mut host = vec![0.0f32; count];
        driver::memcpy_dtoh_async(host.as_mut_ptr() as *mut u8, ptr, bytes, stream).expect("dtoh");
        driver::stream_synchronize(stream).expect("sync");
        host
    }

    unsafe fn alloc_zero(count: usize, stream: cudarc::driver::sys::CUstream) -> *mut u8 {
        let bytes = count * 4;
        let ptr = driver::mem_alloc(bytes).expect("mem_alloc");
        driver::memset_d8(ptr, 0, bytes, stream).expect("memset");
        driver::stream_synchronize(stream).expect("sync");
        ptr
    }

    #[test]
    #[ignore] // requires CUDA GPU
    fn test_cuda_mla_split_kv_a() {
        unsafe {
            let stream = test_init();
            // 2 tokens, kv_lora_rank=3, rope_dim=2 → src is [2, 5]
            let src_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
            let src_ptr = upload(&src_data, stream);
            let latent_ptr = alloc_zero(6, stream); // [2, 3]
            let kpe_ptr = alloc_zero(4, stream); // [2, 2]

            let src = GpuTensor::new(src_ptr, &[2, 5], DType::F32);
            let latent = GpuTensor::new(latent_ptr, &[2, 3], DType::F32);
            let kpe = GpuTensor::new(kpe_ptr, &[2, 2], DType::F32);

            kernels::mla_split_kv_a(src, latent, kpe, 3, 2, stream);
            let lat = download(latent_ptr, 6, stream);
            let kp = download(kpe_ptr, 4, stream);

            assert_eq!(lat, vec![1.0, 2.0, 3.0, 6.0, 7.0, 8.0]);
            assert_eq!(kp, vec![4.0, 5.0, 9.0, 10.0]);
        }
    }

    #[test]
    #[ignore]
    fn test_cuda_mla_extract_write_q_pe() {
        unsafe {
            let stream = test_init();
            // 1 token, 2 heads, qk_head_dim=4, nope=2, rope=2
            // Q layout: [h0_nope0, h0_nope1, h0_pe0, h0_pe1, h1_nope0, h1_nope1, h1_pe0, h1_pe1]
            let q_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
            let q_ptr = upload(&q_data, stream);
            let pe_ptr = alloc_zero(4, stream); // [1, 2*2]

            let q = GpuTensor::new(q_ptr, &[1, 8], DType::F32);
            let pe = GpuTensor::new(pe_ptr, &[1, 4], DType::F32);

            kernels::mla_extract_q_pe(q, pe, 2, 4, 2, 2, stream);
            let pe_host = download(pe_ptr, 4, stream);
            assert_eq!(pe_host, vec![30.0, 40.0, 70.0, 80.0]);

            // Now modify pe and write back
            let new_pe = vec![300.0, 400.0, 700.0, 800.0];
            let new_pe_ptr = upload(&new_pe, stream);
            let new_pe_t = GpuTensor::new(new_pe_ptr, &[1, 4], DType::F32);
            kernels::mla_write_q_pe(new_pe_t, q, 2, 4, 2, 2, stream);
            let q_host = download(q_ptr, 8, stream);
            assert_eq!(
                q_host,
                vec![10.0, 20.0, 300.0, 400.0, 50.0, 60.0, 700.0, 800.0]
            );
        }
    }

    #[test]
    #[ignore]
    fn test_cuda_mla_assemble_k() {
        unsafe {
            let stream = test_init();
            // 1 token, 2 heads, nope=2, rope=1, v_head_dim=2, qk_head_dim=3
            // kv_b: [1, 2*(2+2)] = [1, 8] → [h0_nope0, h0_nope1, h0_v0, h0_v1, h1_nope0, h1_nope1, h1_v0, h1_v1]
            let kv_b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let kv_b_ptr = upload(&kv_b_data, stream);
            // k_pe: [1, 1] (rope_dim=1, single head)
            let kpe_data = vec![99.0];
            let kpe_ptr = upload(&kpe_data, stream);
            // dst: [1, 2*3] = [1, 6]
            let dst_ptr = alloc_zero(6, stream);

            let kv_b = GpuTensor::new(kv_b_ptr, &[1, 8], DType::F32);
            let kpe = GpuTensor::new(kpe_ptr, &[1, 1], DType::F32);
            let dst = GpuTensor::new(dst_ptr, &[1, 6], DType::F32);

            kernels::mla_assemble_k(kv_b, kpe, dst, 2, 2, 1, 2, 3, stream);
            let result = download(dst_ptr, 6, stream);
            // K = [h0_nope0, h0_nope1, k_pe, h1_nope0, h1_nope1, k_pe]
            assert_eq!(result, vec![1.0, 2.0, 99.0, 5.0, 6.0, 99.0]);
        }
    }

    #[test]
    #[ignore]
    fn test_cuda_mla_assemble_v() {
        unsafe {
            let stream = test_init();
            // 1 token, 2 heads, nope=2, v_head_dim=2, qk_head_dim=4
            // kv_b: [1, 2*(2+2)] = [1, 8]
            let kv_b_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let kv_b_ptr = upload(&kv_b_data, stream);
            // dst: [1, 2*4] = [1, 8], pre-zeroed
            let dst_ptr = alloc_zero(8, stream);

            let kv_b = GpuTensor::new(kv_b_ptr, &[1, 8], DType::F32);
            let dst = GpuTensor::new(dst_ptr, &[1, 8], DType::F32);

            kernels::mla_assemble_v(kv_b, dst, 2, 2, 2, 4, stream);
            let result = download(dst_ptr, 8, stream);
            // V[h0] = [v0, v1, 0, 0], V[h1] = [v0, v1, 0, 0]
            // v comes after nope in kv_b: h0_v=[3,4], h1_v=[7,8]
            assert_eq!(result, vec![3.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0]);
        }
    }

    #[test]
    #[ignore]
    fn test_cuda_mla_slice_attn_output() {
        unsafe {
            let stream = test_init();
            // 1 token, 2 heads, qk_head_dim=4, v_head_dim=2
            // src: [1, 2*4] = [1, 8]
            let src_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let src_ptr = upload(&src_data, stream);
            let dst_ptr = alloc_zero(4, stream); // [1, 2*2]

            let src = GpuTensor::new(src_ptr, &[1, 8], DType::F32);
            let dst = GpuTensor::new(dst_ptr, &[1, 4], DType::F32);

            kernels::mla_slice_attn_output(src, dst, 2, 4, 2, stream);
            let result = download(dst_ptr, 4, stream);
            // First v_head_dim=2 elements per head: [1, 2, 5, 6]
            assert_eq!(result, vec![1.0, 2.0, 5.0, 6.0]);
        }
    }
}
