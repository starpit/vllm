// SPDX-License-Identifier: Apache-2.0
//! Qwen3-Next (Qwen3.5) model using `GpuTensor`.
//!
//! Hybrid architecture with two types of attention layers:
//! - **Full attention** (every 4th layer by default): standard KV-cache attention
//!   with QK-norm, partial RoPE, and sigmoid output gating
//! - **GDN linear attention** (remaining layers): Gated Delta Net recurrence
//!   with conv1d, per-head state matrices, and gated output normalization
//!
//! MLP is either dense (LlamaMLP) or MoE (Qwen3-style shared expert).
//!
//! Port of: `vllm/model_executor/models/qwen3_next.py`

#[cfg(feature = "nccl")]
use std::sync::Arc;

use anyhow::Result;

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::dtype::DType;
use crate::kernels;
use crate::kv_cache::KvCachePool;
use crate::layers::{Embedding, Linear, LinearLayer};
use crate::model::gemma2::{GemmaRmsNorm, add_one_to_weight};
use crate::model::llama::{LlamaAttention, LlamaConfig, LlamaMLP, RotaryCache};
use crate::model::qwen3_moe::{Qwen3MoeConfig, Qwen3MoeDecoderLayer, Qwen3MoeMlp};
use crate::tensor::GpuTensor;
use crate::weights::GpuWeights;

#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Qwen3NextConfig {
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

    /// Fraction of head_dim that gets RoPE (e.g. 0.25).
    pub partial_rotary_factor: f64,
    /// Output gating on full attention (sigmoid(gate) * attn_output).
    pub attn_output_gate: bool,

    // GDN fields
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,

    // MoE fields
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub decoder_sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,

    /// Per-layer type: "full_attention" or "linear_attention".
    pub layer_types: Vec<String>,
    /// Whether to apply per-layer scaling (attn_layer_scale, ffn_layer_scale).
    pub layer_scale: bool,
}

impl Qwen3NextConfig {
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types
            .get(layer_idx)
            .is_some_and(|t| t == "full_attention")
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step)
    }

    /// Rotary dimension (even, rounded from partial_rotary_factor * head_dim).
    pub fn rotary_dim(&self) -> usize {
        let rd = (self.head_dim as f64 * self.partial_rotary_factor).round() as usize;
        rd - (rd % 2) // ensure even
    }

    /// Number of full attention layers (for KV cache sizing).
    pub fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| t.as_str() == "full_attention")
            .count()
    }

    /// Indices of full attention layers.
    pub fn full_attention_layer_indices(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| t.as_str() == "full_attention")
            .map(|(i, _)| i)
            .collect()
    }

    /// Convert to LlamaConfig for reusing LlamaAttention loading.
    pub fn as_llama_config(&self) -> LlamaConfig {
        LlamaConfig {
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            head_dim: self.head_dim,
            tie_word_embeddings: self.tie_word_embeddings,
            llama3_rope_scaling: None,
        }
    }

    /// Convert to Qwen3MoeConfig for MoE layer loading.
    pub fn as_moe_config(&self) -> Qwen3MoeConfig {
        Qwen3MoeConfig {
            hidden_size: self.hidden_size,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            moe_intermediate_size: self.moe_intermediate_size,
            shared_expert_intermediate_size: self.shared_expert_intermediate_size,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            head_dim: self.head_dim,
            tie_word_embeddings: self.tie_word_embeddings,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            mlp_only_layers: self.mlp_only_layers.clone(),
        }
    }

    /// GDN key dimension (linear_num_key_heads * linear_key_head_dim).
    pub fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// GDN value dimension (linear_num_value_heads * linear_value_head_dim).
    pub fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// GDN conv1d input dimension (2*key_dim + value_dim).
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

// ---------------------------------------------------------------------------
// GDN recurrent state (GPU-resident)
// ---------------------------------------------------------------------------

/// GPU-resident GDN state pool for all requests across all GDN layers.
///
/// Layout:
/// - `conv_states`: `[num_slots, num_gdn_layers, conv_dim, kernel_size - 1]` (f32)
/// - `ssm_states`:  `[num_slots, num_gdn_layers, num_v_heads, head_v_dim, head_k_dim]` (f32)
///
/// Slots are indexed by request. When a request completes, its slot is freed.
pub struct GdnStatePool {
    /// `[num_slots * num_gdn_layers, conv_dim, kernel_size - 1]` (f32 on GPU).
    pub conv_states: GpuTensor,
    /// `[num_slots * num_gdn_layers, num_v_heads, head_v_dim, head_k_dim]` (f32 on GPU).
    pub ssm_states: GpuTensor,
    pub num_slots: usize,
    pub num_gdn_layers: usize,
    pub conv_dim: usize,
    pub kernel_size: usize,
    pub num_v_heads: usize,
    pub head_v_dim: usize,
    pub head_k_dim: usize,
}

impl GdnStatePool {
    /// Allocate the state pool on GPU. All states are zero-initialized.
    pub unsafe fn new(
        config: &Qwen3NextConfig,
        num_slots: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let num_gdn_layers = config
            .layer_types
            .iter()
            .filter(|t| t.as_str() == "linear_attention")
            .count();
        let conv_dim = config.conv_dim();
        let kernel_size = config.linear_conv_kernel_dim;
        let state_len = kernel_size - 1;
        let num_v_heads = config.linear_num_value_heads;
        let head_v_dim = config.linear_value_head_dim;
        let head_k_dim = config.linear_key_head_dim;

        let conv_elems = num_slots * num_gdn_layers * conv_dim * state_len;
        let ssm_elems = num_slots * num_gdn_layers * num_v_heads * head_v_dim * head_k_dim;

        let conv_bytes = conv_elems * 4; // f32
        let ssm_bytes = ssm_elems * 4;

        let conv_ptr = crate::driver::mem_alloc(conv_bytes)?;
        let ssm_ptr = crate::driver::mem_alloc(ssm_bytes)?;

        // Zero-initialize
        crate::driver::memset_d8(conv_ptr, 0, conv_bytes, stream)?;
        crate::driver::memset_d8(ssm_ptr, 0, ssm_bytes, stream)?;

        let conv_states = GpuTensor::new(
            conv_ptr,
            &[num_slots * num_gdn_layers, conv_dim, state_len],
            DType::F32,
        );
        let ssm_states = GpuTensor::new(
            ssm_ptr,
            &[
                num_slots * num_gdn_layers,
                num_v_heads * head_v_dim,
                head_k_dim,
            ],
            DType::F32,
        );

        Ok(Self {
            conv_states,
            ssm_states,
            num_slots,
            num_gdn_layers,
            conv_dim,
            kernel_size,
            num_v_heads,
            head_v_dim,
            head_k_dim,
        })
    }

    /// Get the conv_state sub-tensor for a given GDN layer.
    /// Returns `[num_slots, conv_dim, state_len]` view.
    pub fn conv_state_for_layer(&self, _gdn_layer_idx: usize) -> GpuTensor {
        // Layout is [num_slots * num_gdn_layers, conv_dim, state_len].
        // Layer `gdn_layer_idx` for slot `s` is at index `s * num_gdn_layers + gdn_layer_idx`.
        // The kernel handles indexing via state_indices; we return the full tensor.
        self.conv_states
    }

    /// Get the ssm_state sub-tensor for a given GDN layer.
    pub fn ssm_state_for_layer(&self, _gdn_layer_idx: usize) -> GpuTensor {
        self.ssm_states
    }

    /// Zero out all GDN state (conv + ssm) for a given slot.
    /// Call this when a new sequence is assigned to the slot.
    pub unsafe fn clear_slot(
        &self,
        slot_idx: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<()> {
        let state_len = self.kernel_size - 1;
        let conv_bytes_per_layer = self.conv_dim * state_len * 4; // f32
        let ssm_bytes_per_layer = self.num_v_heads * self.head_v_dim * self.head_k_dim * 4;

        for layer in 0..self.num_gdn_layers {
            let flat_idx = slot_idx * self.num_gdn_layers + layer;

            // Clear conv state
            let conv_offset = flat_idx * conv_bytes_per_layer;
            let conv_ptr = self.conv_states.raw_ptr().add(conv_offset);
            crate::driver::memset_d8(conv_ptr, 0, conv_bytes_per_layer, stream)?;

            // Clear ssm state
            let ssm_offset = flat_idx * ssm_bytes_per_layer;
            let ssm_ptr = self.ssm_states.raw_ptr().add(ssm_offset);
            crate::driver::memset_d8(ssm_ptr, 0, ssm_bytes_per_layer, stream)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GDN linear attention weights
// ---------------------------------------------------------------------------

/// Weights for a single GDN (Gated Delta Net) linear attention layer.
///
/// All weights are GPU-resident. Small per-head parameters (A_log, dt_bias,
/// conv1d_weight, norm_weight) are stored as f32 GPU tensors.
pub struct GdnWeights {
    /// Combined Q, K, V, Z projection: [2*key_dim + 2*value_dim, hidden_size].
    pub in_proj_qkvz: Linear,
    /// B and A projection: [2*num_v_heads, hidden_size].
    pub in_proj_ba: Linear,
    /// Conv1d weight: [conv_dim, kernel_size] (f32 on GPU).
    pub conv1d_weight: GpuTensor,
    /// A_log: [num_v_heads] (f32 on GPU).
    pub a_log: GpuTensor,
    /// dt_bias: [num_v_heads] (f32 on GPU).
    pub dt_bias: GpuTensor,
    /// Output norm weight: [head_v_dim] (f32 on GPU).
    pub norm_weight: GpuTensor,
    /// Output projection: [hidden_size, value_dim].
    pub out_proj: Linear,
    pub norm_eps: f32,

    // Dimensions
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_dim: usize,
    pub conv_kernel_size: usize,
    /// GDN layer index (for indexing into GdnStatePool).
    pub gdn_layer_idx: usize,
    /// Model dtype (for casting f32 GDN output back to model dtype before out_proj).
    pub model_dtype: DType,
}

impl GdnWeights {
    /// Upload a CPU f32 vec to GPU as a GpuTensor.
    unsafe fn upload_f32(
        data: &[f32],
        shape: &[usize],
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<GpuTensor> {
        let nbytes = data.len() * 4;
        let ptr = crate::driver::mem_alloc(nbytes)?;
        let host = crate::driver::mem_alloc_host(nbytes)?;
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, host, nbytes);
        crate::driver::memcpy_htod_async(ptr, host, nbytes, stream)?;
        crate::driver::stream_synchronize(stream)?;
        crate::driver::mem_free_host(host)?;
        Ok(GpuTensor::new(ptr, shape, DType::F32))
    }

    /// Upload a CPU i32 vec to GPU as a GpuTensor (I32).
    unsafe fn upload_i32(
        data: &[i32],
        shape: &[usize],
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<GpuTensor> {
        let nbytes = data.len() * 4;
        let ptr = crate::driver::mem_alloc(nbytes)?;
        let host = crate::driver::mem_alloc_host(nbytes)?;
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, host, nbytes);
        crate::driver::memcpy_htod_async(ptr, host, nbytes, stream)?;
        crate::driver::stream_synchronize(stream)?;
        crate::driver::mem_free_host(host)?;
        Ok(GpuTensor::new(ptr, shape, DType::I32))
    }

    /// Load GDN weights from safetensors.
    pub fn load(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen3NextConfig,
        gdn_layer_idx: usize,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<Self> {
        let key_dim = config.key_dim();
        let value_dim = config.value_dim();

        // in_proj_qkvz: [2*key_dim + 2*value_dim, hidden]
        let qkvz_name = format!("{prefix}.in_proj_qkvz.weight");
        let qkvz_w = weights.take(&qkvz_name)?;
        let in_proj_qkvz = Linear::new(qkvz_w, None);

        // in_proj_ba: [2*num_v_heads, hidden]
        let ba_name = format!("{prefix}.in_proj_ba.weight");
        let ba_w = weights.take(&ba_name)?;
        let in_proj_ba = Linear::new(ba_w, None);

        // conv1d.weight: [conv_dim, kernel_size] → f32 on GPU
        let conv_name = format!("{prefix}.conv1d.weight");
        let conv_data = weights.take_to_cpu_f32(&conv_name)?;
        let conv_dim = config.conv_dim();
        let kernel_size = config.linear_conv_kernel_dim;
        assert!(
            conv_data.len() >= conv_dim * kernel_size,
            "conv1d weight too small: {} < {}",
            conv_data.len(),
            conv_dim * kernel_size
        );
        let conv1d_weight = unsafe {
            Self::upload_f32(
                &conv_data[..conv_dim * kernel_size],
                &[conv_dim, kernel_size],
                stream,
            )?
        };

        // A_log: [num_v_heads] → f32 on GPU
        let a_log_name = format!("{prefix}.A_log");
        let a_log_data = weights.take_to_cpu_f32(&a_log_name)?;
        let a_log = unsafe { Self::upload_f32(&a_log_data, &[a_log_data.len()], stream)? };

        // dt_bias: [num_v_heads] → f32 on GPU
        let dt_bias_name = format!("{prefix}.dt_bias");
        let dt_bias_data = weights.take_to_cpu_f32(&dt_bias_name)?;
        let dt_bias = unsafe { Self::upload_f32(&dt_bias_data, &[dt_bias_data.len()], stream)? };

        // norm.weight: [head_v_dim] → f32 on GPU
        let norm_name = format!("{prefix}.norm.weight");
        let norm_data = weights.take_to_cpu_f32(&norm_name)?;
        let norm_weight = unsafe { Self::upload_f32(&norm_data, &[norm_data.len()], stream)? };

        // out_proj: [hidden, value_dim]
        let out_proj = Linear::load(weights, &format!("{prefix}.out_proj"))?;
        let model_dtype = out_proj.weight.dtype();

        Ok(Self {
            in_proj_qkvz,
            in_proj_ba,
            conv1d_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
            norm_eps: config.rms_norm_eps,
            num_k_heads: config.linear_num_key_heads,
            num_v_heads: config.linear_num_value_heads,
            head_k_dim: config.linear_key_head_dim,
            head_v_dim: config.linear_value_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size: kernel_size,
            gdn_layer_idx,
            model_dtype,
        })
    }

    /// Forward pass for GDN layer (fully GPU).
    ///
    /// Takes the hidden_states and GDN state pool, runs:
    /// 1. Input projections (GEMM on GPU)
    /// 2. Split QKVZ and BA into components (GPU reshape/split)
    /// 3. Causal conv1d on Q||K||V (GPU kernel)
    /// 4. Fused gating: g, beta (GPU kernel)
    /// 5. Fused recurrence (GPU kernel)
    /// 6. RMSNormGated (GPU kernel)
    /// 7. Output projection (GEMM on GPU)
    ///
    /// # Safety
    /// All GpuTensors must be valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_owned(
        &self,
        hidden_states: GpuTensor,
        gdn_state_pool: &GdnStatePool,
        state_indices: GpuTensor, // [batch_size] i32 — slot indices into pool
        cu_seqlens: GpuTensor,    // [batch_size + 1] i32 — cumulative seq lens
        num_seqs: usize,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        // 1. Input projections on GPU.
        let qkvz =
            self.in_proj_qkvz
                .forward_owned(hidden_states, &mut device.cublas, &mut device.caching);
        let ba =
            self.in_proj_ba
                .forward_owned(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Split QKVZ and BA on GPU using fused kernel (no CPU round-trip).
        let key_dim = self.key_dim;
        let value_dim = self.value_dim;
        let conv_dim = self.conv_dim;
        let num_k_heads = self.num_k_heads;
        let num_v_heads = self.num_v_heads;
        let head_k_dim = self.head_k_dim;
        let head_v_dim = self.head_v_dim;

        let (_q_split, _k_split, _v_split, z_owned, a_owned, b_owned, mixed_owned) =
            kernels::gdn_qkvz_split(
                qkvz.as_gpu_tensor(),
                ba.as_gpu_tensor(),
                num_tokens,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                key_dim,
                value_dim,
                conv_dim,
                &mut device.caching,
                stream,
            );
        drop(qkvz);
        drop(ba);

        let mixed_qkv_gpu = mixed_owned.as_gpu_tensor();
        let a_gpu = a_owned.as_gpu_tensor();
        let b_gpu = b_owned.as_gpu_tensor();
        let z_gpu = z_owned.as_gpu_tensor();

        // Transform state_indices: slot * num_gdn_layers + gdn_layer_idx
        // for proper per-layer indexing into the state pool.
        let num_gdn_layers = gdn_state_pool.num_gdn_layers;
        let gdn_layer_idx = self.gdn_layer_idx;
        let adjusted_indices = if num_gdn_layers > 1 {
            let indices_cpu = download_to_cpu_i32(state_indices, stream);
            let adjusted: Vec<i32> = indices_cpu
                .iter()
                .map(|&s| s * num_gdn_layers as i32 + gdn_layer_idx as i32)
                .collect();
            let t =
                Self::upload_i32(&adjusted, &[num_seqs], stream).expect("upload adjusted_indices");
            Some(t)
        } else {
            None
        };
        let eff_state_indices = adjusted_indices.as_ref().map_or(state_indices, |t| *t);

        // 3. Causal conv1d on GPU.
        let conv_out = device
            .caching
            .alloc_tensor(&[num_tokens, conv_dim], DType::F32);
        if num_seqs == num_tokens {
            // Decode path: one token per sequence — batched update.
            kernels::gdn_conv1d_update(
                gdn_state_pool.conv_states,
                mixed_qkv_gpu,
                self.conv1d_weight,
                conv_out.as_gpu_tensor(),
                eff_state_indices,
                conv_dim,
                self.conv_kernel_size,
                num_seqs,
                stream,
            );
        } else {
            // Prefill path: iterate per sequence using cu_seqlens and state_indices.
            // Read cu_seqlens and state_indices to CPU (small arrays).
            let cu_seqlens_cpu = download_to_cpu_i32(cu_seqlens, stream);
            let state_indices_cpu = download_to_cpu_i32(state_indices, stream);

            for s in 0..num_seqs {
                let seq_start = cu_seqlens_cpu[s] as usize;
                let seq_end = cu_seqlens_cpu[s + 1] as usize;
                let seq_len = seq_end - seq_start;
                if seq_len == 0 {
                    continue;
                }
                let raw_slot = state_indices_cpu[s] as usize;
                let slot_idx = raw_slot * num_gdn_layers + gdn_layer_idx;

                // Slice into the token dimension for this sequence.
                let byte_offset = seq_start * conv_dim * 4; // f32 = 4 bytes
                let x_view = GpuTensor::new(
                    mixed_qkv_gpu.raw_ptr().add(byte_offset),
                    &[seq_len, conv_dim],
                    DType::F32,
                );
                let out_view = GpuTensor::new(
                    conv_out.as_gpu_tensor().raw_ptr().add(byte_offset),
                    &[seq_len, conv_dim],
                    DType::F32,
                );

                kernels::gdn_conv1d_prefill(
                    gdn_state_pool.conv_states,
                    x_view,
                    self.conv1d_weight,
                    out_view,
                    slot_idx,
                    conv_dim,
                    self.conv_kernel_size,
                    seq_len,
                    stream,
                );
            }
        }

        // Split conv output into Q, K, V on GPU (no CPU round-trip).
        let (q_owned, k_owned, v_owned) = kernels::gdn_conv_split(
            conv_out.as_gpu_tensor(),
            num_tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            &mut device.caching,
            stream,
        );
        drop(conv_out);

        let q_gpu = q_owned.as_gpu_tensor();
        let k_gpu = k_owned.as_gpu_tensor();
        let v_gpu = v_owned.as_gpu_tensor();

        // 4. Fused gating: g, beta on GPU.
        let g_gpu = device
            .caching
            .alloc_tensor(&[num_tokens, num_v_heads], DType::F32);
        let beta_gpu = device
            .caching
            .alloc_tensor(&[num_tokens, num_v_heads], DType::F32);
        kernels::gdn_gating(
            g_gpu.as_gpu_tensor(),
            beta_gpu.as_gpu_tensor(),
            self.a_log,
            a_gpu,
            b_gpu,
            self.dt_bias,
            num_v_heads,
            num_tokens,
            stream,
        );

        // 5. Fused recurrence on GPU.
        let o_gpu = device
            .caching
            .alloc_tensor(&[num_tokens, num_v_heads, head_v_dim], DType::F32);
        let scale = 1.0; // scale is applied inside kernel after L2 norm
        kernels::gdn_recurrent_fwd(
            q_gpu,
            k_gpu,
            v_gpu,
            g_gpu.as_gpu_tensor(),
            beta_gpu.as_gpu_tensor(),
            o_gpu.as_gpu_tensor(),
            gdn_state_pool.ssm_states,
            eff_state_indices,
            cu_seqlens,
            scale,
            num_seqs,
            num_tokens,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            stream,
        );
        drop(g_gpu);
        drop(beta_gpu);

        // 6. RMSNormGated on GPU.
        // o_gpu: [num_tokens, num_v_heads, head_v_dim] → reshape to [num_tokens * num_v_heads, head_v_dim]
        // z_gpu: [num_tokens, value_dim] → reshape to [num_tokens * num_v_heads, head_v_dim]
        let total_rows = num_tokens * num_v_heads;
        let o_flat = o_gpu.as_gpu_tensor().reshape(&[total_rows, head_v_dim]);
        let z_flat = z_gpu.reshape(&[total_rows, head_v_dim]);
        let normed = device
            .caching
            .alloc_tensor(&[total_rows, head_v_dim], DType::F32);
        kernels::gdn_rms_norm_gated(
            o_flat,
            z_flat,
            self.norm_weight,
            normed.as_gpu_tensor(),
            self.norm_eps,
            head_v_dim,
            total_rows,
            stream,
        );
        drop(o_gpu);

        // Reshape normed to [num_tokens, value_dim] for output projection.
        let normed_flat = normed.as_gpu_tensor().reshape(&[num_tokens, value_dim]);

        // Cast f32 GDN output back to model dtype before out_proj GEMM
        // (matches Python which stores recurrence output in model dtype).
        let proj_input = if self.model_dtype != DType::F32 {
            kernels::cast_from_f32(normed_flat, self.model_dtype, &mut device.caching, stream)
        } else {
            normed
        };
        // 7. Output projection on GPU.
        let result = self.out_proj.forward_owned(
            proj_input.as_gpu_tensor().reshape(&[num_tokens, value_dim]),
            &mut device.cublas,
            &mut device.caching,
        );
        drop(proj_input);

        result
    }
}

// ---------------------------------------------------------------------------
// Full attention layer (with output gating + partial RoPE)
// ---------------------------------------------------------------------------

/// Full attention for Qwen3-Next.
///
/// Differences from standard LlamaAttention:
/// - Q projection is doubled: first half is Q, second half is gate
/// - QK-norm (GemmaRMSNorm style: weight + 1)
/// - Partial RoPE (only rotary_dim < head_dim gets rotated)
/// - Output gating: sigmoid(gate) * attn_output
pub struct Qwen3NextFullAttention {
    /// Inner LlamaAttention handles QKV GEMM, RoPE, FA2, o_proj.
    /// Q size is doubled to include gate.
    inner: LlamaAttention,
    /// True Q size (without gate).
    true_q_size: usize,
    /// Whether to apply output gating.
    attn_output_gate: bool,
}

impl Qwen3NextFullAttention {
    /// Load with fused QKV weights.
    ///
    /// The Q weight is `[2 * num_heads * head_dim, hidden]` (Q + gate fused).
    /// K and V are standard.
    pub fn load_fused(
        weights: &mut GpuWeights,
        prefix: &str,
        config: &Qwen3NextConfig,
        layer_idx: usize,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let stream = device.compute_stream;
        let num_q_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.head_dim;
        let true_q_size = num_q_heads * head_dim;
        let q_size = if config.attn_output_gate {
            2 * true_q_size // doubled for gate
        } else {
            true_q_size
        };
        let kv_size = num_kv_heads * head_dim;

        // Load Q (doubled), K, V and fuse into single tensor.
        let q_name = format!("{prefix}.qkv_proj.weight");
        // Qwen3-Next uses a single qkv_proj weight, not separate q/k/v.
        // Try qkv_proj first, fall back to separate q/k/v.
        let qkv_proj = if weights.contains(&q_name) {
            let w = weights.take(&q_name)?;
            // Check for bias
            let bias_name = format!("{prefix}.qkv_proj.bias");
            let bias = if weights.contains(&bias_name) {
                Some(weights.take(&bias_name)?)
            } else {
                None
            };
            LinearLayer::Dense(Linear::new(w, bias))
        } else {
            // Separate q_proj, k_proj, v_proj — fuse them.
            let q_proj_name = format!("{prefix}.q_proj.weight");
            let k_proj_name = format!("{prefix}.k_proj.weight");
            let v_proj_name = format!("{prefix}.v_proj.weight");

            let (_, q_dtype) = weights
                .tensor_info(&q_proj_name)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {q_proj_name}"))?;
            let hidden = config.hidden_size;
            let elem = q_dtype.size_bytes();
            let q_bytes = q_size * hidden * elem;
            let kv_bytes = kv_size * hidden * elem;
            let total = q_bytes + 2 * kv_bytes;

            let ptr = unsafe { crate::driver::mem_alloc(total)? };
            unsafe {
                weights.take_into(&q_proj_name, ptr, stream)?;
                weights.take_into(&k_proj_name, ptr.add(q_bytes), stream)?;
                weights.take_into(&v_proj_name, ptr.add(q_bytes + kv_bytes), stream)?;
            }
            let w = unsafe { GpuTensor::new(ptr, &[q_size + 2 * kv_size, hidden], q_dtype) };
            LinearLayer::Dense(Linear::new(w, None))
        };

        let o_proj = LinearLayer::Dense(Linear::load(weights, &format!("{prefix}.o_proj"))?);

        // QK-norm weights (GemmaRMSNorm convention: add +1 at load time).
        let q_norm_name = format!("{prefix}.q_norm.weight");
        let k_norm_name = format!("{prefix}.k_norm.weight");
        let q_norm_weight = if weights.contains(&q_norm_name) {
            let mut w = weights.take(&q_norm_name)?;
            unsafe { add_one_to_weight(&mut w, dtype, device)? };
            Some(w)
        } else {
            None
        };
        let k_norm_weight = if weights.contains(&k_norm_name) {
            let mut w = weights.take(&k_norm_name)?;
            unsafe { add_one_to_weight(&mut w, dtype, device)? };
            Some(w)
        } else {
            None
        };

        let inner = LlamaAttention {
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
            qk_norm_eps: config.rms_norm_eps,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        Ok(Self {
            inner,
            true_q_size,
            attn_output_gate: config.attn_output_gate,
        })
    }

    /// Forward pass with output gating and partial RoPE.
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
        kv_layer_idx: usize,
        rotary: &RotaryCache,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        // 1. QKV projection → [num_tokens, q_size + 2*kv_size].
        //    Where q_size = 2*true_q_size if attn_output_gate.
        let qkv = self.inner.qkv_proj.forward_owned(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
            stream,
        );

        // 2. Split into Q+gate, K, V.
        let q_size = self.inner.q_size;
        let kv_size = self.inner.kv_size;
        let head_dim = self.inner.head_dim;
        let num_q_heads = self.inner.num_q_heads;
        let num_kv_heads = self.inner.num_kv_heads;

        let (q, k, v) = kernels::split_qkv(
            qkv.as_gpu_tensor(),
            q_size,
            kv_size,
            if self.attn_output_gate {
                2 * num_q_heads
            } else {
                num_q_heads
            },
            num_kv_heads,
            head_dim,
            &mut device.caching,
            stream,
        );
        drop(qkv);

        // 3. If output gating, split Q into actual Q and gate.
        let (q, gate) = if self.attn_output_gate {
            // q is [num_tokens, 2*num_q_heads, head_dim].
            // Split into q [num_tokens, num_q_heads, head_dim] and gate [num_tokens, num_q_heads, head_dim].
            let q_tensor = q.as_gpu_tensor();
            let total_elems = num_tokens * num_q_heads * head_dim;
            let elem_bytes = q_tensor.dtype().size_bytes();

            let actual_q = device
                .caching
                .alloc_tensor(&[num_tokens, num_q_heads, head_dim], q_tensor.dtype());
            let gate_tensor = device
                .caching
                .alloc_tensor(&[num_tokens, num_q_heads, head_dim], q_tensor.dtype());

            // Split interleaved: for each token, heads 0..num_q_heads go to Q,
            // next num_q_heads go to gate. But actually the layout from split_qkv
            // is [num_tokens, 2*num_q_heads, head_dim], so the first num_q_heads
            // per token are Q, the next num_q_heads are gate.
            // This is a simple memcpy of two halves per token.
            let bytes_per_half = num_q_heads * head_dim * elem_bytes;
            let stride = 2 * bytes_per_half;
            for t in 0..num_tokens {
                crate::driver::memcpy_dtod_async(
                    actual_q.as_mut_ptr::<u8>().add(t * bytes_per_half),
                    q_tensor.as_ptr::<u8>().add(t * stride),
                    bytes_per_half,
                    stream,
                )
                .expect("q split memcpy");
                crate::driver::memcpy_dtod_async(
                    gate_tensor.as_mut_ptr::<u8>().add(t * bytes_per_half),
                    q_tensor.as_ptr::<u8>().add(t * stride + bytes_per_half),
                    bytes_per_half,
                    stream,
                )
                .expect("gate split memcpy");
            }

            let _ = total_elems;
            drop(q);
            (actual_q, Some(gate_tensor))
        } else {
            (q, None)
        };

        // 4. Apply QK-norm (in-place on Q and K).
        // Note: the qk_norm_rope kernel applies FULL RoPE. For partial RoPE,
        // we need to use the rotary cache which has rotary_dim < head_dim,
        // and the fused_qkv_rope kernel path. Instead, we apply QK-norm separately
        // then partial RoPE.
        if let (Some(q_norm_w), Some(k_norm_w)) =
            (self.inner.q_norm_weight, self.inner.k_norm_weight)
        {
            // Apply per-head RMS norm to Q and K.
            kernels::qk_norm_inplace(
                q.as_gpu_tensor(),
                k.as_gpu_tensor(),
                q_norm_w,
                k_norm_w,
                num_q_heads,
                num_kv_heads,
                head_dim,
                self.inner.qk_norm_eps,
                stream,
            );
        }

        // 5. Apply partial RoPE to Q and K in-place.
        kernels::apply_rope_qk_inplace(
            q.as_gpu_tensor(),
            k.as_gpu_tensor(),
            rotary.cos_sin_cache,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            stream,
        );

        // 6. Write K/V to paged cache.
        kernels::reshape_and_cache(
            k.as_gpu_tensor(),
            v.as_gpu_tensor(),
            kv_cache.k_cache(kv_layer_idx),
            kv_cache.v_cache(kv_layer_idx),
            slot_mapping,
            kv_cache.block_size,
            stream,
        );

        // 7. FlashAttention-2.
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
                self.inner.scale,
                true,
                0.0,
                -1,
                &mut device.caching,
                stream,
            )
        } else {
            drop(k);
            drop(v);
            kernels::flash_attn_paged(
                q.as_gpu_tensor(),
                kv_cache.k_cache(kv_layer_idx),
                kv_cache.v_cache(kv_layer_idx),
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_seqlen_q,
                max_seqlen_k,
                self.inner.scale,
                true,
                kv_cache.block_size,
                device.num_sm,
                &mut device.caching,
                stream,
            )
        };
        drop(q);

        // 8. Output gating: sigmoid(gate) * attn_output.
        let attn_flat = if let Some(gate) = gate {
            // attn_output is [num_tokens, num_q_heads, head_dim]
            // gate is [num_tokens, num_q_heads, head_dim]
            // Apply sigmoid(gate) * attn_output element-wise.
            kernels::sigmoid_mul_inplace(
                attn_output.as_gpu_tensor(),
                gate.as_gpu_tensor(),
                &mut device.caching,
                stream,
            );
            drop(gate);
            attn_output
                .as_gpu_tensor()
                .reshape(&[num_tokens, self.true_q_size])
        } else {
            attn_output
                .as_gpu_tensor()
                .reshape(&[num_tokens, self.true_q_size])
        };

        // 9. Output projection.
        let result = self.inner.o_proj.forward_owned(
            attn_flat,
            &mut device.cublas,
            &mut device.caching,
            stream,
        );
        drop(attn_output);

        // TP all-reduce.
        #[cfg(feature = "nccl")]
        if let Some(ref group) = self.inner.tp_group {
            group
                .all_reduce_inplace(result.as_gpu_tensor())
                .expect("o_proj all_reduce failed");
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

/// A single Qwen3-Next decoder layer (either full attention or GDN).
pub enum Qwen3NextAttnVariant {
    FullAttention(Qwen3NextFullAttention),
    LinearAttention(GdnWeights),
}

#[allow(clippy::large_enum_variant)]
pub enum Qwen3NextMlpVariant {
    Dense(LlamaMLP),
    MoE(Qwen3MoeMlp),
}

pub struct Qwen3NextDecoderLayer {
    pub attn: Qwen3NextAttnVariant,
    pub mlp: Qwen3NextMlpVariant,
    pub input_layernorm: GemmaRmsNorm,
    pub post_attention_layernorm: GemmaRmsNorm,
    /// Optional layer scale applied after attention: x *= scale (pre-offset by +1).
    pub attn_layer_scale: Option<GpuTensor>,
    /// Optional layer scale applied after MLP: x *= scale (pre-offset by +1).
    pub ffn_layer_scale: Option<GpuTensor>,
    /// KV cache layer index (only valid for full attention layers).
    pub kv_layer_idx: Option<usize>,
    /// GDN state index (only valid for linear attention layers).
    pub gdn_state_idx: Option<usize>,
}

impl Qwen3NextDecoderLayer {
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
        gdn_state_pool: &GdnStatePool,
        gdn_state_indices: GpuTensor, // [batch] i32 — slot indices for GDN state
        gdn_cu_seqlens: GpuTensor,    // [batch+1] i32 — cumsum of seq lens
        num_seqs: usize,
        device: &mut GpuDevice,
    ) -> (OwnedTensor, OwnedTensor) {
        let stream = device.compute_stream;

        // Pre-attention norm with fused residual add.
        let (normed, residual) = if let Some(residual) = residual {
            let hs_gpu = *hidden_states;
            let res_gpu = *residual;
            kernels::fused_add_rms_norm_inplace(
                hs_gpu,
                res_gpu,
                self.input_layernorm.inner.weight,
                self.input_layernorm.inner.eps,
                stream,
            );
            (hidden_states, residual)
        } else {
            let normed = kernels::rms_norm(
                *hidden_states,
                self.input_layernorm.inner.weight,
                self.input_layernorm.inner.eps,
                &mut device.caching,
                stream,
            );
            (normed, hidden_states)
        };

        // Attention.
        let attn_output = match &self.attn {
            Qwen3NextAttnVariant::FullAttention(attn) => {
                let kv_idx = self
                    .kv_layer_idx
                    .expect("full attn layer must have kv_layer_idx");
                attn.forward_owned(
                    *normed,
                    positions,
                    slot_mapping,
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    kv_cache,
                    kv_idx,
                    rotary,
                    device,
                )
            }
            Qwen3NextAttnVariant::LinearAttention(gdn) => gdn.forward_owned(
                *normed,
                gdn_state_pool,
                gdn_state_indices,
                gdn_cu_seqlens,
                num_seqs,
                device,
            ),
        };
        drop(normed);

        // Apply attention layer scale before residual add.
        #[cfg(feature = "cuda")]
        if let Some(ref scale) = self.attn_layer_scale {
            kernels::broadcast_mul_inplace(*attn_output, *scale, stream);
        }

        // Post-attention norm + residual.
        let res_gpu = *residual;
        kernels::fused_add_rms_norm_inplace(
            *attn_output,
            res_gpu,
            self.post_attention_layernorm.inner.weight,
            self.post_attention_layernorm.inner.eps,
            stream,
        );

        // MLP.
        let mlp_output = match &self.mlp {
            Qwen3NextMlpVariant::Dense(mlp) => mlp.forward_owned(*attn_output, device),
            Qwen3NextMlpVariant::MoE(moe) => moe.forward_owned(*attn_output, device),
        };
        drop(attn_output);

        // Apply MLP layer scale before residual add (residual is implicit in next layer).
        #[cfg(feature = "cuda")]
        if let Some(ref scale) = self.ffn_layer_scale {
            kernels::broadcast_mul_inplace(*mlp_output, *scale, stream);
        }

        (mlp_output, residual)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Qwen3NextModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen3NextDecoderLayer>,
    pub norm: GemmaRmsNorm,
    pub rotary: RotaryCache,
    pub config: Qwen3NextConfig,
}

impl Qwen3NextModel {
    pub fn load(
        weights: &mut GpuWeights,
        config: &Qwen3NextConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let embed_tokens = Embedding::load(weights, "model.embed_tokens")?;

        let moe_cfg = config.as_moe_config();

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut kv_idx = 0usize;
        let mut gdn_idx = 0usize;

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            let (attn, kv_layer_idx, gdn_state_idx) = if config.is_full_attention(i) {
                let attn = Qwen3NextFullAttention::load_fused(
                    weights,
                    &format!("{prefix}.self_attn"),
                    config,
                    i,
                    dtype,
                    device,
                )?;
                let idx = kv_idx;
                kv_idx += 1;
                (Qwen3NextAttnVariant::FullAttention(attn), Some(idx), None)
            } else {
                let gdn = GdnWeights::load(
                    weights,
                    &format!("{prefix}.linear_attn"),
                    config,
                    gdn_idx,
                    device.compute_stream,
                )?;
                let idx = gdn_idx;
                gdn_idx += 1;
                (Qwen3NextAttnVariant::LinearAttention(gdn), None, Some(idx))
            };

            let mlp = if config.is_moe_layer(i) {
                // Load MoE using Qwen3MoeDecoderLayer's load_moe (package-private).
                // We replicate the pattern from qwen3_moe.rs.
                let moe = Qwen3MoeDecoderLayer::load_moe(
                    weights,
                    &format!("{prefix}.mlp"),
                    &moe_cfg,
                    None,
                    device.compute_stream,
                )?;
                Qwen3NextMlpVariant::MoE(moe)
            } else {
                let dense = LlamaMLP::load_fused(
                    weights,
                    &format!("{prefix}.mlp"),
                    config.intermediate_size,
                    device.compute_stream,
                )?;
                Qwen3NextMlpVariant::Dense(dense)
            };

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

            // Layer scale weights (GemmaRMSNorm convention: +1 at load time).
            let attn_layer_scale = if config.layer_scale {
                let name = format!("{prefix}.attn_layer_scale");
                if weights.contains(&name) {
                    let mut w = weights.take(&name)?;
                    unsafe { add_one_to_weight(&mut w, dtype, device)? };
                    Some(w)
                } else {
                    None
                }
            } else {
                None
            };
            let ffn_layer_scale = if config.layer_scale {
                let name = format!("{prefix}.ffn_layer_scale");
                if weights.contains(&name) {
                    let mut w = weights.take(&name)?;
                    unsafe { add_one_to_weight(&mut w, dtype, device)? };
                    Some(w)
                } else {
                    None
                }
            } else {
                None
            };

            layers.push(Qwen3NextDecoderLayer {
                attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
                attn_layer_scale,
                ffn_layer_scale,
                kv_layer_idx,
                gdn_state_idx,
            });
        }

        let norm = GemmaRmsNorm::load(weights, "model.norm", config.rms_norm_eps, dtype, device)?;
        let rotary = unsafe {
            RotaryCache::new_partial(
                config.head_dim,
                config.rotary_dim(),
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
            config: config.clone(),
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
        gdn_state_pool: &GdnStatePool,
        gdn_state_indices: GpuTensor,
        gdn_cu_seqlens: GpuTensor,
        num_seqs: usize,
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
                gdn_state_pool,
                gdn_state_indices,
                gdn_cu_seqlens,
                num_seqs,
                device,
            );
            hidden_states = hs;
            residual = Some(res);
        }

        // Final norm.
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
// CausalLM wrapper
// ---------------------------------------------------------------------------

pub struct Qwen3NextForCausalLM {
    pub model: Qwen3NextModel,
    pub lm_head: LinearLayer,
}

impl Qwen3NextForCausalLM {
    pub fn load(
        weights: &mut GpuWeights,
        config: &Qwen3NextConfig,
        dtype: DType,
        device: &GpuDevice,
    ) -> Result<Self> {
        let model = Qwen3NextModel::load(weights, config, dtype, device)?;
        let lm_head = if config.tie_word_embeddings {
            LinearLayer::Dense(Linear::new(model.embed_tokens.weight, None))
        } else {
            LinearLayer::Dense(Linear::load(weights, "lm_head")?)
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
        gdn_state_pool: &GdnStatePool,
        gdn_state_indices: GpuTensor,
        gdn_cu_seqlens: GpuTensor,
        num_seqs: usize,
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
            gdn_state_pool,
            gdn_state_indices,
            gdn_cu_seqlens,
            num_seqs,
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

        let logits = self.lm_head.forward_owned(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        );
        logits.into_gpu_tensor()
    }

    /// Number of full attention layers (for KV cache sizing).
    pub fn num_kv_layers(&self) -> usize {
        self.model
            .layers
            .iter()
            .filter(|l| l.kv_layer_idx.is_some())
            .count()
    }

    pub fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    pub fn num_kv_heads(&self) -> usize {
        self.model.config.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.model.config.head_dim
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Download a GPU tensor to CPU as f32 (synchronous).
#[allow(dead_code)]
unsafe fn download_to_cpu_f32(
    tensor: GpuTensor,
    stream: cudarc::driver::sys::CUstream,
) -> Vec<f32> {
    let num_elems = tensor.numel();
    let nbytes = tensor.size_bytes();

    // Allocate pinned host memory.
    let host = crate::driver::mem_alloc_host(nbytes).expect("host alloc");
    crate::driver::memcpy_dtoh_async(host, tensor.as_ptr(), nbytes, stream).expect("dtoh");
    crate::driver::stream_synchronize(stream).expect("sync");

    let result = match tensor.dtype() {
        DType::F32 => {
            let slice = std::slice::from_raw_parts(host as *const f32, num_elems);
            slice.to_vec()
        }
        DType::F16 => {
            let slice = std::slice::from_raw_parts(host as *const half::f16, num_elems);
            slice.iter().map(|x| x.to_f32()).collect()
        }
        DType::BF16 => {
            let slice = std::slice::from_raw_parts(host as *const half::bf16, num_elems);
            slice.iter().map(|x| x.to_f32()).collect()
        }
        _ => panic!("unsupported dtype for download: {:?}", tensor.dtype()),
    };

    crate::driver::mem_free_host(host).expect("free host");
    result
}

/// Download a GPU tensor to CPU as i32 (synchronous). Expects I32 or U32 dtype.
unsafe fn download_to_cpu_i32(
    tensor: GpuTensor,
    stream: cudarc::driver::sys::CUstream,
) -> Vec<i32> {
    let num_elems = tensor.numel();
    let nbytes = num_elems * 4;
    let host = crate::driver::mem_alloc_host(nbytes).expect("host alloc");
    crate::driver::memcpy_dtoh_async(host, tensor.as_ptr(), nbytes, stream).expect("dtoh");
    crate::driver::stream_synchronize(stream).expect("sync");
    let slice = std::slice::from_raw_parts(host as *const i32, num_elems);
    let result = slice.to_vec();
    crate::driver::mem_free_host(host).expect("free host");
    result
}
