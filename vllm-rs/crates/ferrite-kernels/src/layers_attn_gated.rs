// SPDX-License-Identifier: Apache-2.0
//! Qwen3-Next full-attention layer with output gating + partial RoPE.
//!
//! Differences from a standard GQA attention block:
//! 1. **Doubled Q projection.** When `attn_output_gate` is on, the Q
//!    output of the fused QKV gemm is `2 * num_q_heads * head_dim`
//!    wide; the second half is the per-head sigmoid gate that scales
//!    the attention output before `o_proj`.
//! 2. **Per-head Q/K RMSNorm** (Gemma-style: weight + 1) applied to
//!    the post-projection q/k tiles before RoPE.
//! 3. **Q-only RoPE.** Q is rotated explicitly with the rotary cache;
//!    K is left un-rotated in the paged cache and rotated on-the-fly
//!    by FA2 during the matmul (the cache cos/sin pointer is passed
//!    into the attention kernel). RoPE is partial — `rotary_dim < head_dim`
//!    is supported transparently by the underlying kernels.
//! 4. **Sigmoid output gate.** `attn_output *= sigmoid(gate)` before
//!    the final `o_proj`.
//!
//! Mirrors `vllm-cuda/src/model/qwen3_next.rs::Qwen3NextFullAttention`
//! at single-rank (tp=1, BF16). Quantized / TP variants land alongside
//! their dedicated `LinearLayer` flavors in a follow-up.

// Dual-mode like `layers_moe` / `layers_gdn`: struct definition compiles
// without the `cuda` feature so `Instruction::GatedAttention(W,
// Qwen3NextGatedAttentionLayer)` resolves under metal; the impl block is
// `#[cfg(feature = "cuda")]`.

#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::CUstream;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::alloc::OwnedTensor;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::device::GpuDevice;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::driver;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::dtype::DType;
use ferrite_cuda_core::tensor::GpuTensor;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::tensor::TensorView;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::weights::GpuWeights;

#[cfg(feature = "cuda")]
use crate::attention_helpers;
#[cfg(feature = "cuda")]
use crate::kernels;
#[cfg(feature = "cuda")]
use crate::kv_cache::KvCachePool;
use crate::layers::Linear;

/// Per-layer weights for Qwen3-Next full attention with output gating.
pub struct Qwen3NextGatedAttentionLayer {
    /// Fused QKV projection. Output width is
    /// `q_size + 2 * kv_size`, where `q_size = 2 * num_q_heads * head_dim`
    /// when `attn_output_gate` is on (Q + gate fused).
    pub qkv_proj: Linear,
    /// Output projection: `[hidden, num_q_heads * head_dim]`.
    pub o_proj: Linear,
    /// Per-head Q RMSNorm weight: `[head_dim]`. Loaded with the +1
    /// offset baked in (Gemma convention: `weight + 1` at load time).
    pub q_norm_weight: Option<GpuTensor>,
    /// Per-head K RMSNorm weight: `[head_dim]`.
    pub k_norm_weight: Option<GpuTensor>,
    pub qk_norm_eps: f32,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// `2 * num_q_heads * head_dim` when `attn_output_gate`, else
    /// `num_q_heads * head_dim`.
    pub q_size: usize,
    /// `num_kv_heads * head_dim`.
    pub kv_size: usize,
    /// `num_q_heads * head_dim` — Q's true width without the gate.
    pub true_q_size: usize,
    /// `1 / sqrt(head_dim)`.
    pub scale: f32,
    pub attn_output_gate: bool,
}

#[cfg(feature = "cuda")]
impl Qwen3NextGatedAttentionLayer {
    /// Add 1.0 in-place to a Gemma-style RMSNorm weight (`weight + 1`).
    ///
    /// CPU-roundtrip — the QK-norm weight is `[head_dim]`, ~256
    /// elements at most, only run once per layer at load time.
    /// Mirrors `vllm_cuda::model::gemma2::add_one_to_weight`.
    unsafe fn add_one_inplace(weight: &GpuTensor, stream: CUstream) -> Result<()> {
        let n = weight.dim(0);
        let nbytes = weight.size_bytes();
        let host = driver::mem_alloc_host(nbytes)?;
        driver::memcpy_dtoh_async(host, weight.raw_ptr(), nbytes, stream)?;
        driver::stream_synchronize(stream)?;
        match weight.dtype() {
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
            other => anyhow::bail!("unsupported QK-norm dtype: {:?}", other),
        }
        driver::memcpy_htod_async(weight.raw_ptr(), host, nbytes, stream)?;
        driver::stream_synchronize(stream)?;
        driver::mem_free_host(host)?;
        Ok(())
    }

    /// Load Qwen3-Next full-attention weights from safetensors.
    ///
    /// The QKV is a single fused projection — Qwen3-Next ships
    /// `qkv_proj.weight` directly (no separate q/k/v split that
    /// the codegen would need to fuse). `q_norm` / `k_norm` are
    /// Gemma-style (loaded value + 1).
    pub fn load(
        gw: &mut GpuWeights,
        prefix: &str,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        qk_norm_eps: f32,
        attn_output_gate: bool,
        stream: CUstream,
    ) -> Result<Self> {
        let true_q_size = num_q_heads * head_dim;
        let q_size = if attn_output_gate {
            2 * true_q_size
        } else {
            true_q_size
        };
        let kv_size = num_kv_heads * head_dim;

        let qkv_w = gw.take(&format!("{prefix}.qkv_proj.weight"))?;
        let qkv_bias_name = format!("{prefix}.qkv_proj.bias");
        let qkv_bias = if gw.contains(&qkv_bias_name) {
            Some(gw.take(&qkv_bias_name)?)
        } else {
            None
        };
        let qkv_proj = Linear::new(qkv_w, qkv_bias);

        let o_proj = Linear::load(gw, &format!("{prefix}.o_proj"))?;

        let q_norm_name = format!("{prefix}.q_norm.weight");
        let q_norm_weight = if gw.contains(&q_norm_name) {
            let w = gw.take(&q_norm_name)?;
            unsafe { Self::add_one_inplace(&w, stream)? };
            Some(w)
        } else {
            None
        };
        let k_norm_name = format!("{prefix}.k_norm.weight");
        let k_norm_weight = if gw.contains(&k_norm_name) {
            let w = gw.take(&k_norm_name)?;
            unsafe { Self::add_one_inplace(&w, stream)? };
            Some(w)
        } else {
            None
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm_weight,
            k_norm_weight,
            qk_norm_eps,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_size,
            kv_size,
            true_q_size,
            scale: 1.0 / (head_dim as f32).sqrt(),
            attn_output_gate,
        })
    }

    /// Forward pass — returns `[T, num_q_heads * head_dim]` ready for
    /// `o_proj` (already applied here; the returned tensor is the
    /// post-`o_proj` hidden delta).
    ///
    /// `cos_sin_cache` is the model-wide rotary cache (`[max_pos,
    /// rotary_dim]`); `rotary_dim` may be less than `head_dim`
    /// (partial RoPE). The Q-only rotation kernel reads `dim(1)` of
    /// the cache for `rotary_dim`. The same cache pointer is also
    /// threaded into FA2 so K is rotated on read from the paged cache.
    ///
    /// # Safety
    /// Same contract as the underlying kernel calls — every tensor
    /// view must alias live GPU memory; `device.compute_stream` must
    /// be valid.
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
        layer_idx: usize,
        cos_sin_cache: GpuTensor,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        // 1. Fused QKV projection.
        let qkv = self
            .qkv_proj
            .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Split into Q (with optional gate) + K + V tiles.
        let q_heads_for_split = if self.attn_output_gate {
            2 * self.num_q_heads
        } else {
            self.num_q_heads
        };
        let (q_with_gate, k, v) = kernels::split_qkv(
            *qkv.view(),
            self.q_size,
            self.kv_size,
            q_heads_for_split,
            self.num_kv_heads,
            self.head_dim,
            &mut device.caching,
            stream,
        );
        drop(qkv);

        // 3. If `attn_output_gate`, separate Q from gate. The split
        //    layout is `[T, 2 * num_q_heads, head_dim]` — first
        //    `num_q_heads` per token is Q, next `num_q_heads` is gate.
        let (q, gate) = if self.attn_output_gate {
            let q_tensor = *q_with_gate.view();
            let elem_bytes = q_tensor.dtype().size_bytes();
            let bytes_per_half = self.num_q_heads * self.head_dim * elem_bytes;
            let stride = 2 * bytes_per_half;
            let actual_q = device.caching.alloc_tensor(
                &[num_tokens, self.num_q_heads, self.head_dim],
                q_tensor.dtype(),
            );
            let gate_tensor = device.caching.alloc_tensor(
                &[num_tokens, self.num_q_heads, self.head_dim],
                q_tensor.dtype(),
            );
            for t in 0..num_tokens {
                driver::memcpy_dtod_async(
                    actual_q.as_gpu_tensor().raw_ptr().add(t * bytes_per_half),
                    q_tensor.raw_ptr().add(t * stride) as *const u8,
                    bytes_per_half,
                    stream,
                )
                .expect("Q split memcpy");
                driver::memcpy_dtod_async(
                    gate_tensor
                        .as_gpu_tensor()
                        .raw_ptr()
                        .add(t * bytes_per_half),
                    q_tensor.raw_ptr().add(t * stride + bytes_per_half) as *const u8,
                    bytes_per_half,
                    stream,
                )
                .expect("gate split memcpy");
            }
            drop(q_with_gate);
            (actual_q, Some(gate_tensor))
        } else {
            (q_with_gate, None)
        };

        // 4. Per-head Q/K RMSNorm (Gemma convention: weight already
        //    has +1 baked at load time).
        if let (Some(q_norm_w), Some(k_norm_w)) = (self.q_norm_weight, self.k_norm_weight) {
            kernels::qk_norm_inplace(
                *q.view(),
                *k.view(),
                q_norm_w,
                k_norm_w,
                self.num_q_heads,
                self.num_kv_heads,
                self.head_dim,
                self.qk_norm_eps,
                0.0,
                0.0,
                stream,
            );
        }

        // 5. Q-only RoPE (partial-rotary aware via cos_sin_cache.dim(1)).
        let q_flat = q
            .view()
            .reshape(&[num_tokens, self.num_q_heads * self.head_dim]);
        kernels::rotary_embedding_q_only(
            *q_flat,
            *positions,
            cos_sin_cache,
            self.num_q_heads,
            self.head_dim,
            stream,
        );

        // 6. Write K/V to paged cache (un-rotated K — FA2 rotates on read).
        attention_helpers::write_kv_cache(
            k.view(),
            v.view(),
            slot_mapping,
            kv_cache,
            layer_idx,
            stream,
        );

        // 7. FlashAttention-2 with cos/sin passed in for K rotation.
        let attn_output = attention_helpers::attention_standard(
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
            layer_idx,
            device.num_sm,
            &mut device.caching,
            stream,
            cos_sin_cache.raw_ptr() as *const u8,
            cos_sin_cache.dim(1),
            false,
        );
        drop(k);
        drop(v);
        drop(q);

        // 8. Output gating: attn_output *= sigmoid(gate).
        if let Some(gate) = gate.as_ref() {
            kernels::sigmoid_mul_inplace(
                *attn_output.view(),
                *gate.view(),
                &mut device.caching,
                stream,
            );
        }
        drop(gate);

        let attn_flat = attn_output.view().reshape(&[num_tokens, self.true_q_size]);

        // 9. Output projection.
        let result = self
            .o_proj
            .forward(attn_flat, &mut device.cublas, &mut device.caching);
        drop(attn_output);
        result
    }
}

// Metal-side stub. The full gated-attention port (doubled-Q projection,
// per-head Gemma-style QK RMSNorm, partial RoPE, sigmoid output gate)
// lives in a follow-up commit; today the `load` symbol exists only so
// the macro-generated `Weights::load_with` body compiles. Loading a
// Qwen3-Next checkpoint on Metal therefore fails loud at runtime --
// honors `feedback_no_unimplemented_singletons` by resolving the symbol
// totally but refusing to silently produce wrong outputs.
#[cfg(feature = "metal")]
impl Qwen3NextGatedAttentionLayer {
    pub fn load(
        _gw: &mut ferrite_cuda_core::weights::GpuWeights,
        _prefix: &str,
        _num_q_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _qk_norm_eps: f32,
        _attn_output_gate: bool,
        _stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "Qwen3NextGatedAttentionLayer::load not yet implemented on Metal -- \
             port of the gated-attention variant (output gate, partial RoPE) \
             is the next phase"
        )
    }
}
