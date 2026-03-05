// SPDX-License-Identifier: Apache-2.0
//! Quantized Qwen3-Next (Qwen3.5) model architecture using GGUF weights.
//!
//! Parallel to `qwen3_next.rs` but uses `QuantizedLinear` (wrapping `QMatMul`)
//! for linear projections. Non-linear layers (embedding, norms) are dequantized
//! to f32. KV cache remains in f32 (unquantized).
//!
//! Handles the `qwen35` GGUF architecture with:
//! - GDN (linear attention) layers: causal conv1d + gated delta rule recurrence
//! - Full attention layers: output gating, QK norms, partial RoPE
//! - GemmaRmsNorm `(1+w)*x` for all layer norms
//!
//! GGUF tensor name mapping:
//! - GDN layers: `attn_qkv` (fused qkvz), `ssm_alpha`/`ssm_beta` (split in_proj_ba),
//!   `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, `ssm_norm`, `ssm_out`, `attn_gate` (unused?)
//! - Full attention: `attn_q`/`attn_k`/`attn_v`/`attn_output`, `attn_q_norm`/`attn_k_norm`
//! - MLP: `ffn_gate`/`ffn_up`/`ffn_down`

use std::cell::RefCell;

use candle_core::{DType, Device, Module, Tensor};

use vllm_model::error::{ModelError, ModelResult};
use vllm_model::gguf::GgufFile;
use vllm_model::layers::{Embedding, GemmaRmsNorm, QuantizedLinear};
use vllm_model::weight::HfModelConfig;

use crate::attention::attention_with_cache;
use crate::quantized_llama::{QuantizedLlamaMLP, apply_interleaved_rope, precompute_freqs_cis};
use crate::qwen3_next::Qwen3NextConfig;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Dequantize a GGUF tensor to f32 and construct a GemmaRmsNorm.
fn load_gemma_rms_norm(
    gguf: &mut GgufFile,
    name: &str,
    eps: f64,
    device: &Device,
) -> ModelResult<GemmaRmsNorm> {
    let weight = gguf
        .tensor(name, device)?
        .dequantize(device)
        .map_err(|e| ModelError::Other(format!("dequantize norm: {e}")))?;
    GemmaRmsNorm::new(weight, eps).map_err(ModelError::Candle)
}

/// Element-wise sigmoid: 1 / (1 + exp(-x)).
fn tensor_sigmoid(x: &Tensor) -> candle_core::Result<Tensor> {
    (x.neg()?.exp()? + 1.0)?.recip()
}

/// Element-wise softplus: log(1 + exp(x)) with numerical stability.
fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    let threshold = 20.0f64;
    let ones = Tensor::ones_like(x)?;
    let mask = x.ge(&(ones.clone() * threshold)?)?;
    let safe = x.clamp(f64::NEG_INFINITY, threshold)?;
    let sp = (safe.exp()? + 1.0)?.log()?;
    mask.to_dtype(x.dtype())?.broadcast_mul(x)?
        + (ones - mask.to_dtype(x.dtype())?)?.broadcast_mul(&sp)?
}

/// L2 normalize the last dimension of a tensor.
fn l2_normalize(x: &Tensor) -> candle_core::Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?;
    let norm = (norm + 1e-12)?;
    x.broadcast_div(&norm)
}

// ---------------------------------------------------------------------------
// QuantizedQwen3NextAttention (full attention with output gating)
// ---------------------------------------------------------------------------

/// Quantized Qwen3-Next full attention with output gating, QK norms, partial RoPE.
///
/// q_proj output is doubled: first half queries, second half gate.
/// After attention: output = sigmoid(gate) * attn_output.
struct QuantizedQwen3NextAttention {
    q_proj: QuantizedLinear,
    k_proj: QuantizedLinear,
    v_proj: QuantizedLinear,
    o_proj: QuantizedLinear,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    /// Precomputed cos/sin for interleaved RoPE.
    cos: Tensor,
    sin: Tensor,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    scale: f64,
}

impl QuantizedQwen3NextAttention {
    fn load(
        gguf: &mut GgufFile,
        prefix: &str,
        config: &Qwen3NextConfig,
        cos: Tensor,
        sin: Tensor,
        device: &Device,
    ) -> ModelResult<Self> {
        let q_proj = QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_q.weight"), device)?;
        let k_proj = QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_k.weight"), device)?;
        let v_proj = QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_v.weight"), device)?;
        let o_proj =
            QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_output.weight"), device)?;

        let q_norm = load_gemma_rms_norm(
            gguf,
            &format!("{prefix}.attn_q_norm.weight"),
            config.rms_norm_eps,
            device,
        )?;
        let k_norm = load_gemma_rms_norm(
            gguf,
            &format!("{prefix}.attn_k_norm.weight"),
            config.rms_norm_eps,
            device,
        )?;

        let rotary_dim = (config.head_dim as f64 * config.partial_rotary_factor).round() as usize;
        let rotary_dim = rotary_dim - (rotary_dim % 2);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            cos,
            sin,
            num_q_heads: config.num_attention_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            rotary_dim,
            scale: 1.0 / (config.head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<Tensor> {
        let num_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;

        // Q projection (doubled for gate), K, V projections.
        let q_gate = self
            .q_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;
        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?;

        // Split q_gate into q and gate.
        let q_gate = q_gate
            .reshape((num_tokens, self.num_q_heads, 2 * self.head_dim))
            .map_err(ModelError::Candle)?;
        let q = q_gate
            .narrow(2, 0, self.head_dim)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;
        let gate = q_gate
            .narrow(2, self.head_dim, self.head_dim)
            .map_err(ModelError::Candle)?
            .contiguous()
            .map_err(ModelError::Candle)?;

        let k = k
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;

        // Per-head QK norms (GemmaRmsNorm).
        let q = crate::ops::gemma_rms_norm(&q, &self.q_norm).map_err(ModelError::Candle)?;
        let k = crate::ops::gemma_rms_norm(&k, &self.k_norm).map_err(ModelError::Candle)?;

        // Apply partial interleaved RoPE (GGML convention).
        let (q, k) = if self.rotary_dim < self.head_dim {
            let q_rot = q
                .narrow(2, 0, self.rotary_dim)
                .map_err(ModelError::Candle)?
                .contiguous()
                .map_err(ModelError::Candle)?;
            let q_pass = q
                .narrow(2, self.rotary_dim, self.head_dim - self.rotary_dim)
                .map_err(ModelError::Candle)?;
            let k_rot = k
                .narrow(2, 0, self.rotary_dim)
                .map_err(ModelError::Candle)?
                .contiguous()
                .map_err(ModelError::Candle)?;
            let k_pass = k
                .narrow(2, self.rotary_dim, self.head_dim - self.rotary_dim)
                .map_err(ModelError::Candle)?;

            let q_rot = apply_interleaved_rope(&q_rot, &self.cos, &self.sin, positions)?;
            let k_rot = apply_interleaved_rope(&k_rot, &self.cos, &self.sin, positions)?;

            let q = Tensor::cat(&[&q_rot, &q_pass], 2).map_err(ModelError::Candle)?;
            let k = Tensor::cat(&[&k_rot, &k_pass], 2).map_err(ModelError::Candle)?;
            (q, k)
        } else {
            let q = apply_interleaved_rope(&q, &self.cos, &self.sin, positions)?;
            let k = apply_interleaved_rope(&k, &self.cos, &self.sin, positions)?;
            (q, k)
        };

        let v = v
            .reshape((num_tokens, self.num_kv_heads, self.head_dim))
            .map_err(ModelError::Candle)?;

        // Attention with KV cache.
        let attn_output = attention_with_cache(&q, &k, &v, self.scale, kv_cache, None)?;

        // Output gating: sigmoid(gate) * attn_output.
        let gate_sigmoid = tensor_sigmoid(&gate).map_err(ModelError::Candle)?;
        let attn_output = attn_output
            .broadcast_mul(&gate_sigmoid)
            .map_err(ModelError::Candle)?;

        let attn_output = attn_output
            .reshape((num_tokens, self.num_q_heads * self.head_dim))
            .map_err(ModelError::Candle)?;
        self.o_proj
            .forward(&attn_output)
            .map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// QuantizedGatedDeltaNet (GDN linear attention)
// ---------------------------------------------------------------------------

/// Quantized Gated Delta Net linear attention layer.
///
/// GGUF tensor mapping:
/// - `attn_qkv.weight` → fused in_proj_qkvz [hidden, 2*key_dim + 2*value_dim]
/// - `ssm_beta.weight` → beta projection [hidden, num_v_heads]
/// - `ssm_alpha.weight` → alpha projection [hidden, num_v_heads]
/// - `ssm_conv1d.weight` → conv1d [kernel_size, qkvz_dim] (conv on ALL of q+k+v+z)
/// - `ssm_a` → A_log [num_v_heads]
/// - `ssm_dt.bias` → dt_bias [num_v_heads]
/// - `ssm_norm.weight` → RMSNorm weight [head_v_dim]
/// - `ssm_out.weight` → output projection [value_dim, hidden]
///
/// Key difference from safetensors: GGUF conv1d operates on all of q+k+v+z
/// (conv_dim = 2*key_dim + 2*value_dim), while safetensors only convs q+k+v.
struct QuantizedGatedDeltaNet {
    in_proj_qkvz: QuantizedLinear,
    /// Beta projection (separate in GGUF, was fused in_proj_ba in safetensors).
    beta_proj: QuantizedLinear,
    /// Alpha projection (separate in GGUF).
    alpha_proj: QuantizedLinear,
    /// Conv1d kernel weights, shape `[qkvz_dim, kernel_size]`.
    conv1d_weight: Tensor,
    /// Per-head learnable decay parameter.
    a_log: Tensor,
    /// Per-head learnable timestep bias.
    dt_bias: Tensor,
    /// RMSNorm weight for gated normalization, shape `[head_v_dim]`.
    norm_weight: Tensor,
    out_proj: QuantizedLinear,
    norm_eps: f64,

    // Config-derived constants.
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    #[allow(dead_code)]
    key_dim: usize,
    value_dim: usize,
    /// Total qkvz dimension (2*key_dim + 2*value_dim). Conv operates on this.
    qkvz_dim: usize,
    conv_kernel_size: usize,

    // Mutable recurrent state.
    conv_state: RefCell<Option<Tensor>>,
    ssm_state: RefCell<Option<Tensor>>,
}

impl QuantizedGatedDeltaNet {
    fn load(
        gguf: &mut GgufFile,
        prefix: &str,
        config: &Qwen3NextConfig,
        device: &Device,
    ) -> ModelResult<Self> {
        let in_proj_qkvz =
            QuantizedLinear::from_gguf(gguf, &format!("{prefix}.attn_qkv.weight"), device)?;
        let beta_proj =
            QuantizedLinear::from_gguf(gguf, &format!("{prefix}.ssm_beta.weight"), device)?;
        let alpha_proj =
            QuantizedLinear::from_gguf(gguf, &format!("{prefix}.ssm_alpha.weight"), device)?;

        // Conv1d weight: GGUF stores as [kernel_size, qkvz_dim].
        // We need [qkvz_dim, kernel_size] for our element-wise conv implementation.
        let conv1d_raw = gguf
            .tensor(&format!("{prefix}.ssm_conv1d.weight"), device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize conv1d: {e}")))?;
        let conv1d_dims = conv1d_raw.dims().to_vec();
        let conv1d_weight = if conv1d_dims.len() == 3 {
            conv1d_raw.squeeze(1).map_err(ModelError::Candle)?
        } else if conv1d_dims[0] < conv1d_dims[1] {
            // [kernel_size, qkvz_dim] → transpose to [qkvz_dim, kernel_size]
            conv1d_raw.t().map_err(ModelError::Candle)?
        } else {
            conv1d_raw
        };

        // Derive qkvz_dim from conv1d weight shape.
        let qkvz_dim = conv1d_weight.dim(0).map_err(ModelError::Candle)?;

        // Derive head_k_dim from qkvz_dim:
        // qkvz_dim = 2*key_dim + 2*value_dim
        // key_dim = (qkvz_dim - 2*value_dim) / 2
        // head_k_dim = key_dim / num_k_heads
        let value_dim = config.value_dim();
        let key_dim = (qkvz_dim - 2 * value_dim) / 2;
        let head_k_dim = key_dim / config.linear_num_key_heads;

        let a_log = gguf
            .tensor(&format!("{prefix}.ssm_a"), device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize ssm_a: {e}")))?;
        let dt_bias = gguf
            .tensor(&format!("{prefix}.ssm_dt.bias"), device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize dt_bias: {e}")))?;
        let norm_weight = gguf
            .tensor(&format!("{prefix}.ssm_norm.weight"), device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize ssm_norm: {e}")))?;
        let out_proj =
            QuantizedLinear::from_gguf(gguf, &format!("{prefix}.ssm_out.weight"), device)?;

        Ok(Self {
            in_proj_qkvz,
            beta_proj,
            alpha_proj,
            conv1d_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
            norm_eps: config.rms_norm_eps,
            num_k_heads: config.linear_num_key_heads,
            num_v_heads: config.linear_num_value_heads,
            head_k_dim,
            head_v_dim: config.linear_value_head_dim,
            key_dim,
            value_dim,
            qkvz_dim,
            conv_kernel_size: config.linear_conv_kernel_dim,
            conv_state: RefCell::new(None),
            ssm_state: RefCell::new(None),
        })
    }

    /// Reset recurrent state (call between sequences).
    pub fn reset_state(&self) {
        *self.conv_state.borrow_mut() = None;
        *self.ssm_state.borrow_mut() = None;
    }

    /// Extract (take) the recurrent state.
    pub fn extract_state(&self) -> Option<(Tensor, Tensor)> {
        let conv = self.conv_state.borrow_mut().take();
        let ssm = self.ssm_state.borrow_mut().take();
        match (conv, ssm) {
            (Some(c), Some(s)) => Some((c, s)),
            _ => None,
        }
    }

    /// Inject previously-saved recurrent state.
    pub fn inject_state(&self, state: &Option<(Tensor, Tensor)>) {
        match state {
            Some((c, s)) => {
                *self.conv_state.borrow_mut() = Some(c.clone());
                *self.ssm_state.borrow_mut() = Some(s.clone());
            }
            None => {
                *self.conv_state.borrow_mut() = None;
                *self.ssm_state.borrow_mut() = None;
            }
        }
    }

    /// Forward pass.
    ///
    /// GGUF layout: `attn_qkv` projects to `[seq, qkvz_dim]` where
    /// `qkvz_dim = 2*key_dim + 2*value_dim`. Conv1d operates on ALL of qkvz
    /// (unlike safetensors which only convs q+k+v). After conv+SiLU, split
    /// into q, k, v, z.
    fn forward(&self, hidden_states: &Tensor) -> ModelResult<Tensor> {
        let num_tokens = hidden_states.dim(0).map_err(ModelError::Candle)?;
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();

        // --- 1. Input projections ---
        let proj_qkvz = self
            .in_proj_qkvz
            .forward(hidden_states)
            .map_err(ModelError::Candle)?; // [seq, qkvz_dim]

        // Separate alpha/beta projections (GGUF splits them unlike safetensors).
        let beta_out = self
            .beta_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?; // [seq, num_v_heads]
        let alpha_out = self
            .alpha_proj
            .forward(hidden_states)
            .map_err(ModelError::Candle)?; // [seq, num_v_heads]

        // --- 2. Causal conv1d with SiLU on ALL of qkvz ---
        let conv_out = self.causal_conv1d(&proj_qkvz, num_tokens, dtype, device)?;
        // conv_out: [seq, qkvz_dim]

        // --- 3. Split conv output into q, k, v, z ---
        // Layout: [q(key_dim), k(key_dim), v(value_dim), z(value_dim)]
        // We split using the grouped layout matching the safetensors convention.
        let v_per_k = self.num_v_heads / self.num_k_heads;
        let per_group = self.head_k_dim
            + self.head_k_dim
            + v_per_k * self.head_v_dim
            + v_per_k * self.head_v_dim;
        let conv_grouped = conv_out
            .reshape((num_tokens, self.num_k_heads, per_group))
            .map_err(ModelError::Candle)?;

        let q_grouped = conv_grouped
            .narrow(2, 0, self.head_k_dim)
            .map_err(ModelError::Candle)?;
        let k_grouped = conv_grouped
            .narrow(2, self.head_k_dim, self.head_k_dim)
            .map_err(ModelError::Candle)?;
        let v_grouped = conv_grouped
            .narrow(2, 2 * self.head_k_dim, v_per_k * self.head_v_dim)
            .map_err(ModelError::Candle)?;
        let z_grouped = conv_grouped
            .narrow(
                2,
                2 * self.head_k_dim + v_per_k * self.head_v_dim,
                v_per_k * self.head_v_dim,
            )
            .map_err(ModelError::Candle)?;

        let q_heads = q_grouped
            .reshape((num_tokens, self.num_k_heads, self.head_k_dim))
            .map_err(ModelError::Candle)?;
        let k_heads = k_grouped
            .reshape((num_tokens, self.num_k_heads, self.head_k_dim))
            .map_err(ModelError::Candle)?;
        let v_heads = v_grouped
            .reshape((num_tokens, self.num_v_heads, self.head_v_dim))
            .map_err(ModelError::Candle)?;
        let z = z_grouped
            .reshape((num_tokens, self.value_dim))
            .map_err(ModelError::Candle)?;

        let b = beta_out;
        let a = alpha_out;

        // --- 4. Compute gating: g and beta ---
        let a_plus_bias = a.broadcast_add(&self.dt_bias).map_err(ModelError::Candle)?;
        let sp = softplus(&a_plus_bias).map_err(ModelError::Candle)?;
        let a_exp = self.a_log.to_dtype(DType::F32)?.exp()?;
        let g = sp
            .to_dtype(DType::F32)?
            .broadcast_mul(&a_exp)?
            .neg()?
            .to_dtype(dtype)
            .map_err(ModelError::Candle)?;
        let beta = tensor_sigmoid(&b).map_err(ModelError::Candle)?;

        // --- 5. Gated delta rule recurrence ---
        let output = self.gated_delta_recurrence(
            &q_heads, &k_heads, &v_heads, &g, &beta, num_tokens, dtype, device,
        )?;

        // --- 6. RMSNormGated: norm(output) * sigmoid(z) ---
        let z = z
            .reshape((num_tokens, self.num_v_heads, self.head_v_dim))
            .map_err(ModelError::Candle)?;
        let normed = self.rms_norm_gated(&output, &z)?;

        // --- 7. Output projection ---
        let normed_flat = normed
            .reshape((num_tokens, self.value_dim))
            .map_err(ModelError::Candle)?;
        self.out_proj
            .forward(&normed_flat)
            .map_err(ModelError::Candle)
    }

    /// Causal conv1d with SiLU activation. Updates conv_state.
    fn causal_conv1d(
        &self,
        mixed_qkv: &Tensor,
        num_tokens: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Tensor> {
        let k = self.conv_kernel_size;
        let mut conv_st = self.conv_state.borrow_mut();

        if num_tokens == 1 {
            let state = match conv_st.take() {
                Some(s) => s,
                None => Tensor::zeros((k - 1, self.qkvz_dim), dtype, device)
                    .map_err(ModelError::Candle)?,
            };
            let full = Tensor::cat(&[&state, mixed_qkv], 0).map_err(ModelError::Candle)?;
            let full_t = full.t().map_err(ModelError::Candle)?;
            let out = full_t
                .broadcast_mul(&self.conv1d_weight)
                .map_err(ModelError::Candle)?
                .sum(1)
                .map_err(ModelError::Candle)?;
            let out = out.silu().map_err(ModelError::Candle)?;
            *conv_st = Some(full.narrow(0, 1, k - 1).map_err(ModelError::Candle)?);
            out.unsqueeze(0).map_err(ModelError::Candle)
        } else {
            let pad = match conv_st.take() {
                Some(s) => s,
                None => Tensor::zeros((k - 1, self.qkvz_dim), dtype, device)
                    .map_err(ModelError::Candle)?,
            };
            let padded = Tensor::cat(&[&pad, mixed_qkv], 0).map_err(ModelError::Candle)?;
            let mut outputs = Vec::with_capacity(num_tokens);
            for t in 0..num_tokens {
                let window = padded.narrow(0, t, k).map_err(ModelError::Candle)?;
                let window_t = window.t().map_err(ModelError::Candle)?;
                let out = window_t
                    .broadcast_mul(&self.conv1d_weight)?
                    .sum(1)
                    .map_err(ModelError::Candle)?;
                outputs.push(out);
            }
            let output = Tensor::stack(&outputs, 0).map_err(ModelError::Candle)?;
            let output = output.silu().map_err(ModelError::Candle)?;
            let start = num_tokens.saturating_sub(k - 1);
            *conv_st = Some(
                mixed_qkv
                    .narrow(0, start, num_tokens.min(k - 1))
                    .map_err(ModelError::Candle)?,
            );
            Ok(output)
        }
    }

    /// Gated delta rule recurrence.
    #[allow(clippy::too_many_arguments)]
    fn gated_delta_recurrence(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        num_tokens: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Tensor> {
        let mut ssm_st = self.ssm_state.borrow_mut();
        let mut state = match ssm_st.take() {
            Some(s) => s,
            None => Tensor::zeros(
                (self.num_v_heads, self.head_v_dim, self.head_k_dim),
                DType::F32,
                device,
            )
            .map_err(ModelError::Candle)?,
        };

        let mut outputs = Vec::with_capacity(num_tokens);

        for t in 0..num_tokens {
            let q_t = q.get(t).map_err(ModelError::Candle)?;
            let k_t = k.get(t).map_err(ModelError::Candle)?;
            let v_t = v.get(t).map_err(ModelError::Candle)?;
            let g_t = g.get(t).map_err(ModelError::Candle)?;
            let beta_t = beta.get(t).map_err(ModelError::Candle)?;

            let q_t = l2_normalize(&q_t).map_err(ModelError::Candle)?;
            let k_t = l2_normalize(&k_t).map_err(ModelError::Candle)?;

            let g_f32 = g_t.to_dtype(DType::F32).map_err(ModelError::Candle)?;
            let beta_f32 = beta_t.to_dtype(DType::F32).map_err(ModelError::Candle)?;
            let k_f32 = k_t.to_dtype(DType::F32).map_err(ModelError::Candle)?;
            let v_f32 = v_t.to_dtype(DType::F32).map_err(ModelError::Candle)?;
            let q_f32 = q_t.to_dtype(DType::F32).map_err(ModelError::Candle)?;

            let mut head_outputs = Vec::with_capacity(self.num_v_heads);

            for h_v in 0..self.num_v_heads {
                let h_k = h_v * self.num_k_heads / self.num_v_heads;

                let g_h = g_f32
                    .get(h_v)
                    .map_err(ModelError::Candle)?
                    .exp()
                    .map_err(ModelError::Candle)?;
                let beta_h = beta_f32.get(h_v).map_err(ModelError::Candle)?;

                let k_head = k_f32.get(h_k).map_err(ModelError::Candle)?;
                let v_head = v_f32.get(h_v).map_err(ModelError::Candle)?;

                let v_col = v_head.unsqueeze(1).map_err(ModelError::Candle)?;
                let k_row = k_head.unsqueeze(0).map_err(ModelError::Candle)?;
                let outer = v_col.broadcast_mul(&k_row).map_err(ModelError::Candle)?;

                let s_h = state.get(h_v).map_err(ModelError::Candle)?;
                let new_s = (s_h.broadcast_mul(&g_h)? + outer.broadcast_mul(&beta_h)?)
                    .map_err(ModelError::Candle)?;

                let q_head = q_f32.get(h_k).map_err(ModelError::Candle)?;
                let o_h = new_s
                    .matmul(&q_head.unsqueeze(1).map_err(ModelError::Candle)?)
                    .map_err(ModelError::Candle)?
                    .squeeze(1)
                    .map_err(ModelError::Candle)?;

                state = state
                    .slice_assign(
                        &[h_v..h_v + 1, 0..self.head_v_dim, 0..self.head_k_dim],
                        &new_s.unsqueeze(0).map_err(ModelError::Candle)?,
                    )
                    .map_err(ModelError::Candle)?;

                head_outputs.push(o_h.to_dtype(dtype).map_err(ModelError::Candle)?);
            }

            let token_output = Tensor::stack(&head_outputs, 0).map_err(ModelError::Candle)?;
            outputs.push(token_output);
        }

        *ssm_st = Some(state);
        Tensor::stack(&outputs, 0).map_err(ModelError::Candle)
    }

    /// RMSNormGated: rms_norm(x) * weight * sigmoid(z).
    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor) -> ModelResult<Tensor> {
        let x_f32 = x.to_dtype(DType::F32).map_err(ModelError::Candle)?;
        let variance = x_f32
            .sqr()
            .map_err(ModelError::Candle)?
            .mean_keepdim(candle_core::D::Minus1)
            .map_err(ModelError::Candle)?;
        let rsqrt = (variance + self.norm_eps)
            .map_err(ModelError::Candle)?
            .sqrt()
            .map_err(ModelError::Candle)?
            .recip()
            .map_err(ModelError::Candle)?;
        let normed = x_f32
            .broadcast_mul(&rsqrt)
            .map_err(ModelError::Candle)?
            .to_dtype(x.dtype())
            .map_err(ModelError::Candle)?;
        let normed = normed
            .broadcast_mul(&self.norm_weight)
            .map_err(ModelError::Candle)?;
        let z_sigmoid = tensor_sigmoid(z).map_err(ModelError::Candle)?;
        normed.broadcast_mul(&z_sigmoid).map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// QuantizedQwen3NextDecoderLayer
// ---------------------------------------------------------------------------

enum QuantizedAttnVariant {
    FullAttention(QuantizedQwen3NextAttention),
    LinearAttention(QuantizedGatedDeltaNet),
}

struct QuantizedQwen3NextDecoderLayer {
    attn: QuantizedAttnVariant,
    mlp: QuantizedLlamaMLP,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
}

impl QuantizedQwen3NextDecoderLayer {
    fn load(
        gguf: &mut GgufFile,
        prefix: &str,
        config: &Qwen3NextConfig,
        layer_idx: usize,
        cos: Tensor,
        sin: Tensor,
        device: &Device,
    ) -> ModelResult<Self> {
        let attn = if config.is_full_attention(layer_idx) {
            QuantizedAttnVariant::FullAttention(QuantizedQwen3NextAttention::load(
                gguf, prefix, config, cos, sin, device,
            )?)
        } else {
            QuantizedAttnVariant::LinearAttention(QuantizedGatedDeltaNet::load(
                gguf, prefix, config, device,
            )?)
        };

        let mlp = QuantizedLlamaMLP::load(gguf, prefix, device)?;

        // GGUF uses `post_attention_norm` instead of `ffn_norm` for Qwen3.5.
        let input_layernorm = load_gemma_rms_norm(
            gguf,
            &format!("{prefix}.attn_norm.weight"),
            config.rms_norm_eps,
            device,
        )?;
        let post_attention_layernorm = load_gemma_rms_norm(
            gguf,
            &format!("{prefix}.post_attention_norm.weight"),
            config.rms_norm_eps,
            device,
        )?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn is_full_attention(&self) -> bool {
        matches!(self.attn, QuantizedAttnVariant::FullAttention(_))
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
        positions: &Tensor,
        kv_cache: Option<crate::LayerKvHandle<'_>>,
    ) -> ModelResult<(Tensor, Tensor)> {
        // Pre-attention layernorm.
        let (normed, residual) = if let Some(residual) = residual {
            crate::ops::fused_add_gemma_rms_norm(hidden_states, residual, &self.input_layernorm)
                .map_err(ModelError::Candle)?
        } else {
            let normed = crate::ops::gemma_rms_norm(hidden_states, &self.input_layernorm)
                .map_err(ModelError::Candle)?;
            (normed, hidden_states.clone())
        };

        let attn_output = match &self.attn {
            QuantizedAttnVariant::FullAttention(attn) => {
                attn.forward(&normed, positions, kv_cache)?
            }
            QuantizedAttnVariant::LinearAttention(gdn) => gdn.forward(&normed)?,
        };

        // Fused residual add + post-attention layernorm.
        let (normed, residual) = crate::ops::fused_add_gemma_rms_norm(
            &attn_output,
            &residual,
            &self.post_attention_layernorm,
        )
        .map_err(ModelError::Candle)?;

        let mlp_output = self.mlp.forward(&normed).map_err(ModelError::Candle)?;
        Ok((mlp_output, residual))
    }

    fn reset_recurrent_state(&self) {
        if let QuantizedAttnVariant::LinearAttention(gdn) = &self.attn {
            gdn.reset_state();
        }
    }

    fn extract_recurrent_state(&self) -> Option<Option<(Tensor, Tensor)>> {
        match &self.attn {
            QuantizedAttnVariant::LinearAttention(gdn) => Some(gdn.extract_state()),
            QuantizedAttnVariant::FullAttention(_) => None,
        }
    }

    fn inject_recurrent_state(&self, state: &Option<(Tensor, Tensor)>) {
        if let QuantizedAttnVariant::LinearAttention(gdn) = &self.attn {
            gdn.inject_state(state);
        }
    }
}

// ---------------------------------------------------------------------------
// QuantizedQwen3NextModel
// ---------------------------------------------------------------------------

struct QuantizedQwen3NextModel {
    embed_tokens: Embedding,
    layers: Vec<QuantizedQwen3NextDecoderLayer>,
    norm: GemmaRmsNorm,
    num_attn_layers: usize,
}

impl QuantizedQwen3NextModel {
    fn load(gguf: &mut GgufFile, config: &Qwen3NextConfig, device: &Device) -> ModelResult<Self> {
        // Embedding: dequantize to f32.
        let embed_weight = gguf
            .tensor("token_embd.weight", device)?
            .dequantize(device)
            .map_err(|e| ModelError::Other(format!("dequantize embedding: {e}")))?;
        let embed_tokens = Embedding::new(embed_weight);

        // Precompute interleaved RoPE cos/sin tables.
        // Only used by full attention layers — rotary_dim is partial.
        let rotary_dim = (config.head_dim as f64 * config.partial_rotary_factor).round() as usize;
        let rotary_dim = rotary_dim - (rotary_dim % 2);
        let (cos, sin) = precompute_freqs_cis(
            rotary_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(QuantizedQwen3NextDecoderLayer::load(
                gguf,
                &format!("blk.{i}"),
                config,
                i,
                cos.clone(),
                sin.clone(),
                device,
            )?);
        }

        // Final norm: dequantize.
        let norm = load_gemma_rms_norm(gguf, "output_norm.weight", config.rms_norm_eps, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            num_attn_layers: config.num_full_attention_layers(),
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        mut kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let mut hidden_states = self
            .embed_tokens
            .forward(input_ids)
            .map_err(ModelError::Candle)?;

        let mut residual: Option<Tensor> = None;
        let mut kv_slot = 0;
        for layer in &self.layers {
            let layer_handle = if layer.is_full_attention() {
                let h = kv_cache.as_mut().map(|s| s.layer_handle(kv_slot));
                kv_slot += 1;
                h
            } else {
                None
            };
            let (hs, res) =
                layer.forward(&hidden_states, residual.as_ref(), positions, layer_handle)?;
            hidden_states = hs;
            residual = Some(res);
        }

        let (normed, _) = crate::ops::fused_add_gemma_rms_norm(
            &hidden_states,
            residual.as_ref().unwrap(),
            &self.norm,
        )
        .map_err(ModelError::Candle)?;
        Ok(normed)
    }
}

// ---------------------------------------------------------------------------
// QuantizedQwen3NextForCausalLM
// ---------------------------------------------------------------------------

pub struct QuantizedQwen3NextForCausalLM {
    model: QuantizedQwen3NextModel,
    lm_head: QuantizedLinear,
}

impl QuantizedQwen3NextForCausalLM {
    pub fn load(
        gguf: &mut GgufFile,
        config: &Qwen3NextConfig,
        device: &Device,
    ) -> ModelResult<Self> {
        let model = QuantizedQwen3NextModel::load(gguf, config, device)?;

        let has_output = gguf.tensor_names().contains(&"output.weight");
        let lm_head = if has_output {
            QuantizedLinear::from_gguf(gguf, "output.weight", device)?
        } else {
            let qt = gguf.tensor("token_embd.weight", device)?;
            QuantizedLinear::from_qtensor(qt)?
        };

        Ok(Self { model, lm_head })
    }
}

impl crate::Model for QuantizedQwen3NextForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut crate::KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        let hidden_states = self.model.forward(input_ids, positions, kv_cache)?;
        let logits = self
            .lm_head
            .forward(&hidden_states)
            .map_err(ModelError::Candle)?;
        logits.to_dtype(DType::F32).map_err(ModelError::Candle)
    }

    fn num_layers(&self) -> usize {
        self.model.num_attn_layers
    }

    fn hidden_states(&self, input_ids: &Tensor, positions: &Tensor) -> ModelResult<Tensor> {
        self.model.forward(input_ids, positions, None)
    }

    fn reset_recurrent_state(&self) {
        for layer in &self.model.layers {
            layer.reset_recurrent_state();
        }
    }

    fn num_recurrent_layers(&self) -> usize {
        self.model
            .layers
            .iter()
            .filter(|l| matches!(l.attn, QuantizedAttnVariant::LinearAttention(_)))
            .count()
    }

    fn extract_recurrent_state(&self) -> crate::RecurrentState {
        self.model
            .layers
            .iter()
            .filter_map(|l| l.extract_recurrent_state())
            .collect()
    }

    fn inject_recurrent_state(&self, state: &[Option<(Tensor, Tensor)>]) {
        let mut idx = 0;
        for layer in &self.model.layers {
            if matches!(layer.attn, QuantizedAttnVariant::LinearAttention(_)) {
                if let Some(s) = state.get(idx) {
                    layer.inject_recurrent_state(s);
                }
                idx += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Factory function for the GGUF registry
// ---------------------------------------------------------------------------

/// Create a quantized Qwen3-Next model from a GGUF file.
pub fn create_qwen3_next_gguf(
    gguf: &mut GgufFile,
    config: &HfModelConfig,
    device: &Device,
) -> ModelResult<Box<dyn crate::Model>> {
    let next_config = Qwen3NextConfig::from_hf_config(config)?;
    let model = QuantizedQwen3NextForCausalLM::load(gguf, &next_config, device)?;
    Ok(Box::new(model))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen3_next_config_from_hf_gguf_style() {
        // Simulate config derived from GGUF metadata for 0.8B model.
        let mut extra = std::collections::HashMap::new();
        extra.insert("linear_conv_kernel_dim".to_string(), serde_json::json!(4));
        extra.insert("linear_key_head_dim".to_string(), serde_json::json!(64));
        extra.insert("linear_value_head_dim".to_string(), serde_json::json!(128));
        extra.insert("linear_num_key_heads".to_string(), serde_json::json!(16));
        extra.insert("linear_num_value_heads".to_string(), serde_json::json!(16));
        extra.insert("partial_rotary_factor".to_string(), serde_json::json!(0.25));

        let hf_config = HfModelConfig {
            hidden_size: Some(1024),
            num_attention_heads: Some(8),
            num_key_value_heads: Some(2),
            num_hidden_layers: Some(24),
            intermediate_size: Some(3584),
            vocab_size: Some(248320),
            max_position_embeddings: Some(262144),
            rms_norm_eps: Some(1e-6),
            rope_theta: Some(10000000.0),
            head_dim: Some(256),
            extra,
            ..Default::default()
        };

        let config = Qwen3NextConfig::from_hf_config(&hf_config).unwrap();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.linear_conv_kernel_dim, 4);
        assert_eq!(config.linear_num_value_heads, 16);
        assert_eq!(config.key_dim(), 16 * 64); // 1024
        assert_eq!(config.value_dim(), 16 * 128); // 2048
        assert_eq!(config.conv_dim(), 2 * 1024 + 2048); // 4096
        // Default layer_types: every 4th layer (index 3, 7, 11, ...) is full_attention.
        assert!(config.is_linear_attention(0));
        assert!(config.is_full_attention(3));
        assert_eq!(config.num_full_attention_layers(), 6);
    }
}
