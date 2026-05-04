// SPDX-License-Identifier: Apache-2.0
//! Gated Delta Net (GDN) — linear-attention layer used by Qwen3-Next.
//!
//! Two surfaces live here:
//! 1. [`GdnStatePool`] — per-request GPU-resident `(conv_state, ssm_state)`
//!    tensors shared across every GDN layer in the model. Indexed by
//!    `slot * num_gdn_layers + gdn_layer_idx`.
//! 2. [`Qwen3NextGdnLayer`] — per-layer weights + a `forward` that
//!    runs the seven kernel calls mapping the Python
//!    `Qwen3NextGatedDeltaNet` body to GPU code: input projections,
//!    QKVZ split, causal conv1d, fused gating, fused recurrent
//!    delta-rule, RMSNormGated, output projection.
//!
//! Re-exported from `vllm_cuda::model::qwen3_next` for the
//! pre-ferrite hand-written path; the canonical home is here so the
//! ferrite forward dispatcher (`Instruction::GdnAttention`) can name
//! these types without circular crate dependencies.

#![cfg(feature = "cuda")]

use anyhow::Result;
use ferrite_cuda_core::CUstream;
use ferrite_cuda_core::alloc::OwnedTensor;
use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::driver;
use ferrite_cuda_core::dtype::DType;
use ferrite_cuda_core::dump;
use ferrite_cuda_core::tensor::{GpuTensor, TensorView};
use ferrite_cuda_core::weights::GpuWeights;

use crate::kernels;
use crate::layers::Linear;

/// GPU-resident GDN state pool for all requests across all GDN layers.
///
/// Layout:
/// - `conv_states`: `[num_slots * num_gdn_layers, conv_dim, kernel_size - 1]` (f32)
/// - `ssm_states`:  `[num_slots * num_gdn_layers, num_v_heads * head_v_dim, head_k_dim]` (f32)
///
/// Slots are indexed by request. When a request completes, its slot
/// is freed (zeroed via [`GdnStatePool::clear_slot`]).
pub struct GdnStatePool {
    /// `[num_slots * num_gdn_layers, conv_dim, kernel_size - 1]` (f32 on GPU).
    pub conv_states: GpuTensor,
    /// `[num_slots * num_gdn_layers, num_v_heads * head_v_dim, head_k_dim]` (f32 on GPU).
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
    ///
    /// `conv_dim` = `2 * (num_k_heads * head_k_dim) + (num_v_heads * head_v_dim)` —
    /// the caller computes it from the model config and passes it in
    /// so this module stays config-agnostic.
    pub unsafe fn new(
        num_slots: usize,
        num_gdn_layers: usize,
        conv_dim: usize,
        kernel_size: usize,
        num_v_heads: usize,
        head_v_dim: usize,
        head_k_dim: usize,
        stream: CUstream,
    ) -> Result<Self> {
        let state_len = kernel_size - 1;

        let conv_elems = num_slots * num_gdn_layers * conv_dim * state_len;
        let ssm_elems = num_slots * num_gdn_layers * num_v_heads * head_v_dim * head_k_dim;

        // f32 storage for both buffers (matches Python vLLM mamba ssm dtype).
        const F32_BYTES: usize = 4;
        let conv_bytes = conv_elems * F32_BYTES;
        let ssm_bytes = ssm_elems * F32_BYTES;

        let conv_ptr = driver::mem_alloc(conv_bytes)?;
        let ssm_ptr = driver::mem_alloc(ssm_bytes)?;

        driver::memset_d8(conv_ptr, 0, conv_bytes, stream)?;
        driver::memset_d8(ssm_ptr, 0, ssm_bytes, stream)?;

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

    /// Zero out all GDN state (conv + ssm) for a given slot.
    ///
    /// Call this when a new sequence is assigned to the slot — the
    /// recurrent state must start from zeros so a recycled slot
    /// doesn't leak the previous request's history into the new one.
    pub unsafe fn clear_slot(&self, slot_idx: usize, stream: CUstream) -> Result<()> {
        const F32_BYTES: usize = 4;
        let state_len = self.kernel_size - 1;
        let conv_bytes_per_layer = self.conv_dim * state_len * F32_BYTES;
        let ssm_bytes_per_layer = self.num_v_heads * self.head_v_dim * self.head_k_dim * F32_BYTES;

        for layer in 0..self.num_gdn_layers {
            let flat_idx = slot_idx * self.num_gdn_layers + layer;
            let conv_offset = flat_idx * conv_bytes_per_layer;
            let conv_ptr = self.conv_states.raw_ptr().add(conv_offset);
            driver::memset_d8(conv_ptr, 0, conv_bytes_per_layer, stream)?;

            let ssm_offset = flat_idx * ssm_bytes_per_layer;
            let ssm_ptr = self.ssm_states.raw_ptr().add(ssm_offset);
            driver::memset_d8(ssm_ptr, 0, ssm_bytes_per_layer, stream)?;
        }
        Ok(())
    }
}

/// Per-layer weights for a Qwen3-Next Gated Delta Net (GDN) linear
/// attention block. The `forward` method runs the full pipeline; all
/// intermediate buffers are caching-allocator owned so the call site
/// only sees the OwnedTensor output (model dtype, `[T, hidden]`).
pub struct Qwen3NextGdnLayer {
    /// Combined Q, K, V, Z projection: `[2*key_dim + 2*value_dim, hidden]`.
    pub in_proj_qkvz: Linear,
    /// B and A projection: `[2*num_v_heads, hidden]`.
    pub in_proj_ba: Linear,
    /// Conv1d weight: `[conv_dim, kernel_size]` (f32 on GPU).
    pub conv1d_weight: GpuTensor,
    /// `A_log`: `[num_v_heads]` (f32 on GPU).
    pub a_log: GpuTensor,
    /// `dt_bias`: `[num_v_heads]` (f32 on GPU).
    pub dt_bias: GpuTensor,
    /// Output norm weight: `[head_v_dim]` (f32 on GPU).
    pub norm_weight: GpuTensor,
    /// Output projection: `[hidden, value_dim]`.
    pub out_proj: Linear,
    pub norm_eps: f32,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_dim: usize,
    pub conv_kernel_size: usize,
    /// Sequential index among GDN layers (NOT the transformer-layer
    /// index). Used to address into [`GdnStatePool`].
    pub gdn_layer_idx: usize,
    /// Model dtype (so the eval can cast the f32 GDN output back to
    /// the model's working dtype before `out_proj`).
    pub model_dtype: DType,
}

impl Qwen3NextGdnLayer {
    /// Upload a CPU f32 vec to GPU as a `GpuTensor`.
    unsafe fn upload_f32(data: &[f32], shape: &[usize], stream: CUstream) -> Result<GpuTensor> {
        const F32_BYTES: usize = 4;
        let nbytes = data.len() * F32_BYTES;
        let ptr = driver::mem_alloc(nbytes)?;
        let host = driver::mem_alloc_host(nbytes)?;
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, host, nbytes);
        driver::memcpy_htod_async(ptr, host, nbytes, stream)?;
        driver::stream_synchronize(stream)?;
        driver::mem_free_host(host)?;
        Ok(GpuTensor::new(ptr, shape, DType::F32))
    }

    /// Upload a CPU i32 vec to GPU as a `GpuTensor` (I32).
    unsafe fn upload_i32(data: &[i32], shape: &[usize], stream: CUstream) -> Result<GpuTensor> {
        const I32_BYTES: usize = 4;
        let nbytes = data.len() * I32_BYTES;
        let ptr = driver::mem_alloc(nbytes)?;
        let host = driver::mem_alloc_host(nbytes)?;
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, host, nbytes);
        driver::memcpy_htod_async(ptr, host, nbytes, stream)?;
        driver::stream_synchronize(stream)?;
        driver::mem_free_host(host)?;
        Ok(GpuTensor::new(ptr, shape, DType::I32))
    }

    /// Read `[N]` i32 values from a device tensor onto the host.
    unsafe fn download_to_cpu_i32(t: TensorView<'_>, stream: CUstream) -> Vec<i32> {
        const I32_BYTES: usize = 4;
        let n = t.dim(0);
        let nbytes = n * I32_BYTES;
        let host = driver::mem_alloc_host(nbytes).expect("download_to_cpu_i32: alloc host");
        driver::memcpy_dtoh_async(host, t.raw_ptr(), nbytes, stream)
            .expect("download_to_cpu_i32: dtoh");
        driver::stream_synchronize(stream).expect("download_to_cpu_i32: sync");
        let mut out = vec![0i32; n];
        std::ptr::copy_nonoverlapping(host as *const i32, out.as_mut_ptr(), n);
        driver::mem_free_host(host).expect("download_to_cpu_i32: free host");
        out
    }

    /// Load GDN weights from safetensors.
    ///
    /// Mirrors `Qwen3NextGatedDeltaNet.__init__` weight wiring in
    /// `vllm/model_executor/models/qwen3_next.py`. The conv1d kernel,
    /// gating parameters (`A_log`, `dt_bias`), and the gated output
    /// norm are stored as f32 on device because the GDN kernels
    /// consume them at f32 precision.
    pub fn load(
        gw: &mut GpuWeights,
        prefix: &str,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel_size: usize,
        rms_norm_eps: f32,
        gdn_layer_idx: usize,
        stream: CUstream,
    ) -> Result<Self> {
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;

        let qkvz_w = gw.take(&format!("{prefix}.in_proj_qkvz.weight"))?;
        let in_proj_qkvz = Linear::new(qkvz_w, None);

        let ba_w = gw.take(&format!("{prefix}.in_proj_ba.weight"))?;
        let in_proj_ba = Linear::new(ba_w, None);

        let conv_data = gw.take_to_cpu_f32(&format!("{prefix}.conv1d.weight"))?;
        let conv_elems = conv_dim * conv_kernel_size;
        anyhow::ensure!(
            conv_data.len() >= conv_elems,
            "conv1d weight too small: {} < {}",
            conv_data.len(),
            conv_elems,
        );
        let conv1d_weight = unsafe {
            Self::upload_f32(
                &conv_data[..conv_elems],
                &[conv_dim, conv_kernel_size],
                stream,
            )?
        };

        let a_log_data = gw.take_to_cpu_f32(&format!("{prefix}.A_log"))?;
        let a_log = unsafe { Self::upload_f32(&a_log_data, &[a_log_data.len()], stream)? };

        let dt_bias_data = gw.take_to_cpu_f32(&format!("{prefix}.dt_bias"))?;
        let dt_bias = unsafe { Self::upload_f32(&dt_bias_data, &[dt_bias_data.len()], stream)? };

        let norm_data = gw.take_to_cpu_f32(&format!("{prefix}.norm.weight"))?;
        let norm_weight = unsafe { Self::upload_f32(&norm_data, &[norm_data.len()], stream)? };

        let out_proj = Linear::load(gw, &format!("{prefix}.out_proj"))?;
        let model_dtype = out_proj.weight.dtype();

        Ok(Self {
            in_proj_qkvz,
            in_proj_ba,
            conv1d_weight,
            a_log,
            dt_bias,
            norm_weight,
            out_proj,
            norm_eps: rms_norm_eps,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size,
            gdn_layer_idx,
            model_dtype,
        })
    }

    /// GDN forward pass — returns `[T, hidden]` in `model_dtype`.
    ///
    /// Pipeline:
    /// 1. `in_proj_qkvz` + `in_proj_ba` GEMMs.
    /// 2. `gdn_qkvz_split` — fan out QKVZ/BA into Q, K, V, Z, A, B and
    ///    `mixed = concat(Q, K, V)` for the conv1d input.
    /// 3. `gdn_conv1d_update` (decode) / `gdn_conv1d_prefill` (prefill).
    /// 4. `gdn_conv_split` — fan the conv-out back into Q, K, V.
    /// 5. `gdn_gating` — produce `(g, beta)` from `A_log`, `a`, `b`, `dt_bias`.
    /// 6. `gdn_recurrent_fwd` — apply the gated delta-rule recurrence
    ///    against `state_pool.ssm_states`.
    /// 7. `gdn_rms_norm_gated` — z-gated RMSNorm, then cast to model dtype.
    /// 8. `out_proj` GEMM.
    ///
    /// `state_indices` is `[num_seqs] i32` — the per-request slot
    /// IDs. `cu_seqlens` is `[num_seqs + 1] i32`. The per-layer
    /// adjusted index is `slot * num_gdn_layers + gdn_layer_idx`,
    /// computed here when `num_gdn_layers > 1` and passed to the
    /// kernels via a temporary upload (matches the existing
    /// hand-written path; a fully-on-GPU adjustment is a follow-up).
    ///
    /// # Safety
    /// `hidden_states`, `state_indices`, `cu_seqlens`, and the
    /// pool's tensors must all be valid GPU views in the same
    /// CUDA context as `device.compute_stream`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        state_pool: &GdnStatePool,
        state_indices: TensorView<'_>,
        cu_seqlens: TensorView<'_>,
        num_seqs: usize,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        // 1. Input projections.
        let qkvz =
            self.in_proj_qkvz
                .forward(hidden_states, &mut device.cublas, &mut device.caching);
        let ba = self
            .in_proj_ba
            .forward(hidden_states, &mut device.cublas, &mut device.caching);
        dump::dump_tile(
            "gdn.in_proj_qkvz",
            self.gdn_layer_idx as u32,
            &qkvz.as_gpu_tensor(),
            stream,
        );
        dump::dump_tile(
            "gdn.in_proj_ba",
            self.gdn_layer_idx as u32,
            &ba.as_gpu_tensor(),
            stream,
        );

        // 2. QKVZ + BA fan-out (also produces mixed = Q||K||V for conv1d).
        let (_q_split, _k_split, _v_split, z_owned, a_owned, b_owned, mixed_owned) =
            kernels::gdn_qkvz_split(
                *qkvz.view(),
                *ba.view(),
                num_tokens,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.key_dim,
                self.value_dim,
                self.conv_dim,
                &mut device.caching,
                stream,
            );
        drop(qkvz);
        drop(ba);

        let mixed_qkv = *mixed_owned.view();
        let a_gpu = *a_owned.view();
        let b_gpu = *b_owned.view();
        let z_gpu = *z_owned.view();
        dump::dump_tile("gdn.split.z", self.gdn_layer_idx as u32, &z_gpu, stream);
        dump::dump_tile("gdn.split.a", self.gdn_layer_idx as u32, &a_gpu, stream);
        dump::dump_tile("gdn.split.b", self.gdn_layer_idx as u32, &b_gpu, stream);
        dump::dump_tile(
            "gdn.split.mixed_qkv",
            self.gdn_layer_idx as u32,
            &mixed_qkv,
            stream,
        );

        // Per-layer index adjustment for the shared state pool.
        let num_gdn_layers = state_pool.num_gdn_layers;
        let adjusted_indices_owned: Option<GpuTensor> = if num_gdn_layers > 1 {
            let host = Self::download_to_cpu_i32(state_indices, stream);
            let adjusted: Vec<i32> = host
                .iter()
                .map(|&s| s * num_gdn_layers as i32 + self.gdn_layer_idx as i32)
                .collect();
            Some(
                Self::upload_i32(&adjusted, &[num_seqs], stream)
                    .expect("upload adjusted gdn indices"),
            )
        } else {
            None
        };
        let eff_state_indices = adjusted_indices_owned
            .as_ref()
            .map_or(*state_indices, |t| *t);

        // 3. Causal conv1d.
        let conv_out = device
            .caching
            .alloc_tensor(&[num_tokens, self.conv_dim], DType::F32);
        if num_seqs == num_tokens {
            // Decode path.
            kernels::gdn_conv1d_update(
                state_pool.conv_states,
                mixed_qkv,
                self.conv1d_weight,
                *conv_out.view(),
                eff_state_indices,
                self.conv_dim,
                self.conv_kernel_size,
                num_seqs,
                stream,
            );
        } else {
            // Prefill path: per-sequence slice (CPU-driven loop matches
            // the existing hand-written path; collapsing into a single
            // launch is a follow-up optimization).
            const F32_BYTES: usize = 4;
            let cu_seqlens_cpu = Self::download_to_cpu_i32(cu_seqlens, stream);
            let state_indices_cpu = Self::download_to_cpu_i32(state_indices, stream);

            for s in 0..num_seqs {
                let seq_start = cu_seqlens_cpu[s] as usize;
                let seq_end = cu_seqlens_cpu[s + 1] as usize;
                let seq_len = seq_end - seq_start;
                if seq_len == 0 {
                    continue;
                }
                let raw_slot = state_indices_cpu[s] as usize;
                let slot_idx = raw_slot * num_gdn_layers + self.gdn_layer_idx;
                let byte_offset = seq_start * self.conv_dim * F32_BYTES;
                let x_view = GpuTensor::new(
                    mixed_qkv.raw_ptr().add(byte_offset),
                    &[seq_len, self.conv_dim],
                    DType::F32,
                );
                let out_view = GpuTensor::new(
                    conv_out.view().raw_ptr().add(byte_offset),
                    &[seq_len, self.conv_dim],
                    DType::F32,
                );
                kernels::gdn_conv1d_prefill(
                    state_pool.conv_states,
                    x_view,
                    self.conv1d_weight,
                    out_view,
                    slot_idx,
                    self.conv_dim,
                    self.conv_kernel_size,
                    seq_len,
                    stream,
                );
            }
        }

        dump::dump_tile(
            "gdn.conv_out",
            self.gdn_layer_idx as u32,
            &conv_out.view(),
            stream,
        );

        // 4. Conv-output split into post-conv Q, K, V.
        let (q_owned, k_owned, v_owned) = kernels::gdn_conv_split(
            *conv_out.view(),
            num_tokens,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.key_dim,
            self.value_dim,
            self.conv_dim,
            &mut device.caching,
            stream,
        );
        drop(conv_out);

        let q_gpu = *q_owned.view();
        let k_gpu = *k_owned.view();
        let v_gpu = *v_owned.view();
        dump::dump_tile("gdn.post_conv.q", self.gdn_layer_idx as u32, &q_gpu, stream);
        dump::dump_tile("gdn.post_conv.k", self.gdn_layer_idx as u32, &k_gpu, stream);
        dump::dump_tile("gdn.post_conv.v", self.gdn_layer_idx as u32, &v_gpu, stream);

        // 5. Fused gating → (g, beta).
        let g_gpu = device
            .caching
            .alloc_tensor(&[num_tokens, self.num_v_heads], DType::F32);
        let beta_gpu = device
            .caching
            .alloc_tensor(&[num_tokens, self.num_v_heads], DType::F32);
        kernels::gdn_gating(
            *g_gpu.view(),
            *beta_gpu.view(),
            self.a_log,
            a_gpu,
            b_gpu,
            self.dt_bias,
            self.num_v_heads,
            num_tokens,
            stream,
        );
        dump::dump_tile(
            "gdn.gating.g",
            self.gdn_layer_idx as u32,
            &g_gpu.view(),
            stream,
        );
        dump::dump_tile(
            "gdn.gating.beta",
            self.gdn_layer_idx as u32,
            &beta_gpu.view(),
            stream,
        );

        // 6. Fused recurrent forward (gated delta-rule).
        let o_gpu = device
            .caching
            .alloc_tensor(&[num_tokens, self.num_v_heads, self.head_v_dim], DType::F32);
        // Scale = 1.0; the kernel applies its own L2-norm scaling.
        const RECURRENT_SCALE: f32 = 1.0;
        kernels::gdn_recurrent_fwd(
            q_gpu,
            k_gpu,
            v_gpu,
            *g_gpu.view(),
            *beta_gpu.view(),
            *o_gpu.view(),
            state_pool.ssm_states,
            eff_state_indices,
            *cu_seqlens,
            RECURRENT_SCALE,
            num_seqs,
            num_tokens,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            stream,
        );
        dump::dump_tile(
            "gdn.recurrent.o",
            self.gdn_layer_idx as u32,
            &o_gpu.view(),
            stream,
        );
        drop(g_gpu);
        drop(beta_gpu);

        // 7. RMSNormGated → cast f32 → model_dtype.
        let total_rows = num_tokens * self.num_v_heads;
        let o_flat = o_gpu.view().reshape(&[total_rows, self.head_v_dim]);
        let z_flat = z_gpu.reshape(&[total_rows, self.head_v_dim]);
        let normed = device
            .caching
            .alloc_tensor(&[total_rows, self.head_v_dim], DType::F32);
        kernels::gdn_rms_norm_gated(
            *o_flat,
            z_flat,
            self.norm_weight,
            *normed.view(),
            self.norm_eps,
            self.head_v_dim,
            total_rows,
            stream,
        );
        dump::dump_tile(
            "gdn.normed",
            self.gdn_layer_idx as u32,
            &normed.view(),
            stream,
        );
        drop(o_gpu);

        let normed_flat = normed.view().reshape(&[num_tokens, self.value_dim]);
        let proj_input = if self.model_dtype != DType::F32 {
            kernels::cast_from_f32(*normed_flat, self.model_dtype, &mut device.caching, stream)
        } else {
            normed
        };

        // 8. Output projection.
        let result = self.out_proj.forward(
            proj_input.view().reshape(&[num_tokens, self.value_dim]),
            &mut device.cublas,
            &mut device.caching,
        );
        drop(proj_input);

        // The temporary index-adjustment tensor is owned only by us.
        // Release the underlying device buffer; no other path holds it.
        if let Some(t) = adjusted_indices_owned {
            // SAFETY: `adjusted_indices_owned` was allocated above by
            // `upload_i32` via `driver::mem_alloc`; it has no other
            // owner.
            driver::mem_free(t.raw_ptr()).expect("free adjusted gdn indices");
        }

        result
    }
}
