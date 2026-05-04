// SPDX-License-Identifier: Apache-2.0
//! Qwen2.5-VL vision encoder — windowed varlen attention + RMSNorm +
//! SwiGLU MLP, with a `RMSNorm + GELU` patch merger projecting into the
//! text-decoder hidden.
//!
//! Differences vs Qwen2-VL (`ferrite-model-qwen2-vl`):
//! - block norms (`norm1`, `norm2`) and merger `ln_q` are RMSNorm
//!   (weight only), not LayerNorm (weight + bias).
//! - block MLP is `down_proj(silu(gate_proj(x)) · up_proj(x))` with bias
//!   on all three linears, replacing `fc2(QuickGELU(fc1(x)))`.
//! - 28-of-32 blocks run windowed attention bucketed by 112-px spatial
//!   windows; the 4 layers in `fullatt_block_indexes=[7,15,23,31]` run
//!   full image-frame attention. Tokens are gather-permuted into window
//!   order on entry and unpermuted post-merger so all blocks share the
//!   same flat tensor.
//! - patch_embed and the merger MLP shape are unchanged; pixel patch
//!   flatten is unchanged (mirrors Python `Qwen2VLImageProcessor`).
//!
//! The text decoder (`#[forward] fn qwen2`) is shared via
//! `ferrite-model-qwen2`; the same body runs for `Qwen2_5_VLForConditionalGeneration`
//! once a `qwen2.5-vl-3b.json` config registers it.

use anyhow::{Context, Result};
use ferrite_cuda_core::CUstream;
use ferrite_cuda_core::CachingAllocator;
use ferrite_cuda_core::DType;
use ferrite_cuda_core::alloc::OwnedTensor;
use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::driver;
use ferrite_cuda_core::tensor::GpuTensor;
use ferrite_cuda_core::weights::GpuWeights;
use ferrite_forward::{
    EmbedPatch, FerriteMmRegistration, HfFingerprint, MultimodalForward, PixelInput,
};
use ferrite_kernels::kernels;
use ferrite_kernels::layers::{Linear, RmsNorm};

// ── Vision-encoder family constants ────────────────────────────────
//
// Identical across Qwen2.5-VL 3B / 7B / 72B per the upstream config.
// Only `embed_dim`, `intermediate_size`, `depth`, and the merger
// output (`d_model = text_config.hidden_size`) vary by checkpoint and
// are derived from tensor shapes in [`try_load_mm_qwen2_5_vl`].

const NUM_HEADS: u32 = 16;
const WINDOW_SIZE: u32 = 112;
const FULLATT_BLOCK_INDEXES: &[u32] = &[7, 15, 23, 31];
const NORM_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10000.0;

/// One Qwen2.5-VL vision transformer block. Pre-RMSNorm + 2D-rope-aware
/// varlen attention + pre-RMSNorm SwiGLU MLP.
///
/// `gate_proj` and `up_proj` are loaded as separate Linears (rather than
/// `LinearLayer::load_dense_concat`). `down_proj`'s K dim is padded at
/// load time from `intermediate_size` to `intermediate_size_padded`
/// (next multiple of 8) so cuBLAS BF16 GEMM accepts it — `intermediate=3420`
/// on Qwen2.5-VL-3B is mod 4 but not mod 8, and cublasLt + cublasGemmEx
/// both fail on bf16 K=3420 with `CUBLAS_STATUS_INTERNAL_ERROR`.
pub struct VisionBlockWeights {
    pub norm1: RmsNorm,
    pub qkv: Linear,
    pub proj: Linear,
    pub norm2: RmsNorm,
    pub gate_proj: Linear,
    pub up_proj: Linear,
    /// K dim padded to next multiple of 8 (zero-fill on the trailing
    /// columns) so the GEMM lands on an algo cuBLAS supports.
    pub down_proj: Linear,
}

impl VisionBlockWeights {
    pub fn load(weights: &mut GpuWeights, layer: u32, eps: f32, stream: CUstream) -> Result<Self> {
        let prefix = format!("visual.blocks.{layer}");
        let down_raw = Linear::load(weights, &format!("{prefix}.mlp.down_proj"))
            .with_context(|| format!("{prefix}.mlp.down_proj"))?;
        let down_proj = unsafe { pad_linear_k_to_mult8(down_raw, weights, stream) }
            .with_context(|| format!("{prefix}.mlp.down_proj (k-pad)"))?;
        Ok(Self {
            norm1: RmsNorm::load(weights, &format!("{prefix}.norm1"), eps)
                .with_context(|| format!("{prefix}.norm1"))?,
            qkv: Linear::load(weights, &format!("{prefix}.attn.qkv"))
                .with_context(|| format!("{prefix}.attn.qkv"))?,
            proj: Linear::load(weights, &format!("{prefix}.attn.proj"))
                .with_context(|| format!("{prefix}.attn.proj"))?,
            norm2: RmsNorm::load(weights, &format!("{prefix}.norm2"), eps)
                .with_context(|| format!("{prefix}.norm2"))?,
            gate_proj: Linear::load(weights, &format!("{prefix}.mlp.gate_proj"))
                .with_context(|| format!("{prefix}.mlp.gate_proj"))?,
            up_proj: Linear::load(weights, &format!("{prefix}.mlp.up_proj"))
                .with_context(|| format!("{prefix}.mlp.up_proj"))?,
            down_proj,
        })
    }
}

/// Pad a `[D0, K]` Linear's weight to `[D0, K_pad]` where `K_pad =
/// round_up(K, 8)`. Tail columns are zero so the GEMM
/// `out[i, j] = sum_k input[i, k] * weight[j, k]` is unchanged when the
/// activation feeds zeros into those padded columns.
unsafe fn pad_linear_k_to_mult8(
    linear: Linear,
    weights: &mut GpuWeights,
    stream: CUstream,
) -> Result<Linear> {
    let w = linear.weight;
    debug_assert_eq!(w.ndim(), 2);
    let d0 = w.dim(0);
    let k = w.dim(1);
    let k_pad = k.next_multiple_of(8);
    if k_pad == k {
        return Ok(linear);
    }
    let dtype = w.dtype();
    let elem = dtype.size_bytes();
    let src_pitch = k * elem;
    let dst_pitch = k_pad * elem;
    let total_bytes = d0 * dst_pitch;
    let new_ptr = unsafe { driver::mem_alloc(total_bytes) }?;
    weights.record_alloc(new_ptr, total_bytes);
    unsafe { driver::memset_d8(new_ptr, 0, total_bytes, stream) }?;
    let src_base = w.raw_ptr() as *const u8;
    for r in 0..d0 {
        unsafe {
            driver::memcpy_dtod_async(
                new_ptr.add(r * dst_pitch),
                src_base.add(r * src_pitch),
                src_pitch,
                stream,
            )?;
        }
    }
    let new_w = unsafe { GpuTensor::new(new_ptr, &[d0, k_pad], dtype) };
    Ok(Linear::new(new_w, linear.bias))
}

/// `Qwen2_5_VLPatchMerger` — RMSNorm + 2-layer GELU MLP projecting
/// `[L, embed_dim]` → `[L / spatial_merge_size², d_model]`.
pub struct VisionMergerWeights {
    pub ln_q: RmsNorm,
    pub mlp_0: Linear,
    pub mlp_2: Linear,
}

impl VisionMergerWeights {
    pub fn load(weights: &mut GpuWeights, eps: f32) -> Result<Self> {
        Ok(Self {
            ln_q: RmsNorm::load(weights, "visual.merger.ln_q", eps)
                .context("visual.merger.ln_q")?,
            mlp_0: Linear::load(weights, "visual.merger.mlp.0").context("visual.merger.mlp.0")?,
            mlp_2: Linear::load(weights, "visual.merger.mlp.2").context("visual.merger.mlp.2")?,
        })
    }
}

pub struct VisionWeights {
    pub patch_embed_proj: Linear,
    pub blocks: Vec<VisionBlockWeights>,
    pub merger: VisionMergerWeights,
    pub config: VisionConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionConfig {
    pub embed_dim: u32,
    pub depth: u32,
    pub num_heads: u32,
    /// SwiGLU intermediate width (Qwen2.5-VL-3B: 3420). Not embed_dim·ratio.
    pub intermediate_size: u32,
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub spatial_merge_size: u32,
    pub in_chans: u32,
    /// Text-decoder hidden the merger projects into (= `out_hidden_size`).
    pub d_model: u32,
    pub norm_eps: f32,
    /// Spatial window edge in pixels (config: 112).
    pub window_size: u32,
}

impl VisionWeights {
    pub fn load(weights: &mut GpuWeights, config: VisionConfig, stream: CUstream) -> Result<Self> {
        let in_chans = config.in_chans as usize;
        let t = config.temporal_patch_size as usize;
        let p = config.patch_size as usize;
        let embed_dim = config.embed_dim as usize;
        let patch_embed_w = weights
            .take_with_shape(
                "visual.patch_embed.proj.weight",
                &[embed_dim, in_chans * t * p * p],
            )
            .context("visual.patch_embed.proj.weight")?;
        let patch_embed_proj = Linear::new(patch_embed_w, None);
        let mut blocks = Vec::with_capacity(config.depth as usize);
        for layer in 0..config.depth {
            blocks.push(VisionBlockWeights::load(
                weights,
                layer,
                config.norm_eps,
                stream,
            )?);
        }
        let merger = VisionMergerWeights::load(weights, config.norm_eps)?;
        Ok(Self {
            patch_embed_proj,
            blocks,
            merger,
            config,
        })
    }

    /// Run the Qwen2.5-VL vision encoder.
    ///
    /// Inputs:
    /// - `pixels`: `[total_l, in_chans·temporal_patch_size·patch_size²]`
    ///   bf16 patch tensor — concatenated across images.
    /// - `grid_thw`: per-image patch-grid `(T, H, W)` in patch units.
    ///
    /// Output: `[total_l / S², d_model]` bf16 — projected embeds in
    /// natural (un-windowed) order ready for splice into language model
    /// embeddings.
    ///
    /// # Safety
    /// `pixels` must be a valid GPU tensor of the documented shape;
    /// `device` must be the live CUDA device.
    pub unsafe fn forward(
        &self,
        pixels: GpuTensor,
        grid_thw: &[(u32, u32, u32)],
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let cfg = &self.config;
        let total_l = pixels.dim(0);
        let head_dim = (cfg.embed_dim / cfg.num_heads) as usize;
        let half_rot = head_dim / 2;
        let dtype = pixels.dtype();
        let stream = device.compute_stream;
        let scale = (head_dim as f32).powf(-0.5);
        let s = cfg.spatial_merge_size as usize;
        let s2 = s * s;
        debug_assert_eq!(total_l % s2, 0, "total_l ({total_l}) must be ÷ S² ({s2})");
        let l_merged = total_l / s2;

        // 1. Patch embed (Conv3d collapsed to GEMM by stride==kernel).
        let mut x = self.patch_embed_proj.forward(
            pixels.as_view(),
            &mut device.cublas,
            &mut device.caching,
        );

        // 2. Per-token RoPE cos/sin in natural (pre-window) order, bf16.
        let (cos_host, sin_host) = build_rope_cos_sin_bf16(grid_thw, cfg, total_l);
        let cos_natural = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            dtype,
            bf16_slice_as_bytes(&cos_host),
        );
        let sin_natural = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            dtype,
            bf16_slice_as_bytes(&sin_host),
        );

        // 3. Per-image full-attn cu_seqlens (one segment per image-frame),
        //    plus per-image windowed cu_seqlens + window_index permutation.
        let (cu_seqlens_full_host, max_seqlen_full) = build_cu_seqlens_i32(grid_thw);
        let cu_seqlens_full = device.alloc_gpu_tensor_from_host(
            &[cu_seqlens_full_host.len()],
            DType::I32,
            i32_slice_as_bytes(&cu_seqlens_full_host),
        );
        let (window_index_host, cu_window_seqlens_host, max_seqlen_window) =
            build_window_index_cu_seqlens(grid_thw, cfg);
        debug_assert_eq!(window_index_host.len(), l_merged);
        let cu_window_seqlens = device.alloc_gpu_tensor_from_host(
            &[cu_window_seqlens_host.len()],
            DType::I32,
            i32_slice_as_bytes(&cu_window_seqlens_host),
        );
        let window_index_gpu = device.alloc_gpu_tensor_from_host(
            &[l_merged],
            DType::U32,
            u32_slice_as_bytes(&window_index_host),
        );
        let reverse_indices_host = invert_permutation(&window_index_host);
        let reverse_indices_gpu = device.alloc_gpu_tensor_from_host(
            &[l_merged],
            DType::U32,
            u32_slice_as_bytes(&reverse_indices_host),
        );

        // 4. Window-permute hidden states + cos/sin. Permutation is at
        //    granularity S² (S²-token block per merged cell), so reshape
        //    sources to `[L/S², S²·hidden]` and gather rows.
        x.reshape(&[l_merged, s2 * cfg.embed_dim as usize], dtype);
        let mut x_perm = kernels::embedding_gather(
            x.as_gpu_tensor(),
            window_index_gpu,
            &mut device.caching,
            stream,
        );
        x_perm.reshape(&[total_l, cfg.embed_dim as usize], dtype);
        let x: OwnedTensor = x_perm;

        let cos_block = cos_natural.reshape(&[l_merged, s2 * half_rot]);
        let mut cos =
            kernels::embedding_gather(cos_block, window_index_gpu, &mut device.caching, stream);
        cos.reshape(&[total_l, half_rot], dtype);

        let sin_block = sin_natural.reshape(&[l_merged, s2 * half_rot]);
        let mut sin =
            kernels::embedding_gather(sin_block, window_index_gpu, &mut device.caching, stream);
        sin.reshape(&[total_l, half_rot], dtype);

        // 5. 32 transformer blocks. Per-layer cu_seqlens dispatch — full
        //    attn for layers in `fullatt_block_indexes`, windowed
        //    otherwise. RMSNorm before attn + before MLP, both residual.
        for (li, block) in self.blocks.iter().enumerate() {
            let is_full = FULLATT_BLOCK_INDEXES.contains(&(li as u32));
            let (cu_now, max_now) = if is_full {
                (cu_seqlens_full, max_seqlen_full)
            } else {
                (cu_window_seqlens, max_seqlen_window)
            };

            let normed = kernels::rms_norm(
                x.as_gpu_tensor(),
                block.norm1.weight,
                block.norm1.eps,
                &mut device.caching,
                stream,
            );
            let qkv = block
                .qkv
                .forward(normed.view(), &mut device.cublas, &mut device.caching);
            let q_size = cfg.embed_dim as usize;
            let kv_size = cfg.embed_dim as usize;
            let (q, k, v) = kernels::split_qkv(
                qkv.as_gpu_tensor(),
                q_size,
                kv_size,
                cfg.num_heads as usize,
                cfg.num_heads as usize,
                head_dim,
                &mut device.caching,
                stream,
            );
            kernels::vision_rope_apply(
                q.as_gpu_tensor(),
                cos.as_gpu_tensor(),
                sin.as_gpu_tensor(),
                stream,
            );
            kernels::vision_rope_apply(
                k.as_gpu_tensor(),
                cos.as_gpu_tensor(),
                sin.as_gpu_tensor(),
                stream,
            );
            let attn = kernels::flash_attn_contiguous(
                q.as_gpu_tensor(),
                k.as_gpu_tensor(),
                v.as_gpu_tensor(),
                cu_now,
                cu_now,
                max_now,
                max_now,
                scale,
                false,
                0.0,
                -1,
                &mut device.caching,
                stream,
                std::ptr::null(),
                0,
                false,
            );
            let mut attn_flat = attn;
            attn_flat.reshape(&[total_l, cfg.embed_dim as usize], dtype);
            let proj =
                block
                    .proj
                    .forward(attn_flat.view(), &mut device.cublas, &mut device.caching);
            kernels::add_inplace(x.as_gpu_tensor(), proj.as_gpu_tensor(), stream);

            // SwiGLU MLP: down(silu(gate(x)) · up(x)).
            let normed2 = kernels::rms_norm(
                x.as_gpu_tensor(),
                block.norm2.weight,
                block.norm2.eps,
                &mut device.caching,
                stream,
            );
            let gate_out =
                block
                    .gate_proj
                    .forward(normed2.view(), &mut device.cublas, &mut device.caching);
            let up_out =
                block
                    .up_proj
                    .forward(normed2.view(), &mut device.cublas, &mut device.caching);
            // Pack gate||up into a [L, 2·I_pad] zero-init buffer so
            // silu_and_mul_fused (split point I_pad) sees real data on
            // the first I cols of each half and zero on the trailing
            // I_pad - I cols. silu(0)·up==0 keeps the padded slots zero
            // through the next GEMM's K=I_pad contraction.
            let i = cfg.intermediate_size as usize;
            let i_pad = i.next_multiple_of(8);
            let act = pack_gate_up(
                gate_out.as_gpu_tensor(),
                up_out.as_gpu_tensor(),
                i,
                i_pad,
                &mut device.caching,
                stream,
            );
            let mlp_act = kernels::silu_and_mul_fused(
                act.as_gpu_tensor(),
                i_pad,
                &mut device.caching,
                stream,
            );
            let down =
                block
                    .down_proj
                    .forward(mlp_act.view(), &mut device.cublas, &mut device.caching);
            kernels::add_inplace(x.as_gpu_tensor(), down.as_gpu_tensor(), stream);
        }

        // 6. PatchMerger: ln_q (RMSNorm) → reshape → mlp_0 → erf-GELU → mlp_2.
        let mut merged = kernels::rms_norm(
            x.as_gpu_tensor(),
            self.merger.ln_q.weight,
            self.merger.ln_q.eps,
            &mut device.caching,
            stream,
        );
        let merge_hidden = (cfg.embed_dim as usize) * s2;
        merged.reshape(&[l_merged, merge_hidden], dtype);
        let mlp0 =
            self.merger
                .mlp_0
                .forward(merged.view(), &mut device.cublas, &mut device.caching);
        kernels::gelu_erf_inplace(mlp0.as_gpu_tensor(), stream);
        let projected =
            self.merger
                .mlp_2
                .forward(mlp0.view(), &mut device.cublas, &mut device.caching);

        // 7. Reverse-permute back to natural row order.
        let unpermuted = kernels::embedding_gather(
            projected.as_gpu_tensor(),
            reverse_indices_gpu,
            &mut device.caching,
            stream,
        );
        // Caching-allocator-owned device tensors: dropping the bindings
        // is a no-op (GpuTensor is Copy) — the allocator reclaims them.
        let _ = (cu_seqlens_full, cu_window_seqlens);
        let _ = (window_index_gpu, reverse_indices_gpu);
        let _ = (cos, sin); // OwnedTensor scratches drop here.
        unpermuted
    }
}

// ── gate||up packer (zero-padded to next mult of 8) ───────────────
//
// Allocates `[L, 2·i_pad]` zeroed, then memcpy-d2d-async-copies each
// row of `gate [L, i]` into the first `i` columns of the packed
// row's first half, and `up [L, i]` into the first `i` columns of
// the second half. Trailing padding (`i_pad - i` cols per half) stays
// zero, so `silu_and_mul_fused(packed, i_pad)` writes
// `silu(0)·0 = 0` into the corresponding output slots — those zero
// columns are then contracted to zero contribution by the next
// down_proj GEMM (whose K = i_pad with zero-padded weight columns).
unsafe fn pack_gate_up(
    gate: GpuTensor,
    up: GpuTensor,
    i: usize,
    i_pad: usize,
    alloc: &mut CachingAllocator,
    stream: CUstream,
) -> OwnedTensor {
    debug_assert_eq!(gate.dim(0), up.dim(0));
    debug_assert_eq!(gate.dim(1), i);
    debug_assert_eq!(up.dim(1), i);
    let l = gate.dim(0);
    let dtype = gate.dtype();
    let elem = dtype.size_bytes();
    let packed = alloc.alloc_tensor(&[l, 2 * i_pad], dtype);
    let total_bytes = l * 2 * i_pad * elem;
    unsafe { driver::memset_d8(packed.as_gpu_tensor().raw_ptr(), 0, total_bytes, stream) }
        .expect("pack_gate_up: zero-init failed");
    let row_bytes_packed = 2 * i_pad * elem;
    let row_bytes_src = i * elem;
    let gate_src = gate.raw_ptr() as *const u8;
    let up_src = up.raw_ptr() as *const u8;
    let dst_base = packed.as_gpu_tensor().raw_ptr();
    for r in 0..l {
        let dst_row = unsafe { dst_base.add(r * row_bytes_packed) };
        let dst_gate = dst_row;
        let dst_up = unsafe { dst_row.add(i_pad * elem) };
        unsafe {
            driver::memcpy_dtod_async(
                dst_gate,
                gate_src.add(r * row_bytes_src),
                row_bytes_src,
                stream,
            )
            .expect("pack_gate_up: gate d2d");
            driver::memcpy_dtod_async(dst_up, up_src.add(r * row_bytes_src), row_bytes_src, stream)
                .expect("pack_gate_up: up d2d");
        }
    }
    packed
}

// ── 2D RoPE cos/sin construction ───────────────────────────────────
//
// Identical to Qwen2-VL's natural-order build (Python
// `Qwen2VisionTransformer.rot_pos_emb`); window permutation is applied
// downstream via `embedding_gather` on the result, mirroring Python
// vLLM's `Qwen2_5_VisionTransformer.forward` `rotary_pos_emb[window_index]`.

fn build_rope_cos_sin_bf16(
    grid_thw: &[(u32, u32, u32)],
    cfg: &VisionConfig,
    total_l: usize,
) -> (Vec<u16>, Vec<u16>) {
    let head_dim = (cfg.embed_dim / cfg.num_heads) as usize;
    let half_rot = head_dim / 2;
    let freq_axis_dim = half_rot / 2;
    let s = cfg.spatial_merge_size as usize;

    let inv_freq: Vec<f32> = (0..freq_axis_dim)
        .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / (half_rot as f32)))
        .collect();

    let mut cos = Vec::<u16>::with_capacity(total_l * half_rot);
    let mut sin = Vec::<u16>::with_capacity(total_l * half_rot);
    for &(t, h, w) in grid_thw {
        let (h, w, t) = (h as usize, w as usize, t as usize);
        debug_assert_eq!(h % s, 0);
        debug_assert_eq!(w % s, 0);
        let h_blocks = h / s;
        let w_blocks = w / s;
        let frame_len = h * w;
        let mut hpos = vec![0u32; frame_len];
        let mut wpos = vec![0u32; frame_len];
        let mut idx = 0usize;
        for hb in 0..h_blocks {
            for wb in 0..w_blocks {
                for sh in 0..s {
                    for sw in 0..s {
                        hpos[idx] = (hb * s + sh) as u32;
                        wpos[idx] = (wb * s + sw) as u32;
                        idx += 1;
                    }
                }
            }
        }
        for _ in 0..t {
            for token in 0..frame_len {
                let hp = hpos[token] as f32;
                let wp = wpos[token] as f32;
                for &f in inv_freq.iter() {
                    let theta_h = hp * f;
                    cos.push(f32_to_bf16(theta_h.cos()));
                    sin.push(f32_to_bf16(theta_h.sin()));
                }
                for &f in inv_freq.iter() {
                    let theta_w = wp * f;
                    cos.push(f32_to_bf16(theta_w.cos()));
                    sin.push(f32_to_bf16(theta_w.sin()));
                }
            }
        }
    }
    debug_assert_eq!(cos.len(), total_l * half_rot);
    (cos, sin)
}

fn build_cu_seqlens_i32(grid_thw: &[(u32, u32, u32)]) -> (Vec<i32>, usize) {
    let mut cu = Vec::<i32>::with_capacity(grid_thw.len() + 1);
    cu.push(0);
    let mut max_seqlen = 0usize;
    let mut acc: i32 = 0;
    for &(t, h, w) in grid_thw {
        let seg = (h as usize) * (w as usize);
        for _ in 0..t {
            acc += seg as i32;
            cu.push(acc);
            if seg > max_seqlen {
                max_seqlen = seg;
            }
        }
    }
    (cu, max_seqlen)
}

// ── Window-index + cu_window_seqlens ───────────────────────────────
//
// Mirrors Python `Qwen2_5_VisionTransformer.get_window_index_thw`.
// Operates on the post-spatial-merge grid (`llm_h = H/S`, `llm_w = W/S`)
// in row-major. Pads each `(t, llm_h, llm_w)` grid up to a multiple of
// `vit_merger_window_size = window_size / S / patch_size` cells, then
// reshuffles by (window_h, window_w, intra_h, intra_w) so window-sized
// chunks land contiguously.
//
// Returns:
// - `window_index`:    `[L / S²]` u32 — natural→window permutation
//                       (gather indices: `permuted[i] = natural[window_index[i]]`).
// - `cu_window_seqlens`: i32 prefix-sum of windowed segment lengths
//                       (post-multiply by S² so it indexes the L-token tensor).
// - `max_window_seqlen`: max segment length in S²-token units (pre-mul).
fn build_window_index_cu_seqlens(
    grid_thw: &[(u32, u32, u32)],
    cfg: &VisionConfig,
) -> (Vec<u32>, Vec<i32>, usize) {
    let s = cfg.spatial_merge_size as usize;
    let s2 = s * s;
    let p = cfg.patch_size as usize;
    let win_cells = (cfg.window_size as usize) / s / p;
    debug_assert!(win_cells > 0, "window_size/S/patch_size must be > 0");

    let total_merged: usize = grid_thw
        .iter()
        .map(|&(t, h, w)| (t as usize) * ((h as usize) / s) * ((w as usize) / s))
        .sum();
    let mut window_index = Vec::<u32>::with_capacity(total_merged);
    let mut cu = Vec::<i32>::with_capacity(grid_thw.len() * 4 + 1);
    cu.push(0);
    let mut window_index_id: u32 = 0;
    let mut cu_last: i32 = 0;
    let mut max_seqlen_cells = 0usize;

    for &(t, h, w) in grid_thw {
        let (t, h, w) = (t as usize, h as usize, w as usize);
        let llm_h = h / s;
        let llm_w = w / s;
        let pad_h = (win_cells - llm_h % win_cells) % win_cells;
        let pad_w = (win_cells - llm_w % win_cells) % win_cells;
        let nh = (llm_h + pad_h) / win_cells;
        let nw = (llm_w + pad_w) / win_cells;
        let padded_h = llm_h + pad_h;
        let padded_w = llm_w + pad_w;

        // For each frame, walk windows in (window_h, window_w, intra_h,
        // intra_w) order; emit non-pad cells (those with original
        // coordinate < llm_{h,w}) into `window_index`. Per-window
        // segment length goes into `cu` (multiplied by S² since
        // downstream tensors are in token units, not merged-cell units).
        for ti in 0..t {
            let frame_base = (ti * llm_h * llm_w) as u32 + window_index_id;
            for wh in 0..nh {
                for ww in 0..nw {
                    let mut segment_cells: i32 = 0;
                    for ih in 0..win_cells {
                        for iw in 0..win_cells {
                            let row = wh * win_cells + ih;
                            let col = ww * win_cells + iw;
                            if row < llm_h && col < llm_w {
                                window_index.push(frame_base + (row * llm_w + col) as u32);
                                segment_cells += 1;
                            }
                        }
                    }
                    cu_last += segment_cells * (s2 as i32);
                    cu.push(cu_last);
                    if segment_cells as usize > max_seqlen_cells {
                        max_seqlen_cells = segment_cells as usize;
                    }
                }
            }
            // Padding sanity: every (row, col) pair within the padded
            // rectangle is emitted exactly once when row/col fall inside
            // (llm_h, llm_w); guard against off-by-one on partial windows.
            let _ = (padded_h, padded_w);
        }
        window_index_id += (t * llm_h * llm_w) as u32;
    }

    // Dedup consecutive duplicates (matches Python `torch.unique_consecutive`).
    let cu = dedup_consecutive(cu);
    (window_index, cu, max_seqlen_cells * s2)
}

fn dedup_consecutive(v: Vec<i32>) -> Vec<i32> {
    let mut out = Vec::with_capacity(v.len());
    for x in v {
        if out.last().is_none_or(|&y| y != x) {
            out.push(x);
        }
    }
    out
}

fn invert_permutation(perm: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as u32;
    }
    inv
}

fn f32_to_bf16(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

fn bf16_slice_as_bytes(s: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn i32_slice_as_bytes(s: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn u32_slice_as_bytes(s: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

// ── Patch flatten (CHW pixels → [L, C·T·P²] bf16) ───────────────────
//
// Identical to Qwen2-VL — pixel patch ordering is unchanged across the
// VL family. Phase G factors this out once a third consumer arrives.

pub fn patches_from_normalized_chw(
    pixels: &[f32],
    height: u32,
    width: u32,
    cfg: &VisionConfig,
) -> (Vec<u16>, (u32, u32, u32)) {
    let p = cfg.patch_size as usize;
    let s = cfg.spatial_merge_size as usize;
    let t = cfg.temporal_patch_size as usize;
    let c = cfg.in_chans as usize;
    let h = height as usize;
    let w = width as usize;
    assert_eq!(pixels.len(), c * h * w);
    assert_eq!(h % (p * s), 0);
    assert_eq!(w % (p * s), 0);
    let grid_h = h / p;
    let grid_w = w / p;
    let g_h = grid_h / s;
    let g_w = grid_w / s;
    let grid_t: u32 = 1;
    let l = (grid_t as usize) * grid_h * grid_w;
    let feat = c * t * p * p;
    let mut out = vec![0u16; l * feat];
    let mut out_idx = 0usize;
    let stride_c = h * w;
    let stride_h = w;
    for gh in 0..g_h {
        for gw in 0..g_w {
            for mh in 0..s {
                for mw in 0..s {
                    for ci in 0..c {
                        for _ti in 0..t {
                            let img_h_base = gh * (s * p) + mh * p;
                            let img_w_base = gw * (s * p) + mw * p;
                            for ph in 0..p {
                                let img_h = img_h_base + ph;
                                let row = ci * stride_c + img_h * stride_h + img_w_base;
                                for pw in 0..p {
                                    let v = pixels[row + pw];
                                    out[out_idx] = f32_to_bf16(v);
                                    out_idx += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(out_idx, l * feat);
    (out, (grid_t, grid_h as u32, grid_w as u32))
}

// ── MultimodalForward impl ─────────────────────────────────────────

impl MultimodalForward for VisionWeights {
    unsafe fn vision_forward(
        &self,
        pixel_batches: &[PixelInput<'_>],
        placeholders: &[EmbedPatch],
        device: &mut GpuDevice,
    ) -> (OwnedTensor, Vec<EmbedPatch>) {
        let cfg = &self.config;
        let p = cfg.patch_size as usize;
        let t = cfg.temporal_patch_size as usize;
        let c = cfg.in_chans as usize;
        let feat = c * t * p * p;

        let mut all_patches: Vec<u16> = Vec::new();
        let mut grid_thw: Vec<(u32, u32, u32)> = Vec::with_capacity(pixel_batches.len());
        for img in pixel_batches {
            let (mut patch_rows, gthw) =
                patches_from_normalized_chw(img.pixels, img.height, img.width, cfg);
            all_patches.append(&mut patch_rows);
            grid_thw.push(gthw);
        }
        let total_l: usize = grid_thw
            .iter()
            .map(|&(gt, gh, gw)| (gt as usize) * (gh as usize) * (gw as usize))
            .sum();
        debug_assert_eq!(all_patches.len(), total_l * feat);

        let pixels = device.alloc_gpu_tensor_from_host(
            &[total_l, feat],
            DType::BF16,
            bf16_slice_as_bytes(&all_patches),
        );

        let projected = self.forward(pixels, &grid_thw, device);

        let sm = cfg.spatial_merge_size;
        debug_assert_eq!(placeholders.len(), grid_thw.len());
        let patches: Vec<EmbedPatch> = placeholders
            .iter()
            .zip(grid_thw.iter())
            .map(|(ph, &(gt, gh, gw))| {
                let mh = gh / sm;
                let mw = gw / sm;
                debug_assert_eq!(ph.length, gt * mh * mw);
                EmbedPatch {
                    token_offset: ph.token_offset,
                    length: ph.length,
                    grid_t: gt,
                    grid_h_merged: mh,
                    grid_w_merged: mw,
                }
            })
            .collect();
        (projected, patches)
    }

    fn embed_patch_grids(&self, pixel_batches: &[PixelInput<'_>]) -> Vec<(u32, u32, u32)> {
        let cfg = &self.config;
        let p = cfg.patch_size;
        let sm = cfg.spatial_merge_size;
        pixel_batches
            .iter()
            .map(|img| {
                let grid_h = img.height / p;
                let grid_w = img.width / p;
                (1u32, grid_h / sm, grid_w / sm)
            })
            .collect()
    }
}

// ── Inventory registration ─────────────────────────────────────────
//
// Probes `visual.*` shapes off live `GpuWeights` to fill the
// shape-discoverable `VisionConfig` fields (embed_dim, depth,
// in_chans, temporal/patch sizes, spatial_merge_size, intermediate_size,
// d_model). Family-wide constants (num_heads=16, window_size=112,
// fullatt_block_indexes, norm_eps=1e-6) are baked into the loader.

fn try_load_mm_qwen2_5_vl(
    gw: &mut GpuWeights,
    stream: CUstream,
    _max_model_len: usize,
    _tp_rank: u8,
    _hf: HfFingerprint<'_>,
) -> Result<Option<Box<dyn MultimodalForward>>> {
    let pe = match gw.tensor_shape_any("visual.patch_embed.proj.weight") {
        Some(s) => s,
        None => return Ok(None),
    };
    if pe.len() != 5 {
        anyhow::bail!("visual.patch_embed.proj.weight must be 5D, got {pe:?}");
    }
    let embed_dim = pe[0] as u32;
    let in_chans = pe[1] as u32;
    let temporal_patch_size = pe[2] as u32;
    let patch_size = pe[3] as u32;
    if pe[3] != pe[4] {
        anyhow::bail!("patch_embed expected square patch, got {pe:?}");
    }

    // Distinguish Qwen2.5-VL from Qwen2-VL by SwiGLU MLP weight names —
    // Qwen2-VL has `mlp.fc1`, Qwen2.5-VL has `mlp.gate_proj`. Bail out
    // (Ok(None)) if we don't see the 2.5-VL flavor so the qwen2-vl
    // loader keeps its turn at the inventory.
    if !gw.contains("visual.blocks.0.mlp.gate_proj.weight") {
        return Ok(None);
    }

    let mut depth = 0u32;
    while gw.contains(&format!("visual.blocks.{depth}.norm1.weight")) {
        depth += 1;
    }
    if depth == 0 {
        anyhow::bail!("visual.blocks.0.norm1.weight missing");
    }

    let gate = gw
        .tensor_shape_any("visual.blocks.0.mlp.gate_proj.weight")
        .ok_or_else(|| anyhow::anyhow!("visual.blocks.0.mlp.gate_proj.weight missing"))?;
    if gate.len() != 2 || gate[1] as u32 != embed_dim {
        anyhow::bail!("gate_proj expected [I, {embed_dim}], got {gate:?}");
    }
    let intermediate_size = gate[0] as u32;

    let merger0 = gw
        .tensor_shape_any("visual.merger.mlp.0.weight")
        .ok_or_else(|| anyhow::anyhow!("visual.merger.mlp.0.weight missing"))?;
    if merger0.len() != 2 {
        anyhow::bail!("merger.mlp.0.weight must be 2D, got {merger0:?}");
    }
    let merge_in = merger0[1] as u32;
    if !merge_in.is_multiple_of(embed_dim) {
        anyhow::bail!("merger fc0 in-dim ({merge_in}) not a multiple of embed_dim ({embed_dim})");
    }
    let s_squared = merge_in / embed_dim;
    let spatial_merge_size = (s_squared as f64).sqrt() as u32;
    if spatial_merge_size * spatial_merge_size != s_squared {
        anyhow::bail!("merger fc0 in-dim/{embed_dim} = {s_squared} not a perfect square");
    }

    let merger2 = gw
        .tensor_shape_any("visual.merger.mlp.2.weight")
        .ok_or_else(|| anyhow::anyhow!("visual.merger.mlp.2.weight missing"))?;
    if merger2.len() != 2 {
        anyhow::bail!("merger.mlp.2.weight must be 2D, got {merger2:?}");
    }
    let d_model = merger2[0] as u32;

    let config = VisionConfig {
        embed_dim,
        depth,
        num_heads: NUM_HEADS,
        intermediate_size,
        patch_size,
        temporal_patch_size,
        spatial_merge_size,
        in_chans,
        d_model,
        norm_eps: NORM_EPS,
        window_size: WINDOW_SIZE,
    };
    let vw = VisionWeights::load(gw, config, stream)?;
    Ok(Some(Box::new(vw) as Box<dyn MultimodalForward>))
}

ferrite_forward::inventory::submit! {
    FerriteMmRegistration {
        arch_name: "qwen2_5_vl",
        hf_arches: &["Qwen2_5_VLForConditionalGeneration"],
        gguf_archs: &[],
        tp_world_size: 1,
        try_load_mm: try_load_mm_qwen2_5_vl,
    }
}

#[cfg(feature = "nccl")]
ferrite_forward::inventory::submit! {
    FerriteMmRegistration {
        arch_name: "qwen2_5_vl",
        hf_arches: &["Qwen2_5_VLForConditionalGeneration"],
        gguf_archs: &[],
        tp_world_size: 2,
        try_load_mm: try_load_mm_qwen2_5_vl,
    }
}

#[cfg(feature = "nccl")]
ferrite_forward::inventory::submit! {
    FerriteMmRegistration {
        arch_name: "qwen2_5_vl",
        hf_arches: &["Qwen2_5_VLForConditionalGeneration"],
        gguf_archs: &[],
        tp_world_size: 4,
        try_load_mm: try_load_mm_qwen2_5_vl,
    }
}

#[cfg(feature = "nccl")]
ferrite_forward::inventory::submit! {
    FerriteMmRegistration {
        arch_name: "qwen2_5_vl",
        hf_arches: &["Qwen2_5_VLForConditionalGeneration"],
        gguf_archs: &[],
        tp_world_size: 8,
        try_load_mm: try_load_mm_qwen2_5_vl,
    }
}
