// SPDX-License-Identifier: Apache-2.0
//! Qwen2-VL / Qwen2.5-VL vision encoder weights.
//!
//! Hand-coded sibling to the macro-emitted text decoder `Weights` struct.
//! The text decoder math is identical to plain Qwen2 (modulo MROPE_SECTION),
//! so it rides on the existing `#[forward] fn qwen2()` body. The vision
//! encoder doesn't fit the DSL cleanly (varlen attention, 2D rope, packed
//! QKV with bias, QuickGELU, patch-merger reshape), so it lives here as
//! plain Rust composing the `ferrite-kernels` primitives directly.
//!
//! Layout (from a Qwen2-VL-2B-Instruct safetensors checkpoint, 391
//! `visual.*` tensors):
//!
//! ```text
//! visual.patch_embed.proj.weight                  Conv3d weight, flattened
//!                                                 to [embed_dim, in_chans
//!                                                 * temporal_patch_size *
//!                                                 patch_size * patch_size]
//!                                                 = [1280, 1176].
//! visual.blocks.{0..depth-1}.norm1.{weight,bias}  LayerNorm [embed_dim].
//! visual.blocks.{0..}.attn.qkv.{weight,bias}      Packed QKV [3*embed_dim,
//!                                                 embed_dim].
//! visual.blocks.{0..}.attn.proj.{weight,bias}     [embed_dim, embed_dim].
//! visual.blocks.{0..}.norm2.{weight,bias}         LayerNorm [embed_dim].
//! visual.blocks.{0..}.mlp.fc1.{weight,bias}       [mlp_hidden, embed_dim],
//!                                                 mlp_hidden = embed_dim *
//!                                                 mlp_ratio (5120 on 2B).
//! visual.blocks.{0..}.mlp.fc2.{weight,bias}       [embed_dim, mlp_hidden].
//! visual.merger.ln_q.{weight,bias}                LayerNorm [embed_dim].
//! visual.merger.mlp.0.{weight,bias}               [merge_hidden,
//!                                                 merge_hidden],
//!                                                 merge_hidden = embed_dim
//!                                                 * spatial_merge_size**2
//!                                                 (5120 on 2B).
//! visual.merger.mlp.2.{weight,bias}               [d_model, merge_hidden]
//!                                                 (1536 on 2B; matches
//!                                                 text decoder hidden).
//! ```
//!
//! Phase D / E integration:
//!
//! - [`VisionWeights::forward`] — encoder pipeline composing the
//!   per-block kernels (lands as plain Rust, no DSL).
//! - [`build_rope_cos_sin_bf16`] / [`build_cu_seqlens_i32`] —
//!   per-call CPU-built rope grid + varlen segment table.
//! - [`patches_from_normalized_chw`] — patch-flatten matching Python
//!   `Qwen2VLImageProcessor._preprocess`'s 9D transpose.
//! - `impl MultimodalForward for VisionWeights` — bridges
//!   `ferrite_forward::PixelInput { CHW pixels, h, w }` → encoder
//!   `[total_l, in_chans*t*p*p]` patches + per-image `(grid_t, grid_h, grid_w)`.
//! - [`mm_dispatcher`] — `inventory::submit!` of a
//!   `FerriteMmRegistration` claiming `Qwen2VLForConditionalGeneration`
//!   so cuda_worker's `try_load_mm` resolves the encoder when the
//!   live `GpuWeights` carries `visual.*` tensors. Hardcoded
//!   Qwen2-VL-2B vision config for now; per-checkpoint detection via
//!   tensor shape sniff lands when a second VL variant arrives.

use anyhow::{Context, Result};
use ferrite_cuda_core::CUstream;
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
use ferrite_kernels::layers::{LayerNorm, Linear};

/// One Qwen2-VL vision transformer block. 2D-rope-aware varlen attention
/// + 2-layer QuickGELU MLP, with LayerNorm before each (pre-norm).
pub struct VisionBlockWeights {
    pub norm1: LayerNorm,
    pub qkv: Linear,
    pub proj: Linear,
    pub norm2: LayerNorm,
    pub fc1: Linear,
    pub fc2: Linear,
}

impl VisionBlockWeights {
    /// Load one block by 0-indexed `layer` from `visual.blocks.<layer>.*`.
    pub fn load(weights: &mut GpuWeights, layer: u32, eps: f32) -> Result<Self> {
        let prefix = format!("visual.blocks.{layer}");
        Ok(Self {
            norm1: LayerNorm::load(weights, &format!("{prefix}.norm1"), eps)
                .with_context(|| format!("{prefix}.norm1"))?,
            qkv: Linear::load(weights, &format!("{prefix}.attn.qkv"))
                .with_context(|| format!("{prefix}.attn.qkv"))?,
            proj: Linear::load(weights, &format!("{prefix}.attn.proj"))
                .with_context(|| format!("{prefix}.attn.proj"))?,
            norm2: LayerNorm::load(weights, &format!("{prefix}.norm2"), eps)
                .with_context(|| format!("{prefix}.norm2"))?,
            fc1: Linear::load(weights, &format!("{prefix}.mlp.fc1"))
                .with_context(|| format!("{prefix}.mlp.fc1"))?,
            fc2: Linear::load(weights, &format!("{prefix}.mlp.fc2"))
                .with_context(|| format!("{prefix}.mlp.fc2"))?,
        })
    }
}

/// Qwen2-VL `Qwen2VisionPatchMerger` — final projector that maps
/// vision-encoder output `[L, embed_dim]` → `[L / spatial_merge_size**2,
/// d_model]` (text decoder hidden).
pub struct VisionMergerWeights {
    pub ln_q: LayerNorm,
    /// `merger.mlp.0` — [merge_hidden, merge_hidden].
    pub mlp_0: Linear,
    /// `merger.mlp.2` — [d_model, merge_hidden]. (`mlp.1` is GELU, no
    /// weight.)
    pub mlp_2: Linear,
}

impl VisionMergerWeights {
    pub fn load(weights: &mut GpuWeights, eps: f32) -> Result<Self> {
        Ok(Self {
            ln_q: LayerNorm::load(weights, "visual.merger.ln_q", eps)
                .context("visual.merger.ln_q")?,
            mlp_0: Linear::load(weights, "visual.merger.mlp.0").context("visual.merger.mlp.0")?,
            mlp_2: Linear::load(weights, "visual.merger.mlp.2").context("visual.merger.mlp.2")?,
        })
    }
}

/// Qwen2-VL vision encoder weights — 32 transformer blocks bookended by
/// `patch_embed` (Conv3d → flat GEMM) and `merger` (LayerNorm + 2-layer
/// MLP). Loaded by [`VisionWeights::load`] from the `visual.*` namespace
/// of a Qwen2-VL safetensors checkpoint.
pub struct VisionWeights {
    /// `patch_embed.proj.weight` — Conv3d weight reshaped to flat GEMM.
    /// On disk shape is `[embed_dim, in_chans, temporal_patch_size,
    /// patch_size, patch_size]`; the `Linear` view treats it as
    /// `[embed_dim, in_chans * temporal_patch_size * patch_size**2]`
    /// because the runtime kernel collapses the conv to a GEMM (stride
    /// equals kernel size equals input spatial dims, so each "patch" is
    /// the full input window — degenerate conv).
    pub patch_embed_proj: Linear,
    pub blocks: Vec<VisionBlockWeights>,
    pub merger: VisionMergerWeights,

    /// Vision config carried from `config.json::vision_config`. Used by
    /// the encoder forward to size cos/sin grids, varlen cu_seqlens, and
    /// the patch-merge reshape. Stored as plain primitives so the struct
    /// stays decoupled from `vision_config` parsing.
    pub config: VisionConfig,
}

/// Subset of `vision_config` the encoder forward needs at runtime. Filled
/// from `config.json` at the executor layer (Phase E).
#[derive(Clone, Copy, Debug)]
pub struct VisionConfig {
    pub embed_dim: u32,
    pub depth: u32,
    pub num_heads: u32,
    pub mlp_ratio: u32,
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub spatial_merge_size: u32,
    pub in_chans: u32,
    /// Text-decoder hidden the patch-merger projects into. Equals
    /// `text_config.hidden_size` (1536 on Qwen2-VL-2B).
    pub d_model: u32,
    pub norm_eps: f32,
}

impl VisionWeights {
    /// Load every `visual.*` tensor by walking `0..config.depth` blocks
    /// plus the patch_embed and merger. `eps` defaults to `1e-6` per
    /// the Qwen2-VL config (`norm_eps`).
    pub fn load(weights: &mut GpuWeights, config: VisionConfig) -> Result<Self> {
        // patch_embed.proj.weight is a 5D Conv3d weight on disk
        // `[embed_dim, in_chans, t, h, w]` — overflows GpuTensor's
        // MAX_DIMS=4. Flatten to `[embed_dim, in_chans*t*h*w]` at
        // load: stride==kernel==spatial dims means the conv collapses
        // to a GEMM, and `Linear::forward` reads weight as 2D.
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
            blocks.push(VisionBlockWeights::load(weights, layer, config.norm_eps)?);
        }
        let merger = VisionMergerWeights::load(weights, config.norm_eps)?;
        Ok(Self {
            patch_embed_proj,
            blocks,
            merger,
            config,
        })
    }

    /// Run the full Qwen2-VL vision encoder.
    ///
    /// Inputs:
    /// - `pixels`: `[total_l, in_chans * temporal_patch_size * patch_size**2]`
    ///   bf16 patch tensor — concatenated across images, in spatial-merge
    ///   block order (matches Python `Qwen2VLImageProcessor` output).
    ///   `total_l = sum_i (T_i * H_i * W_i)` where each `(T_i, H_i, W_i)`
    ///   is the per-image patch grid.
    /// - `grid_thw`: per-image patch-grid shape `(T, H, W)` in patch units.
    /// - `device`: CUDA device handle (cublas + caching allocator + stream).
    ///
    /// Output: `[total_l / spatial_merge_size**2, d_model]` bf16 — the
    /// projected embeddings to splice into the language model's
    /// `embed_tokens` output at the image-placeholder positions.
    ///
    /// Pipeline matches Python `Qwen2VisionTransformer.forward`:
    ///   1. `patch_embed` (Conv3d → flat GEMM since stride==kernel).
    ///   2. Build 2D RoPE `cos`/`sin` from `grid_thw` (h-axis on first
    ///      `head_dim/2 / 2` slots, w-axis on next).
    ///   3. Build varlen `cu_seqlens` (one segment per image-frame).
    ///   4. 32 transformer blocks: pre-norm attn (qkv + 2D rope + varlen
    ///      flash-attn + proj) + pre-norm QuickGELU MLP, both residual.
    ///   5. PatchMerger: ln_q → reshape `[L, embed_dim] → [L/S^2,
    ///      S^2*embed_dim]` (S = spatial_merge_size) → mlp_0 → erf-GELU
    ///      → mlp_2.
    ///
    /// # Safety
    /// `pixels` must be a valid GPU tensor of the documented shape;
    /// `device` must be the live CUDA device the encoder kernels run on.
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
        let dump = TraceDump::from_env();

        // 1. Patch embed: [L, in_chans*t*h*w] @ [embed_dim, in_chans*t*h*w].T
        //    → [L, embed_dim].
        dump.dump_tensor("ferrite_pixels_in", pixels, stream);
        let x = self.patch_embed_proj.forward(
            pixels.as_view(),
            &mut device.cublas,
            &mut device.caching,
        );
        dump.dump_tensor("ferrite_patch_embed_out", x.as_gpu_tensor(), stream);

        // 2. Build per-token cos/sin rope tables on host, upload as bf16.
        let (cos_host, sin_host) = build_rope_cos_sin_bf16(grid_thw, cfg, total_l);
        let cos_bytes = bf16_slice_as_bytes(&cos_host);
        let sin_bytes = bf16_slice_as_bytes(&sin_host);
        let cos = device.alloc_gpu_tensor_from_host(&[total_l, half_rot], dtype, cos_bytes);
        let sin = device.alloc_gpu_tensor_from_host(&[total_l, half_rot], dtype, sin_bytes);
        dump.dump_tensor("ferrite_cos_half", cos, stream);
        dump.dump_tensor("ferrite_sin_half", sin, stream);

        // 3. Build per-image varlen cu_seqlens.
        let (cu_seqlens_host, max_seqlen) = build_cu_seqlens_i32(grid_thw);
        let cu_bytes = i32_slice_as_bytes(&cu_seqlens_host);
        let cu_seqlens =
            device.alloc_gpu_tensor_from_host(&[cu_seqlens_host.len()], DType::I32, cu_bytes);
        dump.dump_tensor("ferrite_cu_seqlens", cu_seqlens, stream);

        // 4. 32 transformer blocks. Pre-norm + residual on both attn + mlp.
        for (li, block) in self.blocks.iter().enumerate() {
            // Attention.
            let normed = kernels::layer_norm_bias(
                x.as_gpu_tensor(),
                block.norm1.weight,
                block.norm1.bias.expect("vision norm1 must have bias"),
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

            // 2D RoPE on Q and K — neox-style halves over the full
            // head_dim, with cos/sin shape `[L, head_dim/2]`. The kernel
            // does not care that the first / second cos halves come
            // from h / w axes; the math is generic and the structure is
            // baked into `cos`/`sin` by `build_rope_cos_sin_bf16`.
            kernels::vision_rope_apply(q.as_gpu_tensor(), cos, sin, stream);
            kernels::vision_rope_apply(k.as_gpu_tensor(), cos, sin, stream);

            // Varlen flash-attn (no causal mask, no rope inside the
            // kernel — we already applied rope above).
            let attn = kernels::flash_attn_contiguous(
                q.as_gpu_tensor(),
                k.as_gpu_tensor(),
                v.as_gpu_tensor(),
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                scale,
                false, // is_causal
                0.0,   // softcap
                -1,    // window_size_left
                &mut device.caching,
                stream,
                std::ptr::null(), // cos_sin_cache_ptr — rope already applied
                0,                // rotary_dim
                false,            // is_rotary_interleaved
            );
            // attn: [L, num_heads, head_dim] → reshape to [L, embed_dim] for proj.
            let mut attn_flat = attn;
            attn_flat.reshape(&[total_l, cfg.embed_dim as usize], dtype);
            let proj =
                block
                    .proj
                    .forward(attn_flat.view(), &mut device.cublas, &mut device.caching);
            kernels::add_inplace(x.as_gpu_tensor(), proj.as_gpu_tensor(), stream);

            // MLP (QuickGELU).
            let normed2 = kernels::layer_norm_bias(
                x.as_gpu_tensor(),
                block.norm2.weight,
                block.norm2.bias.expect("vision norm2 must have bias"),
                block.norm2.eps,
                &mut device.caching,
                stream,
            );
            let fc1 = block
                .fc1
                .forward(normed2.view(), &mut device.cublas, &mut device.caching);
            kernels::quick_gelu_inplace(fc1.as_gpu_tensor(), stream);
            let fc2 = block
                .fc2
                .forward(fc1.view(), &mut device.cublas, &mut device.caching);
            kernels::add_inplace(x.as_gpu_tensor(), fc2.as_gpu_tensor(), stream);
            if matches!(li, 0 | 1 | 15 | 31) {
                dump.dump_tensor(
                    &format!("ferrite_block_{li}_out"),
                    x.as_gpu_tensor(),
                    stream,
                );
            }
        }

        // 5. PatchMerger: ln_q → reshape → mlp_0 → erf-GELU → mlp_2.
        let mut merged = kernels::layer_norm_bias(
            x.as_gpu_tensor(),
            self.merger.ln_q.weight,
            self.merger.ln_q.bias.expect("merger ln_q must have bias"),
            self.merger.ln_q.eps,
            &mut device.caching,
            stream,
        );
        let merge_factor = (cfg.spatial_merge_size * cfg.spatial_merge_size) as usize;
        let l_out = total_l / merge_factor;
        let merge_hidden = (cfg.embed_dim as usize) * merge_factor;
        merged.reshape(&[l_out, merge_hidden], dtype);
        let mlp0 =
            self.merger
                .mlp_0
                .forward(merged.view(), &mut device.cublas, &mut device.caching);
        kernels::gelu_erf_inplace(mlp0.as_gpu_tensor(), stream);
        let out = self
            .merger
            .mlp_2
            .forward(mlp0.view(), &mut device.cublas, &mut device.caching);
        dump.dump_tensor("ferrite_merger_out", out.as_gpu_tensor(), stream);
        out
    }
}

// ── Trace dump (FERRITE_VIT_DUMP_DIR=<dir>) ────────────────────────
//
// When set, every `dump_tensor` call after the env was first observed
// blocks on `stream`, D2Hs the device tensor into a host buffer, and
// appends `<dir>/<name>.bin` plus a metadata line in `<dir>/dump.json`.
// Used to diff `VisionWeights::forward` intermediates against the
// Python golden generated by `tests/golden_gen_qwen2_vl_vision.py`.

struct TraceDump {
    dir: Option<std::path::PathBuf>,
}

impl TraceDump {
    fn from_env() -> Self {
        let dir = std::env::var("FERRITE_VIT_DUMP_DIR").ok().and_then(|s| {
            if s.is_empty() {
                None
            } else {
                let p = std::path::PathBuf::from(s);
                if let Err(e) = std::fs::create_dir_all(&p) {
                    eprintln!("FERRITE_VIT_DUMP_DIR: cannot create {}: {}", p.display(), e);
                    return None;
                }
                Some(p)
            }
        });
        Self { dir }
    }

    unsafe fn dump_tensor(&self, name: &str, t: GpuTensor, stream: CUstream) {
        let Some(dir) = self.dir.as_ref() else {
            return;
        };
        let bytes = t.size_bytes();
        if bytes == 0 {
            return;
        }
        let mut host = vec![0u8; bytes];
        driver::stream_synchronize(stream).expect("vit-dump: pre-D2H sync");
        driver::memcpy_dtoh_async(host.as_mut_ptr(), t.raw_ptr(), bytes, stream)
            .expect("vit-dump: D2H");
        driver::stream_synchronize(stream).expect("vit-dump: post-D2H sync");
        let bin_path = dir.join(format!("{name}.bin"));
        std::fs::write(&bin_path, &host)
            .unwrap_or_else(|e| eprintln!("vit-dump: write {} failed: {}", bin_path.display(), e));
        let meta = format!(
            "{{\"name\":\"{name}\",\"shape\":{:?},\"dtype\":\"{:?}\",\"bytes\":{bytes}}}\n",
            t.shape().iter().collect::<Vec<_>>(),
            t.dtype()
        );
        let log_path = dir.join("dump.jsonl");
        use std::io::Write as _;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let _ = f.write_all(meta.as_bytes());
        }
        eprintln!("vit-dump: {} {:?} {:?}", name, t.shape(), t.dtype());
    }
}

// ── 2D RoPE cos/sin construction ───────────────────────────────────
//
// Mirrors Python `Qwen2VisionTransformer.rot_pos_emb`:
//   for each (t, h, w) in grid_thw:
//     hpos = arange(h)[:,None].expand(h,w)        # [h, w]
//     wpos = arange(w)[None,:].expand(h,w)        # [h, w]
//     hpos = hpos.reshape(h/S, S, w/S, S).permute(0,2,1,3).flatten()
//     wpos = wpos.reshape(h/S, S, w/S, S).permute(0,2,1,3).flatten()
//     pos = stack([hpos, wpos], -1).repeat(t, 1)  # [t*h*w, 2]
//   pos = cat(per-image)                          # [total_l, 2]
//   inv_freq = 1 / theta**(arange(0, half_rot/2*2, 2) / (half_rot))
//                                                 # length half_rot/2
//   freqs = arange(max_grid).outer(inv_freq)     # [max_grid, half_rot/2]
//   cos_table = freqs.cos(); sin_table = freqs.sin()
//   cos_per_token = cos_table[pos].flatten(1)    # [total_l, half_rot]
// where S = spatial_merge_size, half_rot = head_dim / 2.
//
// Per-token cos has 2 halves: [cos(h_pos * inv_freq[..]), cos(w_pos * inv_freq[..])].
// `vision_rope_apply` reads cos[token, hi] for hi ∈ [0, half_rot), so the
// h/w split lives entirely in the cos/sin layout — kernel is generic.

fn build_rope_cos_sin_bf16(
    grid_thw: &[(u32, u32, u32)],
    cfg: &VisionConfig,
    total_l: usize,
) -> (Vec<u16>, Vec<u16>) {
    let head_dim = (cfg.embed_dim / cfg.num_heads) as usize;
    let half_rot = head_dim / 2;
    let freq_axis_dim = half_rot / 2; // length of inv_freq, per axis (h or w).
    let theta = 10000.0_f32;
    let s = cfg.spatial_merge_size as usize;

    let inv_freq: Vec<f32> = (0..freq_axis_dim)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / (half_rot as f32)))
        .collect();

    let mut cos = Vec::<u16>::with_capacity(total_l * half_rot);
    let mut sin = Vec::<u16>::with_capacity(total_l * half_rot);
    for &(t, h, w) in grid_thw {
        let (h, w, t) = (h as usize, w as usize, t as usize);
        debug_assert_eq!(h % s, 0, "h must be divisible by spatial_merge_size");
        debug_assert_eq!(w % s, 0, "w must be divisible by spatial_merge_size");
        // Build per-(h,w) hpos / wpos with the spatial-merge reshape +
        // permute. Result is a flat list of length h*w with the same
        // ordering Python `Qwen2VLImageProcessor` produces for the
        // patch tensor.
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
        // Repeat per-frame for t time steps.
        for _ in 0..t {
            for token in 0..frame_len {
                let hp = hpos[token] as f32;
                let wp = wpos[token] as f32;
                // First half: h-axis cos/sin; second half: w-axis.
                for &f in inv_freq.iter().take(freq_axis_dim) {
                    let theta_h = hp * f;
                    cos.push(f32_to_bf16(theta_h.cos()));
                    sin.push(f32_to_bf16(theta_h.sin()));
                }
                for &f in inv_freq.iter().take(freq_axis_dim) {
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
    // One varlen segment per (T, H, W) frame — a (t, h, w) tuple
    // contributes `t` segments of length `h * w`.
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

fn f32_to_bf16(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

fn bf16_slice_as_bytes(s: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn i32_slice_as_bytes(s: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

// ── Qwen2VLImageProcessor patch-flatten ────────────────────────────
//
// Mirrors Python `Qwen2VLImageProcessor._preprocess`'s reshape +
// 9D transpose:
//
//   patches: [T, C, H, W]                              (image: T = temporal_patch_size, frames repeated)
//      -> reshape [grid_t, T, C, gH, mH, P, gW, mW, P]
//      -> transpose (0, 3, 6, 4, 7, 2, 1, 5, 8)
//         dims: [grid_t, gH, gW, mH, mW, C, T, P, P]
//      -> reshape [grid_t * grid_h * grid_w, C * T * P * P]
//
// where grid_h = H / P, grid_w = W / P, gH = grid_h / mH, gW = grid_w / mW.
//
// We don't materialize the 9D intermediate — we compute each output
// element's source index directly. For Qwen2-VL on a 392×392 image:
// L = 784 outputs × 1176 features = ~921k elements (~< 4 MB f32 read +
// ~2 MB bf16 write per image, negligible vs the encoder's per-block GEMM
// traffic).
//
// Input pixels are CHW float `[C, H, W]` (post-resize, post-rescale,
// post-mean/std normalization — the engine layer is responsible for
// matching `Qwen2VLImageProcessor`'s do_resize / do_rescale / do_normalize
// before populating `MultimodalData`). For a single image, frames are
// virtually repeated `temporal_patch_size` times — both temporal slots
// read the same pixel data. Output is bf16 to match the encoder dtype.

/// Patch-flatten one normalized CHW image to `[L, C*T*P*P]` bf16
/// patches + `(grid_t, grid_h, grid_w)`. `H` and `W` must be multiples
/// of `patch_size * spatial_merge_size` (Python's smart_resize step
/// guarantees this; the engine layer mirrors it).
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
    assert_eq!(
        pixels.len(),
        c * h * w,
        "patches_from_normalized_chw: pixels.len() = {} but c*h*w = {}",
        pixels.len(),
        c * h * w
    );
    assert_eq!(
        h % (p * s),
        0,
        "image height {h} must be multiple of patch_size*merge_size = {}",
        p * s
    );
    assert_eq!(
        w % (p * s),
        0,
        "image width {w} must be multiple of patch_size*merge_size = {}",
        p * s
    );
    let grid_h = h / p;
    let grid_w = w / p;
    let g_h = grid_h / s;
    let g_w = grid_w / s;
    let grid_t: u32 = 1;
    let l = (grid_t as usize) * grid_h * grid_w;
    let feat = c * t * p * p;
    let mut out = vec![0u16; l * feat];
    // Walk output in (gH, gW, mH, mW, C, T, P_h, P_w) order so the
    // outer index is dense and the inner P_w varies fastest.
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
//
// Bridges the trait's `PixelInput { CHW pixels, h, w }` to the encoder's
// `[total_l, 1176]` patch tensor + per-image `(grid_t, grid_h, grid_w)`.
// Concatenates patches across images into one upload, then calls
// `VisionWeights::forward` once.
//
// `placeholders` describes where each image's slice lands in the token
// sequence — the trait passes them through unchanged. The cuda_worker
// MM seam consumes them to D2D-copy projected embeds into the
// `Instruction::Embed::eval` output rows.

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

        // Fill MRoPE grid info on each returned EmbedPatch so the
        // executor can build `[3, n_tokens]` positions for image-bearing
        // batches. Caller passed in placeholders with token_offset +
        // length already set; we copy those through and zip in the
        // per-image post-spatial-merge grid dims (`grid_t × mh × mw ==
        // length`, the post-merger row count).
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

    /// CPU-only grid computation. Mirrors `patches_from_normalized_chw`'s
    /// grid math (h/patch, w/patch, then merge by `spatial_merge_size`)
    /// without touching pixels or the GPU.
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

/// Probe Qwen2-VL `VisionConfig` from live `GpuWeights` tensor shapes.
///
/// The Qwen2-VL family ships a single vision tower (embed_dim=1280,
/// depth=32, num_heads=16, mlp_ratio=4, patch_size=14, in_chans=3,
/// spatial_merge_size=2) shared across the 2B/7B/72B variants — only
/// `d_model` (= text-decoder hidden size, the merger output dim)
/// changes per variant. Rather than hard-code one variant, we derive
/// every shape-discoverable field from the loaded tensors:
///
/// - `visual.patch_embed.proj.weight` shape `[embed_dim, in_chans, t, h, w]`
///   → `embed_dim`, `in_chans`, `temporal_patch_size`, `patch_size`.
/// - Layer count: walk `visual.blocks.N.norm1.weight` until missing
///   → `depth`.
/// - `visual.blocks.0.mlp.fc1.weight` shape `[mlp_hidden, embed_dim]`
///   → `mlp_ratio = mlp_hidden / embed_dim`.
/// - `visual.merger.mlp.0.weight` shape `[merge_hidden, S²*embed_dim]`
///   → `spatial_merge_size = sqrt(second_dim / embed_dim)`.
/// - `visual.merger.mlp.2.weight` shape `[d_model, merge_hidden]`
///   → `d_model`.
///
/// `num_heads` and `norm_eps` aren't derivable from shapes alone and
/// are constant across the Qwen2-VL family per the upstream config.
fn try_load_mm_qwen2_vl(
    gw: &mut GpuWeights,
    _stream: CUstream,
    _max_model_len: usize,
    _tp_rank: u8,
    _hf: HfFingerprint<'_>,
) -> Result<Option<Box<dyn MultimodalForward>>> {
    let pe = match gw.tensor_shape_any("visual.patch_embed.proj.weight") {
        Some(s) => s,
        None => return Ok(None),
    };
    if pe.len() != 5 {
        anyhow::bail!("visual.patch_embed.proj.weight must be 5D [E, C, T, P, P], got {pe:?}");
    }
    let embed_dim = pe[0] as u32;
    let in_chans = pe[1] as u32;
    let temporal_patch_size = pe[2] as u32;
    let patch_size = pe[3] as u32;
    if pe[3] != pe[4] {
        anyhow::bail!("visual.patch_embed.proj.weight expected square patch (h==w), got {pe:?}");
    }

    let mut depth = 0u32;
    while gw.contains(&format!("visual.blocks.{depth}.norm1.weight")) {
        depth += 1;
    }
    if depth == 0 {
        anyhow::bail!("visual.blocks.0.norm1.weight missing — checkpoint truncated?");
    }

    let fc1 = gw
        .tensor_shape_any("visual.blocks.0.mlp.fc1.weight")
        .ok_or_else(|| anyhow::anyhow!("visual.blocks.0.mlp.fc1.weight missing"))?;
    if fc1.len() != 2 || fc1[1] as u32 != embed_dim {
        anyhow::bail!("visual.blocks.0.mlp.fc1.weight expected [H, {embed_dim}], got {fc1:?}");
    }
    let mlp_hidden = fc1[0] as u32;
    if !mlp_hidden.is_multiple_of(embed_dim) {
        anyhow::bail!("mlp_hidden ({mlp_hidden}) not a multiple of embed_dim ({embed_dim})");
    }
    let mlp_ratio = mlp_hidden / embed_dim;

    let merger0 = gw
        .tensor_shape_any("visual.merger.mlp.0.weight")
        .ok_or_else(|| anyhow::anyhow!("visual.merger.mlp.0.weight missing"))?;
    if merger0.len() != 2 {
        anyhow::bail!("visual.merger.mlp.0.weight expected 2D, got {merger0:?}");
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
        anyhow::bail!("visual.merger.mlp.2.weight expected 2D, got {merger2:?}");
    }
    let d_model = merger2[0] as u32;

    let config = VisionConfig {
        embed_dim,
        depth,
        // num_heads is shape-invisible — fixed across the Qwen2-VL family.
        num_heads: 16,
        mlp_ratio,
        patch_size,
        temporal_patch_size,
        spatial_merge_size,
        in_chans,
        d_model,
        // norm_eps is the upstream default for every Qwen2-VL variant.
        norm_eps: 1e-6,
    };
    let vw = VisionWeights::load(gw, config)?;
    Ok(Some(Box::new(vw) as Box<dyn MultimodalForward>))
}

// Replicated-per-rank vision encoder: at tp>1 each rank loads the
// full `visual.*` weights via `take_with_shape` (no sharding) and
// runs `vision_forward` independently, producing identical
// `mm_embeds`. The post-Embed `SpliceMmEmbeds` pass (see
// `ferrite_forward::Instruction::SpliceMmEmbeds`) D2D-copies those
// mm_embeds into the placeholder rows on every rank AFTER the
// vocab-parallel AllReduce, so the splice isn't summed × tp.
//
// `embed_patches` offsets are in token space (not TP-sharded) and
// MRoPE positions broadcast identically across ranks — the
// executor's `build_mrope_positions_2d` runs on CPU from the image
// grid, independent of any rank.
//
// The `tp_world_size: 1` registration runs at tp=1 (single-rank
// build). The `nccl`-gated registrations at {2, 4, 8} let
// `try_load_mm(arch, tp_world_size, …)` resolve at those ranks
// without forcing a new tp_world_size field on `FerriteMmRegistration`
// (the text-side macro uses the same pattern — one `inventory::submit!`
// per tp value in `tp_sizes`).
ferrite_forward::inventory::submit! {
    FerriteMmRegistration {
        arch_name: "qwen2_vl",
        hf_arches: &["Qwen2VLForConditionalGeneration"],
        gguf_archs: &[],
        tp_world_size: 1,
        try_load_mm: try_load_mm_qwen2_vl,
    }
}

#[cfg(feature = "nccl")]
ferrite_forward::inventory::submit! {
    FerriteMmRegistration {
        arch_name: "qwen2_vl",
        hf_arches: &["Qwen2VLForConditionalGeneration"],
        gguf_archs: &[],
        tp_world_size: 2,
        try_load_mm: try_load_mm_qwen2_vl,
    }
}

#[cfg(feature = "nccl")]
ferrite_forward::inventory::submit! {
    FerriteMmRegistration {
        arch_name: "qwen2_vl",
        hf_arches: &["Qwen2VLForConditionalGeneration"],
        gguf_archs: &[],
        tp_world_size: 4,
        try_load_mm: try_load_mm_qwen2_vl,
    }
}

#[cfg(feature = "nccl")]
ferrite_forward::inventory::submit! {
    FerriteMmRegistration {
        arch_name: "qwen2_vl",
        hf_arches: &["Qwen2VLForConditionalGeneration"],
        gguf_archs: &[],
        tp_world_size: 8,
        try_load_mm: try_load_mm_qwen2_vl,
    }
}
