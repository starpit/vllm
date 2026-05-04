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
//! - Host-side helpers (rope cos/sin tables, varlen cu_seqlens, pixel
//!   patch flatten, trace dump) live in `ferrite-vision` and hang off
//!   `VisionConfig` as methods, shared with `ferrite-model-qwen2-5-vl`.
//! - `impl MultimodalForward for VisionWeights` — bridges
//!   `ferrite_forward::PixelInput { CHW pixels, h, w }` → encoder
//!   `[total_l, in_chans*t*p*p]` patches + per-image `(grid_t, grid_h, grid_w)`.
//! - `inventory::submit!` of a `FerriteMmRegistration` claiming
//!   `Qwen2VLForConditionalGeneration` so cuda_worker's `try_load_mm`
//!   resolves the encoder when the live `GpuWeights` carries `visual.*`
//!   tensors. The vision-config fields are derived from tensor shapes
//!   (no per-variant hardcode).

use anyhow::{Context, Result};
use ferrite_cuda_core::CUstream;
use ferrite_cuda_core::DType;
use ferrite_cuda_core::alloc::OwnedTensor;
use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::tensor::GpuTensor;
use ferrite_cuda_core::weights::GpuWeights;
use ferrite_forward::{
    EmbedPatch, FerriteMmRegistration, HfFingerprint, MultimodalForward, PixelInput,
};
use ferrite_kernels::kernels;
use ferrite_kernels::layers::{LayerNorm, Linear};
use ferrite_vision::{TraceDump, bf16_slice_as_bytes, build_cu_seqlens_i32, i32_slice_as_bytes};

/// Common vision-config surface shared with sibling VL crates, with the
/// host-side helpers (`build_rope_cos_sin_bf16`, `patches_from_normalized_chw`)
/// hung off it as methods.
pub use ferrite_vision::VisionConfig;

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
        let (cos_host, sin_host) = cfg.build_rope_cos_sin_bf16(grid_thw, total_l);
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
                cfg.patches_from_normalized_chw(img.pixels, img.height, img.width);
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
    // mlp_hidden is consumed implicitly via `fc1.weight.dim(0)` at runtime;
    // the divisibility check above is the only thing it gates here.

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
