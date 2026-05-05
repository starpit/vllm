// SPDX-License-Identifier: Apache-2.0
//! Qwen2-VL vision encoder host wrapper.
//!
//! The encoder body itself lives in the DSL: `src/dsl_body.rs` carries
//! `#[vision_forward] fn qwen2_vl()` whose macro expansion emits a
//! per-variant `qwen2_vl_2b::{Weights, load, forward}` (and equivalent
//! `_7b` / `_72b` modules — byte-identical bodies, see G.5.f handoff).
//! This file owns only:
//!
//! - the [`MultimodalForward`] impl bridging
//!   `PixelInput { CHW pixels, h, w }` → encoder
//!   `[total_l, in_chans*t*p*p]` patches + per-image `(grid_t, grid_h, grid_w)`,
//! - host-side cu_seqlens / cos-sin / patches packing (via
//!   [`ferrite_vision::VisionConfig`] methods),
//! - the [`ForwardCtx`] wiring that hands those buffers to the
//!   macro-emitted `forward(...)`,
//! - inventory registration of the `Qwen2VLForConditionalGeneration`
//!   HF arch so cuda_worker's `try_load_mm` resolves this encoder.
//!
//! The macro-emitted three variants (`qwen2_vl_2b`, `_7b`, `_72b`) all
//! emit byte-identical bodies — they differ only in the per-variant
//! `d_model` bound which the macro carves into the dedup signature, and
//! d_model is unused in codegen. Until the macro learns to alias them,
//! we pin the host wrapper to `qwen2_vl_2b`'s entry point. The
//! per-checkpoint `d_model` is recovered at runtime from the on-disk
//! `visual.merger.mlp.2.weight` shape (LinearLayer's loader doesn't
//! validate against the manifest's compile-time bound).

use anyhow::{Context, Result};
use ferrite_cuda_core::CUstream;
use ferrite_cuda_core::DType;
use ferrite_cuda_core::alloc::OwnedTensor;
use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::tensor::GpuTensor;
use ferrite_cuda_core::weights::GpuWeights;
use ferrite_forward::{
    EmbedPatch, FerriteMmRegistration, ForwardCtx, HfFingerprint, MultimodalForward, PixelInput,
};
use ferrite_kernels::kv_cache::KvCachePool;
use ferrite_vision::{TraceDump, bf16_slice_as_bytes, build_cu_seqlens_i32, i32_slice_as_bytes};

use crate::dsl_body::qwen2_vl_2b as canonical;

/// Common vision-config surface shared with sibling VL crates. Host-
/// side helpers (`build_rope_cos_sin_bf16`, `patches_from_normalized_chw`)
/// hang off it as methods.
pub use ferrite_vision::VisionConfig;

/// Qwen2-VL vision encoder wrapper. Holds the macro-emitted weight
/// bundle, the host-side `VisionConfig` (used for cu_seqlens / cos-sin
/// / pixel-patch building), and a placeholder [`KvCachePool`] needed
/// to satisfy [`ForwardCtx::kv_cache`] — vision-body codegen never
/// emits a kv_cache-touching instruction so the placeholder is never
/// dereferenced.
pub struct VisionWeights {
    inner: canonical::Weights,
    config: VisionConfig,
    /// Empty pool placeholder — see [`KvCachePool::empty_for_vision`].
    /// Lives next to `inner` so the `&KvCachePool` borrow stored in
    /// `ForwardCtx::kv_cache` is satisfied by `&self.kv_placeholder`.
    kv_placeholder: KvCachePool,
}

impl VisionWeights {
    /// Load every `visual.*` tensor through the macro-emitted
    /// `Weights::load`, which honors the per-arch `weights.json`
    /// manifest (`__packed_splits__` carves `attn.qkv` → q/k/v) and
    /// prepends the `visual.blocks.<L>.` / `visual.` prefixes.
    pub fn load(gw: &mut GpuWeights, config: VisionConfig, stream: CUstream) -> Result<Self> {
        // `visual.patch_embed.proj.weight` lands on disk as a 5D Conv3d
        // tensor `[E, C, T, P, P]`; the runtime collapses it to a 2D
        // GEMM (stride==kernel==spatial dims). The macro's loader uses
        // `LinearLayer::load_dense_or_ggml` which has no shape-override
        // entry point, so we flatten the entry before the loader sees
        // it. The manifest's `[K, N]` order is `[in_features, embed_dim]`
        // — the loader expects on-disk `[out, in]`, i.e. `[E, C*T*P*P]`.
        let in_chans = config.in_chans as usize;
        let t = config.temporal_patch_size as usize;
        let p = config.patch_size as usize;
        let embed_dim = config.embed_dim as usize;
        gw.reshape_in_place(
            "visual.patch_embed.proj.weight",
            &[embed_dim, in_chans * t * p * p],
        )
        .context("flatten visual.patch_embed.proj.weight 5D → 2D")?;

        let inner = canonical::load(gw, stream, /* max_model_len */ 0, /* tp_rank */ 0)
            .context("dsl_body::qwen2_vl_2b::load")?;
        Ok(Self {
            inner,
            config,
            kv_placeholder: KvCachePool::empty_for_vision(),
        })
    }

    /// Run the full Qwen2-VL vision encoder via the macro-emitted
    /// `forward(...)`. Inputs:
    /// - `pixels`: `[total_l, in_chans * temporal_patch_size * patch_size**2]`
    ///   bf16 patch tensor — concatenated across images, in spatial-
    ///   merge block order (matches Python `Qwen2VLImageProcessor`).
    /// - `grid_thw`: per-image patch-grid shape `(T, H, W)`.
    /// - `device`: CUDA device handle (cublas + caching allocator + stream).
    ///
    /// Output: `[total_l / spatial_merge_size**2, d_model]` bf16 — the
    /// projected embeddings to splice into the language model's
    /// `embed_tokens` output at image-placeholder positions.
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
        let dump = TraceDump::from_env();
        dump.dump_tensor("ferrite_pixels_in", pixels, stream);

        // Per-token cos/sin rope tables on host, uploaded as bf16.
        let (cos_host, sin_host) = cfg.build_rope_cos_sin_bf16(grid_thw, total_l);
        let cos = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            dtype,
            bf16_slice_as_bytes(&cos_host),
        );
        let sin = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            dtype,
            bf16_slice_as_bytes(&sin_host),
        );
        dump.dump_tensor("ferrite_cos_half", cos, stream);
        dump.dump_tensor("ferrite_sin_half", sin, stream);

        // Per-image varlen cu_seqlens + max_seqlen.
        let (cu_seqlens_host, max_seqlen) = build_cu_seqlens_i32(grid_thw);
        let cu_seqlens = device.alloc_gpu_tensor_from_host(
            &[cu_seqlens_host.len()],
            DType::I32,
            i32_slice_as_bytes(&cu_seqlens_host),
        );
        dump.dump_tensor("ferrite_cu_seqlens", cu_seqlens, stream);

        // Build a vision ForwardCtx. Decoder-side fields
        // (input_ids, positions, slot_mapping, seqused_k, block_table,
        // mm_embeds, embed_patches, kv_cache) are unused by the
        // vision body — pass null/empty placeholders. `pixels`,
        // `cu_seqlens_q`, `max_seqlen_q`, `vision_rope_cos`,
        // `vision_rope_sin` carry the actual buffers the body reads.
        let null_view = unsafe { GpuTensor::null(DType::U32).as_view() };
        let pixels_view = unsafe { pixels.as_view() };
        let cu_view = unsafe { cu_seqlens.as_view() };
        let cos_view = unsafe { cos.as_view() };
        let sin_view = unsafe { sin.as_view() };
        let ctx = ForwardCtx {
            input_ids: null_view,
            positions: null_view,
            slot_mapping: null_view,
            cu_seqlens_q: cu_view,
            seqused_k: null_view,
            block_table: null_view,
            max_seqlen_q: max_seqlen,
            max_seqlen_k: 0,
            kv_cache: &self.kv_placeholder,
            mm_embeds: None,
            embed_patches: &[],
            vision_rope_cos: Some(cos_view),
            vision_rope_sin: Some(sin_view),
            pixels: Some(pixels_view),
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        let out = unsafe { canonical::forward(&self.inner, &ctx, device, total_l as u64) };
        dump.dump_tensor("ferrite_merger_out", *out, stream);
        out
    }
}

// ── MultimodalForward impl ─────────────────────────────────────────
//
// Bridges the trait's `PixelInput { CHW pixels, h, w }` to the
// encoder's `[total_l, 1176]` patch tensor + per-image
// `(grid_t, grid_h, grid_w)`. Concatenates patches across images into
// one upload, then calls `VisionWeights::forward` once.
//
// `placeholders` describes where each image's slice lands in the
// token sequence — the trait passes them through unchanged. The
// cuda_worker MM seam consumes them to D2D-copy projected embeds
// into the `Instruction::Embed::eval` output rows.

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
/// `d_model` (= text-decoder hidden, the merger output dim) changes.
/// Every shape-discoverable field is derived from the loaded tensors;
/// `num_heads` and `norm_eps` aren't shape-visible and are constants.
fn try_load_mm_qwen2_vl(
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
    let vw = VisionWeights::load(gw, config, stream)?;
    Ok(Some(Box::new(vw) as Box<dyn MultimodalForward>))
}

// Replicated-per-rank vision encoder: at tp>1 each rank loads the
// full `visual.*` weights (no sharding) and runs `vision_forward`
// independently, producing identical `mm_embeds`. The post-Embed
// `SpliceMmEmbeds` pass (see `ferrite_forward::Instruction::SpliceMmEmbeds`)
// D2D-copies those mm_embeds into the placeholder rows on every rank
// AFTER the vocab-parallel AllReduce, so the splice isn't summed × tp.
//
// `embed_patches` offsets are in token space (not TP-sharded) and
// MRoPE positions broadcast identically across ranks.
//
// One `inventory::submit!` per supported tp value (the text-side
// macro uses the same pattern).

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
