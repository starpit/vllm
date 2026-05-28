// SPDX-License-Identifier: Apache-2.0
//! Generic glue between a `#[vision_forward]`-emitted per-arch
//! `Weights` and the [`MultimodalForward`] trait the executor calls
//! at request time.
//!
//! Each VL/MM arch's macro expansion emits one [`VisionArchWeights`]
//! impl per variant — baking the variant's [`VisionConfig`] from
//! `configs/<variant>.json::vision_*` bounds + scalars, wiring its
//! `pixel_pack` to the user-supplied free fn, and forwarding to the
//! emitted free `forward(...)`. [`VisionWrapper<W>`] provides a
//! single generic [`MultimodalForward`] impl over any
//! `VisionArchWeights` — pixel pack on host, upload to GPU, build
//! cu_seqlens / cos / sin from `grid_thw`, build [`ForwardCtx`],
//! call the emitted forward, fill MRoPE grid info on each returned
//! [`EmbedPatch`].
//!
//! Net effect: per-arch crates drop their hand-written
//! `MultimodalForward` impl + `try_load_mm` shape-probe + inventory
//! rows. The DSL body becomes the only file in the per-arch crate
//! that carries arch-specific code (modulo the per-arch pixel-pack
//! free fn in `ferrite-vision`).

use ferrite_cuda_core::DType;
use ferrite_cuda_core::alloc::OwnedTensor;
use ferrite_cuda_core::device::GpuDevice;
use ferrite_cuda_core::tensor::GpuTensor;
use ferrite_kernels::kv_cache::KvCachePool;
use ferrite_vision::{
    VisionConfig, bf16_slice_as_bytes, build_cu_seqlens_i32, i32_slice_as_bytes,
    permute_rows_block_grouped_bf16, u32_slice_as_bytes,
};

use crate::ForwardCtx;
use crate::dispatcher::{EmbedPatch, MultimodalForward, PixelInput};

/// Per-arch `Weights` glue — the macro emits one impl per variant
/// module. Carries the baked [`VisionConfig`], a pointer to the
/// per-arch host-side pixel-pack fn, and an `unsafe fn vision_forward`
/// that just calls the macro's emitted free `forward(...)`.
///
/// The trait is named `VisionArchWeights` (not just `VisionWeights`)
/// to leave the latter free in per-arch crates if they want a
/// concrete name. In practice they don't — [`VisionWrapper<W>`]
/// covers the whole `MultimodalForward` surface generically.
pub trait VisionArchWeights: Send + Sync + Sized + 'static {
    /// Geometric config, baked at macro-expansion time from the
    /// variant's `vision_*` bounds + `vision_norm_eps` scalar.
    fn vision_config(&self) -> &'static VisionConfig;

    /// CPU pixel pack: CHW pixels → varlen `[T*H*W, C·T·P²]` bf16
    /// patches + `(grid_t, grid_h, grid_w)`. Default delegates to
    /// [`VisionConfig::patches_from_normalized_chw`] — the spatial-
    /// merge order shared by Qwen2-VL / Qwen2.5-VL / any arch with
    /// the same `patch_size · spatial_merge_size` convention. Arches
    /// with different patch ordering (SigLIP raster, etc.) override
    /// via the `pixel_pack = path::to::fn` attribute arg, which the
    /// macro turns into an override of this method.
    fn pixel_pack(
        cfg: &VisionConfig,
        pixels: &[f32],
        height: u32,
        width: u32,
    ) -> (Vec<u16>, (u32, u32, u32)) {
        cfg.patches_from_normalized_chw(pixels, height, width)
    }

    /// Run the encoder body. Macro emits a one-line forwarder to
    /// the variant module's free `forward(...)`.
    ///
    /// # Safety
    /// `ctx` must have `pixels` / `cu_seqlens_q` / `max_seqlen_q` /
    /// `vision_rope_cos` / `vision_rope_sin` populated; `device`
    /// live.
    unsafe fn vision_forward(
        &self,
        ctx: &ForwardCtx<'_>,
        device: &mut GpuDevice,
        num_tokens: u64,
    ) -> OwnedTensor;

    /// Window-attention spatial window edge in pixels, when the body
    /// dispatches per-layer between full-frame and windowed varlen
    /// attention (Qwen2.5-VL family). Default `None` → the wrapper
    /// builds the simple single-cu-seqlens path (Qwen2-VL / SigLIP /
    /// any tower with one attention regime). `Some(window_size)` →
    /// the wrapper builds full + windowed cu_seqlens, the
    /// natural→window-grouped permutation pair, and applies it
    /// host-side to the rope cos/sin tables before upload, populating
    /// the matching `vision_*` fields on [`ForwardCtx`].
    fn windowed_attn_window_size() -> Option<u32> {
        None
    }

    /// Per-arch CPU-side preprocessing metadata declaration. The macro
    /// emits this as `&<crate>::PROCESSOR` so the executor can read
    /// declarative flags (e.g. `mrope_positions`) without naming any
    /// arch.
    fn mm_metadata() -> &'static ferrite_vision::MmMetadata;

    /// Number of learned positional embedding rows the body looks up
    /// via `pos_embed(position_ids, embeddings.position_embedding)`.
    /// `Some(N)` → the wrapper builds `[0..N, 0..N, ...]` u32 per
    /// image and uploads it into [`ForwardCtx::vision_position_ids`];
    /// `None` → no positional embedding (Qwen2-VL / Qwen2.5-VL use 2D
    /// RoPE via `vision_rope` instead). Used by SigLIP / Gemma3-MM.
    /// `N` matches the per-image patch count (image_size/patch_size)²,
    /// fixed at compile time for SigLIP-class arches with a fixed
    /// resize step.
    fn vision_num_positions() -> Option<u32> {
        None
    }
}

/// Generic [`MultimodalForward`] impl over any [`VisionArchWeights`].
/// Owns a `KvCachePool` placeholder because [`ForwardCtx::kv_cache`]
/// is non-optional but vision-body codegen never reads it.
pub struct VisionWrapper<W: VisionArchWeights> {
    pub weights: W,
    kv_placeholder: KvCachePool,
}

impl<W: VisionArchWeights> VisionWrapper<W> {
    pub fn new(weights: W) -> Self {
        Self {
            weights,
            kv_placeholder: KvCachePool::empty_for_vision(),
        }
    }
}

impl<W: VisionArchWeights> MultimodalForward for VisionWrapper<W> {
    unsafe fn vision_forward(
        &self,
        pixel_batches: &[PixelInput<'_>],
        placeholders: &[EmbedPatch],
        device: &mut GpuDevice,
    ) -> (OwnedTensor, Vec<EmbedPatch>) {
        let cfg = self.weights.vision_config();
        let p = cfg.patch_size as usize;
        let t = cfg.temporal_patch_size as usize;
        let c = cfg.in_chans as usize;
        let feat = c * t * p * p;

        let mut all_patches: Vec<u16> = Vec::new();
        let mut grid_thw: Vec<(u32, u32, u32)> = Vec::with_capacity(pixel_batches.len());
        for img in pixel_batches {
            let (mut patch_rows, gthw) = W::pixel_pack(cfg, img.pixels, img.height, img.width);
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

        let half_rot = cfg.half_rot();
        let (cos_host, sin_host) = cfg.build_rope_cos_sin_bf16(&grid_thw, total_l);

        let (cu_seqlens_host, max_seqlen) = build_cu_seqlens_i32(&grid_thw);
        let cu_seqlens = device.alloc_gpu_tensor_from_host(
            &[cu_seqlens_host.len()],
            DType::I32,
            i32_slice_as_bytes(&cu_seqlens_host),
        );

        // ── Window-attention dispatch (Qwen2.5-VL family) ──────────
        //
        // For arches whose `windowed_attn_window_size()` returns Some,
        // build the natural→window-grouped permutation host-side,
        // apply it to cos/sin (S² block-grouped permute), upload all
        // four extras (cu_window_seqlens + window_index +
        // reverse_indices + permuted cos/sin). The body's per-layer
        // `varlen_attention(..., cu_seqlens_{full,window}, ...)`
        // calls then read the matching `ForwardCtx` field via the
        // u8 discriminant baked at codegen.
        let window_dispatch = W::windowed_attn_window_size().map(|window_size| {
            ferrite_vision::build_qwen2_5_window_dispatch(&grid_thw, cfg, window_size)
        });

        // cos/sin live on GPU. When windowed, allocate from the
        // permuted host buffers; else from the natural ones. Same
        // shape `[total_l, half_rot]` either way.
        let (cos_to_upload, sin_to_upload);
        let (cos_buf, sin_buf);
        if let Some(wd) = window_dispatch.as_ref() {
            cos_buf = permute_rows_block_grouped_bf16(
                &cos_host,
                total_l,
                half_rot,
                cfg.spatial_merge_size as usize,
                &wd.window_index,
            );
            sin_buf = permute_rows_block_grouped_bf16(
                &sin_host,
                total_l,
                half_rot,
                cfg.spatial_merge_size as usize,
                &wd.window_index,
            );
            cos_to_upload = &cos_buf[..];
            sin_to_upload = &sin_buf[..];
        } else {
            cos_to_upload = &cos_host[..];
            sin_to_upload = &sin_host[..];
        }
        let cos = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            DType::BF16,
            bf16_slice_as_bytes(cos_to_upload),
        );
        let sin = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            DType::BF16,
            bf16_slice_as_bytes(sin_to_upload),
        );

        // ── Learned positional embeddings (SigLIP / Gemma3-MM) ──────
        //
        // For arches that override `vision_num_positions()` to
        // `Some(N)`, build `[0..N, 0..N, ...]` u32 (one chunk per
        // image) and upload as `vision_position_ids`. The DSL body's
        // `pos_embed(position_ids, weight)` reads this view via
        // `kernels::embedding_gather_masked`. Default `None` arches
        // (Qwen2-VL family) skip the upload — `position_ids_buf`
        // stays None so the GpuTensor isn't allocated.
        //
        // The owned GpuTensor lives in this function's stack frame so
        // the view inside `ctx` doesn't dangle.
        let position_ids_buf = W::vision_num_positions().map(|num_pos| {
            let n = num_pos as usize;
            let mut ids: Vec<u32> = Vec::with_capacity(total_l);
            for _ in 0..pixel_batches.len() {
                ids.extend(0..num_pos);
            }
            debug_assert_eq!(ids.len(), total_l);
            debug_assert_eq!(total_l, pixel_batches.len() * n);
            device.alloc_gpu_tensor_from_host(&[total_l], DType::U32, u32_slice_as_bytes(&ids))
        });
        let position_ids_view = position_ids_buf.as_ref().map(|t| unsafe { t.as_view() });

        let (cu_window_view, window_index_view, reverse_indices_view, max_seqlen_window_opt) =
            if let Some(ref wd) = window_dispatch {
                let l_merged = total_l / (cfg.spatial_merge_size as usize).pow(2);
                let cu_w = device.alloc_gpu_tensor_from_host(
                    &[wd.cu_window_seqlens.len()],
                    DType::I32,
                    i32_slice_as_bytes(&wd.cu_window_seqlens),
                );
                let wi = device.alloc_gpu_tensor_from_host(
                    &[l_merged],
                    DType::U32,
                    u32_slice_as_bytes(&wd.window_index),
                );
                let ri = device.alloc_gpu_tensor_from_host(
                    &[l_merged],
                    DType::U32,
                    u32_slice_as_bytes(&wd.reverse_indices),
                );
                (
                    Some(unsafe { cu_w.as_view() }),
                    Some(unsafe { wi.as_view() }),
                    Some(unsafe { ri.as_view() }),
                    Some(wd.max_seqlen_window),
                )
            } else {
                (None, None, None, None)
            };

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
            // Qwen2.5-VL: `cu_seqlens_full` is the same per-image
            // segmentation as `cu_seqlens_q` above (the window
            // permutation preserves per-image boundaries since
            // `window_index` is built per-image). Reuse the same
            // GpuTensor view; the `cu_seqlens_kind=1` arm reads
            // through this field. `max_seqlen_full` matches.
            vision_cu_seqlens_full: window_dispatch.as_ref().map(|_| cu_view),
            vision_cu_seqlens_window: cu_window_view,
            vision_max_seqlen_full: window_dispatch.as_ref().map(|_| max_seqlen),
            vision_max_seqlen_window: max_seqlen_window_opt,
            vision_window_index: window_index_view,
            vision_reverse_indices: reverse_indices_view,
            vision_position_ids: position_ids_view,
            last_token_indices: None,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        let projected = unsafe { self.weights.vision_forward(&ctx, device, total_l as u64) };

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

    fn mm_metadata(&self) -> &'static ferrite_vision::MmMetadata {
        W::mm_metadata()
    }

    fn embed_patch_grids(&self, pixel_batches: &[PixelInput<'_>]) -> Vec<(u32, u32, u32)> {
        let cfg = self.weights.vision_config();
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
