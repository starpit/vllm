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
use ferrite_vision::{VisionConfig, bf16_slice_as_bytes, build_cu_seqlens_i32, i32_slice_as_bytes};

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
        let cos = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            DType::BF16,
            bf16_slice_as_bytes(&cos_host),
        );
        let sin = device.alloc_gpu_tensor_from_host(
            &[total_l, half_rot],
            DType::BF16,
            bf16_slice_as_bytes(&sin_host),
        );

        let (cu_seqlens_host, max_seqlen) = build_cu_seqlens_i32(&grid_thw);
        let cu_seqlens = device.alloc_gpu_tensor_from_host(
            &[cu_seqlens_host.len()],
            DType::I32,
            i32_slice_as_bytes(&cu_seqlens_host),
        );

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
