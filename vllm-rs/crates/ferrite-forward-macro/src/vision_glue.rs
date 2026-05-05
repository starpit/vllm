// SPDX-License-Identifier: Apache-2.0
//! Per-variant vision-arch glue codegen.
//!
//! For each `#[vision_forward]` variant module the macro emits, this
//! module produces:
//!
//! - `impl ferrite_forward::VisionArchWeights for Weights` baking
//!   the variant's [`ferrite_vision::VisionConfig`] from
//!   `vision_*` bounds + `vision_norm_eps` scalar, forwarding
//!   `pixel_pack` to the user-supplied free fn, and forwarding
//!   `vision_forward` to the variant's local emitted `forward`.
//! - `fn try_load_mm(...)` — d_model fingerprint (the only
//!   per-variant geometric difference within a VL family),
//!   patch_embed 5D-conv flatten, then `Weights::load(...)` and
//!   wrap in [`ferrite_forward::VisionWrapper`].
//! - `inventory::submit!` rows for tp ∈ {1, 2, 4, 8} (2/4/8 under
//!   `#[cfg(feature = "nccl")]`). Vision is replicated per-rank;
//!   the same `try_load_mm` runs at every rank.
//!
//! Net: per-arch crates carry no `MultimodalForward` impl, no
//! `try_load_mm`, no `inventory::submit!` rows. The DSL body and a
//! `pixel_pack = path` attribute arg are the whole arch surface.

use proc_macro2::TokenStream;
use quote::quote;

use crate::config::ModelParams;

/// Read a required `vision_*` bound (or `d_model`) by name. Vision
/// configs always carry these (G.5.b's
/// `qwen2_vl_vision_weights_manifest_anchors_on_vision_bounds` pins
/// the manifest); a missing key is a config bug, not a runtime
/// concern.
fn bound(model: &ModelParams, key: &str) -> u64 {
    *model.bounds.get(key).unwrap_or_else(|| {
        panic!(
            "vision config `{}` is missing required bound `{key}`",
            model.source_stem,
        )
    })
}

/// Read the `vision_norm_eps` scalar. Mandatory in every vision
/// config; missing is a config bug.
fn norm_eps(model: &ModelParams) -> f64 {
    *model.scalars.get("vision_norm_eps").unwrap_or_else(|| {
        panic!(
            "vision config `{}` is missing required scalar `vision_norm_eps`",
            model.source_stem,
        )
    })
}

/// Emit the per-variant vision-arch glue. Returned `TokenStream` is
/// inserted into the variant's `pub mod` body alongside the
/// `Weights` struct + emitted `forward` fn.
///
/// `pixel_pack`: `None` to use the trait's default impl
/// (delegates to [`VisionConfig::patches_from_normalized_chw`] —
/// the spatial-merge order shared by Qwen2-VL / Qwen2.5-VL / any
/// arch with the same `patch_size · spatial_merge_size` convention).
/// `Some(path)` to override with `path` taking
/// `(&VisionConfig, &[f32], u32, u32) -> (Vec<u16>, (u32, u32, u32))`
/// — for arches with different patch ordering (SigLIP raster, etc.).
pub fn emit_per_variant(
    model: &ModelParams,
    arch_name: &str,
    pixel_pack: Option<&syn::Path>,
) -> TokenStream {
    let embed_dim = bound(model, "vision_embed_dim") as u32;
    let depth = bound(model, "vision_depth") as u32;
    let num_heads = bound(model, "vision_num_heads") as u32;
    let patch_size = bound(model, "vision_patch_size") as u32;
    let temporal_patch_size = bound(model, "vision_temporal_patch_size") as u32;
    let spatial_merge_size = bound(model, "vision_spatial_merge_size") as u32;
    let in_chans = bound(model, "vision_in_chans") as u32;
    let d_model = bound(model, "d_model") as u32;
    let eps = norm_eps(model) as f32;

    let embed_dim_lit = proc_macro2::Literal::u32_unsuffixed(embed_dim);
    let depth_lit = proc_macro2::Literal::u32_unsuffixed(depth);
    let num_heads_lit = proc_macro2::Literal::u32_unsuffixed(num_heads);
    let patch_size_lit = proc_macro2::Literal::u32_unsuffixed(patch_size);
    let temporal_patch_size_lit = proc_macro2::Literal::u32_unsuffixed(temporal_patch_size);
    let spatial_merge_size_lit = proc_macro2::Literal::u32_unsuffixed(spatial_merge_size);
    let in_chans_lit = proc_macro2::Literal::u32_unsuffixed(in_chans);
    let d_model_lit = proc_macro2::Literal::u32_unsuffixed(d_model);
    let d_model_usize_lit = proc_macro2::Literal::usize_unsuffixed(d_model as usize);
    let eps_lit = proc_macro2::Literal::f32_suffixed(eps);

    let arch_name_lit = syn::LitStr::new(arch_name, proc_macro2::Span::call_site());
    let hf_arch_lits: Vec<syn::LitStr> = model
        .architectures
        .iter()
        .map(|s| syn::LitStr::new(s, proc_macro2::Span::call_site()))
        .collect();

    // `pixel_pack` override: emit a method body iff the user
    // supplied an explicit path; otherwise let the trait's default
    // impl run (which calls `VisionConfig::patches_from_normalized_chw`).
    let pixel_pack_method = pixel_pack.map(|path| {
        quote! {
            fn pixel_pack(
                cfg: &::ferrite_vision::VisionConfig,
                pixels: &[f32],
                height: u32,
                width: u32,
            ) -> (::std::vec::Vec<u16>, (u32, u32, u32)) {
                #path(cfg, pixels, height, width)
            }
        }
    });

    quote! {
        impl ::ferrite_forward::VisionArchWeights for Weights {
            fn vision_config(&self) -> &'static ::ferrite_vision::VisionConfig {
                static C: ::ferrite_vision::VisionConfig = ::ferrite_vision::VisionConfig {
                    embed_dim: #embed_dim_lit,
                    depth: #depth_lit,
                    num_heads: #num_heads_lit,
                    patch_size: #patch_size_lit,
                    temporal_patch_size: #temporal_patch_size_lit,
                    spatial_merge_size: #spatial_merge_size_lit,
                    in_chans: #in_chans_lit,
                    d_model: #d_model_lit,
                    norm_eps: #eps_lit,
                };
                &C
            }

            #pixel_pack_method

            unsafe fn vision_forward(
                &self,
                ctx: &::ferrite_forward::ForwardCtx<'_>,
                device: &mut ::ferrite_cuda_core::device::GpuDevice,
                num_tokens: u64,
            ) -> ::ferrite_cuda_core::alloc::OwnedTensor {
                unsafe { forward(self, ctx, device, num_tokens) }
            }
        }

        /// Variant try-load. Fingerprints on `d_model` (from on-disk
        /// `visual.merger.mlp.2.weight`'s first dim — the only per-
        /// variant geometric difference within a VL family). Returns
        /// `Ok(None)` when the checkpoint isn't a vision tower or
        /// the d_model doesn't match this variant.
        #[allow(clippy::not_unsafe_ptr_arg_deref)]
        fn try_load_mm(
            gw: &mut ::ferrite_cuda_core::weights::GpuWeights,
            stream: ::ferrite_cuda_core::CUstream,
            max_model_len: usize,
            tp_rank: u8,
            _hf: ::ferrite_forward::HfFingerprint<'_>,
        ) -> ::anyhow::Result<::std::option::Option<
            ::std::boxed::Box<dyn ::ferrite_forward::MultimodalForward>,
        >> {
            // Sniff the merger's output dim — that's d_model.
            let merger2 = match gw.tensor_shape_any("visual.merger.mlp.2.weight") {
                ::std::option::Option::Some(s) => s,
                ::std::option::Option::None => return ::std::result::Result::Ok(::std::option::Option::None),
            };
            if merger2.first().copied() != ::std::option::Option::Some(#d_model_usize_lit) {
                return ::std::result::Result::Ok(::std::option::Option::None);
            }
            // patch_embed.proj.weight ships as 5D `[E, C, T, P, P]`
            // on disk; flatten to `[E, C*T*P*P]` so the macro's
            // `LinearLayer::load_dense_or_ggml` reads a dense 2D
            // weight (it has no shape-override entry point).
            if let ::std::option::Option::Some(pe) =
                gw.tensor_shape_any("visual.patch_embed.proj.weight")
            {
                if pe.len() == 5 {
                    let flat = pe[1] * pe[2] * pe[3] * pe[4];
                    gw.reshape_in_place(
                        "visual.patch_embed.proj.weight",
                        &[pe[0], flat],
                    )?;
                }
            }
            let weights = load(gw, stream, max_model_len, tp_rank)?;
            ::std::result::Result::Ok(::std::option::Option::Some(
                ::std::boxed::Box::new(::ferrite_forward::VisionWrapper::new(weights))
                    as ::std::boxed::Box<dyn ::ferrite_forward::MultimodalForward>,
            ))
        }

        // Vision is replicated per-rank — register at every tp size
        // so cuda_worker's `try_load_mm` finds us regardless of
        // `tp_world_size`. Each rank loads the full `visual.*`
        // weights independently and runs `vision_forward` to
        // produce identical mm_embeds; the post-Embed
        // `SpliceMmEmbeds` D2D-copies them into placeholder rows
        // AFTER the vocab-parallel AllReduce.
        ::ferrite_forward::inventory::submit! {
            ::ferrite_forward::FerriteMmRegistration {
                arch_name: #arch_name_lit,
                hf_arches: &[#(#hf_arch_lits),*],
                gguf_archs: &[],
                tp_world_size: 1,
                try_load_mm,
            }
        }
        #[cfg(feature = "nccl")]
        ::ferrite_forward::inventory::submit! {
            ::ferrite_forward::FerriteMmRegistration {
                arch_name: #arch_name_lit,
                hf_arches: &[#(#hf_arch_lits),*],
                gguf_archs: &[],
                tp_world_size: 2,
                try_load_mm,
            }
        }
        #[cfg(feature = "nccl")]
        ::ferrite_forward::inventory::submit! {
            ::ferrite_forward::FerriteMmRegistration {
                arch_name: #arch_name_lit,
                hf_arches: &[#(#hf_arch_lits),*],
                gguf_archs: &[],
                tp_world_size: 4,
                try_load_mm,
            }
        }
        #[cfg(feature = "nccl")]
        ::ferrite_forward::inventory::submit! {
            ::ferrite_forward::FerriteMmRegistration {
                arch_name: #arch_name_lit,
                hf_arches: &[#(#hf_arch_lits),*],
                gguf_archs: &[],
                tp_world_size: 8,
                try_load_mm,
            }
        }
    }
}
