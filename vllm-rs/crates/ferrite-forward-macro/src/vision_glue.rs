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
use crate::weights_manifest::PadHint;

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
    pad_to_mult8: &[PadHint],
    processor: &syn::Path,
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

    // d_model fingerprint: per-arch on-disk weight + dim that
    // `try_load_mm` reads to discriminate among variants. Default
    // is Qwen's `visual.merger.mlp.2.weight` dim 0 — every existing
    // Qwen2-VL / Qwen2.5-VL config keeps that path byte-identically.
    let fingerprint = model
        .vision_d_model_fingerprint
        .clone()
        .unwrap_or_else(crate::config::VisionDModelFingerprint::qwen_default);
    let fp_key_lit = syn::LitStr::new(&fingerprint.key, proc_macro2::Span::call_site());
    let fp_dim_lit = proc_macro2::Literal::usize_unsuffixed(fingerprint.dim);

    // Patch-embed flatten target: per-arch on-disk conv weight that
    // ships in 4D (SigLIP `[E, C, P, P]`) or 5D (Qwen `[E, C, T, P, P]`)
    // form. We multiply every dim after `leading_dim` to land at a
    // dense 2D `[d_lead, prod_after]`. Default is Qwen's 5D path.
    let flatten = model
        .vision_patch_embed_flatten
        .clone()
        .unwrap_or_else(crate::config::VisionPatchEmbedFlatten::qwen_default);
    let flatten_key_lit = syn::LitStr::new(&flatten.key, proc_macro2::Span::call_site());
    let flatten_lead_lit = proc_macro2::Literal::usize_unsuffixed(flatten.leading_dim);

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

    // Per-block zero-pad expansion for `__pad_to_mult8__` manifest
    // entries. Each PadHint's stem is treated as a per-block weight
    // suffix (matching the per-block convention shared with
    // `__packed_splits__`); the emitted prelude walks
    // `0..vision_depth` and pads the on-disk `.weight` tensor at CPU
    // side before any `Weights::load` runs. Top-level (non-per-block)
    // pads aren't needed today; add a `top_level: bool` PadHint field
    // when a future arch requires it.
    let layered_prefix = {
        let layout = model
            .vision_layout
            .clone()
            .unwrap_or_else(crate::config::VisionSafetensorsLayout::qwen_default);
        format!("{}.{}", layout.default_root, layout.layered_subpath)
    };
    let pad_calls: Vec<TokenStream> = pad_to_mult8
        .iter()
        .map(|hint| {
            let stem = &hint.weight;
            let dim_lit = proc_macro2::Literal::usize_unsuffixed(hint.dim);
            let template = format!("{layered_prefix}.{{l}}.{stem}.weight");
            let template_lit = syn::LitStr::new(&template, proc_macro2::Span::call_site());
            quote! {
                for __l in 0..#depth_lit {
                    let __key = #template_lit.replace("{l}", &__l.to_string());
                    gw.pad_axis_to_mult8(&__key, #dim_lit, 8)?;
                }
            }
        })
        .collect();
    let pad_prelude = if pad_calls.is_empty() {
        quote! {}
    } else {
        quote! {
            // CPU-side weight padding: for each `__pad_to_mult8__`
            // hint, grow the named weight's specified axis to the
            // next multiple of 8 with zero fill. This dodges cuBLAS
            // bf16 GEMM rejecting `K=intermediate_size=3420` on
            // Qwen2.5-VL-3B; the macro's downstream `Linear::load`
            // sees the padded shape directly. No-op if already
            // mult-of-8.
            #(#pad_calls)*
        }
    };

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

    // Window-attention dispatch override: arches with a
    // `vision_window_size` bound (Qwen2.5-VL family) override the
    // trait default `None` to return `Some(window_size)`. The
    // VisionWrapper reads this at request time and builds the
    // window-grouped permutation + permutes cos/sin host-side.
    let windowed_attn_method = model.bounds.get("vision_window_size").map(|window_size| {
        let lit = proc_macro2::Literal::u32_unsuffixed(*window_size as u32);
        quote! {
            fn windowed_attn_window_size() -> ::std::option::Option<u32> {
                ::std::option::Option::Some(#lit)
            }
        }
    });

    // Learned positional embedding override (G.7(c.1)): arches with a
    // `vision_num_positions` bound (SigLIP / Gemma3-MM family) override
    // the trait default `None` to return `Some(num_pos)`. The
    // VisionWrapper reads this at request time and uploads
    // `[0..N, 0..N, ...]` as `vision_position_ids` so the body's
    // `pos_embed(...)` call has its index buffer.
    let pos_embed_method = model.bounds.get("vision_num_positions").map(|num_pos| {
        let lit = proc_macro2::Literal::u32_unsuffixed(*num_pos as u32);
        quote! {
            fn vision_num_positions() -> ::std::option::Option<u32> {
                ::std::option::Option::Some(#lit)
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

            fn mm_metadata() -> &'static ::ferrite_vision::MmMetadata {
                &#processor
            }

            #pixel_pack_method

            #windowed_attn_method

            #pos_embed_method

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
            // Sniff the per-arch d_model fingerprint key + dim —
            // discriminates among variants of one vision arch.
            // Default (Qwen): `visual.merger.mlp.2.weight` dim 0.
            let fp_shape = match gw.tensor_shape_any(#fp_key_lit) {
                ::std::option::Option::Some(s) => s,
                ::std::option::Option::None => return ::std::result::Result::Ok(::std::option::Option::None),
            };
            if fp_shape.get(#fp_dim_lit).copied() != ::std::option::Option::Some(#d_model_usize_lit) {
                return ::std::result::Result::Ok(::std::option::Option::None);
            }
            // Patch-embed conv weight ships as 4D (SigLIP) or 5D
            // (Qwen) on disk; flatten dims after `leading_dim` into
            // one row so the macro's `LinearLayer::load_dense_or_ggml`
            // reads a dense 2D `[d_lead, prod_rest]` (no shape-
            // override entry point on the loader). No-op if the
            // tensor is already 2D.
            if let ::std::option::Option::Some(pe) =
                gw.tensor_shape_any(#flatten_key_lit)
            {
                if pe.len() > 2 {
                    let lead = pe[#flatten_lead_lit];
                    let rest: usize = pe.iter().skip(#flatten_lead_lit + 1).product();
                    gw.reshape_in_place(#flatten_key_lit, &[lead, rest])?;
                }
            }
            #pad_prelude
            let weights = load(gw, stream, max_model_len, tp_rank)?;
            ::std::result::Result::Ok(::std::option::Option::Some(
                ::std::boxed::Box::new(::ferrite_forward::VisionWrapper::new(weights))
                    as ::std::boxed::Box<dyn ::ferrite_forward::MultimodalForward>,
            ))
        }

        // Vision is replicated per-rank — register at every tp size
        // so ferrite_worker's `try_load_mm` finds us regardless of
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
                mm_metadata: #processor,
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
                mm_metadata: #processor,
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
                mm_metadata: #processor,
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
                mm_metadata: #processor,
            }
        }
    }
}
