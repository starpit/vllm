// SPDX-License-Identifier: Apache-2.0
//! Codegen for the `forward!` macro.
//!
//! Walks the solver's DispatchSequence entry by entry. Each entry
//! emits one kernel call. No interpretation, no shortcuts.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::compile_dsl::{ForwardDef, TargetId, WorkloadRange};
use super::dispatch::{DispatchEntry, DispatchSequence, GemmPhase, ImplDispatchKind};
use crate::lowering::BacktrackCpSolver;
use crate::lowering::library::ImplementationLibrary;
use crate::lowering::solver::PlanFamily;
use crate::lowering::tile_graph::TileGraph;
use crate::target_profile::TargetProfile;

pub fn generate(def: &ForwardDef) -> TokenStream {
    if def.is_fully_specialized() {
        generate_fully_specialized(def)
    } else {
        // Runtime paths not yet implemented.
        quote! {}
    }
}

fn generate_fully_specialized(def: &ForwardDef) -> TokenStream {
    let models = def.models.as_static().unwrap();
    let target_id = def.target.as_static().unwrap();

    if models.is_empty() {
        return quote! { compile_error!("forward!: models list is empty"); };
    }

    // For now, use the first model. Multi-model dispatch (from_dims)
    // is a follow-up.
    let model = &models[0];
    let tile_graph = TileGraph::from_model_dag(&def.dag, model.dims);
    let library = build_library(target_id, model.dims);
    let profile = build_profile(target_id);
    let grid = match &def.workloads {
        Some(wl) => build_solve_grid(wl),
        None => PlanFamily::DEFAULT_GRID.to_vec(),
    };
    let family = PlanFamily::solve_grid(&tile_graph, &library, &profile, &BacktrackCpSolver, &grid);

    if family.is_empty() {
        return quote! { compile_error!("forward!: solver found no feasible plans"); };
    }

    let mut bucket_fns = Vec::new();
    let mut match_arms = Vec::new();
    let mut lm_head_bucket_fns = Vec::new();
    let mut lm_head_match_arms = Vec::new();
    let mut prev_upper = 0u32;
    let mut lm_head_prev_upper = 0u32;
    let plans: Vec<_> = family.iter().collect();
    let num_layers = tile_graph.num_layers;

    // Detect whether the DSL includes the post-loop phase (has any
    // tile tagged with layer == num_layers). If so, we emit the
    // lm_head dispatcher; otherwise we skip it (back-compat with the
    // pre-lm_head DSL).
    let has_post_loop = tile_graph.nodes.iter().any(|n| n.layer == num_layers);

    for (i, (seq, plan)) in plans.iter().enumerate() {
        let ds = DispatchSequence::from_plan(plan, &library, &tile_graph);
        let fn_name = format_ident!("solver_layer_bucket_{}", i);
        let stmts = emit_layer_stmts(&ds);

        bucket_fns.push(quote! {
            #[allow(unused_variables, unused_mut, unused_assignments)]
            #[inline(never)]
            unsafe fn #fn_name(
                layer: &LlamaDecoderLayer,
                hidden_states: OwnedTensor,
                residual: Option<OwnedTensor>,
                positions: TensorView<'_>,
                slot_mapping: TensorView<'_>,
                cu_seqlens_q: TensorView<'_>,
                seqused_k: TensorView<'_>,
                block_table: TensorView<'_>,
                max_seqlen_q: usize,
                max_seqlen_k: usize,
                kv_cache: &KvCachePool,
                rotary: &RotaryCache,
                device: &mut GpuDevice,
            ) -> (OwnedTensor, OwnedTensor) {
                let mut hidden_states = hidden_states;
                let mut residual = residual;
                let mut normed: Option<OwnedTensor> = None;
                let mut qkv_out: Option<OwnedTensor> = None;
                let mut k_out: Option<OwnedTensor> = None;
                let mut v_out: Option<OwnedTensor> = None;
                let mut attn_out: Option<OwnedTensor> = None;
                let mut gate_up: Option<OwnedTensor> = None;
                let mut silu_out: Option<OwnedTensor> = None;

                #(#stmts)*

                (hidden_states, residual.unwrap())
            }
        });

        let upper = if i + 1 < plans.len() {
            (*seq + plans[i + 1].0) / 2
        } else {
            u32::MAX
        };
        let lower = prev_upper;
        match_arms.push(quote! {
            #lower ..= #upper => #fn_name(
                layer, hidden_states, residual, positions, slot_mapping,
                cu_seqlens_q, seqused_k, block_table,
                max_seqlen_q, max_seqlen_k, kv_cache, rotary, device,
            ),
        });
        prev_upper = upper.saturating_add(1);

        // ── Post-loop (lm_head) bucket function ──
        if has_post_loop {
            let lm_fn_name = format_ident!("solver_lm_head_bucket_{}", i);
            let lm_stmts = emit_post_loop_stmts(&ds, num_layers);

            lm_head_bucket_fns.push(quote! {
                #[allow(unused_variables, unused_mut, unused_assignments)]
                #[inline(never)]
                unsafe fn #lm_fn_name(
                    lm_head: &LinearLayer,
                    hidden_states: OwnedTensor,
                    device: &mut GpuDevice,
                ) -> OwnedTensor {
                    let mut hidden_states = hidden_states;
                    let mut logits: Option<OwnedTensor> = None;

                    #(#lm_stmts)*

                    logits.expect("lm_head dispatch produced no logits")
                }
            });

            let lm_lower = lm_head_prev_upper;
            lm_head_match_arms.push(quote! {
                #lm_lower ..= #upper => #lm_fn_name(
                    lm_head, hidden_states, device,
                ),
            });
            lm_head_prev_upper = upper.saturating_add(1);
        }
    }

    let lm_head_dispatcher = if has_post_loop {
        quote! {
            pub unsafe fn solver_forward_lm_head(
                lm_head: &LinearLayer,
                num_tokens: u32,
                hidden_states: OwnedTensor,
                device: &mut GpuDevice,
            ) -> OwnedTensor {
                match num_tokens {
                    #(#lm_head_match_arms)*
                }
            }

            #(#lm_head_bucket_fns)*
        }
    } else {
        quote! {}
    };

    let model_hidden_states = if has_post_loop {
        quote! {
            /// Generated backbone: input_ids → hidden_states (post-norm).
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn solver_hidden_states(
                model: &LlamaModel,
                input_ids: TensorView<'_>,
                positions: TensorView<'_>,
                slot_mapping: TensorView<'_>,
                cu_seqlens_q: TensorView<'_>,
                seqused_k: TensorView<'_>,
                block_table: TensorView<'_>,
                max_seqlen_q: usize,
                max_seqlen_k: usize,
                kv_cache: &KvCachePool,
                device: &mut GpuDevice,
            ) -> OwnedTensor {
                let hidden_states = kernels::embedding_gather(
                    model.embed_tokens.weight,
                    *input_ids,
                    &mut device.caching,
                    device.compute_stream,
                );
                let mut hidden_states: OwnedTensor = hidden_states;
                let mut residual: Option<OwnedTensor> = None;
                let num_tokens = hidden_states.dim(0) as u32;

                for layer in model.layers.iter() {
                    let (hs, res) = solver_forward_layer(
                        layer, num_tokens, hidden_states, residual,
                        positions, slot_mapping, cu_seqlens_q, seqused_k,
                        block_table, max_seqlen_q, max_seqlen_k,
                        kv_cache, &model.rotary, device,
                    );
                    hidden_states = hs;
                    residual = Some(res);
                }

                let hs_gpu: GpuTensor = *hidden_states;
                let res_gpu: GpuTensor = residual.as_ref().unwrap().as_gpu_tensor();
                kernels::fused_add_rms_norm_inplace(
                    hs_gpu, res_gpu,
                    model.norm.weight, model.norm.eps,
                    device.compute_stream,
                );
                drop(residual);
                hidden_states
            }
        }
    } else {
        quote! {}
    };

    quote! {
        #model_hidden_states

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn solver_forward_layer(
            layer: &LlamaDecoderLayer,
            num_tokens: u32,
            hidden_states: OwnedTensor,
            residual: Option<OwnedTensor>,
            positions: TensorView<'_>,
            slot_mapping: TensorView<'_>,
            cu_seqlens_q: TensorView<'_>,
            seqused_k: TensorView<'_>,
            block_table: TensorView<'_>,
            max_seqlen_q: usize,
            max_seqlen_k: usize,
            kv_cache: &KvCachePool,
            rotary: &RotaryCache,
            device: &mut GpuDevice,
        ) -> (OwnedTensor, OwnedTensor) {
            match num_tokens {
                #(#match_arms)*
            }
        }

        #(#bucket_fns)*

        #lm_head_dispatcher
    }
}

// ── Entry-by-entry codegen (unchanged from before) ──────────────

fn emit_layer_stmts(ds: &DispatchSequence) -> Vec<TokenStream> {
    ds.entries_for_layer(0).filter_map(emit_entry).collect()
}

/// Emit statements for the post-loop (lm_head) phase.
/// These entries are tagged with `layer == num_layers` in the tile graph.
/// The enclosing function signature is:
///     solver_forward_lm_head_bucket_N(
///         lm_head: &LinearLayer,
///         final_norm_weight: GpuTensor,
///         final_norm_eps: f32,
///         hidden_states: OwnedTensor,
///         residual: Option<OwnedTensor>,
///         device: &mut GpuDevice,
///     ) -> OwnedTensor  // logits
fn emit_post_loop_stmts(ds: &DispatchSequence, num_layers: u16) -> Vec<TokenStream> {
    ds.entries_for_layer(num_layers)
        .filter_map(emit_post_loop_entry)
        .collect()
}

fn emit_post_loop_entry(entry: &DispatchEntry) -> Option<TokenStream> {
    match entry.kind {
        ImplDispatchKind::Noop => None,

        // lm_head GEMM via cuBLAS dispatch (fallback; solver may pick CUTLASS).
        ImplDispatchKind::CublasGemm => Some(quote! {{
            let __out = lm_head.forward(
                hidden_states.view(),
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            drop(hidden_states);
            logits = Some(__out);
        }}),

        // CUTLASS GEMM for lm_head — use the same launch-function pattern
        // as the decoder GEMMs but with `lm_head.dense_weight()` as the
        // weight and `hidden_states` (already-normed) as the input.
        ImplDispatchKind::CutlassGemm {
            tile_m,
            tile_n,
            stages,
        } => {
            let launch_fn = format_ident!("cutlass_gemm_{}x{}_s{}_launch", tile_m, tile_n, stages);
            Some(quote! {{
                let __act: GpuTensor = *hidden_states;
                let __m = __act.dim(0) as i32;
                let __k = __act.dim(1) as i32;
                let __w = lm_head.dense_weight();
                let __n = __w.dim(0) as i32;
                let __out = device.caching.alloc_tensor(
                    &[__m as usize, __n as usize], __act.dtype(),
                );
                let __rc = #launch_fn(
                    __out.as_mut_ptr::<u16>(),
                    __act.as_ptr::<u16>(),
                    __w.as_ptr::<u16>(),
                    __m, __n, __k,
                    1.0f32, 0.0f32,
                    device.compute_stream as u64,
                );
                debug_assert_eq!(__rc, 0, "CUTLASS GEMM failed (lm_head)");
                drop(hidden_states);
                logits = Some(__out);
            }})
        }

        ImplDispatchKind::CutlassGemv => Some(quote! {{
            let __act: GpuTensor = *hidden_states;
            let __m = __act.dim(0) as i32;
            let __k = __act.dim(1) as i32;
            let __w = lm_head.dense_weight();
            let __n = __w.dim(0) as i32;
            let __out = device.caching.alloc_tensor(
                &[__m as usize, __n as usize], __act.dtype(),
            );
            let __rc = cutlass_gemv_launch(
                __out.as_mut_ptr::<u16>(),
                __act.as_ptr::<u16>(),
                __w.as_ptr::<u16>(),
                __m, __n, __k,
                1.0f32, 0.0f32,
                device.compute_stream as u64,
            );
            debug_assert_eq!(__rc, 0, "CUTLASS GEMV failed (lm_head)");
            drop(hidden_states);
            logits = Some(__out);
        }}),

        // Any other kind isn't expected in the post-loop phase.
        _ => None,
    }
}

fn emit_entry(entry: &DispatchEntry) -> Option<TokenStream> {
    match entry.kind {
        ImplDispatchKind::Noop => None,

        ImplDispatchKind::RmsNorm => {
            let is_attn = entry.is_attn_norm.unwrap_or(true);
            if is_attn {
                Some(quote! {
                    let (n, r) = if let Some(res) = residual.take() {
                        let hs_gpu: GpuTensor = *hidden_states;
                        let res_gpu: GpuTensor = *res;
                        kernels::fused_add_rms_norm_inplace(
                            hs_gpu, res_gpu,
                            layer.input_layernorm.weight,
                            layer.input_layernorm.eps,
                            device.compute_stream,
                        );
                        (hidden_states, res)
                    } else {
                        let hs_gpu: GpuTensor = *hidden_states;
                        let n = kernels::rms_norm(
                            hs_gpu,
                            layer.input_layernorm.weight,
                            layer.input_layernorm.eps,
                            &mut device.caching,
                            device.compute_stream,
                        );
                        let r = hidden_states;
                        (n, r)
                    };
                    normed = Some(n);
                    residual = Some(r);
                })
            } else {
                Some(quote! {
                    let hs_gpu: GpuTensor = *hidden_states;
                    let res_gpu: GpuTensor = **residual.as_ref().unwrap();
                    kernels::fused_add_rms_norm_inplace(
                        hs_gpu, res_gpu,
                        layer.post_attention_layernorm.weight,
                        layer.post_attention_layernorm.eps,
                        device.compute_stream,
                    );
                    normed = Some(hidden_states);
                })
            }
        }

        ImplDispatchKind::CublasGemm => {
            let phase = entry.gemm_phase.unwrap();
            if phase == GemmPhase::Up {
                return Some(quote! {
                    if let Some(ref up_proj) = layer.mlp.up_proj {
                        let __out = up_proj.forward(
                            normed.as_ref().unwrap().view(),
                            &mut device.cublas,
                            &mut device.caching,
                            device.compute_stream,
                        );
                        let __gu = gate_up.take().unwrap();
                        let __concat = kernels::concat_dim1(
                            *__gu, *__out,
                            &mut device.caching,
                            device.compute_stream,
                        );
                        drop(__gu);
                        drop(__out);
                        gate_up = Some(__concat);
                    }
                });
            }
            let (input, weight, store) = gemm_operands(phase, entry.fused_residual);
            Some(quote! {{
                let __out = (#weight).forward(
                    #input,
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                #store
            }})
        }

        ImplDispatchKind::CutlassGemm {
            tile_m,
            tile_n,
            stages,
        } => {
            let phase = entry.gemm_phase.unwrap();
            if phase == GemmPhase::Up {
                return Some(quote! {
                    if let Some(ref up_proj) = layer.mlp.up_proj {
                        let __out = up_proj.forward(
                            normed.as_ref().unwrap().view(),
                            &mut device.cublas,
                            &mut device.caching,
                            device.compute_stream,
                        );
                        let __gu = gate_up.take().unwrap();
                        let __concat = kernels::concat_dim1(
                            *__gu, *__out,
                            &mut device.caching,
                            device.compute_stream,
                        );
                        drop(__gu);
                        drop(__out);
                        gate_up = Some(__concat);
                    }
                });
            }
            let (_, weight, store) = gemm_operands(phase, entry.fused_residual);
            let cutlass_input = cutlass_input_expr(phase);
            let launch_fn = format_ident!("cutlass_gemm_{}x{}_s{}_launch", tile_m, tile_n, stages);
            // Always beta=0: output buffer is freshly allocated.
            // Residual accumulation happens in fused_add_rms_norm_inplace.
            Some(quote! {{
                let __act: GpuTensor = #cutlass_input;
                let __m = __act.dim(0) as i32;
                let __k = __act.dim(1) as i32;
                let __w = (#weight).dense_weight();
                let __n = __w.dim(0) as i32;
                let __out = device.caching.alloc_tensor(
                    &[__m as usize, __n as usize], __act.dtype(),
                );
                let __rc = #launch_fn(
                    __out.as_mut_ptr::<u16>(),
                    __act.as_ptr::<u16>(),
                    __w.as_ptr::<u16>(),
                    __m, __n, __k,
                    1.0f32, 0.0f32,
                    device.compute_stream as u64,
                );
                debug_assert_eq!(__rc, 0, "CUTLASS GEMM failed");
                #store
            }})
        }

        ImplDispatchKind::CutlassGemv => {
            // CUTLASS SIMT GEMV: y[N] = W[N,K] @ x[K], only at M=1.
            // Same calling convention as CUTLASS GEMM, just a different launch fn.
            let phase = entry.gemm_phase.unwrap();
            if phase == GemmPhase::Up {
                return Some(quote! {
                    if let Some(ref up_proj) = layer.mlp.up_proj {
                        let __out = up_proj.forward(
                            normed.as_ref().unwrap().view(),
                            &mut device.cublas,
                            &mut device.caching,
                            device.compute_stream,
                        );
                        let __gu = gate_up.take().unwrap();
                        let __concat = kernels::concat_dim1(
                            *__gu, *__out,
                            &mut device.caching,
                            device.compute_stream,
                        );
                        drop(__gu);
                        drop(__out);
                        gate_up = Some(__concat);
                    }
                });
            }
            let (_, weight, store) = gemm_operands(phase, entry.fused_residual);
            let cutlass_input = cutlass_input_expr(phase);
            Some(quote! {{
                let __act: GpuTensor = #cutlass_input;
                let __m = __act.dim(0) as i32;
                let __k = __act.dim(1) as i32;
                let __w = (#weight).dense_weight();
                let __n = __w.dim(0) as i32;
                let __out = device.caching.alloc_tensor(
                    &[__m as usize, __n as usize], __act.dtype(),
                );
                let __rc = cutlass_gemv_launch(
                    __out.as_mut_ptr::<u16>(),
                    __act.as_ptr::<u16>(),
                    __w.as_ptr::<u16>(),
                    __m, __n, __k,
                    1.0f32, 0.0f32,
                    device.compute_stream as u64,
                );
                debug_assert_eq!(__rc, 0, "CUTLASS GEMV failed");
                #store
            }})
        }

        ImplDispatchKind::CutlassNormGemm { .. } => {
            let phase = entry.gemm_phase.unwrap();
            let (input, weight, store) = gemm_operands(phase, entry.fused_residual);
            Some(quote! {{
                let __out = (#weight).forward(
                    #input, &mut device.cublas, &mut device.caching, device.compute_stream,
                );
                #store
            }})
        }

        ImplDispatchKind::FusedQkvRopeCache => Some(quote! {{
            let __qkv = qkv_out.take().unwrap();
            let __q = kernels::fused_qkv_rope_cache(
                __qkv.as_gpu_tensor(),
                *positions, rotary.cos_sin_cache, *slot_mapping,
                *kv_cache.k_cache(layer.self_attn.layer_idx),
                *kv_cache.v_cache(layer.self_attn.layer_idx),
                layer.self_attn.q_size, layer.self_attn.kv_size,
                layer.self_attn.num_q_heads, layer.self_attn.head_dim,
                &mut device.caching, device.compute_stream,
            );
            drop(__qkv);
            qkv_out = Some(__q);
        }}),

        ImplDispatchKind::PrefillRopeCache => Some(quote! {{
            let __qkv = qkv_out.take().unwrap();
            let (__q, __k, __v) = kernels::split_qkv(
                __qkv.as_gpu_tensor(),
                layer.self_attn.q_size, layer.self_attn.kv_size,
                layer.self_attn.num_q_heads, layer.self_attn.num_kv_heads,
                layer.self_attn.head_dim,
                &mut device.caching, device.compute_stream,
            );
            drop(__qkv);
            let __nt = __q.as_gpu_tensor().dim(0);
            kernels::rotary_embedding_inplace(
                __q.as_gpu_tensor().reshape(&[__nt, layer.self_attn.q_size]),
                __k.as_gpu_tensor().reshape(&[__nt, layer.self_attn.kv_size]),
                *positions, rotary.cos_sin_cache,
                layer.self_attn.head_dim, device.compute_stream,
            );
            crate::model::attention_helpers::write_kv_cache(
                __k.view(), __v.view(), slot_mapping,
                kv_cache, layer.self_attn.layer_idx, device.compute_stream,
            );
            qkv_out = Some(__q);
            k_out = Some(__k);
            v_out = Some(__v);
        }}),

        ImplDispatchKind::RotaryEmbedding => Some(quote! {{
            let __qkv = **qkv_out.as_ref().unwrap();
            kernels::fused_qkv_rope(
                __qkv, *positions, rotary.cos_sin_cache,
                layer.self_attn.q_size, layer.self_attn.kv_size,
                layer.self_attn.head_dim, device.compute_stream,
            );
        }}),

        ImplDispatchKind::FlashInferAttention => Some(quote! {{
            let __q = qkv_out.take().unwrap();
            let __a = crate::model::attention_helpers::attention_decode_from_cache(
                __q.view(), cu_seqlens_q, seqused_k, block_table,
                max_seqlen_q, max_seqlen_k, layer.self_attn.scale,
                0.0, -1, kv_cache, layer.self_attn.layer_idx,
                device.num_sm, &mut device.caching, device.compute_stream,
                std::ptr::null(), 0, false,
            );
            drop(__q);
            attn_out = Some(__a);
        }}),

        ImplDispatchKind::FlashInferStandard => Some(quote! {{
            let __q = qkv_out.take().unwrap();
            let __k = k_out.take().unwrap();
            let __v = v_out.take().unwrap();
            let __a = crate::model::attention_helpers::attention_standard(
                __q.view(), __k.view(), __v.view(),
                cu_seqlens_q, seqused_k, block_table,
                max_seqlen_q, max_seqlen_k, layer.self_attn.scale,
                kv_cache, layer.self_attn.layer_idx,
                device.num_sm, &mut device.caching, device.compute_stream,
                std::ptr::null(), 0, false,
            );
            drop(__q); drop(__k); drop(__v);
            attn_out = Some(__a);
        }}),

        ImplDispatchKind::SiluAndMul => Some(quote! {{
            let __gu = gate_up.take().unwrap();
            let __activated = kernels::silu_and_mul_fused(
                __gu.as_gpu_tensor(), layer.mlp.intermediate_size,
                &mut device.caching, device.compute_stream,
            );
            drop(__gu);
            silu_out = Some(__activated);
        }}),
    }
}

fn cutlass_input_expr(phase: GemmPhase) -> TokenStream {
    match phase {
        GemmPhase::Qkv => quote! { **normed.as_ref().unwrap() },
        // attn_out is allocated 3D as [num_tokens, num_q_heads, head_dim]
        // by flash attention; reshape to 2D [num_tokens, q_size] for the
        // CUTLASS GEMM. The cuBLAS path handles this via the reshape in
        // gemm_operands; CUTLASS launches need the same flattened view.
        GemmPhase::OProj => quote! {{
            let __ao = attn_out.as_ref().unwrap();
            let __nt = __ao.dim(0);
            *__ao.view().reshape(&[__nt, layer.self_attn.q_size])
        }},
        GemmPhase::Gate => quote! { **normed.as_ref().unwrap() },
        GemmPhase::Up => quote! { **normed.as_ref().unwrap() },
        GemmPhase::Down => quote! { **silu_out.as_ref().unwrap() },
        GemmPhase::LmHead => quote! { *hidden_states },
    }
}

fn gemm_operands(
    phase: GemmPhase,
    _fused_residual: bool,
) -> (TokenStream, TokenStream, TokenStream) {
    match phase {
        GemmPhase::Qkv => (
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.self_attn.qkv_proj },
            quote! { drop(normed.take()); qkv_out = Some(__out); },
        ),
        GemmPhase::OProj => (
            quote! {{
                let __ao = attn_out.as_ref().unwrap();
                let __nt = __ao.dim(0);
                __ao.view().reshape(&[__nt, layer.self_attn.q_size])
            }},
            quote! { layer.self_attn.o_proj },
            quote! { drop(attn_out.take()); hidden_states = __out; },
        ),
        GemmPhase::Gate => (
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.mlp.gate_up_proj },
            quote! { gate_up = Some(__out); },
        ),
        GemmPhase::Up => (
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.mlp.gate_up_proj },
            quote! {},
        ),
        GemmPhase::Down => (
            quote! { silu_out.as_ref().unwrap().view() },
            quote! { layer.mlp.down_proj },
            quote! { drop(silu_out.take()); hidden_states = __out; },
        ),
        GemmPhase::LmHead => (
            // Input is the final-normed hidden states (residual already folded in
            // by the preceding RmsNorm step; see lm_head's RmsNorm entry below).
            quote! { hidden_states.view() },
            quote! { (*lm_head) },
            quote! { drop(hidden_states); logits = Some(__out); },
        ),
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn build_library(
    target: &TargetId,
    dims: crate::lowering::tile_graph::ModelDims,
) -> ImplementationLibrary {
    match target {
        TargetId::L4Sm89 => ImplementationLibrary::l4_sm89_starter(dims),
        TargetId::L40sSm89 => ImplementationLibrary::l40s_sm89_starter(dims),
        _ => ImplementationLibrary::l4_sm89_starter(dims),
    }
}

fn build_profile(target: &TargetId) -> TargetProfile {
    match target {
        TargetId::L4Sm89 => TargetProfile::l4_sm89(),
        TargetId::L40sSm89 => TargetProfile::l40s_sm89(),
        _ => TargetProfile::l4_sm89(),
    }
}

fn build_solve_grid(workloads: &WorkloadRange) -> Vec<u32> {
    PlanFamily::DEFAULT_GRID
        .iter()
        .copied()
        .filter(|&s| s >= workloads.min_tokens && s <= workloads.max_tokens)
        .collect()
}
