// SPDX-License-Identifier: Apache-2.0
//! Codegen for the `forward!` macro.
//!
//! Walks the solver's DispatchSequence entry by entry. Each entry
//! emits one kernel call. No interpretation, no shortcuts.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::compile_dsl::{ForwardDef, TargetId, WorkloadRange};
use super::dispatch::{DispatchEntry, DispatchSequence, GemmPhase, ImplDispatchKind};
use crate::dag::BufferId;
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
    let mut library = build_library(target_id, model.dims);
    let profile = build_profile(target_id);
    let grid = match &def.workloads {
        Some(wl) => build_solve_grid(wl),
        None => PlanFamily::DEFAULT_GRID.to_vec(),
    };
    let family = PlanFamily::solve_grid(
        &tile_graph,
        &mut library,
        &profile,
        &BacktrackCpSolver::default(),
        &grid,
    );

    if family.is_empty() {
        return quote! { compile_error!("forward!: solver found no feasible plans"); };
    }

    // Fusion decisions must be uniform across ALL buckets (the struct
    // is shared). Check if ANY bucket fuses — if so, all must.
    let plans_vec: Vec<_> = family.iter().collect();
    let qkv_fused = plans_vec.iter().any(|(_, plan)| {
        let ds = DispatchSequence::from_plan(plan, &library, &tile_graph);
        ds.entries_for_layer(0).any(|e| {
            matches!(
                e.kind,
                ImplDispatchKind::FusedQkvGemm | ImplDispatchKind::FusedQkvGemmWithBias
            )
        })
    });
    let gate_up_fused = plans_vec.iter().any(|(_, plan)| {
        let ds = DispatchSequence::from_plan(plan, &library, &tile_graph);
        ds.entries_for_layer(0)
            .any(|e| e.kind == ImplDispatchKind::FusedGateUpGemm)
    });

    // Extract struct fields from the DAG, applying solver fusion decisions.
    let (per_layer_fields, global_fields) =
        extract_weight_fields(&def.dag, qkv_fused, gate_up_fused);
    let struct_defs = emit_structs(&per_layer_fields, &global_fields);
    let model_load = emit_model_load(&per_layer_fields, &global_fields, qkv_fused, gate_up_fused);

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
                layer: &Layer,
                dims: &RuntimeDims,
                layer_idx: usize,
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
                let mut q_out: Option<OwnedTensor> = None;
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
                layer, dims, layer_idx, hidden_states, residual, positions, slot_mapping,
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
                    hidden_states: TensorView<'_>,
                    device: &mut GpuDevice,
                ) -> OwnedTensor {
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
            /// CUTLASS/cuBLAS lm_head dispatch. Takes `TensorView` (borrow)
            /// so the caller keeps the `OwnedTensor` alive — prevents the
            /// caching allocator from freeing the input while the async
            /// GPU kernel is still reading from it.
            pub unsafe fn solver_forward_lm_head(
                lm_head: &LinearLayer,
                num_tokens: u32,
                hidden_states: TensorView<'_>,
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
                model: &Model,
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

                for (layer_idx, layer) in model.layers.iter().enumerate() {
                    let (hs, res) = solver_forward_layer(
                        layer, &model.dims, layer_idx,
                        num_tokens, hidden_states, residual,
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

    let model_forward = if has_post_loop {
        quote! {
            impl Model {
                /// Full forward: input_ids → logits.
                ///
                /// Runs the generated backbone (embed → layers → norm),
                /// optionally gathers last-token hidden states, then
                /// projects through lm_head via solver dispatch.
                #[allow(clippy::too_many_arguments)]
                pub unsafe fn forward(
                    &self,
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
                    last_token_indices: Option<TensorView<'_>>,
                ) -> OwnedTensor {
                    let hs = solver_hidden_states(
                        self, input_ids, positions, slot_mapping,
                        cu_seqlens_q, seqused_k, block_table,
                        max_seqlen_q, max_seqlen_k, kv_cache, device,
                    );
                    // Gather last-token hidden states (scheduler optimization).
                    let hs = if let Some(indices) = last_token_indices {
                        kernels::embedding_gather(
                            hs.as_gpu_tensor(), *indices,
                            &mut device.caching, device.compute_stream,
                        )
                    } else {
                        hs
                    };
                    let num_tokens = hs.dim(0) as u32;
                    let logits = solver_forward_lm_head(
                        &self.lm_head, num_tokens, hs.view(), device,
                    );
                    drop(hs);
                    logits
                }
            }
        }
    } else {
        quote! {}
    };

    quote! {
        #struct_defs

        #model_load

        #model_forward

        #model_hidden_states

        #[allow(clippy::too_many_arguments)]
        pub unsafe fn solver_forward_layer(
            layer: &Layer,
            dims: &RuntimeDims,
            layer_idx: usize,
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
        ImplDispatchKind::Embed => None,
        ImplDispatchKind::FusedQkvGemm => None, // not in post-loop
        ImplDispatchKind::FusedGateUpGemm => None, // not in post-loop
        ImplDispatchKind::BiasAdd => None,
        ImplDispatchKind::CublasGemmWithBias => None,

        // lm_head GEMM via cuBLAS dispatch (fallback; solver may pick CUTLASS).
        ImplDispatchKind::CublasGemm => Some(quote! {{
            let __out = lm_head.forward(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
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
            logits = Some(__out);
        }}),

        // Any other kind isn't expected in the post-loop phase.
        _ => None,
    }
}

fn emit_entry(entry: &DispatchEntry) -> Option<TokenStream> {
    match entry.kind {
        ImplDispatchKind::Noop => None,
        ImplDispatchKind::Embed => None, // handled in pre-loop, not per-layer
        ImplDispatchKind::FusedQkvGemm | ImplDispatchKind::FusedQkvGemmWithBias => {
            // Fused QKV GEMM: one GEMM with concatenated weight.
            // The fused weight field on Layer is `self_attn_qkv_proj`.
            // For FusedQkvGemmWithBias, the LinearLayer's `bias` field
            // holds the concatenated bias; `LinearLayer::forward` applies
            // it automatically via cuBLAS bias epilogue.
            Some(quote! {{
                let __out = layer.self_attn_qkv_proj.forward(
                    normed.as_ref().unwrap().view(),
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(normed.take());
                qkv_out = Some(__out);
            }})
        }
        ImplDispatchKind::FusedGateUpGemm => {
            // Fused gate+up GEMM: one GEMM with concatenated weight.
            Some(quote! {{
                let __out = layer.mlp_gate_up_proj.forward(
                    normed.as_ref().unwrap().view(),
                    &mut device.cublas,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(normed.take());
                gate_up = Some(__out);
            }})
        }
        ImplDispatchKind::BiasAdd => {
            // Standalone bias add on a projection output.
            // Variable names match the bucket function locals:
            // q_out holds unfused Q output, k_out holds K, v_out holds V.
            // (qkv_out holds fused QKV output — BiasAdd only appears
            // in the unfused path.)
            let phase = entry.gemm_phase.unwrap();
            let (buf, weight) = match phase {
                GemmPhase::Q => (quote! { q_out }, quote! { layer.self_attn_q_proj }),
                GemmPhase::K => (quote! { k_out }, quote! { layer.self_attn_k_proj }),
                GemmPhase::V => (quote! { v_out }, quote! { layer.self_attn_v_proj }),
                _ => (quote! { hidden_states }, quote! { layer.self_attn_q_proj }), // fallback
            };
            Some(quote! {{
                let __t = #buf.as_ref().unwrap();
                let __bias = (#weight).dense_bias().expect("BiasAdd: no bias");
                kernels::bias_add_inplace(__t.as_gpu_tensor(), __bias, device.compute_stream);
            }})
        }
        ImplDispatchKind::CublasGemmWithBias => {
            // Fused GEMM+bias via cuBLAS. Same as CublasGemm but the
            // LinearLayer::forward picks the bias path automatically.
            let phase = entry.gemm_phase.unwrap();
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
                    {
                        let __out = layer.mlp_up_proj.forward(
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
                    {
                        let __out = layer.mlp_up_proj.forward(
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
                    {
                        let __out = layer.mlp_up_proj.forward(
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
                *kv_cache.k_cache(layer_idx),
                *kv_cache.v_cache(layer_idx),
                dims.q_size, dims.kv_size,
                dims.num_q_heads, dims.head_dim,
                &mut device.caching, device.compute_stream,
            );
            drop(__qkv);
            qkv_out = Some(__q);
        }}),

        ImplDispatchKind::PrefillRopeCache => Some(quote! {{
            let __qkv = qkv_out.take().unwrap();
            let (__q, __k, __v) = kernels::split_qkv(
                __qkv.as_gpu_tensor(),
                dims.q_size, dims.kv_size,
                dims.num_q_heads, dims.num_kv_heads,
                dims.head_dim,
                &mut device.caching, device.compute_stream,
            );
            drop(__qkv);
            let __nt = __q.as_gpu_tensor().dim(0);
            kernels::rotary_embedding_inplace(
                __q.as_gpu_tensor().reshape(&[__nt, dims.q_size]),
                __k.as_gpu_tensor().reshape(&[__nt, dims.kv_size]),
                *positions, rotary.cos_sin_cache,
                dims.head_dim, device.compute_stream,
            );
            crate::model::attention_helpers::write_kv_cache(
                __k.view(), __v.view(), slot_mapping,
                kv_cache, layer_idx, device.compute_stream,
            );
            qkv_out = Some(__q);
            k_out = Some(__k);
            v_out = Some(__v);
        }}),

        ImplDispatchKind::RotaryEmbedding => Some(quote! {{
            let __qkv = **qkv_out.as_ref().unwrap();
            kernels::fused_qkv_rope(
                __qkv, *positions, rotary.cos_sin_cache,
                dims.q_size, dims.kv_size,
                dims.head_dim, device.compute_stream,
            );
        }}),

        ImplDispatchKind::FlashInferAttention => Some(quote! {{
            let __q = qkv_out.take().unwrap();
            let __a = crate::model::attention_helpers::attention_decode_from_cache(
                __q.view(), cu_seqlens_q, seqused_k, block_table,
                max_seqlen_q, max_seqlen_k, dims.scale,
                0.0, -1, kv_cache, layer_idx,
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
                max_seqlen_q, max_seqlen_k, dims.scale,
                kv_cache, layer_idx,
                device.num_sm, &mut device.caching, device.compute_stream,
                std::ptr::null(), 0, false,
            );
            drop(__q); drop(__k); drop(__v);
            attn_out = Some(__a);
        }}),

        ImplDispatchKind::SiluAndMul => Some(quote! {{
            let __gu = gate_up.take().unwrap();
            let __activated = kernels::silu_and_mul_fused(
                __gu.as_gpu_tensor(), dims.intermediate_size,
                &mut device.caching, device.compute_stream,
            );
            drop(__gu);
            silu_out = Some(__activated);
        }}),

        // TK native attention — codegen is generated by the megakernel
        // compiler, not the per-op emitter. These entries only appear
        // in plans where the entire layer is a single persistent kernel;
        // the megakernel codegen path handles them. Reaching this branch
        // means the per-op emitter was incorrectly asked to emit a
        // DeviceCallable-only op.
        ImplDispatchKind::TkAttentionDecode | ImplDispatchKind::TkAttentionPrefill => {
            panic!("TK attention is DeviceCallable-only; per-op codegen cannot emit it")
        }
    }
}

fn cutlass_input_expr(phase: GemmPhase) -> TokenStream {
    match phase {
        GemmPhase::Q | GemmPhase::K | GemmPhase::V => quote! { **normed.as_ref().unwrap() },
        // attn_out is allocated 3D as [num_tokens, num_q_heads, head_dim]
        // by flash attention; reshape to 2D [num_tokens, q_size] for the
        // CUTLASS GEMM. The cuBLAS path handles this via the reshape in
        // gemm_operands; CUTLASS launches need the same flattened view.
        GemmPhase::OProj => quote! {{
            let __ao = attn_out.as_ref().unwrap();
            let __nt = __ao.dim(0);
            *__ao.view().reshape(&[__nt, dims.q_size])
        }},
        GemmPhase::Gate => quote! { **normed.as_ref().unwrap() },
        GemmPhase::Up => quote! { **normed.as_ref().unwrap() },
        GemmPhase::Down => quote! { **silu_out.as_ref().unwrap() },
        GemmPhase::LmHead => quote! { *hidden_states }, // TensorView derefs to GpuTensor
    }
}

fn gemm_operands(
    phase: GemmPhase,
    _fused_residual: bool,
) -> (TokenStream, TokenStream, TokenStream) {
    match phase {
        GemmPhase::Q => (
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.self_attn_q_proj },
            quote! { q_out = Some(__out); },
        ),
        GemmPhase::K => (
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.self_attn_k_proj },
            quote! { k_out = Some(__out); },
        ),
        GemmPhase::V => (
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.self_attn_v_proj },
            quote! { drop(normed.take()); v_out = Some(__out); },
        ),
        GemmPhase::OProj => (
            quote! {{
                let __ao = attn_out.as_ref().unwrap();
                let __nt = __ao.dim(0);
                __ao.view().reshape(&[__nt, dims.q_size])
            }},
            quote! { layer.self_attn_o_proj },
            quote! { drop(attn_out.take()); hidden_states = __out; },
        ),
        GemmPhase::Gate => (
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.mlp_gate_proj },
            quote! { gate_up = Some(__out); },
        ),
        GemmPhase::Up => (
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.mlp_up_proj },
            quote! {},
        ),
        GemmPhase::Down => (
            quote! { silu_out.as_ref().unwrap().view() },
            quote! { layer.mlp_down_proj },
            quote! { drop(silu_out.take()); hidden_states = __out; },
        ),
        GemmPhase::LmHead => (
            // Input is the final-normed hidden states (residual already folded in
            // by the preceding RmsNorm step; see lm_head's RmsNorm entry below).
            // hidden_states is TensorView (borrow) — caller owns the buffer.
            quote! { hidden_states },
            quote! { (*lm_head) },
            quote! { logits = Some(__out); },
        ),
    }
}

// ── Struct generation from DAG ──────────────────────────────────

/// A field to emit in a generated struct.
struct FieldSpec {
    name: String,
    /// Rust type as a string: "RmsNorm", "LinearLayer", "Embedding", "RotaryCache"
    ty: &'static str,
}

/// Walk the DAG and extract weight buffer references, grouped into
/// per-layer (Layer struct) and global (Model struct) fields.
/// The Rust type is inferred from which op consumes the weight.
fn extract_weight_fields(
    dag: &crate::dag::ModelDag,
    qkv_fused: bool,
    gate_up_fused: bool,
) -> (Vec<FieldSpec>, Vec<FieldSpec>) {
    use crate::dag::{BufferKind, OpKind};
    use std::collections::BTreeMap;

    // Map buffer id → consuming op kind (first consumer).
    let mut weight_op: BTreeMap<&BufferId, &OpKind> = BTreeMap::new();
    for op in &dag.ops {
        for input_id in op.inputs() {
            if let Some(buf) = dag.buffers.get(input_id)
                && buf.kind == BufferKind::Weight
                && !weight_op.contains_key(input_id)
            {
                weight_op.insert(input_id, &op.kind);
            }
        }
    }

    let mut per_layer = Vec::new();
    let mut global = Vec::new();

    for (buf_id, op_kind) in &weight_op {
        let buf = &dag.buffers[*buf_id];
        // Bias weight buffers (e.g. `self_attn.q_proj.bias`) are loaded
        // as part of the parent LinearLayer — they don't need a separate
        // struct field.
        if buf_id.0.ends_with(".bias") {
            continue;
        }
        let ty = match op_kind {
            OpKind::Embed { weights, .. } if weights == *buf_id => "Embedding",
            OpKind::RmsNorm { weights, .. } if weights == *buf_id => "RmsNorm",
            OpKind::Gemm { b, .. } | OpKind::GemmAdd { b, .. } if b == *buf_id => "LinearLayer",
            OpKind::RopeAppend { rotary, .. } if rotary == *buf_id => "RotaryCache",
            _ => continue,
        };

        let spec = FieldSpec {
            name: buf_id.0.clone(),
            ty,
        };
        if buf.per_layer {
            per_layer.push(spec);
        } else {
            global.push(spec);
        }
    }

    // Apply solver fusion decisions to the per-layer fields.
    if qkv_fused {
        per_layer.retain(|f| {
            !f.name.contains("q_proj") && !f.name.contains("k_proj") && !f.name.contains("v_proj")
        });
        per_layer.push(FieldSpec {
            name: "self_attn.qkv_proj".to_string(),
            ty: "LinearLayer",
        });
    }
    if gate_up_fused {
        per_layer.retain(|f| !f.name.contains("gate_proj") && !f.name.contains("up_proj"));
        per_layer.push(FieldSpec {
            name: "mlp.gate_up_proj".to_string(),
            ty: "LinearLayer",
        });
    }
    per_layer.sort_by(|a, b| a.name.cmp(&b.name));

    (per_layer, global)
}

/// Emit the `Layer`, `RuntimeDims`, and `Model` struct definitions
/// from the extracted field specs.
/// Convert a dotted HF path to a valid Rust field ident.
/// `self_attn.q_proj` → `self_attn_q_proj`
fn field_ident(name: &str) -> proc_macro2::Ident {
    format_ident!("{}", name.replace('.', "_"))
}

fn emit_structs(per_layer: &[FieldSpec], global: &[FieldSpec]) -> TokenStream {
    let layer_fields = per_layer.iter().map(|f| {
        let name = field_ident(&f.name);
        let ty = format_ident!("{}", f.ty);
        quote! { pub #name: #ty }
    });

    let model_fields = global.iter().map(|f| {
        let name = field_ident(&f.name);
        let ty = format_ident!("{}", f.ty);
        quote! { pub #name: #ty }
    });

    quote! {
        pub struct Layer {
            #(#layer_fields,)*
        }

        #[derive(Clone, Copy, Debug)]
        pub struct RuntimeDims {
            pub num_q_heads: usize,
            pub num_kv_heads: usize,
            pub head_dim: usize,
            pub q_size: usize,
            pub kv_size: usize,
            pub intermediate_size: usize,
            pub scale: f32,
        }

        pub struct Model {
            pub layers: Vec<Layer>,
            pub dims: RuntimeDims,
            #(#model_fields,)*
        }
    }
}

/// Emit `impl Model { pub unsafe fn load(...) }` — the generated
/// loader that reads weights from `GpuWeights` by their HF paths.
///
/// Fused fields (QKV, gate+up) allocate a contiguous buffer and
/// `take_into` each constituent weight at the right offset.
/// Unfused fields call `Linear::load` / `RmsNorm::load` / etc.
/// All derived from the plan + DAG buffer names — no hardcoded
/// mapping tables.
fn emit_model_load(
    per_layer: &[FieldSpec],
    global: &[FieldSpec],
    qkv_fused: bool,
    gate_up_fused: bool,
) -> TokenStream {
    // Per-layer field loads.
    let layer_loads: Vec<TokenStream> = per_layer.iter().map(|f| {
        let ident = field_ident(&f.name);
        match (f.ty, f.name.as_str()) {
            ("RmsNorm", _) => {
                // Build format string at macro time: "{layer_prefix}.input_layernorm"
                let fmt_str = format!("{{layer_prefix}}.{}", f.name);
                quote! {
                    let #ident = RmsNorm::load(
                        weights,
                        &format!(#fmt_str),
                        config.rms_norm_eps,
                    )?;
                }
            }
            ("LinearLayer", name) if name == "self_attn.qkv_proj" && qkv_fused => {
                // Fused QKV: allocate one buffer, take_into q/k/v.
                quote! {
                    let #ident = {
                        let q_name = format!("{layer_prefix}.self_attn.q_proj.weight");
                        let k_name = format!("{layer_prefix}.self_attn.k_proj.weight");
                        let v_name = format!("{layer_prefix}.self_attn.v_proj.weight");
                        let (q_shape, q_dtype) = weights.tensor_info(&q_name)
                            .ok_or_else(|| anyhow::anyhow!("weight not found: {q_name}"))?;
                        let hidden = if q_shape.len() == 2 { q_shape[1] } else { 1 };
                        let elem = q_dtype.size_bytes();
                        let q_bytes = q_size * hidden * elem;
                        let kv_bytes = kv_size * hidden * elem;
                        let total = q_bytes + 2 * kv_bytes;
                        let ptr = crate::driver::mem_alloc(total)?;
                        weights.record_alloc(ptr, total);
                        weights.take_into(&q_name, ptr, device.compute_stream)?;
                        weights.take_into(&k_name, ptr.add(q_bytes), device.compute_stream)?;
                        weights.take_into(&v_name, ptr.add(q_bytes + kv_bytes), device.compute_stream)?;
                        let w = GpuTensor::new(ptr, &[q_size + 2 * kv_size, hidden], q_dtype);
                        // Fuse bias if present (Qwen2).
                        let q_bias_name = format!("{layer_prefix}.self_attn.q_proj.bias");
                        let bias = if weights.contains(&q_bias_name) {
                            let k_bias_name = format!("{layer_prefix}.self_attn.k_proj.bias");
                            let v_bias_name = format!("{layer_prefix}.self_attn.v_proj.bias");
                            let (qb_shape, qb_dtype) = weights.tensor_info(&q_bias_name).unwrap();
                            let qb_bytes = qb_shape.iter().product::<usize>() * qb_dtype.size_bytes();
                            let (kb_shape, _) = weights.tensor_info(&k_bias_name).unwrap();
                            let kb_bytes = kb_shape.iter().product::<usize>() * qb_dtype.size_bytes();
                            let (vb_shape, _) = weights.tensor_info(&v_bias_name).unwrap();
                            let vb_bytes = vb_shape.iter().product::<usize>() * qb_dtype.size_bytes();
                            let total_bias = qb_bytes + kb_bytes + vb_bytes;
                            let total_elems = total_bias / qb_dtype.size_bytes();
                            let bias_ptr = crate::driver::mem_alloc(total_bias)?;
                            weights.record_alloc(bias_ptr, total_bias);
                            weights.take_into(&q_bias_name, bias_ptr, device.compute_stream)?;
                            weights.take_into(&k_bias_name, bias_ptr.add(qb_bytes), device.compute_stream)?;
                            weights.take_into(&v_bias_name, bias_ptr.add(qb_bytes + kb_bytes), device.compute_stream)?;
                            Some(GpuTensor::new(bias_ptr, &[total_elems], qb_dtype))
                        } else {
                            None
                        };
                        LinearLayer::Dense(Linear::new(w, bias))
                    };
                }
            }
            ("LinearLayer", name) if name == "mlp.gate_up_proj" && gate_up_fused => {
                // Fused gate+up: allocate one buffer, take_into gate and up.
                quote! {
                    let #ident = {
                        let gate_name = format!("{layer_prefix}.mlp.gate_proj.weight");
                        let up_name = format!("{layer_prefix}.mlp.up_proj.weight");
                        let (g_shape, g_dtype) = weights.tensor_info(&gate_name)
                            .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
                        let hidden = if g_shape.len() == 2 { g_shape[1] } else { 1 };
                        let elem = g_dtype.size_bytes();
                        let gate_bytes = g_shape.iter().product::<usize>() * elem;
                        let up_bytes = gate_bytes;
                        let total = gate_bytes + up_bytes;
                        let ptr = crate::driver::mem_alloc(total)?;
                        weights.record_alloc(ptr, total);
                        weights.take_into(&gate_name, ptr, device.compute_stream)?;
                        weights.take_into(&up_name, ptr.add(gate_bytes), device.compute_stream)?;
                        let w = GpuTensor::new(ptr, &[2 * intermediate_size, hidden], g_dtype);
                        LinearLayer::Dense(Linear::new(w, None))
                    };
                }
            }
            ("LinearLayer", _) => {
                let fmt_str = format!("{{layer_prefix}}.{}", f.name);
                quote! {
                    let #ident = LinearLayer::Dense(
                        Linear::load(weights, &format!(#fmt_str))?
                    );
                }
            }
            _ => quote! {},
        }
    }).collect();

    let layer_field_inits: Vec<TokenStream> = per_layer
        .iter()
        .map(|f| {
            let ident = field_ident(&f.name);
            quote! { #ident }
        })
        .collect();

    // Global field loads.
    let global_loads: Vec<TokenStream> = global
        .iter()
        .map(|f| {
            let ident = field_ident(&f.name);
            match f.ty {
                "Embedding" => {
                    let path = format!("model.{}", f.name);
                    quote! {
                        let #ident = Embedding::load(weights, #path)?;
                    }
                }
                "RmsNorm" => {
                    let path = format!("model.{}", f.name);
                    quote! {
                        let #ident = RmsNorm::load(weights, #path, config.rms_norm_eps)?;
                    }
                }
                "LinearLayer" => {
                    let name = &f.name;
                    quote! {
                        let #ident = if config.tie_word_embeddings {
                            LinearLayer::Dense(Linear::new(embed_tokens.weight, None))
                        } else {
                            LinearLayer::Dense(Linear::load(weights, #name)?)
                        };
                    }
                }
                "RotaryCache" => quote! {
                    let #ident = RotaryCache::new(
                        config.head_dim,
                        config.max_position_embeddings,
                        config.rope_theta,
                        config.llama3_rope_scaling.as_ref(),
                        dtype,
                        device,
                    )?;
                    weights.record_alloc(
                        #ident.cos_sin_cache.raw_ptr(),
                        #ident.cos_sin_cache.size_bytes(),
                    );
                },
                _ => quote! {},
            }
        })
        .collect();

    let global_field_inits: Vec<TokenStream> = global
        .iter()
        .map(|f| {
            let ident = field_ident(&f.name);
            quote! { #ident }
        })
        .collect();

    quote! {
        impl Model {
            /// Load weights from a `GpuWeights` holder using HF paths.
            ///
            /// # Safety
            /// `device` must be the live CUDA device; `weights` must own
            /// the safetensors mmap.
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn load(
                weights: &mut GpuWeights,
                config: &LlamaConfig,
                dtype: DType,
                device: &GpuDevice,
            ) -> anyhow::Result<(Self, LinearLayer)> {
                let num_q_heads = config.num_attention_heads;
                let num_kv_heads = config.num_kv_heads;
                let head_dim = config.head_dim;
                let q_size = num_q_heads * head_dim;
                let kv_size = num_kv_heads * head_dim;
                let intermediate_size = config.intermediate_size;
                let scale = 1.0f32 / (head_dim as f32).sqrt();

                #(#global_loads)*

                let mut layers = Vec::with_capacity(config.num_hidden_layers);
                for i in 0..config.num_hidden_layers {
                    let layer_prefix = format!("model.layers.{i}");
                    #(#layer_loads)*
                    layers.push(Layer {
                        #(#layer_field_inits,)*
                    });
                }

                let dims = RuntimeDims {
                    num_q_heads, num_kv_heads, head_dim,
                    q_size, kv_size, intermediate_size, scale,
                };

                // Separate lm_head from the Model struct — caller
                // manages it (cuBLAS lm_head for now).
                let lm_head_layer = if config.tie_word_embeddings {
                    LinearLayer::Dense(Linear::new(embed_tokens.weight, None))
                } else {
                    LinearLayer::Dense(Linear::load(weights, "lm_head")?)
                };

                Ok((Self {
                    layers,
                    dims,
                    #(#global_field_inits,)*
                }, lm_head_layer))
            }
        }
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
