// SPDX-License-Identifier: Apache-2.0
//! Codegen for the `forward!` macro.
//!
//! Walks the solver's DispatchSequence entry by entry. Each entry
//! emits one kernel call. No interpretation, no shortcuts.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::compile_dsl::{CompileDef, ModelSpec, TargetId, WorkloadRange};
use super::dispatch::{DispatchEntry, DispatchSequence, GemmPhase, ImplDispatchKind};
use crate::lowering::BacktrackCpSolver;
use crate::lowering::library::ImplementationLibrary;
use crate::lowering::solver::PlanFamily;
use crate::lowering::tile_graph::TileGraph;
use crate::target_profile::TargetProfile;

pub fn generate(def: &CompileDef) -> TokenStream {
    if def.is_fully_specialized() {
        generate_fully_specialized(def)
    } else {
        // GPU-specialized and fully dynamic not yet implemented.
        quote! {}
    }
}

fn generate_fully_specialized(def: &CompileDef) -> TokenStream {
    let model_id = def.model.as_static().unwrap();
    let target_id = def.target.as_static().unwrap();
    let workloads = def.workloads.as_static().unwrap();

    let tile_graph = build_tile_graph(model_id);
    let library = build_library(target_id);
    let profile = build_profile(target_id);
    let grid = build_solve_grid(workloads);
    let family = PlanFamily::solve_grid(&tile_graph, &library, &profile, &BacktrackCpSolver, &grid);

    if family.is_empty() {
        return quote! { compile_error!("forward!: solver found no feasible plans"); };
    }

    let mut bucket_fns = Vec::new();
    let mut match_arms = Vec::new();
    let mut prev_upper = 0u32;
    let plans: Vec<_> = family.iter().collect();

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
    }

    quote! {
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
    }
}

// ── Entry-by-entry codegen ──────────────────────────────────────

/// Walk layer 0's dispatch entries, emit one statement per entry.
fn emit_layer_stmts(ds: &DispatchSequence) -> Vec<TokenStream> {
    ds.entries_for_layer(0).filter_map(emit_entry).collect()
}

/// One dispatch entry → one kernel call.
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
                // MLP norm: fused_add_rms_norm_inplace mutates both in-place.
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
                // For dense models, Gate already produced [M, 2*intermediate]
                // via the fused gate_up_proj. Up is only needed for quantized
                // (separate up_proj). Guard the entire GEMM at runtime.
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

        ImplDispatchKind::CutlassGemm { tile_m, tile_n } => {
            let phase = entry.gemm_phase.unwrap();
            if phase == GemmPhase::Up {
                // Same as cuBLAS Up: skip for dense, only run for quantized.
                // CUTLASS doesn't apply to quantized, so this is a noop.
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
            let launch_fn = if tile_m == 64 && tile_n == 64 {
                format_ident!("cutlass_gemm_64x64_launch")
            } else {
                format_ident!("cutlass_gemm_128x128_launch")
            };
            // Always beta=0 for CUTLASS — the output buffer is freshly
            // allocated. The fused_residual flag means the solver folded
            // the ResidualAdd tile into this subgraph, but the residual
            // accumulation happens in fused_add_rms_norm_inplace, not
            // in the GEMM epilogue.
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

        ImplDispatchKind::CutlassNormGemm { .. } => {
            // Not yet wired — emit cuBLAS fallback.
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

        ImplDispatchKind::FusedQkvRopeCache => {
            // Decode: fused split + Q RoPE + cache write.
            Some(quote! {{
                let __qkv = qkv_out.take().unwrap();
                let __q = kernels::fused_qkv_rope_cache(
                    __qkv.as_gpu_tensor(),
                    *positions,
                    rotary.cos_sin_cache,
                    *slot_mapping,
                    *kv_cache.k_cache(layer.self_attn.layer_idx),
                    *kv_cache.v_cache(layer.self_attn.layer_idx),
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.num_q_heads,
                    layer.self_attn.head_dim,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(__qkv);
                qkv_out = Some(__q);
            }})
        }

        ImplDispatchKind::PrefillRopeCache => {
            // Prefill: split QKV, rotate both Q and K, write cache.
            Some(quote! {{
                let __qkv = qkv_out.take().unwrap();
                let (__q, __k, __v) = kernels::split_qkv(
                    __qkv.as_gpu_tensor(),
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.num_q_heads,
                    layer.self_attn.num_kv_heads,
                    layer.self_attn.head_dim,
                    &mut device.caching,
                    device.compute_stream,
                );
                drop(__qkv);
                let __nt = __q.as_gpu_tensor().dim(0);
                kernels::rotary_embedding_inplace(
                    __q.as_gpu_tensor().reshape(&[__nt, layer.self_attn.q_size]),
                    __k.as_gpu_tensor().reshape(&[__nt, layer.self_attn.kv_size]),
                    *positions,
                    rotary.cos_sin_cache,
                    layer.self_attn.head_dim,
                    device.compute_stream,
                );
                crate::model::attention_helpers::write_kv_cache(
                    __k.view(), __v.view(), slot_mapping,
                    kv_cache, layer.self_attn.layer_idx,
                    device.compute_stream,
                );
                qkv_out = Some(__q);
                k_out = Some(__k);
                v_out = Some(__v);
            }})
        }

        ImplDispatchKind::RotaryEmbedding => Some(quote! {{
            let __qkv = **qkv_out.as_ref().unwrap();
            kernels::fused_qkv_rope(
                __qkv,
                *positions,
                rotary.cos_sin_cache,
                layer.self_attn.q_size,
                layer.self_attn.kv_size,
                layer.self_attn.head_dim,
                device.compute_stream,
            );
        }}),

        ImplDispatchKind::FlashInferAttention => {
            // Decode: attention from paged KV cache.
            Some(quote! {{
                let __q = qkv_out.take().unwrap();
                let __a = crate::model::attention_helpers::attention_decode_from_cache(
                    __q.view(),
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    layer.self_attn.scale,
                    0.0,
                    -1,
                    kv_cache,
                    layer.self_attn.layer_idx,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                    std::ptr::null(),
                    0,
                    false,
                );
                drop(__q);
                attn_out = Some(__a);
            }})
        }

        ImplDispatchKind::FlashInferStandard => {
            // Prefill: attention_standard with explicit Q, K, V.
            Some(quote! {{
                let __q = qkv_out.take().unwrap();
                let __k = k_out.take().unwrap();
                let __v = v_out.take().unwrap();
                let __a = crate::model::attention_helpers::attention_standard(
                    __q.view(),
                    __k.view(),
                    __v.view(),
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    max_seqlen_k,
                    layer.self_attn.scale,
                    kv_cache,
                    layer.self_attn.layer_idx,
                    device.num_sm,
                    &mut device.caching,
                    device.compute_stream,
                    std::ptr::null(),
                    0,
                    false,
                );
                drop(__q);
                drop(__k);
                drop(__v);
                attn_out = Some(__a);
            }})
        }

        ImplDispatchKind::SiluAndMul => Some(quote! {{
            let __gu = gate_up.take().unwrap();
            let __activated = kernels::silu_and_mul_fused(
                __gu.as_gpu_tensor(),
                layer.mlp.intermediate_size,
                &mut device.caching,
                device.compute_stream,
            );
            drop(__gu);
            silu_out = Some(__activated);
        }}),

        ImplDispatchKind::TkFusedMlpBlock => Some(quote! {
            compile_error!("TK fused MLP codegen not yet implemented");
        }),
    }
}

/// For CUTLASS: return a GpuTensor expression for the input activation.
/// Derefs from the OwnedTensor dataflow variable.
fn cutlass_input_expr(phase: GemmPhase) -> TokenStream {
    match phase {
        GemmPhase::Qkv => quote! { **normed.as_ref().unwrap() },
        GemmPhase::OProj => quote! { **attn_out.as_ref().unwrap() },
        GemmPhase::Gate => quote! { **normed.as_ref().unwrap() },
        GemmPhase::Up => quote! { **normed.as_ref().unwrap() },
        GemmPhase::Down => quote! { **silu_out.as_ref().unwrap() },
    }
}

/// For a GEMM phase, return (input_expr, weight_expr, store_stmt).
/// input_expr is a TensorView (for cuBLAS LinearLayer::forward).
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
            // Reshape attention output from [num_tokens, num_heads, head_dim]
            // to [num_tokens, q_size] for the o_proj GEMM.
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
            // For quantized: use the separate up_proj weight.
            // For dense: gate_up_proj is fused, Gate already produced the full output.
            quote! { layer.mlp.gate_up_proj },
            quote! {
                // Up GEMM — only meaningful for quantized (separate up_proj).
                // For dense, Gate already produced [M, 2*intermediate].
                if layer.mlp.up_proj.is_some() {
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
            },
        ),
        GemmPhase::Down => (
            quote! { silu_out.as_ref().unwrap().view() },
            quote! { layer.mlp.down_proj },
            quote! { drop(silu_out.take()); hidden_states = __out; },
        ),
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn build_tile_graph(model: &ModelSpec) -> TileGraph {
    TileGraph::build_llama_forward(model.num_layers(), model.dims())
}

fn build_library(target: &TargetId) -> ImplementationLibrary {
    match target {
        TargetId::L4Sm89 => ImplementationLibrary::l4_sm89_starter(),
        _ => ImplementationLibrary::l4_sm89_starter(),
    }
}

fn build_profile(target: &TargetId) -> TargetProfile {
    match target {
        TargetId::L4Sm89 => TargetProfile::l4_sm89(),
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
