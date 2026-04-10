// SPDX-License-Identifier: Apache-2.0
//! TokenStream codegen for the `forward!` macro.
//!
//! The codegen is a compiler backend. It walks the solver's
//! `DispatchSequence` entry by entry and emits one Rust call per
//! entry. It does NOT interpret, merge, or second-guess the plan.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::compile_dsl::{CompileDef, ModelId, TargetId, WorkloadRange};
use super::dispatch::{DispatchEntry, DispatchSequence, GemmPhase, ImplDispatchKind};
use crate::lowering::BacktrackCpSolver;
use crate::lowering::library::ImplementationLibrary;
use crate::lowering::solver::PlanFamily;
use crate::lowering::tile_graph::TileGraph;
use crate::target_profile::TargetProfile;

/// Top-level codegen entry point.
pub fn generate(def: &CompileDef) -> TokenStream {
    if def.is_fully_specialized() {
        generate_fully_specialized(def)
    } else if def.is_gpu_specialized() {
        generate_gpu_specialized(def)
    } else {
        generate_fully_dynamic(def)
    }
}

// ── Fully specialized path ──────────────────────────────────────

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
        return quote! {
            compile_error!("forward!: solver found no feasible plans");
        };
    }

    let mut bucket_fns = Vec::new();
    let mut match_arms = Vec::new();
    let mut prev_upper = 0u32;
    let plans: Vec<_> = family.iter().collect();

    for (i, (seq, plan)) in plans.iter().enumerate() {
        let ds = DispatchSequence::from_plan(plan, &library, &tile_graph);
        let fn_name = format_ident!("solver_layer_bucket_{}", i);
        let body = emit_layer_body(&ds);

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
                #body
            }
        });

        let upper = if i + 1 < plans.len() {
            let next_seq = plans[i + 1].0;
            (*seq + next_seq) / 2
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
        /// Solver-generated per-layer dispatch. Each bucket is emitted
        /// directly from the solver's DispatchSequence — one call per entry.
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

/// Walk the dispatch sequence for layer 0 and emit one call per entry.
/// The sequence is already in dependency order (step-sorted by the solver).
fn emit_layer_body(ds: &DispatchSequence) -> TokenStream {
    let layer0: Vec<_> = ds.entries_for_layer(0).collect();

    let mut stmts: Vec<TokenStream> = Vec::new();

    for entry in &layer0 {
        if let Some(stmt) = emit_entry(entry) {
            stmts.push(stmt);
        }
    }

    // The last non-noop entry should produce hidden_states + residual.
    // The layer contract: return (hidden_states, residual).
    quote! {
        // Mutable bindings for the dataflow — each entry reads/writes these.
        let mut hidden_states = hidden_states;
        let mut residual = residual;
        let mut normed: Option<OwnedTensor> = None;
        let mut qkv_out: Option<OwnedTensor> = None;
        let mut attn_out: Option<OwnedTensor> = None;
        let mut gate_up: Option<OwnedTensor> = None;
        let mut silu_out: Option<OwnedTensor> = None;

        #(#stmts)*

        (hidden_states, residual.unwrap())
    }
}

/// Emit one Rust statement for one DispatchEntry.
/// Returns None for Noop entries.
fn emit_entry(entry: &DispatchEntry) -> Option<TokenStream> {
    match entry.kind {
        ImplDispatchKind::Noop => None,

        ImplDispatchKind::RmsNorm => {
            let is_attn = entry.is_attn_norm.unwrap_or(true);
            if is_attn {
                // Attention norm: fused_add_rms_norm if we have a residual,
                // plain rms_norm for layer 0.
                Some(quote! {
                    let (n, r) = if let Some(res) = residual.take() {
                        let hs_gpu = *hidden_states;
                        let res_gpu = *res;
                        kernels::fused_add_rms_norm_inplace(
                            hs_gpu, res_gpu,
                            layer.input_layernorm.weight,
                            layer.input_layernorm.eps,
                            device.compute_stream,
                        );
                        (hidden_states, res)
                    } else {
                        let n = kernels::rms_norm(
                            *hidden_states,
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
                // MLP norm: always fused_add_rms_norm (post-attention).
                Some(quote! {
                    let hs = hidden_states;
                    let res = residual.as_ref().unwrap();
                    let res_gpu = res.as_gpu_tensor();
                    kernels::fused_add_rms_norm_inplace(
                        *hs, res_gpu,
                        layer.post_attention_layernorm.weight,
                        layer.post_attention_layernorm.eps,
                        device.compute_stream,
                    );
                    normed = Some(hs);
                })
            }
        }

        ImplDispatchKind::CublasGemm => Some(emit_gemm(entry, false)),

        ImplDispatchKind::CutlassGemm { .. } => Some(emit_gemm(entry, true)),

        ImplDispatchKind::CutlassNormGemm { .. } => {
            // Fused norm+GEMM — the norm is folded into the GEMM prologue.
            // TODO: wire CUTLASS prologue fusion launcher.
            // For now, emit as separate norm + GEMM.
            Some(emit_gemm(entry, false))
        }

        ImplDispatchKind::FusedQkvRopeCache => {
            Some(quote! {
                // Fused QKV split + RoPE + KV cache write.
                let qkv_tensor = qkv_out.as_ref().unwrap();
                kernels::fused_qkv_rope_cache(
                    qkv_tensor.as_gpu_tensor(),
                    positions,
                    slot_mapping,
                    &layer.self_attn,
                    kv_cache,
                    rotary,
                    device,
                );
            })
        }

        ImplDispatchKind::RotaryEmbedding => Some(quote! {
            kernels::rotary_embedding_inplace(
                qkv_out.as_ref().unwrap().as_gpu_tensor(),
                positions,
                rotary,
                &layer.self_attn,
                device.compute_stream,
            );
        }),

        ImplDispatchKind::FlashInferAttention => {
            Some(quote! {
                // FlashInfer FA2 standalone attention.
                let q_in = qkv_out.take().unwrap();
                let a = layer.self_attn.run_attention(
                    q_in.view(),
                    positions, cu_seqlens_q, seqused_k,
                    block_table, max_seqlen_q, max_seqlen_k,
                    kv_cache, device,
                );
                drop(q_in);
                attn_out = Some(a);
            })
        }

        ImplDispatchKind::SiluAndMul => Some(quote! {
            let gu = gate_up.take().unwrap();
            let activated = kernels::silu_and_mul_fused(
                gu.as_gpu_tensor(),
                layer.mlp.intermediate_size,
                &mut device.caching,
                device.compute_stream,
            );
            drop(gu);
            silu_out = Some(activated);
        }),

        ImplDispatchKind::TkFusedMlpBlock => {
            // TK fused MLP: entire MLP block in one kernel.
            // TODO: emit TK launch. For now, fall back to the
            // individual ops (the solver shouldn't pick this
            // until TK is wired).
            Some(quote! {
                // TK fused MLP — not yet wired, placeholder.
                compile_error!("TK fused MLP codegen not yet implemented");
            })
        }
    }
}

/// Emit a GEMM call for either cuBLAS or CUTLASS, based on the dispatch entry.
/// Routes the result to the right dataflow variable based on GemmPhase.
fn emit_gemm(entry: &DispatchEntry, use_cutlass: bool) -> TokenStream {
    let phase = entry.gemm_phase.unwrap();

    // Determine input/output/weight expressions based on phase.
    let (input_expr, weight_expr, output_binding) = match phase {
        GemmPhase::Qkv => (
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.self_attn.qkv_proj },
            quote! { qkv_out = Some(__out); },
        ),
        GemmPhase::OProj => (
            quote! { attn_out.as_ref().unwrap().view() },
            quote! { layer.self_attn.o_proj },
            quote! { hidden_states = __out; },
        ),
        GemmPhase::Gate => (
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.mlp.gate_up_proj },
            quote! { gate_up = Some(__out); },
        ),
        GemmPhase::Up => (
            // For dense models, gate_up_proj is fused [2*ID, HD].
            // The solver still has a separate Up entry, but the weight
            // is the same fused tensor. The gate entry already produced
            // the full [M, 2*ID] output. Up is a noop for dense.
            // For quantized with separate up_proj, this is a real GEMM.
            quote! { normed.as_ref().unwrap().view() },
            quote! { layer.mlp.up_proj_or_gate_up() },
            quote! {
                // For dense: up is part of fused gate_up, skip.
                // For quantized: concat gate + up.
                if layer.mlp.has_separate_up() {
                    let gu = gate_up.take().unwrap();
                    let concat = kernels::concat_dim1(
                        gu.as_gpu_tensor(),
                        __out.as_gpu_tensor(),
                        &mut device.caching,
                        device.compute_stream,
                    );
                    drop(gu);
                    drop(__out);
                    gate_up = Some(concat);
                }
            },
        ),
        GemmPhase::Down => (
            quote! { silu_out.as_ref().unwrap().view() },
            quote! { layer.mlp.down_proj },
            quote! { hidden_states = __out; },
        ),
    };

    if use_cutlass {
        let launch_fn = match entry.kind {
            ImplDispatchKind::CutlassGemm {
                tile_m: 64,
                tile_n: 64,
            } => {
                format_ident!("cutlass_gemm_64x64_launch")
            }
            _ => format_ident!("cutlass_gemm_128x128_launch"),
        };
        let beta = if entry.fused_residual {
            quote! { 1.0f32 }
        } else {
            quote! { 0.0f32 }
        };
        quote! {{
            let __input = #input_expr;
            let __act = __input.as_gpu_tensor();
            let __m = __act.dim(0) as i32;
            let __k = __act.dim(1) as i32;
            let __w = (#weight_expr).dense_weight();
            let __n = __w.dim(0) as i32;
            let __out = device.caching.alloc_tensor(
                &[__m as usize, __n as usize], __act.dtype(),
            );
            let __rc = #launch_fn(
                __out.as_gpu_tensor().as_mut_ptr::<u16>(),
                __act.as_ptr::<u16>(),
                __w.as_ptr::<u16>(),
                __m, __n, __k,
                1.0f32, #beta,
                device.compute_stream as u64,
            );
            debug_assert_eq!(__rc, 0, "CUTLASS GEMM failed");
            #output_binding
        }}
    } else {
        quote! {{
            let __out = (#weight_expr).forward(
                #input_expr,
                &mut device.cublas,
                &mut device.caching,
                device.compute_stream,
            );
            #output_binding
        }}
    }
}

/// Find the DispatchEntry for a given GEMM phase in a layer's entries.
fn find_gemm_entry<'a>(
    entries: &'a [&'a DispatchEntry],
    phase: GemmPhase,
) -> Option<&'a &'a DispatchEntry> {
    entries
        .iter()
        .find(|e| e.gemm_phase == Some(phase) && e.kind != ImplDispatchKind::Noop)
}

// ── GPU-specialized / fully dynamic (not yet implemented) ───────

fn generate_gpu_specialized(_def: &CompileDef) -> TokenStream {
    quote! {}
}

fn generate_fully_dynamic(_def: &CompileDef) -> TokenStream {
    quote! {}
}

// ── Helpers ─────────────────────────────────────────────────────

fn build_tile_graph(model: &ModelId) -> TileGraph {
    TileGraph::build_llama_forward(model.num_layers())
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
