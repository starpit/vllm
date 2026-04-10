// SPDX-License-Identifier: Apache-2.0
//! TokenStream codegen for the `forward!` macro.
//!
//! Emits the forward function directly — same types as the existing
//! LlamaModel::forward (OwnedTensor, GpuTensor, CublasHandle, etc.).
//! Each workload bucket gets a complete per-layer body with
//! solver-selected kernels.

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

    // Build per-bucket layer functions and the match arms.
    let mut bucket_fns = Vec::new();
    let mut match_arms = Vec::new();
    let mut prev_upper = 0u32;
    let plans: Vec<_> = family.iter().collect();

    for (i, (seq, plan)) in plans.iter().enumerate() {
        let ds = DispatchSequence::from_plan(plan, &library, &tile_graph);
        let fn_name = format_ident!("solver_layer_bucket_{}", i);
        let body = emit_layer_body(&ds);

        bucket_fns.push(quote! {
            #[allow(unused_variables, unused_mut)]
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
        /// Solver-generated per-layer dispatch.
        ///
        /// Selects the optimal kernel mix based on `num_tokens`.
        /// Each bucket is a complete layer body with solver-selected kernels.
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

/// Emit a complete per-layer body from the dispatch sequence.
/// Uses the same ops as LlamaDecoderLayer::forward but with
/// solver-selected GEMM implementations.
fn emit_layer_body(ds: &DispatchSequence) -> TokenStream {
    let layer0: Vec<_> = ds.entries_for_layer(0).collect();

    // Determine which GEMM impl is used for each phase.
    let qkv_gemm = find_gemm_entry(&layer0, GemmPhase::Qkv);
    let oproj_gemm = find_gemm_entry(&layer0, GemmPhase::OProj);
    let gate_gemm = find_gemm_entry(&layer0, GemmPhase::Gate);
    let up_gemm = find_gemm_entry(&layer0, GemmPhase::Up);
    let down_gemm = find_gemm_entry(&layer0, GemmPhase::Down);

    // Check for TK fused MLP (replaces gate+up+silu+down+norm).
    let has_tk_fused_mlp = layer0
        .iter()
        .any(|e| e.kind == ImplDispatchKind::TkFusedMlpBlock);

    // Emit attn norm.
    let attn_norm_code = quote! {
        let (normed, residual) = if let Some(residual) = residual {
            let hs_gpu = *hidden_states;
            let res_gpu = *residual;
            kernels::fused_add_rms_norm_inplace(
                hs_gpu, res_gpu,
                layer.input_layernorm.weight,
                layer.input_layernorm.eps,
                device.compute_stream,
            );
            (hidden_states, residual)
        } else {
            let normed = kernels::rms_norm(
                *hidden_states,
                layer.input_layernorm.weight,
                layer.input_layernorm.eps,
                &mut device.caching,
                device.compute_stream,
            );
            (normed, hidden_states)
        };
    };

    // Emit QKV GEMM.
    let qkv_code = emit_gemm_call(
        qkv_gemm,
        quote! { normed.view() },
        quote! { layer.self_attn.qkv_proj },
    );

    // Emit attention block (rope + cache + FA2 + oproj).
    let oproj_code = emit_gemm_call(
        oproj_gemm,
        quote! { attn_output.view() },
        quote! { layer.self_attn.o_proj },
    );

    let attn_block = quote! {
        let qkv_out = #qkv_code;
        drop(normed);

        let attn_output = layer.self_attn.forward(
            qkv_out.view(),
            positions, slot_mapping, cu_seqlens_q, seqused_k,
            block_table, max_seqlen_q, max_seqlen_k,
            kv_cache, rotary, device,
        );
        drop(qkv_out);

        // Post-attention residual add + oproj.
        // For now, use the standard oproj path.
        // TODO: when the solver picks fused oproj+residual (beta=1),
        // emit the fused variant.
        let oproj_out = #oproj_code;
        drop(attn_output);
    };

    // Emit MLP block.
    // For now, all MLP paths delegate to layer.mlp.forward() which
    // uses the eager cuBLAS path. When CUTLASS standalone launchers
    // and TK fused MLP are wired into vllm-cuda, the codegen will
    // emit direct kernel calls based on the solver's selection.
    let _ = (gate_gemm, up_gemm, down_gemm, has_tk_fused_mlp);
    let mlp_block = quote! {
        let res_gpu = *residual;
        kernels::fused_add_rms_norm_inplace(
            *oproj_out, res_gpu,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.eps,
            device.compute_stream,
        );
        let mlp_output = layer.mlp.forward(oproj_out.view(), device);
        drop(oproj_out);
        (mlp_output, residual)
    };

    quote! {
        #attn_norm_code
        #attn_block
        #mlp_block
    }
}

/// Emit a GEMM call for the solver-selected implementation.
fn emit_gemm_call(
    entry: Option<&&DispatchEntry>,
    input_expr: TokenStream,
    weight_expr: TokenStream,
) -> TokenStream {
    let cublas_fallback = quote! {
        #weight_expr.forward(
            #input_expr,
            &mut device.cublas,
            &mut device.caching,
            device.compute_stream,
        )
    };

    match entry {
        Some(e) => match e.kind {
            ImplDispatchKind::CublasGemm => cublas_fallback,
            ImplDispatchKind::CutlassGemm { tile_m, tile_n } => {
                let launch_fn = if tile_m == 64 && tile_n == 64 {
                    format_ident!("cutlass_gemm_64x64_launch")
                } else {
                    format_ident!("cutlass_gemm_128x128_launch")
                };
                let beta = if e.fused_residual {
                    quote! { 1.0f32 }
                } else {
                    quote! { 0.0f32 }
                };
                quote! {{
                    // CUTLASS standalone GEMM: C = A @ B^T + beta*C
                    let act_tensor = (#input_expr).as_gpu_tensor();
                    let m = act_tensor.dim(0) as i32;
                    let k = act_tensor.dim(1) as i32;
                    let w_tensor = #weight_expr.dense_weight();
                    let n = w_tensor.dim(0) as i32;
                    let out = device.caching.alloc_tensor(
                        &[m as usize, n as usize], act_tensor.dtype(),
                    );
                    let rc = #launch_fn(
                        out.as_gpu_tensor().as_mut_ptr::<u16>(),
                        act_tensor.as_ptr::<u16>(),
                        w_tensor.as_ptr::<u16>(),
                        m, n, k,
                        1.0f32, #beta,
                        device.compute_stream as u64,
                    );
                    debug_assert_eq!(rc, 0, "CUTLASS GEMM failed");
                    out
                }}
            }
            _ => cublas_fallback,
        },
        None => cublas_fallback,
    }
}

/// Find the DispatchEntry for a given GEMM phase in layer 0.
fn find_gemm_entry<'a>(
    entries: &'a [&'a DispatchEntry],
    phase: GemmPhase,
) -> Option<&'a &'a DispatchEntry> {
    entries
        .iter()
        .find(|e| e.gemm_phase == Some(phase) && e.kind != ImplDispatchKind::Noop)
}

// ── GPU-specialized path ────────────────────────────────────────

fn generate_gpu_specialized(def: &CompileDef) -> TokenStream {
    let _model_id = def.model.as_static().unwrap();

    // For the GPU-specialized path, we can't run the solver at
    // compile time (target is runtime). Emit a runtime struct that
    // solves at startup and dispatches via the eager path with
    // solver-selected GEMM overrides.
    // TODO: implement runtime dispatch table.
    quote! {
        // GPU-specialized forward! — solver runs at model load time.
        // Not yet implemented; falls back to eager dispatch.
    }
}

// ── Fully dynamic path ──────────────────────────────────────────

fn generate_fully_dynamic(_def: &CompileDef) -> TokenStream {
    // TODO: implement runtime interpreter dispatch.
    quote! {
        // Fully dynamic forward! — solver runs at model load time.
        // Not yet implemented; falls back to eager dispatch.
    }
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
