// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]
//! Core logic for the `megakernel!` proc-macro.
//!
//! This is a regular library crate (not proc-macro) so it can be used by both:
//! - The `vllm-tk-macros` proc-macro crate (for compile-time expansion)
//! - The `vllm-tk-static` build.rs (for generating CUDA source to compile)

#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::useless_vec
)]
pub mod cpu_golden;
pub mod cuda_codegen;
pub mod dag;
pub mod diagram;
pub mod fused_codegen;
pub mod kernel_library;
pub mod parse;
pub mod reified_dag;
pub mod schedule;
pub mod scheduled_codegen;
pub mod target_profile;
pub mod verify;

/// Generate the complete CUDA source for a LLaMA-like megakernel from DSL source.
///
/// This is the entry point for build.rs: pass the same DSL text that `megakernel!`
/// would parse, get back the CUDA source string.
pub fn generate_cuda_from_dsl(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;

    let dag = parse::build_dag(&def)?;

    let errors = verify::verify(&dag);
    if !errors.is_empty() {
        let mut msg = String::from("verification failed:\n");
        for e in &errors {
            msg.push_str(&format!("  {e}\n"));
        }
        return Err(msg);
    }

    Ok(cuda_codegen::generate_static_kernel(&dag))
}

/// Generate a standalone CUDA test kernel for a single TK op.
///
/// `dsl` is the same DSL text as `generate_cuda_from_dsl`.
/// `op_name` is one of the TK op names (e.g. "attn_norm", "gate_silu").
/// Returns CUDA source with a `test_{op_name}_launch(...)` C entry point.
pub fn generate_single_op_kernel(dsl: &str, op_name: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    cuda_codegen::generate_single_op_kernel(&dag, op_name)
}

/// List all op names that can be tested with `generate_single_op_kernel`.
pub fn available_op_names() -> Vec<&'static str> {
    cuda_codegen::available_op_names()
}

/// Generate an inline RMSNorm kernel (no KVM protocol).
///
/// Uses TK tile primitives + group::sync only. No semaphores, no pages, no warp roles.
pub fn generate_inline_rmsnorm_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    Ok(cuda_codegen::generate_inline_rmsnorm_kernel(&dag))
}

/// Generate an inline GEMM kernel (no KVM protocol).
///
/// Double-buffered K-loop with 8 cooperative warps using TK tile primitives.
pub fn generate_inline_gemm_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    Ok(cuda_codegen::generate_inline_gemm_kernel(&dag))
}

/// Generate a fused MLP block kernel (no KVM protocol).
pub fn generate_fused_mlp_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_mlp_kernel(&dag))
}

/// Generate a fused RMSNorm → GEMM kernel (no KVM protocol, shmem inter-op passing).
pub fn generate_fused_rmsnorm_gemm_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    Ok(cuda_codegen::generate_fused_rmsnorm_gemm_kernel(&dag))
}

/// Generate an inline attention decode kernel (no KVM protocol).
///
/// Each of 8 warps handles 1 KV head with GQA_RATIO query heads.
pub fn generate_inline_attention_decode_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_inline_attention_decode_kernel(&dag))
}

/// Generate a fused single-layer kernel (no KVM protocol).
///
/// Chains: attn_norm → QKV GEMM → [skip attention] → o_proj+residual → MLP block.
pub fn generate_fused_layer_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_layer_kernel(&dag))
}

/// Generate a fused full-layer kernel WITH attention decode (no KVM protocol).
pub fn generate_fused_full_layer_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_full_layer_kernel(&dag))
}

/// Generate a fused multi-layer kernel WITH attention decode (no KVM protocol).
pub fn generate_fused_multi_layer_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_multi_layer_kernel(&dag))
}

/// Generate a fused multi-SM kernel that distributes GEMM tiles across CTAs.
///
/// Uses cross-CTA barriers (atomicAdd + spin) for synchronization between phases.
/// RMSNorm and attention run on CTA 0; GEMMs are partitioned across all CTAs.
pub fn generate_fused_multi_sm_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_multi_sm_kernel(&dag))
}

/// Generate a fused prefill attention kernel (no KVM protocol).
///
/// Phase 1: attention-only. Q from q_post, paged KV cache, output to attn_out.
/// Grid = ceil(num_prefill_tokens / 16) * num_kv_heads CTAs.
pub fn generate_fused_prefill_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;
    Ok(cuda_codegen::generate_fused_prefill_kernel(&dag))
}

use fused_codegen::units::Count;

/// Generate a fused single-layer prefill kernel (no KVM protocol).
///
/// Full layer: attn_norm → QKV GEMM → attention → o_proj+residual → MLP block.
/// Grid = ceil(num_prefill_tokens / 16). Each CTA owns 16 rows through the full layer.
///
/// Backend selection via `TK_FUSED_PREFILL` env var:
/// - unset or `v1`: monolithic v1 codegen (default — known good)
/// - `v2-16row`: template-based v2 codegen, 16-row CTAs (Redundant GEMM)
/// - `v2-32row`: template-based v2 codegen, 32-row CTAs (Cooperative GEMM, 2/8 warps productive)
/// - `v2-64row`: template-based v2 codegen, 64-row CTAs (Cooperative GEMM, 4/8 warps productive)
/// - `v2-128row`: template-based v2 codegen, 128-row CTAs (Cooperative GEMM, 8/8 warps productive)
pub fn generate_fused_prefill_layer_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    let backend = std::env::var("TK_FUSED_PREFILL").unwrap_or_else(|_| "auto".to_string());
    let cfg = match backend.as_str() {
        "auto" => return Ok(fused_codegen::generate_fused_prefill_polyalgorithm(&dag)),
        "v1" => return Ok(cuda_codegen::generate_fused_prefill_layer_kernel(&dag)),
        "v2-16row" => fused_codegen::config::FusedPrefillConfig::rows16(),
        "v2-16row-col4" => fused_codegen::config::FusedPrefillConfig::rows16_col4(),
        "v2-32row" => fused_codegen::config::FusedPrefillConfig::rows32(),
        "v2-64row" => fused_codegen::config::FusedPrefillConfig::rows64(),
        "v2-128row" => fused_codegen::config::FusedPrefillConfig::rows128(),
        "v2-mcta" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows16_col4();
            return Ok(fused_codegen::generate_fused_prefill_mcta(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-col1" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows16();
            return Ok(fused_codegen::generate_fused_prefill_mcta(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128();
            return Ok(fused_codegen::generate_fused_prefill_mcta(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-fused" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-wide" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_wide();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-64row-k128" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows64_k128();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-64row-3stage" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows64_3stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-64row-nosync" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows64_nosync();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-32row-nosync-k128" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows32_nosync_k128();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-64row-wide-3stage" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows64_wide_3stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-64row-k128-grid32" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows64_k128();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(32),
            ));
        }
        "v2-warpspec-112row" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows112_warpspec();
            return Ok(fused_codegen::generate_fused_prefill_mcta_warpspec(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-warpspec-48row" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows48_warpspec();
            return Ok(fused_codegen::generate_fused_prefill_mcta_warpspec(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-32row-wide-nosync" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows32_wide_nosync();
            return Ok(fused_codegen::generate_fused_prefill_mcta(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-64row-k128-grid64" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows64_k128();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(64),
            ));
        }
        // ── gemm_warp_m variants (PFL_GEMM_M > 16) ──
        "v2-mcta-64row-gemm32" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows64_gemm32();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-gemm32" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm32();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-gemm64" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm64();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-gemm32-wide" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm32_wide();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-64row-gemm32-k128" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows64_gemm32_k128();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        // ── Round 2: bigger-CTA / fewer-barrier variants ──
        "v2-mcta-256row-gemm32" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows256_gemm32();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm64" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows256_gemm64();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-gemm32-3stage" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm32_3stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        // ── Round 3: occupancy/per-CTA-work probes ──
        "v2-mcta-192row-gemm32" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows192_gemm32();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-64row-gemm32-nosync" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows64_gemm32_nosync();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-gemm32-narrow" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm32_narrow();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        // ── Round 4: dual-accumulator fused gate+up (A reuse) ──
        "v2-mcta-128row-gemm16-dual" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm16_dual();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-gemm32-dual" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm32_dual();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        // ── Round 5: 1-stage dual_accum (2 CTAs/SM + A reuse) ──
        "v2-mcta-128row-gemm16-dual-1stage" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm16_dual_1stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-gemm32-dual-1stage" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm32_dual_1stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-dual-1stage" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows256_gemm32_dual_1stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        // ── Round 6: k_dim=128 + narrow variants ──
        "v2-mcta-128row-gemm32-dual-1stage-k128" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm32_dual_1stage_k128();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-dual-1stage-k128" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows256_gemm32_dual_1stage_k128();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-dual-1stage-narrow" => {
            let cfg =
                fused_codegen::config::FusedPrefillConfig::rows256_gemm32_dual_1stage_narrow();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        // ── Round 7: col-fixed CTA scheduling (L2 reuse on B tiles) ──
        "v2-mcta-256row-gemm32-dual-1stage-colfix" => {
            let cfg =
                fused_codegen::config::FusedPrefillConfig::rows256_gemm32_dual_1stage_colfix();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-gemm16-dual-1stage-colfix" => {
            let cfg =
                fused_codegen::config::FusedPrefillConfig::rows128_gemm16_dual_1stage_colfix();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        // ── Round 8: K-stripe inner loop (CUTLASS-style) ──
        "v2-mcta-256row-gemm64-kstripe-1stage" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows256_gemm64_kstripe_1stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-128row-gemm64-kstripe-1stage" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows128_gemm64_kstripe_1stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-kstripe-wide-1stage" => {
            let cfg =
                fused_codegen::config::FusedPrefillConfig::rows256_gemm32_kstripe_wide_1stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-dual-1stage-cutlass4" => {
            let cfg =
                fused_codegen::config::FusedPrefillConfig::rows256_gemm32_dual_1stage_cutlass4();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-dual-1stage-cutlass3" => {
            let cfg =
                fused_codegen::config::FusedPrefillConfig::rows256_gemm32_dual_1stage_cutlass3();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-dual-1stage-cutlass-down" => {
            let cfg =
                fused_codegen::config::FusedPrefillConfig::rows256_gemm32_dual_1stage_cutlass_down(
                );
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-dual-1stage-phase-opt" => {
            let cfg =
                fused_codegen::config::FusedPrefillConfig::rows256_gemm32_dual_1stage_phase_opt();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-dual-2stage" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows256_gemm32_dual_2stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-1stage" => {
            let cfg = fused_codegen::config::FusedPrefillConfig::rows256_gemm32_1stage();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        "v2-mcta-256row-gemm32-dual-1stage-kstripe" => {
            let cfg =
                fused_codegen::config::FusedPrefillConfig::rows256_gemm32_dual_1stage_kstripe();
            return Ok(fused_codegen::generate_fused_prefill_mcta_fused_gateup(
                &dag,
                &cfg,
                Count(128),
            ));
        }
        other => {
            return Err(format!(
                "unknown TK_FUSED_PREFILL backend '{other}' \
                 (expected v1, v2-16row, v2-16row-col4, v2-32row, v2-64row, v2-128row, v2-mcta, v2-mcta-col1, v2-mcta-128row, v2-mcta-128row-fused, v2-mcta-128row-wide, v2-mcta-64row-k128, v2-mcta-64row-3stage, or v2-mcta-64row-nosync)"
            ));
        }
    };
    Ok(fused_codegen::generate_fused_prefill_v2(&dag, &cfg))
}

/// Phase 3b — generate a placeholder scheduled megakernel for a tiny test
/// model (2 layers, 32 tokens). Includes the schedule data, placeholder
/// `tile_<phase>` stubs, the persistent CTA loop, and an `extern "C"`
/// `launch_scheduled_megakernel` host helper.
///
/// Returns the complete `.cu` source so build.rs can drop it into the
/// cudaforge static library.
pub fn scheduled_prefill_tiny_dims() -> reified_dag::LlamaDims {
    // Tiny model — keeps the schedule small (~hundreds of nodes) so the
    // validator runs in microseconds and the static array fits in a few KB.
    reified_dag::LlamaDims {
        num_layers: 2,
        hidden_dim: 256,
        intermediate_dim: 512,
        num_attn_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        seq_len: 32,
    }
}

/// CTA pool size used for the tiny scheduled megakernel test fixture.
pub const SCHEDULED_PREFILL_TINY_CTAS: u32 = 4;

/// KV cache page size (slots per page) for the scheduled megakernel.
/// Matches the existing fused prefill kernel's PFL_KV_PAGE_SIZE.
pub const SCHEDULED_PREFILL_KV_PAGE_SIZE: u32 = 16;

pub fn generate_scheduled_prefill_tiny() -> String {
    use crate::kernel_library::coalesce_with_flashinfer_attention as coalesce;
    use crate::reified_dag::{ReifiedDag, TileSizes};
    use crate::schedule::{CostModel, partition_into_waves};
    use crate::scheduled_codegen::emit_scheduled_megakernel_cu;
    use crate::target_profile::TargetProfile;

    let profile = TargetProfile::l4_sm89();
    let dims = scheduled_prefill_tiny_dims();
    let reified = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
    let dag = coalesce(&reified);
    let cost = CostModel::from_dag(&dag);
    // CTA pool size = profile.cooperative_grid_size() so the
    // FlashInfer attention runner inside the megakernel sees a
    // grid that matches its planner's `num_blks_y`.
    let sched = partition_into_waves(&dag, profile.cooperative_grid_size(), &cost, 100);
    emit_scheduled_megakernel_cu(
        &dag,
        &sched,
        SCHEDULED_PREFILL_KV_PAGE_SIZE,
        &profile,
        "tiny",
    )
}

/// Phase 3d — medium variant for scaling validation. NL=4, seq=64,
/// HD=512, ID=1024, NAH=8, NKH=4, HDM=64. Exercises:
///   - multi-page KV cache (seq=64 / page_size=16 = 4 pages per layer)
///   - more layers (NL=4) catching cross-layer cache slot collisions
///   - GQA ratio = 2 (NAH=8, NKH=4)
///   - larger HD/ID/qkv_dim than tiny (4x bigger work per tile)
pub fn scheduled_prefill_medium_dims() -> reified_dag::LlamaDims {
    reified_dag::LlamaDims {
        num_layers: 4,
        hidden_dim: 512,
        intermediate_dim: 1024,
        num_attn_heads: 8,
        num_kv_heads: 4,
        head_dim: 64,
        seq_len: 64,
    }
}

/// CTA pool size for the medium fixture. Bigger than tiny because the work
/// is bigger.
pub const SCHEDULED_PREFILL_MEDIUM_CTAS: u32 = 16;

pub fn generate_scheduled_prefill_medium() -> String {
    use crate::kernel_library::coalesce_with_flashinfer_attention as coalesce;
    use crate::reified_dag::{ReifiedDag, TileSizes};
    use crate::schedule::{CostModel, partition_into_waves};
    use crate::scheduled_codegen::emit_scheduled_megakernel_cu;
    use crate::target_profile::TargetProfile;

    let profile = TargetProfile::l4_sm89();
    let dims = scheduled_prefill_medium_dims();
    let reified = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
    let dag = coalesce(&reified);
    let cost = CostModel::from_dag(&dag);
    let sched = partition_into_waves(&dag, profile.cooperative_grid_size(), &cost, 100);
    emit_scheduled_megakernel_cu(
        &dag,
        &sched,
        SCHEDULED_PREFILL_KV_PAGE_SIZE,
        &profile,
        "medium",
    )
}

// ── Phase 4 step 4: DSL-driven multi-variant codegen ─────────────────────
//
// The DSL is the single source of truth for the supported model variants.
// `LLAMA_DSL` ships with the macros-core crate; `generate_scheduled_prefill
// _variants` parses it, walks the variants block, and produces one .cu
// source per variant.

/// LLaMA architecture + supported model variants. Single source of truth
/// for the scheduled megakernel codegen registry.
pub const LLAMA_DSL: &str = include_str!("../models/llama.dsl");

/// One emitted scheduled megakernel: variant name + CUDA source.
pub struct ScheduledVariant {
    pub name: String,
    pub cu_source: String,
}

/// Pick a CTA pool size for a given variant.
///
/// Always equals `profile.cooperative_grid_size()` — the FlashInfer
/// attention runner inside the megakernel reads
/// `work_indptr[blockIdx.y]`, so the launch grid's `y` dim **must**
/// match the cluster count the FlashInfer planner generates work
/// for. The planner is also given this same number via the shim
/// (no `cudaDeviceGetAttribute` lying happens), so the two stay
/// in sync.
///
/// Smaller variants (tiny, seq=64) end up with many idle CTAs in
/// their non-attention waves — the LPT bin packer distributes only
/// as many row tiles as exist — but the attention wave needs all
/// `cooperative_grid_size()` CTAs, and correctness trumps
/// load-balance efficiency on tiny fixtures.
fn ctas_for_variant(
    profile: &target_profile::TargetProfile,
    _dims: &reified_dag::LlamaDims,
) -> u32 {
    profile.cooperative_grid_size()
}

/// Walk the `variants` block of the LLaMA DSL and emit one scheduled
/// megakernel per variant. Returns `Vec<(name, cu_source)>` ordered by
/// variant declaration order.
///
/// Resolution rule for each variant: start with the kernel header's default
/// params, then apply the variant's overrides. Missing dim parameters are
/// an error.
pub fn generate_scheduled_prefill_variants(dsl: &str) -> Result<Vec<ScheduledVariant>, String> {
    use crate::kernel_library::coalesce_with_flashinfer_attention as coalesce;
    use crate::reified_dag::{LlamaDims, ReifiedDag, TileSizes};
    use crate::schedule::{CostModel, partition_into_waves};
    use crate::scheduled_codegen::emit_scheduled_megakernel_cu;
    use crate::target_profile::TargetProfile;
    use std::collections::HashMap;

    // Hardware target. L4 (sm_89) is the only target wired in
    // today; an `sm_90` build flag would pick a different profile.
    let profile = TargetProfile::l4_sm89();

    let tokens: proc_macro2::TokenStream = dsl.parse().map_err(|e| format!("DSL tokenize: {e}"))?;
    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("DSL parse: {e}"))?;

    if def.variants.is_empty() {
        return Err("DSL has no `variants { ... }` block — nothing to emit".into());
    }

    // Default param table from the kernel header.
    let defaults: HashMap<String, usize> = def
        .params
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();

    let mut out = Vec::new();
    for variant in &def.variants {
        // Resolve params: defaults overlaid with the variant's overrides.
        let mut params = defaults.clone();
        for (k, v) in &variant.params {
            params.insert(k.to_string(), *v);
        }

        // Pull seq_len out — it's a required parametric symbol for the
        // scheduled megakernel since the wave schedule bakes against it.
        let seq_len = *params.get("SEQ_LEN").ok_or_else(|| {
            format!(
                "variant `{}` missing SEQ_LEN (must be in defaults or variant params)",
                variant.name
            )
        })? as u32;

        let dims = LlamaDims::from_params(&params, seq_len)
            .map_err(|e| format!("variant `{}`: {e}", variant.name))?;

        let reified = ReifiedDag::reify_llama(dims, TileSizes::default_v1());
        let dag = coalesce(&reified);
        let cost = CostModel::from_dag(&dag);
        let num_ctas = ctas_for_variant(&profile, &dims);
        let sched = partition_into_waves(&dag, num_ctas, &cost, 100);
        let name = variant.name.to_string();
        let cu_source = emit_scheduled_megakernel_cu(
            &dag,
            &sched,
            SCHEDULED_PREFILL_KV_PAGE_SIZE,
            &profile,
            &name,
        );

        out.push(ScheduledVariant { name, cu_source });
    }

    Ok(out)
}

#[cfg(test)]
mod variant_tests {
    use super::*;

    #[test]
    fn llama_dsl_parses_and_emits_all_variants() {
        let variants =
            generate_scheduled_prefill_variants(LLAMA_DSL).expect("LLAMA_DSL must parse and emit");
        // We expect at least: tiny, medium, llama_3_2_1b_seq1024
        assert!(
            variants.len() >= 3,
            "expected ≥3 variants, got {}",
            variants.len()
        );
        let names: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
        assert!(names.contains(&"tiny"), "missing tiny in {names:?}");
        assert!(names.contains(&"medium"), "missing medium in {names:?}");
        assert!(
            names.contains(&"llama_3_2_1b_seq1024"),
            "missing llama_3_2_1b_seq1024 in {names:?}"
        );

        // Each emitted source should reference its own per-variant namespace
        // and at least the CTA-stream constant name.
        for v in &variants {
            assert!(
                v.cu_source
                    .contains(&format!("namespace pfl_sched_{}", v.name)),
                "variant {} missing per-variant namespace",
                v.name
            );
            assert!(
                v.cu_source.contains("WAVE_OPS"),
                "variant {} missing WAVE_OPS",
                v.name
            );
            assert!(
                v.cu_source
                    .contains(&format!("launch_scheduled_megakernel_{}", v.name)),
                "variant {} missing launch helper",
                v.name
            );
        }
    }

    #[test]
    fn llama_3_2_1b_variant_has_real_dims_baked() {
        let variants = generate_scheduled_prefill_variants(LLAMA_DSL).unwrap();
        let v = variants
            .iter()
            .find(|v| v.name == "llama_3_2_1b_seq1024")
            .expect("missing llama_3_2_1b_seq1024");
        assert!(v.cu_source.contains("MODEL_NUM_LAYERS    = 16"));
        assert!(v.cu_source.contains("MODEL_HIDDEN_DIM    = 2048"));
        assert!(v.cu_source.contains("MODEL_INTERMEDIATE  = 8192"));
        assert!(v.cu_source.contains("MODEL_NUM_ATTN_H    = 32"));
        assert!(v.cu_source.contains("MODEL_NUM_KV_H      = 8"));
        assert!(v.cu_source.contains("MODEL_HEAD_DIM      = 64"));
        assert!(v.cu_source.contains("MODEL_SEQ_LEN       = 1024"));
    }
}

/// Generate a debug variant of the decode kernel that syncs and writes a
/// marker to a debug buffer after each op. Useful for identifying which op
/// crashes in the full megakernel.
pub fn generate_debug_kernel(dsl: &str) -> Result<String, String> {
    let tokens: proc_macro2::TokenStream = dsl
        .parse()
        .map_err(|e| format!("failed to tokenize DSL: {e}"))?;

    let def: parse::MegakernelDef = syn::parse2(tokens).map_err(|e| format!("parse error: {e}"))?;
    let dag = parse::build_dag(&def)?;

    Ok(cuda_codegen::generate_debug_kernel(&dag))
}
