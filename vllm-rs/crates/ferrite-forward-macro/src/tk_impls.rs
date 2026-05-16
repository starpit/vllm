// SPDX-License-Identifier: Apache-2.0
//! TK peer [`Implementation`]s for megakernel-eligible ops.
//!
//! Each `Tk*Impl` in this module is the Instruction-level peer of a
//! backend Impl ([`EmbedRefImpl`], [`RmsNormRefImpl`],
//! [`GemmRefImpl`], [`FusedAddRmsNormImpl`],
//! [`FusedQkvRopeCacheImpl`], [`AttentionViaCacheImpl`],
//! [`FusedGateUpSiluMulImpl`]). The pair exists because of the
//! two-layer solve (see `src/tape_claim.rs`):
//!
//! - **Instruction-level** solve picks a per-tile-pattern claim via
//!   the cost DP. Today Cutlass/cuBLAS peers dominate on sm≥90
//!   because their calibrated CSV rows beat the analytic-roofline
//!   backend peers. A `Tk*` peer won't be picked unless its cost is
//!   lower than the current winner.
//! - **Tape-level** claim in [`crate::tape::tk_mega::TkMegaTapeClaimer`]
//!   gates on `tape_is_all_tk`. So mega only claims the whole tape
//!   when every op on it is a `Tk*` variant — which requires the
//!   Instruction-level DP to have picked `Tk*Impl` at every seed.
//!
//! Each peer here:
//!
//! 1. Gates `target_compatible` on `compute_capability >= 90` (TK
//!    primitives — `wgmma`, `tma::*_async`, `setmaxnreg` — require
//!    sm_90a).
//! 2. Delegates `matches` to its backend peer (identical claim
//!    shape). The peer's `matches` already rejects non-dense weight
//!    storage where appropriate; we inherit that.
//! 3. Reports a constant-epsilon `cost_us` (`TK_COST_US`) so every
//!    `Tk*Impl` beats its backend peer on the DP whenever the target
//!    is sm≥90. Proper calibration — a `tk_<op>` CSV sweep — is a
//!    follow-up; the constant is good enough to land the all-`Tk*`
//!    tape the tape-level claimer needs.
//! 4. Emits an `OpInstance` whose `name` has the `Tk` prefix but
//!    whose field layout is byte-identical to the backend peer.
//!    [`crate::tape::tk_mega::op_emit`] keys off the prefixed name
//!    for `.cu` emission; the host interpreter path (if ever
//!    exercised for a `Tk*` tape — shouldn't happen, but defensive)
//!    maps `Tk*` back to the plain variant in the static-slice
//!    emitter via [`crate::interpreter_codegen::strip_tk_prefix`].
//!
//! No new [`crate::ferrite_forward::Instruction`] variants are
//! added — the prefix strip keeps the runtime enum unchanged.

#![allow(dead_code)]

use std::collections::BTreeMap;

use crate::classified::Program;
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    AttentionViaCacheImpl, CostCtx, CutlassFusedAddRmsNormGemmImpl,
    CutlassFusedAddScalarOffsetRmsNormGemmImpl, CutlassGemmAddImpl, EmbedRefImpl,
    FusedAddRmsNormImpl, FusedAddRmsNormWithOffsetImpl, FusedGateUpGeluMulImpl,
    FusedGateUpSiluMulImpl, FusedQkvRopeCacheImpl, GemmRefImpl, Handoff, Implementation,
    LaunchKind, Layout, MatchContext, MatchInfo, OpcodeShape, Resources, RmsNormRefImpl,
    ScalarMulImpl, ScalarOffsetRmsNormImpl, SlidingAttentionViaCacheImpl, SlotMap, TanhSoftCapImpl,
    WeightAccessor, WorkloadConstraint,
};
use crate::target::TargetProfile;
use ferrite_forward::Instruction;

/// sm version gate shared by every `Tk*Impl::target_compatible`.
/// TK 2.0 uses Hopper-class primitives (`wgmma`, `tma::*_async`,
/// `setmaxnreg`) that require sm_90a — L4 / A100 / sm<90 miss the
/// required PTX.
const TK_SM_MIN: u32 = 90;

/// Cost sentinel for every `Tk*Impl`. Must be below ANY cost the
/// analytic fallback formulas can produce at M=1. The roofline
/// fallback at M=1, D=1 gives ~2e-9 μs (1 GHz clock / 4 flops/cycle).
/// Use 1e-12 (1 picosecond) to stay comfortably below all analytic
/// and calibrated costs while remaining positive (zero is ineligible
/// per the solver's `cost <= 0.0 → None` guard). Swap for a
/// calibrated `tk_<op>` CSV row once the per-op bodies are benched.
///
/// Why 1e-12 and not 1e-3: the old 1e-3 value was occasionally beaten
/// by the analytic cost_attention formula at M=1 (which uses T^2 with
/// T=1, giving ~1e-7 μs). 1e-12 is below any physically achievable
/// compute time and ensures TK always wins on sm≥90 targets.
const TK_COST_US: f64 = 1.0e-12;

/// Map every base `Instruction::X` to its TK peer `Instruction::TkX`.
/// Field order/values preserved verbatim because each Tk peer in
/// `ferrite_forward::Instruction` has the same field tuple as its base
/// (and the few that gain extra fields — `TkSlidingAttentionViaCache`,
/// `TkTanhSoftCap` — append the extra value on the call site, not here).
fn rename_instances_with_tk_prefix(instances: Vec<Instruction>) -> Vec<Instruction> {
    instances
        .into_iter()
        .map(|inst| match inst {
            Instruction::Embed(a) => Instruction::TkEmbed(a),
            Instruction::ScalarMul(a, b, c) => Instruction::TkScalarMul(a, b, c),
            Instruction::RmsNorm(a, b, c) => Instruction::TkRmsNorm(a, b, c),
            Instruction::Gemm(a, b, c, d, e) => Instruction::TkGemm(a, b, c, d, e),
            Instruction::FusedAddRmsNorm(a, b, c) => Instruction::TkFusedAddRmsNorm(a, b, c),
            Instruction::FusedQkvRopeCache(a, b, c, d, e) => {
                Instruction::TkFusedQkvRopeCache(a, b, c, d, e)
            }
            Instruction::AttentionViaCache(a, b, c, d) => {
                Instruction::TkAttentionViaCache(a, b, c, d)
            }
            Instruction::FusedGateUpSiluMul(a, b, c) => Instruction::TkFusedGateUpSiluMul(a, b, c),
            Instruction::FusedGateUpGeluMul(a, b, c) => Instruction::TkFusedGateUpGeluMul(a, b, c),
            Instruction::ScalarOffsetRmsNorm(a, b, c, d) => {
                Instruction::TkScalarOffsetRmsNorm(a, b, c, d)
            }
            Instruction::FusedAddRmsNormWithOffset(a, b, c, d) => {
                Instruction::TkFusedAddRmsNormWithOffset(a, b, c, d)
            }
            Instruction::SpliceMmEmbeds(a) => Instruction::TkSpliceMmEmbeds(a),
            // Variants without a typed Tk peer (e.g. `SlidingAttentionViaCache`
            // which needs an extra `window_size_left` field, `TanhSoftCap`
            // which needs an extra `n_vocab`) are renamed in their own
            // call site by mapping directly to the typed Tk variant. Any
            // other unhandled variant flows through unchanged. Worst case
            // a Tk delegator emits a non-Tk Instruction — the host
            // interpreter still runs it correctly; only the mega tape-
            // level claim would reject it.
            other => other,
        })
        .collect()
}

/// Re-shape a backend peer's [`OpcodeShape`] with `Tk`-prefixed
/// variant ident. Field list preserved verbatim so the per-canonical
/// shape registry still resolves `TkFoo` field count + types to the
/// same layout the op_emit dispatch expects.
fn retag_opcode_shape(base: OpcodeShape) -> OpcodeShape {
    let prefixed_name = format!("Tk{}", base.name);
    OpcodeShape {
        name: syn::Ident::new(&prefixed_name, proc_macro2::Span::call_site()),
        fields: base.fields,
    }
}

// ── TkEmbedImpl — peer of [`EmbedRefImpl`] ───────────────────────

#[derive(Debug, Default)]
pub struct TkEmbedImpl;

impl Implementation for TkEmbedImpl {
    fn name(&self) -> &'static str {
        "tk_embed"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        EmbedRefImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        EmbedRefImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        EmbedRefImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        EmbedRefImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        EmbedRefImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        EmbedRefImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        EmbedRefImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        EmbedRefImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        EmbedRefImpl.required_weights(claimed, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(EmbedRefImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        EmbedRefImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }
}

// ── TkRmsNormImpl — peer of [`RmsNormRefImpl`] ───────────────────

#[derive(Debug, Default)]
pub struct TkRmsNormImpl;

impl Implementation for TkRmsNormImpl {
    fn name(&self) -> &'static str {
        "tk_rmsnorm"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        RmsNormRefImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        RmsNormRefImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        RmsNormRefImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        RmsNormRefImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        RmsNormRefImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        RmsNormRefImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        RmsNormRefImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        RmsNormRefImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        RmsNormRefImpl.required_weights(claimed, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(RmsNormRefImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        RmsNormRefImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }
}

// ── TkScalarMulImpl — peer of [`ScalarMulImpl`] ──────────────────
//
// In-place scale: `x[i] *= scale`. Gemma2/3 applies `sqrt(hidden_size)`
// to the embedding output. This impl renames `ScalarMul(in,out,scale)`
// → `TkScalarMul(in,out,scale)` so the tape stays all-TK and the
// megakernel can claim it. The walker emits an inline loop in the
// storer role that reads/writes the gmem slot directly (no shmem).
// Unity passthrough (scale=1.0) produces an empty op list → no code.

#[derive(Debug, Default)]
pub struct TkScalarMulImpl;

impl Implementation for TkScalarMulImpl {
    fn name(&self) -> &'static str {
        "tk_scalar_mul"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        ScalarMulImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        TK_COST_US.min(ScalarMulImpl.cost_us(m, ctx))
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        ScalarMulImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        ScalarMulImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        ScalarMulImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        ScalarMulImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        ScalarMulImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        ScalarMulImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        ScalarMulImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        ScalarMulImpl.required_weights(claimed, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(ScalarMulImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        ScalarMulImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }
}

// ── TkGemmImpl — peer of [`GemmRefImpl`] ─────────────────────────
//
// Single Gemm opcode at the `Tk*` level — `emit_gemm_gemv` in
// op_emit.rs forks on `num_tokens` internally (`gemv_bf16` for
// M=1, `gemm_bf16` for M>1), so one peer suffices for both decode
// and prefill.

#[derive(Debug, Default)]
pub struct TkGemmImpl;

impl Implementation for TkGemmImpl {
    fn name(&self) -> &'static str {
        "tk_gemm"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        GemmRefImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        GemmRefImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        GemmRefImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        GemmRefImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        GemmRefImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        GemmRefImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        GemmRefImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        GemmRefImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        GemmRefImpl.required_weights(claimed, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(GemmRefImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        GemmRefImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        // `gemv_bf16.cuh` asserts `HIDDEN_DIM % NCW == 0` and
        // `K_PER_WARP % 16 == 0`. NCW = `(head_dim/32).clamp(1, 4)`
        // (matches `FerriteConfig::phase3d`'s `ncw_ceiling`).
        // `pick_chunk_cols` finds the largest ≤512 divisor of K_PER_WARP,
        // so the only hard constraint is K_PER_WARP % 16 == 0.
        let Some(&hidden_dim) = ctx.model.bounds.get("hidden_size") else {
            return false;
        };
        let Some(&intermediate_dim) = ctx.model.bounds.get("intermediate_size") else {
            return false;
        };
        let Some(&head_dim) = ctx.model.bounds.get("head_dim") else {
            return false;
        };
        let ncw = (head_dim / 32).clamp(1, 4);
        let ok = |k: u64| k.is_multiple_of(ncw) && (k / ncw).is_multiple_of(16);
        ok(hidden_dim) && ok(intermediate_dim)
    }
}

// ── TkFusedAddRmsNormImpl — peer of [`FusedAddRmsNormImpl`] ──────

#[derive(Debug, Default)]
pub struct TkFusedAddRmsNormImpl;

impl Implementation for TkFusedAddRmsNormImpl {
    fn name(&self) -> &'static str {
        "tk_fused_add_rms_norm"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        FusedAddRmsNormImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        FusedAddRmsNormImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        FusedAddRmsNormImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        FusedAddRmsNormImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        FusedAddRmsNormImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedAddRmsNormImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedAddRmsNormImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        FusedAddRmsNormImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        FusedAddRmsNormImpl.required_weights(claimed, fuf, program)
    }

    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        FusedAddRmsNormImpl.output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(FusedAddRmsNormImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        FusedAddRmsNormImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }
}

// ── TkFusedQkvRopeCacheImpl — peer of [`FusedQkvRopeCacheImpl`] ──

#[derive(Debug, Default)]
pub struct TkFusedQkvRopeCacheImpl;

impl Implementation for TkFusedQkvRopeCacheImpl {
    fn name(&self) -> &'static str {
        "tk_fused_qkv_rope_cache"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        FusedQkvRopeCacheImpl.workload_constraint()
    }

    fn workload_constraint_for_role(&self, role: crate::solver::ForwardRole) -> WorkloadConstraint {
        match role {
            // Wave G unlocked NUM_TOKENS > 1 for both decode AND prefill.
            // Allow M>1 in either role.
            crate::solver::ForwardRole::Decode | crate::solver::ForwardRole::Prefill => {
                WorkloadConstraint::NumTokensRange {
                    min: 1,
                    max: u32::MAX,
                }
            }
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        FusedQkvRopeCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        FusedQkvRopeCacheImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        FusedQkvRopeCacheImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        FusedQkvRopeCacheImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        FusedQkvRopeCacheImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedQkvRopeCacheImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedQkvRopeCacheImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        FusedQkvRopeCacheImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        FusedQkvRopeCacheImpl.required_weights(claimed, fuf, program)
    }

    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        FusedQkvRopeCacheImpl.output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(FusedQkvRopeCacheImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        FusedQkvRopeCacheImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }
}

// ── TkAttentionViaCacheImpl — peer of [`AttentionViaCacheImpl`] ──

#[derive(Debug, Default)]
pub struct TkAttentionViaCacheImpl;

impl Implementation for TkAttentionViaCacheImpl {
    fn name(&self) -> &'static str {
        "tk_attention_via_cache"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        AttentionViaCacheImpl.workload_constraint()
    }

    fn workload_constraint_for_role(&self, role: crate::solver::ForwardRole) -> WorkloadConstraint {
        AttentionViaCacheImpl.workload_constraint_for_role(role)
    }

    fn accepts_role(&self, role: crate::solver::ForwardRole) -> bool {
        AttentionViaCacheImpl.accepts_role(role)
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        // attention_partial.cuh uses `ferrite::tk::store_n_rows<N>`,
        // generalised across GQA_RATIO ∈ [1, 16]. Gate on that range
        // + divisibility; anything outside falls through to
        // AttentionViaCacheImpl (FA2).
        let Some(&q) = ctx.model.bounds.get("num_attention_heads") else {
            return false;
        };
        let Some(&kv) = ctx.model.bounds.get("num_key_value_heads") else {
            return false;
        };
        if kv == 0 || q % kv != 0 {
            return false;
        }
        let gqa = q / kv;
        (1..=16).contains(&gqa)
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        AttentionViaCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        AttentionViaCacheImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        AttentionViaCacheImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        AttentionViaCacheImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        AttentionViaCacheImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        AttentionViaCacheImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        AttentionViaCacheImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        AttentionViaCacheImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        AttentionViaCacheImpl.required_weights(claimed, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(AttentionViaCacheImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        AttentionViaCacheImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }
}

// ── TkSlidingAttentionViaCacheImpl ────────────────────────────────
//
// TK peer of [`SlidingAttentionViaCacheImpl`] for sliding-window
// decode attention. Delegates the FA2/cost path to the base impl
// and adds `window_size_left` as a 6th field so the mega emitter
// can bake the window size as a literal template argument in each
// `attention_partial` call (allowing global + local attention layers
// to coexist in the same kernel variant with different SLIDING_WINDOW
// values). The host interpreter ignores this extra field and delegates
// to `SlidingAttentionViaCache.eval()` which reads the window size
// from the model config at runtime.

#[derive(Debug, Default)]
pub struct TkSlidingAttentionViaCacheImpl;

impl Implementation for TkSlidingAttentionViaCacheImpl {
    fn name(&self) -> &'static str {
        "tk_sliding_attention_via_cache"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        SlidingAttentionViaCacheImpl.workload_constraint()
    }

    fn workload_constraint_for_role(&self, role: crate::solver::ForwardRole) -> WorkloadConstraint {
        SlidingAttentionViaCacheImpl.workload_constraint_for_role(role)
    }

    fn accepts_role(&self, role: crate::solver::ForwardRole) -> bool {
        SlidingAttentionViaCacheImpl.accepts_role(role)
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        // Require GQA ratio ∈ [1, 16] (same guard as TkAttentionViaCacheImpl)
        // and the model must have a sliding_window in its bounds.
        let Some(&q) = ctx.model.bounds.get("num_attention_heads") else {
            return false;
        };
        let Some(&kv) = ctx.model.bounds.get("num_key_value_heads") else {
            return false;
        };
        if kv == 0 || q % kv != 0 {
            return false;
        }
        let gqa = q / kv;
        if !(1..=16).contains(&gqa) {
            return false;
        }
        // Only claim if the model actually has a finite sliding window.
        ctx.model.bounds.contains_key("sliding_window")
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        SlidingAttentionViaCacheImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        SlidingAttentionViaCacheImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        SlidingAttentionViaCacheImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        SlidingAttentionViaCacheImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        SlidingAttentionViaCacheImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        SlidingAttentionViaCacheImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        SlidingAttentionViaCacheImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        SlidingAttentionViaCacheImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        SlidingAttentionViaCacheImpl.required_weights(claimed, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        // Post-migration shape: in_slot, out_slot, layer, interleaved,
        // window_size_left. The CosSin slot lives on the per-arch
        // `WeightAccessors` impl. The window_size_left field carries
        // the compile-time window size for the mega codegen (baked as
        // a literal template argument); the host interpreter ignores
        // it.
        OpcodeShape::new(
            "TkSlidingAttentionViaCache",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("interleaved", syn::parse_quote!(bool)),
                ("window_size_left", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        let window_size = bounds.get("sliding_window").copied().unwrap_or(0) as u32;
        // Delegate to the base impl to compute in_slot/out_slot/layer/
        // interleaved, then re-pack into the TK variant with the extra
        // compile-time window_size_left field.
        SlidingAttentionViaCacheImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(|ops| {
                ops.into_iter()
                    .map(|inst| match inst {
                        Instruction::SlidingAttentionViaCache(a, b, c, d) => {
                            Instruction::TkSlidingAttentionViaCache(a, b, c, d, window_size)
                        }
                        other => other,
                    })
                    .collect()
            })
    }
}

// ── TkFusedGateUpSiluMulImpl — peer of [`FusedGateUpSiluMulImpl`] ─

#[derive(Debug, Default)]
pub struct TkFusedGateUpSiluMulImpl;

impl Implementation for TkFusedGateUpSiluMulImpl {
    fn name(&self) -> &'static str {
        "tk_fused_gate_up_silu_mul"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn workload_constraint_for_role(&self, role: crate::solver::ForwardRole) -> WorkloadConstraint {
        match role {
            // silu_upgate.cuh now supports NUM_TOKENS > 1 via per-token loop.
            crate::solver::ForwardRole::Decode => WorkloadConstraint::NumTokensRange {
                min: 1,
                max: u32::MAX,
            },
            crate::solver::ForwardRole::Prefill => self.workload_constraint(),
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        FusedGateUpSiluMulImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        FusedGateUpSiluMulImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        FusedGateUpSiluMulImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        FusedGateUpSiluMulImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        FusedGateUpSiluMulImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedGateUpSiluMulImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedGateUpSiluMulImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        FusedGateUpSiluMulImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        FusedGateUpSiluMulImpl.required_weights(claimed, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(FusedGateUpSiluMulImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        FusedGateUpSiluMulImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }
}

// ── TkFusedGateUpGeluMulImpl — peer of [`FusedGateUpGeluMulImpl`] ─
//
// Gemma2/3 GELU variant of TkFusedGateUpSiluMulImpl. Claims the
// `(Gemm[gate], Gelu, Gemm[up], Mul)` compute subgraph and maps it
// onto `gelu_upgate.cuh` where the consumer applies the tanh GELU
// approximation instead of SiLU.

#[derive(Debug, Default)]
pub struct TkFusedGateUpGeluMulImpl;

impl Implementation for TkFusedGateUpGeluMulImpl {
    fn name(&self) -> &'static str {
        "tk_fused_gate_up_gelu_mul"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn workload_constraint_for_role(&self, role: crate::solver::ForwardRole) -> WorkloadConstraint {
        match role {
            crate::solver::ForwardRole::Decode => WorkloadConstraint::NumTokensRange {
                min: 1,
                max: u32::MAX,
            },
            crate::solver::ForwardRole::Prefill => self.workload_constraint(),
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        FusedGateUpGeluMulImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        FusedGateUpGeluMulImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        FusedGateUpGeluMulImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        FusedGateUpGeluMulImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        FusedGateUpGeluMulImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedGateUpGeluMulImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedGateUpGeluMulImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        FusedGateUpGeluMulImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        FusedGateUpGeluMulImpl.required_weights(claimed, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(FusedGateUpGeluMulImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        FusedGateUpGeluMulImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }
}

// ── TkGemmAddImpl — peer of [`CutlassGemmAddImpl`] ───────────────
// NOTE: set cost_us = 2*TK_COST_US so the solver prefers TkGemmImpl (gemv_bf16)
// when both are applicable. TkGemmAdd (down_proj_residual.cuh) processes one
// output row at a time; TkGemm (gemv_bf16) processes 16-row blocks with pipelining.
// TkGemmAdd is ONLY useful for standalone down_proj patterns that don't have a
// co-located FusedAddRmsNorm to handle the residual.
//
// Claims a `(Gemm, Add)` pair where the Add is a residual-stream
// update: `residual += W @ x`. Maps to `down_proj_residual.cuh`'s
// 4-warp-role body in the megakernel tape.
//
// Difference from `CutlassGemmAddImpl`:
// - Handles M=1 (decode path) — `CutlassGemmAddImpl` gates at M≥2.
// - Emits opcode `TkGemmAdd` without CUTLASS tile/stage fields.
// - `target_compatible` gates on sm≥90 (TK primitives).
//
// `output_alias` and most structural methods delegate to the
// singleton `CutlassGemmAddImpl(16, 64, 3)` — tile dimensions are
// irrelevant for alias/match logic.

#[derive(Debug, Default)]
pub struct TkGemmAddImpl;

/// Singleton `CutlassGemmAddImpl` used for delegation. Tile dims
/// are ignored by all methods except `cost_us` and `name`.
fn cutlass_gemm_add_proxy() -> CutlassGemmAddImpl {
    CutlassGemmAddImpl {
        tile_m: 16,
        tile_n: 64,
        stages: 3,
    }
}

impl Implementation for TkGemmAddImpl {
    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US // restored: TkGemmAdd preferred over TkGemmImpl for (Gemm, Add) pairs
    }

    fn name(&self) -> &'static str {
        "tk_gemm_add"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn workload_constraint_for_role(&self, role: crate::solver::ForwardRole) -> WorkloadConstraint {
        match role {
            // down_proj_residual.cuh now supports NUM_TOKENS > 1 via per-token loop.
            crate::solver::ForwardRole::Decode => WorkloadConstraint::NumTokensRange {
                min: 1,
                max: u32::MAX,
            },
            crate::solver::ForwardRole::Prefill => self.workload_constraint(),
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        cutlass_gemm_add_proxy().matches(fuf, seed, profile)
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        cutlass_gemm_add_proxy().resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        // Mega tape — same as all other Tk* peers.
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        cutlass_gemm_add_proxy().input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        cutlass_gemm_add_proxy().output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        cutlass_gemm_add_proxy().is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &crate::classified::Program,
    ) -> Vec<WeightAccessor> {
        cutlass_gemm_add_proxy().required_weights(claimed, fuf, program)
    }

    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        cutlass_gemm_add_proxy().output_alias(claimed_tiles, fuf)
    }

    /// Custom opcode shape: drops CUTLASS-specific tile_m/tile_n/stages
    /// fields since `down_proj_residual.cuh` doesn't use them.
    /// Field 7 (k_offset) + field 8 (k_full) enable K-chunking:
    /// each op processes W[:,k_offset:k_offset+k] where W has row stride k_full.
    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "TkGemmAdd",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
                ("k_offset", syn::parse_quote!(u32)),
                ("k_full", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &crate::classified::Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        use crate::classified::OpKind;
        use crate::codegen::split_base_layer;
        use crate::fuf::FufInput;
        use crate::impl_lib::gemm_nk_from_fuf;

        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("TkGemmAdd: claim contains a Gemm");
        let add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("TkGemmAdd: claim contains an Add");
        let gemm_node = fuf.get(gemm_id);
        let (in_id, in_slot) = match gemm_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!("TkGemmAdd: gemm input 0 must be a Tile (got {other:?})"),
        };
        let in_slot_idx = slots.of(in_id, in_slot);
        let (residual_id, residual_in) = fuf
            .get(add_id)
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Tile { id, slot } if *id != gemm_id => Some((*id, *slot)),
                _ => None,
            })
            .expect("TkGemmAdd: Add has a non-gemm Tile input (residual)");
        let residual_idx = slots.of(residual_id, residual_in);
        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let acc = accessors
            .first()
            .expect("TkGemmAdd: required_weights returned empty");
        let (_base, layer) = split_base_layer(&acc.name.to_string());
        let layer = layer.unwrap_or(0) as u32;
        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("TkGemmAdd: gemm (N, K) must resolve from FUF + bounds");
        // TK "four-chunk" optimization: split large-K down_proj into multiple ops.
        // Each chunk K=chunk_k gives k-chunks=1 per warp (k_per_warp=chunk_k/NCW=256,
        // kChunkCols=256 → 1 chunk), enabling immediate storer overlap per output block.
        // chunk_k = hidden_dim (= N for down_proj: K=intermediate=4×hidden → 4 chunks).
        let chunk_k: u32 = bounds
            .get("hidden_dim")
            .copied()
            .map(|hd| hd as u32)
            .unwrap_or(2048)
            .min(k);
        let num_chunks = if chunk_k > 0 && k > chunk_k && k.is_multiple_of(chunk_k) {
            k / chunk_k
        } else {
            1
        };
        let k_per_chunk = k / num_chunks;
        // k_full = full K dimension of the weight matrix (row stride).
        // For single-chunk ops k_full == k_per_chunk; for 4-chunk split k_full == k.
        let k_full = k;
        Some(
            (0..num_chunks)
                .map(|i| {
                    let k_offset = i * k_per_chunk;
                    Instruction::TkGemmAdd(
                        in_slot_idx,
                        residual_idx,
                        layer,
                        n,
                        k_per_chunk,
                        k_offset,
                        k_full,
                    )
                })
                .collect(),
        )
    }

    /// `applies_to` mirrors `TkGemmImpl`: the K reduction dimension
    /// of every GEMM claimed by this impl must be divisible by
    /// `NCW * kChunkCols` (= NCW * 512). Uses the same divisibility
    /// gate as `TkGemmImpl` since `down_proj_residual.cuh`'s K-chunk
    /// loop shares the same `CHUNK_COLS=512` width.
    fn applies_to(&self, ctx: &MatchContext) -> bool {
        TkGemmImpl.applies_to(ctx)
    }
}

// ── TkFusedAddRmsNormGemmImpl — peer of [`CutlassFusedAddRmsNormGemmImpl`] ──
//
// Claims a `(Add, RmsNorm, Gemm)` 3-tile at the tail of the forward
// pass (last decoder Add → final RmsNorm → lm_head Gemm). Maps to
// `lm_head_fused_residual` in `lm_head.cuh`, which does Add + RmsNorm +
// GEMV in a single 4-warp-role TK body and writes two outputs:
//   - residual_slot: delta + residual (the updated skip connection)
//   - out_slot:      logit scalars [vocab_size]
//
// Differences from `CutlassFusedAddRmsNormGemmImpl`:
// - Restricted to M=1 (decode only; lm_head.cuh has NUM_TOKENS==1 guard).
// - No CUTLASS tile_m/tile_n/stages fields in the opcode shape.
// - target_compatible gates on sm≥90 (TK primitives required).

#[derive(Debug, Default)]
pub struct TkFusedAddRmsNormGemmImpl;

/// Singleton proxy for delegation. Tile dims are ignored by match/alias logic.
fn cutlass_fused_add_rms_norm_gemm_proxy() -> CutlassFusedAddRmsNormGemmImpl {
    CutlassFusedAddRmsNormGemmImpl {
        tile_m: 16,
        tile_n: 64,
        stages: 3,
    }
}

impl Implementation for TkFusedAddRmsNormGemmImpl {
    fn name(&self) -> &'static str {
        "tk_fused_add_rms_norm_gemm"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    /// lm_head is always M=1 (single output position). The fused kernel's
    /// `static_assert(NUM_TOKENS == 1, ...)` gates this at compile time.
    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn workload_constraint_for_role(
        &self,
        _role: crate::solver::ForwardRole,
    ) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        // Reject Gemma2 lm_head case: if the RmsNorm's weight comes from a
        // scalar-offset Add (Weight+Scalar), defer to TkFusedAddScalarOffsetRmsNormGemmImpl.
        // That impl claims all 4 nodes; this impl would only claim 3 and mis-emit (no offset).
        use crate::classified::OpKind;
        use crate::fuf::FufInput;
        let residual_add = fuf.get(seed);
        if residual_add.op == OpKind::Add
            && residual_add
                .inputs
                .iter()
                .all(|i| matches!(i, FufInput::Tile { .. }))
        {
            for rms in &fuf.nodes {
                if rms.op != OpKind::RmsNorm || rms.inputs.len() < 2 {
                    continue;
                }
                if !matches!(rms.inputs[0], FufInput::Tile { id, .. } if id == seed) {
                    continue;
                }
                if let FufInput::Tile { id, .. } = rms.inputs[1] {
                    let wt = fuf.get(id);
                    if wt.op == OpKind::Add
                        && wt.inputs.iter().any(|i| matches!(i, FufInput::Scalar(_)))
                    {
                        return None; // Gemma2 offset case — let 4-node impl handle it
                    }
                }
            }
        }
        cutlass_fused_add_rms_norm_gemm_proxy().matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        cutlass_fused_add_rms_norm_gemm_proxy().resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        cutlass_fused_add_rms_norm_gemm_proxy().input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        cutlass_fused_add_rms_norm_gemm_proxy().output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        cutlass_fused_add_rms_norm_gemm_proxy().is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &crate::classified::Program,
    ) -> Vec<WeightAccessor> {
        cutlass_fused_add_rms_norm_gemm_proxy().required_weights(claimed, fuf, program)
    }

    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        cutlass_fused_add_rms_norm_gemm_proxy().output_alias(claimed_tiles, fuf)
    }

    /// Custom opcode shape: drops CUTLASS tile_m/tile_n/stages fields.
    /// Weight resolution lives on the per-arch `WeightAccessors` impl
    /// (RmsNorm slot 0 + Linear slot 0 at this `op_idx`).
    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "TkFusedAddRmsNormGemm",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &crate::classified::Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        use crate::classified::OpKind;
        use crate::codegen::split_base_layer;
        use crate::fuf::FufInput;
        use crate::impl_lib::gemm_nk_from_fuf;

        let add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Add)
            .expect("TkFusedAddRmsNormGemm: claim contains an Add");
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("TkFusedAddRmsNormGemm: claim contains a Gemm");

        let add_node = fuf.get(add_id);
        let gemm_node = fuf.get(gemm_id);

        let (delta_id, delta_in_slot) = match add_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => {
                panic!("TkFusedAddRmsNormGemm: Add input 0 (delta) must be a Tile (got {other:?})")
            }
        };
        let (residual_id, residual_in_slot) = match add_node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "TkFusedAddRmsNormGemm: Add input 1 (residual) must be a Tile (got {other:?})"
            ),
        };

        let delta_idx = slots.of(delta_id, delta_in_slot);
        let residual_idx = slots.of(residual_id, residual_in_slot);
        let out_slot_idx = slots.of(gemm_id, 0);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let norm_acc = accessors
            .first()
            .expect("TkFusedAddRmsNormGemm: required_weights[0] (norm)");
        let _gemm_acc = accessors
            .get(1)
            .expect("TkFusedAddRmsNormGemm: required_weights[1] (gemm)");

        let (_norm_base, norm_layer) = split_base_layer(&norm_acc.name.to_string());
        let layer = norm_layer.unwrap_or(0) as u32;

        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("TkFusedAddRmsNormGemm: gemm (N, K) must resolve from FUF + bounds");

        Some(vec![Instruction::TkFusedAddRmsNormGemm(
            delta_idx,
            residual_idx,
            out_slot_idx,
            layer,
            n,
            k,
        )])
    }

    /// Same K-divisibility gate as `TkGemmImpl` — the lm_head_fused_residual
    /// consumer uses the same NCW/CHUNK_COLS=512 partitioning as gemv_bf16.
    fn applies_to(&self, ctx: &MatchContext) -> bool {
        TkGemmImpl.applies_to(ctx)
    }
}

// ── TkScalarOffsetRmsNormImpl — peer of [`ScalarOffsetRmsNormImpl`] ─
//
// TK kernel for `rms_norm_offset.cuh`: ScalarOffsetRmsNorm with
// (weight + offset) normalization. Delegates matches/fan_out to
// ScalarOffsetRmsNormImpl and renames the opcode.

#[derive(Debug, Default)]
pub struct TkScalarOffsetRmsNormImpl;

impl Implementation for TkScalarOffsetRmsNormImpl {
    fn name(&self) -> &'static str {
        "tk_scalar_offset_rms_norm"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        ScalarOffsetRmsNormImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        ScalarOffsetRmsNormImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        ScalarOffsetRmsNormImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        ScalarOffsetRmsNormImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        ScalarOffsetRmsNormImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        ScalarOffsetRmsNormImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        ScalarOffsetRmsNormImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        ScalarOffsetRmsNormImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &crate::classified::Program,
    ) -> Vec<WeightAccessor> {
        ScalarOffsetRmsNormImpl.required_weights(claimed, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(ScalarOffsetRmsNormImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &crate::classified::Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        ScalarOffsetRmsNormImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }
}

// ── TkFusedAddRmsNormWithOffsetImpl — peer of [`FusedAddRmsNormWithOffsetImpl`] ─

#[derive(Debug, Default)]
pub struct TkFusedAddRmsNormWithOffsetImpl;

impl Implementation for TkFusedAddRmsNormWithOffsetImpl {
    fn name(&self) -> &'static str {
        "tk_fused_add_rms_norm_with_offset"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        FusedAddRmsNormWithOffsetImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        FusedAddRmsNormWithOffsetImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        FusedAddRmsNormWithOffsetImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        FusedAddRmsNormWithOffsetImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        FusedAddRmsNormWithOffsetImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedAddRmsNormWithOffsetImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        FusedAddRmsNormWithOffsetImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        FusedAddRmsNormWithOffsetImpl.is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &crate::classified::Program,
    ) -> Vec<WeightAccessor> {
        FusedAddRmsNormWithOffsetImpl.required_weights(claimed, fuf, program)
    }

    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        FusedAddRmsNormWithOffsetImpl.output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        retag_opcode_shape(FusedAddRmsNormWithOffsetImpl.opcode_shape())
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &crate::classified::Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        FusedAddRmsNormWithOffsetImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(rename_instances_with_tk_prefix)
    }
}

// ── TkTanhSoftCapImpl — peer of [`TanhSoftCapImpl`] ──────────────
//
// TK storer-only in-place logit softcap. Adds `n_vocab` as a 3rd
// field so op_emit.rs can compute the total element count without
// a separate context lookup. Cap value comes from CuLowerCtx's
// `final_softcap_val_const` which is set per-variant in mod.rs.

#[derive(Debug, Default)]
pub struct TkTanhSoftCapImpl;

impl Implementation for TkTanhSoftCapImpl {
    fn name(&self) -> &'static str {
        "tk_tanh_softcap"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        TanhSoftCapImpl.matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        TanhSoftCapImpl.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        TanhSoftCapImpl.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        TanhSoftCapImpl.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        TanhSoftCapImpl.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        TanhSoftCapImpl.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        TanhSoftCapImpl.output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        TanhSoftCapImpl.is_compute_bound()
    }

    fn consumes_input_tiles(&self, claimed_tiles: &[TileId], fuf: &Fuf) -> Vec<(TileId, u8)> {
        TanhSoftCapImpl.consumes_input_tiles(claimed_tiles, fuf)
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &crate::classified::Program,
    ) -> Vec<WeightAccessor> {
        TanhSoftCapImpl.required_weights(claimed, fuf, program)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "TkTanhSoftCap",
            vec![
                ("in_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("n_vocab", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &crate::classified::Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        let n_vocab = *bounds.get("vocab_size").unwrap_or(&0) as u32;
        TanhSoftCapImpl
            .fan_out(m, fuf, program, bounds, slots)
            .map(|ops| {
                ops.into_iter()
                    .map(|inst| match inst {
                        Instruction::TanhSoftCap(a, b) => Instruction::TkTanhSoftCap(a, b, n_vocab),
                        other => other,
                    })
                    .collect()
            })
    }
}

// ── TkFusedAddScalarOffsetRmsNormGemmImpl ─────────────────────────
//
// Gemma2 lm_head: (Add, ScalarOffsetRmsNorm, Gemm) 4-tile fusion.
// Maps to `lm_head_fused_residual_offset` in `lm_head.cuh`.
// Delegates match/weight/alias to CutlassFusedAddScalarOffsetRmsNormGemmImpl;
// custom opcode_shape drops CUTLASS tile dims and adds `offset`.

fn cutlass_fused_add_scalar_offset_rms_norm_gemm_proxy()
-> CutlassFusedAddScalarOffsetRmsNormGemmImpl {
    CutlassFusedAddScalarOffsetRmsNormGemmImpl {
        tile_m: 16,
        tile_n: 64,
        stages: 3,
    }
}

#[derive(Debug, Default)]
pub struct TkFusedAddScalarOffsetRmsNormGemmImpl;

impl Implementation for TkFusedAddScalarOffsetRmsNormGemmImpl {
    fn name(&self) -> &'static str {
        "tk_fused_add_scalar_offset_rms_norm_gemm"
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.compute_capability >= TK_SM_MIN
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn workload_constraint_for_role(
        &self,
        _role: crate::solver::ForwardRole,
    ) -> WorkloadConstraint {
        WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        if !self.target_compatible(profile) {
            return None;
        }
        cutlass_fused_add_scalar_offset_rms_norm_gemm_proxy().matches(fuf, seed, profile)
    }

    fn cost_us(&self, _m: &MatchInfo, _ctx: &CostCtx) -> f64 {
        TK_COST_US
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        cutlass_fused_add_scalar_offset_rms_norm_gemm_proxy().resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        const H: &[Handoff] = &[Handoff::StreamOrder, Handoff::StreamEvent];
        H
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        cutlass_fused_add_scalar_offset_rms_norm_gemm_proxy().input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        cutlass_fused_add_scalar_offset_rms_norm_gemm_proxy().output_layouts(m)
    }

    fn is_compute_bound(&self) -> bool {
        cutlass_fused_add_scalar_offset_rms_norm_gemm_proxy().is_compute_bound()
    }

    fn required_weights(
        &self,
        claimed: &[TileId],
        fuf: &Fuf,
        program: &crate::classified::Program,
    ) -> Vec<WeightAccessor> {
        cutlass_fused_add_scalar_offset_rms_norm_gemm_proxy()
            .required_weights(claimed, fuf, program)
    }

    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        cutlass_fused_add_scalar_offset_rms_norm_gemm_proxy().output_alias(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        OpcodeShape::new(
            "TkFusedAddScalarOffsetRmsNormGemm",
            vec![
                ("delta_slot", syn::parse_quote!(u32)),
                ("residual_slot", syn::parse_quote!(u32)),
                ("out_slot", syn::parse_quote!(u32)),
                ("layer", syn::parse_quote!(u32)),
                ("offset", syn::parse_quote!(f32)),
                ("n", syn::parse_quote!(u32)),
                ("k", syn::parse_quote!(u32)),
            ],
        )
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &crate::classified::Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<Instruction>> {
        use crate::classified::OpKind;
        use crate::codegen::split_base_layer;
        use crate::fuf::FufInput;
        use crate::impl_lib::gemm_nk_from_fuf;

        // Extract residual-Add, scalar-offset-Add, and Gemm from the 4-node claim.
        let residual_add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| {
                let n = fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().all(|i| matches!(i, FufInput::Tile { .. }))
            })
            .expect("TkFusedAddScalarOffsetRmsNormGemm: claim contains residual Add");
        let scalar_add_id = *m
            .claimed_tiles
            .iter()
            .find(|t| {
                let n = fuf.get(**t);
                n.op == OpKind::Add && n.inputs.iter().any(|i| matches!(i, FufInput::Scalar(_)))
            })
            .expect("TkFusedAddScalarOffsetRmsNormGemm: claim contains scalar-offset Add");
        let gemm_id = *m
            .claimed_tiles
            .iter()
            .find(|t| fuf.get(**t).op == OpKind::Gemm)
            .expect("TkFusedAddScalarOffsetRmsNormGemm: claim contains Gemm");

        let add_node = fuf.get(residual_add_id);
        let scalar_add = fuf.get(scalar_add_id);
        let gemm_node = fuf.get(gemm_id);

        let (delta_id, delta_in_slot) = match add_node.inputs.first() {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "TkFusedAddScalarOffsetRmsNormGemm: Add input 0 must be Tile (got {other:?})"
            ),
        };
        let (residual_id, residual_in_slot) = match add_node.inputs.get(1) {
            Some(FufInput::Tile { id, slot }) => (*id, *slot),
            other => panic!(
                "TkFusedAddScalarOffsetRmsNormGemm: Add input 1 must be Tile (got {other:?})"
            ),
        };
        let offset: f32 = scalar_add
            .inputs
            .iter()
            .find_map(|i| match i {
                FufInput::Scalar(v) => Some(*v as f32),
                _ => None,
            })
            .expect("TkFusedAddScalarOffsetRmsNormGemm: scalar-offset Add has a Scalar");

        let delta_idx = slots.of(delta_id, delta_in_slot);
        let residual_idx = slots.of(residual_id, residual_in_slot);
        let out_slot_idx = slots.of(gemm_id, 0);

        let accessors = self.required_weights(&m.claimed_tiles, fuf, program);
        let norm_acc = accessors
            .first()
            .expect("TkFusedAddScalarOffsetRmsNormGemm: norm weight");
        let _gemm_acc = accessors
            .get(1)
            .expect("TkFusedAddScalarOffsetRmsNormGemm: gemm weight");

        let (_norm_base, norm_layer) = split_base_layer(&norm_acc.name.to_string());
        let layer = norm_layer.unwrap_or(0) as u32;

        let (n, k) = gemm_nk_from_fuf(fuf, gemm_node, bounds)
            .expect("TkFusedAddScalarOffsetRmsNormGemm: gemm (N, K)");

        Some(vec![Instruction::TkFusedAddScalarOffsetRmsNormGemm(
            delta_idx,
            residual_idx,
            out_slot_idx,
            layer,
            offset,
            n,
            k,
        )])
    }

    fn applies_to(&self, ctx: &MatchContext) -> bool {
        TkGemmImpl.applies_to(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sm<90 targets must surface `target_compatible=false` on every
    /// Tk peer. Keeps mega's zoo inert on L4/A100/CPU builds so the
    /// DP never picks a Tk opcode on those.
    #[test]
    fn target_compatibility_gates_on_sm90() {
        use ferrite_cuda_targets::H100_SM90;
        let mut profile = crate::target::from_profile_def(&H100_SM90);
        profile.compute_capability = 89;

        assert!(!TkEmbedImpl.target_compatible(&profile));
        assert!(!TkRmsNormImpl.target_compatible(&profile));
        assert!(!TkGemmImpl.target_compatible(&profile));
        assert!(!TkFusedAddRmsNormImpl.target_compatible(&profile));
        assert!(!TkFusedQkvRopeCacheImpl.target_compatible(&profile));
        assert!(!TkAttentionViaCacheImpl.target_compatible(&profile));
        assert!(!TkFusedGateUpSiluMulImpl.target_compatible(&profile));

        profile.compute_capability = 90;
        assert!(TkEmbedImpl.target_compatible(&profile));
        assert!(TkRmsNormImpl.target_compatible(&profile));
        assert!(TkGemmImpl.target_compatible(&profile));
        assert!(TkFusedAddRmsNormImpl.target_compatible(&profile));
        assert!(TkFusedQkvRopeCacheImpl.target_compatible(&profile));
        assert!(TkAttentionViaCacheImpl.target_compatible(&profile));
        assert!(TkFusedGateUpSiluMulImpl.target_compatible(&profile));
    }

    /// Each Tk peer's `opcode_shape` prepends `Tk` to its backend
    /// peer's variant ident and preserves the field list verbatim.
    /// This is the compile-time contract
    /// `crate::tape::tk_mega::op_emit::op_refs` keys off.
    #[test]
    fn opcode_shapes_prefix_name_and_preserve_fields() {
        fn check(base: OpcodeShape, tk: OpcodeShape) {
            assert_eq!(tk.name.to_string(), format!("Tk{}", base.name));
            assert_eq!(tk.fields.len(), base.fields.len());
            for ((a_n, _), (b_n, _)) in base.fields.iter().zip(tk.fields.iter()) {
                assert_eq!(a_n, b_n);
            }
        }
        check(EmbedRefImpl.opcode_shape(), TkEmbedImpl.opcode_shape());
        check(RmsNormRefImpl.opcode_shape(), TkRmsNormImpl.opcode_shape());
        check(GemmRefImpl.opcode_shape(), TkGemmImpl.opcode_shape());
        check(
            FusedAddRmsNormImpl.opcode_shape(),
            TkFusedAddRmsNormImpl.opcode_shape(),
        );
        check(
            FusedQkvRopeCacheImpl.opcode_shape(),
            TkFusedQkvRopeCacheImpl.opcode_shape(),
        );
        check(
            AttentionViaCacheImpl.opcode_shape(),
            TkAttentionViaCacheImpl.opcode_shape(),
        );
        check(
            FusedGateUpSiluMulImpl.opcode_shape(),
            TkFusedGateUpSiluMulImpl.opcode_shape(),
        );
    }
}
