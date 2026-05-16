// SPDX-License-Identifier: Apache-2.0
//! Tape-level claim trait + library.
//!
//! The Instruction-level solve (FUF → Tape, in
//! [`crate::impl_lib::ImplementationLibrary`]) picks the backend
//! for each claim by cost. That produces a Tape — a flat stream of
//! backend-qualified [`crate::impl_lib::OpInstance`]s.
//!
//! Then a second solve picks an EXECUTOR for the whole Tape.
//! Different executors have different capabilities: the host
//! interpreter can run any Tape (it dispatches `Instruction::eval`
//! per op at runtime), while the TK megakernel executor only
//! claims Tapes whose every op is a `Tk*`-named variant AND the
//! target is sm≥90. A [`TapeClaimer`] answers both questions —
//! "can I run this Tape?" and "what's my cost?" — and, when
//! picked, emits the compile-time artifacts the executor needs.
//!
//! The Instruction-level and Tape-level solves are intentionally
//! separate. Tape-level claim is about *capability first*, cost
//! second. Folding it into the Instruction-level DP would let
//! cost dominate capability, which picks a TkMega-winning per-op
//! claim even when another op on the Tape is cuBLAS-only and
//! mega can't actually execute.

use crate::impl_lib::OpInstance;
use crate::solver::WorkloadPoint;
use crate::target::TargetProfile;
use proc_macro2::{Ident, TokenStream};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Per-canonical artifacts a tape claimer emits at macro-expansion
/// time. The host interpreter emits `TapeEmission::none()` (it
/// runs via the default per-bucket `Instruction::eval` loop that
/// `emit_model` already generates); the TK mega executor emits a
/// `.cu` file + Rust dispatch.
#[derive(Debug, Default)]
pub struct TapeEmission {
    /// Path to the `.cu` written into the cudaforge cache, or
    /// `None` if the executor has no compile-time C++ artifact.
    pub cu_path: Option<PathBuf>,
    /// Rust tokens to splice into the model module — extern decls,
    /// per-canonical forward fns, etc.
    pub rust_decls: TokenStream,
    /// If this executor emits a dedicated per-bucket forward fn
    /// (e.g. `forward_mega_<variant>`), its Rust ident — consumed
    /// by the model-level dispatch table emitter so the bucket
    /// row points at the executor's entry point. `None` means
    /// the bucket uses the default host-interpreter forward.
    pub forward_fn: Option<Ident>,
    /// When an `_ms` (multi-step cooperative) companion kernel was
    /// successfully emitted for this canonical, the ident of its
    /// `LAUNCH_FN_<NAME>_MS` constant. `None` when either the
    /// canonical has no multi-step variant or the `_ms` CU bailed
    /// to `#error` (logits_slot not found). Consumed by codegen.rs
    /// to build `MEGA_FORWARD_TABLE_MULTI_STEP`.
    pub ms_launch_fn: Option<Ident>,
    /// When a persistent decode companion kernel was successfully emitted,
    /// the ident of its `LAUNCH_FN_<NAME>_PERSISTENT_DECODE` constant.
    /// Used by `PersistentDecodeSession` to obtain the fn pointer for a
    /// one-time cooperative launch.
    pub persistent_decode_launch_fn: Option<Ident>,
}

impl TapeEmission {
    pub fn none() -> Self {
        Self::default()
    }
}

/// Evidence (from [`TapeClaimer::matches`]) that an executor can
/// run a given tape. Carried into [`TapeClaimer::emit`] so the
/// emit pass doesn't re-run capability probes.
pub trait TapeMatchInfo: std::any::Any + std::fmt::Debug {}

impl<T: std::any::Any + std::fmt::Debug> TapeMatchInfo for T {}

/// Executor-level claim. Parallels
/// [`crate::impl_lib::Implementation`] but operates on whole
/// Tapes (Instruction sequences), not FUF subgraphs.
pub trait TapeClaimer: std::fmt::Debug {
    /// Identifier for diagnostics / error messages.
    fn name(&self) -> &'static str;

    /// Capability gate: can this executor run THIS tape on THIS
    /// target? Returns `Some(evidence)` if yes — the evidence is
    /// opaque to the caller, passed back into `emit()`. Returns
    /// `None` if the executor cannot claim.
    fn matches(
        &self,
        backbone: &[OpInstance],
        lm_head: &[OpInstance],
        ctx: &TapeEmitCtx<'_>,
    ) -> Option<Box<dyn TapeMatchInfo>>;

    /// Cost estimate (microseconds) for running this tape with
    /// this executor. The tape-level DP picks the cheapest capable
    /// claimer per canonical. Ties broken by first-registered-
    /// wins (library order).
    fn cost_us(
        &self,
        backbone: &[OpInstance],
        lm_head: &[OpInstance],
        ctx: &TapeEmitCtx<'_>,
        info: &dyn TapeMatchInfo,
    ) -> f64;

    /// Compile-time codegen: write any `.cu` files to the
    /// cudaforge cache and return the Rust tokens + per-bucket
    /// forward-fn ident this executor contributes to the model
    /// module.
    fn emit(
        &self,
        canonical_name: &str,
        wp: WorkloadPoint,
        backbone: &[OpInstance],
        lm_head: &[OpInstance],
        terminal_slot: u32,
        ctx: &TapeEmitCtx<'_>,
        info: &dyn TapeMatchInfo,
    ) -> TapeEmission;
}

/// Context passed to [`TapeClaimer::emit`]. Holds the model-level
/// metadata every executor's emit pass needs (shapes, eps, target
/// profile, tp world size, Rust-side accessor types) without each
/// claimer re-deriving it.
pub struct TapeEmitCtx<'a> {
    pub shapes: &'a BTreeMap<String, crate::impl_lib::OpcodeShape>,
    pub rms_norm_eps: f32,
    pub profile: &'a TargetProfile,
    pub tp_world_size: u8,
    pub bounds: &'a BTreeMap<String, u64>,
    /// Float-valued model scalars (e.g. `attn_logit_softcapping`).
    /// Separate from `bounds` (which only carries integer values).
    pub scalars: &'a BTreeMap<String, f64>,
    pub accessor_type_by_base: &'a BTreeMap<String, String>,
}

/// A set of tape claimers, consulted in order. The first claimer
/// to `matches()` wins ties; for multiple matches the `cost_us()`
/// minimum wins. Typically built once per `#[ferrite_forward]`
/// expansion via [`starter_tape_library`].
pub struct TapeClaimerLibrary {
    claimers: Vec<Box<dyn TapeClaimer>>,
}

impl TapeClaimerLibrary {
    pub fn new() -> Self {
        Self {
            claimers: Vec::new(),
        }
    }

    pub fn push(&mut self, c: Box<dyn TapeClaimer>) {
        self.claimers.push(c);
    }

    /// Pick the cheapest capable claimer for a canonical. Returns
    /// `(claimer_idx, match_info)` so the caller can call the
    /// same claimer's `emit()`. `None` means zero claimers matched
    /// — which shouldn't happen in practice because
    /// [`crate::tape::host_interp::HostInterpreterTapeClaimer`]
    /// always matches; treat it as a bug.
    pub fn pick(
        &self,
        backbone: &[OpInstance],
        lm_head: &[OpInstance],
        ctx: &TapeEmitCtx<'_>,
    ) -> Option<(usize, Box<dyn TapeMatchInfo>)> {
        let mut best: Option<(usize, f64, Box<dyn TapeMatchInfo>)> = None;
        for (idx, claimer) in self.claimers.iter().enumerate() {
            let Some(info) = claimer.matches(backbone, lm_head, ctx) else {
                continue;
            };
            let cost = claimer.cost_us(backbone, lm_head, ctx, info.as_ref());
            match &best {
                Some((_, best_cost, _)) if *best_cost <= cost => {}
                _ => best = Some((idx, cost, info)),
            }
        }
        best.map(|(i, _, info)| (i, info))
    }

    pub fn claimer(&self, idx: usize) -> &dyn TapeClaimer {
        self.claimers[idx].as_ref()
    }
}

impl Default for TapeClaimerLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Starter library: host interpreter (universal fallback) + TK
/// mega (sm≥90 + all-`Tk*` tape). Host is pushed first so it wins
/// ties at equal cost.
pub fn starter_tape_library() -> TapeClaimerLibrary {
    let mut lib = TapeClaimerLibrary::new();
    lib.push(Box::new(
        crate::tape::host_interp::HostInterpreterTapeClaimer,
    ));
    lib
}
