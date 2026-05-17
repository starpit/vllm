// SPDX-License-Identifier: Apache-2.0
//! Persistent-envelope variant of `MetalSynthPreAttnImpl`.
//!
//! Wraps the existing pre-attn synth Impl and rewrites its emitted
//! `Instruction::SynthPreAttn` into `Instruction::SynthPreAttnPersistent`
//! (same tuple shape, persistent kernel symbol). The persistent kernel
//! is byte-identical body to the non-persistent variant, plus one
//! appended counter binding + a trailing cross-TG barrier. Single-phase
//! persistent is overhead-only (~5 µs trailing barrier); Phase 2b will
//! fuse across chunks for the real ~10× decode win.
//!
//! Gating: `target_compatible` additionally requires the runtime env
//! var `FERRITE_PERSISTENT_PREATTN=1`. Without it, the solver pool
//! contains only the non-persistent variant (today's behavior). With
//! it, the persistent variant claims the chain and the non-persistent
//! one stays inert (matches the same tiles but the solver picks one).

use std::collections::BTreeMap;

use crate::classified::Program;
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources,
    SlotMap, WeightAccessor, WorkloadConstraint,
};
use crate::metal::synth_pre_attn::MetalSynthPreAttnImpl;
use crate::target::TargetProfile;

/// Persistent-envelope variant. Delegates everything except `name`,
/// `target_compatible` (env gate), `fan_out` (Instruction rewrite),
/// and `opcode_shape` (new variant name).
#[derive(Debug)]
pub struct MetalSynthPreAttnPersistentImpl {
    inner: MetalSynthPreAttnImpl,
}

impl MetalSynthPreAttnPersistentImpl {
    pub fn bf16_gs64() -> Self {
        Self { inner: MetalSynthPreAttnImpl::bf16_gs64() }
    }
    pub fn bf16_gs64_init() -> Self {
        Self { inner: MetalSynthPreAttnImpl::bf16_gs64_init() }
    }

    fn is_enabled() -> bool {
        std::env::var("FERRITE_PERSISTENT_PREATTN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }
}

impl Implementation for MetalSynthPreAttnPersistentImpl {
    fn name(&self) -> &'static str {
        if self.inner.init {
            "metal_synth_pre_attn_init_persistent"
        } else {
            "metal_synth_pre_attn_persistent"
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        self.inner.target_compatible(profile) && Self::is_enabled()
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        self.inner.workload_constraint()
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        self.inner.matches(fuf, seed, profile)
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        // Same as non-persistent — single-phase persistent envelope
        // adds only ~5 µs trailing barrier, which is below cost-model
        // resolution. Solver picks this when env-enabled (and the
        // non-persistent variant is filtered out by target_compatible).
        self.inner.cost_us(m, ctx)
    }

    fn resources(&self, m: &MatchInfo) -> Resources {
        self.inner.resources(m)
    }

    fn launch_kind(&self) -> LaunchKind {
        self.inner.launch_kind()
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        self.inner.supported_input_handoffs()
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        self.inner.supported_output_handoffs()
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        self.inner.input_layouts(m)
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        self.inner.output_layouts(m)
    }

    #[allow(clippy::type_complexity)]
    fn output_alias(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> Vec<((TileId, u8), Option<(TileId, u8)>)> {
        self.inner.output_alias(claimed_tiles, fuf)
    }

    fn kv_layer_io(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
    ) -> (Option<u32>, Option<u32>) {
        self.inner.kv_layer_io(claimed_tiles, fuf)
    }

    fn opcode_shape(&self) -> OpcodeShape {
        // Same field layout as SynthPreAttn — only the canonical
        // name changes so the macro registers a distinct opcode.
        let base = self.inner.opcode_shape();
        OpcodeShape {
            name: syn::Ident::new("SynthPreAttnPersistent", proc_macro2::Span::call_site()),
            fields: base.fields,
        }
    }

    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<ferrite_forward::Instruction>> {
        let inner = self.inner.fan_out(m, fuf, program, bounds, slots)?;
        // Rewrite every SynthPreAttn instruction into SynthPreAttnPersistent
        // with the `_persistent` symbol variant. The non-persistent
        // symbol (e.g. "synth_pre_attn_bfloat_half_gs64") becomes
        // "synth_pre_attn_persistent_bfloat_half_gs64"; init variant
        // becomes "synth_pre_attn_init_persistent_*". Leak the new
        // string so the &'static str on the Instruction outlives the
        // macro invocation, mirroring how the inner impl leaks its
        // symbol.
        Some(
            inner
                .into_iter()
                .map(|instr| match instr {
                    ferrite_forward::Instruction::SynthPreAttn(
                        a, b, c, d, e, f, sym, h,
                    ) => {
                        let new_sym: String = if sym.contains("_init_") {
                            sym.replacen("_init_", "_init_persistent_", 1)
                        } else if let Some(stripped) = sym.strip_prefix("synth_pre_attn_") {
                            format!("synth_pre_attn_persistent_{stripped}")
                        } else {
                            // Defensive: unfamiliar symbol — leave alone (will
                            // fail to resolve at runtime, surfacing the bug).
                            sym.to_string()
                        };
                        let leaked: &'static str = Box::leak(new_sym.into_boxed_str());
                        ferrite_forward::Instruction::SynthPreAttnPersistent(
                            a, b, c, d, e, f, leaked, h,
                        )
                    }
                    other => other,
                })
                .collect(),
        )
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        self.inner.required_weights(claimed_tiles, fuf, program)
    }
}
