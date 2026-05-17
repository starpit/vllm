// SPDX-License-Identifier: Apache-2.0
//! Persistent-envelope variant of `MetalSynthMlpPreDownImpl`.
//!
//! Wraps the existing mlp-pre-down synth Impl and rewrites its emitted
//! `Instruction::SynthMlpPreDown` into `Instruction::SynthMlpPreDownPersistent`
//! (same tuple shape, persistent kernel symbol). The persistent kernel
//! is byte-identical body to the non-persistent variant, plus one
//! appended counter binding + a trailing cross-TG barrier. Pairs with
//! `MetalSynthPreAttnPersistentImpl` so a full layer (pre_attn ⟶
//! attention ⟶ mlp_pre_down) becomes one persistent pre-attn dispatch
//! ⟶ one attention dispatch ⟶ one persistent mlp_pre_down dispatch.
//!
//! Gating: `target_compatible` additionally requires the runtime env
//! var `FERRITE_PERSISTENT_PREATTN=1` (same gate as Phase 2a — both
//! turn on together). Without it, the solver pool contains only the
//! non-persistent variant.

use std::collections::BTreeMap;

use crate::classified::Program;
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpcodeShape, Resources,
    SlotMap, WeightAccessor, WorkloadConstraint,
};
use crate::metal::synth_mlp_pre_down::MetalSynthMlpPreDownImpl;
use crate::target::TargetProfile;

/// Persistent-envelope variant. Delegates everything except `name`,
/// `target_compatible` (env gate), `fan_out` (Instruction rewrite),
/// and `opcode_shape` (new variant name).
#[derive(Debug)]
pub struct MetalSynthMlpPreDownPersistentImpl {
    inner: MetalSynthMlpPreDownImpl,
}

impl MetalSynthMlpPreDownPersistentImpl {
    pub fn bf16_gs64() -> Self {
        Self { inner: MetalSynthMlpPreDownImpl::bf16_gs64() }
    }

    fn is_enabled() -> bool {
        std::env::var("FERRITE_PERSISTENT_PREATTN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }
}

impl Implementation for MetalSynthMlpPreDownPersistentImpl {
    fn name(&self) -> &'static str {
        "metal_synth_mlp_pre_down_persistent"
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
        // ε-tiebreak vs the non-persistent variant — see Phase 2a's
        // MetalSynthPreAttnPersistentImpl::cost_us for the rationale.
        self.inner.cost_us(m, ctx) - 0.001
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

    fn opcode_shape(&self) -> OpcodeShape {
        let base = self.inner.opcode_shape();
        OpcodeShape {
            name: syn::Ident::new("SynthMlpPreDownPersistent", proc_macro2::Span::call_site()),
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
        Some(
            inner
                .into_iter()
                .map(|instr| match instr {
                    ferrite_forward::Instruction::SynthMlpPreDown(
                        a, b, c, d, e, f, sym,
                    ) => {
                        let new_sym: String = if let Some(stripped) =
                            sym.strip_prefix("synth_mlp_pre_down_")
                        {
                            format!("synth_mlp_pre_down_persistent_{stripped}")
                        } else {
                            sym.to_string()
                        };
                        let leaked: &'static str = Box::leak(new_sym.into_boxed_str());
                        ferrite_forward::Instruction::SynthMlpPreDownPersistent(
                            a, b, c, d, e, f, leaked,
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
