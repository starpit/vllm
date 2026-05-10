// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal implementation for Embed (lookup table) operation.
//!
//! Embed performs a lookup table operation: given input_ids [N] and an embedding
//! table [vocab_size, hidden_size], produces output [N, hidden_size] by indexing
//! into the table.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, EmbedRefImpl, Handoff, Implementation, LaunchKind, Layout, MatchInfo, OpInstance,
    OpcodeShape, Resources, SlotMap, WeightAccessor, WorkloadConstraint, default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Metal implementation for Embed operation.
///
/// Signature: `embed(input_ids: [N], table: [vocab_size, hidden_size]) -> [N, hidden_size]`
///
/// This is a memory-bound lookup operation. Cost is proportional to output size.
#[derive(Debug)]
pub struct MetalEmbedImpl {
    dtype: &'static str,
}

impl MetalEmbedImpl {
    pub fn fp16() -> Self {
        Self { dtype: "fp16" }
    }

    pub fn bf16() -> Self {
        Self { dtype: "bf16" }
    }

    /// Analytical cost model for Embed (memory-bound lookup).
    /// Cost = (bytes_read + bytes_written) / bandwidth
    fn analytical_cost_us(&self, num_tokens: u32, hidden_size: u32, bandwidth_gbps: f64) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        let output_elements = (num_tokens * hidden_size) as f64;

        // Read from embedding table + write output
        let bytes_read = output_elements * bytes_per_element;
        let bytes_written = output_elements * bytes_per_element;
        let total_bytes = bytes_read + bytes_written;

        let total_gb = total_bytes / 1e9;
        let time_seconds = total_gb / bandwidth_gbps;
        time_seconds * 1e6 // convert to microseconds
    }
}

impl Implementation for MetalEmbedImpl {
    fn name(&self) -> &'static str {
        match self.dtype {
            "fp16" => "metal_embed_f16",
            "bf16" => "metal_embed_bf16",
            _ => "metal_embed",
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        if node.op != OpKind::Embed {
            return None;
        }

        // Singleton claim
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);

        // Output shape: [num_tokens, hidden_size]
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);

        if let Some(dims) = dims
            && dims.len() == 2
        {
            let num_tokens = dims[0] as u32;
            let hidden_size = dims[1] as u32;

            // Try empirical cost first
            let kernel_name = match self.dtype {
                "fp16" => "embed_f16",
                "bf16" => "embed_bf16",
                _ => "embed_f16",
            };

            if let Some(cost) = ctx
                .profile
                .cost_us_for(kernel_name, num_tokens, hidden_size, 0)
            {
                return cost;
            }

            // Fall back to analytical model
            return self.analytical_cost_us(
                num_tokens,
                hidden_size,
                ctx.profile.memory_bandwidth_gbps,
            );
        }

        // Fallback: conservative estimate
        10.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0,
            regs_per_thread: 8,
            threads_per_cta: 256,
        }
    }

    fn launch_kind(&self) -> LaunchKind {
        LaunchKind::HostCallback
    }

    fn supported_input_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder]
    }

    fn supported_output_handoffs(&self) -> &[Handoff] {
        &[Handoff::StreamOrder]
    }

    fn input_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_inputs.len()]
    }

    fn output_layouts(&self, m: &MatchInfo) -> Vec<Layout> {
        vec![Layout::Any; m.boundary_outputs.len()]
    }

    fn required_weights(
        &self,
        claimed_tiles: &[TileId],
        fuf: &Fuf,
        program: &Program,
    ) -> Vec<WeightAccessor> {
        default_required_weights(claimed_tiles, fuf, program)
    }

    // Host-interpreter codegen: delegate to the CUDA `EmbedRefImpl` so
    // both backends produce structurally identical `Embed` variants
    // and `OpInstance`s. The emission is target-agnostic — it just
    // names a slot index and a `Weights::embed_tokens` accessor — so
    // duplication here would be pure boilerplate. (The CUDA impl is
    // never *registered* in a metal-feature build per 5.F.2's
    // cfg-gated `starter_library`; we only borrow its emission.)
    fn opcode_shape(&self) -> OpcodeShape {
        EmbedRefImpl.opcode_shape()
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        EmbedRefImpl.fan_out(m, fuf, program, bounds, slots)
    }
}
