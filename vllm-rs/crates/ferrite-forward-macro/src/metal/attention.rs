// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal Attention implementation adapter.
//!
//! Wraps Metal attention kernels (basic, paged, multi-head, GQA, optimized)
//! to satisfy ferrite's Implementation trait.

use std::collections::BTreeMap;

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    AttentionPrefillContiguousImpl, AttentionViaCacheImpl, CostCtx, Handoff, Implementation,
    LaunchKind, Layout, MatchInfo, OpInstance, OpcodeShape, Resources, SlidingAttentionPrefillContiguousImpl,
    SlidingAttentionViaCacheImpl, SlotMap, WeightAccessor, WorkloadConstraint,
    default_required_weights,
};
use crate::target::{Backend, TargetProfile};

/// Adapter that wraps Metal Attention kernels to satisfy ferrite's Implementation trait.
///
/// Supports multiple attention variants:
/// - Basic single-head attention
/// - Paged KV cache with block tables
/// - Multi-head attention (MHA)
/// - Grouped-query attention (GQA)
/// - Optimized variants with vectorized loads
#[derive(Debug)]
pub struct MetalAttentionImpl {
    /// Kernel name for cost table lookup
    kernel_name: &'static str,
    /// Data type (fp16 or bf16)
    dtype: &'static str,
    /// Whether this is paged attention (uses block tables)
    is_paged: bool,
    /// Whether this is multi-head attention. Drives the
    /// decode (M=1, false) vs prefill (M>=2, true) workload-constraint
    /// split that mirrors `AttentionViaCacheImpl` /
    /// `AttentionPrefillContiguousImpl` on CUDA.
    is_multihead: bool,
    /// Whether this uses optimized vectorized loads
    is_optimized: bool,
    /// Whether this variant claims `OpKind::SlidingAttention` instead
    /// of `OpKind::Attention`. The two opcodes differ in masking but
    /// share the same FUF tile shape; flagging at impl-construction
    /// time keeps `opcode_shape` deterministic per impl (one variant
    /// per declaration is the trait contract).
    is_sliding: bool,
}

impl MetalAttentionImpl {
    /// Basic single-head attention (fp16)
    pub fn new_basic_fp16() -> Self {
        Self {
            kernel_name: "attention_basic_f16",
            dtype: "fp16",
            is_paged: false,
            is_multihead: false,
            is_optimized: false,
            is_sliding: false,
        }
    }

    /// Paged attention with block tables (fp16)
    pub fn new_paged_fp16() -> Self {
        Self {
            kernel_name: "attention_paged_f16",
            dtype: "fp16",
            is_paged: true,
            is_multihead: false,
            is_optimized: false,
            is_sliding: false,
        }
    }

    /// Multi-head attention (fp16)
    pub fn new_multihead_fp16() -> Self {
        Self {
            kernel_name: "attention_multihead_f16",
            dtype: "fp16",
            is_paged: true,
            is_multihead: true,
            is_optimized: false,
            is_sliding: false,
        }
    }

    /// Optimized multi-head attention with vectorized loads (fp16)
    pub fn new_multihead_optimized_fp16() -> Self {
        Self {
            kernel_name: "attention_multihead_optimized_f16",
            dtype: "fp16",
            is_paged: true,
            is_multihead: true,
            is_optimized: true,
            is_sliding: false,
        }
    }

    /// Sliding-window paged attention, decode (M=1) — Gemma2/Gemma3
    /// alternating-layer attention. Reuses the paged-decode kernel with
    /// the sliding-window mask flag baked into the kernel constants.
    pub fn new_sliding_paged_fp16() -> Self {
        Self {
            kernel_name: "sliding_attention_paged_f16",
            dtype: "fp16",
            is_paged: true,
            is_multihead: false,
            is_optimized: false,
            is_sliding: true,
        }
    }

    /// Sliding-window prefill (M>=2). Same multihead-optimized base
    /// kernel as the non-sliding prefill variant; differs only in mask.
    pub fn new_sliding_multihead_optimized_fp16() -> Self {
        Self {
            kernel_name: "sliding_attention_multihead_optimized_f16",
            dtype: "fp16",
            is_paged: true,
            is_multihead: true,
            is_optimized: true,
            is_sliding: true,
        }
    }

    /// Analytical cost model for attention (memory-bound + compute-bound).
    ///
    /// Attention has two phases:
    /// 1. Q·K^T: Compute-bound, O(seq_len * head_size) FLOPs per token
    /// 2. Softmax + Attention·V: Memory-bound, O(seq_len * head_size) bytes
    ///
    /// Cost = max(compute_cost, memory_cost) since they overlap
    fn analytical_cost_us(
        &self,
        num_heads: u32,
        head_size: u32,
        seq_len: u32,
        bandwidth_gbps: f64,
        compute_tflops: f64,
    ) -> f64 {
        let bytes_per_element = 2.0; // fp16/bf16
        
        // Phase 1: Q·K^T compute cost
        // FLOPs = num_heads * seq_len * head_size * 2 (multiply-add)
        let qk_flops = (num_heads as f64) * (seq_len as f64) * (head_size as f64) * 2.0;
        let compute_cost_us = (qk_flops / (compute_tflops * 1e12)) * 1e6;
        
        // Phase 2: Memory cost (read Q, K, V, write output)
        // Reads: Q [num_heads, head_size] + K [seq_len, head_size] + V [seq_len, head_size]
        // Writes: output [num_heads, head_size]
        let q_bytes = (num_heads as f64) * (head_size as f64) * bytes_per_element;
        let kv_bytes = (seq_len as f64) * (head_size as f64) * bytes_per_element * 2.0;
        let output_bytes = (num_heads as f64) * (head_size as f64) * bytes_per_element;
        let total_bytes = q_bytes + kv_bytes + output_bytes;
        let memory_cost_us = (total_bytes / 1e9 / bandwidth_gbps) * 1e6;
        
        // Return max since phases overlap
        compute_cost_us.max(memory_cost_us)
    }
}

impl Implementation for MetalAttentionImpl {
    fn name(&self) -> &'static str {
        match (self.is_sliding, self.is_optimized, self.is_multihead, self.is_paged) {
            (true, true, _, _) => "metal_sliding_attention_multihead_optimized_f16",
            (true, _, _, _) => "metal_sliding_attention_paged_f16",
            (false, true, _, _) => "metal_attention_multihead_optimized_f16",
            (false, _, true, _) => "metal_attention_multihead_f16",
            (false, _, _, true) => "metal_attention_paged_f16",
            (false, _, _, false) => "metal_attention_basic_f16",
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Only compatible with Metal targets
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        // Mirror the CUDA decode/prefill split. The non-multihead
        // variants (basic, paged) wrap kernels designed for the
        // single-Q-token paged-cache decode path; the multihead
        // variants drive the contiguous-Q/K/V prefill path. Without
        // this split, all 4 variants compete at every bucket and the
        // solver picks one whose `Instruction` shape doesn't match
        // the upstream layout the FUF produces at that bucket.
        if self.is_multihead {
            WorkloadConstraint::NumTokensRange {
                min: 2,
                max: u32::MAX,
            }
        } else {
            WorkloadConstraint::NumTokensRange { min: 1, max: 1 }
        }
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, profile: &TargetProfile) -> Option<MatchInfo> {
        // Defer the matcher (incl. the kv_cache-extern gate that
        // distinguishes decoder from encoder attention) to the CUDA
        // counterpart. `workload_constraint` above already gates
        // decode-only vs prefill-only; `is_sliding` selects between
        // `OpKind::Attention` and `OpKind::SlidingAttention`.
        match (self.is_sliding, self.is_multihead) {
            (false, false) => AttentionViaCacheImpl.matches(fuf, seed, profile),
            (false, true) => AttentionPrefillContiguousImpl.matches(fuf, seed, profile),
            (true, false) => SlidingAttentionViaCacheImpl.matches(fuf, seed, profile),
            (true, true) => SlidingAttentionPrefillContiguousImpl.matches(fuf, seed, profile),
        }
    }

    fn cost_us(&self, m: &MatchInfo, ctx: &CostCtx) -> f64 {
        let tile = m.claimed_tiles[0];
        let node = ctx.fuf.get(tile);
        
        // Get shape: Attention output is [num_heads, head_size]
        let shape = &node.outputs[0];
        let dims = ctx.eval_shape(shape);
        
        if let Some(dims) = dims {
            if dims.len() >= 2 {
                let num_heads = dims[0] as u32;
                let head_size = dims[1] as u32;
                
                // Get sequence length from bounds (context_len or similar)
                let seq_len = ctx.bounds.get("context_len")
                    .or_else(|| ctx.bounds.get("max_seq_len"))
                    .copied()
                    .unwrap_or(2048) as u32;
                
                // Try empirical cost first
                if let Some(cost) = ctx.profile.cost_us_for(self.kernel_name, num_heads, head_size, seq_len) {
                    return cost;
                }
                
                // Fall back to analytical model
                return self.analytical_cost_us(
                    num_heads,
                    head_size,
                    seq_len,
                    ctx.profile.memory_bandwidth_gbps,
                    ctx.profile.peak_tflops_fp16,
                );
            }
        }
        
        // Fallback: conservative estimate (attention is expensive)
        1000.0
    }

    fn resources(&self, _m: &MatchInfo) -> Resources {
        Resources {
            shmem_bytes: 0, // Metal uses threadgroup memory
            regs_per_thread: 64,
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

    fn opcode_shape(&self) -> OpcodeShape {
        match (self.is_sliding, self.is_multihead) {
            (false, false) => AttentionViaCacheImpl.opcode_shape(),
            (false, true) => AttentionPrefillContiguousImpl.opcode_shape(),
            (true, false) => SlidingAttentionViaCacheImpl.opcode_shape(),
            (true, true) => SlidingAttentionPrefillContiguousImpl.opcode_shape(),
        }
    }
    fn fan_out(
        &self,
        m: &MatchInfo,
        fuf: &Fuf,
        program: &Program,
        bounds: &BTreeMap<String, u64>,
        slots: &SlotMap,
    ) -> Option<Vec<OpInstance>> {
        match (self.is_sliding, self.is_multihead) {
            (false, false) => AttentionViaCacheImpl.fan_out(m, fuf, program, bounds, slots),
            (false, true) => AttentionPrefillContiguousImpl.fan_out(m, fuf, program, bounds, slots),
            (true, false) => SlidingAttentionViaCacheImpl.fan_out(m, fuf, program, bounds, slots),
            (true, true) => {
                SlidingAttentionPrefillContiguousImpl.fan_out(m, fuf, program, bounds, slots)
            }
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::target::from_metal_profile;

    #[test]
    fn metal_attention_only_compatible_with_metal_targets() {
        let metal_impl = MetalAttentionImpl::new_multihead_optimized_fp16();
        
        // Metal target - should be compatible
        let metal_profile = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        assert!(metal_impl.target_compatible(&metal_profile));
        
        // CUDA target - should NOT be compatible
        let cuda_profile = crate::target::from_profile_def(&ferrite_cuda_targets::L4_SM89);
        assert!(!metal_impl.target_compatible(&cuda_profile));
    }

    #[test]
    fn metal_attention_analytical_cost_scales_with_seq_len() {
        let impl_fp16 = MetalAttentionImpl::new_multihead_optimized_fp16();
        
        // Short sequence
        let cost_short = impl_fp16.analytical_cost_us(32, 128, 256, 400.0, 10.4);
        
        // Long sequence (4× longer)
        let cost_long = impl_fp16.analytical_cost_us(32, 128, 1024, 400.0, 10.4);
        
        // Cost should scale roughly linearly with seq_len
        let ratio = cost_long / cost_short;
        assert!((ratio - 4.0).abs() < 1.0, "Expected ratio ~4.0, got {}", ratio);
    }

    #[test]
    fn metal_attention_analytical_cost_scales_with_num_heads() {
        let impl_fp16 = MetalAttentionImpl::new_multihead_optimized_fp16();
        
        // Few heads
        let cost_few = impl_fp16.analytical_cost_us(8, 128, 512, 400.0, 10.4);
        
        // Many heads (4× more)
        let cost_many = impl_fp16.analytical_cost_us(32, 128, 512, 400.0, 10.4);
        
        // Cost should scale roughly linearly with num_heads
        let ratio = cost_many / cost_few;
        assert!((ratio - 4.0).abs() < 1.0, "Expected ratio ~4.0, got {}", ratio);
    }
}
