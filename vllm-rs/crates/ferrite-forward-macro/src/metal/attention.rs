// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal Attention implementation adapter.
//!
//! Wraps Metal attention kernels (basic, paged, multi-head, GQA, optimized)
//! to satisfy ferrite's Implementation trait.

use crate::classified::{OpKind, Program};
use crate::fuf::{Fuf, TileId};
use crate::impl_lib::{
    CostCtx, Handoff, Implementation, LaunchKind, Layout, MatchInfo, Resources,
    WeightAccessor, WorkloadConstraint, default_required_weights,
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
    /// Whether this is multi-head attention
    is_multihead: bool,
    /// Whether this uses optimized vectorized loads
    is_optimized: bool,
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
        if self.is_optimized {
            "metal_attention_multihead_optimized_f16"
        } else if self.is_multihead {
            "metal_attention_multihead_f16"
        } else if self.is_paged {
            "metal_attention_paged_f16"
        } else {
            "metal_attention_basic_f16"
        }
    }

    fn target_compatible(&self, profile: &TargetProfile) -> bool {
        // Only compatible with Metal targets
        profile.backend == Backend::Metal
    }

    fn workload_constraint(&self) -> WorkloadConstraint {
        WorkloadConstraint::Any
    }

    fn matches(&self, fuf: &Fuf, seed: TileId, _profile: &TargetProfile) -> Option<MatchInfo> {
        let node = fuf.get(seed);
        
        // Match both Attention and SlidingAttention
        if node.op != OpKind::Attention && node.op != OpKind::SlidingAttention {
            return None;
        }

        // Singleton claim - just this Attention tile
        Some(MatchInfo {
            claimed_tiles: vec![seed],
            boundary_inputs: vec![],
            boundary_outputs: vec![seed],
        })
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
