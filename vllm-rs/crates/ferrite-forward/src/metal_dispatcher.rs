// SPDX-License-Identifier: Apache-2.0
//! Metal-specific dispatcher for ferrite-forward.
//!
//! Parallel to the CUDA dispatcher but simplified for Metal's execution model:
//! - No tensor-parallel support (yet)
//! - Direct Metal buffer management instead of GpuWeights abstraction
//! - Single-device execution (Metal doesn't expose multi-GPU like CUDA)

#![cfg(feature = "metal")]

use crate::interpreter::metal::MetalExecutor;
use crate::{CanonicalParams, ForwardCtx, Instruction};
use ferrite_metal_kernels::device::MetalDevice;
use std::sync::Arc;

/// Metal-specific weights trait.
///
/// Simpler than CUDA's FerriteWeights because:
/// - No TP sharding (single device)
/// - Metal buffers managed directly (no GpuWeights abstraction)
/// - Forward pass returns Metal buffer instead of OwnedTensor
pub trait MetalWeights: Send + Sync {
    fn arch_name(&self) -> &'static str;
    fn num_hidden_layers(&self) -> u64;
    fn hidden_size(&self) -> u64;
    fn intermediate_size(&self) -> u64;
    fn num_attention_heads(&self) -> u64;
    fn num_key_value_heads(&self) -> u64;
    fn head_dim(&self) -> u64;
    fn vocab_size(&self) -> u64;

    /// Execute one forward pass on Metal.
    ///
    /// # Arguments
    /// * `ctx` - Forward context with input tensors
    ///
    /// # Returns
    /// Metal buffer containing output logits [num_tokens, vocab_size]
    fn forward(&self, ctx: &ForwardCtx) -> Result<metal::Buffer, String>;

    /// Backbone-only forward (skips lm_head).
    ///
    /// # Returns
    /// Metal buffer containing hidden states [num_tokens, hidden_size]
    fn forward_backbone(&self, ctx: &ForwardCtx) -> Result<metal::Buffer, String>;
}

/// Metal arch registration - parallel to CUDA's FerriteArchRegistration.
///
/// Each #[forward] macro invocation for Metal will emit one of these.
/// For now, manually constructed for testing.
pub struct MetalArchRegistration {
    pub arch_name: &'static str,
    pub hf_arches: &'static [&'static str],
    pub gguf_archs: &'static [&'static str],
    pub try_load: MetalTryLoadFn,
}

/// Try-load function signature for Metal arches.
///
/// Simpler than CUDA version:
/// - No CUstream (Metal manages command queues internally)
/// - No tp_rank/tp_world_size (single device)
/// - Takes arch hint and returns Box<dyn MetalWeights> on success
pub type MetalTryLoadFn = fn(
    arch_hint: &str,
    max_model_len: usize,
) -> anyhow::Result<Option<Box<dyn MetalWeights>>>;

// TODO: Replace with inventory::collect! when #[forward] macro emits registrations
static METAL_REGISTRATIONS: &[MetalArchRegistration] = &[];

/// Top-level Metal loader.
///
/// Walks registered Metal arches and attempts to load the first match.
/// Returns Ok(None) when no arch claims the hint (caller falls back to CPU/CUDA).
///
/// # Arguments
/// * `arch_hint` - HF architecture string (e.g., "LlamaForCausalLM")
/// * `max_model_len` - Maximum sequence length
///
/// # Returns
/// * `Ok(Some(weights))` - Successfully loaded Metal weights
/// * `Ok(None)` - No Metal arch registered for this hint
/// * `Err(_)` - Load failure (I/O, shape mismatch, etc.)
pub fn try_load_metal(
    arch_hint: &str,
    max_model_len: usize,
) -> anyhow::Result<Option<Box<dyn MetalWeights>>> {
    for reg in METAL_REGISTRATIONS {
        if reg.hf_arches.contains(&arch_hint) || reg.gguf_archs.contains(&arch_hint) {
            match (reg.try_load)(arch_hint, max_model_len) {
                Ok(Some(weights)) => return Ok(Some(weights)),
                Ok(None) => continue, // This arch rejected, try next
                Err(e) => return Err(e), // Hard failure
            }
        }
    }
    Ok(None) // No arch claimed this hint
}

/// Wrapper that implements MetalWeights using MetalExecutor.
///
/// Generic over CanonicalParams so it can wrap any arch's weights.
/// The #[forward] macro will emit one of these per arch.
pub struct MetalWeightsWrapper<W: CanonicalParams> {
    executor: MetalExecutor<W>,
    arch_name: &'static str,
}

impl<W: CanonicalParams> MetalWeightsWrapper<W> {
    /// Create a new wrapper from pre-compiled instructions and weights.
    ///
    /// # Arguments
    /// * `instructions` - Instruction tape from solver
    /// * `weights` - Canonical weights (Metal buffers)
    /// * `num_slots` - Number of tile slots
    /// * `arch_name` - Architecture identifier for logging
    pub fn new(
        _instructions: &[Instruction<W>],
        _weights: W,
        num_slots: usize,
        arch_name: &'static str,
    ) -> Result<Self, String> {
        // Phase 5.6: Create MetalExecutor with simplified API
        let device = ferrite_metal_kernels::detect_device()
            .ok_or("No Metal device found")?;
        let device_arc = std::sync::Arc::new(device);
        
        // Empty instruction tape for Phase 5.6
        let instruction_tape = Vec::new();
        
        let executor = MetalExecutor::new(
            device_arc,
            instruction_tape,
            num_slots,
            0, // num_weight_slots (placeholder)
        )?;
        
        Ok(Self { executor, arch_name })
    }
}

impl<W: CanonicalParams> MetalWeights for MetalWeightsWrapper<W> {
    fn arch_name(&self) -> &'static str {
        self.arch_name
    }

    fn num_hidden_layers(&self) -> u64 {
        // CanonicalParams doesn't expose NUM_LAYERS
        // Return 0 for now - will be populated from actual weights
        0
    }

    fn hidden_size(&self) -> u64 {
        W::Q_SIZE as u64
    }

    fn intermediate_size(&self) -> u64 {
        W::INTERMEDIATE_SIZE as u64
    }

    fn num_attention_heads(&self) -> u64 {
        W::NUM_Q_HEADS as u64
    }

    fn num_key_value_heads(&self) -> u64 {
        W::NUM_KV_HEADS as u64
    }

    fn head_dim(&self) -> u64 {
        W::HEAD_DIM as u64
    }

    fn vocab_size(&self) -> u64 {
        // CanonicalParams doesn't expose VOCAB_SIZE
        // Return 0 for now - will be populated from actual weights
        0
    }

    fn forward(&self, ctx: &ForwardCtx) -> Result<metal::Buffer, String> {
        self.executor.forward(ctx)
    }

    fn forward_backbone(&self, ctx: &ForwardCtx) -> Result<metal::Buffer, String> {
        // TODO: Implement backbone-only path
        // For now, just call forward and return the same buffer
        self.executor.forward(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_load_metal_no_registrations() {
        // With no registrations, should return Ok(None)
        let result = try_load_metal("LlamaForCausalLM", 2048);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    #[ignore] // Requires Metal device and actual weights
    fn test_metal_weights_wrapper() {
        // TODO: Add test with synthetic weights and instruction tape
    }
}
