// SPDX-License-Identifier: Apache-2.0
//! MoE (Mixture of Experts) layers using `GpuTensor` — no candle dependency.
//!
//! Implements the full Python vLLM fused MoE pipeline:
//! 1. Gate → router logits
//! 2. topk_softmax → topk weights + ids
//! 3. moe_align_block_size → sorted tokens by expert
//! 4. fused_moe_gemm (GEMM 1: gate+up)
//! 5. silu_and_mul activation
//! 6. fused_moe_gemm (GEMM 2: down, with routing weight)
//! 7. moe_sum → reduced output

#[cfg(feature = "nccl")]
use std::sync::Arc;

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::kernels;
use crate::layers::Linear;
#[cfg(feature = "nccl")]
use crate::nccl::NcclGroup;
use crate::tensor::GpuTensor;

/// Block size for MoE GEMM tiling. Must match BLOCK_M in fused_moe_gemm_kernels.cu.
const MOE_BLOCK_SIZE: usize = 128;

// ---------------------------------------------------------------------------
// FusedMoELayer
// ---------------------------------------------------------------------------

/// Fused Mixture of Experts layer.
///
/// Weights are stored as stacked `[num_experts, dim, hidden]` tensors.
/// Forward matches Python vLLM's `fused_experts_impl` exactly.
pub struct FusedMoELayer {
    /// Gate projection: `[hidden_size, num_experts]`.
    pub gate: Linear,
    /// Stacked gate+up weights: `[num_experts, 2*intermediate_size, hidden_size]`.
    pub w1: GpuTensor,
    /// Stacked down weights: `[num_experts, hidden_size, intermediate_size]`.
    pub w2: GpuTensor,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub renormalize: bool,
    /// NCCL group for tensor-parallel all-reduce after MoE output.
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

impl FusedMoELayer {
    /// Forward pass — full MoE pipeline.
    ///
    /// * `hidden_states`: `[num_tokens, hidden_size]`
    ///
    /// Returns: `[num_tokens, hidden_size]`
    ///
    /// # Safety
    /// All tensors must be valid GPU memory. Device must be properly initialized.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(&self, hidden_states: GpuTensor, device: &mut GpuDevice) -> GpuTensor {
        self.forward_owned(hidden_states, device).into_gpu_tensor()
    }

    /// Forward pass returning `OwnedTensor`.
    pub unsafe fn forward_owned(
        &self,
        hidden_states: GpuTensor,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        // 1. Gate: router_logits = hidden_states @ gate_weight^T
        //    router_logits: [num_tokens, num_experts]
        let router_logits =
            self.gate
                .forward_owned(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Top-K softmax: select top_k experts per token
        let (topk_weights, topk_ids) = kernels::topk_softmax(
            router_logits.as_gpu_tensor(),
            self.top_k,
            self.renormalize,
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        // 3. Align block size: sort tokens by expert for fused GEMM
        let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
            topk_ids.as_gpu_tensor(),
            self.num_experts,
            MOE_BLOCK_SIZE,
            &mut device.caching,
            stream,
        );
        drop(topk_ids);

        // 4. GEMM 1: hidden_states × w1^T → [num_tokens * top_k, 2*intermediate]
        //    No routing weight applied yet (apply_weights=false).
        let intermediate1 = kernels::fused_moe_gemm(
            hidden_states,
            self.w1,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens,
            self.top_k,
            MOE_BLOCK_SIZE,
            false, // don't apply weights on first GEMM
            &mut device.caching,
            stream,
        );

        // 5. Activation: SiLU(gate) * up → [num_tokens * top_k, intermediate]
        let activated = kernels::silu_and_mul_fused(
            intermediate1.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(intermediate1);

        // 6. GEMM 2: activated × w2^T → [num_tokens * top_k, hidden_size]
        //    Apply routing weight here (apply_weights=true).
        //    top_k=1 because the input is already [num_tokens * top_k, intermediate],
        //    so sorted_token_ids should index directly (token_id / 1 = token_id).
        //    Matches Python vLLM fused_experts_impl line 1907.
        let intermediate2 = kernels::fused_moe_gemm(
            activated.as_gpu_tensor(),
            self.w2,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens * self.top_k, // M*top_k tokens in expanded input
            1,                       // top_k=1: index directly into expanded input
            MOE_BLOCK_SIZE,
            true, // apply routing weights
            &mut device.caching,
            stream,
        );
        drop(activated);
        drop(topk_weights);
        drop(sorted_token_ids);
        drop(expert_ids);
        drop(num_tokens_post_padded);

        // 7. Reduce: sum across top_k experts → [num_tokens, hidden_size]
        let output = kernels::moe_sum(
            intermediate2.as_gpu_tensor(),
            num_tokens,
            self.hidden_size,
            self.top_k,
            &mut device.caching,
            stream,
        );
        drop(intermediate2);

        // 8. TP all-reduce: combine partial expert results across ranks.
        #[cfg(feature = "nccl")]
        if let Some(ref nccl) = self.tp_group {
            nccl.all_reduce_inplace(output.as_gpu_tensor())
                .expect("MoE all_reduce failed");
        }

        output
    }
}

// ---------------------------------------------------------------------------
// SharedFusedMoELayer (Qwen2/3 MoE)
// ---------------------------------------------------------------------------

/// MoE layer with optional shared expert (used by Qwen2 MoE, Qwen3 MoE).
///
/// The shared expert runs in parallel with the MoE routing:
/// ```text
/// output = moe(hidden_states) + shared_expert_gate(hidden_states).sigmoid() * shared_expert(hidden_states)
/// ```
pub struct SharedFusedMoELayer {
    pub moe: FusedMoELayer,
    /// Shared expert: fused gate+up projection `[2*intermediate, hidden]`.
    pub shared_gate_up: Option<Linear>,
    /// Shared expert: down projection `[hidden, intermediate]`.
    pub shared_down: Option<Linear>,
    /// Shared expert gate: `[1, hidden]` — sigmoid gate for shared expert output.
    pub shared_expert_gate: Option<Linear>,
    pub intermediate_size: usize,
}

impl SharedFusedMoELayer {
    /// Forward pass — MoE + shared expert.
    pub unsafe fn forward(&self, hidden_states: GpuTensor, device: &mut GpuDevice) -> GpuTensor {
        self.forward_owned(hidden_states, device).into_gpu_tensor()
    }

    pub unsafe fn forward_owned(
        &self,
        hidden_states: GpuTensor,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let stream = device.compute_stream;

        // MoE path.
        let moe_out = self.moe.forward_owned(hidden_states, device);

        // Shared expert path (if present).
        if let (Some(shared_gate_up), Some(shared_down), Some(shared_gate)) = (
            &self.shared_gate_up,
            &self.shared_down,
            &self.shared_expert_gate,
        ) {
            // shared_gate_up(hidden_states) → [num_tokens, 2*intermediate]
            let shared_gu = shared_gate_up.forward_owned(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
            );
            // SiLU-and-mul → [num_tokens, intermediate]
            let shared_activated = kernels::silu_and_mul_fused(
                shared_gu.as_gpu_tensor(),
                self.intermediate_size,
                &mut device.caching,
                stream,
            );
            drop(shared_gu);

            // down_proj → [num_tokens, hidden]
            let shared_out = shared_down.forward_owned(
                shared_activated.as_gpu_tensor(),
                &mut device.cublas,
                &mut device.caching,
            );
            drop(shared_activated);

            // Gate: sigmoid(shared_expert_gate(hidden_states)) * shared_out
            let gate_logits =
                shared_gate.forward_owned(hidden_states, &mut device.cublas, &mut device.caching);

            // Fused: out = moe_out + sigmoid(gate_logits) * shared_out
            let result = kernels::sigmoid_mul_add(
                moe_out.as_gpu_tensor(),
                shared_out.as_gpu_tensor(),
                gate_logits.as_gpu_tensor(),
                &mut device.caching,
                stream,
            );
            drop(moe_out);
            drop(shared_out);
            drop(gate_logits);

            result
        } else {
            moe_out
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    #[test]
    fn test_fused_moe_layer_sizes() {
        // Just verify struct construction with dummy tensors.
        let gate_w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[8, 4096], DType::BF16) };
        let w1 = unsafe { GpuTensor::new(0x2000 as *mut u8, &[8, 28672, 4096], DType::BF16) };
        let w2 = unsafe { GpuTensor::new(0x3000 as *mut u8, &[8, 4096, 14336], DType::BF16) };

        let layer = FusedMoELayer {
            gate: Linear::new(gate_w, None),
            w1,
            w2,
            num_experts: 8,
            top_k: 2,
            intermediate_size: 14336,
            hidden_size: 4096,
            renormalize: false,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        assert_eq!(layer.num_experts, 8);
        assert_eq!(layer.top_k, 2);
        assert_eq!(layer.intermediate_size, 14336);
    }
}
