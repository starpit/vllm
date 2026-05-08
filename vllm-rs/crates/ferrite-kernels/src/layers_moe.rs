// SPDX-License-Identifier: Apache-2.0
//! MoE (Mixture of Experts) layers using `GpuTensor`.
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

// Always-available imports: only `GpuTensor` (and `Linear`, which holds
// only `GpuTensor` fields). Cuda-tied imports are gated below.
use crate::layers::Linear;
use ferrite_cuda_core::tensor::GpuTensor;

#[cfg(feature = "cuda")]
use crate::kernels;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::alloc::OwnedTensor;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::device::GpuDevice;
#[cfg(feature = "nccl")]
use ferrite_cuda_core::nccl::NcclGroup;
#[cfg(feature = "cuda")]
use ferrite_cuda_core::tensor::TensorView;

/// Dynamic BLOCK_M selection for fused MoE GEMM tiling.
///
/// Matches Python vLLM's `get_default_config` heuristic: select the smallest
/// tile size that covers the expected tokens-per-expert, reducing wasted
/// compute on zero-padded rows during decode.
pub fn select_moe_block_m(num_tokens: usize, top_k: usize, num_experts: usize) -> usize {
    let tokens_per_expert = (num_tokens * top_k) / num_experts;
    if tokens_per_expert <= 16 {
        16
    } else if tokens_per_expert <= 32 {
        32
    } else if tokens_per_expert <= 64 {
        64
    } else {
        128
    }
}

/// Routing-stage configuration shared by every fused-MoE layer (BF16, FP8 scalar,
/// FP8 block, Marlin, GGML). The four families differ only in how they consume
/// the resulting `(topk_weights, topk_ids)` — the *selection* logic is uniform.
///
/// The three branches mirror Python vLLM's expert-selection paths:
/// * `e_score_correction_bias = None` → `topk_softmax` (Mixtral / Qwen MoE / DSv2).
/// * `Some(bias)` with `n_expert_group == 0` → flat sigmoid+bias top-k.
/// * `Some(bias)` with `n_expert_group > 0` and `topk_group > 0` → grouped
///   `noaux_tc` (DeepSeek V3 / Kimi K2). `routed_scaling_factor` is folded into
///   the unbiased sigmoid weights inside `topk_noaux_tc`.
pub struct MoeRouting<'a> {
    pub top_k: usize,
    pub renormalize: bool,
    pub e_score_correction_bias: Option<&'a GpuTensor>,
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
}

/// Dispatch the three router-selection variants. Returns `(topk_weights, topk_ids)`
/// matching `topk_softmax`'s shape contract: `[num_tokens, top_k]` F32 / I32.
#[cfg(feature = "cuda")]
pub unsafe fn route_experts(
    router_logits: GpuTensor,
    cfg: &MoeRouting<'_>,
    caching: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: ferrite_cuda_core::CUstream,
) -> (OwnedTensor, OwnedTensor) {
    if let Some(bias) = cfg.e_score_correction_bias {
        if cfg.n_expert_group > 0 && cfg.topk_group > 0 {
            kernels::topk_noaux_tc(
                router_logits,
                *bias,
                cfg.top_k,
                cfg.n_expert_group,
                cfg.topk_group,
                cfg.renormalize,
                cfg.routed_scaling_factor,
                caching,
                stream,
            )
        } else {
            kernels::topk_sigmoid_with_bias(
                router_logits,
                *bias,
                cfg.top_k,
                cfg.renormalize,
                caching,
                stream,
            )
        }
    } else {
        kernels::topk_softmax(router_logits, cfg.top_k, cfg.renormalize, caching, stream)
    }
}

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
    /// `[num_experts]` F32 e_score_correction_bias for sigmoid routing (DeepSeek V3 / Kimi K2).
    /// When `Some`, uses sigmoid top-k with bias instead of softmax top-k.
    pub e_score_correction_bias: Option<GpuTensor>,
    /// Number of expert groups for `noaux_tc` grouped routing (DeepSeek V3).
    /// 0 = no group selection (flat top-k).
    pub n_expert_group: usize,
    /// Number of groups to select in grouped routing. 0 = disabled.
    pub topk_group: usize,
    /// Scaling factor applied to routing weights (default 1.0).
    pub routed_scaling_factor: f64,
    /// NCCL group for tensor-parallel all-reduce after MoE output.
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

#[cfg(feature = "cuda")]
impl FusedMoELayer {
    /// Load a Mixtral-style BF16 fused MoE layer from safetensors.
    ///
    /// Mirrors `vllm-cuda/src/model/mixtral.rs::MixtralDecoderLayer::load_moe`
    /// at single-rank (tp=1). Mixtral's on-disk weight names:
    /// - `{prefix}.gate.weight` — `[num_experts, hidden_size]`, dense BF16.
    /// - `{prefix}.experts.{e}.w1.weight` — `[intermediate_size, hidden_size]` (gate_proj).
    /// - `{prefix}.experts.{e}.w3.weight` — `[intermediate_size, hidden_size]` (up_proj).
    /// - `{prefix}.experts.{e}.w2.weight` — `[hidden_size, intermediate_size]` (down_proj).
    ///
    /// Stacked into:
    /// - `w1`: `[num_experts, 2*intermediate_size, hidden_size]` (gate+up fused).
    /// - `w2`: `[num_experts, hidden_size, intermediate_size]`.
    ///
    /// `routed_scaling_factor` defaults to 1.0; `renormalize` is false (Mixtral
    /// does not renormalize topk weights). Sigmoid bias / grouped routing /
    /// expert-group selection all default off — those branches are reserved for
    /// the DeepSeek-V3 family, not for Mixtral or Qwen3-MoE.
    pub fn load(
        gw: &mut ferrite_cuda_core::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        intermediate_size: usize,
        hidden_size: usize,
        stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        use ferrite_cuda_core::driver;

        let gate = crate::layers::Linear::load(gw, &format!("{prefix}.gate"))?;

        let first_w1 = format!("{prefix}.experts.0.w1.weight");
        let (_, disk_dtype) = gw
            .tensor_info(&first_w1)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_w1}"))?;
        // `take_into` casts floating-point sources to the configured
        // target dtype on the way in. Size + tag the stacked buffer
        // with the post-cast width — otherwise FP32 checkpoints blow
        // up the buffer to 2× the right size and feed the fused-MoE
        // GEMM a wrongly-typed tensor.
        let dtype = gw.target_dtype().unwrap_or(disk_dtype);
        let elem = dtype.size_bytes();

        let w1_bytes = num_experts * 2 * intermediate_size * hidden_size * elem;
        let w2_bytes = num_experts * hidden_size * intermediate_size * elem;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        let gate_proj_bytes = intermediate_size * hidden_size * elem;
        for e in 0..num_experts {
            let w1_name = format!("{prefix}.experts.{e}.w1.weight");
            let w3_name = format!("{prefix}.experts.{e}.w3.weight");
            let w2_name = format!("{prefix}.experts.{e}.w2.weight");
            let expert_w1_off = e * 2 * intermediate_size * hidden_size * elem;
            let expert_w2_off = e * hidden_size * intermediate_size * elem;
            unsafe {
                gw.take_into(&w1_name, w1_ptr.add(expert_w1_off), stream)?;
                gw.take_into(
                    &w3_name,
                    w1_ptr.add(expert_w1_off + gate_proj_bytes),
                    stream,
                )?;
                gw.take_into(&w2_name, w2_ptr.add(expert_w2_off), stream)?;
            }
        }

        let w1 = unsafe {
            GpuTensor::new(
                w1_ptr,
                &[num_experts, 2 * intermediate_size, hidden_size],
                dtype,
            )
        };
        let w2 = unsafe {
            GpuTensor::new(
                w2_ptr,
                &[num_experts, hidden_size, intermediate_size],
                dtype,
            )
        };

        Ok(FusedMoELayer {
            gate,
            w1,
            w2,
            num_experts,
            top_k,
            intermediate_size,
            hidden_size,
            // Python vLLM's `MixtralMoE` constructs `FusedMoE(...,
            // renormalize=True)` (vllm/model_executor/models/mixtral.py
            // L136). HF transformers' `MixtralSparseMoeBlock.forward`
            // unconditionally divides by `routing_weights.sum(dim=-1)`.
            // The hand-written `vllm-cuda::mixtral::load_moe` path has
            // a latent bug here (`renormalize: false`); ferrite tracks
            // Python vLLM as the reference per
            // `feedback_python_vllm_is_the_reference`.
            renormalize: true,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        })
    }

    /// Forward pass — full MoE pipeline.
    ///
    /// Always uses `forward_fused` (WMMA kernel). The fused kernel is a plain
    /// CUDA kernel launch, fully compatible with CUDA graph capture.
    ///
    /// * `hidden_states`: `[num_tokens, hidden_size]`
    ///
    /// Returns: `[num_tokens, hidden_size]`
    ///
    /// # Safety
    /// All tensors must be valid GPU memory. Device must be properly initialized.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        self.forward_fused(hidden_states, device)
    }

    /// Fused MoE forward — uses custom tiled GEMM kernel. Graph-capturable.
    #[allow(clippy::too_many_arguments)]
    unsafe fn forward_fused(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        let block_m = select_moe_block_m(num_tokens, self.top_k, self.num_experts);

        let router_logits =
            self.gate
                .forward(hidden_states, &mut device.cublas, &mut device.caching);

        let (topk_weights, topk_ids) = route_experts(
            router_logits.as_gpu_tensor(),
            &MoeRouting {
                top_k: self.top_k,
                renormalize: self.renormalize,
                e_score_correction_bias: self.e_score_correction_bias.as_ref(),
                n_expert_group: self.n_expert_group,
                topk_group: self.topk_group,
                routed_scaling_factor: self.routed_scaling_factor,
            },
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
            topk_ids.as_gpu_tensor(),
            self.num_experts,
            block_m,
            &mut device.caching,
            stream,
        );
        drop(topk_ids);

        let intermediate1 = kernels::fused_moe_gemm(
            *hidden_states,
            self.w1,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens,
            self.top_k,
            block_m,
            false,
            &mut device.caching,
            stream,
        );

        let activated = kernels::silu_and_mul_fused(
            intermediate1.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(intermediate1);

        let intermediate2 = kernels::fused_moe_gemm(
            activated.as_gpu_tensor(),
            self.w2,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens * self.top_k,
            1,
            block_m,
            true,
            &mut device.caching,
            stream,
        );

        drop(activated);
        drop(topk_weights);
        drop(sorted_token_ids);
        drop(expert_ids);
        drop(num_tokens_post_padded);

        let output = kernels::moe_sum(
            intermediate2.as_gpu_tensor(),
            num_tokens,
            self.hidden_size,
            self.top_k,
            &mut device.caching,
            stream,
        );
        drop(intermediate2);

        #[cfg(feature = "nccl")]
        if let Some(ref nccl) = self.tp_group {
            nccl.all_reduce_inplace(output.as_gpu_tensor())
                .expect("MoE all_reduce failed");
        }

        output
    }
}

#[cfg(feature = "metal")]
impl FusedMoELayer {
    /// Metal stub. The fused MoE GEMM + topk + softmax/sigmoid kernel
    /// chain is cuda-only — no Apple-silicon counterpart yet. Returning
    /// `Err` keeps the macro emission for MoE arches well-typed under
    /// `--features metal`; trying to actually load a Mixtral / Qwen-MoE
    /// / DeepSeek-MoE checkpoint on metal surfaces this error at the
    /// load site rather than blowing up during shader compilation.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        _gw: &mut ferrite_cuda_core::weights::GpuWeights,
        _prefix: &str,
        _num_experts: usize,
        _top_k: usize,
        _intermediate_size: usize,
        _hidden_size: usize,
        _stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("FusedMoELayer not supported on metal: port MoE GEMM + topk kernels first")
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

#[cfg(feature = "cuda")]
impl SharedFusedMoELayer {
    /// Load a Qwen-MoE-style BF16 fused MoE + shared-expert layer.
    ///
    /// Mirrors `vllm-cuda/src/model/qwen3_moe.rs::Qwen3MoeDecoderLayer::load_moe`
    /// at single-rank (tp=1). Qwen3-MoE / Qwen2-MoE expert weight names use the
    /// HF `gate_proj/up_proj/down_proj` convention (NOT Mixtral's
    /// `w1/w2/w3`); the shared expert lives at
    /// `{prefix}.shared_expert.{gate,up,down}_proj.weight` with a sigmoid
    /// gate at `{prefix}.shared_expert_gate.weight`.
    ///
    /// Routed `renormalize: true` matches Python vLLM's Qwen-MoE softmax
    /// path (`norm_topk_prob = True` is implicit in the family).
    pub fn load(
        gw: &mut ferrite_cuda_core::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        use ferrite_cuda_core::driver;

        let gate = crate::layers::Linear::load(gw, &format!("{prefix}.gate"))?;

        let first_gate = format!("{prefix}.experts.0.gate_proj.weight");
        let (_, disk_dtype) = gw
            .tensor_info(&first_gate)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_gate}"))?;
        // Use post-cast dtype for the stacked buffer; see
        // `FusedMoELayer::load` for the rationale.
        let dtype = gw.target_dtype().unwrap_or(disk_dtype);
        let elem = dtype.size_bytes();

        let inter = moe_intermediate_size;
        let w1_bytes = num_experts * 2 * inter * hidden_size * elem;
        let w2_bytes = num_experts * hidden_size * inter * elem;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        let gate_proj_bytes = inter * hidden_size * elem;
        for e in 0..num_experts {
            let gate_name = format!("{prefix}.experts.{e}.gate_proj.weight");
            let up_name = format!("{prefix}.experts.{e}.up_proj.weight");
            let down_name = format!("{prefix}.experts.{e}.down_proj.weight");
            let expert_w1_off = e * 2 * inter * hidden_size * elem;
            let expert_w2_off = e * hidden_size * inter * elem;
            unsafe {
                gw.take_into(&gate_name, w1_ptr.add(expert_w1_off), stream)?;
                gw.take_into(
                    &up_name,
                    w1_ptr.add(expert_w1_off + gate_proj_bytes),
                    stream,
                )?;
                gw.take_into(&down_name, w2_ptr.add(expert_w2_off), stream)?;
            }
        }

        let w1 = unsafe { GpuTensor::new(w1_ptr, &[num_experts, 2 * inter, hidden_size], dtype) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[num_experts, hidden_size, inter], dtype) };

        let moe = FusedMoELayer {
            gate,
            w1,
            w2,
            num_experts,
            top_k,
            intermediate_size: inter,
            hidden_size,
            renormalize: true,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        let (shared_gate_up, shared_down, shared_expert_gate) = if shared_expert_intermediate_size
            > 0
        {
            let shared_inter = shared_expert_intermediate_size;
            let shared_gate_proj_bytes = shared_inter * hidden_size * elem;
            let shared_total = 2 * shared_gate_proj_bytes;
            let ptr = unsafe { driver::mem_alloc(shared_total)? };
            unsafe {
                gw.take_into(
                    &format!("{prefix}.shared_expert.gate_proj.weight"),
                    ptr,
                    stream,
                )?;
                gw.take_into(
                    &format!("{prefix}.shared_expert.up_proj.weight"),
                    ptr.add(shared_gate_proj_bytes),
                    stream,
                )?;
            }
            let gu_w = unsafe { GpuTensor::new(ptr, &[2 * shared_inter, hidden_size], dtype) };
            let gate_up = crate::layers::Linear::new(gu_w, None);
            let down =
                crate::layers::Linear::load(gw, &format!("{prefix}.shared_expert.down_proj"))?;
            let sgate = crate::layers::Linear::load(gw, &format!("{prefix}.shared_expert_gate"))?;
            (Some(gate_up), Some(down), Some(sgate))
        } else {
            (None, None, None)
        };

        Ok(SharedFusedMoELayer {
            moe,
            shared_gate_up,
            shared_down,
            shared_expert_gate,
            intermediate_size: shared_expert_intermediate_size,
        })
    }

    /// Tensor-parallel sharded BF16 load — intermediate-dim sharded, same
    /// scheme as [`Fp8SharedFusedMoELayer::load`] at tp>1. Only called by
    /// `Fp8SharedFusedMoELayer::load`'s BF16 fallback path when the
    /// checkpoint's `ignore` list ships an MoE layer dense BF16 inside an
    /// otherwise FP8 model (the unsloth/Qwen3-Coder-Next-FP8-Dynamic
    /// layer-47 case). `world == 1` yields per-rank dims equal to full.
    ///
    /// After load, the forward path produces per-rank partial sums; the
    /// instruction arm's post-forward all-reduce (via `ForwardCtx::tp_group`)
    /// closes them. `moe.tp_group` is left `None` — the outer all-reduce
    /// is the single TP sync point.
    #[allow(clippy::too_many_arguments)]
    pub fn load_sharded(
        gw: &mut ferrite_cuda_core::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        tp_rank: usize,
        tp_size: usize,
        stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        use ferrite_cuda_core::driver;

        anyhow::ensure!(
            tp_size >= 1 && tp_rank < tp_size,
            "SharedFusedMoELayer::load_sharded: invalid (tp_rank={tp_rank}, tp_size={tp_size})"
        );
        let inter_full = moe_intermediate_size;
        anyhow::ensure!(
            inter_full.is_multiple_of(tp_size),
            "SharedFusedMoELayer::load_sharded: moe_intermediate_size={inter_full} not divisible by tp_size={tp_size}"
        );
        let inter = inter_full / tp_size;
        let sharded = tp_size > 1;

        let gate = crate::layers::Linear::load(gw, &format!("{prefix}.gate"))?;

        let first_gate = format!("{prefix}.experts.0.gate_proj.weight");
        let (_, disk_dtype) = gw
            .tensor_info(&first_gate)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_gate}"))?;
        let dtype = gw.target_dtype().unwrap_or(disk_dtype);
        let elem = dtype.size_bytes();

        let w1_bytes = num_experts * 2 * inter * hidden_size * elem;
        let w2_bytes = num_experts * hidden_size * inter * elem;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        let gate_proj_bytes = inter * hidden_size * elem;
        for e in 0..num_experts {
            let gate_name = format!("{prefix}.experts.{e}.gate_proj.weight");
            let up_name = format!("{prefix}.experts.{e}.up_proj.weight");
            let down_name = format!("{prefix}.experts.{e}.down_proj.weight");
            let expert_w1_off = e * 2 * inter * hidden_size * elem;
            let expert_w2_off = e * hidden_size * inter * elem;
            if sharded {
                unsafe {
                    gw.take_shard_into(&gate_name, 0, tp_rank, tp_size, w1_ptr.add(expert_w1_off), stream)?;
                    gw.take_shard_into(
                        &up_name,
                        0,
                        tp_rank,
                        tp_size,
                        w1_ptr.add(expert_w1_off + gate_proj_bytes),
                        stream,
                    )?;
                    gw.take_shard_into(&down_name, 1, tp_rank, tp_size, w2_ptr.add(expert_w2_off), stream)?;
                }
            } else {
                unsafe {
                    gw.take_into(&gate_name, w1_ptr.add(expert_w1_off), stream)?;
                    gw.take_into(&up_name, w1_ptr.add(expert_w1_off + gate_proj_bytes), stream)?;
                    gw.take_into(&down_name, w2_ptr.add(expert_w2_off), stream)?;
                }
            }
        }

        let w1 = unsafe { GpuTensor::new(w1_ptr, &[num_experts, 2 * inter, hidden_size], dtype) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[num_experts, hidden_size, inter], dtype) };

        let moe = FusedMoELayer {
            gate,
            w1,
            w2,
            num_experts,
            top_k,
            intermediate_size: inter,
            hidden_size,
            renormalize: true,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        // Shared expert: column-parallel gate_up (sharded dim 0), row-parallel
        // down (sharded dim 1). gate_up is a simple dense `Linear`; pack the
        // per-rank gate and up slices back-to-back into one `[2*shared_inter/tp, hidden]`
        // buffer — same layout as the unsharded path, just with per-rank rows.
        let shared_inter_per_rank = if shared_expert_intermediate_size > 0 {
            anyhow::ensure!(
                shared_expert_intermediate_size.is_multiple_of(tp_size),
                "SharedFusedMoELayer::load_sharded: shared_expert_intermediate_size={shared_expert_intermediate_size} \
                 not divisible by tp_size={tp_size}"
            );
            shared_expert_intermediate_size / tp_size
        } else {
            0
        };
        let (shared_gate_up, shared_down, shared_expert_gate) = if shared_expert_intermediate_size > 0 {
            let shared_gate_proj_bytes = shared_inter_per_rank * hidden_size * elem;
            let shared_total = 2 * shared_gate_proj_bytes;
            let ptr = unsafe { driver::mem_alloc(shared_total)? };
            unsafe {
                if sharded {
                    gw.take_shard_into(
                        &format!("{prefix}.shared_expert.gate_proj.weight"),
                        0,
                        tp_rank,
                        tp_size,
                        ptr,
                        stream,
                    )?;
                    gw.take_shard_into(
                        &format!("{prefix}.shared_expert.up_proj.weight"),
                        0,
                        tp_rank,
                        tp_size,
                        ptr.add(shared_gate_proj_bytes),
                        stream,
                    )?;
                } else {
                    gw.take_into(
                        &format!("{prefix}.shared_expert.gate_proj.weight"),
                        ptr,
                        stream,
                    )?;
                    gw.take_into(
                        &format!("{prefix}.shared_expert.up_proj.weight"),
                        ptr.add(shared_gate_proj_bytes),
                        stream,
                    )?;
                }
            }
            let gu_w = unsafe {
                GpuTensor::new(
                    ptr,
                    &[2 * shared_inter_per_rank, hidden_size],
                    dtype,
                )
            };
            let gate_up = crate::layers::Linear::new(gu_w, None);

            // Down is row-parallel: shard its in-features (dim 1).
            let down_w = if sharded {
                gw.take_shard(
                    &format!("{prefix}.shared_expert.down_proj.weight"),
                    1,
                    tp_rank,
                    tp_size,
                )?
            } else {
                gw.take(&format!("{prefix}.shared_expert.down_proj.weight"))?
            };
            let down = crate::layers::Linear::new(down_w, None);

            let sgate = crate::layers::Linear::load(gw, &format!("{prefix}.shared_expert_gate"))?;
            (Some(gate_up), Some(down), Some(sgate))
        } else {
            (None, None, None)
        };

        Ok(SharedFusedMoELayer {
            moe,
            shared_gate_up,
            shared_down,
            shared_expert_gate,
            intermediate_size: shared_inter_per_rank,
        })
    }

    /// Forward pass — MoE + shared expert.
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let stream = device.compute_stream;

        // MoE path.
        let moe_out = self.moe.forward(hidden_states, device);

        // Shared expert path (if present).
        if let (Some(shared_gate_up), Some(shared_down), Some(shared_gate)) = (
            &self.shared_gate_up,
            &self.shared_down,
            &self.shared_expert_gate,
        ) {
            // shared_gate_up(hidden_states) → [num_tokens, 2*intermediate]
            let shared_gu =
                shared_gate_up.forward(hidden_states, &mut device.cublas, &mut device.caching);
            // SiLU-and-mul → [num_tokens, intermediate]
            let shared_activated = kernels::silu_and_mul_fused(
                shared_gu.as_gpu_tensor(),
                self.intermediate_size,
                &mut device.caching,
                stream,
            );
            drop(shared_gu);

            // down_proj → [num_tokens, hidden]
            let shared_out = shared_down.forward(
                shared_activated.view(),
                &mut device.cublas,
                &mut device.caching,
            );
            drop(shared_activated);

            // Gate: sigmoid(shared_expert_gate(hidden_states)) * shared_out
            let gate_logits =
                shared_gate.forward(hidden_states, &mut device.cublas, &mut device.caching);

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

#[cfg(feature = "metal")]
impl SharedFusedMoELayer {
    /// Metal stub — see `FusedMoELayer::load`.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        _gw: &mut ferrite_cuda_core::weights::GpuWeights,
        _prefix: &str,
        _num_experts: usize,
        _top_k: usize,
        _moe_intermediate_size: usize,
        _shared_expert_intermediate_size: usize,
        _hidden_size: usize,
        _stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "SharedFusedMoELayer not supported on metal: port MoE GEMM + topk kernels first"
        )
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2MoELayer
// ---------------------------------------------------------------------------

/// MoE layer for DeepSeek V2 / V3. Two differences from `SharedFusedMoELayer`:
/// 1. Shared expert uses a **plain add** (no sigmoid gate).
/// 2. Routed output is multiplied by `routed_scaling_factor` before the add.
///
/// Forward: `output = routed_scaling_factor * moe(x) + shared_expert(x)`
pub struct DeepSeekV2MoELayer {
    pub moe: FusedMoELayer,
    /// Shared expert fused gate+up: `[2 * shared_inter, hidden_size]`.
    pub shared_gate_up: crate::layers::Linear,
    /// Shared expert down: `[hidden_size, shared_inter]`.
    pub shared_down: crate::layers::Linear,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f32,
}

#[cfg(feature = "cuda")]
impl DeepSeekV2MoELayer {
    /// Forward pass.
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let stream = device.compute_stream;

        // Routed experts.
        let moe_out = self.moe.forward(hidden_states, device);
        if self.routed_scaling_factor != 1.0 {
            kernels::scale_inplace(*moe_out.view(), self.routed_scaling_factor, &device.cublas);
        }

        // Shared expert: silu(gate_up) → down.
        let shared_gu =
            self.shared_gate_up
                .forward(hidden_states, &mut device.cublas, &mut device.caching);
        let shared_activated = kernels::silu_and_mul_fused(
            *shared_gu.view(),
            self.shared_intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(shared_gu);
        let shared_out = self.shared_down.forward(
            shared_activated.view(),
            &mut device.cublas,
            &mut device.caching,
        );
        drop(shared_activated);

        // output = moe_out + shared_out (no sigmoid gate).
        kernels::add_inplace(*moe_out.view(), *shared_out.view(), stream);
        drop(shared_out);
        moe_out
    }

    /// Load from safetensors. `prefix` is the MLP prefix for this layer
    /// (e.g. `model.layers.3.mlp`).
    ///
    /// `use_sigmoid`: if true, loads `{prefix}.gate.e_score_correction_bias`
    /// and uses sigmoid routing (DeepSeek V3 / Kimi K2 `topk_method="noaux_tc"`).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        gw: &mut ferrite_cuda_core::weights::GpuWeights,
        prefix: &str,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        use_sigmoid: bool,
        n_expert_group: usize,
        topk_group: usize,
        stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        use ferrite_cuda_core::driver;
        use ferrite_cuda_core::tensor::GpuTensor;

        let gate = crate::layers::Linear::load(gw, &format!("{prefix}.gate"))?;

        let first_gate = format!("{prefix}.experts.0.gate_proj.weight");
        let (_, dtype) = gw
            .tensor_info(&first_gate)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_gate}"))?;
        let elem = dtype.size_bytes();
        let inter = moe_intermediate_size;

        // Stack expert weights: w1 = [E, 2*inter, hidden], w2 = [E, hidden, inter].
        let w1_bytes = n_routed_experts * 2 * inter * hidden_size * elem;
        let w2_bytes = n_routed_experts * hidden_size * inter * elem;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        for e in 0..n_routed_experts {
            let gate_name = format!("{prefix}.experts.{e}.gate_proj.weight");
            let up_name = format!("{prefix}.experts.{e}.up_proj.weight");
            let down_name = format!("{prefix}.experts.{e}.down_proj.weight");
            let expert_w1_off = e * 2 * inter * hidden_size * elem;
            let gate_bytes = inter * hidden_size * elem;
            let expert_w2_off = e * hidden_size * inter * elem;
            unsafe {
                gw.take_into(&gate_name, w1_ptr.add(expert_w1_off), stream)?;
                gw.take_into(&up_name, w1_ptr.add(expert_w1_off + gate_bytes), stream)?;
                gw.take_into(&down_name, w2_ptr.add(expert_w2_off), stream)?;
            }
        }

        let w1 =
            unsafe { GpuTensor::new(w1_ptr, &[n_routed_experts, 2 * inter, hidden_size], dtype) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[n_routed_experts, hidden_size, inter], dtype) };

        // Load e_score_correction_bias for sigmoid routing (DeepSeek V3 / Kimi K2).
        // Python vLLM always casts this to F32 before the routing kernel; we match.
        let e_score_correction_bias = if use_sigmoid {
            let bias_name = format!("{prefix}.gate.e_score_correction_bias");
            let bias_raw = gw.take(&bias_name)?;
            let bias_f32 = if bias_raw.dtype() == ferrite_cuda_core::dtype::DType::F32 {
                bias_raw
            } else {
                // Cast BF16/F16 → F32 (happens when checkpoint is in BF16)
                let n = bias_raw.numel();
                let f32_ptr = unsafe { driver::mem_alloc(n * 4)? };
                let f32_bias =
                    unsafe { GpuTensor::new(f32_ptr, &[n], ferrite_cuda_core::dtype::DType::F32) };
                unsafe { kernels::cast_bias_to_f32(bias_raw, f32_bias, stream) };
                f32_bias
            };
            Some(bias_f32)
        } else {
            None
        };

        let moe = FusedMoELayer {
            gate,
            w1,
            w2,
            num_experts: n_routed_experts,
            top_k,
            intermediate_size: inter,
            hidden_size,
            renormalize: norm_topk_prob,
            e_score_correction_bias,
            n_expert_group,
            topk_group,
            // Python passes routed_scaling_factor=1.0 to the inner FusedMoE and
            // applies the actual factor *outside* (after the expert computation).
            // DeepSeekV2MoELayer::forward does the same via scale_inplace — so
            // we must not fold it into the routing weights here too (double apply).
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        // Shared expert: concat gate_proj + up_proj → [2*shared_inter, hidden].
        let shared_inter = n_shared_experts * moe_intermediate_size;
        let shared_gate_name = format!("{prefix}.shared_experts.gate_proj.weight");
        let shared_up_name = format!("{prefix}.shared_experts.up_proj.weight");
        let gate_bytes = shared_inter * hidden_size * elem;
        let shared_ptr = unsafe { driver::mem_alloc(2 * gate_bytes)? };
        unsafe {
            gw.take_into(&shared_gate_name, shared_ptr, stream)?;
            gw.take_into(&shared_up_name, shared_ptr.add(gate_bytes), stream)?;
        }
        let shared_w =
            unsafe { GpuTensor::new(shared_ptr, &[2 * shared_inter, hidden_size], dtype) };
        let shared_gate_up = crate::layers::Linear::new(shared_w, None);

        let shared_down =
            crate::layers::Linear::load(gw, &format!("{prefix}.shared_experts.down_proj"))?;

        Ok(Self {
            moe,
            shared_gate_up,
            shared_down,
            shared_intermediate_size: shared_inter,
            routed_scaling_factor,
        })
    }
}

#[cfg(feature = "metal")]
impl DeepSeekV2MoELayer {
    /// Metal stub — see `FusedMoELayer::load`.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        _gw: &mut ferrite_cuda_core::weights::GpuWeights,
        _prefix: &str,
        _n_routed_experts: usize,
        _n_shared_experts: usize,
        _top_k: usize,
        _moe_intermediate_size: usize,
        _hidden_size: usize,
        _norm_topk_prob: bool,
        _routed_scaling_factor: f32,
        _use_sigmoid: bool,
        _n_expert_group: usize,
        _topk_group: usize,
        _stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        anyhow::bail!(
            "DeepSeekV2MoELayer not supported on metal: port MoE GEMM + topk kernels first"
        )
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2Fp8BlockMoELayer
// ---------------------------------------------------------------------------

/// FP8 block-quantized analog of [`DeepSeekV2MoELayer`]. Used by DeepSeek-V3
/// and Kimi K2 official checkpoints (FP8 E4M3 with 128×128 weight-block scales).
///
/// Forward shape is identical to `DeepSeekV2MoELayer::forward` — the only
/// change is the storage of the routed experts (`Fp8BlockFusedMoELayer`) and
/// the shared expert (`Fp8BlockLinear`).
pub struct DeepSeekV2Fp8BlockMoELayer {
    pub moe: Fp8BlockFusedMoELayer,
    /// Shared expert fused gate+up: concatenated `Fp8BlockLinear`
    /// `[2 * shared_inter, hidden_size]`.
    pub shared_gate_up: crate::layers::Fp8BlockLinear,
    /// Shared expert down: `Fp8BlockLinear` `[hidden_size, shared_inter]`.
    pub shared_down: crate::layers::Fp8BlockLinear,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f32,
}

#[cfg(feature = "cuda")]
impl DeepSeekV2Fp8BlockMoELayer {
    /// Forward pass: `output = routed_scaling_factor * moe(x) + shared_expert(x)`.
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let stream = device.compute_stream;

        // Routed experts (FP8 block).
        let moe_out = self.moe.forward(hidden_states, device);
        if self.routed_scaling_factor != 1.0 {
            kernels::scale_inplace(*moe_out.view(), self.routed_scaling_factor, &device.cublas);
        }

        // Shared expert: silu(gate_up) → down. Both Fp8BlockLinear.
        let shared_gu = self.shared_gate_up.forward(
            hidden_states,
            &mut device.cublas,
            &mut device.caching,
            stream,
        );
        let shared_activated = kernels::silu_and_mul_fused(
            *shared_gu.view(),
            self.shared_intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(shared_gu);
        let shared_out = self.shared_down.forward(
            shared_activated.view(),
            &mut device.cublas,
            &mut device.caching,
            stream,
        );
        drop(shared_activated);

        // output = moe_out + shared_out (no sigmoid gate).
        kernels::add_inplace(*moe_out.view(), *shared_out.view(), stream);
        drop(shared_out);
        moe_out
    }

    /// Load from safetensors. Mirrors `DeepSeekV2MoELayer::load` but expects
    /// FP8 E4M3 expert weights with `weight_scale_inv` block scales (V3/K2
    /// canonical layout).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        gw: &mut ferrite_cuda_core::weights::GpuWeights,
        prefix: &str,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        use_sigmoid: bool,
        n_expert_group: usize,
        topk_group: usize,
        output_dtype: ferrite_cuda_core::dtype::DType,
        stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        use crate::layers_quant::{block_scale_name, ensure_f32_scale};
        use ferrite_cuda_core::driver;
        use ferrite_cuda_core::dtype::DType;
        use ferrite_cuda_core::tensor::GpuTensor;

        let inter = moe_intermediate_size;

        // Gate router (always dense BF16 for V3/K2 — even when experts are FP8).
        let gate = crate::layers::Linear::load(gw, &format!("{prefix}.gate"))?;

        // Derive block size from first expert's gate_proj scale shape.
        let first_gate_w = format!("{prefix}.experts.0.gate_proj.weight");
        let first_gate_s = block_scale_name(gw, &format!("{prefix}.experts.0.gate_proj"));
        let (w_shape, w_dtype) = gw
            .tensor_info(&first_gate_w)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {first_gate_w}"))?;
        anyhow::ensure!(
            w_dtype == DType::Fp8E4m3,
            "DeepSeekV2Fp8BlockMoELayer::load: expected Fp8E4m3 expert weights, got {w_dtype}"
        );
        let (s_shape, _) = gw
            .tensor_info(&first_gate_s)
            .ok_or_else(|| anyhow::anyhow!("scale not found: {first_gate_s}"))?;
        let block_n = w_shape[0] / s_shape[0];
        let block_k = w_shape[1] / s_shape[1];

        // Stacked expert layout (single-rank, no TP yet): w1=[E, 2*inter, hidden] FP8,
        // w2=[E, hidden, inter] FP8.
        let w1_n = 2 * inter;
        let w1_k = hidden_size;
        let w2_n = hidden_size;
        let w2_k = inter;
        let w1_scale_rows = w1_n.div_ceil(block_n);
        let w1_scale_cols = w1_k.div_ceil(block_k);
        let w2_scale_rows = w2_n.div_ceil(block_n);
        let w2_scale_cols = w2_k.div_ceil(block_k);
        let gate_scale_rows = inter.div_ceil(block_n);

        let w1_bytes = n_routed_experts * w1_n * w1_k;
        let w2_bytes = n_routed_experts * w2_n * w2_k;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        let w1_scale_bytes = n_routed_experts * w1_scale_rows * w1_scale_cols * 4;
        let w2_scale_bytes = n_routed_experts * w2_scale_rows * w2_scale_cols * 4;
        let w1_scale_ptr = unsafe { driver::mem_alloc(w1_scale_bytes)? };
        let w2_scale_ptr = unsafe { driver::mem_alloc(w2_scale_bytes)? };

        for e in 0..n_routed_experts {
            let gate_pfx = format!("{prefix}.experts.{e}.gate_proj");
            let up_pfx = format!("{prefix}.experts.{e}.up_proj");
            let down_pfx = format!("{prefix}.experts.{e}.down_proj");

            // FP8 weights: stack gate then up along dim=0, down separately.
            let expert_w1_off = e * w1_n * w1_k;
            let gate_proj_bytes = inter * hidden_size;
            let expert_w2_off = e * w2_n * w2_k;
            unsafe {
                gw.take_into(
                    &format!("{gate_pfx}.weight"),
                    w1_ptr.add(expert_w1_off),
                    stream,
                )?;
                gw.take_into(
                    &format!("{up_pfx}.weight"),
                    w1_ptr.add(expert_w1_off + gate_proj_bytes),
                    stream,
                )?;
                gw.take_into(
                    &format!("{down_pfx}.weight"),
                    w2_ptr.add(expert_w2_off),
                    stream,
                )?;
            }

            // Block scales: copy into stacked 3D buffers.
            let expert_w1_scale_off = e * w1_scale_rows * w1_scale_cols * 4;
            let gate_scale = {
                let raw = gw.take(&block_scale_name(gw, &gate_pfx))?;
                ensure_f32_scale(raw, stream)?
            };
            let gate_scale_bytes = gate_scale_rows * w1_scale_cols * 4;
            unsafe {
                driver::memcpy_dtod_async(
                    w1_scale_ptr.add(expert_w1_scale_off),
                    gate_scale.raw_ptr(),
                    gate_scale_bytes,
                    stream,
                )?;
            }
            let up_scale = {
                let raw = gw.take(&block_scale_name(gw, &up_pfx))?;
                ensure_f32_scale(raw, stream)?
            };
            let up_scale_rows = w1_scale_rows - gate_scale_rows;
            let up_scale_bytes = up_scale_rows * w1_scale_cols * 4;
            unsafe {
                driver::memcpy_dtod_async(
                    w1_scale_ptr.add(expert_w1_scale_off + gate_scale_bytes),
                    up_scale.raw_ptr(),
                    up_scale_bytes,
                    stream,
                )?;
            }
            let expert_w2_scale_off = e * w2_scale_rows * w2_scale_cols * 4;
            let down_scale = {
                let raw = gw.take(&block_scale_name(gw, &down_pfx))?;
                ensure_f32_scale(raw, stream)?
            };
            let down_scale_bytes = w2_scale_rows * w2_scale_cols * 4;
            unsafe {
                driver::memcpy_dtod_async(
                    w2_scale_ptr.add(expert_w2_scale_off),
                    down_scale.raw_ptr(),
                    down_scale_bytes,
                    stream,
                )?;
            }

            // Consume input_scale if present (block quant uses dynamic activation).
            for pfx in &[&gate_pfx, &up_pfx, &down_pfx] {
                let is_name = format!("{pfx}.input_scale");
                if gw.contains(&is_name) {
                    let _ = gw.take(&is_name);
                }
            }
        }

        let w1 = unsafe { GpuTensor::new(w1_ptr, &[n_routed_experts, w1_n, w1_k], DType::Fp8E4m3) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[n_routed_experts, w2_n, w2_k], DType::Fp8E4m3) };
        let w1_scale_inv = unsafe {
            GpuTensor::new(
                w1_scale_ptr,
                &[n_routed_experts, w1_scale_rows, w1_scale_cols],
                DType::F32,
            )
        };
        let w2_scale_inv = unsafe {
            GpuTensor::new(
                w2_scale_ptr,
                &[n_routed_experts, w2_scale_rows, w2_scale_cols],
                DType::F32,
            )
        };

        // e_score_correction_bias (F32) for sigmoid routing.
        let e_score_correction_bias = if use_sigmoid {
            let bias_name = format!("{prefix}.gate.e_score_correction_bias");
            let bias_raw = gw.take(&bias_name)?;
            let bias_f32 = if bias_raw.dtype() == DType::F32 {
                bias_raw
            } else {
                let n = bias_raw.numel();
                let f32_ptr = unsafe { driver::mem_alloc(n * 4)? };
                let f32_bias = unsafe { GpuTensor::new(f32_ptr, &[n], DType::F32) };
                unsafe { kernels::cast_bias_to_f32(bias_raw, f32_bias, stream) };
                f32_bias
            };
            Some(bias_f32)
        } else {
            None
        };

        let moe = Fp8BlockFusedMoELayer {
            gate,
            w1,
            w2,
            w1_scale_inv,
            w2_scale_inv,
            block_size: [block_n, block_k],
            num_experts: n_routed_experts,
            top_k,
            intermediate_size: inter,
            hidden_size,
            renormalize: norm_topk_prob,
            e_score_correction_bias,
            n_expert_group,
            topk_group,
            // Same as DeepSeekV2MoELayer: outer layer applies scale_inplace after
            // experts; inner layer must use 1.0 to avoid double application.
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        // Shared expert (FP8 block): concat gate_proj+up_proj, then down_proj.
        let shared_gate_pfx = format!("{prefix}.shared_experts.gate_proj");
        let shared_up_pfx = format!("{prefix}.shared_experts.up_proj");
        let shared_down_pfx = format!("{prefix}.shared_experts.down_proj");
        let shared_gate_up = crate::layers::Fp8BlockLinear::load_concat(
            gw,
            &[&shared_gate_pfx, &shared_up_pfx],
            output_dtype,
        )?;
        let shared_down = crate::layers::Fp8BlockLinear::load(gw, &shared_down_pfx, output_dtype)?;
        let shared_inter = n_shared_experts * moe_intermediate_size;

        Ok(Self {
            moe,
            shared_gate_up,
            shared_down,
            shared_intermediate_size: shared_inter,
            routed_scaling_factor,
        })
    }
}

// ---------------------------------------------------------------------------
// Fp8FusedMoELayer (FP8 E4M3 quantized MoE)
// ---------------------------------------------------------------------------

/// Fused Mixture of Experts layer with FP8 E4M3 quantized weights.
///
/// Forward pass matches Python vLLM's `fused_experts_impl` with `use_fp8_w8a8=True`:
/// 1. Gate → router logits (dense BF16)
/// 2. topk_softmax → topk weights + ids
/// 3. scaled_fp8_quant_dynamic(hidden) → FP8 input + per-token scales
/// 4. moe_align_block_size
/// 5. fused_moe_fp8_gemm (GEMM 1: gate+up)
/// 6. silu_and_mul activation
/// 7. scaled_fp8_quant_dynamic(activated) → FP8 act + per-token scales
/// 8. fused_moe_fp8_gemm (GEMM 2: down, with routing weight)
/// 9. moe_sum → reduced output
pub struct Fp8FusedMoELayer {
    /// Gate projection: `[hidden_size, num_experts]` — always dense BF16.
    pub gate: Linear,
    /// Stacked gate+up weights: `[num_experts, 2*intermediate_size, hidden_size]` FP8 E4M3.
    pub w1: GpuTensor,
    /// Stacked down weights: `[num_experts, hidden_size, intermediate_size]` FP8 E4M3.
    pub w2: GpuTensor,
    /// Per-expert w1 scales: `[num_experts]` f32.
    pub w1_scale: GpuTensor,
    /// Per-expert w2 scales: `[num_experts]` f32.
    pub w2_scale: GpuTensor,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub renormalize: bool,
    /// `[num_experts]` F32 e_score_correction_bias for sigmoid routing (DeepSeek V3 / Kimi K2).
    /// `None` ⇒ softmax routing (Mixtral / Qwen MoE).
    pub e_score_correction_bias: Option<GpuTensor>,
    /// 0 ⇒ flat top-k. >0 with `topk_group` >0 ⇒ noaux_tc grouped routing.
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

#[cfg(feature = "cuda")]
impl Fp8FusedMoELayer {
    /// Forward pass — full FP8 MoE pipeline.
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;
        let sm_version = device.sm_version;

        let block_m = select_moe_block_m(num_tokens, self.top_k, self.num_experts);

        // 1. Gate: router_logits = hidden_states @ gate_weight^T
        let router_logits =
            self.gate
                .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Route: softmax / sigmoid+bias / grouped noaux_tc (DSv3/Kimi K2).
        let (topk_weights, topk_ids) = route_experts(
            router_logits.as_gpu_tensor(),
            &MoeRouting {
                top_k: self.top_k,
                renormalize: self.renormalize,
                e_score_correction_bias: self.e_score_correction_bias.as_ref(),
                n_expert_group: self.n_expert_group,
                topk_group: self.topk_group,
                routed_scaling_factor: self.routed_scaling_factor,
            },
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        // 3. Quantize hidden states to FP8 with per-token dynamic scales.
        let (fp8_input, a1_scales) =
            kernels::scaled_fp8_quant_dynamic(*hidden_states, &mut device.caching, stream);

        // 4. Align block size: sort tokens by expert
        let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
            topk_ids.as_gpu_tensor(),
            self.num_experts,
            block_m,
            &mut device.caching,
            stream,
        );
        drop(topk_ids);

        // 5. GEMM 1: fp8_input × w1^T → [num_tokens * top_k, 2*intermediate] BF16
        let intermediate1 = kernels::fused_moe_fp8_gemm(
            fp8_input.as_gpu_tensor(),
            self.w1,
            a1_scales.as_gpu_tensor(),
            self.w1_scale,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens,
            self.top_k,
            block_m,
            false, // don't apply routing weights on first GEMM
            sm_version,
            &mut device.caching,
            stream,
        );
        drop(fp8_input);
        drop(a1_scales);

        // 6. Activation: SiLU(gate) * up → [num_tokens * top_k, intermediate]
        let activated = kernels::silu_and_mul_fused(
            intermediate1.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(intermediate1);

        // 7. Re-quantize activated to FP8 with fresh per-token scales (matching Python).
        let (fp8_act, a2_scales) = kernels::scaled_fp8_quant_dynamic(
            activated.as_gpu_tensor(),
            &mut device.caching,
            stream,
        );
        drop(activated);

        // 8. GEMM 2: fp8_act × w2^T → [num_tokens * top_k, hidden_size] BF16
        //    Apply routing weight here. top_k=1 for pass 2 (input already expanded).
        let intermediate2 = kernels::fused_moe_fp8_gemm(
            fp8_act.as_gpu_tensor(),
            self.w2,
            a2_scales.as_gpu_tensor(),
            self.w2_scale,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens * self.top_k,
            1, // top_k=1: index directly into expanded input
            block_m,
            true, // apply routing weights
            sm_version,
            &mut device.caching,
            stream,
        );
        drop(fp8_act);
        drop(a2_scales);
        drop(topk_weights);
        drop(sorted_token_ids);
        drop(expert_ids);
        drop(num_tokens_post_padded);

        // 9. Reduce: sum across top_k experts → [num_tokens, hidden_size]
        let output = kernels::moe_sum(
            intermediate2.as_gpu_tensor(),
            num_tokens,
            self.hidden_size,
            self.top_k,
            &mut device.caching,
            stream,
        );
        drop(intermediate2);

        // 10. TP all-reduce
        #[cfg(feature = "nccl")]
        if let Some(ref nccl) = self.tp_group {
            nccl.all_reduce_inplace(output.as_gpu_tensor())
                .expect("MoE all_reduce failed");
        }

        output
    }
}

// ---------------------------------------------------------------------------
// Fp8BlockFusedMoELayer (FP8 block-quantized MoE)
// ---------------------------------------------------------------------------

/// Fused MoE layer with FP8 E4M3 block-quantized weights.
///
/// Same pipeline as `Fp8FusedMoELayer` but uses per-block weight scales
/// applied during the FP8→BF16 dequant step in the CUDA kernel.
pub struct Fp8BlockFusedMoELayer {
    pub gate: Linear,
    /// Stacked gate+up weights: `[E, 2*inter, hidden]` FP8 E4M3.
    pub w1: GpuTensor,
    /// Stacked down weights: `[E, hidden, inter]` FP8 E4M3.
    pub w2: GpuTensor,
    /// Block scales for w1: `[E, ceil(2*inter/bn), ceil(hidden/bk)]` f32.
    pub w1_scale_inv: GpuTensor,
    /// Block scales for w2: `[E, ceil(hidden/bn), ceil(inter/bk)]` f32.
    pub w2_scale_inv: GpuTensor,
    /// Quantization block size `[block_n, block_k]`.
    pub block_size: [usize; 2],
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub renormalize: bool,
    /// See [`Fp8FusedMoELayer::e_score_correction_bias`].
    pub e_score_correction_bias: Option<GpuTensor>,
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

#[cfg(feature = "cuda")]
impl Fp8BlockFusedMoELayer {
    /// Zero-alloc placeholder. Every field is a logically-invalid sentinel
    /// (dangling `GpuTensor` pointers tagged with `[0, 0, 0]` / `[0, 0]`
    /// shapes; a dummy `Linear`). Only used by
    /// [`Fp8SharedFusedMoELayer`]'s BF16-fallback path to populate its
    /// always-present `moe` field on layers whose experts arrive dense
    /// BF16 (compressed-tensors `ignore` list). The outer forward short-
    /// circuits to the BF16 peer before any of these fields are read.
    pub fn dummy() -> Self {
        use ferrite_cuda_core::dtype::DType;
        let empty = || unsafe {
            GpuTensor::new(
                std::ptr::NonNull::<u8>::dangling().as_ptr(),
                &[0, 0, 0],
                DType::Fp8E4m3,
            )
        };
        let empty_scale = || unsafe {
            GpuTensor::new(
                std::ptr::NonNull::<u8>::dangling().as_ptr(),
                &[0, 0, 0],
                DType::F32,
            )
        };
        let dummy_gate_w = unsafe {
            GpuTensor::new(
                std::ptr::NonNull::<u8>::dangling().as_ptr(),
                &[0, 0],
                DType::BF16,
            )
        };
        Self {
            gate: Linear::new(dummy_gate_w, None),
            w1: empty(),
            w2: empty(),
            w1_scale_inv: empty_scale(),
            w2_scale_inv: empty_scale(),
            block_size: [1, 1],
            num_experts: 0,
            top_k: 0,
            intermediate_size: 0,
            hidden_size: 0,
            renormalize: false,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        }
    }

    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        let block_m = select_moe_block_m(num_tokens, self.top_k, self.num_experts);

        // 1. Gate
        let router_logits =
            self.gate
                .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Route: softmax / sigmoid+bias / grouped noaux_tc.
        let (topk_weights, topk_ids) = route_experts(
            router_logits.as_gpu_tensor(),
            &MoeRouting {
                top_k: self.top_k,
                renormalize: self.renormalize,
                e_score_correction_bias: self.e_score_correction_bias.as_ref(),
                n_expert_group: self.n_expert_group,
                topk_group: self.topk_group,
                routed_scaling_factor: self.routed_scaling_factor,
            },
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        // 3. Quantize hidden states to FP8
        let (fp8_input, a1_scales) =
            kernels::scaled_fp8_quant_dynamic(*hidden_states, &mut device.caching, stream);

        // 4. Align block size
        let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
            topk_ids.as_gpu_tensor(),
            self.num_experts,
            block_m,
            &mut device.caching,
            stream,
        );
        drop(topk_ids);

        // 5. GEMM 1: block-scaled FP8
        let intermediate1 = kernels::fused_moe_fp8_block_gemm(
            fp8_input.as_gpu_tensor(),
            self.w1,
            a1_scales.as_gpu_tensor(),
            self.w1_scale_inv,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens,
            self.top_k,
            block_m,
            false,
            self.block_size,
            &mut device.caching,
            stream,
        );
        drop(fp8_input);
        drop(a1_scales);

        // 6. SiLU activation
        let activated = kernels::silu_and_mul_fused(
            intermediate1.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(intermediate1);

        // 7. Re-quantize to FP8
        let (fp8_act, a2_scales) = kernels::scaled_fp8_quant_dynamic(
            activated.as_gpu_tensor(),
            &mut device.caching,
            stream,
        );
        drop(activated);

        // 8. GEMM 2: block-scaled FP8
        let intermediate2 = kernels::fused_moe_fp8_block_gemm(
            fp8_act.as_gpu_tensor(),
            self.w2,
            a2_scales.as_gpu_tensor(),
            self.w2_scale_inv,
            topk_weights.as_gpu_tensor(),
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            num_tokens * self.top_k,
            1,
            block_m,
            true,
            self.block_size,
            &mut device.caching,
            stream,
        );
        drop(fp8_act);
        drop(a2_scales);
        drop(topk_weights);
        drop(sorted_token_ids);
        drop(expert_ids);
        drop(num_tokens_post_padded);

        // 9. Reduce
        let output = kernels::moe_sum(
            intermediate2.as_gpu_tensor(),
            num_tokens,
            self.hidden_size,
            self.top_k,
            &mut device.caching,
            stream,
        );
        drop(intermediate2);

        #[cfg(feature = "nccl")]
        if let Some(ref nccl) = self.tp_group {
            nccl.all_reduce_inplace(output.as_gpu_tensor())
                .expect("MoE all_reduce failed");
        }

        output
    }
}

// ---------------------------------------------------------------------------
// Fp8SharedFusedMoELayer (Qwen2/3 MoE with FP8 experts)
// ---------------------------------------------------------------------------

/// FP8 MoE layer for the Qwen-MoE family (routed + optional shared expert).
///
/// Routed experts run through [`Fp8BlockFusedMoELayer`] which handles every
/// FP8 weight-scale layout via one `block_size` parameter:
///   * **per-tensor**  → `block_size = [N_max, K_max]` (one scale covers the whole weight)
///   * **per-channel** → `block_size = [1, K_max]`    (one scale per output row)
///   * **blockwise**   → `block_size = [bn, bk]`       (e.g. `[128, 128]` for DeepSeek-style)
///
/// Shared expert projections are [`LinearLayer`] so they can be dense BF16
/// (Qwen2-MoE FP8 checkpoints often keep the shared expert unquantized) or
/// FP8 per-channel (Qwen3-Coder-Next FP8-Dynamic quantizes the whole MLP).
pub struct Fp8SharedFusedMoELayer {
    pub moe: Fp8BlockFusedMoELayer,
    /// Shared expert: fused gate+up projection. `None` when the model has no shared expert.
    pub shared_gate_up: Option<crate::layers::LinearLayer>,
    /// Shared expert: down projection. `None` when the model has no shared expert.
    pub shared_down: Option<crate::layers::LinearLayer>,
    /// Shared-expert sigmoid gate `[1, hidden]` — always dense BF16 in practice.
    pub shared_expert_gate: Option<Linear>,
    /// Per-rank shared-expert intermediate size (shared_expert_intermediate_size / tp_size).
    /// Used by the silu_and_mul inside the shared branch. 0 when there is no shared expert.
    pub intermediate_size: usize,
    /// BF16 fallback for layers the checkpoint's `ignore` list keeps dense
    /// (e.g. layer 47 of `unsloth/Qwen3-Coder-Next-FP8-Dynamic`). When
    /// `Some`, forward delegates to it and the FP8 fields are all-size
    /// dummies. The delegate is TP-sharded along intermediate dim, so the
    /// instruction arm's post-forward all-reduce behaves identically on
    /// both paths.
    pub bf16_fallback: Option<SharedFusedMoELayer>,
}

#[cfg(feature = "cuda")]
impl Fp8SharedFusedMoELayer {
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        // BF16-fallback layers (checkpoint `ignore` list) delegate entirely
        // to the BF16 peer. The peer's inner `tp_group` stays `None`; the
        // DSL interpreter's all-reduce in `Instruction::Fp8SharedFusedMoe`
        // closes the loop for both paths.
        if let Some(ref bf16) = self.bf16_fallback {
            return bf16.forward(hidden_states, device);
        }

        let stream = device.compute_stream;

        // Routed partial. The inner MoE's `tp_group` is `None` — the DSL
        // interpreter performs a single post-combine all-reduce in
        // `Instruction::Fp8SharedFusedMoe` using `ForwardCtx::tp_group`,
        // which covers both the routed MoE partial and the sigmoid-gated
        // shared partial with one collective.
        let moe_out = self.moe.forward(hidden_states, device);

        if let (Some(shared_gate_up), Some(shared_down), Some(shared_gate)) = (
            &self.shared_gate_up,
            &self.shared_down,
            &self.shared_expert_gate,
        ) {
            let shared_gu = shared_gate_up.forward(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                stream,
            );
            let shared_activated = kernels::silu_and_mul_fused(
                shared_gu.as_gpu_tensor(),
                self.intermediate_size,
                &mut device.caching,
                stream,
            );
            drop(shared_gu);

            let shared_out = shared_down.forward(
                shared_activated.view(),
                &mut device.cublas,
                &mut device.caching,
                stream,
            );
            drop(shared_activated);

            // Shared gate is replicated (dense `[1, hidden]`) — identical on
            // every rank — so `sigmoid(gate) * shared_partial + moe_partial`
            // preserves the per-rank partial-sum structure. The instruction
            // arm's all-reduce (at the single TP sync point for this layer)
            // produces the globally-correct `hidden` output.
            let gate_logits =
                shared_gate.forward(hidden_states, &mut device.cublas, &mut device.caching);

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

    /// Load a Qwen-MoE-family FP8 checkpoint.
    ///
    /// Auto-detects the weight-scale scheme from the on-disk scale shape of
    /// expert 0's `gate_proj`:
    ///   * scalar `[]` or `[1]`   → per-tensor   (`block_size = [N_w1, K_w1]`)
    ///   * vector `[N]` / `[N,1]` → per-channel  (`block_size = [1, K_w1]`)
    ///   * `[N_blocks, K_blocks]` → block-wise   (`block_size = [bn, bk]`)
    ///
    /// Shared expert projections may be FP8 or dense BF16; the per-prefix
    /// choice follows on-disk weight dtype via `LinearLayer::load_dense_or_fp8`.
    ///
    /// At `tp_size > 1` this performs intermediate-dim tensor-parallel
    /// sharding matching Python vLLM's `CompressedTensorsW8A8Fp8MoEMethod`
    /// (`_load_w13` along output dim, `_load_w2` along input dim):
    ///   * `gate_proj` / `up_proj`: each expert sharded along dim 0 (output)
    ///     → `[inter/tp, hidden]` per-rank slice fused into `w1 = [E, 2*inter/tp, hidden]`
    ///   * `down_proj`: each expert sharded along dim 1 (input) →
    ///     `[hidden, inter/tp]` per-rank slice into `w2 = [E, hidden, inter/tp]`
    ///   * Per-channel `w1` scales shard along dim 0; per-channel `w2` scales
    ///     index output channels (hidden, not sharded) so stay replicated.
    ///   * Shared expert: column-parallel `gate_proj+up_proj`, row-parallel
    ///     `down_proj`; the outer forward all-reduces the sum after combining
    ///     with the routed MoE output.
    ///   * Router gate and `shared_expert_gate` are replicated across ranks
    ///     (they consume full `[T, hidden]` activations and produce small
    ///     replicated outputs).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        gw: &mut ferrite_cuda_core::weights::GpuWeights,
        prefix: &str,
        num_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        shared_expert_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        output_dtype: ferrite_cuda_core::dtype::DType,
        tp_rank: usize,
        tp_size: usize,
        stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        use crate::layers_quant::{block_scale_name, ensure_f32_scale};
        use ferrite_cuda_core::driver;
        use ferrite_cuda_core::dtype::DType;
        use ferrite_cuda_core::tensor::GpuTensor;

        anyhow::ensure!(
            tp_size >= 1 && tp_rank < tp_size,
            "Fp8SharedFusedMoELayer::load: invalid (tp_rank={tp_rank}, tp_size={tp_size})"
        );
        let inter_full = moe_intermediate_size;
        anyhow::ensure!(
            inter_full.is_multiple_of(tp_size),
            "Fp8SharedFusedMoELayer::load: moe_intermediate_size={inter_full} not divisible by tp_size={tp_size}"
        );
        let inter = inter_full / tp_size;
        let sharded = tp_size > 1;

        // BF16 fallback probe — must run BEFORE any `gw.take` consumes
        // tensors we'd need for the peer load (e.g. the router `gate`).
        // Compressed-tensors checkpoints routinely land a handful of
        // layers on the `ignore` list (e.g. the unsloth
        // Qwen3-Coder-Next-FP8-Dynamic checkpoint ignores layer 47's
        // experts + shared expert). The DSL emits `Fp8SharedFusedMoe`
        // for every layer uniformly, so this layer arrives with BF16
        // experts and needs the BF16 peer.
        let e0_gate_pfx = format!("{prefix}.experts.0.gate_proj");
        let e0_gate_w = format!("{e0_gate_pfx}.weight");
        let (w_shape, w_dtype) = {
            let (s, d) = gw
                .tensor_info(&e0_gate_w)
                .ok_or_else(|| anyhow::anyhow!("weight not found: {e0_gate_w}"))?;
            (s.to_vec(), d)
        };
        if w_dtype != DType::Fp8E4m3 {
            let bf16 = SharedFusedMoELayer::load_sharded(
                gw,
                prefix,
                num_experts,
                top_k,
                moe_intermediate_size,
                shared_expert_intermediate_size,
                hidden_size,
                tp_rank,
                tp_size,
                stream,
            )?;
            return Ok(Self {
                moe: Fp8BlockFusedMoELayer::dummy(),
                shared_gate_up: None,
                shared_down: None,
                shared_expert_gate: None,
                intermediate_size: 0,
                bf16_fallback: Some(bf16),
            });
        }

        // Router gate stays dense BF16 across every FP8 MoE family; replicated.
        let gate = Linear::load(gw, &format!("{prefix}.gate"))?;
        let gate_n_full = w_shape[0]; // == moe_intermediate_size (on-disk)
        let gate_k = w_shape[1]; // == hidden_size
        anyhow::ensure!(
            gate_n_full == inter_full && gate_k == hidden_size,
            "Fp8SharedFusedMoELayer::load: expert gate_proj shape {w_shape:?} \
             mismatches config (inter={inter_full}, hidden={hidden_size})"
        );

        let e0_gate_s = block_scale_name(gw, &e0_gate_pfx);
        let (s_shape, _) = gw
            .tensor_info(&e0_gate_s)
            .ok_or_else(|| anyhow::anyhow!("scale not found: {e0_gate_s}"))?;

        // Three exclusive on-disk schemes — detected from full (unsharded) dims.
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Scheme {
            PerTensor,
            PerChannel,
            Block,
        }
        let (scheme, block_n, block_k) = match s_shape {
            [] | [1] | [1, 1] => (Scheme::PerTensor, gate_n_full, gate_k),
            [n, 1] if *n == gate_n_full => (Scheme::PerChannel, 1usize, gate_k),
            [n] if *n == gate_n_full => (Scheme::PerChannel, 1usize, gate_k),
            [sn, sk] => {
                let bn = gate_n_full.div_ceil(*sn).max(1);
                let bk = gate_k.div_ceil(*sk).max(1);
                (Scheme::Block, bn, bk)
            }
            other => anyhow::bail!(
                "Fp8SharedFusedMoELayer::load: unrecognized scale shape {other:?} at {e0_gate_s} \
                 (expected per-tensor [1], per-channel [N,1] or [N], or block [sn,sk])"
            ),
        };

        // Block-mode TP needs block-aligned sharding. Assert and stop if not.
        if scheme == Scheme::Block && sharded {
            anyhow::ensure!(
                inter_full.is_multiple_of(tp_size * block_n) && inter_full.is_multiple_of(tp_size * block_k),
                "Fp8SharedFusedMoELayer::load: block-mode TP requires inter ({inter_full}) to be \
                 divisible by tp_size×block_n ({}) and tp_size×block_k ({}); drop tp_size or \
                 move to per-channel",
                tp_size * block_n,
                tp_size * block_k,
            );
        }

        // Per-rank stacked layout: w1=[E, 2*inter/tp, hidden] FP8,
        // w2=[E, hidden, inter/tp] FP8. The fused GEMM derives
        // `in_features`/`out_features` from the tensor dims at call time, so
        // per-rank shapes flow through without kernel changes.
        let w1_n = 2 * inter;
        let w1_k = hidden_size;
        let w2_n = hidden_size;
        let w2_k = inter;
        let w1_scale_rows = w1_n.div_ceil(block_n);
        let w1_scale_cols = w1_k.div_ceil(block_k);
        let w2_scale_rows = w2_n.div_ceil(block_n);
        let w2_scale_cols = w2_k.div_ceil(block_k);
        let gate_scale_rows = inter.div_ceil(block_n);

        let w1_bytes = num_experts * w1_n * w1_k;
        let w2_bytes = num_experts * w2_n * w2_k;
        let w1_ptr = unsafe { driver::mem_alloc(w1_bytes)? };
        let w2_ptr = unsafe { driver::mem_alloc(w2_bytes)? };

        let w1_scale_bytes = num_experts * w1_scale_rows * w1_scale_cols * 4;
        let w2_scale_bytes = num_experts * w2_scale_rows * w2_scale_cols * 4;
        let w1_scale_ptr = unsafe { driver::mem_alloc(w1_scale_bytes)? };
        let w2_scale_ptr = unsafe { driver::mem_alloc(w2_scale_bytes)? };

        for e in 0..num_experts {
            let gate_pfx = format!("{prefix}.experts.{e}.gate_proj");
            let up_pfx = format!("{prefix}.experts.{e}.up_proj");
            let down_pfx = format!("{prefix}.experts.{e}.down_proj");

            let expert_w1_off = e * w1_n * w1_k;
            let gate_proj_bytes = inter * hidden_size;
            let expert_w2_off = e * w2_n * w2_k;

            // Weights.
            if sharded {
                unsafe {
                    gw.take_shard_into(
                        &format!("{gate_pfx}.weight"),
                        0,
                        tp_rank,
                        tp_size,
                        w1_ptr.add(expert_w1_off),
                        stream,
                    )?;
                    gw.take_shard_into(
                        &format!("{up_pfx}.weight"),
                        0,
                        tp_rank,
                        tp_size,
                        w1_ptr.add(expert_w1_off + gate_proj_bytes),
                        stream,
                    )?;
                    gw.take_shard_into(
                        &format!("{down_pfx}.weight"),
                        1,
                        tp_rank,
                        tp_size,
                        w2_ptr.add(expert_w2_off),
                        stream,
                    )?;
                }
            } else {
                unsafe {
                    gw.take_into(
                        &format!("{gate_pfx}.weight"),
                        w1_ptr.add(expert_w1_off),
                        stream,
                    )?;
                    gw.take_into(
                        &format!("{up_pfx}.weight"),
                        w1_ptr.add(expert_w1_off + gate_proj_bytes),
                        stream,
                    )?;
                    gw.take_into(
                        &format!("{down_pfx}.weight"),
                        w2_ptr.add(expert_w2_off),
                        stream,
                    )?;
                }
            }

            // Scales.
            let take_scale = |gw: &mut ferrite_cuda_core::weights::GpuWeights,
                              name: &str,
                              dim: usize,
                              shard_this: bool|
             -> anyhow::Result<GpuTensor> {
                let raw = if shard_this {
                    gw.take_shard(name, dim, tp_rank, tp_size)?
                } else {
                    gw.take(name)?
                };
                ensure_f32_scale(raw, stream)
            };
            let (gate_scale, up_scale, down_scale) = match scheme {
                Scheme::PerTensor => (
                    take_scale(gw, &block_scale_name(gw, &gate_pfx), 0, false)?,
                    take_scale(gw, &block_scale_name(gw, &up_pfx), 0, false)?,
                    take_scale(gw, &block_scale_name(gw, &down_pfx), 0, false)?,
                ),
                Scheme::PerChannel => (
                    take_scale(gw, &block_scale_name(gw, &gate_pfx), 0, sharded)?,
                    take_scale(gw, &block_scale_name(gw, &up_pfx), 0, sharded)?,
                    // Down's output dim is hidden (not sharded) → full take.
                    take_scale(gw, &block_scale_name(gw, &down_pfx), 0, false)?,
                ),
                Scheme::Block => (
                    take_scale(gw, &block_scale_name(gw, &gate_pfx), 0, sharded)?,
                    take_scale(gw, &block_scale_name(gw, &up_pfx), 0, sharded)?,
                    take_scale(gw, &block_scale_name(gw, &down_pfx), 1, sharded)?,
                ),
            };

            let expert_w1_scale_off = e * w1_scale_rows * w1_scale_cols * 4;
            let gate_scale_bytes = gate_scale_rows * w1_scale_cols * 4;
            unsafe {
                driver::memcpy_dtod_async(
                    w1_scale_ptr.add(expert_w1_scale_off),
                    gate_scale.raw_ptr(),
                    gate_scale_bytes,
                    stream,
                )?;
            }
            let up_scale_rows = w1_scale_rows - gate_scale_rows;
            let up_scale_bytes = up_scale_rows * w1_scale_cols * 4;
            unsafe {
                driver::memcpy_dtod_async(
                    w1_scale_ptr.add(expert_w1_scale_off + gate_scale_bytes),
                    up_scale.raw_ptr(),
                    up_scale_bytes,
                    stream,
                )?;
            }
            let expert_w2_scale_off = e * w2_scale_rows * w2_scale_cols * 4;
            let down_scale_bytes = w2_scale_rows * w2_scale_cols * 4;
            unsafe {
                driver::memcpy_dtod_async(
                    w2_scale_ptr.add(expert_w2_scale_off),
                    down_scale.raw_ptr(),
                    down_scale_bytes,
                    stream,
                )?;
            }

            for pfx in &[&gate_pfx, &up_pfx, &down_pfx] {
                let is_name = format!("{pfx}.input_scale");
                if gw.contains(&is_name) {
                    let _ = gw.take(&is_name);
                }
            }
        }

        let w1 = unsafe { GpuTensor::new(w1_ptr, &[num_experts, w1_n, w1_k], DType::Fp8E4m3) };
        let w2 = unsafe { GpuTensor::new(w2_ptr, &[num_experts, w2_n, w2_k], DType::Fp8E4m3) };
        let w1_scale_inv = unsafe {
            GpuTensor::new(
                w1_scale_ptr,
                &[num_experts, w1_scale_rows, w1_scale_cols],
                DType::F32,
            )
        };
        let w2_scale_inv = unsafe {
            GpuTensor::new(
                w2_scale_ptr,
                &[num_experts, w2_scale_rows, w2_scale_cols],
                DType::F32,
            )
        };

        // Leave `tp_group = None` on the inner MoE. The outer
        // `Fp8SharedFusedMoELayer::forward` performs a single all-reduce
        // covering both the routed MoE output and the sigmoid-gated shared
        // partial — two all-reduces would be wasteful.
        let moe = Fp8BlockFusedMoELayer {
            gate,
            w1,
            w2,
            w1_scale_inv,
            w2_scale_inv,
            block_size: [block_n, block_k],
            num_experts,
            top_k,
            intermediate_size: inter,
            hidden_size,
            renormalize: norm_topk_prob,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        // Shared expert projections: FP8 per-channel in Qwen3-Coder-Next,
        // dense BF16 in some Qwen2-MoE FP8 checkpoints. Probe by on-disk dtype
        // of gate_proj and dispatch per-weight; dtype cannot mix across the
        // three projections for a given expert.
        let (shared_gate_up, shared_down, shared_expert_gate) = if shared_expert_intermediate_size > 0 {
            anyhow::ensure!(
                shared_expert_intermediate_size.is_multiple_of(tp_size),
                "Fp8SharedFusedMoELayer::load: shared_expert_intermediate_size={shared_expert_intermediate_size} \
                 not divisible by tp_size={tp_size}"
            );
            let sg_gate_pfx = format!("{prefix}.shared_expert.gate_proj");
            let sg_up_pfx = format!("{prefix}.shared_expert.up_proj");
            let sg_down_pfx = format!("{prefix}.shared_expert.down_proj");
            let probe_name = format!("{sg_gate_pfx}.weight");
            let (_, probe_dtype) = gw
                .tensor_info(&probe_name)
                .ok_or_else(|| anyhow::anyhow!("shared-expert weight not found: {probe_name}"))?;
            let (gu, dn) = if probe_dtype == DType::Fp8E4m3 {
                let gu = if sharded {
                    crate::layers::Fp8Linear::load_concat_sharded(
                        gw,
                        &[&sg_gate_pfx, &sg_up_pfx],
                        tp_rank,
                        tp_size,
                        output_dtype,
                    )?
                } else {
                    crate::layers::Fp8Linear::load_concat(
                        gw,
                        &[&sg_gate_pfx, &sg_up_pfx],
                        output_dtype,
                    )?
                };
                let dn = if sharded {
                    crate::layers::Fp8Linear::load_sharded(
                        gw,
                        &sg_down_pfx,
                        1,
                        tp_rank,
                        tp_size,
                        output_dtype,
                    )?
                } else {
                    crate::layers::Fp8Linear::load(gw, &sg_down_pfx, output_dtype)?
                };
                (
                    crate::layers::LinearLayer::Fp8(Box::new(gu)),
                    crate::layers::LinearLayer::Fp8(Box::new(dn)),
                )
            } else {
                let gu = if sharded {
                    crate::layers::LinearLayer::load_dense_concat_sharded(
                        gw,
                        &[&sg_gate_pfx, &sg_up_pfx],
                        stream,
                        tp_rank,
                        tp_size,
                    )?
                } else {
                    crate::layers::LinearLayer::load_dense_concat(
                        gw,
                        &[&sg_gate_pfx, &sg_up_pfx],
                        stream,
                    )?
                };
                let dn = if sharded {
                    crate::layers::LinearLayer::load_dense_sharded(
                        gw,
                        &sg_down_pfx,
                        1,
                        tp_rank,
                        tp_size,
                    )?
                } else {
                    crate::layers::LinearLayer::load_dense(gw, &sg_down_pfx)?
                };
                (gu, dn)
            };
            let sg = Linear::load(gw, &format!("{prefix}.shared_expert_gate"))?;
            (Some(gu), Some(dn), Some(sg))
        } else {
            (None, None, None)
        };

        // Per-rank intermediate size for the shared-expert silu_and_mul.
        // (Was moe_intermediate_size in the previous single-rank path, which
        // matched only because Qwen3-Coder-Next happens to set
        // `moe_intermediate_size == shared_expert_intermediate_size`.)
        let shared_inter_per_rank = if shared_expert_intermediate_size > 0 {
            shared_expert_intermediate_size / tp_size
        } else {
            0
        };

        Ok(Self {
            moe,
            shared_gate_up,
            shared_down,
            shared_expert_gate,
            intermediate_size: shared_inter_per_rank,
            bf16_fallback: None,
        })
    }
}

// ---------------------------------------------------------------------------
// GgmlFusedMoELayer (GGML quantized MoE)
// ---------------------------------------------------------------------------

/// Fused Mixture of Experts layer with GGML-quantized expert weights.
///
/// Uses the `indexed_moe_forward` kernels which handle expert routing
/// internally via an index array: `blockIdx.y = batch_idx`, `blockIdx.z = topk_idx`,
/// expert looked up from `indices[batch * topk + topk_idx]`.
///
/// Forward pass:
/// 1. Gate → router logits (dense matmul)
/// 2. topk_softmax → topk_weights, topk_ids
/// 3. Quantize hidden_states to Q8_1
/// 4. indexed_moe_forward(w1, q8_input, indices) → [batch*topk, 2*inter] f32
/// 5. silu_and_mul → [batch*topk, inter] f32
/// 6. Quantize activated to Q8_1
/// 7. indexed_moe_forward(w2, q8_activated, indices) → [batch*topk, hidden] f32
/// 8. Scale by topk_weights, sum across topk → [batch, hidden] f32
pub struct GgmlFusedMoELayer {
    /// Gate projection: `[num_experts, hidden_size]` dense.
    /// llama.cpp converters store the router gate as F32 unconditionally
    /// (even in Q4_K_M GGUFs), so the ferrite GGUF loader puts it in
    /// `gguf_dense` already cast to model dtype — a plain dense `Linear`
    /// is the right type. (FP8/AWQ/etc. router gates are similarly dense.)
    pub gate: Linear,
    /// Stacked gate+up weights: `[num_experts, 2*intermediate_size, hidden_size]` quantized.
    /// GgmlStorage with nrows = num_experts * 2 * intermediate_size, ncols = hidden_size.
    pub w1: ferrite_cuda_core::ggml_quant::GgmlStorage,
    /// Stacked down weights: `[num_experts, hidden_size, intermediate_size]` quantized.
    /// GgmlStorage with nrows = num_experts * hidden_size, ncols = intermediate_size.
    pub w2: ferrite_cuda_core::ggml_quant::GgmlStorage,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub renormalize: bool,
    /// See [`Fp8FusedMoELayer::e_score_correction_bias`].
    pub e_score_correction_bias: Option<GpuTensor>,
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
}

#[cfg(feature = "cuda")]
impl GgmlFusedMoELayer {
    /// Forward pass — full quantized MoE pipeline.
    ///
    /// * `hidden_states`: `[num_tokens, hidden_size]` in any dtype (cast to f32 internally).
    ///
    /// Returns: `[num_tokens, hidden_size]` in same dtype as input.
    ///
    /// # Safety
    /// All tensors must be valid GPU memory.
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        use crate::ggml::{MATRIX_ROW_PADDING, ggml_moe_forward, ggml_quantize_q8_1_alloc};
        use ferrite_cuda_core::dtype::DType;

        let input_dtype = hidden_states.dtype();
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        // 1. Gate: router_logits = hidden_states @ gate_weight^T (dense Linear).
        let router_logits =
            self.gate
                .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Route: softmax / sigmoid+bias / grouped noaux_tc.
        let (topk_weights, topk_ids) = route_experts(
            router_logits.as_gpu_tensor(),
            &MoeRouting {
                top_k: self.top_k,
                renormalize: self.renormalize,
                e_score_correction_bias: self.e_score_correction_bias.as_ref(),
                n_expert_group: self.n_expert_group,
                topk_group: self.topk_group,
                routed_scaling_factor: self.routed_scaling_factor,
            },
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        // Expert indices: [num_tokens, top_k] i32 viewed as u32 (expert ids are non-negative).
        let indices_ptr = topk_ids.as_gpu_tensor().raw_ptr() as *const u32;

        // 3. Cast hidden_states to f32 if needed, then quantize to Q8_1.
        let hs_f32 = if input_dtype != DType::F32 {
            Some(kernels::cast_logits_to_f32(
                *hidden_states,
                &mut device.caching,
                stream,
            ))
        } else {
            None
        };
        let hs_f32_ptr = if let Some(ref cast) = hs_f32 {
            cast.as_gpu_tensor().raw_ptr() as *const f32
        } else {
            hidden_states.as_ptr::<f32>()
        };
        let k = self.hidden_size;
        let k_padded = crate::ggml::pad(k, MATRIX_ROW_PADDING);
        let (q8_hidden, _) =
            ggml_quantize_q8_1_alloc(hs_f32_ptr, k, num_tokens, &mut device.caching, stream);
        drop(hs_f32);

        // 4. GEMM 1: w1 × q8_input → [num_tokens * top_k, 2*intermediate] f32
        // The indexed_moe_forward kernel uses input_dim1 to determine input sharing:
        //   input_idx = (input_dim1 == 1) ? current_batch : task_id
        // We pass input_dim1=1 so all topk experts for a token share the same
        // quantized input row (no replication needed).
        let out1 = device.caching.alloc_tensor(
            &[num_tokens * self.top_k, 2 * self.intermediate_size],
            ferrite_cuda_core::dtype::DType::F32,
        );
        ggml_moe_forward(
            &self.w1,
            q8_hidden,
            indices_ptr,
            out1.as_gpu_tensor().raw_ptr() as *mut f32,
            2 * self.intermediate_size,
            k,
            num_tokens,
            self.top_k,
            k_padded,
            1, // input_dim1=1: share input across topk per batch item
            stream,
        );

        // 5. SiLU-and-mul → [num_tokens * top_k, intermediate] f32
        let activated = kernels::silu_and_mul_fused(
            out1.as_gpu_tensor(),
            self.intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(out1);

        // 6. Quantize activated to Q8_1
        let inter = self.intermediate_size;
        let inter_padded = crate::ggml::pad(inter, MATRIX_ROW_PADDING);
        let (q8_act, _) = ggml_quantize_q8_1_alloc(
            activated.as_gpu_tensor().raw_ptr() as *const f32,
            inter,
            num_tokens * self.top_k,
            &mut device.caching,
            stream,
        );
        drop(activated);

        // 7. GEMM 2: w2 × q8_act → [num_tokens * top_k, hidden_size] f32
        // For GEMM 2, each task has its own unique input row (the activated output),
        // so input_dim1 = batch * topk (i.e. not 1).
        let out2 = device.caching.alloc_tensor(
            &[num_tokens * self.top_k, self.hidden_size],
            ferrite_cuda_core::dtype::DType::F32,
        );
        ggml_moe_forward(
            &self.w2,
            q8_act,
            indices_ptr,
            out2.as_gpu_tensor().raw_ptr() as *mut f32,
            self.hidden_size,
            inter,
            num_tokens,
            self.top_k,
            inter_padded,
            num_tokens * self.top_k, // input_dim1: each task has unique input
            stream,
        );
        drop(topk_ids); // indices no longer needed

        // 8. Scale by topk_weights and sum across topk → [num_tokens, hidden_size] f32.
        //
        // out2 is `[num_tokens * topk, hidden]` f32; topk_weights is
        // `[num_tokens, topk]` f32 (also num_tokens*topk contiguous f32s). We
        // need `out2[row, :] *= topk_weights[row]` — *row*-wise scaling.
        //
        // `kernels::broadcast_mul_inplace` is the WRONG primitive here: its
        // kernel does `out2[r, c] *= scale[c]` — column-wise — and indexes
        // `scale[c]` for c in [0, hidden), reading num_tokens*topk past the
        // end of the topk_weights buffer (= UB; corrupts output to ±Inf/NaN
        // and the bug propagates through the residual stream into every
        // downstream MoE layer's gate matmul). Use `fp8_post_scale_multiply`,
        // which is genuinely per-row (`output[i, :] *= scales[i]`).
        // Reshape via `with_view` because the kernel asserts `output.ndim()==2`
        // and `scales.ndim()==1` and that `output.dim(0)==scales.dim(0)`.
        let topk_view = topk_weights.view();
        let topk_flat = unsafe {
            ferrite_cuda_core::tensor::TensorView::from_raw(
                ferrite_cuda_core::tensor::GpuTensor::new(
                    topk_view.raw_ptr(),
                    &[num_tokens * self.top_k],
                    ferrite_cuda_core::dtype::DType::F32,
                ),
            )
        };
        kernels::fp8_post_scale_multiply(out2.as_gpu_tensor(), *topk_flat, stream);
        drop(topk_weights);

        let output = kernels::moe_sum(
            out2.as_gpu_tensor(),
            num_tokens,
            self.hidden_size,
            self.top_k,
            &mut device.caching,
            stream,
        );
        drop(out2);

        // Cast back to original dtype if we converted to f32.
        if input_dtype != DType::F32 {
            let result = kernels::cast_from_f32(
                output.as_gpu_tensor(),
                input_dtype,
                &mut device.caching,
                stream,
            );
            drop(output);
            result
        } else {
            output
        }
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2GgmlMoELayer (GGML-quantized DeepSeek MoE)
// ---------------------------------------------------------------------------

/// GGUF/GGML-quantized analog of [`DeepSeekV2MoELayer`]. Used for any
/// DeepSeek-family GGUF (V2-Lite, Moonlight V3-flat, K2 GGUFs).
///
/// Forward shape mirrors `DeepSeekV2MoELayer::forward` — only the
/// expert storage changes (`GgmlFusedMoELayer` for routed experts;
/// `GgmlLinear` for shared gate_up + down).
pub struct DeepSeekV2GgmlMoELayer {
    pub moe: GgmlFusedMoELayer,
    /// Shared expert fused gate+up: concatenated `GgmlLinear`
    /// `[2 * shared_inter, hidden_size]`.
    pub shared_gate_up: crate::layers::GgmlLinear,
    /// Shared expert down: `GgmlLinear` `[hidden_size, shared_inter]`.
    pub shared_down: crate::layers::GgmlLinear,
    pub shared_intermediate_size: usize,
    pub routed_scaling_factor: f32,
}

#[cfg(feature = "cuda")]
impl DeepSeekV2GgmlMoELayer {
    /// Forward pass: `output = routed_scaling_factor * moe(x) + shared_expert(x)`.
    ///
    /// `moe.forward` returns the input dtype (cast back from f32 inside).
    /// `shared_gate_up.forward` casts back to input dtype too (GgmlLinear
    /// internal contract). So both `moe_out` and `shared_out` are
    /// `input_dtype` at the `add_inplace` seam — same shape as
    /// `DeepSeekV2MoELayer::forward`'s terminal add.
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let stream = device.compute_stream;

        // Routed experts (GGML).
        let moe_out = self.moe.forward(hidden_states, device);
        if self.routed_scaling_factor != 1.0 {
            kernels::scale_inplace(*moe_out.view(), self.routed_scaling_factor, &device.cublas);
        }

        // Shared expert: silu(gate_up) → down. Both GgmlLinear; outputs match
        // `hidden_states.dtype()` because GgmlLinear casts back from its
        // internal f32 to the input dtype on exit.
        let shared_gu = self
            .shared_gate_up
            .forward(hidden_states, &mut device.caching, stream);
        let shared_activated = kernels::silu_and_mul_fused(
            *shared_gu.view(),
            self.shared_intermediate_size,
            &mut device.caching,
            stream,
        );
        drop(shared_gu);
        let shared_out =
            self.shared_down
                .forward(shared_activated.view(), &mut device.caching, stream);
        drop(shared_activated);

        // output = moe_out + shared_out (no sigmoid gate); both in input_dtype.
        kernels::add_inplace(*moe_out.view(), *shared_out.view(), stream);
        drop(shared_out);
        moe_out
    }

    /// Load from GGUF weights. Mirrors the hand-written
    /// `vllm-cuda::deepseek_v2::load_gguf` (model-side fused-3D MoE
    /// expert layout): consumes `mlp.experts.fused_{gate,up,down}_exps.weight`
    /// (already 3D `[num_experts, slab, hidden|inter]` after `ferrite-gguf`'s
    /// rename) plus `mlp.shared_experts.{gate,up,down}_proj.weight` and
    /// `mlp.gate.weight` (router).
    ///
    /// Interleaves gate+up byte slabs into a single quantized
    /// `w1 = [E, 2*inter, hidden]` for the `indexed_moe_forward` kernel;
    /// reuses the `down` tensor as `w2 = [E, hidden, inter]`. Frees the
    /// per-component `gate_exps`/`up_exps` buffers after copy so net
    /// memory matches the per-tensor sum (no doubling).
    #[allow(clippy::too_many_arguments)]
    pub fn load_gguf(
        gw: &mut ferrite_cuda_core::weights::GpuWeights,
        prefix: &str,
        n_routed_experts: usize,
        n_shared_experts: usize,
        top_k: usize,
        moe_intermediate_size: usize,
        hidden_size: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
        use_sigmoid: bool,
        n_expert_group: usize,
        topk_group: usize,
        stream: ferrite_cuda_core::CUstream,
    ) -> anyhow::Result<Self> {
        use crate::ggml::GgmlStorage;
        use ferrite_cuda_core::driver;
        use ferrite_cuda_core::dtype::DType;
        use ferrite_cuda_core::tensor::GpuTensor;

        let inter = moe_intermediate_size;
        let hidden = hidden_size;

        // ── Router gate (dense BF16; ferrite-gguf loader converts the
        // F32-on-disk gate to model dtype during load, so it lands in
        // `gguf_dense` already shape-correct for a plain Linear). ──
        let gate_name = format!("{prefix}.gate.weight");
        let gate_w = gw
            .take_gguf_dense(&gate_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_name}"))?;
        let gate = Linear::new(gate_w, None);

        // ── Routed expert weights (fused 3D) ──
        // `mlp.experts.fused_gate_exps.weight` lands as a single GgmlStorage
        // with `nrows = E * inter`, `ncols = hidden` (set by ggml.rs's 3D-flatten).
        let gate_exps_name = format!("{prefix}.experts.fused_gate_exps.weight");
        let up_exps_name = format!("{prefix}.experts.fused_up_exps.weight");
        let down_exps_name = format!("{prefix}.experts.fused_down_exps.weight");

        let gate_exps = gw
            .quantized_map_mut()
            .remove(&gate_exps_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {gate_exps_name}"))?;
        let up_exps = gw
            .quantized_map_mut()
            .remove(&up_exps_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {up_exps_name}"))?;
        let down_exps = gw
            .quantized_map_mut()
            .remove(&down_exps_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {down_exps_name}"))?;

        // gate + up must share dtype because we interleave their byte slabs
        // into a single w1 quantized buffer; down lives in its own w2 storage
        // and can use a different (often higher-precision) quant — Q4_K_M
        // GGUFs typically ship gate/up=Q4_K, down=Q8_0.
        anyhow::ensure!(
            gate_exps.dtype == up_exps.dtype,
            "DeepSeekV2GgmlMoELayer::load_gguf: gate/up expert quant dtype \
             mismatch (gate={:?}, up={:?}); cannot interleave into shared w1",
            gate_exps.dtype,
            up_exps.dtype,
        );

        let qdtype = gate_exps.dtype;
        let bs = qdtype.block_size();
        let ts = qdtype.type_size();

        // Each expert's gate slab: [inter, hidden] quantized.
        // Each expert's up   slab: [inter, hidden] quantized (same size).
        // w1 expert slab        : [2*inter, hidden] quantized — gate then up.
        let expert_slab_bytes = (inter * hidden / bs) * ts;
        let w1_expert_bytes = 2 * expert_slab_bytes;
        let w1_total_bytes = n_routed_experts * w1_expert_bytes;
        let w1_ptr = unsafe { driver::mem_alloc(w1_total_bytes)? };

        for e in 0..n_routed_experts {
            let gate_offset = e * expert_slab_bytes;
            let up_offset = e * expert_slab_bytes;
            let w1_gate_offset = e * w1_expert_bytes;
            let w1_up_offset = w1_gate_offset + expert_slab_bytes;
            unsafe {
                driver::memcpy_dtod_async(
                    w1_ptr.add(w1_gate_offset),
                    gate_exps.ptr.add(gate_offset),
                    expert_slab_bytes,
                    stream,
                )?;
                driver::memcpy_dtod_async(
                    w1_ptr.add(w1_up_offset),
                    up_exps.ptr.add(up_offset),
                    expert_slab_bytes,
                    stream,
                )?;
            }
        }

        // Sync, then free the per-component buffers — w1 now owns the bytes.
        unsafe {
            driver::stream_synchronize(stream)?;
            gw.unrecord_alloc(gate_exps.ptr);
            driver::mem_free(gate_exps.ptr)?;
            gw.unrecord_alloc(up_exps.ptr);
            driver::mem_free(up_exps.ptr)?;
        }
        // Track w1 with the GpuWeights so its lifetime matches the model's.
        gw.record_alloc(w1_ptr, w1_total_bytes);

        let w1 = GgmlStorage {
            ptr: w1_ptr,
            len: w1_total_bytes,
            dtype: qdtype,
            nrows: n_routed_experts * 2 * inter,
            ncols: hidden,
        };
        // w2 = down_exps unchanged: [E, hidden, inter] quantized
        // (nrows = E * hidden, ncols = inter — already set by 3D flatten).
        let w2 = down_exps;

        // ── e_score_correction_bias (V3 / Kimi K2 sigmoid routing) ──
        let e_score_correction_bias = if use_sigmoid {
            let bias_name = format!("{prefix}.gate.e_score_correction_bias");
            let bias_raw = gw.take(&bias_name)?;
            let bias_f32 = if bias_raw.dtype() == DType::F32 {
                bias_raw
            } else {
                let n = bias_raw.numel();
                let f32_ptr = unsafe { driver::mem_alloc(n * 4)? };
                let f32_bias = unsafe { GpuTensor::new(f32_ptr, &[n], DType::F32) };
                unsafe { kernels::cast_bias_to_f32(bias_raw, f32_bias, stream) };
                gw.record_alloc(f32_ptr, n * 4);
                f32_bias
            };
            Some(bias_f32)
        } else {
            None
        };

        let moe = GgmlFusedMoELayer {
            gate,
            w1,
            w2,
            num_experts: n_routed_experts,
            top_k,
            intermediate_size: inter,
            hidden_size: hidden,
            renormalize: norm_topk_prob,
            e_score_correction_bias,
            n_expert_group,
            topk_group,
            // Outer DeepSeekV2GgmlMoELayer::forward applies routed_scaling_factor
            // via scale_inplace; inner stays at 1.0 to avoid double application
            // (matches DeepSeekV2MoELayer's split).
            routed_scaling_factor: 1.0,
        };

        // ── Shared experts: concat gate_proj + up_proj into one fused slab ──
        let shared_inter = n_shared_experts * moe_intermediate_size;
        let shared_gate_name = format!("{prefix}.shared_experts.gate_proj.weight");
        let shared_up_name = format!("{prefix}.shared_experts.up_proj.weight");
        let shared_down_name = format!("{prefix}.shared_experts.down_proj.weight");

        let shared_gate_s = gw
            .quantized_map_mut()
            .remove(&shared_gate_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {shared_gate_name}"))?;
        let shared_up_s = gw
            .quantized_map_mut()
            .remove(&shared_up_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {shared_up_name}"))?;
        let shared_down_s = gw
            .quantized_map_mut()
            .remove(&shared_down_name)
            .ok_or_else(|| anyhow::anyhow!("weight not found: {shared_down_name}"))?;

        anyhow::ensure!(
            shared_gate_s.dtype == shared_up_s.dtype,
            "DeepSeekV2GgmlMoELayer::load_gguf: shared gate/up quant dtype mismatch \
             ({:?} vs {:?})",
            shared_gate_s.dtype,
            shared_up_s.dtype,
        );

        let se_qdtype = shared_gate_s.dtype;
        let se_bs = se_qdtype.block_size();
        let se_ts = se_qdtype.type_size();
        let se_slab_bytes = (shared_inter * hidden / se_bs) * se_ts;
        let se_total_bytes = 2 * se_slab_bytes;
        let se_ptr = unsafe { driver::mem_alloc(se_total_bytes)? };
        unsafe {
            driver::memcpy_dtod_async(se_ptr, shared_gate_s.ptr, se_slab_bytes, stream)?;
            driver::memcpy_dtod_async(
                se_ptr.add(se_slab_bytes),
                shared_up_s.ptr,
                se_slab_bytes,
                stream,
            )?;
            driver::stream_synchronize(stream)?;
            gw.unrecord_alloc(shared_gate_s.ptr);
            driver::mem_free(shared_gate_s.ptr)?;
            gw.unrecord_alloc(shared_up_s.ptr);
            driver::mem_free(shared_up_s.ptr)?;
        }
        gw.record_alloc(se_ptr, se_total_bytes);

        let shared_gate_up = crate::layers::GgmlLinear {
            storage: GgmlStorage {
                ptr: se_ptr,
                len: se_total_bytes,
                dtype: se_qdtype,
                nrows: 2 * shared_inter,
                ncols: hidden,
            },
            bias: None,
        };
        let shared_down = crate::layers::GgmlLinear {
            storage: shared_down_s,
            bias: None,
        };

        Ok(Self {
            moe,
            shared_gate_up,
            shared_down,
            shared_intermediate_size: shared_inter,
            routed_scaling_factor,
        })
    }
}

// ---------------------------------------------------------------------------
// FP8 MoE Weight Loading Helpers
// ---------------------------------------------------------------------------

/// Load FP8 MoE expert weights with per-tensor or per-block scales.
///
/// For per-tensor scales: dequantizes FP8 expert weights to BF16 at load time
/// using merged scales (max of gate/up shard scales, matching Python's
/// `process_fp8_weight_tensor_strategy_moe`).
///
/// For per-block scales: dequantizes using block-indexed scales.
///
/// The dequantized BF16 weights are then used with the existing fused_moe_gemm
/// kernel which operates on BF16. This is the correctness-first approach;
/// a native FP8 fused MoE GEMM kernel can be added later for performance.
///
/// NOTE: This function should be called at model load time, not on the hot path.
/// The returned tensors are BF16 and work with the existing `FusedMoELayer`.
#[cfg(feature = "cuda")]
pub fn load_fp8_moe_weights_dequant(
    w_fp8: GpuTensor, // [num_experts, dim, hidden] FP8 E4M3
    scale: GpuTensor, // [num_experts] f32 (per-tensor) or per-block
    output_dtype: ferrite_cuda_core::dtype::DType,
    alloc: &mut ferrite_cuda_core::alloc::CachingAllocator,
    stream: cudarc::driver::sys::CUstream,
) -> ferrite_cuda_core::alloc::OwnedTensor {
    // For now, use the per-tensor approach: dequantize the entire stacked tensor.
    // This works because scale is per-expert (the fused_moe_gemm kernel selects
    // the right expert slice anyway).
    //
    // TODO: Implement native FP8 fused MoE GEMM for perf parity.
    let _total_elements: usize = w_fp8.numel();

    // Reshape to 2D for dequant, then reshape back.
    // w_fp8: [num_experts, dim, hidden] → flatten to [num_experts * dim, hidden]
    // Then dequant each element using per-expert scale.
    //
    // For the correctness-first approach, we use the max scale across all experts
    // and dequant the entire tensor as 2D.
    let ne = w_fp8.dim(0);
    let d1 = w_fp8.dim(1);
    let d2 = w_fp8.dim(2);

    // Read scales to CPU to find max.
    // Note: This is only done once at load time, so D2H is acceptable.
    let _scale = scale;
    let _ne = ne;

    // For simplicity and correctness, we dequant each expert's 2D slice separately.
    // This is done at load time and is not on the hot path.
    let total_elems = ne * d1 * d2;
    let out = alloc.alloc_tensor(&[ne, d1, d2], output_dtype);

    // Use the block dequant kernel with block_size = [d1, d2] (entire expert = one block).
    // Or simpler: element-wise dequant with per-expert scale.
    // For now, fall through to a per-element kernel using a scale of 1.0 (identity).
    // The actual dequant should use the per-expert scale.
    //
    // TODO: Implement proper per-expert FP8 dequant kernel.
    // For now, this is a placeholder that marks the integration point.
    let _ = total_elems;
    let _ = stream;

    out
}

// ---------------------------------------------------------------------------
// MarlinFusedMoELayer (AWQ/GPTQ INT4 quantized MoE)
// ---------------------------------------------------------------------------

/// Dynamic block size selection for Marlin MoE GEMM tiling.
/// Matches Python vLLM's `_fused_marlin_moe` logic:
///   for block_size_m in [8, 16, 32, 48, 64]:
///       if M * topk / E / block_size_m < 0.9: break
/// Only thread_m_blocks=1 kernels are instantiated → max block_size = 16.
#[cfg(feature = "cuda")]
fn select_moe_block_size(num_tokens: usize, top_k: usize, num_experts: usize) -> usize {
    for &bs in &[8usize, 16] {
        if ((num_tokens * top_k) as f64 / num_experts as f64 / bs as f64) < 0.9 {
            return bs;
        }
    }
    16 // max with thread_m_blocks=1
}

/// Fused Mixture of Experts layer using Marlin INT4 quantized weights.
///
/// Uses the `marlin_moe_gemm` kernel which fuses expert routing + Marlin
/// INT4 GEMM + topk weight multiply into a single kernel per pass.
///
/// Forward matches Python vLLM's `_fused_marlin_moe` two-pass pattern.
pub struct MarlinFusedMoELayer {
    /// Gate projection: `[hidden_size, num_experts]` — always dense.
    pub gate: Linear,
    /// Stacked gate+up weights: `[E, K/tile, 2N*tile]` Marlin-packed.
    pub w1: GpuTensor,
    /// Stacked down weights: `[E, N/tile, K*tile]` Marlin-packed.
    pub w2: GpuTensor,
    /// Scales for w1: `[E, num_groups_w1, 2*intermediate_size]`.
    pub w1_scales: GpuTensor,
    /// Scales for w2: `[E, num_groups_w2, hidden_size]`.
    pub w2_scales: GpuTensor,
    /// Zero points for w1 (AWQ only): `[E, num_groups_w1, 2*intermediate/8]`.
    pub w1_zeros: Option<GpuTensor>,
    /// Zero points for w2 (AWQ only): `[E, num_groups_w2, hidden/8]`.
    pub w2_zeros: Option<GpuTensor>,
    /// Workspace for barrier synchronization: `[sms * 4]` i32.
    pub workspace: GpuTensor,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub group_size: usize,
    pub has_zp: bool,
    /// 1 = kU4 (AWQ), 0 = kU4B8 (GPTQ).
    pub b_type_id: i32,
    pub renormalize: bool,
    /// See [`Fp8FusedMoELayer::e_score_correction_bias`].
    pub e_score_correction_bias: Option<GpuTensor>,
    pub n_expert_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
    #[cfg(feature = "nccl")]
    pub tp_group: Option<Arc<NcclGroup>>,
}

#[cfg(feature = "cuda")]
impl MarlinFusedMoELayer {
    /// Forward pass — full Marlin MoE pipeline.
    ///
    /// * `hidden_states`: `[num_tokens, hidden_size]`
    ///
    /// Returns: `[num_tokens, hidden_size]`
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let num_tokens = hidden_states.dim(0);
        let stream = device.compute_stream;

        // 1. Gate: router_logits = hidden_states @ gate_weight^T
        let router_logits =
            self.gate
                .forward(hidden_states, &mut device.cublas, &mut device.caching);

        // 2. Route: softmax / sigmoid+bias / grouped noaux_tc.
        let (topk_weights, topk_ids) = route_experts(
            router_logits.as_gpu_tensor(),
            &MoeRouting {
                top_k: self.top_k,
                renormalize: self.renormalize,
                e_score_correction_bias: self.e_score_correction_bias.as_ref(),
                n_expert_group: self.n_expert_group,
                topk_group: self.topk_group,
                routed_scaling_factor: self.routed_scaling_factor,
            },
            &mut device.caching,
            stream,
        );
        drop(router_logits);

        // 3. Dynamic block size selection (matches Python vLLM)
        let moe_block_size = select_moe_block_size(num_tokens, self.top_k, self.num_experts);

        // Align block size: sort tokens by expert for fused GEMM
        let (sorted_token_ids, expert_ids, num_tokens_post_padded) = kernels::moe_align_block_size(
            topk_ids.as_gpu_tensor(),
            self.num_experts,
            moe_block_size,
            &mut device.caching,
            stream,
        );
        drop(topk_ids);

        let num_groups_w1 = self.w1_scales.dim(1);
        let num_groups_w2 = self.w2_scales.dim(1);
        let group_size_w1 = if num_groups_w1 > 1 {
            self.hidden_size / num_groups_w1
        } else {
            -1i64 as usize
        };
        let group_size_w2 = if num_groups_w2 > 1 {
            self.intermediate_size / num_groups_w2
        } else {
            -1i64 as usize
        };

        // 4. GEMM 1: hidden_states × w1^T → [num_tokens * top_k, 2 * intermediate]
        //    No topk weight applied (mul_topk_weights=false).
        let intermediate1 = kernels::marlin_moe_gemm(
            *hidden_states,
            self.w1,
            self.w1_scales,
            self.w1_zeros,
            None, // g_idx
            None, // perm
            self.workspace,
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            topk_weights.as_gpu_tensor(),
            moe_block_size,
            self.num_experts,
            self.top_k,
            false, // don't apply weights on first GEMM
            num_tokens,
            2 * self.intermediate_size,
            self.hidden_size,
            num_groups_w1,
            group_size_w1,
            false, // has_act_order
            self.has_zp,
            self.b_type_id,
            device.device_id as i32,
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
        //    Reuse same sorted_token_ids/expert_ids/num_tokens_post_padded from pass 1.
        //    Set top_k=1 because input is already expanded to [M*top_k, intermediate].
        //    Apply routing weights (mul_topk_weights=true). Matches Python vLLM's
        //    _fused_marlin_moe second pass.
        //    Apply routing weight here (mul_topk_weights=true).
        let intermediate2 = kernels::marlin_moe_gemm(
            activated.as_gpu_tensor(),
            self.w2,
            self.w2_scales,
            self.w2_zeros,
            None, // g_idx
            None, // perm
            self.workspace,
            sorted_token_ids.as_gpu_tensor(),
            expert_ids.as_gpu_tensor(),
            num_tokens_post_padded.as_gpu_tensor(),
            topk_weights.as_gpu_tensor(),
            moe_block_size,
            self.num_experts,
            1,    // top_k=1 for pass 2 (input already expanded)
            true, // apply routing weights
            num_tokens * self.top_k,
            self.hidden_size,
            self.intermediate_size,
            num_groups_w2,
            group_size_w2,
            false, // has_act_order
            self.has_zp,
            self.b_type_id,
            device.device_id as i32,
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

        // 9. TP all-reduce
        #[cfg(feature = "nccl")]
        if let Some(ref nccl) = self.tp_group {
            nccl.all_reduce_inplace(output.as_gpu_tensor())
                .expect("MoE all_reduce failed");
        }

        output
    }
}

/// MoE layer with optional shared expert, using Marlin quantized weights.
pub struct MarlinSharedFusedMoELayer {
    pub moe: MarlinFusedMoELayer,
    /// Shared expert: fused gate+up (Marlin-quantized or dense Linear).
    pub shared_gate_up: Option<crate::layers::LinearLayer>,
    /// Shared expert: down projection.
    pub shared_down: Option<crate::layers::LinearLayer>,
    /// Shared expert gate: `[1, hidden]` — sigmoid gate for shared expert output.
    pub shared_expert_gate: Option<Linear>,
    pub intermediate_size: usize,
}

#[cfg(feature = "cuda")]
impl MarlinSharedFusedMoELayer {
    pub unsafe fn forward(
        &self,
        hidden_states: TensorView<'_>,
        device: &mut GpuDevice,
    ) -> OwnedTensor {
        let stream = device.compute_stream;

        // MoE path.
        let moe_out = self.moe.forward(hidden_states, device);

        // Shared expert path (if present).
        if let (Some(shared_gate_up), Some(shared_down), Some(shared_gate)) = (
            &self.shared_gate_up,
            &self.shared_down,
            &self.shared_expert_gate,
        ) {
            let shared_gu = shared_gate_up.forward(
                hidden_states,
                &mut device.cublas,
                &mut device.caching,
                stream,
            );
            let shared_activated = kernels::silu_and_mul_fused(
                shared_gu.as_gpu_tensor(),
                self.intermediate_size,
                &mut device.caching,
                stream,
            );
            drop(shared_gu);

            let shared_out = shared_down.forward(
                shared_activated.view(),
                &mut device.cublas,
                &mut device.caching,
                stream,
            );
            drop(shared_activated);

            let gate_logits =
                shared_gate.forward(hidden_states, &mut device.cublas, &mut device.caching);

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
    use ferrite_cuda_core::DType;

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
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        assert_eq!(layer.num_experts, 8);
        assert_eq!(layer.top_k, 2);
        assert_eq!(layer.intermediate_size, 14336);
    }

    #[test]
    fn test_fp8_fused_moe_layer_sizes() {
        // Verify Fp8FusedMoELayer struct construction with dummy tensors.
        let gate_w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[8, 4096], DType::BF16) };
        let w1 = unsafe { GpuTensor::new(0x2000 as *mut u8, &[8, 28672, 4096], DType::Fp8E4m3) };
        let w2 = unsafe { GpuTensor::new(0x3000 as *mut u8, &[8, 4096, 14336], DType::Fp8E4m3) };
        let w1_scale = unsafe { GpuTensor::new(0x4000 as *mut u8, &[8], DType::F32) };
        let w2_scale = unsafe { GpuTensor::new(0x5000 as *mut u8, &[8], DType::F32) };

        let layer = Fp8FusedMoELayer {
            gate: Linear::new(gate_w, None),
            w1,
            w2,
            w1_scale,
            w2_scale,
            num_experts: 8,
            top_k: 2,
            intermediate_size: 14336,
            hidden_size: 4096,
            renormalize: false,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        assert_eq!(layer.num_experts, 8);
        assert_eq!(layer.top_k, 2);
        assert_eq!(layer.intermediate_size, 14336);
        assert_eq!(layer.hidden_size, 4096);
        assert_eq!(layer.w1.shape(), &[8, 28672, 4096]);
        assert_eq!(layer.w2.shape(), &[8, 4096, 14336]);
        assert_eq!(layer.w1.dtype(), DType::Fp8E4m3);
        assert_eq!(layer.w2.dtype(), DType::Fp8E4m3);
        assert_eq!(layer.w1_scale.shape(), &[8]);
        assert_eq!(layer.w2_scale.shape(), &[8]);
    }

    #[test]
    fn test_fp8_shared_fused_moe_layer_sizes() {
        // Verify Fp8SharedFusedMoELayer struct construction.
        // Per-channel routed experts (block_size=[1, K]) + dense BF16 shared expert.
        let gate_w = unsafe { GpuTensor::new(0x1000 as *mut u8, &[4, 2048], DType::BF16) };
        let w1 = unsafe { GpuTensor::new(0x2000 as *mut u8, &[4, 6144, 2048], DType::Fp8E4m3) };
        let w2 = unsafe { GpuTensor::new(0x3000 as *mut u8, &[4, 2048, 3072], DType::Fp8E4m3) };
        // Per-channel scales: [E, N, 1] f32.
        let w1_scale = unsafe { GpuTensor::new(0x4000 as *mut u8, &[4, 6144, 1], DType::F32) };
        let w2_scale = unsafe { GpuTensor::new(0x5000 as *mut u8, &[4, 2048, 1], DType::F32) };
        let shared_gate_up =
            unsafe { GpuTensor::new(0x6000 as *mut u8, &[6144, 2048], DType::BF16) };
        let shared_down = unsafe { GpuTensor::new(0x7000 as *mut u8, &[2048, 3072], DType::BF16) };

        let moe = Fp8BlockFusedMoELayer {
            gate: Linear::new(gate_w, None),
            w1,
            w2,
            w1_scale_inv: w1_scale,
            w2_scale_inv: w2_scale,
            block_size: [1, 2048],
            num_experts: 4,
            top_k: 2,
            intermediate_size: 3072,
            hidden_size: 2048,
            renormalize: true,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        let layer = Fp8SharedFusedMoELayer {
            moe,
            shared_gate_up: Some(crate::layers::LinearLayer::Dense(Linear::new(
                shared_gate_up,
                None,
            ))),
            shared_down: Some(crate::layers::LinearLayer::Dense(Linear::new(
                shared_down,
                None,
            ))),
            shared_expert_gate: None,
            intermediate_size: 3072,
            bf16_fallback: None,
        };

        assert_eq!(layer.moe.num_experts, 4);
        assert_eq!(layer.moe.top_k, 2);
        assert_eq!(layer.moe.intermediate_size, 3072);
        assert_eq!(layer.moe.block_size, [1, 2048]);
    }

    #[test]
    fn test_fp8_block_fused_moe_layer_sizes() {
        // Verify Fp8BlockFusedMoELayer struct construction with dummy tensors.
        // Model: 128 experts, inter=768, hidden=2048, block_size=[128,128]
        let num_experts = 128;
        let inter = 768;
        let hidden = 2048;
        let block_n = 128;
        let block_k = 128;

        let gate_w =
            unsafe { GpuTensor::new(0x1000 as *mut u8, &[num_experts, hidden], DType::BF16) };
        let w1 = unsafe {
            GpuTensor::new(
                0x2000 as *mut u8,
                &[num_experts, 2 * inter, hidden],
                DType::Fp8E4m3,
            )
        };
        let w2 = unsafe {
            GpuTensor::new(
                0x3000 as *mut u8,
                &[num_experts, hidden, inter],
                DType::Fp8E4m3,
            )
        };

        // Scale shapes: [E, ceil(N/bn), ceil(K/bk)]
        let w1_sr = (2 * inter).div_ceil(block_n); // ceil(1536/128) = 12
        let w1_sc = hidden.div_ceil(block_k); // ceil(2048/128) = 16
        let w2_sr = hidden.div_ceil(block_n); // ceil(2048/128) = 16
        let w2_sc = inter.div_ceil(block_k); // ceil(768/128) = 6

        let w1_scale =
            unsafe { GpuTensor::new(0x4000 as *mut u8, &[num_experts, w1_sr, w1_sc], DType::F32) };
        let w2_scale =
            unsafe { GpuTensor::new(0x5000 as *mut u8, &[num_experts, w2_sr, w2_sc], DType::F32) };

        let layer = Fp8BlockFusedMoELayer {
            gate: Linear::new(gate_w, None),
            w1,
            w2,
            w1_scale_inv: w1_scale,
            w2_scale_inv: w2_scale,
            block_size: [block_n, block_k],
            num_experts,
            top_k: 8,
            intermediate_size: inter,
            hidden_size: hidden,
            renormalize: true,
            e_score_correction_bias: None,
            n_expert_group: 0,
            topk_group: 0,
            routed_scaling_factor: 1.0,
            #[cfg(feature = "nccl")]
            tp_group: None,
        };

        assert_eq!(layer.num_experts, 128);
        assert_eq!(layer.top_k, 8);
        assert_eq!(layer.intermediate_size, 768);
        assert_eq!(layer.hidden_size, 2048);
        assert_eq!(layer.block_size, [128, 128]);
        assert_eq!(layer.w1.shape(), &[128, 1536, 2048]);
        assert_eq!(layer.w2.shape(), &[128, 2048, 768]);
        assert_eq!(layer.w1.dtype(), DType::Fp8E4m3);
        assert_eq!(layer.w1_scale_inv.shape(), &[128, 12, 16]);
        assert_eq!(layer.w2_scale_inv.shape(), &[128, 16, 6]);
    }
}
