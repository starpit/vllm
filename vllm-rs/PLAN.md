# Plan: Qwen3-Next CudaWorker Implementation

## Architecture Summary

Qwen3-Next is a **hybrid** model with three unique components:
1. **Full attention layers** (every 4th layer by default) — standard QKV attention with QK-norm, partial RoPE, and **output gating** (sigmoid gate on Q projection)
2. **GDN linear attention layers** (remaining layers) — Gated Delta Net: conv1d → recurrence with per-head state `[num_v_heads, head_v_dim, head_k_dim]`
3. **MoE MLP** on some layers (reuses existing Qwen3 MoE logic) + dense MLP on others

The `layer_types` array in config determines which layers are `"full_attention"` vs `"linear_attention"`.

## Key Challenges

1. **GDN recurrence state**: Unlike transformer KV cache, GDN layers maintain a `conv_state` (last K-1 tokens of conv input) and `ssm_state` (recurrent `[n_v_heads, head_v_dim, head_k_dim]` matrix per layer). This is fundamentally different from paged KV cache.
2. **Hybrid KV cache**: Full attention layers use standard paged FlashAttention KV cache. GDN layers use recurrent state. The `KvCachePool` must handle both or we need a separate state store.
3. **New CUDA kernels needed**: causal_conv1d, L2 normalization, softplus, gated delta recurrence, RMSNormGated (norm * sigmoid(z))
4. **Partial RoPE**: Only `partial_rotary_factor` fraction of head_dim gets RoPE (e.g., 25%). Need to split, rotate, concat.

## Implementation Plan

### Step 1: Config parsing
- Add `Qwen3NextConfig` struct in `vllm-cuda/src/model/qwen3_next.rs`
- Parse `layer_types`, `partial_rotary_factor`, `linear_*` fields, `attn_output_gate`
- Reuse `Qwen3MoeConfig` fields for MoE layers
- Parse from `HfModelConfig` (same pattern as MLX)

### Step 2: Full attention layer (the easier half)
- `Qwen3NextAttention`: similar to `LlamaAttention` with QK-norm (already have `load_fused_with_qk_norm`)
- **Differences from LLaMA**:
  - Q projection is **doubled** (`num_heads * head_dim * 2`) — first half is Q, second half is gate
  - Partial RoPE: split Q/K at `rotary_dim`, apply RoPE only to first part, concat back
  - Output gating: `sigmoid(gate) * attn_output` before o_proj
  - Uses `GemmaRMSNorm` (weight + 1) for QK norms
- Can reuse `LlamaAttention` internals for the FA2 path, but need custom Q loading and gating
- Weight names: `self_attn.qkv_proj`, `self_attn.o_proj`, `self_attn.q_norm`, `self_attn.k_norm`

### Step 3: GDN linear attention layer (the hard part)
This is the novel component. Forward pass per token:
1. **Input projections**: `in_proj_qkvz` → split into Q, K, V, Z (grouped by k-heads); `in_proj_ba` → split into B, A
2. **Causal conv1d + SiLU**: slide window over [Q‖K‖V] with `conv1d_weight [conv_dim, kernel_size]`
3. **Gating**: `g = -softplus(A + dt_bias) * exp(A_log)`, `beta = sigmoid(B)`
4. **Recurrence** (per v-head): `S[h] = exp(g_h) * S[h] + beta_h * outer(v_h, k_h)`, `o[h] = S[h] @ l2norm(q[h_k])`
5. **RMSNormGated**: `rms_norm(output) * norm_weight * sigmoid(Z)`
6. **Output projection**: `out_proj`

**State management**:
- `conv_state`: `[conv_dim, kernel_size - 1]` per layer — last K-1 inputs for conv sliding window
- `ssm_state`: `[num_v_heads, head_v_dim, head_k_dim]` per layer — recurrent state matrix

**CUDA kernels needed** (can do token-at-a-time on GPU for decode, sequential loop for prefill):
- `causal_conv1d_fwd`: depthwise 1D conv + SiLU activation (or reuse Mamba's causal_conv1d)
- `gdn_recurrence`: fused per-head state update + output computation
- `l2_normalize`: per-head L2 norm for Q/K
- `rms_norm_gated`: fused RMS norm with sigmoid gate
- `softplus`: log(1 + exp(x)) — trivial element-wise kernel

**For initial implementation**: Do the recurrence on CPU (like MLX does token-by-token), move to GPU kernels later. This gets correctness first.

### Step 4: Hybrid state management
- For **full attention layers**: use existing `KvCachePool` (paged FA2)
- For **GDN layers**: need per-layer `conv_state` + `ssm_state` tensors on GPU
- Option A: Store GDN state in a separate `GdnStatePool` alongside `KvCachePool`
- Option B: Encode GDN state as "virtual" KV cache blocks (wasteful)
- **Recommend Option A**: separate `GdnStatePool` with `[batch_size, conv_dim, kernel-1]` and `[batch_size, n_v_heads, head_v_dim, head_k_dim]` per GDN layer. Indexed by request slot.

### Step 5: Wire into CudaWorker
- Add `Qwen3Next` variant to `CudaModel` enum in `cuda_worker.rs`
- Add `"Qwen3NextForCausalLM"` to the model dispatch match
- The `forward` method needs to pass both KV cache (for full attn layers) and GDN state (for linear attn layers)
- `num_layers`, `num_kv_heads`, `head_dim` accessors: only count full attention layers for KV cache sizing
- `execute_model` must manage GDN state allocation/deallocation per request

### Step 6: MLP layer (mostly reuse)
- Dense layers: reuse `LlamaMLP`
- MoE layers: reuse `Qwen3MoeMlp` from `qwen3_moe.rs` (same shared expert pattern)

### Step 7: E2E test
- Need a small Qwen3-Next model on HuggingFace (check what's available)
- Wire into `TestModels` in E2E tests

## Ordering & Effort

| Step | Effort | Dependencies |
|------|--------|-------------|
| 1. Config | Small | None |
| 2. Full attention | Medium | Step 1 |
| 3. GDN (CPU recurrence) | Large | Step 1 |
| 4. Hybrid state mgmt | Medium | Step 3 |
| 5. Wire CudaWorker | Medium | Steps 2-4 |
| 6. MLP | Small | Step 1 (reuse) |
| 7. E2E test | Small | Steps 5-6 |

**Total estimate**: This is a multi-session effort. The GDN recurrence + hybrid state management is the crux. Starting with CPU-side recurrence (like MLX) gets us to correctness, then we can add GPU kernels for performance.

## Open Questions
1. What test model to use? Need to find a small Qwen3-Next on HF.
2. Should we do prefill chunkwise (like Python's `chunk_gated_delta_rule`) or token-by-token? Token-by-token is simpler but O(n²) in state updates for long prefills.
3. CUDA graphs: GDN layers have variable-size state updates — likely incompatible with decode graphs initially.
