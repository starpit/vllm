# Handoff — Session 4

## The goal (unchanged)

Fuse `rms_norm → GEMM` into production, then extend to the full MLP pipeline:
```
rms_norm → GEMM_gate_up → SiLU+mul → GEMM_down → residual_add
```

Current milestone: `rms_norm → GEMM` working in production (`vllm serve` correct text on Qwen2.5-0.5B).

## Where we are

**We have a failing test.** test13c runs the full model (24 layers, real embeddings,
prefill + autoregressive decode) and compares token output between the standard ferrite
path and the fused norm+GEMM path. The fused path produces **wrong tokens** starting
at decode step 1.

```
decode 0: std_token=119069 fused_token=119069 MATCH   hidden_diff=9.00e0
decode 1: std_token=114401 fused_token= 81931 MISMATCH  hidden_diff=1.10e1
decode 2: std_token= 91457 fused_token=144255 MISMATCH  hidden_diff=1.00e1
```

This is the first time the production failure has been reproduced outside `vllm serve`.

## What was done

### Tests written (test_transformer_block.rs)

| Test | What | Result |
|------|------|--------|
| test10 | 1 layer, full block, layer.forward() vs manual fused | Diffs from bf16 precision |
| test10c | QKV-only, shared weights, various M | 0→1.56e-2 (bf16 expected) |
| test10d | Norm precision: fused_add_rms_norm vs add+rms_norm | 1.56e-2 (f32 vs bf16) |
| test10e | 24-layer chain, norm+GEMM only (no attention) | Bounded at ~0.1 |
| test11a | QKV norm+GEMM step only | 0→1.56e-2 |
| test11b | + attention | 1.95e-3 |
| test11c | + MLP gate_up | 3.12e-2 |
| test11d | Full single layer | 1.56e-2 |
| test12 | 24 layers with attention, prefill only | Bounded at ~0.4 |
| test13b | Decode sanity (standard only) | PASS |
| test13c | **Prefill + decode, token comparison** | **FAIL — token mismatch at step 1** |

### What was ruled out

1. **Fused kernel correctness**: 0.00e0 vs separate same-precision path (test9a-k from session 3)
2. **Precision explosion**: bf16 round-trip doesn't compound to garbage over 24 layers (test10e, test12)
3. **Single-layer correctness**: all steps match within bf16 tolerance (test11a-d)
4. **Prefill-only correctness**: 24 layers with attention, bounded diffs (test12)

### The actual failure mode

The fused path diverges during **autoregressive decode**. Each decode step feeds the
previous step's output back as input. The per-step bf16 precision difference (~0.1 in
hidden states, ~9-11 after full model) is large enough to flip the argmax token. Once
a different token is selected, the trajectories diverge completely.

The precision difference comes from the **add→norm boundary**: `fused_add_rms_norm_inplace`
keeps the `residual + hidden_states` sum in f32 for the norm, while `add_inplace` +
`launch_fused_norm_gemm` truncates to bf16 between the two kernels. test10d proves
this directly: 1.56e-2 norm diff at M=64.

### Why it matters for autoregression but not prefill

During prefill, both paths process the same input tokens. The hidden state diffs are
small but don't feed back into themselves. During decode, each step's slightly-wrong
hidden state becomes the next step's input. The error compounds multiplicatively
through the autoregressive loop, not through the layers.

### Architecture note: the two-step split is intentional

The add and norm are split because in the full pipeline:
- The **add** lives in the previous GEMM's **epilogue** (beta=1.0)
- The **norm** lives in the next GEMM's **prologue** (pipeline_fuse!)

These are different GEMMs — the bf16 GMEM boundary between them is inherent to the
two-GEMM-per-block architecture. BUT: the norm prologue could take two inputs
(residual + hidden_states), add them in f32, and norm the f32 sum. This preserves
precision without requiring a single kernel for all three operations.

## What to do next

### Option A: Fix the precision (recommended)

Extend the fused norm+GEMM prologue to accept TWO inputs (residual and hidden_states),
add them in f32 registers, write bf16 back to residual, and norm the f32 sum. This
matches `fused_add_rms_norm_inplace` precision. The `compile!` macro needs to support
a `fused_add` pre-operation in the prologue.

Then re-run test13c — if tokens match, wire into llama.rs.

### Option B: Accept the precision loss, fix via temperature

The bf16 precision path might produce correct text with the right sampling parameters.
The hidden diffs are 9-11, which is close to the argmax boundary. Slightly different
logits might still produce coherent text with temperature > 0. But greedy (argmax)
will diverge. This is fragile and model-dependent — not recommended.

### Option C: Fall back to standard path

Use `fused_add_rms_norm_inplace` + separate `forward_ferrite` GEMM. This is the current
working production path. It doesn't achieve norm→GEMM fusion but still uses ferrite's
CUTLASS GEMMs. No precision issue. Could ship this while working on Option A.

### In production but NOT yet tested

- The first token was correct in test13c but wrong in `vllm serve`. This suggests
  there may be an additional engine-layer issue beyond precision. The test uses fake
  token IDs (not the real "What is 2+2?" tokens). Try with real prompt tokens.
- CUDA graph capture path (not exercised by tests)
- Async scheduling overlap

## Rules (unchanged)

- **NEVER write PTX by hand** — use extracted code
- **NEVER build special-case macros** — extend `compile!`
- **NEVER claim "proven"** without `vllm serve` producing correct text
- **NEVER dismiss divergence** — 1.56e-2 per norm step flips tokens in autoregressive decode
- **Tests ARE the product** — test13c is the real test: do tokens match?
- **No shortcuts**

## Key files

| File | What |
|------|------|
| `tests/test_transformer_block.rs` | All session 4 tests (test10-13) |
| `test13c` | **THE failing test** — prefill+decode token comparison |
| `test10d` | Proves the precision difference at the norm level |
| `HANDOFF_SESSION3.md` | Prior session context |
| `ferrite_gemm_real_weights.rs` | Session 3 kernel tests (test9a-k) |
| `ferrite.rs:375-462` | `launch_fused_norm_gemm` |
| `llama.rs:31-36` | `FUSED_NORM_GEMM` compile! definition |
| `llama.rs:1004-1106` | Standard ferrite forward path (working) |

## Running the tests

```bash
# The failing test (token comparison)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test13c -- --nocapture --test-threads=1

# All test11 (single-layer decomposition)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block "test11" -- --nocapture --test-threads=1

# Multi-layer prefill
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test12 -- --nocapture --test-threads=1
```
