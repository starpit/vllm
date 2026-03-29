# Handoff — Session 4

## The goal

Ferrite fuses CUDA kernels at compile time via escape analysis on PTX. The current
milestone is fusing `rms_norm` into the CUTLASS GEMM prologue so that normalization
and matrix multiply happen in a single kernel launch, eliminating the GMEM round-trip.

The full pipeline target (Phase 5G) is:
```
rms_norm → GEMM_gate_up → SiLU+mul → GEMM_down → residual_add
```
All in one launch. The current milestone is the first piece: `rms_norm → GEMM`.

When this works in production (`vllm serve` produces correct text on Qwen2.5-0.5B),
the architecture extends to the full pipeline.

## What the working path does (llama.rs today)

Per transformer layer (for layers 1-23, layer 0 has no residual):
```
1. fused_add_rms_norm_inplace(hidden_states, residual, norm_weight, eps)
   → residual = residual + hidden_states  (in-place)
   → hidden_states = rms_norm(residual) * norm_weight  (in-place)
   NOTE: this kernel computes res+hs in f32 and norms the f32 sum
2. qkv = forward_ferrite(hidden_states)   // ferrite CUTLASS GEMM
3. attn_output = attention(qkv, ...)
4. fused_add_rms_norm_inplace(attn_output, residual, norm_weight2, eps)
5. gate_up = forward_ferrite(attn_output)  // ferrite CUTLASS GEMM
6. activated = silu_and_mul(gate_up)
7. mlp_output = down_proj(activated)
```

This produces correct text. The ferrite CUTLASS GEMMs match cuBLAS at 0.00e0.

## What the fused path does

Replace steps 1+2 with:
```
1. add_inplace(residual, hidden_states)  // residual += hidden_states (bf16 GMEM write)
2. qkv = launch_fused_norm_gemm(FUSED_NORM_GEMM, residual, ...)
   // reads residual from GMEM (bf16), norms it, feeds to GEMM → qkv
```

Same for steps 4+5.

The architectural reason for the split: in the full pipeline, the add lives in the
previous GEMM's epilogue (beta=1.0) and the norm lives in the next GEMM's prologue
(pipeline_fuse!). These are different GEMMs — the split is intentional.

## Where we are

**We have a failing test.** test13c is the first test to reproduce the production failure
outside `vllm serve`. It runs the full model (24 layers, real embeddings, prefill +
autoregressive decode) and compares token output (argmax over logits):

```
decode 0: std_token=119069 fused_token=119069 MATCH   hidden_diff=9.00e0
decode 1: std_token=114401 fused_token= 81931 MISMATCH  hidden_diff=1.10e1
decode 2: std_token= 91457 fused_token=144255 MISMATCH  hidden_diff=1.00e1
```

Three prior sessions could not reproduce the failure outside production. This session
did, by progressively closing the test gap until tokens were compared via argmax.

## The root cause

The precision difference at the **add→norm boundary**.

`fused_add_rms_norm_inplace` computes `residual + hidden_states` in **f32 registers**
and norms the f32 sum — all in one kernel. The fused path writes `bf16(res + hs)` to
GMEM via `add_inplace`, then reads the bf16 back in `launch_fused_norm_gemm`. The bf16
truncation changes the norm input by up to **1.56e-2** per step (test10d proves this).

This 1.56e-2 is harmless during prefill (both paths process the same tokens, diffs
stay bounded at ~0.4 after 24 layers — test12). But during **autoregressive decode**,
each step feeds the previous step's output back as input. The slightly-wrong hidden
state produces slightly-different logits, which can flip the argmax token. Once a
different token is selected, the trajectories diverge completely.

The fix: the fused prologue needs to accept **two inputs** (residual and hidden_states),
add them in f32, write bf16 back to residual, and norm the f32 sum — matching
`fused_add_rms_norm_inplace` precision.

## Methodology

Session 3 ended with: kernel tests all pass at 0.00e0, production fails, cause unknown.
The handoff prescribed: progressively close the test gap between isolated tests and
production until a test fails.

Session 4 followed this methodology:

1. **test11a-d**: Decompose a single layer into steps (QKV, attention, MLP, full).
   All pass within bf16 tolerance. Establishes that single-layer behavior is correct.

2. **test12**: Chain 24 layers with attention (prefill only). Both paths stay bounded
   (final diff ~0.4). No explosion.

3. **test13b**: Sanity check — decode works with the standard path alone.

4. **test13c**: Prefill + autoregressive decode, 24 layers, comparing **argmax tokens**
   (not just hidden state diffs). **This is the test that fails.**

The key insight that took too long to reach: bounded hidden-state diffs do NOT mean
correct tokens. A diff of 9-11 in hidden states (which looks "bounded") is large enough
to flip argmax over a 151936-token vocabulary. Tests must compare tokens, not just norms.

## Gotchas discovered

### GpuWeights drop frees layer memory
`GpuWeights::from_single_file()` owns the GPU memory for loaded weights via `gpu_allocs`.
`LlamaDecoderLayer::load()` returns `GpuTensor` pointers into that memory. If you
`drop(weights)`, the layer's weight pointers become dangling → `CUDA_ERROR_ILLEGAL_ADDRESS`.
Keep `GpuWeights` alive for the lifetime of the layer.

### Shared CachingAllocator across two paths
Running two forward paths (standard + fused) through the same `device.caching` causes
crashes during decode. The interleaved alloc/free patterns corrupt state. test13c fixes
this by running each path in a **separate GpuDevice** (sequential execution). If you need
both paths in one process, use separate allocators.

### Tests must use --test-threads=1
Multiple tests sharing one GPU deadlock at 300% CPU. Always run with `--test-threads=1`.

### Real vs fake tokens is an untested axis
All tests use synthetic or arbitrary token IDs. Real prompt tokens could produce different
value distributions that trigger edge cases. This is orthogonal to all other test axes —
any test could be upgraded to use real tokens.

### llama.rs FERRITE_FUSED_NORM env var (attempted, reverted)
I added a `FERRITE_FUSED_NORM=1` env var to toggle the fused path in `layer.forward()`.
Ran `vllm serve` with it → garbage (same as session 3). Reverted llama.rs. The env var
approach works mechanically; the fix needs to happen at the precision level first.

## Unexplained gap

Production (`vllm serve`) produces garbage at the **first token**. test13c matches at
decode step 0 and diverges at step 1. Possible explanations:
- test13c uses fake token IDs (not the real prompt). Real prompt tokens may trigger
  the divergence earlier.
- The engine has additional infrastructure (CUDA graphs, async scheduling, `InputBatch`)
  that the test doesn't exercise.
- CUDA graphs were ruled out by session 3 (`--enforce-eager` still fails), but async
  scheduling was not.

## TODOs (in order)

### 1. Close the first-token gap
Run test13c with the **real** tokenized prompt for "What is 2+2? Answer in one word."
(use the Qwen tokenizer to get the actual token IDs). If decode step 0 also diverges,
the test matches production exactly. If it still matches, the first-token failure in
production has a separate cause (engine-layer).

### 2. Try real tokens across other tests
Real vs fake tokens is an untested axis that is orthogonal to every other test dimension.
Any existing test (test11a-d, test12, test13c) could be upgraded to use real tokens.
This rules out value-distribution-dependent bugs.

### 3. Fix the precision in the prologue
Extend `compile!` / the pipeline compiler to support a **fused_add** pre-operation in
the GEMM prologue:
- Prologue accepts TWO input pointers (residual, hidden_states)
- Computes `sum = f32(residual[i]) + f32(hidden_states[i])` in registers
- Writes `bf16(sum)` back to the residual buffer (side-effect writeback)
- Feeds the **f32 sum** (not the bf16 truncation) into the rms_norm reduction
- Norm output feeds into GEMM A-loads as before

This matches `fused_add_rms_norm_inplace` precision. The key change is in
`pipeline_compile.rs` — the prologue currently reads one input; it needs to read two
and fuse the add before the norm.

### 4. Re-run test13c
After the precision fix, re-run test13c. All decode tokens must match. If they don't,
there's a second bug.

### 5. Wire into llama.rs
Replace steps 1+2 in the ferrite forward path with the new fused_add_norm_gemm kernel.
The `FERRITE_FUSED_NORM` env var approach (attempted and reverted this session) works
mechanically — just flip it on once the kernel is correct.

### 6. Test with `vllm serve`
Run `vllm serve` with the fused path enabled. Send "What is 2+2? Answer in one word."
The model must produce coherent, correct text. This is the milestone.

### Alternative: fall back to standard path
If the precision fix is too complex for now, ship the current working path:
`fused_add_rms_norm_inplace` + separate `forward_ferrite` GEMM. This uses ferrite's
CUTLASS GEMMs (faster than cuBLAS) but doesn't fuse norm→GEMM. Work on the full
pipeline fusion (including the precision fix) as a separate effort.

## Rules (non-negotiable)

- **NEVER write PTX by hand** — use extracted code from PTX analysis
- **NEVER build special-case macros** — extend `compile!`
- **NEVER claim "proven"** without `vllm serve` producing correct text
- **NEVER dismiss divergence** — 1.56e-2 per norm step flips tokens in autoregressive decode
- **Tests ARE the product** — test13c is the real test: do tokens match?
- **No shortcuts** — every step on the critical path to the full pipeline
- **Do NOT touch llama.rs** until you have a failing test that you can fix

## Key files

| File | What |
|------|------|
| `tests/test_transformer_block.rs` | All session 4 tests (test10-13) |
| `test13c` | **THE failing test** — prefill+decode token comparison |
| `test10d` | Proves the precision difference at the norm level |
| `HANDOFF_SESSION3.md` | Prior session context |
| `FERRITE.md` | Architecture, roadmap, all proven capabilities |
| `ferrite_gemm_real_weights.rs` | Session 3 kernel tests (test9a-k) |
| `ferrite.rs:375-462` | `launch_fused_norm_gemm` |
| `llama.rs:31-36` | `FUSED_NORM_GEMM` compile! definition |
| `llama.rs:1004-1106` | Standard ferrite forward path (working) |

## Running the tests

```bash
# The failing test (token comparison) — THIS IS THE ONE THAT MATTERS
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test13c -- --nocapture --test-threads=1

# Single-layer decomposition (all pass)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block "test11" -- --nocapture --test-threads=1

# Multi-layer prefill only (passes — failure needs autoregressive decode)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test12 -- --nocapture --test-threads=1

# Decode sanity (standard path only, no fused)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test13b -- --nocapture --test-threads=1
```
