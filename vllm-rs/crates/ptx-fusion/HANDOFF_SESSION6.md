# Handoff — Session 6

## The goal

Same as session 5: fuse `fused_add_rms_norm` into the CUTLASS GEMM prologue so
that residual-add + normalization + matrix multiply happen in a single kernel
launch. Session 5 identified the inter-block aliasing problem. Session 6 fixed
it, then hit a dead end trying to make the split architecture bit-exact with
the standard path.

## What was done

### 1. Proved the root cause: inter-block read/write aliasing

**test14h** (new) compares GEMM output per-block-range for N=256 (grid_n=2):

```
row  blk0_maxdiff  blk1_maxdiff  ratio_gap
  0      0.0000e0     1.5381e-2   2.9240e-1
  1      0.0000e0     9.5215e-3   8.8725e-1
  3      0.0000e0     2.2705e-2   3.4568e-1
  7      0.0000e0     3.6179e-2   9.2555e-1

Summary: blk0=0.00e0  blk1=3.62e-2
→ ALIASING: block 0 correct, block 1 diverged
```

Block 0 (the first N-tile, which does the writeback) is always correct. Block 1
reads from the residual buffer WHILE block 0 writes to it. Block 1 sees a mix
of original zeros and block 0's bf16(hs) values → corrupted sum-of-squares →
wrong inv_rms. The `@%p_rms_wb` guard from session 5 prevented the WRITE race
but not the READ race.

### 2. Implemented Option A: pure-read prologue

Changes in `pipeline_compile.rs`:

- **Skip st.global entirely** for two-input reductions (`has_writeback=true`).
  Previously guarded with `@%p_rms_wb`; now the lines are not emitted at all.
  The prologue reads from both buffers but writes nothing to GMEM. All blocks
  see identical original data → identical inv_rms.

- **Per-site hs_input loading**: At each A-load site, the per-site code now
  loads 8 bf16 from `_ferrite_rms_hs_input` at the same K offset as the A-load.
  Uses `%rd_rms_hs_rb0/rb1` (hs row bases, computed from residual row bases by
  swapping the base pointer). Unpacks to 8 f32 values in `%f_rms_hs0..7`.

- **Per-element add + bf16 truncation + normalize**: Before the existing
  `mul inv_rms, mul weight`, inserts:
  ```ptx
  add.f32          {INPUT}, {INPUT}, %f_rms_hs{ELEM_IDX}  // res + hs in f32
  cvt.rn.bf16.f32  %h_rms_a, {INPUT}                       // truncate to bf16
  cvt.f32.bf16     {INPUT}, %h_rms_a                        // promote back
  ```
  The bf16 round-trip matches the standard path's precision (fused_add_rms_norm
  writes bf16(sum) to memory, pass 2 reads it back). Without this truncation,
  the fused path diverges at layer 3 instead of layer 9.

- **Caller does add_inplace**: `launch_fused_add_norm_gemm` no longer calls
  `add_inplace` internally. The caller must do it after the launch, before
  dropping the hs_input tensor. This avoids a GPU use-after-free when the
  caching allocator reuses hs_input's memory before add_inplace executes.

Changes in `ferrite.rs`:

- Updated `launch_fused_add_norm_gemm` comments documenting the pure-read
  prologue and caller responsibility for add_inplace.
- Added `CU_JIT_FTZ=1` to `cuModuleLoadDataEx` for JIT compilation. (This
  was investigated as a potential fix for the layer-9 drift but had no effect.
  Left in because matching source kernel FTZ is correct regardless.)

### 3. Results after the fix

**test14g** (multi-block sweep): ALL 0.00e0

```
N=   64 grid_n=1 diff=0.00e0 res_diff=0.00e0 PASS
N=  128 grid_n=1 diff=0.00e0 res_diff=0.00e0 PASS
N=  256 grid_n=2 diff=0.00e0 res_diff=0.00e0 PASS
N=  512 grid_n=4 diff=0.00e0 res_diff=0.00e0 PASS
N= 1152 grid_n=9 diff=0.00e0 res_diff=0.00e0 PASS
```

**test14h** (inter-block diagnostic): BOTH blocks 0.00e0

**test15a** (dimension sweep, new): ALL 0.00e0

```
hidden=  896 diff=0.00e0 PASS
hidden= 1024 diff=0.00e0 PASS
hidden= 1536 diff=0.00e0 PASS
hidden= 2048 diff=0.00e0 PASS
hidden= 2560 diff=0.00e0 PASS
hidden= 3584 diff=0.00e0 PASS
```

**test14e** (24-layer end-to-end): 7/8 tokens match, 1 mismatch

```
token 0: MATCH  hidden_diff=0.00e0
token 1: MATCH  hidden_diff=0.00e0
token 2: MATCH  hidden_diff=1.95e-1
token 3: MATCH  hidden_diff=2.65e-1
token 4: MISMATCH  hidden_diff=1.45e-1
token 5: MATCH  hidden_diff=2.70e-1
token 6: MATCH  hidden_diff=2.50e-1
token 7: MATCH  hidden_diff=6.25e-2
```

## The dead end: per-layer drift from the split architecture

### What we investigated

test14i (new) tracks per-layer divergence between the fused and standard paths:

```
layer  qkv_diff  attn_diff  mlp_diff  res_diff  res@qkv
    0    0.00e0     0.00e0    0.00e0    0.00e0    0.00e0
  ...
    8    0.00e0     0.00e0    0.00e0    0.00e0    0.00e0
    9   7.81e-3    7.81e-3   7.81e-3   7.81e-3    0.00e0  ←
   10   3.12e-2    9.77e-3   9.28e-3   3.12e-2   1.56e-2  ←
  ...
   23   1.64e-1    2.66e-1   2.70e-1   5.00e-1   4.06e-1  ←
```

Key observations:
- Layers 0–8: perfectly 0.00e0 on every metric
- Layer 9: first divergence, exactly 7.81e-3 (1 bf16 ULP at ~1.0)
- `res@qkv = 0.00e0` at layer 9: the residuals are identical AFTER the add.
  The divergence is in the QKV GEMM output itself.
- Divergence is only in row 2, spread across multiple blocks/columns
- Standard-vs-standard: 0.00e0 for ALL 24 layers (no non-determinism)

### What we ruled out

| Hypothesis | Test | Result |
|------------|------|--------|
| Thread count mismatch (128 vs 896) | test14g single-layer, all N | 0.00e0 — same work distribution for hidden=896 |
| FTZ mismatch | Added CU_JIT_FTZ=1 to JIT | No change — identical results |
| Non-determinism | Standard-vs-standard 24 layers | 0.00e0 — fully deterministic |
| Inter-block aliasing (still) | res@qkv check at layer 9 | 0.00e0 — residuals identical |
| Multiplication order | Checked source PTX pass 2 | Both do `(input * inv_rms) * weight` |
| bf16 conversion difference | Compared cvt.rn.bf16.f32 vs inline asm | Same instruction |

### What we couldn't pin down

The divergence appears at layer 9 for specific input values (row 2 only). The
per-element ratios across columns are NOT constant, ruling out a pure inv_rms
difference — individual normalized elements differ. CPU computation of the
expected values doesn't match the GPU (because `rsqrt.approx.f32` differs from
IEEE sqrt), making it impossible to use CPU as reference.

The most likely remaining cause: the transplanted prologue's reduction computes
a slightly different sum-of-squares than the C kernel for certain input value
distributions, despite using identical PTX instructions. This could be from
subtle SASS-level differences in how the JIT compiler optimizes the transplanted
code vs how nvcc compiled the original, or from SMEM layout interactions we
haven't identified.

### Why we stopped investigating

The split architecture — prologue (reduction) running separately from per-site
(normalize at A-loads) — is inherently different from the standard path where
both phases run in the same kernel with shared state. This split is **temporary
scaffolding**. The ultimate fused forward pass has norm running BETWEEN two
GEMMs in a persistent kernel:

```
GEMM_down epilogue → residual_add + rms_norm → GEMM_qkv prologue
```

In that architecture, pass 1 and pass 2 stay coupled — no split, no seam, no
per-element hand-written PTX. Making the split bit-exact is wasted effort
because the split won't exist in the final design.

### The hand-written PTX violation

The per-element instructions (`add.f32`, `cvt.rn.bf16`, `mul inv_rms`,
`mul weight`) are **hand-written PTX**, violating Ferrite's core rule: never
write PTX by hand, always transplant from compiled kernel PTX. The
`ReductionDecomposition` already extracts the emit body (pass 2) from the
source kernel — 65 lines of compiled PTX containing the exact normalization
sequence. The pipeline compiler should be using those extracted instructions
rather than generating its own.

This violation is the likely root cause of the per-layer drift. But fixing it
within the split architecture still wouldn't carry forward, because the split
itself is temporary.

### Production wiring attempt

Wired `launch_fused_add_norm_gemm` into llama.rs for Qwen 3B. Results:
- CUDA_ERROR_ILLEGAL_ADDRESS during CUDA graph capture (probably M=1 decode)
- Garbage output with `--enforce-eager`

test15a shows 0.00e0 at all hidden dimensions (896–3584) with M=8, so the
kernel works in isolation. The production failure is in the wiring — wrong
tensor lifetimes, missing edge cases (M=1, first layer with no residual), or
something about the real model's tensor layout. We reverted llama.rs rather
than debug the wiring for a temporary architecture.

## What carries forward to the inter-GEMM pipeline

| Asset | Status | Carries forward? |
|-------|--------|-----------------|
| Aliasing analysis (test14h) | Proven | Yes — any multi-block prologue with writeback has this issue |
| Pure-read prologue pattern | Working | Yes — the pattern of reading without writing is correct |
| Prologue transplant (pass 1 reduction) | Working, 0.00e0 | Yes — the extracted reduction code runs correctly |
| Per-site dual-buffer loading | Working | Partially — the K-offset computation is reusable |
| Per-element hand-written PTX | Working but drifts | **No** — must use transplanted emit body |
| launch_fused_add_norm_gemm | Working in tests | **No** — split architecture is temporary |
| test14e 24-layer comparison | 7/8 match | Yes — useful as regression test |
| test15a dimension sweep | All 0.00e0 | Yes — validates across hidden sizes |

## Key files

| File | What |
|------|------|
| `pipeline_compile.rs` | Option A implementation: pure-read prologue + per-site hs loading |
| `ferrite.rs:473–570` | `launch_fused_add_norm_gemm` (caller does add_inplace) |
| `test_transformer_block.rs` | test14h (aliasing proof), test14i (per-layer tracker), test15a (dim sweep) |
| `HANDOFF_SESSION5.md` | Prior session: root cause analysis, two hypotheses |

## Running the tests

```bash
# Multi-block sweep (THE correctness test — all N values must be 0.00e0)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test14g -- --nocapture --test-threads=1

# Inter-block aliasing diagnostic
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test14h -- --nocapture --test-threads=1

# Dimension sweep (hidden=896 through 3584, all must be 0.00e0)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test15a -- --nocapture --test-threads=1

# 24-layer end-to-end (currently 7/8 tokens match — known drift from split)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test14e -- --nocapture --test-threads=1

# Per-layer divergence tracker (shows where drift starts)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test14i -- --nocapture --test-threads=1

# All pipeline compiler tests (107 must pass)
cargo test -p ptx-fusion-macros
```

## Ultimate goal

**One persistent kernel for the entire transformer forward pass.** Not 6
launches, not 11 — a single kernel launch that runs all layers: norm, GEMM,
attention, MLP, residual connections. Every intermediate value stays in
registers or SMEM. GMEM is only touched for model weights and KV cache.

This is what Ferrite exists to build. Everything else — the pairwise fusion
tests, the prologue/epilogue injection, the pipeline compiler — is scaffolding
toward this single-launch architecture.

## Next: Phase 5G — Inter-GEMM pipeline

The split architecture (norm prologue in one GEMM, norm per-element at A-loads)
is a dead end. **Skip Phase 5E** (wiring the split into llama.rs). Go directly
to Phase 5G: the inter-GEMM pipeline where norm runs between two GEMMs in one
persistent kernel.

The correct architecture for one layer:

```
[persistent kernel, single launch]
  GEMM_qkv → split/RoPE/KV → attention → GEMM_o → residual_add
  → rms_norm → GEMM_gate_up → SiLU+mul → GEMM_down → residual_add
  → [next layer]
```

The norms stay coupled (pass 1 reduction + pass 2 normalize in one phase).
Values flow through SMEM between GEMMs. No split, no hand-written PTX, no
GMEM round-trip for intermediates.

The stepping stone is the two-GEMM case: `GEMM_down → norm → GEMM_qkv` in
one persistent kernel. This requires:

1. **Driver loop**: persistent kernel with work-queue tile dispatch
2. **Global barrier**: synchronize all blocks between GEMM phases
3. **Inter-GEMM SMEM handoff**: down epilogue → SMEM → norm → SMEM → QKV prologue
4. **Residual writeback**: one block writes bf16(res+hs) to GMEM (for downstream
   attention), guarded to avoid races (same pattern as `@%p_rms_wb` but safe
   because norm reads from SMEM, not GMEM)

The persistent kernel infrastructure already exists (Phase 5D proved it with
`pipeline_fuse!`). The two-phase GEMM + global barrier was proven in
`cuda_fuse_general::test_two_phase_persistent`. What's new is the norm phase
between the GEMMs.

## Rules (non-negotiable, carried forward from session 5)

- **NEVER write PTX by hand** — transplant from compiled kernel PTX
- **NEVER build special-case macros** — extend `compile!`
- **NEVER dismiss divergence** — 3e-2 per layer compounds to garbage over 24 layers
- **Tests ARE the product** — test14g/test15a are correctness gates
- **One variable at a time** — every new test changes exactly one thing
- **Handoffs state FACTS** — clearly separate known from unknown
