# Handoff — Session 5

## The goal

Fuse `fused_add_rms_norm` into the CUTLASS GEMM prologue so that residual-add +
normalization + matrix multiply happen in a single kernel launch. This replaces
the current two-step approach (`add_inplace` + `launch_fused_norm_gemm`) which
loses f32 precision at the add→norm boundary.

## What we know (proven by tests)

### Root cause: bf16 truncation at add→norm boundary

- **test14c**: The fused norm+GEMM kernel is **bit-exact** (0.00e0) at every layer
  with any input distribution. The kernel itself is innocent.
- **test14d**: `add_inplace` + standalone `rms_norm` + `ferrite.gemm` (NO fused
  kernel at all) flips the same tokens as the fused path. The ONLY difference from
  the working standard path is the bf16 truncation between add and norm.
- **test14a**: 24-layer prefill with synthetic data + argmax → PASS. The bf16
  truncation is tolerable with small synthetic values.
- **test14b**: Same but real embeddings → FAIL (token 6 mismatch, fused winner is
  rank 5 in standard's logits). Real embedding distributions amplify the truncation.

### The fix requires fusing `fused_add_rms_norm` (not `rms_norm`)

`fused_add_rms_norm_inplace` adds res+hs in f32 registers and computes inv_rms from
the f32 sums. Its pass 2 normalizes bf16(sum) — same as the GEMM per-element code
would. So the ONLY precision-critical part is the inv_rms computation from f32 sums.

The perimeter model doesn't need to become a full DAG yet. It just needs to handle
a two-input reduction with writeback — fan-in at the prologue, fan-out at per-site.

### What we built (all working)

1. **Parser**: Already handles `fused_add_rms_norm` correctly — classifies as
   `Reduction`, decomposes into accumulate/finalize/emit, detects `st.global`
   writeback in accumulation loop. **No parser changes were needed.**

2. **Pipeline compiler**: Extended `build_reduction_computation` to detect two-input
   reductions (via `st.global` in accumulation loop) and map params:
   - `_ferrite_rms_input` → param_1 (residual, GEMM A-ptr, writeback target)
   - `_ferrite_rms_hs_input` → param_0 (hidden_states, second input for add)
   - ptxas validates the fused PTX. 107/107 macro tests pass.

3. **Manifest**: `fused_add_rms_norm` added to `ferrite.toml` with entry_hint
   for bf16 template extraction from `vllm_rms_norm.ptx` (already compiled).

4. **Launch function**: `launch_fused_add_norm_gemm()` in `ferrite.rs` with
   two input pointers (residual + hs_input).

5. **Multi-block writeback race fix**: The prologue's `st.global` writeback is
   guarded with `@%p_rms_wb` so only the first N-tile block per m_tile writes.
   Without this, multiple blocks race on the residual buffer (test14g confirmed:
   values get doubled). With the guard: `res_diff=0.00e0` for all grid sizes.

### What's broken: thread-count mismatch in reduction

The transplanted `fused_add_rms_norm` accumulation code runs with 128 threads
(the GEMM's blockDim) instead of the original kernel's 896 threads (min(hidden, 1024)).
The code is thread-count-agnostic (loops with stride = ntid.x), but:

- Different thread count → different reduction order
- Different reduction order → different floating-point sum-of-squares
- Different sum-of-squares → slightly different inv_rms (~3-5e-2 per layer)
- Over 24 layers: tokens diverge completely

**test14g results** (with writeback race fix):
```
N=64   grid_n=1  qkv_diff=0.00e0  res_diff=0.00e0  PASS  (1 block)
N=128  grid_n=1  qkv_diff=0.00e0  res_diff=0.00e0  PASS
N=256  grid_n=2  qkv_diff=3.73e-2 res_diff=0.00e0  FAIL  (reduction order diff)
N=512  grid_n=4  qkv_diff=5.13e-2 res_diff=0.00e0  FAIL
N=1152 grid_n=9  qkv_diff=4.19e-2 res_diff=0.00e0  FAIL
```

Single-block (N≤128): 0.00e0 — the prologue is correct when 128 threads is
the only execution. Multi-block: ~3-5e-2 per layer from reduction order.

**test14e** (24-layer prefill with fix): ALL tokens mismatch. The 3-5e-2 per-layer
diff compounds fatally, same as the original bf16 truncation problem.

## The remaining problem: inter-block read/write aliasing

The `@%p_rms_wb` guard prevents the WRITE race (only one block writes bf16 sums
to residual). But it does NOT prevent block 1 from READING block 0's partially-written
data during block 1's own prologue accumulation.

### What happens

With multiple GEMM blocks sharing the same m_tile:
1. Block 0's prologue reads res[row,k] (=0) and hs[row,k], computes f32(0+hs)=hs,
   writes bf16(hs) to res[row,k] (guarded by `@%p_rms_wb`), accumulates sum-of-squares.
2. Block 1's prologue reads res[row,k] — but block 0 may have ALREADY written bf16(hs)
   to some elements. Block 1 sees a MIX of original zeros and block 0's hs values.
3. Block 1 computes inv_rms from corrupted (mixed) data → wrong inv_rms → wrong QKV.

**This is NOT a thread-count or reduction-order issue.** Single-block gives 0.00e0
because there's no other block to race with. The 128-thread reduction order is
identical to 896-thread (hidden=896, num_vecs=112, each thread does exactly 1
iteration regardless of ntid.x).

### Why `@%p_rms_wb` doesn't fix it

The guard prevents block 1 from WRITING, but block 1's prologue still READS from
the same buffer that block 0 writes to. The prologue does:
```
ld.global [residual + offset]  // READ — sees block 0's write (partial)
add.f32 ...                     // compute
@%p_rms_wb st.global [residual + offset]  // WRITE — only block 0
fma.rn.f32 ...                  // accumulate sum-of-squares from corrupted read
```

### The fix: eliminate GMEM aliasing in the prologue

The prologue must not write to any buffer it also reads from across blocks.
Two approaches:

**A. On-the-fly sums at per-site (no prologue writeback)**

- Prologue: reads from residual AND hs. Computes inv_rms. Does NOT write to GMEM.
  All blocks see the same original (unmodified) data → all compute the same inv_rms.
- Per-site: at each A-load site, loads from BOTH residual and hs (same K offset),
  adds in f32. The f32 sum goes through per-element normalization (mul inv_rms, mul weight).
  No GMEM write — sums stay in registers, go to SMEM for GEMM.
- Post-kernel: the caller runs `add_inplace(residual, hs)` as a separate kernel
  to update the residual for downstream layers.
- **Pro**: No race, no extra buffer. Prologue reads are pure reads from both buffers.
- **Con**: per-site and per-element need to handle two inputs (adds ~10 PTX lines).
  Post-kernel `add_inplace` is an extra launch (but trivial bandwidth cost).

**B. Separate output buffer**

- Launch function allocates a temp buffer, copies residual → temp.
- Prologue reads from temp (original residual) and hs. Writes bf16 sums to temp.
  All blocks write the same values (benign race on temp).
- A_ptr = temp (GEMM reads normalized sums from temp).
- Post-kernel: copies temp → residual.
- **Pro**: No code changes to prologue or per-site.
- **Con**: Extra alloc + 2 copies. Benign races still happen (all blocks write same
  values, but writes are idempotent). L1 coherency might still cause issues.

### Recommended: Option A

Option A is cleaner — no extra buffers, no copies, no races. It requires:
1. Remove all `st.global` from the transplanted prologue (or skip transplanting them).
2. Add per-site code to load from hs at the same K offset (similar to existing
   weight loading), unpack 8 bf16 → f32 in `%f_rms_hs0..7`.
3. Add per-element instruction: `add.f32 {INPUT}, {INPUT}, %f_rms_hs{ELEM_IDX}`
   before the existing `mul inv_rms, mul weight`.
4. Update `launch_fused_add_norm_gemm` to call `add_inplace(residual, hs)` after
   the kernel launch for the residual writeback.

The per-element add operates on f32(bf16(res)) + f32(bf16(hs)) — the exact f32 sum
of the two bf16 inputs. This is actually BETTER precision than `fused_add_rms_norm_inplace`
pass 2, which normalizes bf16(f32(res)+f32(hs)) — a value that went through an extra
bf16 truncation. For production this is fine; for test comparisons against the standard
path, there may be a small diff from this precision improvement.

### Critical subtlety: why test14c showed 0.00e0

The existing single-input `FUSED_NORM_GEMM` (rms_norm → GEMM) is bit-exact with the
standard path (test14c: 0.00e0 at all M values) because:
1. The old fused path does `add_inplace(res, hs)` → res = bf16(hs+res). Then
   `launch_fused_norm_gemm(res)` reads bf16(hs+res) and norms it.
2. The standard path does `fused_add_rms_norm_inplace(hs, res)` which writes
   bf16(hs+res) to res, then pass 2 reads bf16(hs+res) from res and norms it.
3. Both norm the SAME bf16 values with the SAME inv_rms computation (since both
   read from the SAME bf16 buffer after the add). Hence 0.00e0.

The inv_rms from `fused_add_rms_norm_inplace` uses f32(hs+res) for sum-of-squares,
while the old fused kernel uses bf16(hs+res). For res=0 (test14c initial conditions),
f32(0+hs) = f32(hs) = f32(bf16(hs)) — IDENTICAL, because hs is already bf16. So
inv_rms matches too. The 0.00e0 result is specific to res=0 initial conditions.

## Methodology

**One variable at a time.** Session 5 established this methodology and used it to
isolate the root cause. Every test changes exactly one variable from the last passing
test. The test chain:

```
test14a (24L prefill, synthetic, argmax)     → PASS
test14b (24L prefill, REAL embeddings, argmax) → FAIL (token 6)  ← changed: input data
test14c (1L QKV, real embeddings, per-layer)   → PASS (0.00e0)   ← changed: single layer
test14d (24L prefill, real, NO fused kernel)   → FAIL (same token 6) ← changed: removed fused kernel
test14f (1L QKV, real, FUSED_ADD_NORM_GEMM)   → varies by M      ← changed: new kernel
test14g (1L QKV, real, sweep N for grid size)  → PASS N≤128, FAIL N≥256 ← changed: grid blocks
```

## Key files

| File | What |
|------|------|
| `tests/test_transformer_block.rs` | All session 4+5 tests (test10-14g) |
| `test14g` | **THE diagnostic test** — sweeps N to vary block count |
| `test14e` | 24-layer end-to-end test (currently fails) |
| `test14b` | Original root cause proof (bf16 truncation flips tokens) |
| `test14c` | Proves fused kernel is bit-exact at single layer |
| `test14d` | Proves bf16 truncation alone flips tokens |
| `pipeline_compile.rs` | Two-input reduction support + writeback guard |
| `ferrite.rs:473-597` | `launch_fused_add_norm_gemm` |
| `ferrite.toml` | `fused_add_rms_norm` manifest entry |
| `vllm_rms_norm.ptx` | Contains bf16 `fused_add_rms_norm_kernel` (already compiled) |
| `layernorm_kernels.cu:105-165` | Original CUDA source for `fused_add_rms_norm_kernel` |

## Running the tests

```bash
# Sweep N to vary block count (THE diagnostic test)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test14g -- --nocapture --test-threads=1

# 24-layer end-to-end (currently fails — reduction order diff)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test14e -- --nocapture --test-threads=1

# Single-layer debug (M=8 real embeddings)
cargo test -p vllm-cuda --features ferrite --test test_transformer_block test14f -- --nocapture --test-threads=1

# Root cause proof tests
cargo test -p vllm-cuda --features ferrite --test test_transformer_block "test14[abcd]" -- --nocapture --test-threads=1

# All pipeline compiler tests (should be 107+ pass)
cargo test -p ptx-fusion-macros -- --nocapture
```

## TODOs (in order)

### 1. Implement Option A: on-the-fly sums at per-site

Modify `build_reduction_computation` (in `pipeline_compile.rs`):
- Skip transplanting `st.global` lines in the prologue (the writeback stores)
- Add per-site code: load 8 bf16 from `_ferrite_rms_hs_input` at the same K offset
  as the A-load, unpack to f32 in `%f_rms_hs0..7`
- Add per-element instruction: `add.f32 {INPUT}, {INPUT}, %f_rms_hs{ELEM_IDX}`
  BEFORE the existing `mul inv_rms, mul weight`
- Update `launch_fused_add_norm_gemm` to call `add_inplace(residual, hs)` after launch

### 2. Re-run test14g

All N values should show 0.00e0 (no inter-block aliasing).

### 3. Re-run test14e

24-layer prefill with real embeddings. Tokens should match (or be close enough
for coherent text). If there's a small diff from the precision improvement
(f32(bf16(res)) + f32(bf16(hs)) vs bf16(f32(res)+f32(hs))), that's expected
and should be tolerable.

### 4. Wire into llama.rs + test with `vllm serve`

Only after test14e passes.

## Gotchas

### Shared CachingAllocator in tests

Running multiple forward paths through the same `device.caching` allocator can
cause false test failures. Path C's allocations may reuse memory from path B's
tensors before they're downloaded. Always download results before allocating more
tensors, or use separate GpuDevice instances for each path (see test13c).

### Tests must use --test-threads=1

Multiple tests sharing one GPU deadlock at 300% CPU.

### Cudaforge stale cache

Shared .a files get clobbered across worktrees. Run `rm -f crates/cudaforge/lib*.a`
to force rebuild when switching worktrees or changing kernel code.

## Rules (non-negotiable)

- **NEVER write PTX by hand** — transplant from compiled kernel PTX
- **NEVER build special-case macros** — extend `compile!`
- **NEVER dismiss divergence** — 3e-2 per layer compounds to garbage over 24 layers
- **Tests ARE the product** — test14g is the diagnostic, test14e is the goal
- **One variable at a time** — every new test changes exactly one thing from the last
- **Do NOT touch llama.rs** until test14e passes
- **Handoffs state FACTS** — clearly separate known from unknown, never speculate
