# GPU cost sweep

The ferrite DP solver picks kernels by predicted cost. Predictions
come from `ferrite_forward_macro::target::CostTable`, populated at
proc-macro expansion time from a per-GPU CSV at
`crates/ferrite-cuda-targets/profiles/cost_<gpu>.csv`.

Mispredictions cause wrong picks. Symptoms: a kernel runs much
slower than it should because the DP picked the bigger fused claim
when the singleton was actually faster, or a CUTLASS tile was
chosen when `cutlass_gemv` would have won. The fix is calibration:
re-run the sweep on the target GPU and commit the new CSV.

## When to regenerate

- New GPU joins the fleet (add a `ProfileDef` in
  `crates/ferrite-cuda-targets/src/lib.rs` first, then sweep).
- Sweep grid changed (`NK_SHAPES` / `M_VALUES` in
  `crates/ferrite-cost-sweep/src/gemm_sweep.rs`).
- Kernel set changed (added a CUTLASS tile, new fused epilogue,
  attention backend, etc.).
- `nsys cuda_gpu_kern_sum` shows a kernel taking ≫ predicted time
  (15× off was observed on L40s qwen2.5-3B lm_head before adding
  vocab×hidden rows to the sweep grid).

You don't need to regenerate on every PR — only when the picks
themselves shift, or you're chasing a specific perf gap.

## Running the sweep

```sh
# L40s (sm_89, GDDR6, 142 SMs)
CUDA_PATH=/usr/local/cuda-12.9 \
  cargo run -p ferrite-cost-sweep --features cuda --release --bin gpu_cost_sweep \
  > crates/ferrite-cuda-targets/profiles/cost_l40s_sm89.csv

# L4 (sm_89, GDDR6, 58 SMs)
CUDA_PATH=/usr/local/cuda-12.9 \
  cargo run -p ferrite-cost-sweep --features cuda --release --bin gpu_cost_sweep \
  > crates/ferrite-cuda-targets/profiles/cost_l4_sm89.csv

# H100 (sm_90, HBM3, 132 SMs)
CUDA_PATH=/usr/local/cuda-12.9 \
  cargo run -p ferrite-cost-sweep --features cuda --release --bin gpu_cost_sweep \
  > crates/ferrite-cuda-targets/profiles/cost_h100_sm90.csv
```

The sweep emits `kernel,M,N,K,cost_us` rows to stdout; pipe to the
profile CSV. Stderr carries progress + a final `done` line.

Wall time: ~15-25 minutes per GPU (varies with the `(N, K)` grid
size and the number of `M_VALUES`).

## After regenerating

The CSV is `include_str!`'d at proc-macro expansion time. Rebuild
to pick up the new costs:

```sh
cargo build -p vllm-cli --features cuda --release
```

Sanity check: `vllm ferrite info <arch> <variant>` shows which
Impl was picked per shape — diff vs the pre-sweep dispatch dump to
see what flipped. If a flip is unexpected, sometimes the new
calibration row is noisy (cold-cache outlier) — re-run the sweep
on a quiet GPU. The sweep already runs N warmup iters per shape
and reports the median, but cross-shape contention can still skew
single rows.

## Editing the sweep grid

`NK_SHAPES` in `crates/ferrite-cost-sweep/src/gemm_sweep.rs` lists
every `(N, K)` shape benchmarked. Group additions by arch with a
header comment so future readers can find them. M values come from
`M_VALUES` (currently `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
1024, 2048, 4096]`) and are applied to every shape.

When adding shapes for a new arch:

1. List all GEMM shapes the model uses: `q_proj`, `k_proj`,
   `v_proj` (or packed `(q + 2·kv)` if fused), `o_proj`,
   `gate_proj`, `up_proj`, `down_proj`, `lm_head`.
2. Note `(N, K)` per shape (cuBLAS-convention: weight is `[N, K]`,
   activation is `[M, K]`, output is `[M, N]`).
3. Append rows under an arch header in `NK_SHAPES`.
4. Sweep, commit CSV, rebuild, sanity-check `vllm ferrite info`.

Don't drop existing shapes — other arches share rows by `(N, K)`,
not by name. The CSV is keyed `(kernel, M, N, K)` so duplicates
from independent registrations collapse cleanly.

## Why the predictor's bounding-box matters

`CostTable::get_or_predict` falls back to a fitted 2-param linear
model when the exact `(kernel, M, N, K)` row is absent. The fit is
GLOBAL per-kernel — fitting one set of coefficients across every
`(M, N, K)` point in the sweep. Past the fit's bounding box (with
2× margin), `predict()` returns `None` and the caller falls back
to roofline (see `crates/ferrite-forward-macro/src/target.rs`).

Roofline at out-of-grid shapes is structurally pessimistic but
order-of-magnitude correct, which is better than the previous
behavior (extrapolation off by 15× on lm_head shapes was real).
But roofline can't tell two BW-bound kernels apart at the same
shape — both predict the same time, the DP ties, tiebreak picks
the wrong one. The cure is calibration: add the shape to the
sweep grid.

## File layout

- `crates/ferrite-cost-sweep/src/gemm_sweep.rs` — GEMM-family sweep
  (cuBLAS-replacement Cutlass tiles, GEMV, fused EVT epilogues,
  rms_norm, silu/mul/gelu, fused_qkv_rope_cache).
- `crates/ferrite-cost-sweep/src/attention_sweep.rs` — FA2 +
  FlashInfer attention sweeps over `(num_tokens, sk_bucket)`.
- `crates/ferrite-cost-sweep/src/main.rs` — entrypoint that runs
  both sweeps and emits one merged CSV.
- `crates/ferrite-cuda-targets/profiles/cost_<gpu>.csv` — committed
  per-GPU cost tables; `include_str!`'d at compile time.
- `crates/ferrite-cuda-targets/src/lib.rs` — `ProfileDef` definitions
  binding each `cost_<gpu>.csv` to a GPU-detect signature
  (compute capability + SM count + name match).
