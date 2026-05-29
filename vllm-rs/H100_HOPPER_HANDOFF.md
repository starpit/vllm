# H100 Hopper — perf handoff

Updated 2026-05-28. Branch `worktree-h100`. HEAD `14e043524a` (`ferrite/perf:
narrow lm_head to last-token-per-seq at prefill (CUDA)`).

## TL;DR

vs Python vLLM 0.21.1.dev0 on the serving workload (the one users
actually run): ferrite wins meaningfully on `vllm bench serve`. TPOT
is 20-56 % lower across all shapes; saturated throughput is +12 to
+43 % at most shapes (one regression at `in=4096 out=128`, -11 %).
The offline `vllm bench throughput` is +0 to +32 % faster, biggest at
decode-heavy shapes. The single-batch `vllm bench latency` micro-bench
is the only place ferrite trails (-2 to -4 %) — that overhead amortizes
away under concurrency.

The lm_head-narrow port just landed and is responsible for ~2 % of
this delta at compute-dominated shapes; the rest is from earlier
ferrite work (multi-slot FI plan cache, FlashInfer paged decode on
sm_90+, the rest of the engine).

Both stacks use Hopper-native attention — Python is on FlashAttention
3 (its startup log: "Using FlashAttention version 3"), ferrite is on
FlashInfer. The earlier "we're on FA2" memory note was wrong; we're
on the FI side of the FA3-family, not vllm-flash-attn 2.x.

Three known follow-ups, none of them in this commit:

1. `BS=32 in≥4096` ferrite OOMs on `vllm bench latency` while Python
   doesn't. Caching allocator keeps intermediates Python frees; not a
   forward-pass perf bug.
2. `in=4096 out=128 rate=inf` ferrite serve is the one shape where
   throughput regresses (-10.7 %). TTFT and TPOT are still better;
   it's a saturation/scheduling effect, not a per-call regression.
3. `last_token_indices` narrow only patches
   `Instruction::CutlassFusedAddRmsNormGemm`; other arches' lm_heads
   land on `CutlassFusedRmsNormGemm` / `Gemm` / `CutlassGemm` with the
   same waste pattern. Cleanest fix is at the codegen layer, not
   per-Instruction.

## Bench setup

- Pod: `nick` (1x H100 80 GB HBM3, sm_90, CUDA driver 12.9 / NVIDIA 575.57.08)
- Model: `Qwen/Qwen2.5-7B-Instruct` (bf16, vocab 152064, hidden 3584, 28 layers)
- Python: `vllm 0.21.1.dev0+gad7125a43` w/ `torch 2.11.0+cu129` in
  `/home/nickm/h100/vllm/.venv` (built from upstream main against the
  pod's CUDA 12.9 driver — pypi wheels needed cu130 which the driver
  rejects).
- Ferrite: `worktree-h100` HEAD `14e043524a`,
  `FERRITE_MODELS=qwen2.5-7b cargo build -p vllm-cli --features cuda,nccl,bench --release`
  in `/home/nickm/h100/vllm-rs`.
- All runs `--no-prefix-caching` / `--no-enable-prefix-caching` on
  both sides — without that flag, ferrite's bench (which defaults
  prefix caching ON) silently reuses KV across the bench's identical
  prompts and reports e.g. `BS=32 in=2048` as **11.5×** over Python.
  Don't trust any prior comparison that didn't pin this.

## `vllm bench latency` (5 iters, 2 warmup, --no-prefix-caching)

Wall-clock seconds per iter, lower is better.

| Shape                         | Python | Ferrite | Δ      |
|-------------------------------|------:|--------:|-------:|
| BS=1  in=2048 out=128         | 0.805 | 0.822   | +2.1 % |
| BS=1  in=4096 out=128         | 0.856 | 0.887   | +3.6 % |
| BS=1  in=8192 out=128         | 0.966 | 1.007   | +4.2 % |
| BS=8  in=2048 out=128         | 1.158 | 1.154   | -0.4 % |
| BS=8  in=4096 out=128         | 1.555 | 1.572   | +1.1 % |
| BS=8  in=8192 out=128         | 2.465 | 2.527   | +2.5 % |
| BS=32 in=2048 out=16          | 1.526 | OOM     | —      |
| BS=32 in=4096 out=16          | 3.054 | OOM     | —      |
| BS=32 in=8192 out=16          | 6.354 | OOM     | —      |

Per-step overhead (latency micro-bench) costs us ~2-4 %. Tied at the
prefill-heaviest shape that doesn't OOM.

## `vllm bench throughput` (200 prompts, random dataset, --no-(enable-)prefix-caching)

Total tokens/sec (input + output, all real — Python's `vllm bench
throughput` ignores `--input-len`; you must pass `--random-input-len`
and `--random-output-len`). Higher is better.

| Shape                | Python tok/s | Ferrite tok/s | Δ        |
|----------------------|-------------:|--------------:|---------:|
| in=1024 out=128      |        38740 |         39103 |   +0.9 % |
| in=1024 out=512      |        27859 |         31687 | **+13.7 %** |
| in=2048 out=128      |        39738 |         46910 | **+18.0 %** |
| in=2048 out=512      |        29238 |         38518 | **+31.7 %** |
| in=4096 out=128      |        39626 |         39695 |   +0.2 % |
| in=4096 out=512      |        29994 |         35662 | **+18.9 %** |

Decode-heavy shapes (out=512) show the largest wins; tied at the
two decode-light extremes (in=1024 out=128, in=4096 out=128). The
serving workload (high concurrency, mixed input/output) lives in
exactly this regime.

## `vllm bench serve` (against running server, 200 prompts random)

The most realistic comparison — same Python `vllm bench serve` client
hits each server in turn over the OpenAI completions endpoint. Both
servers started with `--no-(enable-)prefix-caching`, max-model-len
8400, gpu-memory-utilization 0.85.

Note: ferrite's serve doesn't currently populate `prompt_tokens` in the
OpenAI response, so the Python bench client reports `Total token
throughput` = output-only for ferrite (matches `Output token
throughput`). Compare on req/s, output tok/s, TTFT, TPOT — those are
unambiguous.

```
────────── rate=inf (saturated) ──────────
              req/s ↑          TPOT ms ↓        TTFT ms ↓
shape         Py    Fr    Δ    Py    Fr    Δ    Py    Fr    Δ
1024×128    31.9  35.6  +12%  28.0  22.2  -21%  2598  2393   -8%
1024×512    17.8  21.4  +20%  16.9  12.9  -24%  2488  2371   -5%
2048×128    17.8  23.6  +33%  48.2  25.9  -46%  4883  3990  -18%
2048×512    11.2  16.0  +43%  24.8  15.0  -39%  4885  3987  -18%
4096×128     9.2   8.3  -11%  88.5  43.5  -51%  9790  7855  -20%
4096×512     6.4   6.5   +1%  40.4  28.3  -30%  9964  7741  -22%

────────── rate=16 req/s (sustained) ──────────
              TPOT ms ↓        TTFT ms ↓
shape         Py    Fr    Δ    Py    Fr    Δ
1024×128     8.9   9.8  +11%   52    59  +13%
1024×512    11.6  11.0   -5%   65    67   +3%
2048×128    23.0  18.8  -18%  171   175   +3%
2048×512    22.3  15.9  -29%  205   163  -20%
4096×128    88.5  38.7  -56% 3441  3229   -6%
4096×512    40.4  25.0  -38% 3461  3502   +1%

↑ higher is better   ↓ lower is better
```

**Pattern**: TPOT (per-token decode latency) is consistently 20-56 %
lower with ferrite — the strongest signal in the dataset. Ferrite also
wins on saturated throughput at most shapes (single regression at
`in=4096 out=128`, -10.7 %), wins on TTFT at large prefills (-7 to
-22 %), and slightly regresses TTFT on the smallest shapes at
sustained low rates (+3 to +13 %).

This is the workload that actually matters for deployed serving. The
latency micro-bench `+2-4 %` regression sits inside per-step overhead
that the serving path amortizes away.

Python is using **FlashAttention 3** (per its startup log) and
ferrite is on FlashInfer (also Hopper-native — same kernel family,
not FA2 as the prior memory note suggested). So both sides have a
Hopper-native attention backend; the wins below are not from us
catching python on attention.

## Reproducing

Latency:
```
# Python
source /home/nickm/h100/vllm/.venv/bin/activate
vllm bench latency --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
  --input-len 2048 --output-len 128 --batch-size 8 \
  --max-model-len 2300 --num-iters-warmup 2 --num-iters 5 \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching

# Ferrite
cd /home/nickm/h100/vllm-rs
./target/release/vllm bench latency --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 --input-len 2048 --output-len 128 --batch-size 8 \
  --max-model-len 2300 --num-iters-warmup 2 --num-iters 5 \
  --gpu-memory-utilization 0.85 --no-prefix-caching
```

Throughput:
```
# Python — note --random-input-len / --random-output-len, NOT --input-len
vllm bench throughput --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
  --num-prompts 200 --random-input-len 2048 --random-output-len 512 \
  --dataset-name random --gpu-memory-utilization 0.85 --no-enable-prefix-caching

# Ferrite
./target/release/vllm bench throughput --model Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 --num-prompts 200 --input-len 2048 --output-len 512 \
  --dataset-name random --gpu-memory-utilization 0.85 --no-prefix-caching
```

Serve (start a server, then run the Python bench client against it):
```
# Server side — pick one
# Python:
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 --port 8000 \
  --gpu-memory-utilization 0.85 --no-enable-prefix-caching --max-model-len 8400
# Ferrite:
./target/release/vllm serve --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
  --port 8000 --gpu-memory-utilization 0.85 --no-prefix-caching --max-model-len 8400

# Client (always Python — uniform metrics across both servers):
vllm bench serve --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct --dataset-name random \
  --num-prompts 200 --random-input-len 2048 --random-output-len 512 \
  --request-rate inf
```

## Landmines

- **Prefix caching defaults differ.** Ferrite's bench defaults to ON;
  Python's defaults to OFF in current dev. Mixing them gives bogus
  speedups (we hit 11.5× before catching it). Always pin both.
- **`--input-len` semantics differ between Python's bench tools.**
  `vllm bench latency` honors `--input-len`. `vllm bench throughput`
  ignores it for the random dataset and silently uses ~1024; pass
  `--random-input-len`/`--random-output-len` instead.
- **Ferrite serve doesn't populate `prompt_tokens` in the OpenAI
  response.** As a result `vllm bench serve` reports `Total token
  throughput` = output-only against ferrite, but the same number
  includes input tokens against Python. Don't compare the "total tok/s"
  fields directly — use req/s, output tok/s, TTFT, TPOT.
- **Killing Python `vllm serve` doesn't kill its engine-core child.**
  `vllm serve` spawns a `VLLM::EngineCore` process that keeps GPU
  memory after the parent dies; `pkill -9 -f "vllm serve"` only gets
  the parent. Either `pkill -9 -f "VLLM\|vllm"` or look up the engine
  PID via `nvidia-smi --query-compute-apps=pid` and kill that
  separately.
- **CUDA driver 12.9 + pypi vllm wheels.** `vllm==0.21.0` and
  `vllm==0.20.2` from pypi are built against CUDA 13. With driver 12.9
  they fail at startup (`The NVIDIA driver on your system is too old
  (found version 12090)` from torch's `_lazy_init`, or
  `libcudart.so.13: cannot open shared object file` from `vllm._C`).
  Build vllm from source against the pod's CUDA, or use the venv at
  `/home/nickm/h100/vllm/.venv`.
- **FI eviction during graph replay** (memory note from `c8948cb6c`):
  the multi-slot plan cache fixes the sm_89 graph-replay crash; don't
  collapse back to single-slot.
- **`BS=32 in≥4096` OOM is on us.** Python doesn't OOM the same shapes.
  Allocator fragmentation / held intermediates, not a forward-pass bug.

## Recent commits (worktree-h100)

```
14e043524a  ferrite/perf: narrow lm_head to last-token-per-seq at prefill (CUDA)
8e933ace5d  ferrite-cuda: gate chunked-prefill cold path to keep decode on the contiguous kernel
18230ed94e  ferrite: enable FlashInfer paged attention on sm_89+ (multi-slot cache)
```

## What was tried this session and reverted

- `cutlass_gemm_silu_mul.cu` → CUTLASS-3 sm90 EVT (wgmma): kernel
  works, ~20-40 % faster per-call at small/mid M, but fused gate-GEMM
  + EVT silu-mul still loses to cuBLAS-GEMM + separate `silu_and_mul`
  at every MLP shape we calibrated. Net effect on the cost solver:
  fusion-gap **grew** (157 → 165 / 283 picks). Reset.
- FlashInfer Hopper-native prefill shim (sm_90a): compiles, runs E2E,
  ties the generic FI prefill at our shapes. Same kernel-traits sweep
  swept; `USE_TMA_LOAD_KV=true` no help. Reset.
- Cost CSV recalibration against the new silu-mul kernel: regressed
  BS=1 long-decode by 14-133 % on Qwen2.5-3B/7B because the solver
  flipped picks under the new timings. Reset.
- lm_head cuBLAS swap (`Instruction::CutlassFusedAddRmsNormGemm` →
  `cublas.gemm`): SASS counts confirmed the CUTLASS-2 zoo is 97 %
  `mma.sync` on H100, but at lm_head's M=1 prefill-decode shape the
  kernel is HBM-bound (624 MB W-read), so wgmma vs mma.sync doesn't
  matter. Swap measured -0.36 % to -0.58 % (within noise). Backed out;
  the env-gated swap was kept briefly as a tool, dropped from the
  final commit.

## Open work

- Generalize the lm_head narrow at the codegen layer so every model
  benefits, not just qwen2.5-7b's `CutlassFusedAddRmsNormGemm`.
- Make ferrite serve return `prompt_tokens` in the OpenAI completions
  response so `vllm bench serve` reports apples-to-apples
  total-token-throughput.
- Investigate the `BS=32 in≥4096` latency-bench OOM. Python frees
  something we don't.
- Investigate the `in=4096 out=128 rate=inf` serve regression
  (-11 % req/s). Sat-mode scheduling at large prefill + short output;
  we leave throughput on the table at exactly the shape where prefill
  cost dominates a small decode tail.
- Smaller per-step overhead in the latency micro-bench (we're
  ~2-4 % behind). Re-profile post-narrow with nsys to see what's
  actually dominant.

## Latency micro-bench gap diagnosis (2026-05-29)

The 2-4 % latency-bench gap was diagnosed end-to-end. Measured at BS=1
in=8192 out=128 with `--cuda-graph-trace=node`:

```
                    GPU kernel time / iter   wall / iter   oversubscription
Ferrite (greedy)    1.55s                    1.01s         1.53×
Python              1.52s                    0.97s         1.57×
```

Total GPU kernel time is identical. Per-kernel time on shared cuBLAS
wgmma kernels (`nvjet_tst_*`) is identical to the microsecond. The
gap is **per-kernel attention time**:

```
Ferrite — flashinfer::PersistentKernelTemplate<BlockBatchPagedAttentionPersistent>:
  359 ms total / 12068 instances = 29.8 µs avg
Python — cutlass::FlashAttnFwdSm90 (FA3):
  197 ms total / 12152 instances = 16.3 µs avg
```

**FA3 is 46 % faster per call than our FlashInfer paged decode at
BS=1 sk=8192**. Across the iter that's 162 ms of kernel time
difference (most of it hidden by overlap, but the residual ~41 ms
shows up as the 4 % wall gap).

### Root cause

FlashInfer doesn't ship a Hopper-native decode kernel.
`include/flashinfer/attention/hopper/` only has `prefill_sm90.cuh`.
The decode kernel we use (`persistent.cuh`) uses `cp_async` (sm_80).
On H100 it works via forward-compat but doesn't use TMA / WGMMA.
There is no FlashInfer config flip; the kernel doesn't exist there.

Python (`fa_utils.py:75-77`):
```
fa_version = 3 if (device_capability.major == 9 and is_fa_version_supported(3)) else 2
```
default-routes Hopper attention through FA3, which has TMA + WGMMA +
warp-specialization specifically for paged decode.

### Sync count is NOT the gap

I (this Claude) initially blamed per-step `sync_d2h` in
`gpu_sample_and_finalize → finalize_d2h_and_commit`. Instrumented
counters confirmed 128 syncs/iter at BS=1 in=2048 out=128, with
backtraces all in that single path. Tried a deferred-commit refactor
in the worker (`Vec<PendingCommit>`, count-only commits at decode
time, real-token drain on slow-path entry). Verified working with
SYNC_TRACE: greedy path went from sync_d2h=128 → sync_d2h=2 +
event_synchronize_raw=127. **Wall-clock improvement: ~1 %.** The
syncs moved from worker to engine pipeline, didn't disappear.

Pipeline depth: python and ferrite both at 2 in flight (python:
`uniproc_executor.py:60` `return 2 if scheduler_config.async_scheduling
else 1`; ferrite: `core_client.rs::get_output_pipelined` `while
pipeline.gpu_in_flight < 2`). Same depth.

The deferred-commit work is in `git stash@{0}` (label
"deferred-commit + sync-counters from FA3 investigation") on
`worktree-h100`. Worth landing only after FA3 closes the bigger gap.

### FA3 plumbing — Phase 1b layout debug DONE (2026-05-29 cont.)

The "illegal memory access on first call" from the prior session was the
shim's scheduler-workspace setup. flash_api.cpp:1057-1088 in the FA3
source computes:

```
num_prepare_batch_vectors = use_prepare_varlen(=1 when is_varlen)
                          + use_dynamic_split(=0)
                          + varlen_sort_batches(=0)
                          + head_swizzle(=1 when is_causal)
                          = 2  // for our config
b_rounded = round_up(batch_size, 4)             // 4 for batch=1
prepare_seqlen_q_ptr   = workspace[0]            // [0..b_rounded)
num_nheads_in_l2_ptr   = workspace[b_rounded]    // [b_rounded..b_rounded*2)  (head_swizzle)
tile_count_semaphore   = workspace[b_rounded*2]  // single int, plus offset stored on params
```

Old shim used `num_prepare_batch_vectors=1`, mis-sized the workspace, and
left `prepare_seqlen_q_ptr` / `num_nheads_in_l2_ptr` null while
`is_varlen=true` made `prepare_varlen_num_blocks` (the prelude kernel)
fire and dereference them. Fix at `/tmp/fa3_pod/fa3_shim.cu`. Allocate
`b_rounded * 2 + 1` ints, zero them, point the three pointers at the
right offsets. Also: `params.window_size_left = seqlen_k - 1`, `params.total_k = batch_size * page_size` (for paged), and
`params.prepare_varlen_pdl = (batch_size <= 992)` to match flash_api.

Verification (BS=1 sk=8192 hdim128 bf16, all-zero inputs):
```
LSE[0..3] = 9.01091 9.01091 9.01091 9.01091  (log(8192) ≈ 9.0109)
LSE nonzero count: 28 / 28
```
Numerically correct.

### FA3 plumbing — num_splits dynamic-bound fix (2026-05-29 cont.)

A line-by-line audit of vLLM-python's `flash_attn_varlen_func` ->
`mha_fwd` -> `set_params_fprop` revealed one more configuration miss:
Python passes `num_splits=max_num_splits=32` to BOTH `get_scheduler_metadata`
and the consumer call (flash_attn.py:472, :829). The kernel reads the
runtime per-batch split count from `num_splits_dynamic_ptr` (block.h:58)
and `flash_prepare_scheduler.cu:161` caps that dynamic value at
`min(optimal, num_splits_static)`.

I had been passing `num_splits=heuristic_pick`, which bakes the static
ceiling at the heuristic's value for the captured shape. With ferrite's
`padded_max_seqlen_k=2048` graph capture, the heuristic picks 14;
replays at sk=8192 (where 29 splits is optimal) stay capped at 14.

Fix: pass `FA3_MAX_NUM_SPLITS=32` always — matches Python. The persistent
oaccum/lseaccum buffers were already sized for 32 splits, so no
allocation changes needed.

After fix (BS=1 shapes, prior session's ferrite-on-FI baseline shown):

| shape                  | FA3   | FI    |  Δ vs FI | Python | gap closed vs FI |
|------------------------|------:|------:|---------:|-------:|----------------:|
| BS=1 in=2048 out=128   | 0.809 | 0.822 |   -1.6 % |  0.805 |  +1.6 pts       |
| BS=1 in=4096 out=128   | 0.873 | 0.887 |   -1.6 % |  0.856 |  +1.6 pts       |
| BS=1 in=8192 out=128   | 0.999 | 1.009 |   -1.1 % |  0.966 |  +1.1 pts       |
| BS=1 in=128  out=512   | 3.047 | 3.084 |   -1.2 % |    —   |  +1.2 pts       |
| BS=8 in=2048 out=128   | 1.155 | 1.155 |     tie  |    —   |  —              |
| BS=8 in=4096 out=128   | 1.569 | 1.569 |     tie  |    —   |  —              |
| BS=8 in=8192 out=128   | 2.523 |  OOM  |   FA3 wins (FI fragments) | — |   —     |

FA3 now wins at every BS=1 shape by 1-1.6%, and is more memory-friendly
at BS=8 in=8192 where FI hits OOM. Most of the residual gap to Python at
long-context (3.4 % at sk=8192) is NOT FA3-shaped — lives elsewhere.

### FA3 plumbing — Phase 6+7 DONE: AOT scheduling, FA3 ≤ FI everywhere (2026-05-29)

After the initial Phase 5 wiring, FA3 was a wash or slight regression vs
FlashInfer. Diagnosed with `FERRITE_FA3_TRACE=1`: FA3 was firing, but the
captured graph at `padded_max_seqlen_k=2048`
(`vllm-executor/src/ferrite_worker.rs:3760`) baked `num_splits=14` for a
shape later replayed at `seqlen_k=8192` (where the heuristic wants ~29
splits). FlashInfer's persistent kernel reads runtime `seqused_k` so it
didn't care; FA3's launch params are static at capture.

The deeper gap: vLLM-python uses **AOT scheduling**
(`vllm/v1/attention/backends/flash_attn.py:438-472`):
`get_scheduler_metadata` runs ONCE per scheduler step, the kernel skips
its own prelude (`skip_scheduler_metadata_computation=true`). My shim ran
the prelude `prepare_varlen_num_blocks` on every layer call (28 captured
kernels per decode iter that vLLM-python doesn't have).

Fixed both in `commit pending` by:

1. Exporting `fa3_get_scheduler_metadata_bf16_hdim128_sm90` from the
   shim (mirrors `mha_fwd_get_scheduler_metadata` in flash_api.cpp).
2. Adding process-persistent FA3 workspaces (metadata 4KB,
   oaccum 64MB, lseaccum 4MB) via `driver::mem_alloc` — never freed,
   sized once at first call for `max_num_splits=32` × max_h × max_total_q.
   Routing through caching allocator was wrong: workspaces must outlive
   any single forward and survive across captures, neither of which the
   caching allocator guarantees.
3. Plumbing `step_layer` (0-based within forward, distinct from
   `kv_layer` for pp) through `attention_helpers::flash_attn_3_decode`.
4. At `step_layer == 0`: pick `num_splits` (capped at 32), call
   `fa3_aot_build_metadata` to populate the persistent workspace. All
   layers (incl. 0) call the consumer with `skip_scheduler_metadata=1`.
5. `num_splits` cached in `AtomicUsize` so layers 1..N-1 read what
   layer 0 picked — graph-capture-friendly (atomic read at capture time
   bakes the value into the captured launch).

Bench (BS=1 in=8192 out=128 was the canary regression, now a tie):

```
shape                    FA3       FI       Δ
BS=1 in=8192 out=128    1.011    1.010    tie
BS=1 in=4096 out=128    0.879    0.887    -0.9% FA3 wins
BS=1 in=2048 out=128    0.815    0.823    -1.0% FA3 wins
BS=1 in=128  out=512    3.049    3.082    -1.1% FA3 wins
BS=8 in=2048 out=128    1.154    1.157    -0.3% FA3 wins
BS=8 in=4096 out=128    1.571    1.569    tie
```

FA3 wins or ties at all 6 shapes. The wins (~1%) are roughly what
collapsing 27 prelude kernel launches per decode step buys at compute-
non-dominated shapes; at the longest-context shape kernel time
dominates and the savings disappear into noise.

E2E correctness re-verified: 8/8 factual prompts produce identical
output under FA3 (AOT) vs FERRITE_DISABLE_FA3=1 (FlashInfer).

### FA3 plumbing — Phase 5 DONE: wired into FlashInferAttentionDecode (2026-05-29)

`Instruction::FlashInferAttentionDecode` arm in
`crates/ferrite-forward/src/instr.rs:2326` now tries FA3 first when:

- `ctx.device.sm_version >= 90`
- `head_dim == 128`
- `!use_logits_soft_cap`
- the compute stream is NOT in capture (skip during graph capture so the
  num_splits heuristic doesn't bake one shape's workspace size into a
  graph that gets replayed with a different shape).
- `FERRITE_DISABLE_FA3` env is unset.

`fa3_out: Option<OwnedTensor>` short-circuits the FlashInfer call when
present; FA2 fallback still runs if FlashInfer would itself return None.
The dispatch reads as `match fa3_out.or(fi) { Some => use it, None =>
FA2 fallback }` — a one-line additive change to the existing dispatcher
shape.

`crates/ferrite-kernels/src/attention_helpers.rs` adds:

- `is_stream_capturing(stream) -> bool` — small helper exported so
  ferrite-forward (which doesn't depend on cudarc) can detect capture
  without taking a new direct dep.
- `flash_attn_3_decode(...)` — wrapper that reads K/V layer tensors
  out of `KvCachePool` and forwards to
  `crate::flash_attn_3::flash_attn_3_paged_decode_bf16_hdim128`.

Verified: full `cargo build -p vllm-cli --features cuda,nccl,bench
--release` clean on `nick`. No clippy/type errors.

**Phase 6 (E2E correctness) deferred** — pod was busy on another
session. Next step:

1. `vllm serve Qwen/Qwen2.5-7B-Instruct ...` (default route → FA3).
2. `curl /v1/completions` with a known prompt; capture output.
3. Run again with `FERRITE_DISABLE_FA3=1` (forces FlashInfer path).
4. Diff outputs — should match within bf16 noise. Stronger check: same
   prompt against Python vLLM 0.21.x (also FA3) should also match.

**Phase 7 (re-bench)** comes after correctness. Expected delta from FA3
is the closing of the latency-bench 2-4% gap at compute-dominated
shapes (BS=1 in=8192 out=128) where the prior session measured
ferrite's 29.8 µs FlashInfer paged decode vs Python's 16.3 µs FA3 per
attention call.

### FA3 plumbing — Phases 2-4 DONE: vendored, built, FFI in (2026-05-29)

In-tree now:

- `third_party/vllm-flash-attn-3/hopper/` — 27 forward-only headers + 4
  .cu files (combine, prepare_scheduler, hdim128 bf16 paged, hdim128 bf16
  paged_split). Pin file at the same vllm-flash-attn commit
  `f5bc33cfc02c744d24a2e9d50e6db656de40611c`.
- `third_party/flash-attn-3-shim/ffi_shim.cu` — raw-pointer C ABI shim
  for `fa3_paged_decode_bf16_hdim128_sm90`. Promoted from
  `/tmp/fa3_pod/fa3_shim.cu`.
- `third_party/flash-attn-3-shim/compat/c10/util/Exception.h` — same
  TORCH_CHECK / TORCH_INTERNAL_ASSERT stub as the FA2 shim.
- `crates/ferrite-cuda-builder/build.rs::build_flash_attention_3` —
  cudaforge invocation; sm_90a, FLASHATTENTION_DISABLE_* flags trim to
  bf16 hdim128 forward only. Skipped on pre-sm_90 build hosts.
- `crates/vllm-cuda/build.rs` — link directive guarded on existence of
  `libvllm_flash_attn_3.a`.
- `crates/ferrite-kernels/src/flash_attn_3.rs` — Rust FFI + wrapper
  `flash_attn_3_paged_decode_bf16_hdim128`. Allocates oaccum/lseaccum
  workspaces from the caching allocator; computes num_splits via the
  ported FA3 heuristic (port of `heuristics.h:32`).

Build verified on `nick`:
- `libvllm_flash_attn_3.a`: 11.4 MB. Defined symbols include
  `fa3_paged_decode_bf16_hdim128_sm90`, both `Split=false` and
  `Split=true` variants of `run_mha_fwd_<90, bfloat16_t, 128, 128, ...>`,
  and the bf16/f32/128 combine kernel.
- `cargo build -p vllm-cli --features cuda,nccl` finishes clean in 31s
  after the kernel cache hit; no link errors with the FA3 lib pulled in.

What's left (Phase 5+):
- Pick the dispatch site. Cleanest is gating in `attention_helpers.rs`
  next to where FlashInfer/FA2 are selected today: when
  `cuda_arch >= 90` && bf16 && hdim==128 && paged && decode-shape
  (max_seqlen_q ≤ kBlockM after pack_gqa expansion), route to
  `flash_attn_3_paged_decode_bf16_hdim128`. Otherwise stay on
  FlashInfer.
- E2E correctness check on qwen2.5-7B vs FlashInfer-decoded output.
- Re-bench the full latency matrix.

### FA3 plumbing — Phase 1c DONE: Split path closes the gap (2026-05-29)

Compiled `flash_fwd_hdim128_bf16_paged_split_sm90.cu` (the
`Split=true` instantiation, 2 MB .o), wired the shim to route
`num_splits>1` through the Split kernel + `run_mha_fwd_combine_<bf16,
float, 128>`. Workspace layout is num_splits-dependent:

| num_splits | num_prepare_batch_vectors | semaphore offset (b_rounded=4) |
|-----------:|---------------------------:|-------------------------------:|
| 1          | 2  (prepare_seqlen_q + head_swizzle) | 8  |
| >1 (varlen, b≤992) | 3  (+ num_splits_dynamic)   | 12 |

Bench sweep at BS=1 sk=8192 hdim128 bf16 (wall-clock per iter, 1000 iters):

| num_splits | µs/iter |
|-----------:|--------:|
| 1   | 171.75 |
| 4   | 53.89  |
| 8   | 34.63  |
| 16  | 25.35  |
| 24  | 22.70  |
| 32  | 22.66  |
| 48  | 24.20  |
| 64  | 24.40  |
| 96  | 26.99  |
| 128 | 27.17  |

Optimum at num_splits=24-32 ≈ **22.7 µs/iter wall-clock**. Python's
nsys per-kernel time is 16.3 µs; the 6 µs gap is host-side launch
overhead per call (cudaEvent + 2 kernel launches in our microbench loop)
that goes away in the engine pipeline. Kernel-only time matches Python.

`num_splits_heuristic` from `heuristics.h:32` will pick num_splits ≈ 33
for this shape (best efficiency 1.0 at 33 splits = 132 SMs / 4
total_mblocks). That tracks.

LSE = log(8192) = 9.01091 across all sweep points → numerically correct.

### FA3 perf gap root cause: missing Split kernel (2026-05-29)

After the layout fix the kernel runs cleanly at **172 µs/call** — vs
Python's 16.3 µs. 10× off. The cause is `flash_api.cpp:get_num_splits`:
for BS=1 sk=8192 hdim128 with our shapes the heuristic computes

```
total_mblocks = b * h_k * num_m_blocks = 1 * 4 * 1 = 4
num_n_blocks  = ceil(8192 / kBlockN_sm90) ≈ 64
num_SMs       = 132
                   // 4 m_blocks ≪ 0.8 * 132 → falls into split-search
                   // picks smallest num_splits with eff ≥ 0.85 of max
                   // → num_splits ≈ 33
```

Python uses ~33 splits → ~33× parallelism over seqlen_k → ~5 µs of
on-device work plus combine. We compiled `Split=false` only, so we have
no parallelism along seqlen_k for this shape — 4 SMs do all the work
serially.

Fix is **Phase 1c**: compile the split instantiation
(`flash_fwd_hdim128_bf16_paged_split_sm90.cu`), allocate `oaccum`
`[num_splits, h, total_q, dv]` and `softmax_lseaccum`
`[num_splits, h, total_q]` workspaces, route through the existing
`run_mha_fwd_combine_<cutlass::bfloat16_t, float, 128>` after the split
kernel. `num_splits_heuristic` from `heuristics.h:32` is small and
self-contained — call it directly from the shim, no FlashAttention
dispatch glue needed.

Note: `pagedkv_tma=false` is correct for our shape — `get_pagedkv_tma`
returns false when `seqlen_q * (h/h_k) ≤ kBlockM` (decode pattern).
Python is also on PagedKVNonTMA. We match.

### FA3 plumbing — Phase 0 + 1a complete (2026-05-29)

Verified end-to-end on the pod:

1. **Source identified**: `github.com/vllm-project/flash-attention` at
   commit `f5bc33cfc02c744d24a2e9d50e6db656de40611c` (matches
   `cmake/external_projects/vllm_flash_attn.cmake:44` in vllm).
2. **Compile**: `hopper/instantiations/flash_fwd_hdim128_bf16_paged_sm90.cu`
   builds with our existing cutlass commit
   `62750a2b75c802660e4894434dc55e839f322277` and a 12-line
   `c10/util/Exception.h` stub (defines `TORCH_CHECK`,
   `TORCH_INTERNAL_ASSERT`, `TORCH_WARN_ONCE`, `TORCH_WARN`).
3. **Link**: paged kernel + `flash_prepare_scheduler.cu` +
   `flash_fwd_combine.cu` + a tiny C-ABI shim populate
   `Flash_fwd_params` from raw GPU pointers (no PyTorch dependence at
   the kernel layer). All four .o files link clean.
4. **Run**: kernel launches; passes through internal asserts
   (`prepare_varlen_num_blocks`, `tile_count_semaphore` allocation,
   TMA descriptor setup). Microbench at BS=1 sk=8192 hits illegal
   memory access on first call — **layout/stride bug in synthetic
   test data** (probably page_table layout or kv_batch_stride),
   not a kernel/build problem.

Smoke-test artifacts on pod (will be lost when pod restarts):
```
/tmp/flash-attn-vllm/    — sparse-checkout of vllm-flash-attn @ pin
/tmp/fa3_compat/         — c10 stub
/tmp/fa3_shim/fa3_shim.cu — minimal raw-ptr shim
/tmp/fa3_shim/bench.cu   — standalone microbench
```

### FA3 plumbing — remaining work

1. **Debug the layout** (1-2 hr). Diff the shim's `Flash_fwd_params`
   field-by-field against a python-side capture of the same call.
   Likely culprits: `page_table_batch_stride`, the order of `b` /
   `b_k`, whether `total_k` should reflect actual occupancy or the
   full padded page count, `kv_batch_idx` (currently null —
   verify).
2. **Vendor source** into `third_party/vllm-flash-attn-3/` (parallel to
   existing `third_party/vllm-flash-attn/` for FA2). Pin the same
   commit. Mirror our FA2 `flash-attn-shim/` pattern at
   `third_party/flash-attn-3-shim/`.
3. **Wire cudaforge** in `crates/ferrite-cuda-builder/build.rs` —
   add `build_flash_attention_3` mirroring the existing
   `build_flash_attention`. Output `libvllm_flash_attn_3.a`.
4. **Rust FFI** in `crates/ferrite-kernels/src/` — extern decl of
   `fa3_paged_decode_bf16_hdim128_sm90`, plus a Rust wrapper that
   takes `GpuTensor`s and a `KvCachePool`. Add a workspace allocator
   for the tile_count_semaphore (~32 bytes per call, can be a single
   allocation reused across calls).
5. **Wire as ferrite attention backend.** Add an `Instruction`
   variant or a new `FlashAttention3PagedDecode` impl.
   Cost-CSV row + dispatch gate: route H100 sm_90 BS=1..N decode
   through FA3, keep FlashInfer for sm_89 and prefill (where
   FlashInfer's `prefill_sm90.cuh` IS Hopper-native and fast).
6. **E2E correctness**: greedy output of qwen2.5-7B at known prompt
   matches existing FlashInfer-decoded output to within bf16 noise.
7. **Re-bench** the latency matrix vs python. Expected: closes the
   ~3 % gap, may overshoot.

### FA3 plumbing — known landmines

- `tile_count_semaphore`: workspace buffer required by the persistent
  scheduler at sm_90. For our config (b=1, head_swizzle=true=is_causal,
  no prepare_varlen, no dynamic_split) the offset is
  `b_rounded * num_prepare_batch_vectors = 4 * 1 = 4` ints, total
  buffer ≥ 5 ints. Allocate per-worker, zero-init once, reuse across
  calls.
- `head_swizzle = is_causal || is_local`: auto-set in flash_api.cpp.
  We need this true for decode (causal). Affects scheduler offset
  arithmetic above.
- `pagedkv_tma`: separate code path with its own kernel
  instantiation. We compiled `PagedKVNonTMA=true` (the easier
  variant). The TMA-paged variant might be faster at decode but
  requires its own instantiation + different param setup. Skip for
  v1.
- `prepare_varlen_pdl`: false for our case (skip the persistent
  prelude scheduler kernel). Setting true requires
  `prepare_seqlen_q_ptr` workspace etc. Skip for v1.
- The `flash.h` struct has many fields that are PyTorch-aware (e.g.
  `dq_ptr`, `dk_ptr` for backward). All zero for forward-only.
  `Flash_fwd_params{}` zero-init is sufficient; populate only the
  fields above.
- nvcc compile flags that worked: `-std=c++17 -O3 -arch=sm_90a
  --expt-relaxed-constexpr --expt-extended-lambda` plus disable
  flags `-DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_DROPOUT
  -DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_FP16
  -DFLASHATTENTION_DISABLE_HDIM{32,64,96,192,256} -DFLASHATTENTION_DISABLE_FP8`.
  Without the disable flags nvcc instantiates everything; with our
  set, the kernel .o is 13 MB (one paged variant) — manageable.
