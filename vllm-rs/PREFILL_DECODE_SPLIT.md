# Prefill/Decode Split for Mixed Batches

> **SUPERSEDED**: The prefill/decode split was removed. Mixed batches now run
> through a single unified eager forward pass, matching Python vLLM's behavior.
> The `max_num_batched_tokens` default was updated from 1024 to 2048.
> The historical analysis below is kept for reference.

---

## Problem

With chunked prefill enabled (the default), nearly every scheduler step is a
**mixed batch** containing both prefill chunks (q_len > 1) and decode tokens
(q_len = 1). Because any prefill request forces the entire batch through the
eager (non-graph) forward path, decode tokens that could run in ~5ms via a
CUDA graph instead take ~250-330ms embedded in a large eager pass.

With 1000 concurrent prompts on Qwen2.5-3B (L40S), this produced 12.1 req/s
vs Python vLLM's 17.4 req/s.

## Solution

**Split mixed batches into two sequential forward passes** within a single
scheduler step:

1. **Decode pass**: Extract all q_len=1 requests, run through the existing
   CUDA graph (padded to nearest captured size). Cost: ~5ms regardless of
   batch size.

2. **Prefill pass**: Extract all q_len>1 requests, run through the eager
   forward path. Cost: proportional to total prefill tokens only.

3. **Merge**: Scatter both passes' logits back to original request order,
   then sample normally.

This is the approach Python vLLM used before torch.compile piecewise graphs
were added. Both passes reuse existing infrastructure — no model changes
needed.

### Implementation

All changes are in `CudaWorker::execute_model_inner` (`crates/vllm-executor/src/cuda_worker.rs`).
The split is detected after the existing `is_decode` check:

```
is_decode = all q_len == 1  →  CUDA graph (existing path)
is_mixed  = some q_len == 1 AND some q_len > 1  →  split into two passes (new)
otherwise = pure prefill  →  eager (existing path)
```

No other files required changes. The split is transparent to the scheduler,
sampling, and commit logic.

### Piecewise CUDA Graphs (attempted, abandoned)

Before the split approach, we tried **piecewise CUDA graphs** — capturing
per-layer graph segments (pre-attention and post-attention) and replaying them
around eager FA2 calls. This eliminated individual kernel launch overhead from
nsys traces, but throughput actually **worsened** to 11.8 req/s. The 72 graph
launches per step + FA2 calls didn't save enough vs eager, because our
individual fused kernels are already fast — the overhead is the sheer volume
of work at 8192 tokens, not launch latency. This work is stashed as
`piecewise-cuda-graphs-wip`.

## Tuning max_num_batched_tokens

The split's effectiveness depends on **how many prefill tokens** are in each
mixed step. With the default `max_num_batched_tokens=8192`, the scheduler
packs ~8 × 1024-token prefills per step — the eager pass is still ~250ms,
negating the decode graph savings.

Reducing `max_num_batched_tokens` caps prefill tokens per step:

| max_num_batched_tokens | req/s | vs Python (17.4) |
|---|---|---|
| 256 | 14.7 | -15% |
| 512 | 21.0 | +20% |
| **1024** | **21.8** | **+25%** |
| 2048 | 18.8 | +8% |
| 4096 | 13.5 | -22% |
| 8192 | 12.1 | -30% |

**1024 is the sweet spot.** At this value each 1024-token prompt prefills in
exactly 1 chunk (~25ms eager), while decode runs through the graph (~5ms).
Below 1024, too many steps are needed to drain each prompt (e.g. 256 → 4
chunks). Above 1024, the eager pass grows too large.

The default was changed from 8192 to 1024 in `init.rs`. Users can override
with `--max-num-batched-tokens`.

### Latency impact

`bench latency` (single-request) shows no regression — the split only
activates for mixed batches (>1 request with different q_lens).

## Benchmark: Qwen2.5-3B-Instruct, L40S

```
# Before (default 8192, no split)
Throughput: 12.1 req/s

# After (default 1024, with split)
Throughput: 21.8 req/s   (+80%)

# Python vLLM baseline
Throughput: 17.4 req/s
```

## How it works (step by step)

1. `execute_model_inner` checks if the batch is mixed and a CUDA graph exists
   for the decode subset size.

2. **Partition**: Requests are split into `decode_indices` (q_len=1) and
   `prefill_indices` (q_len>1).

3. **Decode pass**: Build padded decode inputs (token IDs, positions, slot
   mapping, cu_seqlens_q, seqused_k, block table). Replay through
   `CudaGraphRunner::replay()`. Result: `[n_decode, vocab]` logits.

4. **Prefill pass**: Build a new `AttentionMetadata` for the prefill subset.
   Run `model.forward_owned()` through the eager path. Result:
   `[n_prefill, vocab]` logits.

5. **Merge**: Allocate `[num_reqs, vocab]` tensor. D2D copy each group's
   logits to their original positions (2 contiguous copies).

6. **Sample**: The merged logits feed into the existing sampling code
   unchanged (greedy argmax, GPU Gumbel/filtered, or CPU fallback).

Both passes write to disjoint KV cache slots (decode at `tokens_before[i]`,
prefill at `tokens_before[i]..tokens_before[i]+q_len[i]`), so there are no
conflicts. Both run on the same CUDA stream sequentially.
