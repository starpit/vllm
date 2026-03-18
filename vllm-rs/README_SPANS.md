# Relocatable KV Cache Blocks (Spans) for vllm-rs

Spans enable **position-independent KV cache block reuse** for RAG and similar
workloads where preloaded document blocks are served in varying order across
requests.

## The Problem

Standard prefix caching requires an exact prefix match. If request 1 prefills
`[doc_A, doc_B, query_1]` and request 2 prefills `[doc_B, doc_A, query_2]`, no
cache reuse occurs because the prefix differs — even though both requests
contain the same documents.

## The Solution

Spans mark document blocks with a special **span token** (`VLLM_V1_SPANS_TOKEN_PLUS`).
Blocks starting with this token are hashed independently of their position in
the sequence (fan-in hashing), so `doc_A` produces the same cache key regardless
of whether it appears at position 0 or position 64.

Keys in span blocks are stored **without rotary position embedding** (RoPE).
Position-specific rotation is applied at attention time, making the cached KV
data valid at any sequence position.

## Benchmark

```
vllm bench spans <MODEL> [OPTIONS]
```

Runs an A/B comparison:

- **Baseline**: prefix caching disabled, full recompute of all tokens
- **Spans**: per-block KV cache reuse — only the query block is recomputed

### Example

```
$ vllm bench spans mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
    --device metal --num-docs 64 --block-size 64 --query-len 32 \
    --num-iters 3 --enforce-eager

=== Spans Benchmark Results ===
Workload: 64 docs x 64 tokens + 32 query tokens
Prefill latency for reversed doc order (ms):
  Baseline = no caching, full recompute
  Spans    = per-block KV cache reuse (only query recomputed)

                       Baseline      Spans
  -------------------- ---------- ----------
  Avg (ms)               23818.83     247.55
  Min (ms)               23587.66     233.12
  Max (ms)               24264.57     275.74

  Speedup: 96.22x
```

### Expected Speedups

Speedup scales with the number of cached document tokens relative to query
tokens. More documents and larger blocks yield higher speedups because more
prefill computation is skipped.

| Model | Docs | Tokens cached | Speedup |
|-------|------|---------------|---------|
| Llama-3.2-3B-4bit | 64 x 64 | 4096 | ~82x |
| Llama-3.1-8B-4bit | 64 x 64 | 4096 | ~96x |

Speedup is primarily a function of `cached_tokens / query_tokens` and model
size. Larger models benefit more because the per-token prefill cost is higher.

### Benchmark Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-docs` | 4 | Number of document blocks |
| `--block-size` | 16 | Tokens per block (each doc = 1 block) |
| `--query-len` | 16 | Query tokens appended after documents |
| `--num-iters` | 5 | Iterations for averaging |
| `--span-token` | 10 | Token ID used as the span marker |
| `--enforce-eager` | false | Disable CUDA graphs |

## Environment Variables

All span configuration is via environment variables, matching the Python vLLM
`envs.VLLM_V1_SPANS_*` convention.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VLLM_V1_SPANS_ENABLED` | bool | `false` | Master switch |
| `VLLM_V1_SPANS_DEBUG` | bool | `false` | Debug logging for span operations |
| `VLLM_V1_SPANS_TOKEN_PLUS` | u32 | None | Fan-in span token ID |
| `VLLM_V1_SPANS_TOKEN_CROSS` | u32 | None | Cross-context span token ID |
| `VLLM_V1_SPANS_DISABLE_REPOSITION` | bool | `false` | Disable RoPE fusion (hash-only mode) |

Boolean values accept `true`, `True`, or `1`.

### Token Semantics

- **`TOKEN_PLUS`**: First token of a relocatable block. Resets the parent hash
  to a sentinel value, making the block's cache key independent of preceding
  blocks. Keys are stored without RoPE.

- **`TOKEN_CROSS`**: First token of a context-dependent block. All preceding
  tokens are folded into the hash, forcing recomputation if any prior context
  changes. Useful for query blocks that depend on the full document set.

## Architecture

### Crates Modified

| Crate | Role |
|-------|------|
| `vllm-config` | `SpansConfig` struct, env var parsing |
| `vllm-core` | Span-aware block hashing in `SimpleBlockTracker` |
| `vllm-mlx` | Per-block KV cache pool, fused RoPE in attention, `hash_blocks` |
| `vllm-bench` | `vllm bench spans` benchmark |

### MLX Implementation (complete)

The MLX path is fully functional:

1. **Span-aware hashing** (`vllm-core/src/scheduler/core.rs`):
   `hash_all_blocks()` chains parent hashes per block. Blocks starting with
   `TOKEN_PLUS` reset the parent to `NONE_HASH` (fan-in). Blocks starting with
   `TOKEN_CROSS` fold all preceding tokens into the hash.

2. **Per-block KV cache pool** (`vllm-mlx/src/worker.rs`):
   On request completion, the KV cache is split into per-block chunks and stored
   in `block_kv_pool` indexed by per-block hash. On new requests, blocks are
   matched individually and reassembled into a contiguous KV cache via
   concatenation.

3. **Fused RoPE** (`vllm-mlx/src/models/llama.rs`):
   When `fuse_rope=true`, keys are stored without rotation. During attention,
   RoPE is applied to the full cached key sequence at position offset 0,
   effectively rotating each key by its actual position.

### CUDA Implementation (stashed, in progress)

The CUDA changes are stashed in `git stash@{0}` ("cuda: spans phase 4 +
per-block rotation tracking"). They include:

1. **Paged RoPE kernel** (`vllm-cuda/csrc/pos_encoding_kernels.cu`):
   `rotary_paged_k_cache_kernel<T, Inverse>` applies (or un-applies) RoPE to K
   stored in paged cache blocks. Accepts per-block flags to selectively process
   only span blocks. Supports F16 and BF16.

2. **Per-block rotation tracking** (`vllm-cuda/src/kv_cache.rs`):
   `KvCachePool` carries two flag arrays:
   - `block_is_span[block_id]` — permanent: this block should be stored unrotated
   - `block_is_unrotated[block_id]` — transient: current rotation state
   Both are mirrored to GPU via H2D copy before each attention step.

3. **Executor integration** (`vllm-executor/src/cuda_worker.rs`):
   Before each forward pass, the executor walks all blocks in the batch, checks
   the first token of each block against `TOKEN_PLUS`, and sets both flags.

4. **LlamaAttention forward** (`vllm-cuda/src/model/llama.rs`):
   Standard fused QKV + RoPE writes rotated K to cache. Before attention, the
   paged RoPE kernel rotates blocks flagged as `is_unrotated`. After attention,
   it un-rotates blocks flagged as `is_span`. This rotate-attend-unrotate pattern
   leaves span blocks in unrotated resting state between steps.

5. **Q-only RoPE wrapper** (`vllm-cuda/src/kernels.rs`):
   `rotary_embedding_q_only()` applies RoPE to Q while skipping K, using the
   existing kernel with `total_k_dim=0`.

### Hashing Flow

```
Request tokens: [PLUS, a, b, c, | PLUS, d, e, f, | CROSS, q, r, s]
                 ---- block 0 --   ---- block 1 --   ---- block 2 --

Block 0: parent = NONE_HASH (fan-in)
         hash = H(NONE_HASH, [PLUS, a, b, c])

Block 1: parent = NONE_HASH (fan-in)
         hash = H(NONE_HASH, [PLUS, d, e, f])

Block 2: parent = block_1_hash (chained)
         hash = H(block_1_hash, [CROSS, q, r, s], all_prior_tokens)
```

Blocks 0 and 1 hash the same regardless of their order in the sequence. Block 2
includes all prior tokens in its hash, so reordering documents changes its key
and forces recomputation.

## TODO / Cleanup

### Share Hashing Logic Between MLX and CUDA

Currently `hash_blocks()` is duplicated between:
- `vllm-core/src/scheduler/core.rs` (`SimpleBlockTracker::hash_all_blocks`)
- `vllm-mlx/src/worker.rs` (`hash_blocks`)

Both implement the same fan-in / cross-context logic. Extract into a shared
function in `vllm-core` (or `vllm-common`) that both paths call.

### MLX Per-Block Cache Eviction

`block_kv_pool` in the MLX worker grows unbounded. Add LRU or size-based
eviction, similar to `PREFIX_CACHE_POOL_MAX` for the whole-prefix pool. Track
total memory usage across cached blocks.

### MLX KV Assembly Overhead

Assembling KV from per-block chunks requires `num_blocks x num_layers`
concatenations. For large context (hundreds of blocks), this can take tens of
milliseconds. Potential improvements:
- Pre-allocate a contiguous buffer and use `slice_update` instead of `concatenate`
- Cache the assembled KV for repeated orderings
- Use MLX's lazy evaluation more aggressively (avoid early `eval()`)

### CUDA: Fuse RoPE into Flash Attention

The current CUDA approach uses rotate-attend-unrotate (3 kernel launches per
layer). A true solution would fuse RoPE into the Flash Attention 2 kernel so
that unrotated K is rotated on-the-fly during the attention inner loop. FA2
already has `rotary_cos_ptr` / `rotary_sin_ptr` fields in `Flash_fwd_params`
and `copy_rotary_contiguous` / `copy_rotary_interleaved` helpers in `rotary.h`,
but these are only used for the `Append_KV` path, not the cached-K read path.

### CUDA: Eliminate Global `fuse_rope` Flag

The `fuse_rope` field on `LlamaAttention` is set at model load time from env
vars. With per-block rotation tracking in place, the rotation logic should be
driven entirely by `block_is_span` / `block_is_unrotated` flags, allowing
mixed span and non-span workloads without a global switch.

### Non-LLaMA Model Support

Only `MlxLlamaAttention` (covering LLaMA, Mistral, Qwen2) has `fuse_rope`
wired up. Other MLX models (Gemma, CommandR, DeepSeek, etc.) need the same
treatment. The pattern is identical: skip K rotation when `fuse_rope=true`,
apply RoPE to full cached K during attention.

### Cross-Context (`TOKEN_CROSS`) Testing

The `TOKEN_CROSS` hashing logic is implemented and unit-tested, but no
end-to-end benchmark exercises it. Add a bench variant that validates
cross-context recomputation (e.g., same documents but different system prompts
should produce different outputs).

### Scheduler / MLX Worker Cache Alignment

The scheduler (`SimpleBlockTracker`) and MLX worker (`block_kv_pool`) maintain
separate cache state. When the scheduler reports `num_computed_tokens > 0` but
the MLX worker has no matching KV data, the worker silently recomputes. Add
instrumentation to detect and warn on cache state divergence.
