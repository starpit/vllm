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

The `/v1/query/execute` endpoint accepts SPNL queries that describe the
structure of relocatable blocks. During tokenization, the endpoint produces a
sparse `BlockAnnotations` map (`BTreeMap<usize, BlockKind>`) alongside the
token sequence. This map tells the block hasher and attention kernel how to
handle each block:

- **`Relocatable`**: Parent hash reset to `NONE_HASH`, making the block
  cacheable regardless of position. On CUDA, the rotate-attend-unrotate cycle
  keeps K position-independent between attention steps.
- **`Prefixed`**: All preceding tokens are folded into the hash, forcing
  recomputation when any prior context differs.

No special tokens are injected into the token stream. Tokens are padded to
block boundaries using the tokenizer's whitespace token.

## Request Lifecycle Flags

Two flags on `Request` control post-completion behavior:

| Flag | Effect | Use case |
|------|--------|----------|
| `seal` | Pad + hash the final partial block on completion so it's cacheable | RAG prefill, inner generates |
| `volatile` | Push blocks to front of free queue for early eviction | Inner generates in nested patterns |

| Pattern | seal | volatile |
|---------|------|----------|
| Inner generate (nested) | true | true |
| RAG document prefill | true | false |
| Normal request | false | false |

## Hashing Flow

```
Request tokens: [a, b, c, d, | e, f, g, h, | q, r, s, t]
                 -- block 0 --  -- block 1 --  -- block 2 --
Annotations:     Relocatable     Relocatable     Prefixed

Block 0: parent = NONE_HASH (Relocatable)
         hash = H(NONE_HASH, [a, b, c, d])

Block 1: parent = NONE_HASH (Relocatable)
         hash = H(NONE_HASH, [e, f, g, h])

Block 2: parent = block_1_hash (chained)
         hash = H(block_1_hash, [q, r, s, t], all_prior_tokens)
```

Blocks 0 and 1 hash the same regardless of their order in the sequence. Block 2
includes all prior tokens in its hash, so reordering documents changes its key
and forces recomputation.

## RoPE Handling

RoPE is applied to K normally during QKV projection — all blocks store K with
position-encoded RoPE. Relocatable blocks then go through a per-block cycle
managed centrally in the attention helpers:

1. **Pre-attention**: rotate blocks marked `is_unrotated` (cached Relocatable
   blocks from prior steps that were un-rotated)
2. **Attention runs** (all K has correct RoPE)
3. **Post-attention**: un-rotate blocks marked `is_relocatable` (remove RoPE,
   back to position-independent for cache reuse)

Non-Relocatable blocks are never touched by this cycle.

The per-block flags (`is_relocatable`, `is_unrotated`) are computed from
`BlockAnnotations` via `compute_block_flags()` in `vllm-common`.

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `VLLM_V1_SPANS_DEBUG` | bool | `false` | Debug logging for span operations |
| `VLLM_V1_SPANS_PAD_TOKEN` | u32 | tokenizer `" "` | Token ID for block-boundary padding |

## Architecture

### Crates Modified

| Crate | Role |
|-------|------|
| `vllm-common` | `BlockKind`, `BlockAnnotations`, `compute_block_flags()`, seal/volatile on `Request` |
| `vllm-config` | `SpansConfig` struct (debug flag) |
| `vllm-core` | Annotation-aware block hashing, seal/volatile in `SimpleBlockTracker` |
| `vllm-cuda` | Per-block rotation flags in `KvCachePool`, rotate-attend-unrotate in attention helpers |
| `vllm-executor` | Block flag computation from annotations in `cuda_worker` |
| `vllm-serve` | `/v1/query/execute` tokenization produces `BlockAnnotations` |
| `vllm-bench` | `vllm bench spans` benchmark using `Prompt::TokenIdsWithAnnotations` |

### CUDA Implementation (functional)

1. **Annotation-aware hashing** (`vllm-core/src/scheduler/core.rs`):
   `hash_all_blocks()` takes an optional `&BlockAnnotations` map. Relocatable
   blocks reset parent hash to `NONE_HASH`. Prefixed blocks fold all prior
   tokens into the hash.

2. **Per-block rotation tracking** (`vllm-cuda/src/kv_cache.rs`):
   `KvCachePool` carries two flag arrays mirrored to GPU:
   - `block_is_span[block_id]` — this block is Relocatable
   - `block_is_unrotated[block_id]` — K is currently stored without RoPE

3. **Block flag computation** (`vllm-executor/src/cuda_worker.rs`):
   Before each forward pass, the executor calls `compute_block_flags()` for
   each block in annotated requests and sets flags via `mark_block()`.

4. **Rotate-attend-unrotate** (`vllm-cuda/src/model/attention_helpers.rs`):
   `with_span_rotation()` wraps attention with pre/post paged RoPE kernels.
   K is always written WITH RoPE (normal QKV projection). The cycle handles
   only Relocatable blocks.

### MLX Implementation (partial)

MLX models preserve `fuse_rope` code paths for future use but default to
`fuse_rope: false` (normal RoPE). Relocatable block support requires:

- Paged or per-block KV cache (MLX currently uses contiguous per-request tensors)
- Annotation-aware hashing in the MLX worker (currently uses flat `hash_prefix`)

Normal (non-span) MLX inference is fully functional.

## Benchmark

```
vllm bench spans <MODEL> [OPTIONS]
```

Runs an A/B comparison:

- **Without spans**: prefix caching with normal block hashing (order-dependent)
- **With spans**: `Prompt::TokenIdsWithAnnotations` with Relocatable blocks
  (order-independent cache reuse)

### Benchmark Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-docs` | 4 | Number of document blocks |
| `--block-size` | 16 | Tokens per block (each doc = 1 block) |
| `--query-len` | 16 | Query tokens appended after documents |
| `--num-iters` | 5 | Iterations for averaging |
| `--enforce-eager` | false | Disable CUDA graphs |

## TODO

### MLX Paged KV Cache

MLX needs per-block KV storage to support Relocatable blocks. Options:
- Per-block KV slice pool indexed by block hash
- Full paged KV (matching CUDA's block abstraction)

### CUDA: Fuse RoPE into Flash Attention

The current rotate-attend-unrotate uses 3 kernel launches per layer. Fusing
RoPE into the FA2 cached-K read path would eliminate the pre/post kernels.

### Nested Generate in /v1/query/execute

The endpoint currently handles `SingleGenerate` queries. Supporting nested
`Generate` nodes in the SPNL tree would enable the inner-outer pattern
(seal + volatile) to be fully managed server-side.

### Prefixed (Cross-Context) E2E Testing

The Prefixed hashing logic is unit-tested but no end-to-end benchmark exercises
it. Add a bench variant that validates cross-context recomputation.
