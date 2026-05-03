# Ferrite GGUF × tensor parallelism — handoff

Branch: `worktree-gguf-tp`, rooted at `ff-interpreter`. Top commit:
`see git log -3`. This doc is the continuation of `GGUF_HANDOFF.md`'s
"TP > 1 untested for GGUF" open follow-up.

## TL;DR

GGUF × `tensor-parallel-size > 1` is now coherent on seven architectures
(tp=1 and tp=2 produce identical greedy output). Three refuse-at-load
edge cases remain — they're real GGUF × TP divisibility limits, not
bugs. Covered by three commits in this branch.

## Verified at tp=2 (greedy "Paris", Q4_K_M unless noted)

Tested on a 2× L40S pod (`oc rsh nick3`). Each model checked at tp=1
and tp=2 for bit-identical greedy output over the chat-templated
"What is the capital of France?" prompt.

| Arch / model                     | Notes                                      |
| -------------------------------- | ------------------------------------------ |
| Llama-3.2-1B-Instruct            | baseline                                   |
| Llama-3.2-3B-Instruct            | —                                          |
| Qwen2.5-7B-Instruct              | biased q/k/v/o (per-rank bias slice path)  |
| Qwen3-0.6B                       | per-head Q/K RMS norm                      |
| Mistral-7B-Instruct-v0.3         | `gguf_archs` family dispatch               |
| Phi-3.5-mini-instruct            | fused `attn_qkv` / `ffn_up` packed split   |
| gemma-3-4b-it                    | `norm_weight_offset=1.0`                   |

## Refuse-at-load (known GGUF × TP divisibility limits)

These models load and run coherently at tp=1 but fail with an explicit
`refuse-at-load` error at tp>1. Each failure is a hard arithmetic
constraint of the on-disk quant format, not a ferrite bug.

| Model                    | Limit                                                                 |
| ------------------------ | --------------------------------------------------------------------- |
| Qwen2.5-0.5B-Instruct    | `intermediate_size / tp = 2432`, not a multiple of Q4_K block (256)   |
| gemma-3-1b-it            | `intermediate_size / tp = 3456`, same block-alignment issue           |
| granite-3.1-{2b,8b}      | `vocab_size = 49155` is odd; `ShardDim0` embed needs even division    |

Rule of thumb when picking a fixture:

* **ShardDim1** (row-parallel: `o_proj`, `down_proj`): per-rank
  `in_features = in / tp` must satisfy `in_features % block_size == 0`.
  For Q4_K / Q5_K / Q6_K the block size is 256; for legacy quants
  (Q4_0, Q8_0) it's 32. The loader checks via
  `ferrite_kernels::ggml::GgufShardKind::ShardDim1` and bails
  explicitly — see `crates/ferrite-kernels/src/ggml.rs` around line
  1486.
* **ShardDim0** (column-parallel: q/k/v/gate/up/embed/lm_head):
  per-rank `out_features = out / tp` must be a positive integer.
  Embedding/lm_head specifically need `vocab_size % tp == 0`.

Within a given family, the arithmetic usually improves with model
size: Qwen2.5 ≥ 7B, Gemma-3 ≥ 4B, Llama-3.2 all sizes divide cleanly.
Granite 3.1's 49155-vocab is shared across every size.

**Follow-up to unlock the 3 refused models**: add the same padding
tricks Python vLLM uses — vocab-pad embed/lm_head up to a multiple of
`tp`, pad `intermediate_size` up to a multiple of `tp * block_size`
for row-parallel down_proj. Not landed here.

## What the fix was

Three commits on `worktree-gguf-tp`:

1. **`39a728ec5` — tp>1 coherent: harvest preloaded tokenizer +
   GGUF-aware sharded loaders.** The root cause of all-garbage output
   was that `initialize_core_tp` (tp>1 stack init) only loaded the
   tokenizer via `try_load_tokenizer(model_dir)`, which returns `None`
   for GGUF (no sibling `tokenizer.json`). The engine silently fell
   back to byte-level encoding; `<|begin_of_text|>` became 15 raw ASCII
   tokens. The tp=1 path went through `initialize_core`, which harvests
   the worker's `preloaded_tokenizer` (reconstructed from GGUF metadata)
   before falling back to disk. Matched the tp=1 flow in
   `vllm-serve/src/init.rs::initialize_core_tp`.

   Also in this commit (prerequisites, not the actual bug):
   * `ferrite-forward-macro/src/codegen.rs::emit_fingerprint_check`
     bakes `vocab_size / tp` for `QuantMethod::Ggml` variants at
     tp>1 so the shape gate accepts the ShardDim0-sliced `embed_tokens`.
   * `ferrite-kernels/src/layers.rs::{Embedding, LinearLayer}::
     load_*_sharded` check the GGUF maps before the safetensors shard
     path. The GGUF loader pre-shards per `gguf_shard_kind_for_hf_name`
     at file-read time, so the per-rank tensor is already the right
     shape and `take_shard` on the safetensors CPU map would miss
     with "weight not found".
   * Row-parallel (dim=1) biases drop on rank>0 to match Python vLLM's
     `RowParallelLinear` post-AllReduce bias rule.

2. **`81de3c98d` — Phi-3.5-mini tp>1 rank-aware packed-split.** Phi-3
   family ships a fused `attn_qkv`/`ffn_up` parent on disk. The
   packed-splits prelude carves per-slice virtual q/k/v/gate/up
   entries. At tp>1 the fused parent's name isn't in the shard-kind
   table, so the GGUF loader replicated it on every rank; the
   prelude's unsharded carve then produced full-size children. My
   `load_dense_concat_sharded` GGUF fast-path picked them up as
   already-per-rank and returned a `GgmlConcat` of full-size branches
   — but `CanonicalParams::Q_SIZE`/`KV_SIZE` were baked per-rank, so
   `fused_qkv_rope_cache` read the qkv buffer with the wrong stride.

   Added `synthesize_packed_row_split_sizes_tp(packed_prefix,
   split_targets, tp_rank, tp_world_size)` in
   `ferrite-cuda-core/src/weights.rs`. At tp>1 with a quantized
   parent it carves each child at `slice_start + tp_rank *
   (slice_rows / tp)` for `slice_rows / tp` rows — zero-copy per-rank
   views into the replicated parent buffer. Codegen's packed-splits
   prelude in `ferrite-forward-macro/src/codegen.rs` now calls this
   variant with `tp_rank` (runtime) + baked `tp_world_size` literal.

3. **(pending commit) — Qwen2.5-7B tp>1: per-rank bias slice on
   ShardDim0.** GGUF biases ship 1D and land in `gguf_dense`
   full-sized on every rank (shard-kind `Replicate` for ndim<2).
   Column-parallel Linear at tp>1 needs a per-rank bias slice that
   matches the weight's per-rank out dim — otherwise
   `bias_add_inplace` reads the wrong slice. Qwen2's biased q/k/v/o
   exercises this; Llama / Mistral / Phi / Gemma don't ship QKV bias.
   Added `per_rank_bias_slice(...)` in
   `ferrite-kernels/src/layers.rs` that narrows the full 1D bias to
   `bias[rank * per_rank : (rank+1) * per_rank]` via `narrow_dim0`.
   Used in both `LinearLayer::load_dense_sharded` (dim=0 path) and
   `LinearLayer::load_dense_concat_sharded` (fused QKV branches).

   Known gap: `load_gguf_dense_concat` (F16/F32 GGUF concat path)
   doesn't yet slice biases per-rank. No fixture in the sweep
   exercises biased QKV on a dense GGUF — marked as follow-up but
   probably low-priority (Qwen2 F16 GGUFs are rare).

## Reproducer

```
# 2-GPU pod (nick3: 2× L40S)
oc rsh nick3 bash -c '
  cd /root/vllm-gguf-tp/vllm-rs && \
  cargo build -p vllm-cli --features cuda,nccl --release
'

# Point at any coherent fixture
oc rsh nick3 ./target/release/vllm chat \
  --model /root/gguf-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --tensor-parallel-size 2 --max-tokens 30 --temperature 0 \
  --prompt "What is the capital of France?"
```

Fixtures on the pod live in `/root/gguf-models/`; the 10 that
ship `-Q4_K_M.gguf` are the sweep inputs above + the three
refuse-at-load cases (Qwen2.5-0.5B, gemma-3-1b-it, granite-3.1-2b,
granite-3.1-8b).

## Follow-ups (not blocking this branch)

1. **E2E smoke test for tp=2.** Extend
   `crates/ferrite-gguf/tests/gguf_inference_smoke` with a `#[ignore]`-
   gated tp=2 variant. Matches the original plan's Phase 3 proposal.
   Needs `nccl` feature + ≥2 GPUs. Fixtures: Llama-3.2-3B (ShardDim1
   bf16 baseline), Qwen2.5-7B (biased QKV), Phi-3.5-mini (packed
   split). Skip DeepSeek-V2-Lite and Moonlight for now — their MoE
   3D experts are `Replicate`-sharded today (expected to work but
   memory-wasteful at tp=2 since per-rank kv cache halves while
   experts stay full).

2. **Vocab/intermediate padding.** Land the three refuse-at-load
   unlocks (Qwen2.5-0.5B, gemma-3-1b-it, Granite 3.1). The vocab pad
   is straightforward — pad `vocab_size` up to `ceil(v/tp)*tp`,
   synthesize zero-rows for the padding into the sharded embed at
   load, allgather after lm_head already covers the out side. The
   intermediate pad needs codegen support (`INTERMEDIATE_SIZE`
   literals must agree with the padded dim) and a corresponding
   unpadded slice after `down_proj` — not a single-file change.

3. **Performance parity with Python vLLM.** Original plan Phase 4.
   Not started. `vllm bench latency` at tp=2 on e.g.
   Llama-3.2-3B-Q4_K_M or Qwen2.5-7B-Q4_K_M against Python's
   `vllm serve` GGUF backend. Iterate on NCCL stream placement,
   AllReduce promotion dtype, cuBLAS plan cache. No known
   specific regression vs. Python today — just hasn't been measured.

4. **tp=4, tp=8.** The macro fanout emits variants at `{1, 2, 4, 8}`
   under the `nccl` feature (confirmed in `strings` output:
   `llama_3_2_3b_ggml_tp8`). Never exercised on hardware. Most of
   the fixes in this branch generalize; expect more divisibility
   edge cases at higher tp.

5. **CommandR at tp>1.** Known broken at tp=1 per the parent
   handoff — tp makes it strictly harder, not worth touching until
   tp=1 works.

## Where to look first

1. `crates/vllm-serve/src/init.rs::initialize_core_tp` — the tokenizer
   harvest that was missing. Patterned after `initialize_core`.
2. `crates/ferrite-kernels/src/layers.rs::per_rank_bias_slice` — the
   1D-bias narrow helper, called from `LinearLayer::load_dense_sharded`
   and `LinearLayer::load_dense_concat_sharded`.
3. `crates/ferrite-cuda-core/src/weights.rs::
   synthesize_packed_row_split_sizes_tp` — rank-aware packed parent
   carve for Phi-3 style fused QKV / gate_up.
4. `crates/ferrite-forward-macro/src/codegen.rs::emit_fingerprint_check`
   — the sharded `vocab_lit` for GGUF variants at tp>1.
5. `crates/ferrite-kernels/src/ggml.rs::gguf_shard_kind_for_hf_name`
   — the HF-name → shard-kind rule table. Only recognizes the per-slice
   HF names (q_proj, k_proj, etc.), so fused parents (`qkv_proj`,
   `gate_up_proj`) fall through to `Replicate` and the packed-splits
   prelude does the per-rank carve above.
