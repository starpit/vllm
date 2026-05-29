# Ferrite-side piecewise CUDA-graph decode at TP>1 — handoff

Branch: `worktree-rm-old-cuda` · Tip: `fb64268b21`
Pod: `nick3` (2× L40S sm_89) · Path: `/root/rm-old-cuda/vllm-rs/`

## Why this exists

`fb64268b21` made ferrite-forward the sole CUDA model-forward path and
deleted the hand-written `vllm_cuda::model::*` types along with the
piecewise CUDA-graph subsystem that depended on them. At TP>1 today,
**decode runs eager** — prefill is still graph-captured.

The carve-out is real: NCCL collectives inside a monolithic CUDA graph
fail with `CUDA_ERROR_ILLEGAL_ADDRESS` on L40S (verified in this branch
by patching `resolve()` to skip the TP>1 downgrade and observing the
capture failure). So the path to graph-accelerated decode at TP>1 is
piecewise capture: split the forward at NCCL boundaries, capture each
segment as a CUDA graph, run the collectives eagerly between graph
replays. Python vLLM does this via `torch.compile`'s graph breaks; the
hand-written code that lived here did it via per-piece executor methods.
Ferrite needs its own implementation.

## Where the boundaries are

`ferrite-forward-macro::tp_lowering` already inserts the right
instructions (this is unchanged):

- `Instruction::AllReduce(slot)` after every gemm whose weight is
  `ShardDim1` (`o_proj`, `down_proj`) and after the vocab-parallel
  `Embed`.
- `Instruction::AllGather(in_slot, out_slot)` after the `lm_head` gemm
  at tp>1.

For Llama-3.2-1B at TP=2 (16 layers × 2 NCCL/layer + post-Embed +
post-lm_head): **34 segments per bucket**.

The eval bodies are in `vllm-rs/crates/ferrite-forward/src/instr.rs`
around lines 1254 (`AllReduce`) and 1266 (`AllGather`), inside the
`#[cfg(feature = "cuda")] impl Instruction { fn eval(...) }` block. The
allreduce mutates the slot in place (easy to capture around). The
allgather allocates a fresh output buffer — that's the awkward case.

## Architecture sketched in this session, not implemented

New module `vllm-rs/crates/ferrite-forward/src/piecewise.rs`:

```rust
pub enum SegmentTerminator {
    /// In-place all-reduce on `slot`; next graph piece reads the same address.
    AllReduce { slot: u32 },
    /// All-gather: `in_slot` → `out_slot`. Output goes to a stable
    /// pre-allocated buffer; next graph piece (or the final caller)
    /// reads `out_slot`.
    AllGather { in_slot: u32, out_slot: u32 },
    /// Last segment (no collective after); its captured tensor at
    /// `terminal_slot` is the output.
    Terminal { terminal_slot: u32 },
}

pub struct Segment<'a> {
    pub instructions: &'a [Instruction],
    pub bucket: u32,
    pub terminator: SegmentTerminator,
}

pub fn tape_segments<'a>(
    backbone: &'a [Instruction],
    backbone_bucket: u32,
    lm_head: &'a [Instruction],
    lm_head_bucket: u32,
    terminal_slot: u32,
) -> Vec<Segment<'a>> { ... }

pub struct PiecewiseRunner {
    captured: Vec<(CudaGraphExec, SegmentTerminator)>,
    /// Slots whose addresses must stay alive for the lifetime of the
    /// runner (the caching-allocator pool that backs them).
    _pool_handle: PrivatePoolGuard,
    /// The very last segment's terminal slot — the runner's output
    /// is read from here at replay's end.
    terminal_slot: u32,
}

pub unsafe fn run_piecewise_capture<W: CanonicalParams>(
    backbone: &[Instruction], backbone_bucket: u32,
    lm_head:  &[Instruction], lm_head_bucket:  u32,
    wm: &W, fwd: &ForwardCtx, device: &mut GpuDevice,
    num_slots: u32, terminal_slot: u32,
) -> PiecewiseRunner { ... }

pub unsafe fn run_piecewise_replay<W: CanonicalParams>(
    runner: &PiecewiseRunner,
    wm: &W, fwd: &ForwardCtx, device: &mut GpuDevice,
) -> OwnedTensor { ... }
```

Capture flow:
1. `device.caching.begin_allocate_to_pool()` — anchor a private pool.
2. Walk the tape with `Loop` expanded once per layer iteration so each
   layer's pre-/post-NCCL chunks become physically distinct segments.
3. For each segment:
   - `stream_begin_capture(stream)`
   - `SuppressNcclGuard::new()` (so the in-segment NCCL no-ops don't
     pollute the captured graph; the in-place AllReduce variant inside
     `eval` already respects this — see `ferrite-cuda-core/src/nccl.rs`).
   - `run_slice(segment, ctx)` — kernels record into the graph,
     allocations land in the private pool with deterministic addresses.
   - `stream_end_capture(stream)` → `graph_instantiate` → `exec`.
   - Note the terminator's slot(s) for replay-side eager NCCL.
4. `end_allocate_to_pool()` — keeps the pool live, addresses pinned.

Replay flow:
1. For each `(exec, term)` in order:
   - `graph_launch(exec, stream)`
   - Run the collective eagerly into the captured-stable buffer:
     - `AllReduce { slot }` →
       `tp_group.all_reduce_inplace_promote(tile_at(slot), …)`. The
       in-place mutation lands in the same address the next segment's
       captured graph expects.
     - `AllGather { in, out }` → see "Open question" below.
2. After the final segment: copy/move the terminal slot's tensor to
   the caller as `OwnedTensor` (same handoff `instr::run` does today
   via `take_owned` / memcpy).

## Open questions, in priority order

### 1. AllGather output stability (the hard one)

`ferrite-cuda-core::NcclGroup::all_gather_last_dim` allocates BOTH a
temp buffer (`temp = self.all_gather(tensor, alloc)`) and a final
output (`out = alloc.alloc_tensor(...)`), then runs a rearrange kernel
that reads `temp`, writes `out`. With NCCL suppressed at capture time,
both allocs still happen (predictable) but the rearrange reads garbage.
The *captured graph piece that follows* reads `out`.

Two design choices:

- **(a) Split the AllGather instruction in lowering** into
  `AllGatherTransfer(in_slot, temp_slot)` (just NCCL) +
  `AllGatherRearrange(temp_slot, out_slot)` (kernel). Make the segment
  terminator be `AllGatherTransfer`, so the rearrange ends up captured
  in the *next* segment. Replay-time eager NCCL writes to `temp_slot`'s
  pre-allocated address; rearrange runs from the captured graph and
  produces the right `out_slot`.

- **(b) Carve out an explicit "AllGather scratch buffer" pinned in
  the runner**. Replay-side code calls a custom transfer-only NCCL
  helper that targets the scratch directly, then replays the segment
  whose graph contains the rearrange.

(a) is cleaner — it generalizes to other split-collectives and keeps
the runner dumb. Plumbs through `tp_lowering.rs` and `info.rs` /
`interpreter_codegen.rs` enum match arms. Worth doing.

### 2. `Loop` instruction and per-layer capture

`run_slice` expands `Loop(count, body_len)` dynamically: it runs the
body `count` times, each iteration with a different `ctx.layer_offset`.
The kernels read per-layer weights via `ctx.wm.<accessor>(bucket,
op_idx, layer, …)`, so the ADDRESS each kernel receives changes per
layer.

For piecewise capture, `Loop` *can't* be captured as one graph and
replayed N times — pointer values are baked in per layer. Two paths:

- **(a) Expand once at capture time**: walk the tape, materialize a
  flat per-iteration sequence with `layer_offset` baked in. Capture
  each (layer × piece-position) combination as its own graph. For
  Llama-3.2-1B: 16 layers × 2 NCCL/layer = 32 in-loop pieces, +1
  pre-loop (Embed+AllReduce), +1 post-loop (final norm + lm_head +
  AllGather) = 34 graphs. Memory: each `cudaGraphExec_t` ≈ a few KB,
  so ~few hundred KB total; fine.

- **(b) Use cudaGraph instantiation flags / node updates** so a single
  graph can be parameterized by layer index. Significantly more
  complex and has its own pointer-stability issues.

Go with (a). It's also what the deleted hand-written piecewise did
(see the per-`layer_idx` arms in the old `execute_pre_attn_piece` /
`execute_post_attn_piece`).

### 3. Multi-bucket capture

`forward(num_tokens)` looks up a bucket from `FORWARD_TABLE` based on
`(num_tokens, max_seqlen_k)`. The tape per bucket has different
shapes. For decode (q_len=1) we typically capture at common batch
sizes (per `cuda_graph_sizes`). The runner needs to be keyed by graph
batch size so the executor can pick the right one.

Likely shape: `HashMap<usize /* batch_size */, PiecewiseRunner>`,
populated in `compile_or_warm_up_model`, looked up in
`execute_model_inner`'s decode path.

### 4. Caching-allocator pool hygiene across captures

The existing prefill graph capture does
`begin_allocate_to_pool` → capture one graph → `end_allocate_to_pool`,
with the pool kept (not reset) for the runner's lifetime. For
piecewise, all 34 segment captures share *one* pool (so addresses
inside a single forward are deterministic). Different buckets use
different pools (so they don't share addresses, which would let
replays of one collide with the other).

Look at `vllm-cuda::graph::PrefillGraphRunner::capture` for the
reference pattern; copy the surrounding bracket into
`run_piecewise_capture` and just add a `for segment in segments {}`
loop inside.

## Concrete first steps

1. **Add `Instruction::AllGather` split in lowering** (open question 1
   path a). Touch:
   - `ferrite-forward/src/instr.rs` — add
     `AllGatherTransfer(u32, u32)` and
     `AllGatherRearrange(u32, u32)` variants (gated `#[cfg(feature =
     "nccl")]`); split `eval` accordingly. Update `info.rs` /
     `interpreter_codegen.rs` `I::AllGather(..)` match arms.
   - `ferrite-forward-macro/src/tp_lowering.rs::insert_lm_head_allgather`
     — emit two ops instead of one. Tests in `tp_lowering.rs` will need
     updating.
   - One-pass test: `cargo test -p ferrite-forward-macro` should pass
     after the split.

2. **Stub the new module**: create
   `ferrite-forward/src/piecewise.rs` with the types above and an
   `unimplemented!()`-bodied `run_piecewise_capture` /
   `run_piecewise_replay`. Wire it under `pub mod piecewise;` in
   `ferrite-forward/src/lib.rs`. Confirm it compiles with the existing
   workspace.

3. **Implement `tape_segments`**: pure function over `&[Instruction]`
   that handles `Loop` expansion. Add a unit test that segments a
   small synthetic tape and asserts segment count and terminators.

4. **Implement capture+replay with eager-only first** (no graph
   capture, just split-and-call). This proves segmentation logic
   matches `instr::run`'s output bit-identically. Reference test:
   compare the output tensor against `instr::run`'s output at TP=1
   (where NCCL is no-op anyway, so the result should match exactly).

5. **Add graph capture**, segment by segment, anchored to a private
   pool. Verify against same reference.

6. **Wire into the macro** (`ferrite-forward-macro/src/codegen.rs`):
   emit `forward_piecewise_capture` / `forward_piecewise_replay`
   alongside the existing `forward`. Match the existing dispatch
   pattern (`find_bucket(FORWARD_TABLE, num_tokens, max_seqlen_k)`).

7. **Wire into `FerriteWeights` trait**
   (`ferrite-forward/src/lib.rs:477`): add
   `forward_piecewise_capture` and `forward_piecewise_replay`
   methods (default impl: panic). Per-arch macro override emits the
   real impl. Per-arch trait impls are emitted by the macro too.

8. **Wire into the executor**
   (`vllm-executor/src/ferrite_worker.rs`):
   - Add `piecewise_runners: HashMap<usize, PiecewiseRunner>` (keyed
     by graph_bs) on the worker.
   - In `compile_or_warm_up_model`, when
     `cuda_graph_mode == Piecewise`, capture once per
     `cuda_graph_sizes` entry. Build a dummy `ForwardCtx` like the
     existing prefill graph capture does.
   - In `execute_model_inner`'s decode path, when batch matches a
     captured size, call `forward_piecewise_replay` instead of
     `forward`. Keep the eager fallback for non-captured batch sizes.

9. **Test on nick3**:
   ```bash
   cd /root/rm-old-cuda/vllm-rs
   FERRITE_MODELS=llama-3.2-1b cargo build --release \
       -p vllm-cli --features cuda,nccl
   ./target/release/vllm serve --model unsloth/Llama-3.2-1B-Instruct \
       --tensor-parallel-size 2 --port 18200
   curl http://localhost:18200/v1/completions -d '{...}'
   ```
   Expect: `loaded LlamaForCausalLM via ferrite-forward`,
   `cuda_graph_mode resolved Auto → Piecewise`, plus a NEW log line
   `Piecewise CUDA graphs captured for batch_size={...}`. Output
   should be coherent.

10. **Compare decode latency** vs the eager-decode baseline this
    branch ships today. Should drop noticeably (graph replay
    eliminates per-kernel launch overhead between AllReduces).

## What this branch does NOT do

- Does not touch the metal piecewise path (irrelevant — metal is
  TP=1 only).
- Does not address MoE arches at TP>1 with graphs. Same node-count
  limit problem the deleted hand-written piecewise also wouldn't have
  helped with. MoE keeps eager.
- Does not handle pipeline parallelism (`pp_size > 1`). PP was
  removed in `fb64268b21`; load_model rejects it. Re-adding PP is
  separate work.

## Files most relevant when picking this back up

- `vllm-rs/crates/ferrite-forward/src/instr.rs` — `Instruction` enum,
  `eval`, `run_slice`, `run`, `run_backbone`. The piecewise module
  goes alongside.
- `vllm-rs/crates/ferrite-forward-macro/src/tp_lowering.rs` —
  `insert_all_reduces`, `insert_lm_head_allgather`. AllGather split
  lives here.
- `vllm-rs/crates/ferrite-forward-macro/src/codegen.rs:7362` — the
  emitted `forward` body. New emission for `forward_piecewise_*`
  goes alongside.
- `vllm-rs/crates/ferrite-forward/src/lib.rs:477` — `FerriteWeights`
  trait. New methods go here.
- `vllm-rs/crates/ferrite-cuda-core/src/nccl.rs:33` —
  `SuppressNcclGuard`, `is_nccl_suppressed`. Keep the
  `SuppressNcclGuard` wrapping the entire capture phase. Already
  respected by `all_reduce_inplace`.
- `vllm-rs/crates/vllm-cuda/src/graph.rs:705` —
  `PrefillGraphRunner::new` / `capture` is the pattern to match for
  the new piecewise runner (private pool, dummy warmup, real capture,
  instantiate). Lift the bracket and re-use.
- `vllm-rs/crates/vllm-executor/src/ferrite_worker.rs` —
  `compile_or_warm_up_model` (line 3262), `execute_model_inner`. The
  piecewise capture branch was deleted in `fb64268b21`; the comment
  there points to where the new branch goes.

## Empirical sanity check — what I confirmed and what I didn't

- **Confirmed**: NCCL inside monolithic CUDA graph fails on L40S
  sm_89 with `CUDA_ERROR_ILLEGAL_ADDRESS` (no surprise, but verified
  in this branch). The April-2026 carve-out is correct.
- **Confirmed**: at TP=2 with `cuda_graph_mode = Piecewise` (the
  current state of this branch), prefill graph capture works (5
  sizes captured), decode runs eager, output is coherent across
  Llama / Qwen2 architectures.
- **Not confirmed**: that ferrite-side piecewise capture actually
  works end-to-end. The design above is plausible but the
  AllGather-output and Loop-expansion details have not been proven
  out empirically yet.
