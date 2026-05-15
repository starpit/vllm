# Ferrite-metal MTL4 migration

Captures the migration plan from `MTLCommandBuffer` + Serial encoder
+ `MTLIndirectCommandBuffer` to MTL4's native compute sequencing.
First documented in `project_ferrite_metal_status.md` under
"TODO 1: MTL4 command-encoding migration"; this file is the
executable plan.

HEAD at planning time: `a3bff17ef`.

## Why

Current per-forward hot path (`pool.rs:626-652`,
`worker.rs:455-553`):

1. `queue.commandBuffer()` — one MTLCommandBuffer per forward.
2. `cmdbuf.computeCommandEncoder()` — one Serial-dispatch encoder
   spans the whole forward.
3. Per step: `enc.setComputePipelineState(pipeline)` →
   `icb.execute_on_encoder(&enc, range)`. The ICB pre-records every
   `setBuffer` at bake time; the encoder only re-binds the pipeline.
4. `cb.commit()` + `cb.waitUntilCompleted()`.

Two structural pains:

- **ICBs are a hack for sequencing.** Apple's compute-ICB API ships
  only `MTLIndirectCommandType::ConcurrentDispatch` — there is no
  serial-ICB primitive. Today's serial guarantee comes from the
  encoder's default `dispatchType=Serial`, which serializes
  consecutive `executeCommandsInBuffer` calls on the same encoder.
  Cross-ICB ordering rides the encoder, not the ICB. Functional but
  bent: we use a parallel-dispatch primitive to express a serial
  pipeline.
- **CPU encode cost.** Synth-attention kernel has ~18 buffer
  bindings per dispatch; even amortized through bake-time ICB
  recording the encoder still pays the `setComputePipelineState`
  call per step + the `executeCommandsInBuffer` jump. CPU encode is
  60–80µs/forward today (post-`717664b58`).

MTL4 closes both:

- `MTL4ComputeCommandEncoder.dispatchThreadgroups_threadsPerThreadgroup`
  is direct compute dispatch with first-class sequencing — no ICB
  serialization workaround needed.
- `MTL4ArgumentTable` is a pre-built binding-set that the encoder
  references via one `setArgumentTable(...)` call. Replaces the
  per-dispatch `setBuffer × N` and the ICB pre-recording machinery
  with one table bind per kernel.

Plus the M1 Max status-5 regression
(`project_ferrite_metal_status.md`) is plausibly an ICB-on-Serial
hazard the M1 driver exposes; MTL4's native sequencing should
sidestep it.

## Gating

- **OS-gated**, not hardware-gated. `MTL4CommandQueue` requires
  macOS 15 (Sequoia) / macOS 26 + Apple Family 7+ (M1 onward). No
  M4 exclusivity.
- **Runtime probe**, not a build-time cut: call
  `device.newMTL4CommandQueue()`; `Some` → MTL4 available, `None` →
  fall back to MTL3 path. Lets a single binary run on macOS 14 OR
  macOS 15+, and on M1 OR M4.
- Hard cut deferred to Scope B once the perf win is measured.

## Leverage surface — what "fully leverage MTL4" means

MTL4 is not just a command-submission swap. The four levers, in
priority order:

| # | Lever | Win type | Where it lands |
|---|---|---|---|
| A | `MTL4ComputeCommandEncoder` + `MTL4ArgumentTable` | CPU encode (60–80µs → ~20µs/forward expected) | Phase A below |
| B | `MTL4Compiler` threaded build + `MTL4PipelineDataSetSerializer` on-disk cache | Startup time (parallel pipeline build + persist across runs) | Phase B below |
| C | Explicit barriers — no auto-hazard-tracking | GPU concurrency across non-dependent steps. **Actual perf lever beyond CPU savings.** | Phase C below |
| D | `MTL4StitchedFunctionDescriptor` — link-time function stitching | Replace source-level `fuse_pass::synthesize_*` with atom-graph stitched at pipeline build. Research-grade. | Tracked follow-up |

A is necessary; B is independent and high-value-for-startup; C is
the actual GPU perf win on top of A (today's Serial encoder
serializes *everything* — conservative). D is the closest MTL4 gets
to a structural shift in the synth pipeline.

## Phase A — side-by-side encoder + argument-table swap

Goal: gated MTL4 path, measure 5-run decode tok/s vs ICB baseline.
Commit gate: **≥5% win** on Llama-3.2-1B-4bit decode → Phase B
(and rip-out of MTL3).

### A.1. Probe + Cargo feature wiring (small)

- Enable objc2-metal MTL4 features in
  `vllm-rs/crates/ferrite-metal-kernels/Cargo.toml` +
  `vllm-rs/crates/ferrite-forward/Cargo.toml`:
  - `MTL4ArgumentTable`, `MTL4BufferRange`, `MTL4CommandAllocator`,
    `MTL4CommandBuffer`, `MTL4CommandEncoder`, `MTL4CommandQueue`,
    `MTL4ComputeCommandEncoder`, `MTL4ComputePipeline`.
- Extend `__re` module with MTL4 type aliases.
- `MetalWorkerPool::new`: probe `device.newMTL4CommandQueue()` once
  at construction, store `Option<MTL4Queue>`. Log
  `"MTL4 available: yes/no"` at info level.
- Env var `FERRITE_METAL_MTL4=1` opt-in. Default = MTL3 (today's
  path). If MTL4 env-on AND probe-yes → use MTL4 path; else MTL3.

### A.2. MTL4 bake-time path (medium)

At bake time (today: `bucket_bakings[i].icb` records every
dispatch's bindings into the ICB), build instead:

```
struct Mtl4BakedStep {
    pipeline: Retained<ProtocolObject<dyn MTL4ComputePipelineState>>,
    arg_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    threadgroups: MTLSize,
    threads_per_threadgroup: MTLSize,
}
```

Per-bucket: `Vec<Mtl4BakedStep>` replacing the ICB for the MTL4
path. Keep both in `BucketBaking` for now — selected at run time by
the env-var gate.

**Pipeline conversion.** `MTL4ComputePipeline` is built from the
same AIR/metallib bytes via `MTL4Compiler`. Per
`MTL4Compiler.newComputePipelineState...`. Per-kernel-symbol
pipeline build at bake time (post-AOT-compile, same metallib
inputs).

**Argument table build.** Per step, one MTL4ArgumentTable:
- `desc.maxBufferBindCount = N` (N = number of buffer bindings for
  the kernel, today 8–18).
- For each binding `i`: `table.setAddress_atIndex(buf.gpuAddress() + offset, i)`.
- Buffer GPU addresses are stable for the buffer's lifetime; arena
  + weight buffers are pinned, so the table is bake-once + reuse.
- Runtime-bindings buffers (`input_ids`, `positions`,
  `slot_mapping`, etc.) are also pinned across forwards — their
  contents change per forward but the GPU address is stable. Same
  bake-once treatment.

### A.3. MTL4 run-time path (medium)

Replacement for `worker.rs:run_bucket`:

```
let cb = queue.commandBuffer();   // MTL4CommandBuffer
let enc = cb.computeCommandEncoder();
for step in &baking.mtl4_steps {
    enc.setArgumentTable(Some(&step.arg_table));
    enc.setComputePipelineState(&step.pipeline);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        step.threadgroups,
        step.threads_per_threadgroup,
    );
}
enc.endEncoding();
cb.commit();
cb.waitUntilCompleted();
```

No ICB. No `executeCommandsInBuffer`. No per-step encoder
end+reopen. MTL4's encoder is natively serializing across
`dispatchThreadgroups` calls — that's the whole reason for the
migration.

**Residency set.** Carries over unchanged
(`MetalResidencySet.attach_to_queue` works on both MTL3 and MTL4
queues per Apple's docs).

### A.4. Bench + decide

5-run greedy decode @ M4:
- `mlx-community/Llama-3.2-1B-Instruct-4bit` 200 tok
- `mlx-community/Llama-3.2-3B-Instruct-4bit` 200 tok

Both paths under `FERRITE_METAL_TRACE=1` for encode/commit/wait
breakdown.

Gate: ≥5% decode tok/s win on either model → land Scope B (rip
out MTL3 + ICB path).

If M1 Max is reachable: also run the MTL4 path there. If MTL4
sidesteps status-5, the M1 user is unblocked independent of perf
results.

## Phase B — MTL4Compiler + serialized pipeline cache

Independent of Phase A's perf gate; lands as a startup-time win.

Today: `SpecializedPipelines::build_*` walks the bucket plan,
calls `device.newComputePipelineStateWithFunction(...)` serially per
unique (symbol × constants) tuple. On Llama-3.2-3B that's dozens of
pipelines; cold pool build is ~150–300ms of pipeline-state
creation alone.

MTL4 offers:

- **`MTL4Compiler`** (created via
  `device.newMTL4CompilerWithDescriptor_error`). Multiple compilers
  can run in parallel; each compiler builds pipelines on its own
  thread. Map this onto a tokio/std `JoinSet` of compile tasks.
- **`MTL4PipelineDataSetSerializer`** + binary archive
  (`MTL4Archive`). Persist all built `MTL4ComputePipelineState`s to
  disk, keyed on the same canonical-hash we already compute for
  AOT metallib paths. Second cold start: deserialize archive →
  `MTL4PipelineState` directly, skip per-pipeline build entirely.

Phasing:

- **B.1.** Move pipeline-state creation to `MTL4Compiler` (still
  serial, single compiler). Smoke-test that AOT metallib bytes feed
  into `MTL4Compiler.newComputePipelineState...` cleanly.
- **B.2.** Parallel compile across pipelines (N compilers, JoinSet).
  Measure cold-start delta on Llama-3.2-3B-4bit.
- **B.3.** Add `MTL4PipelineDataSetSerializer` write at pool-build
  end + read at pool-build start. Cache under
  `$XDG_CACHE_HOME/ferrite/metal-pipelines/<canonical-hash>.archive`.
  Invalidation: archive embeds canonical-hash + AOT metallib hash;
  mismatch → rebuild.

## Phase C — explicit barriers, kill conservative serialization

The **actual** GPU perf lever beyond Phase A's CPU savings.

Today: encoder default `dispatchType=Serial` serializes every
adjacent `executeCommandsInBuffer` (or per Phase A,
`dispatchThreadgroups`) call. Conservative — every step waits for
every prior step to finish.

MTL4 disables auto hazard tracking on the compute encoder;
hazards become the programmer's responsibility via explicit
barriers (`memoryBarrierWithScope:` analogue —
`MTL4ComputeCommandEncoder.memoryBarrierAfterEncoderStages_beforeEncoderStages_afterEncoderQueueStages_beforeEncoderQueueStages`
+ scope variants). This lets non-RAW step pairs run concurrently.

Phasing:

- **C.1.** Per-step read/write slot sets. `BucketStep::Icb` already
  carries `step_resources`; promote to a typed `{reads:
  SlotMask, writes: SlotMask}` shape. Compute at bake time.
- **C.2.** Bake-time barrier insertion. Walk the linearized step
  list; insert a `MemoryBarrier` step between steps `i` and `j>i`
  iff `j.reads ∩ i.writes ≠ ∅` (RAW) or `j.writes ∩ i.writes ≠ ∅`
  (WAW) or `j.writes ∩ i.reads ≠ ∅` (WAR). On the synth decode
  loop the chain (RmsNorm → QKV → RoPE → Attn → o_proj) is fully
  RAW so barrier-density stays high; the wins are in:
  - Cross-bucket-segment boundaries (today serialized
    pessimistically, the boundary is often write-disjoint).
  - Within `SynthMlpPreDown`'s gate/up parallelism (gate-matmul
    and up-matmul read the same input but write disjoint outputs
    — concurrency-safe today but blocked by Serial encoder).
- **C.3.** Validate via direct A/B: `FERRITE_METAL_MTL4_BARRIERS=safe`
  (barrier between every step, equivalent to Serial) vs
  `=optimal` (computed mask). Output must be bit-equivalent.

Expected upside: 5–15% decode tok/s on top of Phase A, model
dependent. Less on Llama (chain-heavy), more on MoE / parallel
branches.

## Phase D — stitched compute pipelines (follow-up, research)

Tracked, not committed. Replace `fuse_pass::synthesize_pre_attn_chunk`'s
source-level synthesis with `MTL4StitchedFunctionDescriptor` +
link-time pipeline stitching. Each atom (`AddRmsNormAtom`,
`AffineQmvAtom`, `RopeAppendAtom`, `SiluMulAtom`) registers as a
standalone MSL function; the stitched descriptor wires the
producer→consumer graph at pipeline-build.

Why interesting: kills the `fuse_pass` source-template machinery
(askama / proc-macro-time codegen) for a smaller, atom-graph-driven
pipeline cache. The hard part is whether the stitched-function
linker performs the same register-resident dataflow our hand-fused
source synth gets today — Apple's docs are thin and there's no
public benchmark of stitched vs source-fused performance. Worth a
1-day probe after Phase C lands.

## Scope-out — final cleanup once A/B/C land

- Delete `BucketBaking.icb` + `baked_resources` ICB-only fields.
- Delete `worker.rs::run_bucket_per_step_*` debug paths.
- Delete `FERRITE_METAL_PER_STEP_CMDBUF` opt-out.
- Hard-require macOS 15 (or keep MTL3 fallback if user telemetry
  shows macOS-14 holdouts).

## Risks + open questions

- **`MTL4Compiler.newComputePipelineState` accepts AOT-compiled
  `.metallib` bytes.** Should — AIR is shared with MTL3. Verify in
  A2.
- **Function-constant specialization.** Today: `MTLFunctionConstantValues`
  + `newFunctionWithName_constantValues`. MTL4 path:
  `MTL4SpecializedFunctionDescriptor`. Same data, different
  builder. Probably mechanical translation but unverified.
- **`useResources` for non-residency-set buffers.** MTL4 has its
  own residency model via `MTL4CommandAllocator`. The current
  residency-set call paths need a translation — should be a 1:1
  swap but worth verifying.
- **Argument-table buffer-count cap.** Today's largest binding
  count is ~18 (synth pre-attn). Default
  `MTL4ArgumentTableDescriptor.maxBufferBindCount = 8`; must bump
  per kernel.
- **NOT `MTL4MachineLearningCommandEncoder` /
  `MTL4MachineLearningPipeline`.** Those wrap CoreML-style ML ops
  and are not our path. Custom MSL stays custom MSL.

## Validation

- `vllm chat --device metal --model
  mlx-community/Llama-3.2-3B-Instruct-4bit -p "What is the capital
  of France?"` → coherent output on both MTL3 and MTL4 paths.
- 8-prompt golden sweep matches the post-P10c baseline (the MLX
  greedy goldens in `vllm-e2e/testdata/golden/`).
- `ferrite-forward --lib` test suite green on both paths.

## Build / verify

```
cd vllm-rs && FERRITE_MODELS=llama-3.2-3b-mlx-affine-b4-g64 \
  cargo build --release -Fmetal --bin vllm

# MTL3 (default, today)
./target/release/vllm chat --device metal \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --bench --max-tokens 200 -p "Write a 200-word poem about the ocean."

# MTL4 (Scope A side-by-side)
FERRITE_METAL_MTL4=1 ./target/release/vllm chat --device metal \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --bench --max-tokens 200 -p "Write a 200-word poem about the ocean."
```

## Hard rules carried forward

- `feedback_no_handcoded_fusion`, `feedback_no_silent_deferrals`,
  `feedback_build_flags`, `feedback_commit_msg_via_file`,
  `feedback_no_gpg_sign`.
- Scope A keeps both paths; do **not** strip the MTL3 path before
  the perf gate passes (`feedback_no_silent_deferrals`).
