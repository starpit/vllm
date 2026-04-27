# KVM_MAPPING — Phase 2 encoder design (P2-1)

> Output of step P2-1 of `MEGA_HANDOFF.md`. Resolves the five
> mapping unknowns and pins the bidirectional table between
> `Instruction<W>` (`vllm-rs/crates/ferrite-forward/src/instr.rs`)
> and the 13 TK throughput opcodes
> (`vllm-rs/third_party/megakernels/demos/cross-gpu-llama/llama.cuh`).
> Read this **before** writing `interpreters/kvm_mega.rs` (P2-2).
>
> Scope: single-GPU Llama-3.2-1B / Llama-style decoder bring-up.
> Multi-GPU (`num_devices > 1`) and barrier-row synthesis stay
> out of scope (P2-7 follow-up).

## Source pins

The row layouts here are read directly from these files; if they
drift, this doc is wrong:

- Vendor opcodes + globals struct:
  `third_party/megakernels/demos/cross-gpu-llama/llama.cuh:10-24,77-294`
- Vendor row-payload reads (per op `parsed_instruction`):
  - `batched_rms_norm.cu:24-46` — Norm rows (Attn / Mlp / LM_Head)
  - `qkv_rope_append.cu:32-39` — QKV+RoPE+KvAppend
  - `attention_decode.cu:70-81` — batched decode
  - `attention_prefill.cu:42-46` — per-Q-block prefill
  - `matmul_adds.cu:18-26,166,199` — O_ProjResidual / DownProjResidual
  - `gate_silu.cu:22-28` — GateSiLU
  - `up_matmul.cu:28-34` — UpMatmul
  - `lm_head.cu:21-27` — LM_Head
  - `all_device_barrier.cu:14-17`, `inc_barriers.cu:17` — out of
    scope for P2 bring-up (multi-GPU only)
- Older-branch encoder reference (the shape we mirror):
  `crates/ferrite-forward/src/tk_instructions.rs` (already ported)
- Instruction IR (the source side):
  `crates/ferrite-forward/src/instr.rs:80-164`

## Row format invariants

- All rows are `[i32; 32]`. Slot 0 is the opcode. Trailing slots
  are zero-padded by the emitter (mirror prim_mega's discipline).
- DAG ordering = tape position. A consumer op's loader spin-waits
  on the producer op's `Bar` counter. The encoder does not insert
  cross-op sync rows for single-GPU.
- "Layer" semantics are per-opcode and not always the model layer
  index — see the per-row tables below (LM_Head reuses slot 1 as
  unused; Attn/Mlp/QKV use it as model layer; in `tk_instructions.rs`
  the LM_Head row pins layer=0 by convention).
- The encoder runs **after** loop unrolling: `Loop` rows do not
  exist in the emitted tape (mirroring prim_mega's `try_encode_bucket`
  which re-expands at encode time).

## Bidirectional table — `Instruction<W>` ↔ TK opcode

| TK opcode | Numeric | Producer `Instruction<W>` variant(s) | Encoder fan-out (single-GPU) |
| --- | ---: | --- | --- |
| `OPCODE_AttnNorm` | 1 | `RmsNorm` (post-residual, attn position) | `batch_size` rows × 1 (one row per token, `num_items=1`) |
| `OPCODE_QKV_RopeAppend` | 2 | `FusedQkvRopeCache` ∨ `FusedQkvRopePrefill` | `(batch_size / matmul_batch_block_size) × ((Q+2KV)*HEAD / matmul_out_block_size)` |
| `OPCODE_GQA_AttentionPrefill` | 3 | `AttentionPrefillContiguous` ∨ `FlashInferAttentionPrefill` (prefill mode only) | `Σ_seq ⌈q_len / 16⌉ × num_kv_heads` |
| `OPCODE_GQA_AttentionDecode` | 4 | `AttentionViaCache` ∨ `FlashInferAttentionDecode` (decode mode only) | `⌈(num_tokens × num_kv_heads) / 14⌉` (14 = (32-3)/2 batched pairs) |
| `OPCODE_O_ProjResidual` | 5 | `CutlassGemmAdd` (attn-post position) | `num_batch_blocks × num_output_blocks` |
| `OPCODE_MlpNorm` | 6 | `RmsNorm` (post-residual, mlp position) | `batch_size` rows × 1 |
| `OPCODE_GateSiLU` | 7 | `FusedGateUpSiluMul` (gate-half slice) | `num_batch_blocks × num_intermediate_blocks` |
| `OPCODE_UpMatmul` | 8 | `FusedGateUpSiluMul` (up-half slice) | `num_batch_blocks × num_intermediate_blocks` |
| `OPCODE_DownProjResidual` | 9 | `CutlassGemmAdd` (mlp-post position) | `num_batch_blocks × num_output_blocks` |
| `OPCODE_LM_HeadNorm` | 10 | `RmsNorm` (final, post-last-DownProj) | `batch_size` rows × 1 |
| `OPCODE_LM_Head` | 11 | `CutlassGemm` ∨ `Gemm` (vocab-projection) | `num_batch_blocks × (vocab_size / matmul_out_block_size)` |
| `OPCODE_Barrier_Inc` | 12 | — | not emitted (multi-GPU only) |
| `OPCODE_AllDeviceBarrier` | 13 | — | not emitted (multi-GPU only) |

`Embed` is NOT a TK opcode — it stays an external pre-megakernel
launch (the older branch's vendor demos start the megakernel after
embeddings are in `g.hidden_states`). The encoder emits no row for
`Embed`; the launcher handles it host-side before kicking the
megakernel (mirrors how vendor's `tp_generate.py` runs embed
outside the kernel).

## Per-opcode row payloads

Each row's slot list. `[i32]` = literal i32 from the IR; `runtime`
= filled by the launcher post-template-copy (matches prim_mega's
`RuntimeSource`); `synth` = encoder-synthesized from tape position.

### `OPCODE_AttnNorm` / `OPCODE_MlpNorm` / `OPCODE_LM_HeadNorm`

```
[opcode, layer_idx, num_items=1, local_batch_idx_0, 0, 0, ..., 0]
```

Source: `batched_rms_norm.cu:29-32`. The kernel can batch multiple
batch indices per row (`num_items > 1`, indices in slots 3..3+num_items),
but the older-branch encoder emits one row per batch index with
`num_items=1` for simplicity. We match that; revisit only if
profiling shows the per-row controller overhead matters.

| Slot | Source | Notes |
| ---: | --- | --- |
| 0 | const `OPCODE_*Norm` | |
| 1 | const `layer_idx` | post-loop-unroll (= baseline + iter) |
| 2 | const `1` | `num_items` |
| 3 | const `local_batch_idx` | one row per padded batch position; emitted for ALL `batch_size` indices, including padding (so downstream `Bar` counters reach the expected count — see `tk_instructions.rs:86-88` comment) |
| 4–31 | const `0` | zero-pad |

### `OPCODE_QKV_RopeAppend`

```
[opcode, layer, local_row, local_col, row, col, 0, ..., 0]
```

Source: `qkv_rope_append.cu:32-39`. For single-GPU `local==global`
so the encoder duplicates: `local_row = row = batch_block`,
`local_col = col = qkv_block`.

The same opcode covers both prefill and decode (the kernel
conditions on `g.num_prefill_tokens` set per launch — see Q2 below).

| Slot | Source | Notes |
| ---: | --- | --- |
| 0 | const `OPCODE_QKV_RopeAppend` | |
| 1 | const `layer` | |
| 2 | const `local_row` | = `batch_block` |
| 3 | const `local_col` | = `qkv_block` |
| 4 | const `row` | = `batch_block` (single-GPU) |
| 5 | const `col` | = `qkv_block` (single-GPU) |
| 6–31 | const `0` | zero-pad |

### `OPCODE_O_ProjResidual` / `OPCODE_DownProjResidual` / `OPCODE_GateSiLU` / `OPCODE_UpMatmul` / `OPCODE_LM_Head`

```
[opcode, layer, local_row, local_col, row, col, 0, ..., 0]
```

Source: `matmul_adds.cu:18-26`, `gate_silu.cu:22-28`,
`up_matmul.cu:28-34`, `lm_head.cu:21-27`. Identical 6-int payload
across all matmul-shape ops. For LM_Head, slot 1 (`layer`) is by
convention `0` (LM_Head runs once at end-of-forward; see
`tk_instructions.rs:243`).

The residual-add is folded into `OPCODE_O_ProjResidual` and
`OPCODE_DownProjResidual` via the matmul's storer reading
`hidden_states` as the C operand (`matmul_adds.cu:166,199`:
`MatMulAddOp<&Globals::*, &Globals::*, &Globals::hidden_states, ...>`).
The norm ops downstream therefore read the post-residual
`hidden_states` directly — there is no separate `Add` row.

### `OPCODE_GQA_AttentionDecode`

```
[opcode, layer, num_entries=2*num_pairs, seq_0, kv_0, seq_1, kv_1, ...]
```

Source: `attention_decode.cu:70-81`. Variable-length payload,
batched up to `(32-3)/2 = 14` `(seq_idx, kv_head_idx)` pairs per
row. `num_entries = num_pairs * 2` (the kernel reads
`s.instruction()[2] / 2` to get the pair count).

| Slot | Source | Notes |
| ---: | --- | --- |
| 0 | const `OPCODE_GQA_AttentionDecode` | |
| 1 | const `layer` | |
| 2 | const `num_entries = 2 * num_pairs` | |
| 3 + 2k | const `seq_idx` | k in `[0, num_pairs)` |
| 4 + 2k | const `kv_head_idx` | k in `[0, num_pairs)` |
| trailing | const `0` | zero-pad |

### `OPCODE_GQA_AttentionPrefill`

```
[opcode, layer, seq_idx, prefill_block_idx, prefill_token_offset, kv_head_idx, 0, ..., 0]
```

Source: `attention_prefill.cu:42-46`. One row per
`(seq, q_block_of_16, kv_head)` triple. `prefill_token_offset =
seqused_k - num_q_tokens` for that sequence (post-history-append).

## The five unknowns — resolution

### Q1 — Residual fusion shape

**Decision: option (b) — match-by-IR-variant; do not peephole-fuse
in the encoder.** The forward DSL must emit the unfused triple
`Gemm + Add + RmsNorm` (or use `CutlassGemmAdd + RmsNorm`) at
positions that feed a TK row. The solver picks per cost+feasibility:

- On Hopper-KvmMega-targeting paths: solver picks
  `KvmCutlassGemmAddImpl` over `(Gemm, Add)` (claims 2 tiles,
  emits `OPCODE_O_ProjResidual` or `OPCODE_DownProjResidual` for
  the projection-with-residual position) and `KvmRmsNormImpl` over
  `RmsNorm` (claims 1 tile, emits `OPCODE_*Norm`). No `Add` rows
  reach the encoder because they're claimed.
- On host paths: solver picks `HostFusedAddRmsNormImpl` over
  `(Add, RmsNorm)` instead — the unfused triple shape lets host
  keep its existing fusion benefit without forcing kvm to deal
  with `FusedAddRmsNorm` as a single IR variant.

**Why this and not (a) "encoder collapses adjacent rows":** the
encoder is a closed match per IR variant per
`feedback_no_special_case_macros` and `feedback_opkind_is_math_not_fusion`.
A peephole-fusion path inside the encoder would be a hand-rolled
optimizer competing with the solver — exactly the anti-pattern
those rules block.

**Action item before P2-2:** verify the llama (and cohere /
gemma3 / etc.) forward DSLs actually emit the unfused triple and
not `FusedAddRmsNorm` directly. If they emit `FusedAddRmsNorm`,
the DSL must be refactored to emit `Add + RmsNorm` first; that
refactor is wholesale across every arch per
`feedback_no_piecemeal_codegen_migration`. **This is a precondition
for P2-2; do not start the encoder code without confirming the
DSL shape first.**

**Why mapping `FusedAddRmsNorm` directly to `OPCODE_AttnNorm` is
wrong:** the TK norm ops do not perform any add. They read from
`g.hidden_states` (already post-residual) and write to a separate
`rms_*_intermediates` buffer (`batched_rms_norm.cu:106,136,257`).
The residual-add lives entirely in the prior matmul-with-residual.
Mapping `FusedAddRmsNorm` to `OPCODE_AttnNorm` directly would
double-add (the prior `OPCODE_O_ProjResidual` already added).

### Q2 — QKV decode/prefill collapse

**Decision: both `FusedQkvRopeCache` and `FusedQkvRopePrefill` map
to `OPCODE_QKV_RopeAppend`.** Identical 6-int row layout. The TK
op conditions on `g.num_prefill_tokens` set per launch
(`qkv_rope_append.cu` reads it from globals — confirmed in handoff
"What's done"). The encoder does not need to know the variant tag
beyond which IR field set to read.

**Constraint for the launcher (not the encoder):** `g.num_prefill_tokens`
must be set to either `num_tokens` (full prefill) or `0` (full
decode) per call. **Mixed prefill+decode in a single megakernel
forward is not supported in P2 bring-up.** This matches the
older branch's verified shipping path. Chunked prefill stays on
host until P2-8 (out of scope here).

### Q3 — Gate/Up split

**Decision: encoder splits `FusedGateUpSiluMul` into two TK
opcodes (`OPCODE_GateSiLU` + `OPCODE_UpMatmul`) at encode time.**

Per matmul fan-out: `num_batch_blocks × num_intermediate_blocks`
rows for each. Total fan-out from one IR `FusedGateUpSiluMul`
tile is `2 × num_batch_blocks × num_intermediate_blocks`.

**Solver claim-mask precondition (per `feedback_solver_claim_mask_size`):**
the picked `KvmFusedGateUpSiluMulImpl`'s `fan_out` is wider than
any current Impl (PrimMega's widest is `OP_CUTLASS_GEMM_SPLITK`'s
single-row claim). Verify K bound + `u8`/`u16` typing in `solver.rs`
covers the new fan-out before P2-3 lands; bump if necessary, with
a unit test pinning it (mirror `solver_picks_dc_siblings_on_h100`).

**Why split, not "register two tiles per Impl":** TK has them as
two opcodes because their data dependencies and storer barriers
differ — UpMatmul's storer is the one that increments `Bar` for
the downstream `OPCODE_DownProjResidual`. Splitting at encode time
matches the kernel-side reality. The IR fusion is a host
optimization; the encoder unwinds it for kvm.

### Q4 — GemmAdd routing (which residual TK opcode?)

**Decision: encoder phase-state machine.** `interpreters/kvm_mega.rs`
maintains a `Phase { Attn, Mlp }` state per layer, transitioning
on the IR's structural landmarks:

```
state := Attn at AttnNorm
       | Mlp  at MlpNorm
       | reset to Attn at start of each new layer
```

On `CutlassGemmAdd`:
- `Phase::Attn` ⇒ emit `OPCODE_O_ProjResidual`
- `Phase::Mlp`  ⇒ emit `OPCODE_DownProjResidual`

**Why not weight_fn name pattern-match (e.g., contains "o_proj"):**
brittle across archs; cohere / gemma3 / qwen3 all use slightly
different naming conventions. State-from-tape-position is structural
and works wholesale.

**Why not plumb the picked Impl identity onto OpInstance:** that's
a refactor with blast radius across every Impl + the host emitter.
Out of P2 scope.

**Edge case — DSLs without a separate MlpNorm landmark:** for
arches that fold MLP norm into Down (none today), the state machine
needs reconsideration. Document the assumption and fail loudly
(panic at codegen) if the IR sequence doesn't contain the expected
norm landmarks. Mirror prim_mega's "no `_` catch-all" discipline.

### Q5 — Barrier synthesis

**Decision: no barrier rows for single-GPU bring-up.** Confirmed:
the older branch's `tk_instructions::build_throughput_instructions`
(now ported at `crates/ferrite-forward/src/tk_instructions.rs`) does
NOT emit `OPCODE_Barrier_Inc` or `OPCODE_AllDeviceBarrier`. The
opcode constants are commented-out in `tk_instructions.rs:26-27`
because they're unused for `num_devices = 1`.

Cross-instruction synchronization on a single device is handled
entirely by:
- Each op's storer increments `g.Bar[dev_idx][{layer, prev_op-1, batch_block, 0}]`
- The next op's loader spin-waits on that counter via
  `wait_on_barrier<Scope::GPU>(...)` (or `Scope::SYS` on multi-GPU)

This is in `batched_rms_norm.cu:285-345` (gmem_waiter pattern).
The encoder emits no rows for it; the kernel-side per-op code
already wires the counters.

**Multi-GPU follow-up (P2-7):** when `LLAMA_NUM_DEVICES > 1`, the
encoder synthesizes barrier rows between sharded phases per the
older branch's NOT-YET-PORTED multi-GPU `build_throughput_instructions`
variant. Out of scope for P2-1..P2-6.

## Mega-ineligible variants (closed-match, no `_` arm)

For parity with prim_mega's discipline (`feedback_no_refusal_chasing`,
`feedback_no_special_case_macros`), `interpreters/kvm_mega.rs`'s
`encode_op_arm` must enumerate every IR variant and either return
`Some(row)` or `None` (canonical kvm-ineligible). Any unlisted
variant panics at macro-expand time, not runtime.

KvmFit-eligible (have a TK opcode):

```
RmsNorm, FusedQkvRopeCache, FusedQkvRopePrefill,
AttentionViaCache, AttentionPrefillContiguous,
FlashInferAttentionDecode, FlashInferAttentionPrefill,
CutlassGemmAdd, CutlassGemm (only at LM_Head position; otherwise None),
FusedGateUpSiluMul, Embed (no row — handled pre-launch)
```

KvmFit-ineligible (no TK opcode in cross-gpu-llama set):

```
LayerNorm, Reshape, Add, ScalarMul, TanhSoftCap,
FusedAddRmsNorm (see Q1 — DSL must un-fuse to be kvm-eligible),
FusedAddRmsNormWithOffset, ScalarOffsetRmsNorm,
FusedGemmBias, FusedGateUpGeluMul, FusedQkvQkNormRopeCache,
SlidingAttentionViaCache, SlidingAttentionPrefillContiguous,
RopeAppend, MlaSplit, MlaAttention, DeepSeekMoe,
CutlassGemmSplitK, CutlassGemv, CutlassFusedGemmBias,
CutlassFusedGateUpSiluMul, Marlin*, Bnb4*, Fp8*
```

A canonical with any KvmFit-ineligible variant in its picked Impl
set has `pick_interpreter` fall back to `Host` (or `PrimMega` if
the bucket is prim-mega-eligible). All-or-nothing per layer per
the existing `MegakernelFit` tier semantics.

## Ordering invariant (sanity-check pin)

Per layer, in tape order (before LM_Head):

```
[batch_size × OPCODE_AttnNorm]
[num_batch_blocks × num_qkv_blocks × OPCODE_QKV_RopeAppend]
[either prefill OR decode attn rows — not mixed in P2]
[num_batch_blocks × num_output_blocks × OPCODE_O_ProjResidual]
[batch_size × OPCODE_MlpNorm]
[num_batch_blocks × num_intermediate_blocks × OPCODE_GateSiLU]
[num_batch_blocks × num_intermediate_blocks × OPCODE_UpMatmul]
[num_batch_blocks × num_output_blocks × OPCODE_DownProjResidual]
```

After all layers:

```
[batch_size × OPCODE_LM_HeadNorm]
[num_batch_blocks × num_logit_blocks × OPCODE_LM_Head]
```

P2-2's encoder tests assert this row-count and ordering for
synthetic OpInstance sequences (mirror
`tk_instructions::tests::basic_instruction_generation`).

## Open items surfaced (block P2-2 if hit)

1. **DSL shape for residual fusion.** Per Q1: confirm the forward
   DSL emits `Gemm + Add + RmsNorm` (unfused) and not
   `FusedAddRmsNorm` directly. If the latter, refactor wholesale
   first.
2. **Solver claim-mask K bound** for `KvmFusedGateUpSiluMulImpl`'s
   `2 × num_batch_blocks × num_intermediate_blocks` fan-out
   (Q3). Add unit test before P2-3.
3. **Mixed prefill/decode in one forward.** Out of P2 scope per
   Q2. If the calling code allows it, the launcher must split
   into two megakernel calls (or fall to host).

## Out of scope (P2-7+)

- Multi-GPU barrier-row synthesis (Q5).
- `OPCODE_Barrier_Inc` / `OPCODE_AllDeviceBarrier` row emission.
- Cross-GPU `redAdd` orchestration in the launcher.
- Cohere / Granite / Gemma3 per-arch megakernel TUs (P2-6).
