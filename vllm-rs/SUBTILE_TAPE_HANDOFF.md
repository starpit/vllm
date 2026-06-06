# Handoff — ff-subtile worktree, post-Commit 8 + §10 metal_tape scrub

## Where to work

- **Worktree:** `/Users/nickm/git/vllm/.claude/worktrees/ff-subtile`
- **Branch:** `worktree-ff-subtile`
- **Crate:** `vllm-rs/crates/ferrite-wavefront/`
- **HEAD:** `c260bd19b1` — §10 dead-arm scrub of metal_tape.rs (-1318 LOC)

`pwd && git branch --show-current` first thing — verify the worktree, per `memory/feedback_handoff_worktree_match.md`.

## State

**Substrate is plan-clean through Commit 8 + §10 metal_tape scrub.** Multi-round audit (workflows `wth12p261`, `w0vid1jxt`) closed every BLOCKER and DRIFT in player triviality / compile-time-or-garbage / K2 / K8 metal_tape clusters. Remaining drifts are all staged future-commit work (§4 commits 7, 10 lower.rs / region_schedule.rs).

What landed in this session:

| Commit | Description |
|---|---|
| `f8a359d71f` | Slot-lifecycle SubtileTape rebuild (hazards-explicit) |
| `56413bdf6f` | Nuke ComputeBody and BufId from TkTape |
| `70e1b89b52` | Align plan + constraints with landed reality |
| `389d43170f` | BLOCKER fix: typestate cur_loop, GEMM k typed witness |
| `0bdfc69414` | BLOCKER fix: NonZeroU32 for n_blocks/head_blocks |
| `a87b2c32f0` | Commit 6: `lower_tape_to_tk` (conservative all-gmem) |
| `a71e6854dd` | Commit 6b: `validate_tk_tape` |
| `62d75c5db8` | Player ≤5 lines: flat Sync/Fence/Commit/Wait Instrs |
| `a9dbfc3e6f` | Player ≤5 lines: flat ForLoopOpen*/Close Instrs |
| `5bcb318528` | DRIFT: debug_assert! host-eval shape checks |
| `7fccd147c0` | Commit 8: full player emit |
| `df8bb449d3` | BLOCKER fix: StoreAsyncTyped tk20 wrapper + split PageBarrierWait |
| `d5c283e5ca` | Bake byte-offset string at tape-build time |
| `1e0fd4b545` | ValidatedGraph<F> typed witness for lowering |
| `6ce6dfcfa7` | Collapse LoadAsync/StoreAsync arms to ≤5 lines |
| `b6f2bfd380` | Drop GemmM1 accum if/else from tk20 helper |
| `c260bd19b1` | §10 dead-arm scrub of metal_tape.rs (1471 → 153 LOC, -1318) |

82 unit + 9 doctests green throughout.

## Remaining work (all staged future commits)

1. **`lower.rs` deletion** (213 LOC) — plan §4 commit 7. Requires `to_wavefront.rs` (proc-macro side) to build SubtileIR directly. Multi-crate change.
2. **`region_schedule.rs` deletion** (590 LOC) — plan §4 commit 10. mega.rs/partition.rs still consume Schedule/TapeInstr/play/schedule_from_assignment/ScheduleParams/partition_roundrobin/schedule_wavefront — requires migrating them to `SubtileTape` + `tk_player`. Substantial.
3. **Commit 9 (E2E on H100)** — Pod-only.

## Substrate shape (current)

### SubtileTape (`subtile_tape.rs`)
- `Instr::{ AllocSlot, Compute{node,writes,reads}, FreeSlot, OpenLoop, CloseLoop }` — the slot-lifecycle hazard model.
- `SlotHandle` (move-only) → `SlotWritten` (move-only) → free.
- `SlotId`, `SlotHandle`, `SlotWritten`, `LoopVarId`, `RuntimeBoundId` all sealed via `sealed::Seal(pub(super) ())`.
- `TapeBuilder<S>` typestate: `S::Loop` associated type (Outside = `()`, InsideLoop = `LoopVarId`) — no `Option`, no runtime `expect`.
- `validate_subtile_tape` enforces: compute well-formedness, loop balance, slot lifecycle, slot id range, **edge coverage** (load-bearing — `Compute.reads`'s set-of-writers equals `predecessors()` set).
- `lower_dag_to_tape(&ValidatedGraph<F>) -> SubtileTape` — the `ValidatedGraph` typed witness elides the runtime validate-the-IR gate at lowering entry.

### TkTape (`tk_tape.rs`)
Flat Instr enum (no nested ComputeBody, no LoopCount, no ParityExpr, no ByteOffsetExpr — all folded into per-Instr variants):

- Sync: `SyncthreadsCta`, `SyncthreadsGroup{n_warps}`
- Fence: `ThreadfenceBlock`, `ThreadfenceDevice`, `ThreadfenceSystem`
- Commit/wait: `CommitGroupBulk`, `WaitGroupBulk{n}`
- Page barrier: `BarrierInit`, `PageBarrierWaitStaticP0` / `PageBarrierWaitStaticP1` (parity is a const-generic split per plan §2 row "Phase (parity)" — never a u8 field), `PageBarrierWaitLoopStart0` / `PageBarrierWaitLoopStart1` (same const-generic split on `start_parity`), `PageBarrierArrive`, `ArriveIfRuntimeEven`
- Memory: `LoadAsync(LoadSpec)`, `StoreAsync(StoreSpec)`, `StoreAsyncTyped`
- Compute: `RmsNorm`, `GemmM1{accum:AccumKind}`, `SiluMul`, `ResidualAdd`, `RopeRotateNeoX` / `RopeRotateInterleaved` (const-generic split per §2 RopeForm row — no runtime `RopeFormTag` field; constructor matches once on `F::TAG` to pick the variant), `AttnDecodeInit/Qkt/Sv/Finalise` (each carries `kv_layout: KvLayoutId`; `head_dim` / `num_kv_heads` come from `tape.kv_layout(id)` per §2 line 92 single-source method), `DebugOpBeginMarker`
- Control flow: `ForLoopOpenConst`, `ForLoopOpenKernelArg`, `ForLoopClose`

`ByteOffset(String)` is a sealed pre-baked CUDA fragment type (no enum dispatch at emit time). Source identifiers are `subtile_ir::TensorId` (no `BufId`).

### Player (`tk_player.rs`)
- One arm per Instr; ≤5 lines each.
- All `kittens::*` strings live in the `tk20` sub-module.
- No emit-time arithmetic — `byte_off.as_str()` and `tile_type.as_str()` are pre-baked at tape-build time.
- Helpers (`barrier_name`, `rope_side_str`, `tk20::accum_str`) are sealed-enum-to-`&'static str` translators only — no formula computation. Rope form is encoded by `Instr` variant identity (`RopeRotateNeoX` / `RopeRotateInterleaved`) instead of a translator helper.

### `lower_tape_to_tk` (`lower_tape_to_tk.rs`)
Conservative all-gmem; no analysis, no lookahead, no shmem decisions (those are §6.5 optimizer-pass territory).
- AllocSlot → mint `PageId` (1:1 with SlotId).
- Compute → exhaustive SubOp dispatch.
- AttnDecode 4-phase split: Init in parent frame BEFORE OpenLoop; Qkt+Sv in body; Finalise + store + arrive in parent AFTER ForLoopClose.
- FreeSlot → release slot→PageId mapping.
- OpenLoop / CloseLoop → flat `ForLoopOpen*` and `ForLoopClose`; body Instrs append into parent frame.
- Witnesses preserved end-to-end: `KvCacheLayout`, `KvCacheProducer`, `SoftmaxStateId`, `RopeForm`.

### `validate_tk_tape` (commit 6b)
- `MissingFenceBeforeArrive`: every `PageBarrierArrive{Done}` on a page with in-flight `StoreAsync` (no fence/wait since) flagged.
- Scaffolding for `WaitWithoutLoad`, `LoopVarMismatch` (filled alongside future passes).
- Wired at `lower_tape_to_tk` exit.

## Hard rules — DO NOT VIOLATE (still binding)

- **Tape is a tape.** No side-tables on SubtileTape.
- **Hazards explicit.** Slot lifecycle + EdgeMismatch validator.
- **Tape runs correctly, if slowly.** `validate_subtile_tape` + `validate_tk_tape` run at lowering exits.
- **Player ≤5 lines per arm, one TK 2.0 call per arm.** No inner-match dispatch beyond sealed-enum-to-token. No emit-time arithmetic.
- **Compile-time-or-garbage.** Typed witnesses (ValidatedGraph, NonZeroU32, S::Loop assoc, KvCacheLayout, GemmK-via-derivation, sealed Seal'd handles).
- **No workers as IR primitives.**
- **`feedback_dogfood_tk20_rust`:** every TK 2.0 call goes through `tk20::*`; no inline `kittens::*` outside that module.

## Build & test commands

- Build: `cargo build -p ferrite-wavefront`
- Test: `cargo test -p ferrite-wavefront`
- Clippy: `cargo clippy -p ferrite-wavefront --tests`
- Audit: `Workflow({scriptPath: "/Users/nickm/.claude/projects/-Users-nickm-git-vllm--claude-worktrees-ff-subtile-vllm-rs/779e6b2f-0507-4b8c-90c3-5424d5968068/workflows/scripts/subtile-conformance-audit-wf_5af277c2-6fc.js"})` — fresh run; check `subagent_tokens > 0` to verify it's not a cache replay.

## Resume command

```
cd /Users/nickm/git/vllm/.claude/worktrees/ff-subtile && pwd && git branch --show-current && git log --oneline -20
```
