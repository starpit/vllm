# Qwen3-Next ferrite support — HANDOFF (append-only)

> **APPEND ONLY.** New entries go at the bottom under a dated
> `## YYYY-MM-DD — <short subject>` header. Never edit or delete an
> existing entry — if a prior note is wrong, append a correction
> entry that names the date and subject of the entry it supersedes.
> The plan (`QWEN3_NEXT_PLAN.md`) is read-only; everything that
> changes — phase status, decisions, deviations, blockers,
> verification results — lives here.
>
> Entry guidance (terse — see
> `MEMORY: feedback_handoff_brevity`): one short paragraph per
> entry. Lead with the fact; the git commit + code carry the detail.
> If an entry claims something is "verified", that means a real
> inference run, not a tokenizer roundtrip
> (`MEMORY: feedback_handoff_verified_means_inference`).

## 2026-05-04 — plan created

Branch `feat/rust` worktree `.claude/worktrees/qwen3-next` off
`13141b7b3`. Read-only plan at `QWEN3_NEXT_PLAN.md`. No code yet.
Phase 1 (OpKind plumbing) is next; nothing started.

## 2026-05-04 — Phases 1+2 merge; landing as one commit

While reading `instr.rs` to scope Phase 1 I confirmed the eval
body for `Instruction::GdnAttention` must dereference
`ctx.fwd.gdn_state`, which is the field added in Phase 2. Shipping
Phase 1 alone would force either an `unimplemented!` placeholder
(banned by `feedback_no_unimplemented_singletons`) or a broken
build (banned by `feedback_no_intentional_errors`). Both phases
land as a single wholesale commit — `feedback_no_piecemeal_codegen_migration`
already prefers wholesale, this just makes the unit explicit.
Plan still names them as distinct phases; treat #1 done iff #2 done.

## 2026-05-04 — relocating GdnStatePool to ferrite-kernels

Plan §"Already in tree" notes `vllm_cuda::model::qwen3_next::GdnStatePool`
exists. To name it from `ferrite-forward::ForwardCtx` I'm moving the
struct to `ferrite-kernels::gdn_state` (new module) and re-exporting
from `vllm_cuda::model::qwen3_next` for the existing hand-written
path. No semantic change; the layout (`[num_slots * num_gdn_layers,
conv_dim, state_len]` + `[num_slots * num_gdn_layers, num_v_heads,
head_v_dim, head_k_dim]`) and the GDN kernels' index math are
unchanged. cuda_worker keeps constructing it through the new
ferrite-kernels path.

## 2026-05-04 — Phases 1+2 complete; ready to commit

Wholesale: new `ferrite-kernels::layers_gdn` module
(`GdnStatePool` relocated + new `Qwen3NextGdnLayer::{load,forward}`),
`OpKind::GdnAttention` (parser, classify, shape sig),
`Instruction::GdnAttention` + eval calling
`Qwen3NextGdnLayer::forward`, `info.rs` naming arm,
`ForwardCtx::{gdn_state, gdn_state_indices}`,
`FieldLoad::GdnAttention` + planner picker (reads
`linear_num_*` / `rms_norm_eps` / `layer_types` from config.json) +
`emit_unindexed_let` arm + `emit_layered_load_body` arm,
`GdnAttentionRefImpl` singleton (gates on `linear_num_value_heads`),
`NON_GEMM_NAMES` += `gdn_attention_ref`. cuda_worker now calls
`make_gdn_state_pool(config, ...)` (struct constructor signature
changed to primitives — single-line edit, not an arch arm).
210/210 macro tests pass; ferrite-models + vllm-cli release build
clean. The Impl is dormant until Phase 5 lands the
`ferrite-model-qwen3-next` crate that emits `gdn_attention(...)`.

Committed: `e617bdfb1`.

## 2026-05-04 — Phase 3 partial-RoPE: already covered

Read of `ferrite-kernels/src/rotary.rs:524-607` shows
`RotaryCache::new_partial_from_stream` ships partial-rotary
support (kernel passes non-rotary tail through unchanged), and
`ferrite-forward-macro/src/codegen.rs:2155-2350` already routes to
the partial constructor when `partial_rotary_factor` is in the
model config. So Phase 3's partial-RoPE half is **free** — Qwen3-Next
gets it just by advertising `partial_rotary_factor=0.25` from the
new arch's `model.json`. No code change needed.

## 2026-05-04 — Phase 3 output gate: design decision

`attn_output_gate=True` doubles the Q-projection output and applies
`attn * sigmoid(gate)` before `o_proj`. This is true math, not a
fusion — `feedback_opkind_is_math_not_fusion` says new math gets
its own OpKind. Decomposing into `split` + `sigmoid` + `mul` would
require new DSL primitives that no other arch needs.

Plan: introduce `OpKind::GatedAttention` paralleling `MlaAttention`
(single tile wrapping multiple kernel calls). Inputs:
`(q_gate, k, v, positions, rotary, kv_cache[layer], block_table)`
where `q_gate: [T, 2 * num_heads * head_dim]`. The Impl extracts
q/gate, applies q_norm/k_norm + partial RoPE, runs flash-attention,
sigmoid-gate-multiplies the output, and returns `[T, num_heads * head_dim]`
ready for `o_proj`. Same structural template as
`GdnAttentionRefImpl` from Phase 1+2.

## 2026-05-04 — Phase 4: SharedFusedMoeRefImpl fence landed

Single-line refinement of the Tier-1 Qwen-MoE Impl's `applies_to`:
adds `&& !b.contains_key("linear_num_value_heads")` to the
exclusion list, plus a comment block explaining the
Qwen3-Next-distinctive bound. 210/210 macro tests pass;
ferrite-models clean (Qwen3-MoE configs unaffected — none ship
that bound). The fence is dormant until Phase 5 introduces a
Qwen3-Next config that would otherwise collide.

Committed: `fbda91186`.

## 2026-05-04 — Phase 3 complete

Wholesale: new `ferrite-kernels::layers_attn_gated` module with
`Qwen3NextGatedAttentionLayer::{load, forward}` (port of
`Qwen3NextFullAttention` — fused QKV + q/gate split + per-head Gemma
RMSNorm + Q-only partial RoPE + paged-cache KV write + FA2 with
on-the-fly K rotation + sigmoid output gate + `o_proj`).
`OpKind::GatedAttention` (parser/classify/shape sig),
`Instruction::GatedAttention(in, out, layer, weight_fn, cos_sin_fn)`
with eval threading `ForwardCtx` runtime args plus the model-wide
rotary cache through to the layer's forward; `info.rs` arm;
`FieldLoad::GatedAttention` planner reading
`num_attention_heads` / `num_key_value_heads` / `head_dim` /
`rms_norm_eps` / `attn_output_gate` from `model.json`; emit
arms for `unindexed_let` and `layered_load_body`;
`GatedAttentionRefImpl` singleton (gates on `linear_num_value_heads`);
`NON_GEMM_NAMES` += `gated_attention_ref`. 210/210 macro tests pass;
ferrite-models build clean. Dormant until Phase 5 emits
`gated_attention(...)` from the Qwen3-Next DSL.

Committed: `c30b1f37d`.

## 2026-05-04 — Phase 4 fence reverted in Phase 5

Phase 4's `linear_num_value_heads` exclusion in
`SharedFusedMoeRefImpl::applies_to` turned out to be too eager: when
the Phase 5 arch crate landed it caused a "no Impl matched tile … op
Moe" error on Qwen3-Next configs because the dedicated peer Impl the
fence was anticipating doesn't exist (the plan deferred it as
"likely unnecessary — math is identical"). Math IS identical, so the
right answer per the plan is to let `SharedFusedMoeRefImpl` claim
Qwen3-Next MoE tiles too. The exclusion is removed in the Phase 5
commit; the comment block stays, recasting the rationale ("Qwen3-Next
reuses the same MoE math, attention is fenced separately").

## 2026-05-04 — Phase 5 complete

New `ferrite-model-qwen3-next` crate with:
- DSL body: hybrid
  `if layer % full_attn_period == full_attn_remainder { gated_attention(...) } else { gdn_attention(...) }`,
  then standard `moe_block(normed2, mlp[layer])` for every layer.
- `gated_attention` DSL form takes
  `(x, attn[layer], positions, rotary, kv_cache[layer], block_table)`
  so the FUF carries `ExternKind::Rotary` + `ExternKind::KvCache` +
  `ExternKind::BlockTable` and the codegen plants the `rotary` field
  on `Weights`. (Initial 2-arg form caused the codegen to omit the
  rotary cache; fixed by widening the shape signature to 6 args.)
- `configs/qwen3-next-4-layer.json` (smoke: 3 GDN + 1 gated layer) →
  27 tiles · 19 waves.
- `configs/qwen3-next-80b-a3b-instruct.json` → 291 tiles · 194 waves.
- Workspace + `ferrite-models` registration with `arch-qwen3-next`
  feature; included in `all-arches`.

210/210 macro tests pass; ferrite-models + vllm-cli release builds
clean. Compile-verified, NOT inference-verified — the executor
still needs a `Qwen3NextForCausalLM → Ferrite` arm in
`cuda_worker.rs` and per-request `ForwardCtx::{gdn_state,
gdn_state_indices}` wiring before Phase 6 can run a real prompt.

Committed: `a257d7aca`.

## 2026-05-04 — clarification: load-side dispatch was already correct

Initial Phase 6 framing said "cuda_worker arm dispatching
Qwen3NextForCausalLM to the Ferrite path" was needed. Reread of
`cuda_worker.rs:5077` — `ferrite_forward::try_load(...)` already
runs unconditionally before any per-arch hand-written switch, so
once `ferrite-model-qwen3-next` registers `Qwen3NextForCausalLM`
in its `architectures` (it does), the load is owned by ferrite.
The hand-written `vllm_cuda::model::qwen3_next::Qwen3NextForCausalLM`
fallback at line ~5741 is unreachable for ferrite-claimed loads.
Phase 6 work is therefore confined to *runtime* wiring.

## 2026-05-04 — Phase 6a: executor runtime wiring (compile-clean)

Three surgical edits to `vllm-executor/src/cuda_worker.rs`:
- After ferrite ownership is set and before kv_cache pool init,
  if `ferrite_weights.arch_name() == "qwen3_next"` populate
  `self.qwen3_next_config` from `qwen3_next_config_from_hf(&hf_config)`.
  `init_kv_cache_pool` then constructs the existing `gdn_state_pool`
  for ferrite loads too — same code path the legacy
  `CudaModel::Qwen3Next` variant uses.
- New `CudaModel::is_qwen3_next()` helper returns true for both the
  legacy hand-written variant and ferrite loads whose `arch_name()`
  matches. The dispatch site (the prepared-batch branch) now
  predicates on this method instead of `matches!(model, CudaModel::Qwen3Next(_))`.
- `forward_qwen3_next` gains a `Self::Ferrite` arm: builds
  `ForwardCtx` with `gdn_state` and `gdn_state_indices` populated,
  calls `m.weights.forward(...)`, and applies the same
  last-token-gather epilog the generic Ferrite branch uses. Legacy
  `Qwen3Next` arm unchanged.

vllm-cli release build clean; fmt + clippy clean. NOT inference-verified
yet — needs a Qwen3-Next checkpoint on disk (none cached locally,
and the official 80B-A3B-Instruct is ~80GB). Compile-clean is the
honest claim until a real run lands.

## 2026-05-04 — Phase 6b: golden landed (red), Python vLLM unusable

The 80B-only architecture forced a smaller fixture for Phase 6.
Survey:
- `Qwen/Qwen3-Next-80B-A3B-*` BF16: ~160 GB. ✗
- AWQ-4bit / int4 / Q4 GGUF variants: 40-50 GB. ✗ (14 GB free)
- `Goekdeniz-Guelmez/Qwen3Next-Dev`: 137 MB, real
  `Qwen3NextForCausalLM`, 4 layers (2 GDN + 2 full),
  hidden_size=8, num_experts=4, `max_position_embeddings=32`.
  ✓ Undertrained → output is gibberish but greedy is deterministic.

Python vLLM is **unusable on this arch** in the current tree —
`vllm/model_executor/layers/layernorm.py:RMSNormGated.forward_cuda`
reads `self.activation` which is never set in `__init__`. The error
fires inside the engine-core child process, so a parent-side
monkey-patch doesn't help. Editing main `vllm/` to fix it would
violate `feedback_edit_in_worktree`. Workaround: generate the golden
via HF transformers' `trust_remote_code` path (the model ships its
own `modeling_qwen3_next.py`). Same architecture, deterministic
greedy decode, same JSON shape as `generate_golden_refs.py` emits.

Golden fixture:
- `crates/vllm-e2e/testdata/golden/qwen3_next_dev.json`: 2 prompts
  (`"hello world"`, `"the quick brown"`), 8 tokens each, top-20
  logprobs per position. Generated by `/tmp/gen_qwen3_next_golden.py`
  (one-shot — not committed; reproducible via the model's
  `modeling_qwen3_next.py`).
- `TestModels::QWEN3_NEXT_DEV` in `vllm-e2e/src/lib.rs`.
- New `run_correctness_test_with_max_len` helper in
  `e_correctness.rs` (the standard helper pins `--max-model-len 2048`,
  which the model rejects against its 32-position embedding).
- `test_cuda_correctness_qwen3_next_dev` test pinned to
  `--max-model-len 24`, threshold=1.

Server boot, model load (via ferrite — confirmed by the load log
`loaded Qwen3NextForCausalLM via ferrite-forward (qwen3_next)`),
forward, and sampling all complete end-to-end. **The test is RED**:
ferrite emits `" World World World World World World World World"`
vs golden `"印刷issance青铜Hello直线距离 OPTION World!\n"`. The
divergence is real GDN/gated-attention math — top-K window doesn't
even contain the golden's first token. Per
`feedback_tdd_red_tests_are_fine`, ship red, fix next, go green.
The structural wiring (Phase 6a) IS correct — the model loads, the
forward path runs, the GDN state pool fires; what's wrong is the
numerics inside the layer-struct `forward` bodies (or weight loading
into them).

## Open follow-ups

- **Phase 6c — go green.** Diagnose why ferrite's GDN + gated
  attention math diverges from the HF transformers reference. Likely
  candidates: (a) `Qwen3NextGdnLayer::forward` — many places where
  layout / dtype assumptions could be off, especially the QKVZ-split,
  conv1d state addressing, and recurrent kernel calls. (b)
  `Qwen3NextGatedAttentionLayer::forward` — the q/gate split is a
  CPU-driven memcpy loop that's easy to get wrong on the per-head
  layout. (c) Weight-loading transposes (HF stores `[out, in]`;
  ferrite expects same — verify each prefix). (d) Gemma "+1"
  RMSNorm offset on `q_norm` / `k_norm`. Start with single-prompt
  + position 0, compare logits tensor element-wise vs HF
  transformers, narrow down to the first divergent op.
- **TP / quant / GGUF.** Out of scope per the plan; will need
  per-format peer Impls and the GDN state pool's TP-aware sharding
  (today the pool is unsharded — `feedback_ferrite_weights_unsharded`
  applies). Document any divergence here when those land.
- **Python vLLM unblock.** Filing or fixing the `RMSNormGated.activation`
  bug upstream would let the standard `generate_golden_refs.py` path
  produce this golden. For now the HF transformers path is the
  reference of record for this fixture.
