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
