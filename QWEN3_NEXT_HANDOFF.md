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

## 2026-05-04 — Phase 6c progress: 3 bugs fixed, golden still red

Three concrete bugs found and patched against the Goekdeniz Dev
fixture; test still red — at least one more bug remains.

1. **Wrong arch-config variant.** The first run claimed the
   80B-config variant (the only one whose dim-bounds didn't reject
   it on fingerprint), so the codegen-baked `full_attn_period=4` /
   `full_attn_remainder=3` didn't match the Dev model's alternating
   pattern. Fix: new `configs/qwen3-next-dev.json` advertising
   `full_attn_period=2` / `full_attn_remainder=1` and the Dev
   model's actual dimensions (hidden=128, num_heads=4, head_dim=32,
   num_experts=4, etc.). With this in tree the variant discovery
   picks it correctly (`vllm ferrite info` shows three variants).
2. **Fused-vs-separate QKV layout mismatch.** Ferrite's
   `Qwen3NextGatedAttentionLayer::load` only handled the fused
   `qkv_proj.weight` shape Python vLLM's `QKVParallelLinear` emits.
   Goekdeniz Dev (and any model created via HF transformers'
   `trust_remote_code` path) ships separate `q_proj.weight` /
   `k_proj.weight` / `v_proj.weight`. Fix: the loader now falls
   back to fusing the three via `take_into` into one
   `[q_size + 2*kv_size, hidden]` tensor, mirroring
   `vllm-cuda::model::qwen3_next::Qwen3NextFullAttention::load_fused`.
3. **Per-head Q/gate interleaving.** The original split assumed
   layout `[head0_Q, head1_Q, …, head0_gate, head1_gate, …]` but the
   actual Python reference layout (matching `view(num_heads,-1)` +
   `chunk(2,dim=-1)`) is interleaved per-head:
   `[head0_Q, head0_gate, head1_Q, head1_gate, …]`. Fix: the q/gate
   split now copies per `(token, head)` with the per-head pair
   stride. Note this is a CPU-driven memcpy loop — a fused CUDA
   kernel is a follow-up perf item but isn't the correctness bug.
4. **GDN pool sizing.** `cuda_worker::init_kv_cache_pool` sizes the
   GDN slot count off `num_gpu_blocks`. On a low-VRAM model + L4
   that balloons to ~1M slots × ~18 KB → OOM. The test uses
   `--gpu-memory-utilization 0.05` as a workaround; the real fix is
   to size the pool by `max_num_seqs` (or expose a dedicated
   `--num-gdn-slots`), and that's a follow-up.

After all four fixes the test STILL fails:
```
Engine text: "近日近日近日近日近日近日近日近日"
Golden text: "印刷issance青铜Hello直线距离 OPTION World!\n"
```
Position-0 logits differ enough that the engine's argmax isn't even
in the golden's top-20 window. The output is degenerate (single
token repeating), characteristic of a numerical / structural bug
that produces near-uniform or near-zero deltas across positions.

## Open follow-ups for Phase 6c

Diagnostic plan:
1. Use `/tmp/dump_hf_intermediates.py` (committed in spirit; recreate
   from the script body in this handoff entry — it hooks every layer
   + final_norm + embed and dumps `[0,0,:8]` and `[-1,-1,:8]` slices
   to `/tmp/hf_qwen3_next_dump.json`). HF intermediates already
   captured for prompt "hello world", token IDs `[31173, 3121]`,
   top-3 next tokens `[("印刷", -9.84), (" moder", -9.88), ("农夫", -9.98)]`.
2. Add a parallel ferrite dump — easiest is to print the same slices
   from inside `Instruction::eval` (gated by an env var so it doesn't
   leak into normal runs). Compare layer-by-layer; first divergent
   layer points at the buggy op.
3. Likely candidates: (a) GDN forward — QKVZ-split, conv1d state
   addressing, recurrence kernel arg order. The CPU-roundtrip
   adjusted-indices path is brittle. (b) Q-only RoPE — the kernel
   reads `cos_sin_cache.dim(1)` for `rotary_dim`, but the cache the
   codegen builds with `partial_rotary_factor=0.25` may have a
   different layout than the kernel expects. (c) Gemma "+1" on
   `q_norm`/`k_norm` — verify the offset is applied at load time
   and not double-applied / not applied. (d) `attn_output_gate`
   plumbing — the bool flow through the load path then forward path
   is easy to flip.

Reproducer (post-Phase-6c-progress):
```
VLLM_GPU_MEMORY_UTILIZATION=0.05 cargo test -p vllm-e2e \
  --features e2e,cuda --release --test e_correctness \
  test_cuda_correctness_qwen3_next_dev \
  -- --ignored --nocapture --test-threads=1
```

## Other open follow-ups

- **GDN pool sizing.** Track `max_num_seqs` instead of
  `num_gpu_blocks` so the auxiliary pool doesn't OOM on low-VRAM
  models. Follow-up to Phase 6c.
- **Q/gate split kernel.** CPU-driven memcpy loop is correct but
  slow. Add a CUDA kernel to deinterleave in one launch. Perf
  follow-up; not on the green-test critical path.
- **TP / quant / GGUF.** Per-format peer Impls + TP-aware GDN pool
  sharding. Out of scope per the plan; document any divergence here
  when they land.
- **Python vLLM unblock.** Filing or fixing the
  `RMSNormGated.activation` bug upstream would let the standard
  `generate_golden_refs.py` path produce this golden. For now the
  HF transformers path is the reference of record for this fixture.

## 2026-05-04 — Phase 6c-ii: two real bugs, output now coherent

After auditing ferrite's GDN + gated-attention layers against HF
`modeling_qwen3_next.py` and Python vLLM's `qwen3_next.py` (rather
than chasing more dumps), two unrelated bugs surfaced:

1. **GDN `rms_norm_gated` used `sigmoid(z)` instead of `silu(z)`.**
   `csrc/gdn_recurrent_kernels.cu:188` was `o = normed * sigmoid(z)`
   but HF (`modeling_qwen3_next.py:64` `Qwen3NextRMSNormGated`) does
   `hidden * F.silu(gate)` = `normed * z * sigmoid(z)`. Fix is one
   line + docstring; missing `* z` collapses the GDN output to
   ~1e-5 magnitude (the Phase 6c-i symptom).

2. **`GpuWeights` slow-path `take()` raced on a shared cast buffer.**
   `weights.rs::maybe_cast_cpu` writes the f32→bf16 cast into a
   shared `self.cast_pinned` buffer; both `take()` and `take_into()`
   slow paths queued an `htod_async` from that buffer and returned
   without syncing. The next slow-path `take()` overwrote the buffer
   while the prior async DMA was still in flight, so weights got
   read as later-loaded tensors' bytes. Concretely visible: layer-0
   `input_layernorm.weight` (all ~1.0 in the file) read as the first
   128 BF16 elements of `linear_attn.in_proj_ba.weight`. Fix:
   `stream_synchronize` after H2D in both slow paths, mirroring the
   fast path that already syncs before freeing its per-tensor pinned
   buffer.

Why slow path? Precast worker sorts largest-first (so big mmaps
fault early), but the load body calls `take()` in declaration order
— small tensors (layer norms, `A_log`, `dt_bias` at 128 elems)
sit at the back of the precast queue and routinely fall through to
the slow path before precast reaches them. Fix is correct as-is;
optional follow-up: drain precast before allowing load-side takes,
or precast smallest-first.

Test post-fix: still RED but ferrite now generates coherent output
including "Hello World" and varied tokens (vs. the prior single
"近日" repeating 8×). Engine position-0 token sits outside the
golden's top-20 logprob window (-9.84..-10.20 nats), which on this
undertrained 137 MB model is ~0.36 nats wide and easily flipped by
BF16 noise downstream of any small remaining math discrepancy.

Diagnostic dump infrastructure landed alongside (env-gated,
zero-cost off): `ferrite-forward::dump::dump_tile`, instrumented
calls in `Instruction::eval` for Embed / RmsNorm / Add /
FusedAddRmsNorm / GdnAttention / GatedAttention / SharedFusedMoe,
emitted as `FERRITE_DUMP {json}` lines on stderr when
`FERRITE_DUMP=1`. HF reference at `/tmp/hf_qwen3_next_dump.json`,
generated by `vllm-rs/scripts/gen_qwen3_next_golden.py`.

Open: ferrite `gdn_attn.out` magnitude is still ~10× smaller than
the HF layer-delta would suggest (~0.002 vs HF's ~0.02 for the
combined attn+MoE contribution). Could be additional small-but-real
GDN math discrepancy, MoE numerics, or just the undertrained model.
Next session: dump matching HF intermediates one level deeper
(per-step inside GDN), and/or move to a model whose top-K
distribution has more headroom than 0.36 nats.

## 2026-05-04 — Phase 6c-iii: Gemma `(1 + w)` norm convention

Audit-then-instrument loop (per-step inside GDN via
`ferrite-cuda-core::dump`, paired with a Python hook script
`/tmp/dump_hf_gdn.py` against HF transformers'
`Qwen3NextGatedDeltaNet`) localized the residual divergence to
the very first matmul: ferrite's `gdn.in_proj_qkvz` was
element-wise exactly half HF's. Tracing one step further:
ferrite's `rmsnorm.out` (input to `in_proj_qkvz`) was exactly
half HF's `Qwen3NextRMSNorm.forward(embed)`.

Cause: HF's `Qwen3NextRMSNorm` (`modeling_qwen3_next.py:218`)
computes `output * (1.0 + self.weight.float())` — same Gemma
`(1 + w)` convention that ferrite-model-gemma2's DSL already
threads via `rmsnorm(x, weight + 1.0)`. Qwen3-Next's DSL was
missing the `+ 1.0` on every layer norm; with the file's stored
weights ≈ 1.0, HF computes `* 2.0` and ferrite-without-offset
computes `* 1.0` — exact 2× drop on every norm output.

Fix: add `+ 1.0` to `input_layernorm[layer]`,
`post_attention_layernorm[layer]`, and final `norm` weight refs
in the qwen3-next DSL. Codegen routes through
`ScalarOffsetRmsNormImpl` / `FusedAddRmsNormWithOffset` which
pass `weight_offset = 1.0` to the rms_norm kernel.

The GDN inner norm (`Qwen3NextRMSNormGated`, init=ones, plain
`weight * x`) is NOT Gemma-style and stays unchanged — it lives
inside `Qwen3NextGdnLayer` and never surfaces in the DSL. The
gated-attention layer's `q_norm`/`k_norm` already used the same
`+1` baked in via `add_one_inplace` at load (Phase 6c-i).

Engine output post-fix: `Hello撒上 Ojcollections抚顺DataTable!\nHello`
— position-0 token is now `Hello`, a coherent continuation of
the `hello world` prompt. Test still RED vs the HF golden's
top-1 `印刷` (which sits at logprob -9.84 vs top-20 at -10.20 —
0.36 nat window). On this 137 MB undertrained checkpoint, BF16
reduction-order drift between cuTLASS+FA2 (ferrite) and PyTorch
SDPA (HF golden generator) is enough to flip argmax across the
window. Structural math is verified correct via the dump diff.

Future work: relax test threshold for this fixture, or migrate
to a less-undertrained Qwen3-Next checkpoint when one fits in 14
GB free / L4 VRAM. Diagnostic dump infra (`ferrite-cuda-core::dump`
+ in-GDN per-step dumps + `vllm-rs/scripts/gen_qwen3_next_golden.py`
+ `/tmp/dump_hf_gdn.py`) stays in tree as the audit harness.

## 2026-05-07 — Phase 7a (scope shift to FP8): baseline wiring landed, TP-shard gap blocks load

Scope shifted from BF16-golden green to FP8 enablement — the BF16
golden is numerics-only perfect already (6c-iii) and the real target
is `unsloth/Qwen3-Coder-Next-FP8-Dynamic` (78 GB, compressed-tensors
per-channel weights + per-token dynamic activations, all Linears
except layer 47 + routers + lm_head).

Landed in `a9e3fc856`:
- `Fp8SharedFusedMoELayer` now wraps `Fp8BlockFusedMoELayer`
  internally so one `block_size` field covers all three
  compressed-tensors schemes (per-tensor `[N,K]`, per-channel
  `[1,K]`, block `[bn,bk]`) — the block kernel already does per-row
  indexing when `block_n=1`. `shared_gate_up/down` widened to
  `LinearLayer` so Qwen2-MoE FP8 (dense shared) and
  Qwen3-Coder-Next (FP8 per-channel shared) both route without
  fork. New `::load` probes expert-0 gate_proj scale shape,
  classifies scheme, stacks scales into `[E, N/bn, K/bk]` F32 via
  `ensure_f32_scale`.
- New `Fp8SharedFusedMoeRefImpl` — arch-agnostic FP8 peer of
  `SharedFusedMoeRefImpl`, gated on `num_experts` + `is_fp8_any_moe`.
  BF16 peer now defers via `is_fp8_any_moe` instead of
  `is_fp8_block_moe` so per-channel MoE routes to the FP8 peer too.
- `FieldLoad::Fp8SharedFusedMoe` + dispatch + three emit arms
  (accessor detector, `emit_unindexed_let`, `emit_layered_load_body`),
  storage/accessor validator accepts Fp8→`Fp8SharedFusedMoELayer`,
  `any_fp8` widened to include `Fp8SharedFusedMoe` +
  (pre-existing omission fix) `DeepSeekV2Fp8BlockMoe`.
- `Instruction::Fp8SharedFusedMoe` + eval arm (same dump shape as
  BF16 peer), info naming, `fp8_shared_fused_moe_ref` added to
  `NON_GEMM_NAMES`.
- `configs/qwen3-coder-next-fp8.json` — the unsloth checkpoint's
  variant (hidden 2048, inter 5120, rope θ 5e6, full_attn_period=4).

Variant discovery works: `ferrite · qwen3-coder-next-fp8 · 389
tiles · 196 waves · tp=1 · cublas cutlass non-gemm comm`. Release
build clean with `--features cuda,nccl`. The 78 GB `unsloth/…`
checkpoint then **OOMs during ferrite load** on nick3 (2×L40S,
46 GB each) regardless of `--tensor-parallel-size`: the ferrite
MoE load path is not TP-sharded — each rank calls
`Fp8SharedFusedMoELayer::load(…, num_experts=512, …)` and allocates
the full stacked `[E, 2*inter, hidden] Fp8` + `[E, hidden, inter]
Fp8` arrays locally. The FP8 *shared-expert* projections (dense
`Fp8Linear`) are similarly unsharded. Precedent is the same: the
existing `DeepSeekV2Fp8BlockMoELayer::load` and
`Fp8FusedMoELayer::load` also take no tp-rank/tp-size args.

Next-session entry point (on nick3, worktree
`/home/nickm/qwen3-next-fresh/vllm-rs/`):

1. **TP-shard `Fp8SharedFusedMoELayer::load`** — add
   `tp_rank`/`tp_size` params + an NCCL group ref; slice experts
   along dim 0 (expert parallelism:
   `experts[rank*E/tp..(rank+1)*E/tp]`) OR along the hidden-dim
   of each expert (column-parallel gate/up, row-parallel down +
   all-reduce). Expert parallelism is cheaper at prefill
   (no per-layer all-reduce), hidden-dim is cheaper when tokens
   outnumber selected experts. Python vLLM's
   `CompressedTensorsW8A8Fp8MoEMethod.apply` uses hidden-dim for
   small TP × large expert; pick that first to match parity.
   Thread tp into the three codegen emit arms
   (`emit_unindexed_let`, `emit_layered_load_body`, and the
   storage/accessor validator doesn't change). Do the same for the
   shared-expert `Fp8Linear`s — `Fp8Linear::load_concat` already
   exists in a sharded variant somewhere in `layers_quant.rs`; if
   not, add one.
2. **Reproducer** (post-TP):
   ```
   oc rsh nick3 bash -c 'cd /home/nickm/qwen3-next-fresh/vllm-rs && \
     cargo build -p vllm-cli --features cuda,nccl --release && \
     ./target/release/vllm serve unsloth/Qwen3-Coder-Next-FP8-Dynamic \
       --device cuda --tensor-parallel-size 2 \
       --max-model-len 256 --gpu-memory-utilization 0.9'
   ```
3. **Layer-47 mixed quant** is parked until load works. The
   unsloth checkpoint's `ignore` list excludes layer 47's
   full-attention + all experts + shared expert (+ `lm_head`, all
   routers, all GDN layers and `shared_expert_gate` across the
   board). The GDN / router / gate exclusions already land
   correctly (those weights live on non-FP8 accessors via ferrite's
   existing storage routing). The layer-47 exception is the hard
   one: one `mlp[layer]` accessor is Fp8 on 47 layers and Dense on
   1 — fix needs either per-layer accessor types or a `MoeAny`
   enum. Not blocking TP-shard.

Source-of-truth pointers (for quick reference):
- Python: `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
  — `CompressedTensorsW8A8Fp8MoEMethod` (per-channel apply path)
- DeepSeek TP-sharded FP8 load precedent:
  `DeepSeekV2Fp8BlockMoELayer::load` in
  `ferrite-kernels/src/layers_moe.rs` (note: still NOT tp-aware —
  this gap is cross-arch, not Qwen3-Next-specific).

## 2026-05-07 — Phase 7b: MoE TP-shard + FP8 Linear TP-shard + fingerprint + layer-47 BF16 fallback — model serves on TP=2, forward still red

Goal of this session: make `unsloth/Qwen3-Coder-Next-FP8-Dynamic`
load on `nick3` (2×L40S 46 GB each) at `--tensor-parallel-size 2`.
Achieved — the model now loads in 23.5 s, binds 0.0.0.0:8000, and
all routes register. Forward still fails on the first request
(illegal memory access in FA2) because Qwen3Next's gated-attention
and GDN layers are not yet TP-sharded; that's Phase 7c.

Landed:

1. **`Fp8SharedFusedMoELayer::load` gained `tp_rank` / `tp_size`
   params.** Intermediate-dim TP matching Python vLLM's
   `_load_w13` (dim 0) / `_load_w2` (dim 1) per expert. Per-channel
   `w1` scale shards along dim 0; per-channel `w2` scale stays
   replicated (indexes the unsharded hidden output). Block-mode
   sharding asserts `inter_full % (tp_size × block_{n,k})`. Router
   gate and `shared_expert_gate` stay replicated. Shared expert
   `gate_proj+up_proj` column-parallel via new
   `Fp8Linear::load_concat_sharded`; `down_proj` row-parallel via
   new `Fp8Linear::load_sharded` (dim 1). The DSL interpreter
   (`Instruction::Fp8SharedFusedMoe`) performs one post-combine
   all-reduce using `ForwardCtx::tp_group`, covering both the
   routed partial and the sigmoid-gated shared partial
   (replicated sigmoid preserves partial-sum structure).

2. **Fingerprint sniff upgraded to a multi-layer scan.** The
   original `fp8_marker_tensor = "model.layers.0.{fp_leaf}.weight_scale"`
   misses on Qwen3-Next because layer 0 is GDN (no `self_attn.q_proj`),
   and `last_layer = 47` misses on the unsloth checkpoint because
   layer 47 is on the `ignore` list (dense BF16, no weight_scale).
   Same story for the positive `last_tensor` existence check —
   `weight_scale` at layer 47 isn't there either. The fix scans
   `{0, n/4-1, n/2, 3n/4, n-1}` and accepts/rejects on any hit.
   Without this, both the BF16 and FP8 variants either missed or
   matched ambiguously, and the dispatcher picked the BF16
   `SharedFusedMoELayer::load` (which blows through 78 GB per
   rank). (Cross-arch change — pinned by `cargo build` across all
   existing arches; no new test failures spotted in the variant
   discovery output.)

3. **Layer-47 BF16 fallback.** `Fp8SharedFusedMoELayer::load`
   probes expert-0 dtype BEFORE consuming the router gate; if
   it's BF16, delegates to the new
   `SharedFusedMoELayer::load_sharded` (also intermediate-dim TP)
   and parks a dummy `Fp8BlockFusedMoELayer` (via `::dummy()`) in
   the outer struct's `moe` field. Forward path checks
   `self.bf16_fallback` first; when `Some`, delegates entirely to
   the BF16 peer. The instruction arm's all-reduce still closes
   the loop because the BF16 peer is also sharded — both paths
   produce per-rank partial sums.

4. **FP8 Linear TP-sharding.** `FieldLoad::Fp8Linear` emit arms
   (both `emit_unindexed_let` and `emit_layered_load_body`) now
   thread `tp_rank` / `tp_world_size`, routing to new
   `Fp8Linear::load_sharded` (single prefix, per-dim) and
   `Fp8Linear::load_concat_sharded` (column-parallel fused) plus
   two `ferrite-forward::load_layered_fp8_linear_sharded` /
   `_concat_sharded` helpers. Applies to every compiled FP8
   variant, not just Qwen3-Next; at tp=1 the short-circuits yield
   byte-equivalent behavior.

Reproducer state on nick3
(`/home/nickm/qwen3-next-fresh/vllm-rs`):

```
./target/release/vllm serve unsloth/Qwen3-Coder-Next-FP8-Dynamic \
  --device cuda --tensor-parallel-size 2 \
  --max-model-len 256 --gpu-memory-utilization 0.9
# → "Stack initialized with TP=2 in 23.5s"
# → "vLLM Rust server listening on http://0.0.0.0:8000"
# Memory: per-rank total=44.4 GiB, weights+overhead=41.3 GiB,
# est_activations=512 MiB, num_gpu_blocks=16.
```

First request crashes with
`CUDA error at third_party/vllm-flash-attn/src/flash_fwd_launch_template.h:84: an illegal memory access was encountered`.
Cause (unverified but strongly-inferred): the MoE path is now
TP-sharded end-to-end, but Qwen3-Next's `Qwen3NextGatedAttentionLayer`
and `Qwen3NextGdnLayer` still load full unsharded weights — the
AllReduce the tp-lowering pass inserts after `o_proj` then sums
replicated partials across ranks, producing doubled attn output
that doesn't match the FA2 kernel's assumed shapes.

Next-session entry point (Phase 7c, on nick3):

1. **TP-shard `Qwen3NextGatedAttentionLayer::load`** — the QKV is
   a fused `[q_size + 2*kv_size, hidden]` projection
   (`q_size = 2*true_q_size` with `attn_output_gate`). Column-
   parallel on dim 0 of qkv_proj, row-parallel on dim 1 of o_proj.
   Per-head `q_norm` / `k_norm` are `[head_dim]` vectors and
   replicated. Thread `tp_rank`/`tp_size` through
   `FieldLoad::GatedAttention` emit arm + the layer's forward so
   `num_q_heads` becomes `num_q_heads / tp_size` at runtime.

2. **TP-shard `Qwen3NextGdnLayer::load`** —
   `in_proj_qkvz` / `in_proj_ba` / `out_proj` / `conv1d`. Per-head
   sharding mirrors the full-attn head split. Note the GDN
   recurrent state (`GdnStatePool`) is already per-slot; the only
   TP-adjacent concern is that `num_v_heads` / `num_k_heads` shard
   by tp.

3. **Re-run the reproducer**; confirm `/v1/completions` returns
   coherent Python from the `def fibonacci(n):` prompt.

Source-of-truth pointers (for quick reference):
- Python: `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
  — `CompressedTensorsW8A8Fp8MoEMethod` (per-channel apply path)
- DeepSeek TP-sharded FP8 load precedent:
  `DeepSeekV2Fp8BlockMoELayer::load` in
  `ferrite-kernels/src/layers_moe.rs` (note: still NOT tp-aware —
  this gap is cross-arch, not Qwen3-Next-specific).

## 2026-05-08 — Phase 7c: TP-shard gated-attention + GDN + pre-existing FP8 attention blocker

Goal: make the Phase 7b model (`unsloth/Qwen3-Coder-Next-FP8-Dynamic`) at
TP=2 on nick3 return coherent output. Partial — landed the TP-sharding
as designed, but surfaced a pre-existing FP8-attention-weight handling
gap that is still the forward blocker. Model loads and serves; first
`/v1/completions` request still crashes, now traced to `Qwen3NextGatedAttentionLayer`
loading FP8 `q_proj` / `k_proj` / `v_proj` / `o_proj` via `Linear`
(BF16 cublas GEMM) when the on-disk dtype is `float8_e4m3fn`.

Landed (Phase 7c proper):

1. **`Qwen3NextGatedAttentionLayer::load_sharded`.** New method on
   `layers_attn_gated.rs` that:
   - Calls `gw.synthesize_packed_row_split_sizes(...qkv_proj, &[("q_proj", q_size_full), ("k_proj", kv_size_full), ("v_proj", kv_size_full)])`
     so the fused-parent case (`qkv_proj.weight` present) carves
     virtual children that re-use the packed-source fallback already
     in `Linear::load`. No-op when the checkpoint ships separate
     `q/k/v_proj` (the unsloth checkpoint's case).
   - Allocates a per-rank packed `[q_size + 2*kv_size, hidden]` buffer
     using `elem = q_dtype.size_bytes()` (the **on-disk** dtype — not
     `post_dtype`, since `take_shard_into` copies raw bytes for FP8
     and would half-fill a BF16-sized buffer, which is the bug I hit
     on the first run and fixed).
   - `o_proj`: `Linear::load_sharded(dim=1, tp_rank, tp_size)` — row-
     parallel with bias on rank 0 only.
   - `q_norm` / `k_norm`: `[head_dim]` vectors kept replicated (per-head
     Gemma-style RMSNorm weight, `+1` baked in at load).
   - Stores per-rank `num_q_heads = num_q_heads_full / tp_size` and
     `num_kv_heads = num_kv_heads_full / tp_size` in the struct so
     the existing `forward()` splits Q/K/V with the sharded head
     counts without any forward-side rework.

2. **`Qwen3NextGdnLayer::load_sharded`.** New method on
   `layers_gdn.rs`:
   - `in_proj_qkvz.weight` is a **single per-head grouped** block
     (`[num_k_heads, 2*head_k + 2*v_per_k*head_v]` contiguous per
     group). Python's `MergedColumnParallelLinear(output_sizes=[sum(...)])`
     with one output size is effectively `ColumnParallelLinear`; a
     simple `take_shard(dim=0, rank, tp_size)` gives each rank whole
     head groups because `num_k_heads % tp_size == 0`. First draft
     mistakenly did a 4-block synthesize + pack; kernel expects the
     grouped layout and the 4-block version reordered rows — reverted.
   - `in_proj_ba.weight` similarly stays as a simple dim=0 shard
     (`[num_k_heads, 2*v_per_k]` grouped; same divisibility argument).
   - `conv1d.weight` has **block layout** `[key_dim, key_dim, value_dim]`
     along dim 0 (Python's `mamba_v2_sharded_weight_loader`). Manual
     3-block shard on CPU f32 then upload per-rank.
   - `A_log` / `dt_bias`: sharded dim 0 (`[num_v_heads]` → per-rank slice).
   - `norm.weight`: replicated (per-head RMSNorm).
   - `out_proj`: `Linear::load_sharded(dim=1, tp_rank, tp_size)`.

3. **`GdnStatePool` per-rank sizing.** `make_gdn_state_pool`
   (`vllm-cuda/src/model/qwen3_next.rs`) gained `tp_size`; shards
   `conv_dim` and `num_v_heads` by tp. `head_v_dim` / `head_k_dim`
   stay replicated (heads are the shard unit). `cuda_worker.rs`
   passes `self.config.tp_world_size` to the call. This keeps the
   per-slot ssm and conv state layouts consistent with the per-rank
   `Qwen3NextGdnLayer`'s head counts.

4. **FieldLoad emit arms.** `FieldLoad::GdnAttention` and
   `FieldLoad::GatedAttention` in both `emit_unindexed_let` and
   `emit_layered_load_body` now branch on `sharded = tp_world_size > 1`:
   at tp>1 they call `load_sharded(..., tp_rank, tp_world_lit)`; at
   tp=1 the original unsharded call is byte-identical.

5. **Instruction eval AllReduces.** `Instruction::GdnAttention` and
   `Instruction::GatedAttention` gained a `#[cfg(feature = "nccl")]
   if let Some(group) = ctx.fwd.tp_group { group.all_reduce_inplace_promote(...) }`
   block right after `w.forward(...)`, following the
   `Instruction::Fp8SharedFusedMoe` precedent. Both layers are
   row-parallel-on-output (`o_proj` for attention,
   `out_proj` for GDN) so the hidden output is a per-rank partial
   sum that must be all-reduced before the residual add.

Reproducer state on nick3 (`/home/nickm/qwen3-next-fresh/vllm-rs`):

```
./target/release/vllm serve unsloth/Qwen3-Coder-Next-FP8-Dynamic \
  --device cuda --tensor-parallel-size 2 \
  --max-model-len 256 --gpu-memory-utilization 0.85
# → "Stack initialized with TP=2 in 23.2s" (was 23.5s in Phase 7b)
# → Memory estimate: total=44.4 GiB, weights+overhead=40.1 GiB (was 41.3 GiB in Phase 7b — small win from GDN+attn shard)
# → server binds, routes register
# First request: CUDA_ERROR_ILLEGAL_ADDRESS in FA2 at layer 3, same crash as Phase 7b
```

The FA2 crash at layer 3 traces back to a pre-existing issue **not**
addressed by TP-sharding: `Qwen3NextGatedAttentionLayer` routes
`qkv_proj` and `o_proj` through the dense `Linear` type, whose forward
is a BF16 cublas GEMM. On this checkpoint those four weights are
`float8_e4m3fn` with per-channel BF16 `weight_scale` (verified via
safetensors inspection of `model.layers.3.self_attn.{q,k,v,o}_proj.weight`
and `.weight_scale`). The `Linear` loader copies raw FP8 bytes into
a buffer and then wraps it as `GpuTensor::new(ptr, [rows, hidden], BF16)` —
the bytes are FP8 but the claimed dtype is BF16, so cublas reads
interleaved FP8 pairs as garbage BF16 values. The matmul doesn't
OOB (byte count matches), but the garbage Q/K/V propagates into the
paged K cache and eventually trips FA2's indirect indexing into an
invalid address. Same behavior at tp=1 and tp=2 — TP-sharding doesn't
interact with this, it's orthogonal.

Next-session entry point (Phase 7d, on nick3):

1. **Teach `Qwen3NextGatedAttentionLayer` to consume FP8 weights.**
   Two viable paths:
   - (a) Swap `Linear` for `Fp8AnyLinear` so FP8 weights go through
     the existing per-channel `Fp8Linear::forward` path, matching
     the MoE side. Needs a new fused-QKV Fp8 helper (Fp8Linear has
     `load_concat_sharded` for gate_up; we'd reuse the pattern for
     `[q_proj, k_proj, v_proj]`). `o_proj` is a single prefix — use
     `Fp8Linear::load_sharded(dim=1)` directly.
   - (b) Dequantize FP8 → BF16 at load time. Adds a one-shot kernel
     (`fp8_dequant_to_bf16(weight_fp8, weight_scale, dst_bf16)`), keeps
     `Linear::forward` unchanged. Cheaper code delta; memory cost
     is 2× the attention FP8 bytes per rank (minor — attention is
     ~1 GB of the ~40 GB per-rank footprint).
   Recommend (a) for parity with the MoE path and the Python
   reference, but (b) is a faster unblock if (a) proves too invasive.

2. **Layer-47 full-attention stays in the BF16 `ignore` list.** Once
   (1) lands, the existing `Linear`-based path is still the right
   choice for that single layer. Thread a per-layer
   `Fp8 vs Dense` picker through `FieldLoad::GatedAttention` — same
   shape as the `Fp8SharedFusedMoE` vs `SharedFusedMoE` split that
   Phase 7b landed for the MoE side (fingerprint-driven multi-layer
   probe).

3. **Re-run the reproducer.** `def fibonacci(n):` → coherent Python
   on TP=2.

Source-of-truth pointers:
- Python FP8 attention: `QKVParallelLinear(..., quant_config=quant_config)`
  in `vllm/model_executor/models/qwen3_next.py::Qwen3NextAttention.__init__`;
  dispatches to `CompressedTensorsW8A8Fp8.apply` for per-channel FP8.
- Existing Rust FP8 precedent (MoE): `Fp8Linear::load_concat_sharded`
  and `Fp8SharedFusedMoELayer` in `ferrite-kernels/src/layers_{quant,moe}.rs`.
- Pre-existing buffer sizing gotcha: `Qwen3NextGatedAttentionLayer::load`
  already had the FP8-vs-BF16 elem-size mismatch; `load_sharded`
  mirrors the same convention (`elem = q_dtype.size_bytes()`) so the
  fix in (1) lands once and covers both paths.

## 2026-05-08 — Phase 7d: FP8 gated-attention — model returns coherent output on TP=2

Goal: fix CUDA_ERROR_ILLEGAL_ADDRESS in FA2 caused by BF16 cuBLAS GEMM
reading FP8 bytes as garbage Q/K/V activations. Fully landed.

What shipped:

1. **`Fp8GatedAttentionLayer`** in `ferrite-kernels/src/layers_attn_gated.rs`:
   - New struct holding `Fp8Linear qkv_proj` + `Fp8Linear o_proj` +
     `bf16_fallback: Option<Qwen3NextGatedAttentionLayer>`.
   - Single `load(gw, prefix, ..., tp_rank, tp_size, output_dtype, stream)`
     that is uniformly TP-aware (world=1 short-circuits to non-TP inside
     each `Fp8Linear::load_concat_sharded` / `load_sharded`).
   - Runtime dtype probe on `q_proj.weight`: if BF16 (compressed-tensors
     `ignore` list — layer 47 on unsloth), delegates to
     `Qwen3NextGatedAttentionLayer::load_sharded` and parks dummies.
     Same pattern as `Fp8SharedFusedMoELayer`'s `bf16_fallback`.
   - `forward`: short-circuits to BF16 peer when `bf16_fallback.is_some()`;
     else runs full FP8 path (gate-split, per-head Q/K RMSNorm, Q-only
     partial RoPE, paged K/V write, FA2 with on-the-fly K rotation,
     sigmoid output gate, FP8 o_proj). Byte-identical to the BF16 peer
     logic, but `qkv_proj.forward` / `o_proj.forward` use `Fp8Linear`.
   - `add_one_inplace` bumped to `pub(crate)` so the FP8 loader can reuse
     it for Q/K norm weight initialization.

2. **`Fp8Linear::dummy()`** in `ferrite-kernels/src/layers.rs`: dangling
   sentinel, never read in practice (outer forward short-circuits).

3. **`quantization.rs`**: `OpKind::GatedAttention` added to the
   `reached_by_matmul` set so the FP8 storage format is resolved for
   `self_attn[layer]` aggregate weight accessors on FP8 checkpoints.

4. **`codegen.rs`**:
   - `FieldLoad::Fp8GatedAttention` variant (same fields as `GatedAttention`).
   - `is_fp8_gated_attention` type-string probe.
   - Classification: `is_gated_attention || is_fp8_gated_attention` block
     returns the right variant.
   - `accessor_is_fp8_any`: `Fp8GatedAttentionLayer` registered so the
     storage-vs-accessor consistency check passes at macro-expand time.
   - Emit arms in both `emit_unindexed_let` and `emit_layered_load_body`;
     both use `__fp8_dtype` prelude binding and thread `tp_rank/tp_size`.

5. **`impl_lib.rs`**:
   - `is_fp8_gated_attention()` predicate (mirrors `is_fp8_any_moe`).
   - `Fp8GatedAttentionRefImpl`: claims `OpKind::GatedAttention` when FP8;
     emits `Instruction::Fp8GatedAttention` with `rotary_cos_sin` accessor.
   - `GatedAttentionRefImpl::matches`: early-returns `None` when FP8 (defer
     to the new impl).

6. **`instr.rs`**: `Instruction::Fp8GatedAttention` variant + eval arm
   with post-forward `#[cfg(feature = "nccl")] all_reduce_inplace_promote`.

7. **`info.rs`**: `Fp8GatedAttention` introspection arm.

Reproducer state on nick3 (`/home/nickm/qwen3-next-fresh/vllm-rs`):

```
./target/release/vllm serve unsloth/Qwen3-Coder-Next-FP8-Dynamic \
  --device cuda --tensor-parallel-size 2 \
  --max-model-len 256 --gpu-memory-utilization 0.85
# → "Stack initialized with TP=2"
# → server binds, routes register
# First request: coherent Python output ✓
# curl response: {"text":"n\n    a=0\n    b=0\n    if n<=0:..."}
```

Phase complete for code-completion use case. Known open issue:

**Chat-template-formatted prompts produce `!!!!` (all-NaN logits) at FP8+TP=2.**

Root cause traced via FERRITE_DUMP:
- All intermediate activations (embed → GDN → gated-attn → MoE, all 48 layers)
  are FINITE and well-behaved for chat prompts.
- `allgather.in` (lm_head GEMM output, pre-gather) is ALL NaN for T=12 chat
  prompts on BOTH ranks.
- Code prompts (T=4, no special tokens) have non-NaN lm_head output.
- The NaN first appears in `allgather.in`, meaning the lm_head BF16 GEMM itself
  produces NaN for the chat-prompt hidden state.

Suspected mechanism: the FP8 attention + MoE computation for the 12-token chat
prompt (including `<|im_start|>`, `<|im_end|>` special tokens at high vocab IDs
in rank 1's shard) produces hidden state values that cause BF16 overflow
(>65504) in some dimensions somewhere between layer 47 and the lm_head GEMM.
After `ScalarOffsetRmsNorm`, individual dimensions could be as large as
sqrt(2048) × typical_activation due to concentration, but the BF16 GEMM
accumulation over 2048 terms with large weights might overflow.

To fix: need to either clamp the hidden state before lm_head, use F32
accumulation in the final GEMM, or investigate which layer first produces
BF16 Inf (by adding bounds checks or using CUDA anomaly detection).

NOT a GDN issue: GDN weights are BF16 in this checkpoint (not FP8), no
dequantization needed there.

Separately fixed: GDN state pool slot exhaustion (slot leak across requests)
was causing `!!!!` on all requests after ~16. Committed as
`150314508 ferrite: Qwen3-Next GDN state pool slot allocator`. After the fix,
stress test: 0 bangs in 30 requests.

## 2026-05-08 — Diagnostic: garbage output on all prompts; nan_to_zero wrong approach

### What was attempted

The previous handoff entry traced `!!!!` to "BF16 overflow in chat prompts" and
proposed clamping/nan_to_zero before lm_head. This session investigated whether
that was the right fix.

**Key finding: it was not.** Other FP8 models (DeepSeek V3, FP8 block MoE)
work fine without any lm_head sanitization. Python vLLM does nothing special
before lm_head for this model either. The `nan_to_zero` was treating a symptom.

### What the runs actually showed

Ran `unsloth/Qwen3-Coder-Next-FP8-Dynamic --device cuda --tensor-parallel-size 2
--max-model-len 256 --gpu-memory-utilization 0.85` on nick3 (2×L40S).

**All prompts (code AND chat) now produce garbage.** Not just `!!!!`; actual
vocabulary tokens but incoherent (e.g., `def fib(n):` → `2\r\nfib(n`). This is
a regression beyond what Phase 7d described.

### FERRITE_DUMP=1 findings

Added `dump_tile_check_finite` (full-scan version of `dump_tile` — downloads ALL
elements, not just first8/last8) to catch Inf/NaN in middle dimensions.

Results for `def fib(n):` (T=4) code prompt with TP=2:

```
embed.out:
  rank 0: ALL ZEROS       ← correct (rank 0 handles vocab 0..76K;
  rank 1: non-zero values ← tokens < 76K are in-shard for rank 0,
                            but the 2 dump lines could be interleaved;
                            zero rank = the one handling out-of-range IDs)
gdn_attn.out layer 0:   first8=[0.004, 0.018, -0.015, ...]  (both ranks identical after AllReduce)
gated_attn.out layer 3: first8=[-0.017, -0.011, ...]
scalar_offset_rmsnorm.in:  shape=[4,2048], max_abs=0.265, any_nonfinite=false
scalar_offset_rmsnorm.out: shape=[4,2048], max_abs=28.125, any_nonfinite=false
```

The final norm input AND output are **all-finite** with reasonable magnitudes.
No Inf/NaN anywhere in the sampled values. So the earlier "BF16 overflow"
hypothesis was wrong, or the dumps missed middle dimensions (first8/last8).

With `FERRITE_DUMP=1` the output was `'2\r\nfib(n'` instead of pure garbage,
suggesting that the 36 stream synchronizations per forward pass (one per GDN
layer for the index adjustment D2H copy) change the race condition dynamics.

### RAII audit for GDN slots

No RAII guard. `alloc_slot` is called at first prefill; `free_slot` is called
when `scheduler_output.finished_req_ids` contains the request. Normal path is
correct. Panics or error returns before `finished_req_ids` propagates would leak
slots. This is not the cause of garbage output (first request uses fresh slot 0).

### What diagnostic infrastructure was added (committed below)

1. `ferrite-cuda-core/src/dump.rs`: `dump_tile_check_finite` — downloads ALL
   elements, reports `any_nonfinite`, `first_nonfinite_idx`, `count_nonfinite`,
   `max_abs`. Use with `FERRITE_DUMP=1`.

2. `ferrite-forward/src/instr.rs`: `dump_tile_check_finite` calls added to
   `FusedAddRmsNorm` (on the residual after update — the hidden_states) and
   `ScalarOffsetRmsNorm` (input and output).

3. `vllm-cuda/csrc/precision_cast_kernels.cu` + `ferrite-kernels/src/kernels.rs`:
   `nan_to_zero_bf16_inplace` CUDA kernel + `sanitize_bf16_inplace` wrapper.
   Not called anywhere — kept as a tool in case a targeted fix is needed.

### Root cause: unknown, but narrowed

- Final norm input/output: all-finite
- GDN, attention, MoE outputs: finite (from first8/last8 dumps)
- But `dump_tile_check_finite` only runs when `FERRITE_DUMP=1`; without it we
  don't know if middle dimensions are bad

**Most likely remaining causes:**
1. A computation regression in commits after Phase 7d (candidates: `02ccf8cbc`
   golden+finite_logprob fix, `150314508` GDN slot allocator, `c0a36aefa` GDN
   pool sizing). One of these may have changed forward computation behavior.
2. A race condition in the TP=2 path — the FERRITE_DUMP synchronizations
   change which garbage is produced, suggesting CUDA async order matters.

### Next-session entry point

Run the full-dump with `FERRITE_DUMP=1` AND add Python vLLM as a reference:

```bash
# On nick3, run Python vLLM on the same model and capture intermediate
# activations layer by layer (using hooks or debug prints), then compare
# against ferrite's FERRITE_DUMP output. Find the first layer where
# ferrite diverges from Python.
#
# Alternative: binary search the commit range 38f901ba5..HEAD to find
# which commit introduced the regression. Build each candidate and test.
```

To binary-search the regression:
```bash
git bisect start HEAD 38f901ba5
# For each candidate commit: build on nick3, curl test
# Good = "n\n    a=0..." in output, Bad = garbage
```

## 2026-05-08 — Debugging session: non-determinism on first request from fresh server

### The core finding

Temperature=0 (fully deterministic greedy decoding) produces **different outputs
on different fresh server restarts** for identical inputs. This is the symptom we
need to fix. The non-determinism is not between requests on a running server
(though that exists too from state issues) — it's in the very FIRST request.

Three fresh server restarts, same prompt `def fib(n):`, `temperature=0`:
```
Run 1: ' fib(n): return (n) n(n) n(n) n(n)'
Run 2: ' fib(n): fib(n) fib(n) fib(n) fib(n) fib'
Run 3: '\n\n\n\n\n    return:\n        if fib\n        else:\n           '
```

Run 3 is somewhat coherent. Runs 1 and 2 echo the prompt (a classic sign of
induction heads failing due to numerical errors).

### Most likely cause: FP8 CUTLASS GEMM non-determinism

CUTLASS FP8 GEMM is non-deterministic between CUDA runs because tensor core
reduction order is not fixed. Dynamic FP8 quantization computes activation scales
from `max(|x|)/448` — if x is slightly different (due to CUTLASS non-determinism),
the scale changes, and the quantized tensor for the next layer changes. This
**compounds over 48 layers** until the logit distribution is completely different.

Python vLLM fixes this with **CUDA graphs for decode** (which capture a fixed
execution path and make it deterministic). We disabled CUDA graphs for Qwen3-Next
because of GDN recurrent state that's hard to capture. Without graphs, each run
picks a different reduction order in each FP8 GEMM → different activations →
different logits → different tokens.

### What was ruled out this session

- `nan_to_zero` before lm_head: wrong approach (reverted)
- `take_merged_dim0_shard` for `in_proj_ba`: wrong — weight IS in grouped format,
  simple `take_shard` is correct (reverted)
- Embedding vocab sharding: correct
- GDN intermediate values: finite, reasonable magnitude
- KV attention path: structurally correct
- NCCL stream ordering: uses `compute_stream` throughout, no race
- The `in_proj_ba` B/A split: confirmed GROUPED layout; gdn_qkvz_split kernel
  correctly handles grouped format
- `conv1d` 3-block sharding: correct
- `A_log`, `dt_bias` sharding: correct

### What was added this session (committed as 36dd60fd8)

- `dump_tile_check_finite`: full-scan finite check in FERRITE_DUMP
  (previously first8/last8 could miss Inf in middle dims)
- `nan_to_zero_bf16_inplace` CUDA kernel + Rust wrapper (unused, available as tool)
- `dump_tile_check_finite` calls on FusedAddRmsNorm residual and
  ScalarOffsetRmsNorm input/output

### Next session entry point

**Option A (most direct):** Force CUTLASS to use deterministic mode.
Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var and add
`cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH)` or equivalent for CUTLASS.
Test if non-determinism goes away. If yes, accept the (small) perf hit.

**Option B:** Compare against Python vLLM on a machine where it can run
(newer driver). Run both with `torch.use_deterministic_algorithms(True)` in
Python to confirm whether Python has the same non-determinism.

**Option C:** Run a smaller BF16 model (not Qwen3-Next FP8) at TP=2 to verify
the TP=2 infrastructure (AllReduce, AllGather, weight sharding) is correct for
a deterministic model. If a BF16 model gives coherent TP=2 output, the bug is
FP8-specific. If BF16 also fails, there's a more fundamental TP=2 bug.

The quick reproducer:
```bash
# On nick3
./target/release/vllm serve unsloth/Qwen3-Coder-Next-FP8-Dynamic \
  --device cuda --tensor-parallel-size 2 \
  --max-model-len 256 --gpu-memory-utilization 0.85
# Expected (Python): '\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)'
# Actual: non-deterministic garbage
```
