# Ferrite-owned TK op headers

Phase 2/3 lands one `.cuh` per op in this directory, per
`FERRITE_TK_PLAN.md`:

- `embed.cuh` (Phase 3f-2e-i: header drafted; 2e-ii: pod smoke
  green; **2e-iii: wired through `emit_cu_variant` via
  `emit_op_block`'s `"Embed"` arm** — pure row-wise gather from the
  embedding table, no math in consumer, single output page. First
  op in every canonical's schedule; lifts the variant to the Qkv
  tier because the `input_ids` kernel arg rides alongside
  `positions` / `slot_mapping` / KV pools.)
- `rms_norm.cuh`
- `gemv_bf16.cuh`
- `gemm_bf16.cuh`
- `fused_add_rms_norm.cuh`
- `fused_qkv_rope_cache.cuh` (Phase 3f-2b-i: header drafted, not yet
  wired through `emit_cu_variant` — BIASED + INTERLEAVED paths gated
  by `static_assert` until follow-up slices implement them; NeoX
  non-biased path is the first-cut target for pod smoke.)
- `rms_qkv_rope_append.cuh`
- `attention_partial.cuh` (Phase 3f-2d-i: header drafted, not yet
  wired through `emit_cu_variant` — `SPLITS != 1`, prefill
  (`NUM_TOKENS > 1`), sliding-window, and softcap paths gated by
  `static_assert` until follow-up slices implement them; decode
  GQA with `SPLITS=1` is the first-cut target for pod smoke.)
- `attention_reduction.cuh` (Phase 3f-2d-ii: stub header — all four
  role functions fire `static_assert(SPLITS > 1)` because in the
  `SPLITS == 1` scope of 2d-i the op is the identity and the
  walker skips emitting its block. The file's template signature,
  page-slot constants, and bar IDs are reserved so 2d-iv's
  codegen dispatch can refer to them symbolically.)
- `o_proj_residual.cuh`
- `silu_upgate.cuh` (Phase 3f-2g: header drafted + standalone pod
  smoke green + **2g-iii: wired through `emit_cu_variant` via
  `emit_op_block`'s `"FusedGateUpSiluMul"` arm**. Decode-only
  (`static_assert(NUM_TOKENS == 1)` on every role) until a
  prefill slice lands. Three pages (x + gate row + up row), one
  consumer-scoped bar.sync [id 13] publishes gate/up partials,
  warp 0 lane 0 fuses `silu(gate) * up` and packs the single bf16
  output. Pod smoke at HIDDEN_DIM=2048, INTERMEDIATE_DIM=8192,
  NCW=4 matches a fp32 CPU reference to bf16 resolution — see
  `../../smoke/ferrite_silu_upgate_smoke.cu`.)
- `down_proj_residual.cuh` (Phase 3f-2h: header + pod smoke +
  `emit_cu_variant` dispatch via `emit_op_block`'s
  `"FusedCublasGemmAdd"` arm, landed in one commit. Decode-only
  (`static_assert(NUM_TOKENS == 1)`). Two pages (x + weight row),
  one consumer bar.sync [id 14] for the warp-tree reduction.
  Storer does scalar RMW on `residual[row]` (read bf16, add fp32
  dot, write bf16). Pod smoke at K=8192, N=2048, NCW=4 matches
  fp32 CPU reference within `TOL=0.02` — see
  `../../smoke/ferrite_down_proj_residual_smoke.cu`.)
- `lm_head.cuh` (Phase 3f-2i: header + pod smoke +
  `emit_cu_variant` dispatch via `emit_op_block`'s
  `"CutlassFusedRmsNormGemm"` arm, landed in one commit. Fused
  final rms_norm + lm_head gemv:
  `out[row] = W_gemm[row, :] · (x * rsqrt(mean(x^2) + eps) *
  norm_weight)` computed on the fly without materializing the
  normed activation to gmem. Three pages (x + norm_weight + one
  gemm weight row). Two-pass consumer: sum_of_squares → rms_scale
  → dot, each via warp shfl_xor + cross-warp scratch + bar.sync.
  **Bar IDs 1/2 reused from rms_norm** — PTX `bar.sync` caps at
  15 and the walker runs ops sequentially per CTA so cross-op id
  aliasing is safe. Decode-only (`static_assert(NUM_TOKENS ==
  1)`). Pod smoke at K=2048, N=8192 (trimmed from 128256 vocab
  to keep run under a second) matches fp32 CPU reference within
  bf16 resolution — see `../../smoke/ferrite_lm_head_smoke.cu`.)

Each header defines four functions in the op's namespace:

```
<op>::consumer(Globals&, SharedState&, int stage, <args>)
<op>::loader  (Globals&, SharedState&, int stage, <args>)
<op>::launcher(Globals&, SharedState&, int stage, <args>)
<op>::storer  (Globals&, SharedState&, int stage, <args>)
```

Phase 1 intentionally ships no op headers — the walker bodies
are empty. This README is a placeholder so the directory exists
in git.
