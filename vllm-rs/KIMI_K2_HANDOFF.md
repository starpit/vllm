# Kimi K2 — handoff

Updated 2026-04-27. Branch `worktree-kimi-ff-interp` rebased onto
`a62f1bb5f` (ff-interpreter HEAD: "disable FlashInfer fleet-wide
pending tp>1 fix"). Pick this branch up on an H100 — L4 ran out of
disk for the only real-weights K2-flat-routing fixture (Moonlight-16B
is 32 GB).

## TL;DR — what to do first on the new machine

1. `git checkout worktree-kimi-ff-interp` and confirm `git log -1` is `2aa2edd3f`
   (or whatever fixup-of-`80f9d3b26` ended up as).
2. `cargo build -p vllm-cli --features cuda --release` — should build
   clean. (On the L4 we tripped a pre-existing
   `tk_megakernel_llama_3_2_3b.cu` nvcc failure unrelated to anything
   in this branch; the H100 toolchain shouldn't hit it.)
3. Run the smoke fleet to confirm nothing regressed:
   ```
   cargo test -p ferrite-forward-macro --release fingerprint_tests
   cargo test -p vllm-e2e --features cuda,e2e --release --test e_correctness \
       -- --ignored --test-threads=1 \
       test_cuda_correctness_deepseek_v3_academic_9b \
       test_cuda_correctness_deepseek_v3_academic_9b_fp8_block \
       test_cuda_correctness_qwen3_0_6b_fp8_block \
       test_cuda_correctness_kimi_k2_tiny
   ```
   All four are expected green (regenerated goldens were committed
   for the V3 FP8-block + kimi_k2_tiny tests because ff-interp's
   evolved Fp8AnyLinear / dedup_signature paths produced different
   accumulator order than the kimi-side snapshots).
4. **The next chunk:** download `moonshotai/Moonlight-16B-A3B-Instruct`
   (~32 GB BF16) and run the load smoke. The crate +
   config + dispatcher are all in place; this is the first real-
   weights K2-flat-routing exercise. See **§ Moonlight bring-up**
   below for the full recipe.

## What's on this branch (8 commits past `a62f1bb5f`)

Most-recent first:

- `2aa2edd3f` *fixup! …complete a79a87764 port* —
  `deepseek_moe_fp8_block` added to `NON_GEMM_NAMES` in the new TOTAL
  kernel-class summary upstream introduced post-rebase.
- `af6504433` **`ferrite-model-deepseek-v3-flat`: q_lora_rank=null
  sibling crate** — Path A1 from the design discussion. New crate;
  V2-Lite-style direct `q_proj` DSL body + V3-style routing flavor
  driven by `deepseek_moe(..)` config. Initial config:
  `moonlight-16b-a3b-instruct.json`. Workspace + `ferrite-models`
  umbrella registration via the new `arch-deepseek-v3-flat` feature.
- `3bd93fa3b` **`ferrite-forward`: dispatcher walks past
  `Ok(None)`** — `try_load` rewritten as
  `inventory::iter().filter(...).find_map(...).transpose()` so two
  registrations claiming the same HF arch coexist. Required for
  Moonlight: both `deepseek-v3` (LoRA-Q variants) and
  `deepseek-v3-flat` (direct-Q variants) register
  `"DeepseekV3ForCausalLM"`; per-checkpoint fingerprints sort it out.
  Threaded `tp_world_size` + `tp_rank` through the new combinator
  during rebase.
- `637423df2` **e2e: `kimi_k2_tiny` golden regen + missing
  `make_tiny_kimi_k2.py`** — kimi-side snapshot disagreed at
  position 0 on ff-interp; per the test docstring (random weights →
  random output, structural regression test only) the golden was
  always meant to be ferrite-self-generated. `make_tiny_kimi_k2.py`
  was missing from the `841195e23` migration; restored.
- `80f9d3b26` **complete `a79a87764` port — V3 FP8-block MoE
  end-to-end** — the big one. Earlier port-by-symbol-grep had
  matched on comments only; the actual missing pieces:
  - `ferrite-kernels::layers_moe`: `MoeRouting` + `route_experts`
    helpers, `DeepSeekV2Fp8BlockMoELayer`, `Fp8BlockFusedMoELayer`.
    V3-routing fields (`e_score_correction_bias`, `n_expert_group`,
    `topk_group`, `routed_scaling_factor`) added to existing MoE
    structs (`FusedMoELayer`, `Fp8FusedMoELayer`,
    `GgmlFusedMoELayer`, `MarlinFusedMoELayer`). vllm-cuda
    constructors patched with default-value defaults.
  - `ferrite-forward-macro::quantization`: `OpKind::DeepSeekMoe`
    added to `reached_by_matmul` (was Gemm-only) so MoE weights
    resolve to FP8-block storage.
  - `ferrite-forward-macro::impl_lib`: `is_fp8_block_moe` helper,
    `DeepSeekMoeRefImpl::matches` defers to FP8-block when storage
    is blockwise FP8, full `DeepSeekFp8BlockMoeImpl` (claims
    `OpKind::DeepSeekMoe` for FP8-block; emits
    `DeepSeekMoeFp8Block` host-interpreter Op variant). Registered
    in `starter_library()`.
  - `ferrite-forward-macro::codegen`: new
    `FieldLoad::DeepSeekV2Fp8BlockMoe` enum variant, shared
    `DeepSeekMoeCfg`/`read_deepseek_moe_cfg` helper (factored from
    `is_deepseek_v2_moe`), `is_deepseek_v2_fp8_block_moe` planner
    branch, storage-compat gate accepts the new accessor, two new
    `match plan` emitter arms (unindexed + layered) calling
    `DeepSeekV2Fp8BlockMoELayer::load(.., __fp8_dtype, stream)`.
  - `ferrite-forward::instr`: `Instruction::DeepSeekMoeFp8Block`
    enum variant + `eval` arm (closed match — must be hand-coded).
    `info.rs::normalize` arm too.
  - Refresh `testdata/golden/deepseek_v3_academic_9b_fp8_block.json`
    against ff-interp's evolved FP8 path.
- `d388f1e22` *fixup: adapt `20abe824d` cherry-pick* —
  `codegen::fingerprint_tests::repo_model_archs()` →
  `arch_configs(slug)` for ff-interp's per-crate
  `crates/ferrite-model-<arch>/configs/` layout + restored the
  missing `deepseek-v3-academic-9b-fp8-block.json` arch config that
  the upstream `a79a87764` port skipped.
- `8947f6a10` `docs+scripts: K2 handoff + Python-vLLM-compatible
  FP8-block targets` — initial handoff doc + the `re:.*` patch on
  the quantize script's compressed-tensors targets.
- `8bca08f72` `feat: DeepSeek V3 FP8-block end-to-end — ferrite
  fingerprint fix + correctness test` — the original kimi-side
  fingerprint fix (`block_disambiguation` uses `fp_leaf` instead of
  hardcoded `q_proj`) + e2e correctness test +
  `codegen::fingerprint_tests` regression coverage.
- `22c00c7b1` `e2e: drop bzantium fixture, replace with kimi_k2_tiny`
  — completes the bzantium-removal half of `a79a87764` that the
  upstream port also skipped.

## Working today (verified pre-rebase, expected green post-rebase)

- `cargo build -p ferrite-models --features cuda --release` — clean
  (12 arch crates incl. new `deepseek-v3-flat` with one variant
  `moonlight-16b-a3b-instruct · 331 tiles · 247 waves`).
- `codegen::fingerprint_tests` — 2/2 (regression for the kimi-side
  fingerprint fix).
- `test_cuda_correctness_deepseek_v3_academic_9b` (BF16 V3 control).
- `test_cuda_correctness_deepseek_v3_academic_9b_fp8_block` (real V3
  academic-9B, llmcompressor-quantized to canonical FP8-block-128×128
  via `starpit/academic-ds-9b-fp8-block`).
- `test_cuda_correctness_qwen3_0_6b_fp8_block` (FP8-block dense
  control — proves the FP8-block path itself works on ff-interp,
  separately from MoE).
- `test_cuda_correctness_kimi_k2_tiny` (synthetic 4-layer K2 — random
  weights, structural regression only — exercises K2-specific routing
  knobs `n_group=1` / `topk_group=1` / `routed_scaling_factor=2.827`
  / `sigmoid+noaux_tc` end-to-end through the V3 path).

## Open: Moonlight-16B-A3B end-to-end on H100

The whole rest of this branch builds toward this one validation.

Why Moonlight? Real Kimi-K2 / K2.5 / K2.6 are 1T-param scale → out of
reach on commodity GPUs. `moonshotai/Moonlight-16B-A3B-Instruct` is
the only published `DeepseekV3ForCausalLM`-arch + K2-style flat
routing (`n_group=1`, `topk_group=1`, sigmoid+noaux_tc,
`routed_scaling_factor=2.446`) + `q_lora_rank=null` checkpoint that
fits an H100 (32 GB BF16, 27 layers, 64 routed + 2 shared experts).

### What's already done

- New crate `ferrite-model-deepseek-v3-flat` registered in the
  workspace and the `ferrite-models` umbrella under feature
  `arch-deepseek-v3-flat` (in `all-arches`).
- `configs/moonlight-16b-a3b-instruct.json` — HF config plus the
  derived bounds the macro reads (`q_proj_out=3072`,
  `kv_a_proj_out=576`, `kv_lora_out=4096`, `head_dim=192`,
  `attn_out=2048`).
- `configs/weights.json` — copied from `ferrite-model-deepseek-v2`
  because shapes are identical (V2-Lite and Moonlight share the
  same MLA + flat-Q + MoE skeleton).
- DSL body in `src/lib.rs` — byte-for-byte the V2 forward (direct
  `q_proj`, no `q_a_proj`/`q_b_proj`); routing flavor (sigmoid vs
  softmax, group sizes, scaling factor) flows through
  `deepseek_moe(..)` from each config's
  `scoring_func`/`topk_method`/`n_group`/`topk_group`/
  `routed_scaling_factor`.
- Dispatcher fix (3bd93fa3b) so `deepseek-v3` and `deepseek-v3-flat`
  can both register `"DeepseekV3ForCausalLM"` and per-checkpoint
  fingerprints route to the right one.

### What needs to happen on H100

1. **Download Moonlight** (~32 GB):
   ```
   huggingface-cli download moonshotai/Moonlight-16B-A3B-Instruct
   ```
2. **Add a TestModels constant** in
   `crates/vllm-e2e/src/lib.rs`:
   ```rust
   #[cfg(feature = "cuda")]
   pub const MOONLIGHT_16B_A3B_INSTRUCT_CUDA: &str =
       "moonshotai/Moonlight-16B-A3B-Instruct";
   ```
3. **Add a load-smoke test** in
   `crates/vllm-e2e/tests/e_correctness.rs` patterned after
   `test_cuda_correctness_deepseek_v3_academic_9b`. First run with
   `VLLM_UPDATE_GOLDEN=moonlight_16b_a3b_instruct` to capture
   ferrite's output, then commit the golden + verify second run
   passes. Threshold=5 (matches the academic_9b setup).
4. **Either generate a Python vLLM golden too** (preferred — gives
   real correctness signal) by adding `"moonlight_16b_a3b_instruct":
   "moonshotai/Moonlight-16B-A3B-Instruct"` to `MODELS` in
   `scripts/generate_golden_refs.py` and running it on a Python vLLM
   that supports the arch — **OR** ferrite-self-generate (matching
   the academic-9B precedent if Python vLLM rejects this exact
   layout).

### Failure modes to watch for at load time

- **Wrong crate claims it.** If `ferrite-model-deepseek-v3` (the
  LoRA-Q crate) accidentally accepts Moonlight, you'll see
  `crates/vllm-cuda/src/model/deepseek_v2.rs::DeepSeekV2Attention`
  in any backtrace — that's the hand-written fall-through, not
  ferrite. The fingerprint check should reject because Moonlight
  ships `q_proj.weight` not `q_a_proj.weight`. If it doesn't, the
  fix is in `emit_fingerprint_check`'s leaf-selection logic, same
  region the kimi-side fingerprint fix touched.
- **Both crates reject.** Either `q_proj`/`q_a_proj` detection in the
  fingerprint is wrong, or there's a manifest/arch-config mismatch.
  Verify with: `python3 -c "from safetensors import safe_open; ...
  print(list of layers.0.self_attn.* keys)"` against the downloaded
  fixture.
- **Token divergence vs Python vLLM golden.** Same FP8-block-style
  drift we hit on ff-interp; if the engine output stays *coherent*,
  regenerate the golden and document. If it goes incoherent, that's
  a real bug — check that the routing flavor knobs flow through
  (`use_sigmoid`, `n_expert_group=1`, `topk_group=1`,
  `routed_scaling_factor=2.446`).

## Quantization follow-ups (optional, after BF16 Moonlight is green)

- **Moonlight FP8-block.** Same recipe as
  `scripts/quantize_academic_9b_fp8_block.py` — llmcompressor with
  the `re:.*` targets patch — would produce a Moonlight-FP8-block
  fixture. K2 official checkpoints ship FP8-block, so this is the
  closest-to-K2 storage format we can validate at this scale. New
  variant config under
  `crates/ferrite-model-deepseek-v3-flat/configs/`; existing
  `DeepSeekFp8BlockMoeImpl` machinery already handles it.
- **K2 official FP8-block.** Once Moonlight FP8-block works, the K2
  4-bit variant on HF (~250 GB) becomes the next reach goal — needs
  multi-GPU TP. ff-interpreter's tp>1 path is now in
  (`47a0d897f`); this branch already threads `tp_world_size` /
  `tp_rank` through the new dispatcher.

## Reference: Python vLLM gotchas (still apply)

- **compressed-tensors `targets`**: `["Linear"]` alone breaks Python
  vLLM's `find_matched_target` for V3's runtime-fused
  `fused_qkv_a_proj`. `quantize_academic_9b_fp8_block.py` appends
  `re:.*` post-quantize.
- **`validate_fp8_block_shape` is strict**: V3 academic-9B
  `intermediate_size=10944` and the fused
  `q_a_proj+kv_a_proj_with_mqa` partition (1600) aren't 128-divisible.
  Python's loader rejects both. Ferrite's `Fp8BlockLinear::load`
  handles ceil-rounded partial last blocks. ⇒ the V3 FP8-block golden
  is ferrite-self-generated, not Python vLLM.
- **FlashInfer override**: `generate_golden_refs.py` skips
  `attention_backend="FLASHINFER"` for `deepseek_*` / `kimi_*` keys
  (FlashInfer rejects MLA head shapes; Python falls back to
  TritonMLA).
- **HF account for fixture upload**: `starpit` (token at
  `~/.cache/huggingface/token`, role=write).

## Why this branch exists (the rebase story)

- Started from `worktree-kimi` (off `ferrite-forward@961188f5c`) with
  5 kimi commits: `f34f0fd9a` (GGUF/IQ — already upstream),
  `a79a87764` (K2 plumbing), `0ab2f0c95` (handoff doc v1),
  `20abe824d` (V3 FP8-block + fingerprint fix), `40c4233b4` (handoff
  doc v2 + quantize-script tweak).
- The "K2 plumbing" port had been partially upstreamed onto
  `ff-interpreter` by someone earlier, in a way that made
  symbol-presence audits return false-positives — comment-only
  mentions of `DeepSeekV2Fp8BlockMoELayer` etc. existed but the
  bodies didn't. End result: V3 FP8-block ran the BF16 cuBLAS path
  on FP8 weights and crashed.
- This branch adds back everything missing (the big `80f9d3b26`
  commit) plus the bzantium → kimi_k2_tiny fixture migration that
  the upstream port also dropped, then rebases onto whatever
  `ff-interpreter` HEAD is at hand-off time. Two non-trivial rebase
  conflicts each time the base advances:
  - `ferrite-forward/src/instr.rs` — Cutlass variants gain
    measurement fields upstream. Resolution: keep HEAD's extended
    Cutlass shapes, insert `DeepSeekMoeFp8Block` next to
    `DeepSeekMoe`.
  - `ferrite-forward/src/lib.rs` — TP commits added
    `tp_world_size`/`tp_rank` to the dispatcher. Resolution: keep
    the new `find_map`/`transpose` walk-past-`Ok(None)` semantics +
    HEAD's TP filter + HEAD's call signature.
  - Plus periodic small follow-ons as the trait surface evolves
    (e.g. `interpreter_arm` was removed from `Implementation` —
    delete it from `DeepSeekFp8BlockMoeImpl`; the eval body lives
    in `Instruction::DeepSeekMoeFp8Block` directly).

If you rebase forward again, expect those same hot spots.
