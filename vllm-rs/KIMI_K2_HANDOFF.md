# Kimi K2 — handoff

Updated 2026-04-29. Branch `worktree-kimi-ff-interp` rebased onto
`a62f1bb5f` (ff-interpreter HEAD: "disable FlashInfer fleet-wide
pending tp>1 fix"). **Moonlight-16B-A3B-Instruct BF16 + FP8-block both
green** — see commits `0627f471b` (BF16) and `aa6c1709c` (FP8-block).

## TL;DR — what to do first on the new machine

1. `git checkout worktree-kimi-ff-interp` and confirm `git log -1` is `aa6c1709c`.
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
       test_cuda_correctness_moonlight_16b_a3b_instruct \
       test_cuda_correctness_moonlight_16b_a3b_instruct_fp8_block
   ```
   All five are expected green. `kimi_k2_tiny` is dropped — Moonlight
   BF16 + FP8-block are the real-model coverage for K2-flat-routing.
4. **The next chunk:** K2 official FP8-block (~250 GB, multi-GPU TP).
   Moonlight FP8-block validates the FP8-block MoE path at 16B scale;
   K2 official is a straight scale-up once TP is plumbed. See
   **§ Quantization follow-ups** below.

## What's on this branch (9 commits past `a62f1bb5f`)

Most-recent first:

- `0627f471b` **`ferrite-model-deepseek-v3-flat`: fix Moonlight garbage
  output (3 bugs)** — root-cause audit of Moonlight producing incoherent
  output; three independent bugs found and fixed:
  1. `routed_scaling_factor` double-apply: `DeepSeekV2MoELayer` and
     `Fp8BlockMoELayer` both folded the factor into `topk_noaux_tc`
     routing weights AND called `scale_inplace` after experts → 2.446²
     ≈ 5.98× instead of 2.446×. Python passes 1.0 to the inner experts.
  2. RoPE cache used `head_dim=192` instead of `qk_rope_head_dim=64` →
     wrong `inv_freq` (1/50000^(2i/192) vs 1/50000^(2i/64)). Fixed in
     `codegen.rs`.
  3. `e_score_correction_bias` is BF16 in Moonlight checkpoints;
     `topk_noaux_tc` asserted F32. Fixed with inline cast.
  Adds `moonlight_16b_a3b_instruct` golden; removes stale `kimi_k2_tiny`
  golden (see smoke-fleet note above).
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
- `test_cuda_correctness_moonlight_16b_a3b_instruct` (green) — real
  16B BF16 MoE model, `q_lora_rank=null`, `sigmoid+noaux_tc`,
  `routed_scaling_factor=2.446`. Exercises the full K2-flat-routing
  path end-to-end with real weights.
- `test_cuda_correctness_moonlight_16b_a3b_instruct_fp8_block`
  (**NEW, green**) — same model quantized to FP8-block-128×128 via
  llmcompressor; hosted at `starpit/moonlight-16b-a3b-instruct-fp8-block`.
  Validates `DeepSeekFp8BlockMoeImpl` + `DeepSeekV2Fp8BlockMoELayer` at
  16B scale with K2-flat routing. Golden captured on L40S (SM89).

## ~~Open~~: Moonlight-16B-A3B end-to-end — **DONE**

Completed in `0627f471b`. Three bugs were blocking coherent output
(double routed_scaling_factor, wrong RoPE head_dim, BF16 bias assert).
Golden committed; test is green.

## Quantization follow-ups

- ~~**Moonlight FP8-block.**~~ **DONE** — `aa6c1709c`. Script at
  `scripts/quantize_moonlight_fp8_block.py`; checkpoint at
  `starpit/moonlight-16b-a3b-instruct-fp8-block` (includes
  `tokenizer.json`). Correctness test green on L40S.
- **K2 official FP8-block.** The K2 official checkpoint on HF (~250 GB)
  becomes the next reach goal — needs multi-GPU TP. ff-interpreter's
  tp>1 path is in (`47a0d897f`); this branch already threads
  `tp_world_size` / `tp_rank` through the new dispatcher.

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
