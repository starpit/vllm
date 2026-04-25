# Kimi K2 — handoff

Updated 2026-04-25, off commit `20abe824d` (V3 FP8-block end-to-end).

## Where we are

K2 plumbing is committed. V3 BF16 + V3 FP8-block are both verified end-to-end on
L4 via ferrite-forward. K2 forward correctness against a real fixture is the
remaining open chunk.

## Working

- Shared `MoeRouting` + `route_experts` helper across all 5 MoE families (`a79a787`).
- `DeepSeekV2Fp8BlockMoELayer` (FP8 E4M3 + 128×128 scales, V3/K2 canonical layout).
- `DeepSeekFp8BlockMoeImpl` claims `OpKind::DeepSeekMoe` for FP8-block storage; `DeepSeekMoeRefImpl` defers to it.
- `FieldLoad::DeepSeekV2Fp8BlockMoe` codegen.
- BF16: `test_cuda_correctness_deepseek_v3_academic_9b` green on L4.
- **FP8-block: `test_cuda_correctness_deepseek_v3_academic_9b_fp8_block` green on L4** via `starpit/academic-ds-9b-fp8-block` (real V3 academic-9B, llmcompressor-quantized to canonical FP8-block-128×128).
- `Fp8GemmImpl` claims V3 dense Linears (`q_a_proj`/`q_b_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`/`o_proj`/dense MLP) — the singleton dispatch already handled it once the fingerprint fix below let ferrite see the variant.

## What was broken (commit 20abe824d fix)

`emit_fingerprint_check`'s `block_disambiguation` arm hardcoded
`q_proj.weight_scale_inv`. MLA arches ship `q_a_proj`, so every V3/K2 FP8-block
checkpoint silently failed the fingerprint and fell back to the hand-written V3
path that loaded FP8 bytes into a BF16 cuBLAS GEMM and crashed on
`kv_a_proj_with_mqa`. Fix: use the existing `fp_leaf` selector in both the
block and per-tensor disambiguation arms. Regression tests cover both branches
(`codegen::fingerprint_tests`).

## Python vLLM gotchas (re-usable for K2)

- **compressed-tensors `targets`**: `["Linear"]` alone breaks Python vLLM's
  `find_matched_target` for the V3 fused `fused_qkv_a_proj` layer. The
  `quantize_academic_9b_fp8_block.py` script now appends `re:.*` to targets
  post-quantize (still gated by `ignore`); the published HF fixture
  (`starpit/academic-ds-9b-fp8-block/config.json`) is patched accordingly.
- **`validate_fp8_block_shape` is strict**: V3 academic-9B has
  `intermediate_size=10944` and a fused `q_a_proj+kv_a_proj_with_mqa` partition
  of 1600 — neither divisible by 128. Python's loader rejects both. Ferrite's
  `Fp8BlockLinear::load` handles ceil-rounded partial last blocks; Python's
  doesn't. **Result:** the V3 FP8-block correctness golden is ferrite-self-
  generated, not Python vLLM. Same convention as the BF16 V3 test.
- **FlashInfer override**: `generate_golden_refs.py` skips
  `attention_backend="FLASHINFER"` for `deepseek_*` / `kimi_*` keys (FlashInfer
  rejects MLA head shapes; Python falls back to TritonMLA).

## Open: K2 forward correctness against a real fixture

The synthetic `kimi-k2-tiny` is load-time smoke only — random weights →
near-uniform output → token-match flake.

Available real K2-arch fixtures surveyed:
- `moonshotai/Moonlight-16B-A3B[-Instruct]` — `DeepseekV3ForCausalLM` arch with
  K2-style flat routing (`n_group=1`, `topk_group=1`, sigmoid+noaux_tc,
  `routed_scaling_factor=2.446`) AT 16GB BF16 → fits L4. **Blocker:**
  `q_lora_rank=None` (V2-style direct `q_proj`), but ferrite's V3 DSL
  unconditionally uses `q_a_proj` → `q_a_layernorm` → `q_b_proj`. Adding a
  `q_lora_rank=None` branch needs either a separate `ferrite-model-deepseek-v2-moe`
  crate or a config-conditional in the DSL.
- Real `Kimi-K2-Instruct` / `Kimi-K2-Base` / `K2.5` / `K2.6` are 1T-param scale
  → too big for L4 even at q4.
- No published `Kimi-K2-tiny` from Moonshot.

Two viable paths forward:
1. **Add q_lora_rank=None branch to ferrite V3 DSL** → covers Moonlight-16B
   (real K2-flat-routing weights) **and** DeepSeek-V2-Lite (currently hand-
   written). High value; bounded scope.
2. **Trim a layer or two from real Kimi K2** like Citaman did with
   command-r-1-layer. Hard — K2's ~1T weights download alone is impractical.

Recommend path 1 in the next session.
