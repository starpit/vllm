#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project

"""Generate golden reference logprobs using Python vLLM.

Run on a GPU pod with Python vLLM installed:
    source /root/vllm/.venv/bin/activate
    cd /root/vllm/vllm-rs
    python3 scripts/generate_golden_refs.py

Generates JSON files in crates/vllm-e2e/testdata/golden/ for use by
Rust E2E correctness tests that compare our engine's output against
Python vLLM's output.
"""

import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

# Prompts from Python vLLM's tests/prompts/example.txt
PROMPTS = [
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
    "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020.",
    "Compare and contrast artificial intelligence with human intelligence in terms of processing information.",
    "Describe the basic components of a neural network and how it can be trained.",
    "Write a short story about a robot that dreams for the first time.",
    "Analyze the impact of the COVID-19 pandemic on global economic structures and future business models.",
    "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in Western versus Eastern societies.",
    "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches the worm.'",
]

MODELS = {
    "qwen2_0_5b": "Qwen/Qwen2.5-0.5B",
    "smollm_135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "gemma2_2b": "unsloth/gemma-2-2b-it",
    "granite_3_3_2b": "ibm-granite/granite-3.3-2b-instruct",
    "qwen3_0_6b": "Qwen/Qwen3-0.6B",
    # AWQ — exercises the ferrite-forward MarlinLinear / marlin_gemm path.
    # Both FERRITE_ENABLED and FERRITE_DISABLE=1 runs should match this
    # golden within the same top-N tolerance used for dense correctness tests.
    "llama_3_2_1b_awq": "AMead10/Llama-3.2-1B-Instruct-AWQ",
    # GPTQ symmetric INT4 — exercises the ferrite-forward GPTQ → Marlin
    # path via `MarlinLinear::load_gptq` / `load_gptq_concat`. Same
    # storage-vs-compute split as AWQ: the kernel is identical post-
    # repack; only the loader differs.
    "qwen2_0_5b_gptq": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
    # Gemma2 GPTQ — same loader, different arch (alternating
    # sliding/full attention + GELU MLP + softcap). Proves ferrite's
    # GPTQ plumbing is arch-agnostic.
    "gemma2_2b_gptq": "qilowoq/gemma-2-2B-it-4Bit-GPTQ",
    # Gemma2 AWQ — dolphin-2.9.4 fine-tune of gemma-2-2b, instruct-
    # formatted. Exercises the Marlin AWQ path on the gemma2 stack
    # (alt sliding/full attention, softcap, fused GELU MLP). This
    # repo ships both `embed_tokens.weight` and `lm_head.weight`
    # explicitly; RichardErkhov's gemma-2-2b-it-awq omits
    # embed_tokens and trips the fingerprint gate, so we picked
    # solidrust's materialization-friendly variant.
    "gemma2_2b_awq": "solidrust/dolphin-2.9.4-gemma2-2b-AWQ",
    # TinyLlama GPTQ desc_act=true — exercises the g_idx argsort +
    # sort_indices → gptq_repack_into perm path that Qwen2.5-0.5B and
    # Gemma2-2B don't reach (both ship desc_act=false).
    "tinyllama_1b_gptq_desc_act": "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
    # Compressed-tensors INT4 (Neural Magic / RedHatAI pack format) —
    # exercises the `GptqLayout::WeightPacked` branch of the shared
    # `MarlinLinear::load_gptq` / `load_gptq_concat` loaders. The
    # loader sniffs `.weight_packed` + `.weight_scale`, transposes
    # them back to AutoGPTQ-native `[K/8, N]` / `[num_groups, N]`
    # before the same repack / permute pipeline AutoGPTQ goes
    # through. Same uint4b8 bits end-to-end.
    "tinyllama_1b_w4a16_ct": "nm-testing/TinyLlama-1.1B-Chat-v1.0-W4A16-e2e",
    # BNB4 NF4 (double-quant) — exercises `Bnb4bitLinear::load` +
    # the singleton + fused BNB4 impl family. Qwen3's per-head
    # QK-norm forces the singleton path for attention-side gemms
    # (the QKV-fused matcher can't walk through rmsnorm), while the
    # MLP gate/up still fuses via `Bnb4FusedGateUpSiluMulImpl`.
    "qwen3_0_6b_bnb_4bit": "unsloth/Qwen3-0.6B-bnb-4bit",
    # BNB4 NF4 Llama-3.2-1B — same NF4+double_quant format as the
    # qwen3 entry, exercising the BNB4 Impl family on Llama's plain
    # (no-QK-norm) QKV-rope fused path, so `Bnb4FusedQkvRope{Cache,Prefill}Impl`
    # claims all three Q/K/V Gemms as one unit instead of falling to
    # the singleton.
    "llama_3_2_1b_bnb_4bit": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "qwen2_0_5b_bnb_4bit": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    "gemma2_2b_w4a16_ct": "RedHatAI/gemma-2-2b-it-quantized.w4a16",
    "granite_3_1_2b_gptq": "sroecker/granite-3.1-2b-instruct-gptq",
    "granite_3_2b_bnb_4bit": "unsloth/granite-3.2-2b-instruct-bnb-4bit",
    # Unsloth mirror — `google/gemma-3-1b-it` is gated. Match this to
    # `TestModels::GEMMA3_1B_IT_CUDA` so engine + golden run on the same
    # weights.
    "gemma3_1b": "unsloth/gemma-3-1b-it",
    # CommandR (CohereForCausalLM) — 1-layer trim of real v01 by
    # Citaman (mergekit). Full v01 dims (hidden=8192, head_dim=128,
    # vocab=256000), single decoder layer → ~5GB bf16, fits L4.
    # Trained weights → real logprob signal, token-equivalence test
    # catches actual math bugs.
    "command_r_1l": "Citaman/command-r-1-layer",
    # Mistral-7B-Instruct-v0.3 — dense bf16, `sliding_window=null`.
    # Matches `TestModels::MISTRAL`. First Mistral-arch golden;
    # exercises the non-sliding branch of ferrite-models/src/mistral.rs
    # against Python vLLM's `MistralForCausalLM`.
    "mistral_7b_instruct_v0_3": "unsloth/mistral-7b-instruct-v0.3",
    # FP8 dynamic-per-tensor — exercises the ferrite-forward
    # `Fp8GemmImpl` / `Fp8FusedGemmBiasImpl` / `Fp8FusedGateUpSiluMulImpl` /
    # `Fp8FusedGateUpGeluMulImpl` / `Fp8FusedQkvRope{Cache,Prefill}Impl`
    # family + `Fp8Linear::{load, load_concat}`. All six checkpoints
    # below ship FP8E4M3 weights with BF16 `.weight_scale [N, 1]`
    # (channel-strategy) + dynamic per-token activation quant — the
    # W8A8 class covered by Slice 1 of the FP8 rollout. Each exercises
    # a different arch's DSL body: SwiGLU vs GELU MLP, QK-norm'd vs
    # plain QKV rope, short vs long `num_hidden_layers`.
    "qwen2_0_5b_fp8_dynamic": "RedHatAI/Qwen2.5-0.5B-FP8-dynamic",
    "llama_3_2_1b_fp8_dynamic": "RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic",
    "qwen3_0_6b_fp8_dynamic": "RedHatAI/Qwen3-0.6B-FP8-dynamic",
    "gemma2_2b_fp8_dynamic": "espressor/google.gemma-2-2b-it_W8A8_FP8",
    "gemma3_1b_fp8_dynamic": "RedHatAI/gemma-3-1b-it-FP8-dynamic",
    "granite_3_1_2b_fp8_dynamic": "RedHatAI/granite-3.1-2b-instruct-FP8-dynamic",
    "mistral_7b_v03_fp8_dynamic": "nm-testing/Mistral-7B-Instruct-v0.3-FP8-Dynamic",
    # FP8 static-per-tensor (Slice 2) — pre-calibrated `input_scale`
    # baked into the checkpoint. Same kernels as dynamic; the
    # `Fp8Linear::forward` branch on `input_scale` presence selects
    # the `cutlass_scaled_mm` static epilogue instead of the
    # per-token dynamic quant kernel. Fingerprint disambiguation
    # keys off the on-disk `.input_scale` tensor to pick between the
    # two compiled variants. `gemma-2-2b-it-FP8` and
    # `Llama-3.2-1B-Instruct-FP8` ship the compressed-tensors shape
    # (`quant_method: "compressed-tensors"`, `dynamic: false`);
    # `Qwen2-1.5B-Instruct-FP8` and `Mistral-7B-Instruct-v0.3-FP8`
    # ship the native `quant_method: "fp8"` shape — parser handles
    # both.
    "qwen2_1_5b_fp8_static": "RedHatAI/Qwen2-1.5B-Instruct-FP8",
    "llama_3_2_1b_fp8_static": "RedHatAI/Llama-3.2-1B-Instruct-FP8",
    "gemma2_2b_fp8_static": "RedHatAI/gemma-2-2b-it-FP8",
    "mistral_7b_v03_fp8_static": "RedHatAI/Mistral-7B-Instruct-v0.3-FP8",
    # FP8 blockwise-128×128 (Slice 3). Weights are FP8 `[N, K]` with a
    # 2-D per-block scale tensor `[ceil(N/128), ceil(K/128)]`; ferrite
    # routes these via `Fp8BlockLinear`. Need `enforce_eager=True`
    # like the per-tensor FP8 entries above (CUDA-graph FP8 is
    # nondeterministic across runs) — the factory override below
    # keys off the `_fp8` suffix which matches block too.
    "qwen3_0_6b_fp8_block": "RedHatAI/Qwen3-0.6B-FP8-BLOCK",
    # Phi-3-mini-4k-instruct — `Phi3ForCausalLM`, dense bf16, MHA,
    # no LongRoPE. Matches `TestModels::PHI3_MINI_4K_CUDA`. First
    # ferrite arch with packed on-disk weights (`qkv_proj`,
    # `gate_up_proj`); exercises `LinearLayer::load_dense`'s
    # packed-split fallback.
    "phi3_mini_4k_instruct": "microsoft/Phi-3-mini-4k-instruct",
    # Phi-3-medium-4k-instruct — `Phi3ForCausalLM`, dense bf16, GQA
    # (40 q-heads, 10 kv-heads, head_dim=128, hidden=5120). Matches
    # `TestModels::PHI3_MEDIUM_4K_CUDA`. Exercises the manifest-driven
    # `__packed_splits__` prelude with unequal per-slice row counts.
    "phi3_medium_4k_instruct": "microsoft/Phi-3-medium-4k-instruct",
    # Phi-3.5-mini-instruct — `Phi3ForCausalLM`, dense bf16, MHA, same
    # shapes as Phi-3-mini-4k but max_position_embeddings=131072 with
    # LongRoPE. Matches `TestModels::PHI3_5_MINI_CUDA`. Short prompts
    # keep positions < 4096 (short_factor regime).
    "phi3_5_mini_instruct": "microsoft/Phi-3.5-mini-instruct",
    # Phi-4-mini-instruct — `Phi3ForCausalLM`, dense bf16 GQA
    # (24 q-heads, 8 kv-heads, head_dim=128), `partial_rotary_factor=0.75`
    # ⇒ rotary_dim=96 combined with LongRoPE (factor vectors length
    # rotary_dim/2 = 48) and `tie_word_embeddings=true`. Matches
    # `TestModels::PHI4_MINI_CUDA`. Fits L4 at bf16 (≈3.8B × 2 ≈ 7.6 GB).
    "phi4_mini_instruct": "microsoft/Phi-4-mini-instruct",
    # Phi-3-mini-128k — MHA + LongRoPE (same-shape variant of
    # mini-4k / 3.5-mini; disambiguated via max_position_embeddings).
    "phi3_mini_128k_instruct": "microsoft/Phi-3-mini-128k-instruct",
    # Phi-3-medium-128k — GQA 40/10 + LongRoPE. 14B bf16, A100-class.
    "phi3_medium_128k_instruct": "microsoft/Phi-3-medium-128k-instruct",
    # Phi-4 (14B, not mini) — GQA 40/10, no scaling. A100-class.
    "phi4": "microsoft/Phi-4",
    # Phi-4-reasoning — GQA 40/10, `partial_rotary_factor=1.0`
    # (normalized to full rotary). A100-class.
    "phi4_reasoning": "microsoft/Phi-4-reasoning",
    # Phi-4-reasoning-plus — same shape as Phi-4-reasoning. A100-class.
    "phi4_reasoning_plus": "microsoft/Phi-4-reasoning-plus",
    # Phi-4-mini-reasoning — GQA 24/8, partial=0.75, longrope, tied
    # (identical shape to Phi-4-mini-instruct). Fits on L4.
    "phi4_mini_reasoning": "microsoft/Phi-4-mini-reasoning",
}

MAX_TOKENS = 32
NUM_LOGPROBS = 20

OUTPUT_DIR = Path(__file__).parent.parent / "crates" / "vllm-e2e" / "testdata" / "golden"


def generate_for_model(model_id: str, output_key: str):
    print(f"Loading {model_id}...")
    # tiny-random-cohere ships fp32 weights; our engine auto-casts to
    # bf16. With initializer_range=0.02 the bf16 mantissa loss flips
    # argmax in the uniform-logits random-init regime — force Python
    # vLLM to bf16 so both engines see the same precision. Other test
    # models all carry native bf16 in their config, so the default path
    # already matches.
    #
    # `enforce_eager=True` disables CUDA graph capture. Graph replay
    # introduces nondeterminism in FP8 paths (verified: two Llama FP8
    # runs with graphs diverge within a few tokens; two runs with
    # enforce_eager=True produce bit-identical output). FP8 goldens
    # MUST be generated with enforce_eager=True so ferrite's
    # deterministic output has a stable reference to compare against.
    # Non-FP8 goldens (dense, AWQ, GPTQ, BNB4) use the default graphs
    # path — their numerics are already stable enough that graph
    # nondeterminism stays below the top-N tolerance.
    is_fp8 = "fp8" in output_key.lower()
    kwargs = {"model": model_id, "max_model_len": 2048}
    if is_fp8:
        kwargs["enforce_eager"] = True
        # Ferrite defaults to FlashInfer for FP8; match that in the
        # golden so the top-N comparison isn't backend-skewed.
        # Replaces the retired `VLLM_ATTENTION_BACKEND` env var.
        kwargs["attention_backend"] = "FLASHINFER"
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=NUM_LOGPROBS,
    )

    # Generate one prompt at a time to avoid batching effects on numerics.
    # Batched chunked-prefill can produce different logits than single-prompt.
    results = []
    for prompt_idx, prompt_text in enumerate(PROMPTS):
        print(f"  Prompt {prompt_idx}: {prompt_text[:60]}...")
        outputs = llm.generate([prompt_text], sampling_params)
        output = outputs[0]
        completion = output.outputs[0]

        output_tokens = []
        logprobs_list = []

        for lp_entry in completion.logprobs:
            top = {}
            for token_id, logprob_obj in lp_entry.items():
                tok_text = logprob_obj.decoded_token
                if tok_text is None:
                    tok_text = tokenizer.decode([token_id])
                top[tok_text] = round(logprob_obj.logprob, 6)
            logprobs_list.append(top)

        for token_id in completion.token_ids:
            output_tokens.append(tokenizer.decode([token_id]))

        results.append({
            "prompt": output.prompt,
            "output_tokens": output_tokens,
            "output_text": completion.text,
            "logprobs": logprobs_list,
        })

    output_data = {
        "model": model_id,
        "max_tokens": MAX_TOKENS,
        "num_logprobs": NUM_LOGPROBS,
        "results": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{output_key}.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Wrote {out_path}")


def main():
    # Optional CLI args: one or more golden keys to regenerate. Defaults to all.
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    for key in selected:
        if key not in MODELS:
            raise SystemExit(f"unknown golden key: {key}. known: {sorted(MODELS)}")
        generate_for_model(MODELS[key], key)
    print("Done!")


if __name__ == "__main__":
    main()
