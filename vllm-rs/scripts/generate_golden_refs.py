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
}

MAX_TOKENS = 32
NUM_LOGPROBS = 20

OUTPUT_DIR = Path(__file__).parent.parent / "crates" / "vllm-e2e" / "testdata" / "golden"


def generate_for_model(model_id: str, output_key: str):
    print(f"Loading {model_id}...")
    llm = LLM(model=model_id, max_model_len=2048)
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
    for key, model_id in MODELS.items():
        generate_for_model(model_id, key)
    print("Done!")


if __name__ == "__main__":
    main()
