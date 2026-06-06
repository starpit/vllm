#!/usr/bin/env python3
"""Greedy text-only generation oracle for Gemma4-12B (mlx-vlm text tower).

Produces the reference continuations ferrite must match verbatim (greedy).

    cd ~/git/mlx-vlm && uv run python \
        <worktree>/vllm-rs/scripts/gemma4/oracle_generate.py [--max-tokens N]

Writes scripts/gemma4/goldens/generations.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _oracle_common import MODEL, chat_ids, greedy_generate, load_text_model

OUT = Path(__file__).resolve().parent / "goldens" / "generations.json"

PROMPTS = {
    "capital": "What is the capital of France?",
    "fox": "Continue the sentence: the quick brown fox",
    "code": "Write a Python function that reverses a string.",
    # >1024 tokens once tokenized — exercises the sliding window (P6).
    "long": "List the numbers from 1 to 400, separated by commas, then "
            "state which of them are perfect squares.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    lm, tokenizer = load_text_model()

    results = {}
    for tag, text in PROMPTS.items():
        ids = chat_ids(tokenizer, text)
        toks = greedy_generate(lm, tokenizer, ids, max_tokens=args.max_tokens)
        out_text = tokenizer.decode(toks)
        results[tag] = {
            "prompt_text": text,
            "prompt_token_ids": list(ids),
            "output_token_ids": toks,
            "output_text": out_text,
        }
        print(f"[{tag}] T={len(ids)} -> {out_text!r}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
