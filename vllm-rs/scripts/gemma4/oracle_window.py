#!/usr/bin/env python3
"""Sliding-window e2e golden: a >1024-token PROMPT so the window mask
actually fires during prefill on all 40 sliding layers (every shorter
prompt degenerates to plain causal — see oracle_goldens.py NOTE), plus
48 greedy continuation tokens whose decode steps run with kv_len > 1024.

Run: cd ~/git/mlx-vlm && uv run python .../oracle_window.py
Writes goldens/window.json {prompt_text, prompt_token_ids, output_*}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _oracle_common import MODEL, chat_ids, greedy_generate, load_text_model

OUT = Path(__file__).resolve().parent / "goldens" / "window.json"


def long_prompt() -> str:
    nums = ", ".join(str(i) for i in range(1, 601))
    return (
        "Here is a list of numbers: " + nums + ". "
        "Question: what is the sum of the first three numbers in the list? "
        "Answer briefly."
    )


def main() -> None:
    lm, tokenizer = load_text_model()
    text = long_prompt()
    ids = chat_ids(tokenizer, text)
    assert len(ids) > 1100, f"prompt only {len(ids)} tokens — window won't fire"
    toks = greedy_generate(lm, tokenizer, ids, max_tokens=48)
    out_text = tokenizer.decode(toks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(
            {
                "model": MODEL,
                "prompt_text": text,
                "prompt_token_ids": list(ids),
                "output_token_ids": toks,
                "output_text": out_text,
            },
            f,
            indent=2,
        )
    print(f"T={len(ids)} out={len(toks)} -> {out_text!r}")


if __name__ == "__main__":
    main()
