#!/usr/bin/env python3
"""P0 oracle goldens for the Gemma4-12B ferrite port.

Runs mlx-vlm (the checkpoint's converter and the Mac reference) on fixed
prompts and dumps per-layer hidden states + logits for cosine comparison
against ferrite-metal.

Run from the mlx-vlm project so its env resolves:
    cd ~/git/mlx-vlm && uv run python \
        <worktree>/vllm-rs/scripts/gemma4/oracle_goldens.py

Outputs (under scripts/gemma4/goldens/<prompt_tag>/):
    meta.json            prompt text, token ids, greedy next token
    embed_scaled.npy     embed(ids) * sqrt(hidden)        [1, T, 3840] f32
    layer_NN.npy         output of layer NN (post layer_scalar, pre final
                         norm)                            [1, T, 3840] f32
    final_norm.npy       model.norm(h)                    [1, T, 3840] f32
    logits_last.npy      softcapped logits at last pos    [1, vocab]   f32
NOTE: prompts are < 1024 tokens, so the sliding window degenerates to
causal; window correctness is exercised separately by a >1024-token
end-to-end greedy comparison (P6).
"""
import json
from pathlib import Path

import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _oracle_common import MODEL, chat_ids, load_text_model  # noqa: E402
OUT_ROOT = Path(__file__).resolve().parent / "goldens"

PROMPTS = {
    "capital": "What is the capital of France?",
    "fox": "Continue the sentence: the quick brown fox",
    "code": "Write a Python function that reverses a string.",
}

NUM_LAYERS = 48


def main():
    lm, tokenizer = load_text_model()

    for tag, text in PROMPTS.items():
        ids = list(chat_ids(tokenizer, text))
        x = mx.array([ids])
        out_dir = OUT_ROOT / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        # embed + scale golden (the pre-layer-0 hidden)
        m = lm.model
        h0 = m.embed_tokens(x) * m.embed_scale
        mx.eval(h0)
        np.save(out_dir / "embed_scaled.npy", np.array(h0.astype(mx.float32)))

        # full prefill with per-layer capture (cache=None: pure causal pass)
        result = lm(inputs=x, capture_layer_ids=list(range(NUM_LAYERS)))
        logits = result.logits
        hiddens = result.hidden_states
        mx.eval(logits, hiddens)
        assert len(hiddens) == NUM_LAYERS, len(hiddens)

        for i, h in enumerate(hiddens):
            np.save(out_dir / f"layer_{i:02d}.npy", np.array(h.astype(mx.float32)))

        final = m.norm(hiddens[-1])
        mx.eval(final)
        np.save(out_dir / "final_norm.npy", np.array(final.astype(mx.float32)))
        np.save(
            out_dir / "logits_last.npy",
            np.array(logits[:, -1, :].astype(mx.float32)),
        )

        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        meta = {
            "model": MODEL,
            "prompt_text": text,
            "token_ids": ids,
            "greedy_next_token": next_tok,
            "greedy_next_piece": tokenizer.decode([next_tok]),
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{tag}] T={len(ids)} next={next_tok!r} "
              f"({meta['greedy_next_piece']!r}) -> {out_dir}")


if __name__ == "__main__":
    main()
