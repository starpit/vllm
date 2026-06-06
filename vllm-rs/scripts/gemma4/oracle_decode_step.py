#!/usr/bin/env python3
"""Decode-step goldens: per-layer hidden at the LAST position for
ids = chat(capital) + [The, capital, of, France, is] (T=25). A pure
causal recompute of position 24 equals the incremental decode step
whose logits pick token 6 (oracle: ' **', margin 2.125 over ' Paris').

Run: cd ~/git/mlx-vlm && uv run python .../oracle_decode_step.py
Outputs goldens/capital_decode5/: layer_NN.npy [1,1,3840] (last pos),
embed_scaled.npy, final_norm.npy, logits_last.npy [1,V].
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _oracle_common import chat_ids, load_text_model  # noqa: E402

OUT = Path(__file__).resolve().parent / "goldens" / "capital_decode5"
GEN = [818, 5279, 529, 7001, 563]  # The capital of France is
NUM_LAYERS = 48


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lm, tok = load_text_model()
    ids = list(chat_ids(tok, "What is the capital of France?")) + GEN
    x = mx.array([ids])
    m = lm.model

    h0 = m.embed_tokens(x) * m.embed_scale
    mx.eval(h0)
    np.save(OUT / "embed_scaled.npy", np.array(h0[:, -1:, :].astype(mx.float32)))

    result = lm(inputs=x, capture_layer_ids=list(range(NUM_LAYERS)))
    logits, hiddens = result.logits, result.hidden_states
    mx.eval(logits, hiddens)
    for i, h in enumerate(hiddens):
        np.save(OUT / f"layer_{i:02d}.npy", np.array(h[:, -1:, :].astype(mx.float32)))
    final = m.norm(hiddens[-1])
    mx.eval(final)
    np.save(OUT / "final_norm.npy", np.array(final[:, -1:, :].astype(mx.float32)))
    np.save(OUT / "logits_last.npy", np.array(logits[:, -1, :].astype(mx.float32)))
    nxt = int(mx.argmax(logits[:, -1, :]).item())
    print(f"T={len(ids)} next={nxt} ({tok.decode([nxt])!r}) -> {OUT}")


if __name__ == "__main__":
    main()
