#!/usr/bin/env python3
"""P6 perf: mlx-vlm oracle TTFT/TPOT for gemma-4-12B-it-4bit.

Mirrors bench_ferrite.py: decode config (code prompt T=22, 64 toks) and
prefill config (window prompt T=2930, 8 toks), N runs each (default 5).
TTFT = wall time of the prefill forward + argmax eval; TPOT = mean
per-token wall time of the incremental decode loop (greedy, no eos stop
so run lengths match).

Run: cd ~/git/mlx-vlm && uv run python .../bench_oracle.py [--runs 5]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _oracle_common import chat_ids, load_text_model

HERE = Path(__file__).resolve().parent


def timed_run(lm, ids, max_tokens):
    cache = lm.make_cache()
    t0 = time.perf_counter()
    out = lm(inputs=mx.array([ids]), cache=cache)
    tok = mx.argmax(out.logits[:, -1, :], axis=-1)
    mx.eval(tok)
    ttft = (time.perf_counter() - t0) * 1000
    tok = int(tok.item())
    t1 = time.perf_counter()
    n = 0
    for _ in range(max_tokens - 1):
        out = lm(inputs=mx.array([[tok]]), cache=cache)
        nxt = mx.argmax(out.logits[:, -1, :], axis=-1)
        mx.eval(nxt)
        tok = int(nxt.item())
        n += 1
    tpot = (time.perf_counter() - t1) / max(n, 1) * 1000
    return ttft, tpot


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    p99 = xs[min(n - 1, int(round(0.99 * (n - 1))))]
    return xs[0], med, p99


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    lm, tokenizer = load_text_model()
    window = json.load(open(HERE / "goldens" / "window.json"))["prompt_text"]
    configs = [
        ("decode T=22 out=64", "Write a Python function that reverses a string.", 64),
        ("prefill T=2930 out=8", window, 8),
    ]
    # warmup
    timed_run(lm, chat_ids(tokenizer, "hello"), 4)
    for label, content, max_tokens in configs:
        ids = chat_ids(tokenizer, content)
        ttfts, tpots = [], []
        for _ in range(args.runs):
            ttft, tpot = timed_run(lm, ids, max_tokens)
            ttfts.append(ttft)
            tpots.append(tpot)
        for name, xs in (("TTFT(ms)", ttfts), ("TPOT(ms)", tpots)):
            mn, med, p99 = stats(xs)
            print(
                f"[mlx]     {label:22s} {name}  min={mn:8.2f} "
                f"median={med:8.2f} p99={p99:8.2f}  runs={[round(x,1) for x in xs]}"
            )


if __name__ == "__main__":
    main()
