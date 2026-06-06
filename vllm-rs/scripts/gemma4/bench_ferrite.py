#!/usr/bin/env python3
"""P6 perf: ferrite-metal serve TTFT/TPOT for gemma-4-12B-it-4bit.

Two configs, N runs each (default 5), /reset_prefix_cache between runs:
  decode : code prompt (T=22), 64 tokens  -> TPOT
  prefill: window prompt (T=2930), 8 tokens -> TTFT

Usage: python3 bench_ferrite.py [--port 8399] [--runs 5]
Server must already be running.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "mlx-community/gemma-4-12B-it-4bit"


def stream_request(port: int, content: str, max_tokens: int):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    stamps = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            delta = d["choices"][0].get("delta", {})
            if delta.get("content"):
                stamps.append(time.perf_counter())
    ttft = (stamps[0] - t0) * 1000 if stamps else float("nan")
    tpot = (
        (stamps[-1] - stamps[0]) / (len(stamps) - 1) * 1000
        if len(stamps) > 1
        else float("nan")
    )
    return ttft, tpot, len(stamps)


def reset_cache(port: int) -> None:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST"
    )
    urllib.request.urlopen(req, timeout=30).read()


def stats(xs):
    xs = sorted(xs)
    n = len(xs)
    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
    p99 = xs[min(n - 1, int(round(0.99 * (n - 1))))]
    return xs[0], med, p99


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8399)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    window = json.load(open(HERE / "goldens" / "window.json"))["prompt_text"]
    configs = [
        ("decode T=22 out=64", "Write a Python function that reverses a string.", 64),
        ("prefill T=2930 out=8", window, 8),
    ]
    # warmup
    stream_request(args.port, "hello", 4)
    for label, content, max_tokens in configs:
        ttfts, tpots = [], []
        for _ in range(args.runs):
            reset_cache(args.port)
            ttft, tpot, n = stream_request(args.port, content, max_tokens)
            ttfts.append(ttft)
            tpots.append(tpot)
        for name, xs in (("TTFT(ms)", ttfts), ("TPOT(ms)", tpots)):
            mn, med, p99 = stats(xs)
            print(
                f"[ferrite] {label:22s} {name}  min={mn:8.2f} "
                f"median={med:8.2f} p99={p99:8.2f}  runs={xs and [round(x,1) for x in xs]}"
            )


if __name__ == "__main__":
    main()
