#!/usr/bin/env python3
"""Verbatim greedy parity vs goldens/generations.json + multiseq check.

Sends each golden prompt (greedy, exact output length) to a running
ferrite server and requires byte-identical output text. Then fires 3
DIFFERENT prompts concurrently and checks each answer lands on its own
sequence (the GDN-era cross-contamination regression class).

Usage: python3 verify_goldens.py [--port 8399]
"""
import argparse
import concurrent.futures
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "mlx-community/gemma-4-12B-it-4bit"


def chat(port: int, content: str, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8399)
    args = ap.parse_args()

    goldens = json.load(open(HERE / "goldens" / "generations.json"))
    failures = 0
    for name, g in goldens.items():
        # The oracle's output_text includes the EOS marker text when
        # generation stopped on EOS; the server's message content
        # (correctly) excludes the stop token.
        want = g["output_text"].removesuffix("<turn|>")
        got = chat(args.port, g["prompt_text"], len(g["output_token_ids"]))
        ok = got == want
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {len(got)} chars")
        if not ok:
            failures += 1
            print(f"  want: {want[:160]!r}")
            print(f"  got : {got[:160]!r}")

    # Multiseq: 3 different prompts in flight together, 5 rounds.
    probes = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of Egypt?", "Cairo"),
    ]
    for round_i in range(5):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(chat, args.port, q, 24) for q, _ in probes]
            outs = [f.result() for f in futs]
        ok = all(want in out for (_, want), out in zip(probes, outs))
        print(f"[{'PASS' if ok else 'FAIL'}] multiseq round {round_i + 1}")
        if not ok:
            failures += 1
            for (q, want), out in zip(probes, outs):
                print(f"  {want}: {out[:80]!r}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
