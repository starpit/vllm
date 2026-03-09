#!/usr/bin/env python3
"""Debug logprob divergence between Python vLLM and Rust engine.

Phase 1 (--dump-python): Run Python vLLM, save logprobs to /tmp/py_logprobs.json
Phase 2 (--compare):     Start Rust engine, query it, compare against saved Python logprobs

Usage:
  python3 scripts/debug_logprobs.py --dump-python
  # then in another invocation (Python vLLM is gone, GPU is free):
  python3 scripts/debug_logprobs.py --compare --rust-port 8000
"""

import argparse
import json
import subprocess
import sys
import time
import requests

PROMPT = "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs."
MODEL = "Qwen/Qwen2.5-0.5B"
MAX_TOKENS = 32
NUM_LOGPROBS = 20
PY_DUMP = "/tmp/py_logprobs.json"


def dump_python():
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, max_model_len=2048)
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, logprobs=NUM_LOGPROBS)
    outputs = llm.generate([PROMPT], params)
    completion = outputs[0].outputs[0]

    positions = []
    for tid, lp_entry in zip(completion.token_ids, completion.logprobs):
        tok_text = tokenizer.decode([tid])
        top = {}
        for tok_id, logprob_obj in lp_entry.items():
            t = logprob_obj.decoded_token or tokenizer.decode([tok_id])
            top[t] = logprob_obj.logprob
        positions.append({"token": tok_text, "token_id": tid, "top": top})

    data = {"text": completion.text, "positions": positions}
    with open(PY_DUMP, "w") as f:
        json.dump(data, f)
    print(f"Python text: {completion.text!r}")
    print(f"Saved to {PY_DUMP}")


def compare(port):
    with open(PY_DUMP) as f:
        py_data = json.load(f)
    py_positions = py_data["positions"]
    py_text = py_data["text"]

    resp = requests.post(
        f"http://127.0.0.1:{port}/v1/completions",
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "logprobs": NUM_LOGPROBS,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    lp = choice["logprobs"]

    rs_positions = []
    for tok, top_lp in zip(lp["tokens"], lp["top_logprobs"]):
        rs_positions.append({"token": tok, "top": top_lp or {}})
    rs_text = choice["text"]

    print(f"Python text: {py_text!r}")
    print(f"Rust text:   {rs_text!r}")
    print()

    min_len = min(len(py_positions), len(rs_positions))
    for i in range(min_len):
        py = py_positions[i]
        rs = rs_positions[i]
        match = py["token"] == rs["token"]

        if not match:
            print(f"\n*** FIRST MISMATCH at position {i} ***")
            print(f"  Python: {py['token']!r}")
            print(f"  Rust:   {rs['token']!r}")

            py_sorted = sorted(py["top"].items(), key=lambda x: x[1], reverse=True)
            rs_sorted = sorted(rs["top"].items(), key=lambda x: x[1], reverse=True)

            print(f"\n  {'Rank':>4s}  {'Python vLLM':>25s}  {'Rust engine':>25s}")
            print(f"  {'':->4s}  {'':->25s}  {'':->25s}")
            for rank in range(max(len(py_sorted), len(rs_sorted))):
                py_str = rs_str = ""
                if rank < len(py_sorted):
                    t, lp_val = py_sorted[rank]
                    py_str = f"{t!r:>12s} {lp_val:>8.4f}"
                if rank < len(rs_sorted):
                    t, lp_val = rs_sorted[rank]
                    rs_str = f"{t!r:>12s} {lp_val:>8.4f}"
                print(f"  {rank+1:4d}  {py_str:>25s}  {rs_str:>25s}")

            py_tok = py["token"]
            rs_tok = rs["token"]
            print(f"\n  Rust token {rs_tok!r} in Python top-{NUM_LOGPROBS}: {rs_tok in py['top']}", end="")
            if rs_tok in py["top"]:
                print(f" (logprob={py['top'][rs_tok]:.4f})")
            else:
                print()
            print(f"  Python token {py_tok!r} in Rust top-{NUM_LOGPROBS}: {py_tok in rs['top']}", end="")
            if py_tok in rs["top"]:
                print(f" (logprob={rs['top'][py_tok]:.4f})")
            else:
                print()
            break
        else:
            # Check if logprobs are close
            py_lp = py["top"].get(py["token"], 0)
            rs_lp = rs["top"].get(rs["token"], 0)
            diff = abs(py_lp - rs_lp)
            flag = " <-- logprob diff!" if diff > 0.1 else ""
            print(f"Position {i:2d}: OK {py['token']!r:>15s}  py={py_lp:>8.4f}  rs={rs_lp:>8.4f}  diff={diff:.4f}{flag}")
    else:
        print("\nAll positions match!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-python", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--rust-port", type=int, default=8000)
    args = parser.parse_args()

    if args.dump_python:
        dump_python()
    elif args.compare:
        compare(args.rust_port)
    else:
        print("Specify --dump-python or --compare")
        sys.exit(1)


if __name__ == "__main__":
    main()
