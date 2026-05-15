"""
Standalone MLX SDPA bench at the Llama-3.2-3B prefill shape.

Mirrors what `model(prompt, cache=cache)` invokes internally — drives MLX's
`mx.fast.scaled_dot_product_attention` directly at:

    B=1, N_q=24, N_kv=8, T_q=T_kv=M, D=128, dtype=bfloat16, mask="causal"

The CPU dispatcher in `mlx/backend/metal/scaled_dot_product_attention.cpp`
picks the `sdpa_full` (steel_attention, tiled FA-2) path whenever
`query_sequence_length > 8`, so at M=1024 this is the steel_attention kernel
the comparison cares about.

Output: best / median µs per call over N iters (after warmup).

Usage:
    ~/.venv/bin/python scripts/bench_attn_mlx.py
    ~/.venv/bin/python scripts/bench_attn_mlx.py --m 512,1024,2048 --iters 50
"""

import argparse
import math
import statistics
import time

import mlx.core as mx


def bench_one(m: int, n_q: int, n_kv: int, d: int, iters: int, warmup: int, dtype):
    scale = 1.0 / math.sqrt(d)
    # Allocate inputs once; shape matches Llama-3.2-3B prefill.
    q = mx.random.normal(shape=(1, n_q, m, d)).astype(dtype)
    k = mx.random.normal(shape=(1, n_kv, m, d)).astype(dtype)
    v = mx.random.normal(shape=(1, n_kv, m, d)).astype(dtype)
    mx.eval(q, k, v)

    # Warmup (compile + first-touch).
    for _ in range(warmup):
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        mx.eval(o)

    samples_us = []
    for _ in range(iters):
        # Re-eval inputs so we measure attention only, not graph dedup.
        mx.eval(q, k, v)
        t0 = time.perf_counter_ns()
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        mx.eval(o)
        t1 = time.perf_counter_ns()
        samples_us.append((t1 - t0) / 1000.0)

    samples_us.sort()
    return {
        "best_us": samples_us[0],
        "median_us": statistics.median(samples_us),
        "p10_us": samples_us[max(0, iters // 10 - 1)],
        "p90_us": samples_us[min(iters - 1, (iters * 9) // 10)],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", default="512,1024,2048", help="comma-separated T_q=T_kv values")
    p.add_argument("--n_q", type=int, default=24)
    p.add_argument("--n_kv", type=int, default=8)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = p.parse_args()

    dtype = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}[args.dtype]

    print(f"# MLX SDPA bench — B=1 N_q={args.n_q} N_kv={args.n_kv} D={args.d} dtype={args.dtype} causal")
    print(f"# warmup={args.warmup} iters={args.iters}")
    print(f"# {'M':>6}  {'best (µs)':>11}  {'p10 (µs)':>11}  {'median (µs)':>13}  {'p90 (µs)':>11}")
    for m_str in args.m.split(","):
        m = int(m_str)
        r = bench_one(m, args.n_q, args.n_kv, args.d, args.iters, args.warmup, dtype)
        print(f"  {m:>6}  {r['best_us']:>11.2f}  {r['p10_us']:>11.2f}  {r['median_us']:>13.2f}  {r['p90_us']:>11.2f}")


if __name__ == "__main__":
    main()
