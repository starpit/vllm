#!/usr/bin/env python3
"""Benchmark Triton SiLU kernel at various sizes."""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def silu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def silu_inplace_kernel(x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    out = x * tl.sigmoid(x)
    tl.store(x_ptr + offs, out, mask=mask)


def bench_silu(N, block_size=1024, warmup=20, iters=200):
    x = torch.randn(N, dtype=torch.float32, device='cuda')
    grid = lambda meta: ((N + meta['BLOCK'] - 1) // meta['BLOCK'],)

    # Warmup
    for _ in range(warmup):
        silu_inplace_kernel[grid](x, N, BLOCK=block_size)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(iters):
        silu_inplace_kernel[grid](x, N, BLOCK=block_size)
    torch.cuda.synchronize()
    elapsed_us = (time.time() - t0) / iters * 1e6

    # Bandwidth: read N*4 bytes + write N*4 bytes
    bytes_total = N * 4 * 2
    bw_gbs = bytes_total / (elapsed_us * 1e-6) / 1e9

    return elapsed_us, bw_gbs


def verify_silu(N=1024):
    x = torch.randn(N, dtype=torch.float32, device='cuda')
    x_copy = x.clone()
    expected = x * torch.sigmoid(x)

    grid = lambda meta: ((N + meta['BLOCK'] - 1) // meta['BLOCK'],)
    out = torch.empty_like(x)
    silu_kernel[grid](x, out, N, BLOCK=1024)
    torch.cuda.synchronize()

    max_err = (out - expected).abs().max().item()
    print(f"Verification: max error = {max_err:.6e} (N={N})")
    assert max_err < 1e-5, f"SiLU verification failed: max_err={max_err}"


if __name__ == '__main__':
    print("Triton SiLU Benchmark")
    print("=" * 50)

    verify_silu()
    print()

    sizes = [1_000_000, 4_000_000, 16_000_000, 64_000_000]
    for N in sizes:
        us, bw = bench_silu(N)
        print(f"N={N:>12,}  {us:8.1f} us  {bw:7.1f} GB/s")
