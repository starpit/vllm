#!/usr/bin/env python3
"""Benchmark Triton RMSNorm kernel at various batch sizes with hidden_size=4096."""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def rmsnorm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    hidden_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight

    One program instance per row.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * hidden_size

    # Accumulate sum of squares in tiles
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += x * x

    # Reduce to get mean(x^2)
    mean_sq = tl.sum(sum_sq, axis=0) / hidden_size
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    # Normalize and multiply by weight
    for off in range(0, hidden_size, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_size
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w
        tl.store(output_ptr + row_start + cols, y.to(tl.float16), mask=mask)


def torch_rmsnorm(x, weight, eps=1e-6):
    """Reference RMSNorm in PyTorch."""
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((x_f32 / rms) * weight.float()).half()


def verify(hidden_size=4096, eps=1e-6):
    """Verify Triton kernel against PyTorch reference."""
    batch = 32
    x = torch.randn(batch, hidden_size, dtype=torch.float16, device='cuda')
    w = torch.randn(hidden_size, dtype=torch.float16, device='cuda')
    out = torch.empty_like(x)

    BLOCK_SIZE = min(hidden_size, 1024)
    grid = (batch,)
    rmsnorm_kernel[grid](out, x, w, hidden_size, eps, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    expected = torch_rmsnorm(x, w, eps)
    max_err = (out.float() - expected.float()).abs().max().item()
    print(f"Verification: max error = {max_err:.6e} (batch={batch}, hidden={hidden_size})")
    assert max_err < 0.01, f"RMSNorm verification failed: max_err={max_err}"


def bench_rmsnorm(batch_size, hidden_size=4096, eps=1e-6, warmup=50, iters=500):
    """Benchmark RMSNorm at given batch size."""
    x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device='cuda')
    w = torch.randn(hidden_size, dtype=torch.float16, device='cuda')
    out = torch.empty_like(x)

    BLOCK_SIZE = min(hidden_size, 1024)
    grid = (batch_size,)

    # Warmup
    for _ in range(warmup):
        rmsnorm_kernel[grid](out, x, w, hidden_size, eps, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(iters):
        rmsnorm_kernel[grid](out, x, w, hidden_size, eps, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    elapsed_us = (time.time() - t0) / iters * 1e6

    # Bandwidth: read input (f16) + read weight (f16) + write output (f16)
    # = batch * hidden * 2 + hidden * 2 + batch * hidden * 2
    bytes_total = batch_size * hidden_size * 2 * 2 + hidden_size * 2  # input + output + weight
    bw_gbs = bytes_total / (elapsed_us * 1e-6) / 1e9

    return elapsed_us, bw_gbs


if __name__ == '__main__':
    print("Triton RMSNorm Benchmark")
    print("=" * 60)

    verify()
    print()

    hidden_size = 4096
    batch_sizes = [1, 32, 256, 1024]

    print(f"{'Batch':>8}  {'Hidden':>8}  {'Time (us)':>10}  {'BW (GB/s)':>10}")
    print("-" * 50)

    for bs in batch_sizes:
        us, bw = bench_rmsnorm(bs, hidden_size)
        print(f"{bs:>8}  {hidden_size:>8}  {us:>10.1f}  {bw:>10.1f}")
