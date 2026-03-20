#!/usr/bin/env python3
"""
Triton comparison benchmark for fused RMSNorm -> GEMM -> SiLU.

Measures:
  1. Unfused baseline: 3 separate PyTorch ops (rmsnorm, matmul, silu)
  2. Fused Triton: single kernel doing all 3 ops

Run:
  python3 triton_fused_bench.py
"""

import torch
import triton
import triton.language as tl
import time
import math


# ===========================================================================
# Reference PyTorch implementation (unfused)
# ===========================================================================

def rmsnorm_pytorch(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight"""
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms * weight.float()).half()


def unfused_rmsnorm_gemm_silu(
    x: torch.Tensor,       # [batch, hidden_size] f16
    w_norm: torch.Tensor,  # [hidden_size] f16
    w_gemm: torch.Tensor,  # [hidden_size, out_features] f16
    eps: float = 1e-6,
):
    """3 separate ops: RMSNorm -> GEMM -> SiLU"""
    normed = rmsnorm_pytorch(x, w_norm, eps)            # [batch, hidden_size] f16
    gemm_out = torch.matmul(normed.float(), w_gemm.float())  # [batch, out_features] f32
    return torch.nn.functional.silu(gemm_out)            # [batch, out_features] f32


# ===========================================================================
# Triton fused kernel
# ===========================================================================

@triton.jit
def fused_rmsnorm_gemm_silu_kernel(
    input_ptr, w_norm_ptr, w_gemm_ptr, output_ptr,
    batch, hidden_size, out_features,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused RMSNorm -> GEMM -> SiLU kernel.

    Each program handles a BLOCK_M x BLOCK_N tile of the output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column ranges for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Phase 1: Compute RMSNorm scale factors for BLOCK_M rows
    # Load full row, compute sum of squares
    acc_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k in range(0, hidden_size, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # input[rm, rk] shape [BLOCK_M, BLOCK_K]
        x_ptrs = input_ptr + rm[:, None] * hidden_size + rk[None, :]
        x = tl.load(x_ptrs).to(tl.float32)
        acc_sq += tl.sum(x * x, axis=1)

    # rsqrt(mean(x^2) + eps)
    scale = 1.0 / tl.sqrt(acc_sq / hidden_size + eps)  # [BLOCK_M]

    # Phase 2: GEMM with inline normalization
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, hidden_size, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load input chunk and normalize
        x_ptrs = input_ptr + rm[:, None] * hidden_size + rk[None, :]
        x = tl.load(x_ptrs).to(tl.float32)
        x_normed = x * scale[:, None]

        # Load norm weights and apply
        wn_ptrs = w_norm_ptr + rk
        wn = tl.load(wn_ptrs).to(tl.float32)
        x_normed = x_normed * wn[None, :]

        # Load GEMM weights: w_gemm[rk, rn]
        wg_ptrs = w_gemm_ptr + rk[:, None] * out_features + rn[None, :]
        wg = tl.load(wg_ptrs).to(tl.float32)

        # Accumulate
        acc += tl.dot(x_normed.to(tl.float16), wg.to(tl.float16)).to(tl.float32)

    # Phase 3: SiLU activation
    sigmoid = 1.0 / (1.0 + tl.exp(-acc))
    result = acc * sigmoid

    # Store
    out_ptrs = output_ptr + rm[:, None] * out_features + rn[None, :]
    tl.store(out_ptrs, result)


# ===========================================================================
# Benchmark harness
# ===========================================================================

def benchmark_fn(fn, warmup=20, iters=100):
    """Benchmark a function, return mean time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / iters * 1e6  # microseconds


def main():
    torch.manual_seed(42)
    device = "cuda"

    hidden_size = 4096
    out_features = 4096
    eps = 1e-6

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    print("Triton Fused RMSNorm->GEMM->SiLU Benchmark")
    print("=" * 60)
    print(f"hidden_size={hidden_size}, out_features={out_features}")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print()

    # Create weight tensors (shared across batch sizes)
    w_norm = torch.randn(hidden_size, device=device, dtype=torch.float16) * 0.1 + 1.0
    w_gemm = torch.randn(hidden_size, out_features, device=device, dtype=torch.float16) * 0.01

    for batch_size in [64, 256, 1024]:
        print(f"--- batch={batch_size} ---")

        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.float16) * 0.1

        # Reference (unfused PyTorch)
        ref_output = unfused_rmsnorm_gemm_silu(x, w_norm, w_gemm, eps)

        # Triton fused kernel
        output = torch.empty(batch_size, out_features, device=device, dtype=torch.float32)
        grid = (batch_size // BLOCK_M, out_features // BLOCK_N)

        def run_triton_fused():
            fused_rmsnorm_gemm_silu_kernel[grid](
                x, w_norm, w_gemm, output,
                batch_size, hidden_size, out_features,
                eps=eps,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )

        run_triton_fused()
        torch.cuda.synchronize()

        # Correctness check
        max_err = (output - ref_output).abs().max().item()
        print(f"  Triton fused max err: {max_err:.4f}")

        # Benchmark: unfused PyTorch
        def run_unfused():
            unfused_rmsnorm_gemm_silu(x, w_norm, w_gemm, eps)

        unfused_us = benchmark_fn(run_unfused)

        # Benchmark: fused Triton
        fused_us = benchmark_fn(run_triton_fused)

        # Benchmark: PyTorch matmul only (for reference)
        def run_matmul_only():
            torch.matmul(x.float(), w_gemm.float())

        matmul_us = benchmark_fn(run_matmul_only)

        flops = 2.0 * batch_size * hidden_size * out_features
        fused_tflops = flops / (fused_us * 1e-6) / 1e12
        unfused_tflops = flops / (unfused_us * 1e-6) / 1e12
        matmul_tflops = flops / (matmul_us * 1e-6) / 1e12

        print(f"  Unfused (PyTorch):     {unfused_us:.1f} us, {unfused_tflops:.1f} TFLOPS")
        print(f"  Fused (Triton):        {fused_us:.1f} us, {fused_tflops:.1f} TFLOPS")
        print(f"  matmul only (PyTorch): {matmul_us:.1f} us, {matmul_tflops:.1f} TFLOPS")
        print(f"  Fused speedup vs unfused: {unfused_us/fused_us:.2f}x")
        print()


if __name__ == "__main__":
    main()
