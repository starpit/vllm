"""Compare Ferrite's fused RmsNorm→GEMM→SiLU against Triton and PyTorch baselines."""
import torch
import triton
import triton.language as tl
import time

# ── Triton fused RmsNorm→MatMul→SiLU ──
@triton.jit
def rms_norm_matmul_silu(
    x_ptr, w_norm_ptr, w_gemm_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    eps: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # RMSNorm: compute norm factor for each row in this tile
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = tl.arange(0, BK)

    # Accumulate for GEMM
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, K, BK):
        # Load x tile
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k + offs_k[None, :]) * stride_xk,
                     mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)

        # Load norm weights for this K chunk
        w_norm = tl.load(w_norm_ptr + k + offs_k, mask=(k + offs_k) < K, other=0.0)

        # Load GEMM weights
        offs_n = pid_n * BN + tl.arange(0, BN)
        w = tl.load(w_gemm_ptr + (k + offs_k[:, None]) * stride_wn + offs_n[None, :],
                     mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)

        # Note: This is NOT a true fused kernel — Triton can't easily fuse
        # RMSNorm (which needs the full row for variance) with per-tile GEMM.
        # A real comparison needs separate Triton kernels.
        acc += tl.dot(x, w)

    # SiLU activation
    acc = acc * tl.sigmoid(acc)

    # Store
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    tl.store(out_ptr + offs_m[:, None] * stride_on + offs_n[None, :],
             acc.to(tl.float16),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def benchmark_pytorch_unfused(batch, hidden, out_feat, warmup=10, iters=100):
    """PyTorch unfused: RMSNorm → Linear → SiLU (3 separate ops)"""
    x = torch.randn(batch, hidden, dtype=torch.float16, device='cuda')
    w_norm = torch.ones(hidden, dtype=torch.float16, device='cuda')
    w_gemm = torch.randn(hidden, out_feat, dtype=torch.float16, device='cuda')
    eps = 1e-6

    def run():
        # RMSNorm
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
        normed = (x.float() / rms).half() * w_norm
        # GEMM
        out = torch.mm(normed, w_gemm)
        # SiLU
        out = out.float() * torch.sigmoid(out.float())
        return out

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    us = (time.time() - t0) / iters * 1e6
    flops = 2 * batch * hidden * out_feat
    tflops = flops / (us * 1e-6) / 1e12
    return us, tflops


def benchmark_triton_unfused(batch, hidden, out_feat, warmup=10, iters=100):
    """Triton unfused: separate RMSNorm kernel + matmul + SiLU"""
    x = torch.randn(batch, hidden, dtype=torch.float16, device='cuda')
    w_norm = torch.ones(hidden, dtype=torch.float16, device='cuda')
    w_gemm = torch.randn(hidden, out_feat, dtype=torch.float16, device='cuda')

    def run():
        # Use PyTorch ops (Triton doesn't have a standard RMSNorm)
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        normed = (x.float() / rms).half() * w_norm
        out = torch.mm(normed, w_gemm)
        out = torch.nn.functional.silu(out.float()).half()
        return out

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    us = (time.time() - t0) / iters * 1e6
    flops = 2 * batch * hidden * out_feat
    tflops = flops / (us * 1e-6) / 1e12
    return us, tflops


print("=" * 60)
print("Comparison: RMSNorm → GEMM → SiLU")
print("=" * 60)

for batch in [256, 1024, 4096]:
    hidden = 4096
    out_feat = 4096
    print(f"\nbatch={batch}, hidden={hidden}, out={out_feat}")
    print(f"  FLOPs: {2*batch*hidden*out_feat/1e9:.1f} GFLOP")

    us_pt, tf_pt = benchmark_pytorch_unfused(batch, hidden, out_feat)
    print(f"  PyTorch unfused: {us_pt:.1f} us, {tf_pt:.1f} TFLOPS (GEMM equiv)")

    us_tr, tf_tr = benchmark_triton_unfused(batch, hidden, out_feat)
    print(f"  Triton/PyTorch unfused: {us_tr:.1f} us, {tf_tr:.1f} TFLOPS")
