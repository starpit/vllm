import torch, triton, triton.language as tl, time

@triton.jit
def mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    om = pm * BM + tl.arange(0, BM)
    on = pn * BN + tl.arange(0, BN)
    ok = tl.arange(0, BK)
    ap = a_ptr + om[:, None] * sam + ok[None, :] * sak
    bp = b_ptr + ok[:, None] * sbk + on[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(ap)
        b = tl.load(bp)
        acc += tl.dot(a, b)
        ap += BK * sak
        bp += BK * sbk
    cp = c_ptr + om[:, None] * scm + on[None, :] * scn
    tl.store(cp, acc.to(tl.float16))

m = n = k = 1024
a = torch.randn(m, k, dtype=torch.float16, device='cuda')
b = torch.randn(k, n, dtype=torch.float16, device='cuda')
c = torch.empty(m, n, dtype=torch.float16, device='cuda')
grid = lambda meta: (m // meta['BM'], n // meta['BN'])

for BM, BN, BK in [(64,64,16), (64,64,32), (128,64,32), (128,128,32)]:
    try:
        for _ in range(10):
            mm[grid](a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            mm[grid](a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK)
        torch.cuda.synchronize()
        us = (time.time() - t0) / 100 * 1e6
        tf = 2 * m * n * k / (us * 1e-6) / 1e12
        print(f'{BM}x{BN}_bk{BK}: {us:.1f} us, {tf:.1f} TFLOPS')
    except Exception as e:
        print(f'{BM}x{BN}_bk{BK}: {e}')
