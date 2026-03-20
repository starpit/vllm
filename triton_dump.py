import torch, triton, triton.language as tl, os

os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache'

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

mm[grid](a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BM=64, BN=64, BK=32)

# Find the PTX in cache
import glob
ptx_files = glob.glob('/tmp/triton_cache/**/*.ptx', recursive=True)
if not ptx_files:
    ptx_files = glob.glob(os.path.expanduser('~/.triton/cache/**/*.ptx'), recursive=True)
if ptx_files:
    ptx_file = max(ptx_files, key=os.path.getmtime)
    ptx = open(ptx_file).read()
    open('/tmp/triton_mm.ptx', 'w').write(ptx)
    print(f'PTX: {ptx_file} ({len(ptx)} bytes)')
    lines = ptx.split('\n')
    mma = sum(1 for l in lines if 'mma.sync' in l)
    ldm = sum(1 for l in lines if 'ldmatrix' in l)
    lds = sum(1 for l in lines if 'ld.shared' in l)
    cpa = sum(1 for l in lines if 'cp.async' in l)
    print(f'mma.sync: {mma}, ldmatrix: {ldm}, ld.shared: {lds}, cp.async: {cpa}')
    print(f'Total lines: {len(lines)}')
else:
    print('No PTX found in cache')
