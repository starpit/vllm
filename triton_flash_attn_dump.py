"""Dump Triton flash attention PTX from the official tutorial."""
import torch
import os, glob, time, sys

os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache_flash'

# Import the tutorial flash attention
sys.path.insert(0, os.path.expanduser('~/triton/python/tutorials'))
import importlib.util
spec = importlib.util.spec_from_file_location('fused_attn',
    os.path.expanduser('~/triton/python/tutorials/06-fused-attention.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Test data: batch=1, heads=1, seq=512, d=64 (simple case)
batch, heads, seq_q, head_dim = 1, 1, 512, 64
q = torch.randn(batch, heads, seq_q, head_dim, dtype=torch.float16, device='cuda')
k = torch.randn(batch, heads, seq_q, head_dim, dtype=torch.float16, device='cuda')
v = torch.randn(batch, heads, seq_q, head_dim, dtype=torch.float16, device='cuda')

# Run attention
sm_scale = 1.0 / (head_dim ** 0.5)
attn = mod.attention
out = attn(q, k, v, False, sm_scale).half()
torch.cuda.synchronize()
print(f"Output shape: {out.shape}")

# Benchmark
for _ in range(10):
    attn(q, k, v, False, sm_scale)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    attn(q, k, v, False, sm_scale)
torch.cuda.synchronize()
us = (time.time() - t0) / 100 * 1e6
flops = 2 * heads * seq_q * seq_q * head_dim * 2
tflops = flops / (us * 1e-6) / 1e12
print(f"Triton flash attn: {us:.1f} us, {tflops:.1f} TFLOPS (seq={seq_q}, d={head_dim})")

# Dump PTX
ptx_files = glob.glob('/tmp/triton_cache_flash/**/*.ptx', recursive=True)
if ptx_files:
    ptx_file = max(ptx_files, key=os.path.getmtime)
    ptx = open(ptx_file).read()
    open('/tmp/triton_flash_attn.ptx', 'w').write(ptx)
    lines = ptx.split('\n')
    mma = sum(1 for l in lines if 'mma.sync' in l)
    ldm = sum(1 for l in lines if 'ldmatrix' in l)
    cpa = sum(1 for l in lines if 'cp.async.cg' in l)
    ex2 = sum(1 for l in lines if 'ex2.approx' in l)
    bars = sum(1 for l in lines if 'bar.sync' in l)
    print(f'PTX: {len(ptx)} bytes, {len(lines)} lines')
    print(f'mma.sync: {mma}, ldmatrix: {ldm}, cp.async: {cpa}, ex2: {ex2}, barriers: {bars}')
    for l in lines:
        if '.reg' in l and ('%r<' in l or '%rd<' in l or '%f<' in l or '%p<' in l):
            print(f'  {l.strip()}')
else:
    print("No PTX found")
