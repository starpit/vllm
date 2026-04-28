# SPDX-License-Identifier: Apache-2.0
"""FERRITE_WEIGHT_DUMP — print per-rank weight slices for diff against ferrite.

Format mirrors vllm-rs/crates/ferrite-cuda-core/src/weights.rs `dump_shard_head`:
  [ferrite-weight-dump] name=X dim=D rank=R/W shape=[..] head_bits=[..] head_vals=[..] [tail_row=N tail_bits=[..] tail_vals=[..]]

Goal: sort + diff between Python and ferrite to verify the loaders read the
same per-rank bytes from the safetensors mmap. bf16 / f16 print 4-hex bits;
f32 prints 8-hex bits.
"""

from __future__ import annotations

import os
import sys

import torch

_ENABLED: bool | None = None


def _enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("FERRITE_WEIGHT_DUMP") == "1"
    return _ENABLED


def _bits_hex(t: torch.Tensor) -> list[str]:
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        bits = t.contiguous().view(torch.uint16).tolist()
        return [f"{b:04x}" for b in bits]
    if t.dtype == torch.float32:
        bits = t.contiguous().view(torch.uint32).tolist()
        return [f"{b:08x}" for b in bits]
    return [f"<{t.dtype}>" for _ in range(t.numel())]


def _vals_str(t: torch.Tensor) -> list[str]:
    return [f"{float(v):+.6e}" for v in t.float().tolist()]


def dump(name: str, dim: int, rank: int, world: int, tensor: torch.Tensor) -> None:
    if not _enabled():
        return
    if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return
    t = tensor.detach().cpu().contiguous()
    shape = list(t.shape)
    total = t.numel()
    if total == 0:
        return
    flat = t.flatten()
    head_n = min(8, total)
    head = flat[:head_n]
    head_bits = _bits_hex(head)
    head_vals = _vals_str(head)
    tail_part = ""
    if t.dim() == 2 and shape[0] > 1:
        last_row = shape[0] - 1
        tail_n = min(8, shape[1])
        tail = t[last_row, :tail_n]
        tail_bits = _bits_hex(tail)
        tail_vals = _vals_str(tail)
        tail_part = (
            f" tail_row={last_row} tail_bits=[{','.join(tail_bits)}]"
            f" tail_vals=[{','.join(tail_vals)}]"
        )
    print(
        f"[ferrite-weight-dump] name={name} dim={dim} rank={rank}/{world} "
        f"shape={shape} head_bits=[{','.join(head_bits)}] "
        f"head_vals=[{','.join(head_vals)}]{tail_part}",
        file=sys.stderr,
        flush=True,
    )
