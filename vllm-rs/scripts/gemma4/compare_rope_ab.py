#!/usr/bin/env python3
"""Byte-compare post-rope Q and attention outputs between the fused
(RopeAppendNormed) and baseline (unfused chain) dumps of the SAME
decode forward. Localizes the rope-prologue divergence to a tensor.

Usage: compare_rope_ab.py <fused_dump_dir> <baseline_dump_dir>
"""
import json
import sys

import numpy as np


def load(d):
    m = json.load(open(d + "/manifest.json"))
    return m["entries"]


def bf16(path, n):
    a = np.fromfile(path, dtype=np.uint16, count=n)
    return (a.astype(np.uint32) << 16).view(np.float32)


def rows(entries, kernel, binding):
    """occurrence-ordered (file, bytes) for kernel@binding."""
    out = []
    for e in entries:
        if e["kernel"] == kernel and e["binding_index"] == binding:
            out.append(e)
    return out


def main():
    fused_dir, base_dir = sys.argv[1], sys.argv[2]
    fe, be = load(fused_dir), load(base_dir)

    # Post-rope Q: fused = RopeAppendNormed b0; baseline = RopeAppend b0.
    fq = rows(fe, "RopeAppendNormed", 0)
    bq = rows(be, "RopeAppend", 0)
    print(f"rope occurrences: fused={len(fq)} baseline={len(bq)}")
    n = min(len(fq), len(bq))
    for i in range(n):
        # valid row 0 only (m=1 decode); q width from slot bytes is
        # bucket-padded — compare the head row by actual q width.
        width = 8192 if fq[i]["bytes"] > 40_000_000 else 4096
        a = bf16(f"{fused_dir}/{fq[i]['file']}", width)
        b = bf16(f"{base_dir}/{bq[i]['file']}", width)
        diff = np.nonzero(a.view(np.uint32) != b.view(np.uint32))[0]
        tag = "GLOBAL" if width == 8192 else "sliding"
        if len(diff):
            d0 = diff[0]
            print(
                f"  rope#{i:2d} {tag}: {len(diff):5d}/{width} lanes differ "
                f"(first lane {d0}: fused={a[d0]:.6f} base={b[d0]:.6f})"
            )
        else:
            print(f"  rope#{i:2d} {tag}: identical")

    # Attention outputs (reads cache K/V -> catches K/V divergence).
    fa = rows(fe, "AttentionViaCache", 0)
    ba = rows(be, "AttentionViaCache", 0)
    print(f"attention occurrences: fused={len(fa)} baseline={len(ba)}")
    for i in range(min(len(fa), len(ba))):
        width = 8192 if fa[i]["bytes"] > 40_000_000 else 4096
        a = bf16(f"{fused_dir}/{fa[i]['file']}", width)
        b = bf16(f"{base_dir}/{ba[i]['file']}", width)
        diff = np.nonzero(a.view(np.uint32) != b.view(np.uint32))[0]
        if len(diff):
            d0 = diff[0]
            print(
                f"  attn#{i:2d}: {len(diff):5d}/{width} lanes differ "
                f"(first lane {d0}: fused={a[d0]:.6f} base={b[d0]:.6f})"
            )
        else:
            print(f"  attn#{i:2d}: identical")


if __name__ == "__main__":
    main()
