#!/usr/bin/env python3
"""Diff ferrite VisionWeights::forward intermediates against the Python golden.

Usage:
    ~/vllm/.venv/bin/python diff_qwen2_vl_vision.py \
        --golden tests/goldens \
        --ferrite /tmp/ferrite_vit_dump

Reports per-stage shape/dtype check + bf16-tolerant L1 / Linf / cosine on a
flattened view. Stages diffed (when both sides have them):
    patch_embed_out, block_0_out, block_1_out, block_15_out, block_31_out,
    merger_out
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_golden(dir_: Path) -> dict[str, np.ndarray]:
    manifest = json.load(open(dir_ / "goldens.json"))
    out = {}
    for name, meta in manifest.items():
        if name in ("model", "input_png", "vision_config"):
            continue
        path = dir_ / f"{name}.bin"
        dtype = np.dtype(meta["dtype"])
        arr = np.fromfile(path, dtype=dtype).reshape(meta["shape"])
        # Goldens store bf16 as uint16; reinterpret on read.
        if dtype == np.uint16 and name not in ("cu_seqlens",):
            arr = bf16_uint16_to_f32(arr)
        out[name] = arr
    return out


def load_ferrite(dir_: Path) -> dict[str, tuple[np.ndarray, dict]]:
    """Returns {name: (arr_f32, meta)} from the dump.jsonl + bin sidecars."""
    log = dir_ / "dump.jsonl"
    if not log.exists():
        print(f"no {log}", file=sys.stderr)
        return {}
    out = {}
    for line in log.read_text().splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        name = meta["name"]
        path = dir_ / f"{name}.bin"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        raw = np.fromfile(path, dtype=np.uint8)
        dtype_str = meta["dtype"]
        if dtype_str in ("BF16",):
            arr = bf16_bytes_to_f32(raw).reshape(meta["shape"])
        elif dtype_str in ("F32",):
            arr = raw.view(np.float32).reshape(meta["shape"])
        elif dtype_str in ("F16",):
            arr = raw.view(np.float16).astype(np.float32).reshape(meta["shape"])
        elif dtype_str in ("I32",):
            arr = raw.view(np.int32).reshape(meta["shape"])
        elif dtype_str in ("I64",):
            arr = raw.view(np.int64).reshape(meta["shape"])
        else:
            print(f"unknown dtype {dtype_str} for {name}", file=sys.stderr)
            continue
        out[name] = (arr, meta)
    return out


def bf16_uint16_to_f32(u: np.ndarray) -> np.ndarray:
    # bf16 = high 16 bits of f32 little-endian.
    extended = np.zeros(u.shape + (2,), dtype=np.uint16)
    extended[..., 1] = u
    return extended.view(np.float32).reshape(u.shape)


def bf16_bytes_to_f32(b: np.ndarray) -> np.ndarray:
    u = b.view(np.uint16)
    return bf16_uint16_to_f32(u)


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    af, bf = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    diff = af - bf
    l1 = float(np.mean(np.abs(diff)))
    linf = float(np.max(np.abs(diff)))
    a_norm = float(np.linalg.norm(af) + 1e-30)
    b_norm = float(np.linalg.norm(bf) + 1e-30)
    cos = float(np.dot(af, bf) / (a_norm * b_norm))
    rel = float(np.mean(np.abs(diff) / (np.abs(af) + np.abs(bf) + 1e-6)))
    return {"l1": l1, "linf": linf, "cos": cos, "rel": rel,
            "abs_a": float(np.mean(np.abs(af))), "abs_b": float(np.mean(np.abs(bf)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--ferrite", type=Path, required=True)
    args = ap.parse_args()

    g = load_golden(args.golden)
    f = load_ferrite(args.ferrite)

    print(f"Golden has: {sorted(g.keys())}")
    print(f"Ferrite has: {sorted(f.keys())}")
    print()

    pairs = [
        ("patches", "ferrite_pixels_in"),
        ("cos", "ferrite_cos_half"),  # golden is full [L,80], ferrite half [L,40]
        ("sin", "ferrite_sin_half"),
        ("cu_seqlens", "ferrite_cu_seqlens"),
        ("patch_embed_out", "ferrite_patch_embed_out"),
        ("block_0_out", "ferrite_block_0_out"),
        ("block_1_out", "ferrite_block_1_out"),
        ("block_15_out", "ferrite_block_15_out"),
        ("block_31_out", "ferrite_block_31_out"),
        ("merger_out", "ferrite_merger_out"),
    ]
    for gname, fname in pairs:
        if gname not in g or fname not in f:
            print(f"  SKIP {gname} ↔ {fname}: missing")
            continue
        ga = g[gname]
        fa, fmeta = f[fname]
        # Special case: cos/sin golden is full [L, head_dim], ferrite is half.
        # Diff first head_dim/2 columns of the golden against ferrite.
        if gname in ("cos", "sin") and ga.shape[-1] == 2 * fa.shape[-1]:
            ga = ga[..., : fa.shape[-1]]
        if gname == "cu_seqlens":
            match = np.array_equal(ga.astype(np.int64), fa.astype(np.int64))
            print(f"{gname:24s} ↔ {fname:32s}: shapes {ga.shape} vs {fa.shape} "
                  f"{'MATCH' if match else 'DIFFER (ga=' + str(ga.tolist()) + ' fa=' + str(fa.tolist()) + ')'}")
            continue
        if ga.shape != fa.shape:
            print(f"{gname:24s} ↔ {fname:32s}: SHAPE MISMATCH ga={ga.shape} fa={fa.shape}")
            continue
        s = stats(ga, fa)
        ok = s["cos"] > 0.99
        flag = " " if ok else " *"
        print(f"{gname:24s} ↔ {fname:32s}: shape={ga.shape} l1={s['l1']:.4g} "
              f"linf={s['linf']:.4g} cos={s['cos']:.6f} rel={s['rel']:.3g} "
              f"|a|={s['abs_a']:.3g} |b|={s['abs_b']:.3g}{flag}")


if __name__ == "__main__":
    main()
