#!/usr/bin/env python3
"""P6 layerwise bisect: ferrite-metal activation dumps vs mlx-vlm goldens.

Ferrite side: run the server with the dump env set and send the golden
prompt once (T must match FERRITE_DUMP_NUM_TOKENS):

    FERRITE_DUMP_DIR=/tmp/g4dump FERRITE_DUMP_NUM_TOKENS=20 \
        ./target/release/vllm serve mlx-community/gemma-4-12B-it-4bit --port 8399

Then:

    python3 scripts/gemma4/compare_dump.py /tmp/g4dump \
        [--goldens scripts/gemma4/goldens/capital]

Mapping (see pool.rs `maybe_run_dump_pass` for the dump contract):
    ScalarMul occ 0       (b0=out) -> embed_scaled.npy   [1, T, H]
    ScalarWeightMul occ i (b0=out) -> layer_{i:02d}.npy  [1, T, H]
    RmsNorm LAST occ      (b0=out) -> final_norm.npy     [1, T, H]
    TanhSoftCap occ 0     (b1=out) -> logits_last.npy    [1, V] (row T-1)

Dump buffers are whole arena slots ([bucket_m rows, width], activation
dtype, only the first T rows live); goldens are f32.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OUT_BINDING = {  # kernel -> output binding_index (lowering.rs contracts)
    "ScalarMul": 0,
    "ScalarWeightMul": 0,
    "RmsNorm": 0,
    "RmsNormUnit": 0,
    "TanhSoftCap": 1,
}


def load_bin(path: Path, dtype: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    if dtype == "bf16":
        return (raw.astype(np.uint32) << 16).view(np.float32)
    if dtype == "f16":
        return raw.view(np.float16).astype(np.float32)
    raise ValueError(f"unsupported dump dtype {dtype!r}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", type=Path)
    ap.add_argument(
        "--goldens",
        type=Path,
        default=Path(__file__).resolve().parent / "goldens" / "capital",
    )
    ap.add_argument("--cos-threshold", type=float, default=0.999)
    args = ap.parse_args()

    manifest = json.loads((args.dump_dir / "manifest.json").read_text())
    T = manifest["num_tokens"]
    dtype = manifest["dtype"]
    entries = manifest["entries"]

    def find(kernel: str, occurrence: int) -> dict | None:
        b = OUT_BINDING[kernel]
        for e in entries:
            if (
                e["kernel"] == kernel
                and e["occurrence"] == occurrence
                and e["binding_index"] == b
            ):
                return e
        return None

    def last_occ(kernel: str) -> int:
        occs = [e["occurrence"] for e in entries if e["kernel"] == kernel]
        return max(occs) if occs else -1

    def ferrite_rows(entry: dict, width: int, rows: int) -> np.ndarray:
        flat = load_bin(args.dump_dir / entry["file"], dtype)
        need = rows * width
        assert flat.size >= need, (
            f"{entry['file']}: {flat.size} elems < {rows}x{width}"
        )
        return flat[:need].reshape(rows, width)

    # (label, golden file, kernel, occurrence, last-row-only)
    plan: list[tuple[str, str, str, int, bool]] = [
        ("embed_scaled", "embed_scaled.npy", "ScalarMul", 0, False)
    ]
    n_layers = last_occ("ScalarWeightMul") + 1
    for i in range(n_layers):
        plan.append((f"layer_{i:02d}", f"layer_{i:02d}.npy", "ScalarWeightMul", i, False))
    plan.append(("final_norm", "final_norm.npy", "RmsNorm", last_occ("RmsNorm"), False))
    plan.append(("logits_last", "logits_last.npy", "TanhSoftCap", 0, True))

    first_bad: str | None = None
    rows_out: list[tuple[str, float, float, int]] = []
    fer_logits = gold_logits = None
    for label, gfile, kernel, occurrence, last_row in plan:
        gpath = args.goldens / gfile
        entry = find(kernel, occurrence)
        if entry is None or not gpath.exists():
            print(f"{label:14s}  MISSING ({'dump' if entry is None else 'golden'})")
            continue
        gold = np.load(gpath).astype(np.float32).reshape(-1)
        if last_row:
            width = gold.size  # [1, V]
            fer = ferrite_rows(entry, width, T)[T - 1]
            fer_logits, gold_logits = fer, gold
        else:
            width = np.load(gpath).shape[-1]
            fer = ferrite_rows(entry, width, T).reshape(-1)
        cos = cosine(fer, gold)
        max_abs = float(np.max(np.abs(fer - gold)))
        nan_ct = int(np.isnan(fer).sum())
        rows_out.append((label, cos, max_abs, nan_ct))
        flag = ""
        if not (cos >= args.cos_threshold):
            flag = "  <-- DIVERGED"
            if first_bad is None:
                first_bad = label
        print(f"{label:14s}  cos={cos:+.6f}  max|d|={max_abs:11.4f}  nan={nan_ct}{flag}")

    if fer_logits is not None and gold_logits is not None:
        fa, ga = np.argsort(-fer_logits)[:8], np.argsort(-gold_logits)[:8]
        print(f"\nferrite top-8: {[(int(i), round(float(fer_logits[i]), 3)) for i in fa]}")
        print(f"golden  top-8: {[(int(i), round(float(gold_logits[i]), 3)) for i in ga]}")
        print(f"argmax: ferrite={int(fa[0])} golden={int(ga[0])} "
              f"{'MATCH' if fa[0] == ga[0] else 'MISMATCH'}")

    if first_bad is not None:
        # Per-token cosine on the first divergent tensor to show where
        # in the sequence it breaks (position-dependent bugs: rope,
        # sliding window, cache indexing).
        label, gfile, kernel, occurrence, _ = next(
            p for p in plan if p[0] == first_bad
        )
        entry = find(kernel, occurrence)
        gold = np.load(args.goldens / gfile).astype(np.float32)
        width = gold.shape[-1]
        gold2 = gold.reshape(-1, width)
        fer2 = ferrite_rows(entry, width, T if gold2.shape[0] != 1 else T)
        print(f"\nper-token cos at first divergence ({first_bad}):")
        for t in range(min(T, gold2.shape[0])):
            print(f"  t={t:02d}  cos={cosine(fer2[t], gold2[t]):+.6f}")
        print(f"\nFIRST DIVERGENT: {first_bad}")
    else:
        print("\nALL TENSORS MATCH (cos >= threshold)")


if __name__ == "__main__":
    main()
