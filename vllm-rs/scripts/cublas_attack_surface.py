#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""cuBLAS attack-surface diagnostic for the cuBLAS-freedom workstream.

For every Cublas-using pick the solver emits, classify by
how close the best non-cuBLAS alternative is on the same workload:

  Margin classes (cuBLAS is FASTER than CUTLASS by …):
    a) ≤ 0.25% — would flip with any honest cost-eval refinement
    b) ≤ 0.5%
    c) ≤ 0.75%
    d) ≤ 1%
    e) ≤ 2%
    f) ≤ 5%
    g) >  5% — cuBLAS truly faster; needs a new kernel to displace

  Other classes:
    fusion_candidate — pick exists at a shape consistent with a
       fusion-decomposable Gemm (q_proj, gate_proj, up_proj, etc.).
       A DSL-level fusion would absorb this pick before any kernel
       work; flagging here directs effort to fusion authoring.
    no_csv_data — neither cublas nor any cutlass standalone tile
       has a calibrated row at this (M, N, K). Predictor extrapolating
       both sides linregly; gap is meaningless.

Counts are DEDUP'd by `(arch, layer, N, K, M-bucket)` to surface
LOGICAL pick decisions, not the M-bucket multiplication that
inflates raw `grep -c Cublas` counts.

Usage:
  vllm ferrite info --color never > /tmp/dump.txt
  python3 vllm-rs/scripts/cublas_attack_surface.py /tmp/dump.txt

Default CSV path is the L4 sm89 profile in ferrite-cuda-targets.
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CSV = "vllm-rs/crates/ferrite-cuda-targets/profiles/cost_l4_sm89.csv"

# Solver workload-grid points. Each M-bucket in the dump contains
# exactly one of these; the cost-eval reads CSV rows at these M values.
WORKLOAD_GRID = [1, 8, 64, 512, 4096]

# Margin buckets — keys ordered low-to-high; "the first one that fits".
MARGIN_BUCKETS = [
    ("a_le_0.25", 0.0025),
    ("b_le_0.50", 0.0050),
    ("c_le_0.75", 0.0075),
    ("d_le_1.00", 0.0100),
    ("e_le_2.00", 0.0200),
    ("f_le_5.00", 0.0500),
    ("g_gt_5.00", float("inf")),
]

LINE_RE = re.compile(
    r"(?P<kind>Cublas|FusedGateUpSiluMul|FusedGateUpGeluMul|"
    r"FusedQkvRopeCache|FusedQkvRopePrefill|FusedGemmBias)\s+"
    r"\[(?P<slots>[^\]]+)\]\s+"
    r"L=(?P<layer>\d+)"
)
SHAPE_RE = re.compile(r"\(M=(?P<m>[^,]+),N=(?P<n>\d+),K=(?P<k>\d+)\)")


def m_range(mr: str):
    if mr.isdigit():
        m = int(mr)
        return (m, m + 1)
    lo_s, hi_s = mr.split("..")
    lo = int(lo_s)
    hi = float("inf") if hi_s == "∞" else int(hi_s)
    return (lo, hi)


def grid_m_for_bucket(lo: int, hi):
    """Return the workload-grid M point that falls inside [lo, hi)."""
    for g in WORKLOAD_GRID:
        if lo <= g < hi:
            return g
    return lo  # bucket smaller than a grid step


def is_fusion_candidate_shape(n: int, k: int) -> bool:
    """Heuristic: shape suggests this Gemm could be claimed by a fusion
    if the model author wrote one, OR if a CUTLASS-fused peer existed.

    We can't see the FUF from a dump, so use shape-sniffing:
    - Plausible q_proj: N ∈ [hidden..2*hidden] and N is a multiple of 64
      (head_dim grid). Without arch context this is noisy.
    - Plausible kv_proj small-N: N ∈ [256, 2048] (GQA num_kv ∈ [1..16]
      × head_dim ∈ [64, 128, 256]) and K is a typical hidden_size.
    - Plausible gate/up: N is intermediate_size (~2-4× hidden).
    - Plausible down: K is intermediate_size.

    All standalone bf16 Gemms that flow into Silu/Mul/RopeAppend in
    the FUF would have been claimed by an existing fusion if its
    matcher accepted them. Since the dump shows them as standalone
    Cublas, the matcher rejected (e.g., bias-mismatch) OR no fusion
    pattern claims this shape (e.g., MLA q_a/q_b decomposition).

    For a script-level proxy, mark any shape where one of N or K is
    a "fusion-typical" intermediate width as a fusion candidate.
    """
    typical_intermediates = {
        # llama / mistral / qwen2 / granite / smollm / tinyllama
        1536, 2560, 2752, 4864, 5632, 6144, 8192, 8960, 9216, 9728,
        10240, 10944, 11008, 12288, 12800, 13824, 14336, 17920, 18432,
        18944, 21504, 22528, 27648, 28672, 29568, 36864,
        # MLA q-projection ranks
        1536, 24576,
    }
    return n in typical_intermediates or k in typical_intermediates


def load_csv(path: Path):
    """Parse cost CSV → {(kernel, m, n, k): us}."""
    table = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("kernel,"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        kernel = parts[0]
        try:
            m, n, k = int(parts[1]), int(parts[2]), int(parts[3])
            us = float(parts[4])
        except ValueError:
            continue
        table[(kernel, m, n, k)] = us
    return table


def best_non_cublas(csv, m: int, n: int, k: int):
    """Min over all non-cuBLAS calibrated rows at exact (M, N, K).
    Excludes `*_add`/`*_bias`/`*_silu_mul` rows since they encode
    bigger claims than a standalone Gemm and never match a Cublas
    pick's claim-shape; their cost wouldn't be substitutable for
    cuBLAS at a standalone-Gemm pick site."""
    best = (float("inf"), None)
    for (kernel, mm, nn, kk), us in csv.items():
        if mm != m or nn != n or kk != k:
            continue
        if kernel == "cublas":
            continue
        if not kernel.startswith("cutlass_"):
            continue
        if kernel.endswith("_add") or "_silu_mul" in kernel:
            continue
        if "_bias_" in kernel:
            continue
        if us < best[0]:
            best = (us, kernel)
    return best


def classify_margin(cublas_us: float, alt_us: float) -> str:
    """Return the margin bucket key (e.g. 'a_le_0.25')."""
    if alt_us == float("inf"):
        return "no_alt"
    gap = (alt_us - cublas_us) / cublas_us  # positive = cuBLAS faster
    for key, ceiling in MARGIN_BUCKETS:
        if gap <= ceiling:
            return key
    return MARGIN_BUCKETS[-1][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "dump",
        nargs="?",
        default="-",
        help="Path to `vllm ferrite info` dump (or '-' to read stdin).",
    )
    ap.add_argument("--csv", default=DEFAULT_CSV, help=f"Cost CSV path (default: {DEFAULT_CSV}).")
    ap.add_argument(
        "--per-arch",
        action="store_true",
        help="Print a per-arch breakdown after the global summary.",
    )
    ap.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="Sample N picks from each margin bucket (default: 3).",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: cost CSV not found at {csv_path}", file=sys.stderr)
        sys.exit(1)
    csv = load_csv(csv_path)

    if args.dump == "-":
        dump_lines = sys.stdin.read().splitlines()
    else:
        dump_lines = Path(args.dump).read_text().splitlines()

    # Per-arch attack surface, dedup'd by (arch, layer, N, K, grid_m).
    # `seen` tracks distinct picks; `by_class` accumulates classification.
    by_class = Counter()
    by_class_arch = defaultdict(Counter)
    samples = defaultdict(list)
    fusion_candidates_by_class = Counter()
    seen = set()

    cur_arch = "(unknown)"
    cur_kind = None
    for raw in dump_lines:
        if raw.startswith("══"):
            cur_arch = raw.replace("══", "").strip()
            continue
        # Restrict to standalone Cublas picks (the dominant class on
        # this branch). Fused-cuBLAS variants (FusedGateUpSiluMul,
        # FusedQkvRopeCache, etc.) lack shape annotations and need a
        # different analysis path.
        m_kind = LINE_RE.search(raw)
        if not m_kind:
            continue
        if m_kind.group("kind") != "Cublas":
            continue
        m_shape = SHAPE_RE.search(raw)
        if not m_shape:
            continue

        layer = int(m_kind.group("layer"))
        lo, hi = m_range(m_shape.group("m"))
        gm = grid_m_for_bucket(lo, hi)
        n = int(m_shape.group("n"))
        k = int(m_shape.group("k"))

        key = (cur_arch, layer, n, k, gm)
        if key in seen:
            continue
        seen.add(key)

        cb_us = csv.get(("cublas", gm, n, k))
        alt_us, alt_kernel = best_non_cublas(csv, gm, n, k)
        if cb_us is None or alt_us == float("inf"):
            cls = "no_csv_data"
        else:
            cls = classify_margin(cb_us, alt_us)

        by_class[cls] += 1
        by_class_arch[cur_arch][cls] += 1
        if is_fusion_candidate_shape(n, k):
            fusion_candidates_by_class[cls] += 1
        if len(samples[cls]) < args.show_samples:
            samples[cls].append(
                (cur_arch, layer, gm, n, k, cb_us, alt_us, alt_kernel)
            )

    # Print global summary.
    total = sum(by_class.values())
    print("=" * 60)
    print(f"cuBLAS attack surface — {total} distinct (arch, layer, N, K, M) picks")
    print("=" * 60)
    cls_order = [k for k, _ in MARGIN_BUCKETS] + ["no_alt", "no_csv_data"]
    print(f"{'class':<14} {'count':>7}  {'%':>6}  fusion-candidate")
    for cls in cls_order:
        c = by_class.get(cls, 0)
        if c == 0:
            continue
        pct = 100.0 * c / max(total, 1)
        fc = fusion_candidates_by_class.get(cls, 0)
        print(f"  {cls:<12} {c:>7}  {pct:>5.1f}%   {fc:>5}")

    print()
    print("Sample picks per class (cuBLAS / best CUTLASS / kernel):")
    for cls in cls_order:
        if not samples.get(cls):
            continue
        print(f"  [{cls}]")
        for arch, layer, gm, n, k, cb, alt, kn in samples[cls]:
            arch_short = arch.split("/", 1)[-1].strip().split()[0]
            cb_str = f"{cb:>7.1f}" if cb is not None else "      ?"
            alt_str = f"{alt:>7.1f}" if alt != float("inf") else "      ?"
            kn_str = kn or "(none)"
            print(
                f"    {arch_short:<32} L={layer:<2} M={gm:<5} N={n:<6} K={k:<6}"
                f"  cb={cb_str}  alt={alt_str}  ({kn_str})"
            )

    if args.per_arch:
        print()
        print("=" * 60)
        print("Per-arch breakdown")
        print("=" * 60)
        arch_totals = sorted(
            ((sum(c.values()), arch, c) for arch, c in by_class_arch.items()),
            reverse=True,
        )
        for total_a, arch, classes in arch_totals:
            if total_a == 0:
                continue
            print(f"\n{arch}  ({total_a} picks)")
            for cls in cls_order:
                c = classes.get(cls, 0)
                if c == 0:
                    continue
                print(f"    {cls:<12} {c:>5}")


if __name__ == "__main__":
    main()
