#!/usr/bin/env python3
"""Generate PARITY.md from parity.csv.

Usage:
    python3 scripts/generate_parity_md.py          # writes PARITY.md
    python3 scripts/generate_parity_md.py --check   # exits non-zero if PARITY.md is stale
"""

import csv
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "parity.csv"
MD_PATH = ROOT / "PARITY.md"

# Sections whose rows use a rust_mlx column
MLX_SECTIONS = {"Model Architectures — Decoder-Only LLMs"}


def load_csv():
    """Return list of row dicts and an ordered dict of section -> [rows]."""
    rows = []
    sections = OrderedDict()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # normalise values
            for key in ("python", "rust", "rust_mlx"):
                v = row.get(key, "").strip().lower()
                row[key] = v if v else ""
            row["notes"] = row.get("notes", "").strip()
            row["deprecated"] = row.get("deprecated", "").strip().lower() == "yes"
            rows.append(row)
            sections.setdefault(row["section"], []).append(row)
    return rows, sections


def active(rows):
    """Filter to non-deprecated rows."""
    return [r for r in rows if not r["deprecated"]]


def status_icon(val):
    return {
        "yes": "\u2705",
        "no": "\u274C",
        "partial": "\u26A0\uFE0F",
        "na": "N/A",
        "": "",
    }.get(val, val)


def count_status(rows, col):
    """Count yes/partial/no/na for a column across rows."""
    yes = sum(1 for r in rows if r[col] == "yes")
    partial = sum(1 for r in rows if r[col] == "partial")
    no = sum(1 for r in rows if r[col] == "no")
    na = sum(1 for r in rows if r[col] in ("na", ""))
    return yes, partial, no, na


def parity_fraction(rows, col="rust"):
    """Return (matched, applicable) where matched = yes+partial vs python yes/partial."""
    applicable = sum(
        1 for r in rows if r["python"] in ("yes", "partial") and r[col] != "na"
    )
    matched = sum(
        1
        for r in rows
        if r["python"] in ("yes", "partial")
        and r[col] in ("yes", "partial")
        and r[col] != "na"
    )
    return matched, applicable


def rust_only_count(rows, col="rust"):
    """Features where python is no/na but rust is yes."""
    return sum(
        1 for r in rows if r["python"] in ("no", "na", "") and r[col] == "yes"
    )


def generate_summary_table(sections):
    lines = []
    lines.append(
        "| Section | Rust ✅ | Rust ⚠️ | Rust ❌ | Rust-only |"
    )
    lines.append("|---|---:|---:|---:|---:|")

    t_yes = t_partial = t_no = t_only = 0

    for name, rows in sections.items():
        ar = active(rows)
        # Count rust status only for features that are python yes/partial (applicable)
        r_yes = sum(
            1 for r in ar
            if r["python"] in ("yes", "partial") and r["rust"] == "yes"
        )
        r_partial = sum(
            1 for r in ar
            if r["python"] in ("yes", "partial") and r["rust"] == "partial"
        )
        r_no = sum(
            1 for r in ar
            if r["python"] in ("yes", "partial") and r["rust"] == "no"
        )
        r_only = rust_only_count(ar)

        t_yes += r_yes
        t_partial += r_partial
        t_no += r_no
        t_only += r_only

        lines.append(
            f"| [{name}](#{slugify(name)}) "
            f"| {r_yes} | {r_partial} | {r_no} | {r_only} |"
        )

    lines.append(
        f"| **Total** "
        f"| **{t_yes}** | **{t_partial}** | **{t_no}** | **{t_only}** |"
    )
    return "\n".join(lines)


def slugify(name):
    """Convert section name to GitHub-flavored markdown anchor."""
    s = name.lower()
    s = s.replace("&", "and")
    out = []
    for ch in s:
        if ch.isalnum() or ch == "-":
            out.append(ch)
        elif ch in (" ", "/"):
            out.append("-")
        # drop everything else (backticks, parens, etc.)
    # collapse multiple dashes
    result = "-".join(part for part in "".join(out).split("-") if part)
    return result


def generate_section(name, rows):
    lines = []
    lines.append(f"## {name}")
    lines.append("")

    has_mlx = name in MLX_SECTIONS

    if has_mlx:
        lines.append(
            "| Architecture | Python | Rust (Candle) | Rust (MLX) | Notes |"
        )
        lines.append("|---|:---:|:---:|:---:|---|")
    else:
        # Use the first row's section to guess column header
        col1 = "Feature"
        if "Model Architectures" in name:
            col1 = "Architecture"
        elif "Quantization" in name:
            col1 = "Method"
        elif "Attention" in name:
            col1 = "Backend"
        elif "CUDA Compute" in name:
            col1 = "Kernel"
        lines.append(f"| {col1} | Python | Rust | Notes |")
        lines.append("|---|:---:|:---:|---|")

    for row in rows:
        feat = row["feature"]
        py = status_icon(row["python"])
        rust = status_icon(row["rust"])
        notes = row["notes"]
        dep = row["deprecated"]

        if dep:
            feat = f"~~{feat}~~"
            notes = f"~~Deprecated in Python V1~~" + (f" {notes}" if notes else "")

        if has_mlx:
            mlx = status_icon(row["rust_mlx"])
            lines.append(f"| {feat} | {py} | {rust} | {mlx} | {notes} |")
        else:
            lines.append(f"| {feat} | {py} | {rust} | {notes} |")

    return "\n".join(lines)


def generate_md():
    _, sections = load_csv()

    parts = []
    parts.append("# vLLM Feature Parity: Python vs Rust")
    parts.append("")
    parts.append(f"> Last updated: {date.today().isoformat()}")
    parts.append("")
    # Compute global rust counts for legend (exclude deprecated)
    all_rows = active([r for rows in sections.values() for r in rows])
    # Count against python-applicable features
    leg_yes = sum(1 for r in all_rows if r["python"] in ("yes", "partial") and r["rust"] == "yes")
    leg_partial = sum(1 for r in all_rows if r["python"] in ("yes", "partial") and r["rust"] == "partial")
    leg_no = sum(1 for r in all_rows if r["python"] in ("yes", "partial") and r["rust"] == "no")
    leg_rust_only = sum(1 for r in all_rows if r["python"] in ("no", "na", "") and r["rust"] == "yes")

    parts.append("| Symbol | Meaning | Count |")
    parts.append("|--------|---------|------:|")
    parts.append(f"| \u2705 | Implemented | {leg_yes} |")
    parts.append(f"| \u26A0\uFE0F | Partial | {leg_partial} |")
    parts.append(f"| \u274C | Not implemented | {leg_no} |")
    parts.append(f"| \u2795 | Rust-only | {leg_rust_only} |")
    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append("## Summary")
    parts.append("")
    parts.append(
        "> Counts are for Rust parity against Python features. "
        "**Rust-only** = features unique to the Rust port."
    )
    parts.append("")
    parts.append(generate_summary_table(sections))
    parts.append("")
    parts.append("---")

    for name, rows in sections.items():
        parts.append("")
        parts.append(generate_section(name, rows))
        parts.append("")
        parts.append("---")

    # Remove trailing ---
    while parts and parts[-1] == "---":
        parts.pop()

    return "\n".join(parts) + "\n"


def main():
    check_mode = "--check" in sys.argv

    md = generate_md()

    if check_mode:
        if not MD_PATH.exists():
            print("PARITY.md does not exist. Run: python3 scripts/generate_parity_md.py")
            sys.exit(1)
        existing = MD_PATH.read_text(encoding="utf-8")
        if existing != md:
            print(
                "PARITY.md is stale. Run: python3 scripts/generate_parity_md.py"
            )
            sys.exit(1)
        print("PARITY.md is up to date.")
        sys.exit(0)

    MD_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote {MD_PATH}")


if __name__ == "__main__":
    main()
