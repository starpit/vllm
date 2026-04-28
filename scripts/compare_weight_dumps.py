#!/usr/bin/env python3
"""Compare FERRITE_WEIGHT_DUMP outputs from ferrite vs Python vLLM.

Both sides emit lines like:
  [ferrite-weight-dump] name=X dim=D rank=R/W shape=[...] head_bits=[...] head_vals=[...]

Names differ (ferrite emits unfused q_proj/k_proj/v_proj; Python emits
qkv_proj.weight[q|k|v]), so we key on head_bits + tail_bits, which are
the actual bytes the loader produced for each per-rank slice. A
matching shard appears on both sides; an unmatched shard (one side
only) is the bug.

Usage:
    python compare_weight_dumps.py /tmp/ferrite-wd.log /tmp/python-wd.log
"""

import re
import sys
from collections import defaultdict

LINE_RE = re.compile(
    r"\[ferrite-weight-dump\]\s+"
    r"name=(?P<name>\S+)\s+"
    r"dim=(?P<dim>\d+)\s+"
    r"rank=(?P<rank>\d+)/(?P<world>\d+)\s+"
    r"shape=(?P<shape>\[[^\]]*\])\s+"
    r"head_bits=(?P<head_bits>\[[^\]]*\])\s+"
    r"head_vals=(?P<head_vals>\[[^\]]*\])"
    r"(?:\s+tail_row=(?P<tail_row>\d+)\s+"
    r"tail_bits=(?P<tail_bits>\[[^\]]*\])\s+"
    r"tail_vals=(?P<tail_vals>\[[^\]]*\]))?"
)


def parse(path):
    """Return list of (key, name, rank, shape, head_bits) records.
    key = (rank, world, head_bits, tail_bits) — uniquely identifies the bytes.
    """
    out = []
    with open(path) as f:
        for line in f:
            if "[ferrite-weight-dump]" not in line:
                continue
            m = LINE_RE.search(line)
            if not m:
                print(f"WARN: unparseable line in {path}: {line.strip()[:200]}",
                      file=sys.stderr)
                continue
            d = m.groupdict()
            key = (
                int(d["rank"]),
                int(d["world"]),
                d["head_bits"],
                d.get("tail_bits") or "",
            )
            out.append({
                "key": key,
                "name": d["name"],
                "rank": int(d["rank"]),
                "world": int(d["world"]),
                "shape": d["shape"],
                "head_bits": d["head_bits"],
                "tail_bits": d.get("tail_bits") or "",
            })
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    ferrite = parse(sys.argv[1])
    python = parse(sys.argv[2])
    print(f"ferrite: {len(ferrite)} dumps")
    print(f"python:  {len(python)} dumps")

    f_keys = defaultdict(list)
    for r in ferrite:
        f_keys[r["key"]].append(r)
    p_keys = defaultdict(list)
    for r in python:
        p_keys[r["key"]].append(r)

    only_ferrite = [r for r in ferrite if r["key"] not in p_keys]
    only_python = [r for r in python if r["key"] not in f_keys]

    print(f"\nferrite-only shards (bytes not present anywhere on Python side): "
          f"{len(only_ferrite)}")
    for r in only_ferrite[:30]:
        print(f"  {r['name']}  rank={r['rank']}/{r['world']}  shape={r['shape']}  "
              f"head_bits={r['head_bits'][:80]}")
    if len(only_ferrite) > 30:
        print(f"  ... and {len(only_ferrite) - 30} more")

    print(f"\npython-only shards (bytes not present anywhere on ferrite side): "
          f"{len(only_python)}")
    for r in only_python[:30]:
        print(f"  {r['name']}  rank={r['rank']}/{r['world']}  shape={r['shape']}  "
              f"head_bits={r['head_bits'][:80]}")
    if len(only_python) > 30:
        print(f"  ... and {len(only_python) - 30} more")

    if not only_ferrite and not only_python:
        print("\nALL per-rank weight bytes match between ferrite and Python.")
    else:
        print("\nMISMATCH: at least one per-rank weight slice differs in bytes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
