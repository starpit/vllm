#!/usr/bin/env python3
"""Clean up corrupt or specific entries from a cudaforge cache.

Usage:
    # Scan for truncated .o files and remove them from the cache:
    python3 scripts/cudaforge_cleanup.py /root/.cache/cudaforge/vllm-cuda

    # Remove a specific kernel by name (substring match):
    python3 scripts/cudaforge_cleanup.py /root/.cache/cudaforge/vllm-cuda --remove flash_fwd_split_hdim64_fp16_sm80

    # Dry run (show what would be removed without changing anything):
    python3 scripts/cudaforge_cleanup.py /root/.cache/cudaforge/vllm-cuda --dry-run

    # Both: scan for truncated + remove a specific kernel:
    python3 scripts/cudaforge_cleanup.py /root/.cache/cudaforge/vllm-cuda --remove flash_fwd_split_hdim64_fp16_sm80
"""

import argparse
import json
import os
import struct
import sys


def is_truncated_elf(path: str) -> bool:
    """Check if an ELF .o file is truncated (section header table past EOF)."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            header = f.read(48)
        if len(header) < 48:
            return True
        magic = header[:4]
        if magic != b"\x7fELF":
            return False  # not ELF, skip
        ei_class = header[4]
        if ei_class == 2:  # 64-bit
            e_shoff = struct.unpack_from("<Q", header, 40)[0]
        elif ei_class == 1:  # 32-bit
            e_shoff = struct.unpack_from("<I", header, 32)[0]
        else:
            return False
        return e_shoff >= size
    except (OSError, struct.error):
        return True


def main():
    parser = argparse.ArgumentParser(description="Clean up cudaforge cache entries")
    parser.add_argument("cache_dir", help="Path to cudaforge cache directory (e.g. /root/.cache/cudaforge/vllm-cuda)")
    parser.add_argument("--remove", metavar="NAME", action="append", default=[],
                        help="Remove cache entries whose source path or object path contains NAME (substring match). Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without making changes")
    args = parser.parse_args()

    cache_json = os.path.join(args.cache_dir, ".cudaforge_cache.json")
    if not os.path.exists(cache_json):
        print(f"Error: {cache_json} not found", file=sys.stderr)
        sys.exit(1)

    with open(cache_json) as f:
        cache = json.load(f)

    entries = cache.get("entries", {})
    to_remove = []  # list of (source_key, object_path, reason)

    # 1) Scan for truncated .o files
    for source_key, entry in entries.items():
        obj_path = entry.get("object_path", "")
        if obj_path and os.path.exists(obj_path) and is_truncated_elf(obj_path):
            to_remove.append((source_key, obj_path, "truncated ELF"))

    # 2) Match --remove patterns
    for pattern in args.remove:
        for source_key, entry in entries.items():
            obj_path = entry.get("object_path", "")
            if pattern in source_key or pattern in obj_path:
                if not any(k == source_key for k, _, _ in to_remove):
                    to_remove.append((source_key, obj_path, f"matched --remove {pattern!r}"))

    if not to_remove:
        print("Nothing to clean up.")
        return

    for source_key, obj_path, reason in to_remove:
        obj_name = os.path.basename(obj_path) if obj_path else "(no object)"
        prefix = "[DRY RUN] " if args.dry_run else ""
        print(f"{prefix}Removing: {obj_name}  ({reason})")
        print(f"  source: {source_key}")

        if not args.dry_run:
            del entries[source_key]
            if obj_path and os.path.exists(obj_path):
                os.remove(obj_path)
                print(f"  deleted: {obj_path}")

    if not args.dry_run:
        with open(cache_json, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"\nUpdated {cache_json} ({len(to_remove)} entries removed, {len(entries)} remaining)")
    else:
        print(f"\n{len(to_remove)} entries would be removed ({len(entries)} total)")


if __name__ == "__main__":
    main()
