#!/usr/bin/env python3
"""P0 checkpoint introspection for mlx-community/gemma-4-12B-4bit.

Dumps safetensors header facts the ferrite port depends on:
  - per-class q/k/v/o projection shapes (sliding layer 0 vs global layer 5)
  - which weights exist on global layers (expect NO v_proj; k_eq_v)
  - scale/bias dtypes for 4-bit vs 8-bit tensors (expect bf16)
  - norm-gain dtypes (expect unquantized bf16)
  - layer_scalar presence + shape + (small) values
  - v_norm absence (RMSNormNoScale has no params)
  - full prefix census (vision/audio tower keys to be skipped)
"""
import json
import struct
import sys
from collections import Counter
from pathlib import Path

SNAP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / (
    ".cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-4bit/snapshots/"
    "8de8ab4d40f6b95a76ffa491e23dd430e1f725b5"
)


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    entries = {}  # name -> (dtype, shape)
    for shard in sorted(SNAP.glob("model-*.safetensors")):
        h = read_header(shard)
        for k, v in h.items():
            if k == "__metadata__":
                continue
            entries[k] = (v["dtype"], v["shape"])
    print(f"total tensors: {len(entries)}")

    # prefix census
    prefixes = Counter(k.split(".")[0] for k in entries)
    print("\n== top-level prefixes ==")
    for p, c in prefixes.most_common():
        print(f"  {p}: {c}")

    # layer 0 (sliding) vs layer 5 (global)
    for layer in (0, 5):
        cls = "SLIDING" if layer % 6 != 5 else "GLOBAL"
        pre = f"language_model.model.layers.{layer}."
        print(f"\n== layer {layer} ({cls}) ==")
        for k in sorted(k for k in entries if k.startswith(pre)):
            d, s = entries[k]
            print(f"  {k[len(pre):]:55s} {d:5s} {s}")

    # embed + final norm + any lm_head
    print("\n== top-level text weights ==")
    for k in sorted(k for k in entries if "layers." not in k and k.startswith("language_model")):
        d, s = entries[k]
        print(f"  {k:60s} {d:5s} {s}")

    # checks
    print("\n== checks ==")
    g = "language_model.model.layers.5.self_attn."
    s0 = "language_model.model.layers.0.self_attn."
    print(f"  global v_proj absent:    {g + 'v_proj.weight' not in entries}")
    print(f"  global v_norm absent:    {not any(k.startswith(g + 'v_norm') for k in entries)}")
    print(f"  sliding v_norm absent:   {not any(k.startswith(s0 + 'v_norm') for k in entries)}")
    ls = [k for k in entries if "layer_scalar" in k]
    print(f"  layer_scalar tensors:    {len(ls)}  e.g. {entries[ls[0]] if ls else 'NONE'}")
    # scale dtypes by bits class: mlp = 8-bit, attn = 4-bit
    for name in (
        s0 + "q_proj", g + "q_proj", g + "k_proj",
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.embed_tokens",
    ):
        w = entries.get(name + ".weight")
        sc = entries.get(name + ".scales")
        bi = entries.get(name + ".biases")
        if w:
            # packed u32 width -> bits: cols_packed = K / (32/bits)
            print(f"  {name.split('language_model.model.')[1]:35s} w={w[0]}{w[1]} scales={sc} biases={bi}")
    # norm gain dtype
    for name in (s0 + "q_norm.weight", g + "q_norm.weight", g + "k_norm.weight",
                 "language_model.model.layers.0.input_layernorm.weight",
                 "language_model.model.norm.weight"):
        if name in entries:
            d, s = entries[name]
            print(f"  {name.split('language_model.model.')[1]:35s} {d} {s}")

    # dump full key list for reference
    out = Path(__file__).parent / "checkpoint_keys.txt"
    with open(out, "w") as f:
        for k in sorted(entries):
            d, s = entries[k]
            f.write(f"{k}\t{d}\t{s}\n")
    print(f"\nfull key list -> {out}")


if __name__ == "__main__":
    sys.exit(main())
