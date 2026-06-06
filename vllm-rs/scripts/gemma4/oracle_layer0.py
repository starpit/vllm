#!/usr/bin/env python3
"""Layer-0 INTERNAL goldens for the P6 intra-layer bisect (capital prompt).

Recomputes layer 0 (sliding class) op by op, mirroring
`mlx_vlm/models/gemma4/language.py` exactly, and dumps every
intermediate. Self-validating: the recomputed layer output must match
the existing goldens/capital/layer_00.npy (asserted, cos > 0.9999).

Run from the mlx-vlm project:
    cd ~/git/mlx-vlm && uv run python \
        <worktree>/vllm-rs/scripts/gemma4/oracle_layer0.py

Outputs under goldens/capital_layer0/*.npy, all f32:
    pre_attn_normed   [1,20,3840]   q_proj [1,20,4096]   k_proj [1,20,2048]
    v_proj [1,20,2048] v_normed     q_normed             k_normed
    q_roped [1,16,20,256] (head-major!)                  k_roped [1,8,20,256]
    attn_out [1,20,4096]  oproj     post_attn_normed     h_after_attn
    pre_ffwd_normed   gate_prelin   up_prelin            geglu_out
    down_out          post_ffwd_normed                   h_after_mlp
    layer0_out (== layer_00.npy)
"""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _oracle_common import chat_ids, load_text_model  # noqa: E402

from mlx_vlm.models.gemma4.language import geglu  # noqa: E402

OUT = Path(__file__).resolve().parent / "goldens" / "capital_layer0"
GOLD = Path(__file__).resolve().parent / "goldens" / "capital"
PROMPT = "What is the capital of France?"


def save(name: str, t: mx.array) -> None:
    mx.eval(t)
    np.save(OUT / f"{name}.npy", np.array(t.astype(mx.float32)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lm, tokenizer = load_text_model()
    ids = list(chat_ids(tokenizer, PROMPT))
    assert len(ids) == 20, len(ids)
    x = mx.array([ids])
    m = lm.model
    B, L = 1, len(ids)

    h = m.embed_tokens(x) * m.embed_scale
    layer = m.layers[0]
    attn = layer.self_attn
    assert attn.is_sliding

    xn = layer.input_layernorm(h)
    save("pre_attn_normed", xn)

    q = attn.q_proj(xn)
    save("q_proj", q)
    k = attn.k_proj(xn)
    save("k_proj", k)
    v = attn.v_proj(xn)
    save("v_proj", v)

    q = attn.q_norm(q.reshape(B, L, attn.n_heads, attn.head_dim))
    save("q_normed", q.reshape(B, L, -1))
    k = attn.k_norm(k.reshape(B, L, attn.n_kv_heads, attn.head_dim))
    save("k_normed", k.reshape(B, L, -1))
    v = attn.v_norm(v.reshape(B, L, attn.n_kv_heads, attn.head_dim))
    save("v_normed", v.reshape(B, L, -1))

    q = attn.rope(q.transpose(0, 2, 1, 3), offset=0)  # [1, 16, 20, 256]
    save("q_roped", q)
    k = attn.rope(k.transpose(0, 2, 1, 3), offset=0)  # [1, 8, 20, 256]
    save("k_roped", k)
    v = v.transpose(0, 2, 1, 3)

    # L=20 < window=1024: sliding mask degenerates to plain causal.
    out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=attn.scale, mask="causal"
    )
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    save("attn_out", out)

    oproj = attn.o_proj(out)
    save("oproj", oproj)

    pa = layer.post_attention_layernorm(oproj)
    save("post_attn_normed", pa)
    h2 = h + pa
    save("h_after_attn", h2)

    pf = layer.pre_feedforward_layernorm(h2)
    save("pre_ffwd_normed", pf)
    gate = layer.mlp.gate_proj(pf)
    save("gate_prelin", gate)
    up = layer.mlp.up_proj(pf)
    save("up_prelin", up)
    gg = geglu(gate, up)
    save("geglu_out", gg)
    down = layer.mlp.down_proj(gg)
    save("down_out", down)

    pff = layer.post_feedforward_layernorm(down)
    save("post_ffwd_normed", pff)
    h3 = h2 + pff
    save("h_after_mlp", h3)

    out0 = h3 * layer.layer_scalar
    save("layer0_out", out0)

    # Self-validation vs the e2e golden.
    ref = np.load(GOLD / "layer_00.npy").ravel().astype(np.float64)
    got = np.array(out0.astype(mx.float32)).ravel().astype(np.float64)
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
    print(f"self-check cos(layer0_out, layer_00 golden) = {cos:.6f}")
    assert cos > 0.9999, cos
    print(f"wrote {len(list(OUT.glob('*.npy')))} tensors -> {OUT}")


if __name__ == "__main__":
    main()
