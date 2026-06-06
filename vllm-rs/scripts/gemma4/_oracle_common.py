"""Shared text-only oracle loader for the Gemma4 ferrite port.

The local mlx-vlm checkout's gemma4 module tree predates this checkpoint's
tower layout (`vision_embedder.*` on disk vs `vision_tower.*` in the
module), so `mlx_vlm.load` cannot load it strictly. The text decoder is
identical, so instantiate ONLY `LanguageModel` and load the
`language_model.*` weights — no mlx-vlm repo modifications.

Quantization mirrors utils.load_model's get_class_predicate: per-path
entries in config["quantization"] (gemma4-12B: mlp.{gate,up,down}_proj at
8-bit g64) override the global 4-bit g64.
"""
import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.gemma4 import language as g4_language
from mlx_vlm.models.gemma4.config import TextConfig

# The -it (instruction-tuned) checkpoint: same arch/quant layout as the
# base gemma-4-12B-4bit, but the right vehicle for chat verification.
MODEL = "mlx-community/gemma-4-12B-it-4bit"


def _snapshot_dir():
    root = Path.home() / (
        ".cache/huggingface/hub/models--"
        + MODEL.replace("/", "--")
        + "/snapshots"
    )
    snaps = sorted(p for p in root.iterdir() if p.is_dir())
    assert snaps, f"checkpoint not downloaded: {MODEL}"
    return snaps[-1]


SNAP = _snapshot_dir()
PREFIX = "language_model."


def load_text_model():
    """Returns (LanguageModel, tokenizer). Strict load of the text tower."""
    with open(SNAP / "config.json") as f:
        config = json.load(f)

    text_cfg = TextConfig.from_dict(config["text_config"])
    lm = g4_language.LanguageModel(text_cfg)

    weights = {}
    for wf in sorted(glob.glob(str(SNAP / "model-*.safetensors"))):
        for k, v in mx.load(wf).items():
            if k.startswith(PREFIX):
                weights[k[len(PREFIX):]] = v

    if hasattr(lm, "sanitize"):
        weights = lm.sanitize(weights)

    quant = config["quantization"]

    def class_predicate(p, m):
        full = PREFIX + p
        if full in quant:                       # per-path override (8-bit MLP)
            return quant[full]
        if not hasattr(m, "to_quantized"):
            return False
        if hasattr(m, "weight") and m.weight.size % 64 != 0:
            return False
        return f"{p}.scales" in weights

    nn.quantize(
        lm,
        group_size=quant["group_size"],
        bits=quant["bits"],
        mode=quant.get("mode", "affine"),
        class_predicate=class_predicate,
    )

    lm.load_weights(list(weights.items()), strict=True)
    mx.eval(lm.parameters())
    lm.eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SNAP)
    if tokenizer.chat_template is None:
        # Gemma 4 ships chat_template.jinja as a separate repo file
        # (HF #45205) that AutoTokenizer doesn't always pick up.
        with open(SNAP / "chat_template.jinja") as f:
            tokenizer.chat_template = f.read()
    return lm, tokenizer


def chat_ids(tokenizer, text):
    """Token ids for a single-turn user prompt via the repo chat template."""
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
    )
    # newer transformers may return a BatchEncoding
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


def greedy_generate(lm, tokenizer, ids, max_tokens=64):
    """Plain greedy decode loop (prefill + step) using lm.make_cache()."""
    cache = lm.make_cache()
    x = mx.array([ids])
    out = lm(inputs=x, cache=cache)
    tok = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
    generated = [tok]
    # -it generation_config: eos_token_id = [1 (<eos>), 106 (<turn|>),
    # 50 (<|tool_response>)]
    eos_ids = {1, 106, 50}
    for _ in range(max_tokens - 1):
        if tok in eos_ids:
            break
        out = lm(inputs=mx.array([[tok]]), cache=cache)
        tok = int(mx.argmax(out.logits[:, -1, :], axis=-1).item())
        generated.append(tok)
    return generated
