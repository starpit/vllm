#!/usr/bin/env python3
"""Launch `mlx_lm.server` on the Gemma4 unified checkpoint.

mlx-lm HEAD ships the `gemma4` text wrapper for unified checkpoints
but (a) lacks the `gemma4_unified` MODEL_REMAPPING entry and (b) its
sanitize predates this checkpoint's `vision_embedder.*` tower naming
(same drift mlx-vlm had). Two-line patch, then exec the stock server.

Run: cd ~/git/mlx-lm && uv run python \
    <worktree>/vllm-rs/scripts/gemma4/mlx_lm_server_gemma4.py \
    --model mlx-community/gemma-4-12B-it-4bit --port 8390
"""
import sys

import mlx_lm.utils as u

u.MODEL_REMAPPING["gemma4_unified"] = "gemma4"

from mlx_lm.models import gemma4  # noqa: E402

_orig_sanitize = gemma4.Model.sanitize


def _sanitize(self, weights):
    weights = {
        k: v
        for k, v in weights.items()
        if not k.removeprefix("model.").startswith("vision_embedder")
    }
    return _orig_sanitize(self, weights)


gemma4.Model.sanitize = _sanitize

from mlx_lm.server import main  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "mlx_lm.server"
    main()
