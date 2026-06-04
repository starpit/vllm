# vision_parity — P-1 oracle harness for VL-on-ferrite-metal

`dump_golden.py` runs the real Qwen3.5-VL-9B vision tower via **mlx-vlm**
(natively on the Mac, no torch) and dumps per-stage tensors as golden
fixtures. The ferrite-metal vision kernels (P1+) are checked cosine>0.99
against these.

Setup: `uv pip install -e ~/git/mlx-vlm --python ~/.venv/bin/python`
Run:   `~/.venv/bin/python vllm-rs/tools/vision_parity/dump_golden.py`
Out:   `golden/*.npy` (gitignored, regenerable) + `manifest.txt`

See `ORACLE.md` for the pinned dims/layouts and `../../VL_METAL_PLAN.md` for the plan.
