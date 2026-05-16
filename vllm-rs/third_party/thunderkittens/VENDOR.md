# Vendor provenance

Source: https://github.com/HazyResearch/ThunderKittens (local clone `~/git/ThunderKittens`)
Upstream commit: `4b0aa30da67ba4466c2079695183f428fd6ce0bf` (2026-04-29, "Add more base ops")

Vendored via rsync of the `include/` tree only, excluding `.claude/`
and `tests/batch-vm/llama_sm89/` (local additions, not upstream).

Clean vendor drop. No patches applied.

## History

- 2026-05-04: Bumped from `cce72c2f5c71c3ab812f27f96d6289e412baed60` →
  `4b0aa30da67ba4466c2079695183f428fd6ce0bf`. New TK tree collapses
  `types/device/pgl.cuh` (pre-TK-2.0 4-bool template) into
  `types/system/pgl.cuh` (`pgl<GL, NUM_DEVICES, MULTICAST, TMA_Types...>`
  — simpler 3-arg template + pack). Vendor `cross-gpu-llama/llama.cuh`
  and ferrite codegen launcher both have to adopt the new signature.
