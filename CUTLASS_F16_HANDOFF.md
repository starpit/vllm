# CUTLASS F16 templatize — DONE

**Tip:** worktree-multimodal post-`68b8c2bda` (uncommitted). Goal achieved: `cutlass::bfloat16_t` no longer hardcoded — every CUTLASS GEMM macro emits paired `_bf16_launch` / `_f16_launch` symbols. F16 arches (LLaVA-1.5 / Vicuna-7B) keep CUTLASS perf instead of falling through to cuBLAS.

**Files touched.**
- `vllm-rs/crates/vllm-cuda/csrc/cutlass_standalone_gemm.cu` — every `*_CONFIG` / `*_LAUNCH` macro takes `(DTYPE_TAG, T)` and emits dtype-suffixed symbols. Public `CUTLASS_GEMM(...)` etc. expand to both bf16+f16. `run_gemm` / `run_gemm_splitk` deduce T via `typename GemmOp::ElementA`. GEMV templated. `namespace sm90_bf16` → `namespace sm90` with both dtype paths.
- `vllm-rs/crates/vllm-cuda/csrc/cutlass_gemm_bias.cu` — same pattern.
- `vllm-rs/crates/vllm-cuda/csrc/cutlass_gemm_silu_mul.cu` — `dispatch_silu_mul_gemm_t<T>` template, two extern launches.
- `vllm-rs/crates/ferrite-kernels/src/cutlass.rs` — extern decls compacted via `gemm_pair!` / `splitk_pair!` / `bias_pair!` Rust macros (~70 fewer lines). `launch_fn_for(tile, dtype)` etc. return the matching fn pointer per dtype. Safe wrappers read `a.dtype()` and pass through.
- `vllm-rs/crates/ferrite-forward-macro/src/impl_lib.rs` — `starter_library_for(bf16_only_kernels)` collapsed to single `starter_library()`. The `bf16_only_kernels` parameter dropped — CUTLASS Impls register unconditionally now that F16 launchers exist.
- `vllm-rs/crates/ferrite-forward-macro/src/lib.rs` — call site simplified.
- `vllm-rs/crates/ferrite-cost-sweep/src/gemm_sweep.rs` — sed-rename `_launch` → `_bf16_launch` (sweep is BF16-only by design).

**Validation (2026-05-07).** `cargo test -p vllm-e2e --features e2e,cuda` on each MM suite:

| Suite | Result |
|---|---|
| `e_llava` | 4 passed; 0 failed |
| `e_qwen2_vl` (skip tp2) | 9 passed; 0 failed |
| `e_qwen2_5_vl` | 8 passed; 0 failed |
| `e_gemma3_mm` | 4 passed; 0 failed |

Total: 25/25. Cost CSV unchanged — F16 / BF16 share the HMMA path on Ada/Hopper, same shape ⇒ same microseconds.

**Pitfall hit, recorded for future sessions.** Cargo proc-macro caching is BLIND to env var changes. Building once with `FERRITE_MODELS=llava` cached an empty `ferrite-model-qwen2-vl` (proc-macro saw the filter and emitted `quote! {}`). Subsequent `cargo test` runs without the env var did NOT re-expand — `FerriteMmRegistration` for Qwen2-VL was missing from the inventory, the server's MM resolver returned None, the vision encoder was silently bypassed for ALL Qwen2-VL / Qwen2.5-VL / Gemma3-MM requests. Symptom looked like a BF16 regression. The cure is a `find vllm-rs/crates -name lib.rs -path '*ferrite-model*' -exec touch {} \;` before re-test; no `cargo:rerun-if-env-changed=FERRITE_MODELS` exists in the workspace.

**Next-session followups (perf, not correctness):**
1. Bench LLaVA-1.5-7B with the new CUTLASS launchers to measure the actual delta vs the prior cuBLAS-fallback baseline. Expected ~5–15 % on fused-EVT epilogues.
2. Consider declaring `cargo:rerun-if-env-changed=FERRITE_MODELS` in `ferrite-cuda-builder/build.rs` or wherever cargo can see it, so the proc-macro env trap stops biting future Claudes.
