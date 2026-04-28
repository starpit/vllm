# tk-mvp megakernel handoff

Worktree: `/home/moosevan/vllm/.claude/worktrees/tk-mvp/`
Branch: `worktree-tk-mvp` (forked off `ff-interpreter` at `a62f1bb5f`).

## North star

Two steps, no sub-steps:

1. **Solver Impls for the TK kernels** — normal Implementations
   the solver picks naturally based on cost.
2. **A host-side "KVM interpreter"** — encodes the lowered
   instruction sequence as a tape, H2Ds it, invokes vendor's
   `mk` megakernel which plays the tape on GPU.

Both interpreters consume the same lowered bucket: the host
interpreter's per-op kernel-launch loop (`ferrite_forward::run`)
is one playback strategy, the GPU megakernel is another.

## Status

| Step | State | Commit |
|------|-------|--------|
| 1. Solver Impls | done | `369a6b618` |
| 2a. Scaffold (mod + types) | done | `a94d1cb17` |
| 2b. emit_model hookup (FERRITE_KVM gated) | done | `b9ae21c04` |
| 2c. Solver actually picks TK | NOT done | — |
| 2d. Smoke test | NOT run | — |

## What's wired

* **Codegen side** —
  `ferrite-forward-macro/src/interpreter/{mod.rs,kvm.rs}`.
  Encodes a `LoweredBucket` into a `[[i32; 32]; N]` tape, emits
  per-canonical extern decl + Rust marshaling wrapper fn, writes
  per-canonical `.cu` source (vendor's `mk<llama_config,
  llama_70b_globals, ops…>` template instantiation +
  `cudaLaunchCooperativeKernel` body that aggregate-inits
  `globals_t`) into `~/.cache/cudaforge/megakernels/`.

* **Runtime side** —
  `ferrite-forward/src/interpreter/{mod.rs,kvm.rs}`.
  `WeightPtrs`, `RopePtrs`, `ShapeConfig`, `ExternLaunch`,
  `KvmWrapperFn<W>`, `launch()`. Allocates scratch from
  `device.caching`, H2Ds the static tape, calls the per-canonical
  extern launcher, returns `(hidden, logits)`.

* **emit_model hookup** — gated on `FERRITE_KVM=1` env var at
  macro expansion time. For each canonical whose every solver-
  picked Impl has `MegakernelFit::Kvm`, emits the artifacts and
  registers the wrapper fn in a parallel `KVM_WRAPPERS` table
  aligned with `FORWARD_TABLE`. `forward()` consults
  `KVM_WRAPPERS[idx]` first; falls through to `ferrite_forward::
  run` when it's `None`.

## What's blocking

**The solver isn't picking the TK Impls.**

A clean build with `FERRITE_KVM=1 FERRITE_GPU=h100
FERRITE_MODELS=llama-3.2-3b cargo build -p ferrite-model-llama
--features cuda` finishes successfully but the per-arch line
prints `… fa2 cublas cutlass non-gemm` — no `kvm` tag.
`kvm_wrapper_idents` ends up empty for every canonical, so
`KVM_WRAPPERS` is not emitted and `forward()` keeps the original
`find_bucket + run` body.

### Root cause (suspected)

Each Tk Impl in `impl_lib.rs:starter_library()` overrides
`launch_kind` to `LaunchKind::DeviceCallable` and `megakernel_fit`
to `MegakernelFit::Kvm`, but inherits `cost_us` from its host
counterpart. The solver's DP doesn't apply any
DeviceCallable-specific discount — `grep "launch_overhead\|
device_callable" solver.rs` returns no hits. So Tk and host
costs are identical, the DP picks the first registered one (the
host Impl, which is registered earlier), and Tk never wins.

The comment in `impl_lib.rs:2090-2093` claims the savings live
in "the per-pick `launch_overhead_us` term" — which doesn't
exist in `solver.rs`. The wiring is missing.

### Next concrete step (2c)

Add a per-pick launch-overhead term in the solver's cost
evaluation: `LaunchKind::HostCallback` adds ~5 µs, `DeviceCallable`
adds 0 µs. Two-line patch in solver.rs's per-Impl cost loop;
matches the design comment in `impl_lib.rs:2090-2093`. **Do
NOT** lower individual TK Impls' `cost_us` — that's the
kind of special-case hack the user has explicitly forbidden
("DP solver — no hand-coded fusion preferences"). The whole
point is that DeviceCallable wins **principledly** because
launch overhead is real and the cost model finally accounts
for it.

After that, rebuild llama-3.2-3b with FERRITE_KVM=1, expect
the per-arch line to show `kvm` in the tag list, and run the
smoke test (2d).

## Smoke test (2d)

Once the solver picks TK Impls, build the full vllm-cli:

```
ulimit -v 16777216
FERRITE_KVM=1 FERRITE_GPU=h100 FERRITE_MODELS=llama-3.2-3b \
  cargo build -p vllm-cli --features cuda --release -j 2
```

Then run vllm chat on llama-3.2-3b with a short prompt. The
ferrite path should compile in the wrapper fn and the
megakernel from `~/.cache/cudaforge/megakernels/
tk_megakernel_llama_3_2_3b_m_*_sk_*.cu`. Expected first failure
mode: KV-cache layout mismatch — vendor expects a single
contiguous `(layer × num_blocks × block_size × num_kv_heads ×
head_dim)` slab, ferrite's `KvCachePool` allocates per-layer.
The runtime helper currently passes the layer-0 view as the
K/V pointer (see `interpreter/kvm.rs` module docstring); for
multi-layer models that's wrong. Wiring a parallel kvm-only
KV pool is the next blocker after 2c.

## Commits on this worktree

* `369a6b618` — step 1: Tk-tier solver Impls.
* `a94d1cb17` — step 2 (scaffold): kvm interpreter modules.
* `b9ae21c04` — step 2: emit_model hookup behind FERRITE_KVM=1.

## Hard rules (LOAD-BEARING)

* The two steps above are the **only** structural changes to
  the macro pipeline. Everything else (KvCachePool layout,
  layered weights, `interpreter_codegen.rs` body) is off-limits
  except to make routines `pub`.
* The host interpreter (`crate::instr::run`) is the live path
  whenever `KVM_WRAPPERS[idx]` is `None`. KVM is **not** a
  replacement; it's a parallel playback strategy on top of the
  same lowered bucket.
* Do NOT add bespoke codegen arms or special-case macros.
  Every TK pick flows through the same `Implementation` trait
  every other Impl uses.
