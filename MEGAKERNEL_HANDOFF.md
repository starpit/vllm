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
| 2c. Solver picks TK (launch overhead + class) | done | `36b3c7d55` |
| 2c'. Field extraction + FusedAddRmsNorm encoding | done | `b361bc9fe` |
| 2d. Smoke test (NVCC + chat) | NOT run | — |

llama-3.2-3b m=8 prefill bucket now produces
`~/.cache/cudaforge/megakernels/tk_megakernel_llama_3_2_3b_m_8_sk_128.cu`
+ a `kvm_wrapper_m_8_sk_128` fn + a populated `KVM_WRAPPERS`
slot. The dispatcher routes m=8 prefill calls through the
wrapper.

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

**Smoke test not run yet.** The .cu file is written, the
wrapper fn is emitted, KVM_WRAPPERS is populated for m=8 —
but ferrite-cuda-builder/build.rs needs to pick up the new
.cu file from `~/.cache/cudaforge/megakernels/` and feed it
to NVCC. The runtime kvm wrapper expects an
`extern "C" fn tk_megakernel_<canonical>_launch` symbol; that
only exists once NVCC compiles the .cu.

Three remaining items to validate:

1. ferrite-cuda-builder picks up .cu files from
   ~/.cache/cudaforge/megakernels/. If it doesn't already,
   that's the next code change.
2. NVCC compiles vendor's mk<...> template with the bounds
   `dims_from_bounds` baked in. Vendor's static_asserts
   must pass. Llama-3.2-3b's hidden=3072 is a multiple of
   256, intermediate=8192/1=8192 is a multiple of 256, so
   the asserts should pass.
3. The runtime helper's KV-cache layout assumption (single
   contiguous `(layer × num_blocks × …)` slab) doesn't match
   ferrite's per-layer KvCachePool. For a single-layer
   smoke run this is fine (we pass the layer-0 view). For
   multi-layer correctness, a parallel kvm-only KV pool is
   needed.

**Decode bucket (m=1) is still kvm-INELIGIBLE.**

A clean build with `FERRITE_KVM=1 FERRITE_GPU=h100
FERRITE_MODELS=llama-3.2-3b cargo build -p ferrite-model-llama
--features cuda` finishes successfully but the per-arch line
prints `… fa2 cublas cutlass non-gemm` — no `kvm` tag.
`kvm_wrapper_idents` ends up empty for every canonical, so
`KVM_WRAPPERS` is not emitted and `forward()` keeps the original
`find_bucket + run` body.

The decode bucket m=1 has `cutlass_gemv` picked for o_proj
(there's no TK decode-matmul_add — `tk_cutlass_32x64_s4_add`
only matches at prefill). The encoder rejects CutlassGemv
inside the body because the megakernel has no structural
opcode for it; only the lm_head's terminal CutlassGemv is
encodable (as OP_LM_HEAD). Until a TK decode-matmul_add
exists OR encode_op gains a body-CutlassGemv arm, m=1 falls
through to the host interpreter.

### Next concrete step (2d)

Run the smoke test:

```
ulimit -v 16777216
FERRITE_KVM=1 FERRITE_GPU=h100 FERRITE_MODELS=llama-3.2-3b \
  cargo build -p vllm-cli --features cuda --release -j 2
```

Expected outcomes (in order of likely failure):

1. **NVCC link-step failure** — `tk_megakernel_<...>_launch`
   undefined. Means ferrite-cuda-builder/build.rs isn't
   scanning ~/.cache/cudaforge/megakernels/. Fix: add the
   dir to its scan list.
2. **NVCC compile failure** on the .cu — vendor static_assert
   trip. Inspect; either fix dims_from_bounds or relax the
   assert.
3. **Build succeeds, runtime panic in launch()** —
   KvCachePool layout doesn't match. For a 1-layer dummy,
   pass; for real llama, need the parallel kvm-only KV pool.

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
