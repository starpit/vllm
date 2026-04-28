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
| 2c. Solver picks TK | done | `36b3c7d55` |
| 2c'. Field extraction + FusedAddRmsNorm encoding | done | `b361bc9fe` |
| 2d. Vendor TK sources + Hopper NVCC flags | done | `4ea25be0d` |
| 2d. NVCC compiles tk_megakernel_<canonical>_launch | done | `15ec71c23` |
| 2e. Smoke test on H100 | blocked on hardware | — |

## What's wired (end-to-end)

1. **Codegen side** (`ferrite-forward-macro/src/interpreter/kvm.rs`):
   * Walks the post-loop-compression `LoweredBucket` and encodes
     each `OpInstance` into vendor's i32 tape format.
   * For each kvm-eligible canonical, emits:
     - per-bucket `[[i32; 32]; N]` tape static;
     - per-canonical `extern "C" fn tk_megakernel_<canonical>_launch`
       declaration matching the C symbol;
     - per-canonical Rust marshaling wrapper that walks the
       lowered IR for weight accessors (looked up by FIELD NAME
       in the OpcodeShape registry, not by hardcoded index) and
       calls `interpreter::kvm::launch`;
     - per-canonical `.cu` source written into
       `~/.cache/cudaforge/megakernels/`. The .cu defines per-
       arch `LLAMA_*` macros, includes vendor's
       `cross-gpu-llama` headers + ops, and aggregate-inits
       `llama_70b_globals` followed by chevron-launch of
       `mk<llama_config, llama_70b_globals, op...>`.

2. **Runtime side** (`ferrite-forward/src/interpreter/kvm.rs`):
   * `WeightPtrs`, `RopePtrs`, `ShapeConfig`, `ExternLaunch`,
     `KvmWrapperFn<W>` types.
   * `launch()` allocates scratch from `device.caching`, H2Ds
     the static tape, builds the C arg pack from the wrapper's
     extracted accessors, and calls the per-canonical extern
     launcher. Returns `(hidden, logits)`.

3. **emit_model hookup** (gated on `FERRITE_KVM=1`):
   * Per canonical, checks if any solver-picked Impl in the
     SFUF has `MegakernelFit::Kvm`; encodes the bucket; if
     `encode_bucket` succeeds, emits the artifacts and registers
     the wrapper in `KVM_WRAPPERS` aligned with `FORWARD_TABLE`.
   * `forward()` consults `KVM_WRAPPERS[idx]`; if `Some`, calls
     the wrapper and `take_owned`s; if `None`, falls through to
     `ferrite_forward::run`.

4. **Solver flips on cost** (`solver.rs`):
   * Per-pick launch-overhead term (HostCallback +5µs,
     DeviceCallable 0µs) makes TK Impls win on cost.

5. **Vendor parameterization patches** (`third_party/megakernels/cross-gpu-llama/`):
   * Every `LLAMA_*` `#define` is `#ifndef`-guarded so the
     proc-macro's per-arch overrides land first.
   * `globals_t::num_devices` reads `LLAMA_NUM_DEVICES`
     (default 8) instead of being hardcoded 8 — TP=1 picks
     `num_devices=1`. `kv_cache_t::r = num_kv_heads/num_devices`
     becomes a positive integer for models with `num_kv_heads<8`.
   * `gl_as_pgl<GL>` shim wraps every TK 1.x `pgl<>` typedef so
     the ops can use `g.field[g.dev_idx]` syntax without TK 2.x
     multicast at TP=1.
   * `qkv_rope_append.cu` / `attention_decode.cu` /
     `attention_prefill.cu` storer + assert generalizations for
     `num_kv_heads/num_devices > 1`.

6. **NVCC build pipeline** (`ferrite-cuda-builder/build.rs`):
   * `build_megakernels()` scans
     `~/.cache/cudaforge/megakernels/` for `.cu` files emitted
     by the macro, compiles them into `libmegakernels.a`.
   * Include paths cover vendored thunderkittens + megakernels
     framework + cross-gpu-llama vendor sources.
   * `compute_cap(arch)` picks `sm_90a` for Hopper (cudaforge
     auto-suffixes `a` for sm_90+).
   * `KITTENS_HOPPER` / `KITTENS_BLACKWELL` defines mirror
     vendor's Makefile.

## Verified

```
$ rm -f ~/.cache/cudaforge/vllm-cuda/libmegakernels*
$ PATH=/usr/local/cuda-12.9/bin:$PATH \
  FERRITE_KVM=1 FERRITE_GPU=h100 FERRITE_MODELS=llama-3.2-3b CUDA_ARCH=90 \
  cargo build -p vllm-cli --features cuda --release -j 2
… 2m38s …
    Finished `release` profile [optimized] target(s) in 2m 38s

$ nm ~/.cache/cudaforge/vllm-cuda/libmegakernels.a \
    | grep "T tk_megakernel.*_launch$"
0000000000004d80 T tk_megakernel_llama_3_2_3b_m_8_sk_128_launch

$ ./target/release/vllm ferrite info | head -5
══ llama / llama-3.2-3b · tp=1 ══
…
```

## What's blocking the smoke test — hardware

```
$ nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
NVIDIA L4, 8.9
```

Vendor's megakernel relies on Hopper-specific PTX (wgmma,
tcgen05, dynamic cluster dims, multicast TMA). KITTENS_HOPPER
gates these. The compiled `libmegakernels.a` contains
`sm_90a` PTX; loading + executing it on the L4 (sm_89) raises
a CUDA driver exception that crosses the FFI boundary into
Rust as "fatal runtime error: Rust cannot catch foreign
exceptions, aborting." `vllm ferrite info` works because it
doesn't load the model; `vllm chat … -q "Hi"` aborts during
model load / first forward.

Smoke test on Hopper:

```
ulimit -v 24000000
PATH=/usr/local/cuda-12.9/bin:$PATH \
FERRITE_KVM=1 FERRITE_GPU=h100 FERRITE_MODELS=llama-3.2-3b CUDA_ARCH=90 \
cargo build -p vllm-cli --features cuda --release -j 2
./target/release/vllm chat unsloth/Llama-3.2-3B-Instruct -q "Hello"
```

## Build prerequisites

* CUDA 12.3+ (multicast — `cuMulticastBindMem` etc.). System
  default `nvcc` is 12.0; this worktree builds with
  `PATH=/usr/local/cuda-12.9/bin:$PATH`.
* Hopper GPU at runtime (sm_90a).
* `unsloth/Llama-3.2-3B-Instruct` HF model.

## Commits on this worktree

* `369a6b618` — step 1: Tk-tier solver Impls.
* `a94d1cb17` — step 2 (scaffold): kvm interpreter modules.
* `b9ae21c04` — step 2: emit_model hookup behind FERRITE_KVM=1.
* `b334d1749` — handoff doc (initial).
* `36b3c7d55` — step 2c: launch-overhead term + tk_ kernel-class.
* `fb8dae8bf` — step 2c: relax kvm-eligibility to any_kvm.
* `b361bc9fe` — step 2c: shape-aware field extraction +
  FusedAddRmsNorm encoding + diag.
* `298837b12` — handoff refresh (m=8 .cu file written).
* `2cc5bf18d` — handoff refresh (build.rs scan identified).
* `4ea25be0d` — vendor cross-gpu-llama + ThunderKittens +
  Hopper NVCC flags.
* `e035c5999` — vendor low-latency-llama (NOT used; pivot
  rejected — see "Pivot rejected" below).
* `983eb6f65` — handoff that proposed the pivot (superseded).
* `15ec71c23` — NVCC compiles tk_megakernel_<canonical>_launch.
  cross-gpu-llama vendor patches (LLAMA_NUM_DEVICES + gl_as_pgl
  + #ifndef-guarded macros) ported from prior ff-mega worktree;
  `emit_cu_source` ported 1:1 (chevron-launch, not
  cudaLaunchCooperativeKernel; correct globals_t aggregate-init
  field order). vllm-cli release binary builds clean.

## Pivot rejected

An earlier note in this handoff proposed pivoting to
`low-latency-llama` (a single-GPU, Llama-1B-fit vendor demo)
because the cross-gpu-llama 8-GPU + 70B hardcoding looked
unworkable. **Rejected.** The 8-GPU and 70B hardcoding are
**parameterizations**, not architectural barriers. The prior
ff-interpreter-mega worktree resolved both via small vendor
patches (preserved in 15ec71c23). low-latency-llama remains
vendored under `third_party/megakernels/low-latency-llama/`
but is not on the path forward.

## Hard rules (LOAD-BEARING)

* The two steps above are the **only** structural changes to
  the macro pipeline.
* The host interpreter (`crate::instr::run`) is the live path
  whenever `KVM_WRAPPERS[idx]` is `None`. KVM is **not** a
  replacement; it's a parallel playback strategy on top of the
  same lowered bucket.
* Do NOT add bespoke codegen arms or special-case macros.
* No pivot. Resolve blockers in cross-gpu-llama directly.
