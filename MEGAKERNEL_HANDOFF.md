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

## Current status

| Step | State | Commit |
|------|-------|--------|
| Step 1: TK solver Impls | done | `369a6b618` |
| Step 2: scaffold (modules + types) | done | `a94d1cb17` |
| Step 2: emit_model hookup gated on FERRITE_KVM=1 | done | `b9ae21c04` |
| Step 2: vendor TK + cross-gpu-llama + Hopper NVCC flags | done | `4ea25be0d` |
| Step 2: NVCC compiles `tk_megakernel_<canonical>_launch` | done | `15ec71c23` |
| Encoder cleanup — total over IR, role-aware opcodes | done | `a0616fe7b` |
| TK cost ≈0 hack (MVP) | done | `f08655c0b` |
| Encode CutlassFusedAddRmsNormGemm + AttentionPrefillContiguous | done | `a8e13fef4` |
| `[build-deps] ferrite-models = { features=["cuda"] }` | done | `a841176a9` |
| Pre-create cudaforge megakernel cache dir | done | `65e74de1c` |
| **Re-enable `build_megakernels` (drop disabled `return;`)** | done | `6dc1e420a` |
| **XDG_CACHE_HOME path consistency macro↔build.rs** | done | `9a0ca2ecb` |
| **Always emit `-lmegakernels` from vllm-cuda/build.rs** | done | `b896434a5` |
| **Build pipeline green end-to-end on H100** | DONE | — |
| Smoke test (vllm chat or memory probe) | **HANGS** | — |

## Where we are

`cargo build -p vllm-cli --features cuda --release -j 2` on
H100 builds clean: proc-macro emits `.cu` files, NVCC compiles
them into `libmegakernels.a` with the expected
`tk_megakernel_<canonical>_launch` symbols, vllm-cli links
without errors. Verified `nm libmegakernels.a | grep
tk_megakernel` returns `T tk_megakernel_llama_3_2_3b_m_8_sk_128_launch`.

`vllm chat ...` HANGS on the memory-probe forward — the
probe runs a dummy forward at large batch (the prefill bucket),
which routes through `KVM_WRAPPERS` to `kvm_wrapper_m_8_sk_128`,
which calls `tk_megakernel_..._launch`. The megakernel
launches but never returns. Last log line before the hang:
```
INFO KvCachePool: 28 layers ...
```

## Most likely cause of the hang

**KV cache layout mismatch.** Vendor's `mk<>` template expects
a SINGLE contiguous slab indexed
`(layer × num_blocks × block_size × num_kv_heads × head_dim)`.
Ferrite's `KvCachePool` allocates per-layer separate buffers.

The runtime helper at
`crates/ferrite-forward/src/interpreter/kvm.rs::launch` passes
`ctx.kv_cache.k_cache(0).raw_ptr()` — layer-0's view — as
`d_k_cache` for ALL 28 layers. Layers 1..27 read garbage from
layer-0 memory, downstream barriers wait on writes that never
happen, kernel spins forever in cooperative-launch mode.

The prior ff-interpreter-mega worktree solved this at
`7c177562b: 6c-iii-a — contiguous KV cache alloc`. That fix
allocates a parallel kvm-only contiguous KV slab in the
forward path; the megakernel reads/writes it; on exit the
data is copied back into ferrite's per-layer cache. ~80 lines.

## TODO (priority order)

1. **Runtime escape hatch.** Add a `FERRITE_KVM_DISABLE=1` env
   var that, at runtime, makes `KVM_WRAPPERS[idx]` lookups
   return `None` regardless of static contents — fall through
   to the host interpreter. Use this to (a) confirm host path
   works on this build (vllm chat produces tokens), and (b)
   bisect any remaining issue between "kvm path" and "build
   itself."

2. **Contiguous kvm-only KV cache allocation.** Port the design
   from `7c177562b`. The runtime `launch()` helper allocates a
   fresh contiguous KV slab sized
   `(num_layers × num_blocks × block_size × num_kv_heads ×
   head_dim)`, copies layer-0..N-1 of `ctx.kv_cache` into it,
   passes the slab to the megakernel, then copies back. For
   the memory probe (which doesn't need real KV state) the
   copy-in can be skipped; only the size needs to be right.
   Once this lands, the hang should resolve and we should see
   a real forward pass complete (correct OR INCORRECT output —
   either is progress).

3. **Verify TK actually wins the solver.** The build output
   per-arch tag line should read:
   ```
   ferrite · llama-3.2-3b · ... · cublas cutlass non-gemm kvm
   ```
   With `kvm` at the end. If `kvm` is missing, the cost hack at
   `f08655c0b` isn't taking effect (proc-macro caching, env-var
   not propagating, etc.) and the wrappers won't be exercised
   even if the build links. The diag log
   `~/.cache/cudaforge/megakernels/_kvm_diag.log` shows
   per-canonical accept/reject decisions — `encoded
   tape_rows=N` means TK won and the bucket is kvm-routed.

4. **Replace the TK cost ≈0 hack with a real cost discount.**
   `f08655c0b` forces TK Impls to cost 0.001µs at the solver,
   which always-wins and isn't honest. The proper fix is a
   wave-parallel discount on `MegakernelFit::Kvm` Impls — they
   run inside a cooperative kernel that overlaps ops via wave
   scheduling, so their effective cost is some fraction of the
   isolated kernel cost. Probably ~0.5× initial guess.
   Re-evaluate after smoke test passes; cost-model fidelity
   matters less than correctness for MVP.

5. **AttentionPrefillContiguous placeholder.** Current encoder
   emits ONE row per (layer, kv_head) with seq_idx=0,
   block_idx=0, token_offset=0 — a dummy that lets the bucket
   be kvm-eligible. For real prefill the rows must be built
   per-call from `cu_seqlens_q`, by the runtime helper. Wire
   "tape splicing" in `launch()`: scan the static tape for
   sentinel rows, splice in real prefill rows, H2D the
   spliced tape. Required before claiming prefill correctness
   (multi-seq batched prefill currently produces wrong output).

6. **Smoke test on real prompt.** Once memory probe completes,
   run `vllm chat unsloth/Llama-3.2-3B-Instruct -q "Hello"`
   with `FERRITE_KVM=1`. Expected: tokens come out (correct or
   garbled). Garbled = next iteration. Coherent = milestone.

## What's wired (cleaned up)

* **Codegen side**
  (`ferrite-forward-macro/src/interpreter/kvm.rs`):
  - **Total over IR** (no `=> None` catch-all). Every
    OpInstance variant has an explicit eligibility decision in
    `variant_kvm_eligible` and a real `encode_op` arm if
    eligible.
  - **Role-aware opcodes**:
    `RmsNorm` weight path → `OP_ATTN_NORM` (1) /
    `OP_MLP_NORM` (6) / `OP_LM_HEAD_NORM` (10);
    `CutlassGemmAdd` weight path → `OP_O_PROJ_RESIDUAL` (5) /
    `OP_DOWN_PROJ_RESIDUAL` (9). Vendor's exact
    `OPCODE_*` numbers from
    `third_party/megakernels/cross-gpu-llama/llama.cuh`.
  - **By-name weight extraction** (not by ordinal). Wrapper
    walks the bucket looking for OpInstances whose weight
    accessor path contains `input_layernorm`,
    `post_attention_layernorm`, `model_norm`, `self_attn_o_proj`,
    `mlp_down_proj`, `lm_head`, etc.
  - **vocab_size on KvmDims** — vocab-block fanout = vocab_size
    / matmul_out_block_size, no hardcoded constants.
  - **Big WARNING comment block** at top of kvm.rs enumerating
    eight forbidden patterns (catch-all None, structural fudge,
    positional weight extraction, lying comments, etc.) so
    they don't creep back in.

* **Runtime side**
  (`ferrite-forward/src/interpreter/kvm.rs`):
  `WeightPtrs`, `RopePtrs`, `ShapeConfig`, `ExternLaunch`,
  `KvmWrapperFn<W>`, `launch()`. Allocates scratch from
  `device.caching`, H2Ds the static tape, builds the C arg
  pack, calls the per-canonical extern launcher.
  Module docstring flags the K/V-pointer-is-layer-0
  shortcoming explicitly.

* **emit_model hookup** (`codegen.rs`, gated on FERRITE_KVM=1):
  Per canonical, `bucket_kvm_eligible` walks both backbone +
  lm_head; if every variant is encodable, `encode_bucket`
  produces the tape, `emit_extern_decl` + `emit_wrapper_fn` +
  `emit_cu_source` + `write_cu_to_cache` produce the artifacts.
  Parallel `KVM_WRAPPERS` table aligned with `FORWARD_TABLE`
  routes the dispatcher.

* **Solver flips on cost**
  (`solver.rs`):
  Per-pick launch-overhead term (HostCallback +5µs,
  DeviceCallable 0µs). MVP cost hack: `MegakernelFit::Kvm`
  Impls return cost 0.001µs at the central evaluator (TODO 4).

* **Vendor parameterization patches**
  (`third_party/megakernels/cross-gpu-llama/`):
  - Every `LLAMA_*` `#define` is `#ifndef`-guarded so the
    proc-macro's per-arch overrides land first.
  - `globals_t::num_devices = LLAMA_NUM_DEVICES` (default 8)
    instead of hardcoded 8 → TP=1 picks `num_devices=1` →
    `kv_cache_t::r = num_kv_heads/num_devices >= 1`.
  - `gl_as_pgl<GL>` shim wraps every TK 1.x `pgl<>` typedef so
    ops can use `g.field[g.dev_idx]` without TK 2.x
    multicast at TP=1.
  - `qkv_rope_append.cu` / `attention_decode.cu` /
    `attention_prefill.cu` storer + assert generalizations for
    `num_kv_heads/num_devices > 1`.

* **NVCC build pipeline** (`ferrite-cuda-builder/build.rs`):
  - `build_megakernels()` scans
    `~/.cache/cudaforge/megakernels/` (XDG-aware) for `.cu`
    files emitted by the macro, compiles them into
    `libmegakernels.a`.
  - Include paths: `vllm-cuda/csrc`, vendored
    `third_party/megakernels/{cross-gpu-llama,include}`,
    `third_party/thunderkittens/include`.
  - `compute_cap(arch)` picks `sm_90a` for Hopper.
  - `KITTENS_HOPPER` / `KITTENS_BLACKWELL` defines mirror
    vendor's Makefile.
  - `ferrite-models` in `[build-dependencies]` with
    `features = ["cuda"]` enforces proc-macro expansion BEFORE
    build.rs runs.

* **Linking** (`vllm-cuda/build.rs`):
  Always emits `cargo:rustc-link-lib=static=megakernels`. The
  prior `if mk_lib.exists()` check ran at vllm-cuda build.rs
  time (parallel with `ferrite-cuda-builder/build.rs`) and
  silently dropped the directive when libmega.a hadn't been
  built yet — caused 2h of "defined-but-undefined-symbol"
  debugging.

## Build prerequisites

* CUDA 12.3+ (for `cuMulticast*`). System default `nvcc` is
  often 12.0; this worktree builds with
  `PATH=/usr/local/cuda-12.9/bin:$PATH`.
* Hopper GPU at runtime (sm_90a). The compiled megakernel
  contains wgmma/tcgen PTX with no sm_89 fallback — L4
  aborts with "Rust cannot catch foreign exceptions" on
  model load.

## Build invocation

```
cd vllm-rs
PATH=/usr/local/cuda-12.9/bin:$PATH \
FERRITE_KVM=1 FERRITE_MODELS=llama-3.2-3b \
  cargo build -p vllm-cli --features cuda --release -j 2
```

## Hard rules (LOAD-BEARING)

* Two structural changes only: TK Impls + KVM interpreter.
  No KvCachePool changes for everyone, no layered-weight
  changes for everyone. The contiguous kvm-only KV slab
  (TODO 2) MUST be a parallel allocation, not a modification
  of the host KvCachePool.
* The host interpreter (`crate::instr::run`) is the live path
  whenever `KVM_WRAPPERS[idx]` is `None`. KVM is a parallel
  playback strategy, not a replacement.
* No bespoke codegen arms or special-case macros.
* The kvm encoder is TOTAL over the IR: every variant has an
  explicit decision in `variant_kvm_eligible`. New IR variants
  fail the build until decided.
* No `=> None` catch-all in `encode_op`.
* No "structural fudge" arms that emit wrong opcodes for
  variants that need role tracking.
* No XDG-ignoring hardcoded `$HOME/.cache/...` paths in either
  the proc-macro or build.rs — both must use the same
  resolution logic.

## Pivot rejected

Earlier in the session I proposed pivoting to
`low-latency-llama` (a single-GPU, Llama-1B-fit vendor demo).
**Rejected** by the user — the cross-gpu-llama 8-GPU + 70B
hardcodings are parameterizations, not architectural
barriers. Vendored both demos under
`third_party/megakernels/`; only `cross-gpu-llama` is on the
path forward.

## Commits on this worktree

```
b896434a5 always emit `-lmegakernels` from vllm-cuda/build.rs
6dc1e420a re-enable build_megakernels (drop the early `return;`)
9a0ca2ecb fix XDG_CACHE_HOME path mismatch macro↔build.rs
65e74de1c pre-create megakernel cache dir
a841176a9 enable cuda feature for ferrite-models build-dep
81c5cfe1a ferrite-models as [build-dependencies] of cuda-builder
a8e13fef4 encode CutlassFusedAddRmsNormGemm + AttentionPrefillContiguous
f08655c0b force TK Impls to cost ≈0 (MVP hack)
a0616fe7b rip the bullshit from the kvm encoder + wrapper
4d4fa95e4 encoder arm for CutlassFusedAddRmsNormGemm + correct opcodes
c12c053df kvm_diag_log always-appends with PID tag
7be005cb6 log raw FERRITE_KVM/MODELS/GPU env vars
04e44b647 file-based kvm_diag (eprintln unreliable in proc-macros)
15ec71c23 NVCC compiles tk_megakernel_<canonical>_launch
4ea25be0d vendor TK sources + Hopper NVCC flags
b9ae21c04 emit_model hookup behind FERRITE_KVM=1
a94d1cb17 scaffold: kvm interpreter modules
369a6b618 step 1: Tk-tier solver Impls
```
