# mk_llama: vllm chat as TK megakernel — handoff

**Goal**: `vllm chat --model=<small-llama>` on H100 routes through ONE `__global__` kernel based on `~/Megakernels/demos/cross-gpu-llama/`. No kernel-library multi-launch path. No hand-written attention. Reuse TK's vendored op sources verbatim where possible.

**Why a handoff doc**: this is multi-session work. I'm committing to a decomposition so future sessions land into a clear next step instead of re-litigating direction. Each session = one bullet below; each ends with a green test or a green build of the next layer down.

## Where things stand right now

Last commit on worktree-ff2 that's relevant to this path:
- `cd99dd86b` — vendored 11 .cu/.cuh files from cross-gpu-llama into `crates/ferrite-stencil/csrc/mk_llama/`. Does NOT compile yet against current `~/ThunderKittens` because of a `pgl<>` template signature drift (cross-gpu-llama was written against an older TK).

Working in parallel (kittens kernel-library path; not the megakernel direction but proves the toolchain end-to-end):
- `e2d701f78` + earlier — `libkittens_kernels.a` builds on H100. `rmsnorm` and `gemm` smoke tests pass on H100 with TK primitives.
- `cef0f3d5a` — `kittens_ferrite_attn::attn_prefill_body` ported as `__device__` from TK's `fwd_attend_ker` (non-megakernel, but a real port).

The vendored mk_llama sources sit dormant — not built, not linked, not callable.

## Session-by-session plan

### Session 1 (THIS ONE if there's time, else next): get mk_llama compiling against current TK

Concrete deliverable: `cargo build -p ferrite-cuda-builder --features cuda` produces `~/.cache/cudaforge/vllm-cuda/libmk_llama.a` on H100.

Sub-steps:

1. **Adapt `llama.cuh` to single-device**:
   - Change `num_devices = 8` → `1`.
   - Replace every `kittens::pgl<GL, NUM_DEVICES, ...>` member with plain `kittens::gl<...>`. (`hidden_states`, `rms_rope_intermediates`, `rms_gate_intermediates`, `attn_out`, the `barriers Bar`.)
   - Remove `dev_idx` member (always 0).
   - Drop `LLAMA_BROADCAST_LM_HEAD_NORM`-conditional `pgl` use of `rms_lm_head_intermediates` — pick the non-broadcast branch.
2. **Find every op .cu reference to pgl features** and resolve:
   - `all_device_barrier.cu` — body becomes `__syncthreads();` no-op stub for single-device.
   - `inc_barriers.cu` — same; the `barrier_inc` op is a single-device no-op.
   - `attention_decode.cu`, `attention_prefill.cu`, `qkv_rope_append.cu`, `gate_silu.cu`, `up_matmul.cu`, `matmul_adds.cu`, `lm_head.cu`, `batched_rms_norm.cu` — anything calling `g.foo[dev_idx]` or `pgl::mc_ptr` or `tma::cluster::*` either drops the `[dev_idx]` (single device) or guards out the multicast paths.
   - `matmul_pipeline.cuh` — same.
3. **Write `mk_llama_single.cu` top-level**:
   - Include all op .cu's (with kittens.cuh + megakernel.cuh up front).
   - Declare an `ops` typedef list.
   - Pick a small Llama variant (Llama-3.2-1B: 16 layers, hidden=2048, intermediate=8192, 32 q heads, 8 kv heads, head_dim=64) — the demo's `LLAMA_*` defines need a smaller-than-70B parallel set (`LLAMA_3_2_1B_*`).
   - Expose `extern "C" cudaError_t launch_mk_llama(...)`. Internally constructs `globals_t<...>` from raw pointers, sets `cudaFuncAttribute(MaxDynamicSharedMemorySize, ...)`, launches `mk<config, globals, ops...><<<grid, block, dyn_shm, stream>>>(g)`.
4. **Add to `ferrite-cuda-builder`'s build.rs**: a `build_mk_llama` function that compiles the single-device sources into `libmk_llama.a`. Same flags as kittens build (gencode=compute_90a + std=c++20 + extended-lambda + relaxed-constexpr) plus `-I ~/Megakernels/include`.

**Ends when**: `libmk_llama.a` exists and `nm` shows `launch_mk_llama` symbol. No correctness yet — just builds.

### Session 2: instruction buffer schema + dummy launch

1. Read `~/Megakernels/megakernels/demos/throughput/scheduler.py` carefully. Each instruction is `int[INSTRUCTION_WIDTH=32]` with `[opcode, arg0, arg1, ...]` layout per op. Document the layout per op (attn_norm, qkv_rope_append, attention_*, o_proj, mlp_norm, gate_silu, up_matmul, downproj, lm_head_norm, lm_head, barrier_inc, all_device_barrier).
2. Rust port: a `crates/ferrite-mk-llama/` (or a module under ferrite-stencil-kernels) that builds a `Vec<[i32; 32]>` instruction buffer for a given `(num_layers, num_tokens, sk_bucket)` configuration. Mirror the throughput Python's `make_schedule` logic — that's where the order of (PreAttnLayerNorm, QKV_MatMulRopeAppend, AttentionPrefill OR AttentionDecode, O_ProjResidual, PreMLP_Norm, GateSilu, UpMatmul, DownProjResidual, ...) per layer + (PreLMHeadRMS, LM_Head) per output gets baked.
3. Rust FFI for `launch_mk_llama` + a Rust integration test that:
   - cudaMallocs zeroed weights + activations + KV cache.
   - Builds an instruction buffer for 2 tokens, 16 layers, batch=1.
   - Calls `launch_mk_llama`.
   - cudaDeviceSynchronizes; checks `cudaSuccess`.
   - Doesn't validate output (zeros in → garbage out is fine).

**Ends when**: the integration test runs on H100 without crashing.

### Session 3: real weights + correctness vs PyTorch reference

1. Add a weight-loader that pulls a real Llama-3.2-1B safetensors checkpoint and lays the weights out per `globals_t::weights_t` etc. (TK uses specific pre-permuted formats; reverse-engineer from `~/Megakernels/megakernels/llama.py`'s `convert_weights`).
2. Run a single forward (`num_tokens = 8`, batch=1) on a known input.
3. Compare output `logits` against `transformers`-loaded Llama-3.2-1B running the same input. Need argmax-equal at minimum, top-k overlap better.

**Ends when**: a Rust test loads real weights, runs the megakernel, and matches HF logits within tolerance for a single prefill step.

### Session 4: vllm-cuda integration

1. Add a `KittensMegakernelForward` path in `cuda_worker.rs`. Routes only when:
   - cc >= 90a.
   - Model arch is Llama (matched by name).
   - `KITTENS_MEGAKERNEL=1` env var set (initial gating).
2. Plug into vllm's standard `ModelRunner` interface — same input shape (input_ids, positions, slot_mapping, kv_cache, …), same output shape (logits).
3. Memory plan: where `globals_t`'s activation buffers live (per-stream scratch?), how KV cache pointers thread through.

**Ends when**: `KITTENS_MEGAKERNEL=1 vllm chat --model=meta-llama/Llama-3.2-1B-Instruct` runs without crashing.

### Session 5: correctness + ship

1. Run vllm chat with the env var, validate coherent output.
2. Measure latency vs the existing path.
3. Decide whether to enable by default.

**Ends when**: chat output is coherent.

## Decomposition rationale

Each session has a binary outcome: either the step ends with a green build/test, or it doesn't and the next session restarts there with the failure mode known. No "I'll keep going" — explicit handoffs.

Total: 5 sessions of focused work to get to coherent vllm chat through TK megakernel on H100. This is the realistic timeline I should have given in session 1 instead of repeatedly promising and failing.

## What I will NOT do across these sessions

- Hand-write any CUDA op (use TK / cross-gpu-llama verbatim wherever possible).
- Reinvent the megakernel composition (`mk<>` from megakernel.cuh is the entry point).
- Run a multi-launch kernel-library as the megakernel path (kittens kernel-library exists already as a separate exploration; it's not on the megakernel critical path).
- Pivot direction without the user explicitly requesting it.
