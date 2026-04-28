# mega — handoff

> Worktree `ff-interpreter-mega`, branch `feat/rust`. This file is
> the source of truth for the megakernel work. Read it before
> changing anything. The host-interpreter pivot
> (`HANDOFF_INTERPRETER.md`) is the prerequisite and is fold-ready;
> mega builds on top of its `Instruction<W>` IR + `static
> BACKBONE_M_<N>: &[Instruction<W>]` slices without modification.

## STATE 2026-04-28 (late) — P2-4b 6c-i…6c-iii-d-1 done; 6c-iii-d-2 (codegen wrapper + flip) is next

Tip `0aa7c4fd8`. Five commits past `e2f9b1eaf`:

- **`08ac9b98c`** — 6c-i: replace per-bucket `KVM_BACKBONE_M_<wp>`
  + `KVM_LM_HEAD_M_<wp>` tape statics with one concatenated
  `KVM_FULL_M_<wp>` (backbone rows then lm_head rows) plus
  `_BACKBONE_LEN` / `_LM_HEAD_LEN` length consts. Pure codegen
  refactor; same 107 .cu files re-emitted. New unit test
  `emit_kvm_full_program_concatenates_with_split_lens` passes;
  total macro suite now 251/255 (4 pre-existing fails).
- **`0dd0e0ce0`** — 6c-ii: `paged_kv` module in
  `ferrite-forward` with the pure host-side CSR builder
  `build_paged_kv_metadata_from_host`. Seven unit tests cover
  prefill / decode / mixed / panic-ordering edges. Algorithm
  ported from vendor `tp_generate.py` + `test_prefill.py`.
- **`7c177562b`** — 6c-iii-a: `KvCachePool::new` now does ONE
  `mem_alloc` per K and per V (`bytes_per_layer * num_layers`)
  with per-layer `GpuTensor` views. Per-layer `mem_alloc` was
  incompatible with vendor's
  `kv_cache_t = gl<bf16, -1, -1, num_kv_heads, head_dim>` which
  strides across the layer dim. Two new accessors
  `k_cache_base_ptr()` / `v_cache_base_ptr()`.
- **`395d869a5`** — 6c-iii-b: layered loaders
  (`load_layered_rms_norm`, `load_layered_linear_dense`,
  `load_layered_linear_dense_concat`) now allocate ONE
  `[L, R, H]` block per accessor and `take_into` each layer.
  Falls back to per-layer load on packed-source synthesis
  (Phi-3), bias presence, or non-uniform shape — none of which
  are kvm-eligible anyway. Same Vec<…> API to callers.
  `ferrite-model-llama` (fast path) and `ferrite-model-phi3`
  (fallback path) both build clean.
- **`cde7fe8c7`** — 6c-iii-c: cuda-gated
  `paged_kv::device::build_paged_kv_metadata_on_device` —
  D2H/H2D wrapper around the 6c-ii host helper. Returns
  `DevicePagedKv` with nine owned int32 tensors + six scalar
  C-arg ints. ~10 KB of round-trip traffic per forward call.
- **`0aa7c4fd8`** — 6c-iii-d-1: `kvm_mega::launch_kvm_mega`
  runtime helper in `ferrite-forward`. Takes the per-canonical
  extern launcher fn pointer, the bucket's `KVM_FULL_M_<wp>`
  tape, three arg structs (`KvmWeightPtrs`, `KvmRopePtrs`,
  `KvmShapeConfig`); allocates Bar / timings / scratch via
  caching allocator, H2Ds the tape, calls
  `build_paged_kv_metadata_on_device`, fires the extern
  launcher with the full ~50-arg list. `stream_synchronize`s at
  the end so OwnedTensor scratch lifetimes outlive the kernel
  (caching allocator is not stream-aware). No callers yet.

  Working-tree dirty (uncommitted): `launch_kvm_mega` signature
  changed to return `(OwnedTensor, OwnedTensor)` for
  (hidden_states, logits) so the codegen wrapper from 6c-iii-d-2
  can place them into the dispatcher's `tiles` slots. Will land
  bundled with the wrapper commit.

251/255 tests pass on `cargo test -p ferrite-forward-macro
--release`; the 4 failures are pre-existing on HEAD
(`dc_zoo_equals_host_zoo`, `dc_cutlass_*_name_covers_full_zoo`,
`swiglu_mlp_claimed_as_fused_subgraph_per_layer`) and unrelated
to mega.

### Step 6c-iii-d-2 — per-canonical wrapper codegen + LAUNCHER_TABLE flip (next)

Add `kvm_mega::emit_kvm_mega_launcher` to
`crates/ferrite-forward-macro/src/interpreters/kvm_mega.rs`.
For each (canonical, bucket) of every kvm-eligible canonical,
emit a `kvm_mega_<canonical>_m_<wp>` Rust wrapper fn that:

1. Extracts layered-weight base pointers from `Weights<W>`
   accessors (`(Weights::self_attn_qkv_proj)(wm, 0).dense_weight()`,
   etc.) — relies on the layer-0-base-equals-block-base
   invariant established by 6c-iii-b.
2. Pulls the kvm-compat full-width RoPE pointers from the
   canonical's Weights (see vendor-fork roadmap below for why
   these exist as separate fields and the path to deleting them).
3. Builds `KvmWeightPtrs` / `KvmRopePtrs` / `KvmShapeConfig`
   structs, with shape constants baked from `kvm_kernel_dims_from_bounds`
   and `batch_size` / `vocab_size` derived at runtime
   (`batch_size = ctx.cu_seqlens_q.numel() - 1`, NOT
   `wp.num_tokens` — the bucket worst-case is fine for
   eligibility but the actual call's batch is the seq count).
4. Calls `launch_kvm_mega(..)`.
5. Places returned `(hidden, logits)` `OwnedTensor`s into
   `tiles[backbone_slot]` / `tiles[terminal_slot]` as
   `TileEntry::Owned`.

The wrapper is per-bucket because the tape static
`KVM_FULL_M_<wp>` is per-bucket, but the C extern fn is per-
canonical. ~108 canonicals × ~5 buckets ≈ ~540 wrappers; emit
each as ~30 lines of Rust delegating to `launch_kvm_mega` —
~16k LoC total, comfortable below the codegen-size cliff.

Then in codegen.rs's `LAUNCHER_TABLE` build loop, replace the
kvm-slot `None` with `Some(kvm_mega_<canonical>_m_<wp>)` for
every bucket of an eligible canonical.

Codegen emission also needs to: at model-load time, for kvm-
eligible canonicals, allocate the two full-width RoPE tables
(`kvm_rope_cos_full` / `kvm_rope_sin_full`, `[max_pos, head_dim]`)
and D2D-mirror the half-width values into both halves of each
row. See vendor-fork roadmap below for the deletion plan.

After this lands → step 6c-iii-e: H100 smoke test
(`FERRITE_FORCE_KVM_MEGA=1 vllm chat -m
meta-llama/Llama-3.2-1B-Instruct`).

### Step 6c — per-canonical marshaling wrapper (REFERENCE — most done; see 6c-iii-d-2 in STATE for current entry point)

**Goal.** Emit `kvm_mega_<canonical>` as a `KvmMegaLauncher<Weights>`
fn that fills the ~50 C args with values from `Weights<W>` /
`ForwardCtx<W>` accessors, then calls the
`tk_megakernel_<canonical>_launch` symbol declared in 6b-i. Flip
the `LAUNCHER_TABLE` kvm slot from `None` to `Some(kvm_mega_<canonical>)`
for every bucket of an eligible canonical. After this lands,
`FERRITE_FORCE_KVM_MEGA=1 vllm chat …` exercises the kvm path
end-to-end.

**Status.** Decomposition below is complete through 6c-iii-d-1.
The remaining piece (codegen wrapper + LAUNCHER_TABLE flip) is
6c-iii-d-2 in STATE; the argument categories below remain useful
reference for what each emitted wrapper marshals.

**Why it's bigger than 6a/6b-i.** Previous handoff text claimed
`tk_paged_kv` and `tk_instructions` modules existed in
`crates/ferrite-forward/src/` from earlier ports — they don't.
Both were ripped out in commits `d22cba7bf` and `8587caddd`
(originated from the forbidden worktree-ferrite-mega; vendor
scheduler.py is the source of truth, not the hand-Rust port).
Step 6c rebuilds the host-side helpers from scratch under our
own commits, justified line-by-line.

**Argument categories** (see C signature in
`kvm_mega::emit_kvm_cu_source` ~ln 1035):

1. **Tape (`d_instructions`, `num_instructions`).** Use the
   IR-emitted `KVM_BACKBONE_M_<wp>` + `KVM_LM_HEAD_M_<wp>`
   static slices from `emit_kvm_program` (already landed). The
   wrapper memcpys the concatenated bytes to a per-call device
   tape buffer. **Open Q:** the C launcher takes ONE tape; we
   emit two per bucket (backbone + lm_head). Either concatenate
   at codegen time into one `KVM_FULL_M_<wp>` static and emit
   that, or memcpy them sequentially into one device buffer.
   Concatenation at codegen is cleaner; lands as a small
   change to `emit_kvm_program` + the codegen call site.

2. **Per-call device scratch (`d_bar`, `d_timings`,
   `d_global_instruction_index`).** Sized from kernel dims. The
   `Bar` array is a per-op semaphore buffer; the C globals's
   gl<> dims (bar_d0..d3) come from the tape's instruction
   count. `d_timings` is a debug timing buffer (`num_timing_rows
   * TIMING_WIDTH * sizeof(int)`). `d_global_instruction_index`
   is one int initialized to 0. Allocate via
   `device.caching.alloc_tensor` (zero-init for the global
   index — needs a memset_async helper or a one-int memcpy).

3. **Layered weights (`d_qkv_weights, qkv_R, …`).** Map vendor's
   fused-tensor names to ferrite's `Weights` accessors. The C
   side wants:
   - `qkv_weights` — vendor expects ONE concat-Q+K+V tensor.
     Ferrite's llama already concats: `self_attn_qkv_proj`
     accessor returns the fused tensor (verify by grepping the
     emitted Weights struct on llama_3_2_1b: search for
     `qkv_proj` field declarations).
   - `up_weights` / `gate_weights` — separate accessors
     (`mlp_up_proj`, `mlp_gate_proj`). Vendor takes them
     separate — no fusion needed.
   - `down_weights`, `attn_norm_weights` (= `input_layernorm`),
     `mlp_norm_weights` (= `post_attention_layernorm`),
     `o_weights` (= `self_attn_o_proj`),
     `lm_head_norm_weights` (= `norm`),
     `lm_head_weights` (= `lm_head`).

   Each accessor is `(Weights::<name>)(wm, layer)` returning
   `&LinearLayer` / `&RmsNorm` / `&Embedding`. The wrapper
   needs the FIRST layer's pointer for layered weights — TK's
   `gl<>` carries `[num_layers, R, hidden]` so passing
   layer-0's raw_ptr gives the base of the contiguous block.
   **Verify** layered weights are stored contiguously (i.e.,
   `Vec<LinearLayer>` allocation behavior). If not, codegen
   needs to bake a contiguous-block load for kvm-eligible
   canonicals (single safetensors load for the full
   `[L, R, H]` tensor instead of per-layer).
   `R` parameters (`qkv_R`, `o_R`, `up_R`, etc.) are runtime
   shape dims — derive from `.dense_weight().shape()[0]` per
   `prim_mega::emit_prim_mega_launcher`'s
   `WeightShapeDim` precedent.

4. **KV cache (`d_k_cache`, `d_v_cache`, `kv_total_pages`,
   `kv_page_size_runtime`).** `ForwardCtx::kv_cache:
   &KvCachePool` exposes per-layer `k_cache(layer)` and
   `v_cache(layer)`; the C side takes ONE base pointer for all
   layers, so we need `k_cache(0).raw_ptr()` and the pool to
   guarantee contiguous-across-layers layout (currently it
   is — `KvCachePool` allocates one big `[L*P, page_size,
   num_kv_heads, head_dim]` block). `kv_total_pages` =
   `pool.num_pages()`; `kv_page_size_runtime` =
   `pool.page_size()`.

5. **RoPE tables (`d_rope_cos`, `d_rope_sin`, `max_pos`).**
   `Weights::rotary_cos_sin(0)` — already exists per the
   `pub fn rotary_cos_sin(&self, _layer: u32)` arm at
   codegen.rs:1884. Returns the `(cos, sin)` GpuTensor pair.
   `max_pos` = first-axis length.

6. **Per-call activations (`d_hidden_states`,
   `d_rms_rope_intermediates`, `d_rms_gate_intermediates`,
   `d_q_post_rope`, `d_attn_out`, `d_silu_out`,
   `d_rms_lm_head_intermediates`, `d_logits`).** Allocated
   per-call via `device.caching.alloc_tensor`. Shapes derive
   from `EncodeCtx`:
   - `hidden_states`, `rms_rope_intermediates`,
     `rms_gate_intermediates`, `attn_out`,
     `rms_lm_head_intermediates` → `[batch_size, hidden_dim]`
     bf16.
   - `q_post_rope` → `[batch_size, num_attention_heads * head_dim]`
     bf16.
   - `silu_out` → `[batch_size, intermediate_dim / num_devices]`
     bf16.
   - `logits` → `[batch_size, vocab_size]` bf16.

   `batch_size_arg` and `vocab_size_arg` are scalar ints.

7. **Paged-KV CSR metadata (`d_position_ids`,
   `d_kv_append_indices`, prefill+decode `qo_indptr` /
   `kv_indptr` / `kv_indices` / `kv_last_page_len`).** This is
   the rebuild-from-scratch piece. Today ferrite's
   `ForwardCtx` carries `block_table` (decode), `seqused_k`
   (decode), `cu_seqlens_q` (prefill), `slot_mapping` (cache
   write indices). We need to derive vendor's CSR triples per
   call:
   - `prefill_qo_indptr` = `cu_seqlens_q` (prefill side, drop
     decode-only entries).
   - `prefill_kv_indptr` / `prefill_kv_indices` /
     `prefill_kv_last_page_len` — derived by walking
     `block_table[prefill_seqs]` and computing prefix sums.
     D2H copy of `block_table` + `seqused_k`, host-side build,
     H2D back. Vendor's
     `Megakernels/demos/cross-gpu-llama/python/host_helpers.py`
     (or equivalent) is the algorithmic reference; the
     forbidden branch's `tk_paged_kv::build_decode_metadata`
     was a port of the same — re-derive from vendor source,
     not from that branch.
   - Decode side same shape, decode-only seqs.
   - `kv_append_indices` ≈ `slot_mapping` after a vendor-
     specific transform (page index = slot / page_size,
     intra-page position = slot % page_size, packed as one int
     per token).
   - `position_ids` = `ctx.positions` (length =
     `num_position_ids`).

   D2H/H2D round-trips are unavoidable for CSR build (block
   table is device-side); pipeline them on a side stream so
   the kernel can launch without waiting for compute_stream
   sync. Land as a new `kvm_mega::build_paged_kv_metadata`
   helper in `ferrite-forward` (NOT proc-macro — runtime
   code), called from the per-canonical wrapper.

8. **Scalars (`attn_scale`, `rms_norm_eps`, `num_pages`,
   `num_prefill_tokens`, `dev_idx`).** `attn_scale` = `1 /
   sqrt(head_dim)` baked at codegen time as a `f32` literal.
   `rms_norm_eps` from `Weights::input_layernorm(0).eps`.
   `num_pages` = `kv_cache.num_pages()` (already computed for
   `kv_total_pages`). `num_prefill_tokens` = `ctx.cu_seqlens_q`
   prefill-side accumulation. `dev_idx` = `0` (TP=1).

9. **Stream (`raw_stream`).** `device.compute_stream as *mut
   c_void`.

**Suggested decomposition into commits.**

- **6c-i.** Concatenate per-bucket `KVM_BACKBONE_M_<wp>` +
  `KVM_LM_HEAD_M_<wp>` into a single `KVM_FULL_M_<wp>` static
  in `emit_kvm_program` / `try_emit_kvm_mega`. Pure codegen
  refactor; tape contents change shape only (bytes are the
  concatenation).
- **6c-ii.** Add `kvm_mega::build_paged_kv_metadata` in
  `ferrite-forward`. Pure host-side helper, unit-testable
  with synthetic block tables. Reference vendor source
  (`~/Megakernels/...`), not the forbidden branch.
- **6c-iii.** Emit `kvm_mega_<canonical>` wrapper fn in
  codegen + flip `LAUNCHER_TABLE` slot. End-to-end smoke test
  with `FERRITE_FORCE_KVM_MEGA=1 vllm chat -m
  meta-llama/Llama-3.2-1B-Instruct` on H100.

### What's green right now

- **`libmegakernels.a` builds end-to-end on H100** (172 MB, 108 .o):
  107 `T tk_megakernel_<canonical>_launch` symbols + `prim_mega.o`.
  Verify: `nm libmegakernels.a | grep "T tk_megakernel_.*_launch$"
  | wc -l` → `107`.
- **Build invocation that works:**
  ```
  PATH=/usr/local/cuda-12.9/bin:$PATH \
  CUDA_PATH=/usr/local/cuda-12.9 \
  FERRITE_GPU=h100 CUDA_ARCH=90 \
    cargo build -p ferrite-cuda-builder --features cuda
  ```
  Both env vars matter: `FERRITE_GPU=h100` makes the proc-macro emit
  KvmMega tape (gates `target_profile.kvm_compatible()`); `CUDA_ARCH=90`
  makes `ferrite-cuda-builder/build.rs::detect_cuda_arch` use sm_90a
  (the cudaforge auto-suffix path) instead of falling back to
  nvidia-smi → sm_89 on this L4 dev box and corrupting the .o cache.
- **Per-canonical `.cu` emission lives in
  `~/.cache/cudaforge/megakernels/tk_megakernel_<canonical>.cu`**,
  written by `kvm_mega::write_kvm_cu_to_cache` during proc-macro
  expansion of every model crate that has a kvm-eligible bucket.
  `build_megakernels` scans that dir + `crates/vllm-cuda/csrc/
  megakernel/` and feeds both into a single nvcc → `libmegakernels.a`
  invocation.
- **Launcher body** (`tk_megakernel_<canonical>_launch`) takes ~50
  flat C-friendly args (per-field raw pointers + scalar shape ints),
  aggregate-inits `llama_70b_globals` mirroring `globals_t<>`
  (llama.cuh:230–292), sets
  `cudaFuncAttributeMaxDynamicSharedMemorySize`, launches
  `mk<llama_config, llama_70b_globals, ops...><<<grid, block, smem,
  stream>>>(g)`, returns `(int)cudaGetLastError()`. Today nothing
  *calls* it from Rust — that's step 6.

### Critical invariants worth knowing before touching anything

- **Vendor `globals_t::num_devices` was hardcoded `8`.** Patched in
  `5f715530e` (now `a53da44ef`) to `LLAMA_NUM_DEVICES`. Without this,
  `kv_cache_t = gl<bf16, -1, -1, num_kv_heads/num_devices, ...>` goes
  to `r=0` for any model with `num_kv_heads<8` (deepseek-v3-tiny,
  gemma3-1b, smollm2) and TK's `gl<>` static_asserts fail at NVCC.
- **`ferrite-cuda-builder` depends on `ferrite-models`** in
  `[dependencies]` (not `[build-dependencies]`). With cargo's
  pipelined compilation, `build_megakernels` *can* run before the
  proc-macro for every model crate has finished writing its `.cu` to
  the cache — empirically observed during this session. **First
  build after a `cargo clean` may compile a partial set of `.cu`
  files; a second `cargo build` picks the rest up via
  rerun-if-changed.** If you see `libmegakernels.manifest`
  containing only `prim_mega.o` after a build that should have
  written 107 launcher symbols, this is the cause. The
  `cargo:warning=build_megakernels picking up N cu files` warning in
  `build.rs` makes this surface immediately.
- **`kvm_kernel_dims_from_bounds` returns `Option<KvmKernelDims>`**;
  canonicals that fail vendor static_asserts (`head_dim % 32 != 0`,
  `num_kv_heads % num_devices != 0`, `kv_col_start % 2 != 0`,
  `hidden_dim % 256 != 0`, `intermediate_dim/num_devices % 256 != 0`)
  return `None` and silently skip `.cu` emission. That's how we
  avoid feeding NVCC source it would reject.
- **`pick_interpreter` returns KvmMega only if every picked Impl
  is Kvm-fit.** Today the cost CSV doesn't favor KvmFit Impls over
  the host counterparts (KvmFit Impls delegate to host), so even
  with `kvm_compatible() = true` the cost summary shows every
  bucket as `→Host`. That's expected — emission of `.cu` files
  happens independently of `pick_interpreter` (gated only on
  `kvm_compatible()`), so the build is still exercising every
  KvmFit encoder arm.

### Steps 6a + 6b-i (done) — pointers to current state

- `KvmMegaLauncher<W>` fn-ptr type + `kvm_mega_forced()`:
  `vllm-rs/crates/ferrite-forward/src/lib.rs`.
- 3-slot `LAUNCHER_TABLE` + kvm dispatch branches in
  `forward()` / `forward_backbone()`:
  `vllm-rs/crates/ferrite-forward-macro/src/codegen.rs` ~ln
  3756–3905.
- Per-canonical extern "C" decl emitter:
  `vllm-rs/crates/ferrite-forward-macro/src/interpreters/kvm_mega.rs::emit_kvm_extern_decl`
  (called from `codegen.rs` ~ln 3589).

After step 6c lands, the verification path is:
`FERRITE_FORCE_KVM_MEGA=1 vllm chat -m
meta-llama/Llama-3.2-1B-Instruct` on H100 → "capital of France is
Paris" coherent. P2-5 (multi-arch coverage) and P2-6 (multi-GPU
barrier opcodes) follow.

### Open follow-ups (not blockers for step 6c)

- **`kv_cache_t` page count.** `globals_t::num_devices = 1` patch
  (TP=1) makes the gl<>-r dim work for `num_kv_heads<8`, but the
  kernel runtime path still does `kv_offset * RT::cols` arithmetic
  that assumed 8-way sharding. Spot-check on Llama-3.2-1B once
  step 6 calls the launcher.
- **Cost CSV doesn't favor Kvm Impls.** Adding KvmFit cost rows
  (or halving the per-row cost vs. host as a placeholder) is
  what flips `pick_interpreter` to actually return KvmMega for
  a bucket. Today every bucket reads `→Host` even though the .cu
  is built. This blocks `pick_interpreter`-driven dispatch (but
  not `FERRITE_FORCE_KVM_MEGA`-driven dispatch).

### Vendor-fork roadmap (we are already maintaining a fork)

Vendor's `~/Megakernels/demos/cross-gpu-llama` is in-tree and
patchable. We've already landed one patch (`a53da44ef`,
`globals_t::num_devices` 8 → `LLAMA_NUM_DEVICES`); more are
inevitable. Tracked here so the "MVP shim vs vendor-correct"
choice for each isn't relitigated every session.

- **RoPE table layout — MVP shim, vendor patch later.** Vendor's
  `rope_table_t = gl<float, 1, 1, max_pos, head_dim>` is full-
  width per row; vendor's `apply_rope_inplace` does
  `warp::mul(input, input, rope_cos)` over the whole `head_dim`.
  Ferrite's `RotaryCache::cos_cache` is `[max_pos, head_dim/2]`,
  which is what the math actually wants — RoPE rotates pairs
  `(x_i, x_{i+head_dim/2})` using one (cos, sin) per pair, so
  there are only `head_dim/2` distinct angles per position.
  Vendor's full-width is just the half-width values mirrored,
  i.e. wasted bandwidth even for vendor.

  **MVP path (step 6c-iii-d-2)**: per-canonical Weights for
  kvm-eligible canonicals load TWO extra GpuTensors
  (`kvm_rope_cos_full` / `kvm_rope_sin_full`, both `[max_pos,
  head_dim]` f32) at model load time, populated by D2D-mirroring
  the half-width cache. ~32 MB per arch one-time. The kvm
  marshaling wrapper hands the pointers to vendor with no
  per-call cost. Clearly labeled `// vendor-compat, delete when
  vendor patched`.

  **For-real path (later)**: patch
  `Megakernels/demos/cross-gpu-llama/qkv_rope_append.cu` so the
  rope kernel reads `[max_pos, head_dim/2]` and broadcasts each
  cos/sin across the rotation pair. Then: delete the per-arch
  `kvm_rope_*_full` fields, the load-time mirroring code, and
  the wrapper's pointer extraction shim. Net effect after the
  vendor patch: zero ferrite-side concession to vendor's layout.

- **`globals_t::num_devices` hardcoding (already patched).**
  See `~/Megakernels/...` patch `a53da44ef` — this is already
  load-bearing for any model with `num_kv_heads < 8`. Keep
  patched as long as our fork lives.

- **Bar buffer dim-3 conventions.** Vendor's
  `barriers[layer, opcode-1, batch_block, out_block]` indexing
  encodes vendor's specific opcode set. Ferrite sizes Bar
  conservatively in `kvm_mega::bar_dims` (16 opcode slots, etc).
  When vendor adds opcodes or changes the indexing convention,
  bar_dims needs to track. Not a blocker today.

- **Eight-way TP hardcoding (other fields).** Per
  `~/Megakernels/CLAUDE.md` and the existing patch in
  `a53da44ef`, several `_per_dev` reductions are still
  derivation-from-`num_devices` rather than parameterized. Will
  bite at TP > 1 — same fork, same deletion path, future
  session's problem.

---

## PIVOT 2026-04-27 — KvmMega is now the perf path

Tip `af4e8a13c`. **Phase 1 (PrimMega-as-perf-path) is closed; we
have pivoted to KvmMega via the vendored TK throughput ops.**

### Why the pivot

1. **Paged attention is non-negotiable.** Decode-path attention reads
   from ferrite's paged KV pool; mega has to honor that. The candidate
   routes are:
   - Vendor TK standalone attention (`~/ThunderKittens/kernels/
     attention/mha_h100/`) — contiguous K/V only. **Disqualified.**
   - FlashInfer DC — has paged KV, but plumbing the plan handle
     into the launcher (set_io / replan / params host→device per
     forward, plus per-tuple shim accessors) is heavy. **Abandoned.**
   - The vendored `Megakernels/demos/cross-gpu-llama/` ops
     (`attention_decode.cu`, `attention_prefill.cu`,
     `qkv_rope_append.cu`) — paged-KV-aware by construction; consume
     the `(kv_indptr, kv_indices, kv_last_page_len)` CSR triple
     directly. **The shipping path on `worktree-ferrite-mega@417e16bda`,
     E2E "capital of France is Paris" verified on H100 Llama-3.2-1B
     at TTFT 9.1ms / ITL 7.5ms.**

2. **TK kernels aren't natively DC-callable.** They're authored as
   KVM-framework structs (`controller / loader / consumer / storer`
   warp specialization). Plugging them into PrimMega's
   `switch(opcode)` body would require unwrapping the warp
   specialization — a rewrite that destroys the perf those kernels
   are designed for.

3. **PrimMega proved its job.** `feedback_prim_mega_is_scaffolding`
   already framed PrimMega as scaffolding for KvmMega; the Embed mega
   arm landing (`2bd945e6a`) demonstrates the encoder + launcher
   emission + `LAUNCHER_TABLE` infra works end-to-end. Adding more
   PrimMega arms (FlashInfer DC, FusedGateUpSiluMul decomp,
   occupancy-derived launch geometry) is investment in the wrong
   target.

### What's done (this session, pivot prep)

Four commits past `468aa7faf`:

- `2bd945e6a` — **Embed mega arm**. Closed the last gap in PrimMega's
  scaffolding proof: `dc_embed<T>` C++ kernel, `OP_EMBED` opcode,
  `encode_embed` arm, `ForwardField::InputIds`, `DcEmbeddingImpl`.
  232/232 tests, models build clean. PrimMega remains structurally
  complete; `LAUNCHER_TABLE` will not see real `Some/Some` entries
  via this path because the FI DC + FusedGateUpSiluMul gap won't be
  closed (per pivot above), but the launcher *symbols* are emitted
  and verifiable.

- `b3f67673e` — **cross-gpu-llama vendor deltas.** Ported four-file
  diff from `worktree-ferrite-mega@417e16bda` onto our vendor:
  - `attention_decode.cu` / `attention_prefill.cu`: relax
    `GQA_RATIO == 8` → `<= 16`; hoist `qkv_kv_blocks` /
    `qkv_q_blocks` as `constexpr` off `Globals` so they generalize
    across (matmul_out_block_size, head_dim, num_devices).
  - `qkv_rope_append.cu`: drop two `static_assert(num_devices == 8,
    "Fix this function.")` markers (the algorithmic dependency is
    still real and stays Phase 2 follow-up).
  - `llama.cuh`: `#pragma once` → `#ifndef LLAMA_CUH_INCLUDED` guard
    so codegen-emitted TUs can include op `.cu` files that include
    `llama.cuh` without redefinition; gate every `LLAMA_<DIM>` on
    `#ifndef` so per-arch defines from codegen take precedence.

- `eb22e4df5` — **`tk_paged_kv` + `tk_instructions` ported** to
  `crates/ferrite-forward/src/`. Both host-side, both already
  KvmMega-shaped:
  - `tk_paged_kv::build_decode_metadata` — D2H `block_table` +
    `seqused_k`, build CSR triple, H2D back. Per-call work,
    no kernel.
  - `tk_instructions::build_throughput_instructions` — emits
    `Vec<i32>` of length `total_instructions * 32` with the 13
    cross-gpu-llama opcodes (`OPCODE_AttnNorm`,
    `OPCODE_QKV_RopeAppend`, …, `OPCODE_AllDeviceBarrier`).
    Work-stealing: kernel atomically increments
    `global_instruction_index`, no per-op grid sync — DAG ordering
    encoded by tape position.

- `af4e8a13c` — **`generate_tk_megakernel` + `TkModelDims`** ported.
  New `cuda_codegen` module in `ferrite-forward-macro` carrying the
  per-arch `.cu` source generator: `LLAMA_<DIM>` macro overrides,
  13-op includes, `ops::` alias block, `mk<llama_config,
  llama_70b_globals, op1..op13>` instantiation, `extern "C" int
  tk_megakernel_<model>_launch(...)` wrapper with ~50 flat C-friendly
  args. Older branch's `DevicePhase`-based `generate_megakernel`
  stripped (depended on data structures from a different IR).

### What's NOT yet done (Phase 2 — KvmMega proper)

- **Encoder reshape.** `interpreters/prim_mega.rs` produces an
  8-opcode flat tape; KvmMega's tape format is a 13-opcode
  work-stealing tape (different opcodes, different row payloads,
  different DAG-encoding mechanism). New file
  `interpreters/kvm_mega.rs` consumes the same `Vec<OpInstance>` the
  PrimMega encoder consumes, emits a TK-shape `Vec<i32>`. Five
  mapping unknowns to resolve before writing code (see "§Phase 2
  work order" below).

- **KvmFit Impls.** New `MegakernelFit::Kvm` Impls in `impl_lib.rs`
  for the variants TK throughput consumes. Per
  `feedback_prim_mega_is_scaffolding`, KvmFit Impls are the actual
  perf-path Impls; they replace (not extend) the PrimMega DC
  siblings on Hopper.

- **`forward!` proc-macro consumer for `generate_tk_megakernel`.**
  Today `cuda_codegen::generate_tk_megakernel` is a library function
  with no caller. Wire it into `codegen.rs`'s per-canonical emit so
  per-arch megakernel `.cu` files land in `~/.cache/cudaforge/
  megakernels/` for `build_megakernels` to pick up.

- **Launcher fn shape.** `PrimMegaLauncher<W>` in
  `ferrite-forward/src/lib.rs` is the right shape but the body
  changes: instead of `prim_mega_llama_launch(tape, len, pt[],
  grid, block, smem, stream)`, it's
  `tk_megakernel_<model>_launch(...)` taking the ~50 flat args
  (weights, activations, KV cache, RoPE tables, paged metadata,
  attn_scale, num_pages, batch_size, num_prefill_tokens). The host
  side builds the metadata via `tk_paged_kv` + the tape via
  `tk_instructions` immediately before each call.

### What's NOT going to happen (recorded so future sessions don't
chase them)

- **No FlashInfer DC arm in prim_mega.cu.** The `dc_flashinfer.cuh`
  template (`cd9fd897c`) stays vendored as a reference for the
  pattern but won't be wired up. Plan-handle threading is too heavy
  for the value vs the KvmMega route.

- **No occupancy-derived launch geometry for prim_mega_llama_kernel.**
  The placeholders `grid_x=132 / block_x=256 / smem=49152` stay; the
  kernel itself is scaffolding and will be retired when KvmMega
  ships. Effort would be wasted.

- **No FusedGateUpSiluMul decomp into Gemm × 2 + SiluMul rows for
  PrimMega forced mode.** Same reason.

- **No per-arch PrimMega TUs.** PrimMega remains a single
  `prim_mega_llama_kernel` (mis-named — actually generic across the
  CUTLASS-DC path on any arch). KvmMega is per-arch by construction
  (`generate_tk_megakernel(model_name, dims)`).

### What stays (the rules still apply)

`§Things-that-must-not-happen` is unchanged and applies fully to the
KvmMega encoder. Specifically:

- No `_` catch-all in `interpreters/kvm_mega.rs`'s match. Missing
  variants surface as compile errors at codegen.
- No runtime fallback ("try mega, fall back if it fails").
- No piecemeal Impl migration. KvmFit Impl audit is wholesale.
- No mega wire types in `ferrite-forward` runtime — same
  macro-emitted launcher + vendored `.cu` discipline.
- North stars (`feedback_ferrite_compiler_stars`): no math leakage,
  no combinatorial explosion, const-prop natural, megakernel
  natural.

The host-interpreter pivot (`HANDOFF_INTERPRETER.md`) remains the
prerequisite. The `Instruction<W>` IR is unchanged — KvmMega is
another consumer of the same IR, not a new IR.

## Phase 2 work order — the actual entry point for the next session

**Step P2-1 — Encoder mapping design (paper, not code).** ✅ Landed
as `KVM_MAPPING.md` (this commit). Five unknowns resolved:
- Q1 residual fusion: option (b) — match-by-IR-variant. DSL must
  emit `Gemm + Add + RmsNorm` unfused; solver picks `KvmCutlassGemmAddImpl`
  (over `Gemm + Add` claim) on Hopper + `KvmRmsNormImpl` (over
  `RmsNorm` alone). `FusedAddRmsNorm` → kvm-ineligible (mapping it
  to `OPCODE_AttnNorm` would double-add since the residual is in
  the prior matmul-with-residual storer).
- Q2 QKV: both `FusedQkvRopeCache` + `FusedQkvRopePrefill` → same
  `OPCODE_QKV_RopeAppend` row template; `g.num_prefill_tokens`
  toggles kernel-side. Mixed prefill+decode in one forward = out
  of P2 scope.
- Q3 Gate/Up: encoder splits `FusedGateUpSiluMul` into
  `OPCODE_GateSiLU` + `OPCODE_UpMatmul` rows at encode time.
  Solver claim-mask K bound likely needs to grow per
  `feedback_solver_claim_mask_size`.
- Q4 GemmAdd routing: encoder phase-state machine (`Phase::Attn`
  after `OPCODE_AttnNorm`, `Phase::Mlp` after `OPCODE_MlpNorm`)
  picks O_Proj vs Down_Proj opcode for `CutlassGemmAdd`.
- Q5 barriers: not emitted for single-GPU. Cross-instruction sync
  is by per-op `Bar` increments + loader spin-waits, not tape rows.

Read `KVM_MAPPING.md` before writing P2-2. Both blocking
preconditions **confirmed satisfied** (2026-04-27):
- Q1 DSL shape: llama already emits unfused `gemm + add + rmsnorm`;
  existing `CutlassGemmAddImpl` already claims `(Gemm, Add)`.
- Q3 solver claim-mask: K=16 bounds tile-spread, not output-row
  fan-out. `FusedGateUpSiluMulImpl` claims 4 tiles; KvmFit
  sibling inherits. No bump needed.

Work for P2-2..P2-5 is in `impl_lib.rs` (new `Kvm*Impl` tier
delegating to host counterparts, parallel to `Dc*Impl`) +
`interpreters/kvm_mega.rs` (encoder) + `codegen.rs` (wire
`generate_tk_megakernel`) + a new `FERRITE_FORCE_KVM_MEGA` env
override. No DSL refactor; no solver refactor.

The five unknowns:

1. **Residual fusion.** TK's `OPCODE_AttnNorm` / `OPCODE_MlpNorm`
   include the cross-layer residual-add as input. We have
   `FusedAddRmsNorm` and `Add` as separate variants. Two options:
   (a) the encoder collapses adjacent `Add` + `RmsNorm` (or fuses
   straight from `FusedAddRmsNorm`) into one TK row at encode time;
   (b) the solver picks claims at TK granularity (a new
   `KvmFusedAddRmsNorm` Impl that claims both tiles together). (b)
   is more principled but bigger refactor. Pick one; document why.

2. **QKV decode/prefill collapse.** TK has one `OPCODE_QKV_RopeAppend`.
   We have `FusedQkvRopeCache` (decode) + `FusedQkvRopePrefill`
   (prefill) as distinct variants. Likely both map to the same
   opcode; the encoder reads the variant tag to decide which row
   payload to emit. Verify the TK op handles both cases (checked
   `cross-gpu-llama/qkv_rope_append.cu` — yes, it conditions on
   `g.num_prefill_tokens`).

3. **Gate/Up split.** TK has `OPCODE_GateSiLU` + `OPCODE_UpMatmul`
   as **two** opcodes. We have `FusedGateUpSiluMul` as **one**
   variant. The encoder must emit two TK rows from one of our rows.
   Splitting at encode time is mechanical; the alternative (split
   the Impl back out) defeats our fusion analysis. Pick (a) encoder
   split.

4. **GemmAdd routing.** TK's `OPCODE_O_ProjResidual` and
   `OPCODE_DownProjResidual` both map to `CutlassGemmAdd` (or
   `Gemm` followed by `Add`). The encoder needs to know *which*
   TK opcode to emit based on position in the tape (post-attention
   vs post-MLP). Either rely on tape order (encoder maintains
   per-bucket "I've seen attn / I've seen MLP" state) or add tile
   metadata that distinguishes them.

5. **Barrier synthesis.** TK's `OPCODE_Barrier_Inc` /
   `OPCODE_AllDeviceBarrier` have no `Instruction<W>` analog —
   they're tape-position-synthesized between phases. The encoder
   inserts them at the right cadence. Match
   `worktree-ferrite-mega@417e16bda`'s `tk_instructions::build_
   throughput_instructions` exactly — that's the pattern that ships
   coherent output.

Output of P2-1: a table mapping each `Instruction<W>` variant + each
TK opcode bidirectionally, with row-payload field-by-field
correspondence. Lands as a single doc commit.

**Step P2-2 — `interpreters/kvm_mega.rs` encoder.** Mirror
`interpreters/prim_mega.rs`'s shape:
- `EncodedRow` reused if applicable, or new `KvmEncodedRow` with
  TK's exact 32-int row payload semantics.
- `try_encode_bucket(arch_opcodes, instances, ctx) ->
  Option<KvmEncodedBucket>` with closed match per
  `Instruction<W>` variant per the P2-1 mapping table.
- `emit_kvm_program` + `emit_kvm_launcher` rendering the static
  tape + launcher fn body.

Tests on synthetic OpInstances per variant.

**Step P2-3 — KvmFit Impls in `impl_lib.rs`.** New
`MegakernelFit::Kvm` Impls covering the variants TK consumes. By
`feedback_no_piecemeal_codegen_migration`, this is wholesale — every
variant the TK throughput tape supports gets a KvmFit Impl in one
commit.

**Step P2-4 — Wire `generate_tk_megakernel` into `codegen.rs`.**
Per-canonical: when the bucket is fully KvmFit, emit a
`tk_megakernel_<canonical>.cu` to the megakernel cache dir (similar
to `prim_mega_llama_kernel`'s build path), emit Rust extern "C" decl
for `tk_megakernel_<canonical>_launch`, populate
`LAUNCHER_TABLE` with a `KvmMegaLauncher` arm.

**Step P2-5 — `FERRITE_FORCE_KVM_MEGA` env override + E2E.**
Parallel to `FERRITE_FORCE_PRIM_MEGA`. Once a Llama-3.2-1B canonical
has full KvmFit coverage, set the env var and confirm "capital of
France is Paris" coherent on H100, byte-equal vs host on a fixed
seed. Matches `worktree-ferrite-mega@417e16bda`'s verification.

**Step P2-6 — Cross-arch coverage.** Cohere, Granite, Gemma3 each
need their own per-arch megakernel TU. The `head_dim=64, GQA=4`
gate the older branch hit (only Llama-3.2-1B / Granite compiled)
applies; non-conforming archs stay on host.

## Where we are

Phase 1 work order steps 1–5 done + cost model picks DC on Hopper /
host on Ada (principled, not push-order) + **DC coverage corrected
to wholesale-per-kernel for every CUTLASS family the kernel-level
Params allows** + step 6 prim_mega encoder match landed + **step 7a

Phase 1 work order steps 1–5 done + cost model picks DC on Hopper /
host on Ada (principled, not push-order) + **DC coverage corrected
to wholesale-per-kernel for every CUTLASS family the kernel-level
Params allows** + step 6 prim_mega encoder match landed + **step 7a
program-static emission landed** + **step 7b/7c launcher emission
+ codegen wiring landed** + **step 8 forward dispatch + step 9
`FERRITE_FORCE_PRIM_MEGA` env override landed (co-dependent, single
commit `0fb11c236`)**. Tip `0fb11c236`. Twenty-three commits on
`feat/rust` past the host-pivot baseline.

**Architecture decision (2026-04-27):** PrimMega stays whole-forward
+ opcode tape (current `prim_mega.cu` shape), NOT per-layer inline
kernels (older `worktree-ferrite-mega` branch's pattern). PrimMega is
**scaffolding for KvmMega**, not a perf path. KvmMega's vendor design
(`~/Megakernels/include/controller/instruction_fetch.cuh`) is a
per-SM instruction tape with opcode dispatch — PrimMega mirrors that
shape so the encoder + launcher + slot conventions transfer when
KvmMega lands. Per-layer inline would build the wrong muscle. See
`feedback_prim_mega_is_scaffolding`. **Do not optimize PrimMega**;
its job is to prove the infra end-to-end with the smallest possible
delta from KvmMega's eventual shape.

**DC coverage corrective sweep (`8d8ab1f9f`..`b080a097d`):** the
"wholesale CUTLASS DC fan-out" framing in `9a66f06b3` shipped a
subset (11 of 16 host tiles + dead s2 entries with no host
counterpart) — violation of `feedback_no_piecemeal_codegen_migration`.
Corrected wholesale-per-kernel:

- `8d8ab1f9f` — bf16-generic family: dropped `CUTLASS_DC_TILE_ZOO`
  carve-out; DC and host registrations both iterate
  `CUTLASS_TILE_ZOO` so drift is structurally impossible. Pin tests
  `dc_zoo_equals_host_zoo` (parses .cuh X-macro, asserts equality)
  + `dc_cutlass_name_covers_full_zoo` (every entry has a non-
  `unknown` `name()` arm).
- `3e4723f10` — `CutlassGemmAdd` × 16 tiles: same `device::Gemm`
  underlying class as bare GEMM (residual-add is a runtime
  `beta=1.0`), reuses existing `dc_gemm` template — pure Rust DC
  sibling, no new C++.
- `61bb767d9` — `CutlassGemv` (singleton M=1): new `dc_gemv` C++
  template wrapping `kernel::Gemv` with `DcGemvKernel_bf16_8`
  matching the standalone's `GemvKernel_8`. Caveat:
  `kernel::Gemv::Arguments` constructor isn't `CUTLASS_HOST_DEVICE`
  (CUTLASS oversight vs `kernel::Gemm`); worked around via raw-
  byte placement + field-by-field assignment. Hand-rolled
  placeholder `dc_gemv` in `megakernel_ops.cuh` deleted.
- `b080a097d` — `CutlassGemmSplitK` × 12 (tile, split_k) tuples:
  two-phase wholesale. New `dc_gemm_splitk` template inlines
  `kernel::GemmSplitKParallel` then `cg::this_grid().sync()` then
  `kernel::ReduceSplitK` — both Params constructors are
  `CUTLASS_HOST_DEVICE` in CUTLASS. Workspace contract: caller-
  supplied `[split_k, M, N]` float scratch, threaded via the
  pointer table at encoder-emission time. `OP_CUTLASS_GEMM_SPLITK`
  opcode + `CUTLASS_DC_SPLITK_LIST` X-macro list pinned to
  `CUTLASS_SPLITK_ZOO` by `dc_splitk_zoo_equals_host_zoo`.

**Hard constraint — three CUTLASS impls stay HostCallback-only:**

- `CutlassFusedGemmBiasImpl` (singleton EVT bias-fused)
- `CutlassFusedGateUpSiluMulImpl` (singleton MLP fusion)
- All sm90 kernels (`Gemm_sm90_*` Cooperative / WS / Pingpong /
  Coop2x1) — the H100-native TMA + warp-specialized path

All three use `cutlass::gemm::device::GemmUniversalAdapter<KernelType>`
whose kernel-level `kernel::GemmUniversal::Params` constructor takes
runtime device properties (`device_sms`, `sm_occupancy`) and is
`__host__` only by CUTLASS design — see `cutlass/gemm/kernel/
gemm_universal.h:303–324` and `UniversalParamsBase`'s scheduler-
partitioning logic. Plus `cudaMallocAsync` workspace via
`gemm_op.get_workspace_size`. Forking CUTLASS to add
`CUTLASS_HOST_DEVICE` would be invasive (affects every CUTLASS
consumer). User has explicitly declined the fork. Per the per-kernel
rule in `feedback_no_piecemeal_codegen_migration`, these stay
HostCallback-only — that's "this kernel doesn't have a DC form",
not a half-DC violation. See `feedback_gemm_universal_not_dc_able`.

**Implication for Hopper:** PrimMega-on-Hopper dispatches to the
sm80-shape `device::Gemm` family (which IS DC-able —
`kernel::Gemm::Params` is `CUTLASS_HOST_DEVICE`). The wgmma+TMA
sm90-native perf path is left to KvmMega (which will author TK
templates DC-friendly from the start, via the vendored
Megakernels infrastructure).

### Earlier nine commits (now-historical foundation):

- `f7249aaae` — vendor `Megakernels` (throughput@`91eaff262`) +
  `ThunderKittens` (`0b55588d2`) under `vllm-rs/third_party/`.
  Megakernels LICENSE restored from upstream `main`'s tip; see
  `third_party/VENDOR.md`.
- `ef265b2a8` — `MegakernelFit::{None, Primitive, Kvm}` enum +
  defaulted `Implementation::megakernel_fit()` →
  `MegakernelFit::None`; `interpreter_codegen.rs` renamed to
  `interpreters/host.rs` with sibling stubs `prim_mega.rs` +
  `kvm_mega.rs`; `interpreters::pick_interpreter` selector +
  `TargetProfile::{prim_mega_compatible, kvm_compatible}` stubs.
- `e9c489c6e` — persistent `__global__` skeleton at
  `vllm-cuda/csrc/megakernel/prim_mega.cu`. Cooperative grid sync
  between phases (forced by existing `dc_*` early-return
  pattern); pointer table side-channel for 64-bit ptrs; arms
  wired to `dc_rms_norm` / `dc_fused_add_rms_norm` /
  `dc_fused_qkv_rope_cache` / `dc_silu_and_mul` / `dc_gemv` from
  the existing `megakernel_ops.cuh`. Build infra extension in
  `ferrite-cuda-builder/build.rs::build_megakernels` to scan the
  tree-resident dir alongside `~/.cache/cudaforge/megakernels/`.
- `a2924e765` — CUTLASS DC pattern proven for one tile. On-device
  Params construction via `CUTLASS_HOST_DEVICE` constructors
  (`gemm.h:99–135`, `threadblock_swizzle.h:64`); zero
  host-side `prepare_*_params` shim needed for non-splitK
  configs (split_k_slices=1 sidesteps the workspace branch in
  `device::Gemm::initialize`). Header at
  `vllm-cuda/csrc/dc_cutlass.cuh`.
- `9a66f06b3` — wholesale CUTLASS DC fan-out via X-macro list
  (`cutlass_gemm_configs.cuh::CUTLASS_DC_GEMM_LIST`) expanded
  three times in prim_mega.cu (typedefs + `CutlassConfig` enum +
  `run_cutlass_gemm` switch arms). 18 workhorse tiles —
  `64x64`, `64x128`, `128x64`, `128x128`, `128x256`, `256x64`,
  `256x128` at stages `s2/s3/s4`. Deep-pipeline (s5+), 32x*
  small-M, K64, W8, swizzle, silu, GEMV, splitK, sm90 not yet
  fanned out (each needs its own typedef macro shape).
- `cd9fd897c` — FI DC template at
  `vllm-cuda/csrc/dc_flashinfer.cuh`. `dc_persistent_attn<
  Runner1, Runner2, Reduction, Params>` mirrors vendor's
  `PersistentKernelTemplate` body
  (`flashinfer/attention/persistent_template.cuh:60–97`).
  Verifies: `Runner1::Run + Runner2::Run +
  cg::this_grid().sync() + Reduction::Run` are all
  `static __device__ __forceinline__` on
  `persistent.cuh:181 + 488`, directly callable from inside
  another `__global__`. `build_megakernels` now pulls the
  FlashInfer headers via the same `with_git_dependency` pin
  (`08ab45d67`) the standalone shim build uses.
- `cf918daec` + `9b4168ed4` — DC siblings landed via thin-
  delegation pattern: `DcRmsNormImpl`, `DcFusedAddRmsNormImpl`,
  `DcFusedQkvRopeCacheImpl`, `DcCutlassGemmImpl × 11 tiles` (the
  `CUTLASS_TILE_ZOO ∩ CUTLASS_DC_GEMM_LIST` intersection — see
  `dc_cutlass_zoo_is_subset_of_host_zoo` test for the invariant).
  Each delegates `matches` / `cost_us` / `fan_out` /
  `opcode_shape` / etc. to a fresh host-counterpart instance and
  overrides only the four mega-relevant methods.
- `1f4bf3192` — step 8 trace half: per-workload `pick_interpreter`
  decision printed alongside the solve summary (commandr today
  reports `Host` across every M because DC sibling coverage is
  too thin for the all-or-nothing tier semantics to flip a whole
  canonical).
- `3f700439f` — solver now picks DC siblings on Hopper, host on
  Ada, **for principled reasons** (not library push-order).
  Reverse-engineered from `worktree-ferrite-mega/f5918269f`:
  - `prim_mega_compatible()` tightened from `>= 80` to `>= 90`.
    Ada's `cg::this_grid().sync()` is ~100us (gmem-atomic spin),
    Hopper's is ~15us (L2 synchronizer + TMA fences). Ada loses
    by construction.
  - `launch_overhead_us(DeviceCallable, _) = 0us` (was
    `InKernelGridSync`). Wave-level mega-launch cost amortizes
    over N picks → ~0.5us per pick at N=10, round to zero. The
    older branch never charged grid-sync per-pick anywhere.
  - Solver effect: host pays +5us/pick, DC pays +0us/pick. DC
    wins where feasible; feasibility gates on `>= 90`.
  - Locked by `cost::tests::solver_picks_dc_siblings_on_h100`
    (asserts ≥1 DC sibling picked on H100 LLAMA_BODY at M=1) +
    `predicted_us_includes_launch_overhead` (asserts zero DC
    picks on L4).

The host interpreter at the parent commit (`8a45b39d6` or later,
after the seam swap + slot-metadata fix) is the platform.

`libmegakernels.a` builds clean (~2.0 MB at 18 CUTLASS tiles +
ferrite DC ops; FlashInfer not yet instantiated). Includes
`prim_mega_llama_kernel` + `prim_mega_llama_launch` exported
symbols.

The pivot already staked the abstractions we need:

- `crates/ferrite-forward-macro/src/impl_lib.rs:135` — `enum
  LaunchKind { HostCallback, RegularLaunch, CooperativeLaunch,
  DeviceCallable }`. Every one of today's 49 Impls returns
  `HostCallback`. Zero `DeviceCallable` in tree.
- `crates/ferrite-forward-macro/src/impl_lib.rs:459` — `fn
  launch_kind(&self) -> LaunchKind` on the `Implementation`
  trait.
- `crates/ferrite-forward-macro/src/cost.rs:17` — comment:
  "consumers (DeviceCallable impls + megakernel emission)".
- `crates/ferrite-forward-macro/src/concurrency.rs:38–44, 119–123`
  — Rule 4 stub for "DeviceCallable in same persistent kernel".
- `Handoff` cost table already values mega-shaped handoffs:
  `KernelBoundary=5us`, `Mbarrier=0.1us`, `SyncThreads=0.5us`.
  Solver naturally prefers cheaper handoffs once Impls are
  mega-fit; no new cost knob needed for Phase 1.

## Locked design

### Two phases, one IR

The host's universal `Instruction<W>` enum + per-bucket `static
BACKBONE_M_<N>` slices are the input to **both** mega backends.
No second IR. No new Instruction variants required for Phase 1.
Phase 2 may add output-tile-bound variants (see §Phase 2).

**Phase 1 — primitive megakernel.** The host interpreter ported
to CUDA. One persistent `__global__` per arch; body is a `switch`
over `[i32; 32]` opcodes; each arm calls a `__device__` fn that
wraps cutlass / flashinfer / TK. No warp specialization, no page
virtual memory, no controller/loader/storer split. Per-op
handoffs are `__syncthreads()` or grid sync. Probably flat-or-
slightly-worse than host on perf — its job is to **prove the
infra** end-to-end with the smallest possible surface.

**Phase 2 — KVM megakernel.** Vendored `~/Megakernels` template
instantiated per arch. Warp specialization (controller / loader /
consumers / storer / launcher), instruction pipelining, page
virtual memory, per-SM instruction tape (see
`include/controller/instruction_fetch.cuh:22,32` —
`get_worker_id()` indexes the second dim of the
`[1, NUM_SMS, ROWS_PER_SM, 32]` instruction tensor). Where the
perf actually comes from. Builds on Phase 1's DeviceCallable
Impls; adds KVM-specific authoring (`release_lid`, semaphore
choreography, output-tile-granular `fan_out`).

### Solver picks first; interpreter follows

Solver behavior is unchanged — it picks the cheapest feasible
Impl per claim by cost. The interpreter is selected **post-solve**
from the picked set:

```text
fn pick_interpreter(picked_impls, profile) -> Interpreter {
    if profile.kvm_compatible()
        && picked_impls.all(|i| i.megakernel_fit() == Kvm) {
        Interpreter::KvmMega
    } else if profile.prim_mega_compatible()
        && picked_impls.all(|i| i.megakernel_fit() >= Primitive) {
        Interpreter::PrimMega
    } else {
        Interpreter::Host
    }
}
```

Both axes must hold: every picked Impl fits the tier, AND the
target supports the tier. If TK kernels are slow on a workload,
solver picks cutlass HostCallback, mega isn't available, host
runs — no force-fit, no perf regression.

### One new Impl method, no other trait surface changes

```rust
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MegakernelFit { None, Primitive, Kvm }

impl Implementation {
    fn megakernel_fit(&self) -> MegakernelFit { MegakernelFit::None }
}
```

`Kvm > Primitive > None` (a KvmFit Impl is implicitly Primitive-
fit). Default `None`; new DeviceCallable wrappings return
`Primitive`; new TK / KVM-template Impls return `Kvm`.

**No** `mega_opcode`. **No** `MegaOp` struct. **No** mega-specific
methods on `Implementation`. The encoder is one match per emitter
file, in the macro. The variant-shape information needed for
encoding comes from the `OpInstance` fields `fan_out` already
emits.

### Three emitters, sibling files

```
crates/ferrite-forward-macro/src/interpreters/
    host.rs       (today's interpreter_codegen.rs body, renamed)
    prim_mega.rs  (Phase 1 emitter)
    kvm_mega.rs   (Phase 2 emitter)
    mod.rs
```

Each consumes the same lowered buckets `colored_slot_map` /
`apply_loop_compression` / `collect_boundary_inputs` already
produce. Shared utilities stay in their current location until
the second consumer reaches for them, then extract.

### Per-emitter encoder match

Each mega emitter contains exactly one exhaustive `match` over
`Instruction<W>` variants → `[i32; 32]`. Variants without an arm
mark the canonical mega-ineligible **at codegen time** (the
emitter just doesn't emit the `MEGA_PROGRAM_M_<N>` static for that
canonical). No `_` catch-all. No runtime "refused" returns.

### Loops: unroll in Phase 1, vendor a LOOP opcode later if needed

KVM's controller is a flat `for kvms.instruction_index = 0..num_iters`
loop over the per-SM tape (`include/controller/controller.cuh:24-29`).
No provision for repeating a range. So `Instruction::Loop { body,
count }` — the row the host's `apply_loop_compression` emits —
has no native KVM analog.

**Phase 1: unroll at encode time.** Mega emitters expand each
`Loop` row back out into `count` copies of `body` before
serializing to the per-SM tape. Program memory: a worst-case
decode bucket on Llama-70B is ~80 layers × ~10 ops ≈ 800 rows ×
32 ints × 4 bytes ≈ 100 KB per SM tape — trivial in device
memory.

**Phase 2 (optional): vendor a LOOP opcode.** Modify vendored
controller to support a backwards-jump opcode that adjusts
`instruction_index`. Keeps the tape compact, lets multiple
layers share one program region, and may help i-cache pressure.
Only worth doing if profiles show program-memory or fetch
overhead. Not on the critical path.

The host emitter's `apply_loop_compression` stays unchanged —
the host runtime *does* benefit from the compressed form
(`__layer + baseline` shadowing, per-row baselines). Mega
emitters consume the compressed form and re-expand on encoding.

### No mega types in `ferrite-forward` runtime

`ferrite-forward` continues to host the universal IR + host
`eval`. All mega artifacts (encoder, scheduler, launcher,
per-arch `globals` struct, vendored .cu) are macro-emitted +
vendor code. Nothing megakernel-shaped leaks into the runtime
crate.

### Vendor in-tree

Copy `~/Megakernels` and `~/ThunderKittens` under
`vllm-rs/third_party/{megakernels,thunderkittens}/` with their
LICENSE files and snapshotted upstream commit hashes. Treat as
ours to tweak. Build path runs them through `vllm-cuda`'s build
system (or a dedicated `vllm-mega-cuda` crate — TBD on first
build).

## Phase 1 — work order

The original PoC-pair-then-widen framing (steps 4 + 10) was
collapsed into a single wholesale audit per the no-piecemeal
rule (`feedback_no_piecemeal_codegen_migration`). The list below
reflects the actual landed sequence + remaining gaps.

✅ 1. Vendor Megakernels + ThunderKittens. `f7249aaae`.
✅ 2. `MegakernelFit` enum + `megakernel_fit()` trait method
   defaulting to `None`. `ef265b2a8`.
✅ 3. `interpreters/{host,prim_mega,kvm_mega}.rs` rename +
   `Interpreter` enum + `pick_interpreter` selector +
   `TargetProfile::{prim_mega_compatible, kvm_compatible}`
   stubs (currently both return `false`). `ef265b2a8`.
✅ 5. **Primitive megakernel `.cu`** at
   `vllm-cuda/csrc/megakernel/prim_mega.cu` —
   `prim_mega_llama_kernel` + `prim_mega_llama_launch` (cooperative
   launch). Arms: rms_norm / fused_add_rms_norm / qkv_rope_cache
   / silu_and_mul / gemv / cutlass_gemm. `e9c489c6e`.
✅ 4 + 10. **DC sibling audit + wholesale-per-kernel wrapping.**
   Every CUTLASS host launcher whose kernel-level `Params` is
   `CUTLASS_HOST_DEVICE` constructible has a DC sibling. The few
   that aren't (GemmUniversalAdapter family) stay HostCallback-only
   per the per-kernel rule.
   - CUTLASS DC pattern (`dc_cutlass.cuh`) — `a2924e765`.
   - bf16-generic 16-tile zoo, wholesale (no subset) — `8d8ab1f9f`.
   - `DcCutlassGemmAddImpl` × 16 tiles (residual+gemm,
     beta=1.0 epilogue, reuses `dc_gemm`) — `3e4723f10`.
   - `DcCutlassGemvImpl` (M=1 GEMV, new `dc_gemv` template
     wrapping `kernel::Gemv` with field-assignment Params
     workaround) — `61bb767d9`.
   - `DcCutlassGemmSplitKImpl` × 12 (two-phase: GEMM grid +
     `cg::this_grid().sync()` + Reduction; new `dc_gemm_splitk`
     template) — `b080a097d`.
   - FI DC template (`dc_flashinfer.cuh`) — `cd9fd897c` (arm in
     prim_mega.cu + Params marshaling pending — see Next session).
   - `DcRmsNormImpl`, `DcFusedAddRmsNormImpl`,
     `DcFusedQkvRopeCacheImpl` — `cf918daec` / `9b4168ed4`.
   Won't-fix (kernel-level constraint):
   - sm90 family (`Gemm_sm90_*` Cooperative / WS / Pingpong /
     Coop2x1) — `device::GemmUniversalAdapter`,
     `kernel::GemmUniversal::Params` is `__host__` only.
   - `CutlassFusedGemmBiasImpl`, `CutlassFusedGateUpSiluMulImpl` —
     same `GemmUniversalAdapter` constraint.
   - See `feedback_gemm_universal_not_dc_able`. User has declined
     the CUTLASS fork.

🟡 6. Encoder match in `prim_mega.rs`. **Match + IR landed; bucket-
   level emission to TokenStream lands with step 7.** Closed match
   over OpInstance variants → [`EncodedRow`] of typed slots:
   - `RowSlot::{Const(i32), ConstExpr(TokenStream),
     Ptr(PtrSpec), Runtime(RuntimeSource)}` covers compile-time
     literals, codegen-time const expressions (e.g.
     `<Weights as CanonicalParams>::HIDDEN_SIZE as i32`),
     pointer indirection (deduped post-pass into a `ptr_plan`),
     and per-call runtime fills (e.g. `eps` from
     `Weights::<fn>(W,layer).eps`).
   - Loop unrolling at encode time threads `layer = baseline +
     iter` through every body row (vendor controller has no LOOP
     opcode; per `MEGA_HANDOFF.md` §"Loops").
   - Free / Alias rows skipped (pt[] is fixed for the duration of
     one cooperative launch).
   - Variants without an arm explicitly enumerated → `try_encode_
     bucket` returns `None`, marking canonical mega-ineligible at
     codegen time. **No `_` catch-all** — adding a new
     `Instruction<W>` variant in `ferrite-forward/src/instr.rs`
     surfaces as a `panic!()` at codegen time pointing at the
     missing arm.
   - 9 unit tests on synthetic OpInstances cover RmsNorm,
     CutlassGemm + GemmAdd, GemmSplitK workspace-per-row,
     Loop-unroll baseline preservation, Free/Alias skip, the
     "Embed has no arm" rejection path, and X-macro ordering of
     `cutlass_gemm_config_id` vs the C++ `CUTLASS_DC_GEMM_LIST`.
   - Slot conventions per prim_mega.cu's `run_*` docstrings +
     dc_cutlass.cuh's per-template caller contracts.
   Remaining: per-bucket `MEGA_PROGRAM_M_<N>: &[[i32; 32]]` static
   emission lands as part of step 7 (the launcher needs to consume
   `EncodedBucket`'s ptr_plan + runtime_fills regardless, so
   emission groups naturally with the launcher fn).
✅ 7. Per-arch globals struct emit + launcher fn.
   **7a (`2519acde1`):** `emit_prim_mega_program(static_ident,
   &EncodedBucket) -> TokenStream` renders resolved rows into
   `static <ident>: [[i32; 32]; N]` matching prim_mega.cu's
   `tape + pc * INSTRUCTION_WIDTH` contract.
   **7b/7c (`984d9e8ff`..`cb0b6d3fa`):** five-phase landing of the
   launcher fn — surfaced `slot_shapes` from colored_slot_map →
   `LoweredBucket` (phase 1, `984d9e8ff`); added
   `RuntimeSource::WeightShapeDim { fn_ident, layer, dim_idx }` for
   CUTLASS N/K runtime resolution (phase 2, `46630dbb3`); extended
   `WorkspaceKind::SplitKScratch` with `(split_k, m, weight_fn,
   weight_layer)` resolved shape source (phase 3, `adde6f37f`);
   `emit_prim_mega_launcher(fn_ident, static_program_ident, bucket,
   slot_shapes, bounds, ctx) -> TokenStream` rendering the full fn
   body inline — tile pre-alloc, split-K workspaces, tape buffer +
   memcpy, per-PtrSpec pt[] fills (TileSlot / Weight with KvCache
   prefix routing / Workspace / Forward), runtime patches for
   eps + WeightShapeDim, extern call (phase 4, `278832120`); wired
   into codegen.rs's per-canonical static_slices loop alongside
   BACKBONE_M_<wp>/LM_HEAD_M_<wp> (phase 5, `cb0b6d3fa`). 14 new
   unit tests; cargo test 230/230 ✓; commandr/llama/qwen2 builds
   clean. **Today every real model's `try_encode_bucket` returns
   None on every canonical** because at least one op in each bucket
   lacks a mega arm (Embed / FusedGateUpSiluMul / attention) — so
   the launcher body is dead in practice. The infra is what matters:
   the launcher symbols are emitted whenever encoding succeeds, so
   step 8's dispatch can call them once DC sibling coverage widens.
   **Placeholders remaining in launcher:** grid_x=132 / block_x=256
   / smem_size=49152 are TODO-marked — derive from
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor` against the
   picked phase's smem requirement.
✅ 8. **Forward dispatch wired (`0fb11c236`, co-landed with step 9).**
   Trace half (`1f4bf3192`) was already in. Runtime half added a
   parallel `LAUNCHER_TABLE: &[(Option<__PrimMegaLauncher>,
   Option<__PrimMegaLauncher>)]` aligned with `FORWARD_TABLE` —
   per-bucket `(bb_launcher, lm_launcher)` populated when the
   canonical's `try_emit_prim_mega` succeeded for both backbone +
   lm_head, `(None, None)` otherwise (all-or-nothing tier semantics).
   `forward()` / `forward_backbone()` now `find_bucket_idx` →
   branch on `prim_mega_forced()` → call launcher pair (alloc
   tiles, run cooperative, `take_owned` terminal slot or DtoD-copy
   backbone) → else fall through to `ferrite_forward::run`. New
   `find_bucket_idx`, `PrimMegaLauncher<W>` type alias,
   `prim_mega_forced()` env-cached gate all live in
   `ferrite-forward/src/lib.rs`.
✅ 9. **`FERRITE_FORCE_PRIM_MEGA` env override (`0fb11c236`).** Co-
   dependent with step 8 (the override is the only consumer of the
   dispatch branch; the branch is dead without the override). Reads
   the env var once, caches via `OnceLock`. When set, every bucket
   whose `LAUNCHER_TABLE[bidx]` is `Some/Some` routes through the
   launcher; partial coverage panics loudly (silent host fallback
   would mask launcher correctness regressions). Today every real
   model's `try_encode_bucket` returns `None` on every canonical
   (Embed has no mega arm), so all `LAUNCHER_TABLE` entries are
   `(None, None)` and forcing panics on any real model. That's the
   intended state until DC sibling coverage widens — the infra is
   done; this gate replaces the original step-9 e2e plan, which
   depended on widening coverage first.
🟡 — **New Impls in impl_lib.rs returning `DeviceCallable +
   Primitive` fit.** Sibling for every CUTLASS launcher we have
   a DC sibling for, every ferrite-owned DC op, every FI config.
   Done (thin-delegation pattern — host-counterpart instance
   answers `matches` / `cost_us` / `fan_out` / `opcode_shape`,
   sibling overrides only `launch_kind`, `megakernel_fit`,
   `target_compatible`, mega-internal handoffs):
   - `DcRmsNormImpl` (host: `RmsNormRefImpl`) — `cf918daec`
   - `DcFusedAddRmsNormImpl` (host: `FusedAddRmsNormImpl`)
   - `DcFusedQkvRopeCacheImpl` (host: `FusedQkvRopeCacheImpl`)
   - `DcCutlassGemmImpl` × 11 tiles via `CUTLASS_DC_TILE_ZOO`
     — intersection of `CUTLASS_TILE_ZOO` ∩
     `CUTLASS_DC_GEMM_LIST` (the host CSV-calibrated zoo and the
     C++ X-macro list); a `dc_cutlass_zoo_is_subset_of_host_zoo`
     test pins the invariant so cost-lookup never misses the CSV.
   `prim_mega_compatible()` lives at `compute_capability >= 90`
   (Hopper+). The earlier `>= 80` was speculative — Ada / Ampere
   `cg::this_grid().sync()` runs ~100us/call (gmem-atomic spin
   counters), so prim-mega's per-op handoff blows out the host's
   ~5us `KernelBoundary` by 20× and host wins by construction on
   those targets. Hopper has hw-accelerated grid sync (~15us, L2-
   synchronizer + TMA fences) — that's where the older
   `worktree-ferrite-mega` branch's "100% megakernel" demo
   actually shipped. Locked by
   `cost::tests::solver_picks_dc_siblings_on_h100` (asserts at
   least one DC sibling picked on H100 LLAMA_BODY at M=1) and
   `predicted_us_includes_launch_overhead` (asserts zero DC picks
   on L4).
   Remaining: DC siblings for `FusedQkvRopePrefillImpl`,
   `CutlassGemvImpl`, silu_mul, FlashInfer attention configs;
   plus the `s2` deep-pipeline / 32×* / W8 / sm90 CUTLASS DC
   tile fan-out (Rust + C++ X-macro both need it).
⏳ 10. **End-to-end byte-equality validation.** Once one canonical
   has wholesale DC coverage (every op has a mega arm), set
   `FERRITE_FORCE_PRIM_MEGA=1` and `vllm chat unsloth/Llama-3.2-3B-
   Instruct --enforce-eager` on H100 — confirm coherent output AND
   diff token-by-token against the host path on a fixed seed.
   Today this gates on the canonical-coverage milestone (Embed,
   FusedGateUpSiluMul, attention DC arms all need to land for at
   least one forward shape). The cost-driven `pick_interpreter`
   path lights up automatically once coverage is whole.

## Next session — what to pick up

> **OBSOLETE PRE-PIVOT (2026-04-27).** This whole section described
> the original PrimMega Phase-1 push (FI DC plan-handle threading,
> FusedGateUpSiluMul decomp, occupancy-derived launch geometry).
> Superseded by the **§PIVOT 2026-04-27 — KvmMega is now the perf
> path** section at the top of this file. Read that section's
> "§Phase 2 work order — the actual entry point for the next
> session" instead.
>
> The text below is preserved for forensics — if you ever want to
> revive PrimMega coverage, this is the punch list. Do not work
> from it without reading the pivot section first.

Architecture is settled (whole-forward + tape, scaffolding for
KvmMega). DC coverage is wholesale-per-kernel for every CUTLASS
family the kernel-level Params allows. Steps 7 (program-static +
launcher emit + codegen wiring) **and 8 + 9 (forward dispatch +
`FERRITE_FORCE_PRIM_MEGA` env override)** are done. Remaining Phase-1
work, in rough order of leverage / effort:

0. **Steps 8 + 9 — forward dispatch + FERRITE_FORCE_PRIM_MEGA
   (LANDED `0fb11c236`).** Co-dependent: parallel
   `LAUNCHER_TABLE` aligned with `FORWARD_TABLE`,
   `forward()`/`forward_backbone()` branch on `prim_mega_forced()`.
   `find_bucket_idx`, `PrimMegaLauncher<W>`, `prim_mega_forced()` all
   live in `ferrite-forward/src/lib.rs`. Today every real model's
   table entries are `(None, None)` (Embed has no mega arm), so
   forcing panics on any real model — that's the intended state
   until DC sibling coverage widens. Reference notes from step 7's
   landing preserved below for the runtime-types contract; future
   launcher tweaks (grid_x/block_x/smem occupancy derivation, etc.)
   should match the same shape.

   **Runtime-types decision (settled):**

   - **`tile_table` type.** `&mut Vec<Option<TileEntry>>` from
     `ferrite_forward::tile_table::TileEntry` — exactly the host's
     `InterpreterCtx::tiles`. The launcher fn signature mirrors
     the host's `run` shape but threads tiles in by `&mut` so step
     8's `pick_interpreter` dispatch is a one-line branch off the
     same per-bucket fn body that calls `ferrite_forward::run`
     today (codegen.rs:3430). Concretely:
     ```rust
     unsafe fn prim_mega_<arch>_launch_m_<wp>(
         wm: &Weights,
         fwd: &ForwardCtx,
         device: &mut GpuDevice,
         tiles: &mut Vec<Option<TileEntry>>,
     );
     ```
     The launcher's caller (the per-canonical `forward` fn in
     codegen.rs ~3420) allocates `tiles = vec![None; num_slots]`
     once, passes it through, and `take_owned`s the terminal slot
     after — same pattern as `ferrite_forward::run`.

   - **ForwardCtx field paths.** `ForwardField::Positions` →
     `ctx.fwd.positions` (TensorView<'a>); `ForwardField::SlotMapping`
     → `ctx.fwd.slot_mapping` (TensorView<'a>). Both already match
     the encoder's `ForwardField` variants. `.raw_ptr()` returns
     the device pointer the launcher writes into pt[].

   - **CachingAllocator API.** `device.caching.alloc_tensor(&shape,
     dtype) -> OwnedTensor` is the only entrypoint the launcher
     uses. Three call sites:
     1. **Per-tile slot pre-allocation.** `tiles[slot] =
        Some(TileEntry::Owned(device.caching.alloc_tensor(&shape,
        dtype)))` — one call per active slot at launch time.
     2. **Tape buffer.** Static program is `[[i32; 32]; N]` in
        rodata; launcher allocates a device-resident copy via
        `caching.alloc_tensor(&[N, 32], DType::I32)`, memcpys the
        static in, then patches `runtime_fills`.
     3. **Workspace.** `WorkspaceKind::SplitKScratch` →
        `caching.alloc_tensor(&[split_k * M * N], DType::F32)`. M
        is `num_tokens`; (N, split_k) come from the encoded row;
        N is unblocked by the WeightShapeDim mechanism below.
     pt[] is a small device array of `*mut c_void`; allocate via
     `caching.alloc_tensor(&[ptr_plan.len() * 8], DType::U8)` then
     reinterpret, host-build the pointer list in a `Vec<*mut
     c_void>`, memcpy_htod_async to the device buffer.

   - **Slot-shape blocker resolution.** `colored_slot_map` already
     tracks `color_shape: HashMap<u32, Shape>` internally
     (interpreters/host.rs:261). Surface it: change `SlotMap` to
     carry parallel `slot_shapes: Vec<Shape>` (or return a tuple
     `(SlotMap, Vec<Shape>)` from `colored_slot_map`). Thread
     through `LoweredBucket` so step 7c's launcher emits per-slot
     pre-allocation. Small infra change; backwards compatible
     because `SlotMap::of` keeps its current contract — the
     parallel vec is additive.

   - **N/K-from-accessor (CutlassGemm + CutlassGemmSplitK +
     CutlassGemv).** Add a new `RuntimeSource` variant:
     ```rust
     RuntimeSource::WeightShapeDim {
         fn_ident: String, layer: u32, dim_idx: u8,
     }
     ```
     Encoder pushes it for the N / K placeholders that currently
     emit `Const(0)`. `assign_ptr_indices` already zeroes
     `Runtime` slots and records the patch into `runtime_fills`,
     so emission infra unchanged. Launcher's runtime-fill pass
     resolves it via `(weight_fn)(W, layer).weight.shape()[dim_idx
     as usize] as i32` and patches the device tape buffer.

   - **kv_cache K/V pointer interception (`OP_QKV_ROPE_CACHE`).**
     The encoder synthesizes `kv_cache_<layer>_{k,v}` PtrSpecs.
     Step 7c emits a special-case arm in the launcher's pt[]
     fill: pattern-match the synthetic `fn_ident` prefix, route
     to `wm.kv_cache.get_k(layer).raw_ptr()` / `.get_v(layer)`.
     The exact KvCache accessor lives on per-arch Weights via
     `ferrite_kernels::kv_cache::KvCachePool` — verify with
     `commandr` first since it's the smallest verify model.

   **Implementation order (one commit per phase, all must land
   together to avoid dead-code violations):**

   1. Surface `slot_shapes` from `colored_slot_map` →
      `LoweredBucket`. Tests on existing `colored_slot_map` cases.
   2. Add `RuntimeSource::WeightShapeDim`; rewrite
      `encode_cutlass_gemm` / `_splitk` / `_gemv` to use it for
      N / K. Update existing arm tests.
   3. Extend `WorkspaceKind::SplitKScratch` to carry resolved
      shape source (`(M_source, N_source, split_k)` triple, where
      sources are `WeightShapeDim`-style for N).
   4. Implement `emit_prim_mega_launcher(static_program_ident,
      bucket, slot_shapes, ctx) -> TokenStream` rendering the
      whole launcher fn body inline. Tests pin the outer shape
      (fn signature + ordering of pt[] fills + runtime patches).
   5. Wire into `codegen.rs` per-canonical emission alongside
      `BACKBONE_M_<wp>` / `LM_HEAD_M_<wp>` static slices: emit
      `prim_mega_backbone_m_<wp>` launcher fn when `try_encode_
      bucket` returns `Some`. Step 8 dispatch swaps
      `ferrite_forward::run(...)` ↔ launcher call based on
      `pick_interpreter`'s decision.

   The plumbing through codegen.rs is the only "one-shot" piece
   — phases 1-4 are local to `interpreters/`. Phase 5 lights up
   PrimMega for any canonical where every pick is `>=
   Primitive`-fit on Hopper. Step 9 e2e validation comes after.

3. **Widen DC sibling coverage so at least one canonical encodes
   wholesale.** Today every real model has at least one op without
   a mega arm in every bucket (Embed, FusedGateUpSiluMul,
   attention), so `try_encode_bucket` returns `None` everywhere
   and `LAUNCHER_TABLE` is all `(None, None)`. The cheapest path
   to a forced-prim-mega-runs validation is probably:
   a. Add the Embed mega arm (it's a single device-side gather; the
      shape can land as `OP_EMBED` with a runtime token-id pt[]
      pointer).
   b. Add the FlashInfer DC arm (next item — template ready).
   c. Add `CutlassFusedGateUpSiluMul` host fallback into mega via
      decomposing into `CutlassGemm × 2 + SiluMul` rows when the
      bucket is forced (costs a slot, fine for scaffolding).
   Once one canonical has wholesale arms, set
   `FERRITE_FORCE_PRIM_MEGA=1` and validate byte-equality against
   host on H100. The cost-driven `pick_interpreter` path then
   automatically lights up.

4. **FlashInfer DC arms in `prim_mega.cu` + Params marshaling.**
   `dc_flashinfer.cuh` template is ready (`cd9fd897c`); needs the
   actual switch arm in `prim_mega.cu` plus host-side Params
   marshaling — copy `plan->params_1` / `params_2` from
   `flashinfer_shim.cu.j2`'s `FlashInferPlan` to a device-resident
   buffer, encoder emits `OP_FI_PAGED_ATTN` row with pt[] indices
   for the two Params blobs + smem offset. Real piece of work
   relative to the other DC additions but unblocks attention DC.

5. **Wave-level launch counter for cost.rs.** Match the older
   branch's `cost.rs:90–112`: walk schedule, count `1 launch` per
   mega wave + `1 launch` per non-mega host pick, multiply by
   `launch_overhead_us(HostCallback, profile)` once. Improves
   `predicted_us` accuracy without changing picks. Small commit.

6. **`DcFusedQkvRopePrefillImpl`** (the prefill counterpart to
   `DcFusedQkvRopeCacheImpl`). Decode-only DC was landed earlier;
   prefill needs its own arm. Mechanical — delegate to host
   counterpart, override 4 methods.

**Won't fix without forking CUTLASS:** sm90 wgmma+TMA family,
`CutlassFusedGemmBiasImpl`, `CutlassFusedGateUpSiluMulImpl`. All
use `device::GemmUniversalAdapter` whose Params is host-only by
design. Documented in code + `feedback_gemm_universal_not_dc_able`.
Hopper-native sm90 perf path is left to KvmMega (TK templates
authored DC-friendly from the start).

## Phase 2 — work order (later)

Authoring sequence per arch:

1. Define output-tile-granular `Instruction<W>` variants matching
   vendor's existing op shapes
   (`QKV_MatMulRopeAppend(layer, batch_start, qkv_block_idx)`,
   etc.). These coexist with today's whole-kernel-call variants;
   they're picked when the solver chooses a KvmFit Impl. Host
   `eval` arms for them either panic (mega-only) or loop over
   tiles (preserves universal-IR principle but adds host-side
   work). Default to mega-only — same canonical compiled
   per-eligibility is already handled by the host/prim_mega/
   kvm_mega split.
2. Author KvmFit Impls — output-tile-granular `fan_out`,
   `release_lid` order, TK-shaped consumer/loader/storer
   templates on the .cu side, opcode-pack-position fixed in
   per-arch `mk<config, globals_arch, NoOp, ops...>`
   instantiation.
3. `interpreters/kvm_mega.rs`: encoder match for the new
   variants, scheduler (round-robin SM assignment first; smarter
   later), per-arch globals + `mk` instantiation, launcher.
4. PyVM differential testing using vendor's `python_vm.py` —
   diff the megakernel output against the PyVM trace per-row,
   per-instruction-stage. Vendor's pattern.
5. End-to-end: `vllm chat` Llama on H100, byte-equal to host /
   prim_mega.

## Known-broken: vendor's TP=8 hardcoding

The vendored cross-GPU Llama path is hard-wired to TP=8 and will
need de-hardcoding before KvmMega ships on anything other than an
8-GPU node. Concrete sites in
`vllm-rs/third_party/megakernels/`:

- `demos/cross-gpu-llama/llama.cuh:111` —
  `constexpr static int num_devices = 8;` on the Globals struct
  (compile-time constant, threads through every kernel that takes
  `Globals`).
- `demos/cross-gpu-llama/qkv_rope_append.cu:188, 287` — two
  `static_assert(Globals::num_devices == 8, "Fix this function.")`
  markers; vendor flagged the algorithmic dependency on
  `num_devices == 8` themselves.
- Python harness defaults: `megakernels/scripts/tp_generate.py:48`,
  `tp_generate_pyvm.py:23`, `tp_diff_test.py:47`,
  `demos/tp_throughput/{bench_cpp_scheduling.py:17,
  test_cpp_scheduling.py:38,144}`.

Resolution path (Phase 2 work, not Phase 1):

1. Make `num_devices` a `KvmMega` per-arch template parameter
   threaded through `globals_arch` rather than baked at the demo
   level. Ferrite emits the instantiation per the active TP
   degree it already tracks (the host pivot's TP plan in
   `project_tp_design_notes.md`).
2. Audit and rewrite the two `qkv_rope_append.cu` arms the vendor
   `static_assert`'d as 8-only. These almost certainly encode a
   head-shard / shuffle pattern that's hand-unrolled for 8.
3. Update `python_vm.py` + the TP scripts to take `num_devices`
   from the same source so PyVM differential testing keeps working
   at TP≠8.

PrimMega is unaffected — the cross-GPU demo paths are only entered
once we instantiate the KVM template.

## Cost-metric refinements (post-Phase 2)

Once the basics work, the solver cost model needs to grow to
properly value mega benefits:

- Persistent-SM cache reuse across instructions (vendor's design
  point — currently invisible to ferrite's CSV-driven cost).
- Cluster-block + DSMEM advantages (sm≥90).
- Wave-level launch counting. Today `launch_overhead_us` zeros
  out per-pick overhead for `DeviceCallable` (the older
  `worktree-ferrite-mega` branch did the same), so a chain of N
  DC ops contributes zero launch overhead even though it's one
  cooperative `cudaLaunchCooperativeKernel` (~5us). The
  approximation is fine for picks (~0.5us per pick at N=10) but
  under-counts `predicted_us`. Match the older branch's
  `cost.rs:90–112`: walk the schedule, count `1 launch` per mega
  wave + `1 launch` per non-mega host pick, multiply by
  `launch_overhead_us` once. Comes online when the per-wave
  scheduler hook (`Wave::is_megakernel`) lands.
- Per-op `Handoff::InKernelGridSync` charge inside a mega wave.
  100us/call on Ada, 15us on Hopper (target-aware in
  `Handoff::cost_us`). Currently uncharged anywhere — fine on
  Hopper where 9 syncs/layer × 32 layers ≈ 4.3ms is small
  relative to the work, less fine if PrimMega ever returns to
  Ada via a per-wave restructure that keeps grid-sync chains
  short.
- KvmFit DC ops should pay [`Handoff::Mbarrier`] (~0.1us) rather
  than the prim-mega grid-sync when running inside a KVM
  megakernel. The wave-level term should split on
  `MegakernelFit` once KvmFit impls appear.
- Per-SM tape length imbalance penalty (idle SMs at end of
  bucket).

What already lands: per-launch overhead is summed into both the
solver's DP candidate cost AND `loop_cost_us` aggregation.
HostCallback / RegularLaunch / CooperativeLaunch each pay one
`KernelBoundary` per pick (~5us — target-agnostic). DeviceCallable
pays zero per pick (wave-level overhead amortized — see
"Wave-level launch counting" above). `Handoff::InKernelGridSync` is
target-aware (100us on Ada, 15us on Hopper) for future consumers
but isn't yet summed anywhere. The combination makes DC strictly
cheaper than host at every seed where DC is feasible; feasibility
gates on `prim_mega_compatible() >= 90`, so on Ada DC is filtered
out at `target_compatible()` and on Hopper it's picked. Verified
by `cost::tests::solver_picks_dc_siblings_on_h100` and
`predicted_us_includes_launch_overhead`.

Don't bolt the rest on before Phase 2 lands.

## Things-that-must-not-happen

- **No** `mega_opcode` / `MegaOp` struct on `Implementation`.
  The encoder is one match per emitter file. Variant payload
  comes from `OpInstance` fields `fan_out` already emits.
- **No** mega wire types or encoder logic in `ferrite-forward`
  runtime. Macro-emitted launcher + vendored .cu only.
- **No** universal opcode registry on the Rust side. Per-arch
  `ops...` pack on the .cu side closes the opcode space
  template-side.
- **No** `_` arm in any encoder match. Missing arms = canonical
  ineligible at codegen, not runtime.
- **No** "compile for mega" target flag forcing solver picks.
  Solver picks by cost; interpreter selection is post-hoc.
- **No** runtime fallback ("try mega, fall back if it fails").
  Eligibility is decided once at startup per canonical.
- **No** piecemeal Impl migration. DeviceCallable wrapping audit
  is wholesale per kernel: a kernel either has both wrappings or
  the HostCallback one only. (Mirrors host pivot's wholesale
  rule.)
- **No** TK-only assumption. CUTLASS device-side and FlashInfer
  device-callable are equally valid sources of DeviceCallable
  Impls. Pick whatever's easiest to wrap first.
- **No** authoring `.cu` templates for ops Phase 1 already covers
  via DeviceCallable cutlass / flashinfer wrappings. Phase 2
  only adds template authoring where output-tile granularity
  buys real perf.

## Reference points

### Vendor

- `~/Megakernels/include/megakernel.cuh` — template entrypoint,
  `mk<config, globals, ops...>` `__global__` and warp dispatch.
- `~/Megakernels/include/config.cuh` — `INSTRUCTION_WIDTH = 32`,
  `NUM_CONSUMER_WARPS = 16`, page size, semaphore count, etc.
- `~/Megakernels/include/controller/instruction_fetch.cuh` —
  per-SM tape indexing via `get_worker_id()`. Termination
  signal: `instruction[0] == -1`.
- `~/Megakernels/megakernels/demos/throughput/instructions.py` —
  vendor's existing Llama op set
  (`PreAttnLayerNorm`, `QKV_MatMulRopeAppend`, `AttentionDecode`,
  `O_ProjResidual`, `PreMLP_Norm`, `GateSilu`, `UpMatMul`,
  `DownProjResidual`, `PreLMHeadRMS`, `LM_Head`).
- `~/Megakernels/megakernels/python_vm.py` — PyVM ground-truth
  reference for differential testing.
- `~/Megakernels/megakernels/scheduler.py` — vendor's per-SM
  assignment logic; the Rust scheduler will mirror it.
- `~/ThunderKittens/kernels/{attention,gemm}/` — H100/B200/B300
  TK kernels. Ampere unsupported as of TK 2.0 — the sm≥90 floor
  is TK's, not arbitrary.

### Tree

- `crates/ferrite-forward/src/instr.rs` — `Instruction<W>` IR.
  Encoder reads from here; nothing mega-shaped lands here.
- `crates/ferrite-forward-macro/src/impl_lib.rs:135` —
  `LaunchKind` (existing).
- `crates/ferrite-forward-macro/src/impl_lib.rs:459` —
  `launch_kind()` trait method (existing).
- `crates/ferrite-forward-macro/src/concurrency.rs:38-44,
  119-123` — Rule 4 stub. Activated by Phase 2.
- `crates/ferrite-forward-macro/src/interpreter_codegen.rs` —
  host emitter, sibling reference for prim_mega + kvm_mega.
- `crates/ferrite-forward-macro/src/target.rs` — `TargetProfile`
  including `compute_capability`. Add `kvm_compatible(&self)`
  and `prim_mega_compatible(&self)` methods here.

## Pre-commit checklist

Re-read this file. Verify:

- [ ] `cargo build -p ferrite-forward-macro` clean.
- [ ] `cargo test -p ferrite-forward-macro --lib` green
      (host-interpreter test count baseline + any new mega
      tests).
- [ ] `cargo build -p ferrite-models --features cuda` clean.
- [ ] `cargo fmt` clean and `cargo clippy --all-targets
      -D warnings` clean on every touched crate.
- [ ] If a Phase 1 commit: `vllm chat
      unsloth/Llama-3.2-3B-Instruct --prompt "why is the sky
      blue" --enforce-eager` produces coherent output AND
      `pick_interpreter` returns `PrimMega` for at least one
      bucket of the canonical. Stderr-trace the selection.
- [ ] No new methods on `Implementation` beyond
      `megakernel_fit()`.
- [ ] No `_` arm or `unsafe { unreachable_unchecked() }` in any
      mega encoder match.
- [ ] No mega wire types in `ferrite-forward`.
- [ ] If you touched the host emitter while doing mega work:
      every host pre-commit check from `HANDOFF_INTERPRETER.md`
      §Pre-commit also passes (curated golden subset, no
      regression).
- [ ] Vendor sources in `vllm-rs/third_party/{megakernels,
      thunderkittens}/` carry their LICENSE + a `VENDOR.md` with
      upstream commit hash + import date.
