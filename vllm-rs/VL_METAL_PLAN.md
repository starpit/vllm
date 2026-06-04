# VL on ferrite-metal — Implementation Plan

**Branch:** `worktree-vl-metal` (off `worktree-ferrite-metal` @ `7cd1cc348`, which carries the working Qwen3.5 GDN-hybrid text backbone on metal).
**Goal:** bring Vision-Language *inference* (vision tower + multimodal pipeline) to the ferrite-metal (Apple Silicon) backend. Today the **text** backbones of VL models build on metal, but the **vision tower + multimodal dispatch** are CUDA-only.

> Produced by an 8-survey → architect → adversarial-critique → revise workflow, then key facts independently re-verified against the codebase + HF configs (2026-06-02). The architect's first-draft thesis ("metal text interpreter does 95% of the work; only 3 new kernels") was **false** and has been replaced; this is the corrected plan.

---

## 1. Corrected thesis: what is reusable vs net-new

**Reusable as-is (verified):**
- The `#[vision_forward]` **macro IR pipeline** (parse → classify → FUF unroll → solver → schedule → Instruction tape) is the *same* path as the text `#[forward]`; the vision `Instruction` variants (`VisionRope`, `VarlenAttention`, `LoadPixels`, `PosEmbed`, `Gelu`, `QuickGelu`, `Pool2d`, `SpliceMmEmbeds`) are **not** `cfg`-gated — they already exist on metal builds.
- Pure-CPU host glue in `ferrite-vision` (image preprocess, `build_rope_cos_sin_bf16`, `build_cu_seqlens_i32`, patch flatten) is backend-neutral.
- The text decoder *after* the embed-splice is proven (== mlx-lm verbatim at HEAD, cos 1.0).
- Generic metal kernels: `gemm`, `add`/`mul`/`sub`/`bias_add`/`silu_mul` (elementwise.metal), `gelu_tanh` (activation.metal), `rmsnorm` (text + Qwen2.5-VL ViT), reshape/alias/free no-ops.

**The catch — what is genuinely net-new on metal (the bulk of the work):**
- **Zero** metal lowering arms and **zero** `ferrite-forward-macro/src/metal/` adapters exist for *any* vision Instruction. The IR is portable; the metal **codegen for it is not written**. Every vision op currently hits the `UnsupportedVariant` catch-all (`lowering.rs:3074`).
- The **dispatch layer** is `#[cfg(cuda)]`: `MultimodalForward` trait (`ferrite-forward/src/lib.rs:908`), `FerriteModel.mm`, `run_mm_vision_forward(&CudaModel)` (`ferrite_worker.rs:2209`), `try_load_mm` / `FerriteMmRegistration`.
- The two ops the first draft called "reuse" **do not exist on metal**: `embedding_gather_masked` (CUDA-only, `kernels.rs:2637`) and the host-upload primitive `alloc_gpu_tensor_from_host` (CUDA `GpuDevice` only; metal uploads go through `MetalAllocator` `StorageModeShared` + `.contents()` memcpy, `worker.rs:700-737`).

---

## 2. Verified ground truth (re-confirmed 2026-06-02)

**Bring-up target = `mlx-community/Qwen3.5-9B-MLX-4bit` (Qwen3.5-VL-9B).** Verified from its HF config + safetensors index:
- ViT `vision_config`: depth **27**, hidden **1152**, heads **16** → **head_dim 72**, intermediate 4304, patch 16, spatial_merge 2, out_hidden **4096**, `gelu_pytorch_tanh`, `num_position_embeddings` 2304 (learned pos-embed present), `deepstack_visual_indexes=[]` (no deepstack), **no `window_size`** (no window attention).
- Weight prefix is **`vision_tower.*`** (NOT `visual.*`).
- ViT is **UNQUANTIZED** — 0 `.scales`/`.biases` markers on vision weights (dense bf16/f16; only the *text* decoder is 4-bit). This **de-risks** the loader (no affine-quant vision GEMM).
- ViT norms are **LayerNorm-WITH-BIAS** (`blocks.N.norm1/norm2`, `merger.norm` each carry `.weight` **and** `.bias`) — NOT RMSNorm. No metal LayerNorm matcher exists → net-new.
- Block structure: `attn.qkv` (fused), `attn.proj`, `mlp.linear_fc1`→gelu→`mlp.linear_fc2` (plain 2-layer, NOT SwiGLU), `patch_embed.proj`, `pos_embed.weight` [2304,1152], `merger.linear_fc1/fc2/norm`.
- Text backbone == our working GDN hybrid on metal.

**Oracle — RESOLVED (2026-06-02):** `~/git/mlx-vlm` (HEAD `549a75a`) is the Mac-native, exact-model, **runnable** reference (no torch needed). `mlx_vlm/models/qwen3_5/` = GDN text (`language.py` + `gated_delta.py`) + vision (`vision.py` re-exports `qwen3_vl/vision.py`, the real 439-line ViT) + `qwen3_5.py` (merge + mrope). (mlx-**lm** is text-only — every `*_vl.py` there pops `vision_tower`; the vision towers live only in mlx-**vlm**.) This gives both the authoritative DSL structure AND per-stage numeric parity by running the real 9B weights on this Mac. Authoritative facts read from it:
- **rope = `rotate_half` / GPT-NeoX** (`vision.py:29,50`) — resolves the interleave ambiguity (was a silent-garbage risk).
- **`mrope_section = [11, 11, 10]`** (`config.py:84`) — the band-split, now pinned.
- **deepstack hard-disabled** for qwen3.5 (`config.py:52-58` forces `deepstack_visual_indexes=[]`).
- **PosEmbed = `fast_pos_embed_interpolate`** (`vision.py:293`): a 4-corner **bilinear interpolation** over a 48×48 (=√2304) learned grid (`pos_embed(idx)*weight` for 4 corners, summed), then tiled/permuted per `grid_thw` — NOT a plain row-gather. Cheap + per-image → compute **host-side** and upload like the cos/sin tables (no special kernel).
- **PatchEmbed = `nn.Conv3d`** with kernel=stride=patch (non-overlapping) → reduces to reshape + dense `gemm` (reuse gemm).
- **Attention** splits q/k/v by `cu_seqlens` then per-segment SDPA (bidirectional, MHA, no window); blocks = LayerNorm(eps 1e-6)→attn→res→LayerNorm→MLP(fc1→gelu→fc2)→res.
- **Image merge = `masked_scatter`** (`qwen3_5.py:114`): replace `inputs_embeds` rows where `input_ids == image_token_index` with the vision features; `get_rope_index` builds the [3,n] mrope positions.

**Qwen2.5-VL (second vehicle, better-verifiable):** on-disk config gives vision_embed 1280, **head_dim 80**, intermediate 3420→**padded 3424**, `window_size` 112, `fullatt_block_indexes=[7,15,23,31]`; ViT uses RMSNorm + SwiGLU (both already on metal). HF transformers definitely supports `qwen2_5_vl`, so it is the cross-check oracle if torch can be installed.

---

## 3. Kernel inventory

### Exists on metal (reuse)
| op | shader | note |
|---|---|---|
| dense gemm (bf16/f16) | `gemm.metal` | patch_embed.proj, fused qkv, proj, fc1/fc2, merger — ViT is unquantized |
| residual add / mul / sub | `elementwise.metal` | residuals; `sub` = one LayerNorm tile |
| bias_add | `elementwise.metal` | qkv/proj/fc/patch_embed/merger biases (LN bias is separate) |
| gelu tanh | `activation.metal` `gelu_tanh_bf16` | exact `gelu_pytorch_tanh` |
| rmsnorm | `rmsnorm.metal` | text decoder + Qwen2.5-VL ViT (NOT Qwen3.5-VL ViT) |
| reshape/alias/free | lowering no-ops (`lowering.rs:3063`) | window-permute, merger reshape, rank2↔rank3 |
| cu_seqlens segment binding | pattern from `gdn_scan_varlen.metal:52` | proves device-int cu_seqlens binding + per-segment bos/eos |
| host upload | `MetalAllocator` StorageModeShared (`worker.rs:700-737`) | the real upload path (NOT `alloc_gpu_tensor_from_host`) |

### Net-new metal kernels / codegen
1. **`vision_rope_2d.metal`** — 2D pair-rotation of q,k reading **two** separate cos/sin buffers `[total_l, half_rot=36]`, index `token*half_rot+hi` (matches host `build_rope_cos_sin_bf16` + CUDA `vision_rope_apply_kernel` `cs_idx=token*half+hi`). Drop the text kernel's single interleaved `cos_sin[pos*rot_dim]`, the paged-KV write (buffers 5-7), and the GQA `group_r` logic (vision is MHA). `HEAD_DIM` a function-constant (72). Clone from `rope.metal` Q-rotation block (~297-307).
2. **`vision_varlen_attn.metal`** *(HIGH RISK)* — cacheless **bidirectional** SDPA over `cu_seqlens` segments. Required because every wired metal attn kernel is **paged-cache + causal** (`AttentionPrefillPaged` binds block_table/K/V), and head_dim 72/80 are absent from `STEEL_PAGED_HEAD_DIMS=[64,96,128,256]`, from non-paged `attention_steel.metal` (bd128 only), and fail `attention_via_cache_v2`'s `head_dim%32==0`. MVP: simple per-(head, query-token) over-segment online-softmax, `head_dim` as `function_constant(0)` so 72/80 need no template explosion; bind `cu_seqlens` at buffer(5). Perf path later = FA2 + steel bd72/bd80 build.rs instantiation. Reference: segment loop of `gdn_scan_varlen.metal` + online-softmax math of `attention_steel.metal`.
3. **PosEmbed (host-side, NO new kernel for the 9B)** — `fast_pos_embed_interpolate` is a 4-corner bilinear interpolation over the 48×48 learned grid; it is cheap + per-image, so compute it **host-side** (like the cos/sin tables), upload the resulting `[total_l, 1152]` buffer, and add it with the existing elementwise `add`. *Net-new `embed_gather.metal` is only needed for Qwen2.5-VL (P5) window_index/reverse_indices permute — defer it to P5.* (The CUDA `embedding_gather_masked` has no metal twin, but the 9B path sidesteps it.)
4. **`vision_layernorm`** (fused, preferred) **or** a 4-tile lowering (mean / sub / normalize-over-centered / scale+bias) — LayerNorm-with-bias; no metal matcher today. Net-new mean+var single pass for fidelity. Clone `rmsnorm.metal` + a mean reduction.
5. **`embed_splice`** — per-`EmbedPatch` D2D copy of `[length, 4096]` bf16 rows from `mm_embeds` into the Embed-gather output at `token_offset`. Metal has no `compute_stream`; implement as an `MTLBlitCommandEncoder` copy loop sequenced after the Embed dispatch (mind the compute↔blit encoder boundary), or a 1-thread-per-row copy kernel inside the compute encoder. Mirror CUDA `instr.rs:1279-1307`.
6. **`vision_mrope`** — band-split rope: reads positions as `[3, n_tokens]` + `mrope_section [u32;3]` function-constants, applies **per-band (t/h/w)** rotation (NOT three full ropes — that is numerically wrong). Distinct kernel symbol from the text rope's `ROPE_*` constants. `mrope_section` is baked in model code, not config JSON — pin in P-1.
7. *(Qwen2.5-VL)* window-attention dispatch: extend the `VarlenAttention` lowering to pick `vision_cu_seqlens_full` vs `_window` by `kind∈{1,2}`; reuse `embed_gather` for permutes; head_dim 80 falls out of the function-constant attn kernel.
8. *(Gemma3-MM, deprioritized)* `avg_pool_2d.metal` — 4×4 spatial average for the SigLIP projector.

Plus: **metal lowering arms + `ferrite-forward-macro/src/metal/` adapters for every vision Instruction** (none exist), and **RuntimeBindingKind::Vision\*** entries in `runtime.rs`/`kernel_bindings.rs`.

---

## 4. Phases (incrementally verifiable on the Mac)

### P-1 — Set up the mlx-vlm parity harness *(risk: low — oracle RESOLVED)*
The oracle is `~/git/mlx-vlm/mlx_vlm/models/qwen3_5` (runs on this Mac, no torch). P-1 is now harness setup, not a search.
- `uv pip install -e ~/git/mlx-vlm` (or add to PYTHONPATH); confirm it loads `mlx-community/Qwen3.5-9B-MLX-4bit` and runs a single-image forward on the Mac.
- Add per-stage tensor dumps to a local copy of `qwen3_vl/vision.py` + `qwen3_5.py` (post patch+pos_embed, post block-0/13, post merger, post masked_scatter, the [3,n] mrope positions). Save as committed golden fixtures for a fixed image (e.g. red_circle_224.png) + fixed seed.
- Transcribe the exact ViT forward from `qwen3_vl/vision.py` (already read: rotate_half rope, LayerNorm eps 1e-6, Conv3d patch, fast_pos_embed_interpolate, per-segment cu_seqlens SDPA, PatchMerger) and the merge/mrope from `qwen3_5.py` (masked_scatter, get_rope_index, mrope_section [11,11,10]).
- **Artifact:** `ORACLE.md` recording the mlx-vlm commit, the dump points, the golden fixtures, and the pinned dims/layouts (head_dim 72, two-buffer cos/sin `token*half_rot+hi`, mrope [11,11,10]). No torch/synthetic-golden fallback needed; CUDA parity is a bonus cross-check only.

### P0 — De-CUDA-gate the dispatch + wiring surface (stub vision_forward) *(risk: medium)*
- `MultimodalForward` trait + `ForwardCtx` vision fields: `#[cfg(cuda)]` → `#[cfg(any(cuda, metal))]`.
- `ferrite-vision`: add a `metal` feature; keep only `pad_linear_k_to_mult8` + `TraceDump` cuda-gated; **add a metal upload shim** (`MetalAllocator` StorageModeShared) behind a backend-neutral method and route `vision_arch.rs`'s `alloc_gpu_tensor_from_host` calls (lines 157/167/213/218/244/251/256/261) through it. *This is the load-bearing refactor the draft mislabeled bookkeeping.*
- `FerriteModel` + `mm` field → `#[cfg(any(cuda, metal))]`; add `#[cfg(metal)] run_mm_vision_forward_metal` stub.
- `engine.rs` image extraction → `#[cfg(all(multimodal, any(cuda, metal)))]` (pure CPU).
- New `ferrite-model-qwen3-5-vl` crate: stub VisionWrapper + PROCESSOR + config JSON (from verified vision_config) — enough to link.
- The stub returns a **deterministic per-row signature** (row r = f(r)), not all-ones, so P4's splice can be smoke-tested for correct offsets.
- **Verify:** `FERRITE_MODELS=qwen3_5_vl cargo build --bin vllm -Fmetal` links.

### P0b — Synthetic-golden harness *(only if P-1 found no Python/CUDA oracle; risk: medium)*
- `vision_ref.rs`: pure-Rust CPU refs (`rope_2d_ref` indexing `token*half_rot+hi`, `bidir_sdpa_ref` block-diagonal softmax, `layernorm_bias_ref`, `posembed_gather_ref`), each cross-derived from the CUDA kernel math **and** the spec to reduce false-green.
- Freeze fixed-seed fixtures. Document in `ORACLE.md` that a synthetic ref can share a layout misconception; whole-ViT cosine vs CUDA (if reachable) is the strong check.

### P1 — Golden kernel: `vision_rope_2d` *(risk: medium)*
- Write `vision_rope_2d.metal`; add `I::VisionRope` lowering arm + macro adapter + `RuntimeBindingKind::{VisionRopeCos,VisionRopeSin}`; reuse `build_rope_cos_sin_bf16`.
- **Verify:** `#[cfg(metal)]` test on 16-tok×16-head×72-dim; cosine > 0.999 vs P-1 oracle **AND** assert the CUDA `cs_idx=token*half+hi` layout (catch the stride trap).

### P2 — Golden kernel: non-paged bidirectional varlen attention @ head_dim 72 *(risk: high)*
- Write `vision_varlen_attn.metal` (function-constant head_dim); `I::VarlenAttention` lowering arm + macro adapter + `RuntimeBindingKind::VisionCuSeqlens`; rank2↔rank3 reshape in the arm.
- **Verify:** two synthetic images, `cu_seqlens=[0,n0,n0+n1]`; cosine > 0.999 vs oracle dense block-diagonal softmax; assert **no cross-segment leakage**.

### P2.5 — Vision-tower loader (dense, unquantized) *(risk: medium)*
- Sniff prefix `vision_tower.*`; map fused `attn.qkv`, `attn.proj`, `mlp.linear_fc1/fc2`, `norm1/norm2` (weight+bias = LayerNorm), `patch_embed.proj`, `pos_embed.weight` [2304,1152], `merger.*`. **No affine-quant path** (verified unquantized).
- **Verify:** load `blocks.0.attn.qkv.weight` + `pos_embed.weight`, D2H a slice, byte-equal vs source safetensors. Gates P3.

### P3 — Full ViT forward + LayerNorm chain + PosEmbed + per-stage cosine *(risk: high)*
- Transcribe the `#[vision_forward]` body exactly from the P-1 oracle: `patch_embed → +pos_embed → 27×{LN(norm1,bias) → vision_rope_2d → varlen_attn(qkv, kind=0) → proj → residual → LN(norm2,bias) → fc1 → gelu_tanh → fc2 → residual} → merger{LN → fc1 → gelu_tanh → fc2}`. **Confirm from the oracle whether pos_embed AND rope both apply, the order, and the interleave (NeoX vs GPT-J)** — wrong = silent garbage.
- Lowering + macro adapters for `LoadPixels`, `PosEmbed`(embed_gather), `Gelu`(gelu_tanh), the LayerNorm-with-bias path.
- `run_mm_vision_forward_metal`: a **second baked metal tape** — the ViT is its own arch with its own `CanonicalParams`, lowering, arena, and per-bucket MTL4 ICBs keyed by vision-token buckets `[256,1024,4096,16384]` (full parallel baking pipeline, not `run_bucket_mtl4` over an arbitrary tape).
- **Net-new metal vision D2H readback helper** (none exists) for the dumps.
- **Verify:** per-stage cosine (post patch+pos_embed / LN-0 / block-0 / block-13 / merger) > 0.99 vs P-1 oracle on a fixed image. If no Python/CUDA oracle: degrade to per-stage golden for the already-tested kernels + magnitude/drift sanity + the P4 caption (explicitly weak; record in ORACLE.md).

### P4 — Embed splice + MRoPE (band-split) + coherent caption *(risk: high)*
- `embed_splice` via blit loop; `I::SpliceMmEmbeds` arm + `RuntimeBindingKind::MmEmbeds`. **The `mm_embeds` buffer must be worker-held (not arena)** so it outlives the blit — use-after-free here is silent corruption.
- `vision_mrope` band-split kernel; reads `[3,n_tokens]` + `mrope_section`. Drop the draft's "three sequential RopeAppend" (numerically wrong).
- Worker: `forward_with_metal_followup` populates `ctx.mm_embeds`/`ctx.embed_patches` (currently hard-None at `lib.rs:385`); `build_mrope_positions_2d` → `#[cfg(any(cuda,metal))]`; cached-prefix rebuilds grid for trailing-token MRoPE.
- **Verify:** new `#[cfg(metal)] e_qwen3_5_vl_metal.rs` — POST an OpenAI chat with a red-circle image, assert the caption mentions red/circle. **This also exercises net-new text-side MRoPE + splice** (the "text == mlx-lm" guarantee does NOT cover these), so also assert prefill-logits cosine vs CUDA if reachable, and that a text-only request stays coherent through the same decoder.

### P5 — Qwen2.5-VL second vehicle (head_dim 80 + window attention) *(risk: high)*
- head_dim 80 falls out of the function-constant attn kernel; `I::VarlenAttention` handles `kind∈{1,2}` (full/window) + `RuntimeBindingKind::{VisionCuSeqlensWindow,VisionWindowIndex,VisionReverseIndices}`; reuse `embed_gather` for permutes.
- ViT uses RMSNorm + SwiGLU (already on metal). **`intermediate_size` 3420→padded 3424** — gate/up GEMM widths must match the padded weight layout (same hazard class as the qmv K-tail over-read).
- **Verify:** coherent caption on Qwen2.5-VL-3B; per-layer cosine on `fullatt=[7,15,23,31]` vs HF transformers (the certain oracle). Use Qwen2.5-VL as the cross-check for the whole VL path.

---

## 5. Metal multimodal worker wiring (step by step)
1. **Load:** build a metal `VisionWrapper<Qwen35VlVisionWeights>` from dense `vision_tower.*`; store `FerriteModel.mm = Some(...)`.
2. **Per request:** CPU pipeline (preprocess + `engine.rs` image parsing) → `MultimodalData` + `EmbedPatch` placeholders.
3. **`run_mm_vision_forward_metal`:** CPU helpers `build_rope_cos_sin_bf16` (two `[total_l,half_rot]` buffers), `build_cu_seqlens_i32`, position_ids, patch flatten.
4. **Upload** each table via the metal `MetalAllocator` shared-buffer shim (NOT `alloc_gpu_tensor_from_host`); write `vision_*` onto a vision-scoped ForwardCtx.
5. **Drive the ViT** via a separate baked metal worker (own CanonicalParams/lowering/arena/ICBs, vision-token buckets).
6. `Binding::Runtime{ Vision* }` resolves against the live vision ForwardCtx (mirror InputIds/Positions in `pool.rs`/`runtime.rs`); panic-on-null.
7. ViT outputs `mm_embeds [total_mm_tokens, 4096]` into a **worker-held** MTLBuffer (outlives the splice). Return `(mm_embeds, Vec<EmbedPatch> with grid_t/h/w)`.
8. Text `forward_with_metal_followup` (~`ferrite_worker.rs:8275`): set `ctx.mm_embeds`/`ctx.embed_patches` (currently None at `lib.rs:385-386`).
9. **MRoPE:** `build_mrope_positions_2d` → `[3,n_tokens]`; the band-split kernel applies per-band rotation.
10. **Run text tape:** Embed gathers text embeddings; `SpliceMmEmbeds` blits each patch's `[length,4096]` from `mm_embeds` at `token_offset`; the GDN decoder runs unchanged.
11. **Decode steps:** ViT does not rerun; on cached-prefix rebuild grid metadata so trailing-token MRoPE matches the encoder.

---

## 6. Verification strategy (3-tier, oracle = mlx-vlm)
- **Oracle = `~/git/mlx-vlm/qwen3_5`** running the real Qwen3.5-9B on this Mac (no torch). P-1 stands up per-stage dumps as committed golden fixtures.
- **Tier 1 (per-kernel golden, P1/P2):** standalone `#[cfg(metal)]` tests vs the oracle; rope test also asserts the CUDA `cs_idx` layout; attn test asserts no cross-segment leakage.
- **Tier 2 (per-stage cosine, P3):** D2H dumps (needs the net-new readback helper) cosine > 0.99; degrades to per-stage golden + drift sanity if no Python/CUDA oracle.
- **Tier 3 (e2e caption, P4/P5):** new metal e2e tests; note the caption also exercises net-new MRoPE+splice, so cross-check prefill logits vs CUDA if reachable.
- **Qwen2.5-VL is the better-verifiable vehicle** (HF support certain) and serves as the cross-check even though Qwen3.5-VL-9B is brought up first.

---

## 7. Open risks (highest first)
> RESOLVED since the first draft: the **oracle** (was #1) — `~/git/mlx-vlm/qwen3_5` runs the real 9B on this Mac; per-stage parity is available, no torch/synthetic-golden needed. The **rope interleave** (was #4) — confirmed `rotate_half`/NeoX. **`mrope_section`** = [11,11,10] (pinned). **PosEmbed** moved host-side (no kernel). The attention/LayerNorm/MRoPE kernels are now the dominant risks.
1. **Non-paged bidirectional varlen attention @ head_dim 72/80** is fully net-new (every wired metal attn is paged+causal; 72/80 uninstantiated everywhere). Dominant kernel risk (~1–1.5 wk).
2. **LayerNorm-with-bias** is net-new (no metal matcher); wrong centering/variance = silent garbage. (Now golden-checkable against mlx-vlm per-stage dumps.)
3. **MRoPE band-split** (`mrope_section=[11,11,10]`) has no metal consumer; the "three sequential RopeAppend" MVP is numerically wrong — must be per-band. (Now checkable against `get_rope_index` + a per-stage dump.)
4. **`mm_embeds` buffer lifecycle** across the compute↔blit boundary — worker-held; use-after-free is silent corruption.
5. **Metal vision D2H readback** for per-stage dumps doesn't exist — must be built before P3 is verifiable (now the primary verification mechanism, so build it early).
6. **Second-worker baking** (own CanonicalParams/lowering/arena/ICBs) is a full parallel pipeline.
7. **PatchEmbed Conv3d → gemm** equivalence (non-overlapping kernel=stride) must reproduce mlx-vlm's patch flattening order exactly (channel/temporal/spatial layout) — golden-check the post-patch_embed dump.
8. Known multi-seq text **batch-non-invariance** may resurface with longer/staggered vision sequences (out of scope for first caption).

---

## 8. Effort
Realistically **4–6 focused weeks** to a coherent Qwen3.5-VL-9B caption on metal (P-1…P4), + ~1 week for Qwen2.5-VL window attention (P5). Critical path: **P-1 (oracle) → P2 (varlen attn @72) → P2.5 (loader) → P3 (full ViT + LayerNorm + per-stage parity) → P4 (splice + band-split MRoPE + e2e)**. De-risked vs the first estimate by the verified unquantized vision tower; re-risked up by the net-new attention/LayerNorm/MRoPE kernels, the absent oracle, the metal D2H helper, and the second baking pipeline. Everything downstream of the splice is proven *except the splice and MRoPE themselves*, which the text path has never exercised.
