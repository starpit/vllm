# TK vs Ferrite — divergence audit

Source: 7 parallel line-by-line audits of ferrite ops vs ThunderKittens reference
(`tests/batch-vm/llama_official/` for prefill, `tests/vm/llama_official/` for decode,
`prototype/vm/{config,vm,util}.cuh` for substrate).

## Severity legend
- **C0** = causes incorrect output or crashes today
- **C1** = silent latent correctness bug (works for current configs only)
- **P0** = blocks the 2× cuBLAS perf goal
- **P1** = perf hit; not blocking
- **A** = architectural/by-design

---

## CORRECTNESS (C0)

### C0-1. qkv_rope_cache: consumer→storer one-way handoff races
- **TK**: `constorer = group<NCW+1>` ping-pong via `constorer::sync(store_bar)` — storer releases consumer back AFTER `tma::store_async_read_wait()`.
- **Ferrite**: `arrive(page_done[kActPageOff])` from consumer warp 0 only. Storer never arrives back. Consumer iter N+1 can overwrite `out_bf_a/_b` before storer iter N has read it.
- File: `fused_qkv_rope_cache.cuh:285,377-378,445,491-493`
- Fix: add a `staging_consumed` mbarrier; storer arrives after `store_async_wait<0>`; consumer waits before next zero/write.

### C0-2. Lane-write staging assumes rv_fl<16> layout
- **TK**: `kittens::warp::store(sv_bf<16>, rt_bf/rv)` — primitives that handle register-tile layout.
- **Ferrite**: `if (laneid() < 16) sv[lane] = __float2bfloat16_rn(rv[0][0])` — assumes lane `l` holds element `l` in `rv[0][0]`. Bypasses TK primitive.
- Files: `fused_qkv_rope_cache.cuh:378-381`, `silu_upgate.cuh:275-277`, `gelu_upgate.cuh:246-248`
- Fix: replace with `kittens::warp::store(sv, rv)`.

### C0-3. cos_sin_cache layout suspected mismatch
- **TK prefill**: separate fp32 `g.rope_cos[pos, head_dim]` and `g.rope_sin[pos, head_dim]` tables, loaded via TMA into scratch+8192 / scratch+8192+sizeof(sv_fl<128>).
- **Ferrite**: combined bf16 `cos_sin_cache[pos, HEAD_DIM]` where `[0..HALF_DIM)`=cos, `[HALF_DIM..HEAD_DIM)`=sin. Loaded into a page (not scratch).
- File: `fused_qkv_rope_cache.cuh:166-181, 281-282`
- Fix: verify host builder emits this exact layout; if not, fix host or kernel to agree.

### C0-4. attention SPLITS>1 reduction is a stub
- **Ferrite**: `attention_reduction.cuh` body = `static_assert(SPLITS>1) ... (void)args; return;`. LSE never produced from partial.
- **Ferrite**: `attention_partial.cuh` static_asserts SPLITS==1 in 4 places — any code path selecting SPLITS>1 breaks.
- Fix: gate dispatch table to never pick SPLITS>1 buckets, OR implement reduction. Pick gating for now.

### C0-5. Vector attention silently broken for HEAD_DIM != 64
- **Ferrite**: vector path indexes `Q_vecs[h].data[0][0]` and `data[1][0]` → assumes 32 lanes × 2 elements = HEAD_DIM=64. No static_assert. LLaMA-3-8B (HEAD_DIM=128) silently processes only first 64 elements.
- File: `attention_partial.cuh:502-503,565-566,604-607,624-625`
- Fix: `static_assert(HEAD_DIM == 64)` on the vector path entry.

### C0-6. Header comment lies about caps in attention
- `attention_partial.cuh:18-23` claims `SLIDING_WINDOW==0` and `HAS_SOFTCAP==0` caps. Code has full untested implementations gated on those template params.
- Fix: add the missing static_asserts OR remove the misleading comment AND test the paths.

---

## CORRECTNESS-LATENT (C1)

### C1-1. attention_partial mbarrier: per-row TMAs vs whole-tile TMA
- **TK**: 1 `tma::load_async<dim::DEPTH>` per K/V page, descriptor for full `st_bf<KV_BLOCK_SIZE, HEAD_DIM>` tile.
- **Ferrite**: 16 row TMAs per page, each contributing `row_bytes` to a single `expect_bytes(page_bytes)` mbarrier. Sum equals expected (16 × HEAD_DIM × 2 == BLOCK_SIZE × HEAD_DIM × 2) — works, but fragile.
- Severity is C1 because correct-by-arithmetic but not by TK contract.

### C1-2. Page semaphore arrival count differs from TK
- **TK**: `page_finished[pid][i]` init'd with count `NCW * (1<<i)` — every consumer warp arrives once.
- **Ferrite**: `page_done[s]` init'd to count 1 — only one warp leader arrives. Fragile against future ops that need cross-consumer synchronization through the page.

### C1-3. No `fence.proxy.async.shared::cta` at boot
- **TK**: explicit proxy fence after semaphore init so TMA loads see the sem state.
- **Ferrite**: only `__syncthreads()`. May work today because loader's first TMA happens after enough other syncthreads, but not guaranteed by spec.

### C1-4. argmax assumes `blockIdx.x==0` without enforcing it
- `argmax.cuh` is documented as CTA-0 only but has no internal guard. Silent UB if invoked from another CTA.

---

## PERFORMANCE (P0 — blocks 2× cuBLAS goal)

### P0-1. No K-pipeline anywhere
- **TK matmul_pipeline**: `INPUT_PIPELINE_STAGES=3`, separate `inputs_arrived[i]` and `inputs_finished[i]` per stage, K-chunks streamed concurrently with launcher MMA.
- **TK matvec_pipeline**: `INPUT_PIPELINE_STAGES=3`, `OUTPUT_PIPELINE_STAGES=3`, `STAGE_PAGES=4` (16 KB weight slice per stage).
- **Ferrite gemm**: 2-phase double buffer.
- **Ferrite gemv**: STAGES = `Config::INSTRUCTION_PIPE_STAGES` (substrate-wide, currently 2-3).
- **Ferrite qkv_rope_cache**: `STAGES=1` hardcoded — no overlap between weight TMA and matvec compute.
- **Ferrite down_proj**: no stages at all.

### P0-2. gemm loader uses raw float4 stores, NOT TMA
- TK loader: lane 0 issues 3 TMA descriptors per stage for full A (128×64) + B (256×64) tiles.
- Ferrite gemm loader: 32 lanes do raw `float4` element loads. Orders of magnitude bandwidth gap.
- File: `gemm_bf16.cuh:138,173,180`

### P0-3. No `tma::store_add_async` / no async-store pipeline
- TK matmul_adds + matvec_adds: residual fused into store via `tma::store_add_async`.
- Ferrite down_proj_residual storer: scalar lane-0 read-modify-write of bf16 (`down_proj_residual.cuh:233-238`). Worst-case perf for residual update.
- TK constorer round-robin: warp i stores into shared scratch, hands off to storer warp via named-bar, storer issues `tma::store_async`, releases warp i back.
- Ferrite: no constorer. Either direct global writes from consumer (gemm) or single-storer serial issue.

### P0-4. gemm per-CTA tile is 64×64 (vs TK's 128×256)
- 8× less work per CTA → 8× more launches, 8× more scheduling overhead.

### P0-5. attention_partial KV pipeline depth
- TK: NUM_STAGES=3 (decode), 6 (prefill).
- Ferrite: STAGES=2 (locked in `variant_cpp.rs::op_page_count`).

### P0-6. attention_partial per-row TMAs
- 16× more TMA descriptor pressure than TK.

### P0-7. No fused up_proj / down_proj
- TK up_matmul steals unreleased pipeline pages via `get_used_page_at(2 + 2*laneid())`, fuses post-matmul × silu_out in same kernel.
- Ferrite: separate kernel launches.

### P0-8. RMS not fused into matvec consumer (decode)
- TK decode: `rms_matvec_pipeline` fuses RMS into matvec consumer — no gmem round-trip for normalised activation.
- Ferrite decode: standalone `rms_norm` op, then gemv — full gmem round-trip.

### P0-9. lm_head granularity
- TK prefill: 128×256 GEMM tile per CTA (TMEM/WGMMA).
- TK decode: 16-element vocab block per inner iter.
- Ferrite: 1 bf16 logit per CTA per iter, scalar store. ~16× to ~256× less work per CTA.

---

## PERFORMANCE (P1)

### P1-1. RMS weight reload per iter (decode persistent loop)
- TK: weight loaded once per CTA per row.
- Ferrite: reloaded every iter. Documented hoisting opportunity.

### P1-2. EVICT_FIRST cache policy missing
- TK decode: explicit `cache_policy::EVICT_FIRST` on weight TMA loads.
- Ferrite: defaults.

### P1-3. Scalar shfl_xor butterfly redundancy in vector attn

### P1-4. argmax uses `__shfl_xor_sync` despite header comment claim of "no `__shfl`"

---

## ARCHITECTURE (A — by design, document)

These reflect ferrite's deliberate deviation from TK's runtime VM model:

- A-1. No instruction stream — straight-line codegen instead.
- A-2. No controller warp role (4 roles: loader/launcher/consumer/storer).
- A-3. No `DYNAMIC_SEMAPHORES[32]` per stage — only `page_ready/page_done/page_consumed`.
- A-4. No logical→physical `pid_order` page mapping.
- A-5. No `INSTRUCTION_PIPELINE_STAGES_BITS` parity-bit page reuse.
- A-6. No `Bar[layer][opcode][row][col]` 4-D atomic counter — flat `barriers[edge_idx]`.
- A-7. NCW=8/4/2/1 vs TK's hardcoded 16 (with `static_assert(NCW==16)` in TK's matmul_pipeline).
- A-8. No `TEVENT_*` / `s.record()` per-event timing.
- A-9. Persistent grid-stride loops vs TK's per-instruction CTAs.
- A-10. Paged KV cache (`block_table[token]`) vs TK's contiguous `[layer, block, kv_head, head_dim]`.
- A-11. Per-token `seq_lens` vs TK's global `pos_id`.

These collectively mean ferrite cannot be a drop-in port of TK ops. Each op needs to be re-derived for ferrite's substrate, not transliterated.

---

## FIX ORDER

Phase 1 (correctness — get coherent output back):
1. C0-2 lane-write → `warp::store` everywhere.
2. C0-3 cos_sin layout — verify and align host/kernel.
3. C0-1 qkv_rope_cache back-pressure mbarrier.
4. C0-5 vector attn HEAD_DIM gate.
5. C0-4 SPLITS>1 dispatch gate.
6. C0-6 attention header comment vs static_assert reconciliation.

Phase 2 (perf — go after the 2× goal):
1. P0-2 gemm TMA loader (huge win).
2. P0-3 async store pipeline + tma::store_add_async for down_proj.
3. P0-1 K-pipeline depth ≥ 3 in qkv, gemv, gemm.
4. P0-6 attention whole-tile TMA.
5. P0-4 gemm 128×256 tile (or at least 128×128).
6. P0-9 lm_head 16-vocab-block granularity.
7. P0-7 fused up_proj / down_proj page-stealing.
8. P0-8 fused RMS+matvec for decode.
