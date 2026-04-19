# Stencil IR — design & vocabulary freeze

Status: pre-implementation design. Dated 2026-04-19.

Sits between FUF (op-level math DAG, fully unrolled, const-propped) and megakernel codegen. Replaces ad-hoc wavefront emission with an explicit IR that makes warp specialization, load/compute overlap, and per-arch lowering fall out of a small fixed vocabulary.

Source of the "megakernel evidence" below: the HazyResearch Megakernels repo at `~/Megakernels/` (SM90 single-kernel LLM inference, built on ThunderKittens).

---

## 1. IR stack

Four layers. Arch differences live only in layer 3.

1. **FUF** — op-level math DAG, fully unrolled across layers. Const-prop natural. *(existing; see `project_fuf_compiler_architecture.md`)*
2. **Stencil IR** — regions + affine iteration domain + role-annotated nodes + typed dep edges. Arch-neutral.
3. **Resource mapping** — per-arch pass that assigns roles to hardware units (warpgroups, CTAs, pipeline stages). Table-driven.
4. **Emitter** — straight-line PTX/SASS/CUDA. Mechanical once layer 3 is decided.

**Region** = straight-line stencil with:
- Iteration domain (e.g. `(q_tile, kv_tile, head_group)`) with optional affine predicates
- Nodes, each with a `role` and a reference to the underlying FUF math op
- Typed dep edges carrying a dep vector on the iteration domain

**Megakernel** = CFG of regions; edges between regions carry a barrier kind. Data-dependent branches appear only on inter-region edges, never inside regions.

---

## 2. Role vocabulary (frozen at 3)

Roles answer *"which hardware unit does this node go on?"* — nothing else. Math never becomes a role.

| Role | What | SM90 lowering | SM89 lowering |
|---|---|---|---|
| `Load` | GMEM → SMEM tile move | dedicated TMA producer warpgroup (Megakernel's `loader`) | inlined into Compute warps via `cp.async.ca.shared.global` |
| `Compute` | tensor-core math on tiles | WGMMA consumer warpgroups (Megakernel's `consumer`, 4 wg × 4 warps) | `mma.sync m16n8k16` in all warps |
| `Store` | SMEM → GMEM incl. atomic variants | dedicated TMA-store warpgroup (Megakernel's `storer`) | inlined, `st.global` / `atomicAdd` |

**Rule for adding a new role later:** (1) describes movement/sync not math, (2) has a hardware unit that can be dedicated to it on some arch, (3) cannot be expressed as existing roles + an edge dep kind. Fail any → it's a fusion or pattern, not a role.

**Explicitly not roles:**
- Math ops (RMSNorm, RoPE, softmax, SiLU, GEMM) — these are FUF `OpKind`.
- Fusions (QkNorm, QkvProj) — `impl_lib` entries.
- Region names (Attention, MLP, Block) — identifiers, not roles.
- Epilogue — `Compute` + `Store` with `Pipeline` edge. Don't add a role for a pattern the stencil already expresses.
- Controller — compiler artifact. We codegen straight-line dispatch; no runtime fetcher warpgroup needed. Reclaims 4 warps vs. Megakernel's runtime VM.

**Deferred:**
- `Launch` role — earns a slot only when SM100 (Blackwell `tcgen05` tensor-memory barriers) lands. Megakernel's `launcher` warpgroup is a no-op on pure SM90.
- `Reduce` as a role — covered by `Store` + `AtomicReduce` edge until proven insufficient.

---

## 3. Edge dep kinds (frozen at 5)

Dep kinds describe *synchronization*, not hardware.

| Dep kind | Meaning | SM90 primitive | SM89 primitive |
|---|---|---|---|
| `Raw` | SMEM-visible true data dep, same CTA | mbarrier phase bit / `__syncthreads` | `__syncthreads` / named barrier |
| `Pipeline` | intra-region producer/consumer, staged | named semaphore (Megakernel's `weights_arrived` / `weights_finished`) | `cp.async.commit_group` + `wait_group` |
| `AtomicReduce` | cross-CTA accumulation | `Store` emits `atomicAdd`; next `Compute` spins on global counter (Megakernel `g.Bar` pattern) | same — `atomicAdd` works on Ada |
| `Barrier` | region boundary, all CTAs must pass | cluster sync if intra-cluster, else global counter; **not** `grid.sync` (occupancy killer) | global counter only |
| `DataDependent` | runtime value decides dispatch (MoE router, spec accept) | region boundary + static-per-invocation dispatch table | same |

Every kind has ≥1 justifying use case across the stress tests.

---

## 4. Clarifications that surfaced (design constraints, not extensions)

These appeared repeatedly across tests; they aren't new vocabulary, they're capabilities the existing vocabulary requires.

### 4.1. Address expressions on `Load` nodes can gather

`Load` addresses are affine in domain coords plus region-entry constants — but one of those "constants" can be a small lookup table hoisted to SMEM at region entry.

Example (paged KV decode):
```
addr(K_j at (b, k, h)) = kv_pool_base
                      + block_table_smem[k / blocks_per_tile] * block_stride
                      + (k % blocks_per_tile) * token_stride
                      + h * head_stride
```

Affine-in-k within a page; the `block_table_smem` lookup happens once per tile. TMA descriptor gets rebuilt per tile using the gathered physical base. No new role. No `GatherLoad` sub-kind.

Constraint: the gather source must be hoistable to region entry. If the gather itself varies per inner-loop iteration in a non-hoistable way, that's a different problem (and probably a different region).

### 4.2. Domain predicates can reference region-entry scalars

Three patterns collapse under one mechanism:

- Causal mask: `k ≤ q_tile` (q_tile is static per-CTA)
- Sliding window (Gemma3 local attn): `k ∈ [q - W, q]` (W baked at compile time)
- Ragged decode: `k < ceil(seq_len[b] / tile_k)` (seq_len[b] read at region entry)

All three are affine predicates on the iteration domain, where coefficients may include region-entry scalars (constant or runtime-loaded). The predicate drives early-exit codegen inside the region, including proper handling under pipelining (don't load past `seq_len`).

### 4.3. Region templates are parametric

Gemma3 alternates local/global attention per layer. Instead of two region types, one `Attn(W)` template with `W = ∞` for global. Specialized at FUF→Stencil lowering time based on a per-layer spec in the architecture definition.

The stencil IR never sees a choice happen. Const-prop handles it.

---

## 5. Megakernel evidence (SM90)

What we learned by reading `~/Megakernels/`, filtered for what informs our IR:

**Hardware partition** (`include/megakernel.cuh:118-140`, `config.cuh`):
- 1 CTA per SM, 20 warps = 640 threads
- Roles are warpgroup-fixed for the whole kernel, not time-multiplexed:
  - 16 warps (4 wg) `consumer` — WGMMA compute
  - 4 warps (1 wg) `loader` — TMA producer
  - 4 warps (1 wg) `storer` — TMA store / epilogue
  - 4 warps (1 wg) `launcher` — SM100 tcgen05 barriers (no-op on SM90)
  - 4 warps (1 wg) `controller` — fetches instructions, manages SMEM pages

**Instruction VM** (`megakernels/instructions.py`, `util.cuh:11-19`):
- 128-byte instruction struct; host-side scheduler builds per-SM queues assigned by greedy cost-weighted heap
- Device reads `instructions[smid, inst_idx, :]`
- Pipeline depth 2 at the instruction level (current + next)

**Overlap is intra-instruction, 3-stage** (`demos/low-latency-llama/matvec_pipeline.cuh`):
- Semaphores: `weights_arrived / weights_finished / outputs_arrived / outputs_finished`
- Loader issues N+1 while consumer WGMMAs on N

**Cross-instruction sync:**
- Per-CTA: instruction_finished semaphore in SMEM
- Cross-CTA: global atomic counter `g.Bar` (increment in storer, spin-wait in next consumer)
- **No `grid.sync`.** Forcing full occupancy would kill performance.

**No work stealing.** Load balance = cost-weighted greedy host-side scheduling.

**Implications for our design:**
- Drop the controller warpgroup: we codegen straight-line instruction dispatch (reclaims 4 warps).
- Drop launcher on SM90 (add back for SM100).
- Keep the host-side per-SM queue + `g.Bar` pattern; it's proven.
- Our compiler advantage over their VM: explicit wavefront via stencil skewing + compile-time const-prop of shapes → tighter pipeline sizing and fewer dynamic fetches.

---

## 6. How SM90 vs SM89 falls out

Same stencil IR. Only the **role → hardware unit** table differs:

| | SM90 (Hopper) | SM89 (Ada) |
|---|---|---|
| `Load` | dedicated TMA warpgroup (1 wg) | inlined in Compute warps, `cp.async` |
| `Compute` | 4 WGMMA warpgroups | all warps, `mma.sync` |
| `Store` | dedicated TMA-store warpgroup (1 wg) | inlined, `st.global` |
| Pipeline depth | 3 (SMEM budget + distributed SMEM) | 2 (less SMEM) |
| Cross-CTA sync | cluster sync or `g.Bar` | `g.Bar` only |
| CTAs per SM | 1 (fixed by 20-warp layout) | more viable — no warpgroup specialization |

The stencil scheduler runs unchanged. The mapping pass is a per-arch rule set. Adding SM100 = extend the table (add `Launch` role + tcgen05 primitives). Adding CDNA (MI300) = extend the table (different tensor-core atom, no warpgroup abstraction — but the Load/Compute/Store split still applies).

---

## 7. Stress tests

Five fused blocks sketched; all hold the vocabulary.

### 7.1. Attention prefill + QKV/RoPE + O_proj (the canonical case)

Three regions: QKV+RoPE → FA2 → O_proj, all connected by `Barrier` edges (epilogue-fuse later).

FA2 region:
- Domain `(q_tile, kv_tile, head_group)` with causal predicate `k ≤ q`
- `Load K_j (q,k,h) ← Compute QK (q,k-P,h)` [Pipeline, depth P] — the overlap edge
- `softmax_update (q,k,h) ← softmax_update (q,k-1,h)` [Raw] — serial chain inside a q_tile
- Parallelism: `q_tile × head_group` across CTAs (free); pipeline staging within CTA (stencil skew on `k, pipe_stage`)

Reproduces Megakernel's `PartialAttention` instruction structurally.

### 7.2. MoE forward

Regions: RMSNorm+Router+TopK → Permute → **Expert MLP (DataDependent entry)** → Unpermute+combine.

The `DataDependent` edge M2→M3 carries a dispatch protocol:
1. All CTAs read `bincount[E]` at barrier
2. Compute prefix-sum → total work units
3. CTA_id → `(expert_id, m_tile_in_expert)` by deterministic mapping
4. Overflow CTAs loop `CTA_id += grid_size`

Not work-stealing. Static-per-invocation. No atomics on the mapping.

Load imbalance mitigation: cost-descending ordering before assignment (Megakernel host-scheduler trick). Still not work-stealing.

No new roles. No new dep kinds.

### 7.3. Gemma3 alternating local/global attention

Static structural `if` handled entirely by FUF unrolling + const-prop.

`Attn(W)` parametric region template: `W=4096` for local, `W=∞` for global. Stencil IR never sees the choice happen. Mapping table doesn't branch on local-vs-global.

Zero additions.

### 7.4. Paged KV decode

Same FA stencil as Test 7.1 with two mutations:
- Domain `(b, kv_tile, head_group)` replacing `q_tile` with batch dim (decode has M=1)
- Load addresses use the gather mechanism (clarification 4.1)
- Per-sequence variable KV range via domain predicate on `seq_len[b]` (clarification 4.2)

No new roles. No new dep kinds. Mapping emits TMA descriptor setup inside the k-loop instead of outside — detail, not abstraction.

### 7.5. Speculative decoding

Composes existing regions: K_spec draft decode passes + 1 target verify (prefill-shape, M=K_spec) + small Compute region computing `accept_length`.

**The `DataDependent` edge lives outside the kernel** — the kernel emits `accept_length`; the host scheduler decides the next forward's shape. Zero in-kernel dispatch.

Tree attention (Medusa) is the one flag: pushes 2D domain predicates harder (custom mask patterns). Predicate mechanism should handle it, but exercises it more than anything else on the roadmap. Worth re-sketching before committing to Medusa-class models.

### 7.6. FA3 split-k

Two regions: split-k partial attention (F1) + partial reduction (F2), connected by `Barrier`.

F1 domain adds `k_split` as a free dim → more CTAs for long-KV/small-Q case.

F2 is a small cross-`k_split` log-sum-exp combine. Deliberately *not* `AtomicReduce` because log-sum-exp isn't cleanly associative under atomics.

Mapping consequence worth noting: F1 and F2 have different CTA utilization. In one persistent launch, F2's extra CTAs early-exit. Acceptable cost for L2 residency of partials.

No new vocabulary.

---

## 8. Verdict

**Stable enough to start building.**

- Roles: **3** (`Load`, `Compute`, `Store`)
- Dep kinds: **5** (`Raw`, `Pipeline`, `AtomicReduce`, `Barrier`, `DataDependent`)
- Clarifications to bake in day-one: gather address expressions, runtime-scalar domain predicates, parametric region templates

All three clarifications appeared multiple times across tests. They're load-bearing constraints, not corner cases.

---

## 9. Known design work (not vocabulary — mechanisms)

These don't threaten the freeze but are real implementation work:

- **MoE barrier dispatch protocol** — static-per-invocation CTA assignment from runtime `bincount`. One concrete protocol to design.
- **Domain predicate codegen under pipelining** — early-exit correctness when some pipeline stages have loaded past the bound.
- **Expert-imbalance ordering** — cost-descending work-unit order for MoE barrier dispatch. Port Megakernel's host-scheduler heuristic.
- **Per-region mapping in one megakernel** — CTAs sized for the highest-parallelism region; lower-parallelism regions eat idle-CTA cost. Verify this is cheaper than separate launches (should be, via L2 residency, but measure).
- **Region templates DSL** — how the arch spec expresses parametric region instantiation (e.g. per-layer `Attn(W)` choice).

---

## 10. Out of scope for v1 — flagged for later

- `Launch` role and SM100 `tcgen05` primitives (add when Blackwell work starts)
- Collective ops (TP/EP cross-device) — will likely need a `CollectiveReduce` edge kind; single-GPU only for now
- Tree attention / Medusa — re-sketch predicate machinery before committing
- Inter-instruction / inter-region overlap (loading for region N+1 while computing region N) — Megakernel doesn't do this; worth it only if profiling justifies
- True work-stealing — decided against for v1; revisit if MoE imbalance proves too severe even with cost-ordering

---

## 11. Next step (before writing IR types)

Draw the `Attn(W)` region template + FA2 prefill mapping table at the level of concrete struct shapes (not just prose). If the types that falls out fit on a page cleanly, start implementation. If they don't, we learned something worth knowing before committing code.
