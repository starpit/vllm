# Stencil IR — §11 struct sketch

Concrete struct shapes for `Attn(W)` region template + FA2 prefill SM90 mapping. Companion to `STENCIL_IR_DESIGN.md`. Dated 2026-04-19.

Goal: prove the frozen vocabulary (3 roles, 5 dep kinds, 3 clarifications) generates a type surface that fits on a page. If it does → implement. If it doesn't → revise vocabulary first.

---

## 1. Core IR types

```rust
// ---------- Iteration domain ----------

pub struct Domain {
    pub axes: Vec<Axis>,
    pub predicates: Vec<Predicate>,
}

pub struct Axis {
    pub name: AxisId,               // q_tile, kv_tile, head_group, k_split, b, ...
    pub bound: Bound,
}

pub enum Bound {
    Const(u32),
    RegionEntryScalar(ScalarId),    // e.g. num_q_tiles(seq_len)
    IndexedScalar(ScalarId, AxisId),// e.g. ceil(seq_len[b] / tile_k)
    Unbounded,                      // for W = ∞
}

// Affine predicate: Σ cᵢ·aᵢ  <op>  offset
pub struct Predicate {
    pub coeffs: SmallVec<[(AxisId, i32); 2]>,
    pub offset: AffineOffset,
    pub op: CmpOp,
}

pub enum AffineOffset {
    Const(i32),
    RegionEntry(ScalarId),                // W, static per region instance
    Indexed(ScalarId, AxisId),            // seq_len[b]
}

pub enum CmpOp { Le, Lt, Ge, Gt, Eq }

// ---------- Nodes & roles ----------

pub struct Node {
    pub id: NodeId,
    pub role: Role,
    pub op: FufOpRef,        // back-ref into FUF DAG; carries math + tile type
    pub addr: Option<LoadAddr>,  // Some for Load/Store, None for Compute
}

pub enum Role { Load, Compute, Store }

// ---------- Edges ----------

pub struct Edge {
    pub src: NodeId,
    pub dst: NodeId,
    pub kind: DepKind,
    pub vector: DepVector,   // offset per axis; e.g. k=-P means "P iters back"
}

pub enum DepKind { Raw, Pipeline, AtomicReduce, Barrier, DataDependent }

pub struct DepVector(pub SmallVec<[(AxisId, i32); 2]>);

// ---------- Load addresses (affine + gather) ----------

pub struct LoadAddr {
    pub base: AddrTerm,
    pub terms: SmallVec<[AddrTerm; 4]>,   // summed
}

pub enum AddrTerm {
    AxisStride    { axis: AxisId, stride: StrideExpr },
    AxisModStride { axis: AxisId, modulus: u32, stride: StrideExpr },
    // paged KV: addr uses block_table_smem[axis/div] as a stride multiplier
    AxisDivGather { axis: AxisId, divisor: u32, table: SmemLookup, stride: StrideExpr },
    RegionEntryConst(ScalarId),
}

pub struct SmemLookup {
    pub source: GmemScalarRef,  // e.g. block_table[b]
    pub hoist:  HoistPoint,     // RegionEntry (only legal choice for v1)
}

// ---------- Region & region template ----------

pub struct Region {
    pub id: RegionId,
    pub domain: Domain,
    pub entry_scalars: Vec<ScalarBinding>, // loaded once at region entry
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

pub struct RegionTemplate {
    pub name: &'static str,                // "Attn"
    pub params: &'static [TemplateParam],  // &[Window]
    pub build: fn(&TemplateArgs) -> Region,
}

pub enum TemplateParam { Window /* W: u32 | ∞ */, HeadDim, NumHeads, /* ... */ }

// ---------- Megakernel CFG ----------

pub struct Megakernel {
    pub regions: Vec<Region>,
    pub control: Vec<ControlEdge>,   // kind ∈ {Barrier, DataDependent}
}
```

**Fits on a page?** Yes — ~60 lines of types, no escape hatches, no `Box<dyn _>`. Every frozen-vocabulary item has exactly one home.

---

## 2. `Attn(W)` template → FA2 prefill region

Instantiation: `Attn { W: ∞, head_dim: 128, tile_q: 128, tile_k: 64, pipe: 3 }`.

```rust
Region {
    id: RegionId("fa2_prefill"),
    domain: Domain {
        axes: vec![
            Axis { name: q_tile,     bound: RegionEntryScalar(num_q_tiles) },
            Axis { name: kv_tile,    bound: RegionEntryScalar(num_kv_tiles) },
            Axis { name: head_group, bound: Const(num_heads / group_size) },
        ],
        predicates: vec![
            // causal: k ≤ q  (tile granularity)
            Predicate {
                coeffs: smallvec![(kv_tile, 1), (q_tile, -1)],
                offset: Const(0),
                op: Le,
            },
            // window: q - W/tile_k ≤ k  (trivially-true when W = ∞; const-prop drops it)
            Predicate {
                coeffs: smallvec![(kv_tile, -1), (q_tile, 1)],
                offset: RegionEntry(window_in_tiles),
                op: Le,
            },
        ],
    },
    entry_scalars: vec![num_q_tiles, num_kv_tiles, window_in_tiles],
    nodes: vec![
        Node { id: n_load_q, role: Load,    op: fuf::load_q_tile,   addr: Some(…) },
        Node { id: n_load_k, role: Load,    op: fuf::load_k_tile,   addr: Some(…) },
        Node { id: n_load_v, role: Load,    op: fuf::load_v_tile,   addr: Some(…) },
        Node { id: n_qk,     role: Compute, op: fuf::qk_matmul,     addr: None    },
        Node { id: n_sm,     role: Compute, op: fuf::softmax_update,addr: None    },
        Node { id: n_pv,     role: Compute, op: fuf::pv_matmul,     addr: None    },
        Node { id: n_store,  role: Store,   op: fuf::store_o_tile,  addr: Some(…) },
    ],
    edges: vec![
        // Pipeline overlap: load K for iter k while computing QK for iter k-P
        Edge { src: n_load_k, dst: n_qk, kind: Pipeline, vector: dv(kv_tile, -P) },
        Edge { src: n_load_v, dst: n_pv, kind: Pipeline, vector: dv(kv_tile, -P) },
        // Raw: softmax running-state chain within a q_tile
        Edge { src: n_sm,     dst: n_sm, kind: Raw,      vector: dv(kv_tile, -1) },
        // Raw same-tile: QK → softmax → PV
        Edge { src: n_qk,     dst: n_sm, kind: Raw,      vector: dv() },
        Edge { src: n_sm,     dst: n_pv, kind: Raw,      vector: dv() },
        // Store fires once per q_tile at kv_tile = last
        Edge { src: n_pv,     dst: n_store, kind: Raw,   vector: dv() },
    ],
}
```

`dv(axis, n)` = dep vector with one entry; `dv()` = zero vector (same-iteration).

Paged-KV decode (§7.4) reuses this region with:
- `q_tile` → `b`, `Const(1)` on q dim (M=1)
- `n_load_k.addr` switched to `AxisDivGather { axis: kv_tile, divisor: blocks_per_tile, table: block_table_smem, … }`
- Extra predicate: `kv_tile < ceil(seq_len[b]/tile_k)` using `AffineOffset::Indexed`

Zero new types.

---

## 3. SM90 mapping table (FA2 prefill)

```rust
pub struct ArchMap {
    pub role: fn(Role, &Region) -> HardwareUnit,
    pub pipe_depth: fn(&Region) -> u32,
    pub barrier:   fn(DepKind)   -> BarrierPrim,
}

pub enum HardwareUnit {
    Warpgroup { first_warp: u8, count: u8 },
    AllWarps,
}

pub enum BarrierPrim {
    Mbarrier,                // Raw intra-CTA
    NamedSem { name: &'static str, depth: u32 },  // Pipeline
    Gbar,                    // Barrier / AtomicReduce cross-CTA
    Cluster,                 // optional Barrier within cluster
    HostRedispatch,          // DataDependent
}
```

SM90 instantiation for FA2 prefill:

| Input | Output |
|---|---|
| `role(Load, _)` | `Warpgroup { first_warp: 16, count: 4 }` (TMA producer) |
| `role(Compute, _)` | `Warpgroup { first_warp: 0, count: 16 }` (4 wg × WGMMA) |
| `role(Store, _)` | `Warpgroup { first_warp: 20, count: 4 }` — wait, 20 warps total; reuses loader or separate. SM90 config has 20 warps = 5 wg; Store uses 1 wg distinct from Load. |
| `pipe_depth(fa2)` | `3` |
| `barrier(Raw)` | `Mbarrier` |
| `barrier(Pipeline)` | `NamedSem { name: "kv_arrived", depth: 3 }` |
| `barrier(Barrier)` | `Gbar` |
| `barrier(AtomicReduce)` | `Gbar` (store emits `atomicAdd`) |
| `barrier(DataDependent)` | `HostRedispatch` |

SM89 instantiation (same region, different table):

| Input | Output |
|---|---|
| `role(Load, _)` | `AllWarps` (inlined `cp.async`) |
| `role(Compute, _)` | `AllWarps` (`mma.sync`) |
| `role(Store, _)` | `AllWarps` |
| `pipe_depth(fa2)` | `2` |
| `barrier(Pipeline)` | `CpAsyncGroup { depth: 2 }` |

Same region IR. Different table.

---

## 4. What didn't fit / revealed

1. **Store-once-per-q_tile** is currently expressed only implicitly (edge `n_pv → n_store` with zero dep vector but n_store has no kv_tile in its address). This needs either (a) an explicit "reduction axis" marker on Store's address, or (b) a rule that Store's address-free axes are implicitly reduced. **(b) is cleaner; no new type.** Document as a lowering rule.

2. **20-warp partition math on SM90.** 5 warpgroups × 4 warps. Roles consume 1+4+1 = 6 wg worth. Either the controller wg (4 warps) is reclaimed as we planned and we run 5 wg exactly, or Store shares a wg with Load (both TMA, different directions). Megakernel keeps them separate. **Decision: reclaim controller, Load/Store separate wg, total 5 wg = 20 warps.** Matches the table above.

3. **`FufOpRef` is a back-reference, not an owner.** Stencil IR layers scheduling on top; FUF owns math + tile types. No duplication.

4. **`RegionTemplate::build: fn(&TemplateArgs) -> Region`** — plain `fn`, no trait object. Template count is small and static; enum-dispatched match at lowering time is fine for v1.

5. **No `DepVector` over non-integer offsets** anywhere in the 6 stress tests. Freeze `i32` per axis; revisit only if a future op needs fractional/symbolic deps.

---

## 5. Verdict

Struct surface fits on a page. Every vocabulary item has one home. FA2 prefill + paged-KV decode share one region with zero new types. SM90↔SM89 differs only in the `ArchMap` table.

**Ready to implement.** Suggested first commit: `ferrite-stencil` crate with the §1 types + `Attn` `RegionTemplate` + an SM90 `ArchMap` stub that round-trips FA2 prefill to a printable mapping — no codegen yet. That's the integration-test-per-phase shape: real FUF op refs in, real mapping out, printed and diffed.
