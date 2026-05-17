# cuda_emit TK 2.0 surface audit

Per `MEGA_IR_PLAN.md` §8.0a workflow. This is the read-only audit
that must precede any emit code. Cite header + line for every
primitive that cuda_emit will splice into emitted `.cu`.

**No emit code yet.** This document lists what TK 2.0 actually
provides, then surfaces the architectural decision points the
rebuild must answer before any line of emit lands.

---

## 1. TK 2.0 is namespace `kittens::group<N>::...`

The single most consequential structural fact:

```cpp
// include/ops/group/group.cuh:114
using warp      = group<1>;
using warpgroup = group<4>;
```

`kittens::warp` is `kittens::group<1>`. Every memory op, every sync
op, every register op lives **inside the `group<N>` struct** (lines
21-117). There is no `kittens::wait(...)` / `kittens::arrive(...)`
free function — all calls are scoped:

| Old (TK 1.0 reference) | TK 2.0 actual |
|---|---|
| `kittens::wait(sem, phase)` | `kittens::group<1>::wait(sem, phase)` or `kittens::warp::wait(sem, phase)` |
| `kittens::arrive(sem)` | `kittens::group<1>::arrive(sem)` (auto-laneid-gates internally) |
| `kittens::tma::load_async(...)` | `kittens::group<1>::tma::load_async(...)` (or `group<N>::tma::...`) |
| `kittens::warp::load(rv, sv)` | `kittens::group<1>::load(rv, sv)` (== `kittens::warp::load`) |
| `kittens::group<NCW>::sync(BAR)` | ✅ same — only thing the previous nuked emit got right at this layer |
| `kittens::laneid()`, `kittens::warpid()` | ✅ free functions in `kittens::` |

The previous nuked cuda_emit emitted `kittens::wait(sem, phase)` and
`kittens::arrive(sem)` as if they were free functions. They aren't.
They're members of `kittens::group<N>::`.

---

## 2. GL descriptors — the heart of TK 2.0 TMA

```cpp
// include/types/global/gl.cuh:113
template<typename _T, int b, int d, int r, int c, typename... TMA_Types>
struct gl {
    T* raw_ptr;
    // dims: compile-time when b/d/r/c > 0, runtime when -1.
    detail::descriptor_dict<TMA_Types...> tma_descs;
    // ...
    __host__ inline gl(T *_data, batch_arg, depth_arg, rows_arg, cols_arg) : ...
};
template<typename _T, int d, int r, int c, typename... TMA_Types> using gl3 = gl<_T, 1, d, r, c, TMA_Types...>;
template<typename _T, int r, int c, typename... TMA_Types>        using gl2 = gl<_T, 1, 1, r, c, TMA_Types...>;
template<typename _T, int c, typename... TMA_Types>               using gl1 = gl<_T, 1, 1, 1, c, TMA_Types...>;
```

Three crucial properties:

1. **The `gl(...)` constructor is `__host__` only** (line 153). The
   `descriptor_dict<TMA_Types...>` member precomputes a `CUtensorMap`
   for each TMA_Type at construction, calling
   `kittens::detail::tma::create_tensor_map<>(...)` which is also
   host-only. **You cannot construct a gl device-side.**
2. **TMA_Types must be listed at gl construction.** When you build
   `gl<bf16, 1, 1, num_tokens, hidden_dim, sv_bf<hidden_dim>>`, the
   gl carries one CUtensorMap precomputed against `sv_bf<hidden_dim>`.
   At call time, `gl::get_tma<sv_bf<hidden_dim>, -1>()` returns that
   map. Calling tensor-TMA against a TMA_Type the gl wasn't
   constructed with → `static_assert` failure: `"SKILL ISSUE:
   Requested a TMA descriptor for a type not initialized in the
   global layout."` (line 73-74).
3. **Coord type matters.** `coord<sv_bf<2048>>(t)` means "row t, in
   units of sv_bf<2048>" → the unit_coord conversion at line 133-155
   turns it into element-level `coord<>` for the TMA call. Wrong
   coord type at the call site → wrong indexing.

### `coord<>` 4D index

```cpp
// include/types/global/util.cuh:120
template<typename _T=ducks::default_type> struct coord {
    int b, d, r, c;  // batch, depth, rows, cols
    coord(_b, _d, _r, _c)  // 4-arg
    coord(_d, _r, _c)      // 3-arg, b=0
    coord(_r, _c)          // 2-arg, b=d=0
    coord(_c)              // 1-arg, b=d=r=0
    coord()                // all 0
};
```

`coord<>` (default type) is element-level. `coord<SV>` / `coord<ST>`
is in units of the SV/ST. `unit_coord<row_axis, col_axis>()`
converts. Critical for vector TMA which uses `coord<SV>` and the
internal `unit_coord<-1, 3>()` to project to default coords.

---

## 3. TK 2.0 TMA — two paths

### 3a. Tensor TMA (`tma::load_async` / `tma::store_async`)

Tile path (`include/ops/group/memory/tile/tma.cuh:124`):

```cpp
template<int axis, cache_policy policy, ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src,
                                          const COORD &idx, semaphore& bar);
```

Vec path (`include/ops/group/memory/vec/tma.cuh:208`):

```cpp
template<cache_policy policy, ducks::sv::all SV, ducks::gl::all GL,
         ducks::coord::vec COORD=coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src,
                                          const COORD &idx, semaphore& bar);
```

Both auto-laneid-gate (`if(laneid() == 0)`). Both require a `gl`
with the SV/ST type registered in TMA_Types. Both index by tile/vec
coord, NOT raw element offset.

### 3b. Non-tensor TMA (`tma::load_async` byte-count overload)

`include/ops/thread/util/tma.cuh:86`:

```cpp
__device__ static inline void load_async(void *dst, void *src,
                                          uint32_t size_bytes, semaphore& bar);
template<typename T>
__device__ static inline void load_async(T &dst, T &src,
                                          uint32_t size_bytes, semaphore& bar);
```

Raw pointers, byte count, semaphore. No tensor-map required. Less
optimized but operates on the existing host ABI directly.

### 3c. Sync support

`include/ops/group/util/tma.cuh:18`:
```cpp
__device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes);

template<typename T, typename... args>
__device__ static inline void expect(semaphore& bar, const T& _1, const args&... _2);
// expand size_bytes<T, args...> and call expect_bytes
```

`include/ops/group/util/tma.cuh:46`:
```cpp
template <int N=0> __device__ static inline void store_async_wait();
```

---

## 4. Mbarrier sync (`include/ops/group/util/sync.cuh`)

```cpp
// :35
__device__ static inline void init_semaphore(semaphore& bar,
                                              int thread_count,
                                              int transaction_count=0);
// :69
__device__ static inline void arrive(semaphore& sem);
// :93
__device__ static inline void arrive(semaphore& sem, uint32_t count);  // Hopper+
// :112
__device__ static inline void wait(semaphore& sem, int kPhaseBit);
```

Auto-laneid-gates internally (line 70 inside arrive). Lives inside
`kittens::group<N>::`. Phase bit is `0` or `1` (parity-based
mbarrier).

### Named PTX bar.sync

`include/ops/group/group.cuh:33`:
```cpp
__device__ static inline void sync(int id) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}
__device__ static inline void arrive(int id) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "n"(GROUP_THREADS));
}
```

`kittens::group<NCW>::sync(BAR_ID)` — exactly what our IR's
`BarRef` fields feed. ✅ already correct in the kept IR work.

---

## 5. Register-vec / register-tile types

```cpp
// include/types/register/rv.cuh:62
template<...> struct rv {
    static constexpr int inner_dim = layout::inner_dim;  // 1 or 2
    static constexpr int outer_dim = is_naive ? (tiles+1)/2 : tiles;
    static constexpr int length = ...;
    dtype data[outer_dim][inner_dim];
    operator[](idx) -> dtype*       // returns row of inner_dim
    operator[](int2 outin) -> dtype& // direct (outer, inner)
};
```

Layouts: `align_l`, `ortho_l`, `naive_l`. `rv_fl<L>` is `rv` with
fp32 dtype + a default layout choice; `rv_bf<L>` for bf16. Layouts
matter for `mma_AB` arg type-checking.

`rt` (register tile) has `tiles[N][M]` arranged as a 2D grid of
16x16 base tiles.

---

## 6. Group memory ops — auto per-warp slicing!

**Major finding** (`include/ops/group/memory/vec/shared_to_register.cuh:84-89`):

```cpp
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load(RV &dst, const SV &src) {
    if constexpr (GROUP_WARPS == 1) {
        static_assert(SV::length == RV::length);
        // ... per-warp load logic
    }
    else {
        static_assert(SV::length == RV::length * GROUP_WARPS);
        auto &_src = src.template subvec<RV::length>(warpid());
        ::kittens::group<1>::load(dst, _src);
    }
}
```

When called as `kittens::group<NCW>::load(rv, full_sv)` with
NCW > 1, TK 2.0 **automatically slices** the full sv into
`subvec<RV::length>(warpid())` and does a per-warp load. The
constraint is `SV::length == RV::length * NCW`.

**This invalidates a huge chunk of the previous nuked emit.** The
old version manually carved per-warp slices via `reinterpret_cast<
sv_bf<K_PER_WARP>*>(... + warpid() * K_PER_WARP * 2)`. That's
unnecessary and (more importantly) wrong against the TK 2.0 type
checker — `kittens::group<NCW>::load(rv, full_sv)` is the canonical
pattern.

The same auto-subvec pattern is in the `store(sv, rv)` direction
(line 100-159).

---

## 7. Register-vec maps (warp scope)

`include/ops/group/register/vec/maps.cuh`:

| Op | Line | Signature |
|---|---|---|
| `add(dst, lhs, rhs)` | 333 | rv-rv elementwise; rhs can be rv or scalar broadcast |
| `sub(dst, lhs, rhs)` | 346 | same |
| `mul(dst, lhs, rhs)` | 359 | same |
| `div(dst, lhs, rhs)` | 372 | same |
| `copy(dst, src)` | 176-177 | rv-rv copy w/ dtype convert |
| `zero(dst)` | 131 | broadcast zero |
| `one(dst)` | 141 | broadcast one |
| `exp(dst, src)` | 187 | elementwise |
| `log(dst, src)` | 221 | elementwise |
| ... | | |

`include/ops/group/register/vec/reductions.cuh`:

| Op | Line | Signature |
|---|---|---|
| `sum(scalar_out, rv)` | 129 | warp-wide sum into scalar |
| `max(scalar_out, rv)` | 93 | warp-wide max |
| `min(scalar_out, rv)` | 111 | warp-wide min |
| `sum(scalar_out, rv, accum)` | 206 | with carry |
| ... | | |

**Convention**: out arg is FIRST (mutated `&`). Scalar reductions
write to `scalar_out` by reference. Add/mul take rhs as either
another rv (elementwise) or a scalar (broadcast — second overload
at line 54-55).

---

## 8. MMA primitives

`include/ops/group/mma/warp.cuh`:

```cpp
// :583
template<row_layout D, row_layout A, col_layout B, row_layout C>
__device__ static inline void mma_AB(D &d, const A &a, const B &b, const C &c);
// :647
template<row_layout D, row_layout A, row_layout B, row_layout C>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, const C &c);
// :711  mma_AtB
// :775  mma_AtBt
```

`d = a · b + c` (or transposed variants). Static asserts on:
- `D::rows == A::rows && D::cols == B::cols` (shape)
- `A::cols == B::rows` (reduction dim)
- dtype combinations (`D=fl, A=bf, B=bf, C=fl` is the bf16
  accumulate-in-fp32 case)

Templates over rt (register tile) layout — `row_layout` /
`col_layout`. Wrong layout = template substitution failure.

---

## 9. Register-budget control (Hopper+)

`include/ops/group/group.cuh:52-61`:

```cpp
template<int n_reg> __device__ static inline void increase_registers();
template<int n_reg> __device__ static inline void decrease_registers();
__device__ static inline void producer_registers();
template<int NCWG> __device__ static inline void consumer_registers();
```

`kittens::group<1>::producer_registers()` decreases to 24.
`kittens::group<1>::consumer_registers<NCWG>()` increases by a
formula. **Replaces** ferrite-substrate's
`set_consumer_registers<Config>()` /
`set_non_consumer_registers<Config>()` (which currently exist in
`ferrite_warp_roles.cuh` as wrappers around the same asm). Either
keep ferrite's wrappers (still valid TK 2.0-compatible asm) or
switch to the TK 2.0 calls; both work.

---

## 10. Ferrite substrate (intact, kept)

`crates/ferrite-kernels/csrc/tk/`:

| File | Provides |
|---|---|
| `ferrite_substrate.cuh` | `SharedState<Config>` (pages, page_ready/done/consumed semaphores, scratch); `init_shared_state` |
| `ferrite_warp_roles.cuh` | `kLoaderSlot` / `kLauncherSlot` / `kStorerSlot`; `set_consumer_registers` / `set_non_consumer_registers` |
| `ferrite_globals.cuh` | dtype aliases (`bf16_ptr`, `f32_ptr`, etc.) |
| `ferrite_barrier.cuh` | `barrier_signal(slot, count)` / `barrier_wait(slot, expected)` |
| `ferrite_tk_helpers.cuh` | `rms_norm_vec` / `rms_norm_scale_from_rv` / `matvec` / `matvec_reduce` / `store_n_rows` / `tanh_softcap_vec` |

**Verification needed**: `ferrite_tk_helpers.cuh` calls
`kittens::warp::copy/mul/sum`, `kittens::group<NCW>::sync(BAR_ID)`.
Those are TK 2.0-valid (verified above). The helper file itself is
intact and compatible.

---

## 11. Host-side ABI — the wall the emit hits

`crates/ferrite-forward/src/interpreter/mega/mod.rs`. Three tiers:

### Tier 1: Base (`LaunchArgs`, `LaunchFn`)
```cpp
extern "C" cudaError_t ferrite_<variant>_launch(
    __nv_bfloat16* const*       act_ptrs,     // [NUM_ACT_SLOTS]
    const __nv_bfloat16* const* weight_ptrs,  // [NUM_WEIGHT_ACCESSORS * NUM_LAYERS]
    int32_t*                    barriers,     // [NUM_EDGES] or null
    int32_t                     trace_level,
    cudaStream_t                stream);
```

### Tier 2: QKV (`LaunchArgsQkv`, `LaunchFnQkv`)
Above + `positions` (u32*), `slot_mapping` (i64*), `key_cache_ptrs`
(bf16**), `value_cache_ptrs` (bf16**).

### Tier 3: Attn (`LaunchArgsAttn`, `LaunchFnAttn`)
QKV tier + `seq_lens` (i32*), `block_table` (u32*).

### The decision the rebuild must answer

The host ABI **passes raw pointers, not `gl<...>` values**. The
`act_ptrs` table is `__nv_bfloat16* const*` — a host-allocated
device array of `NUM_ACT_SLOTS` bf16 pointers. The kernel takes
that array directly. There's no `gl` construction in the host path.

This means TK 2.0 tensor TMA (which requires gls with precomputed
`tma_descs`) is **not directly callable** without one of:

- **Path A — non-tensor TMA throughout.** Use
  `kittens::group<1>::tma::load_async(void* dst, void* src,
  uint32_t size_bytes, semaphore& bar)` for every load. Raw
  pointers in. No gl construction. Loses multi-dim coalescing
  optimizations but IS still a TK 2.0 primitive (`include/ops/
  thread/util/tma.cuh:86` / `include/ops/group/util/tma.cuh:72`).

- **Path B — reshape host ABI to pass gls.** Host constructs gl
  values via `kittens::detail::tma::create_tensor_map<...>`
  (host-only). Kernel signature takes gl values. Big change to
  `LaunchArgs*` and the proc-macro side that emits the launch
  call. Gets full tensor-TMA performance.

- **Path C — hybrid.** E.g., activations → non-tensor TMA;
  weights → gl-based tile TMA (constructed host-side once at
  weight registration time). More moving parts but pragmatic.

---

## 12. Concrete contracts the emit must respect

Pulled from the audit, in priority order:

1. **Every TK call is namespaced under `kittens::group<N>::*`** (or
   `kittens::warp::*` ≡ `kittens::group<1>::*`). No top-level
   `kittens::wait/arrive/...`.
2. **`kittens::group<NCW>::load(rv, sv)` does per-warp slicing
   automatically** when `NCW > 1` and `sv.length == rv.length * NCW`.
   Don't manually carve slices.
3. **TMA paths split by tensor-vs-non-tensor.** Tensor TMA needs
   gls; non-tensor TMA takes raw pointers + byte count. Pick one
   per use site.
4. **Coords are `coord<>` (default, element-level) or `coord<SV>`
   / `coord<ST>` (in tile/vec units).** The vec TMA path uses
   `coord<SV>` and projects internally via
   `unit_coord<-1, 3>()`. Brace-init lists don't work at the
   API surface — must construct a `coord<>` value.
5. **Register layouts (`align_l` / `ortho_l` / `naive_l`) and
   register tile shapes (`rt_fl<R, C>` / `rt_bf<R, C>` row vs col
   layouts) are type-system args of mma_AB.** Get them wrong →
   substitution failure, not a runtime miswire.
6. **Reductions write through scalar mut-ref out arg first**, then
   take rv inputs. `sum(scalar_out, rv)`, not `auto x = sum(rv)`.
7. **Mbarrier ops auto-laneid-gate** internally (`if(laneid() == 0)`
   guards). Don't double-gate — `kittens::group<1>::arrive(sem)`
   from any lane works; only lane 0 actually emits the asm.

---

## 13. What I know I still need to verify

Before any emit code, three more things to nail down:

1. **gl host-construction call site.** `gl(T*, b, d, r, c)` exists,
   but the actual TMA descriptor cache is built via
   `kittens::detail::tma::create_tensor_map<>(...)`. Where does this
   live, what does it require, what's the cost? — affects Path B
   feasibility.
2. **Existing `ferrite_tk_helpers.cuh` audit.** I added
   `tanh_softcap_vec` to it. The pre-existing `rms_norm_vec` calls
   `kittens::warp::copy/mul/sum`, `kittens::group<NCW>::sync(BAR)`.
   Those are TK 2.0-valid per my audit, but I should verify each
   helper's full body compiles against TK 2.0 (not just verify the
   primitive names exist — verify the dtype/layout combos).
3. **Existing `interpreter/mega/mod.rs` extension contract.** When
   the proc-macro currently emits `extern "C" { fn
   ferrite_<variant>_launch(...) }` for some variant, the emitted
   `.cu`'s function signature must match the `LaunchFn*` type
   exactly (positional ABI). Need to read the proc-macro side to
   confirm what it currently emits and how it picks a tier.

---

## 14. Architectural decision points (need user direction)

**Decision 1 — Tensor vs non-tensor TMA.** Path A is the simplest
fit with the existing host ABI; Path B is the long-term right
performance. Recommendation: **Path A for the first emit pass**,
revisit after E2E coherence.

**Decision 2 — Per-warp slicing in consumer bodies.** TK 2.0's
`kittens::group<NCW>::load(rv, sv)` does it automatically. The
ferrite-owned `rms_norm_vec` / `matvec` / etc. helpers in
`ferrite_tk_helpers.cuh` instead take per-warp `sv_bf<K_PER_WARP>`
slices and do their own work. **Two consistent patterns are
possible**: (a) the codegen uses `kittens::group<NCW>::load(...)`
and the helpers take full-length args, OR (b) the codegen passes
already-sliced per-warp args (current ferrite helper convention)
and the codegen does the slicing. Recommendation: **(a) is
canonical TK 2.0**; the helpers may need refactoring.

**Decision 3 — Fate of `ferrite_tk_helpers.cuh`.** The helpers
`rms_norm_vec` / `matvec` / `store_n_rows` are ferrite-owned
wrappers around TK 2.0 primitives. They're not strictly necessary
— the codegen could splice TK 2.0 calls directly. Two options:
(a) keep the helpers as a thin abstraction layer that emit calls
into; (b) inline everything at codegen time, no ferrite-side
`.cuh` helpers (matches the user's framing "TK 2.0 primitives
only"). Recommendation: **(b) — inline at emit time**, drop the
helpers, codegen splices TK 2.0 calls + ferrite substrate scaffold
only.

---

## 15. Proposed sprint sequencing (after this audit lands)

S0 (this doc) — TK 2.0 surface audit ✅

S1 — **Tighten the audit's open items**: gl host-construction
walkthrough; `ferrite_tk_helpers.cuh` line-by-line verification;
proc-macro launch-emit code path. ~half-day read-only.

S2 — **User decisions on the three architectural points** (§14).

S3 — **Handle layer rebuild**: Rust types for `Gl<T, B, D, R, C,
[TMA_Types]>` (or the simpler raw-ptr handles if Path A),
`Coord<TileType>`, `Sv<T, L>`, `Rv<T, L>`, `Rt<T, R, C>`. Each
tk:: function takes typed handles; emits a TK 2.0-namespaced
call.

S4 — **First variant emit (RmsNorm only)**, end-to-end. Compile
the emitted `.cu` against the actual TK 2.0 headers on the pod.
That is the ground truth — not Rust unit tests.

S5+ — One variant per sprint, same as before; this time each
emitted `.cu` must compile on the pod before the variant is
"done."

---

## 16. References (for the rebuild)

```
include/types/global/gl.cuh                ← gl<T,b,d,r,c,TMA_Types...>
include/types/global/util.cuh:120          ← coord<>
include/types/register/rv.cuh:62           ← rv definition
include/types/register/rt.cuh              ← rt definition (not yet read)
include/types/shared/sv.cuh                ← sv definition
include/types/shared/st.cuh                ← st definition
include/ops/group/group.cuh                ← group<N>::*, warp = group<1>
include/ops/group/util/sync.cuh            ← arrive/wait/init_semaphore
include/ops/group/util/tma.cuh             ← expect_bytes / store_async_wait
include/ops/group/memory/tile/tma.cuh      ← tile tma::load/store_async
include/ops/group/memory/vec/tma.cuh       ← vec tma::load/store_async
include/ops/group/memory/vec/shared_to_register.cuh  ← group<N>::load(rv, sv)
include/ops/group/register/vec/maps.cuh    ← warp::add/mul/sub/div/copy/zero/...
include/ops/group/register/vec/reductions.cuh ← warp::sum/max/min
include/ops/group/mma/warp.cuh             ← warp::mma_AB / mma_ABt / ...
include/ops/thread/util/tma.cuh:86         ← non-tensor tma::load_async (raw ptr)
crates/ferrite-kernels/csrc/tk/*           ← ferrite substrate (intact)
crates/ferrite-forward/src/interpreter/mega/mod.rs  ← host ABI
```
