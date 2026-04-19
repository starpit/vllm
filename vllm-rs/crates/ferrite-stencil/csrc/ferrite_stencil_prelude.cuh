// SPDX-License-Identifier: Apache-2.0
//
// Ferrite Stencil megakernel prelude.
//
// Provides the type definitions (StencilFrag, Mbarrier), the
// warpgroup-id macros the emitter references, and the helper
// functions the generated kernel calls. Helpers with fixed
// signatures (the cp.async / wgmma fence-commit-wait fences, the
// cross-CTA `gbar_sync`) lower to real PTX; data-movement and
// compute bodies (tma_load_2d / wgmma.mma_async / mma.sync /
// stg_128) still trap until the emitter threads byte counts and
// mma-fragment descriptors through — item 1 second wave.
//
// Anything still `__trap()` is ACTIVELY unimplemented; a kernel
// that calls it will abort on the device rather than silently
// produce wrong results. Landing real PTX for one helper at a time
// is the discipline — this file is the stable ABI between the
// emitter and nvcc.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

typedef __nv_bfloat16 bf16;

// ─── Warpgroup id constants ──────────────────────────────────────
// 20-warp persistent-CTA layout (SM90): consumer(3 wg) + loader(1) +
// storer(1). SM89's AllWarps mapping ignores these.

#ifndef LOADER_WG
#define LOADER_WG 3u
#endif
#ifndef CONSUMER_WG
#define CONSUMER_WG 0u
#endif
#ifndef STORER_WG
#define STORER_WG 4u
#endif
#ifndef PIPE
#define PIPE 3u
#endif

// ─── Kernel-wide ambient symbols ────────────────────────────────
// These are placeholders the emitter still references outside any
// region scope (norm epsilon, attention softcap, hidden_dim from
// per-model config, paged-KV indirection buffers). Item 4 in
// STENCIL_IR_STATUS.md moves them to kernel params; for now they
// stay as file-scope placeholders so the emitted output parses.

__device__ static constexpr float eps = 1e-6f;
__device__ static constexpr float scale = 1.0f;
__device__ static constexpr float cap = 1.0f;
__device__ static constexpr float rcp_cap = 1.0f;
__device__ static constexpr uint32_t hidden_dim = 1u;
__device__ static constexpr uint32_t blocks_per_tile = 1u;
__device__ static const uint32_t* block_table = nullptr;
__device__ static const uint32_t* token_ids = nullptr;

// ─── Opaque fragment type ───────────────────────────────────────
// Emitted expansions reference `X_frag`, `Q_frag`, etc. as typed
// per-thread locals. Real lowering replaces the type with concrete
// mma fragments (register layouts, accumulator types). Meanwhile
// StencilFrag is opaque enough to parse under arithmetic and
// assignment so the emitted bodies compile.

struct StencilFrag {
    float _pad[1];
    __device__ StencilFrag() = default;
    __device__ StencilFrag& operator=(const StencilFrag&) { return *this; }
    __device__ StencilFrag& operator=(float) { return *this; }
};

__device__ static StencilFrag operator*(float, const StencilFrag& x) { return x; }
__device__ static StencilFrag operator*(const StencilFrag& x, float) { return x; }
__device__ static StencilFrag operator*(const StencilFrag& x, const StencilFrag&) { return x; }
__device__ static StencilFrag operator+(const StencilFrag& x, const StencilFrag&) { return x; }
__device__ static StencilFrag operator-(const StencilFrag& x, float) { return x; }
__device__ static StencilFrag operator-(const StencilFrag& x, const StencilFrag&) { return x; }

// Region-local fragment / smem / mbarrier declarations are emitted
// inside each region's `{}` block by emit_megakernel — see
// emit_ops::local_refs + emit_mega::write_region. The prelude no
// longer hands out shared file-scope placeholders (pre-item-2
// revisions did, which caused every region to collide on the same
// `C_frag`, `smem_a`, `bar_kv`, etc.).

// ─── Load helpers ────────────────────────────────────────────────
// Real PTX now for `cp_async_128<BYTES>` — the helper each warpgroup
// calls to stage a `BYTES`-sized gmem tile into a shared-memory
// destination. TMA + STG variants still trap; they need a tensor-map
// descriptor (TMA) or a concrete fragment register layout (STG), both
// of which are follow-up commits.
//
// Call shape emitted by emit_ops:
//     cp_async_128<SMEM_Q_BYTES>(smem_q, Q_gmem + (q_tile * 16384u + head_group * 128u));
// — a tile-base pointer (not axis indices). `emit_addr::render`
// produces the offset from the Node's `LoadAddr.terms`. The helper
// does NOT know the layout; it copies BYTES contiguous bytes into
// the smem buffer.

template <uint32_t BYTES = 0, class... A> __device__ inline void tma_load_2d(A&&...) { __trap(); }
template <uint32_t BYTES = 0, class... A> __device__ inline void tma_store_2d(A&&...) { __trap(); }
template <uint32_t BYTES = 0, class... A> __device__ inline void generic_load(A&&...) { __trap(); }
template <uint32_t BYTES = 0, class... A> __device__ inline void generic_store(A&&...) { __trap(); }

/// cp.async.ca.shared.global 16B loop. Each warpgroup (128 threads)
/// cooperatively stages a BYTES-sized tile into smem — thread `tid ∈
/// [0, 128)` issues `BYTES / (128 * 16)` cp.async's (rounded up) at
/// 16-byte stride, so the full warpgroup covers the whole tile with
/// coalesced issues.
///
/// BYTES=0 is the "bare" call form (embed-row, cache-store — sites
/// that haven't been updated to name a smem local). Those keep
/// trapping so a runtime hit is loud, not silent.
///
/// BYTES must be a multiple of 16 — cp.async.ca accepts 4 / 8 / 16,
/// and every real region tile (smem_q / smem_k / smem_x / …) is sized
/// as a bf16 array whose byte count rounds to 16. A static_assert
/// catches any future tile_consts entry that forgets this.
///
/// SM80+ only; on SM89 this is the runtime form used by the generic
/// cp.async path (the "sm89_fa2" arch map). On SM90a+ the real target
/// is TMA; cp_async_128 is a fallback for ops where the TMA
/// descriptor plumbing hasn't landed (and a stepping stone for
/// correctness testing — cp.async works on H100 too).
template <uint32_t BYTES>
__device__ inline void cp_async_128(bf16* smem, const bf16* gmem) {
#if __CUDA_ARCH__ >= 800
    if constexpr (BYTES == 0) {
        // Bare call from a not-yet-updated emit_ops site (embed-row /
        // cache-store). Trap so we notice at runtime rather than
        // silently eliding a required load.
        __trap();
    } else {
        static_assert(BYTES % 16u == 0u,
                      "cp_async_128 BYTES must be a multiple of 16 (cp.async 16B issues)");
        constexpr uint32_t CHUNKS = BYTES / 16u;
        constexpr uint32_t THREADS = 128u;
        uint32_t tid = threadIdx.x & (THREADS - 1u);
        uint32_t smem_base = __cvta_generic_to_shared(smem);
        uintptr_t gmem_base = reinterpret_cast<uintptr_t>(gmem);
        #pragma unroll
        for (uint32_t i = tid; i < CHUNKS; i += THREADS) {
            uint32_t offset = i * 16u;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(smem_base + offset), "l"(gmem_base + offset) : "memory");
        }
    }
#else
    (void)smem; (void)gmem;
    __trap();
#endif
}

/// Fall-through overload for sites that haven't been updated to the
/// `(smem, gmem_base_ptr)` shape — embed-row (bare), cache-store (bare).
/// Keeps trapping so future runtime hits are loud. The prelude accepts
/// the call via pack so emit_ops doesn't need a second helper name.
template <uint32_t BYTES = 0, class... A>
__device__ inline void cp_async_128(A&&...) { __trap(); }

/// cp.async.commit_group — closes the current set of in-flight
/// `cp.async` requests into a group that `cp.async.wait_group` can
/// later drain. SM80+ (Ampere). On older archs this is a no-op so
/// emitted SM89-path code still compiles cleanly.
__device__ inline void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

/// cp.async.wait_group N — block until at most `N` pending
/// `cp.async` groups remain. SM80+.
__device__ inline void cp_async_wait_group(uint32_t depth) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" ::"r"(depth) : "memory");
#else
    (void)depth;
#endif
}

// ─── Barrier / sync helpers ──────────────────────────────────────
// Mbarrier-with-pointer forms still trap until the region emitter
// threads a real phase bit (Hopper's `mbarrier.try_wait.shared.b64`
// takes a current phase + deadline). The no-arg `mbarrier_wait()`
// used between regions lowers to a `__syncthreads` barrier — any
// pending shared writes from this CTA finish before the next
// region starts its loads. gbar_sync implements the Hopper-style
// global counter spin-wait used for cross-CTA Barrier edges.

__device__ inline void mbarrier_wait() {
    __syncthreads();
}
template <class T> __device__ inline void mbarrier_wait(T*) { __trap(); }
template <class T> __device__ inline void mbarrier_arrive(T*) { __trap(); }

/// Cross-CTA global barrier. Every CTA calls with the same
/// device-wide counter symbol (`&gbar_counter` in the emitted
/// kernel). Implementation: each CTA elects thread 0, atomic-adds
/// 1 to the counter, and spin-waits on the next multiple of
/// `gridDim.x`. Works on any arch with atomicAdd (SM60+) — no
/// hardware grid sync needed, in line with HazyResearch's
/// `Megakernel`s pattern.
__device__ inline void gbar_sync(uint32_t* counter) {
    __syncthreads();
    if (threadIdx.x == 0u && threadIdx.y == 0u && threadIdx.z == 0u) {
        uint32_t num_ctas = gridDim.x * gridDim.y * gridDim.z;
        uint32_t prev = atomicAdd(counter, 1u);
        uint32_t target = (prev / num_ctas + 1u) * num_ctas;
        while (atomicAdd(counter, 0u) < target) {
            // Spin. Hopper (SM90) could use `st.async` + `red.async`
            // to reduce traffic, but the atomicAdd-0 polling form
            // works on every arch and matches the reference
            // implementation in HazyResearch/ThunderKittens's
            // megakernel.cuh.
        }
    }
    __syncthreads();
}

/// cluster.sync — barrier across the current thread-block cluster
/// (Hopper only). Emitted between regions when `ArchMap::barrier`
/// picks `BarrierPrim::Cluster`.
__device__ inline void cluster_sync() {
#if __CUDA_ARCH__ >= 900
    asm volatile("barrier.cluster.sync.aligned;\n" ::: "memory");
#else
    __syncthreads();
#endif
}

// Named semaphore: placeholder accepting the literal + depth from
// emitted sites; real impl would hash the name at compile time.
struct _NamedSemDepth { int depth; };
#define sem_wait(name, ...) do { (void)(name); __trap(); } while (0)

// ─── Tensor-core compute helpers ─────────────────────────────────
// The compute bodies (wgmma.mma_async / mma.sync) still trap —
// they need accumulator + smem descriptor plumbing from the emitter.
// The wgmma fence / commit / wait fences don't take operands and
// land as real PTX here. SM90+ only; on SM89 they stay no-ops so
// the emitted SM89 code (which never calls them) still compiles.

/// wgmma.fence.sync.aligned — aligns the consumer warpgroup's view
/// of its accumulators before issuing `wgmma.mma_async`. Paired
/// with wgmma_commit_group + wgmma_wait_group.
__device__ inline void wgmma_fence() {
#if __CUDA_ARCH__ >= 900
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}

template <class... A> __device__ inline void wgmma_mma_async(A&&...) { __trap(); }

/// wgmma.wait_group.sync.aligned N — block the warpgroup until at
/// most N wgmma groups remain in-flight. Template N is inlined so
/// the PTX immediate matches the group depth.
template <int N> __device__ inline void wgmma_wait_group() {
#if __CUDA_ARCH__ >= 900
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
#endif
}

/// wgmma.commit_group.sync.aligned — closes the current wgmma
/// group so a subsequent `wait_group<N>` can fence on it.
__device__ inline void wgmma_commit_group() {
#if __CUDA_ARCH__ >= 900
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}

template <class... A> __device__ inline void mma_sync_accumulate(A&&...) { __trap(); }
template <class... A> __device__ inline void mma_accumulate(A&&...) { __trap(); }
template <uint32_t BYTES = 0, class... A> __device__ inline void stmatrix_smem(A&&...) { __trap(); }
template <uint32_t BYTES = 0, class... A> __device__ inline void stg_128(A&&...) { __trap(); }

// ─── Softmax / elementwise primitives ───────────────────────────
// UNIMPLEMENTED: warp-shuffle reductions, exp2 fragment passes.

__device__ inline float rcp(float x) { return 1.0f / x; }

// CUDA math-intrinsic overloads that accept fragments. Real lowering
// will emit vectorized fragment-level ops; this keeps the element-
// wise expansions (`scalar_mul`, `tanh_softcap`) parse-valid.
__device__ inline StencilFrag __tanhf(const StencilFrag& x) { return x; }
__device__ inline StencilFrag rsqrtf(const StencilFrag& x) { return x; }

// Templated fragment-consumer helpers — accept smem arrays, single
// frags, and anything else that appears in expansion call sites.
template <class T> __device__ inline float warp_reduce_sum_of_squares(const T&) { return 0.0f; }
template <class A, class B> __device__ inline StencilFrag frag_mul_g(const A&, const B&) { return StencilFrag{}; }
template <class A, class B> __device__ inline StencilFrag frag_add_g(const A&, const B&) { return StencilFrag{}; }
template <class A, class B> __device__ inline StencilFrag rope_rotate_g(const A&, const B&) { return StencilFrag{}; }
template <class T> __device__ inline StencilFrag silu_g(const T&) { return StencilFrag{}; }
template <class T> __device__ inline float row_max_g(const T&, float init) { return init; }
template <class T> __device__ inline float row_sum_g(const T&) { return 0.0f; }
template <class T> __device__ inline StencilFrag exp2f_frag_g(const T&) { return StencilFrag{}; }

// Thin macros so the pseudocode's bare calls pick up the templated
// forms. Macro approach avoids the "no matching overload" path when
// callers pass arrays vs references.
#define frag_mul(a, b) frag_mul_g((a), (b))
#define frag_add(a, b) frag_add_g((a), (b))
#define rope_rotate(a, b) rope_rotate_g((a), (b))
#define silu(x) silu_g((x))
#define row_max(x, init) row_max_g((x), (init))
#define row_sum(x) row_sum_g((x))
#define exp2f_frag(x) exp2f_frag_g((x))

// Axis-name file-scope placeholders lived here pre-item-3 — when an
// expansion referenced an axis the hosting region didn't own (e.g.
// `load_q_tile` emitting `q_tile` inside `paged_decode` which
// iterates `b`), the reference fell through to `q_tile = 0` in this
// namespace. Item 3 threaded real axis names through `ExpandCtx`, so
// every axis identifier in the emitted body now resolves to an
// in-scope loop variable. No compatibility shim needed — if you see
// an "undeclared identifier" from nvcc, the fix is to extend
// emit_ops's axis lookup, not to re-add a placeholder here.

// ─── Ambient strides / indices ───────────────────────────────────
// Emitted bodies reference `hidden_stride` as loop-invariant context.
// Real lowering will replace it with compile-time constants or kernel
// params; placeholder so parsing succeeds.

__device__ static constexpr uint64_t hidden_stride = 1ull;

// ─── Mbarrier opaque type ───────────────────────────────────────
// Region-local mbarrier declarations (`__shared__ Mbarrier bar_*`)
// are emitted inside each region's `{}` block. The prelude defines
// only the type so those declarations parse; pre-item-2 revisions
// declared every bar_* at file scope, which coupled every region to
// the same handshake objects.

struct Mbarrier { uint64_t _op[1]; };
