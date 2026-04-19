// SPDX-License-Identifier: Apache-2.0
//
// Ferrite Stencil megakernel prelude — SYNTACTIC placeholder.
//
// This header exists so `emit_megakernel`'s output parses cleanly
// under nvcc. The bodies here are NOT real implementations — each
// helper either returns a trivial default or calls `__trap()` to
// ensure silent correctness regressions never ship. Real intrinsic
// lowering (TMA descriptor setup, cp.async.bulk.tensor, wgmma.mma,
// mbarrier PTX, fragment layouts, warp-shuffle reductions, …) lands
// one helper at a time in follow-up commits, with this prelude as
// the stable ABI.
//
// Scope decision: we prioritized compile-time validity of emitted
// sources (so the pipeline has a closed loop up to nvcc) over
// runtime correctness of the placeholders. The emitted code is
// structurally correct and shape-complete; every helper is
// explicitly unimplemented so the next commit has one clear target
// at a time.

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
// UNIMPLEMENTED: TMA descriptor setup + cp.async.bulk.tensor.2d
// on SM90; cp.async.ca.shared.global 16B transactions on SM89.

template <class... A> __device__ inline void tma_load_2d(A&&...) { __trap(); }
template <class... A> __device__ inline void tma_store_2d(A&&...) { __trap(); }
template <class... A> __device__ inline void cp_async_128(A&&...) { __trap(); }
template <class... A> __device__ inline void generic_load(A&&...) { __trap(); }
template <class... A> __device__ inline void generic_store(A&&...) { __trap(); }
__device__ inline void cp_async_commit_group() { __trap(); }
__device__ inline void cp_async_wait_group(uint32_t) { __trap(); }

// ─── Barrier / sync helpers ──────────────────────────────────────
// UNIMPLEMENTED: mbarrier PTX, named semaphores, cross-CTA atomic-
// counter spin-wait for g.Bar.

__device__ inline void mbarrier_wait() { __trap(); }
template <class T> __device__ inline void mbarrier_wait(T*) { __trap(); }
template <class T> __device__ inline void mbarrier_arrive(T*) { __trap(); }
__device__ inline void gbar_sync(uint32_t*) { __trap(); }
__device__ inline void cluster_sync() { __trap(); }
// Named semaphore: placeholder accepting the literal + depth from
// emitted sites; real impl would hash the name at compile time.
struct _NamedSemDepth { int depth; };
#define sem_wait(name, ...) do { (void)(name); __trap(); } while (0)

// ─── Tensor-core compute helpers ─────────────────────────────────
// UNIMPLEMENTED: wgmma on SM90; mma.sync m16n8k16 sweeps on SM89.

template <class... A> __device__ inline void wgmma_fence() { __trap(); }
template <class... A> __device__ inline void wgmma_mma_async(A&&...) { __trap(); }
template <int N> __device__ inline void wgmma_wait_group() { __trap(); }
__device__ inline void wgmma_commit_group() { __trap(); }
template <class... A> __device__ inline void mma_sync_accumulate(A&&...) { __trap(); }
template <class... A> __device__ inline void mma_accumulate(A&&...) { __trap(); }
template <class... A> __device__ inline void stmatrix_smem(A&&...) { __trap(); }
template <class... A> __device__ inline void stg_128(A&&...) { __trap(); }

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

// Axis-name file-scope placeholders. Each for-loop shadows the
// matching identifier, so inside a loop the real iteration variable
// wins; expansions that reference an axis *outside* their own
// region's loop (e.g. load_q_tile emitted inside paged_decode which
// iterates `b`, not `q_tile`) fall through to these zero-valued
// placeholders. Real lowering will thread axis names through
// `ExpandCtx` so mismatched references become impossible.
namespace ferrite_axis_aliases {
__device__ static uint32_t q_tile = 0, kv_tile = 0, head_group = 0, head_tile = 0;
__device__ static uint32_t m_tile = 0, n_tile = 0, k_tile = 0;
__device__ static uint32_t token_tile = 0, inter_tile = 0;
__device__ static uint32_t b = 0, page = 0, slot = 0, row = 0;
}  // namespace ferrite_axis_aliases
using namespace ferrite_axis_aliases;

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
