// SPDX-License-Identifier: Apache-2.0
// ThunderMittens — cross-threadgroup SYNC primitives for the PD-wavefront
// persistent decode megakernel on Apple GPU. Composable METAL_FUNC device
// functions (the per-target primitive library; the IR/encoding stays neutral
// in `ferrite-wavefront`; ≈ ThunderKittens for CUDA).
//
// THE RULE (root-caused in `ferrite-forward/tests/wavefront_sync_probe.rs`):
// on Apple GPU with relaxed-only MSL atomics, NON-atomic device writes are
// NOT reliably visible across threadgroups even with
// `threadgroup_barrier(mem_device)` (intra-TG). ATOMIC device ops ARE
// device-coherent. `flag_sync_sweep` proved TOKEN point-to-point handoff;
// these primitives add the ATOMIC bulk-data handoff that bulk activations
// crossing workers require. Proven by the A2 megakernel
// (`wavefront_qmv_mega_2stage`, bit-exact 50× × P∈{1,2,4,10}).
#pragma once
#include <metal_stdlib>
using namespace metal;

#ifndef METAL_FUNC
#define METAL_FUNC inline
#endif

namespace mittens {

// Hard cap on every spin so a non-co-resident launch fails the result check
// instead of hanging the GPU.
constant constexpr uint WF_SPIN_CAP = 100000000u;

// ── Point-to-point flags ────────────────────────────────────────────
//
// `wf_signal` publishes one worker's readiness; the caller MUST precede it
// with `threadgroup_barrier(mem_flags::mem_device)` (so the producer's data
// writes complete) and gate it to one thread. `wf_wait*` spin on the flag(s).

METAL_FUNC void wf_signal(device atomic_uint* flags, uint i) {
  atomic_store_explicit(&flags[i], 1u, memory_order_relaxed);
}

METAL_FUNC void wf_wait(device atomic_uint* flags, uint i) {
  uint spins = 0u;
  while (atomic_load_explicit(&flags[i], memory_order_relaxed) == 0u) {
    if (++spins > WF_SPIN_CAP) break;
  }
}

// Join: wait on every worker's flag in `[0, n)` (the all-to-all join a
// whole-tensor consumer needs).
METAL_FUNC void wf_wait_all(device atomic_uint* flags, uint n) {
  for (uint i = 0u; i < n; i++) wf_wait(flags, i);
}

// ── Atomic u32-packed bulk-data handoff (16-bit T: bf16 / f16) ───────
//
// Two 16-bit activations share one u32 atomic slot (lo = even index, hi =
// odd). The ATOMIC store/load is the cross-TG-visible path.

template <typename T>
METAL_FUNC uint wf_pack2(T lo, T hi) {
  return uint(as_type<ushort>(lo)) | (uint(as_type<ushort>(hi)) << 16);
}
template <typename T>
METAL_FUNC T wf_unpack_lo(uint v) { return as_type<T>(ushort(v & 0xFFFFu)); }
template <typename T>
METAL_FUNC T wf_unpack_hi(uint v) { return as_type<T>(ushort(v >> 16)); }

// Publish `n_pairs` value-pairs of `src` starting at value index `2*pair0`
// into the coherent `dst[pair0 ..]`, atomically. `src` is the producer's
// (intra-TG-visible) activation buffer. Call from one thread.
template <typename T>
METAL_FUNC void wf_publish_pairs(
    device atomic_uint* dst, const device T* src, uint pair0, uint n_pairs) {
  for (uint q = 0u; q < n_pairs; q++) {
    uint p = pair0 + q;
    atomic_store_explicit(
        &dst[p], wf_pack2<T>(src[2u * p], src[2u * p + 1u]), memory_order_relaxed);
  }
}

// Acquire `n_pairs` from the coherent `src` into `dst` (a 16-bit buffer),
// unpacking. Call from one thread, then `threadgroup_barrier` before the
// threadgroup reads `dst`.
template <typename T>
METAL_FUNC void wf_acquire_pairs(
    device T* dst, const device atomic_uint* src, uint n_pairs) {
  for (uint q = 0u; q < n_pairs; q++) {
    uint v = atomic_load_explicit(&src[q], memory_order_relaxed);
    dst[2u * q] = wf_unpack_lo<T>(v);
    dst[2u * q + 1u] = wf_unpack_hi<T>(v);
  }
}

// Threadgroup-cooperative variants: ALL `nthreads` lanes call these, lane
// `tid` owning the strided pairs `tid, tid+nthreads, ...`. The single-thread
// versions above serialize the whole activation on one lane (~1024 atomics);
// at a P-worker join that is ~P×(publish+acquire) single-lane passes per
// op-boundary and dominates the megakernel. Each pair is an independent
// atomic address, so striping across the TG parallelizes cleanly. The caller's
// existing `threadgroup_barrier(mem_device)` (after the PUBLISH compute, before
// the Signal; and after the ACQUIRE compute, before the consumer reads the
// private copy) still provides the completion fence — no barrier inside here.
template <typename T>
METAL_FUNC void wf_publish_pairs_tg(
    device atomic_uint* dst, const device T* src, uint pair0, uint n_pairs,
    uint tid, uint nthreads) {
  for (uint q = tid; q < n_pairs; q += nthreads) {
    uint p = pair0 + q;
    atomic_store_explicit(
        &dst[p], wf_pack2<T>(src[2u * p], src[2u * p + 1u]), memory_order_relaxed);
  }
}

template <typename T>
METAL_FUNC void wf_acquire_pairs_tg(
    device T* dst, const device atomic_uint* src, uint n_pairs,
    uint tid, uint nthreads) {
  for (uint q = tid; q < n_pairs; q += nthreads) {
    uint v = atomic_load_explicit(&src[q], memory_order_relaxed);
    dst[2u * q] = wf_unpack_lo<T>(v);
    dst[2u * q + 1u] = wf_unpack_hi<T>(v);
  }
}

// ── PAT-4: data-IS-the-flag handoff (NO barrier, NO separate flag) ───
//
// Proven barrier-free in `wavefront_sync_probe.rs` PAT 4: the producer's
// atomic store IS the readiness signal; the consumer spins on that same
// atomic slot until it's written, then reads it. The coherent buffer is
// reset to the SENTINEL (0) before the step, the producer guarantees a
// NON-sentinel store, and the consumer treats 0 as "not yet written".
// This removes the data-before-flag `threadgroup_barrier` and the separate
// Signal/Wait of the PAT-3 path entirely.

// Sentinel-safe pack: a pair of true-zero 16-bit values (`0x00000000`)
// would collide with the empty sentinel and hang the consumer's spin, so
// flush the low half from +0.0 (`0x0000`) to -0.0 (`0x8000`) in that one
// case. -0.0 is numerically identical to +0.0 in the downstream sums/
// products, so this is value-preserving; every other pair is left exact.
template <typename T>
METAL_FUNC uint wf_pack2_nz(T lo, T hi) {
  uint v = wf_pack2<T>(lo, hi);
  return v == 0u ? 0x00008000u : v;
}

// Producer side: publish `n_pairs` of `src` (sentinel-safe) into the
// coherent `dst[pair0..]`, TG-cooperative. No barrier needed afterward —
// the store itself is the readiness signal.
template <typename T>
METAL_FUNC void wf_publish_pairs_nz_tg(
    device atomic_uint* dst, const device T* src, uint pair0, uint n_pairs,
    uint tid, uint nthreads) {
  for (uint q = tid; q < n_pairs; q += nthreads) {
    uint p = pair0 + q;
    atomic_store_explicit(
        &dst[p], wf_pack2_nz<T>(src[2u * p], src[2u * p + 1u]), memory_order_relaxed);
  }
}

// Consumer side: spin on each coherent slot in `[0, n_pairs)` until it's
// non-sentinel (written this step), unpacking into the private `dst`.
// TG-cooperative; the spin per slot IS the wait, so no flag and no barrier
// precede it. The caller still barriers AFTER (before the threadgroup reads
// `dst`) only if `dst` is a shared copy read cross-simdgroup; a consumer
// that reads its own lane's slots needs nothing.
template <typename T>
METAL_FUNC void wf_acquire_pairs_spin_tg(
    device T* dst, const device atomic_uint* src, uint n_pairs,
    uint tid, uint nthreads) {
  for (uint q = tid; q < n_pairs; q += nthreads) {
    uint v = 0u, s = 0u;
    do {
      v = atomic_load_explicit(&src[q], memory_order_relaxed);
    } while (v == 0u && ++s < WF_SPIN_CAP);
    dst[2u * q] = wf_unpack_lo<T>(v);
    dst[2u * q + 1u] = wf_unpack_hi<T>(v);
  }
}

} // namespace mittens
