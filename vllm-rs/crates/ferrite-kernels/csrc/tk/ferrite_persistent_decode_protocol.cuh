// SPDX-License-Identifier: Apache-2.0
// Ferrite KVM pinned-memory protocol.
//
// Defines the communication buffer between the CPU and a persistent
// ferrite KVM decode kernel. The kernel runs indefinitely after a
// single cooperative launch; each decode step is coordinated through
// this buffer, which lives in CUDA pinned (page-locked) memory visible
// to both sides without explicit D2H/H2D copies.
//
// Synchronization protocol (step N):
//   CPU:
//     1. Write input fields (input_ids, positions, seq_lens,
//        slot_mapping, block_table, block_table_stride) for step N.
//     2. __atomic_thread_fence(memory_order_release)
//     3. Increment cpu_step to N+1.
//        ("step N input is ready")
//   Kernel (CTA 0 thread 0 polls; others wait at grid.sync):
//     4. Spin: while (protocol->cpu_step == N) { __nanosleep(100); }
//     5. __threadfence_system()  — acquire fence on input fields.
//     6. grid.sync() — wake all CTAs; check stop_flag.
//     7. Run full decode step (forward pass + argmax).
//     8. grid.sync() — all CTAs finished.
//     9. CTA 0 thread 0: write output_tokens, then:
//        __threadfence_system();
//        protocol->gpu_step = N+1;  — "step N output is ready"
//   CPU:
//    10. Spin: while (protocol->gpu_step == N) { /* poll */ }
//    11. Read output_tokens. Run sampling. Stage step N+1 input.
//
// Stop protocol: CPU writes stop_flag=1 BEFORE incrementing cpu_step.
//   Kernel checks stop_flag after observing cpu_step advance. If set,
//   all CTAs break the persistent loop (no argmax/signal for that step;
//   cpu_step advance is the stop signal, protocol->gpu_step is NOT
//   incremented — this lets CPU detect a clean stop vs stall).
//
// Layout (NUM_TOKENS = batch size, MAX_BLOCKS_PER_SEQ = KV depth):
//
//   [0,  64)  — sync header (cpu_step, gpu_step, stop_flag + padding)
//   [64, 64 + small_fields) — per-step input scalars/arrays
//   [large_offset, ...)    — block_table (large, read per KV page)
//   [last 4*N bytes]       — output_tokens

#pragma once

namespace ferrite {
namespace persistent_decode {

// ------------------------------------------------------------------
// FerritePinnedProtocol<NUM_TOKENS, MAX_BLOCKS_PER_SEQ>
//
// Pinned (page-locked) memory buffer shared between CPU and kernel.
// Instantiated once per persistent decode session and passed as a
// single pointer to the persistent decode kernel. NUM_TOKENS is the decode batch
// size; MAX_BLOCKS_PER_SEQ is the maximum KV blocks per sequence
// (512 * BLOCK_SIZE=16 → 8192 token max context).
// ------------------------------------------------------------------

template <int NUM_TOKENS, int MAX_BLOCKS_PER_SEQ>
struct __attribute__((packed)) FerritePinnedProtocol {

    // ---- Sync header (first 64 bytes / one cache line) ------------
    // Written and read by ONE side only (cpu_step by CPU, gpu_step by
    // kernel), so no CAS is needed. volatile ensures the compiler does
    // not cache/reorder these in registers across the spin loop.

    volatile uint32_t cpu_step;    // CPU increments when step input ready
    volatile uint32_t gpu_step;    // Kernel increments when step output ready
    volatile uint32_t stop_flag;   // CPU sets to 1 to request graceful exit
    uint8_t  _sync_pad[64 - 3 * sizeof(uint32_t)];  // pad to 64 bytes

    // ---- Per-step input: small/scalar fields (cache-line 2) -------
    // CPU writes all fields before the cpu_step increment. Kernel
    // reads after the acquire fence in wait_for_cpu_step.

    uint32_t input_ids [NUM_TOKENS];   // current decode token IDs
    uint32_t positions [NUM_TOKENS];   // absolute position indices
    int32_t  seq_lens  [NUM_TOKENS];   // sequence lengths for attention

    // Explicit padding so slot_mapping (int64_t) is 8-byte aligned.
    // For NUM_TOKENS=1: input/positions/seq_lens = 12 bytes;
    // offset 64+12=76 is not 8-byte aligned → need 4 bytes pad.
    // For NUM_TOKENS=2: 24 bytes; 64+24=88 → already 8-byte aligned
    //   (the padding field evaluates to zero bytes for even NUM_TOKENS).
    // Compile-time: (NUM_TOKENS * 12) % 8 != 0 only when NUM_TOKENS is odd.
    uint8_t  _align_pad[(NUM_TOKENS * 12) % 8 ? 4 : 0];

    int64_t  slot_mapping[NUM_TOKENS]; // per-token KV write slots (i64)

    uint32_t block_table_stride;       // runtime column stride of block_table
    uint32_t _bt_stride_pad;           // align block_table (already u32, ok)

    // ---- Per-step input: block table (large) ----------------------
    // [NUM_TOKENS, MAX_BLOCKS_PER_SEQ] u32 array. Row i = sequence i.
    // block_table_stride is the allocated column count (≤ MAX_BLOCKS).
    // Only columns 0..seq_blocks-1 for each row hold valid block IDs;
    // the rest are zero-padded and never read by the kernel.

    uint32_t block_table[NUM_TOKENS * MAX_BLOCKS_PER_SEQ];

    // ---- Per-step output (kernel writes, CPU reads) ---------------
    // Written by CTA 0 thread 0 before gpu_step increment.
    // For greedy decode: output_tokens[i] is the argmax token for seq i.

    uint32_t output_tokens[NUM_TOKENS];
};

// ------------------------------------------------------------------
// Kernel-side helpers
// ------------------------------------------------------------------

// Called by CTA 0 thread 0 only. Spins until cpu_step > current_step.
// On return the acquire fence ensures subsequent reads of protocol
// input fields see the writes the CPU made before incrementing cpu_step.
template <int NUM_TOKENS, int MAX_BLOCKS_PER_SEQ>
__device__ __forceinline__ void wait_for_cpu_step(
    FerritePinnedProtocol<NUM_TOKENS, MAX_BLOCKS_PER_SEQ>* protocol,
    uint32_t current_step
) {
    while (protocol->cpu_step == current_step) {
        __nanosleep(100);
    }
    __threadfence_system();  // acquire: see CPU writes before cpu_step bump
}

// Called by CTA 0 thread 0 only, after output_tokens are written.
// Publishes the step result to the CPU.
template <int NUM_TOKENS, int MAX_BLOCKS_PER_SEQ>
__device__ __forceinline__ void signal_gpu_step(
    FerritePinnedProtocol<NUM_TOKENS, MAX_BLOCKS_PER_SEQ>* protocol,
    uint32_t new_gpu_step
) {
    __threadfence_system();          // release: CPU sees output_tokens
    protocol->gpu_step = new_gpu_step;
}

// ------------------------------------------------------------------
// Host-side layout constants (used by Rust KvmSession)
// ------------------------------------------------------------------
//
// Rust computes the buffer size as:
//   size_of::<FerritePinnedProtocol<N, M>>()
//   = 64                            (sync header)
//   + 4*N + 4*N + 4*N              (input_ids, positions, seq_lens)
//   + ((N*12)%8 ? 4 : 0)           (alignment pad)
//   + 8*N                          (slot_mapping)
//   + 4 + 4                        (block_table_stride + _bt_stride_pad)
//   + 4*N*M                        (block_table)
//   + 4*N                          (output_tokens)
//
// Field offsets (for N=1, M=512):
//   cpu_step          : 0
//   gpu_step          : 4
//   stop_flag         : 8
//   input_ids[0]      : 64
//   positions[0]      : 68
//   seq_lens[0]       : 72
//   _align_pad        : 76  (4 bytes, N=1 is odd → needs align)
//   slot_mapping[0]   : 80  (8-byte aligned ✓)
//   block_table_stride: 88
//   _bt_stride_pad    : 92
//   block_table[0]    : 96
//   output_tokens[0]  : 96 + 4*512 = 2144
//   total             : 2148 bytes → allocate 4096 (one page)

}  // namespace persistent_decode
}  // namespace ferrite
