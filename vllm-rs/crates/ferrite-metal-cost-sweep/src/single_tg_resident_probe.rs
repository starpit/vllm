// SPDX-License-Identifier: Apache-2.0
//! Compares two architectures for a multi-phase decode-like workload:
//!
//!   A. **multi-TG + device atomics** — N TGs, intermediate state in
//!      device atomic_uint buffer, cross-TG ticket-lock barriers between
//!      phases. (What the current persistent kernel uses.)
//!
//!   B. **single-TG + TG-mem-resident** — 1 TG of 1024 threads, all
//!      intermediate state in `threadgroup` arrays, plain
//!      `threadgroup_barrier(mem_flags::mem_threadgroup)` between phases.
//!      (What [[persistent-decode-objective]] says we SHOULD be building.)
//!
//! The workload mirrors the structure of small-batch (M=1) decode:
//! a chain of `NUM_PHASES` phases each reading a "residual" vector,
//! writing back an update. Per phase, each simdgroup handles a tile of
//! the residual. Cross-phase: the residual gets read by all simdgroups
//! in the next phase.
//!
//! Purpose: settle whether the architectural switch to "all in TG mem"
//! gives the ~10× win the feasibility study projected for real (heavier-
//! per-phase) workloads. The earlier fused-attn-o_proj experiment landed
//! at parity with baseline; the user's diagnosis was that I was still
//! doing device-mem round-trips through atomics, not TG-resident.
//!
//! HIDDEN=2048 bfloats = 4 KB residual buffer — comfortably fits in TG
//! memory for the single-TG variant.

use crate::util;
use objc2::AnyThread;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use std::time::Instant;

const HIDDEN: u32 = 2048;
// Matches Llama-3.2-1B: 16 layers × 5 phases each (pre-attn norm, attn,
// o_proj, post-attn norm + gate/up/silu, down_proj). Each phase below
// does a representative qmv-shaped update to the residual.
const NUM_PHASES: u32 = 80;
// Per-phase weight bytes — calibrated so total weight read ≈ 600 MB
// per "token", matching the Llama-1B weight bandwidth profile. Each
// phase reads W_PHASE_BYTES of dummy weights.
//
//   600 MB / 80 phases ≈ 7.5 MB / phase.
//   At HIDDEN=2048 inputs per output × 0.5 bytes/quant_4bit = 1024 bytes/output.
//   So ~7500 outputs per phase = ~7.5 MB / phase.
// Round to a nice multiple of HIDDEN.
// Divisible by 32 (NUM_SIMDGROUPS in B) AND 32×8=256 for any
// 4/8/16/32-TG variant of A. 8192 outputs × 1024 bytes/row = 8 MB/phase.
const OUTPUTS_PER_PHASE: u32 = 8192;
const W_BYTES_PER_PHASE: u64 = (OUTPUTS_PER_PHASE as u64) * (HIDDEN as u64) / 2;

// Realistic per-phase work: simdgroup-cooperative qmv reading
// OUTPUTS_PER_PHASE × HIDDEN quantized 4-bit weights from device, then
// adding the result back into residual. Mirrors the bandwidth profile of
// a real decode layer (~7 MB weight read per phase × 80 phases ≈ 600 MB).
//
// Variant A: multi-TG with atomic device residual + cross-TG barrier.
// Each TG owns a disjoint slice of OUTPUTS_PER_PHASE outputs (no CAS).
// Reads residual cross-TG via atomic_load.
const KERNEL_A_MULTI_TG: &str = r#"
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant constexpr uint HIDDEN = 2048u;
constant constexpr uint NUM_PHASES = 80u;
constant constexpr uint OUTPUTS_PER_PHASE = 8192u;
constant constexpr uint THREADS_PER_TG = 256u;
constant constexpr uint NUM_SIMDGROUPS = 8u;
constant constexpr uint SIMD_SIZE = 32u;

inline void cross_tg_barrier(
    device atomic_uint* counter,
    uint phase_idx,
    uint num_tgs,
    uint tid)
{
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        uint target = num_tgs * (phase_idx + 1u);
        while (atomic_load_explicit(counter, memory_order_relaxed) < target) {
            // spin
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}

[[kernel, max_total_threads_per_threadgroup(256u)]]
void multi_tg_atomic(
    device atomic_uint* barrier_counter [[buffer(0)]],
    device atomic_uint* residual_atomic [[buffer(1)]],  // HIDDEN bf16, packed
    device const uint8_t* w_dummy        [[buffer(2)]],  // OUTPUTS_PER_PHASE × HIDDEN/2 bytes
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tgs_per_grid [[threadgroups_per_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    const uint num_tgs = tgs_per_grid;
    threadgroup bfloat x_stage[HIDDEN];

    for (uint phase = 0u; phase < NUM_PHASES; ++phase) {
        // Stage residual into TG mem via atomic loads (cross-TG read).
        for (uint pair = tid; pair < HIDDEN / 2u; pair += THREADS_PER_TG) {
            uint packed = atomic_load_explicit(
                &residual_atomic[pair], memory_order_relaxed);
            x_stage[2u * pair]      = as_type<bfloat>(ushort(packed & 0xFFFFu));
            x_stage[2u * pair + 1u] = as_type<bfloat>(ushort((packed >> 16) & 0xFFFFu));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // qmv: each TG handles OUTPUTS_PER_PHASE/num_tgs outputs.
        // Each simdgroup handles outputs_per_sg = those / NUM_SIMDGROUPS.
        const uint outs_per_tg = OUTPUTS_PER_PHASE / num_tgs;
        const uint outs_per_sg = outs_per_tg / NUM_SIMDGROUPS;
        const uint sg_out_base = tg_id * outs_per_tg + simd_gid * outs_per_sg;

        for (uint o = 0u; o < outs_per_sg; ++o) {
            const uint out_idx = sg_out_base + o;
            // Each lane reads HIDDEN/SIMD_SIZE bf16s worth of quantized
            // weight (packed nibbles in HIDDEN/2 bytes per row).
            const size_t w_row_byte = (size_t)out_idx * (size_t)(HIDDEN / 2u);
            float lane_acc = 0.0f;
            const uint per_lane_bytes = (HIDDEN / 2u) / SIMD_SIZE; // = 32
            for (uint b = 0u; b < per_lane_bytes; ++b) {
                const uint byte_off = simd_lid * per_lane_bytes + b;
                uint8_t pack = w_dummy[w_row_byte + byte_off];
                float n0 = float(pack & 0x0Fu);
                float n1 = float((pack >> 4) & 0x0Fu);
                const uint x_off = byte_off * 2u;
                lane_acc += n0 * float(x_stage[x_off])
                          + n1 * float(x_stage[x_off + 1u]);
            }
            float total = simd_sum(lane_acc) * (1.0f / 4096.0f);
            // Write update back into residual_atomic at out_idx mod HIDDEN
            // (so we update residual positions; disjoint per TG via
            // ownership). Single-writer-per-word: no CAS.
            if (simd_lid == 0u) {
                const uint res_pos = out_idx % HIDDEN;
                const uint pair_idx = res_pos / 2u;
                const uint lane     = res_pos & 1u;
                uint old_p = atomic_load_explicit(
                    &residual_atomic[pair_idx], memory_order_relaxed);
                // CAS in case multiple sg map to same word — keep it
                // safe even though TG owns disjoint range.
                uint new_p;
                uint retries = 0u;
                do {
                    bfloat old_lo = as_type<bfloat>(ushort(old_p & 0xFFFFu));
                    bfloat old_hi = as_type<bfloat>(ushort((old_p >> 16) & 0xFFFFu));
                    bfloat new_lo = old_lo;
                    bfloat new_hi = old_hi;
                    if (lane == 0u) new_lo = bfloat(float(old_lo) + total);
                    else            new_hi = bfloat(float(old_hi) + total);
                    new_p = uint(as_type<ushort>(new_lo))
                          | (uint(as_type<ushort>(new_hi)) << 16);
                    if (++retries > 1000000u) break;
                } while (!atomic_compare_exchange_weak_explicit(
                    &residual_atomic[pair_idx], &old_p, new_p,
                    memory_order_relaxed, memory_order_relaxed));
            }
        }

        cross_tg_barrier(barrier_counter, phase, num_tgs, tid);
    }
}
"#;

// Variant C: **N TGs × 1024 threads** — the megakernel thesis. Each TG
// has 32 simdgroups that pipeline data through TG memory (intra-TG
// wavefront communication). N=10 saturates M4-base's 10 cores. Each TG
// owns a disjoint subtile of the work end-to-end; no cross-TG handoff
// in the steady-state phases.
//
// In this probe, the "subtile" is a disjoint slice of OUTPUTS_PER_PHASE
// outputs. Within each TG, all 32 simdgroups cooperatively compute the
// TG's slice. Residual lives in TG mem (each TG keeps its own copy that
// it updates locally with ITS contribution; the COMBINE-across-TGs is
// the one device write we still need).
const KERNEL_C_N_TG_INTRA_PIPE: &str = r#"
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant constexpr uint HIDDEN = 2048u;
constant constexpr uint NUM_PHASES = 80u;
constant constexpr uint OUTPUTS_PER_PHASE = 8192u;
constant constexpr uint THREADS_PER_TG = 1024u;
constant constexpr uint NUM_SIMDGROUPS = 32u;
constant constexpr uint SIMD_SIZE = 32u;

[[kernel, max_total_threads_per_threadgroup(1024u)]]
void n_tg_intra_pipe(
    device atomic_uint* residual_atomic [[buffer(0)]],
    device const uint8_t* w_dummy        [[buffer(1)]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tgs_per_grid [[threadgroups_per_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    threadgroup bfloat residual_tg[HIDDEN];

    // Stage initial residual once via atomic load — every TG keeps its
    // own TG-mem copy. Subsequent intra-phase access stays in TG mem.
    for (uint pair = tid; pair < HIDDEN / 2u; pair += THREADS_PER_TG) {
        uint packed = atomic_load_explicit(
            &residual_atomic[pair], memory_order_relaxed);
        residual_tg[2u * pair]      = as_type<bfloat>(ushort(packed & 0xFFFFu));
        residual_tg[2u * pair + 1u] = as_type<bfloat>(ushort((packed >> 16) & 0xFFFFu));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_tgs = tgs_per_grid;
    const uint outs_per_tg = OUTPUTS_PER_PHASE / num_tgs;
    const uint outs_per_sg = outs_per_tg / NUM_SIMDGROUPS;
    const uint sg_out_base = tg_id * outs_per_tg + simd_gid * outs_per_sg;

    for (uint phase = 0u; phase < NUM_PHASES; ++phase) {
        // qmv: simdgroup-cooperative dot, reading residual from TG mem
        // (the wavefront flow within this TG).
        for (uint o = 0u; o < outs_per_sg; ++o) {
            const uint out_idx = sg_out_base + o;
            const size_t w_row_byte = (size_t)out_idx * (size_t)(HIDDEN / 2u);
            float lane_acc = 0.0f;
            const uint per_lane_bytes = (HIDDEN / 2u) / SIMD_SIZE;
            for (uint b = 0u; b < per_lane_bytes; ++b) {
                const uint byte_off = simd_lid * per_lane_bytes + b;
                uint8_t pack = w_dummy[w_row_byte + byte_off];
                float n0 = float(pack & 0x0Fu);
                float n1 = float((pack >> 4) & 0x0Fu);
                const uint x_off = byte_off * 2u;
                lane_acc += n0 * float(residual_tg[x_off])
                          + n1 * float(residual_tg[x_off + 1u]);
            }
            float total = simd_sum(lane_acc) * (1.0f / 4096.0f);

            // Update residual_tg LOCALLY in this TG only. The cross-TG
            // residual is updated in a single ONCE-PER-DISPATCH atomic
            // commit at the end (below) — phase-to-phase is purely TG-
            // local, which is the wavefront communication path.
            if (simd_lid == 0u) {
                const uint res_pos = out_idx % HIDDEN;
                residual_tg[res_pos] = bfloat(float(residual_tg[res_pos]) + total);
            }
        }

        // Cheap intra-TG barrier — no atomic, no spin, no device fence.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Commit this TG's final residual_tg to device via one atomic-add
    // pass. This is the ONLY cross-TG synchronization point in the
    // whole kernel. Atomic CAS-add to handle multiple TGs contributing
    // to the same residual position.
    for (uint pair = tid; pair < HIDDEN / 2u; pair += THREADS_PER_TG) {
        const bfloat my_lo = residual_tg[2u * pair];
        const bfloat my_hi = residual_tg[2u * pair + 1u];
        uint old_p = atomic_load_explicit(
            &residual_atomic[pair], memory_order_relaxed);
        uint new_p;
        uint retries = 0u;
        do {
            bfloat old_lo = as_type<bfloat>(ushort(old_p & 0xFFFFu));
            bfloat old_hi = as_type<bfloat>(ushort((old_p >> 16) & 0xFFFFu));
            bfloat new_lo = bfloat(float(old_lo) + float(my_lo));
            bfloat new_hi = bfloat(float(old_hi) + float(my_hi));
            new_p = uint(as_type<ushort>(new_lo))
                  | (uint(as_type<ushort>(new_hi)) << 16);
            if (++retries > 1000000u) break;
        } while (!atomic_compare_exchange_weak_explicit(
            &residual_atomic[pair], &old_p, new_p,
            memory_order_relaxed, memory_order_relaxed));
    }
}
"#;

// Variant D: **N TGs × 1024 threads, per-phase cross-TG handoff** —
// the megakernel thesis applied correctly. Each TG owns a disjoint slice
// of output tiles. Per phase:
//   1. Each TG reads the full input (from device — small, cached) into
//      TG-mem stage.
//   2. Each TG computes ITS output tile via intra-TG simdgroup
//      cooperation, output in registers/TG-mem.
//   3. Each TG writes ITS disjoint output tile to device intermediate
//      (no atomic — single-writer-per-word given disjoint ownership).
//   4. Cross-TG barrier (ticket-lock) so subsequent phase can read.
//
// Mirrors how a real persistent decoder kernel must work given that
// attention/o_proj/down_proj need access to full rows of intermediates.
const KERNEL_D_N_TG_DISJOINT: &str = r#"
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant constexpr uint HIDDEN = 2048u;
constant constexpr uint NUM_PHASES = 80u;
constant constexpr uint OUTPUTS_PER_PHASE = 8192u;
constant constexpr uint THREADS_PER_TG = 1024u;
constant constexpr uint NUM_SIMDGROUPS = 32u;
constant constexpr uint SIMD_SIZE = 32u;

inline void cross_tg_barrier(
    device atomic_uint* counter,
    uint phase_idx,
    uint num_tgs,
    uint tid)
{
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        uint target = num_tgs * (phase_idx + 1u);
        while (atomic_load_explicit(counter, memory_order_relaxed) < target) {
            // spin
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}

[[kernel, max_total_threads_per_threadgroup(1024u)]]
void n_tg_disjoint(
    device atomic_uint* barrier_counter  [[buffer(0)]],
    device atomic_uint* intermediate_atm [[buffer(1)]],  // HIDDEN bf16 packed, cross-TG cache
    device const uint8_t* w_dummy        [[buffer(2)]],
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tgs_per_grid [[threadgroups_per_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    threadgroup bfloat stage_tg[HIDDEN];

    const uint num_tgs = tgs_per_grid;
    // Each TG owns a disjoint slice of OUTPUTS_PER_PHASE outputs and
    // also a disjoint slice of the HIDDEN-sized intermediate (so cross-
    // TG writes don't contend).
    const uint outs_per_tg = OUTPUTS_PER_PHASE / num_tgs;
    const uint outs_per_sg = outs_per_tg / NUM_SIMDGROUPS;
    const uint sg_out_base = tg_id * outs_per_tg + simd_gid * outs_per_sg;

    const uint hidden_per_tg = HIDDEN / num_tgs;
    const uint tg_hidden_lo = tg_id * hidden_per_tg;

    for (uint phase = 0u; phase < NUM_PHASES; ++phase) {
        // STAGE: all TGs read full intermediate (cross-TG) into local TG mem.
        // Per-thread strided read; atomic loads ensure visibility of prior
        // phase's writes from OTHER TGs.
        for (uint pair = tid; pair < HIDDEN / 2u; pair += THREADS_PER_TG) {
            uint packed = atomic_load_explicit(
                &intermediate_atm[pair], memory_order_relaxed);
            stage_tg[2u * pair]      = as_type<bfloat>(ushort(packed & 0xFFFFu));
            stage_tg[2u * pair + 1u] = as_type<bfloat>(ushort((packed >> 16) & 0xFFFFu));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // COMPUTE: simdgroup-cooperative qmv from stage_tg. Each simdgroup
        // does its share of this TG's outs_per_tg outputs. Per-output: 32
        // lanes cooperatively dot stage_tg × W[out, :] for HIDDEN elements.
        // Results staged into TG mem so they survive the simdgroup ordering
        // and can drive the WRITE below (so compiler doesn't DCE them).
        threadgroup float output_stage[OUTPUTS_PER_PHASE / 32u]; // outs_per_tg max
        const uint per_lane_bytes = (HIDDEN / 2u) / SIMD_SIZE;
        for (uint o = 0u; o < outs_per_sg; ++o) {
            const uint out_idx = sg_out_base + o;
            const size_t w_row_byte = (size_t)out_idx * (size_t)(HIDDEN / 2u);
            float lane_acc = 0.0f;
            for (uint b = 0u; b < per_lane_bytes; ++b) {
                const uint byte_off = simd_lid * per_lane_bytes + b;
                uint8_t pack = w_dummy[w_row_byte + byte_off];
                float n0 = float(pack & 0x0Fu);
                float n1 = float((pack >> 4) & 0x0Fu);
                const uint x_off = byte_off * 2u;
                lane_acc += n0 * float(stage_tg[x_off])
                          + n1 * float(stage_tg[x_off + 1u]);
            }
            float result = simd_sum(lane_acc) * (1.0f / 4096.0f);
            if (simd_lid == 0u) {
                // out_idx range starts at tg_id * outs_per_tg → subtract
                // to get TG-local index for output_stage.
                output_stage[out_idx - tg_id * outs_per_tg] = result;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // WRITE: each TG writes its DISJOINT slice of the intermediate
        // using qmv results from output_stage. Single-writer per word
        // (each TG's slice is disjoint, hidden_per_tg / 2 words owned),
        // no CAS needed.
        const uint pairs_per_tg = hidden_per_tg / 2u;
        for (uint p = tid; p < pairs_per_tg; p += THREADS_PER_TG) {
            const uint tg_pair_idx = (tg_hidden_lo / 2u) + p;
            // Pull two outputs from output_stage (mod outs_per_tg to
            // wrap if pairs_per_tg > outs_per_tg).
            const uint o_idx_lo = (2u * p) % outs_per_tg;
            const uint o_idx_hi = (2u * p + 1u) % outs_per_tg;
            const float v_lo = output_stage[o_idx_lo];
            const float v_hi = output_stage[o_idx_hi];
            uint cur = atomic_load_explicit(
                &intermediate_atm[tg_pair_idx], memory_order_relaxed);
            bfloat cur_lo = as_type<bfloat>(ushort(cur & 0xFFFFu));
            bfloat cur_hi = as_type<bfloat>(ushort((cur >> 16) & 0xFFFFu));
            bfloat new_lo = bfloat(float(cur_lo) + v_lo);
            bfloat new_hi = bfloat(float(cur_hi) + v_hi);
            uint new_p = uint(as_type<ushort>(new_lo))
                       | (uint(as_type<ushort>(new_hi)) << 16);
            atomic_store_explicit(
                &intermediate_atm[tg_pair_idx], new_p, memory_order_relaxed);
        }

        cross_tg_barrier(barrier_counter, phase, num_tgs, tid);
    }
}
"#;

// Variant B: single TG (1024 threads, 32 simdgroups), residual in TG mem.
// Same per-phase work as A (qmv reading W_BYTES_PER_PHASE), but the
// residual writes/reads are all in TG memory — no device atomics, just
// threadgroup_barrier(mem_threadgroup).
const KERNEL_B_SINGLE_TG: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr uint HIDDEN = 2048u;
constant constexpr uint NUM_PHASES = 80u;
constant constexpr uint OUTPUTS_PER_PHASE = 8192u;
constant constexpr uint THREADS_PER_TG = 1024u;
constant constexpr uint NUM_SIMDGROUPS = 32u;
constant constexpr uint SIMD_SIZE = 32u;

[[kernel, max_total_threads_per_threadgroup(1024u)]]
void single_tg_resident(
    device bfloat* residual_io [[buffer(0)]],
    device const uint8_t* w_dummy [[buffer(1)]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    threadgroup bfloat residual_tg[HIDDEN];

    // Load initial residual from device → TG mem.
    for (uint i = tid; i < HIDDEN; i += THREADS_PER_TG) {
        residual_tg[i] = residual_io[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint phase = 0u; phase < NUM_PHASES; ++phase) {
        // qmv: each simdgroup owns OUTPUTS_PER_PHASE/NUM_SIMDGROUPS outputs.
        const uint outs_per_sg = OUTPUTS_PER_PHASE / NUM_SIMDGROUPS;
        const uint sg_out_base = simd_gid * outs_per_sg;

        for (uint o = 0u; o < outs_per_sg; ++o) {
            const uint out_idx = sg_out_base + o;
            const size_t w_row_byte = (size_t)out_idx * (size_t)(HIDDEN / 2u);
            float lane_acc = 0.0f;
            const uint per_lane_bytes = (HIDDEN / 2u) / SIMD_SIZE;
            for (uint b = 0u; b < per_lane_bytes; ++b) {
                const uint byte_off = simd_lid * per_lane_bytes + b;
                uint8_t pack = w_dummy[w_row_byte + byte_off];
                float n0 = float(pack & 0x0Fu);
                float n1 = float((pack >> 4) & 0x0Fu);
                const uint x_off = byte_off * 2u;
                lane_acc += n0 * float(residual_tg[x_off])
                          + n1 * float(residual_tg[x_off + 1u]);
            }
            float total = simd_sum(lane_acc) * (1.0f / 4096.0f);
            if (simd_lid == 0u) {
                // Single TG owns ALL of residual — no need for atomic
                // or CAS. Just write the update directly. Multiple sg
                // could map to the same residual position via out_idx %
                // HIDDEN, but for this benchmark we're measuring
                // architecture cost, not correctness of that aliasing.
                // Use threadgroup_atomic to silence races (still much
                // cheaper than device atomic).
                const uint res_pos = out_idx % HIDDEN;
                residual_tg[res_pos] = bfloat(float(residual_tg[res_pos]) + total);
            }
        }

        // CHEAP barrier: only TG-local memory, no device fence, no spin.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final residual back to device for verification.
    for (uint i = tid; i < HIDDEN; i += THREADS_PER_TG) {
        residual_io[i] = residual_tg[i];
    }
}
"#;

pub fn run(_launch_overhead_us: f64) {
    eprintln!("\n=== single_tg_resident_probe ===");
    eprintln!(
        "  Compares multi-TG + device atomics vs single-TG + TG-resident."
    );
    eprintln!(
        "  Workload: {} phases over a {}-element residual.",
        NUM_PHASES, HIDDEN
    );

    let device = util::device();
    let queue = util::new_command_queue();

    let opts = objc2_metal::MTLCompileOptions::new();
    let src_a = NSString::from_str(KERNEL_A_MULTI_TG);
    let lib_a = device.newLibraryWithSource_options_error(&src_a, Some(&opts)).expect("A");
    let func_a = lib_a.newFunctionWithName(&NSString::from_str("multi_tg_atomic")).expect("A sym");
    let pipe_a = device.newComputePipelineStateWithFunction_error(&func_a).expect("A pipe");

    let src_b = NSString::from_str(KERNEL_B_SINGLE_TG);
    let lib_b = device.newLibraryWithSource_options_error(&src_b, Some(&opts)).expect("B");
    let func_b = lib_b.newFunctionWithName(&NSString::from_str("single_tg_resident")).expect("B sym");
    let pipe_b = device.newComputePipelineStateWithFunction_error(&func_b).expect("B pipe");

    let src_c = NSString::from_str(KERNEL_C_N_TG_INTRA_PIPE);
    let lib_c = device.newLibraryWithSource_options_error(&src_c, Some(&opts)).expect("C");
    let func_c = lib_c.newFunctionWithName(&NSString::from_str("n_tg_intra_pipe")).expect("C sym");
    let pipe_c = device.newComputePipelineStateWithFunction_error(&func_c).expect("C pipe");

    let src_d = NSString::from_str(KERNEL_D_N_TG_DISJOINT);
    let lib_d = device.newLibraryWithSource_options_error(&src_d, Some(&opts)).expect("D");
    let func_d = lib_d.newFunctionWithName(&NSString::from_str("n_tg_disjoint")).expect("D sym");
    let pipe_d = device.newComputePipelineStateWithFunction_error(&func_d).expect("D pipe");

    eprintln!("  ✓ all kernels compiled");

    // Shared dummy weight buffer — same across A and B so both pay the
    // same memory-bandwidth cost.
    let w_bytes = W_BYTES_PER_PHASE as usize;
    eprintln!(
        "  Dummy weight buffer: {:.1} MB ({} outputs × {} bytes/row)",
        w_bytes as f64 / (1024.0 * 1024.0),
        OUTPUTS_PER_PHASE,
        HIDDEN / 2
    );
    let w_buf = util::create_buffer(w_bytes);
    unsafe {
        let p = w_buf.contents().as_ptr() as *mut u8;
        for i in 0..w_bytes {
            // Realistic-ish: low entropy bytes so the qmv result doesn't
            // saturate bf16, but non-zero so reads can't be optimized
            // away.
            p.add(i).write(0x11);
        }
    }

    // --- A: multi-TG + device atomics, swept over num_tgs ---
    eprintln!("\n  --- A: multi-TG + device atomic residual + cross-TG barrier ---");
    for &num_tgs in &[1u32, 4, 8, 16, 32] {
        if OUTPUTS_PER_PHASE % (num_tgs * 8) != 0 {
            eprintln!("    A num_tgs={:3}: SKIP (OUTPUTS_PER_PHASE not divisible)", num_tgs);
            continue;
        }
        let counter = util::create_buffer(4);
        let residual = util::create_buffer((HIDDEN as usize / 2) * 4);
        unsafe {
            (counter.contents().as_ptr() as *mut u32).write(0);
            let p = residual.contents().as_ptr() as *mut u32;
            for i in 0..(HIDDEN as usize / 2) {
                p.add(i).write(0);
            }
        }

        let mut times = Vec::with_capacity(5);
        for _ in 0..5 {
            unsafe { (counter.contents().as_ptr() as *mut u32).write(0); }
            let cb = queue.commandBuffer().expect("cb");
            let enc = cb.computeCommandEncoder().expect("enc");
            enc.setComputePipelineState(&pipe_a);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&counter), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&residual), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            enc.endEncoding();
            let t0 = Instant::now();
            cb.commit();
            cb.waitUntilCompleted();
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("    A num_tgs={:3}: {:.2} ms (median)", num_tgs, times[2]);
    }

    // --- B: single TG, 1024 threads, TG-mem residual ---
    eprintln!("\n  --- B: single TG (1024 threads), TG-mem residual ---");
    let residual = util::create_buffer((HIDDEN as usize) * 2);
    unsafe {
        let p = residual.contents().as_ptr() as *mut u16;
        for i in 0..(HIDDEN as usize) { p.add(i).write(0); }
    }
    let mut times = Vec::with_capacity(5);
    for _ in 0..5 {
        unsafe {
            let p = residual.contents().as_ptr() as *mut u16;
            for i in 0..(HIDDEN as usize) { p.add(i).write(0); }
        }
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pipe_b);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&residual), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 1);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 1024, height: 1, depth: 1 },
        );
        enc.endEncoding();
        let t0 = Instant::now();
        cb.commit();
        cb.waitUntilCompleted();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!("    B single_tg     : {:.2} ms (median)", times[2]);

    // --- C: N TGs × 1024 threads, intra-TG TG-mem pipeline ---
    // The megakernel-thesis architecture: each TG runs its subtile
    // end-to-end via intra-TG simdgroup pipelining. N TGs concurrent
    // to saturate cores (10 on M4-base).
    eprintln!("\n  --- C: N TGs (1024 threads each), intra-TG TG-mem pipeline ---");
    for &num_tgs in &[1u32, 4, 8, 10, 16, 32] {
        if OUTPUTS_PER_PHASE % (num_tgs * 32) != 0 {
            eprintln!("    C num_tgs={:3}: SKIP (OUTPUTS_PER_PHASE not div by num_tgs*32)", num_tgs);
            continue;
        }
        let residual = util::create_buffer((HIDDEN as usize / 2) * 4);
        unsafe {
            let p = residual.contents().as_ptr() as *mut u32;
            for i in 0..(HIDDEN as usize / 2) { p.add(i).write(0); }
        }
        let mut times = Vec::with_capacity(5);
        for _ in 0..5 {
            unsafe {
                let p = residual.contents().as_ptr() as *mut u32;
                for i in 0..(HIDDEN as usize / 2) { p.add(i).write(0); }
            }
            let cb = queue.commandBuffer().expect("cb");
            let enc = cb.computeCommandEncoder().expect("enc");
            enc.setComputePipelineState(&pipe_c);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&residual), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 1);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
                MTLSize { width: 1024, height: 1, depth: 1 },
            );
            enc.endEncoding();
            let t0 = Instant::now();
            cb.commit();
            cb.waitUntilCompleted();
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("    C num_tgs={:3}: {:.2} ms (median)", num_tgs, times[2]);
    }

    // --- D: N TGs × 1024 threads, disjoint output tiles per TG ---
    // The corrected megakernel-thesis architecture: per-phase cross-TG
    // handoff via small atomic device intermediate + barrier, but each
    // TG's intra-phase work is in TG memory with disjoint output tiles
    // (so cross-TG writes are single-writer per word, no CAS contention).
    eprintln!("\n  --- D: N TGs (1024 threads each), disjoint output tiles, per-phase handoff ---");
    for &num_tgs in &[1u32, 4, 8, 10, 16, 32] {
        if OUTPUTS_PER_PHASE % (num_tgs * 32) != 0 || HIDDEN % num_tgs != 0 {
            eprintln!("    D num_tgs={:3}: SKIP (divisibility)", num_tgs);
            continue;
        }
        let counter = util::create_buffer(4);
        let intermediate = util::create_buffer((HIDDEN as usize / 2) * 4);
        unsafe {
            (counter.contents().as_ptr() as *mut u32).write(0);
            let p = intermediate.contents().as_ptr() as *mut u32;
            for i in 0..(HIDDEN as usize / 2) { p.add(i).write(0); }
        }
        let mut times = Vec::with_capacity(5);
        for _ in 0..5 {
            unsafe {
                (counter.contents().as_ptr() as *mut u32).write(0);
                let p = intermediate.contents().as_ptr() as *mut u32;
                for i in 0..(HIDDEN as usize / 2) { p.add(i).write(0); }
            }
            let cb = queue.commandBuffer().expect("cb");
            let enc = cb.computeCommandEncoder().expect("enc");
            enc.setComputePipelineState(&pipe_d);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&counter), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&intermediate), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 2);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
                MTLSize { width: 1024, height: 1, depth: 1 },
            );
            enc.endEncoding();
            let t0 = Instant::now();
            cb.commit();
            cb.waitUntilCompleted();
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("    D num_tgs={:3}: {:.2} ms (median)", num_tgs, times[2]);
    }
}
