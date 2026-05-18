// SPDX-License-Identifier: Apache-2.0
//! Isolated test of the suspected-buggy pattern in
//! `bn8_attention_oproj_fused_body`: a TG iterates N heads sequentially,
//! and for each head, runs a per-output loop with simdgroup-cooperative
//! simd_sum + lane-0 CAS-add of a bf16 pair into a packed-uint32
//! residual buffer.
//!
//! Goal: reproduce / rule out kernel hangs (watchdog trips, infinite
//! CAS retries, simd_sum divergence) WITHOUT loading a 1B-param model
//! or risking a system beach-ball. Run via
//! `FERRITE_SWEEP=fused_cas_probe ./metal_cost_sweep`.
//!
//! What the kernel does per output position `i`:
//!   - Each lane has a "contribution" value = (head + 1) (constant across lanes)
//!   - simd_sum across 32 lanes → 32 * (head + 1)
//!   - lane 0 CAS-adds bf16(32*(head+1)) into residual[i]
//! Summed across N heads:
//!   expected residual[i] = sum_{h=0..N_HEADS} (32 * (h+1))
//!                        = 32 * N_HEADS * (N_HEADS + 1) / 2
//!
//! For N_HEADS=32: residual[i] = 32 * 528 = 16896 (every position).
//!
//! The kernel uses HIDDEN=2048 output positions; 8 simdgroups split the
//! 2048 outputs (256 per simdgroup), matching the production layout.

use crate::util;
use objc2::AnyThread;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use std::time::Instant;

const HIDDEN: u32 = 2048;
const N_HEADS_PER_TG: u32 = 32;
const THREADS_PER_TG: u32 = 256;
const NUM_SIMDGROUPS: u32 = 8;
const OUTPUTS_PER_SIMDGROUP: u32 = HIDDEN / NUM_SIMDGROUPS; // 256

const KERNEL_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant constexpr uint HIDDEN = 2048u;
constant constexpr uint HEAD_DIM = 64u;
constant constexpr uint N_HEADS_PER_TG = 32u;
constant constexpr uint NUM_SIMDGROUPS = 8u;
constant constexpr uint OUTPUTS_PER_SIMDGROUP = HIDDEN / NUM_SIMDGROUPS;
constant constexpr uint W_ROW_BYTES = HIDDEN / 2u;       // 1024
constant constexpr uint SB_PER_ROW  = HIDDEN / 64u;      // 32 groups per row

[[kernel, max_total_threads_per_threadgroup(256u)]]
void fused_cas_probe(
    device atomic_uint* residual_atomic [[buffer(0)]],
    device const uint32_t* o_w          [[buffer(1)]],  // [HIDDEN, HIDDEN] uint4 packed
    device const half*     o_s          [[buffer(2)]],  // [HIDDEN, HIDDEN/64]
    device const half*     o_b          [[buffer(3)]],  // [HIDDEN, HIDDEN/64]
    uint  tg_id     [[threadgroup_position_in_grid]],
    uint  tid       [[thread_position_in_threadgroup]],
    uint  simd_gid  [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    // Fake "attn_final_o" — per-lane registers, like after attention.
    thread float attn_final_o[2];
    attn_final_o[0] = 0.001f;  // small constant, mimics attention output
    attn_final_o[1] = 0.001f;

    // Iterate heads sequentially in this TG, mirroring fused attn+o_proj.
    for (uint head = 0u; head < N_HEADS_PER_TG; ++head) {
        const uint out_base = simd_gid * OUTPUTS_PER_SIMDGROUP;
        for (uint s = 0u; s < OUTPUTS_PER_SIMDGROUP; ++s) {
            const uint out = out_base + s;

            // Read W_o byte at [out, head*HEAD_DIM + simd_lid*2] (1 byte = 2 nibbles).
            const size_t w_byte_idx =
                  (size_t)out * (size_t)W_ROW_BYTES
                + (size_t)(head * (HEAD_DIM / 2u))
                + (size_t)simd_lid;
            const device uint8_t* w_bytes = (const device uint8_t*)o_w;
            const uchar byte = w_bytes[w_byte_idx];
            const float n0 = float(byte & 0x0Fu);
            const float n1 = float((byte >> 4) & 0x0Fu);

            // Read scale + bias.
            const size_t sb_idx = (size_t)out * (size_t)SB_PER_ROW + (size_t)head;
            const float s_val = float(o_s[sb_idx]);
            const float b_val = float(o_b[sb_idx]);

            // Per-lane partial.
            const float local_dot   = n0 * attn_final_o[0] + n1 * attn_final_o[1];
            const float local_sum_x = attn_final_o[0] + attn_final_o[1];
            const float dot_total   = simd_sum(local_dot);
            const float sumx_total  = simd_sum(local_sum_x);
            const float contrib     = s_val * dot_total + b_val * sumx_total;

            if (simd_lid == 0u) {
                const uint pair_idx = out / 2u;
                const uint lane     = out & 1u;
                uint old = atomic_load_explicit(
                    &residual_atomic[pair_idx], memory_order_relaxed);
                uint new_packed;
                uint retries = 0u;
                do {
                    bfloat old_lo = as_type<bfloat>(ushort(old & 0xFFFFu));
                    bfloat old_hi = as_type<bfloat>(ushort((old >> 16) & 0xFFFFu));
                    bfloat new_lo = old_lo;
                    bfloat new_hi = old_hi;
                    if (lane == 0u) {
                        new_lo = bfloat(float(old_lo) + contrib);
                    } else {
                        new_hi = bfloat(float(old_hi) + contrib);
                    }
                    new_packed = uint(as_type<ushort>(new_lo))
                               | (uint(as_type<ushort>(new_hi)) << 16);
                    retries++;
                    if (retries > 1000000u) break;
                } while (!atomic_compare_exchange_weak_explicit(
                    &residual_atomic[pair_idx], &old, new_packed,
                    memory_order_relaxed, memory_order_relaxed));
            }
        }
    }
}
"#;

pub fn run(_launch_overhead_us: f64) {
    eprintln!("\n=== fused_cas_probe ===");
    eprintln!("  Mirrors the o_proj fused CAS-add inner loop in isolation.");

    let device = util::device();
    let queue = util::new_command_queue();

    // Compile
    let src = NSString::from_str(KERNEL_SOURCE);
    let opts = objc2_metal::MTLCompileOptions::new();
    let lib = match device.newLibraryWithSource_options_error(&src, Some(&opts)) {
        Ok(l) => l,
        Err(e) => {
            panic!("compile failed: {:?}", e);
        }
    };
    let func = lib
        .newFunctionWithName(&NSString::from_str("fused_cas_probe"))
        .expect("symbol");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("pipeline");
    eprintln!(
        "  ✓ compiled (maxThreads/TG = {})",
        pipeline.maxTotalThreadsPerThreadgroup()
    );

    // Allocate dummy o_proj weights/scales/biases.
    // Shape: o_w = [HIDDEN, HIDDEN] uint4 packed → HIDDEN*HIDDEN/2 bytes.
    //        o_s = [HIDDEN, HIDDEN/64] halfs.
    //        o_b = [HIDDEN, HIDDEN/64] halfs.
    let o_w_bytes = (HIDDEN as usize) * (HIDDEN as usize) / 2;
    let o_sb_halfs = (HIDDEN as usize) * (HIDDEN as usize / 64);
    let o_w_buf = util::create_buffer(o_w_bytes);
    let o_s_buf = util::create_buffer(o_sb_halfs * 2);
    let o_b_buf = util::create_buffer(o_sb_halfs * 2);

    // Init dummy values: all quant nibbles = 1, all scales = 0.01 (half),
    // all biases = 0. So each W_o[i, head*HEAD_DIM + d] dequants to 0.01.
    // Per-output contribution: scale * sum_d(1 * attn[d]) + bias * sum_d(attn[d])
    //   = 0.01 * 64 * 0.001 + 0 = 0.00064 per head per output.
    // 32 heads → per output 0.0205 per TG. Total = 0.0205 × num_tgs (rounded by bf16).
    unsafe {
        let p = o_w_buf.contents().as_ptr() as *mut u8;
        for i in 0..o_w_bytes {
            p.add(i).write(0x11); // both nibbles = 1
        }
        let ps = o_s_buf.contents().as_ptr() as *mut u16;
        let half_one_hundredth: u16 = 0x211f; // half(0.01) ≈ 0x211F
        for i in 0..o_sb_halfs {
            ps.add(i).write(half_one_hundredth);
        }
        let pb = o_b_buf.contents().as_ptr() as *mut u16;
        for i in 0..o_sb_halfs {
            pb.add(i).write(0);
        }
    }

    // Sweep TG counts to find where it stalls / wedges
    for &num_tgs in &[1u32, 4, 16, 32] {
        eprintln!("\n  --- num_tgs={} ---", num_tgs);

        // residual: HIDDEN bfloats = HIDDEN/2 uint32 words. Zero-init.
        let residual = util::create_buffer((HIDDEN as usize / 2) * 4);
        unsafe {
            let p = residual.contents().as_ptr() as *mut u32;
            for i in 0..(HIDDEN as usize / 2) {
                p.add(i).write(0);
            }
        }

        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&residual), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&o_w_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&o_s_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&o_b_buf), 0, 3);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: num_tgs as usize, height: 1, depth: 1 },
            MTLSize { width: THREADS_PER_TG as usize, height: 1, depth: 1 },
        );
        enc.endEncoding();

        let t0 = Instant::now();
        cb.commit();
        cb.waitUntilCompleted();
        let elapsed = t0.elapsed();
        eprintln!("  ✓ dispatched + completed in {:.2} ms", elapsed.as_secs_f64() * 1000.0);

        // Verify residual values
        let p = residual.contents().as_ptr() as *const u16;
        let read = |i: u32| -> f32 {
            let bits = unsafe { *p.add(i as usize) };
            f32::from_bits((bits as u32) << 16)
        };

        // Per-output per-head contribution = scale * sum(n0*x0 + n1*x1) + bias * sum(x0 + x1)
        //   = 0.01 * (32 lanes * (1*0.001 + 1*0.001)) + 0
        //   = 0.01 * 32 * 0.002
        //   = 0.00064
        // Per TG (32 heads × per-head) = 32 * 0.00064 = 0.02048
        // Per output across all TGs = 0.02048 × num_tgs
        let per_head = 0.01 * 32.0 * 0.002;
        let per_tg = (N_HEADS_PER_TG as f32) * per_head;
        let expected = per_tg * num_tgs as f32;

        let actual = [read(0), read(1), read(HIDDEN / 2), read(HIDDEN - 1)];
        eprintln!(
            "    expected per-position ≈ {:.6} (per_tg={:.6} × num_tgs={})",
            expected, per_tg, num_tgs
        );
        eprintln!(
            "    actual:  res[0]={:.6} res[1]={:.6} res[HIDDEN/2]={:.6} res[HIDDEN-1]={:.6}",
            actual[0], actual[1], actual[2], actual[3]
        );
    }
}
