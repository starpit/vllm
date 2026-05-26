// SPDX-License-Identifier: Apache-2.0
//! Minimal probe for the PD-wavefront cross-TG **data-before-flag**
//! ordering problem, isolated from the qmv stack.
//!
//! `flag_sync_sweep` proved the flag handoff with a TOKEN (the data IS the
//! atomic value, so coherent by construction). This probes the next
//! requirement: a producer TG writes a SEPARATE buffer, then publishes a
//! flag; a consumer TG spin-waits, then reads the buffer — and must see
//! every producer's write. Each producer `p` writes `data[p] = p+1000`
//! (data is zero-initialized); a consumer sums all slots. Correct ⇒ sum is
//! `Σ(i+1000)`; a race leaves some slot 0 ⇒ a short sum. The race is
//! timing-dependent, so each case is retried.
//!
//! The MSL is compiled at runtime (`newLibraryWithSource`) so fence
//! patterns iterate without rebuilding the metallib. `PAT` selects:
//!   0 — non-atomic data, single TG0 consumer, producer `threadgroup_barrier`
//!   1 — non-atomic data, every-worker consumer (+ barrier2)   [A2-like]
//!   2 — non-atomic data, every-worker, 64-lane read           [A2-like]
//!   3 — ATOMIC data write/read (relaxed) + flag + barriers
//!   4 — data-IS-the-flag: spin until each slot non-zero (the ring pattern)
//!
//! Run: `cargo test -p ferrite-forward -F metal --test wavefront_sync_probe -- --nocapture`
#![cfg(feature = "metal")]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_metal_kernels::device::detect_device;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;
constant constexpr uint CAP = 100000000u;
kernel void data_handoff(
    device uint* data        [[buffer(0)]],
    device atomic_uint* flag [[buffer(1)]],
    device uint* out         [[buffer(2)]],
    constant uint& P         [[buffer(3)]],
    uint tg  [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    // ── Producer: worker `tg` writes its data slot. ──
#if PAT >= 3
    if (tid == 0u) atomic_store_explicit((device atomic_uint*)&data[tg], tg + 1000u, memory_order_relaxed);
#else
    if (tid == 0u) data[tg] = tg + 1000u;
#endif
#if PAT != 4
    threadgroup_barrier(mem_flags::mem_device);                          // fence the data write
    if (tid == 0u) atomic_store_explicit(&flag[tg], 1u, memory_order_relaxed); // publish
#endif

    // ── Consumer ──
#if PAT == 0
    if (tg == 0u && tid == 0u) {
        for (uint p = 0u; p < P; p++) { uint s = 0u; while (atomic_load_explicit(&flag[p], memory_order_relaxed) == 0u) { if (++s > CAP) break; } }
        uint sum = 0u; for (uint p = 0u; p < P; p++) sum += data[p]; out[0] = sum;
    }
#elif PAT == 1
    if (tid == 0u) { for (uint p = 0u; p < P; p++) { uint s = 0u; while (atomic_load_explicit(&flag[p], memory_order_relaxed) == 0u) { if (++s > CAP) break; } } }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) { uint sum = 0u; for (uint p = 0u; p < P; p++) sum += data[p]; out[tg] = sum; }
#elif PAT == 2
    if (tid == 0u) { for (uint p = 0u; p < P; p++) { uint s = 0u; while (atomic_load_explicit(&flag[p], memory_order_relaxed) == 0u) { if (++s > CAP) break; } } }
    threadgroup_barrier(mem_flags::mem_device);
    { uint sum = 0u; for (uint p = 0u; p < P; p++) sum += data[p]; if (tid == 0u) out[tg] = sum; } // 64-lane read
#elif PAT == 3
    if (tid == 0u) { for (uint p = 0u; p < P; p++) { uint s = 0u; while (atomic_load_explicit(&flag[p], memory_order_relaxed) == 0u) { if (++s > CAP) break; } } }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) { uint sum = 0u; for (uint p = 0u; p < P; p++) sum += atomic_load_explicit((device atomic_uint*)&data[p], memory_order_relaxed); out[tg] = sum; }
#elif PAT == 4
    if (tid == 0u) {
        uint sum = 0u;
        for (uint p = 0u; p < P; p++) { uint v = 0u, s = 0u; do { v = atomic_load_explicit((device atomic_uint*)&data[p], memory_order_relaxed); if (++s > CAP) break; } while (v == 0u); sum += v; }
        out[tg] = sum;
    }
#endif
}
"#;

type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Device = Retained<ProtocolObject<dyn MTLDevice>>;

fn zeroed(device: &Device, bytes: usize) -> Buffer {
    let b = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("buffer");
    unsafe { std::ptr::write_bytes(b.contents().as_ptr() as *mut u8, 0, bytes) };
    b
}

fn build_pipeline(device: &Device, pat: u32) -> Pipeline {
    let opts = objc2_metal::MTLCompileOptions::new();
    let src = format!("#define PAT {pat}\n{SRC}");
    let lib: Retained<ProtocolObject<dyn MTLLibrary>> = device
        .newLibraryWithSource_options_error(&NSString::from_str(&src), Some(&opts))
        .expect("compile probe MSL");
    let f = lib
        .newFunctionWithName(&NSString::from_str("data_handoff"))
        .expect("function");
    device
        .newComputePipelineStateWithFunction_error(&f)
        .expect("pipeline")
}

/// Run the probe for `p` workers; return out[0..p].
fn run_probe(device: &Device, pipe: &Pipeline, p: u32) -> Vec<u32> {
    let data = zeroed(device, (p as usize) * 4);
    let flag = zeroed(device, (p as usize) * 4);
    let out = zeroed(device, (p as usize) * 4);
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(pipe);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&data), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&flag), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&out), 0, 2);
        let pv = NonNull::new(&p as *const u32 as *mut c_void).unwrap();
        enc.setBytes_length_atIndex(pv, 4, 3);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: p as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const u32, p as usize) }
        .to_vec()
}

#[test]
fn cross_tg_data_behind_flag() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let mut any_ok = false;
    for pat in [0u32, 1, 2, 3, 4] {
        let pipe = build_pipeline(&device, pat);
        let mut ok = true;
        for p in [1u32, 2, 4, 8, 10] {
            let want: u32 = (0..p).map(|i| i + 1000).sum();
            let consumers = if pat == 0 { 1 } else { p as usize };
            let mut bad = None;
            'retry: for _ in 0..200 {
                let out = run_probe(&device, &pipe, p);
                for (c, &got) in out.iter().take(consumers).enumerate() {
                    if got != want {
                        bad = Some((c, got));
                        break 'retry;
                    }
                }
            }
            match bad {
                None => eprintln!("[probe] PAT={pat} P={p:2}: OK (sum={want})"),
                Some((c, got)) => {
                    ok = false;
                    eprintln!(
                        "[probe] PAT={pat} P={p:2}: RACE at consumer {c} — got {got}, want {want}"
                    );
                }
            }
        }
        eprintln!(
            "[probe] PAT={pat}: {}",
            if ok { "CORRECT for all P" } else { "RACES" }
        );
        if ok {
            any_ok = true;
        }
    }
    assert!(
        any_ok,
        "no fence pattern handed data across TGs — see [probe] lines"
    );
}
