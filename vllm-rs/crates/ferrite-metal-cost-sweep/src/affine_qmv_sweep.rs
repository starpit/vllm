// SPDX-License-Identifier: Apache-2.0
//! Affine int4 GEMV (qmv) cost sweep.
//!
//! Sweeps the three `MetalAffineQmv` variants (`qmv_quad`, `qmv_fast`,
//! `qmv`) across decode-shaped `(M=1, N, K)` tuples covering every
//! Linear shape in the mlx-community 4bit Llama / Qwen / Gemma family.
//! Rows feed `MetalAffineQmmImpl::cost_us` so the solver picks per
//! actual measured cost rather than the analytical roofline.
//!
//! Row format:
//!   `affine_qmv_quad_<dtype>_gs<gs>_d<d>,1,N,K,cost_us`
//!   `affine_qmv_fast_<dtype>_gs<gs>,1,N,K,cost_us`
//!   `affine_qmv_<dtype>_gs<gs>,1,N,K,cost_us`
//!
//! `dtype` is the activation dtype (`bf16` or `f16`); the qmv kernel
//! templates already split symbols on `<T_act, T_scale>` and we sweep
//! the single scale_dtype currently in production (`F16` per
//! `INT4_PARITY_PROBES.md` §7). Group size + bits ride in the kernel
//! name so `MetalAffineQmmImpl::cost_us` can reconstruct the key by
//! re-running the dispatcher decision on `(N, K, bits)`.

use crate::util::{self, Buffer, Device};
use ferrite_metal_kernels::quantized::{
    valid_qmv_kernels, DequantDtype, MetalAffineQmv, QmvKernel, ScaleDtype,
};
use ferrite_metal_kernels::stream::MetalStream;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};

pub fn run(launch_overhead_us: f64) {
    eprintln!("Starting affine_qmv sweep...");

    // Every decode-shape `(N, K)` that appears in Llama-1B, Llama-3B,
    // Qwen2-7B, Qwen3-{1.7,4,8}B, Mistral-7B, Gemma2-{2,9}B,
    // Gemma3-{1,4}B per the model coverage matrix in
    // INT4_PARITY_PLAN.md§Coverage. M=1 (decode); B=1.
    let shapes: &[(u32, u32)] = &[
        // (N, K)
        // ── Llama-3.2-1B-Instruct-4bit (hidden=2048, intermediate=8192) ─
        (2048, 2048), // q_proj, o_proj
        (512, 2048),  // k_proj, v_proj
        (8192, 2048), // gate_proj, up_proj
        (2048, 8192), // down_proj
        // ── Llama-3.2-3B-Instruct-4bit (hidden=3072, intermediate=8192) ─
        (3072, 3072), // q_proj, o_proj
        (1024, 3072), // k_proj, v_proj
        (8192, 3072), // gate_proj, up_proj
        (3072, 8192), // down_proj
        // ── Qwen2-7B / Qwen3-7B (hidden=3584, intermediate=18944) ───────
        (3584, 3584),
        (512, 3584),
        (18944, 3584),
        (3584, 18944),
        // ── Qwen3-1.7B (hidden=2048, intermediate=6144) ────────────────
        (2048, 2048), // dup; HashMap-keyed in CSV so OK
        (256, 2048),  // 8 kv heads × 32 head_dim
        (6144, 2048),
        (2048, 6144),
        // ── Qwen3-4B (hidden=2560, intermediate=9728) ──────────────────
        (2560, 2560),
        (320, 2560), // 8 kv heads × 40? actually 2560/8=320 — sanity hits whatever the model uses
        (9728, 2560),
        (2560, 9728),
        // ── Mistral-7B (hidden=4096, intermediate=14336) ───────────────
        (4096, 4096),
        (1024, 4096),
        (14336, 4096),
        (4096, 14336),
        // ── Gemma-2-2B (hidden=2304, intermediate=9216) ────────────────
        (2304, 2304),
        (1024, 2304),
        (9216, 2304),
        (2304, 9216),
        // ── Gemma-3-1B (hidden=1152, intermediate=6912) ────────────────
        (1152, 1152),
        (256, 1152),
        (6912, 1152),
        (1152, 6912),
        // ── Gemma-3-4B (hidden=2560, intermediate=10240) ───────────────
        (10240, 2560),
        (2560, 10240),
        // ── head_dim=64, K=64 quad-kernel sample (Llama-3.2 attention)  ─
        // Not normally a Linear shape but useful for q-quad coverage of D=64.
        // (Skipped — q_proj is N=hidden, K=hidden; D=64 quad lives inside
        // attention reduce, not in a standalone Linear sweep.)
    ];

    // Production mlx-community 4bit family uses gs=64 uniformly per
    // INT4_PARITY_PROBES.md§5; sweep gs=32 + gs=128 too so the cost
    // table has coverage for future arrivals (e.g. some Qwen3 4bit
    // variants use gs=128 per the same probe).
    let group_sizes: &[u32] = &[32, 64, 128];
    let dtypes: &[DequantDtype] = &[DequantDtype::Bf16, DequantDtype::F16];

    let device = util::device();
    let qmv = MetalAffineQmv::new(device.clone()).expect("MetalAffineQmv::new");

    for &dtype in dtypes {
        for &gs in group_sizes {
            // Deduplicate (N, K) — multiple model families collide on
            // the same shape (e.g. Llama-1B q_proj + Qwen3-1.7B q_proj
            // both 2048×2048). Iterate the dedup'd set so the CSV
            // stays tidy.
            let mut seen: std::collections::BTreeSet<(u32, u32)> =
                std::collections::BTreeSet::new();
            for &(n, k) in shapes {
                if !seen.insert((n, k)) {
                    continue;
                }
                // K must divide group_size for the affine packing to be
                // well-defined; mlx-community 4bit checkpoints satisfy
                // this for their chosen gs (P0 verification), but a
                // sweep tuple may hit a (gs, K) combo that doesn't.
                if !k.is_multiple_of(gs) {
                    continue;
                }
                // Sweep every variant that's *valid* for this shape,
                // not just the heuristic's pick. The CSV is the
                // solver's source of truth — emitting only one row
                // per shape collapses to "the heuristic was always
                // right," which is a tautology, not measurement.
                for kernel in valid_qmv_kernels(n, k, 4) {
                    let cost_us = bench_qmv_variant(
                        &qmv,
                        device,
                        dtype,
                        n,
                        k,
                        gs,
                        kernel,
                        launch_overhead_us,
                    );
                    let name = csv_kernel_name(kernel, dtype, gs);
                    println!("{name},1,{n},{k},{cost_us:.2}");
                }
            }
        }
    }

    eprintln!("affine_qmv sweep complete");
}

/// Stable CSV kernel-name format for a picked qmv variant.
///
/// Mirrors `qmv_kernel_static_name` shape but drops `_s_f16_` (we sweep
/// the single scale dtype in production) and `_batch_0` (B=1 always
/// for the Linear path; gather variants land later under P13 with
/// `affine_gather_*` rows). `MetalAffineQmmImpl::cost_us` reconstructs
/// the same name by re-running `pick_qmv_kernel` on `(N, K, bits)`.
pub fn csv_kernel_name(kernel: QmvKernel, dtype: DequantDtype, gs: u32) -> String {
    let dt = dtype.symbol_infix();
    match kernel {
        QmvKernel::Quad { d } => format!("affine_qmv_quad_{dt}_gs{gs}_d{d}"),
        QmvKernel::Fast => format!("affine_qmv_fast_{dt}_gs{gs}"),
        QmvKernel::Generic => format!("affine_qmv_{dt}_gs{gs}"),
    }
}

fn bench_qmv_variant(
    qmv: &MetalAffineQmv,
    device: &Device,
    dtype: DequantDtype,
    n: u32,
    k: u32,
    gs: u32,
    kernel: QmvKernel,
    launch_overhead_us: f64,
) -> f64 {
    // Match the per-tensor byte sizes used in production:
    //   packed weight: u32 packed_factor=8 → bytes = N*K/2.
    //   scales / biases: (N*K/gs) elements × 2 bytes (F16).
    //   x activation: M*K × 2 bytes (act dtype).
    //   y output: M*N × 2 bytes.
    let packed_bytes = (n as usize) * (k as usize) / 2;
    let scales_elems = (n as usize) * (k as usize) / (gs as usize);
    let scales_bytes = scales_elems * 2;
    let biases_bytes = scales_bytes;
    let x_bytes = 1 * (k as usize) * 2;
    let y_bytes = 1 * (n as usize) * 2;

    let packed = util::create_buffer(packed_bytes);
    let scales = util::create_buffer(scales_bytes);
    let biases = util::create_buffer(biases_bytes);
    let x = util::create_buffer(x_bytes);
    let y = util::create_buffer(y_bytes);

    let mut stream = MetalStream::new(device);
    let per_cb_us = util::time_kernel(launch_overhead_us, 3, 20, || {
        let cb = stream.get_command_buffer().expect("cmd buf").clone();
        let enc = cb.computeCommandEncoder().expect("encoder");
        // BATCH back-to-back qmv dispatches on the SAME encoder —
        // mirrors how production stacks many qmv calls per CB. All
        // dispatches share I/O buffers; consecutive calls write to
        // the same `y` so they serialize on `y` (Serial encoder),
        // but timing isn't sensitive to that — the bench measures
        // the steady-state per-dispatch cost paying ONE per-CB
        // submit/wait fixed cost across BATCH dispatches, which is
        // exactly what production does.
        for _ in 0..BATCH {
            qmv.execute_with_kernel(
                kernel,
                &x as &Buffer,
                &packed as &Buffer,
                &scales as &Buffer,
                &biases as &Buffer,
                &y as &Buffer,
                1,
                n,
                k,
                1,
                gs,
                4,
                dtype,
                ScaleDtype::F16,
                &enc,
            )
            .expect("qmv dispatch");
        }
        enc.endEncoding();
        stream.commit().expect("commit");
        stream.synchronize().expect("sync");
    });
    per_cb_us / (BATCH as f64)
}

/// Number of qmv dispatches encoded into a single command buffer per
/// timing iteration. Production single-stream M=1 decode stacks
/// 100+ qmv dispatches onto one MTL4 encoder per forward — per-CB
/// submit/wait fixed cost is paid ONCE for the whole forward, not
/// per kernel.
///
/// The original sweep dispatched one qmv per CB, so its measured
/// cost was dominated by ~50-200 µs of per-CB overhead, swamping
/// the actual ~1-2 µs of kernel time. That made cross-variant
/// comparisons unstable: the picker was reading noise. This
/// 64-batched version pays the overhead ONCE and divides — relative
/// per-call timings now reflect what production actually pays.
const BATCH: u32 = 64;
