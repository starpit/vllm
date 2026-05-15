// SPDX-License-Identifier: Apache-2.0
//! Affine int4 GEMM (qmm_t) cost sweep.
//!
//! Sweeps `MetalAffineQmmT` (Standard + SplitK variants) across the
//! prefill-shape grid that the production solver hits when M ≥
//! `get_qmv_batch_limit(K, N, arch)`. M values are picked to span
//! production bucket boundaries (8, 16, 32, 64, 128, 256, 512, 1024).
//!
//! Row format:
//!   `affine_qmm_t_<dtype>_gs<gs>,M,N,K,cost_us`
//!   `affine_qmm_t_splitk<k>_<dtype>_gs<gs>,M,N,K,cost_us`
//!
//! Naming convention matches `affine_qmv_*` rows so `MetalAffineQmmImpl::
//! cost_us` reconstructs the key by re-running `pick_qmm_t_kernel(M, N,
//! K, B=1, gs)` and formatting the same way.

use crate::util::{self, Buffer, Device};
use ferrite_metal_kernels::quantized::{
    pick_qmm_t_kernel, DequantDtype, MetalAffineQmmT, QmmTKernel, ScaleDtype,
};
use ferrite_metal_kernels::stream::MetalStream;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};

pub fn run(launch_overhead_us: f64) {
    eprintln!("Starting affine_qmm_t sweep...");

    // Same (N, K) set as qmv. M values span the matmul branch range.
    let shapes_nk: &[(u32, u32)] = &[
        // Llama-3.2-1B
        (2048, 2048),
        (512, 2048),
        (8192, 2048),
        (2048, 8192),
        // Llama-3.2-3B
        (3072, 3072),
        (1024, 3072),
        (8192, 3072),
        (3072, 8192),
        // Qwen2-7B / Qwen3-7B
        (3584, 3584),
        (512, 3584),
        (18944, 3584),
        (3584, 18944),
        // Qwen3-1.7B
        (256, 2048),
        (6144, 2048),
        (2048, 6144),
        // Qwen3-4B
        (2560, 2560),
        (320, 2560),
        (9728, 2560),
        (2560, 9728),
        // Mistral-7B
        (4096, 4096),
        (1024, 4096),
        (14336, 4096),
        (4096, 14336),
        // Gemma-2-2B
        (2304, 2304),
        (1024, 2304),
        (9216, 2304),
        (2304, 9216),
        // Gemma-3-1B
        (1152, 1152),
        (256, 1152),
        (6912, 1152),
        (1152, 6912),
        // Gemma-3-4B
        (10240, 2560),
        (2560, 10240),
    ];

    // Bucket M values: powers-of-two above the qmv vector_limit boundary
    // (18 on M3/M4 per `quantized.cpp:84`). Lower values exercise the
    // SplitK heuristic threshold (when n_tiles × m_tiles is small enough
    // that splitting K wins); higher values converge to Standard.
    let ms: &[u32] = &[8, 16, 32, 64, 128, 256, 512, 1024, 2048];

    let group_sizes: &[u32] = &[32, 64, 128];
    let dtypes: &[DequantDtype] = &[DequantDtype::Bf16, DequantDtype::F16];

    let device = util::device();
    let qmm = MetalAffineQmmT::new(device.clone()).expect("MetalAffineQmmT::new");

    for &dtype in dtypes {
        for &gs in group_sizes {
            let mut seen: std::collections::BTreeSet<(u32, u32, u32)> =
                std::collections::BTreeSet::new();
            for &(n, k) in shapes_nk {
                if !k.is_multiple_of(gs) {
                    continue;
                }
                for &m in ms {
                    if !seen.insert((m, n, k)) {
                        continue;
                    }
                    let cost_us =
                        bench_qmm_t(&qmm, device, dtype, m, n, k, gs, launch_overhead_us);
                    // NAX path is dormant for currently-modelled gens
                    // (`is_nax_capable(_) == false` per
                    // `project_metal_nax_layout_bug.md`); sweep the
                    // non-NAX picker.
                    let kernel = pick_qmm_t_kernel(m, n, k, 1, gs, /* is_nax = */ false);
                    let name = csv_kernel_name(kernel, dtype, gs);
                    println!("{name},{m},{n},{k},{cost_us:.2}");
                }
            }
        }
    }

    eprintln!("affine_qmm_t sweep complete");
}

/// Stable CSV kernel name for a picked qmm_t variant. Mirrors
/// `affine_qmv_sweep::csv_kernel_name` shape — splits Standard /
/// SplitK and folds `split_k` into the kernel column so each variant
/// gets its own cost row.
pub fn csv_kernel_name(kernel: QmmTKernel, dtype: DequantDtype, gs: u32) -> String {
    let dt = dtype.symbol_infix();
    match kernel {
        QmmTKernel::Standard => format!("affine_qmm_t_{dt}_gs{gs}"),
        QmmTKernel::SplitK { split_k, .. } => {
            format!("affine_qmm_t_splitk{split_k}_{dt}_gs{gs}")
        }
        // NAX variant is not exercised by this sweep — the picker is
        // invoked with `is_nax = false` above. Keep the arm exhaustive
        // so future picker variants don't silently fall through.
        QmmTKernel::Nax => format!("affine_qmm_t_nax_{dt}_gs{gs}"),
    }
}

fn bench_qmm_t(
    qmm: &MetalAffineQmmT,
    device: &Device,
    dtype: DequantDtype,
    m: u32,
    n: u32,
    k: u32,
    gs: u32,
    launch_overhead_us: f64,
) -> f64 {
    let packed_bytes = (n as usize) * (k as usize) / 2;
    let scales_elems = (n as usize) * (k as usize) / (gs as usize);
    let scales_bytes = scales_elems * 2;
    let biases_bytes = scales_bytes;
    let x_bytes = (m as usize) * (k as usize) * 2;
    // SplitK kernel writes `[split_k, M, N]` intermediate; allocate the
    // worst case (split_k=32 per quantized.cpp:1453) to be safe.
    let y_bytes = (m as usize) * (n as usize) * 2 * 32;

    let packed = util::create_buffer(packed_bytes);
    let scales = util::create_buffer(scales_bytes);
    let biases = util::create_buffer(biases_bytes);
    let x = util::create_buffer(x_bytes);
    let y = util::create_buffer(y_bytes);

    let mut stream = MetalStream::new(device);
    util::time_kernel(launch_overhead_us, 3, 20, || {
        let cb = stream.get_command_buffer().expect("cmd buf").clone();
        let enc = cb.computeCommandEncoder().expect("encoder");
        qmm.execute(
            &x as &Buffer,
            &packed as &Buffer,
            &scales as &Buffer,
            &biases as &Buffer,
            &y as &Buffer,
            m,
            n,
            k,
            1,
            gs,
            4,
            dtype,
            ScaleDtype::F16,
            &enc,
        )
        .expect("qmm_t dispatch");
        enc.endEncoding();
        stream.commit().expect("commit");
        stream.synchronize().expect("sync");
    })
}
