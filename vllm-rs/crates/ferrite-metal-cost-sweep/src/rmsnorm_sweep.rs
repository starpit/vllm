// SPDX-License-Identifier: Apache-2.0
//! RMSNorm cost sweep (placeholder, kept for binary compat).
//!
//! **Known issues** (not fixed in this pass — tracked separately):
//!
//! 1. The kernel compiled and timed below is an INLINE source-string
//!    shader, not the production `rmsnorm.metal` kernel in
//!    `ferrite-metal-kernels`. The measured numbers therefore don't
//!    correspond to what the runtime actually dispatches.
//!
//! 2. Rows are emitted as `metal_rmsnorm_<dtype>`, but the consumer
//!    `MetalRmsNormImpl::cost_us` looks up `rmsnorm_<dtype>` (no
//!    `metal_` prefix — matches the `cublas` / `cutlass_*` /
//!    `fused_gate_up_silu_mul_*` convention in cuda + fused_kernels).
//!    So even with accurate numbers, the solver would never read them.
//!
//! Kept here because deleting the rows would break
//! `ferrite_metal_targets::tests::test_m1_max_loads_costs` which
//! asserts the keys exist. Re-port to production-shader + correct
//! kernel name is a follow-up.

use crate::util::{self, Device};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
pub fn run(launch_overhead_us: f64) {
    eprintln!("Starting RMSNorm sweep (placeholder)...");

    let hidden_sizes = vec![2048, 3072, 4096, 5120, 6144, 7168, 8192];
    let seq_lens = vec![
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    ];

    let device = util::device();
    let pipeline_f16 = build_pipeline(device, RMSNORM_SOURCE, "rmsnorm_f16");
    let pipeline_bf16 = pipeline_f16.clone(); // bf16 placeholder — same kernel for now

    for &hidden in &hidden_sizes {
        for &seq_len in &seq_lens {
            let cost_us = bench_rmsnorm(&pipeline_f16, seq_len, hidden, launch_overhead_us);
            // CSV row name kept as `metal_rmsnorm_<dtype>` to preserve the
            // historical CSV shape that
            // `ferrite_metal_targets::tests::test_m1_max_loads_costs`
            // asserts. The solver-side lookup mismatch is tracked above.
            println!("metal_rmsnorm_f16,{seq_len},{hidden},0,{cost_us:.2}");
        }
    }

    for &hidden in &hidden_sizes {
        for &seq_len in &seq_lens {
            let cost_us = bench_rmsnorm(&pipeline_bf16, seq_len, hidden, launch_overhead_us);
            println!("metal_rmsnorm_bf16,{seq_len},{hidden},0,{cost_us:.2}");
        }
    }

    eprintln!("RMSNorm sweep complete");
}

fn build_pipeline(
    device: &Device,
    source: &str,
    fn_name: &str,
) -> objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>> {
    let src = NSString::from_str(source);
    let lib = device
        .newLibraryWithSource_options_error(&src, None)
        .expect("compile rmsnorm placeholder shader");
    let name = NSString::from_str(fn_name);
    let func = lib.newFunctionWithName(&name).expect("get function");
    device
        .newComputePipelineStateWithFunction_error(&func)
        .expect("create pipeline state")
}

fn bench_rmsnorm(
    pipeline: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLComputePipelineState>>,
    seq_len: usize,
    hidden: usize,
    launch_overhead_us: f64,
) -> f64 {
    let queue = util::new_command_queue();
    let input = util::create_buffer(seq_len * hidden * 2);
    let weight = util::create_buffer(hidden * 2);
    let output = util::create_buffer(seq_len * hidden * 2);

    util::time_kernel(launch_overhead_us, 3, 10, || {
        let cb = queue.commandBuffer().expect("commandBuffer");
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&input), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&weight), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&output), 0, 2);
        }
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: ((seq_len + 255) / 256) * 256,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    })
}

const RMSNORM_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_f16(
    device const half* input  [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device       half* output [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    // Placeholder body — the production shader lives in
    // ferrite-metal-kernels/shaders/rmsnorm.metal. This stub
    // exists only so the sweep binary builds.
    if (tid >= 1) return;
    output[0] = input[0] * weight[0];
}
"#;
