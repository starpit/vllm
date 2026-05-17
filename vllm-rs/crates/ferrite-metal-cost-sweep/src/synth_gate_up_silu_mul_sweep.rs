// SPDX-License-Identifier: Apache-2.0
//! Synthesized fused gate+up GEMM + SiluMul (large-M) cost sweep.
//!
//! For each production model shape × M bucket, synthesizes the kernel
//! via `ferrite-fusion-synth`, AOT-compiles to metallib, loads into
//! the Metal device, dispatches with all 8 buffer bindings filled
//! with dummy data, and times.
//!
//! Emitted CSV rows match the lookup key in
//! `MetalSynthGateUpSiluMulImpl::cost_us`:
//!
//!   synth_gate_up_silu_mul_large_<act>_<scale>_gs<gs>,M,hidden,0,cost_us
//!
//! Without these rows the Impl returns a 1.0e15 sentinel from
//! `cost_us` and never wins. With the rows merged into
//! `cost_<chip>.csv` the solver gets per-bucket calibrated cost so it
//! picks fused vs unfused on a real cost comparison.

use crate::util::{self, Buffer};
use dispatch2::DispatchData;
use ferrite_fusion_synth::{
    aot::aot_compile_metallib,
    fuse_pass::{synthesize_gate_up_silu_mul_large_chunk, ChunkConstants, SynthesisBackend},
};
use ferrite_metal_kernels::stream::MetalStream;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

/// Production model shape tuple. Only `(hidden, intermediate)` discriminate;
/// the kernel doesn't read num_heads / head_dim / rot_dim.
struct ModelShape {
    name:         &'static str,
    hidden:       u32,
    intermediate: u32,
}

const GROUP_SIZE: u32 = 64;

const SHAPES: &[ModelShape] = &[
    ModelShape { name: "llama_3_2_1b", hidden: 2048, intermediate: 8192 },
    ModelShape { name: "llama_3_2_3b", hidden: 3072, intermediate: 8192 },
    ModelShape { name: "llama_3_1_8b", hidden: 4096, intermediate: 14336 },
];

// `MetalSynthGateUpSiluMulImpl::cost_us` returns a 1.0e15 sentinel for
// `num_tokens > 64`, so the kernel only competes in the decode range.
// Sweep that range plus the 64-token boundary the macro emits.
const M_BUCKETS: &[u32] = &[1, 2, 4, 8, 16, 32, 64];

pub fn run(launch_overhead_us: f64) {
    eprintln!("Starting synth_gate_up_silu_mul sweep...");
    for shape in SHAPES {
        for &m in M_BUCKETS {
            let cost_us = bench_one(shape, m, launch_overhead_us);
            let kind = "synth_gate_up_silu_mul_large_bfloat_half_gs64";
            println!("{kind},{m},{hidden},0,{cost_us:.2}", hidden = shape.hidden);
            eprintln!(
                "  {kind} {} M={m} hidden={} I={}: {cost_us:.2} us",
                shape.name, shape.hidden, shape.intermediate,
            );
        }
    }
    eprintln!("synth_gate_up_silu_mul sweep complete");
}

fn bench_one(s: &ModelShape, m: u32, launch_overhead_us: f64) -> f64 {
    let consts = ChunkConstants {
        hidden:       s.hidden,
        num_q_heads:  0,
        num_kv_heads: 0,
        head_dim:     0,
        rot_dim:      0,
        block_size:   0,
        intermediate: s.intermediate,
        m,
        group_size:   GROUP_SIZE,
        rms_norm_eps: 0.0,
        has_linear_bias: false,
    };
    let synth = synthesize_gate_up_silu_mul_large_chunk(
        SynthesisBackend::Metal, "bfloat", "half", &consts,
    );
    let bytes = aot_compile_metallib(&synth.symbol, &synth.source);
    assert!(!bytes.is_empty(), "AOT compile produced empty metallib for {}", synth.symbol);

    let device = util::device();
    let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let data = DispatchData::from_static_bytes(leaked);
    let library = device
        .newLibraryWithData_error(&data)
        .expect("newLibraryWithData");

    let ns_name = NSString::from_str(&synth.symbol);
    let constants = MTLFunctionConstantValues::new();
    unsafe {
        constants.setConstantValue_type_atIndex(
            NonNull::new(&m as *const u32 as *mut c_void).unwrap(),
            MTLDataType::UInt,
            0,
        );
    }
    let function = library
        .newFunctionWithName_constantValues_error(&ns_name, &constants)
        .expect("newFunctionWithName");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("newComputePipelineState");

    // Per-buffer sizing — mirrors
    // `interpreter/metal/lowering.rs::SynthGateUpSiluMul` (8 bindings).
    let act = 2usize;
    let scale = 2usize;
    let hid = s.hidden as usize;
    let im  = s.intermediate as usize;
    let mu  = m as usize;
    let gs  = GROUP_SIZE as usize;

    let silu_mul_out = util::create_buffer(mu * im * act);
    let x_norm       = util::create_buffer(mu * hid * act);
    // Packed int4: 8 weights per uint32 → im*hid/8 uint32 = im*hid/2 bytes.
    let gate_packed = util::create_buffer(im * hid / 2);
    let gate_scales = util::create_buffer(im * hid / gs * scale);
    let gate_biases = util::create_buffer(im * hid / gs * scale);
    let up_packed   = util::create_buffer(im * hid / 2);
    let up_scales   = util::create_buffer(im * hid / gs * scale);
    let up_biases   = util::create_buffer(im * hid / gs * scale);

    let bufs: [&Buffer; 8] = [
        &silu_mul_out, &x_norm,
        &gate_packed, &gate_scales, &gate_biases,
        &up_packed,   &up_scales,   &up_biases,
    ];

    // Dispatch shape mirrors the lowering: BN=BM=32, TGP=128.
    let tg_n = 32u32;
    let tg_m = 32u32;
    let tg_x = (s.intermediate).div_ceil(tg_n);
    let tg_y = m.div_ceil(tg_m);

    let mut stream = MetalStream::new(device);
    util::time_kernel(launch_overhead_us, 3, 20, || {
        let cb = stream.get_command_buffer().expect("cb").clone();
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pipeline);
        for (i, b) in bufs.iter().enumerate() {
            unsafe {
                enc.setBuffer_offset_atIndex(Some(b), 0, i);
            }
        }
        let tg = MTLSize {
            width:  tg_x as usize,
            height: tg_y as usize,
            depth:  1,
        };
        let tpt = MTLSize {
            width:  128,
            height: 1,
            depth:  1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(tg, tpt);
        enc.endEncoding();
        stream.commit().expect("commit");
        stream.synchronize().expect("sync");
    })
}
