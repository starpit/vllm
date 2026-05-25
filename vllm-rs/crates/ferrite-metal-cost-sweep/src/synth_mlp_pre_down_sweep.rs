// SPDX-License-Identifier: Apache-2.0
//! Synthesized MLP-pre-down megakernel cost sweep.
//!
//! For each production model shape × M bucket, synthesizes the kernel
//! via `ferrite-fusion-synth`, AOT-compiles to metallib, loads into
//! the Metal device, dispatches with all 10 buffer bindings filled
//! with dummy data, and times.
//!
//! Emitted CSV rows match the lookup key in
//! `MetalSynthMlpPreDownImpl::cost_us`:
//!
//!   synth_mlp_pre_down_<act>_<scale>_gs<gs>,M,hidden,0,cost_us
//!
//! Without these rows the Impl rides its component-sum analytical
//! fallback (norm BW + 2×AffineQmm compute roofline + SiluMul BW,
//! ×0.95). With the rows merged into `cost_<chip>.csv` the solver
//! gets per-bucket calibrated cost so it picks fused vs unfused on a
//! real cost comparison (mirrors what synth_pre_attn does today).

use crate::util::{self, Buffer};
use dispatch2::DispatchData;
use ferrite_fusion_synth::{
    aot::aot_compile_metallib,
    fuse_pass::{ChunkConstants, SynthesisBackend, synthesize_mlp_pre_down_chunk},
};
use ferrite_metal_kernels::stream::MetalStream;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

/// Production model shape tuple. The MLP synth kernel only sees
/// `(hidden, intermediate, head_dim)` (head_dim is reused as TILE_N
/// per the lowering); num_q/num_kv/rot_dim are unused — the chain
/// stops at SiluMul.
struct ModelShape {
    name: &'static str,
    hidden: u32,
    intermediate: u32,
    head_dim: u32,
    rms_eps: f32,
}

const GROUP_SIZE: u32 = 64;

const SHAPES: &[ModelShape] = &[
    ModelShape {
        name: "llama_3_2_1b",
        hidden: 2048,
        intermediate: 8192,
        head_dim: 64,
        rms_eps: 1.0e-5,
    },
    ModelShape {
        name: "llama_3_2_3b",
        hidden: 3072,
        intermediate: 8192,
        head_dim: 128,
        rms_eps: 1.0e-5,
    },
    ModelShape {
        name: "llama_3_1_8b",
        hidden: 4096,
        intermediate: 14336,
        head_dim: 128,
        rms_eps: 1.0e-5,
    },
];

// Original decode-range sweep was 1..16. Adding 32 and 64 fills the
// gap between the existing data and the M>64 cutoff in
// `MetalSynthMlpPreDownImpl::cost_us` (commit a01010e05). Inside that
// range the solver currently rides the analytical fallback (norm BW +
// 2×AffineQmm compute roofline + SiluMul BW) which the same commit's
// analysis flagged as under-estimating actual cost — measured rows
// let the solver compare against unfused on a like-for-like basis at
// the 64-token bucket boundary the macro emits.
const M_BUCKETS: &[u32] = &[1, 2, 4, 8, 16, 32, 64];

pub fn run(launch_overhead_us: f64) {
    eprintln!("Starting synth_mlp_pre_down sweep...");
    for shape in SHAPES {
        for &m in M_BUCKETS {
            let cost_us = bench_one(shape, m, launch_overhead_us);
            let kind = "synth_mlp_pre_down_bfloat_half_gs64";
            println!("{kind},{m},{hidden},0,{cost_us:.2}", hidden = shape.hidden);
            eprintln!(
                "  {kind} {} M={m} hidden={} I={}: {cost_us:.2} us",
                shape.name, shape.hidden, shape.intermediate,
            );
        }
    }
    eprintln!("synth_mlp_pre_down sweep complete");
}

fn bench_one(s: &ModelShape, m: u32, launch_overhead_us: f64) -> f64 {
    let consts = ChunkConstants {
        hidden: s.hidden,
        // num_q_heads is overloaded as INTERMEDIATE inside the MLP
        // synth source (see `synthesize_mlp_pre_down_chunk` —
        // "INTERMEDIATE: AtomConstantValue::Uint(consts.num_q_heads)").
        num_q_heads: s.intermediate,
        num_kv_heads: 0,
        head_dim: s.head_dim,
        rot_dim: 0,
        block_size: 0,
        intermediate: s.intermediate,
        m,
        group_size: GROUP_SIZE,
        rms_norm_eps: s.rms_eps,
        has_linear_bias: false,
    };
    let synth = synthesize_mlp_pre_down_chunk(SynthesisBackend::Metal, "bfloat", "half", &consts);
    let bytes = aot_compile_metallib(&synth.symbol, &synth.source);
    assert!(
        !bytes.is_empty(),
        "AOT compile produced empty metallib for {}",
        synth.symbol
    );

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
    // `interpreter/metal/lowering.rs::SynthMlpPreDown` (10 bindings).
    let act = 2usize;
    let scale = 2usize;
    let hid = s.hidden as usize;
    let im = s.intermediate as usize;
    let mu = m as usize;
    let gs = GROUP_SIZE as usize;

    let silu_mul_out = util::create_buffer(mu * im * act);
    let residual = util::create_buffer(mu * hid * act);
    let delta = util::create_buffer(mu * hid * act);
    let rms_w = util::create_buffer(hid * scale);
    let gate_packed = util::create_buffer(im * hid / 2);
    let gate_scales = util::create_buffer(im * hid / gs * scale);
    let gate_biases = util::create_buffer(im * hid / gs * scale);
    let up_packed = util::create_buffer(im * hid / 2);
    let up_scales = util::create_buffer(im * hid / gs * scale);
    let up_biases = util::create_buffer(im * hid / gs * scale);

    let bufs: [&Buffer; 10] = [
        &silu_mul_out,
        &residual,
        &delta,
        &rms_w,
        &gate_packed,
        &gate_scales,
        &gate_biases,
        &up_packed,
        &up_scales,
        &up_biases,
    ];

    let tile_n = s.head_dim;
    let num_tiles = s.intermediate / tile_n;
    let threads_per_tg = 32 * tile_n / 4;

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
            width: m as usize,
            height: num_tiles as usize,
            depth: 1,
        };
        let tpt = MTLSize {
            width: threads_per_tg as usize,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(tg, tpt);
        enc.endEncoding();
        stream.commit().expect("commit");
        stream.synchronize().expect("sync");
    })
}
