// SPDX-License-Identifier: Apache-2.0
//! Synthesized pre-attention megakernel cost sweep.
//!
//! For each production model shape × M bucket × {init, non-init},
//! synthesizes the kernel via `ferrite-fusion-synth`, AOT-compiles
//! to metallib, loads into the Metal device, dispatches with all 18
//! buffer bindings filled with dummy data, and times.
//!
//! Emitted CSV rows match the lookup key in
//! `MetalSynthPreAttnImpl::cost_us`:
//!
//!   synth_pre_attn_<act>_<scale>_gs<gs>,M,hidden,0,cost_us
//!   synth_pre_attn_init_<act>_<scale>_gs<gs>,M,hidden,0,cost_us
//!
//! Once these rows are merged into `cost_<chip>.csv` the solver's
//! `MetalSynthPreAttnImpl` returns measured cost (in place of the
//! 1e9 µs sentinel today), and the temporary `apply_synth_replacement`
//! post-pass + `bucket_m < 2` gate in `codegen.rs` can be removed.

use crate::util::{self, Buffer};
use dispatch2::DispatchData;
use ferrite_fusion_synth::{
    aot::aot_compile_metallib,
    fuse_pass::{
        synthesize_pre_attn_chunk, synthesize_pre_attn_init_chunk, ChunkConstants,
        SynthesisBackend,
    },
};
use ferrite_metal_kernels::stream::MetalStream;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLDataType, MTLDevice,
    MTLFunctionConstantValues, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

/// One per-model shape tuple. Each combination produces its own
/// `synth_pre_attn_*` source — only `(M, hidden)` discriminates in
/// the CSV key, so we cover the per-arch (num_q, num_kv, head_dim,
/// rot_dim) combos that actually ship.
struct ModelShape {
    name:     &'static str,
    hidden:   u32,
    num_q:    u32,
    num_kv:   u32,
    head_dim: u32,
    rot_dim:  u32,
    rms_eps:  f32,
}

const BLOCK_SIZE: u32 = 16;
const GROUP_SIZE: u32 = 64;

const SHAPES: &[ModelShape] = &[
    ModelShape {
        name: "llama_3_2_1b", hidden: 2048, num_q: 32, num_kv: 8,
        head_dim: 64, rot_dim: 64, rms_eps: 1.0e-5,
    },
    ModelShape {
        name: "llama_3_2_3b", hidden: 3072, num_q: 24, num_kv: 8,
        head_dim: 128, rot_dim: 128, rms_eps: 1.0e-5,
    },
    ModelShape {
        name: "llama_3_1_8b", hidden: 4096, num_q: 32, num_kv: 8,
        head_dim: 128, rot_dim: 128, rms_eps: 1.0e-5,
    },
];

const M_BUCKETS: &[u32] = &[1, 2, 4, 8, 16];

pub fn run(launch_overhead_us: f64) {
    eprintln!("Starting synth_pre_attn sweep...");
    for shape in SHAPES {
        for &init in &[false, true] {
            for &m in M_BUCKETS {
                let cost_us = bench_one(shape, m, init, launch_overhead_us);
                let kind = if init {
                    "synth_pre_attn_init_bfloat_half_gs64"
                } else {
                    "synth_pre_attn_bfloat_half_gs64"
                };
                println!("{kind},{m},{hidden},0,{cost_us:.2}", hidden = shape.hidden);
                eprintln!(
                    "  {kind} {} M={m} hidden={}: {cost_us:.2} us",
                    shape.name, shape.hidden,
                );
            }
        }
    }
    eprintln!("synth_pre_attn sweep complete");
}

fn bench_one(s: &ModelShape, m: u32, init: bool, launch_overhead_us: f64) -> f64 {
    let consts = ChunkConstants {
        hidden:       s.hidden,
        num_q_heads:  s.num_q,
        num_kv_heads: s.num_kv,
        head_dim:     s.head_dim,
        rot_dim:      s.rot_dim,
        block_size:   BLOCK_SIZE,
        intermediate: 0,
        m,
        group_size:   GROUP_SIZE,
        rms_norm_eps: s.rms_eps,
    };
    let synth = if init {
        synthesize_pre_attn_init_chunk(SynthesisBackend::Metal, "bfloat", "half", &consts)
    } else {
        synthesize_pre_attn_chunk(SynthesisBackend::Metal, "bfloat", "half", &consts)
    };
    let bytes = aot_compile_metallib(&synth.symbol, &synth.source);
    assert!(!bytes.is_empty(), "AOT compile produced empty metallib for {}", synth.symbol);

    let device = util::device();
    // The DispatchData wrapper expects a 'static slice. The sweep is
    // a short-lived binary that exits after CSV emission, so leaking
    // is fine — keeps the alloc alive for the device's lifetime.
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

    // Per-buffer size analysis — mirrors the binding layout in
    // `interpreter/metal/lowering.rs::SynthPreAttn`.
    let act = 2usize;          // bf16
    let scale = 2usize;        // f16
    let q_n   = (s.num_q  * s.head_dim) as usize;
    let kv_n  = (s.num_kv * s.head_dim) as usize;
    let hid   = s.hidden as usize;
    let mu    = m as usize;
    let gs    = GROUP_SIZE as usize;

    let q_out      = util::create_buffer(mu * q_n * act);
    let residual   = util::create_buffer(mu * hid * act);
    let delta      = util::create_buffer(mu * hid * act);
    let rms_w      = util::create_buffer(hid * scale);
    let q_packed   = util::create_buffer(q_n * hid / 2);
    let q_scales   = util::create_buffer(q_n * hid / gs * scale);
    let q_biases   = util::create_buffer(q_n * hid / gs * scale);
    let k_packed   = util::create_buffer(kv_n * hid / 2);
    let k_scales   = util::create_buffer(kv_n * hid / gs * scale);
    let k_biases   = util::create_buffer(kv_n * hid / gs * scale);
    let v_packed   = util::create_buffer(kv_n * hid / 2);
    let v_scales   = util::create_buffer(kv_n * hid / gs * scale);
    let v_biases   = util::create_buffer(kv_n * hid / gs * scale);
    // RoPE cos/sin cache: production cache is (max_seq, rot_dim) — 4096
    // tokens × rot_dim × act dtype is enough headroom for any positions[t]=0.
    let cos_sin    = util::create_buffer(4096 * (s.rot_dim as usize) * act);
    let positions  = util::create_buffer(mu * 4);
    let slot_map   = util::create_buffer(mu * 4);
    util::zero_buffer(&positions);
    util::zero_buffer(&slot_map);
    // KV cache: at minimum one block per layer. slot_mapping is zeroed
    // so all writes land in block 0.
    let kv_cache_bytes = 1 * (BLOCK_SIZE as usize) * kv_n * act;
    let kv_cache_k = util::create_buffer(kv_cache_bytes);
    let kv_cache_v = util::create_buffer(kv_cache_bytes);

    let bufs: [&Buffer; 18] = [
        &q_out, &residual, &delta, &rms_w,
        &q_packed, &q_scales, &q_biases,
        &k_packed, &k_scales, &k_biases,
        &v_packed, &v_scales, &v_biases,
        &cos_sin, &positions, &slot_map,
        &kv_cache_k, &kv_cache_v,
    ];

    let num_heads_total = s.num_q + 2 * s.num_kv;
    let threads_per_tg  = 32 * s.head_dim / 4; // MK_SIMD_SIZE * HEAD_DIM / MK_ROWS_PER_SIMDGROUP

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
            width:  m as usize,
            height: num_heads_total as usize,
            depth:  1,
        };
        let tpt = MTLSize {
            width:  threads_per_tg as usize,
            height: 1,
            depth:  1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(tg, tpt);
        enc.endEncoding();
        stream.commit().expect("commit");
        stream.synchronize().expect("sync");
    })
}
