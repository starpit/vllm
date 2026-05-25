// SPDX-License-Identifier: Apache-2.0
//! Attention prefill (`attention_prefill_sdpa_v2_paged_bf16_specialized`)
//! cost sweep at Llama-3.2-3B prefill shape.
//!
//! Drives the production `AttentionPrefillSdpaPaged` kernel for the
//! single-sequence case (no prior cached prefix) at M=512/1024/2048 so we
//! can compare head-to-head with `mx.fast.scaled_dot_product_attention` —
//! see `scripts/bench_attn_mlx.py`. This sweep is opt-in
//! (`FERRITE_SWEEP=attention`); the row format
//! `attention_prefill_sdpa_v2_paged_bf16,M,N=num_q_heads,K=head_dim,cost_us`
//! is descriptive only — not consumed by `cost_us_for` (the production
//! attention dispatch path doesn't go through `cost_us_for` today).
//!
//! Shape: B=1, NUM_Q_HEADS=24, NUM_KV_HEADS=8, HEAD_DIM=128, BLOCK_SIZE=16,
//! bfloat16. Matches the Llama-3.2-3B-Instruct-4bit prefill shape that
//! [[project-metal-kernel-codegen-gap]] flagged as the next suspect.

use crate::util::{self, Buffer, Device};
use ferrite_metal_kernels::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use ferrite_metal_kernels::stream::MetalStream;
use half::bf16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};

const HEAD_DIM: u32 = 128;
const NUM_Q_HEADS: u32 = 24;
const NUM_KV_HEADS: u32 = 8;
const ATTN_SCALE: f32 = 0.088_388_35; // 1/sqrt(128)
const BLOCK_SIZE: u32 = 16;
const MAX_BLOCKS_PER_SEQ: u32 = 128; // matches Llama32Probe production constant

pub fn run(launch_overhead_us: f64) {
    eprintln!("Starting attention prefill sweep (Llama-3.2-3B shape)...");

    let device = util::device();
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone())
        .expect("compile standard shaders");

    let key_sdpa = PipelineKey::new(
        "attention",
        "attention_prefill_sdpa_v2_paged_bf16_specialized",
        vec![
            ConstantValue::uint(0, HEAD_DIM),
            ConstantValue::uint(1, NUM_Q_HEADS),
            ConstantValue::uint(2, NUM_KV_HEADS),
            ConstantValue::float(3, ATTN_SCALE),
            ConstantValue::uint(4, BLOCK_SIZE),
            ConstantValue::uint(5, MAX_BLOCKS_PER_SEQ),
        ],
    );
    let pipeline_sdpa = cache
        .get_or_build(&key_sdpa)
        .expect("build attention pipeline");

    // Contiguous steel_attention (MLX FA-2 port, on-disk at
    // shaders/attention_steel.metal). Function constants:
    //   200 align_Q  201 align_K  300 has_mask  301 do_causal  302 has_sinks
    // For our bench: align_Q+align_K=true (M%32==0, M%16==0), no mask,
    // causal=true, no sinks.
    let key_steel = PipelineKey::new(
        "attention_steel",
        "attention_steel_bf16_bq32_bk16_bd128_wm4_wn1",
        vec![
            ConstantValue::boolean(200, true),  // align_Q
            ConstantValue::boolean(201, true),  // align_K
            ConstantValue::boolean(300, false), // has_mask
            ConstantValue::boolean(301, true),  // do_causal
            ConstantValue::boolean(302, false), // has_sinks
        ],
    );
    let pipeline_steel = cache
        .get_or_build(&key_steel)
        .expect("build attention_steel pipeline");

    // Steel paged kernel (the broken one). At M=1024 (kv_len multiple
    // of BK=16) the partial-block load_safe path is NOT exercised; if
    // this matches the contig steel kernel, the bug is exclusively in
    // partial-block handling. At M=10 (kv_rem=10) load_safe DOES fire,
    // exposing the partial-block bug.
    let paged_constants = |debug_mode: u32| {
        vec![
            ConstantValue::uint(0, HEAD_DIM),
            ConstantValue::uint(1, NUM_Q_HEADS),
            ConstantValue::uint(2, NUM_KV_HEADS),
            ConstantValue::float(3, ATTN_SCALE),
            ConstantValue::uint(4, BLOCK_SIZE),
            ConstantValue::uint(5, MAX_BLOCKS_PER_SEQ),
            ConstantValue::uint(99, debug_mode),
        ]
    };
    let key_paged_steel = PipelineKey::new(
        "attention_steel_paged",
        "attention_steel_paged_bf16_bq32_bk16_bd128_wm4_wn1_bs16",
        paged_constants(0),
    );
    let pipeline_paged_steel = cache
        .get_or_build(&key_paged_steel)
        .expect("build attention_steel_paged pipeline");

    // Diagnostic build: same kernel with DEBUG_MODE=1. Replaces store
    // with a per-lane marker write so we can count which (TG, sg, lane)
    // tuples actually reach the store path.
    let key_paged_dbg1 = PipelineKey::new(
        "attention_steel_paged",
        "attention_steel_paged_bf16_bq32_bk16_bd128_wm4_wn1_bs16",
        paged_constants(1),
    );
    let pipeline_paged_dbg1 = cache
        .get_or_build(&key_paged_dbg1)
        .expect("build attention_steel_paged debug-1 pipeline");

    // Diagnostic build: same kernel with DEBUG_MODE=3. Overwrites Otile
    // with constant 1.0 right before the store. If output is 100% 1.0
    // → store path works for ALL elements; bug is upstream. If still
    // 77% zeros → store path itself has the bug.
    let key_paged_dbg3 = PipelineKey::new(
        "attention_steel_paged",
        "attention_steel_paged_bf16_bq32_bk16_bd128_wm4_wn1_bs16",
        paged_constants(3),
    );
    let pipeline_paged_dbg3 = cache
        .get_or_build(&key_paged_dbg3)
        .expect("build attention_steel_paged debug-3 pipeline");

    // Diagnostic build: DEBUG_MODE=4 sets Otile.val_frags[f] = f+1.
    // If output shows the right marker at the right dim-frag, the
    // store path correctly distributes frags. If frag 14 is still
    // zero, store/row_bin_op corrupts that specific frag.
    let key_paged_dbg4 = PipelineKey::new(
        "attention_steel_paged",
        "attention_steel_paged_bf16_bq32_bk16_bd128_wm4_wn1_bs16",
        paged_constants(4),
    );
    let pipeline_paged_dbg4 = cache
        .get_or_build(&key_paged_dbg4)
        .expect("build attention_steel_paged debug-4 pipeline");

    // Correctness sanity: run the contiguous steel kernel at M=64 and
    // M=10 and compare to a pure-CPU SDPA reference. M=10 exercises
    // the partial-Q-tile path (q_tile_full=false → store_safe).
    let max_diff = verify_steel_correctness(device, &pipeline_steel, 64);
    eprintln!(
        "attention_steel_bf16 vs CPU SDPA @ M=64: max |abs diff| = {max_diff:.4} ({})",
        if max_diff < 0.02 { "PASS" } else { "FAIL" }
    );
    let max_diff_partial = verify_steel_correctness(device, &pipeline_steel, 10);
    eprintln!(
        "attention_steel_bf16 vs CPU SDPA @ M=10 (partial-Q tile): max |abs diff| = {max_diff_partial:.4} ({})",
        if max_diff_partial < 0.02 {
            "PASS"
        } else {
            "FAIL"
        }
    );

    // Cross-check: steel_paged at M=1024 (kv_rem=0, partial-block path
    // does NOT fire) — should produce identical results to the
    // bench's paged CPU golden. Test with BOTH thread layouts:
    //   (32, 4, 1) = MLX-style — what we used to verify the new loader
    //   (128, 1, 1) = production lowering style
    // If only one passes, production dispatch shape is wrong.
    let pf = verify_paged_steel_correctness(device, &pipeline_paged_steel, 1024, (32, 4, 1));
    eprintln!(
        "attention_steel_paged_bf16 vs CPU paged-SDPA @ M=1024 (32,4,1): max |abs diff| = {pf:.4} ({})",
        if pf < 0.05 { "PASS" } else { "FAIL" }
    );
    let pf2 = verify_paged_steel_correctness(device, &pipeline_paged_steel, 1024, (128, 1, 1));
    eprintln!(
        "attention_steel_paged_bf16 vs CPU paged-SDPA @ M=1024 (128,1,1) [production]: max |abs diff| = {pf2:.4} ({})",
        if pf2 < 0.05 { "PASS" } else { "FAIL" }
    );
    let pp = verify_paged_steel_correctness(device, &pipeline_paged_steel, 10, (32, 4, 1));
    eprintln!(
        "attention_steel_paged_bf16 vs CPU paged-SDPA @ M=10 (32,4,1): max |abs diff| = {pp:.4} ({})",
        if pp < 0.05 { "PASS" } else { "FAIL" }
    );
    let pp2 = verify_paged_steel_correctness(device, &pipeline_paged_steel, 10, (128, 1, 1));
    eprintln!(
        "attention_steel_paged_bf16 vs CPU paged-SDPA @ M=10 (128,1,1) [production]: max |abs diff| = {pp2:.4} ({})",
        if pp2 < 0.05 { "PASS" } else { "FAIL" }
    );
    // Sweep small M (covers real chat prompt length).
    for m_try in [13u32, 16, 20, 25, 32, 33, 48, 64, 100, 256] {
        let d = verify_paged_steel_correctness(device, &pipeline_paged_steel, m_try, (128, 1, 1));
        eprintln!(
            "attention_steel_paged_bf16 @ M={m_try:>4} (128,1,1): max |abs diff| = {d:.4} ({})",
            if d < 0.05 { "PASS" } else { "FAIL" }
        );
    }

    // CROSS-CHECK: same inputs through OLD (sdpa_paged) and NEW
    // (steel_paged) kernels at multiple data-range scales. Models in
    // production may have wider Q/K values than the bench's default
    // [-1, 1) distribution; if the kernels disagree at higher
    // magnitudes that'd explain the production failure.
    for scale in [1.0_f32, 4.0, 10.0, 30.0] {
        let diff =
            cross_check_paged_kernels(device, &pipeline_sdpa, &pipeline_paged_steel, 36, scale);
        eprintln!(
            "CROSS-CHECK OLD vs NEW @ M=36 data_scale={scale}: max |abs diff| = {diff:.4} ({})",
            if diff < 0.05 { "AGREE" } else { "DISAGREE" }
        );
    }

    // DEBUG_MODE=1 diagnostic: count which simdgroups reach the store
    // path. Each lane writes (simd_group_id + 1) * 10 + 1.0 at its
    // O[(tm+sm)*Q_stride_tok + sn] slot. With BQ=32, BD=128, WM=4:
    //   sg=0 → marker = 11.0
    //   sg=1 → marker = 21.0
    //   sg=2 → marker = 31.0
    //   sg=3 → marker = 41.0
    // 128 lanes per TG × 768 TGs at M=1024 = 98,304 unique write positions.
    // If all 4 simdgroups reach the store, we see all 4 markers.
    diagnose_paged_store_path(device, &pipeline_paged_dbg1, 1024);
    diagnose_paged_otile_store(device, &pipeline_paged_dbg3, 1024);
    diagnose_paged_zero_pattern(device, &pipeline_paged_steel, 1024);
    diagnose_paged_zero_pattern(device, &pipeline_paged_steel, 32);
    diagnose_paged_frag_markers(device, &pipeline_paged_dbg4, 1024);

    let ms: &[u32] = &[512, 1024, 2048];
    for &m in ms {
        let cost_us = bench_attention_prefill(device, &pipeline_sdpa, m, launch_overhead_us);
        println!("attention_prefill_sdpa_v2_paged_bf16,{m},{NUM_Q_HEADS},{HEAD_DIM},{cost_us:.2}");
    }
    for &m in ms {
        let cost_us = bench_attention_steel(device, &pipeline_steel, m, launch_overhead_us);
        println!("attention_steel_bf16,{m},{NUM_Q_HEADS},{HEAD_DIM},{cost_us:.2}");
    }
    for &m in ms {
        let cost_us =
            bench_attention_steel_paged(device, &pipeline_paged_steel, m, launch_overhead_us);
        println!("attention_steel_paged_bf16,{m},{NUM_Q_HEADS},{HEAD_DIM},{cost_us:.2}");
    }

    eprintln!("attention prefill sweep complete");
}

/// Timing-only run of the paged steel attention kernel at the same shape
/// the production lowering would dispatch (single sequence, prefix=0,
/// sequential physical blocks 0..N).
fn bench_attention_steel_paged(
    device: &Device,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: u32,
    launch_overhead_us: f64,
) -> f64 {
    let m_usz = m as usize;
    let num_q = NUM_Q_HEADS as usize;
    let num_kv = NUM_KV_HEADS as usize;
    let head_dim = HEAD_DIM as usize;
    let block_size = BLOCK_SIZE as usize;
    let max_blocks = MAX_BLOCKS_PER_SEQ as usize;
    let num_blocks = m_usz.div_ceil(block_size).max(1);
    const BQ: u32 = 32;

    let q_data: Vec<bf16> = (0..(m_usz * num_q * head_dim))
        .map(|i| bf16::from_f32(((i % 257) as f32) * 0.01))
        .collect();
    let kv_elts = num_blocks * num_kv * block_size * head_dim;
    let k_cache_data: Vec<bf16> = (0..kv_elts)
        .map(|i| bf16::from_f32(((i % 251) as f32) * 0.013))
        .collect();
    let v_cache_data: Vec<bf16> = (0..kv_elts)
        .map(|i| bf16::from_f32(((i % 241) as f32) * 0.017))
        .collect();

    let mut cu_seqlens_q: Vec<u32> = vec![0; m_usz + 2];
    cu_seqlens_q[1] = m;
    let seq_used_k: Vec<u32> = vec![m];
    let mut block_table: Vec<u32> = vec![0; max_blocks];
    for (i, slot) in block_table.iter_mut().take(num_blocks).enumerate() {
        *slot = i as u32;
    }

    let q_buf = upload_bf16(device, &q_data);
    let cu_buf = upload_u32(device, &cu_seqlens_q);
    let seq_used_k_buf = upload_u32(device, &seq_used_k);
    let block_table_buf = upload_u32(device, &block_table);
    let k_cache_buf = upload_bf16(device, &k_cache_data);
    let v_cache_buf = upload_bf16(device, &v_cache_data);
    let o_elts = m_usz * num_q * head_dim;
    let o_buf = util::create_buffer(o_elts * std::mem::size_of::<bf16>());

    let mut stream = MetalStream::new(device);
    util::time_kernel(launch_overhead_us, 3, 20, || {
        let cb = stream.get_command_buffer().expect("cmd buf").clone();
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&cu_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&seq_used_k_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&block_table_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&k_cache_buf), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&v_cache_buf), 0, 6);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: m.div_ceil(BQ) as usize,
                height: NUM_Q_HEADS as usize,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
        enc.endEncoding();
        stream.commit().expect("commit");
        stream.synchronize().expect("sync");
    })
    .max(0.0)
}

/// CPU reference for contiguous causal SDPA at `[B=1, H, T, D]` layout.
/// Pure-f32; bf16-round on input read. Returns max |abs diff| between
/// kernel output and CPU output across all (h, t, d).
fn verify_steel_correctness(
    device: &Device,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: u32,
) -> f32 {
    let m_usz = m as usize;
    let num_q = NUM_Q_HEADS as usize;
    let num_kv = NUM_KV_HEADS as usize;
    let head_dim = HEAD_DIM as usize;
    let group_ratio = num_q / num_kv;
    const BQ: u32 = 32;
    const BK: u32 = 16;

    let q_elts = num_q * m_usz * head_dim;
    let kv_elts = num_kv * m_usz * head_dim;
    let q_data: Vec<bf16> = (0..q_elts)
        .map(|i| bf16::from_f32(((i % 257) as f32) * 0.01))
        .collect();
    let k_data: Vec<bf16> = (0..kv_elts)
        .map(|i| bf16::from_f32(((i % 251) as f32) * 0.013))
        .collect();
    let v_data: Vec<bf16> = (0..kv_elts)
        .map(|i| bf16::from_f32(((i % 241) as f32) * 0.017))
        .collect();

    // CPU reference — read inputs back as f32-from-bf16 (lossy round
    // matches kernel's device-load of bf16 → f32 accumulator).
    let q_f32: Vec<f32> = q_data.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_data.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_data.iter().map(|x| x.to_f32()).collect();
    let mut out_cpu = vec![0.0_f32; q_elts];
    for h in 0..num_q {
        let kv_h = h / group_ratio;
        for t_q in 0..m_usz {
            let q_off = h * m_usz * head_dim + t_q * head_dim;
            let mut scores = vec![0.0_f32; t_q + 1];
            for t_k in 0..=t_q {
                let k_off = kv_h * m_usz * head_dim + t_k * head_dim;
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q_f32[q_off + d] * k_f32[k_off + d];
                }
                scores[t_k] = dot * ATTN_SCALE;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for t_k in 0..=t_q {
                    let v_off = kv_h * m_usz * head_dim + t_k * head_dim;
                    acc += scores[t_k] * v_f32[v_off + d];
                }
                out_cpu[q_off + d] = acc;
            }
        }
    }

    // Same as the paged check — count GPU zero outputs to see if
    // contig kernel also leaves outputs unwritten at partial-Q tiles.
    let params = AttnParams {
        b: 1,
        h: NUM_Q_HEADS as i32,
        d: HEAD_DIM as i32,
        q_l: m as i32,
        k_l: m as i32,
        gqa_factor: group_ratio as i32,
        scale: ATTN_SCALE,
        n_q: m.div_ceil(BQ) as i32,
        n_k: m.div_ceil(BK) as i32,
        n_q_aligned: (m / BQ) as i32,
        n_k_aligned: (m / BK) as i32,
        q_l_rem: (m % BQ) as i32,
        k_l_rem: (m % BK) as i32,
        q_l_off: 0,
        q_strides: [q_elts as i64, (m_usz * head_dim) as i64, head_dim as i64],
        k_strides: [kv_elts as i64, (m_usz * head_dim) as i64, head_dim as i64],
        v_strides: [kv_elts as i64, (m_usz * head_dim) as i64, head_dim as i64],
        o_strides: [q_elts as i64, (m_usz * head_dim) as i64, head_dim as i64],
    };

    let q_buf = upload_bf16(device, &q_data);
    let k_buf = upload_bf16(device, &k_data);
    let v_buf = upload_bf16(device, &v_data);
    let o_buf = util::create_buffer(q_elts * std::mem::size_of::<bf16>());
    let params_buf = upload_bytes(device, unsafe {
        std::slice::from_raw_parts(
            (&params as *const AttnParams) as *const u8,
            std::mem::size_of::<AttnParams>(),
        )
    });

    let mut stream = MetalStream::new(device);
    let cb = stream.get_command_buffer().expect("cmd buf").clone();
    let enc = cb.computeCommandEncoder().expect("encoder");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&v_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&params_buf), 0, 4);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: m.div_ceil(BQ) as usize,
            height: NUM_Q_HEADS as usize,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    );
    enc.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    // D2H + compare.
    let mut max_diff = 0.0_f32;
    let mut n_zero = 0usize;
    let o_ptr = o_buf.contents().as_ptr() as *const bf16;
    for i in 0..q_elts {
        let gpu = unsafe { *o_ptr.add(i) }.to_f32();
        let diff = (gpu - out_cpu[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if gpu == 0.0 {
            n_zero += 1;
        }
    }
    eprintln!("  (contig zero count: {n_zero}/{q_elts})");
    max_diff
}

fn bench_attention_prefill(
    device: &Device,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: u32,
    launch_overhead_us: f64,
) -> f64 {
    let m_usz = m as usize;
    let num_q = NUM_Q_HEADS as usize;
    let num_kv = NUM_KV_HEADS as usize;
    let head_dim = HEAD_DIM as usize;
    let block_size = BLOCK_SIZE as usize;
    let max_blocks = MAX_BLOCKS_PER_SEQ as usize;
    let num_blocks = m_usz.div_ceil(block_size).max(1);

    // Inputs/outputs are filled by initializing as bf16 of `i % 257` so
    // values fit the bf16 range and aren't all zero (which can shortcut
    // softmax). Real-content distribution doesn't matter for timing.
    let q_data: Vec<bf16> = (0..(m_usz * num_q * head_dim))
        .map(|i| bf16::from_f32(((i % 257) as f32) * 0.01))
        .collect();
    let kv_elts = num_blocks * num_kv * block_size * head_dim;
    let k_cache_data: Vec<bf16> = (0..kv_elts)
        .map(|i| bf16::from_f32(((i % 251) as f32) * 0.013))
        .collect();
    let v_cache_data: Vec<bf16> = (0..kv_elts)
        .map(|i| bf16::from_f32(((i % 241) as f32) * 0.017))
        .collect();

    // cu_seqlens_q: single sequence [0, M] + sentinel zero (kernel scans
    // until hi <= lo). Size `bucket+2` per existing tests.
    let mut cu_seqlens_q: Vec<u32> = vec![0; m_usz + 2];
    cu_seqlens_q[1] = m;
    // seq_used_k: total cached K per seq — single seq, all M tokens.
    let seq_used_k: Vec<u32> = vec![m];
    // block_table: contiguous physical blocks 0..num_blocks for this seq.
    let mut block_table: Vec<u32> = vec![0; max_blocks];
    for (i, slot) in block_table.iter_mut().take(num_blocks).enumerate() {
        *slot = i as u32;
    }

    let q_buf = upload_bf16(device, &q_data);
    let cu_buf = upload_u32(device, &cu_seqlens_q);
    let seq_used_k_buf = upload_u32(device, &seq_used_k);
    let block_table_buf = upload_u32(device, &block_table);
    let k_cache_buf = upload_bf16(device, &k_cache_data);
    let v_cache_buf = upload_bf16(device, &v_cache_data);
    let output_buf = util::create_buffer(m_usz * num_q * head_dim * std::mem::size_of::<bf16>());

    let mut stream = MetalStream::new(device);
    util::time_kernel(launch_overhead_us, 3, 20, || {
        let cb = stream.get_command_buffer().expect("cmd buf").clone();
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&output_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&cu_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&seq_used_k_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&block_table_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&k_cache_buf), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&v_cache_buf), 0, 6);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: NUM_Q_HEADS as usize,
                height: m_usz,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        stream.commit().expect("commit");
        stream.synchronize().expect("sync");
    })
}

/// AttnParams layout per mlx_steel_attn/params.h. C++ default packing —
/// 11 × `int` (44 bytes) + 1 × `float` (4 bytes) + 12 × `int64_t`
/// (`Q/K/V/O_strides[3]` flattened) = 144 bytes. Layout-matched here so
/// the const buffer at buffer(4) is read correctly by the kernel.
#[repr(C)]
struct AttnParams {
    b: i32,
    h: i32,
    d: i32,
    q_l: i32,
    k_l: i32,
    gqa_factor: i32,
    scale: f32,
    n_q: i32,
    n_k: i32,
    n_q_aligned: i32,
    n_k_aligned: i32,
    q_l_rem: i32,
    k_l_rem: i32,
    q_l_off: i32,
    q_strides: [i64; 3],
    k_strides: [i64; 3],
    v_strides: [i64; 3],
    o_strides: [i64; 3],
}

fn bench_attention_steel(
    device: &Device,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: u32,
    launch_overhead_us: f64,
) -> f64 {
    let m_usz = m as usize;
    let num_q = NUM_Q_HEADS as usize;
    let num_kv = NUM_KV_HEADS as usize;
    let head_dim = HEAD_DIM as usize;
    const BQ: u32 = 32;
    const BK: u32 = 16;

    // Q: [B=1, N_q, T_q=M, D]
    // K, V: [B=1, N_kv, T_kv=M, D]
    // O: [B=1, N_q, T_q=M, D]
    let q_elts = num_q * m_usz * head_dim;
    let kv_elts = num_kv * m_usz * head_dim;
    let q_data: Vec<bf16> = (0..q_elts)
        .map(|i| bf16::from_f32(((i % 257) as f32) * 0.01))
        .collect();
    let k_data: Vec<bf16> = (0..kv_elts)
        .map(|i| bf16::from_f32(((i % 251) as f32) * 0.013))
        .collect();
    let v_data: Vec<bf16> = (0..kv_elts)
        .map(|i| bf16::from_f32(((i % 241) as f32) * 0.017))
        .collect();

    let params = AttnParams {
        b: 1,
        h: NUM_Q_HEADS as i32,
        d: HEAD_DIM as i32,
        q_l: m as i32,
        k_l: m as i32,
        gqa_factor: (NUM_Q_HEADS / NUM_KV_HEADS) as i32,
        scale: ATTN_SCALE,
        n_q: m.div_ceil(BQ) as i32,
        n_k: m.div_ceil(BK) as i32,
        n_q_aligned: (m / BQ) as i32,
        n_k_aligned: (m / BK) as i32,
        q_l_rem: (m % BQ) as i32,
        k_l_rem: (m % BK) as i32,
        q_l_off: 0,
        // Q strides: [batch, head, seq] for [1, N_q, M, D] contiguous.
        q_strides: [
            (num_q * m_usz * head_dim) as i64,
            (m_usz * head_dim) as i64,
            head_dim as i64,
        ],
        k_strides: [
            (num_kv * m_usz * head_dim) as i64,
            (m_usz * head_dim) as i64,
            head_dim as i64,
        ],
        v_strides: [
            (num_kv * m_usz * head_dim) as i64,
            (m_usz * head_dim) as i64,
            head_dim as i64,
        ],
        o_strides: [
            (num_q * m_usz * head_dim) as i64,
            (m_usz * head_dim) as i64,
            head_dim as i64,
        ],
    };

    let q_buf = upload_bf16(device, &q_data);
    let k_buf = upload_bf16(device, &k_data);
    let v_buf = upload_bf16(device, &v_data);
    let o_buf = util::create_buffer(q_elts * std::mem::size_of::<bf16>());
    let params_buf = upload_bytes(device, unsafe {
        std::slice::from_raw_parts(
            (&params as *const AttnParams) as *const u8,
            std::mem::size_of::<AttnParams>(),
        )
    });

    let mut stream = MetalStream::new(device);
    util::time_kernel(launch_overhead_us, 3, 20, || {
        let cb = stream.get_command_buffer().expect("cmd buf").clone();
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&k_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&v_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&params_buf), 0, 4);
        }
        // Grid: (NQ, H, B) per MLX scaled_dot_product_attention.cpp:160.
        // Threads: (32, wm=4, wn=1) = 128 per TG.
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: m.div_ceil(BQ) as usize,
                height: NUM_Q_HEADS as usize,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        );
        enc.endEncoding();
        stream.commit().expect("commit");
        stream.synchronize().expect("sync");
    })
}

/// DEBUG_MODE=1 diagnostic: how many output positions did each simdgroup
/// actually write? Each lane writes `(sg+1)*10 + 1.0` at its store slot
/// in the marker kernel. We count occurrences of 11.0, 21.0, 31.0, 41.0
/// to see whether all 4 simdgroups reach the store path.
fn diagnose_paged_store_path(
    device: &Device,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: u32,
) {
    let m_usz = m as usize;
    let num_q = NUM_Q_HEADS as usize;
    let num_kv = NUM_KV_HEADS as usize;
    let head_dim = HEAD_DIM as usize;
    let block_size = BLOCK_SIZE as usize;
    let max_blocks = MAX_BLOCKS_PER_SEQ as usize;
    let num_blocks = m_usz.div_ceil(block_size).max(1);
    const BQ: u32 = 32;

    let q_data: Vec<bf16> = (0..(m_usz * num_q * head_dim))
        .map(|i| bf16::from_f32(((i % 257) as f32) * 0.01))
        .collect();
    let kv_elts = num_blocks * num_kv * block_size * head_dim;
    let k_cache_data: Vec<bf16> = vec![bf16::from_f32(0.5); kv_elts];
    let v_cache_data: Vec<bf16> = vec![bf16::from_f32(0.5); kv_elts];

    let mut cu_seqlens_q: Vec<u32> = vec![0; m_usz + 2];
    cu_seqlens_q[1] = m;
    let seq_used_k: Vec<u32> = vec![m];
    let mut block_table: Vec<u32> = vec![0; max_blocks];
    for (i, slot) in block_table.iter_mut().take(num_blocks).enumerate() {
        *slot = i as u32;
    }

    let q_buf = upload_bf16(device, &q_data);
    let cu_buf = upload_u32(device, &cu_seqlens_q);
    let seq_used_k_buf = upload_u32(device, &seq_used_k);
    let block_table_buf = upload_u32(device, &block_table);
    let k_cache_buf = upload_bf16(device, &k_cache_data);
    let v_cache_buf = upload_bf16(device, &v_cache_data);
    let o_elts = m_usz * num_q * head_dim;
    let o_buf = util::create_buffer(o_elts * std::mem::size_of::<bf16>());

    let mut stream = MetalStream::new(device);
    let cb = stream.get_command_buffer().expect("cmd buf").clone();
    let enc = cb.computeCommandEncoder().expect("encoder");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&cu_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&seq_used_k_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&block_table_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&k_cache_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&v_cache_buf), 0, 6);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: m.div_ceil(BQ) as usize,
            height: NUM_Q_HEADS as usize,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    );
    enc.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    // Count occurrences of the 4 simdgroup markers.
    let o_ptr = o_buf.contents().as_ptr() as *const bf16;
    let mut counts = [0usize; 5]; // [0]=other/zero, [1]=sg0, [2]=sg1, [3]=sg2, [4]=sg3
    for i in 0..o_elts {
        let v = unsafe { *o_ptr.add(i) }.to_f32();
        let sg = ((v - 1.0) / 10.0).round() as i32;
        match sg {
            1 => counts[1] += 1,
            2 => counts[2] += 1,
            3 => counts[3] += 1,
            4 => counts[4] += 1,
            _ => counts[0] += 1,
        }
    }
    let total = o_elts;
    eprintln!(
        "DEBUG store-path coverage @ M={m}: sg0={} sg1={} sg2={} sg3={} other(or zero)={} (total={total})",
        counts[1], counts[2], counts[3], counts[4], counts[0]
    );
    let _ = (num_kv, head_dim); // suppress unused warnings if any
}

/// DEBUG_MODE=4: each frag's elements set to (frag_id + 1). Output
/// dim-frag j should contain only values (j + 1). Reveals whether the
/// store correctly distributes frag values to their col range, OR
/// whether some frags are dropped / merged.
fn diagnose_paged_frag_markers(
    device: &Device,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: u32,
) {
    let m_usz = m as usize;
    let num_q = NUM_Q_HEADS as usize;
    let num_kv = NUM_KV_HEADS as usize;
    let head_dim = HEAD_DIM as usize;
    let block_size = BLOCK_SIZE as usize;
    let max_blocks = MAX_BLOCKS_PER_SEQ as usize;
    let num_blocks = m_usz.div_ceil(block_size).max(1);
    const BQ: u32 = 32;

    let q_data: Vec<bf16> = vec![bf16::from_f32(0.5); m_usz * num_q * head_dim];
    let kv_elts = num_blocks * num_kv * block_size * head_dim;
    let k_cache_data: Vec<bf16> = vec![bf16::from_f32(0.5); kv_elts];
    let v_cache_data: Vec<bf16> = vec![bf16::from_f32(0.5); kv_elts];
    let mut cu_seqlens_q: Vec<u32> = vec![0; m_usz + 2];
    cu_seqlens_q[1] = m;
    let seq_used_k: Vec<u32> = vec![m];
    let mut block_table: Vec<u32> = vec![0; max_blocks];
    for (i, slot) in block_table.iter_mut().take(num_blocks).enumerate() {
        *slot = i as u32;
    }
    let q_buf = upload_bf16(device, &q_data);
    let cu_buf = upload_u32(device, &cu_seqlens_q);
    let seq_used_k_buf = upload_u32(device, &seq_used_k);
    let block_table_buf = upload_u32(device, &block_table);
    let k_cache_buf = upload_bf16(device, &k_cache_data);
    let v_cache_buf = upload_bf16(device, &v_cache_data);
    let o_elts = m_usz * num_q * head_dim;
    let o_buf = util::create_buffer(o_elts * std::mem::size_of::<bf16>());

    let mut stream = MetalStream::new(device);
    let cb = stream.get_command_buffer().expect("cmd buf").clone();
    let enc = cb.computeCommandEncoder().expect("encoder");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&cu_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&seq_used_k_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&block_table_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&k_cache_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&v_cache_buf), 0, 6);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: m.div_ceil(BQ) as usize,
            height: NUM_Q_HEADS as usize,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    );
    enc.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let o_ptr = o_buf.contents().as_ptr() as *const bf16;
    eprintln!("DEBUG frag-markers @ M=1024 (expect dim-frag j → value j+1):");
    // Sample first Q-token, first head, all dims.
    for j in 0..(head_dim / 8) {
        let mut counts = std::collections::HashMap::<i32, usize>::new();
        for col in 0..8 {
            let dim = j * 8 + col;
            for q_pos in 0..m_usz {
                let v = unsafe { *o_ptr.add(q_pos * num_q * head_dim + 0 * head_dim + dim) }
                    .to_f32() as i32;
                *counts.entry(v).or_insert(0) += 1;
            }
        }
        let summary: Vec<String> = counts.iter().map(|(v, c)| format!("v={v}:{c}")).collect();
        eprintln!(
            "  dim-frag {j} (cols {}..{}): {}",
            j * 8,
            j * 8 + 7,
            summary.join(" ")
        );
    }
    let _ = (num_kv, head_dim);
}

/// REAL paged kernel run: identify the PATTERN of which output
/// positions are zero. Output is `[total_q, num_q_heads, head_dim]`.
/// Bucket non-zero counts by (head_dim_position, q_token mod 8,
/// q_token mod 32) to spot frag/simdgroup correlations.
fn diagnose_paged_zero_pattern(
    device: &Device,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: u32,
) {
    let m_usz = m as usize;
    let num_q = NUM_Q_HEADS as usize;
    let num_kv = NUM_KV_HEADS as usize;
    let head_dim = HEAD_DIM as usize;
    let block_size = BLOCK_SIZE as usize;
    let max_blocks = MAX_BLOCKS_PER_SEQ as usize;
    let num_blocks = m_usz.div_ceil(block_size).max(1);
    const BQ: u32 = 32;

    // All-ones inputs: expected output = 1.0 at every position.
    // (V=1, softmax weights sum to 1 → output = sum_k(w_k * 1) = 1.)
    // If we still see the same zero pattern, the bug is structural.
    let q_data: Vec<bf16> = vec![bf16::from_f32(1.0); m_usz * num_q * head_dim];
    let kv_elts = num_blocks * num_kv * block_size * head_dim;
    let k_cache_data: Vec<bf16> = vec![bf16::from_f32(1.0); kv_elts];
    let v_cache_data: Vec<bf16> = vec![bf16::from_f32(1.0); kv_elts];

    let mut cu_seqlens_q: Vec<u32> = vec![0; m_usz + 2];
    cu_seqlens_q[1] = m;
    let seq_used_k: Vec<u32> = vec![m];
    let mut block_table: Vec<u32> = vec![0; max_blocks];
    for (i, slot) in block_table.iter_mut().take(num_blocks).enumerate() {
        *slot = i as u32;
    }

    let q_buf = upload_bf16(device, &q_data);
    let cu_buf = upload_u32(device, &cu_seqlens_q);
    let seq_used_k_buf = upload_u32(device, &seq_used_k);
    let block_table_buf = upload_u32(device, &block_table);
    let k_cache_buf = upload_bf16(device, &k_cache_data);
    let v_cache_buf = upload_bf16(device, &v_cache_data);
    let o_elts = m_usz * num_q * head_dim;
    let o_buf = util::create_buffer(o_elts * std::mem::size_of::<bf16>());

    let mut stream = MetalStream::new(device);
    let cb = stream.get_command_buffer().expect("cmd buf").clone();
    let enc = cb.computeCommandEncoder().expect("encoder");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&cu_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&seq_used_k_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&block_table_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&k_cache_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&v_cache_buf), 0, 6);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: m.div_ceil(BQ) as usize,
            height: NUM_Q_HEADS as usize,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    );
    enc.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let o_ptr = o_buf.contents().as_ptr() as *const bf16;

    // Per-dim non-zero count (head 0, q=0..1023). Reveals WHICH
    // cols within each frag have non-zero output.
    eprintln!("All-1.0 input — Real-run zero pattern by exact dim (head=0):");
    let mut by_dim = [0usize; 128];
    for q_pos in 0..m_usz {
        for d in 0..head_dim {
            let v = unsafe { *o_ptr.add(q_pos * num_q * head_dim + 0 * head_dim + d) }.to_f32();
            if v != 0.0 {
                by_dim[d] += 1;
            }
        }
    }
    for j in 0..16 {
        let row: Vec<String> = (0..8)
            .map(|c| format!("{:>4}", by_dim[j * 8 + c]))
            .collect();
        eprintln!(
            "  frag {j:>2} (cols {:>3}..{:>3}): [{}]",
            j * 8,
            j * 8 + 7,
            row.join(" ")
        );
    }
    // Bucket by dim-frag AND q_head to see if frag-coverage depends on head.
    let mut nonzero_per_dim_frag = [0usize; 16];
    let mut nonzero_per_head = vec![0usize; num_q];
    let mut nonzero_head_x_frag = vec![[0usize; 16]; num_q];
    for i in 0..o_elts {
        let v = unsafe { *o_ptr.add(i) }.to_f32();
        if v != 0.0 {
            let dim = i % head_dim;
            let head = (i / head_dim) % num_q;
            let frag = dim / 8;
            nonzero_per_dim_frag[frag] += 1;
            nonzero_per_head[head] += 1;
            nonzero_head_x_frag[head][frag] += 1;
        }
    }
    eprintln!("Real-run zero pattern @ M=1024:");
    eprint!("  non-zero by dim-frag: ");
    for (j, c) in nonzero_per_dim_frag.iter().enumerate() {
        eprint!("[{j}]{c} ");
    }
    eprintln!();
    eprint!("  non-zero by q_head: ");
    for (h, c) in nonzero_per_head.iter().enumerate() {
        eprint!("[{h}]{c} ");
    }
    eprintln!();
    // Show frag coverage for heads 0, 1, 2, 3 (one full GQA group).
    for h in 0..6.min(num_q) {
        eprint!("  head {h} non-zero per frag: ");
        for c in nonzero_head_x_frag[h].iter() {
            eprint!("{c} ");
        }
        eprintln!();
    }
    let _ = (num_kv, head_dim);
}

/// DEBUG_MODE=3 diagnostic: with Otile forced to 1.0 right before the
/// store, count how many output positions end up exactly 1.0. If all
/// → store path works for every element; bug is upstream of the store.
/// If still 77% zero → store path itself fails.
fn diagnose_paged_otile_store(
    device: &Device,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: u32,
) {
    let m_usz = m as usize;
    let num_q = NUM_Q_HEADS as usize;
    let num_kv = NUM_KV_HEADS as usize;
    let head_dim = HEAD_DIM as usize;
    let block_size = BLOCK_SIZE as usize;
    let max_blocks = MAX_BLOCKS_PER_SEQ as usize;
    let num_blocks = m_usz.div_ceil(block_size).max(1);
    const BQ: u32 = 32;

    let q_data: Vec<bf16> = vec![bf16::from_f32(0.5); m_usz * num_q * head_dim];
    let kv_elts = num_blocks * num_kv * block_size * head_dim;
    let k_cache_data: Vec<bf16> = vec![bf16::from_f32(0.5); kv_elts];
    let v_cache_data: Vec<bf16> = vec![bf16::from_f32(0.5); kv_elts];

    let mut cu_seqlens_q: Vec<u32> = vec![0; m_usz + 2];
    cu_seqlens_q[1] = m;
    let seq_used_k: Vec<u32> = vec![m];
    let mut block_table: Vec<u32> = vec![0; max_blocks];
    for (i, slot) in block_table.iter_mut().take(num_blocks).enumerate() {
        *slot = i as u32;
    }

    let q_buf = upload_bf16(device, &q_data);
    let cu_buf = upload_u32(device, &cu_seqlens_q);
    let seq_used_k_buf = upload_u32(device, &seq_used_k);
    let block_table_buf = upload_u32(device, &block_table);
    let k_cache_buf = upload_bf16(device, &k_cache_data);
    let v_cache_buf = upload_bf16(device, &v_cache_data);
    let o_elts = m_usz * num_q * head_dim;
    let o_buf = util::create_buffer(o_elts * std::mem::size_of::<bf16>());

    let mut stream = MetalStream::new(device);
    let cb = stream.get_command_buffer().expect("cmd buf").clone();
    let enc = cb.computeCommandEncoder().expect("encoder");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&cu_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&seq_used_k_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&block_table_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&k_cache_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&v_cache_buf), 0, 6);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: m.div_ceil(BQ) as usize,
            height: NUM_Q_HEADS as usize,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    );
    enc.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let o_ptr = o_buf.contents().as_ptr() as *const bf16;
    let mut n_one = 0usize;
    let mut n_zero = 0usize;
    let mut n_other = 0usize;
    for i in 0..o_elts {
        let v = unsafe { *o_ptr.add(i) }.to_f32();
        if v == 1.0 {
            n_one += 1;
        } else if v == 0.0 {
            n_zero += 1;
        } else {
            n_other += 1;
        }
    }
    eprintln!(
        "DEBUG Otile-forced-to-1 store @ M={m}: ones={n_one}/{o_elts} ({:.1}%) zeros={n_zero} other={n_other}",
        100.0 * n_one as f64 / o_elts as f64
    );
    let _ = (num_kv, head_dim);
}

/// Run both the OLD sdpa_paged kernel and the NEW steel_paged kernel
/// on identical (deterministic) inputs and return the max abs diff
/// between their outputs. Both kernels implement causal SDPA, so they
/// should agree up to bf16 precision.
#[allow(clippy::too_many_lines)]
fn cross_check_paged_kernels(
    device: &Device,
    pipeline_old: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pipeline_new: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: u32,
    data_scale: f32,
) -> f32 {
    let m_usz = m as usize;
    let num_q = NUM_Q_HEADS as usize;
    let num_kv = NUM_KV_HEADS as usize;
    let head_dim = HEAD_DIM as usize;
    let block_size = BLOCK_SIZE as usize;
    let max_blocks = MAX_BLOCKS_PER_SEQ as usize;
    let num_blocks = m_usz.div_ceil(block_size).max(1);
    const BQ: u32 = 32;

    let mix = |i: usize, prime: u64| -> f32 {
        let h = ((i as u64).wrapping_mul(prime)) ^ ((i as u64) << 7);
        let n = (h % 4096) as f32 / 4096.0;
        (n * 2.0 - 1.0) * data_scale
    };
    let q_data: Vec<bf16> = (0..(m_usz * num_q * head_dim))
        .map(|i| bf16::from_f32(mix(i, 257)))
        .collect();
    let kv_elts = num_blocks * num_kv * block_size * head_dim;
    let k_cache_data: Vec<bf16> = (0..kv_elts).map(|i| bf16::from_f32(mix(i, 251))).collect();
    let v_cache_data: Vec<bf16> = (0..kv_elts).map(|i| bf16::from_f32(mix(i, 241))).collect();
    let mut cu_seqlens_q: Vec<u32> = vec![0; m_usz + 2];
    cu_seqlens_q[1] = m;
    let seq_used_k: Vec<u32> = vec![m];
    let mut block_table: Vec<u32> = vec![0; max_blocks];
    for (i, slot) in block_table.iter_mut().take(num_blocks).enumerate() {
        *slot = i as u32;
    }

    let dispatch_kernel = |pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
                           tg: MTLSize,
                           tpt: MTLSize|
     -> Vec<f32> {
        let q_buf = upload_bf16(device, &q_data);
        let cu_buf = upload_u32(device, &cu_seqlens_q);
        let seq_used_k_buf = upload_u32(device, &seq_used_k);
        let block_table_buf = upload_u32(device, &block_table);
        let k_cache_buf = upload_bf16(device, &k_cache_data);
        let v_cache_buf = upload_bf16(device, &v_cache_data);
        let o_elts = m_usz * num_q * head_dim;
        let o_buf = util::create_buffer(o_elts * std::mem::size_of::<bf16>());

        let mut stream = MetalStream::new(device);
        let cb = stream.get_command_buffer().expect("cmd buf").clone();
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&cu_buf), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&seq_used_k_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&block_table_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&k_cache_buf), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&v_cache_buf), 0, 6);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(tg, tpt);
        enc.endEncoding();
        stream.commit().expect("commit");
        stream.synchronize().expect("sync");
        let o_ptr = o_buf.contents().as_ptr() as *const bf16;
        let mut out = vec![0.0_f32; o_elts];
        for i in 0..o_elts {
            out[i] = unsafe { *o_ptr.add(i) }.to_f32();
        }
        out
    };

    // OLD kernel: grid (num_q_heads, total_q, 1) × (1024, 1, 1) threads.
    let old_out = dispatch_kernel(
        pipeline_old,
        MTLSize {
            width: NUM_Q_HEADS as usize,
            height: m_usz,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    // NEW kernel: grid (ceil(M/BQ), num_q_heads, 1) × (128, 1, 1) threads.
    let new_out = dispatch_kernel(
        pipeline_new,
        MTLSize {
            width: m.div_ceil(BQ) as usize,
            height: NUM_Q_HEADS as usize,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );

    let mut max_diff = 0.0_f32;
    let mut max_idx = 0usize;
    for (i, (a, b)) in old_out.iter().zip(new_out.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    let total_per_q = num_q * head_dim;
    let q_pos = max_idx / total_per_q;
    let head = (max_idx % total_per_q) / head_dim;
    let dim = max_idx % head_dim;
    eprintln!(
        "  worst at q_pos={q_pos} head={head} dim={dim}: old={:.4} new={:.4} diff={max_diff:.4}",
        old_out[max_idx], new_out[max_idx]
    );
    max_diff
}

/// CPU reference for paged causal SDPA. Single sequence, prefix_len=0,
/// physical blocks allocated 0..num_blocks. Returns max |abs diff|
/// between kernel output and CPU reference. Caller picks `m` to
/// exercise either the kv_rem=0 (full-block) or kv_rem!=0 (partial-
/// block) path: `m % BK == 0` → kv_rem=0; else partial.
fn verify_paged_steel_correctness(
    device: &Device,
    pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    m: u32,
    tg_threads: (usize, usize, usize),
) -> f32 {
    let m_usz = m as usize;
    let num_q = NUM_Q_HEADS as usize;
    let num_kv = NUM_KV_HEADS as usize;
    let head_dim = HEAD_DIM as usize;
    let block_size = BLOCK_SIZE as usize;
    let max_blocks = MAX_BLOCKS_PER_SEQ as usize;
    let group_ratio = num_q / num_kv;
    let num_blocks = m_usz.div_ceil(block_size).max(1);
    const BQ: u32 = 32;

    // Mixed-sign data closer to real-model bf16 outputs. Splay across
    // [-1, 1] so softmax sees realistic score distributions.
    let mix = |i: usize, prime: u64| -> f32 {
        let h = ((i as u64).wrapping_mul(prime)) ^ ((i as u64) << 7);
        let n = (h % 4096) as f32 / 4096.0; // [0, 1)
        n * 2.0 - 1.0 // [-1, 1)
    };
    let q_data: Vec<bf16> = (0..(m_usz * num_q * head_dim))
        .map(|i| bf16::from_f32(mix(i, 257)))
        .collect();
    let kv_elts = num_blocks * num_kv * block_size * head_dim;
    let k_cache_data: Vec<bf16> = (0..kv_elts).map(|i| bf16::from_f32(mix(i, 251))).collect();
    let v_cache_data: Vec<bf16> = (0..kv_elts).map(|i| bf16::from_f32(mix(i, 241))).collect();

    let mut cu_seqlens_q: Vec<u32> = vec![0; m_usz + 2];
    cu_seqlens_q[1] = m;
    let seq_used_k: Vec<u32> = vec![m];
    let mut block_table: Vec<u32> = vec![0; max_blocks];
    for (i, slot) in block_table.iter_mut().take(num_blocks).enumerate() {
        *slot = i as u32;
    }

    // CPU reference. Layout: Q [M, N_q, D], K/V cache [num_blocks, N_kv,
    // BLOCK_SIZE, D]. For single sequence, prefix_len=0, all tokens
    // are "new" with absolute K position = q position.
    let q_f32: Vec<f32> = q_data.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_cache_data.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_cache_data.iter().map(|x| x.to_f32()).collect();
    let mut out_cpu = vec![0.0_f32; m_usz * num_q * head_dim];
    let kv_block_stride = num_kv * block_size * head_dim;
    for q_pos in 0..m_usz {
        for h in 0..num_q {
            let kv_h = h / group_ratio;
            let q_off = q_pos * num_q * head_dim + h * head_dim;
            let attend_len = q_pos + 1;
            let mut scores = vec![0.0_f32; attend_len];
            for t in 0..attend_len {
                let logical_block = t / block_size;
                let block_offset = t % block_size;
                let physical_block = block_table[logical_block] as usize;
                let k_base = physical_block * kv_block_stride
                    + kv_h * block_size * head_dim
                    + block_offset * head_dim;
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q_f32[q_off + d] * k_f32[k_base + d];
                }
                scores[t] = dot * ATTN_SCALE;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for (t, &score) in scores.iter().enumerate() {
                    let logical_block = t / block_size;
                    let block_offset = t % block_size;
                    let physical_block = block_table[logical_block] as usize;
                    let v_base = physical_block * kv_block_stride
                        + kv_h * block_size * head_dim
                        + block_offset * head_dim;
                    acc += score * v_f32[v_base + d];
                }
                out_cpu[q_off + d] = acc;
            }
        }
    }

    // Kernel run on same inputs. Steel-paged dispatcher binding order
    // (see steel_attention_paged_kernel.h):
    //   buf(0) O, buf(1) Q, buf(2) cu_seqlens_q, buf(3) seq_used_k,
    //   buf(4) block_table, buf(5) k_cache, buf(6) v_cache.
    let q_buf = upload_bf16(device, &q_data);
    let cu_buf = upload_u32(device, &cu_seqlens_q);
    let seq_used_k_buf = upload_u32(device, &seq_used_k);
    let block_table_buf = upload_u32(device, &block_table);
    let k_cache_buf = upload_bf16(device, &k_cache_data);
    let v_cache_buf = upload_bf16(device, &v_cache_data);
    let o_elts = m_usz * num_q * head_dim;
    let o_buf = util::create_buffer(o_elts * std::mem::size_of::<bf16>());

    let mut stream = MetalStream::new(device);
    let cb = stream.get_command_buffer().expect("cmd buf").clone();
    let enc = cb.computeCommandEncoder().expect("encoder");
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&o_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&cu_buf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&seq_used_k_buf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&block_table_buf), 0, 4);
        enc.setBuffer_offset_atIndex(Some(&k_cache_buf), 0, 5);
        enc.setBuffer_offset_atIndex(Some(&v_cache_buf), 0, 6);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: m.div_ceil(BQ) as usize,
            height: NUM_Q_HEADS as usize,
            depth: 1,
        },
        MTLSize {
            width: tg_threads.0,
            height: tg_threads.1,
            depth: tg_threads.2,
        },
    );
    enc.endEncoding();
    stream.commit().expect("commit");
    stream.synchronize().expect("sync");

    let mut max_diff = 0.0_f32;
    let mut max_idx = 0usize;
    let o_ptr = o_buf.contents().as_ptr() as *const bf16;
    let mut n_zero = 0usize;
    for i in 0..o_elts {
        let gpu = unsafe { *o_ptr.add(i) }.to_f32();
        let diff = (gpu - out_cpu[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
        if gpu == 0.0 {
            n_zero += 1;
        }
    }
    // Surface basic shape of the failure: the worst element + how
    // many outputs the kernel left at zero. Near-100% zeros means
    // the kernel mostly didn't write; partial means some
    // threadgroups succeeded and others didn't.
    let total_per_q = num_q * head_dim;
    let q_pos = max_idx / total_per_q;
    let head = (max_idx % total_per_q) / head_dim;
    let dim = max_idx % head_dim;
    let gpu0 = unsafe { *o_ptr.add(max_idx) }.to_f32();
    eprintln!(
        "  worst at q_pos={q_pos} head={head} dim={dim}: gpu={gpu0:.4} cpu={:.4} diff={max_diff:.4} ({n_zero}/{o_elts} zeros)",
        out_cpu[max_idx]
    );
    max_diff
}

fn upload_bytes(device: &Device, data: &[u8]) -> Buffer {
    let bytes = data.len().max(1);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents().as_ptr() as *mut u8,
            data.len(),
        );
    }
    buf
}

fn upload_bf16(device: &Device, data: &[bf16]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(1);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(data),
        );
    }
    buf
}

fn upload_u32(device: &Device, data: &[u32]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(1);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            std::mem::size_of_val(data),
        );
    }
    buf
}
