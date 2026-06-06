// SPDX-License-Identifier: Apache-2.0
//! P1 kernel parity tests for the Gemma4 port:
//!   * `gelu_mul_f16` (decomposed GeGLU q-MLP tail) vs CPU gelu_tanh.
//!   * `tanh_soft_cap_f16_specialized` (final logit softcap, cap as
//!     function constant 1) vs CPU.
//!   * `ATTN_WINDOW` sliding-window mask in the paged SDPA prefill and
//!     decode kernels vs a CPU softmax-attention reference (window 0 ≡
//!     causal; window W masks `q_abs - k >= W`).
//!
//! GPU tests — run with `--test-threads=1` (standing rule).

use ferrite_metal_kernels::detect_device;
use ferrite_metal_kernels::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};
use half::f16;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};

type Device = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLDevice>>;
type Buffer = objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>;

fn buf_f16(device: &Device, data: &[f32]) -> Buffer {
    let h: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    let bytes = (h.len() * 2).max(4);
    let buf = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            h.as_ptr() as *const u8,
            buf.contents().as_ptr() as *mut u8,
            h.len() * 2,
        );
    }
    buf
}

fn buf_u32(device: &Device, data: &[u32]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(4);
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

fn buf_u64(device: &Device, data: &[u64]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(8);
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

fn buf_zero(device: &Device, bytes: usize) -> Buffer {
    let buf = device
        .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
        .expect("newBuffer");
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, bytes) };
    buf
}

fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
    unsafe {
        std::slice::from_raw_parts(buf.contents().as_ptr() as *const f16, n)
            .iter()
            .map(|h| h.to_f32())
            .collect()
    }
}

fn pseudo(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as u32 & 0x00FF_FFFF) as f32 / (1u32 << 23) as f32 - 1.0) * scale
        })
        .collect()
}

fn gelu_tanh_ref(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56;
    const COEFF: f32 = 0.044_715;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + COEFF * x * x * x)).tanh())
}

fn dispatch_1d(
    queue: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    pipeline: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
    bufs: &[&Buffer],
    threads: usize,
) {
    let cb = queue.commandBuffer().expect("commandBuffer");
    let enc = cb.computeCommandEncoder().expect("encoder");
    enc.setComputePipelineState(pipeline);
    for (i, b) in bufs.iter().enumerate() {
        unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, i) };
    }
    let tg = 256usize;
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: threads.div_ceil(tg),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}

#[test]
fn gelu_mul_f16_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let queue = device.newCommandQueue().expect("queue");
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let n = 4096usize;
    let gate = pseudo(7, n, 4.0);
    let up = pseudo(11, n, 2.0);

    let key = PipelineKey::new(
        "silu_mul",
        "gelu_mul_f16",
        vec![ConstantValue::uint(0, n as u32)],
    );
    let pipeline = cache.get_or_build(&key).expect("gelu_mul pipeline");

    let gate_buf = buf_f16(&device, &gate);
    let up_buf = buf_f16(&device, &up);
    let out_buf = buf_zero(&device, n * 2);
    dispatch_1d(&queue, &pipeline, &[&out_buf, &gate_buf, &up_buf], n);

    let got = read_f16(&out_buf, n);
    let mut max_err = 0f32;
    for i in 0..n {
        // reference uses the f16-rounded inputs the kernel actually read
        let g = f16::from_f32(gate[i]).to_f32();
        let u = f16::from_f32(up[i]).to_f32();
        let want = gelu_tanh_ref(g) * u;
        max_err = max_err.max((got[i] - want).abs());
    }
    assert!(max_err < 5e-3, "gelu_mul max_err {max_err}");
}

#[test]
fn tanh_soft_cap_f16_specialized_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let queue = device.newCommandQueue().expect("queue");
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let n = 2048usize;
    let cap = 30.0f32;
    let x = pseudo(13, n, 80.0); // exercises the saturating tail

    let key = PipelineKey::new(
        "elementwise",
        "tanh_soft_cap_f16_specialized",
        // slot 1: slot 0 is BIAS_ADD_NUM_COLS (file-scoped indices).
        vec![ConstantValue::float(1, cap)],
    );
    let pipeline = cache.get_or_build(&key).expect("tanh_soft_cap pipeline");

    let in_buf = buf_f16(&device, &x);
    let out_buf = buf_zero(&device, n * 2);
    dispatch_1d(&queue, &pipeline, &[&in_buf, &out_buf], n);

    let got = read_f16(&out_buf, n);
    let mut max_err = 0f32;
    for i in 0..n {
        let xi = f16::from_f32(x[i]).to_f32();
        let want = cap * (xi / cap).tanh();
        max_err = max_err.max((got[i] - want).abs());
    }
    assert!(max_err < 0.05, "tanh_soft_cap max_err {max_err}");
}

// ── Sliding-window attention parity ─────────────────────────────────

struct AttnCase {
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
    window: i32,
}

/// CPU reference: causal (+ optional window) softmax attention over a
/// contiguous K/V laid out `[block, kv_head, tok_in_block, head_dim]`
/// (the paged layout with an identity block table).
#[allow(clippy::too_many_arguments)]
fn attn_ref(
    q: &[f32],     // [num_q, num_q_heads, head_dim] (one row per query token)
    k: &[f32],     // paged layout, f16-rounded
    v: &[f32],
    case: &AttnCase,
    block_size: usize,
    q_positions: &[usize], // absolute K-axis position per query row
    scale: f32,
) -> Vec<f32> {
    let &AttnCase {
        num_q_heads,
        num_kv_heads,
        head_dim,
        kv_len,
        window,
    } = case;
    let group = num_q_heads / num_kv_heads;
    let kv_at = |tok: usize, kvh: usize, d: usize| -> f32 {
        let blk = tok / block_size;
        let tib = tok % block_size;
        let idx = ((blk * num_kv_heads + kvh) * block_size + tib) * head_dim + d;
        k[idx]
    };
    let vv_at = |tok: usize, kvh: usize, d: usize| -> f32 {
        let blk = tok / block_size;
        let tib = tok % block_size;
        let idx = ((blk * num_kv_heads + kvh) * block_size + tib) * head_dim + d;
        v[idx]
    };
    let mut out = vec![0f32; q_positions.len() * num_q_heads * head_dim];
    for (qi, &q_abs) in q_positions.iter().enumerate() {
        for h in 0..num_q_heads {
            let kvh = h / group;
            let qrow = &q[(qi * num_q_heads + h) * head_dim..][..head_dim];
            let mut scores = Vec::with_capacity(kv_len);
            for t in 0..kv_len {
                if t > q_abs {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                if window > 0 && (q_abs - t) as i64 >= window as i64 {
                    scores.push(f32::NEG_INFINITY);
                    continue;
                }
                let mut s = 0f32;
                for d in 0..head_dim {
                    s += qrow[d] * kv_at(t, kvh, d);
                }
                scores.push(s * scale);
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let orow = &mut out[(qi * num_q_heads + h) * head_dim..][..head_dim];
            for t in 0..kv_len {
                if exps[t] == 0.0 {
                    continue;
                }
                let w = exps[t] / denom;
                for d in 0..head_dim {
                    orow[d] += w * vv_at(t, kvh, d);
                }
            }
        }
    }
    out
}

fn attn_constants(case: &AttnCase, block_size: usize, max_blocks: usize) -> Vec<ConstantValue> {
    vec![
        ConstantValue::uint(0, case.head_dim as u32),
        ConstantValue::uint(1, case.num_q_heads as u32),
        ConstantValue::uint(2, case.num_kv_heads as u32),
        ConstantValue::float(3, 1.0 / (case.head_dim as f32).sqrt()),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, max_blocks as u32),
        ConstantValue::uint(6, 0), // BPC=0 single-buffer fast path
        ConstantValue::int(7, case.window),
    ]
}

fn run_window_case(case: AttnCase) {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let queue = device.newCommandQueue().expect("queue");
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let block_size = 16usize;
    let num_blocks = case.kv_len.div_ceil(block_size);
    let kv_elems = num_blocks * case.num_kv_heads * block_size * case.head_dim;

    let k_host = pseudo(101, kv_elems, 1.0);
    let v_host = pseudo(103, kv_elems, 1.0);
    let k_buf = buf_f16(&device, &k_host);
    let v_buf = buf_f16(&device, &v_host);
    // f16-rounded copies for the CPU reference
    let k_r: Vec<f32> = k_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
    let v_r: Vec<f32> = v_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();

    // chunk-address tables (BPC=0: entry 0 = buffer base)
    let k_tab = buf_u64(&device, &[k_buf.gpuAddress()]);
    let v_tab = buf_u64(&device, &[v_buf.gpuAddress()]);

    let block_table: Vec<u32> = (0..num_blocks as u32).collect();
    let bt_buf = buf_u32(&device, &block_table);
    let scale = 1.0 / (case.head_dim as f32).sqrt();

    // ── decode: 1 query at q_abs = kv_len-1 ──
    {
        let q_host = pseudo(107, case.num_q_heads * case.head_dim, 1.0);
        let q_buf = buf_f16(&device, &q_host);
        let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        let out_buf = buf_zero(&device, case.num_q_heads * case.head_dim * 2);
        let seq_used = buf_u32(&device, &[case.kv_len as u32]);

        let key = PipelineKey::new(
            "attention",
            "attention_via_cache_v2_f16_specialized",
            attn_constants(&case, block_size, num_blocks),
        );
        let pipeline = cache.get_or_build(&key).expect("decode pipeline");

        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&seq_used), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&bt_buf), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&k_tab), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&v_tab), 0, 5);
            // K/V are reached via raw gpuAddress through the chunk
            // table — declare residency explicitly (production uses
            // residency sets; without this the pager can serve stale
            // zero pages).
            enc.useResource_usage(
                k_buf.as_ref() as &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLResource>,
                objc2_metal::MTLResourceUsage::Read,
            );
            enc.useResource_usage(
                v_buf.as_ref() as &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLResource>,
                objc2_metal::MTLResourceUsage::Read,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: case.num_q_heads,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        let got = read_f16(&out_buf, case.num_q_heads * case.head_dim);
        let want = attn_ref(
            &q_r,
            &k_r,
            &v_r,
            &case,
            block_size,
            &[case.kv_len - 1],
            scale,
        );
        let max_err = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        assert!(
            max_err < 2e-2,
            "decode window={} max_err {max_err}",
            case.window
        );
    }

    // ── prefill: all kv_len queries, q_abs = row index ──
    {
        let total_q = case.kv_len;
        let q_host = pseudo(109, total_q * case.num_q_heads * case.head_dim, 1.0);
        let q_buf = buf_f16(&device, &q_host);
        let q_r: Vec<f32> = q_host.iter().map(|&x| f16::from_f32(x).to_f32()).collect();
        let out_buf = buf_zero(&device, total_q * case.num_q_heads * case.head_dim * 2);
        let cu_seqlens = buf_u32(&device, &[0, total_q as u32, 0, 0]);
        let seq_used = buf_u32(&device, &[case.kv_len as u32]);

        let key = PipelineKey::new(
            "attention",
            "attention_prefill_sdpa_v2_paged_f16_specialized",
            attn_constants(&case, block_size, num_blocks),
        );
        let pipeline = cache.get_or_build(&key).expect("prefill pipeline");

        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&q_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&cu_seqlens), 0, 2);
            enc.setBuffer_offset_atIndex(Some(&seq_used), 0, 3);
            enc.setBuffer_offset_atIndex(Some(&bt_buf), 0, 4);
            enc.setBuffer_offset_atIndex(Some(&k_tab), 0, 5);
            enc.setBuffer_offset_atIndex(Some(&v_tab), 0, 6);
            enc.useResource_usage(
                k_buf.as_ref() as &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLResource>,
                objc2_metal::MTLResourceUsage::Read,
            );
            enc.useResource_usage(
                v_buf.as_ref() as &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLResource>,
                objc2_metal::MTLResourceUsage::Read,
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: case.num_q_heads,
                height: total_q,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        let positions: Vec<usize> = (0..total_q).collect();
        let want = attn_ref(&q_r, &k_r, &v_r, &case, block_size, &positions, scale);
        let got = read_f16(&out_buf, total_q * case.num_q_heads * case.head_dim);
        let max_err = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        assert!(
            max_err < 2e-2,
            "prefill window={} max_err {max_err}",
            case.window
        );
    }
}

#[test]
fn attention_window_disabled_matches_causal_ref() {
    run_window_case(AttnCase {
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        kv_len: 50,
        window: 0,
    });
}

#[test]
fn attention_window_masks_old_keys() {
    // window smaller than kv_len: the mask actively drops keys.
    run_window_case(AttnCase {
        num_q_heads: 4,
        num_kv_heads: 2,
        head_dim: 64,
        kv_len: 50,
        window: 8,
    });
}

#[test]
fn attention_window_gemma4_sliding_geometry() {
    // Gemma4 sliding-layer geometry: 16 q-heads, 8 kv-heads, head_dim
    // 256, window 1024 — with kv_len > window so the mask is active.
    run_window_case(AttnCase {
        num_q_heads: 16,
        num_kv_heads: 8,
        head_dim: 256,
        kv_len: 1100,
        window: 1024,
    });
}









#[test]
fn rmsnorm_unit_f16_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let queue = device.newCommandQueue().expect("queue");
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    // Gemma4 v_norm shapes: rows = T*kv_heads, width = head_dim.
    let (rows, width) = (24usize, 256usize);
    let eps = 1e-6f32;
    let x = pseudo(31, rows * width, 2.0);

    let key = PipelineKey::new(
        "rmsnorm",
        "rmsnorm_unit_f16_specialized",
        vec![
            ConstantValue::uint(0, rows as u32),
            ConstantValue::uint(1, width as u32),
            ConstantValue::float(2, eps),
        ],
    );
    let pipeline = cache.get_or_build(&key).expect("rmsnorm_unit pipeline");
    let in_buf = buf_f16(&device, &x);
    let out_buf = buf_zero(&device, rows * width * 2);

    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&out_buf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&in_buf), 0, 1);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize { width: rows, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();

    let got = read_f16(&out_buf, rows * width);
    let mut max_err = 0f32;
    for r in 0..rows {
        let row: Vec<f32> = (0..width)
            .map(|i| f16::from_f32(x[r * width + i]).to_f32())
            .collect();
        let ms = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let rms = (ms + eps).sqrt();
        for i in 0..width {
            let want = row[i] / rms;
            max_err = max_err.max((got[r * width + i] - want).abs());
        }
    }
    assert!(max_err < 5e-3, "rmsnorm_unit max_err {max_err}");
}

#[test]
fn scalar_weight_mul_f16_matches_cpu() {
    let Some(di) = detect_device() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let device = di.device.clone();
    let queue = device.newCommandQueue().expect("queue");
    let cache = SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shaders");

    let n = 3840usize;
    let scalar = 0.937f32; // a plausible layer_scalar value
    let x = pseudo(37, n, 3.0);
    let key = PipelineKey::new(
        "elementwise",
        "scalar_weight_mul_f16_specialized",
        Vec::new(),
    );
    let pipeline = cache.get_or_build(&key).expect("scalar_weight_mul pipeline");
    let in_buf = buf_f16(&device, &x);
    let w_buf = buf_f16(&device, &[scalar]);
    let out_buf = buf_zero(&device, n * 2);
    dispatch_1d(&queue, &pipeline, &[&out_buf, &in_buf, &w_buf], n);

    let got = read_f16(&out_buf, n);
    let w = f16::from_f32(scalar).to_f32();
    let mut max_err = 0f32;
    for i in 0..n {
        let want = f16::from_f32(x[i]).to_f32() * w;
        max_err = max_err.max((got[i] - want).abs());
    }
    assert!(max_err < 5e-3, "scalar_weight_mul max_err {max_err}");
}
