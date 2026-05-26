// SPDX-License-Identifier: Apache-2.0
//! PD-wavefront Milestone A1 — on-device proof of the persistent
//! multi-threadgroup tape-loop.
//!
//! `wavefront_qmv_mega` is a persistent kernel launched with `P`
//! threadgroups (one per GPU core, co-resident). Each TG loops over the
//! qmv's N-block row-groups (`g = tgpos.x, +P, …`), composing the
//! validated `qmv_fast_impl` device function once per group — the
//! persistent-megakernel structure (one TG running many subtile ops in
//! ONE dispatch, not one TG per op). Its output must be BIT-IDENTICAL to
//! the whole `affine_qmv_fast` for every `P` (each output row is the same
//! per-K reduction regardless of which worker computes it).
//!
//! This isolates persistent-multi-TG + on-device atom composition +
//! co-residency, with NO cross-TG sync yet — Milestone A2 adds the
//! spinloop `Wait`/`Signal` flags and the folded shape-class `switch`.
//!
//! Integration test (not a lib `#[cfg(test)]` module) to skip the lib's
//! unrelated test-build rot. Run:
//! `cargo test -p ferrite-forward -F metal --test wavefront_mega_gpu`.
#![cfg(feature = "metal")]

use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_forward::interpreter::metal::__re::{Buffer, ComputePipelineState, Device};
use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{
    DequantDtype, QmvKernel, ScaleDtype, qmv_kernel_static_name,
};
use ferrite_metal_kernels::specialized_pipeline_cache::{
    ConstantValue, PipelineKey, SpecializedPipelineCache,
};

fn buffer_from_bytes(device: &Device, bytes: &[u8]) -> Buffer {
    unsafe {
        device
            .newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
            .expect("newBufferWithBytes")
    }
}

fn zeroed_buffer(device: &Device, n_bytes: usize) -> Buffer {
    device
        .newBufferWithLength_options(n_bytes, MTLResourceOptions::StorageModeShared)
        .expect("newBufferWithLength")
}

fn read_bf16(buf: &Buffer, n: usize) -> Vec<u16> {
    let ptr = buf.contents().as_ptr() as *const u16;
    unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
}

/// Bind w/scales/biases/x/y at args 0..5 and dispatch `grid_tg`
/// threadgroups of `[32,2,1]` (2 simdgroups = what `qmv_fast_impl`
/// expects), waiting for completion.
#[allow(clippy::too_many_arguments)]
fn dispatch_qmv(
    device: &Device,
    pipe: &ComputePipelineState,
    w: &Buffer,
    s: &Buffer,
    b: &Buffer,
    x: &Buffer,
    y: &Buffer,
    grid_tg: (usize, usize, usize),
) {
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(pipe);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(w), 0, 0);
        enc.setBuffer_offset_atIndex(Some(s), 0, 1);
        enc.setBuffer_offset_atIndex(Some(b), 0, 2);
        enc.setBuffer_offset_atIndex(Some(x), 0, 3);
        enc.setBuffer_offset_atIndex(Some(y), 0, 4);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid_tg.0,
            height: grid_tg.1,
            depth: grid_tg.2,
        },
        MTLSize {
            width: 32,
            height: 2,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}

#[test]
fn wavefront_qmv_mega_bit_exact_vs_whole() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // bf16 act / f16 scale, gs=64, 4-bit, M=1, N % 8 == 0 (fast variant).
    let (n, k, gs, bits) = (96u32, 512u32, 64u32, 4u32);
    let n_bytes_packed = (n * k / 2) as usize; // 2 nibbles/byte
    let n_groups = (n * k / gs) as usize;

    let mut s = 0x1234_5678u64;
    let mut byte = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (s >> 33) as u8
    };
    let packed: Vec<u8> = (0..n_bytes_packed).map(|_| byte()).collect();
    let f16le = |v: f32| half::f16::from_f32(v).to_bits().to_le_bytes();
    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let scales: Vec<u8> = (0..n_groups)
        .flat_map(|i| f16le(0.01 + 0.0001 * (i % 17) as f32))
        .collect();
    let biases: Vec<u8> = (0..n_groups)
        .flat_map(|i| f16le(-0.05 + 0.001 * (i % 13) as f32))
        .collect();
    let x: Vec<u8> = (0..k)
        .flat_map(|i| bf16le(((i % 7) as f32 - 3.0) * 0.1))
        .collect();

    let w_buf = buffer_from_bytes(&device, &packed);
    let s_buf = buffer_from_bytes(&device, &scales);
    let b_buf = buffer_from_bytes(&device, &biases);
    let x_buf = buffer_from_bytes(&device, &x);

    // K / N ride as function constants 0 / 1 (IN_VEC_SIZE / OUT_VEC_SIZE),
    // identical for both kernels.
    let constants = vec![
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
    ];

    // Reference: the trusted whole `affine_qmv_fast` (batched=0), grid
    // (M=1, ceil(N/8), 1).
    let fast_sym: &'static str = Box::leak(
        qmv_kernel_static_name(
            QmvKernel::Fast,
            DequantDtype::Bf16,
            ScaleDtype::F16,
            bits,
            gs,
        )
        .to_string()
        .into_boxed_str(),
    );
    let whole_pipe = cache
        .get_or_build(&PipelineKey::new(
            "quantized_qmv",
            fast_sym,
            constants.clone(),
        ))
        .expect("whole qmv pipeline");
    let y_ref = zeroed_buffer(&device, (n * 2) as usize);
    dispatch_qmv(
        &device,
        &whole_pipe,
        &w_buf,
        &s_buf,
        &b_buf,
        &x_buf,
        &y_ref,
        (1, (n as usize).div_ceil(8), 1),
    );
    let ref_bits = read_bf16(&y_ref, n as usize);
    assert!(
        ref_bits.iter().any(|&b| b != 0),
        "reference qmv produced all zeros"
    );

    // The persistent megakernel: P co-resident TGs, grid (P,1,1).
    let mega_pipe = cache
        .get_or_build(&PipelineKey::new(
            "quantized_qmv",
            "wavefront_qmv_mega_bf16_s_f16_gs_64_b_4",
            constants.clone(),
        ))
        .expect("wavefront_qmv_mega pipeline");
    for p in [1usize, 2, 4, 10] {
        let y_mega = zeroed_buffer(&device, (n * 2) as usize);
        dispatch_qmv(
            &device,
            &mega_pipe,
            &w_buf,
            &s_buf,
            &b_buf,
            &x_buf,
            &y_mega,
            (p, 1, 1),
        );
        let mega_bits = read_bf16(&y_mega, n as usize);
        assert_eq!(
            mega_bits, ref_bits,
            "wavefront_qmv_mega P={p} must be bit-exact vs whole affine_qmv_fast"
        );
    }
}

/// Dispatch the 2-stage wavefront megakernel: w0/s0/b0/x→y1, w1/s1/b1/y1→y2,
/// flags[P], y1c (u32-packed handoff), y1r (per-worker scratch) at args 0..12,
/// grid (P,1,1) of [32,2,1] threads.
#[allow(clippy::too_many_arguments)]
fn dispatch_2stage(
    device: &Device,
    pipe: &ComputePipelineState,
    bufs: [&Buffer; 12], // w0,s0,b0,x,y1,w1,s1,b1,y2,flags,y1c,y1r
    p: usize,
) {
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(pipe);
    for (i, b) in bufs.iter().enumerate() {
        unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, i) };
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: p,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 2,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}

// Cross-TG spinloop sync + folded shape-class switch, with the ATOMIC
// u32-packed handoff fix: producers pack their y1 stripe (8 bf16/group → 4
// u32) and `atomic_store` to a coherent buffer; consumers join on all flags,
// `atomic_load` + unpack into a private bf16 copy, then run stage-1 qmv on it.
// The atomic handoff is the cross-TG-visible path (non-atomic device writes
// race — root-caused in `wavefront_sync_probe.rs`, PAT 0/1/2 vs 3/4).
#[test]
fn wavefront_qmv_mega_2stage_bit_exact_vs_sequential() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // Two distinct shape classes: y1 = qmv(W0[512,512], x[512]); then
    // y2 = qmv(W1[1024,512], y1[512]). K must be %512 (fast variant K
    // loop), N %8. Stage 1 reads the WHOLE y1 → every stage-1 block
    // depends on every stage-0 stripe → the all-P-flag join.
    let (gs, bits) = (64u32, 4u32);
    let (k0, n0) = (512u32, 512u32);
    let (k1, n1) = (512u32, 1024u32); // k1 == n0 (y1 is stage-1's activation)

    let mut s = 0xC0FFEEu64;
    let mut byte = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (s >> 33) as u8
    };
    let f16le = |v: f32| half::f16::from_f32(v).to_bits().to_le_bytes();
    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let packed = |n: u32, k: u32, byte: &mut dyn FnMut() -> u8| -> Vec<u8> {
        (0..(n * k / 2) as usize).map(|_| byte()).collect()
    };
    let scal = |n: u32, k: u32, off: f32| -> Vec<u8> {
        (0..(n * k / gs) as usize)
            .flat_map(|i| f16le(off + 0.0001 * (i % 17) as f32))
            .collect()
    };

    let w0 = buffer_from_bytes(&device, &packed(n0, k0, &mut byte));
    let s0 = buffer_from_bytes(&device, &scal(n0, k0, 0.01));
    let b0 = buffer_from_bytes(&device, &scal(n0, k0, -0.05));
    let x: Vec<u8> = (0..k0)
        .flat_map(|i| bf16le(((i % 7) as f32 - 3.0) * 0.1))
        .collect();
    let x = buffer_from_bytes(&device, &x);
    let w1 = buffer_from_bytes(&device, &packed(n1, k1, &mut byte));
    let s1 = buffer_from_bytes(&device, &scal(n1, k1, 0.02));
    let b1 = buffer_from_bytes(&device, &scal(n1, k1, -0.03));

    let fast_sym: &'static str = Box::leak(
        qmv_kernel_static_name(
            QmvKernel::Fast,
            DequantDtype::Bf16,
            ScaleDtype::F16,
            bits,
            gs,
        )
        .to_string()
        .into_boxed_str(),
    );
    // Reference: two sequential whole `affine_qmv_fast` dispatches.
    let whole0 = cache
        .get_or_build(&PipelineKey::new(
            "quantized_qmv",
            fast_sym,
            vec![
                ConstantValue::int(0, k0 as i32),
                ConstantValue::int(1, n0 as i32),
            ],
        ))
        .expect("whole stage0");
    let whole1 = cache
        .get_or_build(&PipelineKey::new(
            "quantized_qmv",
            fast_sym,
            vec![
                ConstantValue::int(0, k1 as i32),
                ConstantValue::int(1, n1 as i32),
            ],
        ))
        .expect("whole stage1");
    let y1_ref = zeroed_buffer(&device, (n0 * 2) as usize);
    dispatch_qmv(
        &device,
        &whole0,
        &w0,
        &s0,
        &b0,
        &x,
        &y1_ref,
        (1, (n0 as usize).div_ceil(8), 1),
    );
    let y2_ref = zeroed_buffer(&device, (n1 * 2) as usize);
    dispatch_qmv(
        &device,
        &whole1,
        &w1,
        &s1,
        &b1,
        &y1_ref,
        &y2_ref,
        (1, (n1 as usize).div_ceil(8), 1),
    );
    let ref_bits = read_bf16(&y2_ref, n1 as usize);
    assert!(
        ref_bits.iter().any(|&b| b != 0),
        "reference produced all zeros"
    );

    // The 2-stage megakernel: P co-resident workers, spinloop join on y1.
    let mega = cache
        .get_or_build(&PipelineKey::new(
            "quantized_qmv",
            "wavefront_qmv_mega_2stage_bf16_s_f16_gs_64_b_4",
            vec![
                ConstantValue::int(2, k0 as i32),
                ConstantValue::int(3, n0 as i32),
                ConstantValue::int(4, k1 as i32),
                ConstantValue::int(5, n1 as i32),
            ],
        ))
        .expect("2stage pipeline");
    for p in [1usize, 2, 4, 10] {
        // The cross-TG handoff race is timing-dependent, so retry — a single
        // pass could be luck. The atomic u32-packed handoff must hold every
        // time.
        for iter in 0..50 {
            let y1 = zeroed_buffer(&device, (n0 * 2) as usize);
            let y2 = zeroed_buffer(&device, (n1 * 2) as usize);
            let flags = buffer_from_bytes(&device, &vec![0u8; p * 4]); // P atomic_uint, zeroed
            let y1c = buffer_from_bytes(&device, &vec![0u8; (n0 / 2) as usize * 4]); // u32-packed handoff
            let y1r = zeroed_buffer(&device, p * (n0 as usize) * 2); // [P*N0] per-worker bf16 scratch
            dispatch_2stage(
                &device,
                &mega,
                [
                    &w0, &s0, &b0, &x, &y1, &w1, &s1, &b1, &y2, &flags, &y1c, &y1r,
                ],
                p,
            );
            let mega_bits = read_bf16(&y2, n1 as usize);
            assert_eq!(
                mega_bits, ref_bits,
                "2-stage megakernel P={p} iter={iter} must be bit-exact vs two sequential \
                 whole qmv (a mismatch = a spinloop sync / data-before-flag ordering bug)"
            );
        }
    }
}

// ── rmsnorm atom (mittens/rmsnorm.h) composition proof ────────────────────

/// Bind output/input/weight at args 0..3 and dispatch `grid_tg` threadgroups
/// of `[tg_size,1,1]` (the rmsnorm reduction owns the whole row), waiting for
/// completion.
fn dispatch_rmsnorm(
    device: &Device,
    pipe: &ComputePipelineState,
    output: &Buffer,
    input: &Buffer,
    weight: &Buffer,
    grid_tg: usize,
    tg_size: usize,
) {
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(pipe);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(output), 0, 0);
        enc.setBuffer_offset_atIndex(Some(input), 0, 1);
        enc.setBuffer_offset_atIndex(Some(weight), 0, 2);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid_tg,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_size,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}

/// CPU RMSNorm reference (f32, sequential reduction): the noise-floor sanity
/// that the whole kernel computes rmsnorm at all (catches a botched atom
/// extraction). bf16 act in, f16 gain in, f32 out.
fn cpu_rmsnorm_bf16_s_f16(input: &[u16], weight: &[u16], m: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for row in 0..m {
        let mut ss = 0f32;
        for i in 0..n {
            let v = half::bf16::from_bits(input[row * n + i]).to_f32();
            ss += v * v;
        }
        let rms = (ss / n as f32 + eps).sqrt();
        for i in 0..n {
            let v = half::bf16::from_bits(input[row * n + i]).to_f32();
            let w = half::f16::from_bits(weight[i]).to_f32();
            out[row * n + i] = (v / rms) * w;
        }
    }
    out
}

// `wavefront_rmsnorm_mega`: P co-resident TGs each loop the rows they own,
// composing the mittens::rmsnorm_impl atom once per row (A1-style, disjoint
// outputs, no flags). Each per-row reduction uses the same tg_size as the
// whole kernel, so the output must be bit-exact vs the whole rmsnorm_specialized
// for every P. A CPU reference gates that the whole kernel (hence the atom) is
// actually correct, not just self-consistent with the mega.
#[test]
fn wavefront_rmsnorm_mega_bit_exact_vs_whole() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // Llama-3.2-1B hidden=2048 (N>=1024 ⇒ tg_size=1024, a power of two so the
    // reduction tree is exact); M=10 rows spread across the P workers.
    let (m, n, eps) = (10u32, 2048u32, 1e-5f32);
    let tg_size = (n as usize).min(1024);

    let mut st = 0xBEEF_F00Du64;
    let mut next = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0 // ~U(-3, 3)
    };
    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let f16le = |v: f32| half::f16::from_f32(v).to_bits().to_le_bytes();
    let input: Vec<u8> = (0..(m * n)).flat_map(|_| bf16le(next())).collect();
    let weight: Vec<u8> = (0..n)
        .flat_map(|i| f16le(0.5 + 0.01 * (i % 7) as f32))
        .collect();

    let in_buf = buffer_from_bytes(&device, &input);
    let w_buf = buffer_from_bytes(&device, &weight);

    // Function constants 0/1/2 = M / HIDDEN_SIZE / EPS (uint, uint, float).
    let constants = vec![
        ConstantValue::uint(0, m),
        ConstantValue::uint(1, n),
        ConstantValue::float(2, eps),
    ];

    // Reference: the whole rmsnorm_specialized (now a thin wrapper over the
    // mittens atom), grid (M,1,1).
    let whole = cache
        .get_or_build(&PipelineKey::new(
            "rmsnorm",
            "rmsnorm_bf16_s_f16_specialized",
            constants.clone(),
        ))
        .expect("whole rmsnorm pipeline");
    let y_ref = zeroed_buffer(&device, (m * n * 2) as usize);
    dispatch_rmsnorm(
        &device, &whole, &y_ref, &in_buf, &w_buf, m as usize, tg_size,
    );
    let ref_bits = read_bf16(&y_ref, (m * n) as usize);
    assert!(
        ref_bits.iter().any(|&b| b != 0),
        "whole rmsnorm produced all zeros"
    );

    // Noise-floor sanity vs the CPU reference (catches a dropped-term extraction
    // bug; tolerates GPU tree-reduction order + bf16 output rounding).
    let in_u16: Vec<u16> = input
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let w_u16: Vec<u16> = weight
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let cpu = cpu_rmsnorm_bf16_s_f16(&in_u16, &w_u16, m as usize, n as usize, eps);
    for (i, &c) in cpu.iter().enumerate() {
        let g = half::bf16::from_bits(ref_bits[i]).to_f32();
        assert!(
            (g - c).abs() <= 0.02 + 0.03 * c.abs(),
            "whole rmsnorm[{i}] gpu={g} vs cpu={c} exceeds noise floor"
        );
    }

    // The persistent megakernel: P co-resident TGs, grid (P,1,1), same tg_size.
    let mega = cache
        .get_or_build(&PipelineKey::new(
            "rmsnorm",
            "wavefront_rmsnorm_mega_bf16_s_f16",
            constants.clone(),
        ))
        .expect("wavefront_rmsnorm_mega pipeline");
    for p in [1usize, 2, 4, 10] {
        let y_mega = zeroed_buffer(&device, (m * n * 2) as usize);
        dispatch_rmsnorm(&device, &mega, &y_mega, &in_buf, &w_buf, p, tg_size);
        let mega_bits = read_bf16(&y_mega, (m * n) as usize);
        assert_eq!(
            mega_bits, ref_bits,
            "wavefront_rmsnorm_mega P={p} must be bit-exact vs whole rmsnorm_specialized"
        );
    }
}

// ── rope rotation atom (mittens/rope.h) composition proof ─────────────────

/// Dispatch the whole `rope_append_*_specialized`: q,k,v,cos_sin,positions,
/// slot_mapping,kv_k,kv_v at args 0..8, grid (tokens, q_heads, 1) of
/// [head_dim,1,1] threads.
fn dispatch_rope_append(
    device: &Device,
    pipe: &ComputePipelineState,
    bufs: [&Buffer; 8],
    grid: (usize, usize),
    head_dim: usize,
) {
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(pipe);
    for (i, b) in bufs.iter().enumerate() {
        unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, i) };
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid.0,
            height: grid.1,
            depth: 1,
        },
        MTLSize {
            width: head_dim,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}

/// Dispatch `wavefront_rope_mega`: q,k,cos_sin,positions at args 0..4, grid
/// (P,1,1) of [head_dim,1,1] threads.
fn dispatch_rope_mega(
    device: &Device,
    pipe: &ComputePipelineState,
    q: &Buffer,
    k: &Buffer,
    cos_sin: &Buffer,
    positions: &Buffer,
    p: usize,
    head_dim: usize,
) {
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(pipe);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(q), 0, 0);
        enc.setBuffer_offset_atIndex(Some(k), 0, 1);
        enc.setBuffer_offset_atIndex(Some(cos_sin), 0, 2);
        enc.setBuffer_offset_atIndex(Some(positions), 0, 3);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: p,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: head_dim,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
}

// `wavefront_rope_mega`: P co-resident TGs each loop the flattened (token,
// q_head) pairs they own, composing the mittens::rope_rotate_pair atom for Q
// (+ K on owning heads). The rotation is per-element with no reduction, so the
// mega's rotated q_inout/k_inout must be bit-exact vs the whole
// rope_append_*_specialized (run here with sentinel slot_mapping so the paged
// cache write is skipped and only the rotation is exercised). A CPU reference
// gates the atom itself (within a noise floor, since -O3 may FMA-contract the
// rotation that the CPU computes with two roundings).
#[test]
fn wavefront_rope_mega_bit_exact_vs_whole() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // Llama-3.2-1B attention geometry: 32 q-heads, 8 kv-heads, head_dim 64,
    // full rotary (rot_dim == head_dim), paged block size 16.
    let (head_dim, num_q, num_kv, rot_dim, block_size) = (64u32, 32u32, 8u32, 64u32, 16u32);
    let num_tokens = 5u32;
    let max_pos = 64u32;
    let half_dim = (rot_dim / 2) as usize;

    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let mut st = 0x5151_2727u64;
    let mut next = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };

    let q: Vec<u8> = (0..num_tokens * num_q * head_dim)
        .flat_map(|_| bf16le(next() * 2.0))
        .collect();
    let k: Vec<u8> = (0..num_tokens * num_kv * head_dim)
        .flat_map(|_| bf16le(next() * 2.0))
        .collect();
    let v: Vec<u8> = (0..num_tokens * num_kv * head_dim)
        .flat_map(|_| bf16le(next()))
        .collect();
    // cos_sin[pos*rot_dim + d]: d in [0,half) = cos, [half,rot) = sin. Arbitrary
    // deterministic values (the bit-exact proof only needs CPU+GPU to share
    // them — not real RoPE frequencies).
    let cos_sin: Vec<u8> = (0..max_pos * rot_dim)
        .flat_map(|i| {
            let d = (i % rot_dim) as usize;
            let ang = 0.07 * (i as f32);
            bf16le(if d < half_dim { ang.cos() } else { ang.sin() })
        })
        .collect();
    let positions: Vec<u8> = (0..num_tokens)
        .flat_map(|t| ((t * 3 + 1) % max_pos).to_le_bytes())
        .collect();
    let slot_mapping: Vec<u8> = (0..num_tokens)
        .flat_map(|_| 0xFFFF_FFFFu32.to_le_bytes())
        .collect();

    let cos_sin_buf = buffer_from_bytes(&device, &cos_sin);
    let pos_buf = buffer_from_bytes(&device, &positions);
    let slot_buf = buffer_from_bytes(&device, &slot_mapping);
    let v_buf = buffer_from_bytes(&device, &v);
    let kv_k = zeroed_buffer(&device, 64);
    let kv_v = zeroed_buffer(&device, 64);

    let consts_whole = vec![
        ConstantValue::uint(0, head_dim),
        ConstantValue::uint(1, num_q),
        ConstantValue::uint(2, num_kv),
        ConstantValue::uint(3, rot_dim),
        ConstantValue::uint(4, block_size),
    ];
    let mut consts_mega = consts_whole.clone();
    consts_mega.push(ConstantValue::uint(5, num_tokens));

    let whole = cache
        .get_or_build(&PipelineKey::new(
            "rope",
            "rope_append_bf16_specialized",
            consts_whole,
        ))
        .expect("whole rope pipeline");
    let mega = cache
        .get_or_build(&PipelineKey::new(
            "rope",
            "wavefront_rope_mega_bf16",
            consts_mega,
        ))
        .expect("wavefront_rope_mega pipeline");

    // Reference: whole rope_append rotates q/k in place (sentinel slot ⇒ no
    // cache write).
    let q_whole = buffer_from_bytes(&device, &q);
    let k_whole = buffer_from_bytes(&device, &k);
    dispatch_rope_append(
        &device,
        &whole,
        [
            &q_whole,
            &k_whole,
            &v_buf,
            &cos_sin_buf,
            &pos_buf,
            &slot_buf,
            &kv_k,
            &kv_v,
        ],
        (num_tokens as usize, num_q as usize),
        head_dim as usize,
    );
    let q_ref = read_bf16(&q_whole, (num_tokens * num_q * head_dim) as usize);
    let k_ref = read_bf16(&k_whole, (num_tokens * num_kv * head_dim) as usize);
    assert!(
        q_ref.iter().any(|&b| b != 0) && k_ref.iter().any(|&b| b != 0),
        "whole rope produced all zeros"
    );

    // CPU rope reference (noise-floor sanity that the atom rotates correctly).
    let cs: Vec<u16> = cos_sin
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let pos: Vec<u32> = positions
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let rope_cpu = |inp: &[u8], heads: u32| -> Vec<f32> {
        let mut out: Vec<f32> = inp
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        for t in 0..num_tokens as usize {
            let p = pos[t] as usize;
            for h in 0..heads as usize {
                let base = (t * heads as usize + h) * head_dim as usize;
                for d in 0..half_dim {
                    let c = half::bf16::from_bits(cs[p * rot_dim as usize + d]).to_f32();
                    let s = half::bf16::from_bits(cs[p * rot_dim as usize + half_dim + d]).to_f32();
                    let x0 = out[base + d];
                    let x1 = out[base + half_dim + d];
                    out[base + d] = half::bf16::from_f32(x0 * c - x1 * s).to_f32();
                    out[base + half_dim + d] = half::bf16::from_f32(x1 * c + x0 * s).to_f32();
                }
            }
        }
        out
    };
    for (i, &c) in rope_cpu(&q, num_q).iter().enumerate() {
        let g = half::bf16::from_bits(q_ref[i]).to_f32();
        assert!(
            (g - c).abs() <= 0.02 + 0.03 * c.abs(),
            "rope Q[{i}] gpu={g} vs cpu={c} exceeds noise floor"
        );
    }
    for (i, &c) in rope_cpu(&k, num_kv).iter().enumerate() {
        let g = half::bf16::from_bits(k_ref[i]).to_f32();
        assert!(
            (g - c).abs() <= 0.02 + 0.03 * c.abs(),
            "rope K[{i}] gpu={g} vs cpu={c} exceeds noise floor"
        );
    }

    // Mega bit-exact vs whole, P co-resident workers.
    for p in [1usize, 2, 4, 10] {
        let q_mega = buffer_from_bytes(&device, &q);
        let k_mega = buffer_from_bytes(&device, &k);
        dispatch_rope_mega(
            &device,
            &mega,
            &q_mega,
            &k_mega,
            &cos_sin_buf,
            &pos_buf,
            p,
            head_dim as usize,
        );
        let q_m = read_bf16(&q_mega, (num_tokens * num_q * head_dim) as usize);
        let k_m = read_bf16(&k_mega, (num_tokens * num_kv * head_dim) as usize);
        assert_eq!(
            q_m, q_ref,
            "wavefront_rope_mega P={p} Q must be bit-exact vs whole rope_append"
        );
        assert_eq!(
            k_m, k_ref,
            "wavefront_rope_mega P={p} K must be bit-exact vs whole rope_append"
        );
    }
}

// ── silu·mul atom (mittens/silu_mul.h) composition proof ──────────────────

/// Bind out/gate/up at args 0..3 and dispatch `grid` threadgroups of [tg,1,1].
fn dispatch_silu_mul(
    device: &Device,
    pipe: &ComputePipelineState,
    out: &Buffer,
    gate: &Buffer,
    up: &Buffer,
    grid: usize,
    tg: usize,
) {
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(pipe);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(out), 0, 0);
        enc.setBuffer_offset_atIndex(Some(gate), 0, 1);
        enc.setBuffer_offset_atIndex(Some(up), 0, 2);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid,
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

// `wavefront_silu_mul_mega`: P co-resident TGs grid-stride over the elements,
// composing the mittens::silu_mul_impl atom. Bit-exact vs the whole silu_mul
// (both share the GPU exp()); a CPU reference gates the atom within a noise
// floor (Metal exp() vs Rust exp() differ in the last bits).
#[test]
fn wavefront_silu_mul_mega_bit_exact_vs_whole() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    let n = 8192u32; // Llama-3.2-1B intermediate_size (one decode token)
    let tg = 256usize;

    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let mut st = 0xABCD_1234u64;
    let mut next = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0 // ~U(-4, 4)
    };
    let gate: Vec<u8> = (0..n).flat_map(|_| bf16le(next())).collect();
    let up: Vec<u8> = (0..n).flat_map(|_| bf16le(next())).collect();
    let gate_buf = buffer_from_bytes(&device, &gate);
    let up_buf = buffer_from_bytes(&device, &up);

    let constants = vec![ConstantValue::uint(0, n)];

    let whole = cache
        .get_or_build(&PipelineKey::new(
            "silu_mul",
            "silu_mul_bf16",
            constants.clone(),
        ))
        .expect("whole silu_mul pipeline");
    let out_ref = zeroed_buffer(&device, (n * 2) as usize);
    dispatch_silu_mul(
        &device,
        &whole,
        &out_ref,
        &gate_buf,
        &up_buf,
        (n as usize).div_ceil(tg),
        tg,
    );
    let ref_bits = read_bf16(&out_ref, n as usize);
    assert!(
        ref_bits.iter().any(|&b| b != 0),
        "whole silu_mul produced all zeros"
    );

    // CPU silu·mul reference (noise-floor sanity).
    let f = |bytes: &[u8], i: usize| {
        half::bf16::from_bits(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])).to_f32()
    };
    for i in 0..n as usize {
        let g = f(&gate, i);
        let u = f(&up, i);
        let c = (g / (1.0 + (-g).exp())) * u;
        let gpu = half::bf16::from_bits(ref_bits[i]).to_f32();
        assert!(
            (gpu - c).abs() <= 0.02 + 0.03 * c.abs(),
            "silu_mul[{i}] gpu={gpu} vs cpu={c} exceeds noise floor"
        );
    }

    let mega = cache
        .get_or_build(&PipelineKey::new(
            "silu_mul",
            "wavefront_silu_mul_mega_bf16",
            constants,
        ))
        .expect("wavefront_silu_mul_mega pipeline");
    for p in [1usize, 2, 4, 10] {
        let out_mega = zeroed_buffer(&device, (n * 2) as usize);
        dispatch_silu_mul(&device, &mega, &out_mega, &gate_buf, &up_buf, p, tg);
        let mega_bits = read_bf16(&out_mega, n as usize);
        assert_eq!(
            mega_bits, ref_bits,
            "wavefront_silu_mul_mega P={p} must be bit-exact vs whole silu_mul"
        );
    }
}

// ── decode attention atom (mittens/attention.h) composition proof ─────────

/// Bind output/q/seq_used_k/block_table/k_cache/v_cache at args 0..6 and
/// dispatch `grid` threadgroups of [tg,1,1] threads (tg = 1024 = 32 simdgroups).
fn dispatch_attention(
    device: &Device,
    pipe: &ComputePipelineState,
    bufs: [&Buffer; 6],
    grid: (usize, usize),
    tg: usize,
) {
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(pipe);
    for (i, b) in bufs.iter().enumerate() {
        unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, i) };
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: grid.0,
            height: grid.1,
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

// `wavefront_attention_mega`: P co-resident TGs (1024 threads each) loop the
// flattened (seq, q_head) pairs they own, composing the
// mittens::attention_decode_impl atom (full 32-simdgroup combine per pair).
// Two checks:
//   (1) composition: random K/V ⇒ mega bit-exact vs the whole kernel for any P;
//   (2) correctness: a CONSTANT-V cache (every cached V row of a kv-head equal)
//       makes attention's softmax-weighted output equal that V row regardless of
//       the scores — a reference-free check that the atom's softmax + weighted-V
//       + paged indexing are right (catches a botched extraction).
#[test]
fn wavefront_attention_mega_bit_exact_vs_whole() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // Llama-3.2-1B attention geometry; 2 decode sequences, paged block size 16.
    let (head_dim, num_q, num_kv, block_size) = (64usize, 32usize, 8usize, 16usize);
    let batch = 2usize;
    let max_blocks = 4usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let group_ratio = num_q / num_kv;
    let kv_lens = [40usize, 24usize]; // ceil(40/16)=3, ceil(24/16)=2 blocks
    let num_blocks = 5usize; // seq0 -> physical {0,1,2}, seq1 -> {3,4}

    let mut block_table_u32 = vec![0u32; batch * max_blocks];
    block_table_u32[0] = 0;
    block_table_u32[1] = 1;
    block_table_u32[2] = 2;
    block_table_u32[max_blocks] = 3;
    block_table_u32[max_blocks + 1] = 4;
    let block_table: Vec<u8> = block_table_u32
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let seq_used: Vec<u8> = kv_lens
        .iter()
        .flat_map(|&l| (l as u32).to_le_bytes())
        .collect();

    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let mut st = 0x7777_3333u64;
    let mut next = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };

    let q: Vec<u8> = (0..batch * num_q * head_dim)
        .flat_map(|_| bf16le(next()))
        .collect();
    let cache_elems = num_blocks * num_kv * block_size * head_dim;
    let k_cache: Vec<u8> = (0..cache_elems).flat_map(|_| bf16le(next())).collect();
    let v_rand: Vec<u8> = (0..cache_elems).flat_map(|_| bf16le(next())).collect();

    // Constant-V cache: v_cache[blk][kvh][tok][d] = v_const[kvh][d] for all
    // blocks/tokens.
    let v_const: Vec<f32> = (0..num_kv * head_dim).map(|_| next()).collect();
    let mut v_const_cache = vec![0u8; cache_elems * 2];
    for blk in 0..num_blocks {
        for kvh in 0..num_kv {
            for tok in 0..block_size {
                for d in 0..head_dim {
                    let idx = ((blk * num_kv + kvh) * block_size + tok) * head_dim + d;
                    let bytes = bf16le(v_const[kvh * head_dim + d]);
                    v_const_cache[idx * 2] = bytes[0];
                    v_const_cache[idx * 2 + 1] = bytes[1];
                }
            }
        }
    }

    let q_buf = buffer_from_bytes(&device, &q);
    let seq_buf = buffer_from_bytes(&device, &seq_used);
    let bt_buf = buffer_from_bytes(&device, &block_table);
    let k_buf = buffer_from_bytes(&device, &k_cache);
    let v_rand_buf = buffer_from_bytes(&device, &v_rand);
    let v_const_buf = buffer_from_bytes(&device, &v_const_cache);

    let consts = vec![
        ConstantValue::uint(0, head_dim as u32),
        ConstantValue::uint(1, num_q as u32),
        ConstantValue::uint(2, num_kv as u32),
        ConstantValue::float(3, scale),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, max_blocks as u32),
    ];
    let mut consts_mega = consts.clone();
    consts_mega.push(ConstantValue::uint(6, batch as u32));

    let whole = cache
        .get_or_build(&PipelineKey::new(
            "attention",
            "attention_via_cache_v2_bf16_specialized",
            consts.clone(),
        ))
        .expect("whole attention pipeline");
    let mega = cache
        .get_or_build(&PipelineKey::new(
            "attention",
            "wavefront_attention_mega_bf16",
            consts_mega,
        ))
        .expect("wavefront_attention_mega pipeline");

    let out_n = batch * num_q * head_dim;
    let out_bytes = out_n * 2;

    // (1) composition: random V, mega bit-exact vs whole.
    let out_ref = zeroed_buffer(&device, out_bytes);
    dispatch_attention(
        &device,
        &whole,
        [&out_ref, &q_buf, &seq_buf, &bt_buf, &k_buf, &v_rand_buf],
        (batch, num_q),
        1024,
    );
    let ref_bits = read_bf16(&out_ref, out_n);
    assert!(
        ref_bits.iter().any(|&b| b != 0),
        "whole attention produced all zeros"
    );
    for &b in &ref_bits {
        assert!(
            half::bf16::from_bits(b).to_f32().is_finite(),
            "whole attention produced a non-finite output"
        );
    }
    for p in [1usize, 2, 4, 10] {
        let out_mega = zeroed_buffer(&device, out_bytes);
        dispatch_attention(
            &device,
            &mega,
            [&out_mega, &q_buf, &seq_buf, &bt_buf, &k_buf, &v_rand_buf],
            (p, 1),
            1024,
        );
        let mega_bits = read_bf16(&out_mega, out_n);
        assert_eq!(
            mega_bits, ref_bits,
            "wavefront_attention_mega P={p} must be bit-exact vs whole attention_via_cache_v2"
        );
    }

    // (2) correctness: constant-V cache ⇒ output == that kv-head's V vector.
    let out_const = zeroed_buffer(&device, out_bytes);
    dispatch_attention(
        &device,
        &whole,
        [&out_const, &q_buf, &seq_buf, &bt_buf, &k_buf, &v_const_buf],
        (batch, num_q),
        1024,
    );
    let const_bits = read_bf16(&out_const, out_n);
    for seq in 0..batch {
        for qh in 0..num_q {
            let kvh = qh / group_ratio;
            for d in 0..head_dim {
                let got =
                    half::bf16::from_bits(const_bits[(seq * num_q + qh) * head_dim + d]).to_f32();
                let want = v_const[kvh * head_dim + d];
                assert!(
                    (got - want).abs() <= 0.02 + 0.03 * want.abs(),
                    "attention const-V seq={seq} qh={qh} d={d}: got {got} want {want}"
                );
            }
        }
    }
}
