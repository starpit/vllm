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
