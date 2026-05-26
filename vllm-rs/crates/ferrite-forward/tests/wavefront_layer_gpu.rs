// SPDX-License-Identifier: Apache-2.0
//! PD-wavefront — trivial tape-player build-up, bit-exact GPU proofs.
//!
//! Step 2: a single `qmv_fast` atom whose `w/scales/biases/x/y` operands are
//! resolved through the bindless `gpuAddress` table (the trivial player's
//! addressing mechanism, proven in `wavefront_addr_probe.rs`) — must be
//! bit-exact vs the normal whole `affine_qmv_fast`. The operand buffers are
//! never `setBuffer`-bound; only the address table is. Each operand is made
//! resident with `useResource` (production reuses the allocator's shared
//! `MetalResidencySet`).
//!
//! Run: `cargo test -p ferrite-forward -F metal --test wavefront_layer_gpu`.
#![cfg(feature = "metal")]

use objc2::msg_send;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLResourceUsage, MTLSize,
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

/// Normal whole `affine_qmv_fast` (batched=0): w/s/b/x/y bound at 0..5, grid
/// (1, ceil(N/8), 1), TG [32,2,1]. The trusted reference.
#[allow(clippy::too_many_arguments)]
fn dispatch_qmv_whole(
    device: &Device,
    pipe: &ComputePipelineState,
    w: &Buffer,
    s: &Buffer,
    b: &Buffer,
    x: &Buffer,
    y: &Buffer,
    n: u32,
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
            width: 1,
            height: (n as usize).div_ceil(8),
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

/// Make `buf` resident for the encoder (it is reached only via its raw
/// gpuAddress, never `setBuffer`-bound).
fn use_resource(
    enc: &objc2::runtime::ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf: &Buffer,
    usage: MTLResourceUsage,
) {
    unsafe {
        let _: () = msg_send![enc, useResource: &**buf, usage: usage];
    }
}

/// The bindless qmv: only the `addrs` table is bound; w/s/b/x/y are reached via
/// their gpuAddresses. Same grid/TG as the whole kernel.
#[allow(clippy::too_many_arguments)]
fn dispatch_qmv_bindless(
    device: &Device,
    pipe: &ComputePipelineState,
    addrs: &Buffer,
    w: &Buffer,
    s: &Buffer,
    b: &Buffer,
    x: &Buffer,
    y: &Buffer,
    n: u32,
) {
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(pipe);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(addrs), 0, 0);
    }
    use_resource(&enc, w, MTLResourceUsage::Read);
    use_resource(&enc, s, MTLResourceUsage::Read);
    use_resource(&enc, b, MTLResourceUsage::Read);
    use_resource(&enc, x, MTLResourceUsage::Read);
    use_resource(&enc, y, MTLResourceUsage::Read | MTLResourceUsage::Write);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: (n as usize).div_ceil(8),
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

#[test]
fn wavefront_qmv_bindless_bit_exact_vs_whole() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // bf16 act / f16 scale, gs=64, 4-bit, M=1, N % 8 == 0 (fast variant).
    let (n, k, gs, bits) = (96u32, 512u32, 64u32, 4u32);
    let n_bytes_packed = (n * k / 2) as usize;
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

    let constants = vec![
        ConstantValue::int(0, k as i32),
        ConstantValue::int(1, n as i32),
    ];

    // Reference: trusted whole affine_qmv_fast (batched=0).
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
    let whole = cache
        .get_or_build(&PipelineKey::new(
            "quantized_qmv",
            fast_sym,
            constants.clone(),
        ))
        .expect("whole qmv pipeline");
    let y_ref = zeroed_buffer(&device, (n * 2) as usize);
    dispatch_qmv_whole(&device, &whole, &w_buf, &s_buf, &b_buf, &x_buf, &y_ref, n);
    let ref_bits = read_bf16(&y_ref, n as usize);
    assert!(ref_bits.iter().any(|&b| b != 0), "reference qmv all zeros");

    // Bindless: only the gpuAddress table is bound; operands reached via address.
    let bindless = cache
        .get_or_build(&PipelineKey::new(
            "wavefront_layer",
            "wavefront_qmv_bindless_bf16_s_f16_gs_64_b_4",
            constants,
        ))
        .expect("wavefront_qmv_bindless pipeline");
    let y_out = zeroed_buffer(&device, (n * 2) as usize);
    let addr_u64 = |b: &Buffer| -> u64 { b.gpuAddress() };
    let addrs_bytes: Vec<u8> = [&w_buf, &s_buf, &b_buf, &x_buf, &y_out]
        .iter()
        .flat_map(|b| addr_u64(b).to_le_bytes())
        .collect();
    let addrs = buffer_from_bytes(&device, &addrs_bytes);
    dispatch_qmv_bindless(
        &device, &bindless, &addrs, &w_buf, &s_buf, &b_buf, &x_buf, &y_out, n,
    );
    let out_bits = read_bf16(&y_out, n as usize);
    assert_eq!(
        out_bits, ref_bits,
        "wavefront_qmv_bindless must be bit-exact vs whole affine_qmv_fast \
         (a mismatch = the gpuAddress operand resolution is wrong)"
    );
}

// ── step 3: the trivial interpret loop + shape-class switch ───────────────

/// Build a 4-bit affine qmv problem (random packed w + f16 scales/biases +
/// bf16 x) for `[n, k]`, returning the four input buffers.
fn make_qmv(
    device: &Device,
    n: u32,
    k: u32,
    gs: u32,
    seed: u64,
) -> (Buffer, Buffer, Buffer, Buffer) {
    let mut s = seed;
    let mut byte = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        (s >> 33) as u8
    };
    let f16le = |v: f32| half::f16::from_f32(v).to_bits().to_le_bytes();
    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let packed: Vec<u8> = (0..(n * k / 2) as usize).map(|_| byte()).collect();
    let n_groups = (n * k / gs) as usize;
    let scales: Vec<u8> = (0..n_groups)
        .flat_map(|i| f16le(0.01 + 0.0001 * (i % 17) as f32))
        .collect();
    let biases: Vec<u8> = (0..n_groups)
        .flat_map(|i| f16le(-0.05 + 0.001 * (i % 13) as f32))
        .collect();
    let x: Vec<u8> = (0..k)
        .flat_map(|i| bf16le(((i % 7) as f32 - 3.0) * 0.1))
        .collect();
    (
        buffer_from_bytes(device, &packed),
        buffer_from_bytes(device, &scales),
        buffer_from_bytes(device, &biases),
        buffer_from_bytes(device, &x),
    )
}

/// Whole `affine_qmv_fast` reference output for one problem.
#[allow(clippy::too_many_arguments)]
fn qmv_ref(
    device: &Device,
    cache: &SpecializedPipelineCache,
    w: &Buffer,
    s: &Buffer,
    b: &Buffer,
    x: &Buffer,
    n: u32,
    k: u32,
    gs: u32,
) -> Vec<u16> {
    let sym: &'static str = Box::leak(
        qmv_kernel_static_name(QmvKernel::Fast, DequantDtype::Bf16, ScaleDtype::F16, 4, gs)
            .to_string()
            .into_boxed_str(),
    );
    let pipe = cache
        .get_or_build(&PipelineKey::new(
            "quantized_qmv",
            sym,
            vec![
                ConstantValue::int(0, k as i32),
                ConstantValue::int(1, n as i32),
            ],
        ))
        .expect("qmv ref pipeline");
    let y = zeroed_buffer(device, (n * 2) as usize);
    dispatch_qmv_whole(device, &pipe, w, s, b, x, &y, n);
    read_bf16(&y, n as usize)
}

/// The interpret loop replays two independent qmv subtiles (a 2-instruction
/// tape) in one persistent worker (P=1), fixed 1024-thread TG, with operands
/// resolved through the gpuAddress table. Each must be bit-exact vs the whole
/// `affine_qmv_fast` — proving the tape loop + shape-class switch + the
/// fixed-TG qmv arm + bindless operands, with zero decisions in the player.
#[test]
fn wavefront_player_two_qmv_bit_exact() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");
    let gs = 64u32;
    let (k0, n0) = (512u32, 96u32);
    let (k1, n1) = (512u32, 64u32);

    let (w0, s0, b0, x0) = make_qmv(&device, n0, k0, gs, 0x1111_2222);
    let (w1, s1, b1, x1) = make_qmv(&device, n1, k1, gs, 0x3333_4444);
    let ref0 = qmv_ref(&device, &cache, &w0, &s0, &b0, &x0, n0, k0, gs);
    let ref1 = qmv_ref(&device, &cache, &w1, &s1, &b1, &x1, n1, k1, gs);
    assert!(ref0.iter().any(|&b| b != 0) && ref1.iter().any(|&b| b != 0));

    let y0 = zeroed_buffer(&device, (n0 * 2) as usize);
    let y1 = zeroed_buffer(&device, (n1 * 2) as usize);

    // operand table: subtile 0 = [w0,s0,b0,x0,y0], subtile 1 = [w1,s1,b1,x1,y1].
    let operand_bufs: Vec<&Buffer> = vec![&w0, &s0, &b0, &x0, &y0, &w1, &s1, &b1, &x1, &y1];
    let operands_bytes: Vec<u8> = operand_bufs
        .iter()
        .flat_map(|b| b.gpuAddress().to_le_bytes())
        .collect();
    let operands = buffer_from_bytes(&device, &operands_bytes);

    // shapes[sc] = 8-u32 record (op_kind=QMV(0), K, N, ...).
    let shapes_u32: Vec<u32> = vec![
        0, k0, n0, 0, 0, 0, 0, 0, // class 0
        0, k1, n1, 0, 0, 0, 0, 0, // class 1
    ];
    let shapes_bytes: Vec<u8> = shapes_u32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shapes = buffer_from_bytes(&device, &shapes_bytes);

    // tape[pc] = (opcode=Compute(0), shape_class, operand_base, flag).
    let tape_u32: Vec<u32> = vec![0, 0, 0, 0, /*instr 1*/ 0, 1, 5, 0];
    let tape_bytes: Vec<u8> = tape_u32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tape = buffer_from_bytes(&device, &tape_bytes);

    // P=1: the single worker runs the whole tape [0, 2).
    let offsets_bytes: Vec<u8> = [0u32, 2u32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let tape_offsets = buffer_from_bytes(&device, &offsets_bytes);

    let player = cache
        .get_or_build(&PipelineKey::new(
            "wavefront_layer",
            "wavefront_player_bf16_s_f16_gs_64_b_4",
            vec![],
        ))
        .expect("wavefront_player pipeline");

    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&player);
    let flags = zeroed_buffer(&device, 4); // unused by this tape (no Signal/Wait)
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&tape), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&shapes), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&operands), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&tape_offsets), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&flags), 0, 4);
    }
    for buf in &operand_bufs {
        use_resource(&enc, buf, MTLResourceUsage::Read | MTLResourceUsage::Write);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
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

    assert_eq!(
        read_bf16(&y0, n0 as usize),
        ref0,
        "player subtile 0 must be bit-exact vs whole qmv"
    );
    assert_eq!(
        read_bf16(&y1, n1 as usize),
        ref1,
        "player subtile 1 must be bit-exact vs whole qmv"
    );
}

/// One qmv `[N, K]` N-block-tiled into `P` subtiles of `nb` rows, each subtile
/// assigned to its own co-resident worker (P=10 TGs, one tape instruction
/// each), operands offset per block via the gpuAddress table. The assembled
/// output must be bit-exact vs the whole `affine_qmv_fast` — proving per-worker
/// tapes + the persistent multi-TG spread + the N-block offset linchpin through
/// bindless operands. This is the bandwidth case (independent blocks, edge-cut
/// 0, no cross-worker handoff) the scheduler targets.
#[test]
fn wavefront_player_nblock_spread_across_workers() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // N=240 split into P=10 blocks of nb=24 rows (nb % 8 == 0 ⇒ fast variant
    // clean; 24/8 = 3 groups/block). K=512, gs=64, 4-bit, bf16 act / f16 scale.
    let (n, k, gs) = (240u32, 512u32, 64u32);
    let p = 10u32;
    let nb = n / p; // 24
    assert_eq!(n % p, 0);
    assert_eq!(nb % 8, 0);

    let (w, s, b, x) = make_qmv(&device, n, k, gs, 0xABCD_0001);
    let y_ref = qmv_ref(&device, &cache, &w, &s, &b, &x, n, k, gs);
    assert!(y_ref.iter().any(|&v| v != 0));

    let y = zeroed_buffer(&device, (n * 2) as usize);

    // Per-row byte strides for the block offsets (4-bit packed w, f16 s/b, bf16 y).
    let w_row = (k / 2) as u64; // K/2 bytes/row (2 nibbles/byte)
    let sb_row = (k / gs) as u64 * 2; // (K/gs) f16 groups/row
    let y_row = 2u64; // bf16

    // operand table: block i → [w+off, s+off, b+off, x (whole), y+off].
    let mut operands_bytes: Vec<u8> = Vec::new();
    for i in 0..p as u64 {
        let r = i * nb as u64;
        let addrs = [
            w.gpuAddress() + r * w_row,
            s.gpuAddress() + r * sb_row,
            b.gpuAddress() + r * sb_row,
            x.gpuAddress(),
            y.gpuAddress() + r * y_row,
        ];
        for a in addrs {
            operands_bytes.extend_from_slice(&a.to_le_bytes());
        }
    }
    let operands = buffer_from_bytes(&device, &operands_bytes);

    // One shape class shared by every block: (QMV, K, nb, ...).
    let shapes_bytes: Vec<u8> = [0u32, k, nb, 0, 0, 0, 0, 0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let shapes = buffer_from_bytes(&device, &shapes_bytes);

    // tape: P instructions, block i at operand_base 5*i. tape_offsets gives
    // worker i exactly instruction i.
    let mut tape_u32: Vec<u32> = Vec::new();
    for i in 0..p {
        tape_u32.extend_from_slice(&[
            0,     /*Compute*/
            0,     /*shape 0*/
            5 * i, /*operand_base*/
            0,
        ]);
    }
    let tape_bytes: Vec<u8> = tape_u32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tape = buffer_from_bytes(&device, &tape_bytes);
    let offsets_u32: Vec<u32> = (0..=p).collect(); // [0,1,...,P]
    let offsets_bytes: Vec<u8> = offsets_u32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tape_offsets = buffer_from_bytes(&device, &offsets_bytes);

    let player = cache
        .get_or_build(&PipelineKey::new(
            "wavefront_layer",
            "wavefront_player_bf16_s_f16_gs_64_b_4",
            vec![],
        ))
        .expect("wavefront_player pipeline");

    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&player);
    let flags = zeroed_buffer(&device, 4); // unused by this tape (no Signal/Wait)
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&tape), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&shapes), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&operands), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&tape_offsets), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&flags), 0, 4);
    }
    use_resource(&enc, &w, MTLResourceUsage::Read);
    use_resource(&enc, &s, MTLResourceUsage::Read);
    use_resource(&enc, &b, MTLResourceUsage::Read);
    use_resource(&enc, &x, MTLResourceUsage::Read);
    use_resource(&enc, &y, MTLResourceUsage::Read | MTLResourceUsage::Write);
    // P co-resident workers, fixed 1024-thread (32-simdgroup) TG.
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: p as usize,
            height: 1,
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

    assert_eq!(
        read_bf16(&y, n as usize),
        y_ref,
        "N-block-spread player output must be bit-exact vs whole affine_qmv_fast"
    );
}

// ── step 4b: cross-worker handoff (Signal/Wait + atomic Publish/Acquire) ──

/// Whole `affine_qmv_fast` dispatched into a caller-provided `y` buffer.
#[allow(clippy::too_many_arguments)]
fn qmv_whole_into(
    device: &Device,
    cache: &SpecializedPipelineCache,
    w: &Buffer,
    s: &Buffer,
    b: &Buffer,
    x: &Buffer,
    y: &Buffer,
    n: u32,
    k: u32,
    gs: u32,
) {
    let sym: &'static str = Box::leak(
        qmv_kernel_static_name(QmvKernel::Fast, DequantDtype::Bf16, ScaleDtype::F16, 4, gs)
            .to_string()
            .into_boxed_str(),
    );
    let pipe = cache
        .get_or_build(&PipelineKey::new(
            "quantized_qmv",
            sym,
            vec![
                ConstantValue::int(0, k as i32),
                ConstantValue::int(1, n as i32),
            ],
        ))
        .expect("qmv whole pipeline");
    dispatch_qmv_whole(device, &pipe, w, s, b, x, y, n);
}

fn zero_buf(buf: &Buffer, len: usize) {
    unsafe { std::ptr::write_bytes(buf.contents().as_ptr() as *mut u8, 0, len) }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A2 reproduced through the generic trivial player: a 2-stage chain
/// y1 = qmv(W0, x); y2 = qmv(W1, y1), where stage-0 is N-block-spread across P
/// workers and stage-1 on EVERY worker joins on all of y1 — the all-to-all
/// cross-worker handoff. The player runs it as: Compute(stage0 block) →
/// Publish(block → coherent y1c) → Signal → Wait(all P) → Acquire(y1c → private
/// copy) → Compute(stage1 block). Must be bit-exact vs two sequential whole
/// qmvs, every retry (the atomic u32-packed handoff + data-before-flag ordering
/// is the one correctness-critical thing). All sync/handoff is DATA in the tape;
/// the player makes zero decisions.
#[test]
fn wavefront_player_two_stage_handoff_bit_exact() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");
    let gs = 64u32;
    let k0 = 512u32; // K must be a multiple of the fast variant's block_size (512)

    let player = cache
        .get_or_build(&PipelineKey::new(
            "wavefront_layer",
            "wavefront_player_bf16_s_f16_gs_64_b_4",
            vec![],
        ))
        .expect("wavefront_player pipeline");

    for p in [1u32, 2, 4, 10] {
        // N0 = K1 must be a multiple of 512 (stage-1 K) AND of P*8 (each stage-0
        // block nb0 = N0/P must be %8 for the fast variant). lcm(512, P*8).
        let n0 = {
            let a = 512u32;
            let bb = p * 8;
            a / gcd(a, bb) * bb
        };
        let nb0 = n0 / p;
        let k1 = n0;
        let nb1 = 64u32; // stage-1 block width (%8); N1 = nb1 * P
        let n1 = nb1 * p;

        // Immutable operands (built once per P).
        let (w0, s0, b0, x) = make_qmv(&device, n0, k0, gs, 0xA200_0000 + p as u64);
        let (w1, s1, b1, _x1) = make_qmv(&device, n1, k1, gs, 0xB100_0000 + p as u64);

        // Reference: two sequential whole qmvs.
        let y1_ref = zeroed_buffer(&device, (n0 * 2) as usize);
        qmv_whole_into(&device, &cache, &w0, &s0, &b0, &x, &y1_ref, n0, k0, gs);
        let y2_ref_buf = zeroed_buffer(&device, (n1 * 2) as usize);
        qmv_whole_into(
            &device,
            &cache,
            &w1,
            &s1,
            &b1,
            &y1_ref,
            &y2_ref_buf,
            n1,
            k1,
            gs,
        );
        let y2_ref = read_bf16(&y2_ref_buf, n1 as usize);
        assert!(y2_ref.iter().any(|&v| v != 0), "P={p} reference all zeros");

        // Mutable buffers (zeroed in place each retry; addresses stay stable so
        // the operand table is built once).
        let y1 = zeroed_buffer(&device, (n0 * 2) as usize);
        let y2 = zeroed_buffer(&device, (n1 * 2) as usize);
        let y1c = zeroed_buffer(&device, (n0 / 2 * 4) as usize); // coherent u32 handoff
        let my_copy = zeroed_buffer(&device, (p * n0 * 2) as usize); // [P, N0] private bf16
        let flags = zeroed_buffer(&device, (p * 4) as usize); // P atomic_uint

        // Per-worker byte strides.
        let w0_row = (k0 / 2) as u64;
        let sb0_row = (k0 / gs) as u64 * 2;
        let w1_row = (k1 / 2) as u64;
        let sb1_row = (k1 / gs) as u64 * 2;

        // operand table: 14 entries per worker (stage0 5, publish 2, acquire 2,
        // stage1 5). Per-worker positioning rides in the addresses.
        let mut operands_bytes: Vec<u8> = Vec::new();
        let mut push = |a: u64| operands_bytes.extend_from_slice(&a.to_le_bytes());
        for i in 0..p as u64 {
            let r0 = i * nb0 as u64; // stage-0 first row
            let r1 = i * nb1 as u64; // stage-1 first row
            let y1_off = r0 * 2;
            let y1c_off = i * (nb0 as u64 / 2) * 4;
            let mc_off = i * n0 as u64 * 2;
            // stage0 [w0,s0,b0,x,y1]
            push(w0.gpuAddress() + r0 * w0_row);
            push(s0.gpuAddress() + r0 * sb0_row);
            push(b0.gpuAddress() + r0 * sb0_row);
            push(x.gpuAddress());
            push(y1.gpuAddress() + y1_off);
            // publish [y1c+, y1+]
            push(y1c.gpuAddress() + y1c_off);
            push(y1.gpuAddress() + y1_off);
            // acquire [my_copy+, y1c(whole)]
            push(my_copy.gpuAddress() + mc_off);
            push(y1c.gpuAddress());
            // stage1 [w1,s1,b1,my_copy(whole worker copy),y2]
            push(w1.gpuAddress() + r1 * w1_row);
            push(s1.gpuAddress() + r1 * sb1_row);
            push(b1.gpuAddress() + r1 * sb1_row);
            push(my_copy.gpuAddress() + mc_off);
            push(y2.gpuAddress() + r1 * 2);
        }
        let operands = buffer_from_bytes(&device, &operands_bytes);

        // 4 shape classes (8-u32 records): QMV stage0, PUBLISH, ACQUIRE, QMV stage1.
        let shapes_u32: Vec<u32> = vec![
            0,
            k0,
            nb0,
            0,
            0,
            0,
            0,
            0, // 0: QMV stage0
            1,
            0,
            nb0 / 2,
            0,
            0,
            0,
            0,
            0, // 1: PUBLISH pair0=0 n_pairs=nb0/2
            2,
            n0 / 2,
            0,
            0,
            0,
            0,
            0,
            0, // 2: ACQUIRE n_pairs=N0/2
            0,
            k1,
            nb1,
            0,
            0,
            0,
            0,
            0, // 3: QMV stage1
        ];
        let shapes = buffer_from_bytes(&device, &bytes_of_u32(&shapes_u32));

        // tape: per worker [stage0, publish, signal, wait*P, acquire, stage1].
        let mut tape_u32: Vec<u32> = Vec::new();
        for i in 0..p {
            let base = 14 * i;
            tape_u32.extend_from_slice(&[0, 0, base, 0]); // Compute QMV stage0
            tape_u32.extend_from_slice(&[0, 1, base + 5, 0]); // Compute PUBLISH
            tape_u32.extend_from_slice(&[1, 0, 0, i]); // Signal flag i
            for j in 0..p {
                tape_u32.extend_from_slice(&[2, 0, 0, j]); // Wait flag j
            }
            tape_u32.extend_from_slice(&[0, 2, base + 7, 0]); // Compute ACQUIRE
            tape_u32.extend_from_slice(&[0, 3, base + 9, 0]); // Compute QMV stage1
        }
        let tape = buffer_from_bytes(&device, &bytes_of_u32(&tape_u32));
        let per_worker = 5 + p;
        let offsets_u32: Vec<u32> = (0..=p).map(|i| i * per_worker).collect();
        let tape_offsets = buffer_from_bytes(&device, &bytes_of_u32(&offsets_u32));

        // The cross-TG handoff is timing-dependent — retry so a pass isn't luck.
        for iter in 0..50 {
            zero_buf(&y1, (n0 * 2) as usize);
            zero_buf(&y2, (n1 * 2) as usize);
            zero_buf(&y1c, (n0 / 2 * 4) as usize);
            zero_buf(&my_copy, (p * n0 * 2) as usize);
            zero_buf(&flags, (p * 4) as usize);

            let queue = device.newCommandQueue().expect("queue");
            let cb = queue.commandBuffer().expect("cb");
            let enc = cb.computeCommandEncoder().expect("enc");
            enc.setComputePipelineState(&player);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&tape), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&shapes), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&operands), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&tape_offsets), 0, 3);
                enc.setBuffer_offset_atIndex(Some(&flags), 0, 4);
            }
            for buf in [&w0, &s0, &b0, &x, &w1, &s1, &b1] {
                use_resource(&enc, buf, MTLResourceUsage::Read);
            }
            for buf in [&y1, &y2, &y1c, &my_copy] {
                use_resource(&enc, buf, MTLResourceUsage::Read | MTLResourceUsage::Write);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: p as usize,
                    height: 1,
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

            assert_eq!(
                read_bf16(&y2, n1 as usize),
                y2_ref,
                "two-stage handoff P={p} iter={iter} must be bit-exact vs sequential whole qmvs \
                 (a mismatch = a Signal/Wait or atomic Publish/Acquire ordering bug)"
            );
        }
    }
}

fn bytes_of_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_of_u64(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

// ── rmsnorm shape-class arm: single subtile through the player ────────────

/// The player runs one rmsnorm subtile (op_kind 3) on the single decode row in
/// the fixed 1024-thread TG — the whole-TG reduction reconciliation — and must
/// be bit-exact vs the whole `rmsnorm_*_specialized`. Operands [out, in, weight]
/// via the gpuAddress table; hidden + eps-bits ride in the shape descriptor.
#[test]
fn wavefront_player_rmsnorm_bit_exact() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // Llama-3.2-1B hidden=2048 ⇒ tg_size=1024 (power-of-two ⇒ exact tree); the
    // player launches 1024 threads and passes tg_size=1024, matching the whole
    // kernel's reduction order.
    let (hidden, eps) = (2048u32, 1e-5f32);
    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let f16le = |v: f32| half::f16::from_f32(v).to_bits().to_le_bytes();
    let mut st = 0xBEEF_F00Du64;
    let mut next = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0
    };
    let input: Vec<u8> = (0..hidden).flat_map(|_| bf16le(next())).collect();
    let weight: Vec<u8> = (0..hidden)
        .flat_map(|i| f16le(0.5 + 0.01 * (i % 7) as f32))
        .collect();
    let in_buf = buffer_from_bytes(&device, &input);
    let w_buf = buffer_from_bytes(&device, &weight);

    // Reference: whole rmsnorm_bf16_s_f16_specialized, grid (1,1,1), TG [1024,1,1].
    let consts = vec![
        ConstantValue::uint(0, 1),
        ConstantValue::uint(1, hidden),
        ConstantValue::float(2, eps),
    ];
    let whole = cache
        .get_or_build(&PipelineKey::new(
            "rmsnorm",
            "rmsnorm_bf16_s_f16_specialized",
            consts,
        ))
        .expect("whole rmsnorm pipeline");
    let y_ref = zeroed_buffer(&device, (hidden * 2) as usize);
    {
        let queue = device.newCommandQueue().expect("queue");
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&whole);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&y_ref), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&in_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&w_buf), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
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
    }
    let ref_bits = read_bf16(&y_ref, hidden as usize);
    assert!(ref_bits.iter().any(|&b| b != 0), "whole rmsnorm all zeros");

    // Player: one rmsnorm subtile.
    let y_out = zeroed_buffer(&device, (hidden * 2) as usize);
    let operand_bufs: Vec<&Buffer> = vec![&y_out, &in_buf, &w_buf];
    let operands_bytes: Vec<u8> = operand_bufs
        .iter()
        .flat_map(|b| b.gpuAddress().to_le_bytes())
        .collect();
    let operands = buffer_from_bytes(&device, &operands_bytes);
    // shape (RMSNORM=3, hidden, eps_bits, ...).
    let shapes = buffer_from_bytes(
        &device,
        &bytes_of_u32(&[3, hidden, eps.to_bits(), 0, 0, 0, 0, 0]),
    );
    let tape = buffer_from_bytes(&device, &bytes_of_u32(&[0, 0, 0, 0])); // Compute, shape 0, base 0
    let tape_offsets = buffer_from_bytes(&device, &bytes_of_u32(&[0, 1]));
    let flags = zeroed_buffer(&device, 4);

    let player = cache
        .get_or_build(&PipelineKey::new(
            "wavefront_layer",
            "wavefront_player_bf16_s_f16_gs_64_b_4",
            vec![],
        ))
        .expect("wavefront_player pipeline");
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&player);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&tape), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&shapes), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&operands), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&tape_offsets), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&flags), 0, 4);
    }
    use_resource(
        &enc,
        &y_out,
        MTLResourceUsage::Read | MTLResourceUsage::Write,
    );
    use_resource(&enc, &in_buf, MTLResourceUsage::Read);
    use_resource(&enc, &w_buf, MTLResourceUsage::Read);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
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

    assert_eq!(
        read_bf16(&y_out, hidden as usize),
        ref_bits,
        "player rmsnorm arm must be bit-exact vs whole rmsnorm_specialized"
    );
}

/// Run the player on a single-instruction tape (one Compute subtile, P=1) and
/// return the output buffer's bf16 bits. `operand_bufs` are made resident; the
/// shape descriptor and operand order are the arm's contract.
#[allow(clippy::too_many_arguments)]
fn run_player_single(
    device: &Device,
    cache: &SpecializedPipelineCache,
    shape: [u32; 8],
    operand_bufs: &[&Buffer],
    out_idx: usize,
    out_n: usize,
    tg: usize,
) -> Vec<u16> {
    let operands_bytes: Vec<u8> = operand_bufs
        .iter()
        .flat_map(|b| b.gpuAddress().to_le_bytes())
        .collect();
    let operands = buffer_from_bytes(device, &operands_bytes);
    let shapes = buffer_from_bytes(device, &bytes_of_u32(&shape));
    let tape = buffer_from_bytes(device, &bytes_of_u32(&[0, 0, 0, 0])); // Compute, shape 0, base 0
    let tape_offsets = buffer_from_bytes(device, &bytes_of_u32(&[0, 1]));
    let flags = zeroed_buffer(device, 4);
    let player = cache
        .get_or_build(&PipelineKey::new(
            "wavefront_layer",
            "wavefront_player_bf16_s_f16_gs_64_b_4",
            vec![],
        ))
        .expect("wavefront_player pipeline");
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&player);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&tape), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&shapes), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&operands), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&tape_offsets), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&flags), 0, 4);
    }
    for buf in operand_bufs {
        use_resource(&enc, buf, MTLResourceUsage::Read | MTLResourceUsage::Write);
    }
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
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
    read_bf16(operand_bufs[out_idx], out_n)
}

/// silu·mul arm bit-exact vs the whole `silu_mul_bf16` (per-element pure ⇒ the
/// player's 1024-thread grid-stride matches the whole kernel's 1-thread/element).
#[test]
fn wavefront_player_silu_mul_bit_exact() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");
    let n = 8192u32; // Llama-3.2-1B intermediate_size

    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let mut st = 0xABCD_1234u64;
    let mut next = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
    };
    let gate: Vec<u8> = (0..n).flat_map(|_| bf16le(next())).collect();
    let up: Vec<u8> = (0..n).flat_map(|_| bf16le(next())).collect();
    let gate_buf = buffer_from_bytes(&device, &gate);
    let up_buf = buffer_from_bytes(&device, &up);

    // Reference: whole silu_mul_bf16.
    let whole = cache
        .get_or_build(&PipelineKey::new(
            "silu_mul",
            "silu_mul_bf16",
            vec![ConstantValue::uint(0, n)],
        ))
        .expect("whole silu_mul pipeline");
    let ref_out = zeroed_buffer(&device, (n * 2) as usize);
    {
        let queue = device.newCommandQueue().expect("queue");
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&whole);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&ref_out), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&gate_buf), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&up_buf), 0, 2);
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (n as usize).div_ceil(256),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }
    let ref_bits = read_bf16(&ref_out, n as usize);
    assert!(ref_bits.iter().any(|&b| b != 0), "whole silu_mul all zeros");

    // Player: operands [out, gate, up], shape (SILU_MUL=4, n, _, _).
    let out = zeroed_buffer(&device, (n * 2) as usize);
    let got = run_player_single(
        &device,
        &cache,
        [4, n, 0, 0, 0, 0, 0, 0],
        &[&out, &gate_buf, &up_buf],
        0,
        n as usize,
        1024,
    );
    assert_eq!(
        got, ref_bits,
        "player silu_mul arm must be bit-exact vs whole silu_mul"
    );
}

/// rope arm bit-exact vs the whole `rope_append_bf16_specialized` rotating Q in
/// place (same `rope_rotate_pair` atom, same cos/sin). The player slices cos/sin
/// out of the cos_sin buffer at the token position via the operand addresses.
#[test]
fn wavefront_player_rope_bit_exact() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // Llama-3.2-1B: head_dim 64, 32 q-heads, full rotary; one decode token.
    let (head_dim, num_q, num_kv, rot_dim, block_size) = (64u32, 32u32, 8u32, 64u32, 16u32);
    let half = (rot_dim / 2) as usize;
    let pos = 7u32;
    let max_pos = 64u32;

    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let mut st = 0x5151_2727u64;
    let mut next = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let q: Vec<u8> = (0..num_q * head_dim)
        .flat_map(|_| bf16le(next() * 2.0))
        .collect();
    let k: Vec<u8> = (0..num_kv * head_dim)
        .flat_map(|_| bf16le(next() * 2.0))
        .collect();
    let v: Vec<u8> = (0..num_kv * head_dim)
        .flat_map(|_| bf16le(next()))
        .collect();
    let cos_sin: Vec<u8> = (0..max_pos * rot_dim)
        .flat_map(|i| {
            let d = (i % rot_dim) as usize;
            let ang = 0.07 * (i as f32);
            bf16le(if d < half { ang.cos() } else { ang.sin() })
        })
        .collect();
    let positions: Vec<u8> = pos.to_le_bytes().to_vec();
    let slot_mapping: Vec<u8> = 0xFFFF_FFFFu32.to_le_bytes().to_vec(); // sentinel ⇒ no cache write
    let cos_sin_buf = buffer_from_bytes(&device, &cos_sin);

    // Reference: whole rope_append rotates q (and k) in place.
    let whole = cache
        .get_or_build(&PipelineKey::new(
            "rope",
            "rope_append_bf16_specialized",
            vec![
                ConstantValue::uint(0, head_dim),
                ConstantValue::uint(1, num_q),
                ConstantValue::uint(2, num_kv),
                ConstantValue::uint(3, rot_dim),
                ConstantValue::uint(4, block_size),
            ],
        ))
        .expect("whole rope pipeline");
    let q_ref_buf = buffer_from_bytes(&device, &q);
    {
        let k_buf = buffer_from_bytes(&device, &k);
        let v_buf = buffer_from_bytes(&device, &v);
        let pos_buf = buffer_from_bytes(&device, &positions);
        let slot_buf = buffer_from_bytes(&device, &slot_mapping);
        let kv_k = zeroed_buffer(&device, 64);
        let kv_v = zeroed_buffer(&device, 64);
        let queue = device.newCommandQueue().expect("queue");
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&whole);
        let bufs = [
            &q_ref_buf,
            &k_buf,
            &v_buf,
            &cos_sin_buf,
            &pos_buf,
            &slot_buf,
            &kv_k,
            &kv_v,
        ];
        for (i, b) in bufs.iter().enumerate() {
            unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, i) };
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: num_q as usize,
                depth: 1,
            },
            MTLSize {
                width: head_dim as usize,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }
    let q_ref = read_bf16(&q_ref_buf, (num_q * head_dim) as usize);
    assert!(
        q_ref.iter().any(|&b| b != 0),
        "whole rope produced all zeros"
    );

    // Player: operands [q, cos, sin] where cos/sin slice cos_sin at the token
    // position. shape (ROPE=5, head_dim, num_q, _).
    let q_buf = buffer_from_bytes(&device, &q);
    let cos_off = (pos * rot_dim) as u64 * 2; // bf16 bytes
    let sin_off = (pos * rot_dim + half as u32) as u64 * 2;
    // Build operands with explicit per-operand offsets (cos/sin into cos_sin).
    let operands_bytes: Vec<u8> = [
        q_buf.gpuAddress(),
        cos_sin_buf.gpuAddress() + cos_off,
        cos_sin_buf.gpuAddress() + sin_off,
    ]
    .iter()
    .flat_map(|a| a.to_le_bytes())
    .collect();
    let operands = buffer_from_bytes(&device, &operands_bytes);
    let shapes = buffer_from_bytes(&device, &bytes_of_u32(&[5, head_dim, num_q, 0, 0, 0, 0, 0]));
    let tape = buffer_from_bytes(&device, &bytes_of_u32(&[0, 0, 0, 0]));
    let tape_offsets = buffer_from_bytes(&device, &bytes_of_u32(&[0, 1]));
    let flags = zeroed_buffer(&device, 4);
    let player = cache
        .get_or_build(&PipelineKey::new(
            "wavefront_layer",
            "wavefront_player_bf16_s_f16_gs_64_b_4",
            vec![],
        ))
        .expect("player pipeline");
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&player);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&tape), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&shapes), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&operands), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&tape_offsets), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&flags), 0, 4);
    }
    use_resource(
        &enc,
        &q_buf,
        MTLResourceUsage::Read | MTLResourceUsage::Write,
    );
    use_resource(&enc, &cos_sin_buf, MTLResourceUsage::Read);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
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

    assert_eq!(
        read_bf16(&q_buf, (num_q * head_dim) as usize),
        q_ref,
        "player rope arm must be bit-exact vs whole rope_append (Q)"
    );
}

/// attention arm bit-exact vs the whole `attention_via_cache_v2_bf16_specialized`.
/// The player loops the q-heads on one worker (each a full 32-simdgroup attention
/// over the paged cache); the whole kernel runs one TG per (seq, q_head). Same
/// atom, same per-head compute ⇒ bit-exact.
#[test]
fn wavefront_player_attention_bit_exact() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // Llama-3.2-1B attention geometry; one decode sequence (seq 0), paged blk 16.
    let (head_dim, num_q, num_kv, block_size) = (64usize, 32usize, 8usize, 16usize);
    let max_blocks = 4usize;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let kv_len = 40usize; // ceil(40/16) = 3 blocks
    let num_blocks = 3usize;

    let mut block_table_u32 = vec![0u32; max_blocks];
    block_table_u32[0] = 0;
    block_table_u32[1] = 1;
    block_table_u32[2] = 2;
    let block_table = buffer_from_bytes(
        &device,
        &block_table_u32
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let seq_used = buffer_from_bytes(&device, &(kv_len as u32).to_le_bytes());

    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let mut st = 0x7777_3333u64;
    let mut next = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let q: Vec<u8> = (0..num_q * head_dim).flat_map(|_| bf16le(next())).collect();
    let cache_elems = num_blocks * num_kv * block_size * head_dim;
    let k_cache: Vec<u8> = (0..cache_elems).flat_map(|_| bf16le(next())).collect();
    let v_cache: Vec<u8> = (0..cache_elems).flat_map(|_| bf16le(next())).collect();
    let q_buf = buffer_from_bytes(&device, &q);
    let k_buf = buffer_from_bytes(&device, &k_cache);
    let v_buf = buffer_from_bytes(&device, &v_cache);

    let consts = vec![
        ConstantValue::uint(0, head_dim as u32),
        ConstantValue::uint(1, num_q as u32),
        ConstantValue::uint(2, num_kv as u32),
        ConstantValue::float(3, scale),
        ConstantValue::uint(4, block_size as u32),
        ConstantValue::uint(5, max_blocks as u32),
    ];
    let whole = cache
        .get_or_build(&PipelineKey::new(
            "attention",
            "attention_via_cache_v2_bf16_specialized",
            consts,
        ))
        .expect("whole attention pipeline");
    let out_n = num_q * head_dim;
    let out_ref = zeroed_buffer(&device, out_n * 2);
    {
        let queue = device.newCommandQueue().expect("queue");
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&whole);
        let bufs = [&out_ref, &q_buf, &seq_used, &block_table, &k_buf, &v_buf];
        for (i, b) in bufs.iter().enumerate() {
            unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, i) };
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: num_q,
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
    }
    let ref_bits = read_bf16(&out_ref, out_n);
    assert!(
        ref_bits.iter().any(|&b| b != 0),
        "whole attention all zeros"
    );

    // Player: one ATTN subtile. operands [output, q, seq_used, block_table,
    // k_cache, v_cache]; shape (ATTN=6, head_dim, num_q, num_kv, scale_bits,
    // block_size, max_blocks).
    let out = zeroed_buffer(&device, out_n * 2);
    let got = run_player_single(
        &device,
        &cache,
        [
            6,
            head_dim as u32,
            num_q as u32,
            num_kv as u32,
            scale.to_bits(),
            block_size as u32,
            max_blocks as u32,
            0,
        ],
        &[&out, &q_buf, &seq_used, &block_table, &k_buf, &v_buf],
        0,
        out_n,
        1024,
    );
    assert_eq!(
        got, ref_bits,
        "player attention arm must be bit-exact vs whole attention_via_cache_v2"
    );
}

/// Residual Add arm bit-exact vs the CPU `bf16(f32(a)+f32(b))` (a plain
/// deterministic add — no transcendental, so CPU is an exact reference).
#[test]
fn wavefront_player_add_bit_exact() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");
    let n = 2048u32;

    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let mut st = 0x0ADD_0ADDu64;
    let mut next = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0
    };
    let a: Vec<u8> = (0..n).flat_map(|_| bf16le(next())).collect();
    let b: Vec<u8> = (0..n).flat_map(|_| bf16le(next())).collect();
    let a_buf = buffer_from_bytes(&device, &a);
    let b_buf = buffer_from_bytes(&device, &b);
    let out = zeroed_buffer(&device, (n * 2) as usize);

    let got = run_player_single(
        &device,
        &cache,
        [7, n, 0, 0, 0, 0, 0, 0],
        &[&out, &a_buf, &b_buf],
        0,
        n as usize,
        1024,
    );

    let af = |bytes: &[u8], i: usize| {
        half::bf16::from_bits(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]])).to_f32()
    };
    for (i, &g) in got.iter().enumerate() {
        let want = half::bf16::from_f32(af(&a, i) + af(&b, i)).to_bits();
        assert_eq!(g, want, "player add arm[{i}] mismatch");
    }
}

/// rope_append (K side) arm: rotate K in place + write rotated K / un-rotated V
/// into the paged cache, bit-exact vs the whole `rope_append_bf16_specialized`
/// — the UNCHANGED oracle, kept as an independent reference so the test can't
/// pass spuriously. This is the op that lets the megakernel's attention read
/// the new token from the cache, matching the non-mega oracle (Tier-B exact).
/// The Q side stays the rotation-only ROPE arm.
#[test]
fn wavefront_player_rope_append_bit_exact() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // Llama-3.2-1B attention geometry; full rope; one decode token at pos 7.
    let (head_dim, num_q, num_kv, rot_dim, block_size) = (64u32, 32u32, 8u32, 64u32, 16u32);
    let half = (rot_dim / 2) as usize;
    let (pos, max_pos, num_blocks) = (7u32, 16u32, 4u32);
    let slot = 37u32; // block 2, offset 5 — a real (non-sentinel) cache slot
    let kvdim = num_kv * head_dim;

    let bf16le = |v: f32| half::bf16::from_f32(v).to_bits().to_le_bytes();
    let mut st = 0x9E37_1234u64;
    let mut next = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((st >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let q: Vec<u8> = (0..num_q * head_dim)
        .flat_map(|_| bf16le(next() * 2.0))
        .collect();
    let k: Vec<u8> = (0..kvdim).flat_map(|_| bf16le(next() * 2.0)).collect();
    let v: Vec<u8> = (0..kvdim).flat_map(|_| bf16le(next())).collect();
    let cos_sin: Vec<u8> = (0..max_pos * rot_dim)
        .flat_map(|i| {
            let d = (i % rot_dim) as usize;
            let ang = 0.07 * (i as f32);
            bf16le(if d < half { ang.cos() } else { ang.sin() })
        })
        .collect();
    let positions: Vec<u8> = pos.to_le_bytes().to_vec();
    let slot_mapping: Vec<u8> = slot.to_le_bytes().to_vec();
    let cos_sin_buf = buffer_from_bytes(&device, &cos_sin);
    let cache_bytes = (num_blocks * num_kv * block_size * head_dim * 2) as usize;

    // Reference: the whole rope_append_bf16_specialized (rotates q + k, writes
    // the paged cache). We compare only its K rotation + cache (Q is the
    // separate rotation-only ROPE arm).
    let whole = cache
        .get_or_build(&PipelineKey::new(
            "rope",
            "rope_append_bf16_specialized",
            vec![
                ConstantValue::uint(0, head_dim),
                ConstantValue::uint(1, num_q),
                ConstantValue::uint(2, num_kv),
                ConstantValue::uint(3, rot_dim),
                ConstantValue::uint(4, block_size),
            ],
        ))
        .expect("rope_append pipeline");
    let k_ref = buffer_from_bytes(&device, &k);
    let kc_ref = zeroed_buffer(&device, cache_bytes);
    let vc_ref = zeroed_buffer(&device, cache_bytes);
    {
        let q_ref = buffer_from_bytes(&device, &q);
        let v_ref = buffer_from_bytes(&device, &v);
        let pos_buf = buffer_from_bytes(&device, &positions);
        let slot_buf = buffer_from_bytes(&device, &slot_mapping);
        let queue = device.newCommandQueue().expect("queue");
        let cb = queue.commandBuffer().expect("cb");
        let enc = cb.computeCommandEncoder().expect("enc");
        enc.setComputePipelineState(&whole);
        let bufs = [
            &q_ref,
            &k_ref,
            &v_ref,
            &cos_sin_buf,
            &pos_buf,
            &slot_buf,
            &kc_ref,
            &vc_ref,
        ];
        for (i, b) in bufs.iter().enumerate() {
            unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, i) };
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: num_q as usize,
                depth: 1,
            },
            MTLSize {
                width: head_dim as usize,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
    }
    let k_rot_ref = read_bf16(&k_ref, kvdim as usize);
    let kc_ref_bits = read_bf16(&kc_ref, cache_bytes / 2);
    let vc_ref_bits = read_bf16(&vc_ref, cache_bytes / 2);
    assert!(
        kc_ref_bits.iter().any(|&b| b != 0),
        "oracle wrote nothing into the K cache"
    );

    // Player: one WL_OP_ROPE_APPEND instruction. operands [k, cos, sin, v,
    // kv_cache_k, kv_cache_v, slot_mapping]; cos/sin slice cos_sin at pos.
    let k_buf = buffer_from_bytes(&device, &k);
    let v_buf = buffer_from_bytes(&device, &v);
    let kc = zeroed_buffer(&device, cache_bytes);
    let vc = zeroed_buffer(&device, cache_bytes);
    let slot_buf = buffer_from_bytes(&device, &slot_mapping);
    let cos_off = (pos * rot_dim) as u64 * 2;
    let sin_off = (pos * rot_dim + half as u32) as u64 * 2;
    let operands_bytes: Vec<u8> = [
        k_buf.gpuAddress(),
        cos_sin_buf.gpuAddress() + cos_off,
        cos_sin_buf.gpuAddress() + sin_off,
        v_buf.gpuAddress(),
        kc.gpuAddress(),
        vc.gpuAddress(),
        slot_buf.gpuAddress(),
    ]
    .iter()
    .flat_map(|a| a.to_le_bytes())
    .collect();
    let operands = buffer_from_bytes(&device, &operands_bytes);
    // shape (ROPE_APPEND=8, head_dim, num_kv, rot_dim, block_size, ...).
    let shapes = buffer_from_bytes(
        &device,
        &bytes_of_u32(&[8, head_dim, num_kv, rot_dim, block_size, 0, 0, 0]),
    );
    let tape = buffer_from_bytes(&device, &bytes_of_u32(&[0, 0, 0, 0]));
    let tape_offsets = buffer_from_bytes(&device, &bytes_of_u32(&[0, 1]));
    let flags = zeroed_buffer(&device, 4);
    let player = cache
        .get_or_build(&PipelineKey::new(
            "wavefront_layer",
            "wavefront_player_bf16_s_f16_gs_64_b_4",
            vec![],
        ))
        .expect("player pipeline");
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&player);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&tape), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&shapes), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&operands), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&tape_offsets), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&flags), 0, 4);
    }
    use_resource(
        &enc,
        &k_buf,
        MTLResourceUsage::Read | MTLResourceUsage::Write,
    );
    use_resource(&enc, &cos_sin_buf, MTLResourceUsage::Read);
    use_resource(&enc, &v_buf, MTLResourceUsage::Read);
    use_resource(&enc, &kc, MTLResourceUsage::Read | MTLResourceUsage::Write);
    use_resource(&enc, &vc, MTLResourceUsage::Read | MTLResourceUsage::Write);
    use_resource(&enc, &slot_buf, MTLResourceUsage::Read);
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
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

    assert_eq!(
        read_bf16(&k_buf, kvdim as usize),
        k_rot_ref,
        "rope_append arm: K rotation must match the whole rope_append"
    );
    assert_eq!(
        read_bf16(&kc, cache_bytes / 2),
        kc_ref_bits,
        "rope_append arm: K paged-cache write must match the whole rope_append"
    );
    assert_eq!(
        read_bf16(&vc, cache_bytes / 2),
        vc_ref_bits,
        "rope_append arm: V paged-cache write must match the whole rope_append"
    );
}

// ── serializer → player end-to-end (the encoding actually drives the GPU) ──

/// A 2-op `qmv → qmv` chain produced by `ferrite_wavefront::mega::serialize`,
/// run through the player with the gpuAddress operand table built from
/// `MegaProgram.operands`, must be bit-exact vs two sequential whole qmvs.
///
/// Every prior test drove the player from a hand-built tape; this is the first
/// to drive it from the *serializer's* output, so it proves the whole path:
/// the emitted tape/shape tables, the `BufId → gpuAddress` resolution, and the
/// intra-worker arena hand-off (qmv0 writes an arena slot that qmv1 reads).
#[test]
fn wavefront_serialized_qmv_chain_bit_exact() {
    use ferrite_wavefront::lower::{InputRef, LoweredOp, LoweringInput, OpDesc};
    use ferrite_wavefront::mega::{Geometry, SourceDesc, serialize};
    use ferrite_wavefront::region::lower_region;
    use ferrite_wavefront::region_schedule::partition_roundrobin;
    use ferrite_wavefront::subtile::SourceShape;
    use ferrite_wavefront::subtile_ir::{BufferRef, WeightBundle, WeightLoc, WeightRole};

    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    let gs = 64u32;
    let (k0, n0) = (512u32, 512u32); // stage0: x[1,512] @ W0[512,512]
    let (k1, n1) = (512u32, 64u32); // stage1: y1[1,512] @ W1[64,512]; k1 == n0

    let (w0, s0, b0, x) = make_qmv(&device, n0, k0, gs, 0x5151_0001);
    let (w1, s1, b1, _x1) = make_qmv(&device, n1, k1, gs, 0x5151_0002);

    // Reference: two sequential whole qmvs.
    let y1_ref = zeroed_buffer(&device, (n0 * 2) as usize);
    qmv_whole_into(&device, &cache, &w0, &s0, &b0, &x, &y1_ref, n0, k0, gs);
    let y2_ref_buf = zeroed_buffer(&device, (n1 * 2) as usize);
    qmv_whole_into(
        &device,
        &cache,
        &w1,
        &s1,
        &b1,
        &y1_ref,
        &y2_ref_buf,
        n1,
        k1,
        gs,
    );
    let y2_ref = read_bf16(&y2_ref_buf, n1 as usize);
    assert!(y2_ref.iter().any(|&v| v != 0));

    // Build + serialize the chain.
    let wl = |op_idx| WeightLoc {
        layer: 0,
        bucket: 0,
        op_idx,
        slot: 0,
    };
    let qw = |op_idx| SourceDesc::QuantWeight {
        weight: BufferRef::Weight {
            bundle: WeightBundle::LinearLayer,
            role: WeightRole::Weight,
            loc: wl(op_idx),
        },
        scales: BufferRef::Weight {
            bundle: WeightBundle::LinearLayer,
            role: WeightRole::AffineScales,
            loc: wl(op_idx),
        },
        biases: BufferRef::Weight {
            bundle: WeightBundle::LinearLayer,
            role: WeightRole::AffineBiases,
            loc: wl(op_idx),
        },
        group_size: gs,
        bits: 4,
        scale_elem: 2,
    };
    let input = LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols: k0 },  // 0 x
            SourceShape { rows: n0, cols: k0 }, // 1 W0
            SourceShape { rows: n1, cols: k1 }, // 2 W1
        ],
        ops: vec![
            OpDesc {
                op: LoweredOp::Gemm { n: n0, k: k0 },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            OpDesc {
                op: LoweredOp::Gemm { n: n1, k: k1 },
                m: 1,
                inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
            },
        ],
        result: 1,
    };
    let sources = vec![
        SourceDesc::Dense {
            buffer: BufferRef::Weight {
                bundle: WeightBundle::Embedding,
                role: WeightRole::Weight,
                loc: wl(0),
            },
            elem: 2,
        },
        qw(1),
        qw(2),
    ];
    let g = lower_region(&input, 1000); // coarse: one block per qmv, P=1
    let sched = partition_roundrobin(&g, 1);
    let prog = serialize(
        &g,
        &sched,
        &sources,
        Geometry {
            act_elem: 2,
            block_size: 16,
            max_blocks: 4,
        },
    )
    .expect("serialize");

    // Resolve each BufId → a synthetic buffer: arena slots are fresh zeroed
    // buffers (sized by arena_bytes), weights/x map to the make_qmv buffers.
    let arena: Vec<Buffer> = prog
        .arena_bytes
        .iter()
        .map(|&b| zeroed_buffer(&device, b as usize))
        .collect();
    let resolved: Vec<Buffer> = prog
        .buffers
        .iter()
        .map(|bref| match bref {
            BufferRef::ArenaSlot(s) => arena[*s as usize].clone(),
            BufferRef::Weight {
                bundle: WeightBundle::Embedding,
                ..
            } => x.clone(),
            BufferRef::Weight {
                bundle: WeightBundle::LinearLayer,
                role,
                loc,
            } => {
                let (w, s, b) = if loc.op_idx == 1 {
                    (&w0, &s0, &b0)
                } else {
                    (&w1, &s1, &b1)
                };
                match role {
                    WeightRole::Weight => w.clone(),
                    WeightRole::AffineScales => s.clone(),
                    WeightRole::AffineBiases => b.clone(),
                    other => panic!("unexpected weight role {other:?}"),
                }
            }
            other => panic!("unexpected buffer ref {other:?}"),
        })
        .collect();

    // Operand table: gpuAddress(buffer) + byte_offset (base 0 — whole tensors).
    let operand_addrs: Vec<u64> = prog
        .operands
        .iter()
        .map(|sl| resolved[sl.buffer.0 as usize].gpuAddress() + sl.byte_offset)
        .collect();
    let operands = buffer_from_bytes(&device, &bytes_of_u64(&operand_addrs));
    let tape = buffer_from_bytes(&device, &prog.tape_bytes());
    let shapes = buffer_from_bytes(&device, &prog.shapes_bytes());
    let tape_offsets = buffer_from_bytes(&device, &prog.tape_offsets_bytes());
    let flags = zeroed_buffer(&device, (prog.num_flags.max(1) * 4) as usize);

    let player = cache
        .get_or_build(&PipelineKey::new(
            "wavefront_layer",
            "wavefront_player_bf16_s_f16_gs_64_b_4",
            vec![],
        ))
        .expect("player pipeline");
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    enc.setComputePipelineState(&player);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&tape), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&shapes), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&operands), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&tape_offsets), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&flags), 0, 4);
    }
    for buf in &resolved {
        use_resource(&enc, buf, MTLResourceUsage::Read | MTLResourceUsage::Write);
    }
    let p = prog.tape_offsets.len() as u32 - 1;
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: p as usize,
            height: 1,
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

    // The result is the qmv1 output arena slot the serializer recorded.
    let result_buf = &resolved[prog.result.0 as usize];
    assert_eq!(
        read_bf16(result_buf, n1 as usize),
        y2_ref,
        "serialized qmv→qmv chain must be bit-exact vs two sequential whole qmvs"
    );
}
