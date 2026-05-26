// SPDX-License-Identifier: Apache-2.0
//! On-device proof of the GPU subtile player linchpin: the SAME affine
//! qmv, dispatched WHOLE (one block, N=n) vs N-block-TILED (nb<n, with
//! offset weight/scales/biases/y bindings + an `OUT_VEC_SIZE=nb` pipeline
//! per block), must produce BIT-IDENTICAL output. Every output element is
//! the same per-K reduction in both — the kernel just sees a standalone
//! `nb × K` matvec at offset base pointers.
//!
//! Both paths run through the real stack: `tile_qmv` (the compiler) →
//! `play` (the trivial player) → `MetalExecutor` (binds + dispatches).
//!
//! Integration test (not a lib `#[cfg(test)]` module) so it compiles
//! against the public API and skips the lib's unrelated test-build rot.
//! Run: `cargo test -p ferrite-forward -F metal --test subtile_player_gpu`.
#![cfg(feature = "metal")]

use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice, MTLResourceOptions,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use ferrite_forward::interpreter::metal::__re::{Buffer, Device};
use ferrite_forward::interpreter::metal::subtile_player::{
    MetalExecutor, ResolvedBuffer, resolve_pipelines,
};
use ferrite_metal_kernels::device::detect_device;
use ferrite_metal_kernels::quantized::{
    DequantDtype, QmvKernel, ScaleDtype, qmv_kernel_static_name,
};
use ferrite_metal_kernels::specialized_pipeline_cache::SpecializedPipelineCache;
use ferrite_wavefront::subtile_ir::{
    BufId, BufferRef, PipelineInterner, QmvKernelInfo, QmvOperands, QmvShape, SubtileIr, play,
    tile_qmv,
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

/// Run a one-matmul `SubtileIr` (the qmv blocks for `y`) through the
/// trivial player; return `y`'s raw bf16 bits.
fn run_ir(
    ir: &SubtileIr,
    bufs: &[ResolvedBuffer],
    cache: &SpecializedPipelineCache,
    device: &Device,
    n_out: usize,
    y: &Buffer,
) -> Vec<u16> {
    let pipelines = resolve_pipelines(ir, cache).expect("resolve pipelines");
    let queue = device.newCommandQueue().expect("queue");
    let cb = queue.commandBuffer().expect("cb");
    let enc = cb.computeCommandEncoder().expect("enc");
    {
        let mut exec = MetalExecutor::new(&enc, bufs, &pipelines);
        play(ir, &mut exec);
    }
    enc.endEncoding();
    cb.commit();
    cb.waitUntilCompleted();
    read_bf16(y, n_out)
}

#[test]
fn nblocked_qmv_bit_exact_vs_whole() {
    let Some(md) = detect_device() else {
        eprintln!("[skip] no Metal device");
        return;
    };
    let device = md.device;
    let cache =
        SpecializedPipelineCache::with_standard_shaders(device.clone()).expect("shader cache");

    // Generic-eligible shape (the generic qmv is valid for any block
    // width, so whole vs ragged-free 4-way split compare cleanly).
    let (n, k, gs, bits, m) = (96u32, 512u32, 64u32, 4u32, 1u32);
    let n_bytes_packed = (n * k / 2) as usize; // 2 nibbles/byte
    let n_groups = (n * k / gs) as usize;

    // Deterministic inputs (values are irrelevant to blocked==whole; a
    // fixed LCG just makes any divergence reproducible).
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

    let packed_buf = buffer_from_bytes(&device, &packed);
    let scales_buf = buffer_from_bytes(&device, &scales);
    let biases_buf = buffer_from_bytes(&device, &biases);
    let x_buf = buffer_from_bytes(&device, &x);
    let y_whole = zeroed_buffer(&device, (n * m) as usize * 2);
    let y_blocked = zeroed_buffer(&device, (n * m) as usize * 2);

    // Force the generic variant for every block (valid for any width).
    let generic_symbol = qmv_kernel_static_name(
        QmvKernel::Generic,
        DequantDtype::Bf16,
        ScaleDtype::F16,
        bits,
        gs,
    )
    .to_string();
    let info_for = |_w: u32| QmvKernelInfo {
        library: "quantized_qmv",
        symbol: generic_symbol.clone(),
        bn: 8,
        tpt: [32, 2, 1],
    };

    let ops = QmvOperands {
        weight: (BufId(0), 0),
        scales: (BufId(1), 0),
        biases: (BufId(2), 0),
        x: (BufId(3), 0),
        y: (BufId(4), 0),
    };
    let shape = QmvShape {
        n,
        k,
        group_size: gs,
        bits,
        m,
    };
    let placeholder = vec![BufferRef::ArenaSlot(0); 5];

    // Whole: one block (nb >= n).
    let mut pl_w = PipelineInterner::default();
    let tape_w = tile_qmv(&ops, shape, n, info_for, 2, 2, &mut pl_w);
    assert_eq!(tape_w.len(), 1, "whole = single block");
    let ir_w = SubtileIr {
        buffers: placeholder.clone(),
        elem_bytes: vec![2, 2, 2, 2, 2],
        pipelines: pl_w.specs,
        num_flags: 0,
        tape: tape_w,
        terminal: BufId(4),
    };
    let bufs_w = [
        (packed_buf.clone(), 0u64),
        (scales_buf.clone(), 0),
        (biases_buf.clone(), 0),
        (x_buf.clone(), 0),
        (y_whole.clone(), 0),
    ];
    let out_whole = run_ir(&ir_w, &bufs_w, &cache, &device, (n * m) as usize, &y_whole);

    // Blocked: nb=24 → 4 blocks of 24.
    let mut pl_b = PipelineInterner::default();
    let tape_b = tile_qmv(&ops, shape, 24, info_for, 2, 2, &mut pl_b);
    assert_eq!(tape_b.len(), 4, "96/24 = 4 blocks");
    let ir_b = SubtileIr {
        buffers: placeholder,
        elem_bytes: vec![2, 2, 2, 2, 2],
        pipelines: pl_b.specs,
        num_flags: 0,
        tape: tape_b,
        terminal: BufId(4),
    };
    let bufs_b = [
        (packed_buf, 0u64),
        (scales_buf, 0),
        (biases_buf, 0),
        (x_buf, 0),
        (y_blocked.clone(), 0),
    ];
    let out_blocked = run_ir(
        &ir_b,
        &bufs_b,
        &cache,
        &device,
        (n * m) as usize,
        &y_blocked,
    );

    assert_eq!(
        out_whole, out_blocked,
        "N-blocked qmv must be bit-exact vs whole qmv"
    );
    assert!(out_whole.iter().any(|&b| b != 0), "output is all zero");
}
