// SPDX-License-Identifier: Apache-2.0
//
// Row-gather kernel CPU-parity tests.
//
// Validates the `row_gather_*` shaders against the CPU reference in
// `ferrite_metal_kernels::row_gather`. Coverage:
//
//   * `idx_divisor == 1` (plain row gather): used by `indices[order]`
//      and `_scatter_unsort: x[inv_order]`.
//   * `idx_divisor == top_k` (sort gather): used by
//      `x.flatten(0,-3)[order // K]` in `_gather_sort`.
//   * Shapes: (M, D) covering MoE-router-driven and decode-friendly
//     dimensions.
//   * dtypes: f32, bf16, u32.

#![cfg(target_os = "macos")]

use ferrite_metal_kernels::detect_device;
use ferrite_metal_kernels::row_gather::{
    dispatch_row_gather, row_gather_cpu_f32, row_gather_cpu_u32, Buffer, Device, RowGatherDtype,
    RowGatherKernels,
};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

fn upload<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let nbytes = std::mem::size_of_val(data);
    let buf = device
        .newBufferWithLength_options(nbytes.max(1), MTLResourceOptions::StorageModeShared)
        .expect("buf");
    if nbytes > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buf.contents().as_ptr() as *mut u8,
                nbytes,
            );
        }
    }
    buf
}

fn download<T: Copy>(buf: &Buffer, n: usize) -> Vec<T> {
    let mut out = vec![unsafe { std::mem::zeroed::<T>() }; n];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents().as_ptr() as *const u8,
            out.as_mut_ptr() as *mut u8,
            n * std::mem::size_of::<T>(),
        );
    }
    out
}

#[test]
fn row_gather_f32_plain() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = RowGatherKernels::new(&device).expect("RowGatherKernels::new");

    // (src_rows, d, m, idx_divisor)
    for &(src_rows, d, m, divisor) in &[
        (8usize, 4096usize, 8usize, 1u32),  // _scatter_unsort tiny
        (16, 2048, 16, 1),                  // small prefill
        (4, 128, 32, 1),                    // M > src_rows
        (64, 768, 64, 1),                   // bigger
    ] {
        let src: Vec<f32> = (0..src_rows * d).map(|i| (i as f32) * 0.001 - 1.5).collect();
        let idx: Vec<u32> = (0..m).map(|i| ((i * 31) as u32) % src_rows as u32).collect();
        let want = row_gather_cpu_f32(&src, &idx, src_rows, d, divisor);

        let src_buf = upload(&device, &src);
        let idx_buf = upload(&device, &idx);
        let out_buf = upload(&device, &vec![0.0f32; m * d]);
        dispatch_row_gather(
            &kernels,
            RowGatherDtype::F32,
            &queue,
            &src_buf,
            &idx_buf,
            &out_buf,
            src_rows as u32,
            m as u32,
            d as u32,
            divisor,
            4,
        )
        .expect("dispatch_row_gather f32");
        let got: Vec<f32> = download(&out_buf, m * d);
        for i in 0..m * d {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "src_rows={src_rows} d={d} m={m} divisor={divisor} i={i}"
            );
        }
    }
}

#[test]
fn row_gather_f32_div_top_k() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = RowGatherKernels::new(&device).expect("RowGatherKernels::new");

    // mimic `x.flatten(0,-3)[order // K]`: x rows = N, idx length =
    // N*K, divisor = K.
    let n = 16usize;
    let k = 8usize;
    let d = 1024usize;
    let m = n * k;
    let src: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.0003 + 0.7).collect();
    // `order` is a permutation of [0, N*K). Build one.
    let mut order: Vec<u32> = (0..m as u32).collect();
    // deterministic shuffle: swap pairs based on hash.
    for i in 0..m {
        let j = ((i.wrapping_mul(2654435761)) as u32 % m as u32) as usize;
        order.swap(i, j);
    }
    let want = row_gather_cpu_f32(&src, &order, n, d, k as u32);

    let src_buf = upload(&device, &src);
    let idx_buf = upload(&device, &order);
    let out_buf = upload(&device, &vec![0.0f32; m * d]);
    dispatch_row_gather(
        &kernels,
        RowGatherDtype::F32,
        &queue,
        &src_buf,
        &idx_buf,
        &out_buf,
        n as u32,
        m as u32,
        d as u32,
        k as u32,
        4,
    )
    .expect("dispatch_row_gather f32 div");
    let got: Vec<f32> = download(&out_buf, m * d);
    for i in 0..m * d {
        assert_eq!(got[i].to_bits(), want[i].to_bits(), "div-top-k i={i}");
    }
}

#[test]
fn row_gather_bf16_plain() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = RowGatherKernels::new(&device).expect("RowGatherKernels::new");

    let src_rows = 24usize;
    let d = 512usize;
    let m = 24usize;
    let src_f32: Vec<f32> = (0..src_rows * d).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let src_bf16: Vec<u16> = src_f32
        .iter()
        .map(|x| half::bf16::from_f32(*x).to_bits())
        .collect();
    let idx: Vec<u32> = (0..m as u32).rev().collect(); // reverse perm
    let want_bf16: Vec<u16> = {
        let src_via_bf16: Vec<f32> = src_bf16
            .iter()
            .map(|b| half::bf16::from_bits(*b).to_f32())
            .collect();
        let want_f32 = row_gather_cpu_f32(&src_via_bf16, &idx, src_rows, d, 1);
        want_f32
            .iter()
            .map(|x| half::bf16::from_f32(*x).to_bits())
            .collect()
    };

    let src_buf = upload(&device, &src_bf16);
    let idx_buf = upload(&device, &idx);
    let out_buf = upload(&device, &vec![0u16; m * d]);
    dispatch_row_gather(
        &kernels,
        RowGatherDtype::BF16,
        &queue,
        &src_buf,
        &idx_buf,
        &out_buf,
        src_rows as u32,
        m as u32,
        d as u32,
        1,
        2,
    )
    .expect("dispatch_row_gather bf16");
    let got: Vec<u16> = download(&out_buf, m * d);
    for i in 0..m * d {
        assert_eq!(got[i], want_bf16[i], "bf16 plain i={i}");
    }
}

#[test]
fn row_gather_u32_d1() {
    // 1D index gather: indices_sorted = indices_flat[order].
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = RowGatherKernels::new(&device).expect("RowGatherKernels::new");

    let n = 24usize; // source length
    let m = 24usize; // gathered length
    let src: Vec<u32> = (0..n as u32).map(|i| (i * 7) % 60).collect();
    let idx: Vec<u32> = {
        let mut v: Vec<u32> = (0..m as u32).collect();
        v.swap(0, m - 1);
        v.swap(3, m - 4);
        v
    };
    let want = row_gather_cpu_u32(&src, &idx, n, 1, 1);

    let src_buf = upload(&device, &src);
    let idx_buf = upload(&device, &idx);
    let out_buf = upload(&device, &vec![0u32; m]);
    dispatch_row_gather(
        &kernels,
        RowGatherDtype::U32,
        &queue,
        &src_buf,
        &idx_buf,
        &out_buf,
        n as u32,
        m as u32,
        1,
        1,
        4,
    )
    .expect("dispatch_row_gather u32 d=1");
    let got: Vec<u32> = download(&out_buf, m);
    assert_eq!(got, want);
}
