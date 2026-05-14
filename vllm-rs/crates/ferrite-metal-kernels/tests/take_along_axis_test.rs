// SPDX-License-Identifier: Apache-2.0
//
// TakeAlongAxis (gather_axis 2D contiguous, axis=-1) CPU-parity tests.

#![cfg(target_os = "macos")]

use ferrite_metal_kernels::argpartition::ArgSortDtype;
use ferrite_metal_kernels::detect_device;
use ferrite_metal_kernels::take_along_axis::{
    dispatch_take_along_axis, take_along_axis_cpu_f32, Buffer, Device, TakeAlongAxisKernels,
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

fn deterministic_distinct(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i.wrapping_mul(2654435761)) as u32 as f32) / (u32::MAX as f32) * 10.0 - 5.0)
        .collect()
}

fn random_indices(batch: usize, axis_size: usize, top_k: usize) -> Vec<u32> {
    // Pick top_k indices per row from a deterministic pseudo-random
    // permutation — avoids duplicates within a row so we can compare
    // values 1:1.
    let mut out = Vec::with_capacity(batch * top_k);
    for b in 0..batch {
        let mut perm: Vec<u32> = (0..axis_size as u32).collect();
        // Fisher-Yates with a deterministic seed per row.
        let mut state = (b as u32).wrapping_mul(2654435761) ^ 0x9E3779B9;
        for i in (1..axis_size).rev() {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (state as usize) % (i + 1);
            perm.swap(i, j);
        }
        out.extend_from_slice(&perm[..top_k]);
    }
    out
}

#[test]
fn take_along_axis_f32_router_shapes() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = TakeAlongAxisKernels::new(&device).expect("TakeAlongAxisKernels::new");

    for &(batch, ax, k) in &[
        (1usize, 8usize, 2usize),
        (1, 128, 8),
        (1, 512, 10),
        (32, 128, 8),
        (32, 512, 10),
    ] {
        let src = deterministic_distinct(batch * ax);
        let idx = random_indices(batch, ax, k);
        let src_buf = upload(&device, &src);
        let idx_buf = upload(&device, &idx);
        let out_buf = upload(&device, &vec![0.0f32; batch * k]);
        dispatch_take_along_axis(
            &kernels,
            ArgSortDtype::F32,
            &queue,
            &src_buf,
            &idx_buf,
            &out_buf,
            batch as u32,
            ax as u32,
            k as u32,
            4,
        )
        .expect("dispatch_take_along_axis f32");
        let got: Vec<f32> = download(&out_buf, batch * k);
        let want = take_along_axis_cpu_f32(&src, &idx, batch, ax, k);
        for i in 0..(batch * k) {
            assert!(
                (got[i] - want[i]).abs() < 1e-6,
                "f32 mismatch ax={ax} k={k} idx={i}: got={} want={}",
                got[i],
                want[i]
            );
        }
    }
}

#[test]
fn take_along_axis_bf16_router_shapes() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = TakeAlongAxisKernels::new(&device).expect("TakeAlongAxisKernels::new");

    let (batch, ax, k) = (32usize, 512usize, 10usize);
    let src_f32 = deterministic_distinct(batch * ax);
    let src_bf16: Vec<u16> = src_f32
        .iter()
        .map(|x| half::bf16::from_f32(*x).to_bits())
        .collect();
    let src_via_bf16: Vec<f32> = src_bf16
        .iter()
        .map(|b| half::bf16::from_bits(*b).to_f32())
        .collect();
    let idx = random_indices(batch, ax, k);

    let src_buf = upload(&device, &src_bf16);
    let idx_buf = upload(&device, &idx);
    let out_buf = upload(&device, &vec![0u16; batch * k]);
    dispatch_take_along_axis(
        &kernels,
        ArgSortDtype::BF16,
        &queue,
        &src_buf,
        &idx_buf,
        &out_buf,
        batch as u32,
        ax as u32,
        k as u32,
        2,
    )
    .expect("dispatch_take_along_axis bf16");
    let got_bf16: Vec<u16> = download(&out_buf, batch * k);
    let got: Vec<f32> = got_bf16
        .iter()
        .map(|b| half::bf16::from_bits(*b).to_f32())
        .collect();
    let want = take_along_axis_cpu_f32(&src_via_bf16, &idx, batch, ax, k);
    for i in 0..(batch * k) {
        assert!(
            (got[i] - want[i]).abs() < 1e-6,
            "bf16 mismatch idx={i}: got={} want={}",
            got[i],
            want[i]
        );
    }
}
