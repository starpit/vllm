// SPDX-License-Identifier: Apache-2.0
//
// Softmax kernel CPU-parity tests.
//
// Validates the `block_softmax_*` shaders (a faithful port of mlx
// `softmax_single_row`) against the CPU reference in
// `ferrite_metal_kernels::softmax::softmax_cpu_f32`. Coverage targets
// the MoE-router shapes used by the Qwen3-MoE / Qwen3-Next forward
// path: num_experts ∈ {8, 128, 512} × batch ∈ {1, 32}.

#![cfg(target_os = "macos")]

use ferrite_metal_kernels::detect_device;
use ferrite_metal_kernels::softmax::{
    dispatch_softmax, softmax_cpu_f32, Buffer, Device, SoftmaxKernels,
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

fn deterministic_inputs(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i.wrapping_mul(2654435761)) as u32 as f32) / (u32::MAX as f32) * 10.0 - 5.0)
        .collect()
}

#[test]
fn softmax_f32_matches_cpu_router_shapes() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = SoftmaxKernels::new(&device).expect("SoftmaxKernels::new");

    // num_experts ∈ {8 (Mixtral), 128 (Qwen3-MoE), 512 (Qwen3-Next)} ×
    // batch ∈ {1 decode, 32 prefill}.
    for &(batch, ax) in &[(1usize, 8usize), (1, 128), (1, 512), (32, 128), (32, 512)] {
        let input = deterministic_inputs(batch * ax);
        let in_buf = upload(&device, &input);
        let out_buf = upload(&device, &vec![0.0f32; input.len()]);
        dispatch_softmax(
            &kernels.f32,
            &queue,
            &in_buf,
            &out_buf,
            batch as u32,
            ax as u32,
            4,
        )
        .expect("dispatch_softmax");
        let got: Vec<f32> = download(&out_buf, input.len());
        let want = softmax_cpu_f32(&input, batch, ax);
        for i in 0..input.len() {
            let diff = (got[i] - want[i]).abs();
            assert!(
                diff < 5e-6,
                "f32 softmax mismatch at batch={batch} ax={ax} idx={i}: got={} want={} diff={diff}",
                got[i],
                want[i]
            );
        }
        for b in 0..batch {
            let s: f32 = got[b * ax..(b + 1) * ax].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "batch={batch} ax={ax} row {b} sum={s}");
        }
    }
}

#[test]
fn softmax_bf16_precise_matches_cpu() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = SoftmaxKernels::new(&device).expect("SoftmaxKernels::new");
    let (batch, ax) = (32usize, 512usize);

    let input_f32 = deterministic_inputs(batch * ax);
    let input_bf16: Vec<u16> = input_f32
        .iter()
        .map(|x| half::bf16::from_f32(*x).to_bits())
        .collect();
    let input_via_bf16: Vec<f32> = input_bf16
        .iter()
        .map(|b| half::bf16::from_bits(*b).to_f32())
        .collect();

    let in_buf = upload(&device, &input_bf16);
    let out_buf = upload(&device, &vec![0u16; input_bf16.len()]);
    dispatch_softmax(
        &kernels.bf16_precise,
        &queue,
        &in_buf,
        &out_buf,
        batch as u32,
        ax as u32,
        2,
    )
    .expect("dispatch_softmax bf16 precise");
    let got_bf16: Vec<u16> = download(&out_buf, input_bf16.len());
    let got: Vec<f32> = got_bf16
        .iter()
        .map(|b| half::bf16::from_bits(*b).to_f32())
        .collect();
    let want = softmax_cpu_f32(&input_via_bf16, batch, ax);
    for i in 0..input_bf16.len() {
        let diff = (got[i] - want[i]).abs();
        assert!(
            diff < 2e-3,
            "bf16 precise softmax mismatch idx={i}: got={} want={} diff={diff}",
            got[i],
            want[i]
        );
    }
}

#[test]
fn softmax_f16_precise_matches_cpu() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = SoftmaxKernels::new(&device).expect("SoftmaxKernels::new");
    let (batch, ax) = (32usize, 128usize);

    let input_f32 = deterministic_inputs(batch * ax);
    let input_f16: Vec<u16> = input_f32
        .iter()
        .map(|x| half::f16::from_f32(*x).to_bits())
        .collect();
    let input_via_f16: Vec<f32> = input_f16
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();

    let in_buf = upload(&device, &input_f16);
    let out_buf = upload(&device, &vec![0u16; input_f16.len()]);
    dispatch_softmax(
        &kernels.f16_precise,
        &queue,
        &in_buf,
        &out_buf,
        batch as u32,
        ax as u32,
        2,
    )
    .expect("dispatch_softmax f16 precise");
    let got_f16: Vec<u16> = download(&out_buf, input_f16.len());
    let got: Vec<f32> = got_f16
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();
    let want = softmax_cpu_f32(&input_via_f16, batch, ax);
    for i in 0..input_f16.len() {
        let diff = (got[i] - want[i]).abs();
        assert!(
            diff < 5e-4,
            "f16 precise softmax mismatch idx={i}: got={} want={} diff={diff}",
            got[i],
            want[i]
        );
    }
}
