// SPDX-License-Identifier: Apache-2.0
//
// ArgPartition (argsort) kernel CPU-parity tests.
//
// Validates the `c_arg_block_sort_*` shaders (a faithful port of mlx
// single-block argsort) against the CPU reference in
// `ferrite_metal_kernels::argpartition::argsort_cpu_f32`. Coverage
// targets the MoE-router shapes used by the Qwen3-MoE / Qwen3-Next
// forward path:
//
//   * num_experts ∈ {8, 60, 128, 512} → exercises bn ∈ {32, 64, 128}
//   * batch ∈ {1 decode, 32 prefill}
//   * dtypes: f32, f16, bf16
//
// Like softmax, this is an integration test under `tests/` to avoid
// the pre-existing test-rot in `src/instruction_executor/test_*.rs`.

#![cfg(target_os = "macos")]

use ferrite_metal_kernels::argpartition::{
    argsort_cpu_f32, argsort_cpu_u32, dispatch_argpartition, ArgPartitionKernels, ArgSortDtype,
    Buffer, Device,
};
use ferrite_metal_kernels::detect_device;
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

/// Distinct deterministic values in `[-5, 5]`. Distinctness avoids
/// tie-break differences between mlx's block-merge-sort (not strictly
/// stable across the merge boundary) and `argsort_cpu_f32` (stable).
fn deterministic_distinct(n: usize) -> Vec<f32> {
    // Knuth multiplicative hash mod 2^32 → [0, 1) → [-5, 5]. n ≤ 32*512
    // for MoE shapes so collisions are negligible.
    (0..n)
        .map(|i| ((i.wrapping_mul(2654435761)) as u32 as f32) / (u32::MAX as f32) * 10.0 - 5.0)
        .collect()
}

/// Top-k accuracy: the trailing `k` indices the GPU produces must
/// describe a value set equal to the CPU's trailing `k` indices'
/// value set (i.e., same top-k by value; internal ordering inside the
/// top-k slice may differ if values are equal — distinct inputs make
/// that a non-issue).
fn assert_topk_matches(
    got: &[u32],
    want: &[u32],
    input: &[f32],
    batch: usize,
    axis_size: usize,
    k: usize,
) {
    for b in 0..batch {
        let g = &got[b * axis_size..(b + 1) * axis_size];
        let w = &want[b * axis_size..(b + 1) * axis_size];
        let row = &input[b * axis_size..(b + 1) * axis_size];

        let mut got_topk: Vec<f32> = g[axis_size - k..]
            .iter()
            .map(|&i| row[i as usize])
            .collect();
        let mut want_topk: Vec<f32> = w[axis_size - k..]
            .iter()
            .map(|&i| row[i as usize])
            .collect();
        got_topk.sort_by(|a, b| a.partial_cmp(b).unwrap());
        want_topk.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..k {
            assert!(
                (got_topk[i] - want_topk[i]).abs() < 1e-6,
                "row={b} k={k} ax={axis_size}: gpu top-k value[{i}]={} cpu={}",
                got_topk[i],
                want_topk[i]
            );
        }
    }
}

#[test]
fn argpartition_f32_router_shapes() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = ArgPartitionKernels::new(&device).expect("ArgPartitionKernels::new");

    for &(batch, ax, k) in &[
        (1usize, 8usize, 2usize),   // Mixtral decode
        (1, 60, 4),                 // Qwen3-MoE-30B-A3B decode
        (1, 128, 8),                // Qwen3-MoE-2x57B decode
        (1, 512, 10),               // Qwen3-Next decode
        (32, 128, 8),               // Qwen3-MoE prefill
        (32, 512, 10),              // Qwen3-Next prefill
    ] {
        let input = deterministic_distinct(batch * ax);
        let in_buf = upload(&device, &input);
        let out_buf = upload(&device, &vec![0u32; input.len()]);
        dispatch_argpartition(
            &kernels,
            ArgSortDtype::F32,
            &queue,
            &in_buf,
            &out_buf,
            batch as u32,
            ax as u32,
            4,
        )
        .expect("dispatch_argpartition f32");
        let got: Vec<u32> = download(&out_buf, input.len());
        let want = argsort_cpu_f32(&input, batch, ax);
        assert_topk_matches(&got, &want, &input, batch, ax, k);
    }
}

#[test]
fn argpartition_bf16_router_shapes() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = ArgPartitionKernels::new(&device).expect("ArgPartitionKernels::new");

    for &(batch, ax, k) in &[(1usize, 60usize, 4usize), (32, 128, 8), (32, 512, 10)] {
        let input_f32 = deterministic_distinct(batch * ax);
        let input_bf16: Vec<u16> = input_f32
            .iter()
            .map(|x| half::bf16::from_f32(*x).to_bits())
            .collect();
        // Round-trip so the CPU reference uses the same bf16-truncated
        // values the GPU sorts against.
        let input_via_bf16: Vec<f32> = input_bf16
            .iter()
            .map(|b| half::bf16::from_bits(*b).to_f32())
            .collect();

        let in_buf = upload(&device, &input_bf16);
        let out_buf = upload(&device, &vec![0u32; input_bf16.len()]);
        dispatch_argpartition(
            &kernels,
            ArgSortDtype::BF16,
            &queue,
            &in_buf,
            &out_buf,
            batch as u32,
            ax as u32,
            2,
        )
        .expect("dispatch_argpartition bf16");
        let got: Vec<u32> = download(&out_buf, input_bf16.len());
        let want = argsort_cpu_f32(&input_via_bf16, batch, ax);
        assert_topk_matches(&got, &want, &input_via_bf16, batch, ax, k);
    }
}

/// Full-row exact match: every position in the argsort must agree.
/// Used for the u32→u32 path where inputs are distinct.
fn assert_argsort_exact_u32(got: &[u32], want: &[u32], batch: usize, axis_size: usize) {
    for b in 0..batch {
        let g = &got[b * axis_size..(b + 1) * axis_size];
        let w = &want[b * axis_size..(b + 1) * axis_size];
        for i in 0..axis_size {
            assert_eq!(
                g[i], w[i],
                "row={b} ax={axis_size} pos={i}: gpu={} cpu={}",
                g[i], w[i]
            );
        }
    }
}

/// uint32 argsort — used by MoE `_gather_sort` to sort flattened
/// expert indices and then sort `order` to produce `inv_order`. Inputs
/// here mirror those two stages.
#[test]
fn argpartition_u32_gather_sort_shapes() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = ArgPartitionKernels::new(&device).expect("ArgPartitionKernels::new");

    // (batch=1, axis_size) flat-sort shapes. Cover bn∈{32,64,128}.
    // _gather_sort sorts an [N*top_k] flat array. For Qwen3-MoE
    // (top_k=8) we hit:
    //   N*K ∈ {8, 16, 64, 128, 256, 512} → bn ∈ {32, 64, 128}.
    for &ax in &[8usize, 16, 32, 64, 128, 256, 512] {
        // Stage 1: sort expert indices (small u32 values in
        // [0, num_experts)). Use a deterministic mix so ties are rare.
        let num_experts = 60u32;
        let input: Vec<u32> = (0..ax)
            .map(|i| ((i.wrapping_mul(2654435761)) as u32) % num_experts)
            .collect();
        let in_buf = upload(&device, &input);
        let out_buf = upload(&device, &vec![0u32; ax]);
        dispatch_argpartition(
            &kernels,
            ArgSortDtype::U32,
            &queue,
            &in_buf,
            &out_buf,
            1,
            ax as u32,
            4,
        )
        .expect("dispatch_argpartition u32 stage1");
        let got: Vec<u32> = download(&out_buf, ax);
        let want = argsort_cpu_u32(&input, 1, ax);
        // Stable order on ties may diverge between merge-sort and a
        // stable std-sort; check the inputs the argsort selects rather
        // than positions, but use a hash so the assertion still has
        // teeth — sorted-value sequences must match exactly.
        let got_vals: Vec<u32> = got.iter().map(|&i| input[i as usize]).collect();
        let want_vals: Vec<u32> = want.iter().map(|&i| input[i as usize]).collect();
        assert_eq!(
            got_vals, want_vals,
            "u32 stage1 mismatch ax={ax}"
        );

        // Stage 2: sort `got` (which is a permutation, hence distinct
        // values) — full-row exact match.
        let order_buf = upload(&device, &got);
        let inv_buf = upload(&device, &vec![0u32; ax]);
        dispatch_argpartition(
            &kernels,
            ArgSortDtype::U32,
            &queue,
            &order_buf,
            &inv_buf,
            1,
            ax as u32,
            4,
        )
        .expect("dispatch_argpartition u32 stage2");
        let inv_got: Vec<u32> = download(&inv_buf, ax);
        let inv_want = argsort_cpu_u32(&got, 1, ax);
        assert_argsort_exact_u32(&inv_got, &inv_want, 1, ax);
    }
}

#[test]
fn argpartition_f16_router_shapes() {
    let device = match detect_device() {
        Some(d) => d.device,
        None => return,
    };
    let queue = device.newCommandQueue().expect("queue");
    let kernels = ArgPartitionKernels::new(&device).expect("ArgPartitionKernels::new");

    let (batch, ax, k) = (32usize, 128usize, 8usize);
    let input_f32 = deterministic_distinct(batch * ax);
    let input_f16: Vec<u16> = input_f32
        .iter()
        .map(|x| half::f16::from_f32(*x).to_bits())
        .collect();
    let input_via_f16: Vec<f32> = input_f16
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();

    let in_buf = upload(&device, &input_f16);
    let out_buf = upload(&device, &vec![0u32; input_f16.len()]);
    dispatch_argpartition(
        &kernels,
        ArgSortDtype::F16,
        &queue,
        &in_buf,
        &out_buf,
        batch as u32,
        ax as u32,
        2,
    )
    .expect("dispatch_argpartition f16");
    let got: Vec<u32> = download(&out_buf, input_f16.len());
    let want = argsort_cpu_f32(&input_via_f16, batch, ax);
    assert_topk_matches(&got, &want, &input_via_f16, batch, ax, k);
}
