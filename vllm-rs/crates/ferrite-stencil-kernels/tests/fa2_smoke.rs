// SPDX-License-Identifier: Apache-2.0
//! Correctness test for the step-3b SM89 smoke kernel.
//!
//! Generates Q/K/V in host memory, computes the expected attention
//! output with a simple CPU reference (scalar FA2), launches the
//! generated kernel on device 0, copies the output back, and
//! compares. Tolerances are loose on purpose — bf16 softmax is
//! noisy and the reference uses f32 throughout.
//!
//! Run with:
//!   cargo test -p ferrite-stencil-kernels --features cuda \
//!     -- --test-threads=1

#![cfg(all(feature = "cuda", stencil_linked))]

use cudarc::driver::result;
use cudarc::driver::sys;
use half::bf16;

use ferrite_stencil_kernels::{SMOKE_HD, SMOKE_SEQ_K, SMOKE_SEQ_Q, launch_smoke_sm89};

fn init_cuda() {
    result::init().expect("cuInit");
    let dev = result::device::get(0).expect("cuDeviceGet");
    let ctx = unsafe { result::primary_ctx::retain(dev) }.expect("cuCtxRetain");
    unsafe { result::ctx::set_current(ctx) }.expect("cuCtxSetCurrent");
}

fn gpu_alloc_copy(host: &[bf16]) -> u64 {
    let bytes = std::mem::size_of_val(host);
    let dptr = unsafe { result::malloc_sync(bytes) }.expect("cuMemAlloc");
    let rc = unsafe { sys::cuMemcpyHtoD_v2(dptr, host.as_ptr() as *const _, bytes) };
    assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "h2d: {rc:?}");
    dptr
}

fn gpu_alloc_zeros(bytes: usize) -> u64 {
    let dptr = unsafe { result::malloc_sync(bytes) }.expect("cuMemAlloc");
    unsafe { result::memset_d8_sync(dptr, 0, bytes) }.expect("cuMemsetD8");
    dptr
}

fn gpu_copy_to_host(dptr: u64, host: &mut [bf16]) {
    let bytes = std::mem::size_of_val(host);
    let rc = unsafe { sys::cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, dptr, bytes) };
    assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "d2h: {rc:?}");
}

fn cpu_fa2_reference(q: &[bf16], k: &[bf16], v: &[bf16], o: &mut [bf16]) {
    // Q: [SEQ_Q, HD], K/V: [SEQ_K, HD], O: [SEQ_Q, HD]. Scalar
    // softmax(Q @ K^T) @ V. All arithmetic in f32.
    for i in 0..SMOKE_SEQ_Q {
        let mut s = vec![0f32; SMOKE_SEQ_K];
        for j in 0..SMOKE_SEQ_K {
            let mut acc = 0f32;
            for d in 0..SMOKE_HD {
                acc += q[i * SMOKE_HD + d].to_f32() * k[j * SMOKE_HD + d].to_f32();
            }
            s[j] = acc;
        }
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p = s.iter().map(|v| (v - m).exp()).collect::<Vec<_>>();
        let sum: f32 = p.iter().sum();
        for x in &mut p {
            *x /= sum;
        }
        for d in 0..SMOKE_HD {
            let mut acc = 0f32;
            for j in 0..SMOKE_SEQ_K {
                acc += p[j] * v[j * SMOKE_HD + d].to_f32();
            }
            o[i * SMOKE_HD + d] = bf16::from_f32(acc);
        }
    }
}

#[test]
fn smoke_kernel_matches_cpu_reference() {
    init_cuda();

    // Deterministic seed — no randomness, fixed values so any delta
    // surfaces in the diff.
    let q: Vec<bf16> = (0..SMOKE_SEQ_Q * SMOKE_HD)
        .map(|i| bf16::from_f32(((i % 13) as f32 - 6.0) * 0.125))
        .collect();
    let k: Vec<bf16> = (0..SMOKE_SEQ_K * SMOKE_HD)
        .map(|i| bf16::from_f32(((i % 11) as f32 - 5.0) * 0.0625))
        .collect();
    let v: Vec<bf16> = (0..SMOKE_SEQ_K * SMOKE_HD)
        .map(|i| bf16::from_f32(((i % 7) as f32 - 3.0) * 0.25))
        .collect();

    let mut expected = vec![bf16::from_f32(0.0); SMOKE_SEQ_Q * SMOKE_HD];
    cpu_fa2_reference(&q, &k, &v, &mut expected);

    let dq = gpu_alloc_copy(&q);
    let dk = gpu_alloc_copy(&k);
    let dv = gpu_alloc_copy(&v);
    let do_ = gpu_alloc_zeros(SMOKE_SEQ_Q * SMOKE_HD * 2);

    unsafe { launch_smoke_sm89(dq, dk, dv, do_) }.expect("launch succeeds");

    let mut actual = vec![bf16::from_f32(0.0); SMOKE_SEQ_Q * SMOKE_HD];
    gpu_copy_to_host(do_, &mut actual);

    for ptr in [dq, dk, dv, do_] {
        unsafe { sys::cuMemFree_v2(ptr) };
    }

    // bf16 softmax × bf16 matmul on scalar paths: ~1e-2 absolute is a
    // generous upper bound on the accumulated error. Tighten once 3d
    // moves to tensor-core accumulation.
    let atol = 5e-2_f32;
    let rtol = 5e-2_f32;
    let mut max_err: f32 = 0.0;
    for i in 0..actual.len() {
        let a = actual[i].to_f32();
        let e = expected[i].to_f32();
        let diff = (a - e).abs();
        let tol = atol + rtol * e.abs();
        max_err = max_err.max(diff);
        assert!(
            diff <= tol,
            "mismatch at [{}]: actual={:.5} expected={:.5} diff={:.5} tol={:.5}",
            i,
            a,
            e,
            diff,
            tol
        );
    }
    eprintln!("max abs err = {max_err:.5}");
}
