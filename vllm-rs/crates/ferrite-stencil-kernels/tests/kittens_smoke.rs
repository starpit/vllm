// SPDX-License-Identifier: Apache-2.0
//! # H100 smoke tests for the kittens-based megakernel FFI.
//!
//! PLAN: compiler emits kittens primitives, static schedule, no VM.
//! Goal: vllm chat on H100. These tests validate the full pipeline
//! (emit_kittens → nvcc → libkittens_kernels.a → Rust FFI → kernel
//! launch) against CPU references.
//!
//! Must run on sm_90a+ hardware — kittens::tma::load_async and
//! warpgroup::mma_async are Hopper-only. On lower arches
//! `libkittens_kernels.a` doesn't exist; the test is gated out by
//! the `kittens_available` helper.
//!
//! Usage on H100:
//!   cargo test -p ferrite-stencil-kernels --features cuda \
//!     --test kittens_smoke -- --test-threads=1
//!
//! `ferrite-cuda-builder`'s build.rs must have run on the same box
//! (or one with matching GPU arch + THUNDERKITTENS_ROOT) first so
//! the static archive is present in the cudaforge cache.

#![cfg(all(feature = "cuda", kittens_linked))]

use cudarc::driver::result;
use cudarc::driver::sys;
use half::bf16;

use ferrite_stencil_kernels::kittens;

fn init_cuda() {
    result::init().expect("cuInit");
    let dev = result::device::get(0).expect("cuDeviceGet");
    let ctx = unsafe { result::primary_ctx::retain(dev) }.expect("cuCtxRetain");
    unsafe { result::ctx::set_current(ctx) }.expect("cuCtxSetCurrent");
}

/// Returns true if the current device is sm_90a+. Tests skip (pass
/// trivially) on lower arches so the suite stays green on the L4
/// dev box.
fn kittens_available() -> bool {
    init_cuda();
    let dev = result::device::get(0).expect("cuDeviceGet");
    let mut major: i32 = 0;
    let mut minor: i32 = 0;
    unsafe {
        sys::cuDeviceGetAttribute(
            &mut major,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            dev,
        );
        sys::cuDeviceGetAttribute(
            &mut minor,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            dev,
        );
    }
    major >= 9
}

fn gpu_alloc_copy_bf16(host: &[bf16]) -> u64 {
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

fn gpu_copy_to_host_bf16(dptr: u64, host: &mut [bf16]) {
    let bytes = std::mem::size_of_val(host);
    let rc = unsafe { sys::cuMemcpyDtoH_v2(host.as_mut_ptr() as *mut _, dptr, bytes) };
    assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "d2h: {rc:?}");
}

fn cpu_rmsnorm(x: &[bf16], w: &[bf16], y: &mut [bf16], num_tokens: usize, d: usize, eps: f32) {
    for t in 0..num_tokens {
        let row = &x[t * d..(t + 1) * d];
        let sq: f32 = row.iter().map(|v| v.to_f32() * v.to_f32()).sum();
        let rsqrt = 1.0 / ((sq / d as f32) + eps).sqrt();
        for i in 0..d {
            y[t * d + i] = bf16::from_f32(row[i].to_f32() * rsqrt * w[i].to_f32());
        }
    }
}

fn cpu_gemm(a: &[bf16], b: &[bf16], c: &mut [bf16], m: usize, n: usize, k: usize) {
    // c = a @ b. a row-major [m, k], b row-major [k, n], c [m, n].
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p].to_f32() * b[p * n + j].to_f32();
            }
            c[i * n + j] = bf16::from_f32(acc);
        }
    }
}

#[test]
fn gemm_matches_cpu_reference() {
    if !kittens_available() {
        eprintln!("skipping: device is not sm_90a+");
        return;
    }

    // Emitted kernel uses TM=64, TN=128, TK=64. Pick small dims that
    // match: M=64, N=128, K=64 — a single CTA, single k_tile. Larger
    // runs work too but take longer; this is a smoke test.
    const M: usize = 64;
    const N: usize = 128;
    const K: usize = 64;

    // Deterministic small-value inputs so f32 accumulation doesn't
    // lose precision too fast in bf16.
    let a: Vec<bf16> = (0..M * K)
        .map(|i| bf16::from_f32(((i as f32) * 0.001).sin() * 0.1))
        .collect();
    let b: Vec<bf16> = (0..K * N)
        .map(|i| bf16::from_f32(((i as f32) * 0.002).cos() * 0.1))
        .collect();

    let mut c_ref = vec![bf16::from_f32(0.0); M * N];
    cpu_gemm(&a, &b, &mut c_ref, M, N, K);

    let d_a = gpu_alloc_copy_bf16(&a);
    let d_b = gpu_alloc_copy_bf16(&b);
    let d_c = gpu_alloc_zeros(M * N * std::mem::size_of::<bf16>());

    let rc = unsafe { kittens::gemm(0, d_a, d_b, d_c, M as u32, N as u32, K as u32) };
    assert_eq!(rc, 0, "kittens::gemm returned cudaError_t={rc}");
    unsafe { sys::cuCtxSynchronize() };

    let mut c_gpu = vec![bf16::from_f32(0.0); M * N];
    gpu_copy_to_host_bf16(d_c, &mut c_gpu);

    // Tolerance for bf16 WGMMA vs f32 CPU: the accumulator inside
    // WGMMA is f32, so small absolute error; but the output is bf16
    // so there's quantization on write. 1% relative or 1e-2 absolute
    // should be plenty for these K=64 small-value dot products.
    let mut max_err = 0.0f32;
    for (r, g) in c_ref.iter().zip(c_gpu.iter()) {
        let e = (r.to_f32() - g.to_f32()).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(
        max_err < 1e-2,
        "gemm output mismatch: max abs error = {max_err}",
    );

    unsafe {
        let _ = result::free_sync(d_a);
        let _ = result::free_sync(d_b);
        let _ = result::free_sync(d_c);
    }
}

#[test]
fn rmsnorm_matches_cpu_reference() {
    if !kittens_available() {
        eprintln!("skipping: device is not sm_90a+");
        return;
    }

    const NUM_TOKENS: usize = 4;
    const D: usize = 4096; // must match the hardcoded `D` in emit_kittens.
    const EPS: f32 = 1e-6;

    // Deterministic inputs: x = sin(i/D), w = 1.0.
    let x: Vec<bf16> = (0..NUM_TOKENS * D)
        .map(|i| bf16::from_f32(((i as f32) / (D as f32)).sin() * 0.1))
        .collect();
    let w: Vec<bf16> = vec![bf16::from_f32(1.0); D];

    let mut y_ref = vec![bf16::from_f32(0.0); NUM_TOKENS * D];
    cpu_rmsnorm(&x, &w, &mut y_ref, NUM_TOKENS, D, EPS);

    // Launch kittens kernel.
    let d_x = gpu_alloc_copy_bf16(&x);
    let d_w = gpu_alloc_copy_bf16(&w);
    let d_y = gpu_alloc_zeros(NUM_TOKENS * D * std::mem::size_of::<bf16>());

    let rc = unsafe {
        kittens::rmsnorm(
            0, // default stream
            d_x,
            d_w,
            d_y,
            NUM_TOKENS as u32,
            D as u32,
            EPS,
        )
    };
    assert_eq!(rc, 0, "kittens::rmsnorm returned cudaError_t={rc}");
    unsafe { sys::cuCtxSynchronize() };

    let mut y_gpu = vec![bf16::from_f32(0.0); NUM_TOKENS * D];
    gpu_copy_to_host_bf16(d_y, &mut y_gpu);

    // Tolerance: bf16 has 7 fractional bits, so output magnitudes
    // near 1.0 carry ~2^-7 ≈ 0.008 quantization noise; a 4096-element
    // reduction adds another factor. Empirically the GPU/CPU delta
    // sits around 0.016 (= 2 bf16 ULPs at the output magnitude), so
    // 4e-2 leaves headroom without masking real divergence.
    let mut max_err = 0.0f32;
    for (a, b) in y_ref.iter().zip(y_gpu.iter()) {
        let e = (a.to_f32() - b.to_f32()).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(
        max_err < 4e-2,
        "rmsnorm output mismatch: max abs error = {max_err}",
    );

    // Cleanup.
    unsafe {
        let _ = result::free_sync(d_x);
        let _ = result::free_sync(d_w);
        let _ = result::free_sync(d_y);
    }
}
