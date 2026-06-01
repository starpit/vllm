// SPDX-License-Identifier: Apache-2.0
//! Ferrite-side host launcher for the orchestrator-emitted megakernel.
//!
//! `tk_codegen::emit_kernel` emits a paired `__global__ void <name>(...)`
//! kernel and a C-linkage host wrapper
//! `extern "C" cudaError_t launch_<name>(void* const* bufs, const
//! uint32_t* u32_args, cudaStream_t stream)`. The wrapper:
//!   - sets `cudaFuncAttributeMaxDynamicSharedMemorySize` to
//!     `NUM_PAGES * PAGE_SIZE`,
//!   - launches with `<<<1, total_threads, DYN_SMEM, stream>>>`,
//!   - forwards each `bufs[i]` cast to the declared kernel arg type
//!     and each `u32_args[j]` to the corresponding runtime u32.
//!
//! This module exposes a Rust FFI binding to that wrapper. Linking is
//! provided by the `cuda` feature (which pulls in `ferrite-cuda-builder`,
//! whose `build.rs` emits `rustc-link-lib=static=megakernels`).
//!
//! No vllm `CudaWorker`, no `CandleWorker`. The path is:
//!   ferrite-wavefront → libmegakernels.a → CUDA driver.

use std::ffi::c_void;

// FFI signature for the orchestrator's emitted launcher.
//
// `bufs` MUST be a contiguous array of device pointers in the same
// order the orchestrator declared them (sources first, then per-op
// output staging buffers). `u32_args` carries the runtime u32 args
// the kernel signature lists in `KernelArgs::u32_args` (today only
// `__num_kv_pages` for AttnDecode-bearing programs).
//
// Returns the underlying `cudaError_t` (0 = `cudaSuccess`).
unsafe extern "C" {
    pub fn launch_tk_decode_one_layer(
        bufs: *const *mut c_void,
        u32_args: *const u32,
        stream: *mut c_void,
    ) -> i32;

    /// Single-op RmsNorm reproducer (emitted by
    /// `bin/tk_emit_rmsnorm`). Same C ABI as
    /// [`launch_tk_decode_one_layer`]; isolates the round protocol of
    /// one op from the 12-op forward so the deadlock audit can pin the
    /// offending wait/arrive imbalance to RmsNorm specifically.
    pub fn launch_tk_decode_rmsnorm(
        bufs: *const *mut c_void,
        u32_args: *const u32,
        stream: *mut c_void,
    ) -> i32;

    /// Single-op Add reproducer (Phase 6 tile-golden harness; emitted
    /// by `bin/tk_emit_add`).
    pub fn launch_tk_decode_add(
        bufs: *const *mut c_void,
        u32_args: *const u32,
        stream: *mut c_void,
    ) -> i32;

    /// Single-op SiluMul reproducer (emitted by `bin/tk_emit_silu_mul`).
    pub fn launch_tk_decode_silu_mul(
        bufs: *const *mut c_void,
        u32_args: *const u32,
        stream: *mut c_void,
    ) -> i32;

    /// Single-op RopeRotate reproducer (emitted by
    /// `bin/tk_emit_rope_rotate`). Takes one runtime u32 arg —
    /// `__decode_position` — for cos/sin row indexing.
    pub fn launch_tk_decode_rope_rotate(
        bufs: *const *mut c_void,
        u32_args: *const u32,
        stream: *mut c_void,
    ) -> i32;

    /// Single-op M=1 GEMM reproducer (emitted by `bin/tk_emit_gemm_m1`).
    pub fn launch_tk_decode_gemm_m1(
        bufs: *const *mut c_void,
        u32_args: *const u32,
        stream: *mut c_void,
    ) -> i32;
}

/// Safe wrapper around [`launch_tk_decode_one_layer`].
///
/// `bufs[i]` must point to valid device memory whose layout matches
/// the source / op-output buffer at position `i` in
/// [`crate::fixtures::one_layer_input`]. `stream` is a `CUstream` /
/// `cudaStream_t` cast to `*mut c_void` (the C ABI is identical).
///
/// # Safety
/// Caller must ensure all `bufs` pointers are valid device-side, the
/// stream belongs to the active CUDA context, and the device has at
/// least `NUM_PAGES * PAGE_SIZE` bytes of dynamic shared memory
/// available (true for any sm_90 device).
pub unsafe fn launch_decode_one_layer(
    bufs: &[*mut c_void],
    u32_args: &[u32],
    stream: *mut c_void,
) -> i32 {
    unsafe { launch_tk_decode_one_layer(bufs.as_ptr(), u32_args.as_ptr(), stream) }
}

/// Safe wrapper around [`launch_tk_decode_rmsnorm`].
///
/// `bufs[i]` must point to valid device memory whose layout matches
/// the source / op-output buffer at position `i` in
/// [`crate::fixtures::rmsnorm_only_input`] (3 buffers: x, rms_w, op0).
///
/// # Safety
/// Same contract as [`launch_decode_one_layer`].
pub unsafe fn launch_decode_rmsnorm(
    bufs: &[*mut c_void],
    u32_args: &[u32],
    stream: *mut c_void,
) -> i32 {
    unsafe { launch_tk_decode_rmsnorm(bufs.as_ptr(), u32_args.as_ptr(), stream) }
}

/// Safe wrapper for the Phase 6 single-op Add kernel.
/// `bufs[i]` matches `fixtures::add_only_input` (3 bufs: a, b, op0).
///
/// # Safety
/// Same contract as [`launch_decode_one_layer`].
pub unsafe fn launch_decode_add(
    bufs: &[*mut c_void],
    u32_args: &[u32],
    stream: *mut c_void,
) -> i32 {
    unsafe { launch_tk_decode_add(bufs.as_ptr(), u32_args.as_ptr(), stream) }
}

/// Safe wrapper for the Phase 6 single-op SiluMul kernel.
/// `bufs[i]` matches `fixtures::silu_mul_only_input` (3 bufs: gate,
/// up, op0).
///
/// # Safety
/// Same contract as [`launch_decode_one_layer`].
pub unsafe fn launch_decode_silu_mul(
    bufs: &[*mut c_void],
    u32_args: &[u32],
    stream: *mut c_void,
) -> i32 {
    unsafe { launch_tk_decode_silu_mul(bufs.as_ptr(), u32_args.as_ptr(), stream) }
}

/// Safe wrapper for the Phase 6 single-op RopeRotate kernel.
/// `bufs[i]` matches `fixtures::rope_rotate_only_input` (4 bufs: x,
/// cos, sin, op0). `u32_args[0]` is `__decode_position` (the cos/sin
/// row index).
///
/// # Safety
/// Same contract as [`launch_decode_one_layer`].
pub unsafe fn launch_decode_rope_rotate(
    bufs: &[*mut c_void],
    u32_args: &[u32],
    stream: *mut c_void,
) -> i32 {
    unsafe { launch_tk_decode_rope_rotate(bufs.as_ptr(), u32_args.as_ptr(), stream) }
}

/// Safe wrapper for the Phase 6 single-op M=1 GEMM kernel.
/// `bufs[i]` matches `fixtures::gemm_m1_only_input` (3 bufs: x, w,
/// op0).
///
/// # Safety
/// Same contract as [`launch_decode_one_layer`].
pub unsafe fn launch_decode_gemm_m1(
    bufs: &[*mut c_void],
    u32_args: &[u32],
    stream: *mut c_void,
) -> i32 {
    unsafe { launch_tk_decode_gemm_m1(bufs.as_ptr(), u32_args.as_ptr(), stream) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Forces the linker to resolve `launch_tk_decode_one_layer` —
    /// proves the orchestrator's emitted launcher made it into
    /// `libmegakernels.a` and the build.rs link directives wired the
    /// archive into our test binary. We never invoke the function
    /// (would need real device memory and a stream), but taking its
    /// address forces the symbol to land in the test binary.
    #[test]
    fn launcher_symbol_resolves() {
        let f: unsafe extern "C" fn(*const *mut c_void, *const u32, *mut c_void) -> i32 =
            launch_tk_decode_one_layer;
        assert_ne!(f as usize, 0);
    }

    /// As [`launcher_symbol_resolves`] but for the RmsNorm-only repro
    /// emit. Proves `bin/tk_emit_rmsnorm` produced a kernel whose
    /// host wrapper is now in `libmegakernels.a`.
    #[test]
    fn rmsnorm_launcher_symbol_resolves() {
        let f: unsafe extern "C" fn(*const *mut c_void, *const u32, *mut c_void) -> i32 =
            launch_tk_decode_rmsnorm;
        assert_ne!(f as usize, 0);
    }

    /// End-to-end smoke harness: allocate the full 26-buffer set of
    /// zero-filled device memory, hand them to the orchestrator's
    /// `launch_tk_decode_one_layer` wrapper with `__num_kv_pages = 1`,
    /// and assert the launch + stream sync both report `cudaSuccess`.
    ///
    /// `#[ignore]`d so the test only runs when explicitly invoked
    /// (it requires H100 + the cudaforge cache to have been
    /// populated by `bin/tk_emit_decode` and `ferrite-cuda-builder`).
    /// Verified passing on H100 in 0.65 s after the three round-
    /// protocol fixes:
    ///   1. `Arrive` lowers to `kittens::group<1>::arrive` so each
    ///      consumer warp's lane 0 fires (8 arrives, not 1).
    ///   2. `wait_loop_parity` takes `start_phase` so iter 0 reads
    ///      the page's static phase, not a hardcoded 0 — fixes
    ///      slot reuse mid-forward when a prior op closed the slot
    ///      at the opposite parity (op4/Gemm hang).
    ///   3. `lower_gemm_m1` skips `complete_round` when
    ///      `n_blocks % 2 == 0` so the typed phase tracks the
    ///      runtime barrier parity at the op boundary
    ///      (op5/Add hang post-op4/Gemm).
    ///
    /// Run manually with:
    ///   cargo test -p ferrite-wavefront --features cuda \
    ///       launcher_runs_on_zeros -- --ignored --nocapture
    #[test]
    #[ignore]
    fn launcher_runs_on_zeros() {
        use crate::fixtures::{buf_byte_sizes, one_layer_input};
        use cudarc::driver::{CudaContext, DevicePtr};

        let ctx = CudaContext::new(0).expect("cuda init");
        let stream = ctx.new_stream().expect("stream create");

        let input = one_layer_input();
        let raw_sizes = buf_byte_sizes(&input);

        // Pad each allocation to at least 1 MiB (covers any single-page
        // TMA load at PAGE_SIZE = 16 KiB with comfortable headroom) and
        // round to 128 bytes (TMA alignment). Real shapes are kept where
        // they exceed the floor so the larger weight tensors get their
        // declared bytes.
        let pad_floor = 1usize << 20;
        let bufs: Vec<cudarc::driver::CudaSlice<u8>> = raw_sizes
            .iter()
            .map(|&n| {
                let n_padded = n.max(pad_floor);
                let n_aligned = n_padded.div_ceil(128) * 128;
                stream.alloc_zeros::<u8>(n_aligned).expect("alloc_zeros")
            })
            .collect();

        // Collect device pointers in BufId order. The `_records` keep
        // the SyncOnDrop guards alive for the duration of the launch so
        // cudarc treats the buffers as live across the kernel call.
        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(bufs.len());
        let mut _records: Vec<cudarc::driver::SyncOnDrop<'_>> = Vec::with_capacity(bufs.len());
        for buf in &bufs {
            let (dptr, rec) = DevicePtr::device_ptr(buf, &stream);
            ptrs.push(dptr as *mut c_void);
            _records.push(rec);
        }

        // Make sure all the alloc_zeros memsets retire before the
        // kernel reads them.
        stream.synchronize().expect("pre-launch sync");

        let u32_args: [u32; 1] = [1]; // __num_kv_pages = 1
        let stream_raw = stream.cu_stream() as *mut c_void;

        let err = unsafe { launch_decode_one_layer(&ptrs, &u32_args, stream_raw) };
        assert_eq!(err, 0, "launch_tk_decode_one_layer returned cudaError {err}");

        stream
            .synchronize()
            .expect("post-launch sync (kernel ran to completion)");

        drop(_records);
        drop(bufs);
    }

    /// Single-op (RmsNorm) reproducer for the deadlock audit. Allocates
    /// the 3-buffer set (x, rms_w, op0_out) of zero-filled device memory
    /// and launches `tk_decode_rmsnorm` (no `__num_kv_pages` arg —
    /// RmsNorm has no AttnDecode child).
    ///
    /// `#[ignore]`d for the same reason as
    /// [`launcher_runs_on_zeros`] — until the round-protocol bug is
    /// fixed it deadlocks. With `TK_EMIT_DEBUG_HANDSHAKE=1` set when
    /// `bin/tk_emit_rmsnorm` was last run, the kernel emits one printf
    /// per loader/consumer/storer wait+arrive; the first WAIT_START
    /// without a matching prior ARRIVE/TMA_LOAD_ISSUED is the bug.
    ///
    /// Run manually with:
    ///   cargo run -p ferrite-wavefront --bin tk_emit_rmsnorm \
    ///       (optional: TK_EMIT_DEBUG_HANDSHAKE=1)
    ///   cargo build -p ferrite-cuda-builder --features cuda
    ///   cargo test -p ferrite-wavefront --features cuda \
    ///       launcher_runs_on_zeros_rmsnorm_only -- --ignored --nocapture
    #[test]
    #[ignore]
    fn launcher_runs_on_zeros_rmsnorm_only() {
        use crate::fixtures::{buf_byte_sizes, rmsnorm_only_input};
        use cudarc::driver::{CudaContext, DevicePtr};

        let ctx = CudaContext::new(0).expect("cuda init");
        let stream = ctx.new_stream().expect("stream create");

        let input = rmsnorm_only_input();
        let raw_sizes = buf_byte_sizes(&input);
        // Sanity: 2 sources + 1 op output = 3 buffers.
        assert_eq!(raw_sizes.len(), 3);

        let pad_floor = 1usize << 20;
        let bufs: Vec<cudarc::driver::CudaSlice<u8>> = raw_sizes
            .iter()
            .map(|&n| {
                let n_padded = n.max(pad_floor);
                let n_aligned = n_padded.div_ceil(128) * 128;
                stream.alloc_zeros::<u8>(n_aligned).expect("alloc_zeros")
            })
            .collect();

        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(bufs.len());
        let mut _records: Vec<cudarc::driver::SyncOnDrop<'_>> = Vec::with_capacity(bufs.len());
        for buf in &bufs {
            let (dptr, rec) = DevicePtr::device_ptr(buf, &stream);
            ptrs.push(dptr as *mut c_void);
            _records.push(rec);
        }

        stream.synchronize().expect("pre-launch sync");

        // RmsNorm has no AttnDecode child → no `__num_kv_pages` runtime arg.
        let u32_args: [u32; 0] = [];
        let stream_raw = stream.cu_stream() as *mut c_void;

        let err = unsafe { launch_decode_rmsnorm(&ptrs, &u32_args, stream_raw) };
        assert_eq!(
            err, 0,
            "launch_tk_decode_rmsnorm returned cudaError {err}"
        );

        stream
            .synchronize()
            .expect("post-launch sync (rmsnorm-only kernel ran to completion)");

        drop(_records);
        drop(bufs);
    }

    /// RmsNorm correctness — synthetic deterministic bf16 input,
    /// kernel output compared element-wise to
    /// `ferrite_forward::cpu_golden::rmsnorm` after a bf16 round-trip.
    ///
    /// Method:
    ///   1. Generate `x` and `weight` as f32 with a fixed PRNG seed.
    ///   2. Round-trip through bf16 host-side; the kernel reads bf16
    ///      from HBM and converts to f32 for compute, so the host
    ///      reference uses the same lossy bf16-back-to-f32 inputs to
    ///      isolate kernel correctness from input-storage rounding.
    ///   3. H2D the bf16 bytes into the source buffers; allocate a
    ///      zero output buffer.
    ///   4. Launch `tk_decode_rmsnorm`, sync.
    ///   5. D2H the output bytes; convert bf16 back to f32.
    ///   6. Run `cpu_golden::rmsnorm` on the matching f32 inputs.
    ///   7. Assert max-abs error within bf16 tolerance.
    ///
    /// Tolerance: bf16 has 8 mantissa bits → ~0.4% relative error
    /// per output. RmsNorm produces values around the same magnitude
    /// as the input scaled by `weight / rms`. For inputs in
    /// `[-1, 1]` and weights near 1.0, output magnitude is similar,
    /// so a 1.5e-2 absolute tolerance comfortably covers the
    /// bf16 storage rounding of the final write.
    ///
    /// Run manually with:
    ///   cargo run -p ferrite-wavefront --bin tk_emit_rmsnorm
    ///   cargo build -p ferrite-cuda-builder --features cuda
    ///   cargo test -p ferrite-wavefront --features cuda \
    ///       rmsnorm_kernel_matches_cpu_golden -- --ignored --nocapture
    #[test]
    #[ignore]
    fn rmsnorm_kernel_matches_cpu_golden() {
        use crate::fixtures::{buf_byte_sizes, rmsnorm_only_input};
        use cudarc::driver::{CudaContext, DevicePtr};
        use ferrite_forward::cpu_golden;
        use half::bf16;

        // Fixture geometry: 1×2048 hidden, eps = 1e-5.
        let input = rmsnorm_only_input();
        let hidden = input.sources[0].cols as usize;
        assert_eq!(hidden, 2048);
        let eps = match input.ops[0].op {
            crate::lower::LoweredOp::RmsNorm { eps } => eps,
            _ => panic!("rmsnorm_only_input op0 must be RmsNorm"),
        };

        // Deterministic synthetic inputs: a simple Lehmer-style PRNG
        // seeded once. Avoids the `rand` dependency. Range chosen so
        // the rms scale stays well-conditioned (`rms ≈ 0.58`, far from
        // 0 or saturating bf16).
        let mut state = 0x1234_5678_u64;
        let mut next_f32 = || -> f32 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Take 24 high bits, normalise to [-1, 1).
            let bits = ((state >> 40) as u32) & 0x00ff_ffff;
            (bits as f32) / (1u32 << 23) as f32 - 1.0
        };

        let mut x_f32 = vec![0.0_f32; hidden];
        let mut w_f32 = vec![0.0_f32; hidden];
        for i in 0..hidden {
            x_f32[i] = next_f32();
            // Weights centred on 1.0 (typical RmsNorm gain), spread ±0.5.
            w_f32[i] = 1.0 + 0.5 * next_f32();
        }

        // bf16 round-trip the inputs so the host reference uses the
        // exact same lossy values the kernel reads back from HBM.
        let x_bf16: Vec<bf16> = x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let w_bf16: Vec<bf16> = w_f32.iter().map(|&v| bf16::from_f32(v)).collect();
        let x_lossy: Vec<f32> = x_bf16.iter().map(|b| b.to_f32()).collect();
        let w_lossy: Vec<f32> = w_bf16.iter().map(|b| b.to_f32()).collect();

        // Pack bf16 → bytes (little-endian u16 = bf16 bit pattern).
        let to_bytes = |slice: &[bf16]| -> Vec<u8> {
            let mut out = Vec::with_capacity(slice.len() * 2);
            for v in slice {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
            out
        };
        let x_bytes = to_bytes(&x_bf16);
        let w_bytes = to_bytes(&w_bf16);

        let ctx = CudaContext::new(0).expect("cuda init");
        let stream = ctx.new_stream().expect("stream create");

        let raw_sizes = buf_byte_sizes(&input);
        assert_eq!(raw_sizes.len(), 3);

        // Same allocation policy as the smoke test: pad to 1 MiB,
        // align to 128. Real shapes fit inside the floor.
        let pad_floor = 1usize << 20;
        let alloc_size = |n: usize| -> usize { n.max(pad_floor).div_ceil(128) * 128 };

        let buf_x: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[0]))
            .expect("alloc x");
        let buf_w: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[1]))
            .expect("alloc weight");
        let buf_out: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[2]))
            .expect("alloc out");

        // H2D copy the synthetic input data. `memcpy_htod` writes
        // only the prefix we hand it; the padding bytes stay zero
        // and the kernel reads only the tile-shaped prefix anyway.
        let mut buf_x_mut = buf_x;
        let mut buf_w_mut = buf_w;
        stream
            .memcpy_htod(&x_bytes, &mut buf_x_mut)
            .expect("h2d x");
        stream
            .memcpy_htod(&w_bytes, &mut buf_w_mut)
            .expect("h2d w");

        // Collect device pointers in fixture-buffer order: 0 = x,
        // 1 = weight, 2 = op0_out (where the kernel writes the
        // RmsNorm result).
        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(3);
        let mut _records: Vec<cudarc::driver::SyncOnDrop<'_>> = Vec::with_capacity(3);
        for buf in [&buf_x_mut, &buf_w_mut, &buf_out] {
            let (dptr, rec) = DevicePtr::device_ptr(buf, &stream);
            ptrs.push(dptr as *mut c_void);
            _records.push(rec);
        }

        stream.synchronize().expect("pre-launch sync");

        let u32_args: [u32; 0] = [];
        let stream_raw = stream.cu_stream() as *mut c_void;
        let err = unsafe { launch_decode_rmsnorm(&ptrs, &u32_args, stream_raw) };
        assert_eq!(err, 0, "launch_tk_decode_rmsnorm cudaError {err}");
        stream.synchronize().expect("post-launch sync");

        // D2H the full padded device buffer; the RmsNorm output
        // occupies only the first `hidden * 2` bytes — the rest is
        // pad (the kernel never writes past the output tile). cudarc's
        // `memcpy_dtoh` requires `dst.len() >= src.len()`, so we
        // allocate to match the buffer size and slice the prefix.
        let mut out_bytes = vec![0u8; buf_out.len()];
        stream
            .memcpy_dtoh(&buf_out, &mut out_bytes)
            .expect("d2h out");
        stream.synchronize().expect("post-d2h sync");

        let mut got_f32 = vec![0.0_f32; hidden];
        for i in 0..hidden {
            let bits = u16::from_le_bytes([out_bytes[2 * i], out_bytes[2 * i + 1]]);
            got_f32[i] = bf16::from_bits(bits).to_f32();
        }

        // Host reference using the bf16-rounded inputs.
        let mut want_f32 = vec![0.0_f32; hidden];
        cpu_golden::rmsnorm(&x_lossy, &w_lossy, &mut want_f32, eps);
        // Round the reference through bf16 too — the kernel stores
        // its result as bf16, so the comparison must apply the same
        // output rounding to keep the tolerance honest.
        for v in &mut want_f32 {
            *v = bf16::from_f32(*v).to_f32();
        }

        // Element-wise check.
        let mut max_abs = 0.0_f32;
        let mut max_idx = 0usize;
        for i in 0..hidden {
            let diff = (got_f32[i] - want_f32[i]).abs();
            if diff > max_abs {
                max_abs = diff;
                max_idx = i;
            }
        }
        let tol = 1.5e-2_f32;
        assert!(
            max_abs <= tol,
            "rmsnorm max-abs error {max_abs} > tol {tol} at i={max_idx} \
             (got {} vs want {})",
            got_f32[max_idx],
            want_f32[max_idx]
        );

        drop(_records);
        drop(buf_x_mut);
        drop(buf_w_mut);
        drop(buf_out);
    }

    // ── Phase 6 tile-golden harness ─────────────────────────────────
    //
    // Shared utilities + per-IType `<op>_kernel_matches_cpu_golden`
    // tests for the four single-op fixtures emitted by `bin/tk_emit_*`.
    // The pattern mirrors `rmsnorm_kernel_matches_cpu_golden` above:
    // PRNG-seeded f32 inputs → bf16 round-trip → H2D → launch → D2H →
    // compare to `cpu_golden::*` within bf16 tolerance (1.5e-2 abs).
    //
    // Run after the matching `tk_emit_<op>` bin produces the .cu and
    // ferrite-cuda-builder compiles libmegakernels.a:
    //   cargo run -p ferrite-wavefront --bin tk_emit_<op>
    //   cargo build -p ferrite-cuda-builder --features cuda
    //   cargo test -p ferrite-wavefront --features cuda \
    //       <op>_kernel_matches_cpu_golden -- --ignored --nocapture

    /// Lehmer PRNG → f32 in `[-1, 1)`. Deterministic across runs (seed
    /// fixed). Produces well-conditioned inputs (away from saturating
    /// bf16, away from underflow on subsequent compute).
    fn prng_f32_seeded(seed: u64, n: usize, scale: f32, offset: f32) -> Vec<f32> {
        let mut state = seed;
        let mut next = || -> f32 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = ((state >> 40) as u32) & 0x00ff_ffff;
            (bits as f32) / (1u32 << 23) as f32 - 1.0
        };
        (0..n).map(|_| offset + scale * next()).collect()
    }

    /// `f32 → bf16 → bytes`. The kernel reads bf16 from HBM; the host
    /// reference uses the same lossy round-tripped values.
    fn f32_to_bf16_bytes(values: &[f32]) -> (Vec<half::bf16>, Vec<f32>, Vec<u8>) {
        let bf16_vec: Vec<half::bf16> = values.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let lossy_f32: Vec<f32> = bf16_vec.iter().map(|b| b.to_f32()).collect();
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for v in &bf16_vec {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        (bf16_vec, lossy_f32, bytes)
    }

    /// `bytes → bf16 → f32`. Reads the first `n` bf16 elements out of a
    /// padded device buffer.
    fn bytes_to_f32_prefix(bytes: &[u8], n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; n];
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
            out[i] = half::bf16::from_bits(bits).to_f32();
        }
        out
    }

    /// Round host-reference f32 results through bf16 — matches the
    /// kernel's bf16 store, so the comparison applies the same output
    /// rounding to keep tolerance honest.
    fn round_through_bf16_inplace(v: &mut [f32]) {
        for x in v {
            *x = half::bf16::from_f32(*x).to_f32();
        }
    }

    /// Element-wise max-abs check. Panics with diagnostic on first
    /// element exceeding `tol`.
    fn assert_max_abs(got: &[f32], want: &[f32], tol: f32, label: &str) {
        assert_eq!(got.len(), want.len());
        let (mut max_abs, mut max_idx) = (0.0_f32, 0usize);
        for i in 0..got.len() {
            let diff = (got[i] - want[i]).abs();
            if diff > max_abs {
                max_abs = diff;
                max_idx = i;
            }
        }
        assert!(
            max_abs <= tol,
            "{label}: max-abs {max_abs} > tol {tol} at i={max_idx} (got {} vs want {})",
            got[max_idx],
            want[max_idx]
        );
    }

    /// `out = a + b` element-wise. Synthetic inputs in `[-1, 1)`.
    /// Output magnitude ≤ 2; bf16 storage error per element ≤ 7.8e-3,
    /// so 1.5e-2 is comfortable.
    #[test]
    #[ignore]
    fn add_kernel_matches_cpu_golden() {
        use crate::fixtures::{add_only_input, buf_byte_sizes};
        use cudarc::driver::{CudaContext, DevicePtr};
        use ferrite_forward::cpu_golden;

        let input = add_only_input();
        let hidden = input.sources[0].cols as usize;

        let a_f32 = prng_f32_seeded(0x1234_5678, hidden, 1.0, 0.0);
        let b_f32 = prng_f32_seeded(0x9abc_def0, hidden, 1.0, 0.0);
        let (_, a_lossy, a_bytes) = f32_to_bf16_bytes(&a_f32);
        let (_, b_lossy, b_bytes) = f32_to_bf16_bytes(&b_f32);

        let ctx = CudaContext::new(0).expect("cuda init");
        let stream = ctx.new_stream().expect("stream create");
        let raw_sizes = buf_byte_sizes(&input);
        assert_eq!(raw_sizes.len(), 3);
        let pad_floor = 1usize << 20;
        let alloc_size = |n: usize| -> usize { n.max(pad_floor).div_ceil(128) * 128 };

        let buf_a: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[0]))
            .expect("alloc a");
        let buf_b: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[1]))
            .expect("alloc b");
        let buf_out: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[2]))
            .expect("alloc out");

        let mut buf_a_mut = buf_a;
        let mut buf_b_mut = buf_b;
        stream.memcpy_htod(&a_bytes, &mut buf_a_mut).expect("h2d a");
        stream.memcpy_htod(&b_bytes, &mut buf_b_mut).expect("h2d b");

        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(3);
        let mut _records: Vec<cudarc::driver::SyncOnDrop<'_>> = Vec::with_capacity(3);
        for buf in [&buf_a_mut, &buf_b_mut, &buf_out] {
            let (dptr, rec) = DevicePtr::device_ptr(buf, &stream);
            ptrs.push(dptr as *mut c_void);
            _records.push(rec);
        }
        stream.synchronize().expect("pre-launch sync");

        let u32_args: [u32; 0] = [];
        let err = unsafe { launch_decode_add(&ptrs, &u32_args, stream.cu_stream() as *mut c_void) };
        assert_eq!(err, 0, "launch_tk_decode_add cudaError {err}");
        stream.synchronize().expect("post-launch sync");

        let mut out_bytes = vec![0u8; buf_out.len()];
        stream.memcpy_dtoh(&buf_out, &mut out_bytes).expect("d2h out");
        stream.synchronize().expect("post-d2h sync");

        let got = bytes_to_f32_prefix(&out_bytes, hidden);
        let mut want = vec![0.0_f32; hidden];
        cpu_golden::add(&a_lossy, &b_lossy, &mut want);
        round_through_bf16_inplace(&mut want);
        assert_max_abs(&got, &want, 1.5e-2, "add");
    }

    /// `out = silu(gate) * up`. Inputs in `[-1, 1)`; SiLU saturates
    /// gracefully, so the magnitude stays ≲ 1, bf16 store error ≤ 4e-3.
    #[test]
    #[ignore]
    fn silu_mul_kernel_matches_cpu_golden() {
        use crate::fixtures::{buf_byte_sizes, silu_mul_only_input};
        use cudarc::driver::{CudaContext, DevicePtr};
        use ferrite_forward::cpu_golden;

        let input = silu_mul_only_input();
        let intermediate = input.sources[0].cols as usize;

        let g_f32 = prng_f32_seeded(0x1111_2222, intermediate, 1.0, 0.0);
        let u_f32 = prng_f32_seeded(0x3333_4444, intermediate, 1.0, 0.0);
        let (_, g_lossy, g_bytes) = f32_to_bf16_bytes(&g_f32);
        let (_, u_lossy, u_bytes) = f32_to_bf16_bytes(&u_f32);

        let ctx = CudaContext::new(0).expect("cuda init");
        let stream = ctx.new_stream().expect("stream create");
        let raw_sizes = buf_byte_sizes(&input);
        assert_eq!(raw_sizes.len(), 3);
        let pad_floor = 1usize << 20;
        let alloc_size = |n: usize| -> usize { n.max(pad_floor).div_ceil(128) * 128 };

        let buf_g: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[0]))
            .expect("alloc gate");
        let buf_u: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[1]))
            .expect("alloc up");
        let buf_out: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[2]))
            .expect("alloc out");

        let mut buf_g_mut = buf_g;
        let mut buf_u_mut = buf_u;
        stream.memcpy_htod(&g_bytes, &mut buf_g_mut).expect("h2d gate");
        stream.memcpy_htod(&u_bytes, &mut buf_u_mut).expect("h2d up");

        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(3);
        let mut _records: Vec<cudarc::driver::SyncOnDrop<'_>> = Vec::with_capacity(3);
        for buf in [&buf_g_mut, &buf_u_mut, &buf_out] {
            let (dptr, rec) = DevicePtr::device_ptr(buf, &stream);
            ptrs.push(dptr as *mut c_void);
            _records.push(rec);
        }
        stream.synchronize().expect("pre-launch sync");

        let u32_args: [u32; 0] = [];
        let err = unsafe {
            launch_decode_silu_mul(&ptrs, &u32_args, stream.cu_stream() as *mut c_void)
        };
        assert_eq!(err, 0, "launch_tk_decode_silu_mul cudaError {err}");
        stream.synchronize().expect("post-launch sync");

        let mut out_bytes = vec![0u8; buf_out.len()];
        stream.memcpy_dtoh(&buf_out, &mut out_bytes).expect("d2h out");
        stream.synchronize().expect("post-d2h sync");

        let got = bytes_to_f32_prefix(&out_bytes, intermediate);
        // cpu_golden has no fused silu_mul; compose silu + mul.
        let mut silu_g = vec![0.0_f32; intermediate];
        cpu_golden::silu(&g_lossy, &mut silu_g);
        let mut want = vec![0.0_f32; intermediate];
        cpu_golden::mul(&silu_g, &u_lossy, &mut want);
        round_through_bf16_inplace(&mut want);
        assert_max_abs(&got, &want, 1.5e-2, "silu_mul");
    }

    /// NeoX RoPE rotate at decode position 0 (cos[0]=1, sin[0]=0 →
    /// identity rotation; the test still exercises the full lane
    /// decompose + cp.async cos/sin path). For non-trivial rotation,
    /// pass `__decode_position` > 0; today we test the simplest case
    /// to isolate the kernel's pair-decompose math from the runtime
    /// position offset (which the kernel adds to the cos/sin TMA load
    /// source pointer).
    #[test]
    #[ignore]
    fn rope_rotate_kernel_matches_cpu_golden() {
        use crate::fixtures::{buf_byte_sizes, rope_rotate_only_input};
        use cudarc::driver::{CudaContext, DevicePtr};
        use ferrite_forward::cpu_golden;

        let input = rope_rotate_only_input();
        let cols = input.sources[0].cols as usize;
        let head_dim = match input.ops[0].op {
            crate::lower::LoweredOp::RopeRotate { head_dim } => head_dim as usize,
            _ => panic!("rope_rotate_only_input op0 must be RopeRotate"),
        };
        assert_eq!(cols % head_dim, 0);
        let num_heads = cols / head_dim;

        // Q row, cos row (head_dim), sin row (head_dim).
        let x_f32 = prng_f32_seeded(0xa1a1_b2b2, cols, 1.0, 0.0);
        // Use a non-trivial rotation: cos = 0.7, sin = 0.7 (≈ 45°) so
        // both halves contribute. cos[2k] = cos[2k+1] for NeoX style.
        let cos_f32 = vec![0.7_f32; head_dim];
        let sin_f32 = vec![0.7_f32; head_dim];

        let (_, x_lossy, x_bytes) = f32_to_bf16_bytes(&x_f32);
        let (_, cos_lossy, cos_bytes) = f32_to_bf16_bytes(&cos_f32);
        let (_, sin_lossy, sin_bytes) = f32_to_bf16_bytes(&sin_f32);

        let ctx = CudaContext::new(0).expect("cuda init");
        let stream = ctx.new_stream().expect("stream create");
        let raw_sizes = buf_byte_sizes(&input);
        assert_eq!(raw_sizes.len(), 4);
        let pad_floor = 1usize << 20;
        let alloc_size = |n: usize| -> usize { n.max(pad_floor).div_ceil(128) * 128 };

        let buf_x: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[0]))
            .expect("alloc x");
        let buf_c: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[1]))
            .expect("alloc cos");
        let buf_s: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[2]))
            .expect("alloc sin");
        let buf_out: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[3]))
            .expect("alloc out");

        let mut buf_x_mut = buf_x;
        let mut buf_c_mut = buf_c;
        let mut buf_s_mut = buf_s;
        stream.memcpy_htod(&x_bytes, &mut buf_x_mut).expect("h2d x");
        stream.memcpy_htod(&cos_bytes, &mut buf_c_mut).expect("h2d cos");
        stream.memcpy_htod(&sin_bytes, &mut buf_s_mut).expect("h2d sin");

        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(4);
        let mut _records: Vec<cudarc::driver::SyncOnDrop<'_>> = Vec::with_capacity(4);
        for buf in [&buf_x_mut, &buf_c_mut, &buf_s_mut, &buf_out] {
            let (dptr, rec) = DevicePtr::device_ptr(buf, &stream);
            ptrs.push(dptr as *mut c_void);
            _records.push(rec);
        }
        stream.synchronize().expect("pre-launch sync");

        // RoPE-bearing kernels expect `__decode_position` as the only
        // u32 arg. Position 0 → cos/sin row 0 → row pointers untouched.
        let u32_args: [u32; 1] = [0];
        let err = unsafe {
            launch_decode_rope_rotate(&ptrs, &u32_args, stream.cu_stream() as *mut c_void)
        };
        assert_eq!(err, 0, "launch_tk_decode_rope_rotate cudaError {err}");
        stream.synchronize().expect("post-launch sync");

        let mut out_bytes = vec![0u8; buf_out.len()];
        stream.memcpy_dtoh(&buf_out, &mut out_bytes).expect("d2h out");
        stream.synchronize().expect("post-d2h sync");

        let got = bytes_to_f32_prefix(&out_bytes, cols);
        // cpu_golden::rope signature: (qk, cos_table, sin_table,
        // positions: &[i32], seq_len, num_heads, head_dim, output).
        // The kernel reads cos/sin from a single `[head_dim]` source
        // and adds `__decode_position * head_dim_bytes` as a runtime
        // offset; with pos=0 there's no offset, so cos_table and
        // sin_table here are 1-row tables == cos_lossy / sin_lossy.
        let positions: [i32; 1] = [0];
        let mut want = vec![0.0_f32; cols];
        cpu_golden::rope(
            &x_lossy,
            &cos_lossy,
            &sin_lossy,
            &positions,
            1,
            num_heads,
            head_dim,
            &mut want,
        );
        round_through_bf16_inplace(&mut want);
        // Tolerance is wider — RoPE composes two multiplies + one add
        // per output, so bf16 rounding accumulates ~2-3× the per-elem
        // worst-case (4e-3 → ~1.2e-2). 1.5e-2 still comfortable.
        assert_max_abs(&got, &want, 1.5e-2, "rope_rotate");
    }

    /// M=1 GEMM `y = x @ w^T`. K=2048 accumulating bf16 dot products
    /// into f32 — error grows with K. With inputs in `[-1, 1)`,
    /// expected output magnitude ~sqrt(K)*0.33 ≈ 15; per-output bf16
    /// store error ≤ 6e-2. Use a wider 1.5e-1 abs tolerance.
    #[test]
    #[ignore]
    fn gemm_m1_kernel_matches_cpu_golden() {
        use crate::fixtures::{buf_byte_sizes, gemm_m1_only_input};
        use cudarc::driver::{CudaContext, DevicePtr};
        use ferrite_forward::cpu_golden;

        let input = gemm_m1_only_input();
        let k = input.sources[0].cols as usize;
        let n = input.sources[1].rows as usize;
        assert_eq!(input.sources[1].cols as usize, k);

        // Tighter input range so the K=2048 sum doesn't overflow bf16
        // storage in either input or accumulator. Inputs in `[-0.1,
        // 0.1)` → output magnitude ~sqrt(2048)*0.033 ≈ 1.5, well-
        // conditioned.
        let x_f32 = prng_f32_seeded(0x5555_6666, k, 0.1, 0.0);
        let w_f32 = prng_f32_seeded(0x7777_8888, n * k, 0.1, 0.0);
        let (_, x_lossy, x_bytes) = f32_to_bf16_bytes(&x_f32);
        let (_, w_lossy, w_bytes) = f32_to_bf16_bytes(&w_f32);

        let ctx = CudaContext::new(0).expect("cuda init");
        let stream = ctx.new_stream().expect("stream create");
        let raw_sizes = buf_byte_sizes(&input);
        assert_eq!(raw_sizes.len(), 3);
        let pad_floor = 1usize << 20;
        let alloc_size = |n: usize| -> usize { n.max(pad_floor).div_ceil(128) * 128 };

        let buf_x: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[0]))
            .expect("alloc x");
        let buf_w: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[1]))
            .expect("alloc w");
        let buf_out: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(alloc_size(raw_sizes[2]))
            .expect("alloc out");

        let mut buf_x_mut = buf_x;
        let mut buf_w_mut = buf_w;
        stream.memcpy_htod(&x_bytes, &mut buf_x_mut).expect("h2d x");
        stream.memcpy_htod(&w_bytes, &mut buf_w_mut).expect("h2d w");

        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(3);
        let mut _records: Vec<cudarc::driver::SyncOnDrop<'_>> = Vec::with_capacity(3);
        for buf in [&buf_x_mut, &buf_w_mut, &buf_out] {
            let (dptr, rec) = DevicePtr::device_ptr(buf, &stream);
            ptrs.push(dptr as *mut c_void);
            _records.push(rec);
        }
        stream.synchronize().expect("pre-launch sync");

        let u32_args: [u32; 0] = [];
        let err = unsafe {
            launch_decode_gemm_m1(&ptrs, &u32_args, stream.cu_stream() as *mut c_void)
        };
        assert_eq!(err, 0, "launch_tk_decode_gemm_m1 cudaError {err}");
        stream.synchronize().expect("post-launch sync");

        let mut out_bytes = vec![0u8; buf_out.len()];
        stream.memcpy_dtoh(&buf_out, &mut out_bytes).expect("d2h out");
        stream.synchronize().expect("post-d2h sync");

        let got = bytes_to_f32_prefix(&out_bytes, n);
        let mut want = vec![0.0_f32; n];
        cpu_golden::gemm(&x_lossy, &w_lossy, &mut want, 1, k, n);
        round_through_bf16_inplace(&mut want);

        // Diagnostic: count zero-output positions to detect short
        // loops (kernel deadlock, runtime offset miscalculation, or
        // page-allocator parity drift terminating the streaming
        // matvec early).
        let n_zeros_got = got.iter().filter(|&&v| v == 0.0).count();
        let n_zeros_want = want.iter().filter(|&&v| v == 0.0).count();
        eprintln!(
            "gemm_m1 diag: got has {n_zeros_got}/{n} zeros, \
             want has {n_zeros_want}/{n} zeros, \
             first nonzero diff at index check below"
        );
        for i in 0..n.min(8) {
            eprintln!("  i={i}: got={} want={}", got[i], want[i]);
        }
        // Find first index where got=0 but want!=0 (early loop exit).
        for i in 0..n {
            if got[i] == 0.0 && want[i].abs() > 1e-2 {
                eprintln!("  first early-exit candidate: i={i} got=0 want={}", want[i]);
                break;
            }
        }

        // K=2048 dot products with bf16 inputs and f32 accumulator —
        // tolerance widens because each accumulator step rounds
        // through bf16 only at the per-input read (kernel does f32
        // accumulate via __shfl_xor_sync reduction). 5e-2 covers the
        // worst-case bf16 storage error on the [-0.1, 0.1) input
        // range output.
        assert_max_abs(&got, &want, 5e-2, "gemm_m1");
    }
}
