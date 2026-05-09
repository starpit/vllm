// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `argmax_f16` — greedy-sample MSL kernel + Rust dispatcher.
//!
//! Per F.4.3 of the ferrite-metal port plan: a standalone MSL kernel
//! the worker fires after the forward pass's logits have been
//! produced, picking the per-row token with the highest logit. The
//! kernel is *not* part of the ICB hot path — sampling is one-shot
//! per step, encoded into its own command buffer outside the
//! per-bucket ICB.
//!
//! Tie-break on smaller index, matching numpy / torch `argmax`.
//! Determinism cross-threadgroup-size is documented in the shader.
//!
//! ## Why a separate kernel
//!
//! CUDA's `gpu_sample_and_finalize` lifts argmax / Gumbel /
//! greedy-or-stochastic dispatch into a single big kernel; for the
//! Metal port F.4.3 ships *only* greedy. Stochastic sampling
//! (top-p / top-k / temperature) lands when the worker needs it,
//! per `feedback_trait_primitives_not_orchestration.md` — kernels
//! grow on the per-backend impl, not on the shared trait surface.
//!
//! ## Coherency contract
//!
//! `dispatch_argmax_f16` commits + waits the dispatch's command
//! buffer before returning. Caller must ensure the upstream forward
//! pass that wrote `logits` has already completed (committed +
//! awaited on the same `CommandQueue` or interlocked with an
//! event) — same contract as the existing per-kernel dispatchers in
//! this crate. Output buffer is `StorageModeShared`-readable
//! immediately on return.

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Library, MTLResourceOptions, MTLSize,
};

use crate::stream::MetalStreamError;

/// Recommended threadgroup size for `argmax_f16`. 256 threads gives
/// `log2(256) = 8` barriers in the tree reduction and saturates a
/// single simdgroup (32) × 8 wave count, which is plenty for the
/// per-step memory-bandwidth-bound load over `[vocab]`.
///
/// Caller-overridable via [`dispatch_argmax_f16_with_tg_size`] for
/// micro-benches; the kernel requires power-of-two ≤ 1024.
pub const ARGMAX_DEFAULT_TG_SIZE: u64 = 256;

/// Compiled `argmax` library + kernel pipelines. One per Device;
/// hold in a long-lived field on whatever owns the compiled
/// kernels (eg. `MetalLogitsProcessor` once F.5 wires it up).
pub struct ArgmaxKernels {
    /// `argmax_f16` pipeline. Reference-counted; clone is cheap.
    pub f16: ComputePipelineState,
    /// `argmax_bf16` pipeline.
    pub bf16: ComputePipelineState,
    _library: Library,
}

impl ArgmaxKernels {
    /// Compile the argmax shader and resolve both `argmax_f16` and
    /// `argmax_bf16` pipelines.
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let source = include_str!("../shaders/argmax.metal");
        let library = device
            .new_library_with_source(source, &CompileOptions::new())
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("argmax.metal: {e:?}"))
            })?;
        let f16_fn = library.get_function("argmax_f16", None).map_err(|e| {
            MetalStreamError::ShaderCompilationFailed(format!("argmax_f16 fn: {e:?}"))
        })?;
        let f16 = device
            .new_compute_pipeline_state_with_function(&f16_fn)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("argmax_f16 pipeline: {e:?}"))
            })?;
        let bf16_fn = library.get_function("argmax_bf16", None).map_err(|e| {
            MetalStreamError::ShaderCompilationFailed(format!("argmax_bf16 fn: {e:?}"))
        })?;
        let bf16 = device
            .new_compute_pipeline_state_with_function(&bf16_fn)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("argmax_bf16 pipeline: {e:?}"))
            })?;
        Ok(Self {
            f16,
            bf16,
            _library: library,
        })
    }
}

/// Greedy-sample `[batch, vocab]` half-precision logits into
/// `[batch]` u32 token ids. Default `tg_size`.
pub fn dispatch_argmax_f16(
    kernels: &ArgmaxKernels,
    queue: &CommandQueue,
    logits: &Buffer,
    output: &Buffer,
    batch: u32,
    vocab: u32,
) -> Result<(), MetalStreamError> {
    dispatch_argmax_f16_with_tg_size(
        kernels,
        queue,
        logits,
        output,
        batch,
        vocab,
        ARGMAX_DEFAULT_TG_SIZE,
    )
}

/// Variant of [`dispatch_argmax_f16`] with a caller-chosen threadgroup
/// size. `tg_size` must be a power of two between 1 and 1024
/// inclusive (the shader's `shared_*` arrays are sized for 1024).
pub fn dispatch_argmax_f16_with_tg_size(
    kernels: &ArgmaxKernels,
    queue: &CommandQueue,
    logits: &Buffer,
    output: &Buffer,
    batch: u32,
    vocab: u32,
    tg_size: u64,
) -> Result<(), MetalStreamError> {
    if batch == 0 || vocab == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_argmax_f16: batch={batch} vocab={vocab}; both must be > 0"
        )));
    }
    if !tg_size.is_power_of_two() || tg_size == 0 || tg_size > 1024 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_argmax_f16: tg_size={tg_size} must be a power of two in [1, 1024]"
        )));
    }
    let logits_bytes_needed = (batch as u64) * (vocab as u64) * 2;
    if logits.length() < logits_bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "logits buffer too small: have {} bytes, need {logits_bytes_needed}",
            logits.length()
        )));
    }
    let output_bytes_needed = (batch as u64) * 4;
    if output.length() < output_bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "output buffer too small: have {} bytes, need {output_bytes_needed}",
            output.length()
        )));
    }

    let cmdbuf = queue.new_command_buffer();
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&kernels.f16);
    enc.set_buffer(0, Some(logits), 0);
    enc.set_buffer(1, Some(output), 0);
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &batch as *const u32 as *const std::ffi::c_void,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &vocab as *const u32 as *const std::ffi::c_void,
    );
    let threadgroups = MTLSize {
        width: batch as u64,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(threadgroups, threads_per_tg);
    enc.end_encoding();
    cmdbuf.commit();
    cmdbuf.wait_until_completed();
    if cmdbuf.status() != metal::MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "argmax_f16 dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    let _ = enc as &ComputeCommandEncoderRef; // silence unused-borrow
    Ok(())
}

/// BF16 counterpart to [`dispatch_argmax_f16`]. Identical contract
/// — `[batch, vocab]` bf16 logits → `[batch]` u32 token ids — with
/// the bf16 pipeline picked from the same `ArgmaxKernels`. Default
/// `tg_size`.
pub fn dispatch_argmax_bf16(
    kernels: &ArgmaxKernels,
    queue: &CommandQueue,
    logits: &Buffer,
    output: &Buffer,
    batch: u32,
    vocab: u32,
) -> Result<(), MetalStreamError> {
    if batch == 0 || vocab == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_argmax_bf16: batch={batch} vocab={vocab}; both must be > 0"
        )));
    }
    let tg_size: u64 = ARGMAX_DEFAULT_TG_SIZE;
    let logits_bytes_needed = (batch as u64) * (vocab as u64) * 2;
    if logits.length() < logits_bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "logits buffer too small: have {} bytes, need {logits_bytes_needed}",
            logits.length()
        )));
    }
    let output_bytes_needed = (batch as u64) * 4;
    if output.length() < output_bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "output buffer too small: have {} bytes, need {output_bytes_needed}",
            output.length()
        )));
    }
    let cmdbuf = queue.new_command_buffer();
    let enc = cmdbuf.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&kernels.bf16);
    enc.set_buffer(0, Some(logits), 0);
    enc.set_buffer(1, Some(output), 0);
    enc.set_bytes(
        2,
        std::mem::size_of::<u32>() as u64,
        &batch as *const u32 as *const std::ffi::c_void,
    );
    enc.set_bytes(
        3,
        std::mem::size_of::<u32>() as u64,
        &vocab as *const u32 as *const std::ffi::c_void,
    );
    let threadgroups = MTLSize {
        width: batch as u64,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };
    enc.dispatch_thread_groups(threadgroups, threads_per_tg);
    enc.end_encoding();
    cmdbuf.commit();
    cmdbuf.wait_until_completed();
    if cmdbuf.status() != metal::MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "argmax_bf16 dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

/// Tiny helper: stage `data` into a fresh `StorageModeShared` buffer.
/// Used by tests and (eventually) the worker before it has its own
/// dedicated runtime allocator. Public so the lifetimes line up
/// without exposing the internal bytes copy.
pub fn upload_shared_buffer<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let nbytes = std::mem::size_of_val(data) as u64;
    let buffer = device.new_buffer(nbytes.max(1), MTLResourceOptions::StorageModeShared);
    if nbytes > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buffer.contents() as *mut u8,
                nbytes as usize,
            );
        }
    }
    buffer
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::detect_device;

    /// Brute-force CPU argmax with the same tie-break as the kernel:
    /// on equal max, keep the smaller index. Used as the cpu_golden
    /// for the kernel's correctness tests.
    fn cpu_argmax_f16(logits: &[half::f16], batch: usize, vocab: usize) -> Vec<u32> {
        let mut out = vec![0u32; batch];
        for b in 0..batch {
            let row = &logits[b * vocab..(b + 1) * vocab];
            let mut best_v = f32::NEG_INFINITY;
            let mut best_i = 0u32;
            for (i, &v) in row.iter().enumerate() {
                let vf = v.to_f32();
                if vf > best_v || (vf == best_v && (i as u32) < best_i) {
                    best_v = vf;
                    best_i = i as u32;
                }
            }
            out[b] = best_i;
        }
        out
    }

    fn try_device() -> Option<Device> {
        Some(detect_device()?.device.clone())
    }

    /// Single-row argmax with a hand-built input where the maximum
    /// is at a known position. Wedges the basic dispatch path before
    /// any randomized testing.
    #[test]
    fn argmax_f16_finds_known_max() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let kernels = ArgmaxKernels::new(&device).expect("compile argmax");
        let queue = device.new_command_queue();

        let vocab = 64usize;
        let mut logits = vec![half::f16::from_f32(0.0); vocab];
        logits[37] = half::f16::from_f32(99.0);
        let logits_buf = upload_shared_buffer(&device, &logits);
        let output_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

        dispatch_argmax_f16(&kernels, &queue, &logits_buf, &output_buf, 1, vocab as u32)
            .expect("dispatch");

        let read = unsafe { *(output_buf.contents() as *const u32) };
        assert_eq!(read, 37);
    }

    /// Multi-row dispatch: each row has its max at a different
    /// position. Confirms threadgroups don't bleed across rows.
    #[test]
    fn argmax_f16_handles_independent_rows() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let kernels = ArgmaxKernels::new(&device).expect("compile argmax");
        let queue = device.new_command_queue();

        let batch = 4usize;
        let vocab = 256usize;
        let positions = [0usize, 100, 255, 7];
        let mut logits = vec![half::f16::from_f32(0.0); batch * vocab];
        for (b, &p) in positions.iter().enumerate() {
            logits[b * vocab + p] = half::f16::from_f32(50.0);
        }
        let logits_buf = upload_shared_buffer(&device, &logits);
        let output_buf =
            device.new_buffer((batch * 4) as u64, MTLResourceOptions::StorageModeShared);

        dispatch_argmax_f16(
            &kernels,
            &queue,
            &logits_buf,
            &output_buf,
            batch as u32,
            vocab as u32,
        )
        .expect("dispatch");

        let read = unsafe {
            std::slice::from_raw_parts(output_buf.contents() as *const u32, batch).to_vec()
        };
        let expected: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        assert_eq!(read, expected);
    }

    /// Tie-break: when multiple indices share the maximum value, the
    /// kernel must return the smallest. The shader documents this
    /// invariant; this test wedges it.
    #[test]
    fn argmax_f16_tie_breaks_to_smaller_index() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let kernels = ArgmaxKernels::new(&device).expect("compile argmax");
        let queue = device.new_command_queue();

        let vocab = 128usize;
        let mut logits = vec![half::f16::from_f32(0.0); vocab];
        // All-zero row → every index ties. Smallest (0) must win.
        let logits_buf = upload_shared_buffer(&device, &logits);
        let output_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        dispatch_argmax_f16(&kernels, &queue, &logits_buf, &output_buf, 1, vocab as u32)
            .expect("dispatch all-zero");
        assert_eq!(unsafe { *(output_buf.contents() as *const u32) }, 0);

        // Two-way tie at index 5 and 50 — smaller must win.
        logits[5] = half::f16::from_f32(7.0);
        logits[50] = half::f16::from_f32(7.0);
        let logits_buf = upload_shared_buffer(&device, &logits);
        let output_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        dispatch_argmax_f16(&kernels, &queue, &logits_buf, &output_buf, 1, vocab as u32)
            .expect("dispatch tie");
        assert_eq!(unsafe { *(output_buf.contents() as *const u32) }, 5);
    }

    /// Random f16 logits, real LLM-shaped vocab. Confirms the kernel
    /// matches `cpu_argmax_f16` over a non-trivial input. Vocab
    /// dwarfs `tg_size`, so each thread takes many strided steps —
    /// exercises the strided-load path.
    #[test]
    fn argmax_f16_matches_cpu_golden_random() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let kernels = ArgmaxKernels::new(&device).expect("compile argmax");
        let queue = device.new_command_queue();

        let batch = 3usize;
        let vocab = 32_000usize; // Llama-class vocab.

        // Deterministic LCG so the test is reproducible without rand.
        let mut state = 0xdead_beef_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let logits: Vec<half::f16> = (0..batch * vocab)
            .map(|_| {
                let r = (next() >> 32) as u32;
                let v = (r as f32) / (u32::MAX as f32) * 20.0 - 10.0;
                half::f16::from_f32(v)
            })
            .collect();

        let logits_buf = upload_shared_buffer(&device, &logits);
        let output_buf =
            device.new_buffer((batch * 4) as u64, MTLResourceOptions::StorageModeShared);

        dispatch_argmax_f16(
            &kernels,
            &queue,
            &logits_buf,
            &output_buf,
            batch as u32,
            vocab as u32,
        )
        .expect("dispatch");

        let kernel = unsafe {
            std::slice::from_raw_parts(output_buf.contents() as *const u32, batch).to_vec()
        };
        let golden = cpu_argmax_f16(&logits, batch, vocab);
        assert_eq!(kernel, golden, "kernel disagrees with cpu_golden");
    }

    /// Same input across two different threadgroup sizes must produce
    /// the same answer — the tie-break invariants are independent of
    /// the reduction tree shape.
    #[test]
    fn argmax_f16_is_threadgroup_size_invariant() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let kernels = ArgmaxKernels::new(&device).expect("compile argmax");
        let queue = device.new_command_queue();

        let batch = 2usize;
        let vocab = 4096usize;
        // Construct a row with a few ties in the top-1 region.
        let mut logits = vec![half::f16::from_f32(-1.0); batch * vocab];
        for b in 0..batch {
            for &i in &[37usize, 1024, 3000] {
                logits[b * vocab + i] = half::f16::from_f32(5.0);
            }
        }
        let logits_buf = upload_shared_buffer(&device, &logits);

        let mut answers: Vec<Vec<u32>> = Vec::new();
        for &tg in &[64u64, 256, 1024] {
            let output_buf =
                device.new_buffer((batch * 4) as u64, MTLResourceOptions::StorageModeShared);
            dispatch_argmax_f16_with_tg_size(
                &kernels,
                &queue,
                &logits_buf,
                &output_buf,
                batch as u32,
                vocab as u32,
                tg,
            )
            .expect("dispatch");
            let read = unsafe {
                std::slice::from_raw_parts(output_buf.contents() as *const u32, batch).to_vec()
            };
            answers.push(read);
        }
        // Smallest tied index per row should win at every tg size.
        let expected = vec![37u32; batch];
        for ans in &answers {
            assert_eq!(*ans, expected);
        }
    }

    /// Validation: zero batch or zero vocab is rejected.
    #[test]
    fn dispatch_rejects_zero_dims() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let kernels = ArgmaxKernels::new(&device).expect("compile argmax");
        let queue = device.new_command_queue();
        let buf = device.new_buffer(16, MTLResourceOptions::StorageModeShared);
        assert!(dispatch_argmax_f16(&kernels, &queue, &buf, &buf, 0, 16).is_err());
        assert!(dispatch_argmax_f16(&kernels, &queue, &buf, &buf, 1, 0).is_err());
    }

    /// Validation: undersized output buffer is rejected before any
    /// GPU work is submitted.
    #[test]
    fn dispatch_rejects_undersized_output() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let kernels = ArgmaxKernels::new(&device).expect("compile argmax");
        let queue = device.new_command_queue();
        let logits = device.new_buffer(64 * 2, MTLResourceOptions::StorageModeShared);
        // batch=2 needs 8 bytes; provide 4.
        let undersized = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        assert!(dispatch_argmax_f16(&kernels, &queue, &logits, &undersized, 2, 32).is_err());
    }

    /// Validation: non-power-of-two `tg_size` is rejected.
    #[test]
    fn dispatch_rejects_non_power_of_two_tg_size() {
        let Some(device) = try_device() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let kernels = ArgmaxKernels::new(&device).expect("compile argmax");
        let queue = device.new_command_queue();
        let logits = device.new_buffer(64 * 2, MTLResourceOptions::StorageModeShared);
        let output = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        assert!(
            dispatch_argmax_f16_with_tg_size(&kernels, &queue, &logits, &output, 1, 32, 100)
                .is_err()
        );
        assert!(
            dispatch_argmax_f16_with_tg_size(&kernels, &queue, &logits, &output, 1, 32, 2048)
                .is_err()
        );
        // Power of two ≤ 1024 succeeds.
        assert!(
            dispatch_argmax_f16_with_tg_size(&kernels, &queue, &logits, &output, 1, 32, 64).is_ok()
        );
    }
}
