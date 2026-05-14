// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Softmax kernel dispatcher.
//!
//! Faithful port of MLX `softmax_single_row` (`mlx/backend/metal/kernels/
//! softmax.h:11`). Used by the MoE router (`mx.softmax(gates, axis=-1,
//! precise=True)` per `~/git/mlx-lm/mlx_lm/models/qwen3_next.py:335`).
//!
//! Layout: input/output are `[batch, axis_size]` row-major. One
//! threadgroup per row processes the full axis with N_READS=4 per
//! thread. The Metal kernel caps `axis_size <= N_READS * max_threads
//! = 4096`; MoE router rows (num_experts ∈ [8, 512]) are well inside.
//! When axis_size exceeds that, the looped variant in mlx softmax.h:101
//! is the right port — not yet needed.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;

/// Number of elements each thread reads — must match `N_READS` in
/// `shaders/softmax.metal`.
pub const SOFTMAX_N_READS: usize = 4;

/// Compiled softmax pipelines, one per `(dtype, precise)` flavor.
///
/// `*_precise` runs the reduction in `float` regardless of input dtype
/// (mlx's `precise=true` mode) — required for router softmax where the
/// downstream `y * scores[..., None]` multiply at qwen3_next.py:344 is
/// sensitive to score precision.
pub struct SoftmaxKernels {
    pub f32: ComputePipelineState,
    pub f16: ComputePipelineState,
    pub f16_precise: ComputePipelineState,
    pub bf16_precise: ComputePipelineState,
    _library: Library,
}

impl SoftmaxKernels {
    /// Loads the four single-row softmax pipelines. No non-precise
    /// `bf16` variant — Metal's stock `simd_max<bfloat>` /
    /// `simd_sum<bfloat>` are absent on the system toolchain (mlx ships
    /// a `bfloat16_t` wrapper with custom overloads to work around
    /// this). bf16 inputs always run through the precise (float-AccT)
    /// path, which is what `mx.softmax(..., precise=True)` does anyway.
    pub fn new(device: &Device) -> Result<Self, MetalStreamError> {
        let library = load_library_from_bytes(device, crate::embedded_metallib!("softmax"))
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load `softmax.metallib`: {e}"))
            })?;
        let f32 = build_pipeline(device, &library, "block_softmax_float32")?;
        let f16 = build_pipeline(device, &library, "block_softmax_float16")?;
        let f16_precise = build_pipeline(device, &library, "block_softmax_precise_float16")?;
        let bf16_precise = build_pipeline(device, &library, "block_softmax_precise_bfloat16")?;
        Ok(Self {
            f32,
            f16,
            f16_precise,
            bf16_precise,
            _library: library,
        })
    }
}

fn build_pipeline(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, MetalStreamError> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| MetalStreamError::ShaderCompilationFailed(format!("{name} fn missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| MetalStreamError::ShaderCompilationFailed(format!("{name} pipeline: {e:?}")))
}

/// Pick the right number of threads per threadgroup for a given
/// `axis_size`. Matches mlx's `softmax.cpp` dispatcher: ceil_div(axis,
/// N_READS) rounded up to the nearest multiple of 32 (one simdgroup),
/// capped at `max_threads_per_threadgroup` (1024 on every Apple-silicon
/// GPU). The shader's threadgroup-mem `local_max[SIMD_SIZE]` array is
/// sized for up to 32 simdgroups (1024 threads), so this cap is a hard
/// invariant.
pub fn softmax_tg_size(axis_size: usize) -> usize {
    let n = axis_size.div_ceil(SOFTMAX_N_READS);
    // Round up to multiple of 32 (one simdgroup).
    let rounded = n.div_ceil(32) * 32;
    rounded.max(32).min(1024)
}

/// Dispatch one block-softmax over `[batch, axis_size]`.
///
/// `pipeline` must come from a [`SoftmaxKernels`] field matching the
/// input dtype + precise flag.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_softmax(
    pipeline: &ComputePipelineState,
    queue: &CommandQueue,
    input: &Buffer,
    output: &Buffer,
    batch: u32,
    axis_size: u32,
    dtype_bytes: usize,
) -> Result<(), MetalStreamError> {
    if batch == 0 || axis_size == 0 {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "dispatch_softmax: batch={batch} axis_size={axis_size}; both must be > 0"
        )));
    }
    let elems = (batch as usize) * (axis_size as usize);
    let bytes_needed = elems * dtype_bytes;
    if input.length() < bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "softmax: input buffer too small: have {} bytes, need {bytes_needed}",
            input.length()
        )));
    }
    if output.length() < bytes_needed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "softmax: output buffer too small: have {} bytes, need {bytes_needed}",
            output.length()
        )));
    }
    let tg_size = softmax_tg_size(axis_size as usize);
    if axis_size as usize > tg_size * SOFTMAX_N_READS {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "softmax: axis_size={axis_size} exceeds single-row capacity ({} = {tg_size} threads × {SOFTMAX_N_READS} reads); port softmax_looped",
            tg_size * SOFTMAX_N_READS
        )));
    }

    let cmdbuf = queue.commandBuffer().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("queue.commandBuffer returned nil".into())
    })?;
    let enc = cmdbuf.computeCommandEncoder().ok_or_else(|| {
        MetalStreamError::ShaderCompilationFailed("computeCommandEncoder returned nil".into())
    })?;
    enc.setComputePipelineState(pipeline);
    unsafe {
        enc.setBuffer_offset_atIndex(Some(input), 0, 0);
        enc.setBuffer_offset_atIndex(Some(output), 0, 1);
        let axis_i32: i32 = axis_size as i32;
        enc.setBytes_length_atIndex(
            NonNull::new(&axis_i32 as *const i32 as *mut c_void).unwrap(),
            std::mem::size_of::<i32>(),
            2,
        );
    }
    let threadgroups = MTLSize {
        width: batch as usize,
        height: 1,
        depth: 1,
    };
    let threads_per_tg = MTLSize {
        width: tg_size,
        height: 1,
        depth: 1,
    };
    enc.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_tg);
    enc.endEncoding();
    cmdbuf.commit();
    cmdbuf.waitUntilCompleted();
    if cmdbuf.status() != MTLCommandBufferStatus::Completed {
        return Err(MetalStreamError::ShaderCompilationFailed(format!(
            "softmax dispatch finished with status {:?}",
            cmdbuf.status()
        )));
    }
    Ok(())
}

// ─── CPU reference for tests ────────────────────────────────────────

/// Stable softmax reference (float-accumulator), row-by-row.
///
/// Mirrors `mlx.core.softmax(x, axis=-1, precise=True)` semantics: the
/// max is subtracted before `exp` to avoid overflow, and the reduction
/// runs in `f32` regardless of the input dtype.
pub fn softmax_cpu_f32(input: &[f32], batch: usize, axis_size: usize) -> Vec<f32> {
    assert_eq!(input.len(), batch * axis_size);
    let mut out = vec![0.0f32; input.len()];
    for b in 0..batch {
        let row = &input[b * axis_size..(b + 1) * axis_size];
        let mut maxval = f32::NEG_INFINITY;
        for &x in row {
            if x > maxval {
                maxval = x;
            }
        }
        let mut sum = 0.0f32;
        let dst = &mut out[b * axis_size..(b + 1) * axis_size];
        for (i, &x) in row.iter().enumerate() {
            let e = (x - maxval).exp();
            dst[i] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for v in dst.iter_mut() {
            *v *= inv;
        }
    }
    out
}

// Test coverage lives in `tests/softmax_test.rs` (integration tests
// under `tests/` compile against the lib's non-test build, side-
// stepping the pre-existing test-rot in
// `src/instruction_executor/test_*.rs` from the pre-objc2-metal era).
