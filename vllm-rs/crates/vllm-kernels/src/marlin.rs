// SPDX-License-Identifier: Apache-2.0
//! Marlin W4A16 fused GEMM kernel bindings.
//!
//! Provides fused INT4-dequantize-in-register + tensor-core GEMM via the
//! Marlin kernel (IST-DASLab). Reads packed INT4 weights directly, dequantizes
//! in registers, and uses tensor core MMA — eliminating the intermediate
//! full-precision weight matrix allocation.

use candle_core::{DType, Device, Tensor};

use crate::error::{KernelError, KernelResult};

// ---------------------------------------------------------------------------
// FFI declarations
// ---------------------------------------------------------------------------

#[allow(non_camel_case_types)]
type cudaStream_t = *mut std::ffi::c_void;

unsafe extern "C" {
    fn marlin_gemm_f16(
        a: *const std::ffi::c_void,
        b_q_weight: *const std::ffi::c_void,
        c: *mut std::ffi::c_void,
        b_scales: *const std::ffi::c_void,
        b_zeros: *const std::ffi::c_void,
        g_idx: *const std::ffi::c_void,
        perm: *const std::ffi::c_void,
        workspace: *mut std::ffi::c_void,
        c_tmp: *mut std::ffi::c_void,
        a_tmp: *mut std::ffi::c_void,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        lda: i32,
        num_groups: i32,
        group_size: i32,
        has_act_order: bool,
        is_k_full: bool,
        has_zp: bool,
        is_zp_float: bool,
        use_fp32_reduce: bool,
        b_type_id: i32,
        stream: cudaStream_t,
        device_id: i32,
    );

    fn marlin_gemm_bf16(
        a: *const std::ffi::c_void,
        b_q_weight: *const std::ffi::c_void,
        c: *mut std::ffi::c_void,
        b_scales: *const std::ffi::c_void,
        b_zeros: *const std::ffi::c_void,
        g_idx: *const std::ffi::c_void,
        perm: *const std::ffi::c_void,
        workspace: *mut std::ffi::c_void,
        c_tmp: *mut std::ffi::c_void,
        a_tmp: *mut std::ffi::c_void,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        lda: i32,
        num_groups: i32,
        group_size: i32,
        has_act_order: bool,
        is_k_full: bool,
        has_zp: bool,
        is_zp_float: bool,
        use_fp32_reduce: bool,
        b_type_id: i32,
        stream: cudaStream_t,
        device_id: i32,
    );

    fn gptq_marlin_repack_4bit(
        b_q_weight: *const u32,
        perm: *const u32,
        out: *mut u32,
        size_k: i32,
        size_n: i32,
        has_perm: bool,
        stream: cudaStream_t,
        device_id: i32,
    );

    fn awq_marlin_repack_4bit(
        b_q_weight: *const u32,
        out: *mut u32,
        size_k: i32,
        size_n: i32,
        stream: cudaStream_t,
        device_id: i32,
    );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the raw device pointer from a CUDA tensor.
fn device_ptr<T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
    tensor: &Tensor,
) -> KernelResult<usize> {
    use cudarc::driver::DevicePtr;
    let cuda_dev = tensor
        .device()
        .as_cuda_device()
        .map_err(|e| KernelError::Other(format!("{e}")))?;
    let stream = cuda_dev.cuda_stream();
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(cuda_storage) => {
            let slice = cuda_storage.as_cuda_slice::<T>()?;
            let view = slice.slice(layout.start_offset()..);
            let (ptr, _sync_guard) = view.device_ptr(&stream);
            Ok(ptr as usize)
        }
        _ => Err(KernelError::Other("expected CUDA tensor".to_string())),
    }
}

/// Get the raw device pointer, auto-dispatching on dtype.
fn device_ptr_auto(tensor: &Tensor) -> KernelResult<usize> {
    match tensor.dtype() {
        DType::F16 => device_ptr::<half::f16>(tensor),
        DType::BF16 => device_ptr::<half::bf16>(tensor),
        DType::F32 => device_ptr::<f32>(tensor),
        DType::I32 => device_ptr::<i32>(tensor),
        DType::U32 => device_ptr::<u32>(tensor),
        dt => Err(KernelError::Other(format!(
            "marlin: unsupported dtype {dt:?}"
        ))),
    }
}

/// Get the CUDA stream raw pointer and device ordinal from candle's CudaDevice.
fn stream_and_device(device: &Device) -> KernelResult<(cudaStream_t, i32)> {
    let cuda_dev = device
        .as_cuda_device()
        .map_err(|e| KernelError::Other(format!("{e}")))?;
    let stream = cuda_dev.cuda_stream();
    let raw_stream = stream.cu_stream() as cudaStream_t;
    // Get device ordinal via CUDA driver API on the current context.
    let mut dev_id: cudarc::driver::sys::CUdevice = 0;
    unsafe {
        cudarc::driver::sys::cuCtxGetDevice(&mut dev_id);
    }
    Ok((raw_stream, dev_id as i32))
}

// ---------------------------------------------------------------------------
// Marlin GEMM type identifiers
// ---------------------------------------------------------------------------

/// Weight type identifier for the C entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarlinWeightType {
    /// GPTQ: kU4B8 (uint4 with bias=8)
    GptqInt4 = 0,
    /// AWQ: kU4 (uint4 with zero-point)
    AwqInt4 = 1,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Marlin fused GEMM: `C = A @ dequant(B)` in a single kernel.
///
/// * `a` — activation tensor `[M, K]` (FP16 or BF16, contiguous, on CUDA)
/// * `b_q_weight` — Marlin-tiled packed INT4 weights (from repack)
/// * `b_scales` — `[num_groups, N]` scale tensor (same dtype as `a`)
/// * `b_zeros` — optional packed zero-point tensor
/// * `g_idx` — optional `[K]` group index for act_order (desc_act)
/// * `perm` — optional `[K]` permutation for act_order
/// * `workspace` — `[num_sms]` i32 workspace tensor
/// * `size_n` — output features
/// * `num_groups` — number of quantization groups
/// * `group_size` — group size (-1 if single group)
/// * `weight_type` — GPTQ or AWQ
///
/// Returns `[M, N]` output tensor.
#[allow(clippy::too_many_arguments)]
pub fn marlin_gemm(
    a: &Tensor,
    b_q_weight: &Tensor,
    b_scales: &Tensor,
    b_zeros: Option<&Tensor>,
    g_idx: Option<&Tensor>,
    perm: Option<&Tensor>,
    workspace: &Tensor,
    size_n: usize,
    num_groups: usize,
    group_size: i32,
    weight_type: MarlinWeightType,
) -> KernelResult<Tensor> {
    let dtype = a.dtype();
    let device = a.device();
    let a = a.contiguous()?;

    let (size_m, size_k) = a.dims2().map_err(KernelError::Candle)?;
    let lda = size_k as i32;

    // Allocate output
    let c = Tensor::zeros((size_m, size_n), dtype, device)?;

    // FP32 reduce buffer
    let use_fp32_reduce = true;
    let c_tmp = if use_fp32_reduce {
        // Conservative upper bound: 256 SMs * 64 * 256
        let max_c_tmp = 256 * 64 * 256;
        Tensor::zeros(max_c_tmp, DType::F32, device)?
    } else {
        Tensor::zeros(0, DType::F32, device)?
    };

    // act_order temp buffer
    let has_act_order = g_idx.is_some() && perm.is_some();
    let a_tmp = if has_act_order {
        Tensor::zeros((size_m, size_k), dtype, device)?
    } else {
        Tensor::zeros(0, dtype, device)?
    };

    let has_zp = b_zeros.is_some();

    // Get device pointers
    let a_ptr = device_ptr_auto(&a)?;
    let bqw_ptr = device_ptr_auto(b_q_weight)?;
    let c_ptr = device_ptr_auto(&c)?;
    let bs_ptr = device_ptr_auto(b_scales)?;
    let bz_ptr = b_zeros.map(device_ptr_auto).transpose()?.unwrap_or(0);
    let gi_ptr = g_idx.map(device_ptr::<i32>).transpose()?.unwrap_or(0);
    let pm_ptr = perm.map(device_ptr::<i32>).transpose()?.unwrap_or(0);
    let ws_ptr = device_ptr::<i32>(workspace)?;
    let ct_ptr = device_ptr::<f32>(&c_tmp)?;
    let at_ptr = device_ptr_auto(&a_tmp)?;

    let (stream, dev_id) = stream_and_device(device)?;
    let is_k_full = has_act_order;

    match dtype {
        DType::F16 => unsafe {
            marlin_gemm_f16(
                a_ptr as *const _,
                bqw_ptr as *const _,
                c_ptr as *mut _,
                bs_ptr as *const _,
                bz_ptr as *const _,
                gi_ptr as *const _,
                pm_ptr as *const _,
                ws_ptr as *mut _,
                ct_ptr as *mut _,
                at_ptr as *mut _,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                lda,
                num_groups as i32,
                group_size,
                has_act_order,
                is_k_full,
                has_zp,
                false, // is_zp_float
                use_fp32_reduce,
                weight_type as i32,
                stream,
                dev_id,
            );
        },
        DType::BF16 => unsafe {
            marlin_gemm_bf16(
                a_ptr as *const _,
                bqw_ptr as *const _,
                c_ptr as *mut _,
                bs_ptr as *const _,
                bz_ptr as *const _,
                gi_ptr as *const _,
                pm_ptr as *const _,
                ws_ptr as *mut _,
                ct_ptr as *mut _,
                at_ptr as *mut _,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                lda,
                num_groups as i32,
                group_size,
                has_act_order,
                is_k_full,
                has_zp,
                false,
                use_fp32_reduce,
                weight_type as i32,
                stream,
                dev_id,
            );
        },
        _ => {
            return Err(KernelError::Other(format!(
                "marlin_gemm: unsupported activation dtype {dtype:?}, expected F16 or BF16"
            )));
        }
    }

    Ok(c)
}

/// Repack GPTQ INT4 weights from HuggingFace layout to Marlin tiled layout.
pub fn gptq_repack(
    b_q_weight: &Tensor,
    perm: Option<&Tensor>,
    size_k: usize,
    size_n: usize,
) -> KernelResult<Tensor> {
    let device = b_q_weight.device();
    let tile_size = 16;
    let pack_factor = 8; // 32 / 4

    let out = Tensor::zeros(
        (size_k / tile_size, size_n * tile_size / pack_factor),
        DType::I32,
        device,
    )?;

    let bqw = b_q_weight.contiguous()?;
    let bqw_ptr = device_ptr::<i32>(&bqw)? as *const u32;
    let out_ptr = device_ptr::<i32>(&out)? as *mut u32;

    let has_perm = perm.is_some();
    let perm_ptr = perm
        .map(|t| {
            let t = t.contiguous()?;
            device_ptr::<i32>(&t)
        })
        .transpose()?
        .map(|p| p as *const u32)
        .unwrap_or(std::ptr::null());

    let (stream, dev_id) = stream_and_device(device)?;

    unsafe {
        gptq_marlin_repack_4bit(
            bqw_ptr,
            perm_ptr,
            out_ptr,
            size_k as i32,
            size_n as i32,
            has_perm,
            stream,
            dev_id,
        );
    }

    Ok(out)
}

/// Repack AWQ INT4 weights from HuggingFace layout to Marlin tiled layout.
pub fn awq_repack(b_q_weight: &Tensor, size_k: usize, size_n: usize) -> KernelResult<Tensor> {
    let device = b_q_weight.device();
    let tile_size = 16;
    let pack_factor = 8;

    let out = Tensor::zeros(
        (size_k / tile_size, size_n * tile_size / pack_factor),
        DType::I32,
        device,
    )?;

    let bqw = b_q_weight.contiguous()?;
    let bqw_ptr = device_ptr::<i32>(&bqw)? as *const u32;
    let out_ptr = device_ptr::<i32>(&out)? as *mut u32;

    let (stream, dev_id) = stream_and_device(device)?;

    unsafe {
        awq_marlin_repack_4bit(
            bqw_ptr,
            out_ptr,
            size_k as i32,
            size_n as i32,
            stream,
            dev_id,
        );
    }

    Ok(out)
}
