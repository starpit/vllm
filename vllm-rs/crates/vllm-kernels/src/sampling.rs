// SPDX-License-Identifier: Apache-2.0
//! GPU-side top-k / top-p / min-p sampling kernel.
//!
//! When logits are on CUDA, the fused kernel avoids transferring the full
//! vocab-sized tensor to CPU. It computes softmax, radix-selects the top-K
//! candidates, applies top-p and min-p cutoffs, and samples a single token
//! entirely on device — returning only a 4-byte token ID.
//!
//! Port of: new kernel (no Python vLLM equivalent — this is a Rust-native
//! optimization).

#[cfg(feature = "cuda")]
use crate::error::{KernelError, KernelResult};
#[cfg(feature = "cuda")]
use candle_core::{DType, Tensor};

// ---------------------------------------------------------------------------
// CUDA FFI
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod cuda_ffi {
    unsafe extern "C" {
        pub fn sample_top_k_top_p_f32(
            output: *mut u32,
            logits: *const f32,
            vocab_size: i32,
            temperature: f32,
            top_k: i32,
            top_p: f32,
            min_p: f32,
            uniform_random: f32,
        );
        pub fn sample_top_k_top_p_f16(
            output: *mut u32,
            logits: *const u16,
            vocab_size: i32,
            temperature: f32,
            top_k: i32,
            top_p: f32,
            min_p: f32,
            uniform_random: f32,
        );
        pub fn sample_top_k_top_p_bf16(
            output: *mut u32,
            logits: *const u16,
            vocab_size: i32,
            temperature: f32,
            top_k: i32,
            top_p: f32,
            min_p: f32,
            uniform_random: f32,
        );
        pub fn sample_batched_f32(
            output: *mut u32,
            logits: *const f32,
            vocab_size: i32,
            batch_size: i32,
            temperatures: *const f32,
            top_ks: *const i32,
            top_ps: *const f32,
            min_ps: *const f32,
            uniform_randoms: *const f32,
        );
        pub fn sample_batched_f16(
            output: *mut u32,
            logits: *const u16,
            vocab_size: i32,
            batch_size: i32,
            temperatures: *const f32,
            top_ks: *const i32,
            top_ps: *const f32,
            min_ps: *const f32,
            uniform_randoms: *const f32,
        );
        pub fn sample_batched_bf16(
            output: *mut u32,
            logits: *const u16,
            vocab_size: i32,
            batch_size: i32,
            temperatures: *const f32,
            top_ks: *const i32,
            top_ps: *const f32,
            min_ps: *const f32,
            uniform_randoms: *const f32,
        );
    }
}

// ---------------------------------------------------------------------------
// CUDA implementation
// ---------------------------------------------------------------------------

/// Sample a single token from logits on CUDA using the fused top-k/top-p/min-p kernel.
///
/// * `logits` — 1-D or 2-D tensor on CUDA (last dim = vocab_size). If 2-D,
///   only the last row is used (caller should narrow before calling).
/// * `temperature` — must be > 0
/// * `top_k` — 0 means disabled (defaults to 1024 internally)
/// * `top_p` — 1.0 means disabled
/// * `min_p` — 0.0 means disabled
/// * `uniform_random` — pre-generated U(0,1) scalar for sampling
///
/// Returns the sampled token ID as `u32`.
#[cfg(feature = "cuda")]
pub fn cuda_sample_top_k_top_p(
    logits: &candle_core::Tensor,
    temperature: f32,
    top_k: i32,
    top_p: f32,
    min_p: f32,
    uniform_random: f32,
) -> KernelResult<u32> {
    use cudarc::driver::DevicePtr;

    let logits = logits.flatten_all()?;
    let logits = logits.contiguous()?;
    let vocab_size = logits.elem_count();
    let dtype = logits.dtype();

    // Allocate a single u32 on device for output.
    let output = Tensor::zeros(1, DType::U32, logits.device())?;

    // Get raw device pointers.
    let cuda_dev = logits
        .device()
        .as_cuda_device()
        .map_err(|e| KernelError::Other(format!("{e}")))?;
    let stream = cuda_dev.cuda_stream();

    let out_ptr = {
        let (storage, layout) = output.storage_and_layout();
        match &*storage {
            candle_core::Storage::Cuda(cs) => {
                let slice = cs.as_cuda_slice::<u32>()?;
                let view = slice.slice(layout.start_offset()..);
                let (ptr, _guard) = view.device_ptr(&stream);
                ptr as usize
            }
            _ => return Err(KernelError::Other("expected CUDA tensor".into())),
        }
    };

    match dtype {
        DType::F32 => {
            let logits_ptr = device_ptr_of::<f32>(&logits)?;
            unsafe {
                cuda_ffi::sample_top_k_top_p_f32(
                    out_ptr as *mut u32,
                    logits_ptr as *const f32,
                    vocab_size as i32,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    uniform_random,
                );
            }
        }
        DType::F16 => {
            let logits_ptr = device_ptr_of::<half::f16>(&logits)?;
            unsafe {
                cuda_ffi::sample_top_k_top_p_f16(
                    out_ptr as *mut u32,
                    logits_ptr as *const u16,
                    vocab_size as i32,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    uniform_random,
                );
            }
        }
        DType::BF16 => {
            let logits_ptr = device_ptr_of::<half::bf16>(&logits)?;
            unsafe {
                cuda_ffi::sample_top_k_top_p_bf16(
                    out_ptr as *mut u32,
                    logits_ptr as *const u16,
                    vocab_size as i32,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    uniform_random,
                );
            }
        }
        _ => {
            return Err(KernelError::Other(format!(
                "unsupported dtype for GPU sampling: {dtype:?}"
            )));
        }
    }

    // Copy back the single u32.
    let token_id = output.to_vec1::<u32>()?[0];
    Ok(token_id)
}

/// Sample multiple requests in a single batched kernel launch.
///
/// * `logits` — 2-D tensor `[batch_size, vocab_size]` on CUDA (contiguous)
/// * `temperatures`, `top_ks`, `top_ps`, `min_ps`, `uniform_randoms` — per-request
///   parameter slices of length `batch_size`
///
/// Returns `Vec<u32>` of sampled token IDs, one per request.
///
/// This avoids `batch_size - 1` extra GPU syncs compared to calling the
/// single-request kernel in a loop.
#[cfg(feature = "cuda")]
pub fn cuda_sample_batched(
    logits: &Tensor,
    temperatures: &[f32],
    top_ks: &[i32],
    top_ps: &[f32],
    min_ps: &[f32],
    uniform_randoms: &[f32],
) -> KernelResult<Vec<u32>> {
    let shape = logits.dims();
    if shape.len() != 2 {
        return Err(KernelError::Other(format!(
            "cuda_sample_batched: expected 2-D logits, got shape {shape:?}"
        )));
    }
    let batch_size = shape[0];
    let vocab_size = shape[1];
    let dtype = logits.dtype();

    if batch_size == 0 {
        return Ok(vec![]);
    }

    // Ensure logits are contiguous.
    let logits = logits.contiguous()?;

    let device = logits.device();

    // Allocate output [batch_size] u32 on device.
    let output = Tensor::zeros(batch_size, DType::U32, device)?;

    // Upload parameter arrays to device as tensors.
    // Cast top_ks (i32) to u32 for Tensor::from_slice, then get raw pointer as i32.
    let top_ks_u32: Vec<u32> = top_ks.iter().map(|&k| k as u32).collect();
    let d_temps = Tensor::from_slice(temperatures, batch_size, device)?;
    let d_top_ks = Tensor::from_slice(&top_ks_u32, batch_size, device)?;
    let d_top_ps = Tensor::from_slice(top_ps, batch_size, device)?;
    let d_min_ps = Tensor::from_slice(min_ps, batch_size, device)?;
    let d_randoms = Tensor::from_slice(uniform_randoms, batch_size, device)?;

    let out_ptr = device_ptr_of::<u32>(&output)?;
    let temps_ptr = device_ptr_of::<f32>(&d_temps)?;
    let top_ks_ptr = device_ptr_of::<u32>(&d_top_ks)?;
    let top_ps_ptr = device_ptr_of::<f32>(&d_top_ps)?;
    let min_ps_ptr = device_ptr_of::<f32>(&d_min_ps)?;
    let randoms_ptr = device_ptr_of::<f32>(&d_randoms)?;

    match dtype {
        DType::F32 => {
            let logits_ptr = device_ptr_of::<f32>(&logits)?;
            unsafe {
                cuda_ffi::sample_batched_f32(
                    out_ptr as *mut u32,
                    logits_ptr as *const f32,
                    vocab_size as i32,
                    batch_size as i32,
                    temps_ptr as *const f32,
                    top_ks_ptr as *const i32,
                    top_ps_ptr as *const f32,
                    min_ps_ptr as *const f32,
                    randoms_ptr as *const f32,
                );
            }
        }
        DType::F16 => {
            let logits_ptr = device_ptr_of::<half::f16>(&logits)?;
            unsafe {
                cuda_ffi::sample_batched_f16(
                    out_ptr as *mut u32,
                    logits_ptr as *const u16,
                    vocab_size as i32,
                    batch_size as i32,
                    temps_ptr as *const f32,
                    top_ks_ptr as *const i32,
                    top_ps_ptr as *const f32,
                    min_ps_ptr as *const f32,
                    randoms_ptr as *const f32,
                );
            }
        }
        DType::BF16 => {
            let logits_ptr = device_ptr_of::<half::bf16>(&logits)?;
            unsafe {
                cuda_ffi::sample_batched_bf16(
                    out_ptr as *mut u32,
                    logits_ptr as *const u16,
                    vocab_size as i32,
                    batch_size as i32,
                    temps_ptr as *const f32,
                    top_ks_ptr as *const i32,
                    top_ps_ptr as *const f32,
                    min_ps_ptr as *const f32,
                    randoms_ptr as *const f32,
                );
            }
        }
        _ => {
            return Err(KernelError::Other(format!(
                "unsupported dtype for batched GPU sampling: {dtype:?}"
            )));
        }
    }

    // Single sync: copy all token IDs back.
    let token_ids = output.to_vec1::<u32>()?;
    Ok(token_ids)
}

// ---------------------------------------------------------------------------
// Pre-allocated sampling buffers
// ---------------------------------------------------------------------------

/// Pre-allocated GPU buffers for batched sampling, eliminating per-step
/// `cudaMalloc` + `cudaFree` overhead. Sized once to `max_batch_size` and
/// reused across all decode steps.
#[cfg(feature = "cuda")]
pub struct SamplingBuffers {
    /// Max batch size these buffers were allocated for.
    pub max_batch: usize,
    /// `[max_batch]` f32 — temperatures
    d_temperatures: Tensor,
    /// `[max_batch]` u32 — top_k values (stored as u32, cast to i32 pointer)
    d_top_ks: Tensor,
    /// `[max_batch]` f32 — top_p values
    d_top_ps: Tensor,
    /// `[max_batch]` f32 — min_p values
    d_min_ps: Tensor,
    /// `[max_batch]` f32 — uniform random values
    d_uniforms: Tensor,
    /// `[max_batch]` u32 — output token IDs
    d_output: Tensor,
    /// Host-side readback buffer (avoids alloc in hot path).
    host_output: Vec<u32>,
}

#[cfg(feature = "cuda")]
impl SamplingBuffers {
    /// Allocate all buffers on the given CUDA device.
    pub fn new(max_batch: usize, device: &candle_core::Device) -> KernelResult<Self> {
        Ok(Self {
            max_batch,
            d_temperatures: Tensor::zeros(max_batch, DType::F32, device)?,
            d_top_ks: Tensor::zeros(max_batch, DType::U32, device)?,
            d_top_ps: Tensor::zeros(max_batch, DType::F32, device)?,
            d_min_ps: Tensor::zeros(max_batch, DType::F32, device)?,
            d_uniforms: Tensor::zeros(max_batch, DType::F32, device)?,
            d_output: Tensor::zeros(max_batch, DType::U32, device)?,
            host_output: vec![0u32; max_batch],
        })
    }

    /// Sample a batch using pre-allocated buffers. Writes parameters via
    /// `memcpy_htod_sync` into existing device memory instead of allocating
    /// new tensors.
    ///
    /// * `logits` — 2-D `[batch_size, vocab_size]` CUDA tensor (contiguous)
    /// * Per-request slices of length `batch_size` (must be <= `max_batch`)
    ///
    /// Returns `&[u32]` slice of sampled token IDs (valid until next call).
    pub fn sample_batched(
        &mut self,
        logits: &Tensor,
        temperatures: &[f32],
        top_ks: &[i32],
        top_ps: &[f32],
        min_ps: &[f32],
        uniform_randoms: &[f32],
    ) -> KernelResult<&[u32]> {
        let shape = logits.dims();
        if shape.len() != 2 {
            return Err(KernelError::Other(format!(
                "SamplingBuffers::sample_batched: expected 2-D logits, got {shape:?}"
            )));
        }
        let batch_size = shape[0];
        let vocab_size = shape[1];
        let dtype = logits.dtype();

        if batch_size == 0 {
            return Ok(&[]);
        }
        if batch_size > self.max_batch {
            return Err(KernelError::Other(format!(
                "batch_size {batch_size} exceeds pre-allocated max_batch {}",
                self.max_batch
            )));
        }

        let logits = logits.contiguous()?;

        // Convert top_ks i32 → u32 on stack (tiny — batch_size elements).
        let top_ks_u32: Vec<u32> = top_ks.iter().map(|&k| k as u32).collect();

        // Write parameters into pre-allocated device buffers via memcpy.
        memcpy_htod_into::<f32>(&self.d_temperatures, temperatures)?;
        memcpy_htod_into::<u32>(&self.d_top_ks, &top_ks_u32)?;
        memcpy_htod_into::<f32>(&self.d_top_ps, top_ps)?;
        memcpy_htod_into::<f32>(&self.d_min_ps, min_ps)?;
        memcpy_htod_into::<f32>(&self.d_uniforms, uniform_randoms)?;

        // Get device pointers (no allocation — tensors already exist).
        let out_ptr = device_ptr_of::<u32>(&self.d_output)?;
        let temps_ptr = device_ptr_of::<f32>(&self.d_temperatures)?;
        let top_ks_ptr = device_ptr_of::<u32>(&self.d_top_ks)?;
        let top_ps_ptr = device_ptr_of::<f32>(&self.d_top_ps)?;
        let min_ps_ptr = device_ptr_of::<f32>(&self.d_min_ps)?;
        let randoms_ptr = device_ptr_of::<f32>(&self.d_uniforms)?;

        match dtype {
            DType::F32 => {
                let logits_ptr = device_ptr_of::<f32>(&logits)?;
                unsafe {
                    cuda_ffi::sample_batched_f32(
                        out_ptr as *mut u32,
                        logits_ptr as *const f32,
                        vocab_size as i32,
                        batch_size as i32,
                        temps_ptr as *const f32,
                        top_ks_ptr as *const i32,
                        top_ps_ptr as *const f32,
                        min_ps_ptr as *const f32,
                        randoms_ptr as *const f32,
                    );
                }
            }
            DType::F16 => {
                let logits_ptr = device_ptr_of::<half::f16>(&logits)?;
                unsafe {
                    cuda_ffi::sample_batched_f16(
                        out_ptr as *mut u32,
                        logits_ptr as *const u16,
                        vocab_size as i32,
                        batch_size as i32,
                        temps_ptr as *const f32,
                        top_ks_ptr as *const i32,
                        top_ps_ptr as *const f32,
                        min_ps_ptr as *const f32,
                        randoms_ptr as *const f32,
                    );
                }
            }
            DType::BF16 => {
                let logits_ptr = device_ptr_of::<half::bf16>(&logits)?;
                unsafe {
                    cuda_ffi::sample_batched_bf16(
                        out_ptr as *mut u32,
                        logits_ptr as *const u16,
                        vocab_size as i32,
                        batch_size as i32,
                        temps_ptr as *const f32,
                        top_ks_ptr as *const i32,
                        top_ps_ptr as *const f32,
                        min_ps_ptr as *const f32,
                        randoms_ptr as *const f32,
                    );
                }
            }
            _ => {
                return Err(KernelError::Other(format!(
                    "unsupported dtype for batched GPU sampling: {dtype:?}"
                )));
            }
        }

        // Single sync: copy output token IDs to host.
        // Use narrow to only read `batch_size` elements from the larger buffer.
        let output_slice = self.d_output.narrow(0, 0, batch_size)?;
        let token_ids = output_slice.to_vec1::<u32>()?;
        self.host_output[..batch_size].copy_from_slice(&token_ids);
        Ok(&self.host_output[..batch_size])
    }
}

/// Write `data` into the first `data.len()` elements of a pre-allocated
/// device tensor via `memcpy_htod_sync` (no allocation).
#[cfg(feature = "cuda")]
fn memcpy_htod_into<T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
    tensor: &Tensor,
    data: &[T],
) -> KernelResult<()> {
    use cudarc::driver::DevicePtr;
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(cs) => {
            let slice = cs.as_cuda_slice::<T>()?;
            let view = slice.slice(layout.start_offset()..);
            let dev_ptr = {
                let cuda_dev = tensor
                    .device()
                    .as_cuda_device()
                    .map_err(|e| KernelError::Other(format!("{e}")))?;
                let stream = cuda_dev.cuda_stream();
                let (ptr, _guard) = view.device_ptr(&stream);
                ptr
            };
            unsafe {
                cudarc::driver::result::memcpy_htod_sync(dev_ptr, data)
                    .map_err(|e| KernelError::Other(format!("memcpy_htod_sync: {e}")))?;
            }
            Ok(())
        }
        _ => Err(KernelError::Other("expected CUDA tensor".into())),
    }
}

/// Extract a raw device pointer from a contiguous CUDA tensor.
#[cfg(feature = "cuda")]
fn device_ptr_of<T: cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use candle_core::{DType, Device, Tensor};

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Device {
        Device::new_cuda(0).expect("CUDA device required for this test")
    }

    /// Test greedy-like sampling: very low temperature should pick the argmax.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_greedy_like() {
        let dev = cuda_device();
        // Create logits where token 3 has the highest value.
        let logits_data: Vec<f32> = vec![1.0, 2.0, 3.0, 10.0, 0.5, 0.1, 0.2, 0.3];
        let logits = Tensor::from_vec(logits_data, (8,), &dev).unwrap();

        // With very low temperature (effectively greedy), should pick token 3.
        let token = super::cuda_sample_top_k_top_p(&logits, 0.01, 0, 1.0, 0.0, 0.5).unwrap();
        assert_eq!(token, 3, "expected argmax token 3");
    }

    /// Test top-k=1 always picks the argmax regardless of random value.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_top_k_1() {
        let dev = cuda_device();
        let logits_data: Vec<f32> = vec![0.0, 0.0, 5.0, 0.0, 0.0];
        let logits = Tensor::from_vec(logits_data, (5,), &dev).unwrap();

        for u in [0.0f32, 0.25, 0.5, 0.75, 0.99] {
            let token = super::cuda_sample_top_k_top_p(&logits, 1.0, 1, 1.0, 0.0, u).unwrap();
            assert_eq!(token, 2, "top_k=1 should always pick argmax (u={u})");
        }
    }

    /// Test that top-p restricts to high-probability tokens.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_top_p_restricts() {
        let dev = cuda_device();
        // Token 0 has ~73% probability after softmax, token 1 ~27%.
        // Others are negligible. With top_p=0.5, only token 0 should survive.
        let logits_data: Vec<f32> = vec![2.0, 1.0, -10.0, -10.0, -10.0];
        let logits = Tensor::from_vec(logits_data, (5,), &dev).unwrap();

        for u in [0.0f32, 0.5, 0.99] {
            let token = super::cuda_sample_top_k_top_p(&logits, 1.0, 0, 0.5, 0.0, u).unwrap();
            assert_eq!(token, 0, "top_p=0.5 should only keep token 0 (u={u})");
        }
    }

    /// Test min_p filtering: only tokens with prob >= min_p * max_prob survive.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_min_p() {
        let dev = cuda_device();
        // After softmax(T=1): token 0 ≈ 0.84, token 1 ≈ 0.11, token 2 ≈ 0.04
        let logits_data: Vec<f32> = vec![3.0, 1.0, 0.0, -20.0, -20.0];
        let logits = Tensor::from_vec(logits_data, (5,), &dev).unwrap();

        // min_p = 0.2 means keep tokens with prob >= 0.2 * 0.84 ≈ 0.17.
        // Only token 0 survives.
        for u in [0.0f32, 0.5, 0.99] {
            let token = super::cuda_sample_top_k_top_p(&logits, 1.0, 0, 1.0, 0.2, u).unwrap();
            assert_eq!(token, 0, "min_p=0.2 should only keep token 0 (u={u})");
        }
    }

    /// Test with bf16 logits.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_bf16() {
        let dev = cuda_device();
        let logits_data: Vec<f32> = vec![0.0, 0.0, 10.0, 0.0];
        let logits = Tensor::from_vec(logits_data, (4,), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let token = super::cuda_sample_top_k_top_p(&logits, 0.1, 0, 1.0, 0.0, 0.5).unwrap();
        assert_eq!(token, 2, "bf16: expected argmax token 2");
    }

    /// Test with f16 logits.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_f16() {
        let dev = cuda_device();
        let logits_data: Vec<f32> = vec![0.0, 0.0, 0.0, 10.0];
        let logits = Tensor::from_vec(logits_data, (4,), &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let token = super::cuda_sample_top_k_top_p(&logits, 0.1, 0, 1.0, 0.0, 0.5).unwrap();
        assert_eq!(token, 3, "f16: expected argmax token 3");
    }

    /// Sampling distribution test: with uniform logits and no filtering,
    /// all tokens should be sampled roughly equally over many draws.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_distribution() {
        let dev = cuda_device();
        let vocab_size = 4;
        let logits_data: Vec<f32> = vec![0.0; vocab_size];
        let logits = Tensor::from_vec(logits_data, (vocab_size,), &dev).unwrap();

        let mut counts = vec![0u32; vocab_size];
        let num_samples = 1000;
        for i in 0..num_samples {
            let u = (i as f32 + 0.5) / num_samples as f32;
            let token =
                super::cuda_sample_top_k_top_p(&logits, 1.0, 0, 1.0, 0.0, u).unwrap() as usize;
            assert!(token < vocab_size, "token {token} out of range");
            counts[token] += 1;
        }

        // Each token should get roughly 25% of samples.
        let expected = num_samples as f32 / vocab_size as f32;
        for (i, &c) in counts.iter().enumerate() {
            let ratio = c as f32 / expected;
            assert!(
                (0.5..1.5).contains(&ratio),
                "token {i}: count {c}, expected ~{expected}, ratio {ratio}"
            );
        }
    }

    /// Test with a realistic vocab size (32K).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_large_vocab() {
        let dev = cuda_device();
        let vocab_size = 32000;
        let mut logits_data: Vec<f32> = vec![0.0; vocab_size];
        logits_data[12345] = 20.0; // dominant token
        let logits = Tensor::from_vec(logits_data, (vocab_size,), &dev).unwrap();

        let token = super::cuda_sample_top_k_top_p(&logits, 1.0, 50, 0.9, 0.0, 0.5).unwrap();
        assert_eq!(token, 12345, "large vocab: expected dominant token 12345");
    }

    /// Test batched sampling: 4 requests, each with a different dominant token.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_batched() {
        let dev = cuda_device();
        let vocab_size = 8;
        let batch_size = 4;

        // Each row has one dominant token at a different position.
        let mut logits_data = vec![0.0f32; batch_size * vocab_size];
        logits_data[0 * vocab_size + 2] = 20.0; // request 0 → token 2
        logits_data[1 * vocab_size + 5] = 20.0; // request 1 → token 5
        logits_data[2 * vocab_size + 0] = 20.0; // request 2 → token 0
        logits_data[3 * vocab_size + 7] = 20.0; // request 3 → token 7

        let logits = Tensor::from_vec(logits_data, (batch_size, vocab_size), &dev).unwrap();

        let temperatures = vec![1.0f32; batch_size];
        let top_ks = vec![1i32; batch_size]; // greedy via top_k=1
        let top_ps = vec![1.0f32; batch_size];
        let min_ps = vec![0.0f32; batch_size];
        let uniforms = vec![0.5f32; batch_size];

        let tokens = super::cuda_sample_batched(
            &logits,
            &temperatures,
            &top_ks,
            &top_ps,
            &min_ps,
            &uniforms,
        )
        .unwrap();

        assert_eq!(tokens.len(), batch_size);
        assert_eq!(tokens[0], 2, "batch[0] should pick token 2");
        assert_eq!(tokens[1], 5, "batch[1] should pick token 5");
        assert_eq!(tokens[2], 0, "batch[2] should pick token 0");
        assert_eq!(tokens[3], 7, "batch[3] should pick token 7");
    }

    /// Test batched sampling with mixed parameters per request.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_batched_mixed_params() {
        let dev = cuda_device();
        let vocab_size = 5;
        let batch_size = 2;

        // Request 0: token 0 dominant, greedy (top_k=1).
        // Request 1: token 3 dominant, with top_p filtering.
        let mut logits_data = vec![0.0f32; batch_size * vocab_size];
        logits_data[0 * vocab_size + 0] = 20.0;
        logits_data[1 * vocab_size + 3] = 20.0;

        let logits = Tensor::from_vec(logits_data, (batch_size, vocab_size), &dev).unwrap();

        let temperatures = vec![1.0, 0.5];
        let top_ks = vec![1, 0];
        let top_ps = vec![1.0, 0.9];
        let min_ps = vec![0.0, 0.0];
        let uniforms = vec![0.5, 0.5];

        let tokens = super::cuda_sample_batched(
            &logits,
            &temperatures,
            &top_ks,
            &top_ps,
            &min_ps,
            &uniforms,
        )
        .unwrap();

        assert_eq!(tokens[0], 0, "batch[0] greedy should pick token 0");
        assert_eq!(tokens[1], 3, "batch[1] should pick dominant token 3");
    }

    /// Test batched sampling with bf16 logits.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_sampling_batched_bf16() {
        let dev = cuda_device();
        let vocab_size = 4;
        let batch_size = 2;

        let mut logits_data = vec![0.0f32; batch_size * vocab_size];
        logits_data[0 * vocab_size + 1] = 20.0;
        logits_data[1 * vocab_size + 3] = 20.0;

        let logits = Tensor::from_vec(logits_data, (batch_size, vocab_size), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let tokens = super::cuda_sample_batched(
            &logits,
            &[0.1, 0.1],
            &[1, 1],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[0.5, 0.5],
        )
        .unwrap();

        assert_eq!(tokens[0], 1);
        assert_eq!(tokens[1], 3);
    }
}
