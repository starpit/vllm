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
}
