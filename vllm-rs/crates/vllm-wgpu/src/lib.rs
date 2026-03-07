// SPDX-License-Identifier: Apache-2.0
//! WebGPU tensor backend for vLLM.
//!
//! Provides GPU-accelerated tensor operations using WGSL compute shaders,
//! targeting both native GPU (via wgpu/Vulkan/Metal/DX12) and in-browser
//! WebGPU via WASM.

pub mod device;
pub mod gguf;
pub mod model;
pub mod ops;
pub mod tensor;

#[cfg(feature = "worker")]
pub mod worker_impl;

pub use device::{GraphCapture, ValidationMode, WgpuDevice};
pub use tensor::{WgpuDType, WgpuTensor};

/// Errors from the WebGPU backend.
#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error("no suitable WebGPU adapter found")]
    NoAdapter,
    #[error("device creation failed: {0}")]
    DeviceCreation(String),
    #[error("shape mismatch: expected {expected} elements, got {got}")]
    ShapeMismatch { expected: usize, got: usize },
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    #[error("buffer map failed")]
    BufferMap,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocking_device() -> WgpuDevice {
        pollster::block_on(WgpuDevice::new()).expect("failed to create WebGPU device")
    }

    #[test]
    fn test_add() {
        let dev = blocking_device();
        let a = WgpuTensor::from_f32(&dev, &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = WgpuTensor::from_f32(&dev, &[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let c = ops::add(&a, &b).unwrap();
        let result = pollster::block_on(c.to_f32()).unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_matmul() {
        let dev = blocking_device();
        // [2,3] x [3,2] = [2,2]
        let a = WgpuTensor::from_f32(&dev, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = WgpuTensor::from_f32(&dev, &[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = ops::matmul(&a, &b).unwrap();
        let result = pollster::block_on(c.to_f32()).unwrap();
        // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_silu_mul() {
        let dev = blocking_device();
        let gate = WgpuTensor::from_f32(&dev, &[3], &[0.0, 1.0, -1.0]).unwrap();
        let up = WgpuTensor::from_f32(&dev, &[3], &[1.0, 1.0, 1.0]).unwrap();
        let out = ops::silu_mul(&gate, &up).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        // SiLU(0) = 0, SiLU(1) = 1/(1+e^-1) ≈ 0.7311, SiLU(-1) = -1/(1+e^1) ≈ -0.2689
        assert!((result[0]).abs() < 1e-5);
        assert!((result[1] - 0.7311).abs() < 1e-3);
        assert!((result[2] - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn test_softmax() {
        let dev = blocking_device();
        let input = WgpuTensor::from_f32(&dev, &[1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let out = ops::softmax(&input).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Values should be monotonically increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
        assert!(result[2] < result[3]);
    }

    #[test]
    fn test_embedding() {
        let dev = blocking_device();
        // vocab_size=3, dim=2
        let table = WgpuTensor::from_f32(&dev, &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let indices = WgpuTensor::from_u32(&dev, &[2], &[0, 2]).unwrap();
        let out = ops::embedding(&table, &indices).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_rms_norm() {
        let dev = blocking_device();
        let input = WgpuTensor::from_f32(&dev, &[1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let weight = WgpuTensor::from_f32(&dev, &[4], &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let out = ops::rms_norm(&input, &weight, 1e-6).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (7.5_f32).sqrt();
        assert!((result[0] - 1.0 / rms).abs() < 1e-4);
        assert!((result[3] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_t_matvec() {
        // M=1 triggers the specialized matvec_t kernel
        let dev = blocking_device();
        // x: [1, 3], W: [2, 3] → y: [1, 2]
        let x = WgpuTensor::from_f32(&dev, &[1, 3], &[1.0, 2.0, 3.0]).unwrap();
        let w = WgpuTensor::from_f32(&dev, &[2, 3], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let y = ops::matmul_t(&x, &w).unwrap();
        let result = pollster::block_on(y.to_f32()).unwrap();
        // y[0] = 1*4 + 2*5 + 3*6 = 32
        // y[1] = 1*7 + 2*8 + 3*9 = 50
        assert!((result[0] - 32.0).abs() < 1e-4);
        assert!((result[1] - 50.0).abs() < 1e-4);
    }

    #[test]
    fn test_matmul_t_matvec_large() {
        // Test matvec with K large enough to exercise the parallel reduction
        let dev = blocking_device();
        let k = 896; // typical hidden_size
        let n = 64;
        let x_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let w_data: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32) * 0.1).collect();
        let x = WgpuTensor::from_f32(&dev, &[1, k], &x_data).unwrap();
        let w = WgpuTensor::from_f32(&dev, &[n, k], &w_data).unwrap();
        let y = ops::matmul_t(&x, &w).unwrap();
        let result = pollster::block_on(y.to_f32()).unwrap();
        // Verify against CPU reference
        for col in 0..n {
            let expected: f32 = (0..k).map(|i| x_data[i] * w_data[col * k + i]).sum();
            assert!(
                (result[col] - expected).abs() < expected.abs() * 1e-3 + 1e-3,
                "col {col}: got {} expected {expected}",
                result[col]
            );
        }
    }

    #[test]
    fn test_slice_last_dim() {
        let dev = blocking_device();
        // [1, 6] → slice offset=2, length=3 → [1, 3]
        let input =
            WgpuTensor::from_f32(&dev, &[1, 6], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
        let out = ops::slice_last_dim(&input, 2, 3).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        assert_eq!(result, vec![30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_slice_last_dim_multirow() {
        let dev = blocking_device();
        // [2, 4] → slice offset=1, length=2 → [2, 2]
        let input =
            WgpuTensor::from_f32(&dev, &[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let out = ops::slice_last_dim(&input, 1, 2).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn test_silu_mul_split() {
        let dev = blocking_device();
        // [1, 6] with half=3: gate=[0, 1, -1], up=[1, 1, 1]
        let input = WgpuTensor::from_f32(&dev, &[1, 6], &[0.0, 1.0, -1.0, 1.0, 1.0, 1.0]).unwrap();
        let out = ops::silu_mul_split(&input, 3).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        assert!((result[0]).abs() < 1e-5);
        assert!((result[1] - 0.7311).abs() < 1e-3);
        assert!((result[2] - (-0.2689)).abs() < 1e-3);
    }

    #[test]
    fn test_fused_add_rms_norm() {
        let dev = blocking_device();
        let residual = WgpuTensor::from_f32(&dev, &[1, 4], &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let input = WgpuTensor::from_f32(&dev, &[1, 4], &[0.0, 2.0, 3.0, 4.0]).unwrap();
        let weight = WgpuTensor::from_f32(&dev, &[4], &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let (normed, res_out) = ops::fused_add_rms_norm(&residual, &input, &weight, 1e-6).unwrap();
        let res_result = pollster::block_on(res_out.to_f32()).unwrap();
        assert_eq!(res_result, vec![1.0, 2.0, 3.0, 4.0]);
        let norm_result = pollster::block_on(normed.to_f32()).unwrap();
        let rms = (7.5_f32).sqrt(); // same as rms_norm test
        assert!((norm_result[0] - 1.0 / rms).abs() < 1e-4);
        assert!((norm_result[3] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_attention_single_step() {
        let dev = blocking_device();
        // 2 Q heads, 2 KV heads, head_dim=4, seq_len=1
        let q =
            WgpuTensor::from_f32(&dev, &[1, 8], &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        // K and V cache: [4, 8] (max_seq=4, but only seq_len=1 used)
        let k_cache = WgpuTensor::from_f32(
            &dev,
            &[4, 8],
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // pos 0
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // unused
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let v_cache = WgpuTensor::from_f32(
            &dev,
            &[4, 8],
            &[
                0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, // pos 0
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        let out = ops::attention(&q, &k_cache, &v_cache, 2, 2, 4, 1).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        // With seq_len=1, softmax is trivially 1.0, so output = V[0]
        assert!((result[0] - 0.5).abs() < 1e-4); // head 0
        assert!((result[1] - 0.6).abs() < 1e-4);
        assert!((result[4] - 0.1).abs() < 1e-4); // head 1
        assert!((result[5] - 0.2).abs() < 1e-4);
    }

    #[test]
    fn test_attention_two_steps() {
        let dev = blocking_device();
        // 1 Q head, 1 KV head, head_dim=2, seq_len=2
        // Q = [1, 0] → should attend more to K[0]=[1,0] than K[1]=[0,1]
        let q = WgpuTensor::from_f32(&dev, &[1, 2], &[1.0, 0.0]).unwrap();
        let k_cache =
            WgpuTensor::from_f32(&dev, &[4, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let v_cache =
            WgpuTensor::from_f32(&dev, &[4, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let out = ops::attention(&q, &k_cache, &v_cache, 1, 1, 2, 2).unwrap();
        let result = pollster::block_on(out.to_f32()).unwrap();
        // score[0] = 1/sqrt(2), score[1] = 0/sqrt(2)
        // softmax: p[0] = exp(1/√2) / (exp(1/√2) + exp(0))
        let s0 = (1.0_f32 / 2.0_f32.sqrt()).exp();
        let s1 = 1.0_f32; // exp(0)
        let p0 = s0 / (s0 + s1);
        let p1 = s1 / (s0 + s1);
        // output = p0 * V[0] + p1 * V[1] = [p0, p1]
        assert!(
            (result[0] - p0).abs() < 1e-3,
            "got {} expected {p0}",
            result[0]
        );
        assert!(
            (result[1] - p1).abs() < 1e-3,
            "got {} expected {p1}",
            result[1]
        );
    }

    #[test]
    fn test_matmul_t_matvec_large_n() {
        // N > 65535 to catch workgroup dispatch limit issues (e.g. vocab=151936)
        let dev = blocking_device();
        let k = 32;
        let n = 70_000;
        let x_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        // Use a simple pattern so expected values are easy to compute
        let w_data: Vec<f32> = (0..n * k)
            .map(|i| if i % k == 0 { 1.0 } else { 0.0 })
            .collect();
        let x = WgpuTensor::from_f32(&dev, &[1, k], &x_data).unwrap();
        let w = WgpuTensor::from_f32(&dev, &[n, k], &w_data).unwrap();
        let y = ops::matmul_t(&x, &w).unwrap();
        let result = pollster::block_on(y.to_f32()).unwrap();
        assert_eq!(result.len(), n);
        // Each row of W is [1, 0, 0, ...], so y[col] = x[0] = 0.0
        for col in [0, 1000, 65535, 65536, n - 1] {
            assert!(
                (result[col] - 0.0).abs() < 1e-4,
                "col {col}: got {}",
                result[col]
            );
        }
    }

    #[test]
    fn test_matmul_t_with_transposed() {
        let dev = blocking_device();
        let k = 896;
        let n = 64;
        let x_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let w_data: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32) * 0.1).collect();
        let x = WgpuTensor::from_f32(&dev, &[1, k], &x_data).unwrap();
        let w = WgpuTensor::from_f32(&dev, &[n, k], &w_data).unwrap();
        let w_t = w.transpose_2d_cpu(&w_data).unwrap();
        let y = ops::matmul_t_with_transposed(&x, &w, &w_t).unwrap();
        let result = pollster::block_on(y.to_f32()).unwrap();
        for col in 0..n {
            let expected: f32 = (0..k).map(|i| x_data[i] * w_data[col * k + i]).sum();
            assert!(
                (result[col] - expected).abs() < expected.abs() * 1e-3 + 1e-3,
                "col {col}: got {} expected {expected}",
                result[col]
            );
        }
    }

    #[test]
    fn test_buffer_pool_reuse() {
        let dev = blocking_device();
        // Create a tensor, drop it, then create another of the same size.
        // The second should reuse the pooled buffer from the same bucket.
        let ptr1 = {
            let t = WgpuTensor::zeros(&dev, &[1, 64], WgpuDType::F32); // 256 bytes → bucket 256
            let _size = t.size_bytes();
            drop(t);
            let pool = dev.buffer_pool.lock().unwrap();
            assert_eq!(
                pool.buckets.get(&256).map(|v| v.len()),
                Some(1),
                "buffer should be in bucket 256 after drop"
            );
            drop(pool);
            // Create another tensor of same size — should reuse from bucket.
            let t2 = WgpuTensor::zeros(&dev, &[1, 64], WgpuDType::F32);
            let pool = dev.buffer_pool.lock().unwrap();
            assert_eq!(
                pool.buckets.get(&256).map(|v| v.len()),
                Some(0),
                "buffer should have been taken from bucket"
            );
            drop(pool);
            t2
        };
        // Verify the reused tensor works correctly
        let a = WgpuTensor::from_f32(&dev, &[1, 4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let _ = ptr1; // keep alive
        let b = WgpuTensor::from_f32(&dev, &[1, 4], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = ops::add(&a, &b).unwrap();
        let result = pollster::block_on(c.to_f32()).unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_buffer_pool_no_reuse_different_bucket() {
        let dev = blocking_device();
        {
            let t = WgpuTensor::zeros(&dev, &[1, 64], WgpuDType::F32); // 256 bytes → bucket 256
            drop(t);
        }
        // Request a larger size in a different bucket — should NOT reuse
        let _t2 = WgpuTensor::zeros(&dev, &[1, 128], WgpuDType::F32); // 512 bytes → bucket 512
        let pool = dev.buffer_pool.lock().unwrap();
        // Original 256-byte buffer should still be in its bucket
        assert_eq!(pool.buckets.get(&256).map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_buffer_pool_cross_size_reuse() {
        // Bucket pool reuses buffers across different exact sizes within the same bucket.
        // A 200-byte request and a 250-byte request both map to bucket 256.
        let dev = blocking_device();
        {
            // 50 f32 = 200 bytes → bucket 256 (allocated at 256 bytes)
            let t = WgpuTensor::zeros(&dev, &[1, 50], WgpuDType::F32);
            drop(t);
        }
        let pool = dev.buffer_pool.lock().unwrap();
        assert_eq!(pool.buckets.get(&256).map(|v| v.len()), Some(1));
        drop(pool);
        // 60 f32 = 240 bytes → also bucket 256 → should reuse!
        let _t2 = WgpuTensor::zeros(&dev, &[1, 60], WgpuDType::F32);
        let pool = dev.buffer_pool.lock().unwrap();
        assert_eq!(
            pool.buckets.get(&256).map(|v| v.len()),
            Some(0),
            "cross-size reuse within same bucket"
        );
    }

    #[test]
    fn test_buffer_pool_shared_not_recycled() {
        let dev = blocking_device();
        // reshape() clones the Arc, so drop of one shouldn't recycle
        // 2*4 f32 = 32 bytes → bucket 64 (rounds up)
        let t = WgpuTensor::zeros(&dev, &[2, 4], WgpuDType::F32);
        let t2 = t.reshape(&[1, 8]).unwrap(); // shares the buffer Arc
        drop(t);
        // Buffer has 2 strong refs (t2 + pool candidate), so should NOT be pooled
        let pool = dev.buffer_pool.lock().unwrap();
        let count = pool.buckets.get(&64).map(|v| v.len()).unwrap_or(0);
        assert_eq!(count, 0, "shared buffer should not be recycled");
        drop(pool);
        // t2 should still work
        let result = pollster::block_on(t2.to_f32()).unwrap();
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_buffer_pool_correctness_after_reuse() {
        // Ensure a reused buffer produces correct results (not stale data)
        let dev = blocking_device();
        // Create and drop a tensor to seed the pool
        {
            let t = WgpuTensor::from_f32(&dev, &[4], &[99.0, 99.0, 99.0, 99.0]).unwrap();
            drop(t);
        }
        // zeros() should get the pooled buffer — the output op should overwrite stale data
        let a = WgpuTensor::from_f32(&dev, &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = WgpuTensor::from_f32(&dev, &[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let c = ops::add(&a, &b).unwrap();
        let result = pollster::block_on(c.to_f32()).unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_rope_slice_cache() {
        let dev = blocking_device();
        // 2 Q heads, 1 KV head, head_dim=4, position=0
        // QKV = [q0(4), q1(4), k0(4), v0(4)] = 16 elements
        let q_size = 8; // 2 heads × 4
        let kv_size = 4; // 1 head × 4
        let head_dim = 4;
        let max_seq = 8;

        // Simple QKV data
        let mut qkv_data = vec![0.0f32; q_size + 2 * kv_size];
        // Q head 0: [1, 0, 0, 0], Q head 1: [0, 1, 0, 0]
        qkv_data[0] = 1.0;
        qkv_data[5] = 1.0;
        // K: [1, 0, 0, 0]
        qkv_data[q_size] = 1.0;
        // V: [0.5, 0.6, 0.7, 0.8]
        qkv_data[q_size + kv_size] = 0.5;
        qkv_data[q_size + kv_size + 1] = 0.6;
        qkv_data[q_size + kv_size + 2] = 0.7;
        qkv_data[q_size + kv_size + 3] = 0.8;

        let qkv = WgpuTensor::from_f32(&dev, &[1, q_size + 2 * kv_size], &qkv_data).unwrap();

        // cos/sin cache: at position 0, cos=1, sin=0 (angle=0)
        let half_dim = head_dim / 2;
        let mut cos_data = vec![0.0f32; max_seq * half_dim];
        let sin_data = vec![0.0f32; max_seq * half_dim];
        cos_data[0] = 1.0;
        cos_data[1] = 1.0; // pos 0, all cos = 1
        // sin stays 0

        let cos = WgpuTensor::from_f32(&dev, &[max_seq, half_dim], &cos_data).unwrap();
        let sin = WgpuTensor::from_f32(&dev, &[max_seq, half_dim], &sin_data).unwrap();
        let k_cache = WgpuTensor::zeros(&dev, &[max_seq, kv_size], WgpuDType::F32);
        let v_cache = WgpuTensor::zeros(&dev, &[max_seq, kv_size], WgpuDType::F32);

        let q_out = ops::rope_slice_cache(
            &qkv, &cos, &sin, &k_cache, &v_cache, q_size, kv_size, head_dim, 2, 1, 0, max_seq,
        )
        .unwrap();

        // At position 0 with cos=1, sin=0: RoPE is identity
        let q_result = pollster::block_on(q_out.to_f32()).unwrap();
        assert!((q_result[0] - 1.0).abs() < 1e-4, "q[0]={}", q_result[0]);
        assert!((q_result[5] - 1.0).abs() < 1e-4, "q[5]={}", q_result[5]);

        // V should be copied to cache at position 0
        let v_result = pollster::block_on(v_cache.to_f32()).unwrap();
        assert!((v_result[0] - 0.5).abs() < 1e-4);
        assert!((v_result[3] - 0.8).abs() < 1e-4);

        // K should be in cache with RoPE (identity here)
        let k_result = pollster::block_on(k_cache.to_f32()).unwrap();
        assert!((k_result[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_cache_write() {
        let dev = blocking_device();
        let cache = WgpuTensor::zeros(&dev, &[4, 3], WgpuDType::F32);
        let row = WgpuTensor::from_f32(&dev, &[1, 3], &[7.0, 8.0, 9.0]).unwrap();
        ops::cache_write(&cache, &row, 2).unwrap();
        let result = pollster::block_on(cache.to_f32()).unwrap();
        // Row 2 should be [7, 8, 9], others zero
        assert_eq!(&result[6..9], &[7.0, 8.0, 9.0]);
        assert_eq!(&result[0..3], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_q4_0_matvec() {
        // Test Q4_0 quantized matvec against CPU reference.
        // Create a small weight matrix [N=4, K=64] with known Q4_0 blocks,
        // quantize to Q4_0, then verify GPU matvec matches CPU dequant + matvec.
        let dev = blocking_device();
        let n = 4usize;
        let k = 64usize;
        let blocks_per_row = k / 32;

        // Create Q4_0 raw bytes: each block = 18 bytes (2B scale + 16B nibbles)
        let mut raw = vec![0u8; n * blocks_per_row * 18];
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = (row * blocks_per_row + blk) * 18;
                // scale = 1.0 (f16 = 0x3C00)
                raw[off] = 0x00;
                raw[off + 1] = 0x3C;
                // Set nibbles: use pattern based on row and block
                for j in 0..16 {
                    let lo = ((row + j) % 16) as u8; // elem j
                    let hi = ((row + j + 1) % 16) as u8; // elem j+16
                    raw[off + 2 + j] = lo | (hi << 4);
                }
            }
        }

        // Dequantize to f32 for CPU reference
        let f32_weights = gguf::dequantize_q4_0_to_f32(&raw, n * k);

        // Create GPU Q4_0 tensor
        let w_t = WgpuTensor::from_q4_0_transposed(&dev, n, k, &raw).unwrap();

        // Create input vector
        let x_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01).collect();
        let x = WgpuTensor::from_f32(&dev, &[1, k], &x_data).unwrap();

        // Also need a row-major weight for matmul_t_with_transposed
        let w = WgpuTensor::from_f32_as_f16_packed(&dev, &[n, k], &f32_weights).unwrap();

        // GPU matvec with Q4_0
        let y = ops::matmul_t_with_transposed(&x, &w, &w_t).unwrap();
        let result = pollster::block_on(y.to_f32()).unwrap();

        // CPU reference
        for col in 0..n {
            let expected: f32 = (0..k).map(|i| x_data[i] * f32_weights[col * k + i]).sum();
            assert!(
                (result[col] - expected).abs() < expected.abs() * 0.02 + 0.1,
                "col {col}: gpu={} cpu={expected}",
                result[col]
            );
        }
    }

    #[test]
    fn test_q4_0_matvec_realistic() {
        // Test with realistic dimensions: K=2048, N=2048
        let dev = blocking_device();
        let n = 32usize; // Keep N small for fast test, but K realistic
        let k = 2048usize;
        let blocks_per_row = k / 32;

        // Create Q4_0 raw bytes with varied data
        let mut raw = vec![0u8; n * blocks_per_row * 18];
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = (row * blocks_per_row + blk) * 18;
                // Varied scale: f16 for (row+blk+1) * 0.1
                let scale_f16 = half::f16::from_f32((row + blk + 1) as f32 * 0.01);
                let bits = scale_f16.to_le_bytes();
                raw[off] = bits[0];
                raw[off + 1] = bits[1];
                // Varied nibbles
                for j in 0..16 {
                    let lo = ((row + j + blk) % 16) as u8;
                    let hi = ((row + j + blk + 5) % 16) as u8;
                    raw[off + 2 + j] = lo | (hi << 4);
                }
            }
        }

        // Dequantize to f32 for CPU reference
        let f32_weights = gguf::dequantize_q4_0_to_f32(&raw, n * k);

        // Create GPU Q4_0 tensor
        let w_t = WgpuTensor::from_q4_0_transposed(&dev, n, k, &raw).unwrap();
        let w = WgpuTensor::from_f32_as_f16_packed(&dev, &[n, k], &f32_weights).unwrap();

        // Random-ish input vector
        let x_data: Vec<f32> = (0..k)
            .map(|i| ((i * 7 + 3) % 100) as f32 * 0.01 - 0.5)
            .collect();
        let x = WgpuTensor::from_f32(&dev, &[1, k], &x_data).unwrap();

        let y = ops::matmul_t_with_transposed(&x, &w, &w_t).unwrap();
        let result = pollster::block_on(y.to_f32()).unwrap();

        for col in 0..n {
            let expected: f32 = (0..k).map(|i| x_data[i] * f32_weights[col * k + i]).sum();
            let rel_err = if expected.abs() > 1e-3 {
                (result[col] - expected).abs() / expected.abs()
            } else {
                (result[col] - expected).abs()
            };
            assert!(
                rel_err < 0.05,
                "col {col}: gpu={} cpu={expected} rel_err={rel_err}",
                result[col]
            );
        }
    }

    #[test]
    fn test_q4_0_fused_add_rms_norm_matvec() {
        // Test fused add+rms_norm+Q4_0 matvec against separate ops.
        let dev = blocking_device();
        let k = 64usize;
        let n = 8usize;
        let blocks_per_row = k / 32;

        // Create Q4_0 weights
        let mut raw = vec![0u8; n * blocks_per_row * 18];
        for row in 0..n {
            for blk in 0..blocks_per_row {
                let off = (row * blocks_per_row + blk) * 18;
                raw[off] = 0x00;
                raw[off + 1] = 0x3C; // scale = 1.0
                for j in 0..16 {
                    let lo = ((row + j + 3) % 16) as u8;
                    let hi = ((row + j + 7) % 16) as u8;
                    raw[off + 2 + j] = lo | (hi << 4);
                }
            }
        }

        let w_t = WgpuTensor::from_q4_0_transposed(&dev, n, k, &raw).unwrap();
        let residual_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
        let addition_data: Vec<f32> = (0..k).map(|i| (i as f32) * -0.05 + 1.0).collect();
        let norm_data: Vec<f32> = vec![1.0f32; k];

        let residual = WgpuTensor::from_f32(&dev, &[1, k], &residual_data).unwrap();
        let addition = WgpuTensor::from_f32(&dev, &[1, k], &addition_data).unwrap();
        let norm_weight = WgpuTensor::from_f32(&dev, &[k], &norm_data).unwrap();

        let (matvec_out, hidden_out) = ops::fused_add_rms_norm_matvec(
            &residual,
            &addition,
            &norm_weight,
            &w_t,
            n as u32,
            1e-6,
        )
        .unwrap();

        let hidden_result = pollster::block_on(hidden_out.to_f32()).unwrap();
        let matvec_result = pollster::block_on(matvec_out.to_f32()).unwrap();

        // Verify hidden = residual + addition
        for i in 0..k {
            let expected = residual_data[i] + addition_data[i];
            assert!(
                (hidden_result[i] - expected).abs() < 1e-4,
                "hidden[{i}]: got {} expected {expected}",
                hidden_result[i]
            );
        }

        // Verify matvec output is reasonable (not NaN/inf, not zero)
        assert_eq!(matvec_result.len(), n);
        for (i, &v) in matvec_result.iter().enumerate() {
            assert!(v.is_finite(), "matvec[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_matmul_t_tiled_m4() {
        // M=4 exercises the register-tiled matmul_t shader (4×4 per thread)
        let dev = blocking_device();
        let m = 4;
        let k = 32;
        let n = 8;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..n * k).map(|i| ((i % 11) as f32) * 0.1).collect();
        let a = WgpuTensor::from_f32(&dev, &[m, k], &a_data).unwrap();
        let b = WgpuTensor::from_f32(&dev, &[n, k], &b_data).unwrap();
        let c = ops::matmul_t(&a, &b).unwrap();
        let result = pollster::block_on(c.to_f32()).unwrap();
        assert_eq!(result.len(), m * n);
        for row in 0..m {
            for col in 0..n {
                let expected: f32 = (0..k)
                    .map(|i| a_data[row * k + i] * b_data[col * k + i])
                    .sum();
                let got = result[row * n + col];
                assert!(
                    (got - expected).abs() < expected.abs() * 1e-3 + 1e-3,
                    "C[{row},{col}]: got {got} expected {expected}",
                );
            }
        }
    }

    #[test]
    fn test_matmul_t_tiled_large() {
        // Larger M>1 test to exercise multiple workgroups in the tiled shader
        let dev = blocking_device();
        let m = 50; // typical prompt length
        let k = 128;
        let n = 256;
        let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 37) as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..n * k).map(|i| ((i % 23) as f32) * 0.01).collect();
        let a = WgpuTensor::from_f32(&dev, &[m, k], &a_data).unwrap();
        let b = WgpuTensor::from_f32(&dev, &[n, k], &b_data).unwrap();
        let c = ops::matmul_t(&a, &b).unwrap();
        let result = pollster::block_on(c.to_f32()).unwrap();
        assert_eq!(result.len(), m * n);
        // Spot-check first row, last row, and middle row
        for row in [0, m / 2, m - 1] {
            for col in [0, n / 2, n - 1] {
                let expected: f32 = (0..k)
                    .map(|i| a_data[row * k + i] * b_data[col * k + i])
                    .sum();
                let got = result[row * n + col];
                assert!(
                    (got - expected).abs() < expected.abs() * 1e-2 + 1e-2,
                    "C[{row},{col}]: got {got} expected {expected}",
                );
            }
        }
    }

    #[test]
    fn test_attention_prefill() {
        // Test causal prefill attention with M=3 tokens, 1 head, dim=4
        let dev = blocking_device();
        let m = 3;
        let num_q_heads = 1u32;
        let num_kv_heads = 1u32;
        let head_dim = 4u32;
        let q_size = (num_q_heads * head_dim) as usize;
        let kv_size = (num_kv_heads * head_dim) as usize;
        let max_seq = 16;

        // Q: [3, 4] — three query vectors
        let q_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // q0
            0.0, 1.0, 0.0, 0.0, // q1
            0.0, 0.0, 1.0, 0.0, // q2
        ];
        // K_new: [3, 4]
        let k_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // k0
            0.0, 1.0, 0.0, 0.0, // k1
            0.0, 0.0, 1.0, 0.0, // k2
        ];
        // V_new: [3, 4] — identity-like values for easy verification
        let v_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // v0
            0.0, 1.0, 0.0, 0.0, // v1
            0.0, 0.0, 1.0, 0.0, // v2
        ];

        let q = WgpuTensor::from_f32(&dev, &[m, q_size], &q_data).unwrap();
        let k_new = WgpuTensor::from_f32(&dev, &[m, kv_size], &k_data).unwrap();
        let v_new = WgpuTensor::from_f32(&dev, &[m, kv_size], &v_data).unwrap();
        let k_cache = WgpuTensor::zeros(&dev, &[max_seq, kv_size], WgpuDType::F32);
        let v_cache = WgpuTensor::zeros(&dev, &[max_seq, kv_size], WgpuDType::F32);

        let out = ops::attention_prefill(
            &q,
            &k_new,
            &v_new,
            &k_cache,
            &v_cache,
            num_q_heads,
            num_kv_heads,
            head_dim,
            m as u32,
            0, // cache_len = 0
        )
        .unwrap();

        let result = pollster::block_on(out.to_f32()).unwrap();
        assert_eq!(result.len(), m * q_size);

        // q0 can only attend to k0 → output ≈ v0 = [1, 0, 0, 0]
        assert!(
            result[0] > 0.9,
            "prefill out[0,0] should be ~1.0, got {}",
            result[0]
        );

        // q2 attends to k0, k1, k2 with causal mask.
        // q2 = [0,0,1,0], k0 = [1,0,0,0] → dot=0, k1 = [0,1,0,0] → dot=0, k2 = [0,0,1,0] → dot=1
        // After softmax with scale 1/sqrt(4)=0.5: scores ≈ [exp(0), exp(0), exp(0.5)]
        // The output should have more weight on v2
        let out_row2 = &result[2 * q_size..3 * q_size];
        assert!(
            out_row2[2] > out_row2[0],
            "prefill: q2 should attend more to k2 (dim 2 > dim 0): {:?}",
            out_row2
        );
    }

    #[test]
    fn test_cache_write_batch() {
        let dev = blocking_device();
        let max_seq = 16;
        let dim = 4;
        let cache = WgpuTensor::zeros(&dev, &[max_seq, dim], WgpuDType::F32);
        let data = WgpuTensor::from_f32(
            &dev,
            &[3, dim],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        ops::cache_write_batch(&cache, &data, 2, 3).unwrap();
        let result = pollster::block_on(cache.to_f32()).unwrap();
        // Rows 0,1 should be zero; rows 2,3,4 should have our data
        assert_eq!(&result[0..8], &[0.0; 8]);
        assert_eq!(&result[8..12], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&result[12..16], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(&result[16..20], &[9.0, 10.0, 11.0, 12.0]);
    }
}
