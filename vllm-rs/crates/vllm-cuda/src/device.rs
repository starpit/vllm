// SPDX-License-Identifier: Apache-2.0
//! `GpuDevice`: the central runtime object combining streams, cuBLAS, and arena.
//!
//! One `GpuDevice` per GPU. All kernel launches go on `compute_stream`.
//! The `transfer_stream` handles async H2D/D2H copies for overlap.

use crate::arena::ScratchArena;
use crate::cublas::CublasHandle;
use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;
use anyhow::Result;
use cudarc::driver::sys::{CUcontext, CUevent, CUstream};

/// Initial scratch arena size (512 MB — grows during warmup if needed).
const INITIAL_ARENA_SIZE: usize = 512 * 1024 * 1024;

/// The GPU device runtime. Owns streams, cuBLAS handle, and scratch arena.
///
/// All model forward passes operate through this struct. One instance per GPU.
pub struct GpuDevice {
    pub device_id: i32,
    pub ctx: CUcontext,
    pub compute_stream: CUstream,
    pub transfer_stream: CUstream,
    pub cublas: CublasHandle,
    pub arena: ScratchArena,
    /// Event for gating CPU reuse of pinned buffers after H2D transfer.
    pub transfer_done: CUevent,
}

impl GpuDevice {
    /// Initialize a GPU device.
    ///
    /// Creates CUDA context, two streams, cuBLAS handle, and scratch arena.
    pub fn new(device_id: i32) -> Result<Self> {
        unsafe {
            driver::init()?;
            let cu_device = driver::device_get(device_id)?;
            let ctx = driver::ctx_create(cu_device)?;

            let compute_stream = driver::stream_create()?;
            let transfer_stream = driver::stream_create()?;
            let transfer_done = driver::event_create_disable_timing()?;

            let cublas = CublasHandle::new(compute_stream)?;
            let arena = ScratchArena::new(INITIAL_ARENA_SIZE)?;

            tracing::info!(
                "GpuDevice initialized: device={}, arena={:.0} MB",
                device_id,
                INITIAL_ARENA_SIZE as f64 / (1024.0 * 1024.0),
            );

            Ok(Self {
                device_id,
                ctx,
                compute_stream,
                transfer_stream,
                cublas,
                arena,
                transfer_done,
            })
        }
    }

    /// Allocate a tensor from the scratch arena.
    pub fn alloc(&mut self, shape: &[usize], dtype: DType) -> GpuTensor {
        self.arena.alloc(shape, dtype)
    }

    /// GEMM: out = a @ b^T, output allocated from arena.
    ///
    /// # Safety
    /// `a` and `b` must be valid GPU tensors.
    pub unsafe fn gemm(&mut self, a: GpuTensor, b: GpuTensor) -> GpuTensor {
        self.cublas.gemm(a, b, &mut self.arena)
    }

    /// Reset the scratch arena. Call once per engine step after sampling.
    pub fn reset_arena(&mut self) {
        self.arena.reset();
    }

    /// Lock the arena after warmup.
    pub fn lock_arena(&mut self) {
        self.arena.lock();
    }

    /// Synchronize the compute stream (block until all compute completes).
    pub fn sync_compute(&self) -> Result<()> {
        unsafe { driver::stream_synchronize(self.compute_stream) }
    }

    /// Synchronize the transfer stream.
    pub fn sync_transfer(&self) -> Result<()> {
        unsafe { driver::stream_synchronize(self.transfer_stream) }
    }

    /// Record an event on the transfer stream and make compute wait for it.
    /// Used after H2D copy to ensure inputs are ready before forward pass.
    pub fn sync_transfer_to_compute(&self) -> Result<()> {
        unsafe {
            driver::event_record(self.transfer_done, self.transfer_stream)?;
            driver::stream_wait_event(self.compute_stream, self.transfer_done)?;
        }
        Ok(())
    }

    /// Allocate persistent device memory (not from arena).
    /// For weight buffers, KV cache, etc.
    ///
    /// # Safety
    /// Caller owns the returned pointer and must free it with `driver::mem_free`.
    pub unsafe fn alloc_persistent(&self, bytes: usize) -> Result<*mut u8> {
        driver::mem_alloc(bytes)
    }

    /// Async D2D copy on compute stream.
    pub unsafe fn copy_dtod(&self, dst: *mut u8, src: *const u8, bytes: usize) -> Result<()> {
        driver::memcpy_dtod_async(dst, src, bytes, self.compute_stream)
    }

    /// Zero GPU memory on compute stream.
    pub unsafe fn memset_zero(&self, ptr: *mut u8, bytes: usize) -> Result<()> {
        driver::memset_d8(ptr, 0, bytes, self.compute_stream)
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        unsafe {
            // Arena is dropped automatically (has its own Drop).
            // CublasHandle is dropped automatically.
            let _ = driver::event_destroy(self.transfer_done);
            let _ = driver::stream_destroy(self.transfer_stream);
            if !self.compute_stream.is_null() {
                let _ = driver::stream_destroy(self.compute_stream);
            }
            // Don't destroy context here — it may be shared.
            // cuCtxDestroy happens when the process exits.
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn test_device_create_and_drop() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        assert_eq!(dev.device_id, 0);
        assert!(!dev.compute_stream.is_null());
        assert!(!dev.transfer_stream.is_null());
        assert_ne!(
            dev.compute_stream as usize, dev.transfer_stream as usize,
            "compute and transfer streams should differ"
        );
        drop(dev);
    }

    #[test]
    fn test_device_arena_alloc() {
        let mut dev = GpuDevice::new(0).expect("GpuDevice::new");
        let t = dev.alloc(&[32, 4096], DType::BF16);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.dim(0), 32);
        assert_eq!(t.dim(1), 4096);
        assert!(!t.is_null());
    }

    #[test]
    fn test_device_arena_reset() {
        let mut dev = GpuDevice::new(0).expect("GpuDevice::new");
        let t1 = dev.alloc(&[64], DType::F32);
        assert!(dev.arena.used() > 0);

        dev.reset_arena();
        assert_eq!(dev.arena.used(), 0);

        let t2 = dev.alloc(&[64], DType::F32);
        assert_eq!(t1.raw_ptr(), t2.raw_ptr());
    }

    #[test]
    fn test_device_arena_lock() {
        let mut dev = GpuDevice::new(0).expect("GpuDevice::new");
        dev.alloc(&[128], DType::F32);
        dev.reset_arena();
        dev.lock_arena();
        assert!(dev.arena.is_locked());

        // Should still be able to alloc within capacity.
        let t = dev.alloc(&[128], DType::F32);
        assert!(!t.is_null());
    }

    #[test]
    fn test_device_sync_compute() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        dev.sync_compute().expect("sync compute");
    }

    #[test]
    fn test_device_sync_transfer() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        dev.sync_transfer().expect("sync transfer");
    }

    #[test]
    fn test_device_sync_transfer_to_compute() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        dev.sync_transfer_to_compute()
            .expect("sync transfer to compute");
    }

    #[test]
    fn test_device_alloc_persistent() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        unsafe {
            let ptr = dev.alloc_persistent(1024).expect("persistent alloc");
            assert!(!ptr.is_null());
            driver::mem_free(ptr).expect("free");
        }
    }

    #[test]
    fn test_device_memset_zero() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        unsafe {
            let ptr = dev.alloc_persistent(256).expect("alloc");
            dev.memset_zero(ptr, 256).expect("memset");
            dev.sync_compute().expect("sync");

            // Verify zeros.
            let host = driver::mem_alloc_host(256).expect("host");
            driver::memcpy_dtoh_async(host, ptr, 256, dev.compute_stream).expect("dtoh");
            driver::stream_synchronize(dev.compute_stream).expect("sync");

            let bytes = std::slice::from_raw_parts(host, 256);
            for (i, &b) in bytes.iter().enumerate() {
                assert_eq!(b, 0, "not zero at byte {i}");
            }

            driver::mem_free_host(host).unwrap();
            driver::mem_free(ptr).unwrap();
        }
    }

    #[test]
    fn test_device_copy_dtod() {
        let dev = GpuDevice::new(0).expect("GpuDevice::new");
        unsafe {
            let src = dev.alloc_persistent(128).expect("src");
            let dst = dev.alloc_persistent(128).expect("dst");

            // Fill src with pattern via host.
            let host = driver::mem_alloc_host(128).expect("host");
            for i in 0..128 {
                *host.add(i) = (i * 5) as u8;
            }
            driver::memcpy_htod_async(src, host, 128, dev.compute_stream).expect("htod");

            // D2D copy.
            dev.copy_dtod(dst, src, 128).expect("dtod");
            dev.sync_compute().expect("sync");

            // Read back.
            let host_out = driver::mem_alloc_host(128).expect("host_out");
            driver::memcpy_dtoh_async(host_out, dst, 128, dev.compute_stream).expect("dtoh");
            driver::stream_synchronize(dev.compute_stream).expect("sync");

            for i in 0..128 {
                assert_eq!(*host_out.add(i), (i * 5) as u8, "mismatch at {i}");
            }

            driver::mem_free_host(host).unwrap();
            driver::mem_free_host(host_out).unwrap();
            driver::mem_free(src).unwrap();
            driver::mem_free(dst).unwrap();
        }
    }

    #[test]
    fn test_device_gemm_f32() {
        let mut dev = GpuDevice::new(0).expect("GpuDevice::new");
        unsafe {
            // Simple 2x2 GEMM: A @ B^T
            let host_a = driver::mem_alloc_host(16).unwrap();
            let host_b = driver::mem_alloc_host(16).unwrap();
            std::slice::from_raw_parts_mut(host_a as *mut f32, 4)
                .copy_from_slice(&[1.0, 0.0, 0.0, 1.0]); // identity
            std::slice::from_raw_parts_mut(host_b as *mut f32, 4)
                .copy_from_slice(&[2.0, 3.0, 4.0, 5.0]);

            let gpu_a = driver::mem_alloc(16).unwrap();
            let gpu_b = driver::mem_alloc(16).unwrap();
            driver::memcpy_htod_async(gpu_a, host_a, 16, dev.compute_stream).unwrap();
            driver::memcpy_htod_async(gpu_b, host_b, 16, dev.compute_stream).unwrap();

            let a = GpuTensor::new(gpu_a, &[2, 2], DType::F32);
            let b = GpuTensor::new(gpu_b, &[2, 2], DType::F32);

            // I @ B^T = B^T = [[2, 4], [3, 5]]
            let c = dev.gemm(a, b);

            let host_c = driver::mem_alloc_host(16).unwrap();
            driver::memcpy_dtoh_async(host_c, c.raw_ptr(), 16, dev.compute_stream).unwrap();
            driver::stream_synchronize(dev.compute_stream).unwrap();

            let result = std::slice::from_raw_parts(host_c as *const f32, 4);
            // A = I, B = [[2,3],[4,5]], so C = I @ B^T = [[2,4],[3,5]]
            let expected = [2.0, 4.0, 3.0, 5.0];
            for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "gemm mismatch at {i}: got {got}, expected {exp}"
                );
            }

            driver::mem_free_host(host_a).unwrap();
            driver::mem_free_host(host_b).unwrap();
            driver::mem_free_host(host_c).unwrap();
            driver::mem_free(gpu_a).unwrap();
            driver::mem_free(gpu_b).unwrap();
        }
    }

    #[test]
    fn test_device_simulated_forward_step() {
        // Simulate multiple allocs + reset + re-alloc (one engine step).
        let mut dev = GpuDevice::new(0).expect("GpuDevice::new");

        // "Step 1": alloc some activations.
        let _t1 = dev.alloc(&[32, 4096], DType::BF16);
        let _t2 = dev.alloc(&[32, 4096], DType::BF16);
        let _t3 = dev.alloc(&[32, 11008], DType::BF16);
        let used_step1 = dev.arena.used();
        assert!(used_step1 > 0);

        dev.reset_arena();
        assert_eq!(dev.arena.used(), 0);

        // "Step 2": same alloc pattern, same addresses.
        let _t4 = dev.alloc(&[32, 4096], DType::BF16);
        let _t5 = dev.alloc(&[32, 4096], DType::BF16);
        let _t6 = dev.alloc(&[32, 11008], DType::BF16);
        let used_step2 = dev.arena.used();
        assert_eq!(used_step1, used_step2);
    }
}
