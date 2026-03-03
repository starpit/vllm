// SPDX-License-Identifier: Apache-2.0
//! CUDA graph pool: manages captured graphs indexed by padded batch size.

use std::collections::HashMap;

use cudarc::driver::CudaGraph;

/// A single captured CUDA graph entry for a specific padded batch size.
pub struct CudaGraphEntry {
    pub graph: CudaGraph,
    pub padded_batch_size: usize,
}

/// Pool of captured CUDA graphs indexed by padded batch size.
///
/// At runtime, the actual batch size is rounded up to the nearest captured
/// size. If the batch size exceeds all captured sizes, the caller falls
/// back to eager (non-graph) execution.
pub struct CudaGraphPool {
    entries: HashMap<usize, CudaGraphEntry>,
    /// Sorted ascending list of capture sizes.
    capture_sizes: Vec<usize>,
}

impl CudaGraphPool {
    pub fn new(capture_sizes: Vec<usize>) -> Self {
        let mut sizes = capture_sizes;
        sizes.sort();
        sizes.dedup();
        Self {
            entries: HashMap::new(),
            capture_sizes: sizes,
        }
    }

    /// The configured capture sizes (sorted ascending).
    pub fn capture_sizes(&self) -> &[usize] {
        &self.capture_sizes
    }

    /// Find the smallest capture size >= `actual_bs`.
    ///
    /// Returns `None` if `actual_bs` exceeds all capture sizes (caller
    /// should fall back to eager execution).
    pub fn padded_size(&self, actual_bs: usize) -> Option<usize> {
        self.capture_sizes.iter().copied().find(|&s| s >= actual_bs)
    }

    /// Store a captured graph for a given padded batch size.
    pub fn insert(&mut self, padded_bs: usize, graph: CudaGraph) {
        self.entries.insert(
            padded_bs,
            CudaGraphEntry {
                graph,
                padded_batch_size: padded_bs,
            },
        );
    }

    /// Get the captured graph for a padded batch size.
    pub fn get(&self, padded_bs: usize) -> Option<&CudaGraphEntry> {
        self.entries.get(&padded_bs)
    }

    /// Number of captured graphs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether any graphs have been captured.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padded_size_exact() {
        let pool = CudaGraphPool::new(vec![1, 2, 4, 8, 16]);
        assert_eq!(pool.padded_size(1), Some(1));
        assert_eq!(pool.padded_size(4), Some(4));
        assert_eq!(pool.padded_size(16), Some(16));
    }

    #[test]
    fn test_padded_size_round_up() {
        let pool = CudaGraphPool::new(vec![1, 2, 4, 8, 16]);
        assert_eq!(pool.padded_size(3), Some(4));
        assert_eq!(pool.padded_size(5), Some(8));
        assert_eq!(pool.padded_size(9), Some(16));
    }

    #[test]
    fn test_padded_size_too_large() {
        let pool = CudaGraphPool::new(vec![1, 2, 4, 8, 16]);
        assert_eq!(pool.padded_size(17), None);
        assert_eq!(pool.padded_size(256), None);
    }

    #[test]
    fn test_padded_size_dedup_and_sort() {
        let pool = CudaGraphPool::new(vec![8, 4, 4, 2, 8, 1]);
        assert_eq!(pool.capture_sizes(), &[1, 2, 4, 8]);
        assert_eq!(pool.padded_size(3), Some(4));
    }

    #[test]
    fn test_empty_pool() {
        let pool = CudaGraphPool::new(vec![1, 2, 4]);
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert!(pool.get(1).is_none());
    }

    /// Test that CUDA graph capture works at the cudarc level.
    ///
    /// Creates a non-blocking stream (same as candle) and verifies
    /// begin_capture + end_capture works after GPU operations (simulating warmup).
    #[test]
    fn test_cuda_stream_capture_basic() {
        use cudarc::driver::{CudaContext, DevicePtrMut, ValidAsZeroBits, sys::CUstreamCaptureMode};

        let ctx = CudaContext::new(0).expect("failed to create CUDA context");
        // new_stream() creates CU_STREAM_NON_BLOCKING — same as candle.
        let stream = ctx.new_stream().expect("failed to create stream");

        // Simulate warmup: allocate, write, free GPU memory.
        for _ in 0..3 {
            let mut buf = stream.alloc_zeros::<f32>(1024).expect("alloc failed");
            let host_data = vec![1.0f32; 1024];
            stream.memcpy_htod(&host_data, &mut buf).expect("memcpy failed");
            drop(buf); // triggers cuMemFreeAsync on this stream
        }
        stream.synchronize().expect("sync failed");

        // Now try to capture.
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .expect("begin_capture failed after warmup");

        // Do a capturable operation: alloc + memset.
        let buf = stream.alloc_zeros::<f32>(256).expect("alloc during capture");

        let flags = cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph = stream
            .end_capture(flags)
            .expect("end_capture failed");

        assert!(graph.is_some(), "graph should not be None after alloc");
        eprintln!("Graph captured successfully with {} nodes", if graph.is_some() { "some" } else { "zero" });

        // Launch the graph.
        graph.as_ref().unwrap().launch().expect("launch failed");
        stream.synchronize().expect("post-launch sync failed");
    }

    /// Test capture after cuBLAS operations (simulating candle's matmul warmup).
    #[test]
    fn test_cuda_stream_capture_after_cublas() {
        use cudarc::driver::{CudaContext, DevicePtrMut, sys::CUstreamCaptureMode};
        use cudarc::cublas::CudaBlas;

        let ctx = CudaContext::new(0).expect("failed to create CUDA context");
        let stream = ctx.new_stream().expect("failed to create stream");
        let blas = CudaBlas::new(stream.clone()).expect("failed to create cuBLAS handle");

        // Simulate warmup: do a cuBLAS GEMM.
        let n = 64usize;
        let mut a = stream.alloc_zeros::<f32>(n * n).expect("alloc A");
        let mut b = stream.alloc_zeros::<f32>(n * n).expect("alloc B");
        let mut c = stream.alloc_zeros::<f32>(n * n).expect("alloc C");

        unsafe {
            let (a_ptr, _ga) = a.device_ptr_mut(&stream);
            let (b_ptr, _gb) = b.device_ptr_mut(&stream);
            let (c_ptr, _gc) = c.device_ptr_mut(&stream);
            cudarc::cublas::result::sgemm(
                *blas.handle(),
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32, n as i32, n as i32,
                &1.0f32 as *const f32,
                a_ptr as *const f32, n as i32,
                b_ptr as *const f32, n as i32,
                &0.0f32 as *const f32,
                c_ptr as *mut f32, n as i32,
            ).expect("sgemm failed");
        }

        drop(a);
        drop(b);
        drop(c);
        stream.synchronize().expect("sync failed");

        // Now try capture after cuBLAS warmup.
        stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .expect("begin_capture failed after cuBLAS warmup");

        let buf = stream.alloc_zeros::<f32>(256).expect("alloc during capture");

        let flags = cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph = stream
            .end_capture(flags)
            .expect("end_capture failed");

        assert!(graph.is_some());
        eprintln!("Graph capture after cuBLAS: OK");
    }

    /// Minimal candle test: just Device::new_cuda and capture.
    #[test]
    fn test_cuda_stream_capture_candle_device_only() {
        use candle_core::Device;
        use cudarc::driver::sys::CUstreamCaptureMode;

        let device = Device::new_cuda(0).expect("device");
        let stream = match &device {
            Device::Cuda(cd) => cd.cuda_stream(),
            _ => panic!("not CUDA"),
        };

        eprintln!("context ordinal: {}", stream.context().ordinal());
        eprintln!("multi-stream: {}", stream.context().is_in_multi_stream_mode());
        eprintln!("event tracking: {}", stream.context().is_event_tracking());
        eprintln!("managing sync: {}", stream.context().is_managing_stream_synchronization());

        stream.synchronize().expect("sync");

        match stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED) {
            Ok(()) => {
                let flags = cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                let _ = stream.end_capture(flags);
                eprintln!("Capture on candle device stream: OK");
            }
            Err(e) => {
                panic!("begin_capture failed on candle device stream: {e}");
            }
        }
    }

    /// Test: does creating TWO streams (multi-stream mode) break capture?
    ///
    /// candle's `Device::new_cuda` calls `context.new_stream()` which
    /// puts the context into multi-stream mode. This test reproduces that.
    #[test]
    fn test_cuda_stream_capture_multi_stream() {
        use cudarc::driver::{CudaContext, sys::CUstreamCaptureMode};

        let ctx = CudaContext::new(0).expect("ctx");

        // First stream — the one we'll use for cuBLAS and capture.
        let stream1 = ctx.new_stream().expect("stream1");
        // Second stream — puts context in multi-stream mode.
        let _stream2 = ctx.new_stream().expect("stream2");

        eprintln!("Multi-stream mode: {}", ctx.is_in_multi_stream_mode());
        eprintln!("Event tracking: {}", ctx.is_event_tracking());

        let _blas = cudarc::cublas::CudaBlas::new(stream1.clone()).expect("blas");

        stream1.synchronize().expect("sync");

        match stream1.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED) {
            Ok(()) => {
                let flags = cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
                let _ = stream1.end_capture(flags);
                eprintln!("Capture in multi-stream mode: OK");
            }
            Err(e) => {
                panic!("begin_capture failed in multi-stream mode: {e}");
            }
        }
    }
}
