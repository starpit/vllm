// SPDX-License-Identifier: Apache-2.0
//! CUDA graph runner: captures and replays CUDA graphs for decode-step acceleration.
//!
//! During warmup, this module captures the entire model `forward_batch()` call for
//! each configured batch size into a CUDA graph. At runtime, decode steps whose batch
//! size fits a captured graph are replayed in a single driver call instead of launching
//! ~150 individual kernels.
//!
//! **Important**: The current implementation captures the full forward pass including
//! KV cache reads and writes. During graph capture, dummy KV cache blocks (block 0) are
//! used. During replay, the graph reads/writes to the same dummy blocks. After replay,
//! the caller must do the real KV cache scatter eagerly. Full correctness requires a
//! graph-compatible attention kernel with GPU-side block tables (future work).

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use cudarc::driver::DevicePtr;
use tracing::info;
use vllm_kernels::cuda_graph::CudaGraphPool;
use vllm_models::{AttentionMetadata, BatchedKvCacheStorage, KvBlockPool, Model};

use crate::error::ExecutorError;

type Result<T> = std::result::Result<T, ExecutorError>;

fn err_init(msg: impl std::fmt::Display) -> ExecutorError {
    ExecutorError::WorkerInit(msg.to_string())
}

fn err_exec(msg: impl std::fmt::Display) -> ExecutorError {
    ExecutorError::WorkerExecution(msg.to_string())
}

/// Pre-allocated tensors for a single graph capture size.
///
/// These tensors live at fixed GPU addresses. Before replay, real data is
/// copied into them via raw CUDA `cuMemcpyHtoD` (same allocation, no realloc).
struct GraphBuffers {
    /// Input token IDs `[padded_bs]`, on GPU.
    #[allow(dead_code)]
    input_ids: Tensor,
    /// Position indices `[padded_bs]`, on GPU.
    #[allow(dead_code)]
    positions: Tensor,
    /// Output logits `[padded_bs, vocab_size]`, captured during graph capture.
    output_logits: Tensor,
    /// Raw device pointer for input_ids (for in-place memcpy before replay).
    input_ids_dev_ptr: u64,
    /// Raw device pointer for positions (for in-place memcpy before replay).
    positions_dev_ptr: u64,
    /// Padded batch size.
    #[allow(dead_code)]
    padded_bs: usize,
}

/// Extract the raw CUDA device pointer from a candle Tensor's U32 storage.
///
/// The tensor must be on a CUDA device with contiguous U32 storage.
fn extract_device_ptr_u32(tensor: &Tensor, device: &candle_core::CudaDevice) -> Result<u64> {
    let (storage, _layout) = tensor.storage_and_layout();
    match &*storage {
        candle_core::Storage::Cuda(cuda_storage) => {
            let slice: &cudarc::driver::CudaSlice<u32> = cuda_storage
                .as_cuda_slice::<u32>()
                .map_err(|e| err_init(format!("failed to get CudaSlice<u32>: {e}")))?;
            let stream = device.cuda_stream();
            let (dev_ptr, _sync_guard) = slice.device_ptr(&stream);
            Ok(dev_ptr)
        }
        _ => Err(err_init("expected CUDA storage")),
    }
}

/// Copy host `u32` data into a pre-allocated GPU buffer at a known device pointer.
///
/// Uses `cudarc::driver::result::memcpy_htod_sync` to write into the tensor's
/// existing GPU allocation without going through candle (which doesn't expose
/// mutable storage access from outside the crate).
///
/// # Safety
/// - `dev_ptr` must point to a valid GPU allocation of at least
///   `data.len() * size_of::<u32>()` bytes.
/// - The caller must ensure no concurrent reads from this allocation.
unsafe fn memcpy_htod_u32_raw(dev_ptr: u64, data: &[u32]) -> Result<()> {
    unsafe {
        cudarc::driver::result::memcpy_htod_sync(dev_ptr, data)
            .map_err(|e| err_exec(format!("cuMemcpyHtoD_v2 failed: {e}")))
    }
}

/// Trim the CUDA memory pool, releasing all cached (freed-but-not-returned) memory.
///
/// After warmup, dropped tensors free GPU memory via `cuMemFreeAsync`. The
/// stream-ordered allocator caches this memory in a pool. Trimming ensures no
/// pool bookkeeping operations are pending on the stream, which could block
/// `cuStreamBeginCapture`.
unsafe fn trim_memory_pool(ordinal: usize) -> Result<()> {
    use cudarc::driver::sys;
    use std::mem::MaybeUninit;

    unsafe {
        let cu_device = ordinal as sys::CUdevice;
        let mut pool = MaybeUninit::<sys::CUmemoryPool>::uninit();
        let res = sys::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), cu_device);
        if res != sys::CUresult::CUDA_SUCCESS {
            return Err(err_init(format!(
                "cuDeviceGetDefaultMemPool failed: {res:?}"
            )));
        }
        let pool = pool.assume_init();
        let res = sys::cuMemPoolTrimTo(pool, 0);
        if res != sys::CUresult::CUDA_SUCCESS {
            return Err(err_init(format!("cuMemPoolTrimTo failed: {res:?}")));
        }
    }
    Ok(())
}

/// Manages CUDA graph capture and replay for decode steps.
pub struct CudaGraphRunner {
    pool: CudaGraphPool,
    /// Pre-allocated buffers per capture size.
    buffers: HashMap<usize, GraphBuffers>,
    /// Model vocab size (needed to pre-allocate output tensor).
    #[allow(dead_code)]
    vocab_size: usize,
    /// Model dtype.
    #[allow(dead_code)]
    dtype: DType,
    /// Device (must be CUDA).
    device: Device,
    /// Number of warmup runs before capture.
    num_warmups: usize,
    /// Whether graphs have been captured.
    captured: bool,
}

// SAFETY: CudaGraph contains raw CUDA pointers which are !Send by default.
// However, CUDA graph objects are safe to move between threads; they only
// require serialized access (no concurrent use). CandleWorker owns the
// runner and processes requests sequentially on a single thread.
unsafe impl Send for CudaGraphRunner {}

impl CudaGraphRunner {
    /// Create a new runner with the given capture sizes.
    ///
    /// Graphs are not captured until `capture_graphs()` is called.
    pub fn new(
        capture_sizes: Vec<usize>,
        vocab_size: usize,
        dtype: DType,
        device: Device,
        num_warmups: usize,
    ) -> Self {
        Self {
            pool: CudaGraphPool::new(capture_sizes),
            buffers: HashMap::new(),
            vocab_size,
            dtype,
            device,
            num_warmups,
            captured: false,
        }
    }

    /// Capture CUDA graphs for all configured batch sizes.
    ///
    /// Must be called after model + KV cache are fully initialized.
    /// Uses dummy data for capture — the graph records kernel launches and
    /// memory access patterns, not actual data values.
    pub fn capture_graphs(
        &mut self,
        model: &dyn Model,
        kv_block_pool: &mut KvBlockPool,
        _block_size: usize,
    ) -> Result<()> {
        use candle_core::backend::BackendDevice;
        use cudarc::driver::sys::CUstreamCaptureMode;

        let cuda_dev = self
            .device
            .as_cuda_device()
            .map_err(|e| err_init(format!("CudaGraphRunner requires a CUDA device: {e}")))?;
        let stream = cuda_dev.cuda_stream();

        let capture_sizes: Vec<usize> = self.pool.capture_sizes().to_vec();

        for &padded_bs in &capture_sizes {
            info!("Capturing CUDA graph for batch size {padded_bs}...");

            // 1. Create pre-allocated input tensors at fixed addresses.
            let input_ids =
                Tensor::zeros(padded_bs, DType::U32, &self.device).map_err(err_init)?;
            let positions =
                Tensor::zeros(padded_bs, DType::U32, &self.device).map_err(err_init)?;

            // Extract raw device pointers for later in-place memcpy.
            let input_ids_dev_ptr = extract_device_ptr_u32(&input_ids, cuda_dev)?;
            let positions_dev_ptr = extract_device_ptr_u32(&positions, cuda_dev)?;

            // 2. Build dummy AttentionMetadata — all decode, 1 token each.
            let dummy_seq_lens = vec![1usize; padded_bs];
            let dummy_block_ids: Vec<Vec<usize>> = vec![vec![0]; padded_bs];
            let dummy_tokens_before = vec![0usize; padded_bs];
            let attn_meta = AttentionMetadata::padded_decode(
                padded_bs,
                padded_bs,
                &dummy_seq_lens,
                &dummy_block_ids,
                &dummy_tokens_before,
            );

            // 3. Build dummy BatchedKvCacheStorage (writes go to block 0, harmless).
            let batch_block_ids: Vec<Vec<usize>> = vec![vec![0]; padded_bs];
            let batch_tokens_before: Vec<usize> = vec![0; padded_bs];

            // 4. Warmup runs — ensures kernel selection is stable.
            for warmup_idx in 0..self.num_warmups {
                let mut storage = BatchedKvCacheStorage::new(
                    kv_block_pool,
                    batch_block_ids.clone(),
                    batch_tokens_before.clone(),
                );
                let _ = model
                    .forward_batch(&input_ids, &positions, &attn_meta, &mut storage)
                    .map_err(|e| err_init(format!("warmup forward failed: {e}")))?;
                storage.flush_all().map_err(err_init)?;
                cuda_dev.synchronize().map_err(err_init)?;
                if warmup_idx == 0 {
                    info!("  warmup {}/{} done", warmup_idx + 1, self.num_warmups);
                }
            }

            // 5. Prepare stream for capture.
            //
            // Synchronize the stream, then trim the CUDA memory pool. The
            // stream-ordered allocator caches freed memory; trimming ensures no
            // pool bookkeeping is pending that could block begin_capture.
            stream
                .synchronize()
                .map_err(|e| err_init(format!("pre-capture sync failed: {e}")))?;
            unsafe { trim_memory_pool(stream.context().ordinal()) }?;

            // Disable cudarc's event tracking during capture. cudarc records
            // CudaEvents on every tensor access for cross-stream synchronization.
            // These event record/wait operations may not be capturable (they can
            // cause CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
            //
            // SAFETY: We only use a single stream, so inter-stream sync is
            // unnecessary. We re-enable tracking after capture completes.
            let ctx = stream.context().clone();
            let was_tracking = ctx.is_event_tracking();
            if was_tracking {
                unsafe { ctx.disable_event_tracking() };
            }

            let mut capture_storage = BatchedKvCacheStorage::new(
                kv_block_pool,
                batch_block_ids,
                batch_tokens_before,
            );

            stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| {
                    // Re-enable tracking on failure.
                    if was_tracking {
                        unsafe { ctx.enable_event_tracking() };
                    }
                    err_init(format!("begin_capture failed: {e}"))
                })?;

            let capture_result = model.forward_batch(
                &input_ids,
                &positions,
                &attn_meta,
                &mut capture_storage,
            );

            let graph_flags = cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;

            let graph_result = match capture_result {
                Ok(output) => {
                    let graph = stream
                        .end_capture(graph_flags)
                        .map_err(|e| err_init(format!("end_capture failed: {e}")))?
                        .ok_or_else(|| {
                            err_init(format!(
                                "CUDA graph capture returned None for bs={padded_bs}"
                            ))
                        })?;
                    Ok((output, graph))
                }
                Err(e) => {
                    // Forward failed during capture — must still end capture to
                    // restore stream state, then propagate the error.
                    let _ = stream.end_capture(graph_flags);
                    Err(err_init(format!("capture forward failed: {e}")))
                }
            };

            // Re-enable event tracking now that capture is done.
            if was_tracking {
                unsafe { ctx.enable_event_tracking() };
            }

            let (output, graph) = graph_result?;

            // 6. Store buffers and graph.
            self.buffers.insert(
                padded_bs,
                GraphBuffers {
                    input_ids,
                    positions,
                    output_logits: output,
                    input_ids_dev_ptr,
                    positions_dev_ptr,
                    padded_bs,
                },
            );
            self.pool.insert(padded_bs, graph);

            info!("  captured CUDA graph for bs={padded_bs}");
        }

        self.captured = true;
        info!(
            "CUDA graph capture complete: {} graphs, vocab_size={}, dtype={:?}",
            self.pool.len(),
            self.vocab_size,
            self.dtype,
        );
        Ok(())
    }

    /// Try to replay a captured graph for a decode batch.
    ///
    /// Returns `Some(logits)` with shape `[actual_bs, vocab_size]` on success,
    /// or `None` if no graph is available (batch too large or not captured).
    /// The caller should fall back to eager `forward_batch()` when this returns `None`.
    pub fn try_replay(
        &self,
        actual_bs: usize,
        input_ids: &[u32],
        positions: &[u32],
    ) -> Result<Option<Tensor>> {
        if !self.captured {
            return Ok(None);
        }

        let padded_bs = match self.pool.padded_size(actual_bs) {
            Some(s) => s,
            None => return Ok(None),
        };

        let entry = match self.pool.get(padded_bs) {
            Some(e) => e,
            None => return Ok(None),
        };

        let buffers = match self.buffers.get(&padded_bs) {
            Some(b) => b,
            None => return Ok(None),
        };

        // Pad input data to padded_bs with zeros.
        let mut padded_ids = vec![0u32; padded_bs];
        padded_ids[..actual_bs].copy_from_slice(input_ids);

        let mut padded_pos = vec![0u32; padded_bs];
        padded_pos[..actual_bs].copy_from_slice(positions);

        // Copy real data into the pre-allocated GPU tensors at fixed addresses.
        // SAFETY: We own these tensors, the graph is not running, and the device
        // pointers were extracted from these tensors during capture_graphs().
        unsafe {
            memcpy_htod_u32_raw(buffers.input_ids_dev_ptr, &padded_ids)?;
            memcpy_htod_u32_raw(buffers.positions_dev_ptr, &padded_pos)?;
        }

        // Replay the captured graph — single CUDA driver call.
        entry
            .graph
            .launch()
            .map_err(|e| err_exec(format!("CUDA graph replay failed: {e}")))?;

        // Return only the first actual_bs rows of the output logits.
        let logits = if actual_bs < padded_bs {
            buffers
                .output_logits
                .narrow(0, 0, actual_bs)
                .map_err(err_exec)?
        } else {
            buffers.output_logits.clone()
        };
        Ok(Some(logits))
    }

    /// Whether graphs have been captured and are ready for replay.
    pub fn is_captured(&self) -> bool {
        self.captured
    }
}
