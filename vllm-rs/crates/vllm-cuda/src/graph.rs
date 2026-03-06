// SPDX-License-Identifier: Apache-2.0
//! CUDA graph capture and replay for decode batches.
//!
//! During decode, every request contributes exactly 1 token (q_len=1), making
//! all tensor shapes deterministic for a given batch size. We capture the
//! entire model forward pass as a CUDA graph during warmup, then replay it
//! during inference — eliminating ~280 kernel launch overheads per step.
//!
//! **How it works:**
//! 1. Persistent input buffers are allocated at max capacity (NOT from the arena).
//! 2. `capture()` resets the arena, copies dummy inputs into the persistent buffers,
//!    begins stream capture, runs the model forward, ends capture, and instantiates
//!    the graph.
//! 3. `replay()` copies real inputs into the same persistent buffers (same device
//!    addresses the graph was captured with), resets the arena (so intermediate
//!    activations land at the same offsets), and launches the graph.
//! 4. The output logits land at a known arena offset (recorded during capture).

use std::collections::HashMap;

use anyhow::Result;
use cudarc::driver::sys::{CUgraphExec, CUstream};

use crate::device::GpuDevice;
use crate::driver;
use crate::dtype::DType;
use crate::kernels;
use crate::tensor::GpuTensor;

/// Maximum number of blocks per sequence in the block table.
/// Covers sequences up to MAX_BLOCKS * block_size tokens.
const MAX_BLOCKS_PER_SEQ: usize = 512;

/// A single captured CUDA graph for a specific batch size.
struct CapturedGraph {
    exec: CUgraphExec,
    /// Where the model wrote output logits during capture (arena address).
    output_ptr: *const u8,
    #[allow(dead_code)]
    output_numel: usize,
    /// Where the argmax kernel wrote token IDs during capture (arena address).
    /// `[batch_size]` u32 tensor. Only present when argmax is captured in graph.
    argmax_ptr: *const u8,
    /// Arena bytes used after the forward pass + argmax. After replay, the arena
    /// offset must be advanced to this value so that post-graph allocations
    /// don't overlap with the graph's outputs.
    arena_used: usize,
}

/// CUDA graph runner for decode batches.
pub struct CudaGraphRunner {
    /// Captured graphs keyed by batch size.
    graphs: HashMap<usize, CapturedGraph>,
    /// Persistent input buffers — same device addresses used for all captures.
    input_ids: *mut u8,
    positions: *mut u8,
    slot_mapping: *mut u8,
    cu_seqlens_q: *mut u8,
    cu_seqlens_k: *mut u8,
    block_table: *mut u8,
    /// Capacity in elements for the 1-D buffers (max batch size).
    max_batch: usize,
    /// Model output dtype.
    dtype: DType,
    /// Vocab size.
    vocab_size: usize,
}

unsafe impl Send for CudaGraphRunner {}

impl CudaGraphRunner {
    /// Allocate persistent input buffers for graph capture.
    ///
    /// # Safety
    /// Requires active CUDA context.
    pub unsafe fn new(max_batch: usize, vocab_size: usize, dtype: DType) -> Result<Self> {
        let input_ids = driver::mem_alloc(max_batch * 4)?; // u32
        let positions = driver::mem_alloc(max_batch * 4)?; // u32
        let slot_mapping = driver::mem_alloc(max_batch * 8)?; // i64
        let cu_seqlens_q = driver::mem_alloc((max_batch + 1) * 4)?; // u32
        let cu_seqlens_k = driver::mem_alloc((max_batch + 1) * 4)?; // u32
        let block_table = driver::mem_alloc(max_batch * MAX_BLOCKS_PER_SEQ * 4)?; // u32

        Ok(Self {
            graphs: HashMap::new(),
            input_ids,
            positions,
            slot_mapping,
            cu_seqlens_q,
            cu_seqlens_k,
            block_table,
            max_batch,
            dtype,
            vocab_size,
        })
    }

    /// Whether we have a captured graph for this batch size (exact match).
    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
    }

    /// Find the smallest captured graph size >= `batch_size`, or None.
    /// This allows padding a smaller batch to use a pre-captured graph.
    pub fn nearest_graph_size(&self, batch_size: usize) -> Option<usize> {
        self.graphs
            .keys()
            .filter(|&&s| s >= batch_size)
            .min()
            .copied()
    }

    /// GpuTensor views into persistent input buffers for a given batch size.
    fn input_tensors(&self, batch_size: usize) -> InputTensors {
        unsafe {
            InputTensors {
                input_ids: GpuTensor::new(self.input_ids, &[batch_size], DType::U32),
                positions: GpuTensor::new(self.positions, &[batch_size], DType::U32),
                slot_mapping: GpuTensor::new(self.slot_mapping, &[batch_size], DType::I64),
                cu_seqlens_q: GpuTensor::new(self.cu_seqlens_q, &[batch_size + 1], DType::U32),
                cu_seqlens_k: GpuTensor::new(self.cu_seqlens_k, &[batch_size + 1], DType::U32),
                block_table: GpuTensor::new(
                    self.block_table,
                    &[batch_size, MAX_BLOCKS_PER_SEQ],
                    DType::U32,
                ),
            }
        }
    }

    /// Capture a CUDA graph for a decode batch of the given size.
    ///
    /// Runs the model forward once under graph capture mode. The arena is reset
    /// before capture so intermediate activations always start at offset 0.
    ///
    /// `forward_fn` should call model.forward with the provided input tensors
    /// and return the output logits tensor.
    ///
    /// # Safety
    /// Requires active CUDA context. Model and KV cache must be initialized.
    pub unsafe fn capture<F>(
        &mut self,
        batch_size: usize,
        device: &mut GpuDevice,
        mut forward_fn: F,
    ) -> Result<()>
    where
        F: FnMut(InputTensors, &mut GpuDevice) -> GpuTensor,
    {
        assert!(batch_size <= self.max_batch);

        // Fill persistent input buffers with dummy decode data.
        let inputs = self.input_tensors(batch_size);
        self.fill_dummy_decode(batch_size, device.compute_stream)?;

        // Warm up: run one eager forward + argmax to populate cuBLAS plans, etc.
        device.arena.reset();
        let warmup_logits = forward_fn(inputs, device);
        let _ = kernels::argmax_batched(warmup_logits, &mut device.arena, device.compute_stream);
        driver::stream_synchronize(device.compute_stream)?;

        // Now capture: forward + argmax + D2D scatter into persistent input_ids.
        device.arena.reset();
        let inputs = self.input_tensors(batch_size);

        driver::stream_begin_capture(device.compute_stream)?;
        let logits = forward_fn(inputs, device);
        let argmax_out = kernels::argmax_batched(logits, &mut device.arena, device.compute_stream);
        // Scatter sampled token IDs into persistent input_ids for the next step.
        driver::memcpy_dtod_async(
            self.input_ids,
            argmax_out.raw_ptr() as *const u8,
            batch_size * 4,
            device.compute_stream,
        )?;
        let graph = driver::stream_end_capture(device.compute_stream)?;

        let exec = driver::graph_instantiate(graph)?;
        driver::graph_destroy(graph)?;

        let output_ptr = logits.raw_ptr() as *const u8;
        let output_numel = logits.numel();
        let argmax_ptr = argmax_out.raw_ptr() as *const u8;
        let arena_used = device.arena.used();
        let captured = CapturedGraph {
            exec,
            output_ptr,
            output_numel,
            argmax_ptr,
            arena_used,
        };
        self.graphs.insert(batch_size, captured);

        tracing::info!(
            "CUDA graph captured for batch_size={} (with argmax), output at {:?} ({} elements), arena_used={} bytes",
            batch_size,
            output_ptr,
            output_numel,
            arena_used,
        );

        // Reset arena after capture (the captured pointers remain valid since
        // arena memory is persistent — only the offset resets).
        device.arena.reset();

        Ok(())
    }

    /// Replay a captured CUDA graph.
    ///
    /// Copies real input data into the persistent buffers, resets the arena
    /// (so intermediate activations land at the same offsets as during capture),
    /// and launches the graph.
    ///
    /// Returns `ReplayOutput` containing the logits tensor and the argmax
    /// token IDs (already computed inside the graph). The argmax result is
    /// also automatically scattered into the persistent `input_ids` buffer
    /// for the next step (captured as part of the graph).
    ///
    /// When `skip_input_ids_h2d` is true, the `input_ids` H2D copy is skipped
    /// because the previous graph replay already scattered argmax results into
    /// the persistent buffer. This is valid when the batch composition hasn't
    /// changed between steps.
    ///
    /// # Safety
    /// Input slices must be correctly sized for `batch_size`.
    pub unsafe fn replay(
        &self,
        batch_size: usize,
        input_ids: &[u32],
        positions: &[u32],
        slot_mapping: &[i64],
        cu_seqlens_q: &[u32],
        cu_seqlens_k: &[u32],
        block_table: &[u32], // flattened [batch_size, num_blocks], padded to MAX_BLOCKS_PER_SEQ
        device: &mut GpuDevice,
        skip_input_ids_h2d: bool,
    ) -> Result<ReplayOutput> {
        let graph = self
            .graphs
            .get(&batch_size)
            .ok_or_else(|| anyhow::anyhow!("no captured graph for batch_size={batch_size}"))?;

        let stream = device.compute_stream;

        // Copy real inputs into persistent buffers.
        if !skip_input_ids_h2d {
            driver::memcpy_htod_async(
                self.input_ids,
                input_ids.as_ptr() as *const u8,
                batch_size * 4,
                stream,
            )?;
        }
        driver::memcpy_htod_async(
            self.positions,
            positions.as_ptr() as *const u8,
            batch_size * 4,
            stream,
        )?;
        driver::memcpy_htod_async(
            self.slot_mapping,
            slot_mapping.as_ptr() as *const u8,
            batch_size * 8,
            stream,
        )?;
        driver::memcpy_htod_async(
            self.cu_seqlens_q,
            cu_seqlens_q.as_ptr() as *const u8,
            (batch_size + 1) * 4,
            stream,
        )?;
        driver::memcpy_htod_async(
            self.cu_seqlens_k,
            cu_seqlens_k.as_ptr() as *const u8,
            (batch_size + 1) * 4,
            stream,
        )?;
        driver::memcpy_htod_async(
            self.block_table,
            block_table.as_ptr() as *const u8,
            batch_size * MAX_BLOCKS_PER_SEQ * 4,
            stream,
        )?;

        // Reset arena so intermediates land at the same offsets as capture.
        device.arena.reset();

        // Launch the captured graph (forward + argmax + D2D scatter).
        driver::graph_launch(graph.exec, stream)?;

        // Advance the arena offset past everything the graph wrote.
        device.arena.set_offset(graph.arena_used);

        let logits = GpuTensor::new(
            graph.output_ptr as *mut u8,
            &[batch_size, self.vocab_size],
            self.dtype,
        );
        let token_ids = GpuTensor::new(graph.argmax_ptr as *mut u8, &[batch_size], DType::U32);

        Ok(ReplayOutput { logits, token_ids })
    }

    /// Fast replay for steady-state decode: update metadata on GPU instead of H2D.
    ///
    /// Instead of building positions/slot_mapping/cu_seqlens_k on CPU and doing
    /// H2D copies, this calls a single GPU kernel to increment them in-place.
    /// Only `block_table` is H2D-copied (and only when `new_block_table` is Some).
    ///
    /// Requirements: the persistent buffers must already contain valid state from
    /// a previous `replay()` call. `cu_seqlens_q` is constant for decode so it
    /// never needs updating after initial setup.
    ///
    /// # Safety
    /// Same requirements as `replay()`.
    pub unsafe fn replay_decode_fast(
        &self,
        batch_size: usize,
        input_ids: Option<&[u32]>,
        new_block_table: Option<&[u32]>,
        block_size: usize,
        device: &mut GpuDevice,
    ) -> Result<ReplayOutput> {
        let graph = self
            .graphs
            .get(&batch_size)
            .ok_or_else(|| anyhow::anyhow!("no captured graph for batch_size={batch_size}"))?;

        let stream = device.compute_stream;

        // H2D input_ids only if provided (skipped when in-graph argmax scattered them).
        if let Some(ids) = input_ids {
            driver::memcpy_htod_async(
                self.input_ids,
                ids.as_ptr() as *const u8,
                batch_size * 4,
                stream,
            )?;
        }

        // H2D block_table only if new blocks were allocated.
        if let Some(bt) = new_block_table {
            driver::memcpy_htod_async(
                self.block_table,
                bt.as_ptr() as *const u8,
                bt.len() * 4,
                stream,
            )?;
        }

        // Update positions, slot_mapping, cu_seqlens_k on GPU in one kernel.
        kernels::update_decode_metadata_gpu(
            self.positions,
            self.slot_mapping,
            self.cu_seqlens_k,
            self.block_table as *const u8,
            batch_size,
            block_size,
            MAX_BLOCKS_PER_SEQ,
            stream,
        );

        // Reset arena so intermediates land at the same offsets as capture.
        device.arena.reset();

        // Launch the captured graph (forward + argmax + D2D scatter).
        driver::graph_launch(graph.exec, stream)?;

        // Advance the arena offset past everything the graph wrote.
        device.arena.set_offset(graph.arena_used);

        let logits = GpuTensor::new(
            graph.output_ptr as *mut u8,
            &[batch_size, self.vocab_size],
            self.dtype,
        );
        let token_ids = GpuTensor::new(graph.argmax_ptr as *mut u8, &[batch_size], DType::U32);

        Ok(ReplayOutput { logits, token_ids })
    }

    /// Fill persistent buffers with dummy decode data for capture.
    unsafe fn fill_dummy_decode(&self, batch_size: usize, stream: CUstream) -> Result<()> {
        // input_ids: all zeros.
        driver::memset_d8(self.input_ids, 0, batch_size * 4, stream)?;
        // positions: [0, 1, 2, ...] — doesn't matter for capture, just needs valid values.
        let positions: Vec<u32> = (0..batch_size as u32).collect();
        driver::memcpy_htod_async(
            self.positions,
            positions.as_ptr() as *const u8,
            batch_size * 4,
            stream,
        )?;
        // slot_mapping: [0, 1, 2, ...].
        let slots: Vec<i64> = (0..batch_size as i64).collect();
        driver::memcpy_htod_async(
            self.slot_mapping,
            slots.as_ptr() as *const u8,
            batch_size * 8,
            stream,
        )?;
        // cu_seqlens_q: [0, 1, 2, ..., batch_size] (each request has q_len=1).
        let cu_q: Vec<u32> = (0..=batch_size as u32).collect();
        driver::memcpy_htod_async(
            self.cu_seqlens_q,
            cu_q.as_ptr() as *const u8,
            (batch_size + 1) * 4,
            stream,
        )?;
        // cu_seqlens_k: same as cu_seqlens_q for dummy (each seq has 1 token).
        driver::memcpy_htod_async(
            self.cu_seqlens_k,
            cu_q.as_ptr() as *const u8,
            (batch_size + 1) * 4,
            stream,
        )?;
        // block_table: all zeros (block 0 for all).
        driver::memset_d8(
            self.block_table,
            0,
            batch_size * MAX_BLOCKS_PER_SEQ * 4,
            stream,
        )?;

        Ok(())
    }

    /// Batch sizes that have been captured.
    pub fn captured_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = self.graphs.keys().copied().collect();
        sizes.sort();
        sizes
    }
}

/// Output from a graph replay: logits + argmax token IDs.
pub struct ReplayOutput {
    /// Logits tensor `[batch_size, vocab_size]` in arena memory.
    pub logits: GpuTensor,
    /// Argmax token IDs `[batch_size]` u32 in arena memory.
    /// These are also scattered into the persistent `input_ids` buffer.
    pub token_ids: GpuTensor,
}

impl Drop for CudaGraphRunner {
    fn drop(&mut self) {
        unsafe {
            for (_, g) in self.graphs.drain() {
                let _ = driver::graph_exec_destroy(g.exec);
            }
            let _ = driver::mem_free(self.input_ids);
            let _ = driver::mem_free(self.positions);
            let _ = driver::mem_free(self.slot_mapping);
            let _ = driver::mem_free(self.cu_seqlens_q);
            let _ = driver::mem_free(self.cu_seqlens_k);
            let _ = driver::mem_free(self.block_table);
        }
    }
}

/// Input tensor views into the graph runner's persistent buffers.
/// These are passed to the model forward function during capture.
#[derive(Clone, Copy)]
pub struct InputTensors {
    pub input_ids: GpuTensor,
    pub positions: GpuTensor,
    pub slot_mapping: GpuTensor,
    pub cu_seqlens_q: GpuTensor,
    pub cu_seqlens_k: GpuTensor,
    pub block_table: GpuTensor,
}

/// The maximum number of blocks per sequence used in graph capture.
pub const GRAPH_MAX_BLOCKS_PER_SEQ: usize = MAX_BLOCKS_PER_SEQ;
