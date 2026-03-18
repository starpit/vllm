// SPDX-License-Identifier: Apache-2.0
//! CUDA graph capture and replay for decode and prefill batches.
//!
//! During decode, every request contributes exactly 1 token (q_len=1), making
//! all tensor shapes deterministic for a given batch size. We capture the
//! entire model forward pass as a CUDA graph during warmup, then replay it
//! during inference — eliminating ~280 kernel launch overheads per step.
//!
//! **How it works (matching Python/PyTorch):**
//! 1. Persistent input buffers are allocated at max capacity.
//! 2. `capture()` enters capture mode on the caching allocator (frees suppressed),
//!    begins stream capture, runs the model forward, ends capture, and instantiates
//!    the graph. All blocks allocated during capture stay permanently allocated
//!    (like PyTorch's private graph pool).
//! 3. `replay()` copies real inputs into the persistent buffers and launches the graph.
//!    No allocator interaction needed — addresses are baked into the graph.

use std::collections::HashMap;

use anyhow::Result;
use cudarc::driver::sys::{CUgraphExec, CUstream};

use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::driver;
use crate::dtype::DType;
use crate::kernels;
use crate::tensor::GpuTensor;

/// Maximum number of blocks per sequence in the block table.
const MAX_BLOCKS_PER_SEQ: usize = 512;

/// A single captured CUDA graph for a specific batch size.
struct CapturedGraph {
    exec: CUgraphExec,
    #[allow(dead_code)]
    batch_size: usize,
}

/// CUDA graph runner for decode batches.
pub struct CudaGraphRunner {
    graphs: HashMap<usize, CapturedGraph>,
    input_ids: *mut u8,
    positions: *mut u8,
    slot_mapping: *mut u8,
    cu_seqlens_q: *mut u8,
    seqused_k: *mut u8,
    block_table: *mut u8,
    /// Shared output buffer for logits — `[max_batch, vocab_size]` in model dtype.
    /// All captured graphs copy their logits here, so we only keep one allocation.
    shared_logits: *mut u8,
    /// Shared output buffer for argmax token IDs — `[max_batch]` in U32.
    shared_argmax: *mut u8,
    max_batch: usize,
    dtype: DType,
    vocab_size: usize,

    // --- FP8 KV cache dequant buffers (None when KV cache is BF16/F16) ---
    /// `[max_total_kv, num_kv_heads, head_dim]` in model dtype — pre-allocated
    /// contiguous K buffer for FP8 dequant during graph capture/replay.
    fp8_k_buf: Option<*mut u8>,
    /// Same shape — pre-allocated contiguous V buffer.
    fp8_v_buf: Option<*mut u8>,
    /// `[max_batch + 1]` i32 — persistent cu_seqlens_k for FP8 dequant.
    fp8_cu_seqlens_k: Option<*mut u8>,
    /// Max total KV tokens = max_batch * padded_max_seqlen_k.
    fp8_max_total_kv: usize,
    fp8_num_kv_heads: usize,
    fp8_head_dim: usize,
    /// Per-layer cached K scales (host-side). Read once during warmup.
    fp8_k_scales: Vec<f32>,
    /// Per-layer cached V scales (host-side).
    fp8_v_scales: Vec<f32>,
}

unsafe impl Send for CudaGraphRunner {}

impl CudaGraphRunner {
    pub unsafe fn new(max_batch: usize, vocab_size: usize, dtype: DType) -> Result<Self> {
        let input_ids = driver::mem_alloc(max_batch * 4)?;
        let positions = driver::mem_alloc(max_batch * 4)?;
        let slot_mapping = driver::mem_alloc(max_batch * 8)?;
        let cu_seqlens_q = driver::mem_alloc((max_batch + 1) * 4)?;
        let seqused_k = driver::mem_alloc(max_batch * 4)?;
        let block_table = driver::mem_alloc(max_batch * MAX_BLOCKS_PER_SEQ * 4)?;
        let shared_logits = driver::mem_alloc(max_batch * vocab_size * dtype.size_bytes())?;
        let shared_argmax = driver::mem_alloc(max_batch * 4)?;

        Ok(Self {
            graphs: HashMap::new(),
            input_ids,
            positions,
            slot_mapping,
            cu_seqlens_q,
            seqused_k,
            block_table,
            shared_logits,
            shared_argmax,
            max_batch,
            dtype,
            vocab_size,
            fp8_k_buf: None,
            fp8_v_buf: None,
            fp8_cu_seqlens_k: None,
            fp8_max_total_kv: 0,
            fp8_num_kv_heads: 0,
            fp8_head_dim: 0,
            fp8_k_scales: Vec::new(),
            fp8_v_scales: Vec::new(),
        })
    }

    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
    }

    /// Allocate FP8 dequant buffers for CUDA graph capture/replay.
    ///
    /// Call once after construction when `kv_cache_is_fp8`. Allocates persistent
    /// K/V contiguous buffers and cu_seqlens_k at max capacity.
    ///
    /// # Safety
    /// CUDA context must be current.
    pub unsafe fn init_fp8_buffers(
        &mut self,
        padded_max_seqlen_k: usize,
        num_kv_heads: usize,
        head_dim: usize,
        model_dtype: DType,
    ) -> Result<()> {
        let max_total_kv = self.max_batch * padded_max_seqlen_k;
        let elem_bytes = model_dtype.size_bytes();
        let buf_bytes = max_total_kv * num_kv_heads * head_dim * elem_bytes;

        let k_buf = driver::mem_alloc(buf_bytes)?;
        let v_buf = driver::mem_alloc(buf_bytes)?;
        let cu_seqlens_k = driver::mem_alloc((self.max_batch + 1) * 4)?;

        self.fp8_k_buf = Some(k_buf);
        self.fp8_v_buf = Some(v_buf);
        self.fp8_cu_seqlens_k = Some(cu_seqlens_k);
        self.fp8_max_total_kv = max_total_kv;
        self.fp8_num_kv_heads = num_kv_heads;
        self.fp8_head_dim = head_dim;

        let buf_mb = (2 * buf_bytes) as f64 / (1024.0 * 1024.0);
        tracing::info!(
            "FP8 graph dequant buffers allocated: {buf_mb:.1} MB \
             (max_total_kv={max_total_kv}, heads={num_kv_heads}, dim={head_dim})"
        );

        Ok(())
    }

    /// Cache per-layer FP8 KV scales from GPU. Call after graph warmup.
    ///
    /// # Safety
    /// Scale pointers must be valid. CUDA context must be current.
    pub unsafe fn cache_fp8_scales(
        &mut self,
        kv_cache: &crate::kv_cache::KvCachePool,
        stream: CUstream,
    ) -> Result<()> {
        let num_layers = kv_cache.num_layers;
        self.fp8_k_scales.resize(num_layers, 1.0);
        self.fp8_v_scales.resize(num_layers, 1.0);

        for layer in 0..num_layers {
            driver::memcpy_dtoh_async(
                &mut self.fp8_k_scales[layer] as *mut f32 as *mut u8,
                kv_cache.k_scale_ptr(layer) as *const u8,
                4,
                stream,
            )?;
            driver::memcpy_dtoh_async(
                &mut self.fp8_v_scales[layer] as *mut f32 as *mut u8,
                kv_cache.v_scale_ptr(layer) as *const u8,
                4,
                stream,
            )?;
        }
        driver::stream_synchronize(stream)?;

        tracing::debug!(
            "Cached FP8 scales for {num_layers} layers: k[0]={}, v[0]={}",
            self.fp8_k_scales[0],
            self.fp8_v_scales[0],
        );
        Ok(())
    }

    /// Build an [`Fp8GraphCtx`] for the thread-local. Returns `None` if FP8
    /// buffers are not allocated.
    pub fn fp8_graph_ctx(&self) -> Option<crate::model::attention_helpers::Fp8GraphCtx> {
        let k_buf = self.fp8_k_buf?;
        let v_buf = self.fp8_v_buf?;
        let cu_seqlens_k = self.fp8_cu_seqlens_k?;
        Some(crate::model::attention_helpers::Fp8GraphCtx {
            k_buf,
            v_buf,
            cu_seqlens_k,
            max_total_kv: self.fp8_max_total_kv,
            num_kv_heads: self.fp8_num_kv_heads,
            head_dim: self.fp8_head_dim,
            output_dtype: self.dtype,
            k_scales: self.fp8_k_scales.as_ptr(),
            v_scales: self.fp8_v_scales.as_ptr(),
        })
    }

    /// Upload cu_seqlens_k to the persistent FP8 buffer. Call before replay.
    ///
    /// # Safety
    /// FP8 buffers must be initialized.
    pub unsafe fn upload_fp8_cu_seqlens_k(
        &self,
        cu_seqlens_k: &[i32],
        stream: CUstream,
    ) -> Result<()> {
        if let Some(ptr) = self.fp8_cu_seqlens_k {
            driver::memcpy_htod_async(
                ptr,
                cu_seqlens_k.as_ptr() as *const u8,
                cu_seqlens_k.len() * 4,
                stream,
            )?;
        }
        Ok(())
    }

    /// Whether FP8 dequant buffers are allocated.
    pub fn has_fp8_buffers(&self) -> bool {
        self.fp8_k_buf.is_some()
    }

    pub fn nearest_graph_size(&self, batch_size: usize) -> Option<usize> {
        self.graphs
            .keys()
            .filter(|&&s| s >= batch_size)
            .min()
            .copied()
    }

    fn input_tensors(&self, batch_size: usize) -> InputTensors {
        unsafe {
            InputTensors {
                input_ids: GpuTensor::new(self.input_ids, &[batch_size], DType::U32),
                positions: GpuTensor::new(self.positions, &[batch_size], DType::U32),
                slot_mapping: GpuTensor::new(self.slot_mapping, &[batch_size], DType::I64),
                cu_seqlens_q: GpuTensor::new(self.cu_seqlens_q, &[batch_size + 1], DType::I32),
                seqused_k: GpuTensor::new(self.seqused_k, &[batch_size], DType::I32),
                block_table: GpuTensor::new(
                    self.block_table,
                    &[batch_size, MAX_BLOCKS_PER_SEQ],
                    DType::I32,
                ),
            }
        }
    }

    /// Capture a CUDA graph for a decode batch.
    ///
    /// Uses the caching allocator in capture mode (frees suppressed) so all
    /// blocks allocated during capture stay permanently allocated — matching
    /// PyTorch's private graph pool mechanism.
    pub unsafe fn capture<F>(
        &mut self,
        batch_size: usize,
        device: &mut GpuDevice,
        mut forward_fn: F,
    ) -> Result<()>
    where
        F: FnMut(InputTensors, &mut GpuDevice) -> OwnedTensor,
    {
        assert!(batch_size <= self.max_batch);

        self.fill_dummy_decode(batch_size, device.compute_stream)?;

        // Warm up: populates cuBLAS plans and fills the pool.
        // The caller must have already called begin_allocate_to_pool().
        let warmup_logits = forward_fn(self.input_tensors(batch_size), device);
        let warmup_argmax = kernels::argmax_batched(
            warmup_logits.as_gpu_tensor(),
            &mut device.caching,
            device.compute_stream,
        );
        driver::stream_synchronize(device.compute_stream)?;
        drop(warmup_argmax);
        drop(warmup_logits);
        let inputs = self.input_tensors(batch_size);

        let logits_bytes = batch_size * self.vocab_size * self.dtype.size_bytes();
        let argmax_bytes = batch_size * 4;

        driver::stream_begin_capture(device.compute_stream)?;
        let logits = forward_fn(inputs, device);
        let argmax_out = kernels::argmax_batched(
            logits.as_gpu_tensor(),
            &mut device.caching,
            device.compute_stream,
        );
        // Copy logits and argmax into shared buffers (recorded in graph).
        driver::memcpy_dtod_async(
            self.shared_logits,
            logits.as_gpu_tensor().raw_ptr() as *const u8,
            logits_bytes,
            device.compute_stream,
        )?;
        driver::memcpy_dtod_async(
            self.shared_argmax,
            argmax_out.as_gpu_tensor().raw_ptr() as *const u8,
            argmax_bytes,
            device.compute_stream,
        )?;
        // Scatter sampled token IDs into persistent input_ids for the next step.
        driver::memcpy_dtod_async(
            self.input_ids,
            self.shared_argmax as *const u8,
            argmax_bytes,
            device.compute_stream,
        )?;
        let graph = driver::stream_end_capture(device.compute_stream)?;

        let exec = driver::graph_instantiate(graph)?;
        driver::graph_destroy(graph)?;
        drop(logits);
        drop(argmax_out);

        let captured = CapturedGraph { exec, batch_size };
        self.graphs.insert(batch_size, captured);

        tracing::info!(
            "CUDA graph captured for batch_size={} (with argmax), shared output ({} logit elements)",
            batch_size,
            batch_size * self.vocab_size,
        );

        Ok(())
    }

    /// Replay a captured CUDA graph.
    pub unsafe fn replay(
        &self,
        batch_size: usize,
        input_ids: &[u32],
        positions: &[u32],
        slot_mapping: &[i64],
        cu_seqlens_q: &[i32],
        seqused_k: &[i32],
        block_table: &[i32],
        device: &mut GpuDevice,
        skip_input_ids_h2d: bool,
    ) -> Result<ReplayOutput> {
        let graph = self
            .graphs
            .get(&batch_size)
            .ok_or_else(|| anyhow::anyhow!("no captured graph for batch_size={batch_size}"))?;

        let xfer = device.transfer_stream;

        if !skip_input_ids_h2d {
            driver::memcpy_htod_async(
                self.input_ids,
                input_ids.as_ptr() as *const u8,
                batch_size * 4,
                xfer,
            )?;
        }
        driver::memcpy_htod_async(
            self.positions,
            positions.as_ptr() as *const u8,
            batch_size * 4,
            xfer,
        )?;
        driver::memcpy_htod_async(
            self.slot_mapping,
            slot_mapping.as_ptr() as *const u8,
            batch_size * 8,
            xfer,
        )?;
        driver::memcpy_htod_async(
            self.cu_seqlens_q,
            cu_seqlens_q.as_ptr() as *const u8,
            (batch_size + 1) * 4,
            xfer,
        )?;
        driver::memcpy_htod_async(
            self.seqused_k,
            seqused_k.as_ptr() as *const u8,
            batch_size * 4,
            xfer,
        )?;
        driver::memcpy_htod_async(
            self.block_table,
            block_table.as_ptr() as *const u8,
            batch_size * MAX_BLOCKS_PER_SEQ * 4,
            xfer,
        )?;

        // FP8: compute cu_seqlens_k from seqused_k (CPU prefix sum) and upload.
        if let Some(cu_k_ptr) = self.fp8_cu_seqlens_k {
            let mut cu_seqlens_k_host = vec![0i32; batch_size + 1];
            for i in 0..batch_size {
                cu_seqlens_k_host[i + 1] = cu_seqlens_k_host[i] + seqused_k[i];
            }
            driver::memcpy_htod_async(
                cu_k_ptr,
                cu_seqlens_k_host.as_ptr() as *const u8,
                (batch_size + 1) * 4,
                xfer,
            )?;
        }

        device.sync_transfer_to_compute()?;

        // No allocator reset needed — graph uses baked-in addresses.
        driver::graph_launch(graph.exec, device.compute_stream)?;

        let logits = GpuTensor::new(
            self.shared_logits,
            &[batch_size, self.vocab_size],
            self.dtype,
        );
        let token_ids = GpuTensor::new(self.shared_argmax, &[batch_size], DType::U32);

        Ok(ReplayOutput { logits, token_ids })
    }

    /// Fast replay for steady-state decode: update metadata on GPU.
    pub unsafe fn replay_decode_fast(
        &self,
        batch_size: usize,
        input_ids: Option<&[u32]>,
        new_block_table: Option<&[i32]>,
        block_size: usize,
        device: &mut GpuDevice,
    ) -> Result<ReplayOutput> {
        let graph = self
            .graphs
            .get(&batch_size)
            .ok_or_else(|| anyhow::anyhow!("no captured graph for batch_size={batch_size}"))?;

        let stream = device.compute_stream;

        if let Some(ids) = input_ids {
            driver::memcpy_htod_async(
                self.input_ids,
                ids.as_ptr() as *const u8,
                batch_size * 4,
                device.transfer_stream,
            )?;
        }

        if let Some(bt) = new_block_table {
            driver::memcpy_htod_async(
                self.block_table,
                bt.as_ptr() as *const u8,
                bt.len() * 4,
                device.transfer_stream,
            )?;
        }

        if input_ids.is_some() || new_block_table.is_some() {
            device.sync_transfer_to_compute()?;
        }

        kernels::update_decode_metadata_gpu(
            self.positions,
            self.slot_mapping,
            self.seqused_k,
            self.block_table as *const u8,
            batch_size,
            block_size,
            MAX_BLOCKS_PER_SEQ,
            stream,
        );

        // FP8: compute cu_seqlens_k from seqused_k on GPU (prefix sum kernel).
        if let Some(cu_k_ptr) = self.fp8_cu_seqlens_k {
            kernels::compute_cu_seqlens_k_gpu(
                self.seqused_k as *const u8,
                cu_k_ptr,
                batch_size,
                stream,
            );
        }

        driver::graph_launch(graph.exec, stream)?;

        let logits = GpuTensor::new(
            self.shared_logits,
            &[batch_size, self.vocab_size],
            self.dtype,
        );
        let token_ids = GpuTensor::new(self.shared_argmax, &[batch_size], DType::U32);

        Ok(ReplayOutput { logits, token_ids })
    }

    unsafe fn fill_dummy_decode(&self, batch_size: usize, stream: CUstream) -> Result<()> {
        driver::memset_d8(self.input_ids, 0, batch_size * 4, stream)?;
        let positions: Vec<u32> = (0..batch_size as u32).collect();
        driver::memcpy_htod_async(
            self.positions,
            positions.as_ptr() as *const u8,
            batch_size * 4,
            stream,
        )?;
        let slots: Vec<i64> = (0..batch_size as i64).collect();
        driver::memcpy_htod_async(
            self.slot_mapping,
            slots.as_ptr() as *const u8,
            batch_size * 8,
            stream,
        )?;
        let cu_q: Vec<i32> = (0..=batch_size as i32).collect();
        driver::memcpy_htod_async(
            self.cu_seqlens_q,
            cu_q.as_ptr() as *const u8,
            (batch_size + 1) * 4,
            stream,
        )?;
        let seqused_k: Vec<i32> = vec![1; batch_size];
        driver::memcpy_htod_async(
            self.seqused_k,
            seqused_k.as_ptr() as *const u8,
            batch_size * 4,
            stream,
        )?;
        driver::memset_d8(
            self.block_table,
            0,
            batch_size * MAX_BLOCKS_PER_SEQ * 4,
            stream,
        )?;

        // Fill FP8 cu_seqlens_k with dummy prefix sums matching seqused_k=[1,1,...].
        if let Some(cu_k_ptr) = self.fp8_cu_seqlens_k {
            let cu_k: Vec<i32> = (0..=batch_size as i32).collect();
            driver::memcpy_htod_async(
                cu_k_ptr,
                cu_k.as_ptr() as *const u8,
                (batch_size + 1) * 4,
                stream,
            )?;
        }

        Ok(())
    }

    pub fn captured_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = self.graphs.keys().copied().collect();
        sizes.sort();
        sizes
    }
}

/// Output from a graph replay.
pub struct ReplayOutput {
    pub logits: GpuTensor,
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
            let _ = driver::mem_free(self.seqused_k);
            let _ = driver::mem_free(self.block_table);
            let _ = driver::mem_free(self.shared_logits);
            let _ = driver::mem_free(self.shared_argmax);
            if let Some(p) = self.fp8_k_buf {
                let _ = driver::mem_free(p);
            }
            if let Some(p) = self.fp8_v_buf {
                let _ = driver::mem_free(p);
            }
            if let Some(p) = self.fp8_cu_seqlens_k {
                let _ = driver::mem_free(p);
            }
        }
    }
}

/// Input tensor views into the graph runner's persistent buffers.
#[derive(Clone, Copy)]
pub struct InputTensors {
    pub input_ids: GpuTensor,
    pub positions: GpuTensor,
    pub slot_mapping: GpuTensor,
    pub cu_seqlens_q: GpuTensor,
    pub seqused_k: GpuTensor,
    pub block_table: GpuTensor,
}

/// The maximum number of blocks per sequence used in graph capture.
pub const GRAPH_MAX_BLOCKS_PER_SEQ: usize = MAX_BLOCKS_PER_SEQ;

// ---------------------------------------------------------------------------
// Prefill graph runner
// ---------------------------------------------------------------------------

struct CapturedPrefillGraph {
    exec: CUgraphExec,
    #[allow(dead_code)]
    num_tokens: usize,
}

pub struct PrefillGraphRunner {
    graphs: HashMap<usize, CapturedPrefillGraph>,
    input_ids: *mut u8,
    positions: *mut u8,
    slot_mapping: *mut u8,
    cu_seqlens_q: *mut u8,
    seqused_k: *mut u8,
    block_table: *mut u8,
    last_token_indices: *mut u8,
    /// Shared output buffer for logits — `[1, vocab_size]` in model dtype (prefill = 1 output).
    shared_logits: *mut u8,
    /// Shared output buffer for argmax — `[1]` in U32.
    shared_argmax: *mut u8,
    max_tokens: usize,
    dtype: DType,
    vocab_size: usize,
}

unsafe impl Send for PrefillGraphRunner {}

#[derive(Clone, Copy)]
pub struct PrefillInputTensors {
    pub input_ids: GpuTensor,
    pub positions: GpuTensor,
    pub slot_mapping: GpuTensor,
    pub cu_seqlens_q: GpuTensor,
    pub seqused_k: GpuTensor,
    pub block_table: GpuTensor,
    pub last_token_indices: GpuTensor,
}

impl PrefillGraphRunner {
    pub unsafe fn new(max_tokens: usize, vocab_size: usize, dtype: DType) -> Result<Self> {
        let input_ids = driver::mem_alloc(max_tokens * 4)?;
        let positions = driver::mem_alloc(max_tokens * 4)?;
        let slot_mapping = driver::mem_alloc(max_tokens * 8)?;
        let cu_seqlens_q = driver::mem_alloc(2 * 4)?;
        let seqused_k = driver::mem_alloc(4)?;
        let block_table = driver::mem_alloc(MAX_BLOCKS_PER_SEQ * 4)?;
        let last_token_indices = driver::mem_alloc(4)?;
        // Prefill extracts 1 token's logits, so shared buffer is [1, vocab_size].
        let shared_logits = driver::mem_alloc(vocab_size * dtype.size_bytes())?;
        let shared_argmax = driver::mem_alloc(4)?;

        Ok(Self {
            graphs: HashMap::new(),
            input_ids,
            positions,
            slot_mapping,
            cu_seqlens_q,
            seqused_k,
            block_table,
            last_token_indices,
            shared_logits,
            shared_argmax,
            max_tokens,
            dtype,
            vocab_size,
        })
    }

    pub fn nearest_graph_size(&self, num_tokens: usize) -> Option<usize> {
        self.graphs
            .keys()
            .filter(|&&s| s >= num_tokens)
            .min()
            .copied()
    }

    fn input_tensors(&self, num_tokens: usize) -> PrefillInputTensors {
        unsafe {
            PrefillInputTensors {
                input_ids: GpuTensor::new(self.input_ids, &[num_tokens], DType::U32),
                positions: GpuTensor::new(self.positions, &[num_tokens], DType::U32),
                slot_mapping: GpuTensor::new(self.slot_mapping, &[num_tokens], DType::I64),
                cu_seqlens_q: GpuTensor::new(self.cu_seqlens_q, &[2], DType::I32),
                seqused_k: GpuTensor::new(self.seqused_k, &[1], DType::I32),
                block_table: GpuTensor::new(self.block_table, &[1, MAX_BLOCKS_PER_SEQ], DType::I32),
                last_token_indices: GpuTensor::new(self.last_token_indices, &[1], DType::U32),
            }
        }
    }

    pub unsafe fn capture<F>(
        &mut self,
        num_tokens: usize,
        device: &mut GpuDevice,
        mut forward_fn: F,
    ) -> Result<()>
    where
        F: FnMut(PrefillInputTensors, &mut GpuDevice) -> OwnedTensor,
    {
        assert!(num_tokens <= self.max_tokens);

        self.fill_dummy_prefill(num_tokens, device.compute_stream)?;

        // Warm up: fills the pool. The caller manages begin/end_allocate_to_pool.
        let inputs = self.input_tensors(num_tokens);
        let warmup_logits = forward_fn(inputs, device);
        let warmup_argmax = kernels::argmax_batched(
            warmup_logits.as_gpu_tensor(),
            &mut device.caching,
            device.compute_stream,
        );
        driver::stream_synchronize(device.compute_stream)?;
        drop(warmup_argmax);
        drop(warmup_logits);
        let inputs = self.input_tensors(num_tokens);

        let logits_bytes = self.vocab_size * self.dtype.size_bytes(); // [1, vocab]
        let argmax_bytes = 4; // [1] u32

        driver::stream_begin_capture(device.compute_stream)?;
        let logits = forward_fn(inputs, device);
        let argmax_out = kernels::argmax_batched(
            logits.as_gpu_tensor(),
            &mut device.caching,
            device.compute_stream,
        );
        // Copy into shared buffers (recorded in graph).
        driver::memcpy_dtod_async(
            self.shared_logits,
            logits.as_gpu_tensor().raw_ptr() as *const u8,
            logits_bytes,
            device.compute_stream,
        )?;
        driver::memcpy_dtod_async(
            self.shared_argmax,
            argmax_out.as_gpu_tensor().raw_ptr() as *const u8,
            argmax_bytes,
            device.compute_stream,
        )?;
        let graph = driver::stream_end_capture(device.compute_stream)?;

        let exec = driver::graph_instantiate(graph)?;
        driver::graph_destroy(graph)?;
        drop(logits);
        drop(argmax_out);

        self.graphs
            .insert(num_tokens, CapturedPrefillGraph { exec, num_tokens });

        tracing::info!(
            "Prefill CUDA graph captured for num_tokens={}, shared output ({} logit elements)",
            num_tokens,
            self.vocab_size,
        );

        Ok(())
    }

    pub unsafe fn replay(
        &self,
        padded_tokens: usize,
        input_ids: &[u32],
        positions: &[u32],
        slot_mapping: &[i64],
        seq_len: usize,
        block_table: &[i32],
        last_token_idx: u32,
        device: &mut GpuDevice,
    ) -> Result<PrefillReplayOutput> {
        let graph = self
            .graphs
            .get(&padded_tokens)
            .ok_or_else(|| anyhow::anyhow!("no prefill graph for num_tokens={padded_tokens}"))?;

        let xfer = device.transfer_stream;
        let num_real = input_ids.len();

        if num_real < padded_tokens {
            driver::memset_d8(self.input_ids, 0, padded_tokens * 4, xfer)?;
        }
        driver::memcpy_htod_async(
            self.input_ids,
            input_ids.as_ptr() as *const u8,
            num_real * 4,
            xfer,
        )?;

        if num_real < padded_tokens {
            driver::memset_d8(self.positions, 0, padded_tokens * 4, xfer)?;
        }
        driver::memcpy_htod_async(
            self.positions,
            positions.as_ptr() as *const u8,
            num_real * 4,
            xfer,
        )?;

        let mut padded_slots = vec![-1i64; padded_tokens];
        padded_slots[..num_real].copy_from_slice(slot_mapping);
        driver::memcpy_htod_async(
            self.slot_mapping,
            padded_slots.as_ptr() as *const u8,
            padded_tokens * 8,
            xfer,
        )?;

        let cu_q: [i32; 2] = [0, padded_tokens as i32];
        let seqused_k_val: [i32; 1] = [seq_len as i32];
        driver::memcpy_htod_async(self.cu_seqlens_q, cu_q.as_ptr() as *const u8, 8, xfer)?;
        driver::memcpy_htod_async(self.seqused_k, seqused_k_val.as_ptr() as *const u8, 4, xfer)?;

        driver::memcpy_htod_async(
            self.block_table,
            block_table.as_ptr() as *const u8,
            block_table.len().min(MAX_BLOCKS_PER_SEQ) * 4,
            xfer,
        )?;

        driver::memcpy_htod_async(
            self.last_token_indices,
            &last_token_idx as *const u32 as *const u8,
            4,
            xfer,
        )?;

        device.sync_transfer_to_compute()?;
        driver::graph_launch(graph.exec, device.compute_stream)?;

        let logits = GpuTensor::new(self.shared_logits, &[1, self.vocab_size], self.dtype);
        let token_ids = GpuTensor::new(self.shared_argmax, &[1], DType::U32);

        Ok(PrefillReplayOutput { logits, token_ids })
    }

    unsafe fn fill_dummy_prefill(&self, num_tokens: usize, stream: CUstream) -> Result<()> {
        driver::memset_d8(self.input_ids, 0, num_tokens * 4, stream)?;
        let positions: Vec<u32> = (0..num_tokens as u32).collect();
        driver::memcpy_htod_async(
            self.positions,
            positions.as_ptr() as *const u8,
            num_tokens * 4,
            stream,
        )?;
        let slots: Vec<i64> = (0..num_tokens as i64).collect();
        driver::memcpy_htod_async(
            self.slot_mapping,
            slots.as_ptr() as *const u8,
            num_tokens * 8,
            stream,
        )?;
        let cu_q: [i32; 2] = [0, num_tokens as i32];
        let seqused_k: [i32; 1] = [num_tokens as i32];
        driver::memcpy_htod_async(self.cu_seqlens_q, cu_q.as_ptr() as *const u8, 8, stream)?;
        driver::memcpy_htod_async(self.seqused_k, seqused_k.as_ptr() as *const u8, 4, stream)?;
        driver::memset_d8(self.block_table, 0, MAX_BLOCKS_PER_SEQ * 4, stream)?;
        let last_idx = (num_tokens - 1) as u32;
        driver::memcpy_htod_async(
            self.last_token_indices,
            &last_idx as *const u32 as *const u8,
            4,
            stream,
        )?;
        Ok(())
    }

    pub fn captured_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = self.graphs.keys().copied().collect();
        sizes.sort();
        sizes
    }
}

pub struct PrefillReplayOutput {
    pub logits: GpuTensor,
    pub token_ids: GpuTensor,
}

impl Drop for PrefillGraphRunner {
    fn drop(&mut self) {
        unsafe {
            for (_, g) in self.graphs.drain() {
                let _ = driver::graph_exec_destroy(g.exec);
            }
            let _ = driver::mem_free(self.input_ids);
            let _ = driver::mem_free(self.positions);
            let _ = driver::mem_free(self.slot_mapping);
            let _ = driver::mem_free(self.cu_seqlens_q);
            let _ = driver::mem_free(self.seqused_k);
            let _ = driver::mem_free(self.block_table);
            let _ = driver::mem_free(self.last_token_indices);
            let _ = driver::mem_free(self.shared_logits);
            let _ = driver::mem_free(self.shared_argmax);
        }
    }
}
