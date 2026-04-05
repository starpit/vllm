// SPDX-License-Identifier: Apache-2.0
//! `TkWorker`: a `Worker` implementation using the TK KVM megakernel.
//!
//! Fuses the entire LLaMA forward pass into a single persistent kernel launch.
//! Supports paged KV cache with CSR-indexed decode attention.

#[cfg(feature = "cuda")]
mod inner {
    use std::collections::HashMap;

    use anyhow::Result;
    use cudarc::driver::sys::CUstream;
    use vllm_cuda::alloc::CachingAllocator;
    use vllm_cuda::device::GpuDevice;
    use vllm_cuda::driver;
    use vllm_cuda::dtype::DType;
    use vllm_cuda::kernels;
    use vllm_cuda::model::llama::LlamaConfig;
    use vllm_cuda::tensor::GpuTensor;
    use vllm_cuda::{OwnedTensor, RawGpuMem};

    use super::build_csr_metadata;
    use crate::scheduler::{self, DecodeInstructions, PrefillSeq, TkModelConfig};
    use crate::weights::TkWeights;
    use vllm_tk_static::*;

    // TK kernel constants (must match llama_sm89.cuh)
    const KV_PAGE_SIZE: usize = 64;
    #[allow(dead_code)]
    const KV_BLOCK_SIZE: usize = 16;

    /// Configuration for a TkWorker.
    #[derive(Debug, Clone)]
    pub struct TkWorkerConfig {
        pub model_path: String,
        pub dtype: String,
        pub hf_token: Option<String>,
        pub device_id: i32,
        pub max_num_batched_tokens: usize,
        pub gpu_memory_utilization: f64,
        pub sm_count: usize,
    }

    /// Activation buffers for the TK megakernel.
    ///
    /// Pre-allocated at max batch size; the megakernel reads/writes these in-place.
    struct TkActivationBuffers {
        hidden_states: OwnedTensor,
        rms_rope: OwnedTensor,
        rms_gate: OwnedTensor,
        q_post_rope: OwnedTensor,
        attn_out: OwnedTensor,
        silu_out: OwnedTensor,
        rms_lm: OwnedTensor,
        logits: OwnedTensor,
    }

    impl TkActivationBuffers {
        fn alloc(max_bs: usize, config: &LlamaConfig, alloc: &mut CachingAllocator) -> Self {
            let hd = config.hidden_size;
            let id = config.intermediate_size;
            let vs = config.vocab_size;
            let dtype = DType::BF16;

            Self {
                hidden_states: alloc.alloc_tensor(&[max_bs, hd], dtype),
                rms_rope: alloc.alloc_tensor(&[max_bs, hd], dtype),
                rms_gate: alloc.alloc_tensor(&[max_bs, hd], dtype),
                q_post_rope: alloc.alloc_tensor(&[max_bs, hd], dtype),
                attn_out: alloc.alloc_tensor(&[max_bs, hd], dtype),
                silu_out: alloc.alloc_tensor(&[max_bs, id], dtype),
                rms_lm: alloc.alloc_tensor(&[max_bs, hd], dtype),
                logits: alloc.alloc_tensor(&[max_bs, vs], dtype),
            }
        }
    }

    /// GPU-side paged KV metadata buffers (CSR format).
    struct TkKvMetadata {
        /// position_ids: [max_bs] i32
        position_ids: OwnedTensor,
        /// decode_kv_indptr: [max_bs + 1] i32
        kv_indptr: OwnedTensor,
        /// decode_kv_indices: [max_pages] i32
        kv_indices: OwnedTensor,
        /// decode_kv_last_page_len: [max_bs] i32
        kv_last_page: OwnedTensor,
        /// kv_append_indices: [max_bs] i32
        kv_append: OwnedTensor,
        // Prefill metadata (CSR format, same structure as decode).
        prefill_qo_indptr: OwnedTensor,
        prefill_kv_indptr: OwnedTensor,
        prefill_kv_indices: OwnedTensor,
        prefill_kv_last_page_len: OwnedTensor,
    }

    impl TkKvMetadata {
        fn alloc(max_bs: usize, max_pages: usize, alloc: &mut CachingAllocator) -> Self {
            let dt = DType::I32;
            Self {
                position_ids: alloc.alloc_tensor(&[max_bs], dt),
                kv_indptr: alloc.alloc_tensor(&[max_bs + 1], dt),
                kv_indices: alloc.alloc_tensor(&[max_pages], dt),
                kv_last_page: alloc.alloc_tensor(&[max_bs], dt),
                kv_append: alloc.alloc_tensor(&[max_bs], dt),
                // Prefill: max_bs sequences, max_pages pages
                prefill_qo_indptr: alloc.alloc_tensor(&[max_bs + 1], dt),
                prefill_kv_indptr: alloc.alloc_tensor(&[max_bs + 1], dt),
                prefill_kv_indices: alloc.alloc_tensor(&[max_pages], dt),
                prefill_kv_last_page_len: alloc.alloc_tensor(&[max_bs], dt),
            }
        }
    }

    /// TK KV cache: single flat tensors (not per-layer).
    ///
    /// Layout: `[num_layers * num_pages, kv_page_size, num_kv_heads, head_dim]`
    pub struct TkKvCache {
        k_cache: GpuTensor,
        v_cache: GpuTensor,
        _k_mem: RawGpuMem,
        _v_mem: RawGpuMem,
        pub num_blocks: usize,
        pub num_layers: usize,
    }

    impl TkKvCache {
        /// Allocate KV cache on GPU.
        unsafe fn alloc(
            num_blocks: usize,
            num_layers: usize,
            num_kv_heads: usize,
            head_dim: usize,
            stream: CUstream,
        ) -> Result<Self> {
            // Depth = kv_page_size (token positions per page).
            // The kernel's warp::load_async<1, false> uses axis=1, so tile rows
            // map to the depth dimension. Each tile row = one token position.
            // st_bf<kv_block_size, head_dim> loads 16 consecutive depth positions.
            let b = num_layers * num_blocks;
            let d = KV_PAGE_SIZE;
            let dtype = DType::BF16;
            let total_bytes = b * d * num_kv_heads * head_dim * dtype.size_bytes();

            let k_ptr = unsafe { driver::mem_alloc(total_bytes)? };
            unsafe { driver::memset_d8(k_ptr, 0, total_bytes, stream)? };
            let k_cache = unsafe { GpuTensor::new(k_ptr, &[b, d, num_kv_heads, head_dim], dtype) };
            let k_mem = unsafe { RawGpuMem::new(k_ptr, total_bytes) };

            let v_ptr = unsafe { driver::mem_alloc(total_bytes)? };
            unsafe { driver::memset_d8(v_ptr, 0, total_bytes, stream)? };
            let v_cache = unsafe { GpuTensor::new(v_ptr, &[b, d, num_kv_heads, head_dim], dtype) };
            let v_mem = unsafe { RawGpuMem::new(v_ptr, total_bytes) };

            Ok(Self {
                k_cache,
                v_cache,
                _k_mem: k_mem,
                _v_mem: v_mem,
                num_blocks,
                num_layers,
            })
        }

        /// Byte size of the entire KV cache (both K and V).
        fn total_bytes(&self) -> usize {
            self._k_mem.size() + self._v_mem.size()
        }
    }

    /// The TK megakernel worker.
    pub struct TkWorker {
        config: TkWorkerConfig,
        llama_config: Option<LlamaConfig>,
        tk_model_config: Option<TkModelConfig>,
        device: Option<GpuDevice>,
        weights: Option<TkWeights>,
        kv_cache: Option<TkKvCache>,
        alloc: Option<CachingAllocator>,
        activations: Option<TkActivationBuffers>,
        kv_meta: Option<TkKvMetadata>,

        /// Pre-built instruction GPU tensors keyed by batch size.
        instruction_cache: HashMap<usize, (OwnedTensor, OwnedTensor, DecodeInstructions)>,

        /// Barrier tensor (zeroed before each launch).
        barrier: Option<OwnedTensor>,

        /// Pinned host staging buffers for async H2D of per-step metadata.
        host_position_ids: Vec<i32>,
        host_kv_indptr: Vec<i32>,
        host_kv_indices: Vec<i32>,
        host_kv_last_page: Vec<i32>,
        host_kv_append: Vec<i32>,

        // Host staging for prefill metadata.
        host_prefill_qo_indptr: Vec<i32>,
        host_prefill_kv_indptr: Vec<i32>,
        host_prefill_kv_indices: Vec<i32>,
        host_prefill_kv_last_page_len: Vec<i32>,

        ctx_set_on_thread: bool,

        /// Compiled kernel variant selected at model load time.
        kernel_variant: Option<vllm_tk_static::KernelVariant>,
    }

    impl TkWorker {
        pub fn new(config: TkWorkerConfig) -> Self {
            Self {
                config,
                llama_config: None,
                tk_model_config: None,
                device: None,
                weights: None,
                kv_cache: None,
                alloc: None,
                activations: None,
                kv_meta: None,
                instruction_cache: HashMap::new(),
                barrier: None,
                host_position_ids: Vec::new(),
                host_kv_indptr: Vec::new(),
                host_kv_indices: Vec::new(),
                host_kv_last_page: Vec::new(),
                host_kv_append: Vec::new(),
                host_prefill_qo_indptr: Vec::new(),
                host_prefill_kv_indptr: Vec::new(),
                host_prefill_kv_indices: Vec::new(),
                host_prefill_kv_last_page_len: Vec::new(),
                ctx_set_on_thread: false,
                kernel_variant: None,
            }
        }

        /// Ensure instructions for `batch_size` are built and cached.
        /// Returns nothing — caller accesses via `self.instruction_cache`.
        fn ensure_instructions(&mut self, batch_size: usize) {
            if self.instruction_cache.contains_key(&batch_size) {
                return;
            }
            let tk_cfg = self.tk_model_config.as_ref().unwrap();
            let decode = scheduler::build_decode_instructions(tk_cfg, batch_size);
            let alloc = self.alloc.as_mut().unwrap();
            let stream = self.device.as_ref().unwrap().compute_stream;

            let sm = decode.sm_count;
            let mps = decode.max_per_sm;
            let iw = scheduler::INSTRUCTION_WIDTH;
            let inst_tensor = alloc.alloc_tensor(&[sm, mps, iw], DType::I32);
            let inst_bytes = decode.instructions.len() * 4;
            unsafe {
                driver::memcpy_htod_async(
                    inst_tensor.as_mut_ptr::<u8>(),
                    decode.instructions.as_ptr() as *const u8,
                    inst_bytes,
                    stream,
                )
                .expect("upload instructions");
            }

            let tw = scheduler::TIMING_WIDTH;
            let timing_tensor = alloc.alloc_tensor(&[sm, mps, tw], DType::I32);
            // Zero timings.
            unsafe {
                driver::memset_d8(
                    timing_tensor.as_ptr::<u8>() as *mut u8,
                    0,
                    sm * mps * tw * 4,
                    stream,
                )
                .expect("zero timings");
            }

            self.instruction_cache
                .insert(batch_size, (inst_tensor, timing_tensor, decode));
        }

        /// Ensure barrier tensor is large enough, then zero it.
        fn ensure_and_zero_barrier(&mut self, n_batch_blocks: usize, max_barrier_cols: usize) {
            let alloc = self.alloc.as_mut().unwrap();
            let stream = self.device.as_ref().unwrap().compute_stream;
            let nl = self.llama_config.as_ref().unwrap().num_hidden_layers;
            let size = nl * scheduler::NUM_OPS * n_batch_blocks * max_barrier_cols;

            if self.barrier.is_none() || self.barrier.as_ref().unwrap().numel() < size {
                self.barrier = Some(alloc.alloc_tensor(&[size], DType::U32));
            }
            let bar = self.barrier.as_ref().unwrap();
            unsafe {
                driver::memset_d8(bar.as_ptr::<u8>() as *mut u8, 0, size * 4, stream)
                    .expect("zero barrier");
            }
        }

        /// Pre-pad barrier values for incomplete last batch block in prefill.
        ///
        /// Matmul ops (QKV, GateSiLU, LmHead) wait for per-token op barriers
        /// to reach `mbb` (matmul_batch_block_size). When total_tokens < mbb,
        /// the last batch block is incomplete. We pre-fill the barrier with the
        /// deficit so the actual token signals push it to the threshold.
        fn pad_prefill_barriers(
            &mut self,
            total_tokens: usize,
            n_batch_blocks: usize,
            max_barrier_cols: usize,
        ) {
            let config = self.llama_config.as_ref().unwrap();
            let mbb = self
                .tk_model_config
                .as_ref()
                .unwrap()
                .matmul_batch_block_size;
            let nl = config.num_hidden_layers;
            let nkh = config.num_kv_heads;
            let remainder = total_tokens % mbb;
            if remainder == 0 {
                return; // all batch blocks are full
            }
            let deficit = mbb - remainder;
            let last_bb = n_batch_blocks - 1;

            // Build host barrier image (all zeros except padding entries).
            let size = nl * scheduler::NUM_OPS * n_batch_blocks * max_barrier_cols;
            let mut host_bar = vec![0u32; size];

            let idx = |layer: usize, op_slot: usize, bb: usize, col: usize| -> usize {
                ((layer * scheduler::NUM_OPS + op_slot) * n_batch_blocks + bb) * max_barrier_cols
                    + col
            };

            for layer in 0..nl {
                // AttnNorm barrier (slot = AttnNorm - 1 = 0): QKV waits for >= mbb
                host_bar[idx(layer, scheduler::OP_ATTN_NORM as usize - 1, last_bb, 0)] =
                    deficit as u32;

                // MlpNorm barrier (slot = MlpNorm - 1): GateSiLU waits for >= mbb
                host_bar[idx(layer, scheduler::OP_MLP_NORM as usize - 1, last_bb, 0)] =
                    deficit as u32;

                // AttentionDecode barrier (slot = AttentionDecode - 1): O_proj waits for >= mbb * nkh
                host_bar[idx(
                    layer,
                    scheduler::OP_GQA_ATTENTION_DECODE as usize - 1,
                    last_bb,
                    0,
                )] = (deficit * nkh) as u32;
            }

            // LmHeadNorm barrier (slot = LmHeadNorm - 1, layer 0): LmHead waits for >= mbb
            host_bar[idx(0, scheduler::OP_LM_HEAD_NORM as usize - 1, last_bb, 0)] = deficit as u32;

            // Upload over the zeroed barrier.
            // NOTE: must sync before host_bar is dropped, since memcpy_htod_async
            // reads from the host buffer asynchronously.
            let bar = self.barrier.as_ref().unwrap();
            let stream = self.device.as_ref().unwrap().compute_stream;
            unsafe {
                driver::memcpy_htod_async(
                    bar.as_ptr::<u8>() as *mut u8,
                    host_bar.as_ptr() as *const u8,
                    size * 4,
                    stream,
                )
                .expect("upload padded barrier");
                driver::stream_synchronize(stream).expect("sync padded barrier");
            }
        }

        /// Fill host staging buffers with paged KV metadata, then upload to GPU.
        pub unsafe fn upload_kv_metadata(
            &mut self,
            block_tables: &[&[usize]],
            positions: &[u32],
            seq_lens: &[usize],
            batch_size: usize,
        ) {
            let stream = self.device.as_ref().unwrap().compute_stream;
            let kv_meta = self.kv_meta.as_ref().unwrap();

            // Pad metadata to the full matmul batch block size (128), not just
            // ATTN_BATCH_BLOCK_SIZE. The QKV storer writes to KV cache for all
            // 128 rows in a batch block, reading position_ids and kv_append_indices
            // for each. Padding with zeros means padding tokens write to page 0
            // offset 0 (benign, won't corrupt real data if page 0 isn't in use).
            let mbb = self
                .tk_model_config
                .as_ref()
                .unwrap()
                .matmul_batch_block_size;
            let attn_bb = self.tk_model_config.as_ref().unwrap().attn_batch_block_size;
            let padded_bs = batch_size.next_multiple_of(mbb.max(attn_bb));

            // Position IDs.
            self.host_position_ids.clear();
            self.host_position_ids
                .extend(positions.iter().map(|&p| p as i32));
            self.host_position_ids.resize(padded_bs, 0);

            // Build CSR metadata on CPU.
            build_csr_metadata(
                block_tables,
                positions,
                seq_lens,
                &mut self.host_kv_indptr,
                &mut self.host_kv_indices,
                &mut self.host_kv_last_page,
                &mut self.host_kv_append,
            );

            // Pad CSR arrays so attention_decode reads 0-page entries for padding items.
            let last_indptr = *self.host_kv_indptr.last().unwrap_or(&0);
            self.host_kv_indptr.resize(padded_bs + 1, last_indptr);
            self.host_kv_last_page.resize(padded_bs, 0);
            self.host_kv_append.resize(padded_bs, 0);

            let total_pages = self.host_kv_indices.len();
            unsafe {
                driver::memcpy_htod_async(
                    kv_meta.position_ids.as_ptr::<u8>() as *mut u8,
                    self.host_position_ids.as_ptr() as *const u8,
                    padded_bs * 4,
                    stream,
                )
                .expect("upload position_ids");
                driver::memcpy_htod_async(
                    kv_meta.kv_indptr.as_ptr::<u8>() as *mut u8,
                    self.host_kv_indptr.as_ptr() as *const u8,
                    (padded_bs + 1) * 4,
                    stream,
                )
                .expect("upload kv_indptr");
                driver::memcpy_htod_async(
                    kv_meta.kv_indices.as_ptr::<u8>() as *mut u8,
                    self.host_kv_indices.as_ptr() as *const u8,
                    total_pages * 4,
                    stream,
                )
                .expect("upload kv_indices");
                driver::memcpy_htod_async(
                    kv_meta.kv_last_page.as_ptr::<u8>() as *mut u8,
                    self.host_kv_last_page.as_ptr() as *const u8,
                    padded_bs * 4,
                    stream,
                )
                .expect("upload kv_last_page");
                driver::memcpy_htod_async(
                    kv_meta.kv_append.as_ptr::<u8>() as *mut u8,
                    self.host_kv_append.as_ptr() as *const u8,
                    padded_bs * 4,
                    stream,
                )
                .expect("upload kv_append");
            }
        }

        /// Upload per-token KV metadata for prefill.
        ///
        /// Expands per-sequence block tables into per-token CSR entries so that
        /// QKV_RopeAppend can index by token_idx rather than seq_idx.
        pub unsafe fn upload_prefill_kv_metadata(
            &mut self,
            prefill_seqs: &[PrefillSeq],
            block_tables: &[&[usize]],
            positions: &[u32],
            seq_lens: &[usize],
            total_tokens: usize,
        ) {
            let stream = self.device.as_ref().unwrap().compute_stream;
            let kv_meta = self.kv_meta.as_ref().unwrap();

            // Position IDs — already per-token, just copy.
            self.host_position_ids.clear();
            self.host_position_ids
                .extend(positions.iter().map(|&p| p as i32));

            // Build per-token CSR: each token gets its own indptr entry pointing
            // to the SAME page list as its parent sequence, plus its own
            // last_page_len and append_index.
            self.host_kv_indptr.clear();
            self.host_kv_indices.clear();
            self.host_kv_last_page.clear();
            self.host_kv_append.clear();

            let mut page_offset = 0i32;
            self.host_kv_indptr.push(0);

            let mut token_idx = 0usize;
            for (seq_i, seq) in prefill_seqs.iter().enumerate() {
                let blocks = block_tables[seq_i];
                let sl = seq_lens[seq_i];
                // Number of pages needed for the full sequence.
                let num_pages = sl.div_ceil(KV_PAGE_SIZE);

                // last_page_len is the same for all tokens in this sequence.
                let lpl = {
                    let rem = sl % KV_PAGE_SIZE;
                    if rem == 0 { KV_PAGE_SIZE } else { rem }
                };

                for _t in 0..seq.chunk_len {
                    // Each token points to the same page list.
                    for &b in &blocks[..num_pages] {
                        self.host_kv_indices.push(b as i32);
                    }
                    page_offset += num_pages as i32;
                    self.host_kv_indptr.push(page_offset);

                    self.host_kv_last_page.push(lpl as i32);

                    // Append index: where this token's KV goes in the flat cache.
                    let pos = positions[token_idx] as usize;
                    let page_id = blocks[pos / KV_PAGE_SIZE];
                    let offset_in_page = pos % KV_PAGE_SIZE;
                    self.host_kv_append
                        .push((page_id * KV_PAGE_SIZE + offset_in_page) as i32);

                    token_idx += 1;
                }
            }

            // Pad to matmul_batch_block_size (128) for QKV kernel safety.
            // QKV processes 16 rows per warp. Out-of-bounds tokens need valid
            // kv_append_indices to avoid illegal memory access. Pad with slot 0.
            let mbb = 128; // matmul_batch_block_size
            let padded_tokens = total_tokens.next_multiple_of(mbb);
            while self.host_kv_append.len() < padded_tokens {
                self.host_kv_append.push(0); // safe: writes to cache slot 0
            }
            while self.host_kv_last_page.len() < padded_tokens {
                self.host_kv_last_page.push(KV_PAGE_SIZE as i32);
            }
            while self.host_position_ids.len() < padded_tokens {
                self.host_position_ids.push(0);
            }
            // indptr: pad remaining entries pointing to the same page set end
            while self.host_kv_indptr.len() < padded_tokens + 1 {
                let last = *self.host_kv_indptr.last().unwrap();
                self.host_kv_indptr.push(last);
            }

            let total_kv_indices = self.host_kv_indices.len();
            tracing::info!(
                "prefill KV metadata: positions={:?}, indptr={:?}, indices={:?}, last_page={:?}, append={:?}",
                &self.host_position_ids,
                &self.host_kv_indptr,
                &self.host_kv_indices,
                &self.host_kv_last_page,
                &self.host_kv_append,
            );
            unsafe {
                driver::memcpy_htod_async(
                    kv_meta.position_ids.as_ptr::<u8>() as *mut u8,
                    self.host_position_ids.as_ptr() as *const u8,
                    padded_tokens * 4,
                    stream,
                )
                .expect("upload position_ids");
                driver::memcpy_htod_async(
                    kv_meta.kv_indptr.as_ptr::<u8>() as *mut u8,
                    self.host_kv_indptr.as_ptr() as *const u8,
                    (padded_tokens + 1) * 4,
                    stream,
                )
                .expect("upload kv_indptr");
                driver::memcpy_htod_async(
                    kv_meta.kv_indices.as_ptr::<u8>() as *mut u8,
                    self.host_kv_indices.as_ptr() as *const u8,
                    total_kv_indices * 4,
                    stream,
                )
                .expect("upload kv_indices");
                driver::memcpy_htod_async(
                    kv_meta.kv_last_page.as_ptr::<u8>() as *mut u8,
                    self.host_kv_last_page.as_ptr() as *const u8,
                    padded_tokens * 4,
                    stream,
                )
                .expect("upload kv_last_page");
                driver::memcpy_htod_async(
                    kv_meta.kv_append.as_ptr::<u8>() as *mut u8,
                    self.host_kv_append.as_ptr() as *const u8,
                    padded_tokens * 4,
                    stream,
                )
                .expect("upload kv_append");
            }
        }

        /// Launch the TK megakernel for a decode batch.
        ///
        /// `input_ids_gpu`: [batch_size] u32 token IDs on GPU.
        /// After return, logits are in `self.activations.logits`.
        pub unsafe fn launch_decode(&mut self, input_ids_gpu: GpuTensor, batch_size: usize) {
            tracing::info!("launch_decode: start, batch_size={}", batch_size);
            let stream = self.device.as_ref().unwrap().compute_stream;
            let config = self.llama_config.as_ref().unwrap().clone();

            // 1. Embedding gather → copy into hidden_states activation buffer.
            tracing::info!("launch_decode: embedding gather");
            let alloc = self.alloc.as_mut().unwrap();
            let embed = self.weights.as_ref().unwrap().embed_tokens;
            let hidden = unsafe { kernels::embedding_gather(embed, input_ids_gpu, alloc, stream) };
            tracing::info!(
                "launch_decode: embedding done, hidden ptr={:?}",
                hidden.as_ptr::<u8>()
            );

            let hd = config.hidden_size;
            let copy_bytes = batch_size * hd * DType::BF16.size_bytes();
            tracing::info!("launch_decode: copy hidden_states, {} bytes", copy_bytes);
            unsafe {
                driver::memcpy_dtod_async(
                    self.activations
                        .as_ref()
                        .unwrap()
                        .hidden_states
                        .as_mut_ptr::<u8>(),
                    hidden.as_ptr::<u8>(),
                    copy_bytes,
                    stream,
                )
                .expect("copy hidden_states");
            }
            drop(hidden);
            tracing::info!("launch_decode: hidden copied, building instructions");

            // 2. Ensure instructions are built.
            self.ensure_instructions(batch_size);
            let (ref inst, ref timing, ref decode_info) = self.instruction_cache[&batch_size];
            tracing::info!(
                "launch_decode: instructions built, sm={}, mps={}, n_batch_blocks={}, max_barrier_cols={}",
                decode_info.sm_count,
                decode_info.max_per_sm,
                decode_info.n_batch_blocks,
                decode_info.max_barrier_cols
            );
            // Log first few instructions
            {
                let iw = scheduler::INSTRUCTION_WIDTH;
                let total_inst = decode_info.sm_count * decode_info.max_per_sm * iw;
                let mut host_inst = vec![0i32; total_inst];
                unsafe {
                    driver::memcpy_dtoh_async(
                        host_inst.as_mut_ptr() as *mut u8,
                        inst.as_ptr::<u8>(),
                        total_inst * 4,
                        stream,
                    )
                    .ok();
                    driver::stream_synchronize(stream).ok();
                }
                for sm in 0..3.min(decode_info.sm_count) {
                    for ip in 0..5.min(decode_info.max_per_sm) {
                        let base = (sm * decode_info.max_per_sm + ip) * iw;
                        let instr = &host_inst[base..base + iw];
                        if instr[0] == 0 {
                            break;
                        }
                        tracing::info!(
                            "  decode SM[{}] inst[{}]: {:?}",
                            sm,
                            ip,
                            &instr[..6.min(iw)]
                        );
                    }
                }
                // Dump instructions for SMs of interest
                for sm in [49usize, 50, 51, 56, 67, 69, 138]
                    .iter()
                    .copied()
                    .filter(|&s| s < decode_info.sm_count)
                {
                    for ip in 0..3.min(decode_info.max_per_sm) {
                        let base = (sm * decode_info.max_per_sm + ip) * iw;
                        let instr = &host_inst[base..base + iw];
                        if instr[0] == 0 {
                            break;
                        }
                        tracing::info!(
                            "  decode SM[{}] inst[{}]: {:?}",
                            sm,
                            ip,
                            &instr[..6.min(iw)]
                        );
                    }
                }
            }

            let n_batch_blocks = decode_info.n_batch_blocks;
            let max_barrier_cols = decode_info.max_barrier_cols;
            let sm_count = decode_info.sm_count;
            let max_per_sm = decode_info.max_per_sm;
            // Extract raw pointers before mutable borrows below.
            let inst_ptr = inst.as_ptr::<u8>() as *mut u8;
            let timing_ptr = timing.as_ptr::<u8>() as *mut u8;

            // 3. Zero barrier, then pad for incomplete last batch block.
            self.ensure_and_zero_barrier(n_batch_blocks, max_barrier_cols);
            self.pad_prefill_barriers(batch_size, n_batch_blocks, max_barrier_cols);
            let bar_ptr = self.barrier.as_ref().unwrap().as_ptr::<u8>() as *mut u8;
            let nl = config.num_hidden_layers;

            // 4. Build typed LaunchArgs and launch via static megakernel.
            let weights = self.weights.as_ref().unwrap();
            let kv = self.kv_cache.as_ref().unwrap();
            let kv_meta = self.kv_meta.as_ref().unwrap();
            let act = self.activations.as_ref().unwrap();
            let num_pages = kv.num_blocks;

            let attn_scale = 1.0 / (config.head_dim as f32).sqrt();
            tracing::info!(
                "launch_decode: num_pages={}, attn_scale={}, batch_size={}",
                num_pages,
                attn_scale,
                batch_size
            );

            let args = unsafe {
                LaunchArgs {
                    barrier: GpuBarrier::from_raw(bar_ptr),
                    instructions: GpuVmLayout::from_raw(inst_ptr),
                    timings: GpuVmLayout::from_raw(timing_ptr),
                    qkv_weights: GpuWeight::from_raw(weights.qkv_proj.as_ptr::<u8>() as *mut u8),
                    attn_norm: GpuNormWeight::from_raw(weights.attn_norm.as_ptr::<u8>() as *mut u8),
                    o_proj: GpuWeight::from_raw(weights.o_proj.as_ptr::<u8>() as *mut u8),
                    mlp_norm: GpuNormWeight::from_raw(weights.mlp_norm.as_ptr::<u8>() as *mut u8),
                    up_weights: GpuWeight::from_raw(weights.up_proj.as_ptr::<u8>() as *mut u8),
                    gate_weights: GpuWeight::from_raw(weights.gate_proj.as_ptr::<u8>() as *mut u8),
                    down_proj: GpuWeightBig::from_raw(weights.down_proj.as_ptr::<u8>() as *mut u8),
                    lm_head_norm: GpuNormWeight::from_raw(
                        weights.lm_head_norm.as_ptr::<u8>() as *mut u8
                    ),
                    lm_head: GpuWeight::from_raw(weights.lm_head.as_ptr::<u8>() as *mut u8),
                    k_cache: GpuKvCache::from_raw(kv.k_cache.as_ptr::<u8>() as *mut u8),
                    v_cache: GpuKvCache::from_raw(kv.v_cache.as_ptr::<u8>() as *mut u8),
                    rope_cos: GpuRopeTable::from_raw(weights.rope_cos.as_ptr::<u8>() as *mut u8),
                    rope_sin: GpuRopeTable::from_raw(weights.rope_sin.as_ptr::<u8>() as *mut u8),
                    hidden_states: GpuActivation::from_raw(
                        act.hidden_states.as_ptr::<u8>() as *mut u8
                    ),
                    rms_rope: GpuActivation::from_raw(act.rms_rope.as_ptr::<u8>() as *mut u8),
                    rms_gate: GpuActivation::from_raw(act.rms_gate.as_ptr::<u8>() as *mut u8),
                    q_post_rope: GpuActivation::from_raw(act.q_post_rope.as_ptr::<u8>() as *mut u8),
                    attn_out: GpuActivation::from_raw(act.attn_out.as_ptr::<u8>() as *mut u8),
                    silu_out: GpuActivationBig::from_raw(act.silu_out.as_ptr::<u8>() as *mut u8),
                    rms_lm: GpuActivation::from_raw(act.rms_lm.as_ptr::<u8>() as *mut u8),
                    logits: GpuLogits::from_raw(act.logits.as_ptr::<u8>() as *mut u8),
                    position_ids: GpuMetaVec::from_raw(
                        kv_meta.position_ids.as_ptr::<u8>() as *mut u8
                    ),
                    kv_indptr: GpuMetaVec::from_raw(kv_meta.kv_indptr.as_ptr::<u8>() as *mut u8),
                    kv_indices: GpuMetaVec::from_raw(kv_meta.kv_indices.as_ptr::<u8>() as *mut u8),
                    kv_last_page: GpuMetaVec::from_raw(
                        kv_meta.kv_last_page.as_ptr::<u8>() as *mut u8
                    ),
                    kv_append: GpuMetaVec::from_raw(kv_meta.kv_append.as_ptr::<u8>() as *mut u8),
                    // Dummy prefill metadata for decode — reuse kv_last_page as valid non-null ptr
                    prefill_qo_indptr: GpuMetaVec::from_raw(
                        kv_meta.kv_last_page.as_ptr::<u8>() as *mut u8
                    ),
                    prefill_kv_indptr: GpuMetaVec::from_raw(
                        kv_meta.kv_last_page.as_ptr::<u8>() as *mut u8
                    ),
                    prefill_kv_indices: GpuMetaVec::from_raw(
                        kv_meta.kv_last_page.as_ptr::<u8>() as *mut u8
                    ),
                    prefill_kv_last_page_len: GpuMetaVec::from_raw(
                        kv_meta.kv_last_page.as_ptr::<u8>() as *mut u8,
                    ),
                    attn_scale,
                    rms_norm_eps: config.rms_norm_eps,
                    num_pages: num_pages as i32,
                    num_layers: nl,
                    prefill_num_seqs: 0,
                    prefill_num_kv_pages: 0,
                }
            };

            let barrier_shape = [nl, scheduler::NUM_OPS, n_batch_blocks, max_barrier_cols];
            let inst_shape = [1, sm_count, max_per_sm, scheduler::INSTRUCTION_WIDTH];
            let timing_shape = [1, sm_count, max_per_sm, scheduler::TIMING_WIDTH];

            let variant = self
                .kernel_variant
                .as_ref()
                .expect("kernel variant not set");
            let rc = unsafe {
                MegakernelLlamaSm89::launch_decode(
                    &args,
                    variant,
                    DecodeBatchSize(batch_size as i32),
                    NumTokens(0), // num_prefill_tokens
                    barrier_shape,
                    inst_shape,
                    timing_shape,
                    stream as u64,
                )
            };
            if rc != 0 {
                panic!("launch_decode failed: rc={rc}");
            }
            unsafe {
                driver::stream_synchronize(stream).expect("decode stream sync");
            }
            tracing::info!("launch_decode: decode kernel completed");
        }

        /// Launch the TK megakernel for a prefill batch.
        ///
        /// `input_ids_gpu`: [total_tokens] u32 token IDs on GPU.
        /// `prefill_seqs`: per-sequence metadata for instruction generation.
        /// `block_tables`: per-sequence block tables (physical page IDs).
        /// `positions`: per-token position IDs.
        /// `seq_lens`: per-sequence total lengths (including new tokens).
        ///
        /// After return, logits are in `self.activations.logits`.
        pub unsafe fn launch_prefill(
            &mut self,
            input_ids_gpu: GpuTensor,
            prefill_seqs: &[PrefillSeq],
            block_tables: &[&[usize]],
            positions: &[u32],
            seq_lens: &[usize],
            total_tokens: usize,
        ) {
            let stream = self.device.as_ref().unwrap().compute_stream;
            let config = self.llama_config.as_ref().unwrap().clone();
            let hd = config.hidden_size;

            // 1. Embedding gather → hidden_states.
            let alloc = self.alloc.as_mut().unwrap();
            let embed = self.weights.as_ref().unwrap().embed_tokens;
            let hidden = unsafe { kernels::embedding_gather(embed, input_ids_gpu, alloc, stream) };

            let copy_bytes = total_tokens * hd * DType::BF16.size_bytes();
            unsafe {
                driver::memcpy_dtod_async(
                    self.activations
                        .as_ref()
                        .unwrap()
                        .hidden_states
                        .as_mut_ptr::<u8>(),
                    hidden.as_ptr::<u8>(),
                    copy_bytes,
                    stream,
                )
                .expect("copy hidden_states");
            }
            drop(hidden);

            // 2. Build prefill instructions.
            let tk_cfg = self.tk_model_config.as_ref().unwrap();
            let prefill_info = scheduler::build_prefill_instructions(tk_cfg, prefill_seqs);

            let alloc = self.alloc.as_mut().unwrap();
            let sm = prefill_info.sm_count;
            let mps = prefill_info.max_per_sm;
            let iw = scheduler::INSTRUCTION_WIDTH;
            let inst_tensor = alloc.alloc_tensor(&[sm, mps, iw], DType::I32);
            let inst_bytes = prefill_info.instructions.len() * 4;
            unsafe {
                driver::memcpy_htod_async(
                    inst_tensor.as_mut_ptr::<u8>(),
                    prefill_info.instructions.as_ptr() as *const u8,
                    inst_bytes,
                    stream,
                )
                .expect("upload prefill instructions");
            }

            let tw = scheduler::TIMING_WIDTH;
            let timing_tensor = alloc.alloc_tensor(&[sm, mps, tw], DType::I32);
            unsafe {
                driver::memset_d8(
                    timing_tensor.as_ptr::<u8>() as *mut u8,
                    0,
                    sm * mps * tw * 4,
                    stream,
                )
                .expect("zero timings");
            }

            // 3. Zero barrier, then pad partial last batch block.
            self.ensure_and_zero_barrier(
                prefill_info.n_batch_blocks,
                prefill_info.max_barrier_cols,
            );
            self.pad_prefill_barriers(
                total_tokens,
                prefill_info.n_batch_blocks,
                prefill_info.max_barrier_cols,
            );
            let bar_ptr = self.barrier.as_ref().unwrap().as_ptr::<u8>() as *mut u8;
            let nl = config.num_hidden_layers;

            // 4. Upload per-token KV metadata for QKV_RopeAppend.
            // Unlike decode (1 token per seq), prefill has multiple tokens per seq.
            // QKV_RopeAppend reads kv_append_indices[token_idx] per token, so we
            // must expand per-sequence block tables into per-token CSR entries.
            unsafe {
                self.upload_prefill_kv_metadata(
                    prefill_seqs,
                    block_tables,
                    positions,
                    seq_lens,
                    total_tokens,
                );
            }

            // 5. Build and upload prefill-specific CSR metadata.
            // prefill_qo_indptr: CSR pointers into q_post_rope for each sequence.
            // prefill_kv_indptr: CSR pointers into prefill_kv_indices for KV pages.
            self.host_prefill_qo_indptr.clear();
            self.host_prefill_kv_indptr.clear();
            self.host_prefill_kv_indices.clear();
            self.host_prefill_kv_last_page_len.clear();

            let mut qo_offset = 0i32;
            let mut kv_offset = 0i32;
            self.host_prefill_qo_indptr.push(0);
            self.host_prefill_kv_indptr.push(0);

            for (i, seq) in prefill_seqs.iter().enumerate() {
                qo_offset += seq.chunk_len as i32;
                self.host_prefill_qo_indptr.push(qo_offset);

                // KV pages for this sequence: all pages in the block table
                // up to the total sequence length (extend_offset + chunk_len).
                let total_seq_len = seq.extend_offset + seq.chunk_len;
                let num_kv_pages = total_seq_len.div_ceil(KV_PAGE_SIZE);
                let blocks = block_tables[i];
                for &b in &blocks[..num_kv_pages] {
                    self.host_prefill_kv_indices.push(b as i32);
                }
                kv_offset += num_kv_pages as i32;
                self.host_prefill_kv_indptr.push(kv_offset);

                // Last page length.
                let last_page_len = total_seq_len % KV_PAGE_SIZE;
                self.host_prefill_kv_last_page_len
                    .push(if last_page_len == 0 {
                        KV_PAGE_SIZE as i32
                    } else {
                        last_page_len as i32
                    });
            }

            let kv_meta = self.kv_meta.as_ref().unwrap();
            let num_prefill_seqs = prefill_seqs.len();
            let total_prefill_kv_pages = self.host_prefill_kv_indices.len();

            unsafe {
                driver::memcpy_htod_async(
                    kv_meta.prefill_qo_indptr.as_ptr::<u8>() as *mut u8,
                    self.host_prefill_qo_indptr.as_ptr() as *const u8,
                    (num_prefill_seqs + 1) * 4,
                    stream,
                )
                .expect("upload prefill_qo_indptr");
                driver::memcpy_htod_async(
                    kv_meta.prefill_kv_indptr.as_ptr::<u8>() as *mut u8,
                    self.host_prefill_kv_indptr.as_ptr() as *const u8,
                    (num_prefill_seqs + 1) * 4,
                    stream,
                )
                .expect("upload prefill_kv_indptr");
                driver::memcpy_htod_async(
                    kv_meta.prefill_kv_indices.as_ptr::<u8>() as *mut u8,
                    self.host_prefill_kv_indices.as_ptr() as *const u8,
                    total_prefill_kv_pages * 4,
                    stream,
                )
                .expect("upload prefill_kv_indices");
                driver::memcpy_htod_async(
                    kv_meta.prefill_kv_last_page_len.as_ptr::<u8>() as *mut u8,
                    self.host_prefill_kv_last_page_len.as_ptr() as *const u8,
                    num_prefill_seqs * 4,
                    stream,
                )
                .expect("upload prefill_kv_last_page_len");
            }

            // 6. Upload per-sequence chunk_lens and extend_offsets for prefill attention.
            let alloc = self.alloc.as_mut().unwrap();
            let host_chunk_lens: Vec<i32> =
                prefill_seqs.iter().map(|s| s.chunk_len as i32).collect();
            let host_extend_offsets: Vec<i32> = prefill_seqs
                .iter()
                .map(|s| s.extend_offset as i32)
                .collect();
            let chunk_lens_gpu = alloc.alloc_tensor(&[num_prefill_seqs], DType::I32);
            let extend_offsets_gpu = alloc.alloc_tensor(&[num_prefill_seqs], DType::I32);
            unsafe {
                driver::memcpy_htod_async(
                    chunk_lens_gpu.as_ptr::<u8>() as *mut u8,
                    host_chunk_lens.as_ptr() as *const u8,
                    num_prefill_seqs * 4,
                    stream,
                )
                .expect("upload seq_chunk_lens");
                driver::memcpy_htod_async(
                    extend_offsets_gpu.as_ptr::<u8>() as *mut u8,
                    host_extend_offsets.as_ptr() as *const u8,
                    num_prefill_seqs * 4,
                    stream,
                )
                .expect("upload seq_extend_offsets");
            }
            let seq_chunk_lens =
                unsafe { GpuMetaVec::from_raw(chunk_lens_gpu.as_ptr::<u8>() as *mut u8) };
            let seq_extend_offsets =
                unsafe { GpuMetaVec::from_raw(extend_offsets_gpu.as_ptr::<u8>() as *mut u8) };

            // 7. Build typed LaunchArgs and launch via static megakernel.
            let weights = self.weights.as_ref().unwrap();
            let kv = self.kv_cache.as_ref().unwrap();
            let act = self.activations.as_ref().unwrap();
            let num_pages = kv.num_blocks;
            let attn_scale = 1.0 / (config.head_dim as f32).sqrt();

            tracing::info!(
                "TK prefill launch: total_tokens={}, n_batch_blocks={}, sm={}, mps={}",
                total_tokens,
                prefill_info.n_batch_blocks,
                sm,
                mps,
            );

            let args = unsafe {
                LaunchArgs {
                    barrier: GpuBarrier::from_raw(bar_ptr),
                    instructions: GpuVmLayout::from_raw(inst_tensor.as_ptr::<u8>() as *mut u8),
                    timings: GpuVmLayout::from_raw(timing_tensor.as_ptr::<u8>() as *mut u8),
                    qkv_weights: GpuWeight::from_raw(weights.qkv_proj.as_ptr::<u8>() as *mut u8),
                    attn_norm: GpuNormWeight::from_raw(weights.attn_norm.as_ptr::<u8>() as *mut u8),
                    o_proj: GpuWeight::from_raw(weights.o_proj.as_ptr::<u8>() as *mut u8),
                    mlp_norm: GpuNormWeight::from_raw(weights.mlp_norm.as_ptr::<u8>() as *mut u8),
                    up_weights: GpuWeight::from_raw(weights.up_proj.as_ptr::<u8>() as *mut u8),
                    gate_weights: GpuWeight::from_raw(weights.gate_proj.as_ptr::<u8>() as *mut u8),
                    down_proj: GpuWeightBig::from_raw(weights.down_proj.as_ptr::<u8>() as *mut u8),
                    lm_head_norm: GpuNormWeight::from_raw(
                        weights.lm_head_norm.as_ptr::<u8>() as *mut u8
                    ),
                    lm_head: GpuWeight::from_raw(weights.lm_head.as_ptr::<u8>() as *mut u8),
                    k_cache: GpuKvCache::from_raw(kv.k_cache.as_ptr::<u8>() as *mut u8),
                    v_cache: GpuKvCache::from_raw(kv.v_cache.as_ptr::<u8>() as *mut u8),
                    rope_cos: GpuRopeTable::from_raw(weights.rope_cos.as_ptr::<u8>() as *mut u8),
                    rope_sin: GpuRopeTable::from_raw(weights.rope_sin.as_ptr::<u8>() as *mut u8),
                    hidden_states: GpuActivation::from_raw(
                        act.hidden_states.as_ptr::<u8>() as *mut u8
                    ),
                    rms_rope: GpuActivation::from_raw(act.rms_rope.as_ptr::<u8>() as *mut u8),
                    rms_gate: GpuActivation::from_raw(act.rms_gate.as_ptr::<u8>() as *mut u8),
                    q_post_rope: GpuActivation::from_raw(act.q_post_rope.as_ptr::<u8>() as *mut u8),
                    attn_out: GpuActivation::from_raw(act.attn_out.as_ptr::<u8>() as *mut u8),
                    silu_out: GpuActivationBig::from_raw(act.silu_out.as_ptr::<u8>() as *mut u8),
                    rms_lm: GpuActivation::from_raw(act.rms_lm.as_ptr::<u8>() as *mut u8),
                    logits: GpuLogits::from_raw(act.logits.as_ptr::<u8>() as *mut u8),
                    position_ids: GpuMetaVec::from_raw(
                        kv_meta.position_ids.as_ptr::<u8>() as *mut u8
                    ),
                    kv_indptr: GpuMetaVec::from_raw(kv_meta.kv_indptr.as_ptr::<u8>() as *mut u8),
                    kv_indices: GpuMetaVec::from_raw(kv_meta.kv_indices.as_ptr::<u8>() as *mut u8),
                    kv_last_page: GpuMetaVec::from_raw(
                        kv_meta.kv_last_page.as_ptr::<u8>() as *mut u8
                    ),
                    kv_append: GpuMetaVec::from_raw(kv_meta.kv_append.as_ptr::<u8>() as *mut u8),
                    prefill_qo_indptr: GpuMetaVec::from_raw(
                        kv_meta.prefill_qo_indptr.as_ptr::<u8>() as *mut u8,
                    ),
                    prefill_kv_indptr: GpuMetaVec::from_raw(
                        kv_meta.prefill_kv_indptr.as_ptr::<u8>() as *mut u8,
                    ),
                    prefill_kv_indices: GpuMetaVec::from_raw(
                        kv_meta.prefill_kv_indices.as_ptr::<u8>() as *mut u8,
                    ),
                    prefill_kv_last_page_len: GpuMetaVec::from_raw(
                        kv_meta.prefill_kv_last_page_len.as_ptr::<u8>() as *mut u8,
                    ),
                    attn_scale,
                    rms_norm_eps: config.rms_norm_eps,
                    num_pages: num_pages as i32,
                    num_layers: nl,
                    prefill_num_seqs: num_prefill_seqs,
                    prefill_num_kv_pages: total_prefill_kv_pages,
                }
            };

            let barrier_shape = [
                nl,
                scheduler::NUM_OPS,
                prefill_info.n_batch_blocks,
                prefill_info.max_barrier_cols,
            ];
            let inst_shape = [1, sm, mps, iw];
            let timing_shape = [1, sm, mps, tw];

            unsafe {
                driver::stream_synchronize(stream).expect("pre-launch sync");
                tracing::info!("TK prefill: all uploads complete, launching kernel");
                let variant = self
                    .kernel_variant
                    .as_ref()
                    .expect("kernel variant not set");
                let rc = MegakernelLlamaSm89::launch_prefill(
                    &args,
                    variant,
                    NumSeqs(num_prefill_seqs as i32),
                    NumTokens(total_tokens as i32),
                    &seq_chunk_lens,
                    &seq_extend_offsets,
                    barrier_shape,
                    inst_shape,
                    timing_shape,
                    stream as u64,
                );
                if rc != 0 {
                    panic!("launch_prefill failed: rc={rc}");
                }
                tracing::info!("TK prefill: kernel launched, syncing...");
                driver::stream_synchronize(stream).expect("post-launch sync");
                tracing::info!("TK prefill: kernel completed successfully");
            }
        }

        // ---------------------------------------------------------------
        // High-level lifecycle methods (called by TkWorkerAdapter)
        // ---------------------------------------------------------------

        /// Initialize the GPU device and detect SM count.
        pub fn init_device(&mut self) -> Result<()> {
            let device = GpuDevice::new(self.config.device_id)?;

            // Detect SM count via cuDeviceGetAttribute.
            let sm_count = if self.config.sm_count > 0 {
                self.config.sm_count
            } else {
                let mut count: i32 = 0;
                unsafe {
                    let result = cudarc::driver::sys::cuDeviceGetAttribute(
                        &mut count as *mut i32,
                        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                        self.config.device_id,
                    );
                    if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        anyhow::bail!("cuDeviceGetAttribute failed: {:?}", result);
                    }
                }
                count as usize
            };

            tracing::info!(
                "TkWorker: device {} initialized, {} SMs",
                self.config.device_id,
                sm_count
            );
            self.config.sm_count = sm_count;
            self.device = Some(device);
            Ok(())
        }

        /// Load model weights from a resolved model directory.
        ///
        /// `llama_config` should be parsed from config.json via
        /// `gpu_worker_base::llama_config_from_hf`.
        pub fn load_model(
            &mut self,
            model_dir: &std::path::Path,
            llama_config: LlamaConfig,
        ) -> Result<()> {
            let device = self
                .device
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("device not initialized"))?;

            let mut gpu_weights =
                vllm_cuda::weights::GpuWeights::from_dir(model_dir, device.compute_stream)?;

            let weights = unsafe { TkWeights::load(&mut gpu_weights, &llama_config, device)? };

            let gqa_ratio = llama_config.num_attention_heads / llama_config.num_kv_heads;
            let tk_model_config = TkModelConfig {
                num_hidden_layers: llama_config.num_hidden_layers,
                hidden_dim: llama_config.hidden_size,
                intermediate_dim: llama_config.intermediate_size,
                num_attention_heads: llama_config.num_attention_heads,
                num_kv_heads: llama_config.num_kv_heads,
                head_dim: llama_config.head_dim,
                vocab_size: llama_config.vocab_size,
                sm_count: self.config.sm_count,
                matmul_batch_block_size: 128,
                attn_batch_block_size: gqa_ratio,
            };

            let variant = vllm_tk_static::KernelVariant::from_dims(
                llama_config.hidden_size,
                llama_config.intermediate_size,
                llama_config.head_dim,
                llama_config.num_attention_heads,
                llama_config.num_kv_heads,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            tracing::info!("TkWorker: selected kernel variant {:?}", variant);

            self.weights = Some(weights);
            self.llama_config = Some(llama_config);
            self.tk_model_config = Some(tk_model_config);
            self.kernel_variant = Some(variant);
            Ok(())
        }

        /// Query GPU memory after weights are loaded.
        ///
        /// Returns `(free_bytes, total_bytes)`.
        pub fn query_gpu_memory(&self) -> Result<(usize, usize)> {
            let device = self
                .device
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("device not initialized"))?;
            unsafe { driver::ctx_set_current(device.ctx)? };
            let (free, total) = cudarc::driver::result::mem_get_info()
                .map_err(|e| anyhow::anyhow!("cuMemGetInfo: {e}"))?;
            Ok((free, total))
        }

        /// Allocate KV cache, activation buffers, and KV metadata.
        pub fn initialize_cache(&mut self, num_blocks: usize) -> Result<()> {
            let config = self
                .llama_config
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("model not loaded"))?
                .clone();
            let device = self
                .device
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("device not initialized"))?;
            let stream = device.compute_stream;

            // KV cache.
            let kv_cache = unsafe {
                TkKvCache::alloc(
                    num_blocks,
                    config.num_hidden_layers,
                    config.num_kv_heads,
                    config.head_dim,
                    stream,
                )?
            };
            tracing::info!(
                "TkWorker: KV cache allocated: {} blocks, {:.1} MiB",
                num_blocks,
                kv_cache.total_bytes() as f64 / 1_048_576.0,
            );

            // Allocations for activations and metadata.
            let max_bs = self.config.max_num_batched_tokens;
            let max_pages = num_blocks; // conservative upper bound
            let mut alloc = CachingAllocator::new();

            let activations = TkActivationBuffers::alloc(max_bs, &config, &mut alloc);
            let kv_meta = TkKvMetadata::alloc(max_bs, max_pages, &mut alloc);

            // Pre-allocate host staging buffers.
            self.host_position_ids = Vec::with_capacity(max_bs);
            self.host_kv_indptr = Vec::with_capacity(max_bs + 1);
            self.host_kv_indices = Vec::with_capacity(max_pages);
            self.host_kv_last_page = Vec::with_capacity(max_bs);
            self.host_kv_append = Vec::with_capacity(max_bs);

            self.kv_cache = Some(kv_cache);
            self.activations = Some(activations);
            self.kv_meta = Some(kv_meta);
            self.alloc = Some(alloc);
            Ok(())
        }

        /// Ensure CUDA context is set on the current thread (idempotent).
        pub fn ensure_ctx(&mut self) -> Result<()> {
            if !self.ctx_set_on_thread {
                if let Some(ref dev) = self.device {
                    unsafe { driver::ctx_set_current(dev.ctx)? };
                }
                self.ctx_set_on_thread = true;
            }
            Ok(())
        }

        // ---------------------------------------------------------------
        // Accessors
        // ---------------------------------------------------------------

        pub fn device(&self) -> Option<&GpuDevice> {
            self.device.as_ref()
        }

        pub fn llama_config(&self) -> Option<&LlamaConfig> {
            self.llama_config.as_ref()
        }

        pub fn stream(&self) -> Option<CUstream> {
            self.device.as_ref().map(|d| d.compute_stream)
        }

        pub fn alloc(&mut self) -> Option<&mut CachingAllocator> {
            self.alloc.as_mut()
        }

        /// Returns the logits activation buffer (valid after `launch_decode`).
        pub fn logits_tensor(&self) -> Option<GpuTensor> {
            self.activations.as_ref().map(|a| a.logits.as_gpu_tensor())
        }

        /// Number of KV cache blocks allocated.
        pub fn num_kv_blocks(&self) -> usize {
            self.kv_cache.as_ref().map_or(0, |kv| kv.num_blocks)
        }
    }

    // SAFETY: TkWorker is used from a single thread at a time (same as CudaWorker).
    // The raw CUDA pointers are !Send but are only accessed on the worker thread.
    unsafe impl Send for TkWorker {}
}

/// KV page size (must match llama_sm89.cuh).
const KV_PAGE_SIZE: usize = 64;

/// Build CSR paged KV metadata from block tables (pure CPU, no GPU dependency).
///
/// Populates the output vectors with:
/// - `indptr`: CSR row pointers `[0, num_pages_0, num_pages_0+num_pages_1, ...]`
/// - `indices`: flat list of physical page IDs
/// - `last_page_len`: tokens in the last page per sequence
/// - `append_indices`: flat slot index for writing new KV entries
pub fn build_csr_metadata(
    block_tables: &[&[usize]],
    positions: &[u32],
    seq_lens: &[usize],
    indptr: &mut Vec<i32>,
    indices: &mut Vec<i32>,
    last_page_len: &mut Vec<i32>,
    append_indices: &mut Vec<i32>,
) {
    indptr.clear();
    indices.clear();
    last_page_len.clear();
    append_indices.clear();

    let mut page_offset = 0i32;
    indptr.push(0);
    for (i, blocks) in block_tables.iter().enumerate() {
        for &block_id in *blocks {
            indices.push(block_id as i32);
        }
        page_offset += blocks.len() as i32;
        indptr.push(page_offset);

        // Last page length: tokens in last page.
        let sl = seq_lens[i];
        let lpl = if sl == 0 {
            0
        } else {
            let rem = sl % KV_PAGE_SIZE;
            if rem == 0 { KV_PAGE_SIZE } else { rem }
        };
        last_page_len.push(lpl as i32);

        // Append index: flat slot for new KV entry.
        let pos = positions[i] as usize;
        let page_idx = blocks.last().copied().unwrap_or(0);
        let offset_in_page = pos % KV_PAGE_SIZE;
        append_indices.push((page_idx * KV_PAGE_SIZE + offset_in_page) as i32);
    }
}

#[cfg(feature = "cuda")]
pub use inner::{TkKvCache, TkWorker, TkWorkerConfig};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_single_sequence() {
        // 1 sequence, 10 tokens, 1 page (page_size=64).
        let blocks: &[&[usize]] = &[&[5]];
        let positions = &[9u32]; // pos=9 (0-indexed, next token at position 9)
        let seq_lens = &[10usize]; // total tokens in KV including current

        let mut indptr = Vec::new();
        let mut indices = Vec::new();
        let mut last_page = Vec::new();
        let mut append = Vec::new();

        build_csr_metadata(
            blocks,
            positions,
            seq_lens,
            &mut indptr,
            &mut indices,
            &mut last_page,
            &mut append,
        );

        assert_eq!(indptr, vec![0, 1]);
        assert_eq!(indices, vec![5]);
        assert_eq!(last_page, vec![10]); // 10 tokens in only page
        assert_eq!(append, vec![(5 * 64 + 9) as i32]); // page 5, offset 9
    }

    #[test]
    fn test_csr_multi_page_sequence() {
        // 1 sequence, 100 tokens across 2 pages (page_size=64).
        let blocks: &[&[usize]] = &[&[3, 7]]; // pages 3, 7
        let positions = &[99u32]; // next token at position 99
        let seq_lens = &[100usize];

        let mut indptr = Vec::new();
        let mut indices = Vec::new();
        let mut last_page = Vec::new();
        let mut append = Vec::new();

        build_csr_metadata(
            blocks,
            positions,
            seq_lens,
            &mut indptr,
            &mut indices,
            &mut last_page,
            &mut append,
        );

        assert_eq!(indptr, vec![0, 2]);
        assert_eq!(indices, vec![3, 7]);
        assert_eq!(last_page, vec![36]); // 100 % 64 = 36
        // Append: page 7 (last block), offset = 99 % 64 = 35
        assert_eq!(append, vec![(7 * 64 + 35) as i32]);
    }

    #[test]
    fn test_csr_batch_of_3() {
        // 3 sequences with different page counts.
        let b0: &[usize] = &[0]; // 1 page, 30 tokens
        let b1: &[usize] = &[1, 2]; // 2 pages, 64 tokens (exactly full first page)
        let b2: &[usize] = &[4, 5, 6]; // 3 pages, 150 tokens
        let blocks: &[&[usize]] = &[b0, b1, b2];
        let positions = &[29u32, 63, 149];
        let seq_lens = &[30usize, 64, 150];

        let mut indptr = Vec::new();
        let mut indices = Vec::new();
        let mut last_page = Vec::new();
        let mut append = Vec::new();

        build_csr_metadata(
            blocks,
            positions,
            seq_lens,
            &mut indptr,
            &mut indices,
            &mut last_page,
            &mut append,
        );

        // Indptr: cumulative page counts.
        assert_eq!(indptr, vec![0, 1, 3, 6]);

        // Indices: all page IDs flattened.
        assert_eq!(indices, vec![0, 1, 2, 4, 5, 6]);

        // Last page lengths.
        assert_eq!(last_page[0], 30); // 30 % 64 = 30
        assert_eq!(last_page[1], 64); // 64 % 64 = 0 → KV_PAGE_SIZE = 64
        assert_eq!(last_page[2], 22); // 150 % 64 = 22

        // Append indices.
        assert_eq!(append[0], (0 * 64 + 29) as i32); // page 0, offset 29
        assert_eq!(append[1], (2 * 64 + 63) as i32); // page 2, offset 63%64=63
        assert_eq!(append[2], (6 * 64 + 21) as i32); // page 6, offset 149%64=21
    }

    #[test]
    fn test_csr_exact_page_boundary() {
        // Sequence with exactly page_size tokens (boundary case).
        let blocks: &[&[usize]] = &[&[10]];
        let positions = &[63u32]; // last token at pos 63
        let seq_lens = &[64usize]; // exactly 1 full page

        let mut indptr = Vec::new();
        let mut indices = Vec::new();
        let mut last_page = Vec::new();
        let mut append = Vec::new();

        build_csr_metadata(
            blocks,
            positions,
            seq_lens,
            &mut indptr,
            &mut indices,
            &mut last_page,
            &mut append,
        );

        assert_eq!(last_page, vec![64]); // full page → KV_PAGE_SIZE
        assert_eq!(append, vec![(10 * 64 + 63) as i32]);
    }

    #[test]
    fn test_csr_reuse_clears_buffers() {
        // Call twice to verify buffers are properly cleared.
        let mut indptr = Vec::new();
        let mut indices = Vec::new();
        let mut last_page = Vec::new();
        let mut append = Vec::new();

        // First call.
        build_csr_metadata(
            &[&[0, 1, 2]],
            &[130],
            &[131],
            &mut indptr,
            &mut indices,
            &mut last_page,
            &mut append,
        );
        assert_eq!(indptr.len(), 2);
        assert_eq!(indices.len(), 3);

        // Second call with different data.
        build_csr_metadata(
            &[&[5]],
            &[10],
            &[11],
            &mut indptr,
            &mut indices,
            &mut last_page,
            &mut append,
        );
        assert_eq!(indptr, vec![0, 1]);
        assert_eq!(indices, vec![5]);
        assert_eq!(last_page, vec![11]);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::inner::*;
    use vllm_cuda::alloc::CachingAllocator;
    use vllm_cuda::device::GpuDevice;
    use vllm_cuda::dtype::DType;
    use vllm_cuda::model::llama::LlamaConfig;

    fn llama_1b_config() -> LlamaConfig {
        LlamaConfig {
            hidden_size: 2048,
            num_attention_heads: 32,
            num_kv_heads: 8,
            num_hidden_layers: 16,
            intermediate_size: 8192,
            vocab_size: 128256,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            head_dim: 64,
            tie_word_embeddings: true,
            llama3_rope_scaling: None,
        }
    }

    #[test]
    #[ignore] // requires CUDA GPU
    fn test_tk_kv_cache_shapes() {
        let device = GpuDevice::new(0).expect("GpuDevice::new");
        let num_blocks = 64;
        let num_layers = 16;
        let nkh = 8;
        let hdm = 64;
        let kv = unsafe {
            TkKvCache::alloc(num_blocks, num_layers, nkh, hdm, device.compute_stream)
                .expect("alloc KV cache")
        };

        // Shape: [NL * num_blocks, page_size/kv_block_size, NKH, HDM]
        //      = [16*64, 64/16, 8, 64] = [1024, 4, 8, 64]
        assert_eq!(kv.k_cache.shape(), &[1024, 4, 8, 64]);
        assert_eq!(kv.v_cache.shape(), &[1024, 4, 8, 64]);
        assert_eq!(kv.num_blocks, 64);
        assert_eq!(kv.num_layers, 16);

        // Total bytes: 2 * 1024 * 4 * 8 * 64 * 2(bf16) = 8388608
        let expected = 2 * 1024 * 4 * 8 * 64 * 2;
        assert_eq!(kv.total_bytes(), expected);
    }

    #[test]
    #[ignore] // requires CUDA GPU
    fn test_tk_activation_buffer_shapes() {
        let device = GpuDevice::new(0).expect("GpuDevice::new");
        let config = llama_1b_config();
        let mut alloc = CachingAllocator::new();
        let max_bs = 128;
        let act = TkActivationBuffers::alloc(max_bs, &config, &mut alloc);

        assert_eq!(act.hidden_states.shape(), &[128, 2048]);
        assert_eq!(act.rms_rope.shape(), &[128, 2048]);
        assert_eq!(act.rms_gate.shape(), &[128, 2048]);
        assert_eq!(act.q_post_rope.shape(), &[128, 2048]);
        assert_eq!(act.attn_out.shape(), &[128, 2048]);
        assert_eq!(act.silu_out.shape(), &[128, 8192]);
        assert_eq!(act.rms_lm.shape(), &[128, 2048]);
        assert_eq!(act.logits.shape(), &[128, 128256]);
    }

    #[test]
    #[ignore] // requires CUDA GPU
    fn test_tk_instruction_caching() {
        let device = GpuDevice::new(0).expect("GpuDevice::new");
        let mut alloc = CachingAllocator::new();
        let tk_cfg = crate::scheduler::TkModelConfig {
            num_hidden_layers: 16,
            hidden_dim: 2048,
            intermediate_dim: 8192,
            head_dim: 64,
            num_attention_heads: 32,
            num_kv_heads: 8,
            vocab_size: 128256,
            sm_count: 142,
            matmul_batch_block_size: 128,
            attn_batch_block_size: 4,
        };

        let mut worker = TkWorker::new(TkWorkerConfig {
            model_path: String::new(),
            dtype: "bf16".into(),
            hf_token: None,
            device_id: 0,
            max_num_batched_tokens: 512,
            gpu_memory_utilization: 0.9,
            sm_count: 142,
        });
        worker.device = Some(device);
        worker.alloc = Some(alloc);
        worker.tk_model_config = Some(tk_cfg);
        worker.llama_config = Some(llama_1b_config());

        // First call builds instructions.
        worker.ensure_instructions(128);
        assert!(worker.instruction_cache.contains_key(&128));

        // Second call reuses cached.
        let ptr_before = worker.instruction_cache[&128].0.as_ptr::<u8>();
        worker.ensure_instructions(128);
        let ptr_after = worker.instruction_cache[&128].0.as_ptr::<u8>();
        assert_eq!(
            ptr_before, ptr_after,
            "cached instructions should be reused"
        );

        // Different batch size creates new entry.
        worker.ensure_instructions(256);
        assert!(worker.instruction_cache.contains_key(&256));
        assert_eq!(worker.instruction_cache.len(), 2);

        // Verify instruction tensor shapes.
        let (inst, timing, decode) = &worker.instruction_cache[&128];
        assert_eq!(inst.shape()[0], 142); // sm_count
        assert_eq!(inst.shape()[2], 32); // INSTRUCTION_WIDTH
        assert_eq!(timing.shape()[2], 128); // TIMING_WIDTH
        assert_eq!(decode.sm_count, 142);
    }

    #[test]
    #[ignore] // requires CUDA GPU
    fn test_tk_barrier_sizing() {
        let device = GpuDevice::new(0).expect("GpuDevice::new");
        let mut alloc = CachingAllocator::new();

        let mut worker = TkWorker::new(TkWorkerConfig {
            model_path: String::new(),
            dtype: "bf16".into(),
            hf_token: None,
            device_id: 0,
            max_num_batched_tokens: 512,
            gpu_memory_utilization: 0.9,
            sm_count: 142,
        });
        worker.device = Some(device);
        worker.alloc = Some(alloc);
        worker.llama_config = Some(llama_1b_config());

        // BS=128: n_batch_blocks=1, max_barrier_cols=128
        worker.ensure_and_zero_barrier(1, 128);
        let bar = worker.barrier.as_ref().unwrap();
        // size = 16 * 10 * 1 * 128 = 20480
        assert_eq!(bar.numel(), 20480);

        // BS=256: n_batch_blocks=2, max_barrier_cols=128 → bigger
        worker.ensure_and_zero_barrier(2, 128);
        let bar = worker.barrier.as_ref().unwrap();
        // size = 16 * 10 * 2 * 128 = 40960
        assert_eq!(bar.numel(), 40960);
    }
}
