// SPDX-License-Identifier: Apache-2.0
//! TkWorkerAdapter: Worker trait implementation backed by the TK KVM megakernel.
//!
//! Thin adapter that composes:
//! - `vllm_tk::worker::TkWorker` — kernel launch, KV cache, activations
//! - `vllm_executor::input_batch::InputBatch` — persistent batch state
//! - `vllm_executor::gpu_worker_base` — shared GPU worker utilities
//!
//! Mirrors Python vLLM's architecture where `Worker` handles shared plumbing
//! and delegates model-specific execution to a model runner.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::info;
use vllm_core::scheduler::output::SchedulerOutput;
use vllm_cuda::driver;
use vllm_cuda::dtype::DType;
use vllm_cuda::model::llama::LlamaConfig;
use vllm_engine::executor::ModelRunnerOutput;
use vllm_executor::error::{ExecutorError, ExecutorResult};
use vllm_executor::gpu_worker_base;
use vllm_executor::input_batch::InputBatch;
use vllm_executor::worker::Worker;
use vllm_model::weight::HfModelConfig;
use vllm_tk::scheduler::PrefillSeq;
use vllm_tk::worker::{TkWorker, TkWorkerConfig};

/// Configuration for a TkWorkerAdapter.
#[derive(Debug, Clone)]
pub struct TkWorkerAdapterConfig {
    pub model_path: String,
    pub dtype: String,
    pub hf_token: Option<String>,
    pub device_id: i32,
    pub max_num_batched_tokens: usize,
    pub gpu_memory_utilization: f64,
}

/// Thin wrapper that adapts `TkWorker` to the `Worker` trait.
///
/// `TkWorker` handles device init, weight loading, KV cache, and kernel launch.
/// This adapter adds `InputBatch` for batch state and implements the `Worker`
/// trait's `execute_model` flow.
pub struct TkWorkerAdapter {
    config: TkWorkerAdapterConfig,
    inner: TkWorker,
    input_batch: InputBatch,
    hf_config: Option<HfModelConfig>,
    llama_config: Option<LlamaConfig>,
    model_dir: Option<PathBuf>,
}

impl TkWorkerAdapter {
    pub fn new(config: TkWorkerAdapterConfig) -> Self {
        Self {
            inner: TkWorker::new(TkWorkerConfig {
                model_path: config.model_path.clone(),
                dtype: config.dtype.clone(),
                hf_token: config.hf_token.clone(),
                device_id: config.device_id,
                max_num_batched_tokens: config.max_num_batched_tokens,
                gpu_memory_utilization: config.gpu_memory_utilization,
                sm_count: 0, // auto-detected in init_device
            }),
            config,
            input_batch: InputBatch::new(),
            hf_config: None,
            llama_config: None,
            model_dir: None,
        }
    }

    pub fn hf_config(&self) -> Option<&HfModelConfig> {
        self.hf_config.as_ref()
    }

    pub fn model_dir(&self) -> Option<&Path> {
        self.model_dir.as_deref()
    }

    pub fn resolved_dtype_elem_bytes(&self) -> usize {
        DType::BF16.size_bytes()
    }
}

impl Worker for TkWorkerAdapter {
    fn init_device(&mut self) -> ExecutorResult<()> {
        self.inner
            .init_device()
            .map_err(|e| ExecutorError::WorkerInit(format!("TK init_device: {e}")))
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        // Resolve model path using shared utility.
        let model_dir = gpu_worker_base::resolve_model_path(
            &self.config.model_path,
            self.config.hf_token.as_deref(),
            None, // TK doesn't support GGUF
        )?;

        // Parse config.json → LlamaConfig.
        let hf_config = HfModelConfig::from_dir(&model_dir)
            .map_err(|e| ExecutorError::WorkerInit(format!("parse config.json: {e}")))?;
        let llama_config = gpu_worker_base::llama_config_from_hf(&hf_config)?;

        info!(
            "TK backend: loading {} ({}L, h={}, kv_h={}, id={})",
            self.config.model_path,
            llama_config.num_hidden_layers,
            llama_config.hidden_size,
            llama_config.num_kv_heads,
            llama_config.intermediate_size,
        );

        // Delegate weight loading to TkWorker.
        self.inner
            .load_model(&model_dir, llama_config.clone())
            .map_err(|e| ExecutorError::WorkerInit(format!("TK load_model: {e}")))?;

        self.hf_config = Some(hf_config);
        self.llama_config = Some(llama_config);
        self.model_dir = Some(model_dir);
        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        let (free, total) = self
            .inner
            .query_gpu_memory()
            .map_err(|e| ExecutorError::WorkerInit(format!("TK memory query: {e}")))?;

        // Estimate peak activation memory for KV budget computation.
        let config = self.llama_config.as_ref().unwrap();
        let max_bs = self.config.max_num_batched_tokens;
        let peak_activation_estimate = {
            let hd = config.hidden_size;
            let id = config.intermediate_size;
            let vs = config.vocab_size;
            // 7 buffers of [max_bs, hd] + [max_bs, id] + [max_bs, vs], bf16
            (7 * max_bs * hd + max_bs * id + max_bs * vs) * DType::BF16.size_bytes()
        };

        let weights_and_overhead = total.saturating_sub(free);
        let available = gpu_worker_base::compute_available_kv_bytes(
            total,
            weights_and_overhead,
            peak_activation_estimate,
            self.config.gpu_memory_utilization,
        );

        info!(
            "TK memory: total={:.1} GiB, free={:.1} GiB, \
             est_activations={:.0} MiB, available_kv={:.1} GiB",
            total as f64 / 1_073_741_824.0,
            free as f64 / 1_073_741_824.0,
            peak_activation_estimate as f64 / 1_048_576.0,
            available as f64 / 1_073_741_824.0,
        );
        Ok(available)
    }

    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        self.inner
            .initialize_cache(num_gpu_blocks)
            .map_err(|e| ExecutorError::WorkerInit(format!("TK initialize_cache: {e}")))
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        tracing::info!(
            "TK execute_model: new_reqs={}, cached_reqs={}, finished={}",
            scheduler_output.scheduled_new_reqs.len(),
            scheduler_output.scheduled_cached_reqs.req_ids.len(),
            scheduler_output.finished_req_ids.len(),
        );

        self.inner
            .ensure_ctx()
            .map_err(|e| ExecutorError::WorkerExecution(format!("TK ensure_ctx: {e}")))?;

        // 1. Remove finished requests.
        self.input_batch
            .remove_finished(&scheduler_output.finished_req_ids);

        // 2. Add new requests.
        for new_req in &scheduler_output.scheduled_new_reqs {
            let num_tokens = scheduler_output
                .num_scheduled_tokens
                .get(&new_req.req_id)
                .copied()
                .unwrap_or(0);
            if num_tokens == 0 {
                continue;
            }
            let prompt_ids = new_req.prompt_token_ids.as_deref().unwrap_or(&[]);
            let start = new_req.num_computed_tokens as usize;
            let end = (start + num_tokens).min(prompt_ids.len());
            let tokens_to_use = &prompt_ids[start..end];

            let block_ids = new_req.block_ids.first().cloned().unwrap_or_default();
            self.input_batch.add_request(
                new_req.req_id.clone(),
                tokens_to_use,
                block_ids,
                new_req.num_computed_tokens,
            );
        }

        // 3. Update cached requests' block tables.
        for (i, req_id) in scheduler_output
            .scheduled_cached_reqs
            .req_ids
            .iter()
            .enumerate()
        {
            if let Some(Some(new_blocks)) =
                scheduler_output.scheduled_cached_reqs.new_block_ids.get(i)
                && let Some(group0) = new_blocks.first()
            {
                self.input_batch.update_blocks(req_id, group0.clone());
            }
        }

        // 4. Prepare inputs.
        let no_spec = HashMap::new();
        let prepared = self.input_batch.prepare_inputs(&no_spec);
        let batch_size = prepared.attn_meta.num_reqs;
        if batch_size == 0 {
            self.input_batch.reclaim_buffers(prepared);
            return Ok(ModelRunnerOutput::from_ordered(vec![], vec![]));
        }

        let has_prefill = prepared.attn_meta.is_prefill.iter().any(|&p| p);
        let has_decode = prepared.attn_meta.is_prefill.iter().any(|&p| !p);

        tracing::info!(
            "TK execute_model: batch_size={}, has_prefill={}, has_decode={}, total_tokens={}, req_ids={:?}",
            batch_size,
            has_prefill,
            has_decode,
            prepared.flat_token_ids.len(),
            prepared.attn_meta.req_ids,
        );

        let stream = self
            .inner
            .stream()
            .ok_or_else(|| ExecutorError::WorkerExecution("no stream".into()))?;

        // Determine total token count for upload.
        let total_tokens = prepared.flat_token_ids.len();

        // 5. Upload token IDs to GPU.
        let input_ids_tensor = {
            let alloc = self
                .inner
                .alloc()
                .ok_or_else(|| ExecutorError::WorkerExecution("no allocator".into()))?;
            let t = alloc.alloc_tensor(&[total_tokens], DType::U32);
            unsafe {
                driver::memcpy_htod_async(
                    t.as_mut_ptr::<u8>(),
                    prepared.flat_token_ids.as_ptr() as *const u8,
                    total_tokens * 4,
                    stream,
                )
                .map_err(|e| ExecutorError::WorkerExecution(format!("H2D token_ids: {e}")))?;
            }
            t
        };
        let input_ids_gpu = input_ids_tensor.as_gpu_tensor();

        let block_tables: Vec<&[usize]> = prepared
            .attn_meta
            .block_ids
            .iter()
            .map(|v| v.as_slice())
            .collect();

        if has_prefill && !has_decode {
            // Pure prefill batch.
            let prefill_seqs: Vec<PrefillSeq> = prepared
                .req_inputs
                .iter()
                .zip(prepared.attn_meta.is_prefill.iter())
                .filter(|&(_, &is_pf)| is_pf)
                .map(|(ri, _)| PrefillSeq {
                    chunk_len: ri.token_count,
                    extend_offset: prepared.attn_meta.tokens_before[prepared
                        .attn_meta
                        .req_ids
                        .iter()
                        .position(|id| id == &ri.req_id)
                        .unwrap()],
                })
                .collect();

            unsafe {
                self.inner.launch_prefill(
                    input_ids_gpu,
                    &prefill_seqs,
                    &block_tables,
                    &prepared.flat_positions,
                    &prepared.attn_meta.seq_lens,
                    total_tokens,
                );
            }
        } else if !has_prefill && has_decode {
            // Pure decode batch.
            tracing::info!(
                "TK decode: positions={:?}, seq_lens={:?}, block_tables={:?}, token_ids={:?}",
                prepared.flat_positions,
                prepared.attn_meta.seq_lens,
                block_tables,
                prepared.flat_token_ids
            );
            // Check for any async CUDA errors from previous operations.
            unsafe {
                if let Err(e) = driver::stream_synchronize(stream) {
                    return Err(ExecutorError::WorkerExecution(format!(
                        "CUDA async error before decode: {e}"
                    )));
                }
            }
            tracing::info!(
                "TK decode: CUDA sync clean, launching decode for batch_size={}",
                batch_size
            );
            unsafe {
                self.inner.upload_kv_metadata(
                    &block_tables,
                    &prepared.flat_positions,
                    &prepared.attn_meta.seq_lens,
                    batch_size,
                );
                tracing::info!(
                    "TK decode metadata: positions={:?}, seq_lens={:?}, block_table_lens={:?}",
                    &prepared.flat_positions[..batch_size.min(4)],
                    &prepared.attn_meta.seq_lens[..batch_size.min(4)],
                    block_tables
                        .iter()
                        .take(batch_size.min(4))
                        .map(|b| b.len())
                        .collect::<Vec<_>>(),
                );
                self.inner.launch_decode(input_ids_gpu, batch_size);
                tracing::info!("TK decode: kernel launched, syncing...");
                if let Err(e) = driver::stream_synchronize(stream) {
                    return Err(ExecutorError::WorkerExecution(format!(
                        "CUDA error after decode: {e}"
                    )));
                }
                tracing::info!("TK decode: kernel completed successfully");
            }
        } else {
            // Mixed prefill+decode: run prefill first, then decode.
            // For now, warn and skip — mixed batches require splitting inputs.
            tracing::warn!(
                "TK backend: mixed prefill+decode not yet supported, running decode only"
            );
            unsafe {
                self.inner.upload_kv_metadata(
                    &block_tables,
                    &prepared.flat_positions,
                    &prepared.attn_meta.seq_lens,
                    batch_size,
                );
                self.inner.launch_decode(input_ids_gpu, batch_size);
            }
        }

        // 8. Greedy argmax on logits.
        // For prefill: logits has total_tokens rows; we need the last token per seq.
        // For decode: logits has batch_size rows (one per request).
        let logits_full = self
            .inner
            .logits_tensor()
            .ok_or_else(|| ExecutorError::WorkerExecution("logits not available".into()))?;

        let stream = self.inner.stream().unwrap();
        let alloc = self.inner.alloc().unwrap();

        let host_tokens = if has_prefill && !has_decode {
            // Prefill: extract last-token logits per sequence, then argmax.
            // Last token indices: cumsum of token_counts - 1.
            let mut last_token_indices: Vec<usize> = Vec::with_capacity(batch_size);
            let mut offset = 0;
            for ri in &prepared.req_inputs {
                offset += ri.token_count;
                last_token_indices.push(offset - 1);
            }

            // Argmax each last-token's logit row individually.
            let mut tokens = vec![0u32; batch_size];
            for (i, &idx) in last_token_indices.iter().enumerate() {
                let logit_row = logits_full.narrow_dim0(idx, 1);
                let argmax_one =
                    unsafe { vllm_cuda::kernels::argmax_batched(logit_row, alloc, stream) };
                let mut tok = [0u32];
                unsafe {
                    driver::memcpy_dtoh_async(
                        tok.as_mut_ptr() as *mut u8,
                        argmax_one.as_ptr::<u8>(),
                        4,
                        stream,
                    )
                    .map_err(|e| ExecutorError::WorkerExecution(format!("D2H token: {e}")))?;
                    driver::stream_synchronize(stream)
                        .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;
                }
                tokens[i] = tok[0];
            }
            tokens
        } else {
            // Decode: straightforward argmax on all batch_size rows.
            let logits = logits_full.narrow_dim0(0, batch_size);
            let argmax_result =
                unsafe { vllm_cuda::kernels::argmax_batched(logits, alloc, stream) };
            let mut tokens = vec![0u32; batch_size];
            unsafe {
                driver::memcpy_dtoh_async(
                    tokens.as_mut_ptr() as *mut u8,
                    argmax_result.as_ptr::<u8>(),
                    batch_size * 4,
                    stream,
                )
                .map_err(|e| ExecutorError::WorkerExecution(format!("D2H tokens: {e}")))?;
                driver::stream_synchronize(stream)
                    .map_err(|e| ExecutorError::WorkerExecution(format!("sync: {e}")))?;
            }
            tokens
        };

        // 10. Commit step and build output.
        tracing::info!(
            "TK execute_model: host_tokens={:?}, has_prefill={}, has_decode={}",
            host_tokens,
            has_prefill,
            has_decode
        );
        let req_ids: Vec<String> = prepared.attn_meta.req_ids.clone();
        for (i, req_id) in req_ids.iter().enumerate() {
            let token_count = prepared.req_inputs[i].token_count;
            self.input_batch
                .commit_step(req_id, &[host_tokens[i]], token_count, false);
        }
        self.input_batch.reclaim_buffers(prepared);

        Ok(ModelRunnerOutput::from_ordered(req_ids, host_tokens))
    }

    fn compile_or_warm_up_model(&mut self) -> ExecutorResult<()> {
        // TK megakernel is a single persistent kernel — no graph capture needed.
        Ok(())
    }

    fn shutdown(&mut self) {
        info!("TK backend: shutting down");
    }

    fn rank(&self) -> usize {
        0
    }

    fn local_rank(&self) -> usize {
        0
    }

    fn is_driver_worker(&self) -> bool {
        true
    }

    fn architecture(&self) -> Option<String> {
        Some("LlamaForCausalLM".to_string())
    }
}
