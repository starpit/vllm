// SPDX-License-Identifier: Apache-2.0
//! Model architecture implementations.
//!
//! This crate contains:
//! - **`Model` trait** — the core abstraction for all model architectures
//! - **`Sampler`** — greedy and random sampling from logits
//! - **`ModelRegistry`** — maps HuggingFace architecture names to model constructors
//! - **Model implementations** — LLaMA, Mistral, Qwen2, etc.
//!
//! Port of: `vllm/model_executor/models/`

pub mod attention;
pub mod attention_metadata;
pub mod awq_llama;
pub mod bnb_llama;
pub mod commandr;
pub mod deepseek_v2;
pub mod embedding;
pub mod gemma2;
pub mod gemma3;
pub mod gemma3_mm;
pub mod gptq_llama;
#[cfg(feature = "guided-decoding")]
pub mod grammar;
pub mod granite;
pub mod kv_block_pool;
pub mod llama;
pub mod marlin_linear;
pub mod mixtral;
pub mod ops;
pub mod quantized_gemma3;
pub mod quantized_granite;
pub mod quantized_llama;
pub mod quantized_qwen2;
pub mod quantized_qwen3_next;
pub mod qwen2;
pub mod qwen2_vl;
pub mod qwen3_moe;
pub mod qwen3_next;
pub mod registry;
pub mod sampler;
pub mod siglip;

use candle_core::Tensor;
use vllm_model::ModelResult;

// Re-export for convenience.
pub use attention_metadata::AttentionMetadata;
pub use kv_block_pool::KvBlockPool;
pub use registry::ModelRegistry;
pub use sampler::Sampler;

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// Per-GDN-layer recurrent state: `(conv_state, ssm_state)`.
///
/// - `conv_state`: `[kernel_size-1, conv_dim]`
/// - `ssm_state`: `[num_v_heads, head_v_dim, head_k_dim]`
///
/// `None` entries indicate a layer with no state yet (first call).
/// The vector length equals the number of recurrent (GDN) layers.
pub type RecurrentState = Vec<Option<(Tensor, Tensor)>>;

/// Per-layer KV cache entry: `(key, value)` tensors.
///
/// Keys and values have shape `[cached_seq_len, num_kv_heads, head_dim]`.
/// On prefill the cache is populated with the full prompt K/V.
/// On decode the new token's K/V is concatenated to the existing cache.
pub type LayerKvCache = (Tensor, Tensor);

/// KV cache for a model: one `Option<(key, value)>` per layer.
///
/// - `None` entries indicate a layer with no cached K/V yet (first call).
/// - `Some((k, v))` entries hold the accumulated K/V for that layer.
///
/// The vector length must equal the number of layers in the model.
pub type KvCache = Vec<Option<LayerKvCache>>;

// ---------------------------------------------------------------------------
// KvCacheStorage — unified cache abstraction
// ---------------------------------------------------------------------------

/// A deferred write operation for paged KV cache.
///
/// Attention layers produce these during forward; they are flushed to the
/// block pool after the forward pass completes.
pub enum PendingWrite {
    /// Full scatter (prefill or legacy decode): write all new tokens from
    /// a contiguous K/V tensor.
    FullScatter {
        layer: usize,
        k_full: Tensor,
        v_full: Tensor,
    },
    /// Single-token write (paged decode): write one new K/V token.
    /// The tensors have shape `[num_kv_heads, head_dim]`.
    NewToken {
        layer: usize,
        k_token: Tensor,
        v_token: Tensor,
    },
}

/// References to paged KV block tensors for direct-read attention.
///
/// Returned by `LayerKvHandle::paged_block_refs()` so that attention can
/// read K/V directly from blocks without gathering into a contiguous tensor.
pub struct PagedKvBlockRefs {
    /// K block tensors in sequence order, each `[block_size, num_kv_heads, head_dim]`.
    pub k_blocks: Vec<Tensor>,
    /// V block tensors in sequence order, each `[block_size, num_kv_heads, head_dim]`.
    pub v_blocks: Vec<Tensor>,
    /// Total number of cached tokens across all blocks.
    pub num_tokens: usize,
    /// Number of token slots per block.
    pub block_size: usize,
}

/// KV cache storage passed to `Model::forward`.
///
/// Abstracts over legacy contiguous per-request caches and the paged block pool.
/// Attention layers extract per-layer handles via `layer_handle()`.
///
/// For the paged variant, gathers happen lazily per-layer inside forward,
/// but scatters are deferred to a batch flush after forward completes.
/// Call `flush()` after forward to write new tokens to the block pool.
pub enum KvCacheStorage<'a> {
    /// Legacy contiguous per-request cache.
    Contiguous(&'a mut KvCache),
    /// Paged: reads K/V from the block pool during forward, defers writes
    /// to `pending_writes` which are flushed after forward.
    Paged {
        pool: &'a mut KvBlockPool,
        block_ids: &'a [usize],
        tokens_before: usize,
        /// Deferred write operations flushed after forward.
        pending_writes: Vec<PendingWrite>,
    },
}

impl<'a> KvCacheStorage<'a> {
    /// Create a new paged storage (convenience constructor).
    pub fn paged(pool: &'a mut KvBlockPool, block_ids: &'a [usize], tokens_before: usize) -> Self {
        KvCacheStorage::Paged {
            pool,
            block_ids,
            tokens_before,
            pending_writes: Vec::new(),
        }
    }

    /// Extract a per-layer handle for use in an attention layer.
    pub fn layer_handle(&mut self, layer: usize) -> LayerKvHandle<'_> {
        match self {
            KvCacheStorage::Contiguous(cache) => LayerKvHandle::Contiguous(&mut cache[layer]),
            KvCacheStorage::Paged {
                pool,
                block_ids,
                tokens_before,
                pending_writes,
            } => LayerKvHandle::Paged {
                pool,
                block_ids,
                tokens_before: *tokens_before,
                layer,
                pending_writes,
            },
        }
    }

    /// Flush deferred write operations to the block pool.
    ///
    /// Must be called after `Model::forward()` for paged storage.
    /// No-op for contiguous storage.
    pub fn flush(&mut self) -> ModelResult<()> {
        if let KvCacheStorage::Paged {
            pool,
            block_ids,
            tokens_before,
            pending_writes,
        } = self
        {
            for pw in pending_writes.drain(..) {
                match pw {
                    PendingWrite::FullScatter {
                        layer,
                        k_full,
                        v_full,
                    } => {
                        pool.scatter_new_kv(layer, block_ids, *tokens_before, &k_full, &v_full)?;
                    }
                    PendingWrite::NewToken {
                        layer,
                        k_token,
                        v_token,
                    } => {
                        // Write the single new token to the next slot.
                        let global_pos = *tokens_before;
                        let block_offset = global_pos / pool.block_size();
                        let position_in_block = global_pos % pool.block_size();
                        if block_offset < block_ids.len() {
                            let bid = block_ids[block_offset];
                            pool.write_kv(layer, bid, position_in_block, &k_token, &v_token)?;
                            // Update tokens_in_block for the affected block.
                            let new_fill = position_in_block + 1;
                            if new_fill > pool.tokens_stored(bid) {
                                pool.set_tokens_stored(bid, new_fill);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LayerKvHandle — per-layer cache handle for attention
// ---------------------------------------------------------------------------

/// Per-layer handle extracted from `KvCacheStorage`.
///
/// Attention layers receive this instead of `Option<&mut Option<(Tensor, Tensor)>>`.
/// For the paged variant, `take_cached()` gathers from the block pool and
/// `store()` defers the scatter to be flushed after forward completes.
pub enum LayerKvHandle<'a> {
    /// Legacy contiguous slot.
    Contiguous(&'a mut Option<LayerKvCache>),
    /// Paged: reads from pool, defers writes to pending_writes.
    Paged {
        pool: &'a KvBlockPool,
        block_ids: &'a [usize],
        tokens_before: usize,
        layer: usize,
        pending_writes: &'a mut Vec<PendingWrite>,
    },
}

impl LayerKvHandle<'_> {
    /// Retrieve previously cached K/V for this layer.
    ///
    /// - **Contiguous**: takes the cached tensors (leaving `None`).
    /// - **Paged**: gathers from the block pool, or returns `None` if
    ///   `tokens_before == 0` (first call / prefill).
    pub fn take_cached(&mut self) -> ModelResult<Option<(Tensor, Tensor)>> {
        match self {
            LayerKvHandle::Contiguous(slot) => Ok(slot.take()),
            LayerKvHandle::Paged {
                pool,
                block_ids,
                tokens_before,
                layer,
                ..
            } => {
                if *tokens_before == 0 {
                    Ok(None)
                } else {
                    let (k, v) = pool.gather_kv(*layer, block_ids, *tokens_before)?;
                    Ok(Some((k, v)))
                }
            }
        }
    }

    /// Get read-only references to paged KV block tensors for direct-read attention.
    ///
    /// Returns `Some(PagedKvBlockRefs)` when this is a paged handle with
    /// `tokens_before > 0` (i.e., there are cached tokens to read).
    /// Returns `None` for contiguous handles or when there are no cached tokens.
    pub fn paged_block_refs(&self) -> Option<PagedKvBlockRefs> {
        match self {
            LayerKvHandle::Contiguous(_) => None,
            LayerKvHandle::Paged {
                pool,
                block_ids,
                tokens_before,
                layer,
                ..
            } => {
                if *tokens_before == 0 {
                    return None;
                }
                let block_size = pool.block_size();
                let mut k_blocks = Vec::new();
                let mut v_blocks = Vec::new();
                let mut remaining = *tokens_before;
                for &bid in block_ids.iter() {
                    if remaining == 0 {
                        break;
                    }
                    k_blocks.push(pool.k_block(*layer, bid));
                    v_blocks.push(pool.v_block(*layer, bid));
                    remaining = remaining.saturating_sub(block_size);
                }
                Some(PagedKvBlockRefs {
                    k_blocks,
                    v_blocks,
                    num_tokens: *tokens_before,
                    block_size,
                })
            }
        }
    }

    /// Store the full K/V tensors after attention (legacy/prefill path).
    ///
    /// - **Contiguous**: stores the tensors in the cache slot immediately.
    /// - **Paged**: defers the scatter to `pending_writes`. Call
    ///   `KvCacheStorage::flush()` after forward to write to the block pool.
    pub fn store(&mut self, k_full: Tensor, v_full: Tensor) -> ModelResult<()> {
        match self {
            LayerKvHandle::Contiguous(slot) => {
                **slot = Some((k_full, v_full));
                Ok(())
            }
            LayerKvHandle::Paged {
                layer,
                pending_writes,
                ..
            } => {
                pending_writes.push(PendingWrite::FullScatter {
                    layer: *layer,
                    k_full,
                    v_full,
                });
                Ok(())
            }
        }
    }

    /// Store a single new K/V token (paged decode path).
    ///
    /// The tensors should have shape `[num_kv_heads, head_dim]`.
    /// For contiguous handles, this falls back to the full store path
    /// (caller must provide the full concatenated tensors via `store` instead).
    pub fn store_new_token(&mut self, k_token: Tensor, v_token: Tensor) -> ModelResult<()> {
        match self {
            LayerKvHandle::Contiguous(_) => Err(vllm_model::error::ModelError::Other(
                "store_new_token not supported for contiguous cache".into(),
            )),
            LayerKvHandle::Paged {
                layer,
                pending_writes,
                ..
            } => {
                pending_writes.push(PendingWrite::NewToken {
                    layer: *layer,
                    k_token,
                    v_token,
                });
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BatchedKvCacheStorage — multi-request paged cache for batched forward
// ---------------------------------------------------------------------------

/// KV cache storage for a batch of requests using the paged block pool.
///
/// Holds per-request block IDs, tokens-before counts, and per-request pending
/// write vectors. Individual `LayerKvHandle::Paged` handles are extracted per
/// (request, layer) pair via `request_layer_handle()`, reusing the existing
/// attention code unchanged.
///
/// After `forward_batch()` completes, call `flush_all()` to write deferred
/// scatters to the block pool.
pub struct BatchedKvCacheStorage<'a> {
    pool: &'a mut KvBlockPool,
    block_ids: Vec<Vec<usize>>,
    tokens_before: Vec<usize>,
    /// Per-request pending writes. Each inner vec collects writes for one request.
    pending_writes: Vec<Vec<PendingWrite>>,
}

impl<'a> BatchedKvCacheStorage<'a> {
    /// Create batched storage for multiple requests.
    pub fn new(
        pool: &'a mut KvBlockPool,
        block_ids: Vec<Vec<usize>>,
        tokens_before: Vec<usize>,
    ) -> Self {
        let num_reqs = block_ids.len();
        Self {
            pool,
            block_ids,
            tokens_before,
            pending_writes: (0..num_reqs).map(|_| Vec::new()).collect(),
        }
    }

    /// Extract a per-layer handle for one request. Returns a standard
    /// `LayerKvHandle::Paged` so existing attention code works unchanged.
    ///
    /// Handles are valid one at a time — drop the previous handle before
    /// requesting the next (enforced by the borrow checker).
    pub fn request_layer_handle(&mut self, req_idx: usize, layer: usize) -> LayerKvHandle<'_> {
        LayerKvHandle::Paged {
            pool: self.pool,
            block_ids: &self.block_ids[req_idx],
            tokens_before: self.tokens_before[req_idx],
            layer,
            pending_writes: &mut self.pending_writes[req_idx],
        }
    }

    /// Create a per-request `KvCacheStorage::Paged` for use in `forward()`.
    ///
    /// Borrows `self` for the duration of the returned storage. Drop the
    /// storage (and call its `flush()`) before requesting the next request.
    pub fn request_storage(&mut self, req_idx: usize) -> KvCacheStorage<'_> {
        KvCacheStorage::paged(
            self.pool,
            &self.block_ids[req_idx],
            self.tokens_before[req_idx],
        )
    }

    /// Flush all deferred writes to the block pool.
    pub fn flush_all(&mut self) -> ModelResult<()> {
        for req_idx in 0..self.block_ids.len() {
            let block_ids = &self.block_ids[req_idx];
            let tokens_before = self.tokens_before[req_idx];
            for pw in self.pending_writes[req_idx].drain(..) {
                match pw {
                    PendingWrite::FullScatter {
                        layer,
                        k_full,
                        v_full,
                    } => {
                        self.pool.scatter_new_kv(
                            layer,
                            block_ids,
                            tokens_before,
                            &k_full,
                            &v_full,
                        )?;
                    }
                    PendingWrite::NewToken {
                        layer,
                        k_token,
                        v_token,
                    } => {
                        let global_pos = tokens_before;
                        let block_offset = global_pos / self.pool.block_size();
                        let position_in_block = global_pos % self.pool.block_size();
                        if block_offset < block_ids.len() {
                            let bid = block_ids[block_offset];
                            self.pool.write_kv(
                                layer,
                                bid,
                                position_in_block,
                                &k_token,
                                &v_token,
                            )?;
                            let new_fill = position_in_block + 1;
                            if new_fill > self.pool.tokens_stored(bid) {
                                self.pool.set_tokens_stored(bid, new_fill);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Model trait
// ---------------------------------------------------------------------------

/// Core trait for all model architectures.
///
/// A `Model` takes token IDs and positions and produces logits over the
/// vocabulary. This corresponds to the full model pipeline:
/// embedding → transformer layers → lm_head → logits.
///
/// Port of: `vllm/model_executor/models/interfaces_base.py::VllmModelForTextGeneration`
pub trait Model: Send {
    /// Inject LoRA adapter weights into this model.
    ///
    /// Walks the model's linear layers and attaches LoRA A/B pairs to those
    /// whose names match the adapter's `target_modules`.
    ///
    /// Default implementation: no-op (model doesn't support LoRA).
    fn inject_lora(&mut self, _adapter: &vllm_model::lora::LoraAdapter) -> ModelResult<()> {
        Ok(())
    }

    /// Run the model forward pass (single request).
    ///
    /// * `input_ids` — token IDs, shape `[num_tokens]`
    /// * `positions` — position indices, shape `[num_tokens]`
    /// * `kv_cache` — optional KV cache storage. When `Some`, the model
    ///   reads cached K/V from previous steps and writes updated K/V back.
    ///   When `None`, no caching is performed (full recompute).
    ///
    /// Returns logits of shape `[num_tokens, vocab_size]`.
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor>;

    /// Number of transformer layers in this model.
    ///
    /// Used by callers to create an appropriately-sized KV cache.
    /// For hybrid models (e.g., Qwen3-Next), returns the number of layers
    /// that use KV cache (full attention layers only).
    fn num_layers(&self) -> usize;

    /// Reset recurrent state for hybrid models (e.g., GDN linear attention).
    ///
    /// Called before each request's forward pass so that recurrent layers
    /// (conv state, SSM state) start fresh. Default: no-op for standard
    /// transformer models.
    fn reset_recurrent_state(&self) {}

    /// Number of recurrent (GDN) layers in this model.
    ///
    /// Returns 0 for standard transformer models. For hybrid models,
    /// returns the count of GDN / linear-attention layers whose state
    /// must be tracked per-request.
    fn num_recurrent_layers(&self) -> usize {
        0
    }

    /// Extract recurrent state from all GDN layers.
    ///
    /// Returns a vector of `Option<(conv_state, ssm_state)>` with one entry
    /// per recurrent layer. The model's internal state is taken (moved out),
    /// leaving `None` in the RefCells.
    fn extract_recurrent_state(&self) -> RecurrentState {
        vec![]
    }

    /// Inject previously-saved recurrent state into all GDN layers.
    ///
    /// The slice must have length `num_recurrent_layers()`. Each entry
    /// replaces the corresponding GDN layer's internal state.
    fn inject_recurrent_state(&self, _state: &[Option<(Tensor, Tensor)>]) {}

    /// Run the model backbone and return hidden states (before lm_head).
    ///
    /// Used for embedding: a single prefill pass with no KV cache, returning
    /// the transformer output before the language model head projection.
    ///
    /// Default implementation returns an error for models that haven't
    /// overridden this method.
    fn hidden_states(&self, _input_ids: &Tensor, _positions: &Tensor) -> ModelResult<Tensor> {
        Err(vllm_model::error::ModelError::Other(
            "hidden_states not supported by this model".into(),
        ))
    }

    /// Run the model forward pass from pre-computed embeddings (for VLM models).
    ///
    /// * `inputs_embeds` — merged text+image embeddings, shape `[num_tokens, hidden_size]`
    /// * `positions` — position indices, shape `[num_tokens]`
    /// * `kv_cache` — optional KV cache storage.
    ///
    /// Returns logits of shape `[num_tokens, vocab_size]`.
    ///
    /// Default implementation returns an error. VLM-wrapped text models override this.
    fn forward_embeds(
        &self,
        _inputs_embeds: &Tensor,
        _positions: &Tensor,
        _kv_cache: Option<&mut KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        Err(vllm_model::error::ModelError::Other(
            "forward_embeds not supported by this model".into(),
        ))
    }

    /// Inject a tensor-parallel process group into this model's parallel layers.
    ///
    /// Walks the model's `RowParallelLinear` layers and sets their process group
    /// so that NCCL all-reduce is performed after each row-parallel matmul.
    ///
    /// Default implementation: no-op (model doesn't use tensor parallelism).
    fn inject_tp_group(
        &mut self,
        _group: std::sync::Arc<dyn vllm_model::process_group::ProcessGroup>,
    ) -> ModelResult<()> {
        Ok(())
    }

    /// Provide multimodal data (images) for the next forward pass.
    ///
    /// VLM models store this internally and consume it during `forward()`.
    /// Text-only models ignore this (default no-op).
    fn set_mm_data(&mut self, _mm_data: Option<vllm_common::MultimodalData>) {}

    /// Batched forward pass across multiple requests.
    ///
    /// * `input_ids` — flat token IDs from all requests, shape `[total_tokens]`
    /// * `positions` — flat position indices, shape `[total_tokens]`
    /// * `attn_meta` — per-request slicing info for the attention layers
    /// * `kv_storage` — batched paged KV cache for all requests
    ///
    /// Returns logits of shape `[num_reqs, vocab_size]` — one row per request
    /// (the last token's logits for prefill, the single token's logits for decode).
    ///
    /// The default implementation falls back to per-request `forward()` calls.
    /// Architectures opt into real batching by overriding this method to batch
    /// embedding, projections, norms, and MLP while keeping attention per-request.
    fn forward_batch(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        attn_meta: &AttentionMetadata,
        kv_storage: &mut BatchedKvCacheStorage<'_>,
    ) -> ModelResult<Tensor> {
        use vllm_model::error::ModelError;
        // Default: per-request loop calling forward() and flushing immediately.
        // Only keep the last token's logits per request to save memory.
        let mut logits_parts = Vec::with_capacity(attn_meta.num_reqs);
        for req_idx in 0..attn_meta.num_reqs {
            let (start, q_len) = attn_meta.request_slice(req_idx);
            let req_ids = input_ids
                .narrow(0, start, q_len)
                .map_err(ModelError::Candle)?;
            let req_pos = positions
                .narrow(0, start, q_len)
                .map_err(ModelError::Candle)?;

            let mut storage = kv_storage.request_storage(req_idx);
            let logits = self.forward(&req_ids, &req_pos, Some(&mut storage))?;
            storage.flush()?;
            // Take only the last row (the token we'll sample from).
            let last_logits = if q_len > 1 {
                logits.narrow(0, q_len - 1, 1).map_err(ModelError::Candle)?
            } else {
                logits
            };
            logits_parts.push(last_logits);
        }
        Tensor::cat(&logits_parts, 0).map_err(ModelError::Candle)
    }
}

/// Factory function type for constructing a model from weights and config.
///
/// Used by `ModelRegistry` to create model instances from architecture names.
/// `rank` and `world_size` control tensor-parallel weight sharding (0/1 for single-GPU).
pub type ModelFactory = fn(
    weights: &mut vllm_model::weight::ModelWeights,
    config: &vllm_model::weight::HfModelConfig,
    dtype: candle_core::DType,
    device: &candle_core::Device,
    rank: usize,
    world_size: usize,
) -> ModelResult<Box<dyn Model>>;
