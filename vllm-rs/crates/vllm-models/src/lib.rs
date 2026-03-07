// SPDX-License-Identifier: Apache-2.0
//! Shared model infrastructure: sampler, attention metadata, KV cache, embeddings.
//!
//! Model architecture implementations have moved to the vllm-cuda backend.
//! This crate provides types shared across backends (CUDA, MLX).

pub mod attention_metadata;
pub mod embedding;
#[cfg(feature = "guided-decoding")]
pub mod grammar;
pub mod kv_block_pool;
pub mod sampler;

use candle_core::Tensor;
use vllm_model::ModelResult;

// Re-export for convenience.
pub use attention_metadata::AttentionMetadata;
pub use kv_block_pool::KvBlockPool;
pub use sampler::Sampler;

// ---------------------------------------------------------------------------
// KV Cache
// ---------------------------------------------------------------------------

/// Per-layer KV cache entry: `(key, value)` tensors.
pub type LayerKvCache = (Tensor, Tensor);

/// KV cache for a model: one `Option<(key, value)>` per layer.
pub type KvCache = Vec<Option<LayerKvCache>>;

// ---------------------------------------------------------------------------
// KvCacheStorage — unified cache abstraction
// ---------------------------------------------------------------------------

/// A deferred write operation for paged KV cache.
pub enum PendingWrite {
    FullScatter {
        layer: usize,
        k_full: Tensor,
        v_full: Tensor,
    },
    NewToken {
        layer: usize,
        k_token: Tensor,
        v_token: Tensor,
    },
}

pub struct PagedKvBlockRefs {
    pub k_blocks: Vec<Tensor>,
    pub v_blocks: Vec<Tensor>,
    pub num_tokens: usize,
    pub block_size: usize,
}

pub enum KvCacheStorage<'a> {
    Contiguous(&'a mut KvCache),
    Paged {
        pool: &'a mut KvBlockPool,
        block_ids: &'a [usize],
        tokens_before: usize,
        pending_writes: Vec<PendingWrite>,
    },
}

impl<'a> KvCacheStorage<'a> {
    pub fn paged(pool: &'a mut KvBlockPool, block_ids: &'a [usize], tokens_before: usize) -> Self {
        KvCacheStorage::Paged {
            pool,
            block_ids,
            tokens_before,
            pending_writes: Vec::new(),
        }
    }

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
                        let global_pos = *tokens_before;
                        let block_offset = global_pos / pool.block_size();
                        let position_in_block = global_pos % pool.block_size();
                        if block_offset < block_ids.len() {
                            let bid = block_ids[block_offset];
                            pool.write_kv(layer, bid, position_in_block, &k_token, &v_token)?;
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

pub enum LayerKvHandle<'a> {
    Contiguous(&'a mut Option<LayerKvCache>),
    Paged {
        pool: &'a KvBlockPool,
        block_ids: &'a [usize],
        tokens_before: usize,
        layer: usize,
        pending_writes: &'a mut Vec<PendingWrite>,
    },
}

impl LayerKvHandle<'_> {
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

pub struct BatchedKvCacheStorage<'a> {
    pub pool: &'a mut KvBlockPool,
    block_ids: Vec<Vec<usize>>,
    tokens_before: Vec<usize>,
    pending_writes: Vec<Vec<PendingWrite>>,
}

impl<'a> BatchedKvCacheStorage<'a> {
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

    pub fn request_layer_handle(&mut self, req_idx: usize, layer: usize) -> LayerKvHandle<'_> {
        LayerKvHandle::Paged {
            pool: self.pool,
            block_ids: &self.block_ids[req_idx],
            tokens_before: self.tokens_before[req_idx],
            layer,
            pending_writes: &mut self.pending_writes[req_idx],
        }
    }

    pub fn request_storage(&mut self, req_idx: usize) -> KvCacheStorage<'_> {
        KvCacheStorage::paged(
            self.pool,
            &self.block_ids[req_idx],
            self.tokens_before[req_idx],
        )
    }

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

/// Core trait for all model architectures.
pub trait Model: Send {
    fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_cache: Option<&mut KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor>;

    fn num_layers(&self) -> usize;

    fn inject_lora(&mut self, _adapter: &vllm_model::lora::LoraAdapter) -> ModelResult<()> {
        Ok(())
    }

    fn reset_recurrent_state(&self) {}
    fn num_recurrent_layers(&self) -> usize { 0 }
    fn extract_recurrent_state(&self) -> Vec<Option<(Tensor, Tensor)>> { vec![] }
    fn inject_recurrent_state(&self, _state: &[Option<(Tensor, Tensor)>]) {}
    fn hidden_states(&self, _input_ids: &Tensor, _positions: &Tensor) -> ModelResult<Tensor> {
        Err(vllm_model::error::ModelError::Other("hidden_states not supported".into()))
    }
    fn forward_embeds(
        &self,
        _inputs_embeds: &Tensor,
        _positions: &Tensor,
        _kv_cache: Option<&mut KvCacheStorage<'_>>,
    ) -> ModelResult<Tensor> {
        Err(vllm_model::error::ModelError::Other("forward_embeds not supported".into()))
    }
    fn inject_tp_group(
        &mut self,
        _group: std::sync::Arc<dyn vllm_model::process_group::ProcessGroup>,
    ) -> ModelResult<()> {
        Ok(())
    }
    fn set_mm_data(&mut self, _mm_data: Option<vllm_common::MultimodalData>) {}
    fn forward_batch(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        attn_meta: &AttentionMetadata,
        kv_storage: &mut BatchedKvCacheStorage<'_>,
    ) -> ModelResult<Tensor> {
        use vllm_model::error::ModelError;
        let mut logits_parts = Vec::with_capacity(attn_meta.num_reqs);
        for req_idx in 0..attn_meta.num_reqs {
            let (start, q_len) = attn_meta.request_slice(req_idx);
            let req_ids = input_ids.narrow(0, start, q_len).map_err(ModelError::Candle)?;
            let req_pos = positions.narrow(0, start, q_len).map_err(ModelError::Candle)?;
            let mut storage = kv_storage.request_storage(req_idx);
            let logits = self.forward(&req_ids, &req_pos, Some(&mut storage))?;
            storage.flush()?;
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
pub type ModelFactory = fn(
    weights: &mut vllm_model::weight::ModelWeights,
    config: &vllm_model::weight::HfModelConfig,
    dtype: candle_core::DType,
    device: &candle_core::Device,
    rank: usize,
    world_size: usize,
) -> ModelResult<Box<dyn Model>>;
