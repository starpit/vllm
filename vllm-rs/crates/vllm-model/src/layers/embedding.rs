// SPDX-License-Identifier: Apache-2.0
//! Embedding layers.
//!
//! Port of: `vllm/model_executor/layers/vocab_parallel_embedding.py`

use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};

use crate::error::{ModelError, ModelResult};
use crate::process_group::ProcessGroup;
use crate::tensor;
use crate::weight::ModelWeights;

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Token embedding lookup table.
///
/// Stores a weight matrix of shape `[vocab_size, hidden_size]` and performs
/// lookup by token IDs.
pub struct Embedding {
    weight: Tensor,
}

impl Embedding {
    /// Create from an explicit weight tensor.
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    /// Load from model weights.
    ///
    /// Looks for `{prefix}.weight`.
    pub fn load(weights: &mut ModelWeights, prefix: &str, dtype: DType) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let weight = weights.take_cast(&weight_name, dtype)?;
        Ok(Self { weight })
    }

    /// Create with zeros (for testing).
    pub fn zeros(
        vocab_size: usize,
        hidden_size: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let weight = tensor::zeros(&[vocab_size, hidden_size], dtype, device)?;
        Ok(Self { weight })
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.weight.dim(0).unwrap_or(0)
    }

    /// Hidden dimension.
    pub fn hidden_size(&self) -> usize {
        self.weight.dim(1).unwrap_or(0)
    }

    /// Access the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Look up embeddings for the given token IDs.
    pub fn forward_ids(&self, ids: &Tensor) -> ModelResult<Tensor> {
        self.weight.embedding(ids).map_err(ModelError::Candle)
    }
}

impl Module for Embedding {
    fn forward(&self, ids: &Tensor) -> candle_core::Result<Tensor> {
        self.weight.embedding(ids)
    }
}

// ---------------------------------------------------------------------------
// VocabParallelEmbedding (for tensor parallelism)
// ---------------------------------------------------------------------------

/// Embedding that splits the vocabulary across tensor-parallel ranks.
///
/// Each rank holds `vocab_size / world_size` rows of the embedding table.
/// Input token IDs outside this rank's range produce zeros.
///
/// Port of: `vllm/model_executor/layers/vocab_parallel_embedding.py`
pub struct VocabParallelEmbedding {
    inner: Embedding,
    /// Start index for this rank's vocab shard.
    vocab_start: usize,
    /// End index (exclusive) for this rank's vocab shard.
    vocab_end: usize,
    /// NCCL process group for all-reduce (used when TP > 1).
    tp_group: Option<Arc<dyn ProcessGroup>>,
}

impl VocabParallelEmbedding {
    /// Create from an already-sharded embedding.
    pub fn new(embedding: Embedding, vocab_start: usize, vocab_end: usize) -> Self {
        Self {
            inner: embedding,
            vocab_start,
            vocab_end,
            tp_group: None,
        }
    }

    /// Load from model weights, sharding along the vocab dimension.
    pub fn load(
        weights: &mut ModelWeights,
        prefix: &str,
        dtype: DType,
        rank: usize,
        world_size: usize,
    ) -> ModelResult<Self> {
        let weight_name = format!("{}.weight", prefix);
        let full_weight = weights.take_cast(&weight_name, dtype)?;
        let vocab_size = full_weight.dim(0).map_err(ModelError::Candle)?;
        let shard = tensor::shard_tensor(&full_weight, 0, rank, world_size)?;

        let shard_size = vocab_size / world_size;
        let vocab_start = rank * shard_size;
        let vocab_end = vocab_start + shard_size;

        Ok(Self {
            inner: Embedding::new(shard),
            vocab_start,
            vocab_end,
            tp_group: None,
        })
    }

    /// This rank's vocab range.
    pub fn vocab_range(&self) -> (usize, usize) {
        (self.vocab_start, self.vocab_end)
    }

    /// Access the inner embedding.
    pub fn inner(&self) -> &Embedding {
        &self.inner
    }

    /// Set the NCCL process group for tensor-parallel communication.
    pub fn set_tp_group(&mut self, group: Arc<dyn ProcessGroup>) {
        self.tp_group = Some(group);
    }

    /// Forward pass: offset IDs to local range, lookup, zero out-of-range, all-reduce.
    ///
    /// For TP=1, this is equivalent to a normal embedding lookup.
    /// For TP>1, each rank looks up its shard and they all-reduce the results.
    pub fn forward(&self, ids: &Tensor) -> ModelResult<Tensor> {
        if self.tp_group.is_none() || self.tp_group.as_ref().is_some_and(|g| g.world_size() == 1) {
            // TP=1: just do normal lookup (IDs are within range).
            return self.inner.forward_ids(ids);
        }

        // TP>1: offset IDs to local shard, lookup, mask out-of-range, all-reduce.
        let shard_size = self.vocab_end - self.vocab_start;
        let device = ids.device();

        // Create offset: local_ids = global_ids - vocab_start
        let offset = Tensor::new(&[self.vocab_start as u32], device)
            .map_err(ModelError::Candle)?
            .broadcast_as(ids.shape())
            .map_err(ModelError::Candle)?;

        // Compute local IDs (may underflow for out-of-range — we'll mask those).
        // Use i64 to handle negative values from subtraction.
        let ids_i64 = ids.to_dtype(DType::I64).map_err(ModelError::Candle)?;
        let offset_i64 = offset.to_dtype(DType::I64).map_err(ModelError::Candle)?;
        let local_ids = ids_i64.sub(&offset_i64).map_err(ModelError::Candle)?;

        // Build mask: true where IDs are in this rank's range [0, shard_size).
        let zeros =
            Tensor::zeros(local_ids.shape(), DType::I64, device).map_err(ModelError::Candle)?;
        let shard_max = Tensor::new(&[shard_size as i64], device)
            .map_err(ModelError::Candle)?
            .broadcast_as(local_ids.shape())
            .map_err(ModelError::Candle)?;
        let in_range = local_ids
            .ge(&zeros)
            .map_err(ModelError::Candle)?
            .mul(&local_ids.lt(&shard_max).map_err(ModelError::Candle)?)
            .map_err(ModelError::Candle)?;

        // Clamp local IDs to valid range for lookup (out-of-range will be zeroed).
        let clamped = local_ids
            .clamp(0i64, (shard_size - 1) as i64)
            .map_err(ModelError::Candle)?
            .to_dtype(DType::U32)
            .map_err(ModelError::Candle)?;

        // Lookup embeddings.
        let embeddings = self.inner.forward_ids(&clamped)?;

        // Zero out embeddings for out-of-range IDs.
        let mask_f = in_range
            .to_dtype(embeddings.dtype())
            .map_err(ModelError::Candle)?
            .unsqueeze(candle_core::D::Minus1)
            .map_err(ModelError::Candle)?;
        let masked = embeddings.mul(&mask_f).map_err(ModelError::Candle)?;

        // All-reduce across ranks to combine shards.
        let group = self.tp_group.as_ref().unwrap();
        group.all_reduce(&masked).map_err(ModelError::Candle)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup() {
        // Vocab of 4, hidden_size of 3
        let weight = Tensor::new(
            &[
                [1.0f32, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            &Device::Cpu,
        )
        .unwrap();

        let emb = Embedding::new(weight);
        assert_eq!(emb.vocab_size(), 4);
        assert_eq!(emb.hidden_size(), 3);

        let ids = Tensor::new(&[0u32, 2, 3], &Device::Cpu).unwrap();
        let out = emb.forward(&ids).unwrap();
        assert_eq!(out.dims(), &[3, 3]);

        let vals = out.to_vec2::<f32>().unwrap();
        assert_eq!(vals[0], vec![1.0, 0.0, 0.0]); // token 0
        assert_eq!(vals[1], vec![0.0, 0.0, 1.0]); // token 2
        assert_eq!(vals[2], vec![1.0, 1.0, 1.0]); // token 3
    }

    #[test]
    fn test_embedding_batch() {
        let emb = Embedding::zeros(100, 64, DType::F32, &Device::Cpu).unwrap();
        let ids = Tensor::new(&[1u32, 5, 99], &Device::Cpu).unwrap();
        let out = emb.forward(&ids).unwrap();
        assert_eq!(out.dims(), &[3, 64]);
    }

    #[test]
    fn test_embedding_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        // 4x2 embedding
        let w_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        crate::weight::tests_helper::create_safetensors_file(
            &path,
            &[("embed.weight", vec![4, 2], DType::F32, &w_data)],
        );

        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        let emb = Embedding::load(&mut weights, "embed", DType::F32).unwrap();
        assert_eq!(emb.vocab_size(), 4);
        assert_eq!(emb.hidden_size(), 2);

        let ids = Tensor::new(&[0u32, 3], &Device::Cpu).unwrap();
        let out = emb.forward(&ids).unwrap();
        let vals = out.to_vec2::<f32>().unwrap();
        assert_eq!(vals[0], vec![1.0, 2.0]); // token 0
        assert_eq!(vals[1], vec![7.0, 8.0]); // token 3
    }

    #[test]
    fn test_vocab_parallel_embedding() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        // 4x2 embedding, split across 2 ranks
        let w_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        crate::weight::tests_helper::create_safetensors_file(
            &path,
            &[("embed.weight", vec![4, 2], DType::F32, &w_data)],
        );

        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();

        let emb0 = VocabParallelEmbedding::load(&mut weights, "embed", DType::F32, 0, 2).unwrap();
        assert_eq!(emb0.vocab_range(), (0, 2));
        assert_eq!(emb0.inner().vocab_size(), 2);

        // Reload for rank 1 (take consumed rank 0's tensor).
        let mut weights = ModelWeights::from_single_file(&path, &Device::Cpu).unwrap();
        let emb1 = VocabParallelEmbedding::load(&mut weights, "embed", DType::F32, 1, 2).unwrap();
        assert_eq!(emb1.vocab_range(), (2, 4));
        assert_eq!(emb1.inner().vocab_size(), 2);
    }
}
