// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Pre-allocated pool of KV cache block tensors for paged attention.
//!
//! `KvBlockPool` maps the scheduler's arena block indices to fixed-size
//! GPU/CPU tensors. Instead of per-request contiguous KV caches, all
//! requests share a global pool of blocks. This enables:
//!
//! - **Prefix sharing**: Two requests with the same prompt prefix share
//!   the same KV blocks (the scheduler's `BlockPool` tracks ref-counts).
//! - **O(1) decode writes**: Only the current block's next slot is written,
//!   instead of copying the entire cache.
//!
//! Each block stores `block_size` token slots of K and V tensors for every
//! layer. The shape of each block tensor is
//! `[block_size, num_kv_heads, head_dim]`.

use candle_core::{DType, Device, Tensor};
use vllm_model::ModelResult;
use vllm_model::error::ModelError;

/// Pre-allocated pool of KV cache block tensors.
///
/// Indexed by the same arena block indices used by `BlockPool` in
/// `vllm-core`. The CandleWorker translates between scheduler block IDs
/// and tensor storage via this pool.
pub struct KvBlockPool {
    /// Per-layer Vec of K block tensors.
    /// `k_blocks[layer][block_idx]` has shape `[block_size, num_kv_heads, head_dim]`.
    k_blocks: Vec<Vec<Tensor>>,
    /// Per-layer Vec of V block tensors.
    /// `v_blocks[layer][block_idx]` has shape `[block_size, num_kv_heads, head_dim]`.
    v_blocks: Vec<Vec<Tensor>>,
    /// Number of token slots currently filled in each block.
    /// `tokens_in_block[block_idx]` is in `0..=block_size`.
    tokens_in_block: Vec<usize>,
    block_size: usize,
    num_layers: usize,
    num_blocks: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
    device: Device,
}

impl KvBlockPool {
    /// Allocate a new pool with `num_blocks` pre-zeroed KV block tensors.
    pub fn new(
        num_blocks: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        dtype: DType,
        device: &Device,
    ) -> ModelResult<Self> {
        let mut k_blocks = Vec::with_capacity(num_layers);
        let mut v_blocks = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let mut layer_k = Vec::with_capacity(num_blocks);
            let mut layer_v = Vec::with_capacity(num_blocks);
            for _ in 0..num_blocks {
                layer_k.push(
                    Tensor::zeros((block_size, num_kv_heads, head_dim), dtype, device)
                        .map_err(ModelError::Candle)?,
                );
                layer_v.push(
                    Tensor::zeros((block_size, num_kv_heads, head_dim), dtype, device)
                        .map_err(ModelError::Candle)?,
                );
            }
            k_blocks.push(layer_k);
            v_blocks.push(layer_v);
        }

        Ok(Self {
            k_blocks,
            v_blocks,
            tokens_in_block: vec![0; num_blocks],
            block_size,
            num_layers,
            num_blocks,
            num_kv_heads,
            head_dim,
            dtype,
            device: device.clone(),
        })
    }

    /// Block size (tokens per block).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of blocks in the pool.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// How many token slots are filled in a block.
    pub fn tokens_stored(&self, block_idx: usize) -> usize {
        self.tokens_in_block[block_idx]
    }

    /// Set the token count for a block.
    pub fn set_tokens_stored(&mut self, block_idx: usize, n: usize) {
        self.tokens_in_block[block_idx] = n;
    }

    /// Write K/V for a single token into a specific block slot.
    ///
    /// * `layer` — transformer layer index
    /// * `block_idx` — block arena index
    /// * `position_in_block` — slot within the block (0..block_size)
    /// * `k_token` — K tensor of shape `[num_kv_heads, head_dim]`
    /// * `v_token` — V tensor of shape `[num_kv_heads, head_dim]`
    pub fn write_kv(
        &mut self,
        layer: usize,
        block_idx: usize,
        position_in_block: usize,
        k_token: &Tensor,
        v_token: &Tensor,
    ) -> ModelResult<()> {
        let k_3d = k_token.unsqueeze(0).map_err(ModelError::Candle)?;
        let v_3d = v_token.unsqueeze(0).map_err(ModelError::Candle)?;
        self.k_blocks[layer][block_idx] = self.k_blocks[layer][block_idx]
            .slice_scatter0(&k_3d, position_in_block)
            .map_err(ModelError::Candle)?;
        self.v_blocks[layer][block_idx] = self.v_blocks[layer][block_idx]
            .slice_scatter0(&v_3d, position_in_block)
            .map_err(ModelError::Candle)?;
        Ok(())
    }

    /// Gather K/V from a list of blocks into contiguous tensors for attention.
    ///
    /// Returns `(K, V)` each of shape `[num_tokens, num_kv_heads, head_dim]`.
    ///
    /// * `layer` — transformer layer index
    /// * `block_ids` — ordered block arena indices for the request
    /// * `num_tokens` — total number of tokens to gather
    pub fn gather_kv(
        &self,
        layer: usize,
        block_ids: &[usize],
        num_tokens: usize,
    ) -> ModelResult<(Tensor, Tensor)> {
        if num_tokens == 0 {
            let k = Tensor::zeros(
                (0, self.num_kv_heads, self.head_dim),
                self.dtype,
                &self.device,
            )
            .map_err(ModelError::Candle)?;
            let v = Tensor::zeros(
                (0, self.num_kv_heads, self.head_dim),
                self.dtype,
                &self.device,
            )
            .map_err(ModelError::Candle)?;
            return Ok((k, v));
        }

        let mut k_parts = Vec::new();
        let mut v_parts = Vec::new();
        let mut remaining = num_tokens;

        for &bid in block_ids {
            if remaining == 0 {
                break;
            }
            let n = remaining.min(self.block_size);
            k_parts.push(
                self.k_blocks[layer][bid]
                    .narrow(0, 0, n)
                    .map_err(ModelError::Candle)?,
            );
            v_parts.push(
                self.v_blocks[layer][bid]
                    .narrow(0, 0, n)
                    .map_err(ModelError::Candle)?,
            );
            remaining -= n;
        }

        let k = Tensor::cat(&k_parts, 0).map_err(ModelError::Candle)?;
        let v = Tensor::cat(&v_parts, 0).map_err(ModelError::Candle)?;
        Ok((k, v))
    }

    /// Scatter newly computed K/V tokens into the appropriate block positions.
    ///
    /// After a forward pass, the model produces a full K/V cache (old + new).
    /// This method extracts only the new tokens (from `tokens_before` onward)
    /// and writes them into the correct block slots.
    ///
    /// * `layer` — transformer layer index
    /// * `block_ids` — ordered block arena indices for the request
    /// * `tokens_before` — number of tokens that were already in the cache
    /// * `k_full` — full K cache from forward pass, shape `[total_tokens, num_kv_heads, head_dim]`
    /// * `v_full` — full V cache from forward pass, same shape
    pub fn scatter_new_kv(
        &mut self,
        layer: usize,
        block_ids: &[usize],
        tokens_before: usize,
        k_full: &Tensor,
        v_full: &Tensor,
    ) -> ModelResult<()> {
        let total_tokens = k_full.dims()[0];
        if total_tokens <= tokens_before {
            return Ok(());
        }
        let new_count = total_tokens - tokens_before;

        // Extract only the new tokens.
        let new_k = k_full
            .narrow(0, tokens_before, new_count)
            .map_err(ModelError::Candle)?;
        let new_v = v_full
            .narrow(0, tokens_before, new_count)
            .map_err(ModelError::Candle)?;

        // Write each new token into the correct block and slot.
        for i in 0..new_count {
            let global_pos = tokens_before + i;
            let block_offset = global_pos / self.block_size;
            let position_in_block = global_pos % self.block_size;

            if block_offset >= block_ids.len() {
                break;
            }
            let bid = block_ids[block_offset];

            let k_token = new_k.narrow(0, i, 1).map_err(ModelError::Candle)?;
            let v_token = new_v.narrow(0, i, 1).map_err(ModelError::Candle)?;

            self.k_blocks[layer][bid] = self.k_blocks[layer][bid]
                .slice_scatter0(&k_token, position_in_block)
                .map_err(ModelError::Candle)?;
            self.v_blocks[layer][bid] = self.v_blocks[layer][bid]
                .slice_scatter0(&v_token, position_in_block)
                .map_err(ModelError::Candle)?;
        }

        // Update tokens_in_block for affected blocks.
        let last_global = tokens_before + new_count - 1;
        let last_block_offset = last_global / self.block_size;
        for i in 0..=last_block_offset {
            if i >= block_ids.len() {
                break;
            }
            let bid = block_ids[i];
            if i < last_block_offset {
                // Fully filled block.
                self.tokens_in_block[bid] = self.block_size;
            } else {
                // Partial — set to position + 1.
                let fill = (last_global % self.block_size) + 1;
                self.tokens_in_block[bid] = self.tokens_in_block[bid].max(fill);
            }
        }

        Ok(())
    }

    /// Copy all layers of a block (for copy-on-write).
    pub fn copy_block(&mut self, src_idx: usize, dst_idx: usize) -> ModelResult<()> {
        for layer in 0..self.num_layers {
            let src_k = self.k_blocks[layer][src_idx].clone();
            let src_v = self.v_blocks[layer][src_idx].clone();
            self.k_blocks[layer][dst_idx] = src_k;
            self.v_blocks[layer][dst_idx] = src_v;
        }
        self.tokens_in_block[dst_idx] = self.tokens_in_block[src_idx];
        Ok(())
    }

    /// Reset a block to zeros (for reuse after free).
    pub fn reset_block(&mut self, block_idx: usize) -> ModelResult<()> {
        for layer in 0..self.num_layers {
            self.k_blocks[layer][block_idx] = Tensor::zeros(
                (self.block_size, self.num_kv_heads, self.head_dim),
                self.dtype,
                &self.device,
            )
            .map_err(ModelError::Candle)?;
            self.v_blocks[layer][block_idx] = Tensor::zeros(
                (self.block_size, self.num_kv_heads, self.head_dim),
                self.dtype,
                &self.device,
            )
            .map_err(ModelError::Candle)?;
        }
        self.tokens_in_block[block_idx] = 0;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    const NUM_BLOCKS: usize = 8;
    const NUM_LAYERS: usize = 2;
    const NUM_KV_HEADS: usize = 4;
    const HEAD_DIM: usize = 8;
    const BLOCK_SIZE: usize = 4;

    fn make_pool() -> KvBlockPool {
        KvBlockPool::new(
            NUM_BLOCKS,
            NUM_LAYERS,
            NUM_KV_HEADS,
            HEAD_DIM,
            BLOCK_SIZE,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
    }

    #[test]
    fn test_pool_dimensions() {
        let pool = make_pool();
        assert_eq!(pool.num_blocks(), NUM_BLOCKS);
        assert_eq!(pool.num_layers(), NUM_LAYERS);
        assert_eq!(pool.block_size(), BLOCK_SIZE);
    }

    #[test]
    fn test_tokens_stored_initially_zero() {
        let pool = make_pool();
        for i in 0..NUM_BLOCKS {
            assert_eq!(pool.tokens_stored(i), 0);
        }
    }

    #[test]
    fn test_set_tokens_stored() {
        let mut pool = make_pool();
        pool.set_tokens_stored(3, 2);
        assert_eq!(pool.tokens_stored(3), 2);
        assert_eq!(pool.tokens_stored(0), 0);
    }

    #[test]
    fn test_write_and_read_single_token() {
        let mut pool = make_pool();

        // Write a token into block 0, position 0, layer 0.
        let k = Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap();
        let v = (Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap() * 2.0)
            .unwrap();

        pool.write_kv(0, 0, 0, &k, &v).unwrap();
        pool.set_tokens_stored(0, 1);

        // Gather 1 token from block 0.
        let (k_out, v_out) = pool.gather_kv(0, &[0], 1).unwrap();
        assert_eq!(k_out.dims(), &[1, NUM_KV_HEADS, HEAD_DIM]);
        assert_eq!(v_out.dims(), &[1, NUM_KV_HEADS, HEAD_DIM]);

        // Verify values.
        let k_vals = k_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((k_vals[0] - 1.0).abs() < 1e-5);

        let v_vals = v_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v_vals[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_write_full_block_and_read() {
        let mut pool = make_pool();

        // Fill block 1 with 4 tokens (full block).
        for pos in 0..BLOCK_SIZE {
            let val = (pos + 1) as f32;
            let k = (Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap()
                * val as f64)
                .unwrap();
            let v = (Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap()
                * (val * 10.0) as f64)
                .unwrap();
            pool.write_kv(0, 1, pos, &k, &v).unwrap();
        }
        pool.set_tokens_stored(1, BLOCK_SIZE);

        let (k_out, _v_out) = pool.gather_kv(0, &[1], BLOCK_SIZE).unwrap();
        assert_eq!(k_out.dims(), &[BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]);

        // Verify first token's k value is 1.0 and last is 4.0.
        let k_flat = k_out.to_vec3::<f32>().unwrap();
        assert!((k_flat[0][0][0] - 1.0).abs() < 1e-5);
        assert!((k_flat[3][0][0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_gather_multiple_blocks() {
        let mut pool = make_pool();

        // Fill block 0 fully (4 tokens) and block 2 with 2 tokens.
        for pos in 0..BLOCK_SIZE {
            let k = Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap();
            let v = Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap();
            pool.write_kv(0, 0, pos, &k, &v).unwrap();
        }
        pool.set_tokens_stored(0, BLOCK_SIZE);

        for pos in 0..2 {
            let k = (Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap()
                * 2.0)
                .unwrap();
            let v = (Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap()
                * 2.0)
                .unwrap();
            pool.write_kv(0, 2, pos, &k, &v).unwrap();
        }
        pool.set_tokens_stored(2, 2);

        // Gather 6 tokens from blocks [0, 2]: 4 from block 0 + 2 from block 2.
        let (k_out, _v_out) = pool.gather_kv(0, &[0, 2], 6).unwrap();
        assert_eq!(k_out.dims(), &[6, NUM_KV_HEADS, HEAD_DIM]);

        let k_flat = k_out.to_vec3::<f32>().unwrap();
        // First 4 tokens from block 0 (value 1.0).
        assert!((k_flat[0][0][0] - 1.0).abs() < 1e-5);
        assert!((k_flat[3][0][0] - 1.0).abs() < 1e-5);
        // Next 2 tokens from block 2 (value 2.0).
        assert!((k_flat[4][0][0] - 2.0).abs() < 1e-5);
        assert!((k_flat[5][0][0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_gather_zero_tokens() {
        let pool = make_pool();
        let (k, v) = pool.gather_kv(0, &[], 0).unwrap();
        assert_eq!(k.dims(), &[0, NUM_KV_HEADS, HEAD_DIM]);
        assert_eq!(v.dims(), &[0, NUM_KV_HEADS, HEAD_DIM]);
    }

    #[test]
    fn test_scatter_prefill() {
        let mut pool = make_pool();

        // Simulate prefill: model produces K/V for 6 tokens (blocks [0, 1]).
        // tokens_before = 0 (no previous cache).
        let k_full = Tensor::ones(&[6, NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap();
        let v_full =
            (Tensor::ones(&[6, NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap() * 3.0)
                .unwrap();

        pool.scatter_new_kv(0, &[0, 1], 0, &k_full, &v_full)
            .unwrap();

        // Block 0 should have 4 tokens, block 1 should have 2.
        assert_eq!(pool.tokens_stored(0), BLOCK_SIZE);
        assert_eq!(pool.tokens_stored(1), 2);

        // Verify values via gather.
        let (k_out, v_out) = pool.gather_kv(0, &[0, 1], 6).unwrap();
        assert_eq!(k_out.dims(), &[6, NUM_KV_HEADS, HEAD_DIM]);

        let k_vals = k_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((k_vals[0] - 1.0).abs() < 1e-5);

        let v_vals = v_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v_vals[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_scatter_decode_step() {
        let mut pool = make_pool();

        // First prefill 3 tokens into block 0.
        let k_prefill =
            Tensor::ones(&[3, NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap();
        let v_prefill =
            Tensor::ones(&[3, NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap();
        pool.scatter_new_kv(0, &[0], 0, &k_prefill, &v_prefill)
            .unwrap();
        assert_eq!(pool.tokens_stored(0), 3);

        // Now simulate decode: model produces 4-token cache (3 old + 1 new).
        let k_full =
            (Tensor::ones(&[4, NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap() * 5.0)
                .unwrap();
        let v_full =
            (Tensor::ones(&[4, NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap() * 5.0)
                .unwrap();
        pool.scatter_new_kv(0, &[0], 3, &k_full, &v_full).unwrap();

        // Block 0 should now have 4 tokens.
        assert_eq!(pool.tokens_stored(0), BLOCK_SIZE);

        // Verify the newly written token (position 3).
        let (k_out, _) = pool.gather_kv(0, &[0], 4).unwrap();
        let k_flat = k_out.to_vec3::<f32>().unwrap();
        assert!((k_flat[3][0][0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_scatter_across_block_boundary() {
        let mut pool = make_pool();

        // Prefill 3 tokens into block 0.
        let k3 = Tensor::ones(&[3, NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap();
        let v3 = Tensor::ones(&[3, NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap();
        pool.scatter_new_kv(0, &[0, 1], 0, &k3, &v3).unwrap();

        // Decode 3 more tokens (positions 3, 4, 5): crosses from block 0 to block 1.
        let k6 = (Tensor::ones(&[6, NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap()
            * 7.0)
            .unwrap();
        let v6 = (Tensor::ones(&[6, NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap()
            * 7.0)
            .unwrap();
        pool.scatter_new_kv(0, &[0, 1], 3, &k6, &v6).unwrap();

        // Block 0 should be full (4), block 1 should have 2.
        assert_eq!(pool.tokens_stored(0), BLOCK_SIZE);
        assert_eq!(pool.tokens_stored(1), 2);
    }

    #[test]
    fn test_copy_block() {
        let mut pool = make_pool();

        // Write some data to block 0.
        let k = (Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap() * 9.0)
            .unwrap();
        let v = (Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap() * 9.0)
            .unwrap();
        pool.write_kv(0, 0, 0, &k, &v).unwrap();
        pool.set_tokens_stored(0, 1);

        // Copy block 0 → block 3.
        pool.copy_block(0, 3).unwrap();

        assert_eq!(pool.tokens_stored(3), 1);
        let (k_out, _) = pool.gather_kv(0, &[3], 1).unwrap();
        let vals = k_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_reset_block() {
        let mut pool = make_pool();

        // Write to block 2.
        let k = Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap();
        pool.write_kv(0, 2, 0, &k, &v).unwrap();
        pool.set_tokens_stored(2, 1);

        // Reset block 2.
        pool.reset_block(2).unwrap();
        assert_eq!(pool.tokens_stored(2), 0);

        // Values should be zeros.
        let (k_out, _) = pool.gather_kv(0, &[2], 1).unwrap();
        let vals = k_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0]).abs() < 1e-5);
    }

    #[test]
    fn test_multi_layer_gather() {
        let mut pool = make_pool();

        // Write different values per layer.
        for layer in 0..NUM_LAYERS {
            let val = (layer + 1) as f64;
            let k = (Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap()
                * val)
                .unwrap();
            let v = (Tensor::ones(&[NUM_KV_HEADS, HEAD_DIM], DType::F32, &Device::Cpu).unwrap()
                * val)
                .unwrap();
            pool.write_kv(layer, 0, 0, &k, &v).unwrap();
        }
        pool.set_tokens_stored(0, 1);

        // Layer 0 should have value 1.0, layer 1 should have value 2.0.
        let (k0, _) = pool.gather_kv(0, &[0], 1).unwrap();
        let (k1, _) = pool.gather_kv(1, &[0], 1).unwrap();

        let v0 = k0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let v1 = k1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v0[0] - 1.0).abs() < 1e-5);
        assert!((v1[0] - 2.0).abs() < 1e-5);
    }
}
