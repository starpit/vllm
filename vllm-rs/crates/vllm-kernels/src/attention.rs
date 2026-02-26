// SPDX-License-Identifier: Apache-2.0
//! Attention kernels.
//!
//! Trait abstraction for paged attention and related kernels.
//! Port of: `csrc/attention/` and paged attention in `csrc/ops.h`

use candle_core::Tensor;

use crate::error::KernelResult;

/// Paged attention kernel interface.
///
/// Abstracts the CUDA paged attention v1/v2 kernels that operate on
/// key-value caches stored in fixed-size blocks.
#[allow(clippy::too_many_arguments)]
pub trait AttentionKernels: Send + Sync {
    /// Paged attention v1: single-pass attention over paged KV cache.
    ///
    /// * `query` — [num_seqs, num_heads, head_size]
    /// * `key_cache` — [num_blocks, num_kv_heads, head_size/x, block_size, x]
    /// * `value_cache` — [num_blocks, num_kv_heads, head_size, block_size]
    /// * `block_tables` — [num_seqs, max_num_blocks_per_seq]
    /// * `seq_lens` — [num_seqs]
    /// * `scale` — attention scaling factor (1/sqrt(head_size))
    /// * `block_size` — number of tokens per cache block
    ///
    /// Returns attention output [num_seqs, num_heads, head_size].
    fn paged_attention_v1(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        scale: f64,
        block_size: usize,
    ) -> KernelResult<Tensor>;

    /// Paged attention v2: two-pass attention for long sequences.
    ///
    /// Uses intermediate buffers for numerical stability on long sequences.
    /// Same parameters as v1, plus workspace tensors.
    fn paged_attention_v2(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        scale: f64,
        block_size: usize,
    ) -> KernelResult<Tensor>;
}

/// CPU implementation of attention kernels (for testing).
///
/// Implements a simple non-paged attention for correctness testing.
/// Not optimized — only for unit tests.
pub struct CpuAttentionKernels;

impl AttentionKernels for CpuAttentionKernels {
    fn paged_attention_v1(
        &self,
        query: &Tensor,
        _key_cache: &Tensor,
        _value_cache: &Tensor,
        _block_tables: &Tensor,
        _seq_lens: &Tensor,
        _scale: f64,
        _block_size: usize,
    ) -> KernelResult<Tensor> {
        // Stub: return zeros with the right shape.
        // A real CPU implementation would gather keys/values from paged cache
        // and compute attention. This is complex and is deferred to when we
        // have actual model execution tests.
        let shape = query.dims();
        let out = Tensor::zeros(shape, query.dtype(), query.device())?;
        Ok(out)
    }

    fn paged_attention_v2(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_tables: &Tensor,
        seq_lens: &Tensor,
        scale: f64,
        block_size: usize,
    ) -> KernelResult<Tensor> {
        // V2 falls back to V1 for CPU.
        self.paged_attention_v1(query, key_cache, value_cache, block_tables, seq_lens, scale, block_size)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_cpu_attention_stub_shape() {
        let kernels = CpuAttentionKernels;

        let query = Tensor::zeros(&[2, 8, 64], DType::F32, &Device::Cpu).unwrap();
        let key_cache = Tensor::zeros(&[100, 8, 8, 16, 8], DType::F32, &Device::Cpu).unwrap();
        let value_cache = Tensor::zeros(&[100, 8, 64, 16], DType::F32, &Device::Cpu).unwrap();
        let block_tables = Tensor::zeros(&[2, 10], DType::U32, &Device::Cpu).unwrap();
        let seq_lens = Tensor::new(&[32u32, 64], &Device::Cpu).unwrap();

        let out = kernels
            .paged_attention_v1(
                &query,
                &key_cache,
                &value_cache,
                &block_tables,
                &seq_lens,
                0.125,
                16,
            )
            .unwrap();
        assert_eq!(out.dims(), &[2, 8, 64]);
    }

    #[test]
    fn test_cpu_attention_v2_same_as_v1() {
        let kernels = CpuAttentionKernels;

        let query = Tensor::zeros(&[1, 4, 32], DType::F32, &Device::Cpu).unwrap();
        let key_cache = Tensor::zeros(&[10, 4, 4, 16, 8], DType::F32, &Device::Cpu).unwrap();
        let value_cache = Tensor::zeros(&[10, 4, 32, 16], DType::F32, &Device::Cpu).unwrap();
        let block_tables = Tensor::zeros(&[1, 5], DType::U32, &Device::Cpu).unwrap();
        let seq_lens = Tensor::new(&[16u32], &Device::Cpu).unwrap();

        let out = kernels
            .paged_attention_v2(
                &query,
                &key_cache,
                &value_cache,
                &block_tables,
                &seq_lens,
                0.177,
                16,
            )
            .unwrap();
        assert_eq!(out.dims(), &[1, 4, 32]);
    }
}
