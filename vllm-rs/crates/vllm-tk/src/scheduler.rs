// SPDX-License-Identifier: Apache-2.0
//! Instruction generation for the TK KVM megakernel.
//!
//! Builds the `[SM_COUNT, max_per_sm, INSTRUCTION_WIDTH]` i32 instruction tensor
//! and the `[NL, NUM_OPS, N_BATCH_BLOCKS, max_barrier_cols]` u32 barrier tensor
//! that the megakernel reads at runtime.
//!
//! The scheduler uses round-robin SM assignment, matching the Python reference.

/// Width of each instruction vector (int32 elements).
pub const INSTRUCTION_WIDTH: usize = 32;

/// Width of each timing vector.
pub const TIMING_WIDTH: usize = 128;

/// Number of opcodes (AttnNorm=1 through LM_Head=11).
pub const NUM_OPS: usize = 11;

// Opcodes — must match OPCODE_* defines in llama_sm89.cuh
pub const OP_ATTN_NORM: i32 = 1;
pub const OP_QKV_ROPE_APPEND: i32 = 2;
pub const OP_GQA_ATTENTION_PREFILL: i32 = 3;
pub const OP_GQA_ATTENTION_DECODE: i32 = 4;
pub const OP_O_PROJ_RESIDUAL: i32 = 5;
pub const OP_MLP_NORM: i32 = 6;
pub const OP_GATE_SILU: i32 = 7;
pub const OP_UP_MATMUL: i32 = 8;
pub const OP_DOWN_PROJ_RESIDUAL: i32 = 9;
pub const OP_LM_HEAD_NORM: i32 = 10;
pub const OP_LM_HEAD: i32 = 11;

/// Compute the optimal GEMM output tile width, matching `optimal_out_block` in llama_sm89.cuh.
///
/// Picks the largest power-of-2 in `[min_block, 128]` such that
/// `output_dim / tile >= sm_count / 2`, falling back to the smallest valid tile.
pub fn optimal_out_block(output_dim: usize, sm_count: usize, min_block: usize) -> usize {
    let target = sm_count / 2;
    let mut ob = 128;
    while ob >= min_block {
        if output_dim.is_multiple_of(ob) && output_dim / ob >= target {
            return ob;
        }
        ob /= 2;
    }
    ob = min_block;
    while ob <= 128 {
        if output_dim.is_multiple_of(ob) {
            return ob;
        }
        ob *= 2;
    }
    min_block
}

/// Model dimensions needed for instruction generation.
#[derive(Debug, Clone)]
pub struct TkModelConfig {
    pub num_hidden_layers: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub sm_count: usize,
    pub matmul_batch_block_size: usize, // typically 128
    pub attn_batch_block_size: usize,   // typically GQA_RATIO = nah / nkh
}

/// Pre-computed per-op tile sizes and instruction counts.
#[derive(Debug, Clone)]
pub struct TkOpConfig {
    pub nbh: usize,
    pub qkv_out_block: usize,
    pub o_proj_out_block: usize,
    pub gate_out_block: usize,
    pub up_out_block: usize,
    pub down_out_block: usize,
    pub lm_head_out_block: usize,
    pub n_cols_hd: usize,
    pub n_cols_id: usize,
    pub n_cols_vs: usize,
}

impl TkOpConfig {
    pub fn new(cfg: &TkModelConfig) -> Self {
        let nbh = cfg.num_attention_heads + 2 * cfg.num_kv_heads;
        let qkv_out_block = optimal_out_block(nbh * cfg.head_dim, cfg.sm_count, cfg.head_dim);
        let o_proj_out_block = optimal_out_block(cfg.hidden_dim, cfg.sm_count, 32);
        let gate_out_block = optimal_out_block(cfg.intermediate_dim, cfg.sm_count, 32);
        let up_out_block = gate_out_block;
        let down_out_block = o_proj_out_block;
        let lm_head_out_block = 128;

        Self {
            nbh,
            qkv_out_block,
            o_proj_out_block,
            gate_out_block,
            up_out_block,
            down_out_block,
            lm_head_out_block,
            n_cols_hd: cfg.hidden_dim / o_proj_out_block,
            n_cols_id: cfg.intermediate_dim / gate_out_block,
            n_cols_vs: cfg.vocab_size / lm_head_out_block,
        }
    }
}

/// Build decode instructions for a given batch size.
///
/// Returns `(instructions, timings, barrier)` as flat CPU vectors, plus shapes.
/// - `instructions`: `[sm_count, max_per_sm, INSTRUCTION_WIDTH]` i32
/// - `timings`: same shape, all zeros
/// - `barrier`: `[NL, NUM_OPS, n_batch_blocks, max_barrier_cols]` u32, all zeros
pub fn build_decode_instructions(cfg: &TkModelConfig, batch_size: usize) -> DecodeInstructions {
    let ops = TkOpConfig::new(cfg);
    let attn_bb = cfg.attn_batch_block_size;
    let nl = cfg.num_hidden_layers;
    let sm_count = cfg.sm_count;

    // Collect instructions per SM via round-robin.
    let mut sm_lists: Vec<Vec<[i32; INSTRUCTION_WIDTH]>> = vec![vec![]; sm_count];
    let mut sm_ctr: usize = 0;

    let mut add = |fields: &[i32]| {
        let mut inst = [0i32; INSTRUCTION_WIDTH];
        for (i, &f) in fields.iter().enumerate() {
            inst[i] = f;
        }
        sm_lists[sm_ctr % sm_count].push(inst);
        sm_ctr += 1;
    };

    // Per-layer ops (1-8)
    for layer in 0..nl as i32 {
        // Op 1: AttnNorm — one per token
        for t in 0..batch_size as i32 {
            add(&[OP_ATTN_NORM, layer, t]);
        }
        // Op 2: QKV_RopeAppend — one per head-block
        for col in 0..ops.nbh as i32 {
            add(&[OP_QKV_ROPE_APPEND, layer, 0, col]);
        }
        // Op 3: GQA_AttentionDecode — one per (attn_batch_block, kv_head)
        for bb in 0..batch_size.div_ceil(attn_bb) as i32 {
            for kv_h in 0..cfg.num_kv_heads as i32 {
                add(&[OP_GQA_ATTENTION_DECODE, layer, bb, kv_h]);
            }
        }
        // Op 4: O_ProjResidual — one per output col block
        for col in 0..ops.n_cols_hd as i32 {
            add(&[OP_O_PROJ_RESIDUAL, layer, 0, col]);
        }
        // Op 5: MlpNorm — one per token
        for t in 0..batch_size as i32 {
            add(&[OP_MLP_NORM, layer, t]);
        }
        // Op 6: GateSiLU — one per output col block
        for col in 0..ops.n_cols_id as i32 {
            add(&[OP_GATE_SILU, layer, 0, col]);
        }
        // Op 7: UpMatmul — one per output col block
        for col in 0..ops.n_cols_id as i32 {
            add(&[OP_UP_MATMUL, layer, 0, col]);
        }
        // Op 8: DownProjResidual — one per output col block
        for col in 0..ops.n_cols_hd as i32 {
            add(&[OP_DOWN_PROJ_RESIDUAL, layer, 0, col]);
        }
    }

    // Op 9: LM_HeadNorm — one per token
    for t in 0..batch_size as i32 {
        add(&[OP_LM_HEAD_NORM, 0, t]);
    }

    // Op 10: LM_Head — one per output col block
    for col in 0..ops.n_cols_vs as i32 {
        add(&[OP_LM_HEAD, 0, col]);
    }

    // Pad all SM lists to max length with NOPs.
    let max_per_sm = sm_lists.iter().map(|v| v.len()).max().unwrap_or(0);
    for sm in &mut sm_lists {
        sm.resize(max_per_sm, [0i32; INSTRUCTION_WIDTH]);
    }

    // Flatten to contiguous buffer [sm_count, max_per_sm, IW].
    let total = sm_count * max_per_sm * INSTRUCTION_WIDTH;
    let mut instructions = vec![0i32; total];
    for (sm_idx, sm) in sm_lists.iter().enumerate() {
        for (ins_idx, inst) in sm.iter().enumerate() {
            let base = (sm_idx * max_per_sm + ins_idx) * INSTRUCTION_WIDTH;
            instructions[base..base + INSTRUCTION_WIDTH].copy_from_slice(inst);
        }
    }

    // Barrier tensor: [NL, NUM_OPS, n_batch_blocks, max_barrier_cols]
    let n_batch_blocks = batch_size.div_ceil(cfg.matmul_batch_block_size);
    let max_barrier_cols = ops.nbh.max(ops.n_cols_id);
    let barrier_size = nl * NUM_OPS * n_batch_blocks * max_barrier_cols;

    DecodeInstructions {
        instructions,
        max_per_sm,
        sm_count,
        n_batch_blocks,
        max_barrier_cols,
        barrier_size,
        ops,
    }
}

/// Result of building decode instructions — CPU-side data ready for GPU upload.
pub struct DecodeInstructions {
    /// Flattened `[sm_count, max_per_sm, INSTRUCTION_WIDTH]` i32.
    pub instructions: Vec<i32>,
    pub max_per_sm: usize,
    pub sm_count: usize,
    /// Number of batch blocks for the barrier tensor.
    pub n_batch_blocks: usize,
    /// Barrier tensor column count.
    pub max_barrier_cols: usize,
    /// Total elements in the barrier tensor.
    pub barrier_size: usize,
    /// Op config used to generate these instructions.
    pub ops: TkOpConfig,
}

/// A prefill sequence: chunk_len tokens to process, with extend_offset tokens already in KV cache.
#[derive(Debug, Clone)]
pub struct PrefillSeq {
    /// Number of new tokens to prefill in this chunk.
    pub chunk_len: usize,
    /// Number of tokens already in KV cache before this chunk (for chunked prefill).
    pub extend_offset: usize,
}

/// Result of building prefill instructions.
pub struct PrefillInstructions {
    /// Flattened `[sm_count, max_per_sm, INSTRUCTION_WIDTH]` i32.
    pub instructions: Vec<i32>,
    pub max_per_sm: usize,
    pub sm_count: usize,
    pub n_batch_blocks: usize,
    pub max_barrier_cols: usize,
    pub barrier_size: usize,
    pub ops: TkOpConfig,
    /// Total number of prefill tokens across all sequences.
    pub total_tokens: usize,
}

impl PrefillInstructions {
    pub fn num_active_instructions(&self) -> usize {
        self.instructions
            .chunks(INSTRUCTION_WIDTH)
            .filter(|inst| inst[0] != 0)
            .count()
    }
}

/// Build prefill instructions for a batch of prefill sequences.
///
/// Each sequence contributes `chunk_len` tokens. The total batch_size for matmul ops
/// is `sum(chunk_len)`, and attention prefill instructions are generated per 16-token block.
pub fn build_prefill_instructions(
    cfg: &TkModelConfig,
    prefill_seqs: &[PrefillSeq],
) -> PrefillInstructions {
    let ops = TkOpConfig::new(cfg);
    let nl = cfg.num_hidden_layers;
    let sm_count = cfg.sm_count;
    let mbb = cfg.matmul_batch_block_size;

    let total_tokens: usize = prefill_seqs.iter().map(|s| s.chunk_len).sum();
    assert!(
        total_tokens > 0,
        "prefill batch must have at least one token"
    );

    let mut sm_lists: Vec<Vec<[i32; INSTRUCTION_WIDTH]>> = vec![vec![]; sm_count];
    let mut sm_ctr: usize = 0;

    let mut add = |fields: &[i32]| {
        let mut inst = [0i32; INSTRUCTION_WIDTH];
        for (i, &f) in fields.iter().enumerate() {
            inst[i] = f;
        }
        sm_lists[sm_ctr % sm_count].push(inst);
        sm_ctr += 1;
    };

    // Number of matmul batch blocks (for QKV, O_proj, MLP, etc.)
    let n_matmul_blocks = total_tokens.div_ceil(mbb);

    for layer in 0..nl as i32 {
        // Op 1: AttnNorm — one per token
        for t in 0..total_tokens as i32 {
            add(&[OP_ATTN_NORM, layer, t]);
        }

        // Op 2: QKV_RopeAppend — one per (batch_block, head_col)
        for bb in 0..n_matmul_blocks as i32 {
            for col in 0..ops.nbh as i32 {
                add(&[OP_QKV_ROPE_APPEND, layer, bb, col]);
            }
        }

        // Op 3: AttentionPrefill — for each seq, ceil(chunk_len/16) blocks × num_kv_heads
        for (seq_idx, seq) in prefill_seqs.iter().enumerate() {
            let num_q_blocks = seq.chunk_len.div_ceil(16);
            for block_idx in 0..num_q_blocks as i32 {
                for kv_h in 0..cfg.num_kv_heads as i32 {
                    add(&[
                        OP_GQA_ATTENTION_PREFILL,
                        layer,
                        seq_idx as i32,
                        block_idx,
                        seq.extend_offset as i32,
                        kv_h,
                    ]);
                }
            }
        }

        // Op 5: O_ProjResidual
        for bb in 0..n_matmul_blocks as i32 {
            for col in 0..ops.n_cols_hd as i32 {
                add(&[OP_O_PROJ_RESIDUAL, layer, bb, col]);
            }
        }

        // Op 6: MlpNorm — one per token
        for t in 0..total_tokens as i32 {
            add(&[OP_MLP_NORM, layer, t]);
        }

        // Op 7: GateSiLU
        for bb in 0..n_matmul_blocks as i32 {
            for col in 0..ops.n_cols_id as i32 {
                add(&[OP_GATE_SILU, layer, bb, col]);
            }
        }

        // Op 8: UpMatmul
        for bb in 0..n_matmul_blocks as i32 {
            for col in 0..ops.n_cols_id as i32 {
                add(&[OP_UP_MATMUL, layer, bb, col]);
            }
        }

        // Op 9: DownProjResidual
        for bb in 0..n_matmul_blocks as i32 {
            for col in 0..ops.n_cols_hd as i32 {
                add(&[OP_DOWN_PROJ_RESIDUAL, layer, bb, col]);
            }
        }
    }

    // Op 10: LM_HeadNorm — one per token
    for t in 0..total_tokens as i32 {
        add(&[OP_LM_HEAD_NORM, 0, t]);
    }

    // Op 11: LM_Head
    for col in 0..ops.n_cols_vs as i32 {
        add(&[OP_LM_HEAD, 0, col]);
    }

    // Pad all SM lists to max length with NOPs.
    let max_per_sm = sm_lists.iter().map(|v| v.len()).max().unwrap_or(0);
    for sm in &mut sm_lists {
        sm.resize(max_per_sm, [0i32; INSTRUCTION_WIDTH]);
    }

    // Flatten to contiguous buffer [sm_count, max_per_sm, IW].
    let total_elems = sm_count * max_per_sm * INSTRUCTION_WIDTH;
    let mut instructions = vec![0i32; total_elems];
    for (sm_idx, sm) in sm_lists.iter().enumerate() {
        for (ins_idx, inst) in sm.iter().enumerate() {
            let base = (sm_idx * max_per_sm + ins_idx) * INSTRUCTION_WIDTH;
            instructions[base..base + INSTRUCTION_WIDTH].copy_from_slice(inst);
        }
    }

    let n_batch_blocks = total_tokens.div_ceil(mbb);
    let max_barrier_cols = ops.nbh.max(ops.n_cols_id);
    let barrier_size = nl * NUM_OPS * n_batch_blocks * max_barrier_cols;

    PrefillInstructions {
        instructions,
        max_per_sm,
        sm_count,
        n_batch_blocks,
        max_barrier_cols,
        barrier_size,
        ops,
        total_tokens,
    }
}

impl DecodeInstructions {
    /// Count of non-NOP instructions.
    pub fn num_active_instructions(&self) -> usize {
        self.instructions
            .chunks(INSTRUCTION_WIDTH)
            .filter(|inst| inst[0] != 0)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_1b_config() -> TkModelConfig {
        TkModelConfig {
            num_hidden_layers: 16,
            hidden_dim: 2048,
            intermediate_dim: 8192,
            head_dim: 64,
            num_attention_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            sm_count: 142,
            matmul_batch_block_size: 128,
            attn_batch_block_size: 4, // GQA_RATIO = 32/8 = 4
        }
    }

    #[test]
    fn test_optimal_out_block() {
        // LLaMA 1B values (SM_COUNT=142, target=71)
        assert_eq!(optimal_out_block(48 * 64, 142, 64), 64); // QKV: 3072/64=48 < 71, fallback=64
        assert_eq!(optimal_out_block(2048, 142, 32), 32); // O_proj: all < 71, fallback=32
        assert_eq!(optimal_out_block(8192, 142, 32), 64); // Gate: 8192/64=128 >= 71 ✓
    }

    #[test]
    fn test_op_config() {
        let cfg = llama_1b_config();
        let ops = TkOpConfig::new(&cfg);
        assert_eq!(ops.nbh, 48);
        assert_eq!(ops.qkv_out_block, 64);
        assert_eq!(ops.o_proj_out_block, 32); // 2048: fallback to 32
        assert_eq!(ops.gate_out_block, 64); // 8192/64=128 >= 71
        assert_eq!(ops.n_cols_hd, 64); // 2048 / 32
        assert_eq!(ops.n_cols_id, 128); // 8192 / 64
        assert_eq!(ops.n_cols_vs, 250); // 32000 / 128
    }

    #[test]
    fn test_decode_instructions_bs128() {
        let cfg = llama_1b_config();
        let result = build_decode_instructions(&cfg, 128);

        // Per layer: 128(norm) + 48(qkv) + 32*8(attn=256) + 64(oproj) + 128(mlpnorm)
        //          + 128(gate) + 128(up) + 64(down) = 944
        // 16 layers = 15104 + 128(lm_norm) + 250(lm_head) = 15482
        assert_eq!(result.num_active_instructions(), 15482);

        // Shape checks
        assert_eq!(result.sm_count, 142);
        assert_eq!(
            result.instructions.len(),
            142 * result.max_per_sm * INSTRUCTION_WIDTH
        );

        // Barrier shape
        assert_eq!(result.n_batch_blocks, 1); // 128 / 128 = 1
        assert_eq!(result.max_barrier_cols, 128); // max(48, 128)
    }

    #[test]
    fn test_decode_instructions_bs256() {
        let cfg = llama_1b_config();
        let result = build_decode_instructions(&cfg, 256);

        // Per layer: 256 + 48 + 64*8(=512) + 64 + 256 + 128 + 128 + 64 = 1456
        // 16 layers = 23296 + 256 + 250 = 23802
        assert_eq!(result.num_active_instructions(), 23802);
        assert_eq!(result.n_batch_blocks, 2); // 256 / 128 = 2
    }

    #[test]
    fn test_round_robin_distribution() {
        let cfg = llama_1b_config();
        let result = build_decode_instructions(&cfg, 128);

        // Check that instructions are distributed across SMs.
        let total = result.num_active_instructions();
        let per_sm_min = total / 142;
        let per_sm_max = per_sm_min + 1;

        // Count non-NOP per SM
        for sm in 0..142 {
            let count = (0..result.max_per_sm)
                .filter(|&i| {
                    let base = (sm * result.max_per_sm + i) * INSTRUCTION_WIDTH;
                    result.instructions[base] != 0
                })
                .count();
            assert!(
                count >= per_sm_min && count <= per_sm_max,
                "SM {sm} has {count} instructions, expected {per_sm_min}..={per_sm_max}"
            );
        }
    }

    #[test]
    fn test_prefill_instructions_single_seq() {
        let cfg = llama_1b_config();
        // Single sequence, 64 tokens, no extend offset
        let seqs = vec![PrefillSeq {
            chunk_len: 64,
            extend_offset: 0,
        }];
        let result = build_prefill_instructions(&cfg, &seqs);

        // Per layer:
        //   64(norm) + 48(qkv, 1 batch block) + ceil(64/16)*8 = 4*8 = 32(prefill_attn)
        //   + 64(oproj) + 64(mlpnorm) + 128(gate) + 128(up) + 64(down) = 592
        // 16 layers = 9472 + 64(lm_norm) + 250(lm_head) = 9786
        assert_eq!(result.total_tokens, 64);
        assert_eq!(result.n_batch_blocks, 1); // 64 / 128 = 1

        // Verify prefill attention instructions exist
        let prefill_count: usize = result
            .instructions
            .chunks(INSTRUCTION_WIDTH)
            .filter(|inst| inst[0] == OP_GQA_ATTENTION_PREFILL)
            .count();
        // 16 layers × 4 blocks × 8 kv_heads = 512
        assert_eq!(prefill_count, 16 * 4 * 8);
    }

    #[test]
    fn test_prefill_instructions_multi_seq() {
        let cfg = llama_1b_config();
        // Two sequences: 32 and 48 tokens
        let seqs = vec![
            PrefillSeq {
                chunk_len: 32,
                extend_offset: 0,
            },
            PrefillSeq {
                chunk_len: 48,
                extend_offset: 100,
            },
        ];
        let result = build_prefill_instructions(&cfg, &seqs);
        assert_eq!(result.total_tokens, 80);

        // Prefill attention: seq0 has 2 blocks, seq1 has 3 blocks = 5 blocks × 8 kv_heads × 16 layers
        let prefill_count: usize = result
            .instructions
            .chunks(INSTRUCTION_WIDTH)
            .filter(|inst| inst[0] == OP_GQA_ATTENTION_PREFILL)
            .count();
        assert_eq!(prefill_count, 16 * (2 + 3) * 8);
    }

    #[test]
    fn test_instruction_ordering() {
        // Verify that within each SM's list, instructions appear in the correct
        // global order: layer 0 ops 1-8, layer 1 ops 1-8, ..., op 9, op 10.
        let cfg = llama_1b_config();
        let result = build_decode_instructions(&cfg, 128);

        // Reconstruct global instruction order by round-robin interleaving.
        let mut global_ops = Vec::new();
        let max = result.max_per_sm;
        for step in 0..max {
            for sm in 0..142 {
                let base = (sm * max + step) * INSTRUCTION_WIDTH;
                let op = result.instructions[base];
                if op != 0 {
                    global_ops.push(op);
                }
            }
        }

        // The first instruction should be AttnNorm (layer 0)
        assert_eq!(global_ops[0], OP_ATTN_NORM);
        // The last instruction should be LM_Head
        assert_eq!(*global_ops.last().unwrap(), OP_LM_HEAD);
    }
}
