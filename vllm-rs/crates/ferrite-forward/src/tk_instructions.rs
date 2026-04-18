// SPDX-License-Identifier: Apache-2.0
//! Runtime instruction generation for the TK throughput megakernel.
//!
//! Work-stealing: instructions are a flat `[total_instructions][32]` tensor.
//! The megakernel atomically increments `global_instruction_index` to grab
//! the next instruction. Order within the array implicitly encodes the DAG
//! — all ops for layer N precede layer N+1.
//!
//! Opcodes and instruction formats match `llama.cuh` in
//! `third_party/Megakernels/demos/cross-gpu-llama/`.

const INST_WIDTH: usize = 32;

// Opcodes matching llama.cuh exactly.
const OPCODE_ATTN_NORM: i32 = 1;
const OPCODE_QKV_ROPE_APPEND: i32 = 2;
const OPCODE_ATTENTION_PREFILL: i32 = 3;
const OPCODE_ATTENTION_DECODE: i32 = 4;
const OPCODE_O_PROJ_RESIDUAL: i32 = 5;
const OPCODE_MLP_NORM: i32 = 6;
const OPCODE_GATE_SILU: i32 = 7;
const OPCODE_UP_MATMUL: i32 = 8;
const OPCODE_DOWN_PROJ_RESIDUAL: i32 = 9;
const OPCODE_LM_HEAD_NORM: i32 = 10;
const OPCODE_LM_HEAD: i32 = 11;
// const OPCODE_BARRIER_INC: i32 = 12;
// const OPCODE_ALL_DEVICE_BARRIER: i32 = 13;

fn push_inst(out: &mut Vec<i32>, words: &[i32]) {
    let base = out.len();
    out.resize(base + INST_WIDTH, 0);
    for (i, &w) in words.iter().enumerate().take(INST_WIDTH) {
        out[base + i] = w;
    }
}

/// Build the flat instruction tensor for the throughput megakernel.
///
/// Returns a `Vec<i32>` of length `total_instructions * 32`.
/// `batch_size` must be a multiple of `matmul_batch_block_size`.
///
/// For the initial implementation, all tokens are treated as decode
/// (single-token sequences). Prefill instructions are NOT generated —
/// that will come when we split prefill vs decode in ForwardCtx.
#[allow(clippy::too_many_arguments)]
/// Per-sequence info for instruction generation.
/// `(num_q_tokens, token_offset)` where token_offset = seqused_k - num_q_tokens.
pub type SeqInfo = Vec<(usize, usize)>;

pub fn build_throughput_instructions(
    batch_size: usize,
    num_tokens: usize,
    num_layers: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    vocab_size: usize,
    matmul_batch_block_size: usize,
    matmul_out_block_size: usize,
    // If Some, generate prefill attention instructions. Each entry is (q_len, token_offset).
    prefill_seq_info: Option<&SeqInfo>,
) -> Vec<i32> {
    assert!(
        batch_size % matmul_batch_block_size == 0,
        "batch_size ({batch_size}) must be a multiple of matmul_batch_block_size ({matmul_batch_block_size})"
    );
    assert!(
        num_tokens <= batch_size,
        "num_tokens ({num_tokens}) must be <= batch_size ({batch_size})"
    );

    let num_batch_blocks = batch_size / matmul_batch_block_size;
    let qkv_dim = (num_attention_heads + 2 * num_kv_heads) * head_dim;
    let num_qkv_blocks = qkv_dim / matmul_out_block_size;
    let num_output_blocks = hidden_dim / matmul_out_block_size;
    let num_intermediate_blocks = intermediate_dim / matmul_out_block_size;

    let mut out = Vec::new();

    for layer in 0..num_layers as i32 {
        // Op 1: AttnNorm — [opcode, layer, num_items, batch_idx_0, ...]
        // Each instruction handles a batch of sequence indices.
        // Must emit for ALL padded tokens so barriers reach the expected count.
        for bidx in 0..batch_size as i32 {
            push_inst(&mut out, &[OPCODE_ATTN_NORM, layer, 1, bidx]);
        }

        // Op 2: QKV_RopeAppend — matmul format:
        // [opcode, layer, local_row, local_col, row, col]
        // For single-GPU: local_row = row (batch block), local_col = col (output block).
        for batch_block in 0..num_batch_blocks as i32 {
            for qkv_block in 0..num_qkv_blocks as i32 {
                push_inst(&mut out, &[
                    OPCODE_QKV_ROPE_APPEND,
                    layer,
                    batch_block, qkv_block, // local_row, local_col
                    batch_block, qkv_block, // row, col (same for single-GPU)
                ]);
            }
        }

        // Attention: prefill (op 3) or decode (op 4) depending on batch type.
        if let Some(seq_info) = prefill_seq_info {
            // Op 3: AttentionPrefill — [opcode, layer, seq_idx, prefill_block_idx, token_offset, kv_head_idx]
            // Each instruction handles a 16-token Q block for one KV head.
            for (seq_idx, &(q_len, token_offset)) in seq_info.iter().enumerate() {
                let num_q_blocks = q_len.div_ceil(16);
                for kv_head in 0..num_kv_heads {
                    for q_block in 0..num_q_blocks {
                        push_inst(&mut out, &[
                            OPCODE_ATTENTION_PREFILL,
                            layer,
                            seq_idx as i32,
                            q_block as i32,
                            token_offset as i32,
                            kv_head as i32,
                        ]);
                    }
                }
            }
        } else {
            // Op 4: AttentionDecode — batched format:
            // [opcode, layer, num_entries, (global_seq_idx, kv_head_idx)*]
            let max_seqs_per_inst = (INST_WIDTH - 3) / 2;
            let mut pairs: Vec<(i32, i32)> = Vec::new();
            for seq_idx in 0..num_tokens as i32 {
                for kv_head in 0..num_kv_heads as i32 {
                    pairs.push((seq_idx, kv_head));
                }
            }
            for chunk in pairs.chunks(max_seqs_per_inst) {
                let num_entries = (chunk.len() * 2) as i32;
                let mut words = vec![OPCODE_ATTENTION_DECODE, layer, num_entries];
                for &(seq, kv) in chunk {
                    words.push(seq);
                    words.push(kv);
                }
                push_inst(&mut out, &words);
            }
        }

        // Op 5: O_ProjResidual — matmul format:
        // [opcode, layer, local_row, local_col, row, col]
        for batch_block in 0..num_batch_blocks as i32 {
            for o_block in 0..num_output_blocks as i32 {
                push_inst(&mut out, &[
                    OPCODE_O_PROJ_RESIDUAL,
                    layer,
                    batch_block, o_block,
                    batch_block, o_block,
                ]);
            }
        }

        // Op 6: MlpNorm — [opcode, layer, num_items, batch_idx_0, ...]
        // Must emit for ALL padded tokens so barriers reach the expected count.
        for bidx in 0..batch_size as i32 {
            push_inst(&mut out, &[OPCODE_MLP_NORM, layer, 1, bidx]);
        }

        // Op 7: GateSilu — matmul format
        for batch_block in 0..num_batch_blocks as i32 {
            for block in 0..num_intermediate_blocks as i32 {
                push_inst(&mut out, &[
                    OPCODE_GATE_SILU,
                    layer,
                    batch_block, block,
                    batch_block, block,
                ]);
            }
        }

        // Op 8: UpMatmul — matmul format
        for batch_block in 0..num_batch_blocks as i32 {
            for block in 0..num_intermediate_blocks as i32 {
                push_inst(&mut out, &[
                    OPCODE_UP_MATMUL,
                    layer,
                    batch_block, block,
                    batch_block, block,
                ]);
            }
        }

        // Op 9: DownProjResidual — matmul format
        for batch_block in 0..num_batch_blocks as i32 {
            for down_block in 0..num_output_blocks as i32 {
                push_inst(&mut out, &[
                    OPCODE_DOWN_PROJ_RESIDUAL,
                    layer,
                    batch_block, down_block,
                    batch_block, down_block,
                ]);
            }
        }
    }

    // Op 10: LM_HeadNorm — [opcode, layer=0, num_items, batch_idx_0, ...]
    // Must emit for ALL padded tokens so barriers reach the expected count.
    for bidx in 0..batch_size as i32 {
        push_inst(&mut out, &[OPCODE_LM_HEAD_NORM, 0, 1, bidx]);
    }

    // Op 11: LM_Head — matmul format: [opcode, layer=0, local_row, local_col, row, col]
    // Note: for LM_Head, "layer" is actually batch_block (row), matching scheduler.py
    let num_logit_blocks = vocab_size / matmul_out_block_size;
    for batch_block in 0..num_batch_blocks as i32 {
        for logit_block in 0..num_logit_blocks as i32 {
            push_inst(&mut out, &[
                OPCODE_LM_HEAD,
                0, // layer (unused for lm_head)
                batch_block, logit_block,
                batch_block, logit_block,
            ]);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_instruction_generation() {
        let inst = build_throughput_instructions(
            128,   // batch_size (1 block)
            6,     // num_tokens (actual)
            2,     // num_layers
            32,    // num_attention_heads
            8,     // num_kv_heads
            128,   // head_dim
            4096,  // hidden_dim
            14336, // intermediate_dim (divisible by 256)
            32000, // vocab_size
            128,   // matmul_batch_block_size
            256,   // matmul_out_block_size
            None,  // decode mode
        );
        assert!(!inst.is_empty());
        assert!(inst.len() % 32 == 0);
        // First instruction should be AttnNorm (opcode 1)
        assert_eq!(inst[0], OPCODE_ATTN_NORM);
    }

    #[test]
    fn instruction_opcodes_match_llama_cuh() {
        // Verify our opcodes match the #defines in llama.cuh
        assert_eq!(OPCODE_ATTN_NORM, 1);
        assert_eq!(OPCODE_QKV_ROPE_APPEND, 2);
        assert_eq!(OPCODE_ATTENTION_PREFILL, 3);
        assert_eq!(OPCODE_ATTENTION_DECODE, 4);
        assert_eq!(OPCODE_O_PROJ_RESIDUAL, 5);
        assert_eq!(OPCODE_MLP_NORM, 6);
        assert_eq!(OPCODE_GATE_SILU, 7);
        assert_eq!(OPCODE_UP_MATMUL, 8);
        assert_eq!(OPCODE_DOWN_PROJ_RESIDUAL, 9);
        assert_eq!(OPCODE_LM_HEAD_NORM, 10);
        assert_eq!(OPCODE_LM_HEAD, 11);
    }

    #[test]
    fn matmul_instructions_have_six_words() {
        let inst = build_throughput_instructions(
            128, 6, 1, 32, 8, 128, 4096, 14336, 32000, 128, 256, None,
        );
        // Find QKV instruction (after 128 AttnNorm insts for batch_size=128)
        let qkv_start = 128 * INST_WIDTH;
        assert_eq!(inst[qkv_start], OPCODE_QKV_ROPE_APPEND);
        // Should have 6 fields: opcode, layer, local_row, local_col, row, col
        assert_eq!(inst[qkv_start + 1], 0); // layer
        assert_eq!(inst[qkv_start + 2], 0); // local_row
        assert_eq!(inst[qkv_start + 3], 0); // local_col
        assert_eq!(inst[qkv_start + 4], 0); // row (same for single-GPU)
        assert_eq!(inst[qkv_start + 5], 0); // col (same for single-GPU)
    }
}
