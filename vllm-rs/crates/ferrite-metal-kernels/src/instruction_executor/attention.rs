// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Attention instruction recording for Metal ICB.

use super::{Buffer, RecordingContext};

pub fn record_attention(
    _ctx: &mut RecordingContext,
    _q_buffer: &Buffer,
    _k_cache: &Buffer,
    _v_cache: &Buffer,
    _output_buffer: &Buffer,
    _num_tokens: usize,
    _num_heads: usize,
    _head_dim: usize,
    _max_seq_len: usize,
) -> Result<(), String> {
    Err("Attention ICB recording not yet implemented".to_string())
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "Attention recording not yet implemented"]
    fn test_attention_recording() {}
}
