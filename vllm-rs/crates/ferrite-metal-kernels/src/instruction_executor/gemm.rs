// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! GEMM instruction recording for Metal ICB.

use super::{Buffer, RecordingContext};

pub fn record_gemm(
    _ctx: &mut RecordingContext,
    _input_buffer: &Buffer,
    _weight_buffer: &Buffer,
    _output_buffer: &Buffer,
    _m: usize,
    _k: usize,
    _n: usize,
    _dtype: &str,
) -> Result<(), String> {
    Err("GEMM ICB recording not yet implemented".to_string())
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "GEMM recording not yet implemented"]
    fn test_gemm_recording() {}
}
