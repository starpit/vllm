// SPDX-License-Identifier: Apache-2.0
//! BitsAndBytes quantization config types.

/// Quantization type for BitsAndBytes 4-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BnbQuantType {
    NF4,
    FP4,
}

/// Layer-level BnB NF4 configuration.
#[derive(Debug, Clone)]
pub struct BnbNf4Config {
    pub quant_type: BnbQuantType,
    pub blocksize: usize,
    pub double_quant: bool,
}

impl Default for BnbNf4Config {
    fn default() -> Self {
        Self {
            quant_type: BnbQuantType::NF4,
            blocksize: 64,
            double_quant: false,
        }
    }
}

/// Layer-level BnB configuration — either NF4 or INT8.
#[derive(Debug, Clone)]
pub enum BnbLayerConfig {
    Nf4(BnbNf4Config),
    Int8,
}
