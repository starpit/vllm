// SPDX-License-Identifier: Apache-2.0
//! AWQ (Activation-aware Weight Quantization) config.

/// AWQ quantization parameters parsed from `quant_config.json`.
#[derive(Debug, Clone)]
pub struct AwqConfig {
    pub bits: usize,
    pub group_size: usize,
    pub zero_point: bool,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            zero_point: true,
        }
    }
}
