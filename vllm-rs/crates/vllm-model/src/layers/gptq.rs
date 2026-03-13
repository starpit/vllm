// SPDX-License-Identifier: Apache-2.0
//! GPTQ quantization config.

/// GPTQ quantization parameters parsed from `quantize_config.json`.
#[derive(Debug, Clone)]
pub struct GptqConfig {
    pub bits: usize,
    pub group_size: usize,
    pub desc_act: bool,
    pub sym: bool,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            desc_act: false,
            sym: true,
        }
    }
}
