// SPDX-License-Identifier: Apache-2.0
//! Simplified in-browser inference engine.
//!
//! Wraps `WgpuWorker` with prefill/decode state management and stats
//! for the gears panel.

use serde::{Deserialize, Serialize};

use vllm_wgpu::model::{ModelConfig, WgpuWorker};

/// Stats exposed to the gears panel via JS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GearsStats {
    pub tokens_per_sec: f64,
    pub seq_position: usize,
    pub kv_cache_used: usize,
    pub kv_cache_total: usize,
    pub gpu_memory_bytes: usize,
    pub layer_times_ms: Vec<f64>,
    pub last_token: Option<String>,
    pub last_token_prob: f64,
}

impl Default for GearsStats {
    fn default() -> Self {
        Self {
            tokens_per_sec: 0.0,
            seq_position: 0,
            kv_cache_used: 0,
            kv_cache_total: 0,
            gpu_memory_bytes: 0,
            layer_times_ms: Vec::new(),
            last_token: None,
            last_token_prob: 0.0,
        }
    }
}

/// The in-browser engine holding a loaded model and generating tokens.
pub struct BrowserEngine {
    pub worker: WgpuWorker,
    pub config: ModelConfig,
    pub stats: GearsStats,
    pub token_ids: Vec<u32>,
    pub max_gen_len: usize,
    pub prefill_pos: usize,
}

impl BrowserEngine {
    pub fn new(worker: WgpuWorker, config: ModelConfig) -> Self {
        let kv_total = config.max_position_embeddings;
        Self {
            worker,
            config,
            stats: GearsStats {
                kv_cache_total: kv_total,
                ..Default::default()
            },
            token_ids: Vec::new(),
            max_gen_len: 512,
            prefill_pos: 0,
        }
    }

    /// Prefill: process all prompt tokens to build KV cache.
    pub async fn prefill(&mut self) -> Result<(), String> {
        self.prefill_pos = 0;
        for i in 0..self.token_ids.len() {
            let _next = self
                .worker
                .forward_one(self.token_ids[i], i)
                .await
                .map_err(|e| format!("{e}"))?;
            self.prefill_pos = i + 1;
        }
        self.stats.seq_position = self.token_ids.len();
        self.stats.kv_cache_used = self.token_ids.len();
        Ok(())
    }

    /// Decode one token.
    pub async fn step(&mut self) -> Result<u32, String> {
        let pos = self.prefill_pos;
        let input_token = self.token_ids.last().copied().unwrap_or(1);
        let next_token = self
            .worker
            .forward_one(input_token, pos)
            .await
            .map_err(|e| format!("{e}"))?;

        self.token_ids.push(next_token);
        self.prefill_pos = pos + 1;
        self.stats.seq_position = self.token_ids.len();
        self.stats.kv_cache_used = self.token_ids.len();
        Ok(next_token)
    }
}
