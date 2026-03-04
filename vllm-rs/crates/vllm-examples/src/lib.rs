// SPDX-License-Identifier: Apache-2.0
//! In-browser LLM chat with WebGPU.
//!
//! This is the WASM entry point. It exports functions to JavaScript for:
//! - Initializing the WebGPU device
//! - Loading a model from HuggingFace
//! - Running chat inference
//! - Reading gears panel stats

pub mod engine;
pub mod worker;

use std::cell::RefCell;

use wasm_bindgen::prelude::*;
use web_sys::console;

use engine::{BrowserEngine, GearsStats};
use vllm_wgpu::model::{ModelConfig, WgpuWorker};

thread_local! {
    /// Global engine state (single-threaded WASM — `RefCell` is safe).
    static ENGINE: RefCell<Option<BrowserEngine>> = const { RefCell::new(None) };
}

fn with_engine<R>(f: impl FnOnce(&mut BrowserEngine) -> R) -> Result<R, JsValue> {
    ENGINE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let engine = borrow
            .as_mut()
            .ok_or_else(|| JsValue::from_str("not initialized"))?;
        Ok(f(engine))
    })
}

fn with_engine_ref<R>(f: impl FnOnce(&BrowserEngine) -> R) -> Result<R, JsValue> {
    ENGINE.with(|cell| {
        let borrow = cell.borrow();
        let engine = borrow
            .as_ref()
            .ok_or_else(|| JsValue::from_str("not initialized"))?;
        Ok(f(engine))
    })
}

fn default_config() -> ModelConfig {
    ModelConfig {
        hidden_size: 576,
        num_attention_heads: 9,
        num_key_value_heads: 3,
        num_hidden_layers: 30,
        intermediate_size: 1536,
        vocab_size: 49152,
        max_position_embeddings: 2048,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
    }
}

/// Initialize the WebGPU device. Must be called first.
#[wasm_bindgen]
pub async fn init_device() -> Result<(), JsValue> {
    let device = vllm_wgpu::WgpuDevice::new()
        .await
        .map_err(|e| JsValue::from_str(&format!("WebGPU init failed: {e}")))?;
    console::log_1(&"WebGPU device initialized".into());

    let config = default_config();
    ENGINE.with(|cell| {
        *cell.borrow_mut() = Some(BrowserEngine::new(
            WgpuWorker::new(device, config.clone()),
            config,
        ));
    });
    Ok(())
}

/// Load a model by parsing its config.json and fetching weights.
#[wasm_bindgen]
pub async fn load_model(config_json: &str) -> Result<(), JsValue> {
    let config: ModelConfig = serde_json::from_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Failed to parse config.json: {e}")))?;

    console::log_1(
        &format!(
            "Model config: {}L, {}H, vocab={}",
            config.num_hidden_layers, config.hidden_size, config.vocab_size
        )
        .into(),
    );

    with_engine(|engine| {
        let device = engine.worker.device.clone();
        engine.config = config.clone();
        engine.worker = WgpuWorker::new(device, config.clone());
        engine
            .worker
            .init_rope_cache()
            .map_err(|e| JsValue::from_str(&format!("RoPE cache init failed: {e}")))
            .unwrap();
        engine.stats.kv_cache_total = config.max_position_embeddings;
    })?;

    console::log_1(&"Model config loaded. Waiting for weights...".into());
    Ok(())
}

/// Generate the next token given the current context.
#[wasm_bindgen]
pub async fn generate_next() -> Result<u32, JsValue> {
    // Extract what we need, then do the async work outside the borrow
    let (_last_token, _pos) = with_engine(|engine| {
        let pos = engine.token_ids.len();
        let last = engine.token_ids.last().copied().unwrap_or(1);
        (last, pos)
    })?;

    // forward_one needs &mut worker — we must borrow mutably for the async call.
    // Since WASM is single-threaded and we won't re-enter, we take the engine out
    // temporarily.
    let mut engine_taken = ENGINE
        .with(|cell| cell.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("not initialized"))?;

    let result = engine_taken.step().await;

    ENGINE.with(|cell| *cell.borrow_mut() = Some(engine_taken));

    result.map_err(|e| JsValue::from_str(&e))
}

/// Set the initial prompt token IDs.
#[wasm_bindgen]
pub fn set_prompt(token_ids: &[u32]) {
    with_engine(|engine| {
        engine.token_ids = token_ids.to_vec();
    })
    .unwrap();
}

/// Get current gears stats as a JSON string.
#[wasm_bindgen]
pub fn get_stats() -> String {
    with_engine_ref(|engine| serde_json::to_string(&engine.stats).unwrap_or_default())
        .unwrap_or_default()
}

/// Get the current token sequence.
#[wasm_bindgen]
pub fn get_tokens() -> Vec<u32> {
    with_engine_ref(|engine| engine.token_ids.clone()).unwrap_or_default()
}

/// Reset the generation state.
#[wasm_bindgen]
pub fn reset() {
    with_engine(|engine| {
        engine.token_ids.clear();
        engine.prefill_pos = 0;
        engine.stats = GearsStats::default();
        engine.stats.kv_cache_total = engine.config.max_position_embeddings;
    })
    .unwrap();
}
