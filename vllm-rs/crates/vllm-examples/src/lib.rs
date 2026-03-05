// SPDX-License-Identifier: Apache-2.0
//! In-browser LLM chat with WebGPU — all logic in Rust/WASM.
//!
//! The entire application runs from `#[wasm_bindgen(start)]`:
//! - DOM wiring (event listeners, UI updates)
//! - Model downloading from HuggingFace CDN with progress
//! - Tokenization via `tokenizers` crate
//! - Inference via `WgpuWorker`
//!
//! The HTML page only needs:
//! ```html
//! <script type="module">
//!   import init from './pkg/vllm_examples.js';
//!   await init();
//! </script>
//! ```

pub mod chat;
pub mod engine;
pub mod fetch;
pub mod worker;

use std::cell::RefCell;

use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use web_sys::console;

use chat::ChatFormat;
use engine::{BrowserEngine, GearsStats};
use vllm_wgpu::WgpuDevice;
use vllm_wgpu::gguf::GgufReader;
use vllm_wgpu::model::{ModelConfig, WgpuWorker};

// ---------------------------------------------------------------------------
// Global state (single-threaded WASM — RefCell is safe)
// ---------------------------------------------------------------------------

struct AppState {
    device: WgpuDevice,
    engine: Option<BrowserEngine>,
    tokenizer: Option<Tokenizer>,
    chat_format: ChatFormat,
    generating: bool,
    model_id: String,
}

thread_local! {
    static STATE: RefCell<Option<AppState>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut AppState) -> R) -> Result<R, JsValue> {
    STATE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let state = borrow
            .as_mut()
            .ok_or_else(|| JsValue::from_str("not initialized"))?;
        Ok(f(state))
    })
}

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

fn document() -> web_sys::Document {
    web_sys::window().unwrap().document().unwrap()
}

fn get_el(id: &str) -> web_sys::HtmlElement {
    document()
        .get_element_by_id(id)
        .unwrap_or_else(|| panic!("missing element: {id}"))
        .dyn_into()
        .unwrap()
}

fn set_text(id: &str, text: &str) {
    get_el(id).set_inner_text(text);
}

fn set_status(text: &str) {
    set_text("load-status", text);
}

fn add_message(role: &str, text: &str) -> web_sys::HtmlElement {
    let doc = document();
    let messages = get_el("messages");

    let div = doc.create_element("div").unwrap();
    div.set_class_name(&format!("message {role}"));

    let role_el = doc.create_element("div").unwrap();
    role_el.set_class_name("role");
    role_el.set_text_content(Some(role));
    div.append_child(&role_el).unwrap();

    let content = doc
        .create_element("div")
        .unwrap()
        .dyn_into::<web_sys::HtmlElement>()
        .unwrap();
    content.set_text_content(Some(text));
    div.append_child(&content).unwrap();

    messages.append_child(&div).unwrap();
    messages.set_scroll_top(messages.scroll_height());

    content
}

// ---------------------------------------------------------------------------
// Boot: #[wasm_bindgen(start)]
// ---------------------------------------------------------------------------

#[wasm_bindgen(start)]
pub async fn main() -> Result<(), JsValue> {
    console::log_1(&"vLLM WebGPU: initializing...".into());

    // 1. Init WebGPU device
    set_status("Initializing WebGPU...");
    let device = WgpuDevice::new()
        .await
        .map_err(|e| JsValue::from_str(&format!("WebGPU init failed: {e}")))?;
    console::log_1(&"WebGPU device initialized".into());

    // 2. Store global state
    STATE.with(|cell| {
        *cell.borrow_mut() = Some(AppState {
            device,
            engine: None,
            tokenizer: None,
            chat_format: ChatFormat::Raw,
            generating: false,
            model_id: String::new(),
        });
    });

    set_status("Ready — select a model and click Load");

    // 3. Wire up event listeners
    wire_load_button();
    wire_send_button();
    wire_stop_button();
    wire_input_enter();

    Ok(())
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

fn wire_load_button() {
    let btn = get_el("load-btn");
    let closure = Closure::wrap(Box::new(move || {
        wasm_bindgen_futures::spawn_local(load_model());
    }) as Box<dyn Fn()>);
    btn.set_onclick(Some(closure.as_ref().unchecked_ref()));
    closure.forget();
}

fn wire_send_button() {
    let btn = get_el("send-btn");
    let closure = Closure::wrap(Box::new(move || {
        wasm_bindgen_futures::spawn_local(start_generation());
    }) as Box<dyn Fn()>);
    btn.set_onclick(Some(closure.as_ref().unchecked_ref()));
    closure.forget();
}

fn wire_stop_button() {
    let btn = get_el("stop-btn");
    let closure = Closure::wrap(Box::new(move || {
        let _ = with_state(|s| s.generating = false);
    }) as Box<dyn Fn()>);
    btn.set_onclick(Some(closure.as_ref().unchecked_ref()));
    closure.forget();
}

fn wire_input_enter() {
    let textarea = get_el("user-input");
    let closure = Closure::wrap(Box::new(move |e: web_sys::KeyboardEvent| {
        if e.key() == "Enter" && !e.shift_key() {
            e.prevent_default();
            let send_btn = get_el("send-btn");
            // Check if send button is enabled by checking the disabled property
            let btn: &web_sys::HtmlButtonElement = send_btn.unchecked_ref();
            if !btn.disabled() {
                wasm_bindgen_futures::spawn_local(start_generation());
            }
        }
    }) as Box<dyn Fn(web_sys::KeyboardEvent)>);
    textarea
        .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())
        .unwrap();
    closure.forget();
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

async fn load_model() {
    if let Err(e) = load_model_inner().await {
        let msg = e.as_string().unwrap_or_else(|| format!("{e:?}"));
        set_status(&format!("Load failed: {msg}"));
        console::error_1(&e);
    }
}

async fn load_model_inner() -> Result<(), JsValue> {
    let select: web_sys::HtmlSelectElement = get_el("model-select").unchecked_into();
    let model_id = select.value();

    // Disable load button during loading
    let load_btn: web_sys::HtmlButtonElement = get_el("load-btn").unchecked_into();
    load_btn.set_disabled(true);

    set_status(&format!("Fetching config for {model_id}..."));

    // 1. Fetch config.json
    let config_url = fetch::hf_url(&model_id, "config.json");
    let config_text = fetch::fetch_text(&config_url).await?;
    let config: ModelConfig = serde_json::from_str(&config_text)
        .map_err(|e| JsValue::from_str(&format!("parse config.json: {e}")))?;
    console::log_1(
        &format!(
            "Config: {}L, {}H, vocab={}",
            config.num_hidden_layers, config.hidden_size, config.vocab_size
        )
        .into(),
    );

    // 2. Fetch tokenizer.json
    set_status("Fetching tokenizer...");
    let tok_url = fetch::hf_url(&model_id, "tokenizer.json");
    let tok_text = fetch::fetch_text(&tok_url).await?;
    let tokenizer = Tokenizer::from_bytes(tok_text.as_bytes())
        .map_err(|e| JsValue::from_str(&format!("parse tokenizer: {e}")))?;

    // 3. Fetch tokenizer_config.json (best-effort for chat template)
    let chat_format =
        match fetch::fetch_text(&fetch::hf_url(&model_id, "tokenizer_config.json")).await {
            Ok(text) => chat::detect_format(&text),
            Err(_) => ChatFormat::Raw,
        };
    console::log_1(&format!("Chat format: {:?}", chat_format).into());

    // 4. Create worker with config
    let device = with_state(|s| s.device.clone())?;
    let mut worker = WgpuWorker::new(device, config.clone());
    worker
        .init_rope_cache()
        .map_err(|e| JsValue::from_str(&format!("RoPE init: {e}")))?;

    // 5. Determine weight files and fetch them
    // Try single model.safetensors first; if that 404s, try sharded via index.json;
    // if that also fails, try GGUF.
    let weights_loaded = load_safetensors_weights(&model_id, &mut worker).await;
    if let Err(st_err) = weights_loaded {
        console::log_1(&format!("Safetensors failed ({st_err:?}), trying GGUF...").into());
        load_gguf_weights(&model_id, &mut worker).await?;
    }

    // 6. Store everything
    let kv_total = config.max_position_embeddings;
    let engine = BrowserEngine::new(worker, config);

    with_state(|s| {
        s.engine = Some(engine);
        s.tokenizer = Some(tokenizer);
        s.chat_format = chat_format;
        s.model_id = model_id.clone();
    })?;

    // Enable send button
    let send_btn: web_sys::HtmlButtonElement = get_el("send-btn").unchecked_into();
    send_btn.set_disabled(false);
    load_btn.set_disabled(false);

    set_status(&format!("{model_id} loaded"));
    set_text("stat-kv", &format!("0 / {kv_total}"));

    Ok(())
}

async fn load_safetensors_weights(model_id: &str, worker: &mut WgpuWorker) -> Result<(), JsValue> {
    // Try single shard first
    let single_url = fetch::hf_url(model_id, "model.safetensors");
    if let Ok(data) = fetch_shard_with_progress(&single_url, "model.safetensors", 1, 1).await {
        set_status("Uploading weights to GPU...");
        worker
            .load_weights_from_bytes(&[&data])
            .map_err(|e| JsValue::from_str(&e))?;
        return Ok(());
    }

    // Try sharded via index.json
    let index_url = fetch::hf_url(model_id, "model.safetensors.index.json");
    let index_text = fetch::fetch_text(&index_url).await?;
    let index: serde_json::Value = serde_json::from_str(&index_text)
        .map_err(|e| JsValue::from_str(&format!("parse index: {e}")))?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| JsValue::from_str("no weight_map in index.json"))?;

    let mut shard_names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    shard_names.sort();
    shard_names.dedup();

    let total_shards = shard_names.len();
    let mut all_shard_data: Vec<Vec<u8>> = Vec::with_capacity(total_shards);

    for (i, shard_name) in shard_names.iter().enumerate() {
        let url = fetch::hf_url(model_id, shard_name);
        let data = fetch_shard_with_progress(&url, shard_name, i + 1, total_shards).await?;
        all_shard_data.push(data);
    }

    set_status("Uploading weights to GPU...");
    let shard_refs: Vec<&[u8]> = all_shard_data.iter().map(|v| v.as_slice()).collect();
    worker
        .load_weights_from_bytes(&shard_refs)
        .map_err(|e| JsValue::from_str(&e))?;

    Ok(())
}

async fn load_gguf_weights(model_id: &str, worker: &mut WgpuWorker) -> Result<(), JsValue> {
    // Find GGUF files by checking the HF API
    let api_url = format!("https://huggingface.co/api/models/{model_id}");
    let api_text = fetch::fetch_text(&api_url).await?;
    let api: serde_json::Value = serde_json::from_str(&api_text)
        .map_err(|e| JsValue::from_str(&format!("parse HF API: {e}")))?;

    let siblings = api
        .get("siblings")
        .and_then(|v| v.as_array())
        .ok_or_else(|| JsValue::from_str("no siblings in HF API response"))?;

    let mut gguf_files: Vec<String> = siblings
        .iter()
        .filter_map(|s| s.get("rfilename")?.as_str().map(|s| s.to_string()))
        .filter(|name| name.ends_with(".gguf"))
        .collect();

    if gguf_files.is_empty() {
        return Err(JsValue::from_str(
            "no .safetensors or .gguf files found in repo",
        ));
    }

    // Prefer Q4_0 for bandwidth
    let preferred = ["Q4_0", "Q4_K_M", "Q4_K_S", "Q8_0", "F16"];
    let chosen = preferred
        .iter()
        .find_map(|pref| gguf_files.iter().find(|f| f.contains(pref)).cloned())
        .unwrap_or_else(|| {
            gguf_files.sort();
            gguf_files.first().unwrap().clone()
        });

    let url = fetch::hf_url(model_id, &chosen);
    let data = fetch_shard_with_progress(&url, &chosen, 1, 1).await?;

    set_status("Parsing GGUF...");
    let gguf =
        GgufReader::parse(&data).map_err(|e| JsValue::from_str(&format!("parse GGUF: {e}")))?;

    // Update worker config from GGUF metadata
    let config = gguf
        .model_config()
        .map_err(|e| JsValue::from_str(&format!("GGUF config: {e}")))?;

    let device = worker.device.clone();
    *worker = WgpuWorker::new(device, config);
    worker
        .init_rope_cache()
        .map_err(|e| JsValue::from_str(&format!("RoPE init: {e}")))?;

    set_status("Uploading GGUF weights to GPU...");
    worker
        .load_weights_gguf(&gguf)
        .map_err(|e| JsValue::from_str(&e))?;

    Ok(())
}

async fn fetch_shard_with_progress(
    url: &str,
    name: &str,
    shard_num: usize,
    total_shards: usize,
) -> Result<Vec<u8>, JsValue> {
    let name = name.to_string();
    let data = fetch::fetch_bytes_with_progress(url, move |loaded, total| {
        let pct = if total > 0 {
            (loaded as f64 / total as f64 * 100.0) as u32
        } else {
            0
        };
        let mb = loaded as f64 / 1024.0 / 1024.0;
        let total_mb = total as f64 / 1024.0 / 1024.0;
        let shard_info = if total_shards > 1 {
            format!(" [{shard_num}/{total_shards}]")
        } else {
            String::new()
        };
        let status = if total > 0 {
            format!("Downloading {name}{shard_info}: {mb:.1} / {total_mb:.1} MB ({pct}%)")
        } else {
            format!("Downloading {name}{shard_info}: {mb:.1} MB")
        };
        set_status(&status);
        // Update progress bar
        if let Some(el) = document().get_element_by_id("download-progress") {
            let pct_str = if total > 0 {
                format!("{pct}%")
            } else {
                "0%".to_string()
            };
            el.dyn_ref::<web_sys::HtmlElement>()
                .map(|h| h.style().set_property("width", &pct_str));
        }
    })
    .await?;
    Ok(data)
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

async fn start_generation() {
    if let Err(e) = generate_inner().await {
        let msg = e.as_string().unwrap_or_else(|| format!("{e:?}"));
        console::error_1(&JsValue::from_str(&msg));
    }
    // Reset UI state
    get_el("send-btn")
        .style()
        .set_property("display", "")
        .unwrap();
    get_el("stop-btn")
        .style()
        .set_property("display", "none")
        .unwrap();
    let _ = with_state(|s| s.generating = false);
}

async fn generate_inner() -> Result<(), JsValue> {
    let is_generating = with_state(|s| s.generating)?;
    if is_generating {
        return Ok(());
    }

    // Get user input
    let textarea: web_sys::HtmlTextAreaElement = get_el("user-input").unchecked_into();
    let prompt = textarea.value().trim().to_string();
    if prompt.is_empty() {
        return Ok(());
    }
    textarea.set_value("");

    with_state(|s| s.generating = true)?;
    get_el("send-btn")
        .style()
        .set_property("display", "none")
        .unwrap();
    get_el("stop-btn")
        .style()
        .set_property("display", "")
        .unwrap();

    add_message("user", &prompt);

    // Apply chat template and tokenize
    let (token_ids, eos_id) = with_state(|s| {
        let tokenizer = s
            .tokenizer
            .as_ref()
            .ok_or_else(|| JsValue::from_str("no tokenizer loaded"))?;
        let formatted = chat::apply_template(&s.chat_format, &prompt);
        let encoding = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| JsValue::from_str(&format!("tokenize: {e}")))?;
        let ids = encoding.get_ids().to_vec();
        let eos = chat::eos_token_id(&s.chat_format, tokenizer);
        Ok::<_, JsValue>((ids, eos))
    })??;

    // Reset engine and set prompt
    // Take engine out for async work (single-threaded WASM, safe to do)
    let mut engine = with_state(|s| {
        s.engine
            .take()
            .ok_or_else(|| JsValue::from_str("no model loaded"))
    })??;

    engine.token_ids = token_ids;
    engine.prefill_pos = 0;
    engine.worker.reset_kv();
    engine.stats = GearsStats {
        kv_cache_total: engine.config.max_position_embeddings,
        ..Default::default()
    };

    // Prefill
    set_status("Prefilling...");
    engine
        .prefill()
        .await
        .map_err(|e| JsValue::from_str(&format!("prefill: {e}")))?;

    let content_el = add_message("assistant", "");
    let mut generated_text = String::new();
    let perf = web_sys::window().unwrap().performance().unwrap();
    let start_time = perf.now();
    let mut token_count = 0u32;

    // Decode loop
    loop {
        let still_generating = with_state(|s| s.generating)?;
        if !still_generating || token_count >= 512 {
            break;
        }

        match engine.step().await {
            Ok(token_id) => {
                token_count += 1;

                if token_id == eos_id || token_id == 0 {
                    break;
                }

                // Decode token to text
                let text = with_state(|s| {
                    s.tokenizer
                        .as_ref()
                        .unwrap()
                        .decode(&[token_id], false)
                        .unwrap_or_default()
                })?;

                generated_text.push_str(&text);
                content_el.set_text_content(Some(&generated_text));

                // Scroll messages
                let messages = get_el("messages");
                messages.set_scroll_top(messages.scroll_height());

                // Update stats
                let elapsed = (perf.now() - start_time) / 1000.0;
                if elapsed > 0.0 {
                    engine.stats.tokens_per_sec = token_count as f64 / elapsed;
                }
                engine.stats.last_token = Some(text);
                update_stats_display(&engine.stats);

                // Yield to browser
                yield_to_browser().await;
            }
            Err(e) => {
                generated_text.push_str(&format!("\n[Error: {e}]"));
                content_el.set_text_content(Some(&generated_text));
                break;
            }
        }
    }

    // Final stats update
    update_stats_display(&engine.stats);
    let elapsed = (perf.now() - start_time) / 1000.0;
    set_status(&format!(
        "Done: {} tokens in {:.1}s ({:.1} tok/s)",
        token_count,
        elapsed,
        if elapsed > 0.0 {
            token_count as f64 / elapsed
        } else {
            0.0
        }
    ));

    // Put engine back
    with_state(|s| {
        s.engine = Some(engine);
        s.generating = false;
    })?;

    Ok(())
}

fn update_stats_display(stats: &GearsStats) {
    set_text("stat-tps", &format!("{:.1}", stats.tokens_per_sec));
    set_text("stat-position", &stats.seq_position.to_string());
    set_text(
        "stat-last-token",
        stats.last_token.as_deref().unwrap_or("\u{2014}"),
    );
    let prob_text = if stats.last_token_prob > 0.0 {
        format!("{:.1}%", stats.last_token_prob * 100.0)
    } else {
        "\u{2014}".to_string()
    };
    set_text("stat-prob", &prob_text);
    set_text(
        "stat-kv",
        &format!("{} / {}", stats.kv_cache_used, stats.kv_cache_total),
    );

    // KV cache bar
    if let Some(el) = document().get_element_by_id("cache-bar") {
        let pct = if stats.kv_cache_total > 0 {
            stats.kv_cache_used as f64 / stats.kv_cache_total as f64 * 100.0
        } else {
            0.0
        };
        el.dyn_ref::<web_sys::HtmlElement>()
            .map(|h| h.style().set_property("width", &format!("{pct:.0}%")));
    }

    // GPU memory
    if stats.gpu_memory_bytes > 0 {
        set_text(
            "stat-memory",
            &format!("{:.1} MB", stats.gpu_memory_bytes as f64 / 1024.0 / 1024.0),
        );
    }

    // Layer times
    if !stats.layer_times_ms.is_empty() {
        let html: String = stats
            .layer_times_ms
            .iter()
            .enumerate()
            .map(|(i, t)| {
                format!(
                    "<div class=\"layer-time-row\"><span>L{i}</span><span>{t:.1} ms</span></div>"
                )
            })
            .collect();
        get_el("layer-times").set_inner_html(&html);
    }
}

/// Yield to the browser event loop (equivalent to `setTimeout(0)`).
async fn yield_to_browser() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
            .unwrap();
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}
