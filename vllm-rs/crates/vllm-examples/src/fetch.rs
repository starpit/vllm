// SPDX-License-Identifier: Apache-2.0
//! HuggingFace CDN fetch helpers for WASM.
//!
//! All networking uses `web_sys::fetch()` — no JS glue needed.

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

/// Build a HuggingFace CDN URL for a file in a model repo.
pub fn hf_url(model_id: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        model_id, filename
    )
}

/// Fetch a URL and return the response body as a String.
pub async fn fetch_text(url: &str) -> Result<String, JsValue> {
    let resp = fetch_response(url).await?;
    let text = JsFuture::from(resp.text()?).await?;
    text.as_string()
        .ok_or_else(|| JsValue::from_str("response body is not a string"))
}

/// Fetch a URL and return the response body as bytes.
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let resp = fetch_response(url).await?;
    let ab = JsFuture::from(resp.array_buffer()?).await?;
    let u8arr = js_sys::Uint8Array::new(&ab);
    Ok(u8arr.to_vec())
}

/// Fetch a URL with streaming progress reporting.
///
/// Calls `on_progress(bytes_loaded, total_bytes)` as chunks arrive.
/// `total_bytes` is 0 if the server doesn't send Content-Length.
pub async fn fetch_bytes_with_progress(
    url: &str,
    on_progress: impl Fn(u64, u64),
) -> Result<Vec<u8>, JsValue> {
    let resp = fetch_response(url).await?;

    let total: u64 = resp
        .headers()
        .get("content-length")
        .ok()
        .flatten()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let body = resp
        .body()
        .ok_or_else(|| JsValue::from_str("response has no body"))?;
    let reader = body
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()?;

    let mut buf = if total > 0 {
        Vec::with_capacity(total as usize)
    } else {
        Vec::new()
    };

    loop {
        let result = JsFuture::from(reader.read()).await?;
        let done = js_sys::Reflect::get(&result, &JsValue::from_str("done"))?
            .as_bool()
            .unwrap_or(true);
        if done {
            break;
        }
        let chunk = js_sys::Reflect::get(&result, &JsValue::from_str("value"))?;
        let u8arr = js_sys::Uint8Array::new(&chunk);
        let len = u8arr.length() as usize;
        let offset = buf.len();
        buf.resize(offset + len, 0);
        u8arr.copy_to(&mut buf[offset..]);
        on_progress(buf.len() as u64, total);
    }

    Ok(buf)
}

/// Internal: perform a fetch and return the Response, checking for HTTP errors.
async fn fetch_response(url: &str) -> Result<Response, JsValue> {
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)?;
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("no window"))?;
    let resp_val = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_val.dyn_into()?;

    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "HTTP {} fetching {}",
            resp.status(),
            url
        )));
    }
    Ok(resp)
}
