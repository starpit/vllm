// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Qwen2-VL / Qwen2.5-VL multimodal E2E tests.
//!
//! Validates that Qwen2-VL models can load, serve text-only requests,
//! and process image+text requests end-to-end.
//!
//! Candle: `unsloth/Qwen2-VL-2B-Instruct` (~3.8 GB BF16)
//! MLX: `mlx-community/Qwen2-VL-7B-4bit` (~4.6 GB 4-bit)
//!
//! Run with:
//! ```bash
//! # Candle (CPU/CUDA):
//! cargo test -p vllm-e2e --features e2e --test e_qwen2_vl -- --ignored --test-threads=1
//! # MLX (Apple Silicon):
//! cargo test -p vllm-e2e --features e2e,metal --test e_qwen2_vl -- --ignored --test-threads=1
//! ```

#![cfg(feature = "e2e")]

use vllm_e2e::assertions::{assert_coherent_text, assert_valid_chat_response, assert_valid_stream};
use vllm_e2e::{Client, TestModels, TestServer};
use vllm_serve::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pre-computed 4x4 red PNG as base64.
const TINY_RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg==";

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

fn user_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(serde_json::Value::String(content.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn user_msg_with_image(text: &str, image_base64: &str) -> ChatCompletionMessageParam {
    let content = serde_json::json!([
        {
            "type": "image_url",
            "image_url": {
                "url": format!("data:image/png;base64,{}", image_base64)
            }
        },
        {
            "type": "text",
            "text": text
        }
    ]);
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(content),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

async fn start_qwen2_vl() -> (TestServer, Client) {
    // Use MLX model on metal, Candle model otherwise.
    let model = if cfg!(feature = "metal") {
        TestModels::QWEN2_VL_7B_4BIT
    } else {
        TestModels::QWEN2_VL_2B_INSTRUCT
    };
    let server = TestServer::builder(model)
        .start()
        .await
        .expect("Qwen2-VL server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// ===========================================================================
// Tests
// ===========================================================================

/// Server starts with Qwen2VLForConditionalGeneration architecture.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_server_starts() {
    let (server, client) = start_qwen2_vl().await;

    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("Qwen2-VL"),
        "model name should contain 'Qwen2-VL', got: {}",
        models.data[0].id,
    );

    drop(server);
}

/// Text-only chat works through the VLM (delegates to text backbone).
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_text_only_chat() {
    let (_server, client) = start_qwen2_vl().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is 2 + 2?")],
        max_tokens: Some(32),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 1);
}

/// Chat completion with a base64-encoded image + text prompt.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_image_chat() {
    let (_server, client) = start_qwen2_vl().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "What color is this image?",
            TINY_RED_PNG_BASE64,
        )],
        max_tokens: Some(64),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 1);
}

/// Streaming chat with image.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_image_stream() {
    let (_server, client) = start_qwen2_vl().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image(
            "Describe this image briefly.",
            TINY_RED_PNG_BASE64,
        )],
        max_tokens: Some(32),
        temperature: Some(0.0),
        stream: true,
        ..default_chat_request()
    };

    let chunks = client.chat_completion_stream(&request).await.unwrap();
    assert_valid_stream(&chunks);

    let full_text: String = chunks
        .iter()
        .filter_map(|c| c.choices.first().and_then(|ch| ch.delta.content.as_deref()))
        .collect();
    assert_coherent_text(&full_text, 1);
}

/// max_tokens is respected with image input.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen2_vl_image_max_tokens() {
    let (_server, client) = start_qwen2_vl().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg_with_image("What do you see?", TINY_RED_PNG_BASE64)],
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let completion_tokens = resp.usage.completion_tokens.unwrap_or(0);
    assert!(
        completion_tokens <= 5,
        "completion_tokens ({completion_tokens}) should be <= 5",
    );
}
