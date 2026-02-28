// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Phase E1: Basic serving smoke tests.
//!
//! Validates that each model architecture loads, starts serving,
//! and generates coherent text.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e1_basic_serving -- --ignored`

#![cfg(feature = "e2e")]

use vllm_e2e::assertions::{
    assert_coherent_text, assert_valid_chat_response, assert_valid_completion_response,
};
use vllm_e2e::{Client, TestModels, TestServer};
use vllm_serve::protocol::{
    ChatCompletionMessageParam, ChatCompletionRequest, CompletionPrompt, CompletionRequest,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn user_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "user".to_string(),
        content: Some(serde_json::Value::String(content.to_string())),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn simple_chat_request(content: &str, max_tokens: Option<u32>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![user_msg(content)],
        max_tokens,
        temperature: Some(0.0),
        ..default_chat_request()
    }
}

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

fn simple_completion_request(prompt: &str, max_tokens: u32) -> CompletionRequest {
    CompletionRequest {
        prompt: Some(CompletionPrompt::Single(prompt.to_string())),
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        ..default_completion_request()
    }
}

fn default_completion_request() -> CompletionRequest {
    serde_json::from_str(r#"{}"#).unwrap()
}

// ===========================================================================
// SmolLM-135M-Instruct-4bit (LlamaForCausalLM, quantized) — Tier 1
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_smollm_server_starts() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_4BIT)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    // /health
    assert!(client.health().await.unwrap(), "server should be healthy");

    // /v1/models
    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("SmolLM"),
        "model name should contain 'SmolLM', got: {}",
        models.data[0].id
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_smollm_chat_basic() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_smollm_completion_basic() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_smollm_max_tokens() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Write a long story about a cat.", Some(5));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) <= 5,
        "completion_tokens should be <= 5, got: {:?}",
        resp.usage.completion_tokens
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_smollm_version() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let version = client.version().await.unwrap();
    assert!(!version.version.is_empty(), "version should not be empty");
    assert!(
        version.version.contains("rust"),
        "version should contain 'rust', got: {}",
        version.version
    );
}

// ===========================================================================
// Qwen2.5-0.5B-Instruct-4bit (Qwen2ForCausalLM) — Tier 1
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_qwen2_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_4BIT)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_qwen2_chat_basic() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_qwen2_completion_basic() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(!resp.choices[0].text.is_empty());
}

// ===========================================================================
// Qwen3-0.6B-4bit (Qwen3ForCausalLM) — Tier 1
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_qwen3_server_starts() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_4BIT)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_qwen3_chat_basic() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Llama-3.2-1B-Instruct-4bit (LlamaForCausalLM) — Tier 2
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_llama3_server_starts() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_4BIT)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_llama3_chat_basic() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_llama3_completion_basic() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(!resp.choices[0].text.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_llama3_max_tokens() {
    let server = TestServer::builder(TestModels::LLAMA_3_2_1B_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Write a long story.", Some(5));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(resp.usage.completion_tokens.unwrap_or(0) <= 5);
}

// ===========================================================================
// Nightly: Gemma2-2B (Gemma2ForCausalLM) — Tier 3
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_gemma2_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_4BIT)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_gemma2_chat_basic() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Nightly: Phi-3.5-mini (Phi3ForCausalLM) — Tier 3
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_phi3_server_starts() {
    let server = TestServer::builder(TestModels::PHI3_5_MINI_4BIT)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_phi3_chat_basic() {
    let server = TestServer::builder(TestModels::PHI3_5_MINI_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Weekly: Mistral-7B (MistralForCausalLM) — Tier 4
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t4_mistral_server_starts() {
    let server = TestServer::builder(TestModels::MISTRAL_7B_4BIT)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t4_mistral_chat_basic() {
    let server = TestServer::builder(TestModels::MISTRAL_7B_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// Weekly: DeepSeek-V2-Lite (DeepseekV2ForCausalLM) — Tier 4
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t4_deepseek_server_starts() {
    let server = TestServer::builder(TestModels::DEEPSEEK_V2_LITE_4BIT)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t4_deepseek_chat_basic() {
    let server = TestServer::builder(TestModels::DEEPSEEK_V2_LITE_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

// ===========================================================================
// E1b: Float16 vs quantized comparison
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_float16_server_starts() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_F16)
        .start()
        .await
        .expect("float16 server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t1_float16_chat_basic() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_F16)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}
