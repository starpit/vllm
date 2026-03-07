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

fn assistant_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "assistant".to_string(),
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
// Gemma3-270M (Gemma3ForCausalLM, quantized) — Tier 2
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_gemma3_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA3_270M_4BIT)
        .start()
        .await
        .expect("Gemma3 server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t2_gemma3_chat_basic() {
    let server = TestServer::builder(TestModels::GEMMA3_270M_4BIT)
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
async fn test_t2_gemma3_completion_basic() {
    let server = TestServer::builder(TestModels::GEMMA3_270M_4BIT)
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
async fn test_t2_gemma3_max_tokens() {
    let server = TestServer::builder(TestModels::GEMMA3_270M_4BIT)
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
// Nightly: Phi-4-mini (Phi3ForCausalLM + LongRoPE) — Tier 3
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_phi4_mini_server_starts() {
    let server = TestServer::builder(TestModels::PHI4_MINI_4BIT)
        .start()
        .await
        .expect("Phi-4 mini server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_t3_phi4_mini_chat_basic() {
    let server = TestServer::builder(TestModels::PHI4_MINI_4BIT)
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
async fn test_t3_phi4_mini_completion_basic() {
    let server = TestServer::builder(TestModels::PHI4_MINI_4BIT)
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
async fn test_t3_phi4_mini_max_tokens() {
    let server = TestServer::builder(TestModels::PHI4_MINI_4BIT)
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
// Qwen3 MoE 4x0.6B (Qwen3MoeForCausalLM) — MoE architecture
// ===========================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen3_moe_server_starts() {
    let server = TestServer::builder(TestModels::QWEN3_MOE_4X06B_4BIT)
        .start()
        .await
        .expect("Qwen3 MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_qwen3_moe_chat_basic() {
    let server = TestServer::builder(TestModels::QWEN3_MOE_4X06B_4BIT)
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
async fn test_qwen3_moe_completion_basic() {
    let server = TestServer::builder(TestModels::QWEN3_MOE_4X06B_4BIT)
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
async fn test_qwen3_moe_max_tokens() {
    let server = TestServer::builder(TestModels::QWEN3_MOE_4X06B_4BIT)
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

// ---------------------------------------------------------------------------
// Sync scheduling path (verify both scheduling modes work)
// ---------------------------------------------------------------------------

/// Smoke test: sync scheduling path still works end-to-end.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_sync_scheduling_smollm_chat() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_4BIT)
        .with_sync_scheduling()
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
// Granite (IBM) — GraniteForCausalLM, MLX 4-bit quantized (~1.3 GB)
// ===========================================================================
// Granite is architecturally identical to LLaMA with 4 scalar multipliers.
// MLX 4-bit quantized model for Apple Silicon testing.
//
// Run with: cargo test -p vllm-e2e --features e2e,metal --release --test e1_basic_serving test_granite -- --ignored

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_granite_server_starts() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_4BIT)
        .start()
        .await
        .expect("Granite server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
    assert!(
        models.data[0].id.contains("granite"),
        "model name should contain 'granite', got: {}",
        models.data[0].id
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_granite_completion() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_4BIT)
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

/// Granite is a chat/instruct model (EOS=token 0) — bare completions may
/// immediately stop. Test with a second completion prompt for variety.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_granite_completion_coherent() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_4BIT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("Once upon a time", 30);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "completion should not be empty"
    );
}

// ===========================================================================
// CUDA E2E tests — safetensors models that actually run on GPU
// ===========================================================================
// These use non-quantized safetensors models (not MLX 4-bit) so that
// model weights load onto the CUDA device and GPU kernels are exercised.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda -- --ignored

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_smollm_server_starts() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("CUDA SmolLM server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_smollm_completion() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
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

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_smollm_chat() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
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

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen2_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
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
    assert_coherent_text(&resp.choices[0].text, 3);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen2_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "response should not be empty");
    assert_coherent_text(text, 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen3_completion() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_CUDA)
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

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen3_chat() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "response should not be empty");
}

// ===========================================================================
// CUDA GGUF E2E tests — quantized GGUF models on GPU
// ===========================================================================
// TODO: Re-enable when vllm-cuda backend supports GGUF quantized models.
// These previously used candle's QCudaStorage for GGUF + CUDA inference.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gguf -- --ignored

/*
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_gemma3_1b_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_GGUF)
        .start()
        .await
        .expect("CUDA Gemma3 1B GGUF server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_gemma3_1b_completion() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_GGUF)
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

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_gemma3_1b_chat() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    // Note: small quantized GGUF models may generate low-quality text;
    // we only assert the response structure is valid.
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) > 0,
        "should generate at least one token"
    );
}

// Qwen2.5 GGUF (qwen2 architecture with QKV bias)

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen2_0_5b_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GGUF)
        .start()
        .await
        .expect("CUDA Qwen2.5 0.5B GGUF server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen2_0_5b_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) > 0,
        "should generate at least one token"
    );
}

// Qwen3 GGUF (qwen2 architecture in GGUF, LLaMA-compatible)

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen3_0_6b_server_starts() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_GGUF)
        .start()
        .await
        .expect("CUDA Qwen3 0.6B GGUF server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen3_0_6b_chat() {
    let server = TestServer::builder(TestModels::QWEN3_0_6B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) > 0,
        "should generate at least one token"
    );
}

// ---------------------------------------------------------------------------
// CUDA GGUF: Qwen3.5 (Qwen3-Next) — hybrid GDN + full attention
// ---------------------------------------------------------------------------

/// Qwen3.5-0.8B GGUF: server starts and health check passes.
///
/// Run with:
///   cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gguf_qwen3_next -- --ignored --test-threads=1
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen3_next_server_starts() {
    let server = TestServer::builder(TestModels::QWEN3_NEXT_0_8B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

/// Qwen3.5-0.8B GGUF: completion generates non-empty text.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen3_next_completion() {
    let server = TestServer::builder(TestModels::QWEN3_NEXT_0_8B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 30);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(!text.is_empty(), "should produce non-empty text");
}

/// Qwen3.5-0.8B GGUF: chat endpoint works.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gguf_qwen3_next_chat() {
    let server = TestServer::builder(TestModels::QWEN3_NEXT_0_8B_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) > 0,
        "should generate at least one token"
    );
}
*/
// end GGUF block comment

// ---------------------------------------------------------------------------
// Tensor Parallelism (TP=2) tests — require 2 CUDA GPUs + NCCL
// ---------------------------------------------------------------------------

/// TP=2 Qwen2.5-0.5B: server starts, health check passes, completion works.
///
/// Uses Qwen2.5-0.5B (14 Q heads, 2 KV heads — both divisible by 2).
/// SmolLM-135M has 9/3 heads which don't divide evenly by 2.
///
/// Run on nick2 pod (2x L40S):
///   cargo test -p vllm-e2e --features e2e,nccl --release --test e1_basic_serving test_cuda_tp2 -- --ignored --test-threads=1
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_qwen2_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .with_tensor_parallel_size(2)
        .start()
        .await
        .expect("TP=2 Qwen2.5-0.5B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "TP=2 completion should produce non-empty text"
    );
}

/// DeepSeek V2 Lite with TP=2 — validates MLA attention TP sharding.
///
/// Run on nick2 pod (2x L40S):
///   cargo test -p vllm-e2e --features e2e,nccl --release --test e1_basic_serving test_cuda_tp2_deepseek_v2 -- --ignored --test-threads=1
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_deepseek_v2_completion() {
    let server = TestServer::builder(TestModels::DEEPSEEK_V2_LITE_CUDA)
        .with_tensor_parallel_size(2)
        .start()
        .await
        .expect("TP=2 DeepSeek-V2-Lite server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "TP=2 DeepSeek-V2-Lite completion should produce non-empty text"
    );
}

// ===========================================================================
// CUDA Granite — safetensors BF16 (~4.5 GB) on GPU
// ===========================================================================
// TODO: Re-enable when vllm-cuda backend supports GraniteForCausalLM.
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_granite -- --ignored
/*
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_granite_chat() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_INSTRUCT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is the capital of France?", Some(20));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let content = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!content.is_empty(), "chat response should not be empty");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_granite_chat_coherent() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_INSTRUCT)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Tell me a short story", Some(30));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let content = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!content.is_empty(), "chat response should not be empty");
}
*/ // end Granite safetensors block comment

// ===========================================================================
// CUDA Granite GGUF — quantized on GPU
// ===========================================================================
// TODO: Re-enable when vllm-cuda backend supports GGUF + GraniteForCausalLM.
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_granite_gguf -- --ignored
/*
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_granite_gguf_chat() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_INSTRUCT_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is the capital of France?", Some(20));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let content = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!content.is_empty(), "chat response should not be empty");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_granite_gguf_chat_coherent() {
    let server = TestServer::builder(TestModels::GRANITE_3_3_2B_INSTRUCT_GGUF)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Tell me a short story", Some(30));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let content = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!content.is_empty(), "chat response should not be empty");
}
*/ // end Granite GGUF block comment

// ===========================================================================
// CUDA Marlin W4A16 E2E tests — GPTQ and AWQ quantized models
// ===========================================================================
// TODO: Re-enable when vllm-cuda backend supports Marlin GPTQ/AWQ quantization.
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_marlin -- --ignored --test-threads=1
/*
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gptq_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GPTQ_INT4)
        .start()
        .await
        .expect("CUDA GPTQ Marlin server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gptq_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "GPTQ Marlin completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gptq_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "GPTQ Marlin chat should not be empty");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_awq_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_AWQ)
        .start()
        .await
        .expect("CUDA AWQ Marlin server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_awq_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_AWQ)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "AWQ Marlin completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_awq_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_AWQ)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "AWQ Marlin chat should not be empty");
}
*/ // end Marlin block comment

// ===========================================================================
// CUDA MoE E2E tests — commented out, needs ≥80GB GPU
// ===========================================================================
// The smallest MoE safetensors models (Qwen1.5-MoE-A2.7B-Chat, Mixtral-8x7B)
// are 14B+ total params (~31GB BF16) — too large for L40S (48GB).
// MoE CUDA kernels are covered by 5 unit tests in vllm-kernels/src/moe.rs.
// Uncomment when A100-80GB or H100 is available.
//
// #[cfg(feature = "cuda")]
// #[tokio::test(flavor = "multi_thread")]
// #[ignore]
// async fn test_cuda_moe_server_starts() { ... }
// async fn test_cuda_moe_completion() { ... }
// async fn test_cuda_moe_chat() { ... }

// ===========================================================================
// CUDA semantic correctness + multi-turn + non-greedy tests
// ===========================================================================
// These tests validate output quality beyond "non-empty", catching bugs like
// paged FA2 prefill corruption, KV cache continuity issues, and GPU sampling errors.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_correctness -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_completion_semantic() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = resp.choices[0].text.to_lowercase();
    assert!(
        text.contains("paris"),
        "expected 'paris' in completion of 'The capital of France is', got: {}",
        text
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_multi_turn_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Turn 1: establish a fact
    let req1 = ChatCompletionRequest {
        messages: vec![user_msg(
            "My name is Claude. Please remember that. Reply with just 'OK'.",
        )],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp1 = client.chat_completion(&req1).await.unwrap();
    assert_valid_chat_response(&resp1);

    // Turn 2: query the fact — requires correct KV cache from turn 1 prefill
    let turn1_text = resp1.choices[0].message.content.clone().unwrap_or_default();
    let req2 = ChatCompletionRequest {
        messages: vec![
            user_msg("My name is Claude. Please remember that. Reply with just 'OK'."),
            assistant_msg(&turn1_text),
            user_msg("What is my name?"),
        ],
        max_tokens: Some(20),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp2 = client.chat_completion(&req2).await.unwrap();
    assert_valid_chat_response(&resp2);
    let text = resp2.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    assert!(
        text.contains("claude"),
        "turn 2 should remember 'Claude', got: {}",
        text
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_correctness_nongreedy_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Say hello in one sentence.")],
        max_tokens: Some(50),
        temperature: Some(0.7),
        ..default_chat_request()
    };
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 5);
    let word_count = text.split_whitespace().count();
    assert!(word_count >= 3, "expected at least 3 words, got: {}", text);
}
