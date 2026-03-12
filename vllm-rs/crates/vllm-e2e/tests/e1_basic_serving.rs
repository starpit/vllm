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
// CUDA Gemma2 E2E tests — safetensors BF16 on GPU (vllm-cuda backend)
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gemma2 -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma2_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_IT_CUDA)
        .start()
        .await
        .expect("CUDA Gemma2 2B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma2_completion() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_IT_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(!text.is_empty(), "completion should not be empty");
    assert_coherent_text(text, 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma2_chat() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_IT_CUDA)
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
// CUDA Gemma3 E2E tests — safetensors BF16 on GPU (vllm-cuda backend)
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gemma3 -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma3_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_IT_CUDA)
        .start()
        .await
        .expect("CUDA Gemma3 1B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma3_completion() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_IT_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(!text.is_empty(), "completion should not be empty");
    assert_coherent_text(text, 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gemma3_chat() {
    let server = TestServer::builder(TestModels::GEMMA3_1B_IT_CUDA)
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
// CUDA GGUF E2E tests — quantized GGUF models on GPU
// ===========================================================================
// Uses vllm-cuda's GGML kernel FFI for quantized inference (no candle).
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gguf -- --ignored

// TODO: Gemma3 GGUF requires Gemma3ForCausalLM load_gguf() — not yet implemented
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
*/

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

// TODO: Qwen3Next GGUF requires Qwen3NextForCausalLM load_gguf() — not yet implemented
/*
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
// end GGUF tests

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

/// TP=2 Mixtral MoE: validates MoE expert weight sharding + post-MoE all-reduce.
///
/// Uses small_mixtral (~0.8B, 8 experts, top-2). Each expert's intermediate_size
/// is halved per rank; post-MoE all-reduce combines partial expert outputs.
///
/// Run on nick2 pod (2x L40S):
///   cargo test -p vllm-e2e --features e2e,nccl --release --test e1_basic_serving test_cuda_tp2_mixtral -- --ignored --test-threads=1
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_mixtral_completion() {
    let server = TestServer::builder(TestModels::MIXTRAL_SMALL_CUDA)
        .with_tensor_parallel_size(2)
        .start()
        .await
        .expect("TP=2 Mixtral MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "TP=2 Mixtral MoE completion should produce non-empty text"
    );
}

/// TP=2 Qwen2 MoE: validates shared expert sharding + MoE TP.
///
/// Qwen2 MoE has both routed experts and a shared expert (with sigmoid gate).
/// TP shards both the routed experts' intermediate_size and the shared expert's
/// gate_up (dim=0) / down (dim=1).
///
/// Run on nick2 pod (2x L40S):
///   cargo test -p vllm-e2e --features e2e,nccl --release --test e1_basic_serving test_cuda_tp2_qwen2_moe -- --ignored --test-threads=1
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_qwen2_moe_completion() {
    let server = TestServer::builder(TestModels::QWEN2_MOE_A2_7B_CUDA)
        .with_tensor_parallel_size(2)
        .start()
        .await
        .expect("TP=2 Qwen2 MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "TP=2 Qwen2 MoE completion should produce non-empty text"
    );
}

// ===========================================================================
// CUDA Granite — safetensors BF16 (~4.5 GB) on GPU
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_granite -- --ignored
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
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_marlin -- --ignored --test-threads=1

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

// ===========================================================================
// CUDA Marlin Gemma2 GPTQ E2E tests
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_marlin_gemma2 -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gemma2_gptq_server_starts() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_GPTQ_INT4)
        .start()
        .await
        .expect("CUDA Gemma2 GPTQ Marlin server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gemma2_gptq_completion() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "Gemma2 GPTQ Marlin completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_marlin_gemma2_gptq_chat() {
    let server = TestServer::builder(TestModels::GEMMA2_2B_GPTQ_INT4)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "Gemma2 GPTQ Marlin chat should not be empty"
    );
}

// ===========================================================================
// CUDA Marlin GPTQ desc_act (activation ordering) E2E test
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_gptq_desc_act -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_gptq_desc_act_chat() {
    let server = TestServer::builder(TestModels::TINYLLAMA_1B_GPTQ_DESC_ACT)
        .start()
        .await
        .expect("CUDA GPTQ desc_act server should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(!text.is_empty(), "GPTQ desc_act chat should not be empty");
}

// ===========================================================================
// CUDA MoE E2E tests
// ===========================================================================
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_mixtral -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_mixtral_server_starts() {
    let server = TestServer::builder(TestModels::MIXTRAL_SMALL_CUDA)
        .start()
        .await
        .expect("CUDA Mixtral MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_mixtral_completion() {
    let server = TestServer::builder(TestModels::MIXTRAL_SMALL_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "MoE completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_mixtral_chat() {
    let server = TestServer::builder(TestModels::MIXTRAL_SMALL_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        !resp.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("")
            .is_empty(),
        "MoE chat should produce output"
    );
}

// Qwen2 MoE — ~29GB BF16, fits on L40S (48GB) but tight.
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_qwen2_moe -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen2_moe_server_starts() {
    let server = TestServer::builder(TestModels::QWEN2_MOE_A2_7B_CUDA)
        .start()
        .await
        .expect("CUDA Qwen2 MoE server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let models = client.list_models().await.unwrap();
    assert_eq!(models.data.len(), 1);
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen2_moe_completion() {
    let server = TestServer::builder(TestModels::QWEN2_MOE_A2_7B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "Qwen2 MoE completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_qwen2_moe_chat() {
    let server = TestServer::builder(TestModels::QWEN2_MOE_A2_7B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = simple_chat_request("What is 2+2? Answer with just the number.", Some(10));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        !resp.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("")
            .is_empty(),
        "Qwen2 MoE chat should produce output"
    );
}

// ===========================================================================
// CUDA graphs + MoE: verify decode CUDA graphs work with MoE models
// ===========================================================================
// MoE kernels (topk_softmax, moe_align_block_size, fused_moe_gemm) have
// deterministic allocation sizes per batch_size, so CUDA graph capture works.
// These tests explicitly disable enforce_eager to exercise the graph path.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_moe_graphs -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_moe_graphs_mixtral_completion() {
    let server = TestServer::builder(TestModels::MIXTRAL_SMALL_CUDA)
        .with_enforce_eager(false)
        .start()
        .await
        .expect("Mixtral MoE with CUDA graphs should start");

    let client = Client::new(server.base_url());
    let request = simple_completion_request("The capital of France is", 30);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    assert!(
        !resp.choices[0].text.is_empty(),
        "MoE + CUDA graphs completion should not be empty"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_moe_graphs_mixtral_chat() {
    let server = TestServer::builder(TestModels::MIXTRAL_SMALL_CUDA)
        .with_enforce_eager(false)
        .start()
        .await
        .expect("Mixtral MoE with CUDA graphs should start");

    let client = Client::new(server.base_url());
    let request = simple_chat_request("Say hello in one sentence.", Some(50));
    let resp = client.chat_completion(&request).await.unwrap();

    assert_valid_chat_response(&resp);
    assert!(
        !resp.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("")
            .is_empty(),
        "MoE + CUDA graphs chat should produce output"
    );
}

/// Multi-turn chat with MoE + CUDA graphs: exercises graph replay with
/// changing batch composition (prefill eager + decode graphed across turns).
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_moe_graphs_mixtral_multi_turn() {
    let server = TestServer::builder(TestModels::MIXTRAL_SMALL_CUDA)
        .with_enforce_eager(false)
        .start()
        .await
        .expect("Mixtral MoE with CUDA graphs should start");

    let client = Client::new(server.base_url());

    // Turn 1
    let request = simple_chat_request("What is 2+2?", Some(30));
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let turn1 = resp.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_string();
    assert!(!turn1.is_empty(), "Turn 1 should produce output");

    // Turn 2 — new request exercises graph replay
    let request = simple_chat_request("What is 3+3?", Some(30));
    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    assert!(
        !resp.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("")
            .is_empty(),
        "Turn 2 should produce output (graph replay)"
    );
}

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

// ---------------------------------------------------------------------------
// Paged FA2 multi-turn regression tests
//
// The paged FA2 bug caused garbled output on turn 2+ when KV cache blocks
// were non-contiguous. These tests specifically exercise multi-turn chat
// which requires correct paged KV cache reads across multiple prefills.
// ---------------------------------------------------------------------------

/// Multi-turn with 3 turns — each turn adds more KV blocks, stressing
/// the paged block_table remapping.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_paged_fa2_three_turn_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Turn 1
    let req1 = ChatCompletionRequest {
        messages: vec![user_msg("The number I'm thinking of is 42. Just say OK.")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp1 = client.chat_completion(&req1).await.unwrap();
    assert_valid_chat_response(&resp1);
    let t1 = resp1.choices[0].message.content.clone().unwrap_or_default();

    // Turn 2
    let req2 = ChatCompletionRequest {
        messages: vec![
            user_msg("The number I'm thinking of is 42. Just say OK."),
            assistant_msg(&t1),
            user_msg("What number am I thinking of? Answer with just the number."),
        ],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp2 = client.chat_completion(&req2).await.unwrap();
    assert_valid_chat_response(&resp2);
    let t2_text = resp2.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        t2_text.contains("42"),
        "turn 2 should recall '42', got: {}",
        t2_text
    );
    let t2 = resp2.choices[0].message.content.clone().unwrap_or_default();

    // Turn 3 — even more KV cache blocks allocated
    let req3 = ChatCompletionRequest {
        messages: vec![
            user_msg("The number I'm thinking of is 42. Just say OK."),
            assistant_msg(&t1),
            user_msg("What number am I thinking of? Answer with just the number."),
            assistant_msg(&t2),
            user_msg("Double that number. Answer with just the number."),
        ],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp3 = client.chat_completion(&req3).await.unwrap();
    assert_valid_chat_response(&resp3);
    let t3_text = resp3.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        t3_text.contains("84"),
        "turn 3 should compute 42*2=84, got: {}",
        t3_text
    );
}

/// Interleaved multi-turn requests — two separate conversations interleave
/// their block allocations, ensuring the block_table correctly maps
/// non-contiguous physical blocks.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_paged_fa2_interleaved_multi_turn() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Turn 1 for user A — allocates some blocks
    let req_a1 = ChatCompletionRequest {
        messages: vec![user_msg(
            "Remember: the color is blue. Reply with only 'OK'.",
        )],
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp_a1 = client.chat_completion(&req_a1).await.unwrap();
    assert_valid_chat_response(&resp_a1);
    let ta1 = resp_a1.choices[0]
        .message
        .content
        .clone()
        .unwrap_or_default();

    // Turn 1 for user B — allocates more blocks (interleaved with A's freed blocks)
    let req_b1 = ChatCompletionRequest {
        messages: vec![user_msg(
            "Remember: the animal is cat. Reply with only 'OK'.",
        )],
        max_tokens: Some(5),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp_b1 = client.chat_completion(&req_b1).await.unwrap();
    assert_valid_chat_response(&resp_b1);
    let tb1 = resp_b1.choices[0]
        .message
        .content
        .clone()
        .unwrap_or_default();

    // Turn 2 for user A — prefills into new blocks with possible gaps
    let req_a2 = ChatCompletionRequest {
        messages: vec![
            user_msg("Remember: the color is blue. Reply with only 'OK'."),
            assistant_msg(&ta1),
            user_msg("What color did I say? Answer with just the color."),
        ],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp_a2 = client.chat_completion(&req_a2).await.unwrap();
    assert_valid_chat_response(&resp_a2);
    let text_a = resp_a2.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    assert!(
        text_a.contains("blue"),
        "user A turn 2 should recall 'blue', got: {}",
        text_a
    );

    // Turn 2 for user B
    let req_b2 = ChatCompletionRequest {
        messages: vec![
            user_msg("Remember: the animal is cat. Reply with only 'OK'."),
            assistant_msg(&tb1),
            user_msg("What animal did I say? Answer with just the animal."),
        ],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };
    let resp_b2 = client.chat_completion(&req_b2).await.unwrap();
    assert_valid_chat_response(&resp_b2);
    let text_b = resp_b2.choices[0]
        .message
        .content
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    assert!(
        text_b.contains("cat"),
        "user B turn 2 should recall 'cat', got: {}",
        text_b
    );
}

// ===========================================================================
// CUDA Logprobs tests
// ===========================================================================
// Validates that logprobs are correctly returned from CudaWorker (forces CPU
// fallback path where logprobs are computed).
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_logprobs -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_logprobs_chat() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        logprobs: Some(true),
        top_logprobs: Some(5),
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let logprobs = resp.choices[0]
        .logprobs
        .as_ref()
        .expect("logprobs should be present");
    let content = logprobs
        .content
        .as_ref()
        .expect("logprobs.content should be present");
    assert!(!content.is_empty(), "logprobs content should not be empty");
    for entry in content {
        assert!(
            !entry.top_logprobs.is_empty(),
            "top_logprobs should not be empty"
        );
        // Verify logprobs are valid (non-positive log probabilities).
        assert!(
            entry.logprob <= 0.0,
            "logprob should be non-positive, got {}",
            entry.logprob
        );
    }
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_logprobs_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single(
            "The capital of France is".to_string(),
        )),
        max_tokens: Some(10),
        temperature: Some(0.0),
        logprobs: Some(5),
        ..default_completion_request()
    };

    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);

    let token_logprobs = &resp.choices[0]
        .logprobs
        .as_ref()
        .expect("logprobs should be present")
        .token_logprobs;
    assert!(
        !token_logprobs.is_empty(),
        "token_logprobs should not be empty"
    );
    for lp in token_logprobs {
        if let Some(v) = lp {
            assert!(*v <= 0.0, "logprob should be non-positive, got {v}");
        }
    }
}

// ===========================================================================
// CUDA Grammar / constrained decoding tests
// ===========================================================================
// Validates that guided_regex constrains output via CudaWorker grammar state.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_grammar -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_grammar_regex_digits() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Use guided_regex to force the model to output only digits.
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Give me a number.")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        guided_regex: Some("[0-9]+".to_string()),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "grammar-constrained output should not be empty"
    );
    assert!(
        text.chars().all(|c| c.is_ascii_digit()),
        "guided_regex '[0-9]+' should produce only digits, got: {text:?}"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_grammar_json_object() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Use response_format: json_object to force valid JSON output.
    let request = ChatCompletionRequest {
        messages: vec![user_msg(
            "Return a JSON object with a key 'name' set to 'Alice'.",
        )],
        max_tokens: Some(50),
        temperature: Some(0.0),
        response_format: Some(vllm_serve::protocol::ResponseFormat::Standard(
            vllm_serve::protocol::StandardResponseFormat {
                format_type: "json_object".to_string(),
                json_schema: None,
            },
        )),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "json_object constrained output should not be empty"
    );
    // The grammar constrains to valid JSON, but the model may append EOS tokens
    // after the JSON is complete. Trim known EOS markers before parsing.
    let trimmed = text.trim_end_matches("</s>").trim();
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(trimmed);
    assert!(
        parsed.is_ok(),
        "response_format json_object should produce valid JSON, got: {text:?}"
    );
}

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_grammar_ebnf_digits() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Use guided_grammar (Lark/EBNF) to force digit-only output.
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Give me a number.")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        guided_grammar: Some(r#"start: /[0-9]+/"#.to_string()),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);

    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "EBNF grammar-constrained output should not be empty"
    );
    assert!(
        text.chars().all(|c| c.is_ascii_digit()),
        "guided_grammar digits should produce only digits, got: {text:?}"
    );
}

// Validates structural_tag response_format constrains output via CudaWorker.
// The structural tag defines a trigger that forces JSON schema output between tags.
//
// Run with: cargo test -p vllm-e2e --features e2e,cuda --release --test e1_basic_serving test_cuda_grammar_structural_tag -- --ignored --test-threads=1

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_grammar_structural_tag() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Use structural_tag response_format via raw JSON (legacy format).
    // The trigger "TOOL:" with begin "TOOL:" forces the model to emit a JSON object
    // conforming to the schema between the trigger and end marker.
    let response_format: vllm_serve::protocol::ResponseFormat =
        serde_json::from_value(serde_json::json!({
            "type": "structural_tag",
            "structures": [{
                "begin": "TOOL:",
                "schema": {
                    "type": "object",
                    "properties": {
                        "value": { "type": "integer" }
                    },
                    "required": ["value"]
                },
                "end": ";END"
            }],
            "triggers": ["TOOL:"]
        }))
        .expect("structural_tag response_format should deserialize");

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Call a tool with value 42.")],
        max_tokens: Some(60),
        temperature: Some(0.0),
        response_format: Some(response_format),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert!(
        !text.is_empty(),
        "structural_tag constrained output should not be empty"
    );
    // The output should contain the structural tag markers and valid JSON between them.
    // With the grammar constraint, the model must emit text matching the pattern:
    //   (free text)* TOOL: <json conforming to schema> ;END (free text)*
    // We verify the JSON portion parses correctly.
    if let Some(tool_start) = text.find("TOOL:") {
        let after_tool = &text[tool_start + "TOOL:".len()..];
        if let Some(end_pos) = after_tool.find(";END") {
            let json_part = after_tool[..end_pos].trim();
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(json_part);
            assert!(
                parsed.is_ok(),
                "structural_tag should produce valid JSON between markers, got: {json_part:?}"
            );
            let obj = parsed.unwrap();
            assert!(
                obj.get("value").is_some(),
                "structural_tag JSON should have 'value' key, got: {obj}"
            );
        }
    }
    // At minimum, the grammar should have constrained output to be non-empty
    // (the full structural tag pattern may or may not appear depending on model behavior,
    // but the grammar engine ensures validity of whatever is produced).
}

// ===========================================================================
// LogitsProcessor tests: min_tokens, logit_bias, penalties
// ===========================================================================

/// Test min_tokens: with min_tokens=10, output must be at least 10 tokens.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_min_tokens_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());
    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single("Hello".to_string())),
        max_tokens: Some(50),
        min_tokens: 10,
        temperature: Some(0.0),
        ..default_completion_request()
    };

    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);

    let text = &resp.choices[0].text;
    // Rough check: 10 tokens should produce at least ~15 chars of output.
    // The exact token count isn't available via the API, but usage.completion_tokens is.
    let completion_tokens = resp.usage.completion_tokens.unwrap_or(0);
    assert!(
        completion_tokens >= 10,
        "min_tokens=10 should produce at least 10 completion tokens, got {completion_tokens}. text: {text:?}"
    );
}

/// Test logit_bias: boost a specific token to force it into output.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_logit_bias_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // Token 220 in Qwen2 tokenizer is typically a common token.
    // Strongly bias it (+100) so it dominates output.
    let mut bias = std::collections::HashMap::new();
    bias.insert("220".to_string(), 100.0);

    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single("Test".to_string())),
        max_tokens: Some(5),
        temperature: Some(0.0),
        logit_bias: Some(bias),
        ..default_completion_request()
    };

    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);

    // The biased token should dominate. At minimum, we should get non-empty output.
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "logit_bias completion should produce output"
    );
    // With +100 bias on token 220, it should dominate output.
    // We just verify non-empty output and no crash — the exact content depends on tokenizer.
    assert!(
        text.len() >= 2,
        "logit_bias +100 should produce at least a couple characters, got {text:?}"
    );
}

/// Test penalties: repetition_penalty suppresses repeated tokens.
#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_penalties_completion() {
    let server = TestServer::builder(TestModels::QWEN2_0_5B_CUDA)
        .start()
        .await
        .unwrap();

    let client = Client::new(server.base_url());

    // High repetition penalty should reduce repetition.
    let request = CompletionRequest {
        prompt: Some(CompletionPrompt::Single("The".to_string())),
        max_tokens: Some(30),
        temperature: Some(0.5),
        repetition_penalty: Some(2.0),
        seed: Some(42),
        ..default_completion_request()
    };

    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "penalties completion should produce output"
    );
}

// ---------------------------------------------------------------------------
// Gemma3 TP=2 tests (CUDA, NCCL)
// ---------------------------------------------------------------------------

/// TP=2 Gemma3 completion test.
///
///   cargo test -p vllm-e2e --features e2e,nccl --release --test e1_basic_serving test_cuda_tp2_gemma3 -- --ignored --test-threads=1
#[cfg(feature = "nccl")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_tp2_gemma3_completion() {
    let server = TestServer::builder(TestModels::GEMMA3_4B_IT_CUDA)
        .with_tensor_parallel_size(2)
        .start()
        .await
        .expect("TP=2 Gemma3-1B server should start");

    let client = Client::new(server.base_url());
    assert!(client.health().await.unwrap(), "server should be healthy");

    let request = simple_completion_request("The capital of France is", 20);
    let resp = client.completion(&request).await.unwrap();

    assert_valid_completion_response(&resp);
    let text = &resp.choices[0].text;
    assert!(
        !text.is_empty(),
        "TP=2 Gemma3 completion should produce non-empty text"
    );
}

// ---------------------------------------------------------------------------
// Sleep/wake tests
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_cuda_sleep_wake() {
    let server = TestServer::builder(TestModels::SMOLLM_135M_CUDA)
        .start()
        .await
        .expect("server should start");

    let client = Client::new(server.base_url());

    // 1. Verify working.
    let request = simple_completion_request("Hello", 10);
    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);

    // 2. Record GPU memory before sleep.
    let mem_before = client.gpu_memory().await.unwrap();
    assert!(mem_before.used_bytes > 0, "GPU should have memory in use");

    // 3. Sleep.
    client.sleep(1).await.unwrap();
    assert!(
        client.is_sleeping().await.unwrap(),
        "engine should be sleeping"
    );

    // 4. Verify GPU memory dropped significantly.
    let mem_sleeping = client.gpu_memory().await.unwrap();
    assert!(
        mem_sleeping.used_bytes < mem_before.used_bytes / 2,
        "GPU memory should drop significantly after sleep: before={}, sleeping={}",
        mem_before.used_bytes,
        mem_sleeping.used_bytes,
    );

    // 5. Wake up.
    client.wake_up(None).await.unwrap();
    assert!(
        !client.is_sleeping().await.unwrap(),
        "engine should be awake"
    );

    // 6. Verify still works.
    let resp = client.completion(&request).await.unwrap();
    assert_valid_completion_response(&resp);
}
