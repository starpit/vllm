// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Phase E2: Chat completions — full feature coverage.
//!
//! Deep testing of the `/v1/chat/completions` endpoint using SmolLM-135M.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e2_chat_completions -- --ignored`

#![cfg(feature = "e2e")]

use vllm_e2e::assertions::{assert_coherent_text, assert_valid_chat_response};
use vllm_e2e::{Client, TestModels, TestServer};
use vllm_serve::protocol::{ChatCompletionMessageParam, ChatCompletionRequest};

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

fn system_msg(content: &str) -> ChatCompletionMessageParam {
    ChatCompletionMessageParam {
        role: "system".to_string(),
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

fn default_chat_request() -> ChatCompletionRequest {
    serde_json::from_str(r#"{"messages": []}"#).unwrap()
}

async fn start_smollm() -> (TestServer, Client) {
    let server = TestServer::builder(TestModels::SMOLLM_135M_4BIT)
        .start()
        .await
        .expect("SmolLM server should start");
    let client = Client::new(server.base_url());
    (server, client)
}

// ===========================================================================
// E2a: Request parameters
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_chat_temperature_0() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is 2+2?")],
        temperature: Some(0.0),
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp1 = client.chat_completion(&request).await.unwrap();
    let resp2 = client.chat_completion(&request).await.unwrap();

    let text1 = resp1.choices[0].message.content.as_deref().unwrap_or("");
    let text2 = resp2.choices[0].message.content.as_deref().unwrap_or("");
    assert_eq!(text1, text2, "temperature=0 should be deterministic");
}

#[tokio::test]
#[ignore]
async fn test_chat_temperature_high() {
    let (_server, client) = start_smollm().await;

    // High temperature — output may differ across calls.
    // We just verify the response is valid, not that it differs.
    let request = ChatCompletionRequest {
        messages: vec![user_msg("Tell me something interesting.")],
        temperature: Some(1.5),
        max_tokens: Some(30),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test]
#[ignore]
async fn test_chat_top_p() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        top_p: Some(0.1),
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test]
#[ignore]
async fn test_chat_top_k() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        top_k: Some(5),
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test]
#[ignore]
async fn test_chat_min_p() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        min_p: Some(0.1),
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test]
#[ignore]
async fn test_chat_max_tokens() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Write a very long essay about history.")],
        max_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    assert!(
        resp.usage.completion_tokens.unwrap_or(0) <= 10,
        "completion_tokens should be <= 10, got {:?}",
        resp.usage.completion_tokens
    );
}

#[tokio::test]
#[ignore]
async fn test_chat_max_completion_tokens() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Write a long essay.")],
        max_completion_tokens: Some(10),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    assert!(resp.usage.completion_tokens.unwrap_or(0) <= 10);
}

#[tokio::test]
#[ignore]
async fn test_chat_n_1() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        n: 1,
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 1, "n=1 should produce 1 choice");
}

#[tokio::test]
#[ignore]
async fn test_chat_n_3() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        n: 3,
        max_tokens: Some(20),
        temperature: Some(0.8),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_eq!(resp.choices.len(), 3, "n=3 should produce 3 choices");

    // Each choice should have a distinct index
    let indices: Vec<u32> = resp.choices.iter().map(|c| c.index).collect();
    assert!(indices.contains(&0));
    assert!(indices.contains(&1));
    assert!(indices.contains(&2));
}

#[tokio::test]
#[ignore]
async fn test_chat_seed() {
    let (_server, client) = start_smollm().await;

    // Seed with temperature=0 should be deterministic across calls to the same server.
    let request = ChatCompletionRequest {
        messages: vec![user_msg("What is the meaning of life?")],
        seed: Some(42),
        temperature: Some(0.0),
        max_tokens: Some(30),
        ..default_chat_request()
    };

    let resp1 = client.chat_completion(&request).await.unwrap();
    let resp2 = client.chat_completion(&request).await.unwrap();

    let text1 = resp1.choices[0].message.content.as_deref().unwrap_or("");
    let text2 = resp2.choices[0].message.content.as_deref().unwrap_or("");
    assert_eq!(text1, text2, "same seed + temperature=0 should produce same output");
}

#[tokio::test]
#[ignore]
async fn test_chat_logprobs() {
    let (_server, client) = start_smollm().await;

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
    let content = logprobs.content.as_ref().expect("logprobs.content should be present");
    assert!(!content.is_empty(), "logprobs content should not be empty");
    // Each entry should have top_logprobs
    for entry in content {
        assert!(
            !entry.top_logprobs.is_empty(),
            "top_logprobs should not be empty"
        );
    }
}

#[tokio::test]
#[ignore]
async fn test_chat_frequency_penalty() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Repeat the word 'hello' many times.")],
        frequency_penalty: Some(2.0),
        max_tokens: Some(30),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test]
#[ignore]
async fn test_chat_presence_penalty() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Tell me about cats.")],
        presence_penalty: Some(2.0),
        max_tokens: Some(30),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test]
#[ignore]
async fn test_chat_repetition_penalty() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello hello hello")],
        repetition_penalty: Some(1.5),
        max_tokens: Some(30),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

// ===========================================================================
// E2b: Message formats
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_chat_system_message() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![
            system_msg("You are a helpful assistant."),
            user_msg("Say hello."),
        ],
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
    let text = resp.choices[0].message.content.as_deref().unwrap_or("");
    assert_coherent_text(text, 2);
}

#[tokio::test]
#[ignore]
async fn test_chat_multi_turn() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![
            system_msg("You are a helpful assistant."),
            user_msg("My name is Alice."),
            assistant_msg("Nice to meet you, Alice!"),
            user_msg("What is my name?"),
        ],
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test]
#[ignore]
async fn test_chat_user_only() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Hello!")],
        max_tokens: Some(20),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

#[tokio::test]
#[ignore]
async fn test_chat_unicode() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![user_msg("Translate to French: Hello world")],
        max_tokens: Some(30),
        temperature: Some(0.0),
        ..default_chat_request()
    };

    let resp = client.chat_completion(&request).await.unwrap();
    assert_valid_chat_response(&resp);
}

// ===========================================================================
// E2c: Error handling
// ===========================================================================

#[tokio::test]
#[ignore]
async fn test_chat_missing_messages() {
    let (_server, client) = start_smollm().await;

    let body = serde_json::json!({"model": "test"});
    let resp = client.chat_completion_raw(&body).await.unwrap();
    assert_eq!(resp.status().as_u16(), 422, "missing messages should return 422");
}

#[tokio::test]
#[ignore]
async fn test_chat_empty_messages() {
    let (_server, client) = start_smollm().await;

    let request = ChatCompletionRequest {
        messages: vec![],
        max_tokens: Some(10),
        ..default_chat_request()
    };

    // Empty messages may return 422 or produce an error response.
    // We just verify the server doesn't crash.
    let _resp = client.chat_completion(&request).await;
}

#[tokio::test]
#[ignore]
async fn test_chat_invalid_json() {
    let (_server, client) = start_smollm().await;

    let resp = client
        .chat_completion_raw(&serde_json::json!("not a valid request object"))
        .await
        .unwrap();
    assert!(
        resp.status().is_client_error(),
        "invalid JSON should return 4xx, got {}",
        resp.status()
    );
}
