// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! E2E tests for the offline batch `LLM` API.
//!
//! These tests exercise `LLM::new()`, `LLM::generate()`, and `LLM::chat()`
//! directly — no HTTP server involved.
//!
//! Run with: `cargo test -p vllm-e2e --features e2e --test e_llm_api -- --ignored`

#![cfg(feature = "e2e")]

use vllm_e2e::TestModels;
use vllm_serve::llm::{ChatMessage, LLM, SamplingParams};

// ---------------------------------------------------------------------------
// generate()
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_llm_generate_basic() {
    vllm_common::telemetry::init_tracing("off");

    let llm = LLM::new(TestModels::SMOLLM_135M_4BIT).expect("LLM should initialize");
    assert!(llm.model_name().contains("SmolLM"));

    let outputs = llm
        .generate(&["The capital of France is"], None)
        .expect("generate should succeed");

    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];
    assert!(output.finished);
    assert_eq!(output.outputs.len(), 1);
    assert!(
        !output.outputs[0].text.is_empty(),
        "generated text should not be empty"
    );
}

#[test]
#[ignore]
fn test_llm_generate_multiple_prompts() {
    vllm_common::telemetry::init_tracing("off");

    let llm = LLM::new(TestModels::SMOLLM_135M_4BIT).expect("LLM should initialize");

    let prompts = &["Hello, world!", "The meaning of life is"];
    let params = SamplingParams {
        max_tokens: Some(20),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let outputs = llm
        .generate(prompts, Some(params))
        .expect("generate should succeed");

    assert_eq!(outputs.len(), 2);
    for (i, output) in outputs.iter().enumerate() {
        assert!(output.finished, "output {i} should be finished");
        assert!(
            !output.outputs[0].text.is_empty(),
            "output {i} should have text"
        );
        assert_eq!(
            output.prompt.as_deref(),
            Some(prompts[i]),
            "prompt should be preserved"
        );
    }
}

#[test]
#[ignore]
fn test_llm_generate_max_tokens() {
    vllm_common::telemetry::init_tracing("off");

    let llm = LLM::new(TestModels::SMOLLM_135M_4BIT).expect("LLM should initialize");

    let params = SamplingParams {
        max_tokens: Some(3),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let outputs = llm
        .generate(&["Write a very long story about a dragon."], Some(params))
        .expect("generate should succeed");

    assert_eq!(outputs.len(), 1);
    let output = &outputs[0];
    assert!(output.finished);
    assert!(
        output.outputs[0].finish_reason.as_deref() == Some("length"),
        "should stop due to length, got: {:?}",
        output.outputs[0].finish_reason
    );
}

// ---------------------------------------------------------------------------
// chat()
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_llm_chat_basic() {
    vllm_common::telemetry::init_tracing("off");

    let llm = LLM::new(TestModels::SMOLLM_135M_4BIT).expect("LLM should initialize");

    let messages = vec![ChatMessage::user("Say hello in one sentence.")];
    let params = SamplingParams {
        max_tokens: Some(50),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let output = llm
        .chat(&messages, Some(params))
        .expect("chat should succeed");

    assert!(output.finished);
    assert_eq!(output.outputs.len(), 1);
    let text = &output.outputs[0].text;
    assert!(!text.is_empty(), "chat response should not be empty");
    assert!(text.len() >= 2, "chat response should be at least 2 chars");
}

#[test]
#[ignore]
fn test_llm_chat_with_system_message() {
    vllm_common::telemetry::init_tracing("off");

    let llm = LLM::new(TestModels::SMOLLM_135M_4BIT).expect("LLM should initialize");

    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("What is 2+2?"),
    ];
    let params = SamplingParams {
        max_tokens: Some(30),
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let output = llm
        .chat(&messages, Some(params))
        .expect("chat should succeed");

    assert!(output.finished);
    assert!(!output.outputs[0].text.is_empty());
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_llm_builder() {
    vllm_common::telemetry::init_tracing("off");

    let llm = LLM::builder(TestModels::SMOLLM_135M_4BIT)
        .max_model_len(512)
        .build()
        .expect("LLM builder should succeed");

    assert_eq!(llm.max_model_len(), 512);

    let outputs = llm
        .generate(&["Hello"], None)
        .expect("generate should succeed");
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].finished);
}
