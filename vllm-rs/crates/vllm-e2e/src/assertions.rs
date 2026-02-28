// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Assertion helpers for validating OpenAI-compatible API responses.

use vllm_serve::protocol::{
    ChatCompletionResponse, ChatCompletionStreamResponse, CompletionResponse,
};

/// Assert that a chat completion response is well-formed.
pub fn assert_valid_chat_response(resp: &ChatCompletionResponse) {
    // ID should start with "chatcmpl-"
    assert!(
        resp.id.starts_with("chatcmpl-"),
        "id should start with 'chatcmpl-', got: {}",
        resp.id
    );
    assert_eq!(resp.object, "chat.completion");
    assert!(resp.created > 0, "created timestamp should be positive");
    assert!(!resp.model.is_empty(), "model should not be empty");
    assert!(!resp.choices.is_empty(), "should have at least one choice");

    for choice in &resp.choices {
        assert_eq!(choice.message.role, "assistant");
        // finish_reason should be set
        assert!(
            choice.finish_reason.is_some(),
            "finish_reason should be set"
        );
    }

    // Usage should be populated
    assert!(resp.usage.prompt_tokens > 0, "prompt_tokens should be > 0");
    assert!(resp.usage.total_tokens > 0, "total_tokens should be > 0");
}

/// Assert that a streaming chat completion response is well-formed.
///
/// `chunks` should be the parsed SSE chunks (excluding `[DONE]`).
pub fn assert_valid_stream(chunks: &[ChatCompletionStreamResponse]) {
    assert!(!chunks.is_empty(), "stream should have at least one chunk");

    // All chunks should have the same id and model
    let id = &chunks[0].id;
    let model = &chunks[0].model;
    for chunk in chunks {
        assert_eq!(&chunk.id, id, "all chunks should have the same id");
        assert_eq!(&chunk.model, model, "all chunks should have the same model");
        assert_eq!(chunk.object, "chat.completion.chunk");
    }

    // Last chunk should have finish_reason
    let last = chunks.last().unwrap();
    assert!(
        !last.choices.is_empty(),
        "last chunk should have at least one choice"
    );
    let last_choice = &last.choices[0];
    assert!(
        last_choice.finish_reason.is_some(),
        "last chunk should have finish_reason, got: {:?}",
        last_choice
    );
}

/// Collect all text content from stream chunks into a single string.
pub fn collect_stream_text(chunks: &[ChatCompletionStreamResponse]) -> String {
    let mut text = String::new();
    for chunk in chunks {
        for choice in &chunk.choices {
            if let Some(ref content) = choice.delta.content {
                text.push_str(content);
            }
        }
    }
    text
}

/// Assert that text output is coherent (not empty, not just special tokens).
pub fn assert_coherent_text(text: &str, min_len: usize) {
    let trimmed = text.trim();
    assert!(
        trimmed.len() >= min_len,
        "text should be at least {min_len} chars, got {} chars: {:?}",
        trimmed.len(),
        trimmed
    );

    // Check it's not just `<unk>` or `<token_N>` tokens
    let unk_only = trimmed
        .replace("<unk>", "")
        .replace("<pad>", "")
        .trim()
        .is_empty();
    assert!(!unk_only, "text should not be only <unk>/<pad> tokens");
}

/// Assert that a completion response is well-formed.
pub fn assert_valid_completion_response(resp: &CompletionResponse) {
    assert!(
        resp.id.starts_with("cmpl-"),
        "id should start with 'cmpl-', got: {}",
        resp.id
    );
    assert_eq!(resp.object, "text_completion");
    assert!(resp.created > 0);
    assert!(!resp.model.is_empty());
    assert!(!resp.choices.is_empty());

    for choice in &resp.choices {
        assert!(
            choice.finish_reason.is_some(),
            "finish_reason should be set"
        );
    }

    assert!(resp.usage.prompt_tokens > 0);
    assert!(resp.usage.total_tokens > 0);
}

/// Assert that tool calls are well-formed.
pub fn assert_valid_tool_calls(tool_calls: &[vllm_serve::protocol::ToolCall]) {
    assert!(!tool_calls.is_empty(), "should have at least one tool call");
    for tc in tool_calls {
        assert!(!tc.id.is_empty(), "tool call id should not be empty");
        assert_eq!(
            tc.call_type, "function",
            "tool call type should be 'function'"
        );
        assert!(
            !tc.function.name.is_empty(),
            "function name should not be empty"
        );
        // Arguments should be valid JSON
        let _: serde_json::Value =
            serde_json::from_str(&tc.function.arguments).unwrap_or_else(|e| {
                panic!(
                    "tool call arguments should be valid JSON: {e}\narguments: {}",
                    tc.function.arguments
                )
            });
    }
}

/// Assert that text is parseable as JSON.
pub fn assert_json_parseable(text: &str) {
    let _: serde_json::Value = serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("text should be valid JSON: {e}\ntext: {text}"));
}
