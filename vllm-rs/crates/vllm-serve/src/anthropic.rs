// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Anthropic Messages API (`/v1/messages`) — translation layer over the
//! internal chat completion engine.

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::StreamExt as _;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::info;

use crate::engine::StreamDelta;
use crate::protocol;
use crate::server::AppState;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Content of an Anthropic message — either a plain string or array of blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A single content block inside a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: Value },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<ToolResultContent>,
    },
}

/// Content of a tool_result block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolResultBlock {
    #[serde(rename = "text")]
    Text { text: String },
}

/// System parameter — string or array of system blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemParam {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBlock {
    pub text: String,
    #[serde(default)]
    pub cache_control: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
}

/// An Anthropic tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Anthropic Messages API request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<SystemParam>,
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ---------------------------------------------------------------------------
// Conversion: Anthropic request → ChatCompletionRequest
// ---------------------------------------------------------------------------

fn convert_request(req: MessagesRequest) -> protocol::ChatCompletionRequest {
    let mut messages = Vec::new();

    // System message
    if let Some(sys) = req.system {
        let text = match sys {
            SystemParam::Text(t) => t,
            SystemParam::Blocks(blocks) => blocks
                .into_iter()
                .map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n"),
        };
        messages.push(protocol::ChatCompletionMessageParam {
            role: "system".to_string(),
            content: Some(Value::String(text)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    // Convert messages
    for msg in req.messages {
        match msg.content {
            MessageContent::Text(text) => {
                messages.push(protocol::ChatCompletionMessageParam {
                    role: msg.role,
                    content: Some(Value::String(text)),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            MessageContent::Blocks(blocks) => {
                let mut text_parts = Vec::new();
                let mut tool_calls = Vec::new();
                let mut tool_results: Vec<(String, String)> = Vec::new();

                for block in blocks {
                    match block {
                        ContentBlock::Text { text } => text_parts.push(text),
                        ContentBlock::Image { .. } => {}
                        ContentBlock::ToolUse { id, name, input } => {
                            tool_calls.push(protocol::ToolCall {
                                id,
                                call_type: "function".to_string(),
                                function: protocol::FunctionCall {
                                    name,
                                    arguments: serde_json::to_string(&input).unwrap_or_default(),
                                },
                            });
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                        } => {
                            let text = match content {
                                Some(ToolResultContent::Text(t)) => t,
                                Some(ToolResultContent::Blocks(bs)) => bs
                                    .into_iter()
                                    .map(|b| match b {
                                        ToolResultBlock::Text { text } => text,
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                                None => String::new(),
                            };
                            tool_results.push((tool_use_id, text));
                        }
                    }
                }

                // Emit one "tool" message per tool result
                for (tool_use_id, text) in tool_results {
                    messages.push(protocol::ChatCompletionMessageParam {
                        role: "tool".to_string(),
                        content: Some(Value::String(text)),
                        name: None,
                        tool_calls: None,
                        tool_call_id: Some(tool_use_id),
                    });
                }

                // Emit the main message (text + tool_calls) if present
                if !text_parts.is_empty() || !tool_calls.is_empty() {
                    let content = if text_parts.is_empty() {
                        None
                    } else {
                        Some(Value::String(text_parts.join("\n")))
                    };
                    let tc = if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    };
                    messages.push(protocol::ChatCompletionMessageParam {
                        role: msg.role,
                        content,
                        name: None,
                        tool_calls: tc,
                        tool_call_id: None,
                    });
                }
            }
        }
    }

    // Convert tools
    let tools = req.tools.map(|ts| {
        ts.into_iter()
            .map(|t| protocol::ChatCompletionToolsParam {
                tool_type: "function".to_string(),
                function: protocol::FunctionDefinition {
                    name: t.name,
                    description: t.description,
                    parameters: Some(t.input_schema),
                },
            })
            .collect()
    });

    // Convert tool_choice
    let tool_choice = req.tool_choice.map(|tc| {
        if let Some(s) = tc.as_str() {
            match s {
                "auto" => Value::String("auto".to_string()),
                "any" => Value::String("required".to_string()),
                "none" => Value::String("none".to_string()),
                _ => tc,
            }
        } else if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
            serde_json::json!({
                "type": "function",
                "function": { "name": name }
            })
        } else {
            tc
        }
    });

    let stop = req.stop_sequences.map(protocol::StopCondition::Multiple);

    protocol::ChatCompletionRequest {
        model: req.model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        n: 1,
        max_tokens: Some(req.max_tokens),
        max_completion_tokens: None,
        stream: false,
        stream_options: None,
        stop,
        frequency_penalty: None,
        presence_penalty: None,
        logit_bias: None,
        logprobs: None,
        top_logprobs: None,
        prompt_logprobs: None,
        seed: None,
        response_format: None,
        tools,
        tool_choice,
        user: None,
        top_k: req.top_k.map(|k| k as i32),
        min_p: None,
        repetition_penalty: None,
        min_tokens: 0,
        stop_token_ids: Vec::new(),
        include_stop_str_in_output: false,
        ignore_eos: false,
        skip_special_tokens: true,
        priority: 0,
        cache_salt: None,
        request_id: None,
        guided_regex: None,
        guided_grammar: None,
        allowed_token_ids: None,
        bad_words: None,
        truncate_prompt_tokens: None,
        include_reasoning: true,
    }
}

// ---------------------------------------------------------------------------
// Conversion: ChatCompletionResponse → MessagesResponse
// ---------------------------------------------------------------------------

fn map_finish_reason(reason: Option<&str>) -> Option<String> {
    reason.map(|r| {
        match r {
            "stop" => "end_turn",
            "length" => "max_tokens",
            "tool_calls" => "tool_use",
            _ => "end_turn",
        }
        .to_string()
    })
}

fn convert_response(resp: protocol::ChatCompletionResponse) -> MessagesResponse {
    let choice = resp.choices.into_iter().next();

    let (content, stop_reason) = match choice {
        Some(c) => {
            let mut blocks = Vec::new();

            if let Some(text) = c.message.content.filter(|t| !t.is_empty()) {
                blocks.push(ResponseContentBlock::Text { text });
            }

            if let Some(tool_calls) = c.message.tool_calls {
                for tc in tool_calls {
                    let input: Value = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(Value::Object(serde_json::Map::new()));
                    blocks.push(ResponseContentBlock::ToolUse {
                        id: tc.id,
                        name: tc.function.name,
                        input,
                    });
                }
            }

            let stop_reason = map_finish_reason(c.finish_reason.as_deref());
            (blocks, stop_reason)
        }
        None => (vec![], Some("end_turn".to_string())),
    };

    MessagesResponse {
        id: format!("msg_{}", &resp.id),
        response_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: resp.model,
        stop_reason,
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: resp.usage.prompt_tokens,
            output_tokens: resp.usage.completion_tokens.unwrap_or(0),
        },
    }
}

// ---------------------------------------------------------------------------
// Handler: POST /v1/messages
// ---------------------------------------------------------------------------

/// POST /v1/messages — Anthropic Messages API.
pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(request): Json<MessagesRequest>,
) -> Response {
    info!(
        "POST /v1/messages: model={:?}, max_tokens={}, stream={}",
        request.model, request.max_tokens, request.stream
    );

    let is_stream = request.stream;
    let mut chat_request = convert_request(request);

    if is_stream {
        chat_request.stream = true;
        match state.engine.chat_completion_stream(chat_request).await {
            Ok((request_id, model, rx)) => {
                stream_messages_response(request_id, model, rx).into_response()
            }
            Err(e) => e.into_response(),
        }
    } else {
        match state.engine.chat_completion(chat_request).await {
            Ok(response) => {
                let anthropic_response = convert_response(response);
                Json(anthropic_response).into_response()
            }
            Err(e) => e.into_response(),
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming SSE
// ---------------------------------------------------------------------------

fn stream_messages_response(
    request_id: String,
    model: String,
    rx: tokio::sync::mpsc::UnboundedReceiver<StreamDelta>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let (tx, out_rx) = tokio::sync::mpsc::unbounded_channel::<Result<Event, Infallible>>();

    tokio::spawn(async move {
        let msg_id = format!("msg_{request_id}");
        let mut started = false;
        let mut block_started = false;
        let mut output_tokens: u32 = 0;
        let mut stream = UnboundedReceiverStream::new(rx);

        while let Some(delta) = stream.next().await {
            output_tokens += delta.new_token_ids.len() as u32;

            if !started {
                started = true;
                let _ = tx.send(Ok(Event::default()
                    .event("message_start")
                    .json_data(serde_json::json!({
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": model,
                            "stop_reason": null,
                            "stop_sequence": null,
                            "usage": { "input_tokens": 0, "output_tokens": 0 }
                        }
                    }))
                    .unwrap()));
                let _ = tx.send(Ok(Event::default()
                    .event("ping")
                    .json_data(serde_json::json!({"type": "ping"}))
                    .unwrap()));
            }

            let has_tool_calls = delta.tool_call_deltas.is_some();
            let text = if has_tool_calls {
                None
            } else {
                delta.text.filter(|t| !t.is_empty())
            };

            if !block_started && (text.is_some() || has_tool_calls) {
                block_started = true;
                let block = if has_tool_calls {
                    serde_json::json!({"type": "tool_use", "id": "", "name": "", "input": {}})
                } else {
                    serde_json::json!({"type": "text", "text": ""})
                };
                let _ = tx.send(Ok(Event::default()
                    .event("content_block_start")
                    .json_data(serde_json::json!({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": block
                    }))
                    .unwrap()));
            }

            if let Some(ref t) = text {
                let _ = tx.send(Ok(Event::default()
                    .event("content_block_delta")
                    .json_data(serde_json::json!({
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": { "type": "text_delta", "text": t }
                    }))
                    .unwrap()));
            }

            if delta.finish_reason.is_some() {
                let stop_reason = if has_tool_calls {
                    "tool_use"
                } else {
                    "end_turn"
                };

                if block_started {
                    let _ = tx.send(Ok(Event::default()
                        .event("content_block_stop")
                        .json_data(serde_json::json!({
                            "type": "content_block_stop",
                            "index": 0
                        }))
                        .unwrap()));
                }

                let _ = tx.send(Ok(Event::default()
                    .event("message_delta")
                    .json_data(serde_json::json!({
                        "type": "message_delta",
                        "delta": { "stop_reason": stop_reason, "stop_sequence": null },
                        "usage": { "output_tokens": output_tokens }
                    }))
                    .unwrap()));

                let _ = tx.send(Ok(Event::default()
                    .event("message_stop")
                    .json_data(serde_json::json!({"type": "message_stop"}))
                    .unwrap()));
            }
        }
    });

    Sse::new(UnboundedReceiverStream::new(out_rx)).keep_alive(KeepAlive::default())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_simple_request() {
        let json = r#"{
            "model": "claude-3-sonnet",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.messages.len(), 1);
        assert!(!req.stream);
    }

    #[test]
    fn test_deserialize_content_blocks() {
        let json = r#"{
            "max_tokens": 50,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "text", "text": "Tell me more."}
                ]
            }]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            MessageContent::Blocks(blocks) => assert_eq!(blocks.len(), 2),
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_deserialize_system_string() {
        let json = r#"{
            "max_tokens": 10,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match req.system.unwrap() {
            SystemParam::Text(t) => assert_eq!(t, "You are helpful."),
            _ => panic!("expected text system"),
        }
    }

    #[test]
    fn test_deserialize_system_blocks() {
        let json = r#"{
            "max_tokens": 10,
            "system": [{"text": "Be concise."}, {"text": "Be accurate."}],
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match req.system.unwrap() {
            SystemParam::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert_eq!(blocks[0].text, "Be concise.");
            }
            _ => panic!("expected system blocks"),
        }
    }

    #[test]
    fn test_deserialize_tools() {
        let json = r#"{
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
    }

    #[test]
    fn test_deserialize_tool_result() {
        let json = r#"{
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_123", "content": "72°F"}
                ]
            }]
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            MessageContent::Blocks(blocks) => match &blocks[0] {
                ContentBlock::ToolResult { tool_use_id, .. } => {
                    assert_eq!(tool_use_id, "call_123");
                }
                _ => panic!("expected tool_result"),
            },
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_deserialize_stream_flag() {
        let json = r#"{
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert!(req.stream);
    }

    #[test]
    fn test_convert_simple_request() {
        let req = MessagesRequest {
            model: Some("test-model".into()),
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Text("Hello".into()),
            }],
            system: None,
            max_tokens: 100,
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.max_tokens, Some(100));
        assert_eq!(chat.temperature, Some(0.7));
    }

    #[test]
    fn test_convert_system_message() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Text("Hi".into()),
            }],
            system: Some(SystemParam::Text("Be helpful.".into())),
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.messages.len(), 2);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(
            chat.messages[0].content,
            Some(Value::String("Be helpful.".into()))
        );
    }

    #[test]
    fn test_convert_stop_sequences() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Text("Hi".into()),
            }],
            system: None,
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Some(vec!["END".into(), "STOP".into()]),
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        match chat.stop.unwrap() {
            protocol::StopCondition::Multiple(v) => {
                assert_eq!(v, vec!["END", "STOP"]);
            }
            _ => panic!("expected Multiple stop"),
        }
    }

    #[test]
    fn test_convert_tool_use_blocks() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "assistant".into(),
                content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "get_weather".into(),
                    input: serde_json::json!({"city": "NYC"}),
                }]),
            }],
            system: None,
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.messages.len(), 1);
        let tc = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].id, "call_1");
        assert_eq!(tc[0].function.name, "get_weather");
    }

    #[test]
    fn test_convert_tool_result_blocks() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "call_1".into(),
                    content: Some(ToolResultContent::Text("72°F".into())),
                }]),
            }],
            system: None,
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "tool");
        assert_eq!(chat.messages[0].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn test_convert_tools() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Text("Hi".into()),
            }],
            system: None,
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: Some(vec![AnthropicTool {
                name: "search".into(),
                description: Some("Search the web".into()),
                input_schema: serde_json::json!({"type": "object"}),
            }]),
            tool_choice: None,
            metadata: None,
        };
        let chat = convert_request(req);
        let tools = chat.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "search");
        assert_eq!(tools[0].tool_type, "function");
    }

    #[test]
    fn test_convert_tool_choice_any() {
        let req = MessagesRequest {
            model: None,
            messages: vec![AnthropicMessage {
                role: "user".into(),
                content: MessageContent::Text("Hi".into()),
            }],
            system: None,
            max_tokens: 50,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            tools: None,
            tool_choice: Some(Value::String("any".into())),
            metadata: None,
        };
        let chat = convert_request(req);
        assert_eq!(chat.tool_choice, Some(Value::String("required".into())));
    }

    #[test]
    fn test_convert_simple_response() {
        let resp = protocol::ChatCompletionResponse {
            id: "chatcmpl-123".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "test".into(),
            choices: vec![protocol::ChatCompletionResponseChoice {
                index: 0,
                message: protocol::ChatMessage {
                    role: "assistant".into(),
                    content: Some("Hello!".into()),
                    refusal: None,
                    tool_calls: None,
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("stop".into()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            system_fingerprint: None,
            usage: protocol::UsageInfo {
                prompt_tokens: 10,
                total_tokens: 15,
                completion_tokens: Some(5),
                prompt_tokens_details: None,
            },
        };
        let anthropic = convert_response(resp);
        assert_eq!(anthropic.response_type, "message");
        assert_eq!(anthropic.role, "assistant");
        assert_eq!(anthropic.stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(anthropic.usage.input_tokens, 10);
        assert_eq!(anthropic.usage.output_tokens, 5);
        match &anthropic.content[0] {
            ResponseContentBlock::Text { text } => assert_eq!(text, "Hello!"),
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_convert_tool_calls_response() {
        let resp = protocol::ChatCompletionResponse {
            id: "chatcmpl-456".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "test".into(),
            choices: vec![protocol::ChatCompletionResponseChoice {
                index: 0,
                message: protocol::ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    refusal: None,
                    tool_calls: Some(vec![protocol::ToolCall {
                        id: "call_1".into(),
                        call_type: "function".into(),
                        function: protocol::FunctionCall {
                            name: "get_weather".into(),
                            arguments: r#"{"city":"NYC"}"#.into(),
                        },
                    }]),
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("tool_calls".into()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            system_fingerprint: None,
            usage: protocol::UsageInfo {
                prompt_tokens: 20,
                total_tokens: 30,
                completion_tokens: Some(10),
                prompt_tokens_details: None,
            },
        };
        let anthropic = convert_response(resp);
        assert_eq!(anthropic.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(anthropic.content.len(), 1);
        match &anthropic.content[0] {
            ResponseContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "NYC");
            }
            _ => panic!("expected tool_use block"),
        }
    }

    #[test]
    fn test_convert_max_tokens_finish() {
        let resp = protocol::ChatCompletionResponse {
            id: "chatcmpl-789".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "test".into(),
            choices: vec![protocol::ChatCompletionResponseChoice {
                index: 0,
                message: protocol::ChatMessage {
                    role: "assistant".into(),
                    content: Some("truncated...".into()),
                    refusal: None,
                    tool_calls: None,
                    reasoning: None,
                },
                logprobs: None,
                finish_reason: Some("length".into()),
                stop_reason: None,
                prompt_logprobs: None,
            }],
            system_fingerprint: None,
            usage: protocol::UsageInfo {
                prompt_tokens: 5,
                total_tokens: 105,
                completion_tokens: Some(100),
                prompt_tokens_details: None,
            },
        };
        let anthropic = convert_response(resp);
        assert_eq!(anthropic.stop_reason.as_deref(), Some("max_tokens"));
    }

    #[test]
    fn test_response_serialization() {
        let resp = MessagesResponse {
            id: "msg_123".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![ResponseContentBlock::Text { text: "Hi".into() }],
            model: "test".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 1,
                output_tokens: 1,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "Hi");
        assert_eq!(json["stop_reason"], "end_turn");
        assert_eq!(json["usage"]["input_tokens"], 1);
    }

    #[test]
    fn test_finish_reason_mapping() {
        assert_eq!(map_finish_reason(Some("stop")).as_deref(), Some("end_turn"));
        assert_eq!(
            map_finish_reason(Some("length")).as_deref(),
            Some("max_tokens")
        );
        assert_eq!(
            map_finish_reason(Some("tool_calls")).as_deref(),
            Some("tool_use")
        );
        assert_eq!(map_finish_reason(None), None);
    }
}
