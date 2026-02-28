// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Tool call parsing for model outputs.
//!
//! When models generate text containing tool calls (e.g.
//! `<tool_call>{"name":"search","arguments":{"q":"foo"}}</tool_call>`),
//! these parsers detect and extract them into structured `ToolCall` objects.
//!
//! Supports both non-streaming (full text extraction) and streaming
//! (incremental delta) modes, matching Python vLLM behavior.
//!
//! Port of: `vllm/entrypoints/openai/tool_parsers/` (subset)

use std::sync::Arc;

use uuid::Uuid;

use crate::protocol;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Result of parsing tool calls from complete model output (non-streaming).
#[derive(Debug, Clone)]
pub struct ExtractedToolCallInfo {
    /// Whether any tool calls were found.
    pub tools_called: bool,
    /// Extracted tool calls.
    pub tool_calls: Vec<protocol::ToolCall>,
    /// Text before/outside tool calls (None if empty).
    pub content: Option<String>,
}

/// A streaming tool call delta (mirrors OpenAI DeltaToolCall).
#[derive(Debug, Clone)]
pub struct DeltaToolCall {
    /// Index of this tool call in the array.
    pub index: u32,
    /// Tool call ID (set on first delta for this tool).
    pub id: Option<String>,
    /// Call type ("function"), set on first delta.
    pub call_type: Option<String>,
    /// Function name (set once name is parsed).
    pub function_name: Option<String>,
    /// Argument fragment diff (incremental argument text).
    pub function_arguments: Option<String>,
}

/// A streaming delta that can be either content or tool calls.
#[derive(Debug, Clone)]
pub enum ToolParserDelta {
    /// Regular text content (before tool calls).
    Content(String),
    /// One or more tool call deltas.
    ToolCalls(Vec<DeltaToolCall>),
    /// Skip this token (buffering partial tags).
    None,
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Trait for non-streaming tool call extraction.
pub trait ToolCallParser: Send + Sync {
    /// Extract tool calls from complete model output text.
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo;

    /// Create a new per-request streaming state machine.
    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send>;
}

/// Per-request streaming state machine for incremental tool call parsing.
pub trait StreamingToolParserState: Send {
    /// Process a new text delta.
    ///
    /// - `previous_text`: all text before this delta
    /// - `current_text`: all text including this delta (previous_text + delta_text)
    /// - `delta_text`: the new text fragment
    fn process_delta(
        &mut self,
        previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta;
}

// ---------------------------------------------------------------------------
// Hermes tool parser
// ---------------------------------------------------------------------------

const HERMES_TOOL_CALL_OPEN: &str = "<tool_call>";
const HERMES_TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Hermes-style tool call parser.
///
/// Detects tool calls wrapped in `<tool_call>...</tool_call>` tags.
/// The content between tags should be JSON with `name` and `arguments` fields.
#[derive(Default)]
pub struct HermesToolParser;

impl HermesToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for HermesToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        hermes_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(HermesStreamingState::new())
    }
}

/// Extract tool calls from Hermes-formatted text.
fn hermes_extract(text: &str) -> ExtractedToolCallInfo {
    let mut tool_calls = Vec::new();
    let mut content_before = String::new();

    // Find text before first <tool_call> tag.
    let first_open = text.find(HERMES_TOOL_CALL_OPEN);
    if let Some(pos) = first_open {
        let before = text[..pos].trim();
        if !before.is_empty() {
            content_before = before.to_string();
        }
    } else {
        // No tool call tags at all.
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    }

    // Extract all <tool_call>...</tool_call> blocks.
    let mut search_start = 0;
    while let Some(open_pos) = text[search_start..].find(HERMES_TOOL_CALL_OPEN) {
        let open_pos = search_start + open_pos;
        let json_start = open_pos + HERMES_TOOL_CALL_OPEN.len();

        // Find closing tag or end of string (unclosed tag).
        let json_end = text[json_start..]
            .find(HERMES_TOOL_CALL_CLOSE)
            .map(|p| json_start + p)
            .unwrap_or(text.len());

        let json_str = text[json_start..json_end].trim();
        if let Some(tc) = parse_tool_call_json(json_str) {
            tool_calls.push(tc);
        }

        search_start = if json_end < text.len() {
            json_end + HERMES_TOOL_CALL_CLOSE.len()
        } else {
            text.len()
        };
    }

    if tool_calls.is_empty() {
        // Tags found but JSON was malformed — fall back to content.
        ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        }
    } else {
        ExtractedToolCallInfo {
            tools_called: true,
            tool_calls,
            content: if content_before.is_empty() {
                None
            } else {
                Some(content_before)
            },
        }
    }
}

/// Parse a JSON string into a ToolCall.
///
/// Expects `{"name": "...", "arguments": {...}}` format.
fn parse_tool_call_json(json_str: &str) -> Option<protocol::ToolCall> {
    let val: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let name = val.get("name")?.as_str()?.to_string();
    let arguments = val.get("arguments")?;
    let arguments_str = if arguments.is_string() {
        arguments.as_str().unwrap().to_string()
    } else {
        serde_json::to_string(arguments).ok()?
    };

    Some(protocol::ToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: protocol::FunctionCall {
            name,
            arguments: arguments_str,
        },
    })
}

// ---------------------------------------------------------------------------
// Hermes streaming state machine
// ---------------------------------------------------------------------------

struct HermesStreamingState {
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Previously parsed tool call JSON objects.
    prev_tool_call_arr: Vec<serde_json::Value>,
    /// Streamed argument characters for each tool call (for diffing).
    streamed_args_for_tool: Vec<String>,
    /// Buffer for partial tag tokens.
    buffer: String,
    /// Number of tool_call open tags seen so far.
    num_open_tags: usize,
    /// Number of tool_call close tags seen so far.
    num_close_tags: usize,
}

impl HermesStreamingState {
    fn new() -> Self {
        Self {
            current_tool_id: -1,
            current_tool_name_sent: false,
            prev_tool_call_arr: Vec::new(),
            streamed_args_for_tool: Vec::new(),
            buffer: String::new(),
            num_open_tags: 0,
            num_close_tags: 0,
        }
    }
}

impl StreamingToolParserState for HermesStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        // Append delta to buffer.
        self.buffer.push_str(delta_text);

        // Check if buffer contains partial open/close tags that need more tokens.
        if is_partial_tag(&self.buffer) {
            return ToolParserDelta::None;
        }

        // Drain the buffer.
        let text_to_process = std::mem::take(&mut self.buffer);

        // Count open/close tags in the full text so far.
        let new_open_count = current_text.matches(HERMES_TOOL_CALL_OPEN).count();
        let new_close_count = current_text.matches(HERMES_TOOL_CALL_CLOSE).count();

        // If no tool call tags yet, emit as content.
        if new_open_count == 0 {
            return ToolParserDelta::Content(text_to_process);
        }

        // Check if a new tool call just started.
        if new_open_count > self.num_open_tags {
            self.num_open_tags = new_open_count;
            self.current_tool_id += 1;
            self.current_tool_name_sent = false;
            self.prev_tool_call_arr.push(serde_json::Value::Null);
            self.streamed_args_for_tool.push(String::new());

            // The text_to_process might contain text before <tool_call>.
            // If current_tool_id == 0, there might be leading content.
            if self.current_tool_id == 0 {
                // Find content before the first <tool_call> in current_text.
                if let Some(pos) = current_text.find(HERMES_TOOL_CALL_OPEN) {
                    let before = current_text[..pos].to_string();
                    if !before.is_empty() {
                        // We already emitted this as content in previous deltas.
                        // Just skip — the content was already streamed.
                    }
                }
            }
        }

        // Update close tag count.
        self.num_close_tags = new_close_count;

        // Extract the JSON content for the current tool call.
        let tool_idx = self.current_tool_id as usize;

        // Get text after the last <tool_call> open tag.
        let after_last_open = {
            let mut start = 0;
            for _ in 0..self.num_open_tags {
                if let Some(pos) = current_text[start..].find(HERMES_TOOL_CALL_OPEN) {
                    start = start + pos + HERMES_TOOL_CALL_OPEN.len();
                }
            }
            // If there's a close tag, take text up to it.
            if let Some(close_pos) = current_text[start..].find(HERMES_TOOL_CALL_CLOSE) {
                &current_text[start..start + close_pos]
            } else {
                &current_text[start..]
            }
        };

        // Try partial JSON parse of the tool call content.
        let parsed = partial_json_parse(after_last_open.trim());

        if let Some(ref val) = parsed {
            // Try to extract name.
            if !self.current_tool_name_sent
                && let Some(name) = val.get("name").and_then(|n| n.as_str())
            {
                self.current_tool_name_sent = true;
                self.prev_tool_call_arr[tool_idx] = val.clone();

                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: Some(format!("call_{}", Uuid::new_v4().simple())),
                    call_type: Some("function".to_string()),
                    function_name: Some(name.to_string()),
                    function_arguments: Some(String::new()),
                }]);
            }

            // Stream arguments diff.
            if self.current_tool_name_sent {
                let current_args = val
                    .get("arguments")
                    .map(|a| {
                        if a.is_string() {
                            a.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(a).unwrap_or_default()
                        }
                    })
                    .unwrap_or_default();

                let prev_args = &self.streamed_args_for_tool[tool_idx];
                if current_args.len() > prev_args.len() {
                    let diff = current_args[prev_args.len()..].to_string();
                    self.streamed_args_for_tool[tool_idx] = current_args;
                    self.prev_tool_call_arr[tool_idx] = val.clone();

                    return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                        index: tool_idx as u32,
                        id: None,
                        call_type: None,
                        function_name: None,
                        function_arguments: Some(diff),
                    }]);
                }
            }
        }

        ToolParserDelta::None
    }
}

/// Check if text ends with a partial `<tool_call>` or `</tool_call>` tag.
fn is_partial_tag(text: &str) -> bool {
    // Check suffixes of <tool_call> and </tool_call>.
    for tag in [HERMES_TOOL_CALL_OPEN, HERMES_TOOL_CALL_CLOSE] {
        for i in 1..tag.len() {
            if text.ends_with(&tag[..i]) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// LLaMA JSON tool parser
// ---------------------------------------------------------------------------

const LLAMA_PYTHON_TAG: &str = "<|python_tag|>";

/// LLaMA-style JSON tool call parser.
///
/// Detects tool calls as raw JSON objects in the output, optionally
/// prefixed with `<|python_tag|>`.
#[derive(Default)]
pub struct LlamaJsonToolParser;

impl LlamaJsonToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for LlamaJsonToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        llama_json_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(LlamaJsonStreamingState::new())
    }
}

/// Extract tool calls from LLaMA JSON-formatted text.
fn llama_json_extract(text: &str) -> ExtractedToolCallInfo {
    // Strip <|python_tag|> if present.
    let text = text
        .strip_prefix(LLAMA_PYTHON_TAG)
        .unwrap_or(text)
        .trim_start();

    // Try to find JSON objects.
    let mut tool_calls = Vec::new();
    let mut content_before = String::new();
    let mut found_first_json = false;

    let mut pos = 0;
    while pos < text.len() {
        if let Some(brace_pos) = text[pos..].find('{') {
            let abs_pos = pos + brace_pos;
            if !found_first_json {
                let before = text[..abs_pos].trim();
                if !before.is_empty() {
                    content_before = before.to_string();
                }
                found_first_json = true;
            }

            // Try to parse a JSON object starting here.
            if let Some((val, end_pos)) = try_parse_json_object(&text[abs_pos..]) {
                if let Some(tc) = json_value_to_tool_call(&val) {
                    tool_calls.push(tc);
                }
                pos = abs_pos + end_pos;
            } else {
                pos = abs_pos + 1;
            }
        } else {
            break;
        }
    }

    if tool_calls.is_empty() {
        ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        }
    } else {
        ExtractedToolCallInfo {
            tools_called: true,
            tool_calls,
            content: if content_before.is_empty() {
                None
            } else {
                Some(content_before)
            },
        }
    }
}

/// Try to parse a JSON object from the start of `text`.
/// Returns (value, bytes_consumed) on success.
fn try_parse_json_object(text: &str) -> Option<(serde_json::Value, usize)> {
    // Use serde's streaming deserializer to find where the JSON ends.
    let mut de = serde_json::Deserializer::from_str(text).into_iter::<serde_json::Value>();
    if let Some(Ok(val)) = de.next() {
        let end = de.byte_offset();
        if val.is_object() {
            return Some((val, end));
        }
    }
    None
}

/// Convert a JSON value with "name" and "arguments" into a ToolCall.
fn json_value_to_tool_call(val: &serde_json::Value) -> Option<protocol::ToolCall> {
    let name = val.get("name")?.as_str()?.to_string();
    let arguments = val.get("arguments")?;
    // Also accept "parameters" as an alias.
    let arguments = if arguments.is_null() {
        val.get("parameters")?
    } else {
        arguments
    };

    let arguments_str = if arguments.is_string() {
        arguments.as_str().unwrap().to_string()
    } else {
        serde_json::to_string(arguments).ok()?
    };

    Some(protocol::ToolCall {
        id: format!("call_{}", Uuid::new_v4().simple()),
        call_type: "function".to_string(),
        function: protocol::FunctionCall {
            name,
            arguments: arguments_str,
        },
    })
}

// ---------------------------------------------------------------------------
// LLaMA JSON streaming state machine
// ---------------------------------------------------------------------------

struct LlamaJsonStreamingState {
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Streamed argument characters for each tool call.
    streamed_args_for_tool: Vec<String>,
    /// Whether we've seen the start of JSON (first `{`).
    json_started: bool,
    /// Brace depth for tracking JSON object boundaries.
    brace_depth: i32,
    /// Start position of the current JSON object in the full text.
    current_json_start: usize,
}

impl LlamaJsonStreamingState {
    fn new() -> Self {
        Self {
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            json_started: false,
            brace_depth: 0,
            current_json_start: 0,
        }
    }
}

impl StreamingToolParserState for LlamaJsonStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        // Strip python tag from current_text for analysis.
        let effective_text = current_text
            .strip_prefix(LLAMA_PYTHON_TAG)
            .unwrap_or(current_text);

        // If we haven't seen a `{` yet, emit as content.
        if !self.json_started {
            if let Some(brace_pos) = effective_text.find('{') {
                self.json_started = true;
                self.brace_depth = 0;
                self.current_json_start = brace_pos;

                // Count braces in the text from brace_pos.
                for ch in effective_text[brace_pos..].chars() {
                    match ch {
                        '{' => self.brace_depth += 1,
                        '}' => self.brace_depth -= 1,
                        _ => {}
                    }
                }

                // Start a new tool call.
                self.current_tool_id += 1;
                self.current_tool_name_sent = false;
                self.streamed_args_for_tool.push(String::new());

                // Content before the first `{` in delta_text.
                let delta_stripped = delta_text
                    .strip_prefix(LLAMA_PYTHON_TAG)
                    .unwrap_or(delta_text);
                if let Some(pos) = delta_stripped.find('{') {
                    let before = &delta_stripped[..pos];
                    if !before.is_empty() {
                        return ToolParserDelta::Content(before.to_string());
                    }
                }
            } else {
                // No JSON yet, strip python_tag from delta for content.
                let delta_stripped = delta_text
                    .strip_prefix(LLAMA_PYTHON_TAG)
                    .unwrap_or(delta_text);
                if delta_stripped.is_empty() {
                    return ToolParserDelta::None;
                }
                return ToolParserDelta::Content(delta_stripped.to_string());
            }
        } else {
            // Track brace depth for new characters.
            for ch in delta_text.chars() {
                match ch {
                    '{' => self.brace_depth += 1,
                    '}' => self.brace_depth -= 1,
                    _ => {}
                }
            }

            // If depth returns to 0, current JSON object is complete.
            if self.brace_depth == 0 {
                // A new JSON object might start.
                // Check if there's another `{` after current close.
                let remainder = effective_text[self.current_json_start..].trim();
                if let Some((val, end)) = try_parse_json_object(remainder) {
                    let after = remainder[end..].trim_start();
                    if after.starts_with('{') {
                        // New tool call starting.
                        self.current_tool_id += 1;
                        self.current_tool_name_sent = false;
                        self.streamed_args_for_tool.push(String::new());
                        self.current_json_start += end + remainder[end..].find('{').unwrap_or(0);
                        self.brace_depth = 1; // for the new opening brace
                    }
                    let _ = val; // processed below via partial parse
                }
            }
        }

        // Try partial JSON parse of the current tool call's text.
        let tool_idx = self.current_tool_id as usize;
        let json_text = &effective_text[self.current_json_start..];
        let parsed = partial_json_parse(json_text.trim());

        if let Some(ref val) = parsed {
            if !self.current_tool_name_sent
                && let Some(name) = val.get("name").and_then(|n| n.as_str())
            {
                self.current_tool_name_sent = true;
                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: Some(format!("call_{}", Uuid::new_v4().simple())),
                    call_type: Some("function".to_string()),
                    function_name: Some(name.to_string()),
                    function_arguments: Some(String::new()),
                }]);
            }

            if self.current_tool_name_sent {
                let current_args = val
                    .get("arguments")
                    .map(|a| {
                        if a.is_string() {
                            a.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(a).unwrap_or_default()
                        }
                    })
                    .unwrap_or_default();

                let prev_args = &self.streamed_args_for_tool[tool_idx];
                if current_args.len() > prev_args.len() {
                    let diff = current_args[prev_args.len()..].to_string();
                    self.streamed_args_for_tool[tool_idx] = current_args;

                    return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                        index: tool_idx as u32,
                        id: None,
                        call_type: None,
                        function_name: None,
                        function_arguments: Some(diff),
                    }]);
                }
            }
        }

        ToolParserDelta::None
    }
}

// ---------------------------------------------------------------------------
// Partial JSON helper
// ---------------------------------------------------------------------------

/// Attempt to parse potentially incomplete JSON by closing open braces/brackets.
///
/// Tries parsing as-is first, then attempts to fix by appending closing
/// characters for unmatched `{` and `[`.
pub fn partial_json_parse(input: &str) -> Option<serde_json::Value> {
    // Try parsing as-is first.
    if let Ok(v) = serde_json::from_str(input) {
        return Some(v);
    }

    // Count unmatched braces/brackets (outside strings).
    let mut open_braces = 0i32;
    let mut open_brackets = 0i32;
    let mut in_string = false;
    let mut prev_backslash = false;

    for ch in input.chars() {
        if in_string {
            if ch == '\\' && !prev_backslash {
                prev_backslash = true;
                continue;
            }
            if ch == '"' && !prev_backslash {
                in_string = false;
            }
            prev_backslash = false;
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => open_braces += 1,
            '}' => open_braces -= 1,
            '[' => open_brackets += 1,
            ']' => open_brackets -= 1,
            _ => {}
        }
        prev_backslash = false;
    }

    if open_braces <= 0 && open_brackets <= 0 {
        return None; // Not fixable by closing braces.
    }

    // Try closing the string if we're inside one, then close braces/brackets.
    let mut fixed = input.to_string();

    // If we ended inside a string, close it.
    if in_string {
        fixed.push('"');
    }

    // Close brackets then braces (inner-to-outer order).
    for _ in 0..open_brackets {
        fixed.push(']');
    }
    for _ in 0..open_braces {
        fixed.push('}');
    }

    serde_json::from_str(&fixed).ok()
}

// ---------------------------------------------------------------------------
// Parser registry
// ---------------------------------------------------------------------------

/// Get a tool call parser by name.
pub fn get_tool_parser(name: &str) -> Result<Arc<dyn ToolCallParser>, String> {
    match name {
        "hermes" => Ok(Arc::new(HermesToolParser::new())),
        "llama3_json" | "llama4_json" => Ok(Arc::new(LlamaJsonToolParser::new())),
        other => Err(format!("Unknown tool call parser: {other}")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Hermes non-streaming tests --

    #[test]
    fn test_hermes_single_tool_call() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"city":"SF"}"#);
        assert_eq!(result.tool_calls[0].call_type, "function");
        assert!(result.tool_calls[0].id.starts_with("call_"));
        assert!(result.content.is_none());
    }

    #[test]
    fn test_hermes_two_tool_calls() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>{"name":"search","arguments":{"q":"rust"}}</tool_call><tool_call>{"name":"fetch","arguments":{"url":"https://example.com"}}</tool_call>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "search");
        assert_eq!(result.tool_calls[1].function.name, "fetch");
    }

    #[test]
    fn test_hermes_text_before_tool_call() {
        let parser = HermesToolParser::new();
        let output = r#"I'll help you with that.
<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content.unwrap(), "I'll help you with that.");
    }

    #[test]
    fn test_hermes_no_tool_call() {
        let parser = HermesToolParser::new();
        let output = "The weather in SF is sunny and 72°F.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), output);
    }

    #[test]
    fn test_hermes_malformed_json() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>not json at all</tool_call>"#;
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        // Falls back to treating it as content.
        assert!(result.content.is_some());
    }

    #[test]
    fn test_hermes_unclosed_tag() {
        let parser = HermesToolParser::new();
        let output = r#"<tool_call>{"name":"f","arguments":{"a":1}}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "f");
    }

    // -- LLaMA JSON non-streaming tests --

    #[test]
    fn test_llama_single_json() {
        let parser = LlamaJsonToolParser::new();
        let output = r#"{"name":"get_weather","arguments":{"city":"SF"}}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_llama_two_json_objects() {
        let parser = LlamaJsonToolParser::new();
        let output =
            r#"{"name":"search","arguments":{"q":"rust"}}{"name":"fetch","arguments":{"url":"x"}}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "search");
        assert_eq!(result.tool_calls[1].function.name, "fetch");
    }

    #[test]
    fn test_llama_python_tag() {
        let parser = LlamaJsonToolParser::new();
        let output = r#"<|python_tag|>{"name":"get_weather","arguments":{"city":"SF"}}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_llama_plain_text() {
        let parser = LlamaJsonToolParser::new();
        let output = "Just a plain text response with no JSON.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.content.is_some());
    }

    // -- Partial JSON tests --

    #[test]
    fn test_partial_json_incomplete() {
        let input = r#"{"name": "f", "arguments": {"q": "he"#;
        let result = partial_json_parse(input);
        assert!(result.is_some());
        let val = result.unwrap();
        assert_eq!(val["name"], "f");
        assert_eq!(val["arguments"]["q"], "he");
    }

    #[test]
    fn test_partial_json_complete() {
        let input = r#"{"name": "f", "arguments": {"q": "hello"}}"#;
        let result = partial_json_parse(input);
        assert!(result.is_some());
        let val = result.unwrap();
        assert_eq!(val["name"], "f");
    }

    // -- Hermes streaming tests --

    #[test]
    fn test_hermes_streaming_basic() {
        let parser = HermesToolParser::new();
        let mut state = parser.create_streaming_state();

        // Simulate tokens for: <tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>
        let tokens = vec![
            "<tool_call>",
            r#"{"name":"#,
            r#""get_weather","#,
            r#""arguments":{"#,
            r#""city":"SF"#,
            r#"}}"#,
            "</tool_call>",
        ];

        let mut accumulated = String::new();
        let mut got_name = false;
        let mut got_args = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            match state.process_delta(&prev, &accumulated, token) {
                ToolParserDelta::ToolCalls(calls) => {
                    for call in &calls {
                        if call.function_name.is_some() {
                            got_name = true;
                            assert_eq!(call.function_name.as_ref().unwrap(), "get_weather");
                            assert_eq!(call.index, 0);
                            assert!(call.id.is_some());
                        }
                        if call.function_arguments.is_some()
                            && !call.function_arguments.as_ref().unwrap().is_empty()
                        {
                            got_args = true;
                        }
                    }
                }
                _ => {}
            }
        }

        assert!(got_name, "Should have received tool name");
        assert!(got_args, "Should have received argument fragments");
    }

    #[test]
    fn test_hermes_streaming_content_then_tool() {
        let parser = HermesToolParser::new();
        let mut state = parser.create_streaming_state();

        let mut accumulated = String::new();
        let mut got_content = false;
        let mut got_tool = false;

        // First, some content.
        let prev = accumulated.clone();
        accumulated.push_str("Sure, ");
        match state.process_delta(&prev, &accumulated, "Sure, ") {
            ToolParserDelta::Content(text) => {
                assert_eq!(text, "Sure, ");
                got_content = true;
            }
            _ => {}
        }

        // Then tool call.
        let tokens = vec![
            "<tool_call>",
            r#"{"name":"f","arguments":{}}"#,
            "</tool_call>",
        ];
        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            match state.process_delta(&prev, &accumulated, token) {
                ToolParserDelta::ToolCalls(calls) => {
                    got_tool = true;
                    for call in &calls {
                        if let Some(name) = &call.function_name {
                            assert_eq!(name, "f");
                        }
                    }
                }
                _ => {}
            }
        }

        assert!(got_content, "Should have received content");
        assert!(got_tool, "Should have received tool call");
    }

    // -- Registry tests --

    #[test]
    fn test_registry_hermes() {
        assert!(get_tool_parser("hermes").is_ok());
    }

    #[test]
    fn test_registry_llama3_json() {
        assert!(get_tool_parser("llama3_json").is_ok());
    }

    #[test]
    fn test_registry_llama4_json() {
        assert!(get_tool_parser("llama4_json").is_ok());
    }

    #[test]
    fn test_registry_unknown() {
        assert!(get_tool_parser("unknown").is_err());
    }
}
