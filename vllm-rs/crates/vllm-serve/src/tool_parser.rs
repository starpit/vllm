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

    // Track unmatched delimiters in order (outside strings).
    let mut delimiter_stack: Vec<char> = Vec::new();
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
            '{' => delimiter_stack.push('{'),
            '}' => {
                delimiter_stack.pop();
            }
            '[' => delimiter_stack.push('['),
            ']' => {
                delimiter_stack.pop();
            }
            _ => {}
        }
        prev_backslash = false;
    }

    if delimiter_stack.is_empty() && !in_string {
        return None; // Not fixable by closing delimiters.
    }

    // Try closing the string if we're inside one, then close delimiters
    // in reverse order (innermost first).
    let mut fixed = input.to_string();

    if in_string {
        fixed.push('"');
    }

    for &delim in delimiter_stack.iter().rev() {
        match delim {
            '{' => fixed.push('}'),
            '[' => fixed.push(']'),
            _ => {}
        }
    }

    serde_json::from_str(&fixed).ok()
}

// ---------------------------------------------------------------------------
// Kimi K2 tool parser
// ---------------------------------------------------------------------------

/// Markers for Kimi K2 tool call format.
const KIMI_SECTION_BEGIN: &str = "<|tool_calls_section_begin|>";
const KIMI_SECTION_BEGIN_SINGULAR: &str = "<|tool_call_section_begin|>";
const KIMI_SECTION_END: &str = "<|tool_calls_section_end|>";
const KIMI_SECTION_END_SINGULAR: &str = "<|tool_call_section_end|>";
const KIMI_CALL_BEGIN: &str = "<|tool_call_begin|>";
const KIMI_CALL_ARG_BEGIN: &str = "<|tool_call_argument_begin|>";
const KIMI_CALL_END: &str = "<|tool_call_end|>";

/// Kimi K2-style tool call parser.
///
/// Format:
/// ```text
/// <|tool_calls_section_begin|>
/// <|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|> {"city": "SF"} <|tool_call_end|>
/// <|tool_calls_section_end|>
/// ```
#[derive(Default)]
pub struct KimiK2ToolParser;

impl KimiK2ToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for KimiK2ToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        kimi_k2_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(KimiK2StreamingState::new())
    }
}

/// Extract the content before the tool calls section.
fn kimi_k2_find_section_start(text: &str) -> Option<usize> {
    text.find(KIMI_SECTION_BEGIN)
        .or_else(|| text.find(KIMI_SECTION_BEGIN_SINGULAR))
}

/// Parse a Kimi K2 tool call ID into a function name.
///
/// `functions.get_weather:0` → `get_weather`
fn kimi_k2_parse_function_name(tool_call_id: &str) -> String {
    let name_part = tool_call_id
        .rsplit_once(':')
        .map(|(name, _)| name)
        .unwrap_or(tool_call_id);
    name_part
        .rsplit_once('.')
        .map(|(_, name)| name)
        .unwrap_or(name_part)
        .to_string()
}

/// Extract tool calls from Kimi K2-formatted text.
fn kimi_k2_extract(text: &str) -> ExtractedToolCallInfo {
    let section_start = match kimi_k2_find_section_start(text) {
        Some(pos) => pos,
        None => {
            return ExtractedToolCallInfo {
                tools_called: false,
                tool_calls: Vec::new(),
                content: Some(text.to_string()),
            };
        }
    };

    let content_before = text[..section_start].trim();
    let content = if content_before.is_empty() {
        None
    } else {
        Some(content_before.to_string())
    };

    // Extract individual tool calls using the markers.
    let mut tool_calls = Vec::new();
    let mut search_pos = section_start;

    while let Some(call_begin) = text[search_pos..].find(KIMI_CALL_BEGIN) {
        let call_begin = search_pos + call_begin + KIMI_CALL_BEGIN.len();

        // Find the argument section.
        let arg_begin = match text[call_begin..].find(KIMI_CALL_ARG_BEGIN) {
            Some(pos) => call_begin + pos,
            None => break,
        };
        let tool_call_id = text[call_begin..arg_begin].trim();
        let args_start = arg_begin + KIMI_CALL_ARG_BEGIN.len();

        // Find the end of this tool call.
        let call_end = text[args_start..]
            .find(KIMI_CALL_END)
            .map(|p| args_start + p)
            .unwrap_or(text.len());
        let args_str = text[args_start..call_end].trim();

        let function_name = kimi_k2_parse_function_name(tool_call_id);

        // Validate JSON arguments.
        if serde_json::from_str::<serde_json::Value>(args_str).is_ok() {
            tool_calls.push(protocol::ToolCall {
                id: format!("call_{}", Uuid::new_v4().simple()),
                call_type: "function".to_string(),
                function: protocol::FunctionCall {
                    name: function_name,
                    arguments: args_str.to_string(),
                },
            });
        }

        search_pos = if call_end < text.len() {
            call_end + KIMI_CALL_END.len()
        } else {
            text.len()
        };
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
            content,
        }
    }
}

// ---------------------------------------------------------------------------
// Kimi K2 streaming state machine
// ---------------------------------------------------------------------------

struct KimiK2StreamingState {
    /// Whether we're inside a tool_calls_section.
    in_tool_section: bool,
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Streamed argument characters for each tool call.
    streamed_args_for_tool: Vec<String>,
    /// Buffer for partial tag tokens.
    buffer: String,
    /// Number of tool_call_begin tags seen.
    num_call_begins: usize,
    /// Number of tool_call_end tags seen.
    num_call_ends: usize,
}

impl KimiK2StreamingState {
    fn new() -> Self {
        Self {
            in_tool_section: false,
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            buffer: String::new(),
            num_call_begins: 0,
            num_call_ends: 0,
        }
    }
}

/// Check if text ends with a partial Kimi K2 tag.
fn is_partial_kimi_tag(text: &str) -> bool {
    for tag in [
        KIMI_SECTION_BEGIN,
        KIMI_SECTION_BEGIN_SINGULAR,
        KIMI_SECTION_END,
        KIMI_SECTION_END_SINGULAR,
        KIMI_CALL_BEGIN,
        KIMI_CALL_ARG_BEGIN,
        KIMI_CALL_END,
    ] {
        for i in 1..tag.len() {
            if text.ends_with(&tag[..i]) {
                return true;
            }
        }
    }
    false
}

impl StreamingToolParserState for KimiK2StreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        self.buffer.push_str(delta_text);

        // Wait for more tokens if we might be in a partial tag.
        if is_partial_kimi_tag(&self.buffer) {
            return ToolParserDelta::None;
        }

        let text_to_process = std::mem::take(&mut self.buffer);

        // Check if the section has started.
        if !self.in_tool_section {
            if current_text.contains(KIMI_SECTION_BEGIN)
                || current_text.contains(KIMI_SECTION_BEGIN_SINGULAR)
            {
                self.in_tool_section = true;
                // Emit any content before the section marker in this delta.
                let before_marker = if let Some(pos) = text_to_process.find("<|tool_call") {
                    &text_to_process[..pos]
                } else {
                    ""
                };
                if !before_marker.is_empty() {
                    return ToolParserDelta::Content(before_marker.to_string());
                }
                return ToolParserDelta::None;
            }
            // No section yet — emit as content.
            return ToolParserDelta::Content(text_to_process);
        }

        // We're inside the tool calls section.
        // Count tool_call_begin/end tags in full text.
        let new_begins = current_text.matches(KIMI_CALL_BEGIN).count();
        let new_ends = current_text.matches(KIMI_CALL_END).count();

        // New tool call started?
        if new_begins > self.num_call_begins {
            self.num_call_begins = new_begins;
            self.current_tool_id += 1;
            self.current_tool_name_sent = false;
            self.streamed_args_for_tool.push(String::new());
        }
        self.num_call_ends = new_ends;

        if self.current_tool_id < 0 {
            return ToolParserDelta::None;
        }

        let tool_idx = self.current_tool_id as usize;

        // Extract the current tool call's text from current_text.
        // Find the Nth tool_call_begin tag.
        let mut search = 0;
        for _ in 0..self.num_call_begins {
            if let Some(pos) = current_text[search..].find(KIMI_CALL_BEGIN) {
                search = search + pos + KIMI_CALL_BEGIN.len();
            }
        }
        let after_last_begin = &current_text[search..];

        // Try to extract function name (before <|tool_call_argument_begin|>).
        if !self.current_tool_name_sent
            && let Some(arg_pos) = after_last_begin.find(KIMI_CALL_ARG_BEGIN)
        {
            let tool_call_id = after_last_begin[..arg_pos].trim();
            if !tool_call_id.is_empty() {
                let function_name = kimi_k2_parse_function_name(tool_call_id);
                self.current_tool_name_sent = true;

                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: Some(format!("call_{}", Uuid::new_v4().simple())),
                    call_type: Some("function".to_string()),
                    function_name: Some(function_name),
                    function_arguments: Some(String::new()),
                }]);
            }
        }

        // Stream argument diffs.
        if self.current_tool_name_sent {
            // Get text after <|tool_call_argument_begin|>.
            if let Some(arg_start_pos) = after_last_begin.find(KIMI_CALL_ARG_BEGIN) {
                let args_text_start = arg_start_pos + KIMI_CALL_ARG_BEGIN.len();
                let args_text = if let Some(end_pos) =
                    after_last_begin[args_text_start..].find(KIMI_CALL_END)
                {
                    &after_last_begin[args_text_start..args_text_start + end_pos]
                } else {
                    &after_last_begin[args_text_start..]
                };
                let args_text = args_text.trim_start();

                let prev_args = &self.streamed_args_for_tool[tool_idx];
                if args_text.len() > prev_args.len() {
                    let diff = args_text[prev_args.len()..].to_string();
                    self.streamed_args_for_tool[tool_idx] = args_text.to_string();

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
// Mistral tool parser
// ---------------------------------------------------------------------------

const MISTRAL_BOT_TOKEN: &str = "[TOOL_CALLS]";

/// Generate a 9-character alphanumeric random ID matching Mistral's format.
fn mistral_generate_id() -> String {
    use rand::Rng;
    const ALPHANUMERIC: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::thread_rng();
    (0..9)
        .map(|_| ALPHANUMERIC[rng.gen_range(0..ALPHANUMERIC.len())] as char)
        .collect()
}

/// Mistral-style tool call parser.
///
/// Supports two formats:
/// - v11+: `[TOOL_CALLS]func_name{"arg":"val"}[TOOL_CALLS]func2{"arg2":"val2"}`
/// - Pre-v11: `[TOOL_CALLS] [{"name":"func","arguments":{"arg":"val"}}]`
///
/// Format is auto-detected by checking if text after `[TOOL_CALLS]` starts with `[`.
#[derive(Default)]
pub struct MistralToolParser;

impl MistralToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for MistralToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        mistral_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(MistralStreamingState::new())
    }
}

fn mistral_extract(text: &str) -> ExtractedToolCallInfo {
    if !text.contains(MISTRAL_BOT_TOKEN) {
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    }

    let parts: Vec<&str> = text.splitn(2, MISTRAL_BOT_TOKEN).collect();
    let content = parts[0];
    let rest = parts.get(1).unwrap_or(&"");

    // Auto-detect format: if rest starts with `[` (after trimming), it's pre-v11
    let trimmed_rest = rest.trim_start();
    let is_pre_v11 = trimmed_rest.starts_with('[');

    let tool_calls = if is_pre_v11 {
        // Pre-v11: `[{"name":"func","arguments":{...}}]`
        mistral_extract_pre_v11(trimmed_rest)
    } else {
        // v11+: split on [TOOL_CALLS] for multiple tools
        // rest is everything after the first [TOOL_CALLS], may contain more [TOOL_CALLS] delimiters
        let full_tool_text = &text[parts[0].len() + MISTRAL_BOT_TOKEN.len()..];
        let segments: Vec<&str> = full_tool_text.split(MISTRAL_BOT_TOKEN).collect();
        let mut calls = Vec::new();
        for segment in segments {
            if let Some(brace_pos) = segment.find('{') {
                let name = &segment[..brace_pos];
                let args = &segment[brace_pos..];
                calls.push(protocol::ToolCall {
                    id: mistral_generate_id(),
                    call_type: "function".to_string(),
                    function: protocol::FunctionCall {
                        name: name.to_string(),
                        arguments: args.to_string(),
                    },
                });
            }
        }
        calls
    };

    if tool_calls.is_empty() {
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    }

    ExtractedToolCallInfo {
        tools_called: true,
        tool_calls,
        content: if content.is_empty() {
            None
        } else {
            Some(content.to_string())
        },
    }
}

fn mistral_extract_pre_v11(json_text: &str) -> Vec<protocol::ToolCall> {
    // Try direct JSON parse first
    let parsed: Result<Vec<serde_json::Value>, _> = serde_json::from_str(json_text);
    let arr = match parsed {
        Ok(arr) => arr,
        Err(_) => {
            // Fallback: find `[{...}]` pattern (matching Python's regex r"\[{.*}\]")
            let start = json_text.find("[{");
            let end = json_text.rfind("}]");
            if let (Some(s), Some(e)) = (start, end) {
                let substr = &json_text[s..e + 2];
                match serde_json::from_str::<Vec<serde_json::Value>>(substr) {
                    Ok(arr) => arr,
                    Err(_) => return Vec::new(),
                }
            } else {
                return Vec::new();
            }
        }
    };

    arr.into_iter()
        .filter_map(|val| {
            let name = val.get("name")?.as_str()?.to_string();
            let arguments = val.get("arguments")?;
            let arguments_str = if arguments.is_string() {
                arguments.as_str().unwrap().to_string()
            } else {
                serde_json::to_string(arguments).ok()?
            };
            Some(protocol::ToolCall {
                id: mistral_generate_id(),
                call_type: "function".to_string(),
                function: protocol::FunctionCall {
                    name,
                    arguments: arguments_str,
                },
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Mistral streaming state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MistralStreamFormat {
    Unknown,
    V11,
    PreV11,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MistralStreamState {
    WaitingForToolStart,
    ParsingName,
    ParsingArguments,
}

struct MistralStreamingState {
    format: MistralStreamFormat,
    state: MistralStreamState,
    current_tool_id: i32,
    current_tool_name: String,
    /// Buffer for accumulating text until we can determine format or complete a parse unit.
    buffer: String,
    /// For pre-v11: brace depth for JSON streaming.
    brace_depth: i32,
    /// For pre-v11: whether we're inside a JSON string.
    in_string: bool,
    /// For pre-v11: previous char was backslash (escape).
    escape_next: bool,
    /// For pre-v11: accumulated arguments JSON for current tool.
    pre_v11_args_buf: String,
    /// For pre-v11: accumulated name.
    pre_v11_key: Option<String>,
    /// For pre-v11: current JSON key being parsed.
    pre_v11_parse_state: PreV11ParseState,
    /// Whether we've seen the bot token at all.
    bot_token_seen: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreV11ParseState {
    /// Looking for the start of an object `{`.
    WaitingForObject,
    /// Inside an object, looking for keys.
    InObject,
    /// Parsing the "name" string value.
    ParsingNameValue,
    /// Parsing the "arguments" object.
    ParsingArgumentsValue,
    /// Object complete, waiting for next or end of array.
    ObjectComplete,
    /// Array complete.
    Done,
}

impl MistralStreamingState {
    fn new() -> Self {
        Self {
            format: MistralStreamFormat::Unknown,
            state: MistralStreamState::WaitingForToolStart,
            current_tool_id: -1,
            current_tool_name: String::new(),
            buffer: String::new(),
            brace_depth: 0,
            in_string: false,
            escape_next: false,
            pre_v11_args_buf: String::new(),
            pre_v11_key: None,
            pre_v11_parse_state: PreV11ParseState::WaitingForObject,
            bot_token_seen: false,
        }
    }

    /// Process v11+ format streaming.
    fn process_v11(&mut self, text: &str) -> Vec<DeltaToolCall> {
        let mut deltas = Vec::new();
        let mut remaining = text;

        loop {
            match self.state {
                MistralStreamState::WaitingForToolStart => {
                    // Look for [TOOL_CALLS] token
                    if let Some(pos) = remaining.find(MISTRAL_BOT_TOKEN) {
                        remaining = &remaining[pos + MISTRAL_BOT_TOKEN.len()..];
                        self.current_tool_id += 1;
                        self.current_tool_name.clear();
                        self.state = MistralStreamState::ParsingName;
                    } else {
                        break;
                    }
                }
                MistralStreamState::ParsingName => {
                    if let Some(brace_pos) = remaining.find('{') {
                        let name_part = &remaining[..brace_pos];
                        self.current_tool_name.push_str(name_part);
                        remaining = &remaining[brace_pos..];
                        self.state = MistralStreamState::ParsingArguments;

                        // Emit name delta with ID
                        deltas.push(DeltaToolCall {
                            index: self.current_tool_id as u32,
                            id: Some(mistral_generate_id()),
                            call_type: Some("function".to_string()),
                            function_name: Some(self.current_tool_name.clone()),
                            function_arguments: None,
                        });
                    } else {
                        // Buffer the name fragment, don't emit yet
                        self.current_tool_name.push_str(remaining);
                        break;
                    }
                }
                MistralStreamState::ParsingArguments => {
                    // Check if there's another [TOOL_CALLS] — means current tool is done
                    if let Some(pos) = remaining.find(MISTRAL_BOT_TOKEN) {
                        let args_part = &remaining[..pos];
                        if !args_part.is_empty() {
                            deltas.push(DeltaToolCall {
                                index: self.current_tool_id as u32,
                                id: None,
                                call_type: None,
                                function_name: None,
                                function_arguments: Some(args_part.to_string()),
                            });
                        }
                        remaining = &remaining[pos..]; // keep [TOOL_CALLS] for next iteration
                        self.state = MistralStreamState::WaitingForToolStart;
                    } else {
                        // All remaining text is arguments
                        if !remaining.is_empty() {
                            deltas.push(DeltaToolCall {
                                index: self.current_tool_id as u32,
                                id: None,
                                call_type: None,
                                function_name: None,
                                function_arguments: Some(remaining.to_string()),
                            });
                        }
                        break;
                    }
                }
            }
        }

        deltas
    }

    /// Process pre-v11 format streaming using brace-counting.
    fn process_pre_v11(&mut self, text: &str) -> Vec<DeltaToolCall> {
        let mut deltas = Vec::new();

        for ch in text.chars() {
            match self.pre_v11_parse_state {
                PreV11ParseState::WaitingForObject => {
                    if ch == '{' {
                        self.pre_v11_parse_state = PreV11ParseState::InObject;
                        self.current_tool_id += 1;
                        self.pre_v11_key = None;
                        self.current_tool_name.clear();
                        self.pre_v11_args_buf.clear();
                        self.brace_depth = 0;
                        self.in_string = false;
                        self.escape_next = false;
                    }
                    // skip [ , whitespace etc.
                }
                PreV11ParseState::InObject => {
                    // We're inside the top-level object, looking for "name" or "arguments" keys
                    // Simple approach: accumulate into buffer until we identify key-value pairs
                    self.buffer.push(ch);

                    // Check if we've accumulated a complete key
                    if self.buffer.contains("\"name\"") && self.buffer.ends_with(':') {
                        self.buffer.clear();
                        self.pre_v11_parse_state = PreV11ParseState::ParsingNameValue;
                        self.in_string = false;
                    } else if self.buffer.contains("\"arguments\"") && self.buffer.ends_with(':') {
                        self.buffer.clear();
                        self.pre_v11_parse_state = PreV11ParseState::ParsingArgumentsValue;
                        self.brace_depth = 0;
                        self.in_string = false;
                        self.escape_next = false;
                    } else if ch == '}' && !self.buffer.contains('"') {
                        // End of object without finding expected keys
                        self.buffer.clear();
                        self.pre_v11_parse_state = PreV11ParseState::ObjectComplete;
                    }
                }
                PreV11ParseState::ParsingNameValue => {
                    // Parse a JSON string value for the name
                    if ch == '"' && !self.in_string {
                        self.in_string = true;
                    } else if self.in_string {
                        if self.escape_next {
                            self.current_tool_name.push(ch);
                            self.escape_next = false;
                        } else if ch == '\\' {
                            self.escape_next = true;
                        } else if ch == '"' {
                            // Name complete — emit it
                            deltas.push(DeltaToolCall {
                                index: self.current_tool_id as u32,
                                id: Some(mistral_generate_id()),
                                call_type: Some("function".to_string()),
                                function_name: Some(self.current_tool_name.clone()),
                                function_arguments: None,
                            });
                            self.in_string = false;
                            self.pre_v11_parse_state = PreV11ParseState::InObject;
                            self.buffer.clear();
                        } else {
                            self.current_tool_name.push(ch);
                        }
                    }
                }
                PreV11ParseState::ParsingArgumentsValue => {
                    // Stream arguments using brace counting
                    if ch == '{' && !self.in_string {
                        self.brace_depth += 1;
                        self.pre_v11_args_buf.push(ch);
                        if self.brace_depth == 1 {
                            // Emit the opening brace as first args delta
                            deltas.push(DeltaToolCall {
                                index: self.current_tool_id as u32,
                                id: None,
                                call_type: None,
                                function_name: None,
                                function_arguments: Some("{".to_string()),
                            });
                            self.pre_v11_args_buf.clear();
                        }
                    } else if ch == '}' && !self.in_string {
                        self.brace_depth -= 1;
                        if self.brace_depth == 0 {
                            // Arguments complete
                            if !self.pre_v11_args_buf.is_empty() {
                                deltas.push(DeltaToolCall {
                                    index: self.current_tool_id as u32,
                                    id: None,
                                    call_type: None,
                                    function_name: None,
                                    function_arguments: Some(self.pre_v11_args_buf.clone()),
                                });
                                self.pre_v11_args_buf.clear();
                            }
                            deltas.push(DeltaToolCall {
                                index: self.current_tool_id as u32,
                                id: None,
                                call_type: None,
                                function_name: None,
                                function_arguments: Some("}".to_string()),
                            });
                            self.pre_v11_parse_state = PreV11ParseState::InObject;
                            self.buffer.clear();
                        } else {
                            self.pre_v11_args_buf.push(ch);
                        }
                    } else {
                        // Handle strings for correct brace counting
                        if ch == '"' && !self.escape_next {
                            self.in_string = !self.in_string;
                        }
                        self.escape_next = ch == '\\' && self.in_string && !self.escape_next;
                        if self.brace_depth > 0 {
                            self.pre_v11_args_buf.push(ch);
                        }
                    }
                }
                PreV11ParseState::ObjectComplete => {
                    if ch == '{' {
                        // Next object
                        self.pre_v11_parse_state = PreV11ParseState::InObject;
                        self.current_tool_id += 1;
                        self.current_tool_name.clear();
                        self.pre_v11_args_buf.clear();
                        self.brace_depth = 0;
                        self.buffer.clear();
                    } else if ch == ']' {
                        self.pre_v11_parse_state = PreV11ParseState::Done;
                    }
                }
                PreV11ParseState::Done => {}
            }
        }

        // Flush any accumulated args for pre-v11 in-progress arguments
        if self.pre_v11_parse_state == PreV11ParseState::ParsingArgumentsValue
            && self.brace_depth > 0
            && !self.pre_v11_args_buf.is_empty()
        {
            deltas.push(DeltaToolCall {
                index: self.current_tool_id as u32,
                id: None,
                call_type: None,
                function_name: None,
                function_arguments: Some(self.pre_v11_args_buf.clone()),
            });
            self.pre_v11_args_buf.clear();
        }

        deltas
    }
}

impl StreamingToolParserState for MistralStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        if delta_text.is_empty() {
            return ToolParserDelta::None;
        }

        // If we haven't seen the bot token yet, check if it's in current_text
        if !self.bot_token_seen {
            if !current_text.contains(MISTRAL_BOT_TOKEN) {
                return ToolParserDelta::Content(delta_text.to_string());
            }
            self.bot_token_seen = true;

            // Extract content before [TOOL_CALLS]
            if delta_text.contains(MISTRAL_BOT_TOKEN) {
                let parts: Vec<&str> = delta_text.splitn(2, MISTRAL_BOT_TOKEN).collect();
                if !parts[0].is_empty() {
                    // Buffer the post-bot-token text for format detection
                    let after = parts.get(1).unwrap_or(&"");
                    if !after.is_empty() {
                        self.buffer.push_str(after);
                    }
                    // Try to detect format now
                    return self.try_detect_and_flush(Some(parts[0].to_string()));
                }
                let after = parts.get(1).unwrap_or(&"");
                if !after.is_empty() {
                    self.buffer.push_str(after);
                }
            }

            return self.try_detect_and_flush(None);
        }

        // If format not yet detected, buffer and try again
        if self.format == MistralStreamFormat::Unknown {
            self.buffer.push_str(delta_text);
            return self.try_detect_and_flush(None);
        }

        // Format detected, process normally
        let deltas = match self.format {
            MistralStreamFormat::V11 => self.process_v11(delta_text),
            MistralStreamFormat::PreV11 => self.process_pre_v11(delta_text),
            MistralStreamFormat::Unknown => Vec::new(),
        };

        if deltas.is_empty() {
            ToolParserDelta::None
        } else {
            ToolParserDelta::ToolCalls(deltas)
        }
    }
}

impl MistralStreamingState {
    /// Try to detect format from buffered text. If detected, flush buffer through parser.
    fn try_detect_and_flush(&mut self, content_before: Option<String>) -> ToolParserDelta {
        let trimmed = self.buffer.trim_start();
        if trimmed.is_empty() {
            // Not enough data to detect format yet
            if let Some(content) = content_before {
                return ToolParserDelta::Content(content);
            }
            return ToolParserDelta::None;
        }

        if trimmed.starts_with('[') {
            self.format = MistralStreamFormat::PreV11;
        } else {
            self.format = MistralStreamFormat::V11;
        }

        // Flush buffered text through the appropriate parser
        // For V11, we need to prepend [TOOL_CALLS] since process_v11 expects it
        let buffered = std::mem::take(&mut self.buffer);
        let deltas = match self.format {
            MistralStreamFormat::V11 => {
                let with_token = format!("{}{}", MISTRAL_BOT_TOKEN, buffered);
                self.process_v11(&with_token)
            }
            MistralStreamFormat::PreV11 => self.process_pre_v11(&buffered),
            MistralStreamFormat::Unknown => Vec::new(),
        };

        match (content_before, deltas.is_empty()) {
            (Some(content), true) => ToolParserDelta::Content(content),
            (_, false) => ToolParserDelta::ToolCalls(deltas),
            (None, true) => ToolParserDelta::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Jamba tool parser
// ---------------------------------------------------------------------------

const JAMBA_TOOL_CALLS_OPEN: &str = "<tool_calls>";
const JAMBA_TOOL_CALLS_CLOSE: &str = "</tool_calls>";

/// Jamba-style tool call parser.
///
/// Detects tool calls wrapped in `<tool_calls>...</tool_calls>` tags.
/// The content between tags is a JSON **array** of objects, each with
/// `name` and `arguments` fields.
///
/// Port of: `vllm/tool_parsers/jamba_tool_parser.py`
#[derive(Default)]
pub struct JambaToolParser;

impl JambaToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for JambaToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        jamba_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(JambaStreamingState::new())
    }
}

/// Extract tool calls from Jamba-formatted text.
fn jamba_extract(text: &str) -> ExtractedToolCallInfo {
    // Check for the open tag.
    let Some(open_pos) = text.find(JAMBA_TOOL_CALLS_OPEN) else {
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    };

    // Content before the open tag.
    let before = text[..open_pos].trim();
    let content = if before.is_empty() {
        None
    } else {
        Some(before.to_string())
    };

    // Extract the JSON array between tags.
    let json_start = open_pos + JAMBA_TOOL_CALLS_OPEN.len();
    let json_end = text[json_start..]
        .find(JAMBA_TOOL_CALLS_CLOSE)
        .map(|p| json_start + p)
        .unwrap_or(text.len());
    let json_str = text[json_start..json_end].trim();

    // Parse as a JSON array.
    let arr: Vec<serde_json::Value> = match serde_json::from_str(json_str) {
        Ok(arr) => arr,
        Err(_) => {
            return ExtractedToolCallInfo {
                tools_called: false,
                tool_calls: Vec::new(),
                content: Some(text.to_string()),
            };
        }
    };

    let tool_calls: Vec<protocol::ToolCall> = arr
        .iter()
        .filter_map(|val| {
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
        })
        .collect();

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
            content,
        }
    }
}

// ---------------------------------------------------------------------------
// Jamba streaming state machine
// ---------------------------------------------------------------------------

struct JambaStreamingState {
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Previously parsed tool call array.
    prev_tool_call_arr: Vec<serde_json::Value>,
    /// Streamed argument characters for each tool call (for diffing).
    streamed_args_for_tool: Vec<String>,
    /// Buffer for partial tag tokens.
    buffer: String,
    /// Whether we've seen the open tag yet.
    seen_open_tag: bool,
}

impl JambaStreamingState {
    fn new() -> Self {
        Self {
            current_tool_id: -1,
            current_tool_name_sent: false,
            prev_tool_call_arr: Vec::new(),
            streamed_args_for_tool: Vec::new(),
            buffer: String::new(),
            seen_open_tag: false,
        }
    }
}

impl StreamingToolParserState for JambaStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        // Buffer delta for partial tag detection.
        self.buffer.push_str(delta_text);

        // Check for partial open/close tags.
        if is_partial_jamba_tag(&self.buffer) {
            return ToolParserDelta::None;
        }

        let text_to_process = std::mem::take(&mut self.buffer);

        // If we haven't seen the open tag yet, check for it.
        if !self.seen_open_tag {
            if current_text.contains(JAMBA_TOOL_CALLS_OPEN) {
                self.seen_open_tag = true;
                // Suppress the open tag token itself.
                if text_to_process.contains(JAMBA_TOOL_CALLS_OPEN) {
                    // There might be content before the tag in previous deltas
                    // (already streamed). Just suppress this delta.
                    return ToolParserDelta::None;
                }
            } else {
                // No tool calls yet — emit as content.
                return ToolParserDelta::Content(text_to_process);
            }
        }

        // We're inside the <tool_calls> region. Extract the parsable array.
        let parsable_arr = current_text
            .split(JAMBA_TOOL_CALLS_OPEN)
            .last()
            .unwrap_or("")
            .split(JAMBA_TOOL_CALLS_CLOSE)
            .next()
            .unwrap_or("");

        // Try partial JSON parse of the array.
        let parsed = partial_json_parse(parsable_arr.trim());

        let Some(val) = parsed else {
            return ToolParserDelta::None;
        };

        let Some(tool_call_arr) = val.as_array() else {
            return ToolParserDelta::None;
        };

        // Empty array — nothing to stream yet.
        if tool_call_arr.is_empty() {
            return ToolParserDelta::None;
        }

        // Check if a new tool call started (array grew past our cursor).
        if tool_call_arr.len() as i32 > self.current_tool_id + 1 {
            // Flush remaining args for the previous tool if any.
            let flush_delta = if self.current_tool_id >= 0 {
                let prev_idx = self.current_tool_id as usize;
                let prev_call = &tool_call_arr[prev_idx];
                let cur_args = prev_call
                    .get("arguments")
                    .map(|a| {
                        if a.is_string() {
                            a.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(a).unwrap_or_default()
                        }
                    })
                    .unwrap_or_default();
                let prev_streamed = &self.streamed_args_for_tool[prev_idx];
                if cur_args.len() > prev_streamed.len() {
                    let diff = cur_args[prev_streamed.len()..].to_string();
                    self.streamed_args_for_tool[prev_idx] = cur_args;
                    Some(DeltaToolCall {
                        index: prev_idx as u32,
                        id: None,
                        call_type: None,
                        function_name: None,
                        function_arguments: Some(diff),
                    })
                } else {
                    None
                }
            } else {
                None
            };

            // Advance to the new tool.
            self.current_tool_id = tool_call_arr.len() as i32 - 1;
            self.current_tool_name_sent = false;
            self.streamed_args_for_tool.push(String::new());
            self.prev_tool_call_arr = tool_call_arr.clone();

            if let Some(flush) = flush_delta {
                return ToolParserDelta::ToolCalls(vec![flush]);
            }
            // Fall through to try sending the name for the new tool.
        }

        let tool_idx = self.current_tool_id as usize;
        let current_tool_call = &tool_call_arr[tool_idx];

        // Try to send the name if not yet sent.
        if !self.current_tool_name_sent {
            if let Some(name) = current_tool_call.get("name").and_then(|n| n.as_str()) {
                self.current_tool_name_sent = true;
                self.prev_tool_call_arr = tool_call_arr.clone();
                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: Some(format!("call_{}", Uuid::new_v4().simple())),
                    call_type: Some("function".to_string()),
                    function_name: Some(name.to_string()),
                    function_arguments: Some(String::new()),
                }]);
            }
            return ToolParserDelta::None;
        }

        // Stream arguments diff.
        let current_args = current_tool_call
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
            self.prev_tool_call_arr = tool_call_arr.clone();

            return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                index: tool_idx as u32,
                id: None,
                call_type: None,
                function_name: None,
                function_arguments: Some(diff),
            }]);
        }

        self.prev_tool_call_arr = tool_call_arr.clone();
        ToolParserDelta::None
    }
}

/// Check if text ends with a partial `<tool_calls>` or `</tool_calls>` tag.
fn is_partial_jamba_tag(text: &str) -> bool {
    for tag in [JAMBA_TOOL_CALLS_OPEN, JAMBA_TOOL_CALLS_CLOSE] {
        for i in 1..tag.len() {
            if text.ends_with(&tag[..i]) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Granite tool parser
// ---------------------------------------------------------------------------

/// Granite 3.0 special token prefix.
const GRANITE_BOT_TOKEN: &str = "<|tool_call|>";
/// Granite 3.1 string prefix.
const GRANITE_BOT_STRING: &str = "<tool_call>";

/// Granite-style tool call parser.
///
/// Detects tool calls prefixed by `<|tool_call|>` (Granite 3.0) or
/// `<tool_call>` (Granite 3.1), followed by a JSON array of objects
/// with `name` and `arguments` fields.
///
/// Port of: `vllm/tool_parsers/granite_tool_parser.py`
#[derive(Default)]
pub struct GraniteToolParser;

impl GraniteToolParser {
    pub fn new() -> Self {
        Self
    }
}

impl ToolCallParser for GraniteToolParser {
    fn extract_tool_calls(&self, model_output: &str) -> ExtractedToolCallInfo {
        granite_extract(model_output)
    }

    fn create_streaming_state(&self) -> Box<dyn StreamingToolParserState + Send> {
        Box::new(GraniteStreamingState::new())
    }
}

/// Strip Granite prefix tokens and leading whitespace, returning the
/// remaining text. Returns `None` if the stripped text doesn't start with `[`.
fn granite_strip_prefix(text: &str) -> Option<&str> {
    let mut s = text.trim_start();
    if let Some(rest) = s.strip_prefix(GRANITE_BOT_TOKEN) {
        s = rest.trim_start();
    }
    if let Some(rest) = s.strip_prefix(GRANITE_BOT_STRING) {
        s = rest.trim_start();
    }
    if s.starts_with('[') { Some(s) } else { None }
}

/// Extract tool calls from Granite-formatted text.
fn granite_extract(text: &str) -> ExtractedToolCallInfo {
    let Some(stripped) = granite_strip_prefix(text) else {
        return ExtractedToolCallInfo {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(text.to_string()),
        };
    };

    // Parse as a JSON array.
    let arr: Vec<serde_json::Value> = match serde_json::from_str(stripped) {
        Ok(arr) => arr,
        Err(_) => {
            return ExtractedToolCallInfo {
                tools_called: false,
                tool_calls: Vec::new(),
                content: Some(text.to_string()),
            };
        }
    };

    let tool_calls: Vec<protocol::ToolCall> = arr
        .iter()
        .filter_map(|val| {
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
        })
        .collect();

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
            content: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Granite streaming state machine
// ---------------------------------------------------------------------------

struct GraniteStreamingState {
    /// Current tool call index (-1 = no tool call started).
    current_tool_id: i32,
    /// Whether the name for the current tool has been sent.
    current_tool_name_sent: bool,
    /// Previously parsed tool call array.
    prev_tool_call_arr: Vec<serde_json::Value>,
    /// Streamed argument characters for each tool call (for diffing).
    streamed_args_for_tool: Vec<String>,
    /// Whether we've found the `[` start of the JSON array.
    array_started: bool,
    /// Byte offset where the JSON array begins in current_text.
    array_start_offset: usize,
}

impl GraniteStreamingState {
    fn new() -> Self {
        Self {
            current_tool_id: -1,
            current_tool_name_sent: false,
            prev_tool_call_arr: Vec::new(),
            streamed_args_for_tool: Vec::new(),
            array_started: false,
            array_start_offset: 0,
        }
    }
}

impl StreamingToolParserState for GraniteStreamingState {
    fn process_delta(
        &mut self,
        _previous_text: &str,
        current_text: &str,
        delta_text: &str,
    ) -> ToolParserDelta {
        // Find the start of the JSON array if not yet found.
        if !self.array_started {
            // Skip prefix tokens and whitespace.
            let mut s = current_text.trim_start();
            if let Some(rest) = s.strip_prefix(GRANITE_BOT_TOKEN) {
                s = rest.trim_start();
            }
            if let Some(rest) = s.strip_prefix(GRANITE_BOT_STRING) {
                s = rest.trim_start();
            }
            let offset = current_text.len() - s.len();

            if s.starts_with('[') {
                self.array_started = true;
                self.array_start_offset = offset;
            } else if s.is_empty() {
                // Still buffering prefix/whitespace.
                return ToolParserDelta::None;
            } else {
                // Not a tool call — regular content.
                return ToolParserDelta::Content(delta_text.to_string());
            }
        }

        // Parse the JSON array portion.
        let array_text = &current_text[self.array_start_offset..];
        let parsed = partial_json_parse(array_text.trim());

        let Some(val) = parsed else {
            return ToolParserDelta::None;
        };

        let Some(tool_call_arr) = val.as_array() else {
            return ToolParserDelta::None;
        };

        if tool_call_arr.is_empty() {
            return ToolParserDelta::None;
        }

        // Check completeness of the last element: if the full array_text
        // parses as valid JSON, the last element is complete.
        let last_is_complete = serde_json::from_str::<serde_json::Value>(array_text.trim()).is_ok();

        // Check if a new tool call started (array grew past cursor).
        if tool_call_arr.len() as i32 > self.current_tool_id + 1 {
            // Flush remaining args for the previous tool.
            let flush_delta = if self.current_tool_id >= 0 {
                let prev_idx = self.current_tool_id as usize;
                let prev_call = &tool_call_arr[prev_idx];
                let cur_args = prev_call
                    .get("arguments")
                    .map(|a| {
                        if a.is_string() {
                            a.as_str().unwrap().to_string()
                        } else {
                            serde_json::to_string(a).unwrap_or_default()
                        }
                    })
                    .unwrap_or_default();
                let prev_streamed = &self.streamed_args_for_tool[prev_idx];
                if cur_args.len() > prev_streamed.len() {
                    let diff = cur_args[prev_streamed.len()..].to_string();
                    self.streamed_args_for_tool[prev_idx] = cur_args;
                    Some(DeltaToolCall {
                        index: prev_idx as u32,
                        id: None,
                        call_type: None,
                        function_name: None,
                        function_arguments: Some(diff),
                    })
                } else {
                    None
                }
            } else {
                None
            };

            self.current_tool_id = tool_call_arr.len() as i32 - 1;
            self.current_tool_name_sent = false;
            self.streamed_args_for_tool.push(String::new());
            self.prev_tool_call_arr = tool_call_arr.clone();

            if let Some(flush) = flush_delta {
                return ToolParserDelta::ToolCalls(vec![flush]);
            }
            // Fall through to try sending the name.
        }

        let tool_idx = self.current_tool_id as usize;
        let current_tool_call = &tool_call_arr[tool_idx];

        // Try to send the name if not yet sent.
        if !self.current_tool_name_sent {
            if let Some(name) = current_tool_call.get("name").and_then(|n| n.as_str()) {
                self.current_tool_name_sent = true;
                self.prev_tool_call_arr = tool_call_arr.clone();
                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: Some(format!("call_{}", Uuid::new_v4().simple())),
                    call_type: Some("function".to_string()),
                    function_name: Some(name.to_string()),
                    function_arguments: Some(String::new()),
                }]);
            }
            return ToolParserDelta::None;
        }

        // Stream arguments diff.
        let cur_arguments = current_tool_call.get("arguments");
        if let Some(cur_args_val) = cur_arguments {
            let cur_args_json = if cur_args_val.is_string() {
                cur_args_val.as_str().unwrap().to_string()
            } else {
                serde_json::to_string(cur_args_val).unwrap_or_default()
            };

            let sent = self.streamed_args_for_tool[tool_idx].len();

            // When the tool call JSON is complete, we can send the rest.
            // When incomplete, use common-prefix diffing to avoid streaming
            // close-brackets prematurely.
            let argument_diff = if last_is_complete || tool_idx < tool_call_arr.len() - 1 {
                // Complete or not the last element — safe to send remainder.
                if cur_args_json.len() > sent {
                    Some(cur_args_json[sent..].to_string())
                } else {
                    None
                }
            } else {
                // Last element, incomplete — use common prefix with prev.
                let prev_args_val = self
                    .prev_tool_call_arr
                    .get(tool_idx)
                    .and_then(|v| v.get("arguments"));
                if let Some(prev_val) = prev_args_val {
                    let prev_args_json = if prev_val.is_string() {
                        prev_val.as_str().unwrap().to_string()
                    } else {
                        serde_json::to_string(prev_val).unwrap_or_default()
                    };
                    if cur_args_json != prev_args_json {
                        let prefix_len = find_common_prefix_len(&prev_args_json, &cur_args_json);
                        if prefix_len > sent {
                            Some(cur_args_json[sent..prefix_len].to_string())
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else if cur_args_json.len() > sent {
                    // No previous args — send what we have.
                    Some(cur_args_json[sent..].to_string())
                } else {
                    None
                }
            };

            if let Some(diff) = argument_diff {
                self.streamed_args_for_tool[tool_idx].push_str(&diff);
                self.prev_tool_call_arr = tool_call_arr.clone();
                return ToolParserDelta::ToolCalls(vec![DeltaToolCall {
                    index: tool_idx as u32,
                    id: None,
                    call_type: None,
                    function_name: None,
                    function_arguments: Some(diff),
                }]);
            }
        }

        self.prev_tool_call_arr = tool_call_arr.clone();
        ToolParserDelta::None
    }
}

/// Find the length of the common prefix between two strings.
fn find_common_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

// ---------------------------------------------------------------------------
// Parser registry
// ---------------------------------------------------------------------------

/// Get a tool call parser by name.
pub fn get_tool_parser(name: &str) -> Result<Arc<dyn ToolCallParser>, String> {
    match name {
        "hermes" => Ok(Arc::new(HermesToolParser::new())),
        "llama3_json" | "llama4_json" => Ok(Arc::new(LlamaJsonToolParser::new())),
        "kimi_k2" => Ok(Arc::new(KimiK2ToolParser::new())),
        "mistral" => Ok(Arc::new(MistralToolParser::new())),
        "jamba" => Ok(Arc::new(JambaToolParser::new())),
        "granite" => Ok(Arc::new(GraniteToolParser::new())),
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
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        got_name = true;
                        assert_eq!(name, "get_weather");
                        assert_eq!(call.index, 0);
                        assert!(call.id.is_some());
                    }
                    if let Some(args) = &call.function_arguments
                        && !args.is_empty()
                    {
                        got_args = true;
                    }
                }
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
        if let ToolParserDelta::Content(text) = state.process_delta(&prev, &accumulated, "Sure, ") {
            assert_eq!(text, "Sure, ");
            got_content = true;
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
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                got_tool = true;
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        assert_eq!(name, "f");
                    }
                }
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
    fn test_registry_kimi_k2() {
        assert!(get_tool_parser("kimi_k2").is_ok());
    }

    #[test]
    fn test_registry_unknown() {
        assert!(get_tool_parser("unknown").is_err());
    }

    // -- Kimi K2 non-streaming tests --

    #[test]
    fn test_kimi_k2_single_tool_call() {
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_calls_section_begin|>\n<|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|> {\"city\": \"SF\"} <|tool_call_end|>\n<|tool_calls_section_end|>";
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(
            result.tool_calls[0].function.arguments,
            "{\"city\": \"SF\"}"
        );
        assert_eq!(result.tool_calls[0].call_type, "function");
        assert!(result.tool_calls[0].id.starts_with("call_"));
        assert!(result.content.is_none());
    }

    #[test]
    fn test_kimi_k2_two_tool_calls() {
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_calls_section_begin|>\n<|tool_call_begin|> functions.search:0 <|tool_call_argument_begin|> {\"q\": \"rust\"} <|tool_call_end|>\n<|tool_call_begin|> functions.fetch:1 <|tool_call_argument_begin|> {\"url\": \"https://example.com\"} <|tool_call_end|>\n<|tool_calls_section_end|>";
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "search");
        assert_eq!(result.tool_calls[1].function.name, "fetch");
    }

    #[test]
    fn test_kimi_k2_text_before_section() {
        let parser = KimiK2ToolParser::new();
        let output = "I'll check the weather for you.\n<|tool_calls_section_begin|>\n<|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|> {\"city\": \"SF\"} <|tool_call_end|>\n<|tool_calls_section_end|>";
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content.unwrap(), "I'll check the weather for you.");
    }

    #[test]
    fn test_kimi_k2_no_tool_call() {
        let parser = KimiK2ToolParser::new();
        let output = "The weather in SF is sunny and 72°F.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), output);
    }

    #[test]
    fn test_kimi_k2_singular_section_markers() {
        // Support both singular and plural section markers.
        let parser = KimiK2ToolParser::new();
        let output = "<|tool_call_section_begin|>\n<|tool_call_begin|> functions.get_weather:0 <|tool_call_argument_begin|> {\"city\": \"SF\"} <|tool_call_end|>\n<|tool_call_section_end|>";
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_kimi_k2_function_name_parsing() {
        assert_eq!(
            kimi_k2_parse_function_name("functions.get_weather:0"),
            "get_weather"
        );
        assert_eq!(kimi_k2_parse_function_name("functions.search:1"), "search");
        assert_eq!(kimi_k2_parse_function_name("get_weather:0"), "get_weather");
        assert_eq!(kimi_k2_parse_function_name("get_weather"), "get_weather");
        assert_eq!(kimi_k2_parse_function_name("a.b.c:2"), "c");
    }

    // -- Kimi K2 streaming tests --

    #[test]
    fn test_kimi_k2_streaming_basic() {
        let parser = KimiK2ToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec![
            "<|tool_calls_section_begin|>",
            "\n",
            "<|tool_call_begin|>",
            " functions.get_weather:0 ",
            "<|tool_call_argument_begin|>",
            " {\"city\":",
            " \"SF\"}",
            " <|tool_call_end|>",
            "\n",
            "<|tool_calls_section_end|>",
        ];

        let mut accumulated = String::new();
        let mut got_name = false;
        let mut got_args = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        got_name = true;
                        assert_eq!(name, "get_weather");
                        assert_eq!(call.index, 0);
                        assert!(call.id.is_some());
                    }
                    if let Some(args) = &call.function_arguments
                        && !args.is_empty()
                    {
                        got_args = true;
                    }
                }
            }
        }

        assert!(got_name, "Should have received tool name");
        assert!(got_args, "Should have received argument fragments");
    }

    #[test]
    fn test_kimi_k2_streaming_content_then_tool() {
        let parser = KimiK2ToolParser::new();
        let mut state = parser.create_streaming_state();

        let mut accumulated = String::new();
        let mut got_content = false;
        let mut got_tool = false;

        // Content before tools.
        let prev = accumulated.clone();
        accumulated.push_str("Let me check. ");
        if let ToolParserDelta::Content(text) =
            state.process_delta(&prev, &accumulated, "Let me check. ")
        {
            assert_eq!(text, "Let me check. ");
            got_content = true;
        }

        // Tool section.
        let tokens = vec![
            "<|tool_calls_section_begin|>",
            "\n<|tool_call_begin|>",
            " functions.f:0 ",
            "<|tool_call_argument_begin|>",
            " {}",
            " <|tool_call_end|>",
            "\n<|tool_calls_section_end|>",
        ];
        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                got_tool = true;
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        assert_eq!(name, "f");
                    }
                }
            }
        }

        assert!(got_content, "Should have received content");
        assert!(got_tool, "Should have received tool call");
    }

    // -- Mistral non-streaming tests --

    #[test]
    fn test_registry_mistral() {
        assert!(get_tool_parser("mistral").is_ok());
    }

    #[test]
    fn test_registry_jamba() {
        assert!(get_tool_parser("jamba").is_ok());
    }

    #[test]
    fn test_mistral_id_format() {
        let id = mistral_generate_id();
        assert_eq!(id.len(), 9);
        assert!(id.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn test_mistral_no_tools() {
        let parser = MistralToolParser::new();
        let output = "The weather is sunny today.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), output);
    }

    #[test]
    fn test_mistral_single_tool_v11() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]get_weather{"city":"SF"}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"city":"SF"}"#);
        assert_eq!(result.tool_calls[0].call_type, "function");
        assert_eq!(result.tool_calls[0].id.len(), 9);
        assert!(result.content.is_none());
    }

    #[test]
    fn test_mistral_single_tool_pre_v11() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"SF"}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"city":"SF"}"#);
    }

    #[test]
    fn test_mistral_multiple_tools_v11() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]get_weather{"city":"SF"}[TOOL_CALLS]search{"q":"rust"}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[1].function.name, "search");
        assert_eq!(result.tool_calls[1].function.arguments, r#"{"q":"rust"}"#);
    }

    #[test]
    fn test_mistral_multiple_tools_pre_v11() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS] [{"name":"get_weather","arguments":{"city":"SF"}},{"name":"search","arguments":{"q":"rust"}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[1].function.name, "search");
    }

    #[test]
    fn test_mistral_content_before_tools() {
        let parser = MistralToolParser::new();
        let output = r#"Let me help you.[TOOL_CALLS]get_weather{"city":"SF"}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content.unwrap(), "Let me help you.");
    }

    #[test]
    fn test_mistral_complex_arguments() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS]create_event{"title":"Meeting","nested":{"key":"val\"ue"},"list":[1,2,3]}"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls[0].function.name, "create_event");
        assert!(result.tool_calls[0].function.arguments.contains("nested"));
    }

    #[test]
    fn test_mistral_pre_v11_malformed_json_fallback() {
        let parser = MistralToolParser::new();
        // Malformed JSON with extra text — the `[{...}]` pattern should be found by fallback
        let output = r#"[TOOL_CALLS] [{"name":"f","arguments":{"a":1}}] extra text"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "f");
    }

    #[test]
    fn test_mistral_pre_v11_arguments_before_name() {
        let parser = MistralToolParser::new();
        let output = r#"[TOOL_CALLS] [{"arguments":{"city":"SF"},"name":"get_weather"}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"city":"SF"}"#);
    }

    // -- Mistral streaming tests --

    #[test]
    fn test_mistral_streaming_no_tools() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec!["Hello", " world", "!"];
        let mut accumulated = String::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            match state.process_delta(&prev, &accumulated, token) {
                ToolParserDelta::Content(c) => assert_eq!(c, token),
                other => panic!("Expected Content, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_mistral_streaming_single_tool_v11() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec!["[TOOL_CALLS]", "get_weather", r#"{"city":"#, r#""SF"}"#];
        let mut accumulated = String::new();
        let mut got_name = false;
        let mut args = String::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        got_name = true;
                        assert_eq!(name, "get_weather");
                        assert_eq!(call.index, 0);
                        assert!(call.id.is_some());
                    }
                    if let Some(a) = &call.function_arguments {
                        args.push_str(a);
                    }
                }
            }
        }

        assert!(got_name, "Should have received function name");
        assert_eq!(args, r#"{"city":"SF"}"#);
    }

    #[test]
    fn test_mistral_streaming_multiple_tools_v11() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec![
            "[TOOL_CALLS]",
            "get_weather",
            r#"{"city":"SF"}"#,
            "[TOOL_CALLS]",
            "search",
            r#"{"q":"rust"}"#,
        ];
        let mut accumulated = String::new();
        let mut names = Vec::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        names.push(name.clone());
                    }
                }
            }
        }

        assert_eq!(names, vec!["get_weather", "search"]);
    }

    #[test]
    fn test_mistral_streaming_content_then_tool() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec!["Sure!", "[TOOL_CALLS]", "f", r#"{"a":1}"#];
        let mut accumulated = String::new();
        let mut got_content = false;
        let mut got_tool = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            match state.process_delta(&prev, &accumulated, token) {
                ToolParserDelta::Content(c) => {
                    got_content = true;
                    assert_eq!(c, "Sure!");
                }
                ToolParserDelta::ToolCalls(calls) => {
                    for call in &calls {
                        if call.function_name.is_some() {
                            got_tool = true;
                        }
                    }
                }
                _ => {}
            }
        }

        assert!(got_content);
        assert!(got_tool);
    }

    #[test]
    fn test_mistral_streaming_single_tool_pre_v11() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec![
            "[TOOL_CALLS]",
            r#" [{"name"#,
            r#"": "get_weather", "arguments": {"#,
            r#""city": "SF""#,
            "}",
            "}]",
        ];
        let mut accumulated = String::new();
        let mut got_name = false;
        let mut got_args = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        got_name = true;
                        assert_eq!(name, "get_weather");
                    }
                    if call.function_arguments.is_some() {
                        got_args = true;
                    }
                }
            }
        }

        assert!(got_name, "Should have received function name");
        assert!(got_args, "Should have received arguments");
    }

    #[test]
    fn test_mistral_streaming_multiple_tools_pre_v11() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = vec![
            r#"[TOOL_CALLS] [{"name": "f1", "arguments": {"a": 1}}, {"name": "f2", "arguments": {"b": 2}}]"#,
        ];
        let mut accumulated = String::new();
        let mut names = Vec::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        names.push(name.clone());
                    }
                }
            }
        }

        assert_eq!(names, vec!["f1", "f2"]);
    }

    #[test]
    fn test_mistral_streaming_one_chunk_v11() {
        let parser = MistralToolParser::new();
        let mut state = parser.create_streaming_state();

        let full = r#"[TOOL_CALLS]get_weather{"city":"SF"}"#;
        let mut got_name = false;

        match state.process_delta("", full, full) {
            ToolParserDelta::ToolCalls(calls) => {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        got_name = true;
                        assert_eq!(name, "get_weather");
                    }
                }
            }
            other => panic!("Expected ToolCalls, got {:?}", other),
        }

        assert!(got_name);
    }

    // -- Jamba non-streaming tests --

    #[test]
    fn test_jamba_single_tool_call() {
        let parser = JambaToolParser::new();
        let output =
            r#"<tool_calls>[{"name":"get_weather","arguments":{"city":"SF"}}]</tool_calls>"#;
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
    fn test_jamba_two_tool_calls() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>[{"name":"search","arguments":{"q":"rust"}},{"name":"fetch","arguments":{"url":"https://example.com"}}]</tool_calls>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "search");
        assert_eq!(result.tool_calls[1].function.name, "fetch");
    }

    #[test]
    fn test_jamba_text_before_tool_calls() {
        let parser = JambaToolParser::new();
        let output = r#"I'll help you with that.
<tool_calls>[{"name":"get_weather","arguments":{"city":"SF"}}]</tool_calls>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.content.unwrap(), "I'll help you with that.");
    }

    #[test]
    fn test_jamba_no_tool_call() {
        let parser = JambaToolParser::new();
        let output = "The weather in SF is sunny and 72°F.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), output);
    }

    #[test]
    fn test_jamba_malformed_json() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>not json at all</tool_calls>"#;
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert!(result.content.is_some());
    }

    #[test]
    fn test_jamba_unclosed_tag() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>[{"name":"f","arguments":{"a":1}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "f");
    }

    #[test]
    fn test_jamba_string_arguments() {
        let parser = JambaToolParser::new();
        let output = r#"<tool_calls>[{"name":"f","arguments":"{\"a\":1}"}]</tool_calls>"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.arguments, r#"{"a":1}"#);
    }

    // -- Jamba streaming tests --

    #[test]
    fn test_jamba_streaming_content_then_tool() {
        let parser = JambaToolParser::new();
        let mut state = parser.create_streaming_state();

        // First token: content.
        let d1 = state.process_delta("", "Hello", "Hello");
        assert!(matches!(d1, ToolParserDelta::Content(ref s) if s == "Hello"));

        // Open tag token.
        let d2 = state.process_delta("Hello", "Hello<tool_calls>", "<tool_calls>");
        assert!(matches!(d2, ToolParserDelta::None));

        // Start of JSON array with tool name.
        let d3 = state.process_delta(
            "Hello<tool_calls>",
            r#"Hello<tool_calls>[{"name":"get_weather""#,
            r#"[{"name":"get_weather""#,
        );
        // Should get the name.
        match d3 {
            ToolParserDelta::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function_name.as_deref(), Some("get_weather"));
                assert!(calls[0].id.is_some());
            }
            other => panic!("Expected ToolCalls with name, got {:?}", other),
        }

        // Stream arguments.
        let d4 = state.process_delta(
            r#"Hello<tool_calls>[{"name":"get_weather""#,
            r#"Hello<tool_calls>[{"name":"get_weather","arguments":{"city":"SF"#,
            r#","arguments":{"city":"SF"#,
        );
        match d4 {
            ToolParserDelta::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert!(calls[0].function_arguments.is_some());
            }
            other => panic!("Expected ToolCalls with args, got {:?}", other),
        }
    }

    #[test]
    fn test_jamba_streaming_two_tools() {
        let parser = JambaToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = [
            "<tool_calls>",
            r#"[{"name":"f1""#,
            r#","arguments":{"a":1}}"#,
            r#",{"name":"f2""#,
            r#","arguments":{"b":2}}"#,
            "]</tool_calls>",
        ];

        let mut accumulated = String::new();
        let mut names = Vec::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        names.push(name.clone());
                    }
                }
            }
        }

        assert_eq!(names, vec!["f1", "f2"]);
    }

    // -- Granite non-streaming tests --

    #[test]
    fn test_granite_single_tool_call_30() {
        // Granite 3.0 format: <|tool_call|> prefix.
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|>[{"name":"get_weather","arguments":{"city":"SF"}}]"#;
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
    fn test_granite_single_tool_call_31() {
        // Granite 3.1 format: <tool_call> prefix.
        let parser = GraniteToolParser::new();
        let output = r#"<tool_call>[{"name":"get_weather","arguments":{"city":"SF"}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_granite_two_tool_calls() {
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|>[{"name":"search","arguments":{"q":"rust"}},{"name":"fetch","arguments":{"url":"https://example.com"}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].function.name, "search");
        assert_eq!(result.tool_calls[1].function.name, "fetch");
    }

    #[test]
    fn test_granite_no_tool_call() {
        let parser = GraniteToolParser::new();
        let output = "The weather in SF is sunny and 72°F.";
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.tool_calls.is_empty());
        assert_eq!(result.content.unwrap(), output);
    }

    #[test]
    fn test_granite_malformed_json() {
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|>not json"#;
        let result = parser.extract_tool_calls(output);
        assert!(!result.tools_called);
        assert!(result.content.is_some());
    }

    #[test]
    fn test_granite_whitespace_between_prefix_and_array() {
        let parser = GraniteToolParser::new();
        let output = "  <|tool_call|>  [  {\"name\":\"f\",\"arguments\":{\"a\":1}}  ]";
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "f");
    }

    #[test]
    fn test_granite_both_prefixes() {
        // Both prefixes present (unlikely but handled).
        let parser = GraniteToolParser::new();
        let output = r#"<|tool_call|><tool_call>[{"name":"f","arguments":{}}]"#;
        let result = parser.extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
    }

    #[test]
    fn test_registry_granite() {
        assert!(get_tool_parser("granite").is_ok());
    }

    // -- Granite streaming tests --

    #[test]
    fn test_granite_streaming_basic() {
        let parser = GraniteToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = [
            "<|tool_call|>",
            r#"[{"name":"get_weather""#,
            r#","arguments":{"city":"SF"}}]"#,
        ];

        let mut accumulated = String::new();
        let mut got_name = false;
        let mut got_args = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        assert_eq!(name, "get_weather");
                        got_name = true;
                    }
                    if let Some(args) = &call.function_arguments
                        && !args.is_empty()
                    {
                        got_args = true;
                    }
                }
            }
        }

        assert!(got_name);
        assert!(got_args);
    }

    #[test]
    fn test_granite_streaming_no_tool() {
        let parser = GraniteToolParser::new();
        let mut state = parser.create_streaming_state();

        let d = state.process_delta("", "Hello world", "Hello world");
        assert!(matches!(d, ToolParserDelta::Content(ref s) if s == "Hello world"));
    }

    #[test]
    fn test_granite_streaming_two_tools() {
        let parser = GraniteToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = [
            "<|tool_call|>",
            r#"[{"name":"f1""#,
            r#","arguments":{"a":1}}"#,
            r#",{"name":"f2""#,
            r#","arguments":{"b":2}}"#,
            "]",
        ];

        let mut accumulated = String::new();
        let mut names = Vec::new();

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if let Some(name) = &call.function_name {
                        names.push(name.clone());
                    }
                }
            }
        }

        assert_eq!(names, vec!["f1", "f2"]);
    }

    #[test]
    fn test_granite_streaming_31_prefix() {
        // Granite 3.1 uses <tool_call> instead of <|tool_call|>.
        let parser = GraniteToolParser::new();
        let mut state = parser.create_streaming_state();

        let tokens = [
            "<tool_call>",
            r#"[{"name":"f""#,
            r#","arguments":{"x":1}}]"#,
        ];

        let mut accumulated = String::new();
        let mut got_name = false;

        for token in tokens {
            let prev = accumulated.clone();
            accumulated.push_str(token);
            if let ToolParserDelta::ToolCalls(calls) =
                state.process_delta(&prev, &accumulated, token)
            {
                for call in &calls {
                    if call.function_name.as_deref() == Some("f") {
                        got_name = true;
                    }
                }
            }
        }

        assert!(got_name);
    }
}
