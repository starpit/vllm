// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Chat template application using HuggingFace Jinja2 templates.
//!
//! HuggingFace models store a Jinja2 chat template in `tokenizer_config.json`.
//! This module parses that template and applies it to a list of chat messages,
//! producing the formatted prompt string that the model expects.
//!
//! Uses `minijinja` (a Rust Jinja2 engine) for template rendering.
//! The `Environment` and compiled template are built once at construction time
//! and reused for every request, avoiding per-request template compilation.

use std::path::Path;

use minijinja::Environment;
use serde::{Deserialize, Serialize};

use crate::error::ServeError;

// ---------------------------------------------------------------------------
// ChatTemplate
// ---------------------------------------------------------------------------

/// A parsed chat template that can format messages for a specific model.
///
/// The minijinja `Environment` (with the compiled template) is built once
/// at construction time and reused on every `apply()` call.
pub struct ChatTemplate {
    /// Pre-compiled minijinja environment with the "chat" template loaded.
    env: Environment<'static>,
    /// Optional BOS token string (e.g. "<s>", "<|begin_of_text|>").
    bos_token: Option<String>,
    /// Optional EOS token string (e.g. "</s>", "<|end_of_text|>").
    eos_token: Option<String>,
    /// The raw Jinja2 template string (kept for `template_str()` accessor).
    template_str: String,
}

/// A single chat message for template rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMessage {
    pub role: String,
    pub content: String,
}

/// Partial representation of `tokenizer_config.json` for extracting the
/// chat template and special tokens.
#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    chat_template: Option<serde_json::Value>,
    bos_token: Option<serde_json::Value>,
    eos_token: Option<serde_json::Value>,
}

/// Build a minijinja Environment with the given template string compiled as "chat".
fn build_env(template_str: &str) -> Result<Environment<'static>, ServeError> {
    let mut env = Environment::new();

    // Enable Python string/dict/list methods (startswith, endswith, etc.)
    // that HuggingFace Jinja2 chat templates commonly use.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // Add a `raise_exception` function that Jinja2 templates often use.
    env.add_function("raise_exception", raise_exception);

    // Add `strftime_now` — used by HuggingFace transformers chat templates
    // (e.g. granite, llama4) to inject the current date/time.
    env.add_function("strftime_now", strftime_now);

    // Identity filter for HF's `{% generation %}` ... `{% endgeneration %}`
    // markers. The Python implementation
    // (`transformers/utils/chat_template_utils.py:399 AssistantTracker`)
    // is registered as a `jinja2.ext.Extension` with custom `tags =
    // {"generation"}` — minijinja has no equivalent statement-extension
    // API. The standard workaround is to rewrite each pair into a
    // `{% filter generation %}...{% endfilter %}` block (see
    // `strip_generation_markers`); this filter is the registered hook
    // the rewritten template lands on. Today it's identity (return the
    // body verbatim, matching HF's behavior — the extension's `parse`
    // doesn't transform render output, only records span indices for
    // `return_assistant_tokens_mask`); future code can grow it into a
    // proper assistant-token tracker by accumulating offsets here.
    env.add_filter("generation", |s: String| s);

    env.add_template_owned("chat", template_str.to_owned())
        .map_err(|e| ServeError::Internal(format!("invalid chat template: {e}")))?;

    Ok(env)
}

/// Rewrite HuggingFace `{% generation %}` / `{% endgeneration %}`
/// pairs into `{% filter generation %}` / `{% endfilter %}` so the
/// `generation` filter registered in [`build_env`] is the runtime
/// hook for the body content.
///
/// **Why a rewrite is needed.** `{% generation %}` is a Jinja2 tag
/// that HF's `AssistantTracker` extension
/// (`transformers/utils/chat_template_utils.py:399`) registers via
/// `tags = {"generation"}` for `return_assistant_tokens_mask`. The
/// extension's `parse` doesn't transform render output; it only
/// records span indices. minijinja has no equivalent
/// statement-extension API — its parser has a closed keyword set and
/// errors with `unknown statement generation` before any user
/// callback can run (`pycompat::unknown_method_callback` only fires
/// for unknown METHOD calls on values at evaluation time, not for
/// unknown tags at parse time).
///
/// **Why filter blocks.** Jinja's built-in `{% filter NAME %}...{%
/// endfilter %}` *is* a statement minijinja parses, and `add_filter`
/// is a real extension surface. The rewrite preserves render output
/// exactly — the filter is identity today (matching HF's
/// no-render-side-effect behavior) — and exposes a hook that future
/// code could grow into a proper assistant-token tracker by
/// accumulating output offsets in the closure. This puts the work
/// on a supported API rather than parse-suppression.
///
/// LLaVA-1.5's `chat_template.json` (and Llama-3.2's, and a growing
/// number of HF chat templates) use these markers; without the
/// rewrite, the loader falls back to "plain concatenation",
/// multimodal content lands in the prompt as JSON-ish text, the
/// prompt blows past `max_model_len`, and the scheduler hangs.
fn strip_generation_markers(template_str: &str) -> String {
    let mut out = String::with_capacity(template_str.len());
    let bytes = template_str.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'{' && i + 1 < bytes.len() && bytes[i + 1] == b'%' {
            // Find the matching `%}`.
            let block_start = i;
            let mut j = i + 2;
            while j + 1 < bytes.len() && !(bytes[j] == b'%' && bytes[j + 1] == b'}') {
                j += 1;
            }
            if j + 1 < bytes.len() {
                let inner = &template_str[i + 2..j];
                let trimmed = inner.trim_matches(|c: char| c == '-' || c.is_whitespace());
                if trimmed == "generation" {
                    out.push_str("{% filter generation %}");
                    i = j + 2;
                    continue;
                }
                if trimmed == "endgeneration" {
                    out.push_str("{% endfilter %}");
                    i = j + 2;
                    continue;
                }
                // Not our tag — copy through verbatim.
                out.push_str(&template_str[block_start..j + 2]);
                i = j + 2;
                continue;
            }
        }
        let ch = template_str[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

impl ChatTemplate {
    /// Create a `ChatTemplate` from a raw Jinja2 template string.
    pub fn new(template_str: String) -> Result<Self, ServeError> {
        let template_str = strip_generation_markers(&template_str);
        let env = build_env(&template_str)?;
        Ok(Self {
            env,
            bos_token: None,
            eos_token: None,
            template_str,
        })
    }

    /// Set the BOS token string used by the template.
    pub fn with_bos_token(mut self, bos: String) -> Self {
        self.bos_token = Some(bos);
        self
    }

    /// Set the EOS token string used by the template.
    pub fn with_eos_token(mut self, eos: String) -> Self {
        self.eos_token = Some(eos);
        self
    }

    /// Load a `ChatTemplate` from a `tokenizer_config.json` file.
    ///
    /// Returns `None` if the file doesn't exist or doesn't contain a
    /// `chat_template` field.
    pub fn from_tokenizer_config(path: &Path) -> Result<Option<Self>, ServeError> {
        if !path.exists() {
            return Ok(None);
        }

        let data = std::fs::read_to_string(path)
            .map_err(|e| ServeError::Internal(format!("failed to read {}: {e}", path.display())))?;

        let config: TokenizerConfig = serde_json::from_str(&data).map_err(|e| {
            ServeError::Internal(format!("failed to parse {}: {e}", path.display()))
        })?;

        let template_str = match config.chat_template {
            Some(serde_json::Value::String(s)) => s,
            Some(serde_json::Value::Array(arr)) => {
                // Some models have an array of templates; take the first
                // (or the one named "default").
                arr.iter()
                    .find_map(|v| {
                        let obj = v.as_object()?;
                        let name = obj.get("name")?.as_str()?;
                        if name == "default" {
                            obj.get("template")?.as_str().map(|s| s.to_string())
                        } else {
                            None
                        }
                    })
                    .or_else(|| {
                        // Fall back to the first template.
                        arr.first()
                            .and_then(|v| {
                                v.as_object()
                                    .and_then(|obj| obj.get("template"))
                                    .and_then(|t| t.as_str())
                                    .map(|s| s.to_string())
                            })
                            .or_else(|| arr.first().and_then(|v| v.as_str()).map(|s| s.to_string()))
                    })
                    .unwrap_or_default()
            }
            _ => return Ok(None),
        };

        if template_str.is_empty() {
            return Ok(None);
        }

        let bos_token = extract_token_string(config.bos_token);
        let eos_token = extract_token_string(config.eos_token);

        let mut tpl = ChatTemplate::new(template_str)?;
        if let Some(bos) = bos_token {
            tpl = tpl.with_bos_token(bos);
        }
        if let Some(eos) = eos_token {
            tpl = tpl.with_eos_token(eos);
        }

        Ok(Some(tpl))
    }

    /// Apply the chat template to rich JSON messages, with optional tool definitions
    /// and extra template kwargs (e.g. `enable_thinking`).
    ///
    /// Messages are `serde_json::Value` objects so templates can access any field
    /// (`tool_calls`, `tool_call_id`, `name`, etc.) without needing Rust struct changes.
    ///
    /// Returns the formatted prompt string ready for tokenization.
    pub fn apply(
        &self,
        messages: &[serde_json::Value],
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
    ) -> Result<String, ServeError> {
        self.apply_with_kwargs(messages, add_generation_prompt, tools, None)
    }

    /// Like [`apply`] but with additional template keyword arguments.
    ///
    /// `extra_kwargs` are merged into the Jinja context so the template can
    /// access them (e.g. `enable_thinking`, `reasoning_effort`).
    pub fn apply_with_kwargs(
        &self,
        messages: &[serde_json::Value],
        add_generation_prompt: bool,
        tools: Option<&serde_json::Value>,
        extra_kwargs: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> Result<String, ServeError> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| ServeError::Internal(format!("failed to get template: {e}")))?;

        // Today's date string — used by some templates (e.g. LLaMA 3.1).
        let date_string = {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            // Simple UTC date: days since epoch.
            let days = now / 86400;
            // 1970-01-01 is a Thursday (day 4).
            let (year, month, day) = days_to_ymd(days);
            format!(
                "{:02} {month_name} {year}",
                day,
                month_name = MONTH_NAMES[month as usize - 1],
                year = year
            )
        };

        // Build the base context.
        let mut ctx_map = std::collections::BTreeMap::<String, minijinja::Value>::new();
        ctx_map.insert(
            "messages".into(),
            minijinja::Value::from_serialize(messages),
        );
        ctx_map.insert(
            "add_generation_prompt".into(),
            minijinja::Value::from(add_generation_prompt),
        );
        ctx_map.insert(
            "bos_token".into(),
            minijinja::Value::from(self.bos_token.as_deref().unwrap_or("")),
        );
        ctx_map.insert(
            "eos_token".into(),
            minijinja::Value::from(self.eos_token.as_deref().unwrap_or("")),
        );
        if let Some(tools) = tools {
            ctx_map.insert("tools".into(), minijinja::Value::from_serialize(tools));
        }
        ctx_map.insert("date_string".into(), minijinja::Value::from(date_string));

        // Merge extra kwargs (e.g. enable_thinking, reasoning_effort).
        if let Some(kwargs) = extra_kwargs {
            for (key, value) in kwargs {
                ctx_map.insert(key.clone(), minijinja::Value::from_serialize(value));
            }
        }

        let ctx = minijinja::Value::from_serialize(&ctx_map);

        let rendered = tmpl
            .render(ctx)
            .map_err(|e| ServeError::Internal(format!("chat template render failed: {e}")))?;

        Ok(rendered)
    }

    /// Convenience wrapper: apply with simple `TemplateMessage` slices and no tools.
    ///
    /// Used by tests and callers that don't need tool support.
    pub fn apply_simple(
        &self,
        messages: &[TemplateMessage],
        add_generation_prompt: bool,
    ) -> Result<String, ServeError> {
        let values: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::to_value(m).unwrap())
            .collect();
        self.apply(&values, add_generation_prompt, None)
    }

    /// Get the raw template string.
    pub fn template_str(&self) -> &str {
        &self.template_str
    }
}

/// Extract a token string from the `bos_token` / `eos_token` field in
/// tokenizer_config.json. These can be either a plain string or an object
/// with a `content` field.
fn extract_token_string(value: Option<serde_json::Value>) -> Option<String> {
    match value {
        Some(serde_json::Value::String(s)) => Some(s),
        Some(serde_json::Value::Object(mut obj)) => obj.remove("content").and_then(|v| match v {
            serde_json::Value::String(s) => Some(s),
            _ => None,
        }),
        _ => None,
    }
}

const MONTH_NAMES: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    // Civil calendar algorithm (simplified Euclidean).
    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let month_days: [u64; 12] = if is_leap(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 0u64;
    for (i, &md) in month_days.iter().enumerate() {
        if days < md {
            month = i as u64 + 1;
            break;
        }
        days -= md;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400))
}

/// A raise_exception function for Jinja2 compatibility.
/// Many HF chat templates use `{% raise_exception(...) %}` for error handling.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

/// `strftime_now(format)` — returns the current local time formatted with the
/// given strftime format string. Used by HuggingFace transformers chat templates
/// (granite, llama4, etc.) to inject the current date.
fn strftime_now(format: String) -> String {
    chrono::Local::now().format(&format).to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_generation_markers_passthrough() {
        // No `{% generation %}` tags → identity transform.
        let src = "{% for m in messages %}{{ m.content }}{% endfor %}";
        assert_eq!(strip_generation_markers(src), src);
    }

    #[test]
    fn test_strip_generation_markers_basic() {
        // HF AssistantTracker pair → minijinja `{% filter %}` block.
        // Body content between markers renders verbatim through the
        // identity `generation` filter registered in build_env.
        let src = "{% generation %}{{ x }}{% endgeneration %}";
        let out = strip_generation_markers(src);
        assert_eq!(out, "{% filter generation %}{{ x }}{% endfilter %}");
    }

    #[test]
    fn test_strip_generation_markers_with_whitespace_trim() {
        // `{%- generation -%}` whitespace-trim form must also be
        // recognized (HF chat templates use both forms).
        let src = "{%- generation -%}body{%- endgeneration -%}";
        let out = strip_generation_markers(src);
        assert_eq!(out, "{% filter generation %}body{% endfilter %}");
    }

    #[test]
    fn test_strip_generation_markers_renders_body_through_filter() {
        // End-to-end: rewritten template parses, renders, and the
        // body inside the `{% filter generation %}` block ends up in
        // the output verbatim (because the filter is identity).
        let tpl = ChatTemplate::new(
            "{% for m in messages %}{{ m.role }}: \
             {% generation %}{{ m.content }}{% endgeneration %}\n\
             {% endfor %}"
                .to_string(),
        )
        .expect("rewritten template must parse");
        let messages = vec![TemplateMessage {
            role: "assistant".to_string(),
            content: "hello".to_string(),
        }];
        let result = tpl.apply_simple(&messages, false).unwrap();
        assert!(
            result.contains("assistant: hello"),
            "filter-generation body must render verbatim: got {result:?}",
        );
    }

    #[test]
    fn test_strip_generation_markers_llava_template_compiles() {
        // The actual LLaVA-1.5 chat template (from
        // `llava-hf/llava-1.5-7b-hf/chat_template.json`) uses
        // `{% generation %}` blocks. Without the rewrite, minijinja
        // rejects with `unknown statement generation`; after the
        // rewrite, ChatTemplate::new must accept it cleanly.
        let llava_template = "{% for message in messages %}\
            {% if message['role'] != 'system' %}\
            {{ message['role'].upper() + ': '}}\
            {% endif %}\
            {% for content in message['content'] | selectattr('type', 'equalto', 'image') %}\
            {{ '<image>\\n' }}\
            {% endfor %}\
            {% if message['role'] != 'assistant' %}\
            {% for content in message['content'] | selectattr('type', 'equalto', 'text') %}\
            {{ content['text'] + ' '}}\
            {% endfor %}\
            {% else %}\
            {% for content in message['content'] | selectattr('type', 'equalto', 'text') %}\
            {% generation %}{{ content['text'] + ' '}}{% endgeneration %}\
            {% endfor %}\
            {% endif %}\
            {% endfor %}\
            {% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}";
        ChatTemplate::new(llava_template.to_string())
            .expect("rewritten LLaVA template must parse under minijinja");
    }

    #[test]
    fn test_simple_template() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}".to_string(),
        ).unwrap();

        let messages = vec![TemplateMessage {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        }];

        let result = tpl.apply_simple(&messages, true).unwrap();
        assert!(result.contains("user: Hello!"));
        assert!(result.contains("assistant:"));
    }

    #[test]
    fn test_chatml_template() {
        // ChatML format used by Qwen, Yi, etc.
        let tpl = ChatTemplate::new(
            "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}".to_string(),
        ).unwrap();

        let messages = vec![
            TemplateMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            TemplateMessage {
                role: "user".to_string(),
                content: "Hi!".to_string(),
            },
        ];

        let result = tpl.apply_simple(&messages, true).unwrap();
        assert!(result.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHi!<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_bos_eos_tokens() {
        let tpl = ChatTemplate::new(
            "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}{{ eos_token }}".to_string(),
        ).unwrap()
        .with_bos_token("<s>".to_string())
        .with_eos_token("</s>".to_string());

        let messages = vec![TemplateMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        let result = tpl.apply_simple(&messages, false).unwrap();
        assert_eq!(result, "<s>Hello</s>");
    }

    #[test]
    fn test_no_generation_prompt() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}GENERATE{% endif %}".to_string(),
        ).unwrap();

        let messages = vec![TemplateMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];

        let with = tpl.apply_simple(&messages, true).unwrap();
        let without = tpl.apply_simple(&messages, false).unwrap();

        assert!(with.contains("GENERATE"));
        assert!(!without.contains("GENERATE"));
    }

    #[test]
    fn test_empty_messages() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        )
        .unwrap();

        let result = tpl.apply_simple(&[], false).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_multi_turn_conversation() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}[{{ message.role|upper }}] {{ message.content }}\n{% endfor %}".to_string(),
        ).unwrap();

        let messages = vec![
            TemplateMessage {
                role: "user".to_string(),
                content: "What is 2+2?".to_string(),
            },
            TemplateMessage {
                role: "assistant".to_string(),
                content: "4".to_string(),
            },
            TemplateMessage {
                role: "user".to_string(),
                content: "Thanks!".to_string(),
            },
        ];

        let result = tpl.apply_simple(&messages, false).unwrap();
        assert!(result.contains("[USER] What is 2+2?"));
        assert!(result.contains("[ASSISTANT] 4"));
        assert!(result.contains("[USER] Thanks!"));
    }

    #[test]
    fn test_from_tokenizer_config_missing_file() {
        let result =
            ChatTemplate::from_tokenizer_config(Path::new("/nonexistent/tokenizer_config.json"))
                .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_token_string_plain() {
        let val = Some(serde_json::Value::String("<s>".to_string()));
        assert_eq!(extract_token_string(val), Some("<s>".to_string()));
    }

    #[test]
    fn test_extract_token_string_object() {
        let obj = serde_json::json!({"content": "</s>", "lstrip": false});
        let val = Some(obj);
        assert_eq!(extract_token_string(val), Some("</s>".to_string()));
    }

    #[test]
    fn test_extract_token_string_none() {
        assert_eq!(extract_token_string(None), None);
    }

    #[test]
    fn test_tools_passed_to_template() {
        // Template that dumps tools via tojson.
        let tpl = ChatTemplate::new(
            "{% if tools %}TOOLS:{{ tools | tojson }}{% endif %}{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}".to_string(),
        ).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let tools = serde_json::json!([
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}}
        ]);

        let result = tpl.apply(&messages, false, Some(&tools)).unwrap();
        assert!(result.contains("TOOLS:"));
        assert!(result.contains("get_weather"));
    }

    #[test]
    fn test_tool_calls_in_message() {
        // Template that accesses tool_calls on a message.
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.role }}:{% if message.tool_calls %} CALL={{ message.tool_calls[0].function.name }}{% endif %} {{ message.content }}\n{% endfor %}".to_string(),
        ).unwrap();

        let messages = vec![
            serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "NYC"}}
                }]
            }),
            serde_json::json!({
                "role": "tool",
                "content": "{\"temp\": 72}",
                "tool_call_id": "call_1"
            }),
        ];

        let result = tpl.apply(&messages, false, None).unwrap();
        assert!(result.contains("CALL=get_weather"));
        assert!(result.contains("tool:"));
    }

    #[test]
    fn test_no_tools_omitted_from_template() {
        let tpl =
            ChatTemplate::new("{% if tools %}HAS_TOOLS{% else %}NO_TOOLS{% endif %}".to_string())
                .unwrap();

        let result = tpl.apply(&[], false, None).unwrap();
        assert!(result.contains("NO_TOOLS"));

        let tools = serde_json::json!([{"type": "function", "function": {"name": "f"}}]);
        let result = tpl.apply(&[], false, Some(&tools)).unwrap();
        assert!(result.contains("HAS_TOOLS"));
    }

    #[test]
    fn test_date_string_in_template() {
        let tpl = ChatTemplate::new("DATE:{{ date_string }}".to_string()).unwrap();
        let result = tpl.apply(&[], false, None).unwrap();
        // Should contain a date like "28 Feb 2026".
        assert!(result.starts_with("DATE:"));
        assert!(result.len() > 5);
    }

    #[test]
    fn test_extra_kwargs_enable_thinking_true() {
        // Simplified Qwen3-style template that checks enable_thinking.
        let tpl = ChatTemplate::new(
            "{% set enable_thinking = enable_thinking | default(true) %}{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}".to_string(),
        ).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];

        // No extra kwargs → enable_thinking defaults to true → <think> present.
        let result = tpl.apply(&messages, true, None).unwrap();
        assert!(
            result.contains("<think>"),
            "expected <think> with default enable_thinking=true, got: {result}"
        );

        // Explicit enable_thinking=true.
        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
        let result = tpl
            .apply_with_kwargs(&messages, true, None, Some(&kwargs))
            .unwrap();
        assert!(
            result.contains("<think>"),
            "expected <think> with enable_thinking=true, got: {result}"
        );
    }

    #[test]
    fn test_extra_kwargs_enable_thinking_false() {
        // Simplified Qwen3-style template that checks enable_thinking.
        let tpl = ChatTemplate::new(
            "{% set enable_thinking = enable_thinking | default(true) %}{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}".to_string(),
        ).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];

        // enable_thinking=false → no <think>.
        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert(
            "enable_thinking".to_string(),
            serde_json::Value::Bool(false),
        );
        let result = tpl
            .apply_with_kwargs(&messages, true, None, Some(&kwargs))
            .unwrap();
        assert!(
            !result.contains("<think>"),
            "expected no <think> with enable_thinking=false, got: {result}"
        );
    }

    #[test]
    fn test_extra_kwargs_do_not_override_builtins() {
        // Extra kwargs should be able to add new variables but built-in context
        // (messages, add_generation_prompt, etc.) should still work.
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.content }}{% endfor %}{% if custom_var %}CUSTOM{% endif %}".to_string(),
        ).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "Hello"})];
        let mut kwargs = std::collections::HashMap::new();
        kwargs.insert("custom_var".to_string(), serde_json::Value::Bool(true));
        let result = tpl
            .apply_with_kwargs(&messages, false, None, Some(&kwargs))
            .unwrap();
        assert!(result.contains("Hello"), "messages should still render");
        assert!(result.contains("CUSTOM"), "custom_var should be accessible");
    }

    #[test]
    fn test_apply_with_kwargs_none_is_same_as_apply() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        )
        .unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let a = tpl.apply(&messages, false, None).unwrap();
        let b = tpl.apply_with_kwargs(&messages, false, None, None).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_extra_kwargs_empty_map() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        )
        .unwrap();
        let messages = vec![serde_json::json!({"role": "user", "content": "Hi"})];
        let kwargs = std::collections::HashMap::new();
        let result = tpl
            .apply_with_kwargs(&messages, false, None, Some(&kwargs))
            .unwrap();
        assert_eq!(result, "Hi");
    }
}
