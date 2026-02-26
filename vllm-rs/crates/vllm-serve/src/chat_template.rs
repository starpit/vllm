// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Chat template application using HuggingFace Jinja2 templates.
//!
//! HuggingFace models store a Jinja2 chat template in `tokenizer_config.json`.
//! This module parses that template and applies it to a list of chat messages,
//! producing the formatted prompt string that the model expects.
//!
//! Uses `minijinja` (a Rust Jinja2 engine) for template rendering.

use std::path::Path;

use minijinja::Environment;
use serde::{Deserialize, Serialize};

use crate::error::ServeError;

// ---------------------------------------------------------------------------
// ChatTemplate
// ---------------------------------------------------------------------------

/// A parsed chat template that can format messages for a specific model.
#[derive(Clone)]
pub struct ChatTemplate {
    /// The raw Jinja2 template string.
    template_str: String,
    /// Optional BOS token string (e.g. "<s>", "<|begin_of_text|>").
    bos_token: Option<String>,
    /// Optional EOS token string (e.g. "</s>", "<|end_of_text|>").
    eos_token: Option<String>,
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

impl ChatTemplate {
    /// Create a `ChatTemplate` from a raw Jinja2 template string.
    pub fn new(template_str: String) -> Self {
        Self {
            template_str,
            bos_token: None,
            eos_token: None,
        }
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

        let data = std::fs::read_to_string(path).map_err(|e| {
            ServeError::Internal(format!("failed to read {}: {e}", path.display()))
        })?;

        let config: TokenizerConfig = serde_json::from_str(&data).map_err(|e| {
            ServeError::Internal(format!(
                "failed to parse {}: {e}",
                path.display()
            ))
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

        let bos_token = extract_token_string(&config.bos_token);
        let eos_token = extract_token_string(&config.eos_token);

        let mut tpl = ChatTemplate::new(template_str);
        if let Some(bos) = bos_token {
            tpl = tpl.with_bos_token(bos);
        }
        if let Some(eos) = eos_token {
            tpl = tpl.with_eos_token(eos);
        }

        Ok(Some(tpl))
    }

    /// Apply the chat template to a list of messages.
    ///
    /// Returns the formatted prompt string ready for tokenization.
    pub fn apply(
        &self,
        messages: &[TemplateMessage],
        add_generation_prompt: bool,
    ) -> Result<String, ServeError> {
        let mut env = Environment::new();

        // Add a `raise_exception` function that Jinja2 templates often use.
        env.add_function("raise_exception", raise_exception);

        env.add_template("chat", &self.template_str)
            .map_err(|e| ServeError::Internal(format!("invalid chat template: {e}")))?;

        let tmpl = env.get_template("chat").map_err(|e| {
            ServeError::Internal(format!("failed to get template: {e}"))
        })?;

        // Build the context.
        let ctx = minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => self.bos_token.as_deref().unwrap_or(""),
            eos_token => self.eos_token.as_deref().unwrap_or(""),
        };

        let rendered = tmpl.render(ctx).map_err(|e| {
            ServeError::Internal(format!("chat template render failed: {e}"))
        })?;

        Ok(rendered)
    }

    /// Get the raw template string.
    pub fn template_str(&self) -> &str {
        &self.template_str
    }
}

/// Extract a token string from the `bos_token` / `eos_token` field in
/// tokenizer_config.json. These can be either a plain string or an object
/// with a `content` field.
fn extract_token_string(value: &Option<serde_json::Value>) -> Option<String> {
    match value {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Object(obj)) => {
            obj.get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        }
        _ => None,
    }
}

/// A raise_exception function for Jinja2 compatibility.
/// Many HF chat templates use `{% raise_exception(...) %}` for error handling.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_template() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}".to_string(),
        );

        let messages = vec![
            TemplateMessage {
                role: "user".to_string(),
                content: "Hello!".to_string(),
            },
        ];

        let result = tpl.apply(&messages, true).unwrap();
        assert!(result.contains("user: Hello!"));
        assert!(result.contains("assistant:"));
    }

    #[test]
    fn test_chatml_template() {
        // ChatML format used by Qwen, Yi, etc.
        let tpl = ChatTemplate::new(
            "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}".to_string(),
        );

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

        let result = tpl.apply(&messages, true).unwrap();
        assert!(result.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHi!<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_bos_eos_tokens() {
        let tpl = ChatTemplate::new(
            "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}{{ eos_token }}".to_string(),
        )
        .with_bos_token("<s>".to_string())
        .with_eos_token("</s>".to_string());

        let messages = vec![TemplateMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        let result = tpl.apply(&messages, false).unwrap();
        assert_eq!(result, "<s>Hello</s>");
    }

    #[test]
    fn test_no_generation_prompt() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}GENERATE{% endif %}".to_string(),
        );

        let messages = vec![TemplateMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];

        let with = tpl.apply(&messages, true).unwrap();
        let without = tpl.apply(&messages, false).unwrap();

        assert!(with.contains("GENERATE"));
        assert!(!without.contains("GENERATE"));
    }

    #[test]
    fn test_empty_messages() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}{{ message.content }}{% endfor %}".to_string(),
        );

        let result = tpl.apply(&[], false).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_multi_turn_conversation() {
        let tpl = ChatTemplate::new(
            "{% for message in messages %}[{{ message.role|upper }}] {{ message.content }}\n{% endfor %}".to_string(),
        );

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

        let result = tpl.apply(&messages, false).unwrap();
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
        assert_eq!(extract_token_string(&val), Some("<s>".to_string()));
    }

    #[test]
    fn test_extract_token_string_object() {
        let obj = serde_json::json!({"content": "</s>", "lstrip": false});
        let val = Some(obj);
        assert_eq!(extract_token_string(&val), Some("</s>".to_string()));
    }

    #[test]
    fn test_extract_token_string_none() {
        assert_eq!(extract_token_string(&None), None);
    }
}
