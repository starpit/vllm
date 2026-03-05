// SPDX-License-Identifier: Apache-2.0
//! Chat template application for common formats.
//!
//! Parses `tokenizer_config.json` to detect the chat template, then
//! applies it to format user messages. Falls back to raw prompt if
//! no known template is detected.

use serde::Deserialize;

#[derive(Debug, Clone)]
pub enum ChatFormat {
    /// ChatML: `<|im_start|>role\ncontent<|im_end|>`
    ChatMl,
    /// Llama 3: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,
    /// No template detected — use raw prompt.
    Raw,
}

#[derive(Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    chat_template: Option<String>,
}

/// Detect the chat format from `tokenizer_config.json` contents.
pub fn detect_format(tokenizer_config_json: &str) -> ChatFormat {
    if let Ok(TokenizerConfig {
        chat_template: Some(ref tmpl),
    }) = serde_json::from_str::<TokenizerConfig>(tokenizer_config_json)
    {
        if tmpl.contains("im_start") {
            return ChatFormat::ChatMl;
        }
        if tmpl.contains("start_header_id") {
            return ChatFormat::Llama3;
        }
    }
    ChatFormat::Raw
}

/// Format a user message using the detected chat template.
/// Returns the full prompt string ready for tokenization.
pub fn apply_template(format: &ChatFormat, user_message: &str) -> String {
    match format {
        ChatFormat::ChatMl => format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n{user_message}<|im_end|>\n\
             <|im_start|>assistant\n"
        ),
        ChatFormat::Llama3 => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n\
             You are a helpful assistant.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n\
             {user_message}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        ChatFormat::Raw => user_message.to_string(),
    }
}

/// Detect the EOS token ID for the given chat format.
pub fn eos_token_id(format: &ChatFormat, tokenizer: &tokenizers::Tokenizer) -> u32 {
    match format {
        ChatFormat::Llama3 => tokenizer.token_to_id("<|eot_id|>").unwrap_or(2),
        ChatFormat::ChatMl => tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2),
        ChatFormat::Raw => tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2),
    }
}
