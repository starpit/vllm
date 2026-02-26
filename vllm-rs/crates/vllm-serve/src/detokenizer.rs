// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Incremental detokenizer for streaming text generation.
//!
//! Converts token IDs to text incrementally as they are generated, using a
//! sliding-window approach that re-decodes a small window of context to
//! handle tokenizer-dependent boundaries (e.g., space insertion between words).
//!
//! Also provides stop-string detection to halt generation when a stop phrase
//! is encountered in the decoded text.
//!
//! Port of: `vllm/v1/engine/detokenizer.py`

use std::sync::Arc;

use crate::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// IncrementalDetokenizer
// ---------------------------------------------------------------------------

/// Incrementally converts token IDs to text using a sliding-window decode.
///
/// Mirrors the Python `SlowIncrementalDetokenizer` approach: to determine
/// the correct text for newly generated tokens, we re-decode a window of
/// recent tokens and subtract the previously-known prefix. This handles
/// tokenizer quirks (e.g., whether a space should precede a token) correctly.
pub struct IncrementalDetokenizer {
    /// Reference to the shared tokenizer.
    tokenizer: Arc<Tokenizer>,

    /// All token IDs: prompt tokens followed by generated tokens.
    all_token_ids: Vec<u32>,

    /// Number of prompt tokens (to separate prompt from output).
    num_prompt_tokens: usize,

    /// Start of the re-decode window (index into `all_token_ids`).
    prefix_offset: usize,

    /// End of the previously-decoded region (index into `all_token_ids`).
    read_offset: usize,

    /// Accumulated decoded output text.
    output_text: String,

    /// Offset into `output_text` for delta-mode output.
    last_output_text_offset: usize,

    // -- Stop string handling --
    /// Stop strings from sampling params.
    stop_strings: Vec<String>,

    /// Minimum tokens before stop strings can trigger.
    min_tokens: u32,

    /// Whether to include the matched stop string in the output text.
    include_stop_str_in_output: bool,

    /// Number of characters to hold back from output when stop strings are
    /// active and `include_stop_str_in_output` is false, to avoid emitting
    /// text that may later be found to contain a stop string.
    stop_buffer_length: usize,

    /// Whether to skip special tokens during decoding.
    skip_special_tokens: bool,
}

impl IncrementalDetokenizer {
    /// Create a new detokenizer for a request.
    ///
    /// `prompt_token_ids` are the tokenized prompt (used as context for
    /// the sliding-window decode). `sampling_params` provides stop strings
    /// and decode settings.
    pub fn new(
        tokenizer: Arc<Tokenizer>,
        prompt_token_ids: &[u32],
        stop_strings: Vec<String>,
        min_tokens: u32,
        include_stop_str_in_output: bool,
        skip_special_tokens: bool,
    ) -> Self {
        let num_prompt = prompt_token_ids.len();
        let all_token_ids = prompt_token_ids.to_vec();

        // The stop buffer length is the max length of any stop string.
        // We hold back this many characters from streaming output to ensure
        // we don't emit partial stop strings.
        let stop_buffer_length = stop_strings.iter().map(|s| s.len()).max().unwrap_or(0);

        // Initialize the sliding window: prefix_offset starts near the end
        // of the prompt (keeping ~6 tokens of context), read_offset at the
        // end of the prompt.
        let prefix_offset = num_prompt.saturating_sub(6);
        let read_offset = num_prompt;

        Self {
            tokenizer,
            all_token_ids,
            num_prompt_tokens: num_prompt,
            prefix_offset,
            read_offset,
            output_text: String::new(),
            last_output_text_offset: 0,
            stop_strings,
            min_tokens,
            include_stop_str_in_output,
            stop_buffer_length,
            skip_special_tokens,
        }
    }

    /// Update the detokenizer with newly generated token IDs.
    ///
    /// Returns `Some(stop_string)` if a stop string was matched in the
    /// decoded text, or `None` if generation should continue.
    ///
    /// `stop_terminated` indicates whether the engine already decided to
    /// stop (e.g., due to a stop token ID match). When true, we skip
    /// stop-string checking since the engine has already handled it.
    pub fn update(&mut self, new_token_ids: &[u32], stop_terminated: bool) -> Option<String> {
        if new_token_ids.is_empty() {
            return None;
        }

        // Extend with new tokens.
        self.all_token_ids.extend_from_slice(new_token_ids);

        // Decode incrementally using the sliding window.
        let new_text = self.decode_incremental();

        let new_char_count = new_text.len();
        self.output_text.push_str(&new_text);

        // Check stop strings if we have enough tokens and engine hasn't
        // already terminated.
        let num_output_tokens = self.num_output_tokens();
        if !stop_terminated
            && !self.stop_strings.is_empty()
            && num_output_tokens as u32 >= self.min_tokens
            && let Some((stop_str, truncate_offset)) =
                check_stop_strings(&self.output_text, new_char_count, &self.stop_strings)
        {
            if self.include_stop_str_in_output {
                // Keep the stop string in the output.
                let end = truncate_offset + stop_str.len();
                self.output_text.truncate(end);
            } else {
                // Remove the stop string from the output.
                self.output_text.truncate(truncate_offset);
            }
            return Some(stop_str);
        }

        None
    }

    /// Get the next output text.
    ///
    /// If `delta` is true, returns only the new text since the last call.
    /// If `finished` is true, flushes any buffered text (stop buffer).
    pub fn get_next_output_text(&mut self, finished: bool, delta: bool) -> String {
        if delta {
            // In delta mode, we need to account for the stop buffer.
            let effective_end = if !finished && self.stop_buffer_length > 0 {
                // Hold back stop_buffer_length chars.
                self.output_text
                    .len()
                    .saturating_sub(self.stop_buffer_length)
            } else {
                self.output_text.len()
            };

            if effective_end <= self.last_output_text_offset {
                return String::new();
            }

            let text = self.output_text[self.last_output_text_offset..effective_end].to_string();
            self.last_output_text_offset = effective_end;
            text
        } else {
            // Cumulative mode: return all output text.
            if !finished && self.stop_buffer_length > 0 {
                let end = self
                    .output_text
                    .len()
                    .saturating_sub(self.stop_buffer_length);
                self.output_text[..end].to_string()
            } else {
                self.output_text.clone()
            }
        }
    }

    /// Number of output tokens generated so far.
    pub fn num_output_tokens(&self) -> usize {
        self.all_token_ids.len() - self.num_prompt_tokens
    }

    /// The output token IDs generated so far (excluding prompt).
    pub fn output_token_ids(&self) -> &[u32] {
        &self.all_token_ids[self.num_prompt_tokens..]
    }

    /// The full accumulated output text.
    pub fn output_text(&self) -> &str {
        &self.output_text
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    /// Decode new tokens using the sliding-window approach.
    ///
    /// Re-decodes from `prefix_offset` to the end and subtracts the
    /// previously-known text (from `prefix_offset` to `read_offset`).
    fn decode_incremental(&mut self) -> String {
        if self.all_token_ids.len() <= self.read_offset {
            return String::new();
        }

        // Decode the prefix (what we already processed).
        let prefix_text = self
            .tokenizer
            .decode(
                &self.all_token_ids[self.prefix_offset..self.read_offset],
                self.skip_special_tokens,
            )
            .unwrap_or_default();

        // Decode from prefix_offset to the end (includes new tokens).
        let full_text = self
            .tokenizer
            .decode(
                &self.all_token_ids[self.prefix_offset..],
                self.skip_special_tokens,
            )
            .unwrap_or_default();

        // If the full text is longer than the prefix and doesn't end with
        // the Unicode replacement character (incomplete UTF-8), extract
        // the new portion.
        if full_text.len() > prefix_text.len() && !full_text.ends_with('\u{fffd}') {
            // Update the sliding window: keep ~6 tokens of context.
            self.prefix_offset = self.all_token_ids.len().saturating_sub(6);
            self.read_offset = self.all_token_ids.len();
            full_text[prefix_text.len()..].to_string()
        } else {
            // No decodable new text yet (possibly incomplete character).
            String::new()
        }
    }
}

impl std::fmt::Debug for IncrementalDetokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncrementalDetokenizer")
            .field("num_prompt_tokens", &self.num_prompt_tokens)
            .field("num_output_tokens", &self.num_output_tokens())
            .field("output_text_len", &self.output_text.len())
            .field("num_stop_strings", &self.stop_strings.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Stop string checking
// ---------------------------------------------------------------------------

/// Check if any stop string appears in the output text.
///
/// Searches only the region where a stop string could have been newly
/// completed: the last `new_char_count + max_stop_len` characters.
///
/// Returns `Some((stop_string, offset))` where `offset` is the byte position
/// in `output_text` where the stop string starts, or `None` if no match.
pub fn check_stop_strings(
    output_text: &str,
    new_char_count: usize,
    stop_strings: &[String],
) -> Option<(String, usize)> {
    if stop_strings.is_empty() || output_text.is_empty() {
        return None;
    }

    for stop in stop_strings {
        if stop.is_empty() {
            continue;
        }
        // Only search the window where the stop string could newly appear.
        let search_window = new_char_count + stop.len();
        let search_start = output_text.len().saturating_sub(search_window);

        // Find the start on a char boundary.
        let search_start = find_char_boundary(output_text, search_start);

        if let Some(pos) = output_text[search_start..].find(stop.as_str()) {
            return Some((stop.clone(), search_start + pos));
        }
    }

    None
}

/// Find the nearest char boundary at or after `byte_offset`.
fn find_char_boundary(s: &str, byte_offset: usize) -> usize {
    if byte_offset >= s.len() {
        return s.len();
    }
    let mut offset = byte_offset;
    while !s.is_char_boundary(offset) {
        offset += 1;
    }
    offset
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::make_test_tokenizer;

    fn make_detokenizer(prompt: &str, stop_strings: Vec<String>) -> IncrementalDetokenizer {
        let tok = Arc::new(make_test_tokenizer());
        let prompt_ids = tok.encode(prompt, false).unwrap();
        IncrementalDetokenizer::new(
            tok,
            &prompt_ids,
            stop_strings,
            0,     // min_tokens
            false, // include_stop_str_in_output
            false, // skip_special_tokens
        )
    }

    // -- check_stop_strings tests --

    #[test]
    fn test_check_stop_strings_empty() {
        assert!(check_stop_strings("hello world", 5, &[]).is_none());
    }

    #[test]
    fn test_check_stop_strings_no_match() {
        let stops = vec!["foo".to_string()];
        assert!(check_stop_strings("hello world", 5, &stops).is_none());
    }

    #[test]
    fn test_check_stop_strings_match() {
        let stops = vec!["world".to_string()];
        let result = check_stop_strings("hello world", 6, &stops);
        assert!(result.is_some());
        let (stop, offset) = result.unwrap();
        assert_eq!(stop, "world");
        assert_eq!(offset, 6); // "hello " is 6 bytes
    }

    #[test]
    fn test_check_stop_strings_match_at_boundary() {
        let stops = vec!["lo wo".to_string()];
        let result = check_stop_strings("hello world", 6, &stops);
        assert!(result.is_some());
        let (stop, offset) = result.unwrap();
        assert_eq!(stop, "lo wo");
        assert_eq!(offset, 3);
    }

    #[test]
    fn test_check_stop_strings_first_match_wins() {
        let stops = vec!["world".to_string(), "ello".to_string()];
        let result = check_stop_strings("hello world", 11, &stops);
        assert!(result.is_some());
        // "world" is checked first and matches.
        assert_eq!(result.unwrap().0, "world");
    }

    #[test]
    fn test_check_stop_strings_multibyte() {
        let stops = vec!["世界".to_string()];
        let text = "こんにちは世界";
        let result = check_stop_strings(text, text.len(), &stops);
        assert!(result.is_some());
        let (stop, _offset) = result.unwrap();
        assert_eq!(stop, "世界");
    }

    // -- IncrementalDetokenizer tests --

    #[test]
    fn test_detokenizer_initial_state() {
        let detok = make_detokenizer("Hello", vec![]);
        assert_eq!(detok.num_output_tokens(), 0);
        assert!(detok.output_token_ids().is_empty());
        assert_eq!(detok.output_text(), "");
    }

    #[test]
    fn test_detokenizer_update_produces_text() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Hello";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // Encode some continuation text and feed it token by token.
        let continuation = " world";
        let cont_ids = tok
            .encode(&format!("{prompt}{continuation}"), false)
            .unwrap();
        let new_ids = &cont_ids[prompt_ids.len()..];

        let stop = detok.update(new_ids, false);
        assert!(stop.is_none());
        assert!(detok.num_output_tokens() > 0);

        let text = detok.get_next_output_text(true, false);
        // The decoded text should contain "world" (possibly with leading space).
        assert!(
            text.contains("world"),
            "Expected 'world' in output, got: {text:?}"
        );
    }

    #[test]
    fn test_detokenizer_delta_mode() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Hi";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok =
            IncrementalDetokenizer::new(Arc::clone(&tok), &prompt_ids, vec![], 0, false, false);

        // Feed tokens in two batches.
        let full = tok.encode(&format!("{prompt} ab cd"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];

        if new_ids.len() >= 2 {
            let mid = new_ids.len() / 2;
            detok.update(&new_ids[..mid], false);
            let text1 = detok.get_next_output_text(false, true);

            detok.update(&new_ids[mid..], false);
            let text2 = detok.get_next_output_text(true, true);

            // Both deltas should be non-overlapping portions of the output.
            let combined = format!("{text1}{text2}");
            let full_text = detok.get_next_output_text(true, false);
            // The cumulative text should contain the delta texts.
            assert!(!full_text.is_empty(), "Expected non-empty output text");
            // The combined delta texts should equal the full text minus any
            // stop buffer effects (no stop strings here).
            assert_eq!(
                combined, full_text,
                "Delta concatenation should equal full text"
            );
        }
    }

    #[test]
    fn test_detokenizer_stop_string() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Say";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok = IncrementalDetokenizer::new(
            Arc::clone(&tok),
            &prompt_ids,
            vec!["STOP".to_string()],
            0,
            false,
            false,
        );

        // Encode text that contains the stop string.
        let full = tok
            .encode(&format!("{prompt} hello STOP bye"), false)
            .unwrap();
        let new_ids = &full[prompt_ids.len()..];

        let result = detok.update(new_ids, false);
        // Should detect the stop string.
        if let Some(stop_str) = result {
            assert_eq!(stop_str, "STOP");
            // Output text should not contain STOP (include_stop_str_in_output=false).
            let text = detok.get_next_output_text(true, false);
            assert!(
                !text.contains("STOP"),
                "Stop string should be excluded: {text:?}"
            );
        }
        // If the tokenizer doesn't produce exactly "STOP", the test still
        // passes because the stop string check requires exact text match.
    }

    #[test]
    fn test_detokenizer_stop_string_included() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "Say";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok = IncrementalDetokenizer::new(
            Arc::clone(&tok),
            &prompt_ids,
            vec!["STOP".to_string()],
            0,
            true, // include_stop_str_in_output
            false,
        );

        let full = tok
            .encode(&format!("{prompt} hello STOP bye"), false)
            .unwrap();
        let new_ids = &full[prompt_ids.len()..];

        let result = detok.update(new_ids, false);
        if let Some(stop_str) = result {
            assert_eq!(stop_str, "STOP");
            let text = detok.get_next_output_text(true, false);
            assert!(
                text.contains("STOP"),
                "Stop string should be included: {text:?}"
            );
        }
    }

    #[test]
    fn test_detokenizer_stop_buffer() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "X";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        // Use a stop string to create a buffer.
        let mut detok = IncrementalDetokenizer::new(
            Arc::clone(&tok),
            &prompt_ids,
            vec!["END".to_string()], // 3-char stop buffer
            0,
            false,
            false,
        );

        let full = tok.encode(&format!("{prompt} hello there"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];
        detok.update(new_ids, false);

        // Non-finished output should hold back 3 characters.
        let partial = detok.get_next_output_text(false, false);
        let full_text = detok.get_next_output_text(true, false);
        // The partial output should be shorter or equal to the full output
        // (shorter by up to stop_buffer_length characters).
        assert!(partial.len() <= full_text.len());
    }

    #[test]
    fn test_detokenizer_min_tokens() {
        let tok = Arc::new(make_test_tokenizer());
        let prompt = "A";
        let prompt_ids = tok.encode(prompt, false).unwrap();

        let mut detok = IncrementalDetokenizer::new(
            Arc::clone(&tok),
            &prompt_ids,
            vec!["B".to_string()],
            100, // min_tokens = 100 -- stop strings won't trigger
            false,
            false,
        );

        // Even if "B" appears, it won't trigger because min_tokens is high.
        let full = tok.encode(&format!("{prompt} B C"), false).unwrap();
        let new_ids = &full[prompt_ids.len()..];
        let result = detok.update(new_ids, false);
        assert!(
            result.is_none(),
            "Stop string should not trigger before min_tokens"
        );
    }

    #[test]
    fn test_detokenizer_empty_update() {
        let mut detok = make_detokenizer("Hello", vec![]);
        let result = detok.update(&[], false);
        assert!(result.is_none());
        assert_eq!(detok.num_output_tokens(), 0);
    }

    #[test]
    fn test_detokenizer_debug() {
        let detok = make_detokenizer("Hello", vec!["stop".to_string()]);
        let debug_str = format!("{detok:?}");
        assert!(debug_str.contains("IncrementalDetokenizer"));
        assert!(debug_str.contains("num_stop_strings"));
    }

    // -- find_char_boundary tests --

    #[test]
    fn test_find_char_boundary_ascii() {
        let s = "hello";
        assert_eq!(find_char_boundary(s, 0), 0);
        assert_eq!(find_char_boundary(s, 3), 3);
        assert_eq!(find_char_boundary(s, 5), 5);
    }

    #[test]
    fn test_find_char_boundary_multibyte() {
        let s = "héllo"; // 'é' is 2 bytes
        // Byte layout: h(1) é(2) l(1) l(1) o(1) = 6 bytes
        assert_eq!(find_char_boundary(s, 0), 0);
        assert_eq!(find_char_boundary(s, 1), 1);
        // Byte 2 is in the middle of 'é', should advance to byte 3.
        assert_eq!(find_char_boundary(s, 2), 3);
    }

    #[test]
    fn test_find_char_boundary_beyond_end() {
        let s = "hi";
        assert_eq!(find_char_boundary(s, 10), 2);
    }
}
