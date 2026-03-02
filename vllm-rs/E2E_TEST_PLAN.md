# E2E Test Plan: vLLM Rust Port

## Overview

End-to-end tests validate the full stack — from HTTP request to model inference to HTTP response — using real models downloaded from HuggingFace. Unlike unit tests (665 today) which mock out the model/worker layer, E2E tests prove that `vllm serve <model>` actually starts, loads weights, generates coherent text, streams correctly, handles tool calls, obeys structured output constraints, and returns well-formed OpenAI-compatible responses.

**Strategy**: Tests construct the full inference stack in-process via `vllm_serve::init::initialize_stack()`, spawn the HTTP server on a random port, send requests using `reqwest`, and validate responses. Tests are `#[ignore]`-tagged and gated behind `--features e2e,metal` so they don't run in normal `cargo test`. A CI job downloads models once and caches them.

**Backend focus**: MLX (Apple Silicon) first, since that's the primary development target. Candle backend E2E tests are a future phase.

---

## How to Run

### Prerequisites

1. **Apple Silicon Mac required** for the MLX backend (`--features metal`). CPU-only and CUDA E2E tests are future work.

2. **Models are downloaded on first run** from HuggingFace Hub. Tier 1 models (~76–335 MB each) are fast; larger tiers take longer. Downloads are cached in `~/.cache/huggingface/hub/`.

### Architecture

Tests run **in-process** — no separate CLI binary needed. Each test constructs a `VllmConfig`, calls `vllm_serve::init::initialize_stack()` to build the full inference stack, then spawns the HTTP server on a random port. This is faster than spawning a child process and avoids binary discovery issues.

### Run commands

```bash
# All E2E tests (Tier 1–4, all phases):
cargo test -p vllm-e2e --features e2e,metal -- --ignored --test-threads=1

# Single phase:
cargo test -p vllm-e2e --features e2e,metal --test e1_basic_serving -- --ignored --test-threads=1
cargo test -p vllm-e2e --features e2e,metal --test e2_chat_completions -- --ignored --test-threads=1
cargo test -p vllm-e2e --features e2e,metal --test e3_streaming -- --ignored --test-threads=1
cargo test -p vllm-e2e --features e2e,metal --test e_lora --release -- --ignored --test-threads=1
cargo test -p vllm-e2e --features e2e,metal --test e_llm_api -- --ignored --test-threads=1

# Single test:
cargo test -p vllm-e2e --features e2e,metal --test e1_basic_serving test_t1_smollm_chat_basic -- --ignored

# PR tier only (Tier 1+2 models — SmolLM, Qwen2, Qwen3, Llama3):
cargo test -p vllm-e2e --features e2e,metal --test e1_basic_serving -- --ignored --test-threads=1 \
  test_t1 test_t2

# With verbose logging (model download progress, weight loading, etc.):
RUST_LOG=info cargo test -p vllm-e2e --features e2e,metal -- --ignored --test-threads=1
```

### Key details

- **`--features e2e,metal`** — both required. `e2e` gates test compilation (`#![cfg(feature = "e2e")]`); `metal` enables the MLX backend.
- **`-- --ignored`** — required. All E2E tests are `#[ignore]`-tagged so they don't run during normal `cargo test`.
- **`--test-threads=1`** — recommended. Each test loads a model into GPU memory; parallel tests cause excessive memory pressure.
- **Logging**: Silent by default. Set `RUST_LOG=info` (or `debug`, `trace`) to see model download progress, weight loading, cache initialization, and request handling.
- **Startup timeout**: 120 seconds per server (configurable via `TestServerBuilder::with_timeout`). First run may be slower due to HF model download.
- **Tokio runtime**: Tests use `#[tokio::test(flavor = "multi_thread")]` because the engine step loop requires `block_in_place`.
- **Async scheduling**: All E2E tests exercise the async scheduling path by default (executor on a dedicated OS thread, overlapping GPU execution with CPU scheduling). Use `TestServerBuilder::with_sync_scheduling()` to test the synchronous path.

---

## Test Infrastructure (Phase E0)

### E0a. Test harness crate

Create `vllm-rs/crates/vllm-e2e/` — a test-only crate with:

- **`TestServer`** helper: initializes the full stack in-process via `vllm_serve::init::initialize_stack()`, spawns the HTTP server on a random port, waits for `/health` to return 200, provides `base_url()`, aborts the server task on drop.
  ```rust
  let server = TestServer::builder("mlx-community/SmolLM-135M-Instruct-4bit")
      .with_tool_call_parser("kimi_k2")  // optional
      .start().await?;
  let url = server.base_url(); // "http://127.0.0.1:{port}"
  // ... send requests ...
  drop(server); // aborts server task
  ```
- **`Client`** wrapper: thin `reqwest`-based client with helpers for:
  - `chat_completion(request) -> ChatCompletionResponse`
  - `chat_completion_stream(request) -> Vec<ChatCompletionStreamChunk>`
  - `completion(request) -> CompletionResponse`
  - `list_models() -> ModelList`
  - `health() -> bool`
- **Assertion helpers**:
  - `assert_valid_chat_response(resp)` — checks id format, object type, choice count, finish_reason, usage fields
  - `assert_valid_stream(chunks)` — checks first chunk has role, last chunk has finish_reason, `[DONE]` sentinel
  - `assert_valid_completion_response(resp)` — similar for completions
  - `assert_coherent_text(text, min_len)` — checks text isn't empty/garbled (not just `<unk>` tokens)
  - `assert_valid_tool_calls(tool_calls)` — checks id, type, function.name, function.arguments parse as JSON
  - `assert_json_parseable(text)` — for structured output validation

### E0b. Model cache management

- **Model cache**: Uses the standard HuggingFace Hub cache at `~/.cache/huggingface/hub/`. No separate cache directory needed.
- **Cargo features**: `e2e` gates test compilation; `metal` enables the MLX backend.
- **CI integration**: GitHub Actions job with HF cache (keyed by model list hash)

### E0c. Test configuration

```rust
/// Test models by architecture (smallest available for CI).
struct TestModels;
impl TestModels {
    // Tier 1: Tiny (<500 MB) — run on every PR
    const SMOLLM_135M_4BIT: &str = "mlx-community/SmolLM-135M-Instruct-4bit";     // ~76 MB, LlamaForCausalLM
    const QWEN2_0_5B_4BIT: &str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit";    // ~276 MB, Qwen2ForCausalLM
    const QWEN3_0_6B_4BIT: &str = "mlx-community/Qwen3-0.6B-4bit";               // ~335 MB, Qwen3ForCausalLM

    // Tier 2: Small (<1 GB) — run on every PR
    const LLAMA_3_2_1B_4BIT: &str = "mlx-community/Llama-3.2-1B-Instruct-4bit";   // ~680 MB, LlamaForCausalLM
    const GEMMA3_270M_4BIT: &str = "mlx-community/gemma-3-270m-it-qat-4bit";       // ~900 MB, Gemma3ForCausalLM

    // Tier 3: Medium (1–3 GB) — nightly only
    const GEMMA2_2B_4BIT: &str = "mlx-community/gemma-2-2b-it-4bit";              // ~1.4 GB, Gemma2ForCausalLM
    const PHI3_5_MINI_4BIT: &str = "mlx-community/Phi-3.5-mini-instruct-4bit";    // ~2.15 GB, Phi3ForCausalLM
    const PHI4_MINI_4BIT: &str = "mlx-community/Unsloth-Phi-4-mini-instruct-4bit"; // ~2.3 GB, Phi3ForCausalLM (LongRoPE + partial_rotary_factor)

    // Tier 4: Large (3+ GB) — weekly/manual only
    const MISTRAL_7B_4BIT: &str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit";  // ~3.8 GB, MistralForCausalLM
    const COMMANDR_7B_4BIT: &str = "mlx-community/c4ai-command-r7b-12-2024-4bit"; // ~4.2 GB, Cohere2ForCausalLM (NOTE: needs Cohere2 arch)
    const DEEPSEEK_V2_LITE_4BIT: &str = "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"; // ~8.2 GB, DeepseekV2ForCausalLM

    // MoE models
    const QWEN3_MOE_4X06B_4BIT: &str = "justneedsomeavailableusername/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2-mlx-4Bit"; // ~1.5 GB, Qwen3MoeForCausalLM

    // Float16 variants for non-quantized testing
    const SMOLLM_135M_F16: &str = "mlx-community/SmolLM2-135M-Instruct";          // ~255 MB, LlamaForCausalLM

    // GPTQ quantized models (candle + MLX backends)
    const QWEN2_0_5B_GPTQ_INT4: &str = "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4";  // ~459 MB, Qwen2ForCausalLM

    // AWQ quantized models (candle + MLX backends)
    const QWEN2_0_5B_AWQ: &str = "Qwen/Qwen2.5-0.5B-Instruct-AWQ";              // ~393 MB, Qwen2ForCausalLM

    // Multimodal (vision-language) models
    const GEMMA3_4B_IT_QAT_3BIT: &str = "mlx-community/gemma-3-4b-it-qat-3bit";  // ~2.8 GB, Gemma3ForConditionalGeneration (MLX)
    const GEMMA3_4B_IT: &str = "google/gemma-3-4b-it";                            // ~8 GB BF16, Gemma3ForConditionalGeneration (Candle)
}
```

**Deliverables**: `vllm-e2e` crate, `TestServer`, `Client`, assertion helpers, model download script, CI workflow.

---

## Phase E1: Basic Serving — Smoke Tests

Validate that each model architecture loads, starts serving, and generates coherent text.

### E1a. Server lifecycle per architecture

For each model in the test matrix:

| Test | Description |
|------|-------------|
| `test_{arch}_server_starts` | Server starts, `/health` returns 200, `/v1/models` lists the model |
| `test_{arch}_chat_basic` | Simple "Say hello" → non-empty response, finish_reason = "stop" or "length" |
| `test_{arch}_completion_basic` | `POST /v1/completions` with "The capital of France is" → non-empty text |
| `test_{arch}_max_tokens` | `max_tokens: 5` → response has ≤ 5 completion tokens |
| `test_{arch}_version` | `/version` returns valid version string |

**Models for E1a** (Tier 1+2, run on every PR):

| Model | Architecture | Quantized |
|-------|-------------|-----------|
| SmolLM-135M-Instruct-4bit | LlamaForCausalLM | Yes (4-bit) |
| Qwen2.5-0.5B-Instruct-4bit | Qwen2ForCausalLM | Yes (4-bit) |
| Qwen3-0.6B-4bit | Qwen3ForCausalLM | Yes (4-bit) |
| Llama-3.2-1B-Instruct-4bit | LlamaForCausalLM | Yes (4-bit) |
| gemma-3-270m-it-qat-4bit | Gemma3ForCausalLM | Yes (4-bit) |

**Models for E1a-nightly** (Tier 3, nightly):

| Model | Architecture | Quantized |
|-------|-------------|-----------|
| gemma-2-2b-it-4bit | Gemma2ForCausalLM | Yes (4-bit) |
| Phi-3.5-mini-instruct-4bit | Phi3ForCausalLM | Yes (4-bit) |
| Unsloth-Phi-4-mini-instruct-4bit | Phi3ForCausalLM (LongRoPE) | Yes (4-bit) |
| Qwen3-MOE-4x0.6B-2.4B-mlx-4Bit | Qwen3MoeForCausalLM | Yes (4-bit) |

**Models for E1a-weekly** (Tier 4, weekly):

| Model | Architecture | Quantized |
|-------|-------------|-----------|
| Mistral-7B-Instruct-v0.3-4bit | MistralForCausalLM | Yes (4-bit) |
| DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx | DeepseekV2ForCausalLM | Yes (4-bit) |

### E1b. Float16 vs quantized comparison

| Test | Description |
|------|-------------|
| `test_float16_server_starts` | SmolLM2-135M-Instruct (float16) loads and serves |
| `test_float16_chat_basic` | Float16 model generates coherent text |
| `test_float16_vs_quantized_both_work` | Both float16 and 4-bit variants produce non-empty responses for same prompt |

### E1c. Scheduling mode coverage — DONE

All E2E tests exercise the **async scheduling** path by default (enabled in `initialize_stack()`, matching Python vLLM V1). One explicit sync-path test validates the synchronous step loop still works end-to-end.

| Test | Description |
|------|-------------|
| **`test_sync_scheduling_smollm_chat`** | **DONE** — `with_sync_scheduling()` builder → SmolLM chat works with synchronous step loop |

**Deliverables**: ~30 tests covering server lifecycle for all architectures + 1 sync scheduling test.

---

## Phase E2: Chat Completions — Full Feature Coverage

Deep testing of the `/v1/chat/completions` endpoint using SmolLM-135M (fastest model, ~76 MB).

### E2a. Request parameters

| Test | Description |
|------|-------------|
| `test_chat_temperature_0` | temperature=0 → deterministic output (2 calls same result) |
| `test_chat_temperature_high` | temperature=1.5 → output differs across calls |
| `test_chat_top_p` | top_p=0.1 → output is valid, tends to be less diverse |
| `test_chat_top_k` | top_k=5 → output is valid |
| `test_chat_min_p` | min_p=0.1 → output is valid |
| `test_chat_max_tokens` | max_tokens=10 → completion_tokens ≤ 10 |
| `test_chat_max_completion_tokens` | max_completion_tokens=10 → works identically |
| `test_chat_n_1` | n=1 → 1 choice |
| `test_chat_n_3` | n=3 → 3 choices, each with distinct index and text |
| `test_chat_stop_string` | stop=["world"] → output truncated before "world" |
| `test_chat_stop_token` | stop_token_ids with EOS → output stops |
| `test_chat_seed` | seed=42 → deterministic across calls |
| `test_chat_logprobs` | logprobs=true, top_logprobs=5 → logprobs in response |
| **`test_chat_prompt_logprobs`** | **DONE** — prompt_logprobs=3 → per-prompt-token logprobs in response, position 0 is None |
| `test_chat_logit_bias` | logit_bias suppresses specific token → token doesn't appear |
| `test_chat_frequency_penalty` | frequency_penalty=2.0 → less repetition |
| `test_chat_presence_penalty` | presence_penalty=2.0 → output is valid |
| `test_chat_repetition_penalty` | repetition_penalty=1.5 → less repetition |

### E2b. Message formats

| Test | Description |
|------|-------------|
| `test_chat_system_message` | system + user messages → coherent response |
| `test_chat_multi_turn` | system + user + assistant + user → contextual response |
| `test_chat_user_only` | Single user message → works |
| `test_chat_empty_content` | User message with empty string → doesn't crash |
| `test_chat_long_context` | 500+ token prompt → processes correctly |
| `test_chat_unicode` | Unicode characters in messages → preserved in response |

### E2c. Error handling

| Test | Description |
|------|-------------|
| `test_chat_missing_messages` | No messages field → 422 |
| `test_chat_empty_messages` | Empty messages array → 422 or error response |
| `test_chat_invalid_json` | Malformed JSON → 400 |
| `test_chat_invalid_role` | Unknown role → error |
| `test_chat_negative_max_tokens` | max_tokens=-1 → validation error |
| `test_chat_temperature_out_of_range` | temperature=5.0 → validation error |

**Deliverables**: ~30 tests validating chat completion parameters and edge cases.

---

## Phase E3: Streaming

Validate SSE streaming for chat completions.

### E3a. Basic streaming

| Test | Description |
|------|-------------|
| `test_stream_basic` | stream=true → SSE events, last event has `[DONE]` |
| `test_stream_content_matches_nonstream` | Same prompt, stream vs non-stream → same text (with seed) |
| `test_stream_first_chunk_has_role` | First chunk has `delta.role = "assistant"` |
| `test_stream_last_chunk_has_finish_reason` | Final chunk has `finish_reason` set |
| `test_stream_intermediate_chunks` | Middle chunks have content, no finish_reason |
| `test_stream_event_format` | Each event is `data: {json}\n\n` format |
| `test_stream_max_tokens` | stream + max_tokens=5 → ≤ 5 content chunks, finish_reason="length" |

### E3b. Streaming with n>1

| Test | Description |
|------|-------------|
| `test_stream_n2` | n=2 → chunks have index 0 and 1, both complete |
| `test_stream_n3_all_finish` | n=3 → all 3 choices eventually get finish_reason |
| `test_stream_n2_interleaved` | n=2 → chunks from different choices may interleave |

### E3c. Streaming edge cases

| Test | Description |
|------|-------------|
| `test_stream_stop_string` | stop=["world"] → stream ends at stop string |
| `test_stream_client_disconnect` | Client drops connection mid-stream → server doesn't crash |
| `test_stream_empty_response` | max_tokens=1 → at least one content chunk |
| `test_stream_logprobs` | stream + logprobs=true → logprobs in each chunk |

**Deliverables**: ~14 streaming tests.

---

## Phase E4: Text Completions

Validate the `/v1/completions` endpoint.

### E4a. Basic completions

| Test | Description |
|------|-------------|
| `test_completion_basic` | Single string prompt → non-empty text |
| `test_completion_max_tokens` | max_tokens=5 → completion_tokens ≤ 5 |
| `test_completion_temperature` | temperature=0 → deterministic |
| `test_completion_logprobs` | logprobs=5 → logprobs in response |
| `test_completion_echo` | echo=true → response includes prompt |

### E4b. Multi-prompt completions

| Test | Description |
|------|-------------|
| `test_completion_multi_prompt` | Array of 3 prompts → 3 choices |
| `test_completion_multi_prompt_n2` | 2 prompts × n=2 → 4 choices |
| `test_completion_token_ids_prompt` | Token ID array prompt → valid completion |

### E4c. Completion streaming

| Test | Description |
|------|-------------|
| `test_completion_stream` | stream=true → SSE events (when implemented) |

**Deliverables**: ~9 completion tests.

---

## Phase E5: Tool Calling

Validate tool call extraction from model output.

### E5a. Non-streaming tool calls (Hermes format)

Use a model with Hermes-style tool call support (Llama-3.2-1B-Instruct or Qwen2.5-0.5B-Instruct).

| Test | Description |
|------|-------------|
| `test_tool_call_basic` | Provide tools + "What's the weather?" → tool_calls in response |
| `test_tool_call_has_id` | Each tool call has a unique `id` field |
| `test_tool_call_has_function` | tool_calls[].function has `name` and `arguments` |
| `test_tool_call_arguments_valid_json` | `arguments` field parses as valid JSON |
| `test_tool_call_finish_reason` | finish_reason = "tool_calls" |
| `test_tool_call_no_content` | When tool calls present, `content` is null |
| `test_tool_call_none` | tool_choice="none" → no tool calls, regular text response |
| `test_tool_call_without_tools` | No tools provided → regular text response |
| `test_tool_call_multiple` | Prompt that should trigger multiple tool calls → multiple entries |

### E5b. Streaming tool calls

| Test | Description |
|------|-------------|
| `test_tool_call_stream_basic` | stream=true + tools → streaming tool call deltas |
| `test_tool_call_stream_has_id` | First delta for each tool has `id` and `type` |
| `test_tool_call_stream_function_name` | Function name appears in early delta |
| `test_tool_call_stream_arguments_accumulate` | Argument fragments concatenate to valid JSON |
| `test_tool_call_stream_finish_reason` | Last chunk has finish_reason = "tool_calls" |
| `test_tool_call_stream_n2` | n=2 + tools + stream → tool calls for both choices |

### E5c. Tool call parsers

| Test | Description |
|------|-------------|
| `test_hermes_parser_e2e` | `--tool-call-parser hermes` → extracts `<tool_call>` tags |
| `test_llama_json_parser_e2e` | `--tool-call-parser llama3_json` → extracts raw JSON |
| **`test_kimi_k2_server_starts`** | **DONE** — `kimi_k2` parser + SmolLM → server healthy |
| **`test_kimi_k2_chat_with_tools`** | **DONE** — non-streaming + tools → valid response |
| **`test_kimi_k2_chat_without_tools`** | **DONE** — non-streaming, no tools → coherent text |
| **`test_kimi_k2_stream_with_tools`** | **DONE** — streaming + tools → valid SSE chunks |
| **`test_kimi_k2_stream_without_tools`** | **DONE** — streaming, no tools → coherent streamed text |

### E5d. Tool message round-trip

| Test | Description |
|------|-------------|
| `test_tool_message_roundtrip` | user → assistant(tool_call) → tool(result) → assistant(summary) |

**Deliverables**: ~18 tool calling tests.

---

## Phase E6: Structured Output / Constrained Decoding

Validate `response_format` for JSON output.

### E6a. JSON object mode

| Test | Description |
|------|-------------|
| `test_json_object_mode` | response_format={"type":"json_object"} → output is valid JSON |
| `test_json_object_stream` | stream + json_object → concatenated stream is valid JSON |
| `test_json_object_multiple_keys` | JSON has multiple keys (not just a single string) |

### E6b. JSON schema mode

| Test | Description |
|------|-------------|
| `test_json_schema_basic` | response_format with json_schema → output matches schema |
| `test_json_schema_required_fields` | Schema with required fields → all present |
| `test_json_schema_enum` | Schema with enum → value is one of allowed values |
| `test_json_schema_nested` | Schema with nested objects → valid nested JSON |
| `test_json_schema_array` | Schema with array type → valid array |
| `test_json_schema_stream` | stream + json_schema → concatenated output matches schema |

### E6c. Structured output edge cases

| Test | Description |
|------|-------------|
| `test_json_schema_with_tools` | Both tools and response_format → structured output respected |
| `test_json_object_max_tokens` | json_object + short max_tokens → may be truncated but starts valid |

### E6d. Regex-constrained decoding

| Test | Description |
|------|-------------|
| `test_guided_regex_digits` | `guided_regex: "[0-9]+"` → output is all digits |
| `test_guided_regex_stream` | stream + `guided_regex` → concatenated output matches pattern |
| `test_guided_regex_conflict` | Both `response_format` and `guided_regex` → 400 error |

**Deliverables**: ~14 structured output tests.

---

## Phase E7: Sampling Features

Validate sampling parameters work end-to-end with real models.

### E7a. Logprobs

| Test | Description |
|------|-------------|
| `test_logprobs_chat` | logprobs=true → response has logprobs for each token |
| `test_logprobs_top_5` | top_logprobs=5 → each token has 5 top alternatives |
| `test_logprobs_stream` | stream + logprobs → logprobs in each chunk |
| `test_logprobs_completion` | completions + logprobs → logprobs in response |
| **`test_logprobs_prompt`** | **DONE** — prompt_logprobs=3 → per-prompt-token logprobs, first is None (see `test_chat_prompt_logprobs` in E2) |

### E7b. Determinism

| Test | Description |
|------|-------------|
| `test_seed_deterministic` | seed=42, temperature=0.5 → same result twice |
| `test_seed_different` | seed=42 vs seed=99 → different results |
| `test_temperature_0_deterministic` | temperature=0 → same result twice (no seed needed) |

### E7c. Penalty features

| Test | Description |
|------|-------------|
| `test_repetition_penalty_reduces_repeat` | repetition_penalty=2.0 → less token repetition than 1.0 |
| `test_frequency_penalty_works` | frequency_penalty=1.0 → valid output |
| `test_presence_penalty_works` | presence_penalty=1.0 → valid output |

**Deliverables**: ~10 sampling tests.

---

## Phase E8: Multi-Architecture Correctness

Validate that different architectures produce coherent output for the same prompts.

### E8a. Cross-architecture smoke tests

Run a standard prompt ("Explain what a compiler does in one sentence.") across all architectures and verify:

| Test | Description |
|------|-------------|
| `test_cross_arch_smollm` | SmolLM-135M → coherent English output |
| `test_cross_arch_qwen2` | Qwen2.5-0.5B → coherent output |
| `test_cross_arch_qwen3` | Qwen3-0.6B → coherent output |
| `test_cross_arch_llama3` | Llama-3.2-1B → coherent output |
| `test_cross_arch_gemma2` | Gemma2-2B → coherent output (nightly) |
| `test_cross_arch_phi3` | Phi-3.5-mini → coherent output (nightly) |
| `test_cross_arch_mistral` | Mistral-7B → coherent output (weekly) |
| `test_cross_arch_deepseek` | DeepSeek-V2-Lite → coherent output (weekly) |

### E8b. Architecture-specific features

| Test | Description |
|------|-------------|
| `test_qwen3_qk_norms` | Qwen3 with QK norms generates correctly |
| `test_gemma2_softcapping` | Gemma2 with logit softcapping generates correctly |
| `test_gemma2_4_norms` | Gemma2 4-norm architecture works correctly |
| `test_deepseek_moe` | DeepSeek V2 MoE routing produces coherent output (weekly) |
| `test_qwen2_sliding_window` | Qwen2.5-0.5B with prompt exceeding sliding window size → coherent output, no crash |
| `test_qwen2_max_window_layers` | Qwen2 config with `max_window_layers < num_hidden_layers` → sliding window correctly disabled, model still generates (nightly) |
| `test_phi3_sliding_window` | Phi-3.5-mini with long prompt near/beyond sliding window → correct generation (nightly) |
| `test_phi3_mlx_sliding_window` | MLX Phi-3 quantized model with sliding window config → output matches non-windowed for short prompts, no crash on long prompts (nightly, `--features metal`) |
| `test_mistral_sliding_window` | Mistral-7B with >4096 token context → handles gracefully (weekly) |
| `test_mistral_array_sliding_window` | Mistral 3.x model with array-format `sliding_window: [null, 4096, ...]` → config parsed correctly, model generates (nightly) |
| `test_gemma2_interleaved_sliding_window` | Gemma2 model with `layer_types` interleaved pattern → sliding window applied only to alternating layers, correct generation (nightly) |

**Deliverables**: ~18 cross-architecture tests.

---

## Phase E9: Concurrent Requests & Load

Validate the server handles concurrent requests correctly.

### E9a. Concurrent clients

| Test | Description |
|------|-------------|
| `test_concurrent_2_requests` | 2 simultaneous requests → both complete correctly |
| `test_concurrent_5_requests` | 5 simultaneous requests → all complete, no deadlock |
| `test_concurrent_stream_and_nonstream` | 1 streaming + 1 non-streaming simultaneously → both work |
| `test_concurrent_different_params` | Concurrent requests with different temperatures → each respects its params |

### E9b. Sequential request isolation

| Test | Description |
|------|-------------|
| `test_sequential_requests_isolated` | 10 sequential requests → each gets independent response |
| `test_request_after_long_generation` | Long generation (100 tokens) then short (5 tokens) → both correct |

**Deliverables**: ~6 concurrency tests.

---

## Phase E10: Observability & Operations

### E10a. Metrics

| Test | Description |
|------|-------------|
| `test_metrics_endpoint` | `--enable-metrics` → `/metrics` returns Prometheus format |
| `test_metrics_request_count` | After N requests, `vllm_requests_total` ≥ N |
| `test_metrics_prompt_tokens` | After request, `vllm_prompt_tokens_total` > 0 |

### E10b. Health & model info

| Test | Description |
|------|-------------|
| `test_health_before_request` | `/health` → "ok" immediately after startup |
| `test_models_shows_correct_name` | `/v1/models` → model ID matches served model |
| `test_models_has_max_model_len` | `/v1/models` → max_model_len is populated |

**Deliverables**: ~6 observability tests.

---

## Phase E11: CLI & Configuration

### E11a. CLI flags

| Test | Description |
|------|-------------|
| `test_cli_port_flag` | `--port 9999` → server binds to 9999 |
| `test_cli_host_flag` | `--host 127.0.0.1` → only localhost accessible |
| `test_cli_max_model_len` | `--max-model-len 512` → model reports 512 |
| `test_cli_block_size` | `--block-size 8` → server starts (different block size) |
| `test_cli_gpu_memory_utilization` | `--gpu-memory-utilization 0.5` → server starts |
| `test_cli_log_level` | `--log-level debug` → server starts (more verbose) |
| `test_cli_dtype_auto` | `--dtype auto` (default) → server starts |

### E11b. Error cases

| Test | Description |
|------|-------------|
| `test_cli_no_model` | No model arg → process exits with error |
| `test_cli_invalid_model` | Nonexistent model → process exits with error |
| `test_cli_invalid_port` | `--port 99999` → process exits with error |

**Deliverables**: ~10 CLI tests.

---

## Phase E12: Embedding — DONE

Test file: `e_embedding.rs`

Tests decoder models as embedding models via the `/v1/embeddings` endpoint. Supports last-token, mean, and CLS pooling strategies with L2 normalization. Pooling strategy is configurable via `--pooling-strategy` and auto-detected from `1_Pooling/config.json`.

| Test | Model | Description |
|------|-------|-------------|
| `test_embedding_single_string` | SmolLM-135M-4bit | Single string → 1 embedding, usage.prompt_tokens > 0 |
| `test_embedding_multiple_strings` | SmolLM-135M-4bit | 3 strings → 3 embedding objects, same dimensions |
| `test_embedding_dimensions` | SmolLM-135M-4bit | `dimensions: 32` → embedding truncated to 32 floats |
| `test_embedding_normalized` | SmolLM-135M-4bit | Embedding L2 norm ≈ 1.0 |
| `test_embedding_different_inputs` | SmolLM-135M-4bit | Two different strings → cosine similarity < 1.0 |
| `test_embedding_mean_pooling` | SmolLM-135M-4bit | `--pooling-strategy mean` → valid normalized embedding |
| `test_embedding_cls_pooling` | SmolLM-135M-4bit | `--pooling-strategy cls` → valid normalized embedding |
| `test_embedding_mean_vs_last_differ` | SmolLM-135M-4bit | Mean and last pooling produce different embeddings (cosine < 1.0) |
| `test_embedding_qwen2` | Qwen2.5-0.5B-4bit | Qwen2 arch produces valid normalized embeddings |
| `test_embedding_llama3` | Llama-3.2-1B-4bit | LLaMA3 arch produces valid normalized embeddings |

**Deliverables**: 10 E2E tests (all implemented).

---

## Phase E13: LoRA Adapters — DONE

Test file: `e_lora.rs`

Tests single-adapter LoRA support. Creates a synthetic LoRA adapter (random A/B weights targeting q_proj + v_proj) at test time — no external adapter download needed. Runs on both Candle CPU and MLX backends depending on `--features metal`.

| Test | Model | Description |
|------|-------|-------------|
| `test_lora_synthetic_server_starts` | SmolLM-135M-F16 | Server with synthetic LoRA adapter starts, /health + /v1/models work |
| `test_lora_synthetic_chat` | SmolLM-135M-F16 | Chat completion with LoRA adapter returns non-empty response |
| `test_lora_synthetic_completion` | SmolLM-135M-F16 | Text completion with LoRA adapter returns non-empty text |
| `test_lora_synthetic_output_differs` | SmolLM-135M-F16 | Same prompt with vs without LoRA produces different outputs |

**Deliverables**: 4 E2E tests (all implemented). Self-contained — generates synthetic adapter in tempdir.

---

## Phase E14: GPTQ Quantization — DONE

Test file: `e_gptq.rs`

Tests GPTQ INT4 quantized model loading and inference. GPTQ is the most popular GPU quantization format on HuggingFace. Supports both candle (CPU dequantize-per-forward) and MLX (dequantize-at-load-time on Metal) backends.

| Test | Model | Description |
|------|-------|-------------|
| `test_gptq_qwen2_server_starts` | Qwen2.5-0.5B-Instruct-GPTQ-Int4 | Server starts, /health + /v1/models work |
| `test_gptq_qwen2_chat_basic` | Qwen2.5-0.5B-Instruct-GPTQ-Int4 | Chat completion returns coherent text |
| `test_gptq_qwen2_completion_basic` | Qwen2.5-0.5B-Instruct-GPTQ-Int4 | Text completion returns non-empty text |
| `test_gptq_qwen2_max_tokens` | Qwen2.5-0.5B-Instruct-GPTQ-Int4 | max_tokens=5 → completion_tokens ≤ 5 |

Run commands:
```bash
# MLX backend (fast, ~4s):
cargo test -p vllm-e2e --features e2e,metal --test e_gptq --release -- --ignored --test-threads=1

# Candle CPU backend (slower, ~120s):
cargo test -p vllm-e2e --features e2e --test e_gptq --release -- --ignored --test-threads=1
```

**Deliverables**: 4 E2E tests (all implemented). Uses official Qwen GPTQ model (~459 MB).

---

## Phase E14b: AWQ Quantization — DONE

Test file: `e_awq.rs`

Tests AWQ INT4 quantized model loading and inference. AWQ (Activation-aware Weight Quantization) is widely used on HuggingFace (`TheBloke/*-AWQ`, `casperhansen/*-awq`). Supports both candle (CPU dequantize-per-forward) and MLX (dequantize-at-load-time on Metal) backends.

| Test | Model | Description |
|------|-------|-------------|
| `test_awq_qwen2_server_starts` | Qwen2.5-0.5B-Instruct-AWQ | Server starts, /health + /v1/models work |
| `test_awq_qwen2_chat_basic` | Qwen2.5-0.5B-Instruct-AWQ | Chat completion returns coherent text |
| `test_awq_qwen2_completion_basic` | Qwen2.5-0.5B-Instruct-AWQ | Text completion returns non-empty text |
| `test_awq_qwen2_max_tokens` | Qwen2.5-0.5B-Instruct-AWQ | max_tokens=5 → completion_tokens ≤ 5 |

Run commands:
```bash
# MLX backend (fast, ~10s):
cargo test -p vllm-e2e --features e2e,metal --test e_awq --release -- --ignored --test-threads=1

# Candle CPU backend (slower):
cargo test -p vllm-e2e --features e2e --test e_awq --release -- --ignored --test-threads=1
```

**Deliverables**: 4 E2E tests (all implemented). Uses official Qwen AWQ model (~393 MB).

---

## Phase E15: Offline Batch LLM API — DONE

Test file: `e_llm_api.rs`

Tests the programmatic `LLM` struct — offline batch inference without an HTTP server. `LLM::new()` initializes the full stack (worker → executor → engine) and provides synchronous `generate()` / `chat()` methods. This is the foundation for the `vllm-pyo3` Python binding layer.

| Test | Model | Description |
|------|-------|-------------|
| `test_llm_generate_basic` | SmolLM-135M-4bit | Single prompt → non-empty output, model name correct |
| `test_llm_generate_multiple_prompts` | SmolLM-135M-4bit | Two prompts → two outputs, prompt text preserved |
| `test_llm_generate_max_tokens` | SmolLM-135M-4bit | max_tokens=3 → finish_reason="length" |
| `test_llm_chat_basic` | SmolLM-135M-4bit | Single user message → non-empty chat response |
| `test_llm_chat_with_system_message` | SmolLM-135M-4bit | System + user messages → non-empty response |
| `test_llm_builder` | SmolLM-135M-4bit | `LLM::builder().max_model_len(512).build()` → generates correctly |

Run command:
```bash
cargo test -p vllm-e2e --features e2e,metal --test e_llm_api -- --ignored --test-threads=1
```

**Deliverables**: 6 E2E tests (all implemented). Tests use `#[test]` (not `#[tokio::test]`) since LLM owns its own runtime.

---

## Phase E16: Batch Processing — DONE

Test file: `e_batch.rs`

Tests the `vllm batch` offline batch processing command. Reads a JSONL input file, processes requests through the engine (no HTTP server), and writes a JSONL output file. Tests call the batch runner logic directly using `initialize_stack()` + `AsyncEngine` methods. Supports `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings` endpoints.

| Test | Model | Description |
|------|-------|-------------|
| `test_batch_chat_completions` | SmolLM-135M-4bit | 3 chat requests → 3 outputs with correct custom_ids and response bodies |
| `test_batch_completions` | SmolLM-135M-4bit | 2 text completion requests → non-empty completion text |
| `test_batch_embeddings` | SmolLM-135M-4bit | 2 embedding requests → non-empty embedding vectors |
| `test_batch_mixed_endpoints` | SmolLM-135M-4bit | Chat + completion + embedding in single batch → each routed correctly |
| `test_batch_invalid_url` | SmolLM-135M-4bit | Unsupported URL → error in output (not a crash), good request still succeeds |
| `test_batch_malformed_body` | SmolLM-135M-4bit | Missing required fields → error in output (not a crash) |

Run command:
```bash
cargo test -p vllm-e2e --features e2e,metal --release --test e_batch -- --ignored --test-threads=1
```

**Deliverables**: 6 E2E tests (all implemented). No HTTP server — tests exercise the engine directly.

---

## Test Matrix Summary

| Phase | Tests | Models Used | Run Frequency | Estimated Time |
|-------|------:|-------------|---------------|----------------|
| E0. Infrastructure | 0 | — | — | — |
| E1. Basic Serving | ~30 (+1 sync sched) | All tiers | PR / nightly / weekly | 5 min (Tier 1+2) |
| E2. Chat Completions | ~30 | SmolLM-135M | Every PR | 3 min |
| E3. Streaming | ~14 | SmolLM-135M | Every PR | 2 min |
| E4. Text Completions | ~9 | SmolLM-135M | Every PR | 1 min |
| E5. Tool Calling | ~23 (5 done) | SmolLM-135M / Llama-3.2-1B | Every PR | 3 min |
| E6. Structured Output | ~14 | SmolLM-135M | Every PR | 2 min |
| E7. Sampling Features | ~10 | SmolLM-135M | Every PR | 2 min |
| E8. Multi-Architecture | ~18 | All tiers | PR / nightly / weekly | 5 min (Tier 1+2) |
| E9. Concurrency | ~6 | SmolLM-135M | Every PR | 2 min |
| E10. Observability | ~6 | SmolLM-135M | Every PR | 1 min |
| E11. CLI & Config | ~10 | SmolLM-135M | Every PR | 2 min |
| E12. Embedding | 10 (done) | SmolLM / Qwen2 / Llama3 | Every PR | 2 min |
| E13. LoRA Adapters | 4 (done) | SmolLM-135M-F16 | Every PR | <1 min |
| E14. GPTQ Quantization | 4 (done) | Qwen2.5-0.5B-GPTQ-Int4 | Every PR | <1 min (MLX) |
| E14b. AWQ Quantization | 4 (done) | Qwen2.5-0.5B-AWQ | Every PR | <1 min (MLX) |
| E15. Offline Batch LLM API | 6 (done) | SmolLM-135M-4bit | Every PR | <1 min |
| E16. Batch Processing | 6 (done) | SmolLM-135M-4bit | Every PR | <1 min |
| E17. Multimodal VLM | 8 (done) | Gemma3-4B (MLX + Candle) | Nightly / Weekly | ~2 min |
| **Total** | **~204** | | | **~33 min** |

### CI Tiers

| Tier | Trigger | Models | Download Size | Tests |
|------|---------|--------|---------------|-------|
| **PR** | Every pull request | SmolLM-135M-4bit, Qwen2.5-0.5B-4bit, Qwen3-0.6B-4bit, Llama-3.2-1B-4bit, Gemma3-270M-4bit | ~2.3 GB | ~124 |
| **Nightly** | Scheduled (daily) | + Gemma2-2B-4bit, Phi-3.5-mini-4bit | +3.5 GB | ~130 |
| **Weekly** | Scheduled (weekly) | + Mistral-7B-4bit, DeepSeek-V2-Lite-4bit | +12 GB | ~156 |

---

## Model Architecture Coverage Matrix

| Architecture | HF Arch Name | Test Model | Size | Tier | Float | Quantized |
|---|---|---|---|---|---|---|
| LLaMA (generic) | LlamaForCausalLM | SmolLM-135M-Instruct-4bit | 76 MB | PR | SmolLM2-135M-Instruct (255 MB) | Yes |
| LLaMA 3 | LlamaForCausalLM | Llama-3.2-1B-Instruct-4bit | 680 MB | PR | — | Yes |
| Qwen2 | Qwen2ForCausalLM | Qwen2.5-0.5B-Instruct-4bit | 276 MB | PR | — | Yes |
| Qwen3 | Qwen3ForCausalLM | Qwen3-0.6B-4bit | 335 MB | PR | — | Yes |
| Gemma3 | Gemma3ForCausalLM | gemma-3-270m-it-qat-4bit | 900 MB | PR | — | Yes |
| Gemma2 | Gemma2ForCausalLM | gemma-2-2b-it-4bit | 1.4 GB | Nightly | — | Yes |
| Phi-3 | Phi3ForCausalLM | Phi-3.5-mini-instruct-4bit | 2.15 GB | Nightly | — | Yes |
| Phi-4 | Phi3ForCausalLM (LongRoPE) | Unsloth-Phi-4-mini-instruct-4bit | 2.3 GB | Nightly | — | Yes |
| Mistral | MistralForCausalLM | Mistral-7B-Instruct-v0.3-4bit | 3.8 GB | Weekly | — | Yes |
| DeepSeek V2 | DeepseekV2ForCausalLM | DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx | 8.2 GB | Weekly | — | Yes |
| Qwen3 MoE | Qwen3MoeForCausalLM | Qwen3-MOE-4x0.6B-2.4B-mlx-4Bit | ~1.5 GB | Nightly | — | Yes |
| Mixtral MoE | MixtralForCausalLM | Mixtral-SlimOrca-8x7B-3bit | ~18 GB | Manual | — | Yes |
| Command R | CohereForCausalLM | c4ai-command-r-08-2024-4bit | 16.9 GB | Manual | — | Yes |
| Gemma v1 | GemmaForCausalLM | (deferred — 2B model at 2 GB) | — | — | — | — |

| Gemma3 VLM (MLX) | Gemma3ForConditionalGeneration | gemma-3-4b-it-qat-3bit | 2.8 GB | Nightly | — | Yes |
| Gemma3 VLM (Candle) | Gemma3ForConditionalGeneration | google/gemma-3-4b-it | 8 GB | Weekly | Yes (BF16) | — |
|
| GPTQ Qwen2 | Qwen2ForCausalLM | Qwen2.5-0.5B-Instruct-GPTQ-Int4 | 459 MB | PR | — | Yes (GPTQ INT4) |
| AWQ Qwen2 | Qwen2ForCausalLM | Qwen2.5-0.5B-Instruct-AWQ | 393 MB | PR | — | Yes (AWQ INT4) |

**Note on Command R**: The smallest `CohereForCausalLM` is 35B (16.9 GB). The 7B variant uses `Cohere2ForCausalLM` which is a different architecture not yet implemented. Command R E2E tests are manual-only until either (a) a smaller CohereForCausalLM model appears, or (b) we implement Cohere2ForCausalLM.

**Note on Mixtral**: The smallest quantized Mixtral 8x7B is ~18 GB (3-bit). No smaller MixtralForCausalLM models exist. Mixtral E2E tests are manual-only. The architecture is covered by a synthetic weights unit test (`test_mixtral_model_from_weights`) which exercises the full load→forward path with tiny dimensions.

---

## Implementation Order

| Priority | Phase | Rationale |
|----------|-------|-----------|
| 1 | **E0** (Infrastructure) | Must exist before any tests |
| 2 | **E1a** (Basic serving, Tier 1+2) | Proves each arch works at all |
| 3 | **E2** (Chat completions) | Most-used endpoint, validates core functionality |
| 4 | **E3** (Streaming) | Critical for real-world usage |
| 5 | **E5** (Tool calling) | Key differentiator, complex parsing logic |
| 6 | **E6** (Structured output) | New feature, needs validation |
| 7 | **E4** (Text completions) | Important but simpler endpoint |
| 8 | **E7** (Sampling features) | Validates parameter handling |
| 9 | **E9** (Concurrency) | Important for production readiness |
| 10 | **E8** (Multi-architecture, nightly/weekly tiers) | Broader coverage |
| 11 | **E10** (Observability) | Lower priority |
| 12 | **E11** (CLI & config) | Lower priority |
| 13 | **E1b** (Float16 comparison) | Nice to have |

---

## Phase E17: Multimodal / Vision-Language — DONE (verified 2026-03-02)

Test files: `e_multimodal.rs` (MLX), `e_gemma3_vlm.rs` (Candle)

Tests Gemma3ForConditionalGeneration — a VLM with SigLIP vision encoder + AvgPool2d → GemmaRMSNorm → projection + Gemma3 text backbone. Supports both float and quantized (4-bit) models on MLX. Weight prefixes: `vision_tower.vision_model.*`, `multi_modal_projector.*`, `language_model.model.*`. Unit tests in `gemma3_mm.rs` validate weight names against real HF checkpoints to catch prefix mismatches without downloading models.

### MLX path (`e_multimodal.rs` — Tier 3, nightly)

| Test | Model | Description |
|------|-------|-------------|
| `test_vlm_gemma3_server_starts` | gemma-3-4b-it-qat-3bit | Server starts, /health + /v1/models work |
| `test_vlm_gemma3_text_only_chat` | gemma-3-4b-it-qat-3bit | Text-only chat through VLM backbone |
| `test_vlm_gemma3_image_chat` | gemma-3-4b-it-qat-3bit | Image + text chat completion |
| `test_vlm_gemma3_image_stream` | gemma-3-4b-it-qat-3bit | Streaming image + text chat |
| `test_vlm_gemma3_image_max_tokens` | gemma-3-4b-it-qat-3bit | max_tokens respected with image input |

### Candle path (`e_gemma3_vlm.rs` — Tier 4, weekly)

| Test | Model | Description |
|------|-------|-------------|
| `test_gemma3_vlm_candle_server_starts` | google/gemma-3-4b-it | Server starts with SafeTensors BF16 model |
| `test_gemma3_vlm_candle_text_only_chat` | google/gemma-3-4b-it | Text-only chat through VLM backbone |
| `test_gemma3_vlm_candle_max_tokens` | google/gemma-3-4b-it | max_tokens respected |

Run commands:
```bash
# MLX backend (~2.8 GB model, use --release for Metal performance):
cargo test -p vllm-e2e --features e2e,metal --release --test e_multimodal -- --ignored --test-threads=1

# Candle backend (~8 GB BF16 model):
cargo test -p vllm-e2e --features e2e,metal --release --test e_gemma3_vlm -- --ignored --test-threads=1
```

**Deliverables**: 8 E2E tests (all implemented and verified). Covers both MLX quantized and Candle BF16 paths. Candle tests verified 2026-03-02 (3/3 passed in 7.4s with --release).

---

## Future Extensions (not in initial scope)

- **Candle backend E2E**: Same test suite but with `--features candle-metal` or CPU-only, using GGUF models
- **CUDA backend E2E**: Run on Linux CI with GPU, test CUDA worker path
- **Performance regression tests**: Track TTFT, ITL, throughput across commits
- **Python vLLM comparison tests**: Same prompts to Python vLLM and Rust, compare output quality
- **LoRA E2E**: When adapter support is added
- **Long-context E2E**: Test with 32K+ token contexts
- **Memory pressure tests**: Run until OOM, verify graceful degradation
