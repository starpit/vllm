// SPDX-License-Identifier: Apache-2.0
//! Ferrite model architectures — umbrella crate. Each architecture
//! lives in its own `ferrite-model-<arch>` crate so cargo can compile
//! the `#[forward]` invocations in parallel. This umbrella pulls them
//! all in and re-exports their top-level modules so downstream
//! consumers can keep using `ferrite_models::<arch>::…` paths.
//!
//! The `extern crate … as _` lines force the linker to keep each
//! per-arch crate even if nothing in the umbrella's public API
//! references a symbol from it — the `#[forward]`-emitted
//! `inventory::submit!` registrations must end up in the final binary.

extern crate ferrite_model_commandr as _keep_commandr;
extern crate ferrite_model_deepseek_v2 as _keep_deepseek_v2;
extern crate ferrite_model_deepseek_v3 as _keep_deepseek_v3;
extern crate ferrite_model_gemma2 as _keep_gemma2;
extern crate ferrite_model_gemma3 as _keep_gemma3;
extern crate ferrite_model_granite as _keep_granite;
extern crate ferrite_model_llama as _keep_llama;
extern crate ferrite_model_mistral as _keep_mistral;
extern crate ferrite_model_phi3 as _keep_phi3;
extern crate ferrite_model_qwen2 as _keep_qwen2;
extern crate ferrite_model_qwen3 as _keep_qwen3;

pub use ferrite_model_commandr as commandr;
pub use ferrite_model_deepseek_v2 as deepseek_v2;
pub use ferrite_model_deepseek_v3 as deepseek_v3;
pub use ferrite_model_gemma2 as gemma2;
pub use ferrite_model_gemma3 as gemma3;
pub use ferrite_model_granite as granite;
pub use ferrite_model_llama as llama;
pub use ferrite_model_mistral as mistral;
pub use ferrite_model_phi3 as phi3;
pub use ferrite_model_qwen2 as qwen2;
pub use ferrite_model_qwen3 as qwen3;
