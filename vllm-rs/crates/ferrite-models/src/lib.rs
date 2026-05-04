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

#[cfg(feature = "arch-commandr")]
extern crate ferrite_model_commandr as _keep_commandr;
#[cfg(feature = "arch-deepseek-v2")]
extern crate ferrite_model_deepseek_v2 as _keep_deepseek_v2;
#[cfg(feature = "arch-deepseek-v3")]
extern crate ferrite_model_deepseek_v3 as _keep_deepseek_v3;
#[cfg(feature = "arch-deepseek-v3-flat")]
extern crate ferrite_model_deepseek_v3_flat as _keep_deepseek_v3_flat;
#[cfg(feature = "arch-gemma2")]
extern crate ferrite_model_gemma2 as _keep_gemma2;
#[cfg(feature = "arch-gemma3")]
extern crate ferrite_model_gemma3 as _keep_gemma3;
#[cfg(feature = "arch-gemma3-mm")]
extern crate ferrite_model_gemma3_mm as _keep_gemma3_mm;
#[cfg(feature = "arch-granite")]
extern crate ferrite_model_granite as _keep_granite;
#[cfg(feature = "arch-llama")]
extern crate ferrite_model_llama as _keep_llama;
#[cfg(feature = "arch-mistral")]
extern crate ferrite_model_mistral as _keep_mistral;
#[cfg(feature = "arch-mixtral")]
extern crate ferrite_model_mixtral as _keep_mixtral;
#[cfg(feature = "arch-modernbert")]
extern crate ferrite_model_modernbert as _keep_modernbert;
#[cfg(feature = "arch-phi3")]
extern crate ferrite_model_phi3 as _keep_phi3;
#[cfg(feature = "arch-qwen2")]
extern crate ferrite_model_qwen2 as _keep_qwen2;
#[cfg(feature = "arch-qwen2-5-vl")]
extern crate ferrite_model_qwen2_5_vl as _keep_qwen2_5_vl;
#[cfg(feature = "arch-qwen2-moe")]
extern crate ferrite_model_qwen2_moe as _keep_qwen2_moe;
#[cfg(feature = "arch-qwen2-vl")]
extern crate ferrite_model_qwen2_vl as _keep_qwen2_vl;
#[cfg(feature = "arch-qwen3")]
extern crate ferrite_model_qwen3 as _keep_qwen3;
#[cfg(feature = "arch-qwen3-moe")]
extern crate ferrite_model_qwen3_moe as _keep_qwen3_moe;
#[cfg(feature = "arch-qwen3-next")]
extern crate ferrite_model_qwen3_next as _keep_qwen3_next;

#[cfg(feature = "arch-commandr")]
pub use ferrite_model_commandr as commandr;
#[cfg(feature = "arch-deepseek-v2")]
pub use ferrite_model_deepseek_v2 as deepseek_v2;
#[cfg(feature = "arch-deepseek-v3")]
pub use ferrite_model_deepseek_v3 as deepseek_v3;
#[cfg(feature = "arch-deepseek-v3-flat")]
pub use ferrite_model_deepseek_v3_flat as deepseek_v3_flat;
#[cfg(feature = "arch-gemma2")]
pub use ferrite_model_gemma2 as gemma2;
#[cfg(feature = "arch-gemma3")]
pub use ferrite_model_gemma3 as gemma3;
#[cfg(feature = "arch-gemma3-mm")]
pub use ferrite_model_gemma3_mm as gemma3_mm;
#[cfg(feature = "arch-granite")]
pub use ferrite_model_granite as granite;
#[cfg(feature = "arch-llama")]
pub use ferrite_model_llama as llama;
#[cfg(feature = "arch-mistral")]
pub use ferrite_model_mistral as mistral;
#[cfg(feature = "arch-mixtral")]
pub use ferrite_model_mixtral as mixtral;
#[cfg(feature = "arch-modernbert")]
pub use ferrite_model_modernbert as modernbert;
#[cfg(feature = "arch-phi3")]
pub use ferrite_model_phi3 as phi3;
#[cfg(feature = "arch-qwen2")]
pub use ferrite_model_qwen2 as qwen2;
#[cfg(feature = "arch-qwen2-5-vl")]
pub use ferrite_model_qwen2_5_vl as qwen2_5_vl;
#[cfg(feature = "arch-qwen2-moe")]
pub use ferrite_model_qwen2_moe as qwen2_moe;
#[cfg(feature = "arch-qwen2-vl")]
pub use ferrite_model_qwen2_vl as qwen2_vl;
#[cfg(feature = "arch-qwen3")]
pub use ferrite_model_qwen3 as qwen3;
#[cfg(feature = "arch-qwen3-moe")]
pub use ferrite_model_qwen3_moe as qwen3_moe;
#[cfg(feature = "arch-qwen3-next")]
pub use ferrite_model_qwen3_next as qwen3_next;
