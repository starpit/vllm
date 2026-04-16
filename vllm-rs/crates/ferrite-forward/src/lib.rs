// SPDX-License-Identifier: Apache-2.0
//! Consumer-facing crate for the `#[forward]` attribute macro.
//!
//! Re-exports the proc-macro and exposes runtime support types
//! the generated code depends on: most importantly [`ForwardCtx`],
//! the ambient-args bundle the emitted forward fn takes.

pub use ferrite_forward_macro::forward;

pub mod cpu_golden;

#[cfg(feature = "cuda")]
mod ctx {
    use ferrite_cuda_core::tensor::TensorView;
    use ferrite_kernels::kv_cache::KvCachePool;
    use ferrite_kernels::rotary::RotaryCache;

    /// Ambient runtime args the emitted forward fn needs. The
    /// caller builds a `ForwardCtx` per forward call and passes
    /// it in. Fields are the union of what any ported kernel
    /// needs at invocation time; new kernels can reference new
    /// fields, which is the extension point.
    pub struct ForwardCtx<'a> {
        pub input_ids: TensorView<'a>,
        pub positions: TensorView<'a>,
        pub slot_mapping: TensorView<'a>,
        pub cu_seqlens_q: TensorView<'a>,
        pub seqused_k: TensorView<'a>,
        pub block_table: TensorView<'a>,
        pub max_seqlen_q: usize,
        pub max_seqlen_k: usize,
        pub kv_cache: &'a KvCachePool,
        pub rotary: &'a RotaryCache,
    }
}
#[cfg(feature = "cuda")]
pub use ctx::ForwardCtx;

// ── Cross-arch dispatcher (cuda only) ────────────────────────────
//
// Every `#[forward] fn <arch>()` macro invocation auto-emits an
// `impl FerriteWeights for Weights` + an `inventory::submit!`
// registration, so `try_load` discovers every compiled arch
// without a hand-written central list. Adding a new arch touches
// only its source file plus a single `pub mod <arch>;` in the
// calling crate's lib.rs (Rust module system requirement).

#[cfg(feature = "cuda")]
mod dispatcher {
    use ferrite_cuda_core::CUstream;
    use ferrite_cuda_core::alloc::OwnedTensor;
    use ferrite_cuda_core::device::GpuDevice;
    use ferrite_cuda_core::weights::GpuWeights;

    use super::ForwardCtx;

    /// Arch-agnostic handle for a ferrite-loaded model. Every
    /// `#[forward] fn <arch>()` emits an `impl FerriteWeights` for
    /// its per-arch `Weights` type; callers hold
    /// `Box<dyn FerriteWeights>` and never need to know which arch
    /// they got.
    pub trait FerriteWeights: Send + Sync {
        fn arch_name(&self) -> &'static str;
        fn num_hidden_layers(&self) -> u64;
        fn hidden_size(&self) -> u64;
        fn intermediate_size(&self) -> u64;
        fn num_attention_heads(&self) -> u64;
        fn num_key_value_heads(&self) -> u64;
        fn head_dim(&self) -> u64;
        fn vocab_size(&self) -> u64;

        /// # Safety
        /// All tensors in `ctx` must be valid GPU memory; `device`
        /// must be the live CUDA device. Same invariants as each
        /// per-arch `forward`.
        unsafe fn forward(
            &self,
            ctx: &ForwardCtx,
            device: &mut GpuDevice,
            num_tokens: u64,
        ) -> OwnedTensor;

        /// Backbone-only forward (skips lm_head). Same safety.
        unsafe fn forward_backbone(
            &self,
            ctx: &ForwardCtx,
            device: &mut GpuDevice,
            num_tokens: u64,
        ) -> OwnedTensor;
    }

    /// One registration per `#[forward] fn <arch>()`. The macro
    /// emits an `inventory::submit!` block that constructs this.
    pub struct FerriteArchRegistration {
        /// The arch's identifier — `"llama"`, `"qwen2"`, … — from
        /// the carrier fn name. Used in logs.
        pub arch_name: &'static str,
        /// HF `architectures` strings this arch claims to handle.
        /// Harvested by the macro from the union of every compiled
        /// model's `config.json` `architectures: [..]` field.
        pub hf_arches: &'static [&'static str],
        /// Try to load the arch's compiled variants. Internally
        /// iterates per-variant fingerprint sniffs; returns
        /// `Err(..)` if arch was matched but no variant matched
        /// the live `GpuWeights`.
        pub try_load: fn(&mut GpuWeights, CUstream) -> ::anyhow::Result<Box<dyn FerriteWeights>>,
    }

    inventory::collect!(FerriteArchRegistration);

    /// Top-level ferrite loader. Walks every `#[forward]`-registered
    /// arch; the first whose `hf_arches` list contains `arch_hint`
    /// wins and loads. Returns `Ok(None)` if no registered arch
    /// claims `arch_hint` — caller falls through to hand-written.
    pub fn try_load(
        gw: &mut GpuWeights,
        stream: CUstream,
        arch_hint: &str,
    ) -> ::anyhow::Result<Option<Box<dyn FerriteWeights>>> {
        for reg in inventory::iter::<FerriteArchRegistration>() {
            if reg.hf_arches.iter().any(|a| *a == arch_hint) {
                return (reg.try_load)(gw, stream).map(Some);
            }
        }
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
pub use dispatcher::{FerriteArchRegistration, FerriteWeights, try_load};

/// Re-export `inventory` so the `#[forward]`-macro-emitted
/// `inventory::submit!` block resolves without the consuming crate
/// having to add its own `inventory` dep.
#[cfg(feature = "cuda")]
pub use inventory;
