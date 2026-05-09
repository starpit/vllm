// SPDX-License-Identifier: Apache-2.0
//! Metal interpreter — Phase 5.A surface.
//!
//! Public API at this phase is the [`lower`] function that translates
//! one bucket's `Instruction<W>` tape into a buffer-pointer-free
//! [`LoweredMetalTape`]. Worker pool / pipeline cache / `forward()`
//! land in subsequent sub-phases (5.B–5.G) per
//! `FERRITE_METAL_ARCHITECTURE.md`.

pub mod lowered;
pub mod lowering;
pub mod pipelines;

#[cfg(feature = "metal")]
pub mod forward;
#[cfg(feature = "metal")]
pub mod pool;
#[cfg(feature = "metal")]
pub mod runtime;
#[cfg(feature = "metal")]
pub mod worker;

pub use lowered::{
    Binding, DispatchShape, KernelId, LoweredCommand, LoweredMetalTape, LoweringError, MetalDtype,
    RuntimeBindingKind, WeightBundleKind, WeightTensor,
};
pub use lowering::{lower, lower_pair};
pub use pipelines::{PipelineLookupError, SpecializedPipelines, constants_for};

#[cfg(feature = "metal")]
pub use forward::{ForwardError, ForwardInputs};
#[cfg(feature = "metal")]
pub use pool::{
    MetalBucketSpec, MetalWorkerPool, PoolBuildError, PooledWorker, RuntimeFactory, WorkerGuard,
};
#[cfg(feature = "metal")]
pub use runtime::RuntimeBindings;
#[cfg(feature = "metal")]
pub use worker::{ArenaLayout, BoundBuffer, BucketBaking, BucketStep, MetalWorker, WorkerError};

// Re-export `metal::Device` so per-model crates whose macro expansion
// emits a `metal_pool(...)` constructor signature can name the type
// without taking a direct `ferrite-metal-kernels` dep. Per-arch crates
// already depend on `ferrite-forward`, so all macro-emitted paths
// route through this crate.
#[cfg(feature = "metal")]
#[doc(hidden)]
pub mod __re {
    pub use ::ferrite_metal_kernels::metal::{Buffer, CommandQueue, Device, MTLResourceOptions};
}
