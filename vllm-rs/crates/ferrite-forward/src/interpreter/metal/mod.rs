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
pub mod model_meta;
#[cfg(feature = "metal")]
pub mod pool;
#[cfg(feature = "metal")]
pub mod runtime;
#[cfg(feature = "metal")]
pub mod worker;

pub use lowered::{
    Binding, DispatchShape, KernelId, LoweredCommand, LoweredMetalTape, LoweringError,
    RuntimeBindingKind, WeightBundleKind, WeightTensor,
};
pub use lowering::{lower, lower_pair};
pub use pipelines::{
    constants_for, KernelExtras, PipelineLookupError, SpecializedPipelines,
};

#[cfg(feature = "metal")]
pub use forward::{ForwardError, ForwardInputs};
#[cfg(feature = "metal")]
pub use model_meta::{BufferRef, MetalModelMeta};
#[cfg(feature = "metal")]
pub use pool::{MetalWorkerPool, PooledWorker, RuntimeFactory, WorkerGuard};
#[cfg(feature = "metal")]
pub use runtime::RuntimeBindings;
#[cfg(feature = "metal")]
pub use worker::{ArenaLayout, BoundBuffer, BucketBaking, BucketStep, MetalWorker, WorkerError};
