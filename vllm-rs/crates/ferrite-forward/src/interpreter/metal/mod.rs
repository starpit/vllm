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

pub use lowered::{
    Binding, DispatchShape, KernelId, LoweredCommand, LoweredMetalTape, LoweringError,
    RuntimeBindingKind, WeightBundleKind, WeightTensor,
};
pub use lowering::lower;
