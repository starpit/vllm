// SPDX-License-Identifier: Apache-2.0
//! Metal interpreter — Phase 5.A surface.
//!
//! Public API at this phase is the [`lower`] function that translates
//! one bucket's `Instruction<W>` tape into a buffer-pointer-free
//! [`LoweredMetalTape`]. Worker pool / pipeline cache / `forward()`
//! land in subsequent sub-phases (5.B–5.G) per
//! `FERRITE_METAL_ARCHITECTURE.md`.

pub mod ids;
pub mod kernel_bindings;
pub mod kernel_constants;
pub mod kernel_identity;
pub mod lowered;
pub mod lowering;
pub mod pipelines;

#[cfg(feature = "metal")]
pub mod forward;
#[cfg(feature = "metal")]
pub mod mtl4;
#[cfg(feature = "metal")]
pub mod pool;
#[cfg(feature = "metal")]
pub mod runtime;
#[cfg(feature = "metal")]
pub mod worker;

/// Re-export of `ferrite_metal_kernels::quantized::ScaleDtype` so the
/// macro-emitted `impl CanonicalParams for Weights` block can name it
/// without per-arch crates pulling `ferrite-metal-kernels` directly.
/// Mirrors the [`MetalDtype`] re-export above.
#[cfg(feature = "metal")]
pub use ferrite_metal_kernels::quantized::ScaleDtype;
pub use ids::{
    ArenaSlotIdx, BindingIdx, BucketM, ConstSlot, LayerId, LogicalBlockIdx, NumTokens,
    PhysicalBlockIdx, QTokenIdx, SeqIdx, SlotInBlock,
};
pub use lowered::{
    Binding, DispatchShape, KernelId, LoweredCommand, LoweredMetalTape, LoweringError, MetalDtype,
    RuntimeBindingKind, WeightBundleKind, WeightTensor,
};
pub use lowering::{lower, lower_pair};
pub use pipelines::{PipelineLookupError, SpecializedPipelines};

/// Reactive (chunked) KV pool granularity — re-exported single source
/// of truth so the worker (chunk-pool sizing) and the lowering
/// (chunk-table `[[function_constant]]`) share the exact value the
/// macro-generated `SynthPreAttn` bakes. See
/// [`ferrite_fusion_synth::BLOCKS_PER_CHUNK`].
#[cfg(feature = "metal")]
pub use ferrite_fusion_synth::BLOCKS_PER_CHUNK;
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

// Centralized type aliases for the objc2-metal `Retained` wrapper
// types so per-model crates whose macro expansion emits e.g.
// `metal_pool(...)` constructor signatures can name `Buffer` / `Device`
// without hand-spelling `Retained<ProtocolObject<dyn MTLBuffer>>` at
// each call site. Per-arch crates already depend on `ferrite-forward`,
// so all macro-emitted paths route through this crate.
#[cfg(feature = "metal")]
#[doc(hidden)]
pub mod __re {
    use ::objc2::rc::Retained;
    use ::objc2::runtime::ProtocolObject;
    pub use ::objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState,
        MTLDataType, MTLDevice, MTLFunction, MTLFunctionConstantValues, MTLLibrary,
        MTLPipelineOption, MTLResourceOptions, MTLSize,
    };
    // MTL4 surfaces re-exported for the Scope-A side-by-side path
    // (see `FERRITE_METAL_MTL4_MIGRATION.md`). All four are optional
    // at runtime: `MetalWorkerPool::new` probes
    // `device.newMTL4CommandQueue()` once and stores the result.
    pub use ::objc2_metal::{
        MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
        MTL4CommandQueue, MTL4ComputeCommandEncoder, MTL4CounterHeap, MTL4CounterHeapDescriptor,
        MTL4CounterHeapType, MTL4TimestampGranularity, MTLEvent, MTLSharedEvent,
    };
    pub type Mtl4CounterHeap = Retained<ProtocolObject<dyn MTL4CounterHeap>>;
    pub type Mtl4Queue = Retained<ProtocolObject<dyn MTL4CommandQueue>>;
    pub type Mtl4Allocator = Retained<ProtocolObject<dyn MTL4CommandAllocator>>;
    pub type Mtl4CommandBuffer = Retained<ProtocolObject<dyn MTL4CommandBuffer>>;
    pub type Mtl4ComputeEncoder = Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>;
    pub type Mtl4ArgTable = Retained<ProtocolObject<dyn MTL4ArgumentTable>>;
    pub type SharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;
    pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
    pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
    pub type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    pub type CommandBufferRef = ProtocolObject<dyn MTLCommandBuffer>;
    pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
    pub type ComputeCommandEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
    pub type ComputeCommandEncoderRef = ProtocolObject<dyn MTLComputeCommandEncoder>;
    pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
}
