// SPDX-License-Identifier: Apache-2.0
//! NVTX profiling annotations for nsys/Nsight GPU traces.
//!
//! When the `profiling` feature is enabled, functions in this module emit
//! NVTX push/pop calls that appear as named ranges in `nsys profile` traces.
//! Without the feature, all functions compile to zero-cost no-ops.
//!
//! # Usage
//!
//! ```ignore
//! use vllm_kernels::profiling;
//!
//! fn forward_pass() {
//!     let _guard = profiling::range("forward");
//!     // ... work ...
//!     // Range ends when _guard drops
//! }
//!
//! fn per_layer(i: usize) {
//!     let _guard = profiling::range_fmt(format_args!("layer_{i}"));
//!     // ...
//! }
//! ```
//!
//! # Building
//!
//! ```bash
//! cargo build --features profiling   # implies cuda
//! nsys profile --trace=cuda,nvtx ./target/release/vllm bench latency ...
//! ```

// ---------------------------------------------------------------------------
// When profiling is enabled: real NVTX ranges via cudarc
// ---------------------------------------------------------------------------

#[cfg(feature = "profiling")]
pub use cudarc::nvtx::safe::Range;

/// Create a named NVTX range that ends when the returned guard is dropped.
#[cfg(feature = "profiling")]
#[inline]
pub fn range(name: &str) -> Range {
    cudarc::nvtx::safe::scoped_range(name)
}

/// Create a named NVTX range with a dynamic (format_args!) name.
#[cfg(feature = "profiling")]
#[inline]
pub fn range_fmt(args: std::fmt::Arguments<'_>) -> Range {
    cudarc::nvtx::safe::scoped_range(args.to_string())
}

/// Mark an instant event in the NVTX timeline.
#[cfg(feature = "profiling")]
#[inline]
pub fn mark(name: &str) {
    cudarc::nvtx::safe::mark(name);
}

// ---------------------------------------------------------------------------
// When profiling is disabled: zero-cost no-ops
// ---------------------------------------------------------------------------

/// Zero-size guard returned when profiling is off. Optimized away entirely.
#[cfg(not(feature = "profiling"))]
pub struct NoopRange;

/// No-op range (compiles to nothing when profiling is off).
#[cfg(not(feature = "profiling"))]
#[inline(always)]
pub fn range(_name: &str) -> NoopRange {
    NoopRange
}

/// No-op range_fmt (compiles to nothing when profiling is off).
#[cfg(not(feature = "profiling"))]
#[inline(always)]
pub fn range_fmt(_args: std::fmt::Arguments<'_>) -> NoopRange {
    NoopRange
}

/// No-op mark (compiles to nothing when profiling is off).
#[cfg(not(feature = "profiling"))]
#[inline(always)]
pub fn mark(_name: &str) {}
