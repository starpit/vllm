// SPDX-License-Identifier: Apache-2.0
//! Consumer-facing crate for the `#[forward]` attribute macro.
//!
//! Re-exports the proc-macro and holds any runtime types the
//! generated code depends on.

pub use ferrite_forward_macro::forward;
