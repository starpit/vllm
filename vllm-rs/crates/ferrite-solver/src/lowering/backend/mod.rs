// SPDX-License-Identifier: Apache-2.0
//! Backend: transforms a solved [`ExecutionPlan`] into dispatch-ready
//! data structures and Rust source code.
//!
//! ## Modules
//!
//! - [`dispatch`] — flattens an ExecutionPlan into a step-ordered
//!   dispatch sequence with typed entries.
//! - [`compile_dsl`] — parses the `compile!` macro DSL (binding modes).
//! - [`codegen`] — emits Rust TokenStream from the DSL + solver output.

pub mod codegen;
#[cfg(test)]
mod codegen_dump_test;
#[cfg(test)]
mod codegen_test;
pub mod compile_dsl;
pub mod cuda_codegen;
pub mod dispatch;
pub mod fuf_codegen;

pub use compile_dsl::ForwardDef;
pub use dispatch::{DispatchEntry, DispatchSequence, ImplDispatchKind};
