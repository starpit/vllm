// SPDX-License-Identifier: Apache-2.0
//! Interpreter codegen — one submodule per interpreter that
//! consumes the solver's lowered bucket. The host interpreter's
//! codegen still lives in `crate::interpreter_codegen` (will
//! migrate to `interpreter::host` later); the kvm interpreter's
//! codegen lives here in [`kvm`].

pub mod kvm;
