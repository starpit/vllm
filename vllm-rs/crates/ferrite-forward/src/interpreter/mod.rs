// SPDX-License-Identifier: Apache-2.0
//! Interpreter runtime support — sibling of the macro-side
//! `ferrite_forward_macro::interpreter` codegen modules. The host
//! interpreter's runtime support still lives in [`crate::instr`]
//! (will migrate to `interpreter::host` later); the kvm
//! interpreter's runtime helpers live here in [`kvm`].

#[cfg(feature = "cuda")]
pub mod kvm;
