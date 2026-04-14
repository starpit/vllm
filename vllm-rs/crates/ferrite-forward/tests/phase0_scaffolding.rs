// SPDX-License-Identifier: Apache-2.0
//! Phase 0 invariant: `#[forward]` on an empty carrier fn compiles
//! and produces an empty fn. No DSL parsing yet.

use ferrite_forward::forward;

#[forward]
fn llama() {}

#[forward]
fn qwen2() {}

#[test]
fn carrier_fns_are_callable() {
    // The macro should have produced empty fns we can call.
    llama();
    qwen2();
}
