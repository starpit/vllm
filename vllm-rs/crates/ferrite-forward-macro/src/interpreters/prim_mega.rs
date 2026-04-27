// SPDX-License-Identifier: Apache-2.0
//! Primitive megakernel emitter (Phase 1).
//!
//! One persistent `__global__` per arch; body is a `switch` over
//! `[i32; 32]` opcodes; each arm calls a `__device__` fn that
//! wraps a vendor kernel (cutlass / flashinfer / TK). No warp
//! specialization, no page virtual memory, no controller / loader
//! / storer split. Per-op handoffs are `__syncthreads()` or
//! grid sync.
//!
//! Job: prove the infra (DeviceCallable Impl audit, encoder /
//! scheduler / launcher plumbing) end-to-end with the smallest
//! possible surface. Probably perf-flat or slightly worse than
//! host on Phase 1.
//!
//! Today this file is a skeleton. The encoder match, the per-arch
//! `globals` struct emit, and the launcher fn land as the Phase 1
//! work order in `MEGA_HANDOFF.md` proceeds. The eligibility
//! predicate already lives on [`TargetProfile::prim_mega_compatible`]
//! and returns `false` until a target carries a primitive
//! megakernel `.cu`.

#![allow(dead_code)]
