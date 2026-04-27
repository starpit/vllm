// SPDX-License-Identifier: Apache-2.0
//! KVM megakernel emitter (Phase 2).
//!
//! Vendored `~/Megakernels` template instantiated per arch.
//! Warp specialization (controller / loader / consumers / storer
//! / launcher), instruction pipelining, page virtual memory,
//! per-SM instruction tape (see vendor's
//! `include/controller/instruction_fetch.cuh` —
//! `get_worker_id()` indexes the second dim of the
//! `[1, NUM_SMS, ROWS_PER_SM, 32]` instruction tensor).
//!
//! This is where the perf comes from. Builds on Phase 1's
//! DeviceCallable Impls; adds KVM-specific authoring (output-
//! tile-granular `fan_out`, `release_lid`, semaphore
//! choreography). Activates the Rule 4 stub at
//! `concurrency.rs:38–44, 119–123`.
//!
//! See the Phase 2 work order in `MEGA_HANDOFF.md`. Today this
//! file is a skeleton; do not author KVM templates for ops that
//! Phase 1 already covers via DeviceCallable cutlass /
//! flashinfer wrappings — Phase 2 only adds template authoring
//! where output-tile granularity buys real perf.

#![allow(dead_code)]
