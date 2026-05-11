// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! CLI subcommand implementations.

pub mod batch;
#[cfg(feature = "bench")]
pub mod bench;
pub mod chat;
pub mod collect_env;
pub mod convert;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub mod ferrite;
#[cfg(feature = "gce")]
pub mod gce;
#[cfg(feature = "k8s")]
pub mod k8s;
pub mod model;
pub mod pull;
pub mod serve;
#[cfg(feature = "top")]
pub mod top;
