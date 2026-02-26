// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Tracing subscriber initialization.
//!
//! Provides a one-shot `init_tracing()` that sets up `tracing_subscriber::fmt`
//! with an `EnvFilter` for controlling log levels.

use std::sync::Once;

use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

/// Initialize the global tracing subscriber.
///
/// Uses `RUST_LOG` env var if set, otherwise falls back to `log_level`.
/// Safe to call multiple times — only the first call takes effect.
pub fn init_tracing(log_level: &str) {
    INIT.call_once(|| {
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level));

        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .compact()
            .with_target(false)
            .init();
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_tracing_idempotent() {
        // Multiple calls should not panic.
        init_tracing("warn");
        init_tracing("debug");
    }
}
