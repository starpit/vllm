// SPDX-License-Identifier: Apache-2.0
//! Shared quantization preset definitions. Each per-arch crate
//! depends on this and the `#[forward]` macro discovers
//! `presets/*.json` by walking up from the consuming crate's
//! manifest dir to the workspace root, then into
//! `crates/ferrite-quantizations/presets/`.
