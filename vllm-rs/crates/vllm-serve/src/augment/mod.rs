mod cluster;
pub(crate) mod embed;
pub(crate) mod index;
mod options;
mod raptor;
mod retrieve;
mod sidecar;
pub(crate) mod summarize;

pub use index::index;
pub use options::{AugmentOptions, Indexer};
pub use retrieve::retrieve;
pub use sidecar::SidecarManager;
