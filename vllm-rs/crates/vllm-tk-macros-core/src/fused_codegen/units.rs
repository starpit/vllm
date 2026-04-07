// SPDX-License-Identifier: Apache-2.0
//! Newtypes for fused codegen quantities.
//!
//! Prevents mixing up byte sizes, element dimensions, iteration counts,
//! tile counts, and discrete counts at compile time. Each implements
//! `Display` for Askama template rendering and comparison with `usize`
//! for Askama template conditionals.

use std::fmt;

/// Define a newtype wrapper around `usize` with Display, comparison, and Ord.
macro_rules! newtype {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(pub usize);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        /// Allow Askama template conditionals like `{% if field > 1 %}`.
        impl PartialEq<usize> for $name {
            fn eq(&self, other: &usize) -> bool {
                self.0 == *other
            }
        }

        impl PartialOrd<usize> for $name {
            fn partial_cmp(&self, other: &usize) -> Option<std::cmp::Ordering> {
                self.0.partial_cmp(other)
            }
        }
    };
}

newtype!(
    /// Model dimension in elements (HD, ID, HDM, QKV_DIM, cta_rows, k_dim, out_block, kv_page_size).
    Dim
);

newtype!(
    /// Byte size (shmem, tile bytes, a_size, b_size, stage_size, byte offsets).
    Bytes
);

newtype!(
    /// Count of discrete items (warps, heads, layers, stages, GQA ratio, threads, grid CTAs).
    Count
);

newtype!(
    /// Number of iterations (K-loop iters, attention passes, iters_per_page).
    Iters
);

newtype!(
    /// Number of tiles (col_tiles, col_batch).
    Tiles
);
