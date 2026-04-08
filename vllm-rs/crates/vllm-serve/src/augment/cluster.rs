//! Clustering for RAPTOR multi-level indexing.
//!
//! The current implementation is a hand-rolled cosine k-means with k-means++
//! initialization. It is intentionally hidden behind a `ClusterStrategy` enum
//! and a single `cluster_vectors` entry point so that swapping in fancier
//! techniques (GMM, UMAP+GMM, soft-assignment HDBSCAN, …) is a body-only
//! change to one function.
//!
//! Pivoting to `linfa-clustering`:
//!   - Add `linfa = { ... }` and `linfa-clustering = { ... }` to the `rag`
//!     feature in `vllm-serve/Cargo.toml`.
//!   - Implement `cluster_vectors_linfa_kmeans` / `cluster_vectors_linfa_gmm`
//!     below; both return the same `Vec<Vec<usize>>` shape so the
//!     `cross_index` call site stays untouched.
//!   - Note: vectors here are L2-normalized up front, so linfa's default L2
//!     KMeans ranks identically to cosine k-means and the swap is
//!     behavior-preserving for the typical normalized-embedding case.

use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// Which clustering algorithm to apply at each RAPTOR level.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClusterStrategy {
    /// Cosine k-means with k-means++ init. Hand-rolled, no extra deps.
    #[default]
    KMeans,
    /// Gaussian Mixture Model with soft assignment.
    ///
    /// Wiring is staged but the implementation falls back to `KMeans` at
    /// runtime: `linfa-clustering` 0.7 depends on `ndarray = 0.15`, while
    /// this workspace pins `ndarray = 0.17`. To enable GMM for real:
    ///   1. Bump `ndarray` workspace pin to a version supported by the
    ///      `linfa-clustering` release you target (or vice versa).
    ///   2. Add `linfa = "0.7"` and `linfa-clustering = "0.7"` to the
    ///      `rag` feature in `vllm-serve/Cargo.toml`.
    ///   3. Replace the `Gmm` arm of `cluster_vectors` with a call to
    ///      `GaussianMixtureModel::params(k).fit(&dataset)` and threshold
    ///      the soft assignments into `Vec<Vec<usize>>`.
    Gmm,
    /// UMAP dimensionality reduction followed by GMM, matching the
    /// canonical RAPTOR Python reference. No rust UMAP crate has the
    /// numerical stability we'd want for embedding-space reduction yet,
    /// so this falls back to `KMeans` at runtime as well.
    UmapGmm,
}

impl ClusterStrategy {
    /// Parse from an environment-variable string. Returns `None` for
    /// unknown values so the caller can warn.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "kmeans" | "k-means" => Some(Self::KMeans),
            "gmm" => Some(Self::Gmm),
            "umap-gmm" | "umap_gmm" | "umapgmm" => Some(Self::UmapGmm),
            _ => None,
        }
    }
}

/// Cluster `n` vectors of dimension `dim` (row-major flat layout) into `k`
/// groups. Returns groups of row indices (a row may appear in multiple
/// groups when `soft_overlap > 1`). Empty groups are dropped.
///
/// `soft_overlap == 1` is hard k-means; `soft_overlap == T` assigns each
/// point to its top-T nearest centroids — a cheap stand-in for true GMM
/// soft membership that requires no extra dependency. Future strategies
/// (GMM, UMAP+GMM) can return overlapping groups via the same shape.
///
/// Vectors are L2-normalized internally; the expected input is already
/// unit-norm (matches what HNSW stores) so the renormalization is cheap.
pub(crate) fn cluster_vectors(
    strategy: ClusterStrategy,
    vectors: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    soft_overlap: usize,
) -> Vec<Vec<usize>> {
    debug_assert_eq!(vectors.len(), n * dim);
    if n == 0 || k == 0 {
        return vec![];
    }
    if k >= n {
        return (0..n).map(|i| vec![i]).collect();
    }
    let overlap = soft_overlap.clamp(1, k);
    match strategy {
        ClusterStrategy::KMeans => kmeans_cosine(vectors, n, dim, k, overlap),
        // GMM / UMAP+GMM are stubbed pending the ndarray version bump
        // documented on the enum variants. Falling back to k-means
        // (rather than panicking) keeps the request flow intact even if
        // someone sets `VLLM_RAG_RAPTOR_CLUSTER=gmm` on a build that
        // hasn't pulled in linfa yet.
        ClusterStrategy::Gmm | ClusterStrategy::UmapGmm => {
            tracing::warn!(
                strategy = ?strategy,
                "RAPTOR cluster strategy not yet implemented; falling back to KMeans"
            );
            kmeans_cosine(vectors, n, dim, k, overlap)
        }
    }
}

/// Hand-rolled cosine k-means.
///
/// Assumes (and re-normalizes) unit vectors so that cosine distance reduces
/// to negative dot product. K-means++ init, Lloyd iterations, deterministic
/// PRNG seeded from a fixed value so that reruns of the same corpus produce
/// identical trees.
fn kmeans_cosine(
    vectors: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    soft_overlap: usize,
) -> Vec<Vec<usize>> {
    const MAX_ITERS: usize = 15;
    const EPSILON: f32 = 1e-4;

    // Defensive normalization. The HNSW stored vectors should already be
    // unit-norm, but cheap to enforce and avoids surprises if a future
    // embedding model emits non-normalized output. Parallel over rows.
    let mut data = vectors.to_vec();
    data.par_chunks_mut(dim).for_each(|row| {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    });

    // K-means++ init. The init pass itself is sequential over k centroids
    // (each pick depends on the previous min-distance vector), but the
    // per-row distance update inside each step parallelizes cleanly.
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x4150_4F52u64);
    let mut centroids: Vec<f32> = Vec::with_capacity(k * dim);
    let first: usize = rng.gen_range(0..n);
    centroids.extend_from_slice(&data[first * dim..(first + 1) * dim]);

    let mut min_dist_sq = vec![f32::INFINITY; n];
    for _ in 1..k {
        let last_centroid = &centroids[centroids.len() - dim..];
        // Parallel distance update against the just-added centroid.
        data.par_chunks(dim)
            .zip(min_dist_sq.par_iter_mut())
            .for_each(|(row, slot)| {
                // For unit vectors, ||a-b||^2 = 2 - 2*(a·b).
                let dot: f32 = row.iter().zip(last_centroid).map(|(a, b)| a * b).sum();
                let d = (2.0 - 2.0 * dot).max(0.0);
                if d < *slot {
                    *slot = d;
                }
            });
        let total: f32 = min_dist_sq.iter().sum();
        let pick = if total > 0.0 {
            let mut t = rng.r#gen::<f32>() * total;
            let mut chosen = n - 1;
            for (i, d) in min_dist_sq.iter().enumerate() {
                t -= d;
                if t <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            chosen
        } else {
            rng.gen_range(0..n)
        };
        centroids.extend_from_slice(&data[pick * dim..(pick + 1) * dim]);
    }

    // Lloyd iterations.
    let mut assign = vec![0usize; n];
    for _ in 0..MAX_ITERS {
        // Parallel assignment: each row picks its best centroid independently.
        // We compute fresh assignments into a temp vec, then check for changes.
        let new_assign: Vec<usize> = data
            .par_chunks(dim)
            .map(|row| {
                let mut best = 0usize;
                let mut best_dot = f32::NEG_INFINITY;
                for c in 0..k {
                    let cent = &centroids[c * dim..(c + 1) * dim];
                    let dot: f32 = row.iter().zip(cent).map(|(a, b)| a * b).sum();
                    if dot > best_dot {
                        best_dot = dot;
                        best = c;
                    }
                }
                best
            })
            .collect();
        let changed = new_assign.iter().zip(assign.iter()).any(|(a, b)| a != b);
        assign = new_assign;

        // Recompute centroids: per-row contributions accumulated in
        // per-thread partials, then reduced. Avoids the lock-step
        // contention of a single shared sum.
        let (new_centroids, counts) = data
            .par_chunks(dim)
            .zip(assign.par_iter())
            .fold(
                || (vec![0.0f32; k * dim], vec![0usize; k]),
                |(mut sums, mut counts), (row, &c)| {
                    counts[c] += 1;
                    let dst = &mut sums[c * dim..(c + 1) * dim];
                    for (d, s) in dst.iter_mut().zip(row) {
                        *d += *s;
                    }
                    (sums, counts)
                },
            )
            .reduce(
                || (vec![0.0f32; k * dim], vec![0usize; k]),
                |(mut a_sums, mut a_counts), (b_sums, b_counts)| {
                    for (a, b) in a_sums.iter_mut().zip(b_sums.iter()) {
                        *a += *b;
                    }
                    for (a, b) in a_counts.iter_mut().zip(b_counts.iter()) {
                        *a += *b;
                    }
                    (a_sums, a_counts)
                },
            );
        let mut new_centroids = new_centroids;
        let mut max_shift: f32 = 0.0;
        for c in 0..k {
            let cent = &mut new_centroids[c * dim..(c + 1) * dim];
            if counts[c] == 0 {
                // Reseed empty cluster from a random point so it can recover.
                let pick = rng.gen_range(0..n);
                cent.copy_from_slice(&data[pick * dim..(pick + 1) * dim]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for x in cent.iter_mut() {
                    *x *= inv;
                }
            }
            let norm = cent.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in cent.iter_mut() {
                    *x /= norm;
                }
            }
            // Track shift vs old centroid.
            let old = &centroids[c * dim..(c + 1) * dim];
            let shift: f32 = cent.iter().zip(old).map(|(a, b)| (a - b).powi(2)).sum();
            if shift > max_shift {
                max_shift = shift;
            }
        }
        centroids = new_centroids;
        if !changed || max_shift.sqrt() < EPSILON {
            break;
        }
    }

    // Hard assignment from the last Lloyd pass: every point lives in
    // exactly one group based on best centroid.
    let mut groups: Vec<Vec<usize>> = (0..k).map(|_| Vec::new()).collect();
    for (i, &c) in assign.iter().enumerate() {
        groups[c].push(i);
    }

    // Soft overlap: also push each point into its `soft_overlap - 1`
    // *next* best centroids. Cheap stand-in for GMM soft assignment;
    // boosts coverage for ambiguous points without changing the API.
    if soft_overlap > 1 {
        for i in 0..n {
            let row = &data[i * dim..(i + 1) * dim];
            // Score every centroid, then pick the top `soft_overlap` skipping
            // the already-assigned one. k is typically O(sqrt(n)) so the
            // O(k log k) sort is cheap.
            let mut scored: Vec<(usize, f32)> = (0..k)
                .map(|c| {
                    let cent = &centroids[c * dim..(c + 1) * dim];
                    let dot: f32 = row.iter().zip(cent).map(|(a, b)| a * b).sum();
                    (c, dot)
                })
                .collect();
            scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut added = 0usize;
            for (c, _) in scored.into_iter() {
                if c == assign[i] {
                    continue;
                }
                groups[c].push(i);
                added += 1;
                if added + 1 >= soft_overlap {
                    break;
                }
            }
        }
    }

    groups.retain(|g| !g.is_empty());
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build n unit-norm vectors of dim 2 from a list of (x, y) pairs.
    fn make2d(points: &[(f32, f32)]) -> Vec<f32> {
        let mut out = Vec::with_capacity(points.len() * 2);
        for &(x, y) in points {
            let n = (x * x + y * y).sqrt().max(1e-9);
            out.push(x / n);
            out.push(y / n);
        }
        out
    }

    #[test]
    fn empty_input_returns_empty() {
        assert!(cluster_vectors(ClusterStrategy::KMeans, &[], 0, 4, 3, 1).is_empty());
        assert!(cluster_vectors(ClusterStrategy::KMeans, &[1.0; 4], 1, 4, 0, 1).is_empty());
    }

    #[test]
    fn k_ge_n_one_per_point() {
        let v = make2d(&[(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]);
        let g = cluster_vectors(ClusterStrategy::KMeans, &v, 3, 2, 5, 1);
        assert_eq!(g.len(), 3);
        for group in &g {
            assert_eq!(group.len(), 1);
        }
    }

    #[test]
    fn two_well_separated_blobs() {
        // Five points around (1,0); five points around (-1,0). Tight jitter.
        let v = make2d(&[
            (1.00, 0.02),
            (1.01, -0.01),
            (0.99, 0.00),
            (1.02, 0.01),
            (0.98, -0.02),
            (-1.00, 0.02),
            (-1.01, -0.01),
            (-0.99, 0.00),
            (-1.02, 0.01),
            (-0.98, -0.02),
        ]);
        let g = cluster_vectors(ClusterStrategy::KMeans, &v, 10, 2, 2, 1);
        assert_eq!(g.len(), 2);
        // Each group should be entirely from one blob (indices 0..5 vs 5..10).
        for group in &g {
            let all_left = group.iter().all(|&i| i < 5);
            let all_right = group.iter().all(|&i| i >= 5);
            assert!(
                all_left || all_right,
                "blob bled across clusters: {group:?}"
            );
        }
    }

    #[test]
    fn deterministic_under_reseed() {
        let v = make2d(&[
            (1.0, 0.1),
            (0.9, -0.1),
            (-0.9, 0.1),
            (-1.0, -0.1),
            (0.1, 1.0),
            (-0.1, 0.9),
        ]);
        let a = cluster_vectors(ClusterStrategy::KMeans, &v, 6, 2, 3, 1);
        let b = cluster_vectors(ClusterStrategy::KMeans, &v, 6, 2, 3, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn soft_overlap_increases_membership() {
        let v = make2d(&[
            (1.00, 0.02),
            (1.01, -0.01),
            (-1.00, 0.02),
            (-1.01, -0.01),
            (0.05, 1.00), // ambiguous third "blob" of one
            (-0.05, 1.00),
        ]);
        let hard = cluster_vectors(ClusterStrategy::KMeans, &v, 6, 2, 3, 1);
        let soft = cluster_vectors(ClusterStrategy::KMeans, &v, 6, 2, 3, 2);
        let hard_total: usize = hard.iter().map(|g| g.len()).sum();
        let soft_total: usize = soft.iter().map(|g| g.len()).sum();
        assert_eq!(hard_total, 6);
        // Soft overlap of 2 should approximately double membership.
        assert!(
            soft_total > hard_total,
            "soft overlap did not increase membership: hard={hard_total} soft={soft_total}"
        );
    }

    #[test]
    fn gmm_falls_back_to_kmeans() {
        let v = make2d(&[(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]);
        // Should not panic; should produce a valid partition.
        let g = cluster_vectors(ClusterStrategy::Gmm, &v, 4, 2, 2, 1);
        let total: usize = g.iter().map(|g| g.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn strategy_parse() {
        assert_eq!(
            ClusterStrategy::parse("kmeans"),
            Some(ClusterStrategy::KMeans)
        );
        assert_eq!(
            ClusterStrategy::parse("k-means"),
            Some(ClusterStrategy::KMeans)
        );
        assert_eq!(ClusterStrategy::parse("gmm"), Some(ClusterStrategy::Gmm));
        assert_eq!(
            ClusterStrategy::parse("umap-gmm"),
            Some(ClusterStrategy::UmapGmm)
        );
        assert_eq!(ClusterStrategy::parse("nonsense"), None);
    }
}
