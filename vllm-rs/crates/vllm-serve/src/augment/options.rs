use std::sync::Arc;

use crate::tokenizer::Tokenizer;

use super::cluster::ClusterStrategy;
use super::embed::TokenEmbedder;
use super::sidecar::SidecarManager;
use super::summarize::Summarizer;

/// Which indexing strategy to apply when building a corpus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Indexer {
    /// Single-level: chunk → embed → flat HNSW index.
    Layer1,
    /// Multi-level RAPTOR: layer1, then for each passage retrieve neighbors,
    /// LLM-summarize the neighborhood, and rebuild the index with originals
    /// plus all summaries. Requires an in-process engine for summarization
    /// (server path only).
    Raptor,
}

/// Options controlling RAG augmentation behavior.
#[derive(Clone)]
pub struct AugmentOptions {
    /// Max fragments to retrieve per Augment node. Also used by RAPTOR as the
    /// neighborhood size when summarizing each passage.
    pub max_aug: usize,
    /// Directory where LEANN indexes are stored.
    pub index_dir: String,
    /// Chunk size for sentence-based chunking (characters).
    pub chunk_size: usize,
    /// Chunk overlap for sentence-based chunking (characters).
    pub chunk_overlap: usize,
    /// Model name served by the current engine (for in-process embedding).
    pub current_model: Option<String>,
    /// In-process token embedder (set when the engine can compute embeddings).
    pub embedder: Option<Arc<dyn TokenEmbedder>>,
    /// Tokenizer for in-process embedding.
    pub tokenizer: Option<Arc<Tokenizer>>,
    /// Sidecar manager — spawns vllm-rs embedding processes on demand when
    /// the current model cannot serve embeddings in-process.
    pub sidecar_manager: Option<Arc<SidecarManager>>,
    /// LLM summarizer used by RAPTOR phase-2 cluster summarization. Set
    /// by the server path; left `None` on the offline LLM path (which
    /// then errors if `Indexer::Raptor` is requested).
    pub summarizer: Option<Arc<dyn Summarizer>>,
    /// Which indexing strategy to use. Defaults to `Layer1`.
    pub indexer: Indexer,
    /// Target branching factor for RAPTOR levels: each cluster covers
    /// approximately this many nodes from the level below.
    pub raptor_branching: usize,
    /// Maximum tree depth (number of summary levels above the originals).
    /// The level loop also stops early once a level shrinks to <= 1 node.
    pub raptor_max_depth: usize,
    /// Clustering algorithm used at each RAPTOR level.
    pub raptor_cluster_strategy: ClusterStrategy,
    /// Soft cluster membership: each node is also pushed into its
    /// `raptor_soft_overlap - 1` next-best clusters. `1` means hard
    /// k-means (default); `2`–`3` is a cheap stand-in for GMM soft
    /// assignment that improves coverage of ambiguous nodes.
    pub raptor_soft_overlap: usize,
    /// At retrieve time, replace each retrieved summary hit with its
    /// descendant leaves (BFS over `raptor_children`). Off by default —
    /// canonical RAPTOR "collapsed tree" mode returns summaries and
    /// originals together as-is, which empirically works better for
    /// fact-retrieval QA where the matched summary is itself the
    /// strongest signal. Opt in via `VLLM_RAG_RAPTOR_EXPAND_HITS=1`.
    pub raptor_expand_hits: bool,
    /// Max tokens for each RAPTOR summary generation.
    pub raptor_summary_max_tokens: u32,
    /// Sampling temperature for each RAPTOR summary generation.
    pub raptor_summary_temperature: f32,
    /// Maximum concurrent in-flight summary generations during a level.
    pub raptor_concurrency: usize,
}

impl Default for AugmentOptions {
    fn default() -> Self {
        Self {
            max_aug: 10,
            index_dir: "data/leann".to_string(),
            chunk_size: 512,
            chunk_overlap: 50,
            current_model: None,
            embedder: None,
            tokenizer: None,
            sidecar_manager: None,
            summarizer: None,
            indexer: Indexer::Layer1,
            raptor_branching: 10,
            raptor_max_depth: 3,
            raptor_cluster_strategy: ClusterStrategy::KMeans,
            raptor_soft_overlap: 1,
            raptor_expand_hits: false,
            raptor_summary_max_tokens: 100,
            raptor_summary_temperature: 0.2,
            raptor_concurrency: 32,
        }
    }
}

impl AugmentOptions {
    /// Whether the current engine can serve embeddings for the given model
    /// in-process (model matches and embedder is available).
    pub fn can_embed_in_process(&self, embedding_model: &str) -> bool {
        self.current_model.as_deref() == Some(embedding_model)
            && self.embedder.is_some()
            && self.tokenizer.is_some()
    }

    /// Apply environment-variable overrides on top of the current options.
    ///
    /// Recognized variables:
    ///   - `VLLM_RAG_INDEXER` — `layer1` (default) or `raptor`.
    ///   - `VLLM_RAG_RAPTOR_BRANCHING` — target cluster size (default 10).
    ///   - `VLLM_RAG_RAPTOR_MAX_DEPTH` — max tree depth (default 3).
    ///   - `VLLM_RAG_RAPTOR_SUMMARY_MAX_TOKENS` — per-summary cap (default 100).
    ///   - `VLLM_RAG_RAPTOR_CONCURRENCY` — bounded summary concurrency (default 32).
    ///
    /// Unknown values for `VLLM_RAG_INDEXER` are ignored with a warning so a
    /// typo doesn't silently fall back to a different mode than intended.
    pub fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("VLLM_RAG_INDEXER") {
            match v.to_ascii_lowercase().as_str() {
                "layer1" | "" => self.indexer = Indexer::Layer1,
                "raptor" => self.indexer = Indexer::Raptor,
                other => tracing::warn!(
                    "ignoring VLLM_RAG_INDEXER={other:?}; expected `layer1` or `raptor`"
                ),
            }
        }
        if let Ok(v) = std::env::var("VLLM_RAG_RAPTOR_BRANCHING")
            && let Ok(n) = v.parse::<usize>()
        {
            self.raptor_branching = n;
        }
        if let Ok(v) = std::env::var("VLLM_RAG_RAPTOR_MAX_DEPTH")
            && let Ok(n) = v.parse::<usize>()
        {
            self.raptor_max_depth = n;
        }
        if let Ok(v) = std::env::var("VLLM_RAG_RAPTOR_SUMMARY_MAX_TOKENS")
            && let Ok(n) = v.parse::<u32>()
        {
            self.raptor_summary_max_tokens = n;
        }
        if let Ok(v) = std::env::var("VLLM_RAG_RAPTOR_CONCURRENCY")
            && let Ok(n) = v.parse::<usize>()
        {
            self.raptor_concurrency = n;
        }
        if let Ok(v) = std::env::var("VLLM_RAG_RAPTOR_SOFT_OVERLAP")
            && let Ok(n) = v.parse::<usize>()
        {
            self.raptor_soft_overlap = n;
        }
        if let Ok(v) = std::env::var("VLLM_RAG_RAPTOR_CLUSTER")
            && let Some(strategy) = ClusterStrategy::parse(&v)
        {
            self.raptor_cluster_strategy = strategy;
        }
        if let Ok(v) = std::env::var("VLLM_RAG_RAPTOR_EXPAND_HITS") {
            self.raptor_expand_hits = matches!(v.as_str(), "1" | "true" | "yes" | "on");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All env-override cases run in one test to avoid race conditions
    /// from `std::env::set_var` (which is process-global).
    #[test]
    fn env_overrides_apply_correctly() {
        let keys = [
            "VLLM_RAG_INDEXER",
            "VLLM_RAG_RAPTOR_BRANCHING",
            "VLLM_RAG_RAPTOR_MAX_DEPTH",
            "VLLM_RAG_RAPTOR_SUMMARY_MAX_TOKENS",
            "VLLM_RAG_RAPTOR_CONCURRENCY",
            "VLLM_RAG_RAPTOR_SOFT_OVERLAP",
            "VLLM_RAG_RAPTOR_CLUSTER",
        ];
        for k in &keys {
            // SAFETY: setenv is unsafe in Rust 2024; tests are single-threaded
            // for env vars by convention and we restore at the end.
            unsafe {
                std::env::remove_var(k);
            }
        }

        // Defaults preserved when nothing is set.
        let mut o = AugmentOptions::default();
        o.apply_env_overrides();
        assert_eq!(o.indexer, Indexer::Layer1);
        assert_eq!(o.raptor_branching, 10);
        assert_eq!(o.raptor_max_depth, 3);
        assert_eq!(o.raptor_summary_max_tokens, 100);
        assert_eq!(o.raptor_concurrency, 32);
        assert_eq!(o.raptor_soft_overlap, 1);
        assert_eq!(o.raptor_cluster_strategy, ClusterStrategy::KMeans);

        // Valid overrides applied.
        unsafe {
            std::env::set_var("VLLM_RAG_INDEXER", "raptor");
            std::env::set_var("VLLM_RAG_RAPTOR_BRANCHING", "25");
            std::env::set_var("VLLM_RAG_RAPTOR_MAX_DEPTH", "5");
            std::env::set_var("VLLM_RAG_RAPTOR_SUMMARY_MAX_TOKENS", "200");
            std::env::set_var("VLLM_RAG_RAPTOR_CONCURRENCY", "8");
            std::env::set_var("VLLM_RAG_RAPTOR_SOFT_OVERLAP", "3");
            std::env::set_var("VLLM_RAG_RAPTOR_CLUSTER", "gmm");
        }
        let mut o = AugmentOptions::default();
        o.apply_env_overrides();
        assert_eq!(o.indexer, Indexer::Raptor);
        assert_eq!(o.raptor_branching, 25);
        assert_eq!(o.raptor_max_depth, 5);
        assert_eq!(o.raptor_summary_max_tokens, 200);
        assert_eq!(o.raptor_concurrency, 8);
        assert_eq!(o.raptor_soft_overlap, 3);
        assert_eq!(o.raptor_cluster_strategy, ClusterStrategy::Gmm);

        // Unknown indexer value warns and leaves the field at its prior
        // value (we set it back to Layer1 first to make this observable).
        unsafe {
            std::env::set_var("VLLM_RAG_INDEXER", "what-is-this");
        }
        let mut o = AugmentOptions::default();
        o.apply_env_overrides();
        assert_eq!(
            o.indexer,
            Indexer::Layer1,
            "unknown indexer value must not change default"
        );

        // Layer1 is also accepted explicitly.
        unsafe {
            std::env::set_var("VLLM_RAG_INDEXER", "layer1");
        }
        let mut o = AugmentOptions {
            indexer: Indexer::Raptor,
            ..Default::default()
        };
        o.apply_env_overrides();
        assert_eq!(o.indexer, Indexer::Layer1);

        // Cleanup.
        for k in &keys {
            unsafe {
                std::env::remove_var(k);
            }
        }
    }
}
