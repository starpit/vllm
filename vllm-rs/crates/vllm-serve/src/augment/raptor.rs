//! RAPTOR multi-level indexing.
//!
//! Builds a tree of summaries on top of an existing layer-1 LEANN index by
//! repeatedly clustering the current level's embedding vectors, summarizing
//! each cluster with the in-process engine, embedding the summaries, and
//! using those as the next level's input. The loop stops when a level
//! shrinks to one node or `raptor_max_depth` is reached.
//!
//! At the end, every node from every level (originals + all summaries) is
//! merged into a single rebuilt HNSW index. Retrieval stays flat — tree
//! traversal can be added later behind a feature flag without changing the
//! index format.
//!
//! Differences vs. the canonical RAPTOR Python reference:
//!   - clustering: hand-rolled cosine k-means (no UMAP, no GMM yet — see
//!     `cluster.rs` for the pivot point);
//!   - retrieval: flat HNSW over all levels (canonical does tree traversal);
//!   - soft assignment: each node belongs to exactly one cluster.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures_util::{StreamExt, TryStreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use spnl_core::ir::Message;
use spnl_core::optimizer::llo::llir::NonGenerateInput;
use tracing::info;

use leann_core::embedding::EmbeddingProvider;
use leann_core::hnsw::{graph::VectorStorage, io::read_hnsw_index};
use leann_core::index::{IndexPaths, PassageSource};
use leann_core::passages::{PassageManager, load_id_map};

use super::cluster::cluster_vectors;
use super::embed::{HttpEmbeddingProvider, InProcessEmbeddingProvider};
use super::index::{IndexedCorpus, resolve_embedding_base_url};
use super::options::AugmentOptions;
use super::summarize::Summarizer;

/// Build a RAPTOR tree on top of an already-built layer-1 corpus.
///
/// Constructs the embedding provider from `options` (in-process or sidecar)
/// and delegates to [`cross_index_with_provider`]. The split exists so tests
/// can inject a mock provider without going through the full provider
/// selection logic, which requires a real `Tokenizer`.
pub(crate) async fn cross_index(
    corpus: IndexedCorpus,
    options: &AugmentOptions,
    summarizer: &dyn Summarizer,
) -> Result<()> {
    let provider: Arc<dyn EmbeddingProvider + Send + Sync> =
        if options.can_embed_in_process(&corpus.embedding_model) {
            let embedder = options.embedder.clone().unwrap();
            let tokenizer = options.tokenizer.clone().unwrap();
            Arc::new(InProcessEmbeddingProvider::new(
                corpus.embedding_model.clone(),
                corpus.dimensions,
                embedder,
                tokenizer,
            ))
        } else {
            let base_url = resolve_embedding_base_url(&corpus.embedding_model, options)?;
            Arc::new(HttpEmbeddingProvider::with_base_url(
                &corpus.embedding_model,
                corpus.dimensions,
                base_url,
            ))
        };
    cross_index_with_provider(corpus, options, summarizer, provider).await
}

/// Inner cross_index that takes a pre-built embedding provider. Public to
/// the crate for testing.
pub(crate) async fn cross_index_with_provider(
    corpus: IndexedCorpus,
    options: &AugmentOptions,
    summarizer: &dyn Summarizer,
    provider: Arc<dyn EmbeddingProvider + Send + Sync>,
) -> Result<()> {
    let IndexedCorpus {
        filename,
        index_path,
        enclosing_model,
        embedding_model,
        dimensions,
    } = corpus;

    let file_base_name = std::path::Path::new(&filename)
        .file_name()
        .ok_or_else(|| anyhow!("Could not determine base name for {filename}"))?
        .to_string_lossy()
        .into_owned();

    let paths = IndexPaths::new(&index_path);

    // Load the layer-1 HNSW graph and unpack stored vectors. Reusing them
    // means we never re-embed the originals at any point in the tree build.
    let mut index_file = std::fs::File::open(paths.index_file_path())?;
    let graph = read_hnsw_index(&mut index_file)?;
    let stored_vectors: Vec<f32> = match &graph.vector_storage {
        VectorStorage::Raw { data, .. } => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        VectorStorage::Null => {
            return Err(anyhow!(
                "HNSW index has no stored vectors for RAPTOR cross-indexing"
            ));
        }
    };
    if graph.dimensions != dimensions {
        return Err(anyhow!(
            "dimension mismatch: index meta {dimensions} vs hnsw {}",
            graph.dimensions
        ));
    }

    // Load original passages and id map.
    let passage_source = PassageSource {
        source_type: "jsonl".to_string(),
        path: paths.passages_path().to_string_lossy().to_string(),
        index_path: paths.offset_path().to_string_lossy().to_string(),
        path_relative: None,
        index_path_relative: None,
    };
    let passage_mgr = PassageManager::load(&[passage_source], None)?;
    let id_map = load_id_map(&paths.id_map_path())?;
    let n0 = id_map.len();

    let original_texts: Vec<String> = (0..n0)
        .map(|i| {
            passage_mgr
                .get_passage_by_index(i)
                .map(|p| p.text)
                .unwrap_or_default()
        })
        .collect();

    // ------------------------------------------------------------------
    // Level loop: bottom-up tree construction.
    // ------------------------------------------------------------------
    let branching = options.raptor_branching.max(2);
    let max_depth = options.raptor_max_depth;

    /// One summary node added to the tree at some level above the originals.
    struct TreeSummary {
        id: String,
        text: String,
        level: usize,
        children: Vec<String>,
    }

    let mut all_summaries: Vec<TreeSummary> = Vec::new();

    // Working set for the current level. Level 0 starts as the originals,
    // sharing the layer-1 stored vectors and ids.
    let mut current_vectors: Vec<f32> = stored_vectors;
    let mut current_texts: Vec<String> = original_texts.clone();
    let mut current_ids: Vec<String> = id_map.clone();

    for level in 0..max_depth {
        let n = current_texts.len();
        if n <= 1 {
            break;
        }
        let k = n.div_ceil(branching);
        if k <= 1 {
            // One cluster covering everything — produces a single root and
            // we're done.
        }

        info!(
            filename = %filename,
            level,
            nodes = n,
            clusters = k,
            "RAPTOR level"
        );

        // ---- clustering (CPU, parallelized) ----
        let cluster_pb = ProgressBar::new_spinner();
        cluster_pb.set_style(
            ProgressStyle::default_spinner()
                .template(
                    "  {spinner:.magenta} RAPTOR L{prefix} clustering {msg} ({elapsed_precise})",
                )
                .unwrap(),
        );
        cluster_pb.set_prefix(format!("{}", level + 1));
        cluster_pb.set_message(format!(
            "{file_base_name} ({n} nodes → {k} clusters, {strategy:?})",
            strategy = options.raptor_cluster_strategy,
        ));
        cluster_pb.enable_steady_tick(std::time::Duration::from_millis(100));

        // Hand-rolled k-means is rayon-parallel CPU work. Run it on the
        // blocking pool so the async runtime stays responsive (the
        // current_thread runtime would otherwise stall the entire level
        // loop here for many seconds on large corpora).
        let strategy = options.raptor_cluster_strategy;
        let soft_overlap = options.raptor_soft_overlap;
        let cluster_vectors_clone = current_vectors.clone();
        let dim_local = dimensions;
        let n_local = n;
        let k_local = k;
        let groups: Vec<Vec<usize>> = tokio::task::spawn_blocking(move || {
            cluster_vectors(
                strategy,
                &cluster_vectors_clone,
                n_local,
                dim_local,
                k_local,
                soft_overlap,
            )
        })
        .await?;
        cluster_pb.finish_and_clear();

        // ---- per-cluster summarization (GPU, via summarizer) ----
        let level_pb = ProgressBar::new(groups.len() as u64).with_style(
            ProgressStyle::default_bar()
                .template(
                    "  {spinner:.cyan} RAPTOR L{prefix} {msg} [{bar:30.cyan/blue}] {pos}/{len} ({elapsed_precise})",
                )
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏ "),
        );
        level_pb.set_prefix(format!("{}", level + 1));
        level_pb.set_message(format!(
            "{file_base_name} ({n} nodes → {} clusters)",
            groups.len()
        ));
        level_pb.enable_steady_tick(std::time::Duration::from_millis(100));

        // Per-cluster summarization, bounded-concurrent.
        // Summary level number for ids/metadata is `level + 1` (level 0 is originals).
        let summary_level = level + 1;
        let summary_futures = groups
            .into_iter()
            .enumerate()
            .map(|(cluster_idx, members)| {
                let texts: Vec<NonGenerateInput> = members
                    .iter()
                    .map(|&i| NonGenerateInput::Message(Message::User(current_texts[i].clone())))
                    .collect();
                let children: Vec<String> =
                    members.iter().map(|&i| current_ids[i].clone()).collect();
                let id = format!("@raptor-{file_base_name}-L{summary_level}-C{cluster_idx}");
                let enclosing_model = enclosing_model.clone();
                let pb = level_pb.clone();
                async move {
                    let summary =
                        generate_summary(summarizer, &enclosing_model, texts, options).await?;
                    pb.inc(1);
                    Ok::<TreeSummary, anyhow::Error>(TreeSummary {
                        id,
                        text: summary,
                        level: summary_level,
                        children,
                    })
                }
            });

        let level_summaries: Vec<TreeSummary> = futures_util::stream::iter(summary_futures)
            .buffer_unordered(options.raptor_concurrency.max(1))
            .try_collect()
            .await?;

        level_pb.finish_with_message(format!(
            "{file_base_name} summarized {} clusters",
            level_summaries.len()
        ));

        if level_summaries.is_empty() {
            break;
        }

        // Embed all level summaries in one batched provider call (sync,
        // run on the blocking pool).
        let embed_pb = ProgressBar::new_spinner();
        embed_pb.set_style(
            ProgressStyle::default_spinner()
                .template("  {spinner:.green} RAPTOR L{prefix} embedding {msg}")
                .unwrap(),
        );
        embed_pb.set_prefix(format!("{summary_level}"));
        embed_pb.set_message(format!(
            "{file_base_name} ({} summaries)",
            level_summaries.len()
        ));
        embed_pb.enable_steady_tick(std::time::Duration::from_millis(100));

        let summary_texts: Vec<String> = level_summaries.iter().map(|s| s.text.clone()).collect();
        let provider_clone = Arc::clone(&provider);
        let arr: Array2<f32> = tokio::task::spawn_blocking(move || {
            provider_clone.compute_embeddings(&summary_texts, None)
        })
        .await??;
        embed_pb.finish_and_clear();

        let next_vectors: Vec<f32> = arr.iter().copied().collect();
        let next_texts: Vec<String> = level_summaries.iter().map(|s| s.text.clone()).collect();
        let next_ids: Vec<String> = level_summaries.iter().map(|s| s.id.clone()).collect();
        all_summaries.extend(level_summaries);

        if next_texts.len() <= 1 {
            break;
        }
        current_texts = next_texts;
        current_vectors = next_vectors;
        current_ids = next_ids;
    }
    drop(current_texts);
    drop(current_vectors);
    drop(current_ids);

    info!(
        filename = %filename,
        summaries = all_summaries.len(),
        "RAPTOR rebuild: originals + all summaries"
    );

    // ------------------------------------------------------------------
    // Final rebuild: originals + every summary from every level.
    // ------------------------------------------------------------------
    let mut builder = leann_core::LeannBuilder::new(&embedding_model, Some(dimensions), "spnl")
        .with_recompute(false)
        .with_compact(false);

    for (idx, id) in id_map.iter().enumerate() {
        if let Ok(p) = passage_mgr.get_passage_by_index(idx) {
            let mut metadata = HashMap::new();
            metadata.insert("id".to_string(), serde_json::Value::String(id.clone()));
            metadata.insert("raptor_level".to_string(), serde_json::Value::from(0));
            builder.add_text(&p.text, metadata);
        }
    }
    for s in &all_summaries {
        let mut metadata = HashMap::new();
        metadata.insert("id".to_string(), serde_json::Value::String(s.id.clone()));
        metadata.insert("raptor_level".to_string(), serde_json::Value::from(s.level));
        metadata.insert(
            "raptor_children".to_string(),
            serde_json::Value::Array(
                s.children
                    .iter()
                    .cloned()
                    .map(serde_json::Value::String)
                    .collect(),
            ),
        );
        builder.add_text(&s.text, metadata);
    }

    let rebuild_pb = ProgressBar::new_spinner();
    rebuild_pb.set_style(
        ProgressStyle::default_spinner()
            .template("  {spinner:.yellow} RAPTOR rebuild {msg} ({elapsed_precise})")
            .unwrap(),
    );
    rebuild_pb.set_message(format!(
        "{file_base_name} ({} originals + {} summaries)",
        id_map.len(),
        all_summaries.len()
    ));
    rebuild_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let index_path_clone = index_path.clone();
    let provider_clone = Arc::clone(&provider);
    tokio::task::spawn_blocking(move || builder.build_index(&index_path_clone, &*provider_clone))
        .await??;
    rebuild_pb.finish_with_message(format!("{file_base_name} done"));

    // Mark this corpus done — only after a successful phase-3 rebuild.
    let done_file = index_path.with_extension("ok");
    std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&done_file)?;

    Ok(())
}

/// Drive a single summary generation through the supplied `Summarizer`.
async fn generate_summary(
    summarizer: &dyn Summarizer,
    enclosing_model: &str,
    similar_texts: Vec<NonGenerateInput>,
    options: &AugmentOptions,
) -> Result<String> {
    use spnl_core::ir::GenerateMetadata;
    use spnl_core::optimizer::llo::llir::SingleGenerate;

    let spec = SingleGenerate {
        metadata: GenerateMetadata {
            model: enclosing_model.to_string(),
            max_tokens: Some(options.raptor_summary_max_tokens as i32),
            temperature: Some(options.raptor_summary_temperature),
        },
        input: NonGenerateInput::Cross(vec![
            NonGenerateInput::Message(Message::System("You are a helpful assistant.".into())),
            NonGenerateInput::Message(Message::User(
                "Write a summary of the following, including as many key details as possible:"
                    .into(),
            )),
            NonGenerateInput::Plus(similar_texts),
        ]),
    };

    summarizer.summarize(&spec).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use leann_core::embedding::EmbeddingProvider;
    use ndarray::Array2;
    use spnl_core::optimizer::llo::llir::SingleGenerate;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::TempDir;

    /// Deterministic mock embedder: derives a unit-norm dim=4 vector from
    /// the SHA-ish hash of each input text. Different texts → different
    /// vectors → real clustering work for k-means.
    struct MockProvider {
        dim: usize,
        calls: AtomicUsize,
    }

    impl EmbeddingProvider for MockProvider {
        fn compute_embeddings(
            &self,
            chunks: &[String],
            _progress: Option<&dyn leann_core::hnsw::IndexProgress>,
        ) -> anyhow::Result<Array2<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let n = chunks.len();
            let mut data = Vec::with_capacity(n * self.dim);
            for text in chunks {
                let mut h: u64 = 0xcbf29ce484222325;
                for b in text.bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                let mut v = vec![0.0f32; self.dim];
                for (i, slot) in v.iter_mut().enumerate() {
                    let bits = h.rotate_left((i * 17) as u32);
                    *slot = ((bits & 0xffff) as f32 / 32768.0) - 1.0;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                for x in &mut v {
                    *x /= norm;
                }
                data.extend_from_slice(&v);
            }
            Ok(Array2::from_shape_vec((n, self.dim), data)?)
        }

        fn dimensions(&self) -> usize {
            self.dim
        }

        fn name(&self) -> &str {
            "mock-embed"
        }
    }

    /// Mock summarizer: returns a synthetic summary that includes the
    /// number of inputs so we can sanity-check call shape, and a counter
    /// so we can assert how many summary calls happened.
    struct MockSummarizer {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl Summarizer for MockSummarizer {
        async fn summarize(&self, spec: &SingleGenerate) -> anyhow::Result<String> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            // Count Plus children at the bottom of the canonical raptor prompt.
            let inputs = match &spec.input {
                NonGenerateInput::Cross(items) => match items.last() {
                    Some(NonGenerateInput::Plus(p)) => p.len(),
                    _ => 0,
                },
                _ => 0,
            };
            Ok(format!("[summary #{n} of {inputs} items]"))
        }
    }

    /// Build a layer-1 leann index in `tmp` from `texts` using the mock
    /// provider, then return an `IndexedCorpus` ready for cross_index.
    fn build_layer1(tmp: &TempDir, texts: &[&str], provider: &MockProvider) -> IndexedCorpus {
        let dim = provider.dimensions();
        let index_dir = tmp.path().to_path_buf();
        let index_path = index_dir.join("test.leann");

        let mut builder = leann_core::LeannBuilder::new("mock-embed", Some(dim), "test")
            .with_recompute(false)
            .with_compact(false);
        for (i, t) in texts.iter().enumerate() {
            let mut md = HashMap::new();
            md.insert(
                "id".to_string(),
                serde_json::Value::String(format!("@base-test-{i}")),
            );
            builder.add_text(t, md);
        }
        builder.build_index(&index_path, provider).unwrap();

        IndexedCorpus {
            filename: "test-doc.txt".to_string(),
            index_path,
            enclosing_model: "mock-llm".to_string(),
            embedding_model: "mock-embed".to_string(),
            dimensions: dim,
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cross_index_builds_tree_with_metadata() {
        let tmp = TempDir::new().unwrap();
        let texts: Vec<&str> = (0..20)
            .map(|i| match i % 4 {
                0 => "alpha bravo charlie delta echo foxtrot",
                1 => "the quick brown fox jumps over the lazy dog",
                2 => "lorem ipsum dolor sit amet consectetur",
                _ => "indexing tree retrieval augmentation embedding",
            })
            .collect();
        // Make each text unique so the mock provider gives distinct vectors.
        let texts_owned: Vec<String> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| format!("{t} #{i}"))
            .collect();
        let texts_ref: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();

        let provider = Arc::new(MockProvider {
            dim: 4,
            calls: AtomicUsize::new(0),
        });
        let corpus = build_layer1(&tmp, &texts_ref, &provider);

        let summarizer = MockSummarizer {
            calls: AtomicUsize::new(0),
        };
        let mut options = AugmentOptions {
            index_dir: tmp.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        options.indexer = super::super::options::Indexer::Raptor;
        options.raptor_branching = 5; // 20 → 4 clusters → 1
        options.raptor_max_depth = 3;
        options.raptor_concurrency = 1;
        options.raptor_summary_max_tokens = 32;

        let provider_dyn: Arc<dyn EmbeddingProvider + Send + Sync> = provider.clone();
        cross_index_with_provider(corpus.clone(), &options, &summarizer, provider_dyn)
            .await
            .expect("cross_index should succeed");

        // Done-marker present.
        let done = corpus.index_path.with_extension("ok");
        assert!(done.exists(), "done marker should be written");

        // We expect at least one summary call (level 1) and probably a
        // level-2 root summary too.
        let summary_calls = summarizer.calls.load(Ordering::SeqCst);
        assert!(
            summary_calls >= 4,
            "expected ≥4 summary calls (level 1 with branching=5 → 4 clusters), got {summary_calls}"
        );

        // Re-open the rebuilt index and verify metadata.
        let paths = leann_core::index::IndexPaths::new(&corpus.index_path);
        let id_map = leann_core::passages::load_id_map(&paths.id_map_path()).unwrap();
        assert!(
            id_map.len() > texts_owned.len(),
            "rebuild should add summary passages: {} → {}",
            texts_owned.len(),
            id_map.len()
        );

        let source = leann_core::index::PassageSource {
            source_type: "jsonl".to_string(),
            path: paths.passages_path().to_string_lossy().to_string(),
            index_path: paths.offset_path().to_string_lossy().to_string(),
            path_relative: None,
            index_path_relative: None,
        };
        let pmgr = leann_core::passages::PassageManager::load(&[source], None).unwrap();

        let mut saw_level1 = false;
        let mut originals_with_level0 = 0;
        for i in 0..id_map.len() {
            let p = pmgr.get_passage_by_index(i).unwrap();
            let level = p
                .metadata
                .get("raptor_level")
                .and_then(|v| v.as_u64())
                .unwrap_or(99);
            if level == 0 {
                originals_with_level0 += 1;
            } else if level >= 1 {
                saw_level1 = true;
                let children = p
                    .metadata
                    .get("raptor_children")
                    .and_then(|v| v.as_array())
                    .expect("summary must carry raptor_children");
                assert!(!children.is_empty(), "summary children must be non-empty");
                // Every child id must resolve to an existing id_map entry.
                for c in children {
                    let cid = c.as_str().expect("child id must be string");
                    assert!(
                        id_map.iter().any(|s| s == cid),
                        "child id {cid} not in rebuilt id_map"
                    );
                }
            }
        }
        assert_eq!(
            originals_with_level0,
            texts_owned.len(),
            "all originals should be tagged level 0"
        );
        assert!(saw_level1, "at least one level-≥1 summary expected");
    }
}
