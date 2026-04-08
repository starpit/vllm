use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};
use leann_core::hnsw::IndexProgress;
use spnl_core::ir::{Augment, Document, Generate, GenerateMetadata, Query};
use tracing::info;

use super::embed::{HttpEmbeddingProvider, InProcessEmbeddingProvider};
use super::options::{AugmentOptions, Indexer};

/// Metadata about a layer-1 indexed corpus, returned by `process_document_layer1`
/// so RAPTOR phase 2/3 can re-open and rebuild the index.
#[derive(Clone)]
pub(crate) struct IndexedCorpus {
    pub filename: String,
    pub index_path: PathBuf,
    pub enclosing_model: String,
    pub embedding_model: String,
    pub dimensions: usize,
}

/// Adapts an indicatif `ProgressBar` to the `BuildProgress` trait.
///
/// The bar starts as a spinner (during embedding computation) and switches
/// to a determinate progress bar once the HNSW build begins.
struct IndicatifProgress {
    pb: ProgressBar,
    filename: String,
}

impl IndexProgress for IndicatifProgress {
    fn phase(&self, name: &str, total: usize) {
        if total > 0 {
            self.pb.set_style(
                ProgressStyle::default_bar()
                    .template("  {spinner:.green} {msg} [{bar:30.cyan/blue}] {pos}/{len}")
                    .unwrap()
                    .progress_chars("█▉▊▋▌▍▎▏ "),
            );
            self.pb.set_length(total as u64);
            self.pb.set_position(0);
        } else {
            self.pb.set_style(
                ProgressStyle::default_spinner()
                    .template("  {spinner:.green} {msg}")
                    .unwrap(),
            );
        }
        let label = match name {
            "embedding" => "Embedding",
            "building" => "Building index for",
            _ => name,
        };
        self.pb.set_message(format!("{label} {}", self.filename));
    }

    fn progress(&self, completed: usize) {
        self.pb.set_position(completed as u64);
    }
}

/// Sanitize a name for use as a filesystem path component.
fn sanitize_name(name: &str) -> String {
    name.replace(['/', ':'], "_")
}

/// Walk the query tree and collect all `Augment` nodes with their enclosing model.
fn extract_augments(query: &Query, enclosing_model: &Option<String>) -> Vec<(String, Augment)> {
    match (query, enclosing_model) {
        (
            Query::Generate(Generate {
                input,
                metadata: GenerateMetadata { model, .. },
            }),
            _,
        ) => extract_augments(input, &Some(model.clone())),
        (Query::Plus(v) | Query::Cross(v) | Query::Seq(v), _) => v
            .iter()
            .flat_map(|q| extract_augments(q, enclosing_model))
            .collect(),
        (Query::Augment(a), Some(m)) => vec![(m.clone(), a.clone())],
        _ => vec![],
    }
}

/// Scan the query for `Augment` nodes and build LEANN indexes for any that
/// don't already have one on disk.
///
/// When `options.indexer == Indexer::Raptor`, `options.summarizer` must be
/// set; the server path populates it with an `AppStateSummarizer` that
/// drives generation through the in-process async engine.
pub async fn index(query: &Query, options: &AugmentOptions) -> Result<()> {
    let augments = extract_augments(query, &None);
    for augmentation in &augments {
        let corpus = process_document_layer1(augmentation, options).await?;
        if let (Indexer::Raptor, Some(corpus)) = (options.indexer, corpus) {
            let summarizer = options.summarizer.as_ref().ok_or_else(|| {
                anyhow!(
                    "RAPTOR indexing requires a summarizer; the offline LLM path does not yet support it"
                )
            })?;
            super::raptor::cross_index(corpus, options, summarizer.as_ref()).await?;
        }
    }
    Ok(())
}

/// Build a layer-1 LEANN index for a single document. Returns `Some(corpus)`
/// describing the resulting index (used by RAPTOR for cross-indexing), or
/// `None` if the index was already present on disk.
async fn process_document_layer1(
    (enclosing_model, a): &(String, Augment),
    options: &AugmentOptions,
) -> Result<Option<IndexedCorpus>> {
    let (filename, content) = &a.doc;

    let index_name = sanitize_name(&format!(
        "default.{}.{filename}.{:?}",
        a.embedding_model, options.indexer,
    ));
    let index_dir = PathBuf::from(&options.index_dir);
    let index_path = index_dir.join(format!("{index_name}.leann"));
    let done_file = index_dir.join(format!("{index_name}.ok"));

    if done_file.exists() {
        // Already fully indexed. For raptor this means cross_index has
        // completed — its done-marker is written *after* phase 3 rebuild,
        // so a present marker is the authoritative "skip everything"
        // signal regardless of indexer mode. Returning `None` here
        // prevents the caller from re-running cross_index per query
        // (which would re-cluster the rebuilt tree and grow it without
        // bound).
        return Ok(None);
    }

    // Extract text from document
    let text = match content {
        Document::Text(t) => t.clone(),
        _ => return Err(anyhow!("Unsupported document type for: {filename}")),
    };

    // Chunk
    let chunks = leann_core::chunking::chunk_text(&text, options.chunk_size, options.chunk_overlap);
    if chunks.is_empty() {
        return Err(anyhow!("No chunks produced from document: {filename}"));
    }

    let n_chunks = chunks.len();
    info!(
        filename,
        chunks = n_chunks,
        "Indexing document for RAG augmentation"
    );

    // Start as a spinner during embedding computation; switches to a
    // progress bar once the HNSW build begins (via BuildProgress::started).
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("  {spinner:.green} Indexing {msg}")
            .unwrap(),
    );
    pb.set_message(format!("{filename} ({n_chunks} chunks)"));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    let progress = std::sync::Arc::new(IndicatifProgress {
        pb: pb.clone(),
        filename: filename.clone(),
    });

    let file_base_name = std::path::Path::new(filename)
        .file_name()
        .ok_or(anyhow!("Could not determine base name"))?
        .to_string_lossy()
        .to_string();

    // Build LEANN index — choose in-process or HTTP provider
    let use_inprocess = options.can_embed_in_process(&a.embedding_model);

    let dimensions = if use_inprocess {
        let embedder = options.embedder.clone().unwrap();
        let tokenizer = options.tokenizer.clone().unwrap();
        tokio::task::spawn_blocking(move || {
            InProcessEmbeddingProvider::probe_dimensions(&*embedder, &tokenizer)
        })
        .await??
    } else {
        let base_url = resolve_embedding_base_url(&a.embedding_model, options)?;
        tokio::task::spawn_blocking({
            let model = a.embedding_model.clone();
            move || {
                let provider = HttpEmbeddingProvider::with_base_url(&model, 0, base_url);
                let resp = provider.call_api(&["probe".to_string()])?;
                resp.first()
                    .map(|v| v.len())
                    .ok_or_else(|| anyhow!("probe returned no embeddings"))
            }
        })
        .await??
    };

    let mut builder = leann_core::LeannBuilder::new(&a.embedding_model, Some(dimensions), "spnl")
        .with_recompute(false)
        .with_compact(false)
        .with_progress(progress);

    for (idx, chunk) in chunks.iter().enumerate() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "id".to_string(),
            serde_json::Value::String(format!("@base-{file_base_name}-{idx}")),
        );
        builder.add_text(chunk, metadata);
    }

    std::fs::create_dir_all(&index_dir)?;

    let embedding_model = a.embedding_model.clone();
    let index_path_clone = index_path.clone();

    if use_inprocess {
        let embedder = options.embedder.clone().unwrap();
        let tokenizer = options.tokenizer.clone().unwrap();
        tokio::task::spawn_blocking(move || {
            let provider =
                InProcessEmbeddingProvider::new(embedding_model, dimensions, embedder, tokenizer);
            builder.build_index(&index_path_clone, &provider)
        })
        .await??;
    } else {
        let base_url = resolve_embedding_base_url(&embedding_model, options)?;
        tokio::task::spawn_blocking(move || {
            let provider =
                HttpEmbeddingProvider::with_base_url(&embedding_model, dimensions, base_url);
            builder.build_index(&index_path_clone, &provider)
        })
        .await??;
    }

    pb.finish_and_clear();

    // For RAPTOR we defer marking the corpus done until phase 3 rebuild
    // completes; that way a crash mid-cross-index re-runs cleanly.
    if options.indexer == Indexer::Layer1 {
        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&done_file)?;
    }

    Ok(Some(IndexedCorpus {
        filename: filename.clone(),
        index_path,
        enclosing_model: enclosing_model.clone(),
        embedding_model: a.embedding_model.clone(),
        dimensions,
    }))
}

/// Resolve the base URL for an embedding model: use a sidecar if available,
/// otherwise fall back to the env-var default.
pub(crate) fn resolve_embedding_base_url(model: &str, options: &AugmentOptions) -> Result<String> {
    if let Some(mgr) = &options.sidecar_manager {
        let sidecar = mgr.get_or_spawn(model)?;
        Ok(sidecar.base_url.clone())
    } else {
        Ok(std::env::var("VLLM_EMBEDDING_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string()))
    }
}
