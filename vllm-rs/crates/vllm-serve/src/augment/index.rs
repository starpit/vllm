use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use spnl_core::ir::{Augment, Document, Generate, GenerateMetadata, Query};
use tracing::info;

use super::embed::{HttpEmbeddingProvider, InProcessEmbeddingProvider};
use super::options::AugmentOptions;

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
pub async fn index(query: &Query, options: &AugmentOptions) -> Result<()> {
    let augments = extract_augments(query, &None);
    for augmentation in &augments {
        process_document(augmentation, options).await?;
    }
    Ok(())
}

/// Build a LEANN index for a single document if not already indexed.
async fn process_document(
    (_enclosing_model, a): &(String, Augment),
    options: &AugmentOptions,
) -> Result<()> {
    let (filename, content) = &a.doc;

    let index_name = sanitize_name(&format!(
        "default.{}.{filename}.SimpleEmbedRetrieve",
        a.embedding_model,
    ));
    let index_dir = PathBuf::from(&options.index_dir);
    let index_path = index_dir.join(format!("{index_name}.leann"));
    let done_file = index_dir.join(format!("{index_name}.ok"));

    if done_file.exists() {
        return Ok(());
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

    info!(
        filename,
        chunks = chunks.len(),
        "Indexing document for RAG augmentation"
    );

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
        tokio::task::spawn_blocking({
            let model = a.embedding_model.clone();
            move || HttpEmbeddingProvider::probe_dimensions(&model)
        })
        .await??
    };

    let mut builder = leann_core::LeannBuilder::new(&a.embedding_model, Some(dimensions), "spnl")
        .with_recompute(false)
        .with_compact(false);

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
        tokio::task::spawn_blocking(move || {
            let provider = HttpEmbeddingProvider::new(&embedding_model, dimensions);
            builder.build_index(&index_path_clone, &provider)
        })
        .await??;
    }

    // Mark as done
    std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&done_file)?;

    Ok(())
}
