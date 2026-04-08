use std::path::PathBuf;

use anyhow::Result;
use itertools::Itertools;
use spnl_core::ir::{Document, Query};

use leann_core::hnsw::{
    graph::VectorStorage,
    io::read_hnsw_index,
    search::{SearchParams, search_hnsw},
};
use leann_core::index::IndexPaths;
use leann_core::passages::{PassageManager, load_id_map};

use std::collections::{HashMap, HashSet, VecDeque};

use super::embed::{HttpEmbeddingProvider, contentify};
use super::options::{AugmentOptions, Indexer};

/// Sanitize a name for use as a filesystem path component.
fn sanitize_name(name: &str) -> String {
    name.replace(['/', ':'], "_")
}

/// Retrieve relevant document fragments using HNSW vector search.
///
/// Returns formatted strings like `"Relevant Document @base-doc.txt-3: <text>"`.
pub async fn retrieve(
    embedding_model: &str,
    body: &Query,
    (filename, _content): &(String, Document),
    options: &AugmentOptions,
) -> Result<Vec<String>> {
    let max_matches = options.max_aug;

    // Derive the index path (must match index.rs naming).
    let index_name = sanitize_name(&format!(
        "default.{embedding_model}.{filename}.{:?}",
        options.indexer,
    ));
    let index_dir = PathBuf::from(&options.index_dir);
    let index_path = index_dir.join(format!("{index_name}.leann"));
    let paths = IndexPaths::new(&index_path);

    // Load HNSW graph from disk
    let mut index_file = std::fs::File::open(paths.index_file_path())?;
    let graph = read_hnsw_index(&mut index_file)?;

    // Extract stored vectors
    let stored_vectors: Vec<f32> = match &graph.vector_storage {
        VectorStorage::Raw { data, .. } => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        VectorStorage::Null => {
            anyhow::bail!("HNSW index has no stored vectors; was it built with is_recompute=false?")
        }
    };

    // Load passage manager and ID map
    let passage_source = leann_core::index::PassageSource {
        source_type: "jsonl".to_string(),
        path: paths.passages_path().to_string_lossy().to_string(),
        index_path: paths.offset_path().to_string_lossy().to_string(),
        path_relative: None,
        index_path_relative: None,
    };
    let passages = PassageManager::load(&[passage_source], None)?;
    let id_map = load_id_map(&paths.id_map_path())?;

    // Embed the query body — in-process if model matches, otherwise HTTP
    let body_texts = contentify(body);
    let body_vectors: Vec<Vec<f32>> = if options.can_embed_in_process(embedding_model) {
        let embedder = options.embedder.clone().unwrap();
        let tokenizer = options.tokenizer.clone().unwrap();
        tokio::task::spawn_blocking(move || {
            let token_id_seqs: Vec<Vec<u32>> = body_texts
                .iter()
                .map(|text| tokenizer.encode(text, false))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
            embedder.embed_tokens(token_id_seqs)
        })
        .await??
    } else {
        let base_url = resolve_embedding_base_url(embedding_model, options)?;
        let embedding_model_owned = embedding_model.to_string();
        tokio::task::spawn_blocking(move || {
            let provider =
                HttpEmbeddingProvider::with_base_url(&embedding_model_owned, 0, base_url);
            provider.call_api(&body_texts)
        })
        .await??
    };

    // Search for each query vector. For RAPTOR indexes we additionally
    // expand any retrieved summary node down to its leaves so the LLM
    // sees the underlying source text, not just the summary.
    let params = SearchParams::default();
    let matching_labels: Vec<usize> = body_vectors
        .into_iter()
        .flat_map(|query_vec| {
            let (labels, _distances) =
                search_hnsw(&graph, &query_vec, max_matches, &stored_vectors, &params);
            labels.into_iter()
        })
        .unique()
        .collect();

    // Default for raptor indexes is canonical "collapsed tree" retrieval:
    // return summaries and originals together as scored by HNSW. Tree
    // expansion (replacing each summary hit by its leaves) is opt-in via
    // `VLLM_RAG_RAPTOR_EXPAND_HITS=1` because for fact-retrieval QA the
    // matched summary is itself the strongest signal — replacing it with
    // its similarity-clustered leaves dilutes the result set.
    let expanded_labels: Vec<usize> =
        if options.indexer == Indexer::Raptor && options.raptor_expand_hits {
            expand_raptor_hits(&matching_labels, &passages, &id_map)?
        } else {
            matching_labels
        };

    // Resolve passage text — reversed so most relevant is closest to query
    let fragments: Vec<String> = expanded_labels
        .into_iter()
        .rev()
        .filter_map(|label| {
            let id = id_map.get(label).map(|s| s.as_str()).unwrap_or("?");
            passages
                .get_passage_by_index(label)
                .ok()
                .map(|p| format!("Relevant Document {id}: {}", p.text))
        })
        .collect();

    Ok(fragments)
}

/// Tree-aware expansion for RAPTOR indexes. Each retrieved label whose
/// passage is a level-≥1 summary is replaced by its descendant leaves
/// (level 0 originals). Level-0 hits pass through. Insertion order is
/// preserved so the most-relevant hit's leaves come first.
///
/// Falls back to the raw HNSW labels if the index has no `raptor_children`
/// metadata (e.g. an older index built before metadata tracking landed),
/// preserving correct behavior on legacy data.
fn expand_raptor_hits(
    labels: &[usize],
    passages: &PassageManager,
    id_map: &[String],
) -> Result<Vec<usize>> {
    // Build id → row index once. Avoids quadratic scans on the
    // ancestor → descendant walk.
    let id_to_row: HashMap<&str, usize> = id_map
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let mut out: Vec<usize> = Vec::with_capacity(labels.len() * 4);
    let mut seen: HashSet<usize> = HashSet::new();

    for &start in labels {
        // BFS down the children pointer chain. We treat any node lacking
        // `raptor_children` (or with `raptor_level == 0`) as a leaf.
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(start);
        while let Some(row) = queue.pop_front() {
            let p = match passages.get_passage_by_index(row) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let level = p
                .metadata
                .get("raptor_level")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let children = p
                .metadata
                .get("raptor_children")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            if level == 0 || children.is_empty() {
                if seen.insert(row) {
                    out.push(row);
                }
                continue;
            }
            for child in children {
                if let Some(child_id) = child.as_str()
                    && let Some(&child_row) = id_to_row.get(child_id)
                {
                    queue.push_back(child_row);
                }
            }
        }
    }

    if out.is_empty() {
        // No metadata at all (legacy index): degrade gracefully.
        return Ok(labels.to_vec());
    }
    Ok(out)
}

/// Resolve the base URL for an embedding model: use a sidecar if available,
/// otherwise fall back to the env-var default.
fn resolve_embedding_base_url(model: &str, options: &AugmentOptions) -> Result<String> {
    if let Some(mgr) = &options.sidecar_manager {
        let sidecar = mgr.get_or_spawn(model)?;
        Ok(sidecar.base_url.clone())
    } else {
        Ok(std::env::var("VLLM_EMBEDDING_BASE_URL")
            .unwrap_or_else(|_| "http://localhost:11434/v1".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use leann_core::passages::{Passage, write_id_map, write_passages};
    use std::collections::HashMap as StdHashMap;
    use tempfile::TempDir;

    /// Materialize a tiny on-disk passage corpus from `(id, text, metadata)`
    /// triples and return a `(PassageManager, Vec<id>)` ready for the
    /// expansion logic. Mirrors what `LeannBuilder.build_index` would write.
    fn make_passages(
        items: &[(&str, &str, StdHashMap<String, serde_json::Value>)],
    ) -> (PassageManager, Vec<String>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let pas_path = tmp.path().join("test.passages.jsonl");
        let off_path = tmp.path().join("test.passages.idx");
        let id_path = tmp.path().join("test.ids.txt");

        let passages: Vec<Passage> = items
            .iter()
            .map(|(id, text, md)| Passage {
                id: id.to_string(),
                text: text.to_string(),
                metadata: md.clone(),
            })
            .collect();
        write_passages(&passages, &pas_path, &off_path).unwrap();

        let ids: Vec<String> = items.iter().map(|(id, _, _)| id.to_string()).collect();
        write_id_map(&ids, &id_path).unwrap();

        let source = leann_core::index::PassageSource {
            source_type: "jsonl".to_string(),
            path: pas_path.to_string_lossy().to_string(),
            index_path: off_path.to_string_lossy().to_string(),
            path_relative: None,
            index_path_relative: None,
        };
        let mgr = PassageManager::load(&[source], None).unwrap();
        (mgr, ids, tmp)
    }

    fn leaf(id: &str) -> (&str, &str, StdHashMap<String, serde_json::Value>) {
        let mut md = StdHashMap::new();
        md.insert("id".to_string(), serde_json::Value::String(id.to_string()));
        md.insert("raptor_level".to_string(), serde_json::Value::from(0u64));
        (id, "leaf-text", md)
    }

    fn summary(
        id: &'static str,
        level: u64,
        children: &[&str],
    ) -> (
        &'static str,
        &'static str,
        StdHashMap<String, serde_json::Value>,
    ) {
        let mut md = StdHashMap::new();
        md.insert("id".to_string(), serde_json::Value::String(id.to_string()));
        md.insert("raptor_level".to_string(), serde_json::Value::from(level));
        md.insert(
            "raptor_children".to_string(),
            serde_json::Value::Array(
                children
                    .iter()
                    .map(|s| serde_json::Value::String(s.to_string()))
                    .collect(),
            ),
        );
        (id, "summary-text", md)
    }

    #[test]
    fn level_0_hit_passes_through() {
        let items = vec![leaf("@base-doc-0"), leaf("@base-doc-1")];
        let (mgr, ids, _tmp) = make_passages(&items);
        let out = expand_raptor_hits(&[0, 1], &mgr, &ids).unwrap();
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn level_1_summary_expands_to_leaves() {
        let items = vec![
            leaf("@base-doc-0"),
            leaf("@base-doc-1"),
            leaf("@base-doc-2"),
            summary("@raptor-doc-L1-C0", 1, &["@base-doc-0", "@base-doc-1"]),
        ];
        let (mgr, ids, _tmp) = make_passages(&items);
        // Hit the summary at row 3.
        let out = expand_raptor_hits(&[3], &mgr, &ids).unwrap();
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn level_2_summary_expands_transitively() {
        let items = vec![
            leaf("@base-doc-0"),
            leaf("@base-doc-1"),
            leaf("@base-doc-2"),
            leaf("@base-doc-3"),
            summary("@raptor-doc-L1-C0", 1, &["@base-doc-0", "@base-doc-1"]),
            summary("@raptor-doc-L1-C1", 1, &["@base-doc-2", "@base-doc-3"]),
            summary(
                "@raptor-doc-L2-C0",
                2,
                &["@raptor-doc-L1-C0", "@raptor-doc-L1-C1"],
            ),
        ];
        let (mgr, ids, _tmp) = make_passages(&items);
        let out = expand_raptor_hits(&[6], &mgr, &ids).unwrap();
        // BFS order may interleave but the set must be {0,1,2,3}.
        let mut got = out.clone();
        got.sort();
        assert_eq!(got, vec![0, 1, 2, 3]);
    }

    #[test]
    fn duplicate_leaves_dedupe() {
        let items = vec![
            leaf("@base-doc-0"),
            leaf("@base-doc-1"),
            summary("@raptor-doc-L1-C0", 1, &["@base-doc-0", "@base-doc-1"]),
            summary("@raptor-doc-L1-C1", 1, &["@base-doc-0", "@base-doc-1"]),
        ];
        let (mgr, ids, _tmp) = make_passages(&items);
        let out = expand_raptor_hits(&[2, 3], &mgr, &ids).unwrap();
        assert_eq!(out.len(), 2, "leaves should dedupe across summaries");
        let mut sorted = out.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1]);
    }

    #[test]
    fn order_preserved_most_relevant_first() {
        let items = vec![
            leaf("@base-doc-A"),
            leaf("@base-doc-B"),
            leaf("@base-doc-C"),
            leaf("@base-doc-D"),
            summary("@raptor-doc-L1-C0", 1, &["@base-doc-A", "@base-doc-B"]),
            summary("@raptor-doc-L1-C1", 1, &["@base-doc-C", "@base-doc-D"]),
        ];
        let (mgr, ids, _tmp) = make_passages(&items);
        // Hit C0 first, then C1. C0's leaves must come before C1's.
        let out = expand_raptor_hits(&[4, 5], &mgr, &ids).unwrap();
        let pos_a = out.iter().position(|&i| i == 0).unwrap();
        let pos_d = out.iter().position(|&i| i == 3).unwrap();
        assert!(pos_a < pos_d, "C0's leaves must precede C1's");
    }

    #[test]
    fn legacy_index_falls_back() {
        // No raptor_level, no raptor_children — pre-metadata index. Build one
        // passage with empty metadata so the BFS-walk can't find leaves.
        let items: Vec<(&str, &str, StdHashMap<String, serde_json::Value>)> =
            vec![("legacy-0", "old", StdHashMap::new())];
        let (mgr, ids, _tmp) = make_passages(&items);
        let out = expand_raptor_hits(&[0], &mgr, &ids).unwrap();
        // Level defaults to 0 → treated as a leaf, passes through.
        assert_eq!(out, vec![0]);
    }
}
