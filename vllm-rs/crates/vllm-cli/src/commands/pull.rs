// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm pull` — download a model from HuggingFace Hub.

use std::path::{Path, PathBuf};

use tracing::info;

use crate::args::PullArgs;

pub async fn run_pull(args: PullArgs) -> anyhow::Result<()> {
    let model = &args.model;
    let path = Path::new(model);

    // Local .gguf file — nothing to download.
    if path.is_file() && path.extension().is_some_and(|e| e == "gguf") {
        println!("Model already available locally: {}", path.display());
        return Ok(());
    }

    // Local directory — nothing to download.
    if path.is_dir() {
        println!("Model already available locally: {}", path.display());
        return Ok(());
    }

    // Download from HuggingFace Hub.
    let model = model.clone();
    let hf_token = args.hf_token.clone();
    let gguf_file = args.gguf_file.clone();
    let quantization = args.quantization.clone();

    let result_path = tokio::task::spawn_blocking(move || {
        download_model(
            &model,
            hf_token.as_deref(),
            gguf_file.as_deref(),
            quantization.as_deref(),
        )
    })
    .await??;

    println!("Model downloaded to: {}", result_path.display());
    Ok(())
}

fn download_model(
    model_id: &str,
    hf_token: Option<&str>,
    gguf_file: Option<&str>,
    quantization: Option<&str>,
) -> anyhow::Result<PathBuf> {
    info!("Downloading model from HuggingFace Hub: {model_id}");

    let mut builder = hf_hub::api::sync::ApiBuilder::from_env();
    if let Some(token) = hf_token {
        builder = builder.with_token(Some(token.to_string()));
    }
    let api = builder.build()?;
    let repo = api.model(model_id.to_string());

    // GGUF download: explicit filename, --quantization match, or auto-detect.
    let gguf_filename = gguf_file.map(String::from).or_else(|| {
        // If --quantization is set, always try to find a matching GGUF file.
        // Otherwise, only auto-detect for repos with "GGUF" in the name.
        if quantization.is_none() && !model_id.to_ascii_uppercase().contains("GGUF") {
            return None;
        }
        let info = repo.info().ok()?;
        let mut gguf_files: Vec<_> = info
            .siblings
            .iter()
            .filter(|s| s.rfilename.ends_with(".gguf"))
            .collect();
        if gguf_files.is_empty() {
            return None;
        }
        // If user specified a quantization, match case-insensitively.
        if let Some(quant) = quantization {
            let quant_upper = quant.to_ascii_uppercase();
            if let Some(f) = gguf_files
                .iter()
                .find(|s| s.rfilename.to_ascii_uppercase().contains(&quant_upper))
            {
                return Some(f.rfilename.clone());
            }
            // No match — warn but fall through to default selection.
            eprintln!("Warning: no GGUF file matching '{quant}' found, using default selection");
        }
        for pattern in &["Q4_K_M", "Q4_K_S", "Q4_K", "Q4_0", "Q8_0"] {
            if let Some(f) = gguf_files.iter().find(|s| s.rfilename.contains(pattern)) {
                return Some(f.rfilename.clone());
            }
        }
        gguf_files.sort_by(|a, b| a.rfilename.cmp(&b.rfilename));
        Some(gguf_files[0].rfilename.clone())
    });

    if let Some(ref gguf_file) = gguf_filename {
        info!("Downloading GGUF file: {gguf_file}");
        let gguf_path = repo.get(gguf_file)?;
        // Best-effort tokenizer download.
        let _ = repo.get("tokenizer.json");
        let _ = repo.get("tokenizer_config.json");
        return Ok(gguf_path);
    }

    // Safetensors model.
    let config_path = repo.get("config.json")?;
    let model_dir = config_path.parent().unwrap().to_path_buf();

    // Best-effort tokenizer download.
    let _ = repo.get("tokenizer.json");
    let _ = repo.get("tokenizer_config.json");

    // Single file model.
    if repo.get("model.safetensors").is_ok() {
        return Ok(model_dir);
    }

    // Sharded model.
    if let Ok(index_path) = repo.get("model.safetensors.index.json") {
        let index = vllm_model::weight::SafeTensorsIndex::from_file(&index_path)?;
        let sorted_shards = index.shard_files();
        let total = sorted_shards.len();

        let needed: Vec<&String> = sorted_shards
            .iter()
            .filter(|s| !model_dir.join(s).exists())
            .collect();

        if needed.is_empty() {
            info!("All {total} shard files already cached");
        } else {
            info!(
                "Downloading {} of {total} shard files (up to 8 in parallel)",
                needed.len()
            );

            let multi = indicatif::MultiProgress::new();
            const MAX_PARALLEL: usize = 8;
            let repo = &repo;
            let multi = &multi;

            for chunk in needed.chunks(MAX_PARALLEL) {
                let results: Vec<anyhow::Result<()>> = std::thread::scope(|s| {
                    let handles: Vec<_> = chunk
                        .iter()
                        .map(|shard| {
                            let bar = multi.add(indicatif::ProgressBar::new(0));
                            s.spawn(move || {
                                repo.download_with_progress(shard, bar)
                                    .map(|_| ())
                                    .map_err(|e| anyhow::anyhow!("failed to download {shard}: {e}"))
                            })
                        })
                        .collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                for result in results {
                    result?;
                }
            }
        }
        return Ok(model_dir);
    }

    anyhow::bail!("no safetensors weights found for {model_id}");
}
