// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Thin wrappers converting CLI args into `vllm_gce` config structs.

use anyhow::Result;

use crate::args::{GceDownArgs, GceImageBuildArgs, GceImageListArgs, GceUpArgs};

pub async fn run_up(args: GceUpArgs) -> Result<()> {
    // Resolve project early — needed for image lookup.
    let project = args
        .project
        .clone()
        .or_else(|| std::env::var("GCP_PROJECT").ok())
        .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok());

    // Resolve image: explicit --image wins, otherwise auto-select latest.
    let image = match args.image {
        Some(img) => Some(img),
        None => {
            let proj = project.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "GCP project required to auto-select image: set --project, GCP_PROJECT, or GOOGLE_CLOUD_PROJECT"
                )
            })?;
            // --dev → dev images, otherwise prod images.
            let tag = if args.dev.is_some() {
                "vllm-rs-dev"
            } else {
                "vllm-rs-prod"
            };
            let latest = vllm_gce::image::latest_image(proj, tag).await?;
            match latest {
                Some(img) => {
                    eprintln!("  Auto-selected image: {img}");
                    Some(img)
                }
                None => {
                    anyhow::bail!(
                        "no READY image found with tag '{tag}' in project '{proj}'. \
                         Build one with: vllm gce image build <source_dir>"
                    );
                }
            }
        }
    };

    vllm_gce::up::up(vllm_gce::up::GceUpConfig {
        nodes: args.nodes,
        gpu_count: args.gpu_count,
        gpu_class: args.gpu_class,
        dev: args.dev.map(std::path::PathBuf::from),
        local_port: args.local_port,
        image,
        zone: args.zone,
        project,
        name: args.name,
        gcp_credentials: args.gcp_credentials,
        gcp_service_account: args.gcp_service_account,
        hf_token: args.hf_token,
        preemptible: args.preemptible,
        sccache_gcs_bucket: args.sccache_gcs_bucket,
        sccache_gcs_key_prefix: Some(args.sccache_gcs_key_prefix.unwrap_or_else(|| {
            let user = std::env::var("USER")
                .or_else(|_| std::env::var("USERNAME"))
                .unwrap_or_else(|_| "unknown".to_string());
            format!("vllm-rs-dev-{user}")
        })),
        model: args.model,
        serve_args: args.serve_args,
    })
    .await
}

pub async fn run_down(args: GceDownArgs) -> Result<()> {
    vllm_gce::down::down(vllm_gce::down::GceDownConfig {
        name: args.name,
        zone: args.zone,
        project: args.project,
        force: args.force,
    })
    .await
}

pub async fn run_image_build(args: GceImageBuildArgs) -> Result<()> {
    // Validate source directory.
    let source_dir = std::path::PathBuf::from(&args.source_dir);
    if !source_dir.is_dir() {
        anyhow::bail!("source directory '{}' does not exist", args.source_dir);
    }
    if !source_dir.join("Cargo.toml").exists() {
        anyhow::bail!(
            "source directory '{}' has no Cargo.toml — expected a Rust workspace",
            args.source_dir
        );
    }

    // Compute defaults based on --production.
    let prefix = if args.production { "prod" } else { "dev" };
    let tag = args.tag.unwrap_or_else(|| format!("vllm-rs-{prefix}"));
    let version = args.version.unwrap_or_else(|| {
        let now = chrono::Local::now();
        format!("{prefix}-{}", now.format("%Y%m%d-%H%M%S"))
    });

    vllm_gce::image::build(vllm_gce::image::ImageBuildConfig {
        image: args.image,
        tag,
        version,
        source_dir,
        production: args.production,
        project: args.project,
        gcp_credentials: args.gcp_credentials,
        gcp_service_account: args.gcp_service_account,
        sccache_gcs_bucket: args.sccache_gcs_bucket,
        sccache_gcs_key_prefix: Some(args.sccache_gcs_key_prefix.unwrap_or_else(|| {
            let user = std::env::var("USER")
                .or_else(|_| std::env::var("USERNAME"))
                .unwrap_or_else(|_| "unknown".to_string());
            format!("vllm-rs-dev-{user}")
        })),
    })
    .await
}

pub async fn run_image_list(args: GceImageListArgs) -> Result<()> {
    vllm_gce::image::list(vllm_gce::image::ImageListConfig {
        tag: args.tag,
        project: args.project,
        gcp_credentials: args.gcp_credentials,
    })
    .await
}
