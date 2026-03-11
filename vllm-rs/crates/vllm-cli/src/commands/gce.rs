// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Thin wrappers converting CLI args into `vllm_gce` config structs.

use anyhow::Result;

use crate::args::{GceDownArgs, GceUpArgs};

pub async fn run_up(args: GceUpArgs) -> Result<()> {
    vllm_gce::up::up(vllm_gce::up::GceUpConfig {
        nodes: args.nodes,
        gpu_count: args.gpu_count,
        gpu_class: args.gpu_class,
        dev: args.dev.map(std::path::PathBuf::from),
        local_port: args.local_port,
        image: args.image,
        zone: args.zone,
        project: args.project,
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
