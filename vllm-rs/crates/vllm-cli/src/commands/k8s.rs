// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Thin wrappers converting CLI args into `vllm_k8s` config structs.

use anyhow::Result;

use crate::args::{K8sDownArgs, K8sPreloadArgs, K8sUpArgs};

pub async fn run_up(args: K8sUpArgs) -> Result<()> {
    vllm_k8s::up::up(vllm_k8s::up::K8sUpConfig {
        name: args.name,
        namespace: args.namespace,
        nodes: args.nodes,
        gpu_count: args.gpu_count,
        image: args.image,
        model: args.model,
        hf_token: args.hf_token,
        local_port: args.local_port,
        serve_args: args.serve_args,
        shm_size: args.shm_size,
        preload_pvc: args.preload,
    })
    .await
}

pub async fn run_preload(args: K8sPreloadArgs) -> Result<()> {
    vllm_k8s::preload::preload(vllm_k8s::preload::K8sPreloadConfig {
        name: args.pvc_name,
        namespace: args.namespace,
        models: args.models,
        size: args.size,
        storage_class: args.storage_class,
        image: args.image,
        hf_token: args.hf_token,
    })
    .await
}

pub async fn run_down(args: K8sDownArgs) -> Result<()> {
    vllm_k8s::down::down(vllm_k8s::down::K8sDownConfig {
        name: args.name,
        namespace: args.namespace,
        force: args.force,
    })
    .await
}
