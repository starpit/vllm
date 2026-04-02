// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm k8s up` — deploy vLLM to Kubernetes with optional multi-node TP.

use anyhow::Result;
use futures::{AsyncBufReadExt, StreamExt, TryStreamExt};
use k8s_openapi::api::apps::v1::{Deployment, StatefulSet};
use k8s_openapi::api::core::v1::{Pod, Service};
use kube::ResourceExt;
use kube::api::{ListParams, LogParams, PostParams};
use kube::runtime::wait::{await_condition, conditions::is_deployment_completed};

use crate::manifest::{self, ManifestConfig, Manifests};

/// Default container image for vLLM-rs.
pub const DEFAULT_IMAGE: &str = "quay.io/starpit/vllmrs:dev-20260402-cuda12.9.1-sm89";

/// Configuration for `vllm k8s up`.
pub struct K8sUpConfig {
    /// Resource name (used for Deployment/StatefulSet and teardown).
    pub name: String,
    /// Kubernetes namespace (None = default).
    pub namespace: Option<String>,
    /// Number of nodes (>1 = multi-node TP with StatefulSet).
    pub nodes: u32,
    /// Number of GPUs per node.
    pub gpu_count: u32,
    /// Container image.
    pub image: String,
    /// Model to serve (HuggingFace model ID or path).
    pub model: String,
    /// HuggingFace token.
    pub hf_token: Option<String>,
    /// Local port for port forwarding.
    pub local_port: u16,
    /// Extra arguments for `vllm serve`.
    pub serve_args: Vec<String>,
    /// Shared memory size (default "16Gi").
    pub shm_size: String,
    /// PVC name for preloaded models (mounted read-only).
    pub preload_pvc: Option<String>,
}

/// Deploy vLLM to Kubernetes.
pub async fn up(config: K8sUpConfig) -> Result<()> {
    let c = crate::client().await?;
    let instance_id = uuid::Uuid::new_v4().to_string();
    let name = config.name.clone();
    let namespace = config.namespace.clone();
    let local_port = config.local_port;

    let manifest_config = ManifestConfig {
        name: config.name.clone(),
        instance_id: instance_id.clone(),
        namespace: config.namespace.clone(),
        nodes: config.nodes,
        gpu_count: config.gpu_count,
        image: config.image,
        model: config.model,
        hf_token: config.hf_token,
        serve_args: config.serve_args,
        shm_size: config.shm_size,
        preload_pvc: config.preload_pvc,
    };

    let manifests = manifest::build(manifest_config);

    // Create resources.
    match &manifests {
        Manifests::SingleNode { deployment } => {
            eprintln!("  Creating Deployment '{name}'...");
            let api = crate::api::<Deployment>(c.clone(), &namespace)?;
            api.create(&PostParams::default(), deployment).await?;
        }
        Manifests::MultiNode {
            service,
            stateful_set,
        } => {
            eprintln!(
                "  Creating headless Service '{}-headless' + StatefulSet '{name}' ({} nodes)...",
                name,
                stateful_set
                    .spec
                    .as_ref()
                    .and_then(|s| s.replicas)
                    .unwrap_or(0)
            );
            let svc_api = crate::api::<Service>(c.clone(), &namespace)?;
            svc_api.create(&PostParams::default(), service).await?;
            let ss_api = crate::api::<StatefulSet>(c.clone(), &namespace)?;
            ss_api.create(&PostParams::default(), stateful_set).await?;
        }
    }

    let pod_api = crate::api::<Pod>(c.clone(), &namespace)?;
    let label_selector =
        format!("app.kubernetes.io/name={name},app.kubernetes.io/instance={instance_id}");

    // Wait for pods to appear.
    eprintln!("  Waiting for pods...");
    let pods = loop {
        let names = pod_api
            .list(&ListParams::default().labels(&label_selector))
            .await?
            .items
            .into_iter()
            .map(|pod| pod.name_any())
            .collect::<Vec<_>>();
        if !names.is_empty() {
            break names;
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    };
    eprintln!("  Found {} pod(s): {}", pods.len(), pods.join(", "));

    // Stream logs from all pods.
    let log_handles: Vec<tokio::task::JoinHandle<_>> = pods
        .into_iter()
        .map(|pod_name| {
            let api = pod_api.clone();
            tokio::spawn(async move {
                stream_pod_logs(&api, &pod_name).await;
            })
        })
        .collect();

    // For Deployments, wait for completion; for StatefulSets, wait for pods Ready.
    match &manifests {
        Manifests::SingleNode { .. } => {
            let dep_api = crate::api::<Deployment>(c.clone(), &namespace)?;
            eprintln!("  Awaiting deployment completion...");
            loop {
                if await_condition(dep_api.clone(), &name, is_deployment_completed())
                    .await
                    .is_ok()
                {
                    break;
                }
            }
            eprintln!("  READY");
        }
        Manifests::MultiNode { .. } => {
            eprintln!("  Waiting for all pods to be Ready...");
            wait_all_pods_ready(&pod_api, &label_selector).await?;
            eprintln!("  All {} pods READY", config.nodes);
        }
    }

    // Port-forward to rank-0 pod.
    let ready_pod = find_ready_pod(&pod_api, &label_selector).await?;
    eprintln!(
        "  Port forwarding: localhost:{} -> {}:8000",
        local_port, ready_pod
    );

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], local_port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    let pf_pod_api = pod_api.clone();
    let pf_pod_name = ready_pod.clone();
    tokio::spawn(async move {
        let server = tokio_stream::wrappers::TcpListenerStream::new(listener)
            .take_until(tokio::signal::ctrl_c())
            .try_for_each(|client_conn| {
                let api = pf_pod_api.clone();
                let pod = pf_pod_name.clone();
                async move {
                    tokio::spawn(async move {
                        if let Err(e) = forward_connection(&api, &pod, 8000, client_conn).await {
                            eprintln!("  Port-forward error: {e}");
                        }
                    });
                    Ok(())
                }
            });

        if let Err(e) = server.await {
            eprintln!("  Port forwarding error: {e}");
        }
    });

    // Wait for logs or ctrl-c.
    tokio::select! {
        result = futures::future::try_join_all(log_handles) => {
            result?;
        }
        _ = tokio::signal::ctrl_c() => {
            eprintln!("\n  Received Ctrl+C, shutting down...");
        }
    }

    Ok(())
}

async fn stream_pod_logs(api: &kube::api::Api<Pod>, pod_name: &str) {
    let mut last_time: Option<std::time::Instant> = None;
    let mut done = false;
    while !done {
        match api
            .log_stream(
                pod_name,
                &LogParams {
                    follow: true,
                    since_seconds: last_time.map(|t| t.elapsed().as_secs() as i64),
                    ..LogParams::default()
                },
            )
            .await
        {
            Ok(stream) => {
                let mut lines = stream.lines();
                loop {
                    match lines.try_next().await {
                        Ok(Some(line)) => {
                            last_time = Some(std::time::Instant::now());
                            eprintln!("  [{pod_name}] {line}");
                        }
                        Ok(None) => {
                            done = true;
                            break;
                        }
                        Err(_) => break,
                    }
                }
            }
            Err(_) => {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
        }
    }
}

async fn wait_all_pods_ready(api: &kube::api::Api<Pod>, label_selector: &str) -> Result<()> {
    loop {
        let pods = api
            .list(&ListParams::default().labels(label_selector))
            .await?;
        let all_ready = !pods.items.is_empty()
            && pods.items.iter().all(|pod| {
                pod.status
                    .as_ref()
                    .and_then(|s| s.conditions.as_ref())
                    .is_some_and(|conds| {
                        conds
                            .iter()
                            .any(|c| c.type_ == "Ready" && c.status == "True")
                    })
            });
        if all_ready {
            return Ok(());
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }
}

async fn find_ready_pod(api: &kube::api::Api<Pod>, label_selector: &str) -> Result<String> {
    loop {
        let pods = api
            .list(&ListParams::default().labels(label_selector))
            .await?;
        if let Some(pod) = pods.items.iter().find(|pod| {
            pod.status
                .as_ref()
                .and_then(|s| s.conditions.as_ref())
                .is_some_and(|conds| {
                    conds
                        .iter()
                        .any(|c| c.type_ == "Ready" && c.status == "True")
                })
        }) {
            return Ok(pod.name_any());
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
}

async fn forward_connection(
    api: &kube::api::Api<Pod>,
    pod_name: &str,
    port: u16,
    mut client_conn: impl tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
) -> Result<()> {
    let mut forwarder = api.portforward(pod_name, &[port]).await?;
    let mut upstream = forwarder
        .take_stream(port)
        .ok_or_else(|| anyhow::anyhow!("port not found in forwarder"))?;
    tokio::io::copy_bidirectional(&mut client_conn, &mut upstream).await?;
    drop(upstream);
    forwarder.join().await?;
    Ok(())
}
