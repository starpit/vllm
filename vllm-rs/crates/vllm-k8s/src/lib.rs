// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Kubernetes deployment for vLLM — `vllm k8s up` / `vllm k8s down`.

pub mod down;
pub mod manifest;
pub mod preload;
pub mod up;

use anyhow::Result;
use kube::{Client, Resource, api::Api};

/// Create a kube client from the default kubeconfig.
pub(crate) async fn client() -> Result<Client> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    Ok(Client::try_default().await?)
}

/// Create a namespaced (or default-namespaced) API handle.
pub(crate) fn api<K>(c: Client, namespace: &Option<String>) -> Result<Api<K>>
where
    <K as Resource>::DynamicType: Default,
    K: Resource<Scope = kube::core::NamespaceResourceScope>,
{
    Ok(match namespace {
        Some(ns) => Api::namespaced(c, ns),
        None => Api::default_namespaced(c),
    })
}
