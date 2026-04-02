// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm k8s down` — tear down vLLM Kubernetes resources.

use anyhow::Result;
use k8s_openapi::api::apps::v1::{Deployment, StatefulSet};
use k8s_openapi::api::core::v1::Service;
use kube::api::DeleteParams;

/// Configuration for `vllm k8s down`.
pub struct K8sDownConfig {
    /// Resource name to delete.
    pub name: String,
    /// Kubernetes namespace (None = default).
    pub namespace: Option<String>,
    /// Skip confirmation / treat not-found as success.
    pub force: bool,
}

/// Delete vLLM Kubernetes resources (Deployment or StatefulSet + headless Service).
pub async fn down(config: K8sDownConfig) -> Result<()> {
    let c = crate::client().await?;
    let dp = DeleteParams::default();

    // Try Deployment first, then StatefulSet.
    let dep_api = crate::api::<Deployment>(c.clone(), &config.namespace)?;
    let ss_api = crate::api::<StatefulSet>(c.clone(), &config.namespace)?;
    let svc_api = crate::api::<Service>(c.clone(), &config.namespace)?;

    let mut found = false;

    match dep_api.delete(&config.name, &dp).await {
        Ok(_) => {
            eprintln!("  Deleted Deployment '{}'", config.name);
            found = true;
        }
        Err(kube::Error::Api(ref e)) if e.code == 404 => {}
        Err(e) => return Err(e.into()),
    }

    if !found {
        match ss_api.delete(&config.name, &dp).await {
            Ok(_) => {
                eprintln!("  Deleted StatefulSet '{}'", config.name);
                found = true;
            }
            Err(kube::Error::Api(ref e)) if e.code == 404 => {}
            Err(e) => return Err(e.into()),
        }
    }

    // Always try to delete the headless service.
    let svc_name = format!("{}-headless", config.name);
    match svc_api.delete(&svc_name, &dp).await {
        Ok(_) => {
            eprintln!("  Deleted Service '{svc_name}'");
        }
        Err(kube::Error::Api(ref e)) if e.code == 404 => {}
        Err(e) => return Err(e.into()),
    }

    if !found && !config.force {
        anyhow::bail!("no Deployment or StatefulSet named '{}' found", config.name);
    }

    if !found && config.force {
        eprintln!("  '{}' not found (ignored due to --force)", config.name);
    }

    Ok(())
}
