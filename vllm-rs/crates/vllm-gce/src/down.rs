// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm gce down` — tear down a GCE instance via Compute API.

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};

/// Zones to search when the instance isn't found in the specified zone.
const SEARCH_ZONES: &[&str] = &[
    "us-west1-a",
    "us-west1-b",
    "us-west1-c",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east1-a",
    "us-east1-b",
    "us-east1-c",
];

/// Returns true if the error indicates the resource was not found (HTTP 404).
fn is_not_found(err: &google_cloud_compute_v1::Error) -> bool {
    if err.http_status_code() == Some(404) {
        return true;
    }
    if let Some(status) = err.status()
        && status.message.contains("was not found")
    {
        return true;
    }
    false
}

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb.set_message(msg.to_string());
    pb
}

/// Configuration for `vllm gce down`.
pub struct GceDownConfig {
    /// Instance name to delete.
    pub name: String,
    /// GCE zone.
    pub zone: String,
    /// GCE project (None = use env fallback).
    pub project: Option<String>,
    /// Skip confirmation / treat not-found as success.
    pub force: bool,
}

/// Try to find the instance in the given zone, then fall back to searching other zones.
/// Returns the zone where the instance was found.
async fn find_instance_zone(
    client: &google_cloud_compute_v1::client::Instances,
    project: &str,
    name: &str,
    preferred_zone: &str,
) -> Result<Option<String>> {
    // Try the preferred zone first.
    match client
        .get()
        .set_project(project)
        .set_zone(preferred_zone)
        .set_instance(name)
        .send()
        .await
    {
        Ok(_) => return Ok(Some(preferred_zone.to_string())),
        Err(e) if is_not_found(&e) => {}
        Err(e) => return Err(e.into()),
    }

    // Search fallback zones.
    for &zone in SEARCH_ZONES {
        if zone == preferred_zone {
            continue;
        }
        match client
            .get()
            .set_project(project)
            .set_zone(zone)
            .set_instance(name)
            .send()
            .await
        {
            Ok(_) => return Ok(Some(zone.to_string())),
            Err(e) if is_not_found(&e) => continue,
            Err(e) => return Err(e.into()),
        }
    }

    Ok(None)
}

/// Find all instances belonging to this name (single-node or multi-node cluster).
/// Returns a list of (instance_name, zone) pairs.
async fn find_cluster_instances(
    client: &google_cloud_compute_v1::client::Instances,
    project: &str,
    name: &str,
    preferred_zone: &str,
) -> Result<Vec<(String, String)>> {
    let mut found = Vec::new();

    // Check single-node instance name.
    if let Some(zone) = find_instance_zone(client, project, name, preferred_zone).await? {
        found.push((name.to_string(), zone));
        return Ok(found);
    }

    // Check multi-node names: {name}-0, {name}-1, ... up to a reasonable max.
    // First find node 0 to confirm it's a multi-node cluster.
    let node0 = format!("{name}-0");
    let zone = match find_instance_zone(client, project, &node0, preferred_zone).await? {
        Some(z) => z,
        None => return Ok(found), // No instances found at all.
    };

    found.push((node0, zone.clone()));

    // Search for remaining nodes in the same zone (multi-node must be co-located).
    for i in 1..64 {
        let node_name = format!("{name}-{i}");
        match client
            .get()
            .set_project(project)
            .set_zone(&zone)
            .set_instance(&node_name)
            .send()
            .await
        {
            Ok(_) => found.push((node_name, zone.clone())),
            Err(e) if is_not_found(&e) => break,
            Err(e) => return Err(e.into()),
        }
    }

    Ok(found)
}

/// Delete GCE instance(s).
///
/// Handles both single-node (`{name}`) and multi-node (`{name}-0`, `{name}-1`, ...)
/// clusters. When `force` is true, a "not found" error is treated as success.
pub async fn down(config: GceDownConfig) -> Result<()> {
    use google_cloud_compute_v1::client::Instances;
    use google_cloud_lro::Poller;

    let project = config
        .project
        .or_else(|| std::env::var("GCP_PROJECT").ok())
        .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "GCP project required: set --project, GCP_PROJECT, or GOOGLE_CLOUD_PROJECT"
            )
        })?;

    let sp = spinner(&format!("Looking up instance(s) '{}'...", config.name));
    let client = Instances::builder().build().await?;

    let instances = find_cluster_instances(&client, &project, &config.name, &config.zone).await?;

    sp.finish_and_clear();

    if instances.is_empty() {
        if config.force {
            eprintln!(
                "  Instance(s) '{}' not found in any zone (ignored due to --force)",
                config.name
            );
            return Ok(());
        }
        eprintln!("  Instance(s) '{}' not found", config.name);
        return Err(anyhow::anyhow!(
            "instance(s) '{}' not found in any zone",
            config.name
        ));
    }

    // Show what will be deleted.
    for (name, zone) in &instances {
        eprintln!("  Found '{name}' in {zone}");
    }

    if !config.force {
        if instances.len() == 1 {
            eprint!(
                "  Delete instance '{}' in {}? [y/N] ",
                instances[0].0, instances[0].1
            );
        } else {
            eprint!("  Delete {} instances? [y/N] ", instances.len());
        }
        std::io::Write::flush(&mut std::io::stderr()).ok();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            eprintln!("  Aborted.");
            return Ok(());
        }
    }

    // Delete all instances in parallel.
    let sp = spinner(&format!("Deleting {} instance(s)...", instances.len()));

    let mut futs = Vec::new();
    for (name, zone) in &instances {
        let client = &client;
        let project = &project;
        futs.push(async move {
            let result = client
                .delete()
                .set_project(project)
                .set_zone(zone)
                .set_instance(name)
                .poller()
                .until_done()
                .await;
            match result {
                Ok(op) => match op.to_result() {
                    Ok(_) => Ok(name.clone()),
                    Err(e) => Err(anyhow::anyhow!("failed to delete '{name}': {e}")),
                },
                Err(e) => Err(anyhow::anyhow!("failed to delete '{name}': {e}")),
            }
        });
    }

    let results = futures::future::join_all(futs).await;
    sp.finish_and_clear();

    let mut errors = Vec::new();
    for result in results {
        match result {
            Ok(name) => eprintln!("  Instance '{name}' deleted"),
            Err(e) => errors.push(e),
        }
    }

    if !errors.is_empty() {
        for e in &errors {
            eprintln!("  Error: {e}");
        }
        return Err(anyhow::anyhow!(
            "{} instance(s) failed to delete",
            errors.len()
        ));
    }

    Ok(())
}
