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

/// Delete a GCE instance.
///
/// When `force` is true, a "not found" error is treated as success.
/// If the instance isn't in the specified zone, searches fallback zones.
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

    let sp = spinner(&format!("Looking up instance '{}'...", config.name));
    let client = Instances::builder().build().await?;

    let zone = match find_instance_zone(&client, &project, &config.name, &config.zone).await? {
        Some(z) => {
            sp.finish_and_clear();
            if z != config.zone {
                eprintln!(
                    "  Instance '{}' found in {} (not {})",
                    config.name, z, config.zone
                );
            } else {
                eprintln!("  Instance '{}' found in {}", config.name, z);
            }
            z
        }
        None => {
            sp.finish_and_clear();
            if config.force {
                eprintln!(
                    "  Instance '{}' not found in any zone (ignored due to --force)",
                    config.name
                );
                return Ok(());
            }
            eprintln!("  Instance '{}' not found", config.name);
            return Err(anyhow::anyhow!(
                "instance '{}' not found in any zone",
                config.name
            ));
        }
    };

    if !config.force {
        eprint!("  Delete instance '{}' in {}? [y/N] ", config.name, zone);
        std::io::Write::flush(&mut std::io::stderr()).ok();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            eprintln!("  Aborted.");
            return Ok(());
        }
    }

    let sp = spinner(&format!(
        "Deleting instance '{}' in {}...",
        config.name, zone
    ));
    client
        .delete()
        .set_project(&project)
        .set_zone(&zone)
        .set_instance(&config.name)
        .poller()
        .until_done()
        .await?
        .to_result()?;

    sp.finish_and_clear();
    eprintln!("  Instance '{}' deleted", config.name);
    Ok(())
}
