// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Shared helpers for GCE provisioning: zone fallback, error classification, spinners.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

/// Candidate zones for fallback when a zone's resource pool is exhausted.
pub const FALLBACK_ZONES: &[(&str, &str)] = &[
    ("us-west1", "a"),
    ("us-west1", "b"),
    ("us-west1", "c"),
    ("us-central1", "a"),
    ("us-central1", "b"),
    ("us-central1", "c"),
    ("us-east1", "a"),
    ("us-east1", "b"),
    ("us-east1", "c"),
];

/// Error type to distinguish zone-exhaustion from fatal errors.
pub enum CreateError {
    ZoneExhausted(String),
    Fatal(anyhow::Error),
}

/// Check if an operation error is a zone-exhaustion, and extract a human-readable message.
pub fn zone_exhausted_message(
    err: &google_cloud_compute_v1::errors::OperationError,
) -> Option<String> {
    use google_cloud_compute_v1::errors::OperationError;
    match err {
        OperationError::Generic(g) => {
            let is_exhausted = g.status_code == Some(503)
                || g.details.as_ref().is_some_and(|d| {
                    d.errors.iter().any(|e| {
                        e.code
                            .as_deref()
                            .is_some_and(|c| c.contains("ZONE_RESOURCE_POOL_EXHAUSTED"))
                    })
                });
            if !is_exhausted {
                return None;
            }
            if let Some(details) = &g.details {
                for err_entry in &details.errors {
                    for detail in &err_entry.error_details {
                        if let Some(ref lm) = detail.localized_message
                            && let Some(ref msg) = lm.message
                        {
                            return Some(msg.clone());
                        }
                    }
                }
            }
            Some("zone resource pool exhausted".to_string())
        }
        _ => None,
    }
}

/// Build a prioritized zone list: the configured zone first, then fallbacks.
pub fn build_zone_list(configured_zone: &str) -> Vec<(String, String)> {
    let configured_parts: Vec<&str> = configured_zone.rsplitn(2, '-').collect();
    let configured_pair = if configured_parts.len() == 2 {
        Some((
            configured_parts[1].to_string(),
            configured_parts[0].to_string(),
        ))
    } else {
        None
    };

    let mut zones: Vec<(String, String)> = Vec::new();
    if let Some(ref pair) = configured_pair {
        zones.push(pair.clone());
    }
    for &(region, suffix) in FALLBACK_ZONES {
        if configured_pair
            .as_ref()
            .is_some_and(|p| p.0 == region && p.1 == suffix)
        {
            continue;
        }
        zones.push((region.to_string(), suffix.to_string()));
    }
    zones
}

/// Try to create an instance via the GCE API, classifying errors as zone-exhaustion or fatal.
pub async fn try_create_instance(
    client: &google_cloud_compute_v1::client::Instances,
    project: &str,
    zone: &str,
    instance: google_cloud_compute_v1::model::Instance,
) -> Result<(), CreateError> {
    use google_cloud_lro::Poller;

    let result = client
        .insert()
        .set_project(project)
        .set_zone(zone)
        .set_body(instance)
        .poller()
        .until_done()
        .await;

    let operation = match result {
        Ok(op) => op,
        Err(e) => {
            let code = e.http_status_code();
            if code == Some(503) || code == Some(403) {
                let msg = e
                    .status()
                    .map(|s| s.message.clone())
                    .unwrap_or_else(|| format!("zone unavailable (HTTP {})", code.unwrap_or(0)));
                return Err(CreateError::ZoneExhausted(msg));
            }
            return Err(CreateError::Fatal(e.into()));
        }
    };

    match operation.to_result() {
        Ok(_) => Ok(()),
        Err(e) => {
            if let Some(msg) = zone_exhausted_message(&e) {
                Err(CreateError::ZoneExhausted(msg))
            } else {
                Err(CreateError::Fatal(e.into()))
            }
        }
    }
}

/// Delete a single instance (best-effort, used for cleanup).
pub async fn delete_instance(
    client: &google_cloud_compute_v1::client::Instances,
    project: &str,
    zone: &str,
    name: &str,
) -> anyhow::Result<()> {
    use google_cloud_lro::Poller;
    client
        .delete()
        .set_project(project)
        .set_zone(zone)
        .set_instance(name)
        .poller()
        .until_done()
        .await?
        .to_result()?;
    Ok(())
}

/// Read the user's SSH public key and format it as GCE `ssh-keys` metadata.
pub fn read_ssh_public_key() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let home_path = std::path::PathBuf::from(home);
    let username = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "user".to_string());

    for name in [
        "google_compute_engine.pub",
        "id_ed25519.pub",
        "id_rsa.pub",
        "id_ecdsa.pub",
    ] {
        let path = home_path.join(".ssh").join(name);
        if let Ok(contents) = std::fs::read_to_string(&path) {
            let key = contents.trim();
            return Some(format!("{username}:{key}"));
        }
    }
    None
}

pub fn spinner(multi: &MultiProgress, msg: &str) -> ProgressBar {
    let pb = multi.add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb.set_message(msg.to_string());
    pb
}

/// Finish a spinner: clear it from multi and print the final message as a
/// static line with consistent indentation.
pub fn finish_spinner(multi: &MultiProgress, sp: ProgressBar, msg: impl Into<String>) {
    sp.finish_and_clear();
    let _ = multi.println(format!("  {}", msg.into()));
}
