// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm gce up` — provision a GCE VM with GPUs for dev/inference.

use std::sync::Arc;

use anyhow::Result;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::ssh_tunnel::SshSession;

/// Candidate zones for fallback when a zone's resource pool is exhausted.
const FALLBACK_ZONES: &[(&str, &str)] = &[
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
enum CreateError {
    ZoneExhausted(String),
    Fatal(anyhow::Error),
}

/// Check if an operation error is a zone-exhaustion, and extract a human-readable message.
fn zone_exhausted_message(err: &google_cloud_compute_v1::errors::OperationError) -> Option<String> {
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

fn spinner(multi: &MultiProgress, msg: &str) -> ProgressBar {
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
fn finish_spinner(multi: &MultiProgress, sp: ProgressBar, msg: impl Into<String>) {
    sp.finish_and_clear();
    let _ = multi.println(format!("  {}", msg.into()));
}

/// Configuration for `vllm gce up`.
pub struct GceUpConfig {
    /// Instance name.
    pub name: String,
    /// Number of nodes (currently only 1 supported).
    pub nodes: u32,
    /// Number of GPUs per node.
    pub gpu_count: u32,
    /// GPU class: "l40s", "a100-40", "a100-80", "h100".
    pub gpu_class: String,
    /// Local directory to transfer and build (dev mode). None = non-dev.
    pub dev: Option<std::path::PathBuf>,
    /// Local port for SSH tunnel to remote port 8000.
    pub local_port: u16,
    /// GCE boot image (full self-link).
    pub image: String,
    /// GCE zone (starting zone for fallback search).
    pub zone: String,
    /// GCE project.
    pub project: Option<String>,
    /// Path to GCP service account credentials JSON.
    pub gcp_credentials: Option<String>,
    /// GCP service account email for the instance.
    pub gcp_service_account: Option<String>,
    /// HuggingFace token (passed to the VM).
    pub hf_token: Option<String>,
    /// Use preemptible/SPOT VMs.
    pub preemptible: bool,
    /// GCS bucket for sccache.
    pub sccache_gcs_bucket: Option<String>,
    /// GCS key prefix for sccache.
    pub sccache_gcs_key_prefix: Option<String>,
    /// Model to serve (HuggingFace model ID or path).
    pub model: String,
    /// Extra arguments for `vllm serve` on the remote VM.
    pub serve_args: Vec<String>,
}

/// Zone-independent parameters for instance creation.
struct CreateInstanceParams<'a> {
    client: &'a google_cloud_compute_v1::client::Instances,
    project: &'a str,
    instance_name: &'a str,
    gpu: &'a crate::gpu_map::GpuConfig,
    source_image: &'a str,
    service_account: Option<&'a str>,
    cloud_config: &'a str,
    preemptible: bool,
    disk_size_gb: i64,
    ssh_keys_metadata: Option<String>,
}

async fn create_instance_in_zone(
    params: &CreateInstanceParams<'_>,
    zone: &str,
    region: &str,
) -> Result<(), CreateError> {
    use google_cloud_compute_v1::model::{
        AcceleratorConfig, AccessConfig, AttachedDisk, AttachedDiskInitializeParams, Instance,
        Metadata, NetworkInterface, Scheduling, metadata::Items as MetadataItems,
    };
    use google_cloud_lro::Poller;

    let mut instance = Instance::new()
        .set_name(params.instance_name)
        .set_machine_type(format!(
            "zones/{}/machineTypes/{}",
            zone, params.gpu.machine_type
        ))
        .set_disks([AttachedDisk::new()
            .set_boot(true)
            .set_auto_delete(true)
            .set_initialize_params(
                AttachedDiskInitializeParams::new()
                    .set_source_image(params.source_image)
                    .set_disk_size_gb(params.disk_size_gb)
                    .set_disk_type(format!("zones/{}/diskTypes/pd-ssd", zone)),
            )
            .set_mode("READ_WRITE")])
        .set_network_interfaces([NetworkInterface::new()
            .set_subnetwork(format!("regions/{}/subnetworks/default", region))
            .set_access_configs([AccessConfig::new().set_network_tier("PREMIUM")])
            .set_stack_type("IPV4_ONLY")])
        .set_guest_accelerators([AcceleratorConfig::new()
            .set_accelerator_count(params.gpu.accelerator_count as i32)
            .set_accelerator_type(format!(
                "zones/{}/acceleratorTypes/{}",
                zone, params.gpu.accelerator_type
            ))])
        .set_metadata({
            let mut items = vec![
                MetadataItems::default()
                    .set_key("user-data")
                    .set_value(params.cloud_config),
                MetadataItems::default()
                    .set_key("enable-osconfig")
                    .set_value("TRUE"),
            ];
            if let Some(ref ssh_keys) = params.ssh_keys_metadata {
                items.push(
                    MetadataItems::default()
                        .set_key("ssh-keys")
                        .set_value(ssh_keys.as_str()),
                );
            }
            Metadata::default().set_items(items)
        })
        .set_can_ip_forward(false)
        .set_deletion_protection(false);

    if params.preemptible {
        instance = instance.set_scheduling(
            Scheduling::new()
                .set_automatic_restart(false)
                .set_on_host_maintenance("TERMINATE")
                .set_preemptible(true)
                .set_provisioning_model("SPOT")
                .set_instance_termination_action("STOP"),
        );
    } else {
        instance = instance.set_scheduling(
            Scheduling::new()
                .set_automatic_restart(true)
                .set_on_host_maintenance("TERMINATE"),
        );
    }

    {
        use google_cloud_compute_v1::model::ServiceAccount;
        let mut sa = ServiceAccount::new().set_scopes([
            "https://www.googleapis.com/auth/devstorage.read_write",
            "https://www.googleapis.com/auth/logging.write",
            "https://www.googleapis.com/auth/monitoring.write",
        ]);
        if let Some(sa_email) = params.service_account {
            sa = sa.set_email(sa_email);
        }
        instance = instance.set_service_accounts([sa]);
    }

    let result = params
        .client
        .insert()
        .set_project(params.project)
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

/// Read the user's SSH public key and format it as GCE `ssh-keys` metadata.
fn read_ssh_public_key() -> Option<String> {
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

/// Build cloud-init for dev mode.
fn build_dev_cloud_config(
    username: &str,
    model: &str,
    serve_args: &[String],
    hf_token: Option<&str>,
    sccache_gcs_bucket: Option<&str>,
    sccache_gcs_key_prefix: Option<&str>,
) -> String {
    let home = if username == "root" {
        "/root".to_string()
    } else {
        format!("/home/{username}")
    };

    let mut lines = vec![
        "#cloud-config".to_string(),
        "package_update: true".to_string(),
        "packages:".to_string(),
        "  - build-essential".to_string(),
        "  - pkg-config".to_string(),
        "  - libssl-dev".to_string(),
        "  - nvidia-cuda-toolkit".to_string(),
        "write_files:".to_string(),
        "  - path: /etc/apt/apt.conf.d/99force-ipv4".to_string(),
        "    content: 'Acquire::ForceIPv4 \"true\";'".to_string(),
        "runcmd:".to_string(),
        "  - growpart /dev/sda 1 2>/dev/null || true".to_string(),
        "  - resize2fs /dev/sda1 2>/dev/null || true".to_string(),
        format!(
            "  - su - {username} -c \"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y\""
        ),
        // Install sccache from official release (apt version lacks GCS support).
        "  - curl -fsSL https://github.com/mozilla/sccache/releases/download/v0.9.1/sccache-v0.9.1-x86_64-unknown-linux-musl.tar.gz | tar xz -C /usr/local/bin --strip-components=1 sccache-v0.9.1-x86_64-unknown-linux-musl/sccache".to_string(),
    ];

    // Append env vars to ~/.cargo/env — the ONE file guaranteed to be sourced
    // by both SSH sessions and `su -` (rustup's .bash_profile sources it).
    let mut env_exports = vec![
        "export RUSTC_WRAPPER=sccache".to_string(),
        "export SCCACHE_GCS_RW_MODE=READ_WRITE".to_string(),
    ];
    if let Some(bucket) = sccache_gcs_bucket {
        env_exports.push(format!("export SCCACHE_GCS_BUCKET={bucket}"));
    }
    if let Some(prefix) = sccache_gcs_key_prefix {
        env_exports.push(format!("export SCCACHE_GCS_KEY_PREFIX={prefix}"));
    }
    if let Some(token) = hf_token {
        env_exports.push(format!("export HF_TOKEN={token}"));
    }
    for export in &env_exports {
        lines.push(format!("  - echo '{export}' >> {home}/.cargo/env"));
    }

    // Inline env prefix for su - commands (non-interactive shells skip .bashrc).
    let mut env_prefix = "RUSTC_WRAPPER=sccache SCCACHE_GCS_RW_MODE=READ_WRITE".to_string();
    if let Some(bucket) = sccache_gcs_bucket {
        env_prefix.push_str(&format!(" SCCACHE_GCS_BUCKET={bucket}"));
    }
    if let Some(prefix) = sccache_gcs_key_prefix {
        env_prefix.push_str(&format!(" SCCACHE_GCS_KEY_PREFIX={prefix}"));
    }
    if let Some(token) = hf_token {
        env_prefix.push_str(&format!(" HF_TOKEN={token}"));
    }

    lines.push(format!(
        "  - echo 'Waiting for {home}/vllm-rs to appear...'"
    ));
    lines.push(format!(
        "  - while [ ! -d {home}/vllm-rs ]; do sleep 2; done"
    ));
    lines.push(format!(
        "  - echo '{home}/vllm-rs found, starting build...'"
    ));

    lines.push(format!(
        "  - su - {username} -c 'cd {home}/vllm-rs && {env_prefix} cargo build -Fcuda --release -p vllm-cli'"
    ));

    let extra = if serve_args.is_empty() {
        String::new()
    } else {
        format!(" {}", serve_args.join(" "))
    };
    lines.push(if let Some(token) = hf_token {
        format!("  - su - {username} -c 'cd {home}/vllm-rs && HF_TOKEN={token} ./target/release/vllm serve --model {model}{extra}'")
    } else {
        format!("  - su - {username} -c 'cd {home}/vllm-rs && ./target/release/vllm serve --model {model}{extra}'")
    });

    lines.join("\n")
}

/// Build cloud-init for non-dev mode.
fn build_cloud_config() -> String {
    "#cloud-config\n\
     runcmd:\n\
       - growpart /dev/sda 1 2>/dev/null || true\n\
       - resize2fs /dev/sda1 2>/dev/null || true"
        .to_string()
}

/// Strip all known prefixes from a serial console line, returning the core content.
fn strip_serial_line(line: &str) -> &str {
    let mut s = line;

    // Strip "TIMESTAMP INSTANCE " prefix (two space-delimited fields).
    {
        let mut space_count = 0;
        for (i, c) in s.char_indices() {
            if c == ' ' {
                space_count += 1;
                if space_count == 2 {
                    s = &s[i + 1..];
                    break;
                }
            }
        }
    }

    // Strip kernel timestamp prefix like "141.977650] ".
    if let Some(idx) = s.find("] ") {
        let before = &s[..idx];
        if before.bytes().all(|b| b.is_ascii_digit() || b == b'.') {
            s = &s[idx + 2..];
        }
    }

    s.trim()
}

/// Strip the `cloud-init[N]: ` prefix from a line, if present.
fn strip_cloud_init_prefix(s: &str) -> &str {
    if let Some(idx) = s.find("cloud-init[")
        && let Some(colon) = s[idx..].find(':')
    {
        s[idx + colon + 1..].trim_start()
    } else {
        s
    }
}

/// Colorize a serial line for display in the ring buffer dump.
fn colorize_serial_line(stripped: &str) -> String {
    const BLUE: &str = "\x1b[34m";
    const DIM: &str = "\x1b[2m";
    const RESET: &str = "\x1b[0m";

    if stripped.contains("cloud-init[") {
        let content = strip_cloud_init_prefix(stripped);
        return format!("{BLUE}{content}{RESET}");
    }

    format!("{DIM}{stripped}{RESET}")
}

const CONSOLE_BUFFER_SIZE: usize = 30;

/// Stream serial console output as a single rolling status line.
/// Keeps a 30-line buffer; dumps the buffer if an error is detected.
async fn stream_serial_console(
    multi: MultiProgress,
    console_pb: ProgressBar,
    client: google_cloud_compute_v1::client::Instances,
    instance_name: String,
    zone: String,
    project: String,
    server_ready: Arc<tokio::sync::Notify>,
) -> Result<()> {
    let mut notified_ready = false;
    let mut last_start = 0i64;
    let mut consecutive_errors = 0;
    let mut ring: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut partial_line = String::new();

    loop {
        match client
            .get_serial_port_output()
            .set_project(&project)
            .set_zone(&zone)
            .set_instance(&instance_name)
            .set_port(1)
            .set_start(last_start)
            .send()
            .await
        {
            Ok(output) => {
                consecutive_errors = 0;
                if let Some(contents) = output.contents
                    && !contents.is_empty()
                {
                    let mut saw_error = false;

                    // Prepend any leftover partial line from previous poll.
                    let full = if partial_line.is_empty() {
                        contents.clone()
                    } else {
                        let mut s = std::mem::take(&mut partial_line);
                        s.push_str(&contents);
                        s
                    };

                    // If the data doesn't end with newline, the last "line"
                    // is partial — save it for next poll.
                    if !full.ends_with('\n') {
                        if let Some(last_nl) = full.rfind('\n') {
                            partial_line = full[last_nl + 1..].to_string();
                        } else {
                            partial_line = full;
                            if let Some(next) = output.next {
                                last_start = next;
                            }
                            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                            continue;
                        }
                    }

                    for line in full.lines() {
                        let stripped = strip_serial_line(line);
                        if stripped.is_empty() {
                            continue;
                        }

                        // Deduplicate: the serial console often emits each
                        // line twice (once with kernel timestamp, once without).
                        if !seen.insert(stripped.to_string()) {
                            continue;
                        }
                        if seen.len() > 1000 {
                            seen.clear();
                        }

                        // Add to ring buffer.
                        let colored = colorize_serial_line(stripped);
                        if ring.len() >= CONSOLE_BUFFER_SIZE {
                            ring.pop_front();
                        }
                        ring.push_back(colored);

                        // Update spinner with colorized last line.
                        let display_text = strip_cloud_init_prefix(stripped);
                        let truncated: String = if display_text.chars().count() > 120 {
                            let mut s: String = display_text.chars().take(117).collect();
                            s.push_str("...");
                            s
                        } else {
                            display_text.to_string()
                        };
                        let is_cloud_init = stripped.contains("cloud-init[");
                        let colored_display = if is_cloud_init {
                            format!("\x1b[34m{truncated}\x1b[0m")
                        } else {
                            format!("\x1b[2m{truncated}\x1b[0m")
                        };
                        console_pb.set_message(colored_display);

                        // Detect server ready.
                        if !notified_ready && stripped.contains("server listening on") {
                            notified_ready = true;
                            server_ready.notify_one();
                        }

                        // Only flag errors from cloud-init or cargo/rustc output.
                        let lower = stripped.to_ascii_lowercase();
                        if (stripped.contains("cloud-init[")
                            && (lower.contains("error:") || lower.contains("fatal:")))
                            || lower.starts_with("error[")
                            || lower.starts_with("error:")
                        {
                            saw_error = true;
                        }
                    }

                    // Dump the buffer on error.
                    if saw_error {
                        let _ = multi
                            .println("\x1b[31m  --- console output (last 30 lines) ---\x1b[0m");
                        for line in &ring {
                            let _ = multi.println(format!("\x1b[31m  {line}\x1b[0m"));
                        }
                        let _ = multi.println("  --- end ---");
                    }
                }
                if let Some(next) = output.next {
                    last_start = next;
                }
            }
            Err(_) => {
                consecutive_errors += 1;
                if consecutive_errors > 10 {
                    console_pb.finish_and_clear();
                    let _ = multi.println("  Serial console: stopped (too many errors)");
                    break;
                }
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }

    Ok(())
}

/// Build the ordered list of zones to try.
fn build_zone_list(configured_zone: &str) -> Vec<(String, String)> {
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

/// Provision a GCE VM, optionally transfer dev dir and start SSH tunnel.
pub async fn up(config: GceUpConfig) -> Result<()> {
    use google_cloud_compute_v1::client::Instances;

    if config.nodes > 1 {
        todo!("multi-node provisioning not yet implemented");
    }

    let gpu = crate::gpu_map::resolve(&config.gpu_class, config.gpu_count)?;

    let project = config
        .project
        .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "GCP project required: set --project, GCP_PROJECT, or GOOGLE_CLOUD_PROJECT"
            )
        })?;

    let instance_name = &config.name;

    let username = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "user".to_string());

    let remote_home = if username == "root" {
        "/root".to_string()
    } else {
        format!("/home/{username}")
    };

    // Build cloud-init based on mode.
    let cloud_config = if config.dev.is_some() {
        build_dev_cloud_config(
            &username,
            &config.model,
            &config.serve_args,
            config.hf_token.as_deref(),
            config.sccache_gcs_bucket.as_deref(),
            config.sccache_gcs_key_prefix.as_deref(),
        )
    } else {
        build_cloud_config()
    };

    let disk_size_gb = if config.dev.is_some() { 300 } else { 100 };

    let zones_to_try = build_zone_list(&config.zone);

    let ssh_keys_metadata = read_ssh_public_key();

    let multi = MultiProgress::new();

    // --- Stage 1: Create GCE API client ---
    let sp = spinner(&multi, "Authenticating with GCE API...");
    let client = Instances::builder().build().await?;
    finish_spinner(&multi, sp, "Authenticated with GCE API");

    let create_params = CreateInstanceParams {
        client: &client,
        project: &project,
        instance_name,
        gpu: &gpu,
        source_image: &config.image,
        service_account: config.gcp_service_account.as_deref(),
        cloud_config: &cloud_config,
        preemptible: config.preemptible,
        disk_size_gb,
        ssh_keys_metadata,
    };

    // --- Stage 2: Create instance (with zone fallback) ---
    let mut succeeded_zone = None;
    for (region, suffix) in &zones_to_try {
        let zone = format!("{region}-{suffix}");
        let mode = if config.preemptible {
            "SPOT"
        } else {
            "on-demand"
        };
        let sp = spinner(
            &multi,
            &format!(
                "Creating instance '{instance_name}' in {zone} ({} x{} {mode})...",
                gpu.machine_type, config.gpu_count
            ),
        );

        match create_instance_in_zone(&create_params, &zone, region).await {
            Ok(()) => {
                finish_spinner(
                    &multi,
                    sp,
                    format!("Instance '{instance_name}' created in {zone}"),
                );
                succeeded_zone = Some(zone);
                break;
            }
            Err(CreateError::ZoneExhausted(msg)) => {
                finish_spinner(&multi, sp, format!("Zone {zone} exhausted: {msg}"));
                continue;
            }
            Err(CreateError::Fatal(e)) => {
                finish_spinner(&multi, sp, format!("Failed in {zone}"));
                return Err(e);
            }
        }
    }

    let zone = succeeded_zone
        .ok_or_else(|| anyhow::anyhow!("failed to create instance in any zone — all exhausted"))?;

    // --- Stage 3: Get external IP ---
    let sp = spinner(&multi, "Fetching instance external IP...");
    let instance_info = client
        .get()
        .set_project(&project)
        .set_zone(&zone)
        .set_instance(instance_name)
        .send()
        .await?;

    let external_ip = instance_info
        .network_interfaces
        .iter()
        .find_map(|ni| ni.access_configs.iter().find_map(|ac| ac.nat_ip.as_ref()))
        .ok_or_else(|| anyhow::anyhow!("no external IP found for instance"))?
        .clone();
    finish_spinner(&multi, sp, format!("External IP: {external_ip}"));

    // --- Stage 4: SSH connect ---
    let sp = spinner(&multi, &format!("Connecting via SSH to {external_ip}..."));
    let session = Arc::new(SshSession::connect(&external_ip, 22).await?);
    finish_spinner(&multi, sp, format!("SSH connected to {external_ip}"));

    // --- Stage 5: Upload dev directory (if dev mode) ---
    if let Some(ref dev_path) = config.dev {
        let remote_dir = format!("{remote_home}/vllm-rs");
        let sp = spinner(&multi, &format!("Packing {}...", dev_path.display()));
        let (file_count, total_bytes) =
            session.upload_dir(dev_path, &remote_dir, Some(&sp)).await?;
        finish_spinner(
            &multi,
            sp,
            format!(
                "Uploaded {file_count} files ({}) to {remote_dir}",
                crate::ssh_tunnel::humanize_bytes(total_bytes),
            ),
        );
    }

    // --- Stage 6: Info lines ---
    finish_spinner(
        &multi,
        spinner(&multi, ""),
        format!("Tear down with: \x1b[36mvllm gce down {instance_name} --zone={zone}\x1b[0m"),
    );

    // --- Stage 7: Stream serial console (background, at bottom) ---
    let console_pb = multi.add(ProgressBar::new_spinner());
    console_pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.dim} {msg}")
            .unwrap(),
    );
    console_pb.enable_steady_tick(std::time::Duration::from_millis(80));
    console_pb.set_message("Waiting for console output...");
    let server_ready = Arc::new(tokio::sync::Notify::new());
    {
        let stream_client = Instances::builder().build().await?;
        let stream_multi = multi.clone();
        let stream_pb = console_pb.clone();
        let stream_name = instance_name.to_string();
        let stream_zone = zone.clone();
        let stream_project = project.clone();
        let ready = server_ready.clone();
        tokio::spawn(async move {
            let _ = stream_serial_console(
                stream_multi,
                stream_pb,
                stream_client,
                stream_name,
                stream_zone,
                stream_project,
                ready,
            )
            .await;
        });
    }

    // --- Stage 8: Wait for server ready, then SSH tunnel + Ctrl+C ---
    let local_port = config.local_port;
    server_ready.notified().await;

    let tunnel = session.tunnel(local_port, "localhost".to_string(), 8000);
    let tunnel = Arc::new(tunnel);
    let tunnel_clone = tunnel.clone();
    tokio::spawn(async move {
        if let Err(e) = tunnel_clone.start().await {
            eprintln!("SSH tunnel error: {e}");
        }
    });
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    finish_spinner(
        &multi,
        spinner(&multi, ""),
        format!("SSH tunnel: http://localhost:{local_port} — Ctrl+C to stop"),
    );

    tokio::signal::ctrl_c().await?;
    let _ = tunnel.close().await;

    Ok(())
}
