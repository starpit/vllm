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
    /// Number of nodes (1 = single node, >1 = multi-node cluster).
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

/// Multi-node environment variables for cloud-init.
struct MultiNodeEnv {
    node_rank: u32,
    nnodes: u32,
    head_ip: String,
}

/// Build cloud-init for dev mode.
fn build_dev_cloud_config(
    username: &str,
    model: &str,
    serve_args: &[String],
    hf_token: Option<&str>,
    sccache_gcs_bucket: Option<&str>,
    sccache_gcs_key_prefix: Option<&str>,
    multi_node: Option<&MultiNodeEnv>,
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
    if let Some(mn) = multi_node {
        // TODO(multi-node-perf): For A3/A3+ (H100/H200), create a compact placement
        // policy with `groupPlacementPolicy.collocation = "COLLOCATED"` for lowest-latency
        // NVLink/NVSwitch interconnect.
        // TODO(multi-node-perf): For GPUDirect-TCPX/TCPXO on A3 machines, configure
        // multiple NetworkInterface entries per instance, dedicated subnets per NIC,
        // install NCCL GPUDirect plugin, and set NCCL_NET=GPUDirectTCPX.
        env_exports.push(format!("export VLLM_NODE_RANK={}", mn.node_rank));
        env_exports.push(format!("export VLLM_NNODES={}", mn.nnodes));
        if mn.head_ip.is_empty() {
            // Node 0 discovers its own internal IP at runtime via GCE metadata.
            env_exports.push(
                "export VLLM_HEAD_IP=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip)".to_string()
            );
        } else {
            env_exports.push(format!("export VLLM_HEAD_IP={}", mn.head_ip));
        }
        env_exports.push("export VLLM_NCCL_PORT=29400".to_string());
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
    if let Some(mn) = multi_node {
        if mn.head_ip.is_empty() {
            // Node 0: resolve head IP from GCE metadata at runtime.
            env_prefix.push_str(&format!(
                " VLLM_NODE_RANK={} VLLM_NNODES={} VLLM_HEAD_IP=$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip) VLLM_NCCL_PORT=29400",
                mn.node_rank, mn.nnodes
            ));
        } else {
            env_prefix.push_str(&format!(
                " VLLM_NODE_RANK={} VLLM_NNODES={} VLLM_HEAD_IP={} VLLM_NCCL_PORT=29400",
                mn.node_rank, mn.nnodes, mn.head_ip
            ));
        }
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

    // Only node 0 (or single-node) runs `vllm serve`. Worker nodes build and wait.
    let is_head = multi_node.is_none_or(|mn| mn.node_rank == 0);
    if is_head {
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
    } else {
        lines.push(format!(
            "  - echo 'Node {} build complete. Waiting for head node to coordinate.'",
            multi_node.unwrap().node_rank
        ));
        // Worker nodes sleep indefinitely — they'll be used when multi-node TP
        // is wired to GCE. For now they just need to stay alive.
        lines.push("  - sleep infinity".to_string());
    }

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
struct ConsoleStreamConfig {
    multi: MultiProgress,
    console_pb: ProgressBar,
    client: google_cloud_compute_v1::client::Instances,
    instance_name: String,
    zone: String,
    project: String,
    server_ready: Option<Arc<tokio::sync::Notify>>,
    label: Option<String>,
}

async fn stream_serial_console(cfg: ConsoleStreamConfig) -> Result<()> {
    let ConsoleStreamConfig {
        multi,
        console_pb,
        client,
        instance_name,
        zone,
        project,
        server_ready,
        label,
    } = cfg;
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
                        let msg = if let Some(ref lbl) = label {
                            format!("{lbl} {colored_display}")
                        } else {
                            colored_display
                        };
                        console_pb.set_message(msg);

                        // Detect server ready.
                        if !notified_ready && stripped.contains("server listening on") {
                            notified_ready = true;
                            if let Some(ref ready) = server_ready {
                                ready.notify_one();
                            }
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

/// Delete a single instance (best-effort, used for cleanup on partial failures).
async fn delete_instance(
    client: &google_cloud_compute_v1::client::Instances,
    project: &str,
    zone: &str,
    name: &str,
) -> Result<()> {
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

/// Provision GCE VM(s), optionally transfer dev dir and start SSH tunnel.
pub async fn up(config: GceUpConfig) -> Result<()> {
    use google_cloud_compute_v1::client::Instances;

    let num_nodes = config.nodes.max(1);

    let gpu = crate::gpu_map::resolve(&config.gpu_class, config.gpu_count)?;

    let project = config
        .project
        .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "GCP project required: set --project, GCP_PROJECT, or GOOGLE_CLOUD_PROJECT"
            )
        })?;

    let username = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "user".to_string());

    let remote_home = if username == "root" {
        "/root".to_string()
    } else {
        format!("/home/{username}")
    };

    // Instance names: single-node = "{name}", multi-node = "{name}-0" .. "{name}-{N-1}".
    let instance_names: Vec<String> = if num_nodes == 1 {
        vec![config.name.clone()]
    } else {
        (0..num_nodes)
            .map(|i| format!("{}-{i}", config.name))
            .collect()
    };

    let disk_size_gb = if config.dev.is_some() { 300 } else { 100 };
    let zones_to_try = build_zone_list(&config.zone);
    let ssh_keys_metadata = read_ssh_public_key();

    let multi = MultiProgress::new();

    // --- Stage 1: Create GCE API client ---
    let sp = spinner(&multi, "Authenticating with GCE API...");
    let client = Instances::builder().build().await?;
    finish_spinner(&multi, sp, "Authenticated with GCE API");

    // --- Stage 2: Create instances (with zone fallback, all-or-nothing per zone) ---
    // For multi-node, all VMs must be in the same zone for NCCL latency.
    // TODO(multi-node-perf): For custom VPCs, add a `--create-firewall-rule` flag
    // to ensure TCP 29400-29500 is open between nodes.
    let mut succeeded_zone = None;

    'zone_loop: for (region, suffix) in &zones_to_try {
        let zone = format!("{region}-{suffix}");
        let mode = if config.preemptible {
            "SPOT"
        } else {
            "on-demand"
        };

        if num_nodes == 1 {
            // Single-node: original path.
            let instance_name = &instance_names[0];
            // Build cloud-init (no multi-node env).
            let cloud_config = if config.dev.is_some() {
                build_dev_cloud_config(
                    &username,
                    &config.model,
                    &config.serve_args,
                    config.hf_token.as_deref(),
                    config.sccache_gcs_bucket.as_deref(),
                    config.sccache_gcs_key_prefix.as_deref(),
                    None,
                )
            } else {
                build_cloud_config()
            };

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
                ssh_keys_metadata: ssh_keys_metadata.clone(),
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
        } else {
            // Multi-node: create all VMs in parallel, all-or-nothing per zone.
            let sp = spinner(
                &multi,
                &format!(
                    "Creating {num_nodes} instances in {zone} ({} x{} {mode})...",
                    gpu.machine_type, config.gpu_count
                ),
            );

            // We need internal IPs for cloud-init, but we don't have them until VMs
            // are created. Strategy: create all VMs with a placeholder cloud-config
            // first, then fetch internal IPs. The VLLM_* env vars are written to
            // ~/.cargo/env so cloud-init picks them up before the build starts.
            // Actually, we can use a two-phase approach: node 0 first to get its IP,
            // then the rest in parallel.

            // Phase 1: Create node 0 to get its internal IP.
            let node0_name = &instance_names[0];
            // Temporary cloud config without head IP (will be set via env in ~/.cargo/env).
            let node0_cloud_config = if config.dev.is_some() {
                // Node 0 discovers its own internal IP via GCE metadata at runtime.
                build_dev_cloud_config(
                    &username,
                    &config.model,
                    &config.serve_args,
                    config.hf_token.as_deref(),
                    config.sccache_gcs_bucket.as_deref(),
                    config.sccache_gcs_key_prefix.as_deref(),
                    Some(&MultiNodeEnv {
                        node_rank: 0,
                        nnodes: num_nodes,
                        head_ip: String::new(), // empty = self-discover
                    }),
                )
            } else {
                build_cloud_config()
            };

            let node0_params = CreateInstanceParams {
                client: &client,
                project: &project,
                instance_name: node0_name,
                gpu: &gpu,
                source_image: &config.image,
                service_account: config.gcp_service_account.as_deref(),
                cloud_config: &node0_cloud_config,
                preemptible: config.preemptible,
                disk_size_gb,
                ssh_keys_metadata: ssh_keys_metadata.clone(),
            };

            match create_instance_in_zone(&node0_params, &zone, region).await {
                Ok(()) => {}
                Err(CreateError::ZoneExhausted(msg)) => {
                    finish_spinner(&multi, sp, format!("Zone {zone} exhausted: {msg}"));
                    continue;
                }
                Err(CreateError::Fatal(e)) => {
                    finish_spinner(&multi, sp, format!("Failed in {zone}"));
                    return Err(e);
                }
            }

            // Fetch node 0's internal IP.
            let node0_info = client
                .get()
                .set_project(&project)
                .set_zone(&zone)
                .set_instance(node0_name)
                .send()
                .await?;
            let head_ip = node0_info
                .network_interfaces
                .iter()
                .find_map(|ni| ni.network_ip.as_ref())
                .ok_or_else(|| anyhow::anyhow!("no internal IP found for node 0"))?
                .clone();

            // Update node 0's metadata with the real head IP via SSH after connect.
            // For now, node 0 knows its own IP via the cloud-init env vars — we'll
            // fix the PENDING placeholder via SSH in stage 5.

            // Phase 2: Create remaining nodes in parallel with head IP known.
            // Pre-build cloud configs so they live long enough for the futures.
            let worker_cloud_configs: Vec<String> = (1..num_nodes)
                .map(|i| {
                    if config.dev.is_some() {
                        build_dev_cloud_config(
                            &username,
                            &config.model,
                            &config.serve_args,
                            config.hf_token.as_deref(),
                            config.sccache_gcs_bucket.as_deref(),
                            config.sccache_gcs_key_prefix.as_deref(),
                            Some(&MultiNodeEnv {
                                node_rank: i,
                                nnodes: num_nodes,
                                head_ip: head_ip.clone(),
                            }),
                        )
                    } else {
                        build_cloud_config()
                    }
                })
                .collect();

            let worker_params: Vec<CreateInstanceParams<'_>> = (1..num_nodes as usize)
                .map(|i| CreateInstanceParams {
                    client: &client,
                    project: &project,
                    instance_name: &instance_names[i],
                    gpu: &gpu,
                    source_image: &config.image,
                    service_account: config.gcp_service_account.as_deref(),
                    cloud_config: &worker_cloud_configs[i - 1],
                    preemptible: config.preemptible,
                    disk_size_gb,
                    ssh_keys_metadata: ssh_keys_metadata.clone(),
                })
                .collect();

            let futs: Vec<_> = worker_params
                .iter()
                .map(|params| create_instance_in_zone(params, &zone, region))
                .collect();

            let results = futures::future::join_all(futs).await;

            // Check results — if any failed with ZoneExhausted, tear down all and retry.
            let mut any_exhausted = false;
            let mut fatal_error = None;
            for (i, result) in results.into_iter().enumerate() {
                match result {
                    Ok(()) => {}
                    Err(CreateError::ZoneExhausted(msg)) => {
                        any_exhausted = true;
                        let _ = multi.println(format!("  Node {} zone exhausted: {msg}", i + 1));
                    }
                    Err(CreateError::Fatal(e)) => {
                        fatal_error = Some(e);
                    }
                }
            }

            if any_exhausted || fatal_error.is_some() {
                // Tear down any instances we created in this zone.
                let _ = multi.println(format!("  Cleaning up instances in {zone}..."));
                let mut delete_futs = Vec::new();
                for name in &instance_names {
                    delete_futs.push(async {
                        let _ = delete_instance(&client, &project, &zone, name).await;
                    });
                }
                futures::future::join_all(delete_futs).await;

                if let Some(e) = fatal_error {
                    finish_spinner(&multi, sp, format!("Failed in {zone}"));
                    return Err(e);
                }
                finish_spinner(&multi, sp, format!("Zone {zone} exhausted"));
                continue 'zone_loop;
            }

            finish_spinner(
                &multi,
                sp,
                format!("{num_nodes} instances created in {zone}"),
            );
            succeeded_zone = Some(zone);
            break;
        }
    }

    let zone = succeeded_zone
        .ok_or_else(|| anyhow::anyhow!("failed to create instances in any zone — all exhausted"))?;

    // --- Stage 3: Get external IP of head node (node 0) ---
    let head_instance = &instance_names[0];
    let sp = spinner(&multi, "Fetching instance external IP...");
    let instance_info = client
        .get()
        .set_project(&project)
        .set_zone(&zone)
        .set_instance(head_instance)
        .send()
        .await?;

    let external_ip = instance_info
        .network_interfaces
        .iter()
        .find_map(|ni| ni.access_configs.iter().find_map(|ac| ac.nat_ip.as_ref()))
        .ok_or_else(|| anyhow::anyhow!("no external IP found for instance"))?
        .clone();

    if num_nodes > 1 {
        let internal_ip = instance_info
            .network_interfaces
            .iter()
            .find_map(|ni| ni.network_ip.as_ref());
        if let Some(ip) = internal_ip {
            finish_spinner(
                &multi,
                sp,
                format!("Head node: external={external_ip} internal={ip}"),
            );
        } else {
            finish_spinner(&multi, sp, format!("External IP: {external_ip}"));
        }
    } else {
        finish_spinner(&multi, sp, format!("External IP: {external_ip}"));
    }

    // --- Stage 4: SSH connect to head node ---
    let sp = spinner(&multi, &format!("Connecting via SSH to {external_ip}..."));
    let session = Arc::new(SshSession::connect(&external_ip, 22).await?);
    finish_spinner(&multi, sp, format!("SSH connected to {external_ip}"));

    // --- Stage 5: Upload dev directory (if dev mode) ---
    if let Some(ref dev_path) = config.dev {
        let remote_dir = format!("{remote_home}/vllm-rs");

        if num_nodes > 1 {
            // Upload to all nodes in parallel. Connect SSH to each worker node.
            let sp = spinner(
                &multi,
                &format!("Uploading {} to {num_nodes} nodes...", dev_path.display()),
            );

            // Upload to head node first (we already have a session).
            let (file_count, total_bytes) =
                session.upload_dir(dev_path, &remote_dir, Some(&sp)).await?;

            // Upload to worker nodes in parallel.
            let mut upload_futs = Vec::new();
            for i in 1..num_nodes {
                let worker_name = &instance_names[i as usize];
                let worker_info = client
                    .get()
                    .set_project(&project)
                    .set_zone(&zone)
                    .set_instance(worker_name)
                    .send()
                    .await?;
                let worker_ip = worker_info
                    .network_interfaces
                    .iter()
                    .find_map(|ni| ni.access_configs.iter().find_map(|ac| ac.nat_ip.as_ref()))
                    .ok_or_else(|| anyhow::anyhow!("no external IP found for {worker_name}"))?
                    .clone();

                let dev_path = dev_path.clone();
                let remote_dir = remote_dir.clone();
                upload_futs.push(async move {
                    let worker_session = SshSession::connect(&worker_ip, 22).await?;
                    worker_session
                        .upload_dir(&dev_path, &remote_dir, None)
                        .await
                });
            }

            let worker_results = futures::future::join_all(upload_futs).await;
            for result in worker_results {
                result?;
            }

            finish_spinner(
                &multi,
                sp,
                format!(
                    "Uploaded {file_count} files ({}) to {num_nodes} nodes",
                    crate::ssh_tunnel::humanize_bytes(total_bytes),
                ),
            );
        } else {
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
    }

    // --- Stage 6: Info lines ---
    let teardown_name = &config.name;
    finish_spinner(
        &multi,
        spinner(&multi, ""),
        format!("Tear down with: \x1b[36mvllm gce down {teardown_name} --zone={zone}\x1b[0m"),
    );

    // --- Stage 7: Stream serial console (one line per node, max 20) ---
    let server_ready = Arc::new(tokio::sync::Notify::new());
    let console_nodes = num_nodes.min(20) as usize;
    for (idx, node_instance_name) in instance_names.iter().enumerate().take(console_nodes) {
        let console_pb = multi.add(ProgressBar::new_spinner());
        console_pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.dim} {msg}")
                .unwrap(),
        );
        console_pb.enable_steady_tick(std::time::Duration::from_millis(80));

        let node_name = node_instance_name.clone();
        let label = if num_nodes > 1 {
            let tag = format!("[node {idx}]");
            console_pb.set_message(format!("{tag} Waiting for console output..."));
            Some(tag)
        } else {
            console_pb.set_message("Waiting for console output...".to_string());
            None
        };

        // Only node 0 triggers the server_ready notification.
        let ready = if idx == 0 {
            Some(server_ready.clone())
        } else {
            None
        };

        let stream_client = Instances::builder().build().await?;
        let cfg = ConsoleStreamConfig {
            multi: multi.clone(),
            console_pb,
            client: stream_client,
            instance_name: node_name,
            zone: zone.clone(),
            project: project.clone(),
            server_ready: ready,
            label,
        };
        tokio::spawn(async move {
            let _ = stream_serial_console(cfg).await;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_names_single_node() {
        let names: Vec<String> = vec!["myvm".to_string()];
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "myvm");
    }

    #[test]
    fn test_instance_names_multi_node() {
        let name = "cluster";
        let num_nodes = 3u32;
        let names: Vec<String> = (0..num_nodes).map(|i| format!("{name}-{i}")).collect();
        assert_eq!(names, vec!["cluster-0", "cluster-1", "cluster-2"]);
    }

    #[test]
    fn test_build_zone_list_configured_first() {
        let zones = build_zone_list("us-central1-b");
        assert_eq!(zones[0], ("us-central1".to_string(), "b".to_string()));
        // Configured zone should not appear again in the list.
        let count = zones
            .iter()
            .filter(|(r, s)| r == "us-central1" && s == "b")
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_dev_cloud_config_single_node() {
        let config = build_dev_cloud_config("user", "meta/llama", &[], None, None, None, None);
        assert!(config.contains("#cloud-config"));
        assert!(config.contains("vllm serve"));
        assert!(!config.contains("VLLM_NODE_RANK"));
        assert!(!config.contains("VLLM_NNODES"));
    }

    #[test]
    fn test_dev_cloud_config_head_node() {
        let mn = MultiNodeEnv {
            node_rank: 0,
            nnodes: 2,
            head_ip: String::new(), // empty = self-discover
        };
        let config = build_dev_cloud_config("user", "meta/llama", &[], None, None, None, Some(&mn));
        assert!(config.contains("VLLM_NODE_RANK=0"));
        assert!(config.contains("VLLM_NNODES=2"));
        // Head IP is resolved via GCE metadata.
        assert!(config.contains("metadata.google.internal"));
        assert!(config.contains("VLLM_NCCL_PORT=29400"));
        // Head node runs vllm serve.
        assert!(config.contains("vllm serve"));
    }

    #[test]
    fn test_dev_cloud_config_worker_node() {
        let mn = MultiNodeEnv {
            node_rank: 1,
            nnodes: 2,
            head_ip: "10.128.0.5".to_string(),
        };
        let config = build_dev_cloud_config("user", "meta/llama", &[], None, None, None, Some(&mn));
        assert!(config.contains("VLLM_NODE_RANK=1"));
        assert!(config.contains("VLLM_NNODES=2"));
        assert!(config.contains("VLLM_HEAD_IP=10.128.0.5"));
        assert!(config.contains("VLLM_NCCL_PORT=29400"));
        // Worker node does NOT run vllm serve.
        assert!(!config.contains("vllm serve"));
        assert!(config.contains("build complete"));
        assert!(config.contains("sleep infinity"));
    }
}
