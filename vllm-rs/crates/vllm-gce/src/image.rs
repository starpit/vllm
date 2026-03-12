// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm gce image build` — build a GCE image with dev toolchain pre-installed.
//! `vllm gce image list` — list GCE images tagged for vllm-rs.

use anyhow::Result;
use indicatif::MultiProgress;

use crate::common::{
    CreateError, build_zone_list, delete_instance, finish_spinner, spinner, try_create_instance,
};

// ---------------------------------------------------------------------------
// Image build
// ---------------------------------------------------------------------------

/// Configuration for `vllm gce image build`.
pub struct ImageBuildConfig {
    /// Base GCE boot image (full self-link).
    pub image: String,
    /// Image family tag (label value and name prefix).
    pub tag: String,
    /// Image version string (appended to name).
    pub version: String,
    /// Local source directory to upload and build on the VM.
    pub source_dir: std::path::PathBuf,
    /// Build a production image (install binary, remove toolchain).
    pub production: bool,
    /// GCE project.
    pub project: Option<String>,
    /// Path to GCP service account credentials JSON.
    pub gcp_credentials: Option<String>,
    /// GCP service account email for the builder instance.
    pub gcp_service_account: Option<String>,
    /// GCS bucket for sccache.
    pub sccache_gcs_bucket: Option<String>,
    /// GCS key prefix for sccache.
    pub sccache_gcs_key_prefix: Option<String>,
}

/// Build a cloud-init that installs everything `gce up --dev` does, minus
/// the final `vllm serve`. Writes a sentinel file when done.
fn build_image_cloud_config(
    username: &str,
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

    // Append env vars to ~/.cargo/env.
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
    for export in &env_exports {
        lines.push(format!("  - echo '{export}' >> {home}/.cargo/env"));
    }

    // Write sentinel so we know cloud-init finished.
    lines.push("  - echo 'boot-finished-vllm'".to_string());

    lines.join("\n")
}

/// Build a GCE Instance object for the builder VM in the given zone.
/// Mirrors `up`'s instance creation: SSH keys metadata, service account with
/// storage scopes (for sccache GCS), etc.
fn build_builder_instance(
    builder_name: &str,
    zone: &str,
    region: &str,
    source_image: &str,
    cloud_config: &str,
    service_account: Option<&str>,
    ssh_keys_metadata: Option<&str>,
) -> google_cloud_compute_v1::model::Instance {
    use google_cloud_compute_v1::model::{
        AcceleratorConfig, AccessConfig, AttachedDisk, AttachedDiskInitializeParams, Instance,
        Metadata, NetworkInterface, Scheduling, ServiceAccount, metadata::Items as MetadataItems,
    };

    let mut metadata_items = vec![
        MetadataItems::default()
            .set_key("user-data")
            .set_value(cloud_config),
        MetadataItems::default()
            .set_key("enable-osconfig")
            .set_value("TRUE"),
    ];
    if let Some(ssh_keys) = ssh_keys_metadata {
        metadata_items.push(
            MetadataItems::default()
                .set_key("ssh-keys")
                .set_value(ssh_keys),
        );
    }

    let mut instance = Instance::new()
        .set_name(builder_name)
        .set_machine_type(format!("zones/{zone}/machineTypes/g2-standard-4"))
        .set_disks([AttachedDisk::new()
            .set_boot(true)
            .set_auto_delete(true)
            .set_initialize_params(
                AttachedDiskInitializeParams::new()
                    .set_source_image(source_image)
                    .set_disk_size_gb(300)
                    .set_disk_type(format!("zones/{zone}/diskTypes/pd-ssd")),
            )
            .set_mode("READ_WRITE")])
        .set_network_interfaces([NetworkInterface::new()
            .set_subnetwork(format!("regions/{region}/subnetworks/default"))
            .set_access_configs([AccessConfig::new().set_network_tier("PREMIUM")])
            .set_stack_type("IPV4_ONLY")])
        .set_guest_accelerators([AcceleratorConfig::new()
            .set_accelerator_count(1)
            .set_accelerator_type(format!("zones/{zone}/acceleratorTypes/nvidia-l4"))])
        .set_metadata(Metadata::default().set_items(metadata_items))
        .set_can_ip_forward(false)
        .set_deletion_protection(false)
        // Use SPOT to save cost on the builder VM.
        .set_scheduling(
            Scheduling::new()
                .set_automatic_restart(false)
                .set_on_host_maintenance("TERMINATE")
                .set_preemptible(true)
                .set_provisioning_model("SPOT")
                .set_instance_termination_action("STOP"),
        );

    // Always attach a service account with storage scopes (for sccache GCS).
    // Same pattern as `up`: devstorage.read_write + logging + monitoring.
    {
        let mut sa = ServiceAccount::new().set_scopes([
            "https://www.googleapis.com/auth/devstorage.read_write",
            "https://www.googleapis.com/auth/logging.write",
            "https://www.googleapis.com/auth/monitoring.write",
        ]);
        if let Some(sa_email) = service_account {
            sa = sa.set_email(sa_email);
        }
        instance = instance.set_service_accounts([sa]);
    }

    instance
}

/// Build a GCE image by:
/// 1. Spinning up a temporary VM with cloud-init that installs the dev toolchain.
/// 2. Waiting for cloud-init to complete (sentinel file via serial console).
/// 3. Stopping the VM.
/// 4. Creating an image from the VM's disk.
/// 5. Deleting the temporary VM.
pub async fn build(config: ImageBuildConfig) -> Result<()> {
    use google_cloud_compute_v1::client::{Images, Instances};

    let project = config
        .project
        .or_else(|| std::env::var("GCP_PROJECT").ok())
        .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "GCP project required: set --project, GCP_PROJECT, or GOOGLE_CLOUD_PROJECT"
            )
        })?;

    let username = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "user".to_string());

    let image_name = format!("{}-{}", config.tag, config.version);
    let builder_name = format!("{image_name}-builder");

    let multi = MultiProgress::new();

    // --- Authenticate ---
    let sp = spinner(&multi, "Authenticating with GCE API...");
    let instances_client = Instances::builder().build().await?;
    let images_client = Images::builder().build().await?;
    finish_spinner(&multi, sp, "Authenticated with GCE API");

    // --- Create builder VM (with zone fallback) ---
    let cloud_config = build_image_cloud_config(
        &username,
        config.sccache_gcs_bucket.as_deref(),
        config.sccache_gcs_key_prefix.as_deref(),
    );

    let ssh_keys_metadata = crate::common::read_ssh_public_key();

    let zones_to_try = build_zone_list("us-central1-a");
    let sp = spinner(
        &multi,
        &format!("Creating builder instance '{builder_name}'..."),
    );

    let mut succeeded_zone = None;
    for (region, suffix) in &zones_to_try {
        let zone = format!("{region}-{suffix}");
        sp.set_message(format!(
            "Creating builder instance '{builder_name}' in {zone} (SPOT)..."
        ));

        let instance = build_builder_instance(
            &builder_name,
            &zone,
            region,
            &config.image,
            &cloud_config,
            config.gcp_service_account.as_deref(),
            ssh_keys_metadata.as_deref(),
        );

        match try_create_instance(&instances_client, &project, &zone, instance).await {
            Ok(()) => {
                succeeded_zone = Some(zone);
                break;
            }
            Err(CreateError::ZoneExhausted(msg)) => {
                let _ = multi.println(format!("  {zone}: {msg}"));
                continue;
            }
            Err(CreateError::Fatal(e)) => {
                finish_spinner(&multi, sp, format!("Failed to create builder in {zone}"));
                return Err(e);
            }
        }
    }

    let zone = match succeeded_zone {
        Some(z) => z,
        None => {
            finish_spinner(&multi, sp, "All zones exhausted");
            anyhow::bail!(
                "could not create builder instance in any zone — all resource pools exhausted"
            );
        }
    };

    finish_spinner(
        &multi,
        sp,
        format!("Builder instance '{builder_name}' created in {zone}"),
    );

    // Run the remaining stages, cleaning up the builder VM on Ctrl+C or failure.
    let result = run_image_stages(
        &instances_client,
        &images_client,
        &project,
        &zone,
        &builder_name,
        &image_name,
        &config.image,
        &config.tag,
        &config.version,
        &config.source_dir,
        config.production,
        config.sccache_gcs_bucket.as_deref(),
        config.sccache_gcs_key_prefix.as_deref(),
        &multi,
    )
    .await;

    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            eprintln!("  Cleaning up builder instance '{builder_name}'...");
            let _ = delete_instance(&instances_client, &project, &zone, &builder_name).await;
            eprintln!("  Builder instance '{builder_name}' deleted");
            Err(e)
        }
    }
}

/// Stages after VM creation: wait for cloud-init, SSH in, upload source, build,
/// optionally clean up for production, stop, create image, delete builder.
/// Wrapped in select! so Ctrl+C returns an error and the caller cleans up.
#[allow(clippy::too_many_arguments)]
async fn run_image_stages(
    instances_client: &google_cloud_compute_v1::client::Instances,
    images_client: &google_cloud_compute_v1::client::Images,
    project: &str,
    zone: &str,
    builder_name: &str,
    image_name: &str,
    base_image: &str,
    tag: &str,
    version: &str,
    source_dir: &std::path::Path,
    production: bool,
    sccache_gcs_bucket: Option<&str>,
    sccache_gcs_key_prefix: Option<&str>,
    multi: &MultiProgress,
) -> Result<()> {
    use google_cloud_compute_v1::model::Image;
    use google_cloud_lro::Poller;

    let work = async {
        // --- Wait for cloud-init to finish ---
        let _ = multi.println(
            "  Waiting for cloud-init to finish (installing packages, rustup, sccache)...",
        );

        let sentinel_ok =
            wait_for_cloud_init(instances_client, project, zone, builder_name, multi).await;

        if !sentinel_ok {
            let _ = multi.println("  Cloud-init did not complete successfully");
            anyhow::bail!("cloud-init failed on builder instance");
        }

        let _ = multi.println("  Cloud-init completed successfully");

        // --- Get external IP and SSH connect ---
        let sp = spinner(multi, "Fetching builder external IP...");
        let instance_info = instances_client
            .get()
            .set_project(project)
            .set_zone(zone)
            .set_instance(builder_name)
            .send()
            .await?;

        let external_ip = instance_info
            .network_interfaces
            .iter()
            .find_map(|ni| ni.access_configs.iter().find_map(|ac| ac.nat_ip.as_ref()))
            .ok_or_else(|| anyhow::anyhow!("no external IP found for builder instance"))?
            .clone();
        finish_spinner(multi, sp, format!("Builder IP: {external_ip}"));

        let sp = spinner(multi, &format!("Connecting via SSH to {external_ip}..."));
        let session = crate::ssh_tunnel::SshSession::connect(&external_ip, 22).await?;
        finish_spinner(multi, sp, format!("SSH connected to {external_ip}"));

        // --- Upload source ---
        let sp = spinner(multi, &format!("Uploading {}...", source_dir.display()));
        let (file_count, total_bytes) = session
            .upload_dir(source_dir, "~/vllm-rs", Some(&sp))
            .await?;
        finish_spinner(
            multi,
            sp,
            format!(
                "Uploaded {file_count} files ({})",
                crate::ssh_tunnel::humanize_bytes(total_bytes),
            ),
        );

        // --- Build on VM ---
        let _ = multi.println("  Building vllm-rs on VM (this may take a while)...");
        // Inline env vars on the command line, same as `up --dev` does.
        let mut env_prefix = "RUSTC_WRAPPER=sccache SCCACHE_GCS_RW_MODE=READ_WRITE".to_string();
        if let Some(bucket) = sccache_gcs_bucket {
            env_prefix.push_str(&format!(" SCCACHE_GCS_BUCKET={bucket}"));
        }
        if let Some(prefix) = sccache_gcs_key_prefix {
            env_prefix.push_str(&format!(" SCCACHE_GCS_KEY_PREFIX={prefix}"));
        }
        let build_cmd = format!(
            "source ~/.cargo/env && cd ~/vllm-rs && {env_prefix} cargo build --release -p vllm-cli --features cuda"
        );
        let exit_code = session.exec_streaming(&build_cmd, Some(multi)).await?;
        if exit_code != 0 {
            anyhow::bail!("remote build failed with exit code {exit_code}");
        }
        let _ = multi.println("  Build completed successfully");

        // --- Production cleanup (optional) ---
        if production {
            let _ = multi.println("  Production mode: installing binary and cleaning up...");
            let cleanup_cmds = concat!(
                "sudo cp ~/vllm-rs/target/release/vllm /usr/local/bin/vllm && ",
                "rm -rf ~/vllm-rs && ",
                "sudo rm -f /usr/local/bin/sccache && ",
                "sudo apt-get remove -y build-essential pkg-config libssl-dev && ",
                "sudo apt-get autoremove -y && ",
                "source ~/.cargo/env && rustup self uninstall -y"
            );
            let exit_code = session.exec_streaming(cleanup_cmds, Some(multi)).await?;
            if exit_code != 0 {
                let _ = multi.println(format!(
                    "  Warning: production cleanup exited with code {exit_code}"
                ));
            }
            let _ = multi.println("  Production cleanup completed");
        }

        // --- Stop the VM ---
        let sp = spinner(
            multi,
            &format!("Stopping builder instance '{builder_name}'..."),
        );
        instances_client
            .stop()
            .set_project(project)
            .set_zone(zone)
            .set_instance(builder_name)
            .poller()
            .until_done()
            .await?
            .to_result()?;
        finish_spinner(
            multi,
            sp,
            format!("Builder instance '{builder_name}' stopped"),
        );

        // --- Create image from the stopped VM's disk ---
        let sp = spinner(multi, &format!("Creating image '{image_name}'..."));

        let image = Image::new()
            .set_name(image_name)
            .set_description(format!("vllm-rs dev image built from {base_image}"))
            .set_source_disk(format!(
                "projects/{project}/zones/{zone}/disks/{builder_name}"
            ))
            .set_labels([
                ("vllm-rs-tag".to_string(), tag.to_string()),
                ("vllm-rs-version".to_string(), version.to_string()),
            ]);

        images_client
            .insert()
            .set_project(project)
            .set_body(image)
            .poller()
            .until_done()
            .await?
            .to_result()?;

        finish_spinner(multi, sp, format!("Image '{image_name}' created"));

        // --- Delete the builder VM ---
        let sp = spinner(
            multi,
            &format!("Deleting builder instance '{builder_name}'..."),
        );
        delete_instance(instances_client, project, zone, builder_name).await?;
        finish_spinner(
            multi,
            sp,
            format!("Builder instance '{builder_name}' deleted"),
        );

        eprintln!();
        eprintln!("  Image ready: projects/{project}/global/images/{image_name}");
        eprintln!(
            "  Use with: vllm gce up <name> --image projects/{project}/global/images/{image_name} ..."
        );

        Ok(())
    };

    tokio::select! {
        result = work => result,
        _ = tokio::signal::ctrl_c() => {
            eprintln!("\n  Ctrl+C received, cancelling...");
            anyhow::bail!("interrupted by Ctrl+C")
        }
    }
}

/// Strip timestamp and hostname prefix from serial console lines for readability.
/// Handles syslog format: "2026-03-12T14:28:43.855... hostname cloud-init[658]: message"
/// and kernel format: "[    0.294885] message"
fn strip_serial_prefix(line: &str) -> &str {
    let s = line.trim();

    // Syslog format: "TIMESTAMP HOSTNAME REST" — skip first two space-delimited fields.
    if s.starts_with("20") {
        let mut spaces = 0;
        for (i, c) in s.char_indices() {
            if c == ' ' {
                spaces += 1;
                if spaces == 2 {
                    return s[i + 1..].trim();
                }
            }
        }
    }

    // Kernel format: "[  TIMESTAMP] message"
    if s.starts_with('[')
        && let Some(idx) = s.find("] ")
    {
        return s[idx + 2..].trim();
    }

    s
}

/// Poll serial console output for the cloud-init sentinel file.
async fn wait_for_cloud_init(
    client: &google_cloud_compute_v1::client::Instances,
    project: &str,
    zone: &str,
    instance_name: &str,
    multi: &MultiProgress,
) -> bool {
    let mut last_start = 0i64;
    let mut consecutive_errors = 0;
    let mut seen_lines: std::collections::HashSet<String> = std::collections::HashSet::new();

    loop {
        match client
            .get_serial_port_output()
            .set_project(project)
            .set_zone(zone)
            .set_instance(instance_name)
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
                    if let Some(next) = output.next {
                        last_start = next;
                    }

                    // Check for our sentinel or cloud-init finished message.
                    if contents.contains("boot-finished-vllm")
                        || contents.contains("Cloud-init v.") && contents.contains("finished at")
                    {
                        return true;
                    }

                    // Print only cloud-init and runcmd lines (skip kernel boot spam).
                    let mut saw_error = false;
                    for line in contents.lines() {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }

                        // Strip prefix for dedup and display.
                        let display = strip_serial_prefix(trimmed);

                        // Dedup — serial console polls can return overlapping content.
                        if !seen_lines.insert(display.to_string()) {
                            continue;
                        }

                        if display.contains("CRITICAL") || display.contains("FATAL") {
                            saw_error = true;
                            let _ = multi.println(format!("    {display}"));
                            continue;
                        }
                        // Show cloud-init progress, apt, rustup, sccache, cargo lines.
                        let dominated_by_cloud_init = display.contains("cloud-init")
                            || display.contains("runcmd")
                            || display.contains("apt")
                            || display.contains("rustup")
                            || display.contains("sccache")
                            || display.contains("cargo")
                            || display.contains("curl")
                            || display.contains("growpart")
                            || display.contains("resize2fs")
                            || display.contains("boot-finished");
                        if dominated_by_cloud_init {
                            let _ = multi.println(format!("    {display}"));
                        }
                    }

                    if saw_error {
                        return false;
                    }
                }
            }
            Err(_) => {
                consecutive_errors += 1;
                // Cloud-init can take 10+ minutes (nvidia-cuda-toolkit, etc.).
                // The serial console API often returns transient errors during
                // long quiet periods. Be patient: 120 × 5s = 10 minutes of
                // consecutive errors before giving up.
                if consecutive_errors > 120 {
                    let _ = multi.println("  Too many serial console errors, giving up");
                    return false;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }
}

// ---------------------------------------------------------------------------
// Image list
// ---------------------------------------------------------------------------

/// Configuration for `vllm gce image list`.
pub struct ImageListConfig {
    /// Image family tag to filter on. None = show all vllm-rs images.
    pub tag: Option<String>,
    /// GCE project.
    pub project: Option<String>,
    /// Path to GCP service account credentials JSON.
    pub gcp_credentials: Option<String>,
}

/// List GCE images with the given tag label.
pub async fn list(config: ImageListConfig) -> Result<()> {
    use google_cloud_compute_v1::client::Images;

    let project = config
        .project
        .or_else(|| std::env::var("GCP_PROJECT").ok())
        .or_else(|| std::env::var("GOOGLE_CLOUD_PROJECT").ok())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "GCP project required: set --project, GCP_PROJECT, or GOOGLE_CLOUD_PROJECT"
            )
        })?;

    let client = Images::builder().build().await?;

    // Filter: specific tag or any vllm-rs-tagged image.
    let filter = match &config.tag {
        Some(tag) => format!("labels.vllm-rs-tag={tag}"),
        None => "labels.vllm-rs-tag:*".to_string(),
    };

    let response = client
        .list()
        .set_project(&project)
        .set_filter(filter)
        .send()
        .await?;

    let images = response.items;
    if images.is_empty() {
        match &config.tag {
            Some(tag) => eprintln!("No images found with tag '{tag}'"),
            None => eprintln!("No vllm-rs images found"),
        }
        return Ok(());
    }

    // Sort by creation time descending (newest first).
    let mut entries: Vec<(String, String, String, String, String)> = images
        .into_iter()
        .map(|img| {
            let name = img.name.unwrap_or_default();
            let tag = img.labels.get("vllm-rs-tag").cloned().unwrap_or_default();
            let status = format!("{:?}", img.status.unwrap_or_default());
            let created = img.creation_timestamp.unwrap_or_default();
            let version = img
                .labels
                .get("vllm-rs-version")
                .cloned()
                .unwrap_or_default();
            (name, tag, version, status, created)
        })
        .collect();
    entries.sort_by(|a, b| b.4.cmp(&a.4));

    // Print as table.
    let header = format!(
        "{:<50} {:<15} {:<20} {:<10} {}",
        "NAME", "TAG", "VERSION", "STATUS", "CREATED"
    );
    eprintln!("{header}");
    for (name, tag, version, status, created) in &entries {
        eprintln!("{name:<50} {tag:<15} {version:<20} {status:<10} {created}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Latest image lookup
// ---------------------------------------------------------------------------

/// Find the latest READY image with the given tag label.
/// Returns the full self-link (`projects/{project}/global/images/{name}`).
pub async fn latest_image(project: &str, tag: &str) -> Result<Option<String>> {
    use google_cloud_compute_v1::client::Images;

    let client = Images::builder().build().await?;

    let response = client
        .list()
        .set_project(project)
        .set_filter(format!("labels.vllm-rs-tag={tag} AND status=READY"))
        .send()
        .await?;

    let images = response.items;
    if images.is_empty() {
        return Ok(None);
    }

    // Find newest by creation_timestamp (lexicographic sort works for ISO timestamps).
    let newest = images
        .into_iter()
        .filter_map(|img| {
            let name = img.name?;
            let ts = img.creation_timestamp.unwrap_or_default();
            Some((name, ts))
        })
        .max_by(|a, b| a.1.cmp(&b.1));

    Ok(newest.map(|(name, _)| format!("projects/{project}/global/images/{name}")))
}
