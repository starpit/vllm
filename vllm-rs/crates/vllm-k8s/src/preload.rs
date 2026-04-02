// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm k8s preload` — create a PVC and download models into it via a Job.

use anyhow::Result;
use futures::{AsyncBufReadExt, TryStreamExt};
use k8s_openapi::api::batch::v1::{Job, JobSpec};
use k8s_openapi::api::core::v1::{
    Container, EnvVar, PersistentVolumeClaim, PersistentVolumeClaimSpec,
    PersistentVolumeClaimVolumeSource, PodSpec, PodTemplateSpec, Volume, VolumeMount,
    VolumeResourceRequirements,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;
use kube::ResourceExt;
use kube::api::{ListParams, LogParams, PostParams};
use std::collections::BTreeMap;

/// Configuration for `vllm k8s preload`.
pub struct K8sPreloadConfig {
    /// PVC name.
    pub name: String,
    /// Kubernetes namespace (None = default).
    pub namespace: Option<String>,
    /// Models to download.
    pub models: Vec<String>,
    /// PVC storage size.
    pub size: String,
    /// Storage class (None = cluster default).
    pub storage_class: Option<String>,
    /// Container image (used for the download Job).
    pub image: String,
    /// HuggingFace token.
    pub hf_token: Option<String>,
}

fn build_pvc(config: &K8sPreloadConfig) -> PersistentVolumeClaim {
    PersistentVolumeClaim {
        metadata: ObjectMeta {
            name: Some(config.name.clone()),
            namespace: config.namespace.clone(),
            ..Default::default()
        },
        spec: Some(PersistentVolumeClaimSpec {
            access_modes: Some(vec!["ReadWriteOnce".into()]),
            resources: Some(VolumeResourceRequirements {
                requests: Some(BTreeMap::from([(
                    "storage".into(),
                    Quantity(config.size.clone()),
                )])),
                ..Default::default()
            }),
            storage_class_name: config.storage_class.clone(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn build_download_job(config: &K8sPreloadConfig) -> Job {
    // Build a shell script that downloads each model via `vllm pull`.
    let mut script = String::from("set -e\n");
    for model in &config.models {
        script.push_str(&format!("echo 'Downloading {model}...'\n"));
        script.push_str(&format!("vllm pull '{model}'\n"));
        script.push_str(&format!("echo 'Done: {model}'\n"));
    }
    script.push_str("echo 'All models downloaded.'\n");

    let mut env = Vec::new();
    if let Some(ref token) = config.hf_token {
        env.push(EnvVar {
            name: "HF_TOKEN".into(),
            value: Some(token.clone()),
            ..Default::default()
        });
    }
    // Set HOME to PVC so $HOME/.cache/huggingface lands on the volume.
    // Also set HF_HOME explicitly for libraries that check it.
    env.push(EnvVar {
        name: "HOME".into(),
        value: Some("/models".into()),
        ..Default::default()
    });
    env.push(EnvVar {
        name: "HF_HOME".into(),
        value: Some("/models/.cache/huggingface".into()),
        ..Default::default()
    });

    let job_name = format!("{}-preload", config.name);

    Job {
        metadata: ObjectMeta {
            name: Some(job_name),
            namespace: config.namespace.clone(),
            ..Default::default()
        },
        spec: Some(JobSpec {
            backoff_limit: Some(2),
            template: PodTemplateSpec {
                spec: Some(PodSpec {
                    restart_policy: Some("Never".into()),
                    containers: vec![Container {
                        name: "preload".into(),
                        image: Some(config.image.clone()),
                        command: Some(vec!["/bin/sh".into(), "-c".into(), script]),
                        env: Some(env),
                        volume_mounts: Some(vec![VolumeMount {
                            name: "models".into(),
                            mount_path: "/models".into(),
                            ..Default::default()
                        }]),
                        ..Default::default()
                    }],
                    volumes: Some(vec![Volume {
                        name: "models".into(),
                        persistent_volume_claim: Some(PersistentVolumeClaimVolumeSource {
                            claim_name: config.name.clone(),
                            read_only: Some(false),
                        }),
                        ..Default::default()
                    }]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Create a PVC and run a Job to download models into it.
pub async fn preload(config: K8sPreloadConfig) -> Result<()> {
    let c = crate::client().await?;
    let namespace = config.namespace.clone();
    let pvc_name = config.name.clone();
    let job_name = format!("{}-preload", config.name);

    // Create PVC (ignore AlreadyExists).
    let pvc_api = crate::api::<PersistentVolumeClaim>(c.clone(), &namespace)?;
    let pvc = build_pvc(&config);
    match pvc_api.create(&PostParams::default(), &pvc).await {
        Ok(_) => eprintln!("  Created PVC '{pvc_name}' ({})", config.size),
        Err(kube::Error::Api(ref e)) if e.code == 409 => {
            eprintln!("  PVC '{pvc_name}' already exists, reusing");
        }
        Err(e) => return Err(e.into()),
    }

    // Create download Job (delete any stale one first).
    let job_api = crate::api::<Job>(c.clone(), &namespace)?;
    let dp = kube::api::DeleteParams {
        propagation_policy: Some(kube::api::PropagationPolicy::Background),
        ..Default::default()
    };
    match job_api.delete(&job_name, &dp).await {
        Ok(_) => {
            eprintln!("  Cleaned up previous Job '{job_name}'");
            // Brief wait for deletion to propagate.
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        }
        Err(kube::Error::Api(ref e)) if e.code == 404 => {}
        Err(e) => return Err(e.into()),
    }

    let job = build_download_job(&config);
    eprintln!(
        "  Creating download Job '{job_name}' for {} model(s)...",
        config.models.len()
    );
    job_api.create(&PostParams::default(), &job).await?;

    // Wait for pod to appear, then stream logs.
    let pod_api = crate::api::<k8s_openapi::api::core::v1::Pod>(c.clone(), &namespace)?;
    let label_selector = format!("job-name={job_name}");

    eprintln!("  Waiting for download pod...");
    let pod_name = loop {
        let pods = pod_api
            .list(&ListParams::default().labels(&label_selector))
            .await?;
        if let Some(pod) = pods.items.first() {
            break pod.name_any();
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    };
    // Wait for pod to be running (not Pending).
    eprintln!("  Waiting for pod '{pod_name}' to start...");
    loop {
        let pod = pod_api.get(&pod_name).await?;
        let phase = pod
            .status
            .as_ref()
            .and_then(|s| s.phase.as_deref())
            .unwrap_or("Unknown");
        match phase {
            "Running" | "Succeeded" => break,
            "Failed" => anyhow::bail!("download pod failed before starting"),
            _ => {
                // Still Pending — show reason if available.
                if let Some(reason) = pod
                    .status
                    .as_ref()
                    .and_then(|s| s.conditions.as_ref())
                    .and_then(|c| c.iter().find(|c| c.type_ == "PodScheduled"))
                    .and_then(|c| c.message.as_deref())
                {
                    eprintln!("  Pending: {reason}");
                }
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            }
        }
    }

    eprintln!("  Streaming logs from '{pod_name}'...");

    // Stream logs once with follow=true. When the stream ends, the pod is done.
    match pod_api
        .log_stream(
            &pod_name,
            &LogParams {
                follow: true,
                ..LogParams::default()
            },
        )
        .await
    {
        Ok(stream) => {
            let mut lines = stream.lines();
            while let Ok(Some(line)) = lines.try_next().await {
                eprintln!("  [{pod_name}] {line}");
            }
        }
        Err(e) => {
            eprintln!("  Warning: could not stream logs: {e}");
        }
    }

    // Check job outcome.
    // Poll briefly — the job controller may not have updated status yet.
    let mut attempts = 0;
    let succeeded = loop {
        let job = job_api.get(&job_name).await?;
        let succeeded = job.status.as_ref().and_then(|s| s.succeeded).unwrap_or(0);
        let failed = job.status.as_ref().and_then(|s| s.failed).unwrap_or(0);
        if succeeded > 0 {
            break true;
        }
        if failed > 0 {
            break false;
        }
        attempts += 1;
        if attempts > 15 {
            break false;
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    };

    // Clean up the job (and its pods) regardless of outcome.
    let dp = kube::api::DeleteParams {
        propagation_policy: Some(kube::api::PropagationPolicy::Background),
        ..Default::default()
    };
    if let Err(e) = job_api.delete(&job_name, &dp).await {
        eprintln!("  Warning: failed to clean up job '{job_name}': {e}");
    }

    if succeeded {
        eprintln!(
            "  Preload complete. PVC '{pvc_name}' ready for use with: vllm k8s up -P {pvc_name}"
        );
    } else {
        anyhow::bail!("preload job failed — check pod logs");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> K8sPreloadConfig {
        K8sPreloadConfig {
            name: "mymodels".into(),
            namespace: None,
            models: vec!["Qwen/Qwen2.5-0.5B".into(), "meta-llama/Llama-3-8B".into()],
            size: "100Gi".into(),
            storage_class: None,
            image: crate::up::DEFAULT_IMAGE.into(),
            hf_token: Some("hf_test".into()),
        }
    }

    #[test]
    fn pvc_has_rwo() {
        let pvc = build_pvc(&test_config());
        let modes = pvc.spec.as_ref().unwrap().access_modes.as_ref().unwrap();
        assert_eq!(modes, &["ReadWriteOnce".to_string()]);
    }

    #[test]
    fn pvc_has_requested_size() {
        let pvc = build_pvc(&test_config());
        let requests = pvc
            .spec
            .as_ref()
            .unwrap()
            .resources
            .as_ref()
            .unwrap()
            .requests
            .as_ref()
            .unwrap();
        assert_eq!(requests.get("storage").unwrap().0, "100Gi");
    }

    #[test]
    fn job_downloads_all_models() {
        let job = build_download_job(&test_config());
        let spec = job.spec.as_ref().unwrap();
        let container = &spec.template.spec.as_ref().unwrap().containers[0];
        let cmd = container.command.as_ref().unwrap().join(" ");
        assert!(cmd.contains("vllm pull 'Qwen/Qwen2.5-0.5B'"));
        assert!(cmd.contains("vllm pull 'meta-llama/Llama-3-8B'"));
    }

    #[test]
    fn job_mounts_pvc() {
        let job = build_download_job(&test_config());
        let pod_spec = job.spec.as_ref().unwrap().template.spec.as_ref().unwrap();

        // Volume references PVC
        let vol = &pod_spec.volumes.as_ref().unwrap()[0];
        assert_eq!(
            vol.persistent_volume_claim.as_ref().unwrap().claim_name,
            "mymodels"
        );

        let mount = &pod_spec.containers[0].volume_mounts.as_ref().unwrap()[0];
        assert_eq!(mount.mount_path, "/models");
    }

    #[test]
    fn job_sets_hf_home_and_token() {
        let job = build_download_job(&test_config());
        let env = job
            .spec
            .as_ref()
            .unwrap()
            .template
            .spec
            .as_ref()
            .unwrap()
            .containers[0]
            .env
            .as_ref()
            .unwrap();
        assert!(
            env.iter()
                .any(|e| e.name == "HF_TOKEN" && e.value.as_deref() == Some("hf_test"))
        );
        assert!(env.iter().any(
            |e| e.name == "HF_HOME" && e.value.as_deref() == Some("/models/.cache/huggingface")
        ));
    }
}
