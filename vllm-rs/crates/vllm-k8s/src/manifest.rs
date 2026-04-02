// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Build Kubernetes manifests programmatically for single-node and multi-node TP.

use std::collections::BTreeMap;

use k8s_openapi::api::apps::v1::{
    Deployment, DeploymentSpec, StatefulSet, StatefulSetSpec, StatefulSetUpdateStrategy,
};
use k8s_openapi::api::core::v1::{
    Container, ContainerPort, EmptyDirVolumeSource, EnvVar, HTTPGetAction,
    PersistentVolumeClaimVolumeSource, PodSpec, PodTemplateSpec, Probe, ResourceRequirements,
    Service, ServicePort, ServiceSpec, Volume, VolumeMount,
};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{LabelSelector, ObjectMeta};
use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;

/// Configuration for manifest generation.
pub struct ManifestConfig {
    pub name: String,
    pub instance_id: String,
    pub namespace: Option<String>,
    pub nodes: u32,
    pub gpu_count: u32,
    pub image: String,
    pub model: String,
    pub hf_token: Option<String>,
    pub serve_args: Vec<String>,
    pub shm_size: String,
    /// PVC name for preloaded models (mounted read-only at /root/.cache/huggingface).
    pub preload_pvc: Option<String>,
}

/// The set of K8s resources to create.
pub enum Manifests {
    /// Single-node: just a Deployment.
    SingleNode { deployment: Box<Deployment> },
    /// Multi-node TP: headless Service + StatefulSet.
    MultiNode {
        service: Box<Service>,
        stateful_set: Box<StatefulSet>,
    },
}

fn labels(name: &str, instance_id: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("app.kubernetes.io/name".into(), name.into()),
        ("app.kubernetes.io/instance".into(), instance_id.into()),
        ("app.kubernetes.io/component".into(), "vllm".into()),
    ])
}

fn selector_labels(name: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("app.kubernetes.io/name".into(), name.into()),
        ("app.kubernetes.io/component".into(), "vllm".into()),
    ])
}

fn env_vars(config: &ManifestConfig) -> Vec<EnvVar> {
    let mut env = vec![EnvVar {
        name: "MODEL".into(),
        value: Some(config.model.clone()),
        ..Default::default()
    }];
    if let Some(ref token) = config.hf_token {
        env.push(EnvVar {
            name: "HF_TOKEN".into(),
            value: Some(token.clone()),
            ..Default::default()
        });
    }
    env
}

fn startup_probe() -> Probe {
    Probe {
        http_get: Some(HTTPGetAction {
            path: Some("/v1/models".into()),
            port: IntOrString::Int(8000),
            ..Default::default()
        }),
        initial_delay_seconds: Some(15),
        period_seconds: Some(5),
        timeout_seconds: Some(5),
        failure_threshold: Some(120), // 10 min max startup
        ..Default::default()
    }
}

fn readiness_probe() -> Probe {
    Probe {
        http_get: Some(HTTPGetAction {
            path: Some("/health".into()),
            port: IntOrString::Int(8000),
            ..Default::default()
        }),
        period_seconds: Some(30),
        timeout_seconds: Some(2),
        failure_threshold: Some(3),
        ..Default::default()
    }
}

fn shm_volume(size: &str) -> Volume {
    Volume {
        name: "dshm".into(),
        empty_dir: Some(EmptyDirVolumeSource {
            medium: Some("Memory".into()),
            size_limit: Some(Quantity(size.into())),
        }),
        ..Default::default()
    }
}

fn shm_mount() -> VolumeMount {
    VolumeMount {
        name: "dshm".into(),
        mount_path: "/dev/shm".into(),
        ..Default::default()
    }
}

fn gpu_resources(gpu_count: u32) -> ResourceRequirements {
    ResourceRequirements {
        limits: Some(BTreeMap::from([(
            "nvidia.com/gpu".into(),
            Quantity(gpu_count.to_string()),
        )])),
        ..Default::default()
    }
}

/// Build the container command for `vllm serve`.
fn serve_command(config: &ManifestConfig) -> Vec<String> {
    let mut extra = String::new();
    for arg in &config.serve_args {
        extra.push(' ');
        extra.push_str(arg);
    }
    vec![
        "/bin/sh".into(),
        "-c".into(),
        format!("vllm serve --model $MODEL{extra}"),
    ]
}

/// Build the container command for multi-node TP.
/// Each pod computes its rank from the StatefulSet ordinal in the hostname.
fn multi_node_command(config: &ManifestConfig) -> Vec<String> {
    let svc = format!("{}-headless", config.name);
    let ns_suffix = config
        .namespace
        .as_deref()
        .map(|ns| format!(".{ns}.svc"))
        .unwrap_or_default();

    let mut extra = String::new();
    for arg in &config.serve_args {
        extra.push(' ');
        extra.push_str(arg);
    }

    // Shell script: extract ordinal from hostname, set TP env vars.
    let script = format!(
        r#"ORDINAL=${{HOSTNAME##*-}}
export VLLM_NODE_RANK=$ORDINAL
export VLLM_NNODES={nodes}
export VLLM_HEAD_IP={name}-0.{svc}{ns_suffix}
export VLLM_NCCL_PORT=29400
if [ "$ORDINAL" = "0" ]; then
  exec vllm serve --model $MODEL{extra}
else
  echo "Worker node $ORDINAL waiting for head node..."
  sleep infinity
fi"#,
        nodes = config.nodes,
        name = config.name,
    );

    vec!["/bin/sh".into(), "-c".into(), script]
}

fn container(config: &ManifestConfig, multi_node: bool) -> Container {
    let command = if multi_node {
        multi_node_command(config)
    } else {
        serve_command(config)
    };

    let mut mounts = vec![shm_mount()];
    let mut env = env_vars(config);

    // Writable home for OpenShift (random UID has no home dir).
    env.push(EnvVar {
        name: "HOME".into(),
        value: Some("/tmp".into()),
        ..Default::default()
    });

    if config.preload_pvc.is_some() {
        // Mount preloaded models read-only, point HF_HOME at the cache
        // the preload job wrote (under /models/.cache/huggingface).
        mounts.push(VolumeMount {
            name: "preload-models".into(),
            mount_path: "/models".into(),
            read_only: Some(true),
            ..Default::default()
        });
        env.push(EnvVar {
            name: "HF_HOME".into(),
            value: Some("/models/.cache/huggingface".into()),
            ..Default::default()
        });
    }

    Container {
        name: "vllm".into(),
        image: Some(config.image.clone()),
        command: Some(command),
        ports: Some(vec![ContainerPort {
            container_port: 8000,
            protocol: Some("TCP".into()),
            ..Default::default()
        }]),
        env: Some(env),
        resources: Some(gpu_resources(config.gpu_count)),
        volume_mounts: Some(mounts),
        startup_probe: Some(startup_probe()),
        readiness_probe: Some(readiness_probe()),
        ..Default::default()
    }
}

fn pod_spec(config: &ManifestConfig, multi_node: bool) -> PodSpec {
    let mut volumes = vec![shm_volume(&config.shm_size)];

    if let Some(ref pvc_name) = config.preload_pvc {
        volumes.push(Volume {
            name: "preload-models".into(),
            persistent_volume_claim: Some(PersistentVolumeClaimVolumeSource {
                claim_name: pvc_name.clone(),
                read_only: Some(true),
            }),
            ..Default::default()
        });
    }

    PodSpec {
        containers: vec![container(config, multi_node)],
        volumes: Some(volumes),
        ..Default::default()
    }
}

fn pod_template(config: &ManifestConfig, multi_node: bool) -> PodTemplateSpec {
    PodTemplateSpec {
        metadata: Some(ObjectMeta {
            labels: Some(labels(&config.name, &config.instance_id)),
            ..Default::default()
        }),
        spec: Some(pod_spec(config, multi_node)),
    }
}

fn build_deployment(config: &ManifestConfig) -> Deployment {
    Deployment {
        metadata: ObjectMeta {
            name: Some(config.name.clone()),
            namespace: config.namespace.clone(),
            labels: Some(labels(&config.name, &config.instance_id)),
            ..Default::default()
        },
        spec: Some(DeploymentSpec {
            replicas: Some(1),
            selector: LabelSelector {
                match_labels: Some(selector_labels(&config.name)),
                ..Default::default()
            },
            template: pod_template(config, false),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn build_headless_service(config: &ManifestConfig) -> Service {
    Service {
        metadata: ObjectMeta {
            name: Some(format!("{}-headless", config.name)),
            namespace: config.namespace.clone(),
            ..Default::default()
        },
        spec: Some(ServiceSpec {
            cluster_ip: Some("None".into()),
            selector: Some(selector_labels(&config.name)),
            ports: Some(vec![ServicePort {
                port: 8000,
                target_port: Some(IntOrString::Int(8000)),
                protocol: Some("TCP".into()),
                ..Default::default()
            }]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn build_stateful_set(config: &ManifestConfig) -> StatefulSet {
    StatefulSet {
        metadata: ObjectMeta {
            name: Some(config.name.clone()),
            namespace: config.namespace.clone(),
            labels: Some(labels(&config.name, &config.instance_id)),
            ..Default::default()
        },
        spec: Some(StatefulSetSpec {
            replicas: Some(config.nodes as i32),
            service_name: Some(format!("{}-headless", config.name)),
            selector: LabelSelector {
                match_labels: Some(selector_labels(&config.name)),
                ..Default::default()
            },
            template: pod_template(config, true),
            pod_management_policy: Some("Parallel".into()),
            update_strategy: Some(StatefulSetUpdateStrategy {
                type_: Some("RollingUpdate".into()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Build the K8s manifests for the given configuration.
pub fn build(config: ManifestConfig) -> Manifests {
    if config.nodes <= 1 {
        Manifests::SingleNode {
            deployment: Box::new(build_deployment(&config)),
        }
    } else {
        Manifests::MultiNode {
            service: Box::new(build_headless_service(&config)),
            stateful_set: Box::new(build_stateful_set(&config)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(nodes: u32) -> ManifestConfig {
        ManifestConfig {
            name: "test-vllm".into(),
            instance_id: "test-uuid".into(),
            namespace: None,
            nodes,
            gpu_count: 2,
            image: "vllm:latest".into(),
            model: "Qwen/Qwen2.5-0.5B".into(),
            hf_token: Some("hf_test".into()),
            serve_args: vec![],
            shm_size: "16Gi".into(),
            preload_pvc: None,
        }
    }

    #[test]
    fn single_node_produces_deployment() {
        let manifests = build(test_config(1));
        match manifests {
            Manifests::SingleNode { deployment } => {
                assert_eq!(deployment.metadata.name.as_deref(), Some("test-vllm"));
                let spec = deployment.spec.as_ref().unwrap();
                assert_eq!(spec.replicas, Some(1));

                let pod_spec = spec.template.spec.as_ref().unwrap();
                let container = &pod_spec.containers[0];
                let limits = container
                    .resources
                    .as_ref()
                    .unwrap()
                    .limits
                    .as_ref()
                    .unwrap();
                assert_eq!(limits.get("nvidia.com/gpu").unwrap().0, "2");
            }
            _ => panic!("expected SingleNode"),
        }
    }

    #[test]
    fn multi_node_produces_statefulset_and_service() {
        let manifests = build(test_config(2));
        match manifests {
            Manifests::MultiNode {
                service,
                stateful_set,
            } => {
                // Headless service
                assert_eq!(service.metadata.name.as_deref(), Some("test-vllm-headless"));
                assert_eq!(
                    service.spec.as_ref().unwrap().cluster_ip.as_deref(),
                    Some("None")
                );

                // StatefulSet
                assert_eq!(stateful_set.metadata.name.as_deref(), Some("test-vllm"));
                let spec = stateful_set.spec.as_ref().unwrap();
                assert_eq!(spec.replicas, Some(2));
                assert_eq!(spec.service_name.as_deref(), Some("test-vllm-headless"));
                assert_eq!(spec.pod_management_policy.as_deref(), Some("Parallel"));

                // Check multi-node command extracts ordinal
                let container = &spec.template.spec.as_ref().unwrap().containers[0];
                let cmd = container.command.as_ref().unwrap().join(" ");
                assert!(cmd.contains("VLLM_NODE_RANK"));
                assert!(cmd.contains("VLLM_NNODES=2"));
                assert!(cmd.contains("test-vllm-0.test-vllm-headless"));
            }
            _ => panic!("expected MultiNode"),
        }
    }

    #[test]
    fn shm_volume_is_present() {
        let config = test_config(1);
        let manifests = build(config);
        if let Manifests::SingleNode { deployment } = manifests {
            let pod_spec = deployment
                .spec
                .as_ref()
                .unwrap()
                .template
                .spec
                .as_ref()
                .unwrap();
            let vols = pod_spec.volumes.as_ref().unwrap();
            let dshm = vols.iter().find(|v| v.name == "dshm").unwrap();
            let empty = dshm.empty_dir.as_ref().unwrap();
            assert_eq!(empty.medium.as_deref(), Some("Memory"));
            assert_eq!(empty.size_limit.as_ref().unwrap().0, "16Gi");

            let mounts = pod_spec.containers[0].volume_mounts.as_ref().unwrap();
            let shm = mounts.iter().find(|m| m.name == "dshm").unwrap();
            assert_eq!(shm.mount_path, "/dev/shm");
        }
    }

    #[test]
    fn env_includes_model_and_token() {
        let config = test_config(1);
        let manifests = build(config);
        if let Manifests::SingleNode { deployment } = manifests {
            let env = deployment.spec.unwrap().template.spec.unwrap().containers[0]
                .env
                .as_ref()
                .unwrap()
                .clone();
            assert!(
                env.iter()
                    .any(|e| e.name == "MODEL" && e.value.as_deref() == Some("Qwen/Qwen2.5-0.5B"))
            );
            assert!(
                env.iter()
                    .any(|e| e.name == "HF_TOKEN" && e.value.as_deref() == Some("hf_test"))
            );
        }
    }

    #[test]
    fn serve_args_are_appended() {
        let mut config = test_config(1);
        config.serve_args = vec![
            "--enforce-eager".into(),
            "--max-model-len".into(),
            "4096".into(),
        ];
        let manifests = build(config);
        if let Manifests::SingleNode { deployment } = manifests {
            let cmd = deployment.spec.unwrap().template.spec.unwrap().containers[0]
                .command
                .as_ref()
                .unwrap()
                .join(" ");
            assert!(cmd.contains("--enforce-eager --max-model-len 4096"));
        }
    }
}
