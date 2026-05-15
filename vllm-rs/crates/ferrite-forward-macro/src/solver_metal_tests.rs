// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Solver tests for Metal backend integration.
//!
//! Verifies that the solver correctly selects Metal implementations
//! when the target profile is Metal, and rejects them for CUDA targets.

#[cfg(all(test, feature = "metal"))]
mod tests {
    use crate::classified::OpKind;
    use crate::config::ModelParams;
    use crate::fuf::unroll;
    use crate::impl_lib::starter_library;
    use crate::parse::parse_block;
    use crate::shape::infer;
    use crate::solver::solve;
    use crate::target::{from_metal_profile, from_profile_def};
    use std::collections::BTreeMap;

    fn build_simple_rmsnorm_fuf() -> (crate::fuf::Fuf, crate::shape::Inferred, ModelParams) {
        let src = r#"
            hidden_states = embed(input_ids, embed_tokens);
            normed = rmsnorm(hidden_states, input_layernorm);
        "#;

        let file: syn::File = syn::parse_str(&format!("fn _c() {{ {src} }}")).expect("parse");
        let block = match &file.items[0] {
            syn::Item::Fn(f) => &*f.block,
            _ => unreachable!(),
        };

        let ast = parse_block(block).unwrap();
        let program = crate::classify::classify(&ast).unwrap();

        let mut params = ModelParams {
            name: "test_model".to_string(),
            source_stem: "test_model".to_string(),
            source_path: std::path::PathBuf::from("test"),
            bounds: BTreeMap::new(),
            scalars: BTreeMap::new(),
            quantization: None,
            tie_word_embeddings: false,
            architectures: vec![],
            extra_tracked_paths: vec![],
            rope_scaling: None,
            rope_scaling_hash: None,
            mrope_section: None,
        };
        params.bounds.insert("num_hidden_layers".into(), 1);
        params.bounds.insert("hidden_size".into(), 4096);
        params.bounds.insert("vocab_size".into(), 32000);

        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &BTreeMap::new(),
        )
        .unwrap();

        let cfg = crate::cfg::build_cfg(&program, &params).unwrap();
        let fuf = unroll(&cfg, &inferred).unwrap();

        (fuf, inferred, params)
    }

    #[test]
    fn solver_picks_metal_rmsnorm_for_metal_target() {
        let (fuf, inferred, params) = build_simple_rmsnorm_fuf();
        let lib = starter_library();

        // Use Metal M1 target
        let metal_target = from_metal_profile(&ferrite_metal_targets::M1_8CORE);

        let workloads = solve(
            &fuf,
            &lib,
            &metal_target,
            &inferred,
            &params.bounds,
            &[1, 512],
            &[],
        )
        .expect("solve should succeed with Metal target");

        // Verify we got assignments for both workload points
        assert_eq!(workloads.per_workload.len(), 2);

        // Check that RMSNorm tiles are covered
        for (wp, sfuf) in workloads.per_workload.iter() {
            assert!(
                sfuf.is_cover_complete(fuf.len()),
                "cover incomplete at num_tokens={}",
                wp.num_tokens
            );

            // Find RMSNorm subgraphs and verify they use Metal implementations
            for sg in sfuf.subgraphs() {
                let tiles = sfuf.tiles_in_subgraph(sg);
                let ops: Vec<OpKind> = tiles.iter().map(|t| fuf.get(*t).op).collect();

                if ops.contains(&OpKind::RmsNorm) {
                    let impl_id = sfuf.impl_of(sg).unwrap();
                    let impl_name = lib.get(impl_id).name();

                    // Should be one of the Metal implementations
                    assert!(
                        impl_name.starts_with("metal_rmsnorm"),
                        "Expected Metal RMSNorm impl, got {} at num_tokens={}",
                        impl_name,
                        wp.num_tokens
                    );
                }
            }
        }
    }

    #[test]
    fn solver_rejects_metal_rmsnorm_for_cuda_target() {
        let (fuf, inferred, params) = build_simple_rmsnorm_fuf();
        let lib = starter_library();

        // Use CUDA L4 target
        let cuda_target = from_profile_def(&ferrite_cuda_targets::L4_SM89);

        let workloads = solve(
            &fuf,
            &lib,
            &cuda_target,
            &inferred,
            &params.bounds,
            &[1, 512],
            &[],
        )
        .expect("solve should succeed with CUDA target");

        // Verify we got assignments
        assert_eq!(workloads.per_workload.len(), 2);

        // Check that RMSNorm tiles use CUDA implementations, NOT Metal
        for (wp, sfuf) in workloads.per_workload.iter() {
            for sg in sfuf.subgraphs() {
                let tiles = sfuf.tiles_in_subgraph(sg);
                let ops: Vec<OpKind> = tiles.iter().map(|t| fuf.get(*t).op).collect();

                if ops.contains(&OpKind::RmsNorm) {
                    let impl_id = sfuf.impl_of(sg).unwrap();
                    let impl_name = lib.get(impl_id).name();

                    // Should NOT be Metal implementation
                    assert!(
                        !impl_name.starts_with("metal_"),
                        "CUDA target should not pick Metal impl, got {} at num_tokens={}",
                        impl_name,
                        wp.num_tokens
                    );

                    // Should be the CUDA reference implementation
                    assert_eq!(
                        impl_name, "rmsnorm_ref",
                        "Expected CUDA rmsnorm_ref, got {} at num_tokens={}",
                        impl_name, wp.num_tokens
                    );
                }
            }
        }
    }

    #[test]
    fn metal_rmsnorm_cost_varies_by_device() {
        let (fuf, inferred, params) = build_simple_rmsnorm_fuf();
        let lib = starter_library();

        // Solve for M1 (68.25 GB/s)
        let m1_target = from_metal_profile(&ferrite_metal_targets::M1_8CORE);
        let m1_workloads = solve(
            &fuf,
            &lib,
            &m1_target,
            &inferred,
            &params.bounds,
            &[512],
            &[],
        )
        .unwrap();
        let m1_cost = m1_workloads.get_nt(512).unwrap().predicted_us;

        // Solve for M2 (100 GB/s - 1.46× faster)
        let m2_target = from_metal_profile(&ferrite_metal_targets::M2_10CORE);
        let m2_workloads = solve(
            &fuf,
            &lib,
            &m2_target,
            &inferred,
            &params.bounds,
            &[512],
            &[],
        )
        .unwrap();
        let m2_cost = m2_workloads.get_nt(512).unwrap().predicted_us;

        // M1 should be slower than M2 (lower bandwidth)
        assert!(
            m1_cost > m2_cost,
            "M1 cost ({} µs) should be higher than M2 cost ({} µs) due to lower bandwidth",
            m1_cost,
            m2_cost
        );

        // Cost ratio should roughly match bandwidth ratio (within 20% tolerance)
        let cost_ratio = m1_cost / m2_cost;
        let bandwidth_ratio = 100.0 / 68.25; // M2 / M1
        let ratio_diff = (cost_ratio - bandwidth_ratio).abs() / bandwidth_ratio;
        assert!(
            ratio_diff < 0.2,
            "Cost ratio ({:.2}) should roughly match bandwidth ratio ({:.2}), diff: {:.1}%",
            cost_ratio,
            bandwidth_ratio,
            ratio_diff * 100.0
        );
    }

    #[test]
    fn metal_implementations_registered_in_library() {
        let lib = starter_library();

        // Count Metal implementations
        let metal_impls: Vec<_> = lib
            .iter_enumerated()
            .filter(|(_, imp)| imp.name().starts_with("metal_"))
            .collect();

        // Should have at least 2 Metal RMSNorm implementations (fp16 + bf16)
        assert!(
            metal_impls.len() >= 2,
            "Expected at least 2 Metal implementations, found {}",
            metal_impls.len()
        );

        // Verify specific implementations are present
        let names: Vec<&str> = metal_impls.iter().map(|(_, imp)| imp.name()).collect();
        assert!(
            names.contains(&"metal_rmsnorm_f16"),
            "metal_rmsnorm_f16 not found in library"
        );
        assert!(
            names.contains(&"metal_rmsnorm_bf16"),
            "metal_rmsnorm_bf16 not found in library"
        );
    }
}
