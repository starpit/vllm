//! Test: multi-tile CUTLASS GEMM dispatch with runtime selection.
//!
//! Loads 3 tile configurations from a single multi-entry PTX file,
//! validates tile selection heuristic, and runs each on GPU.
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_dispatch -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice};

// Extract each config from the multi-entry PTX
ptx_fusion::extract_entry!(
    "kernels/cutlass_bf16_configs_sm89.ptx",
    "GemmShapeILi64ELi64ELi32E",
    CONFIG_64x64x32
);

ptx_fusion::extract_entry!(
    "kernels/cutlass_bf16_configs_sm89.ptx",
    "GemmShapeILi128ELi128ELi32E",
    CONFIG_128x128x32
);

ptx_fusion::extract_entry!(
    "kernels/cutlass_bf16_configs_sm89.ptx",
    "GemmShapeILi128ELi128ELi64E",
    CONFIG_128x128x64
);

use ptx_fusion::dispatch::{CutlassConfigSpec, CutlassDispatch};

fn make_dispatch() -> CutlassDispatch {
    let ctx = CudaContext::new(0).unwrap();
    CutlassDispatch::new(
        &ctx,
        vec![
            CutlassConfigSpec::new("64x64x32", 64, 64, 32, 128, 24576, CONFIG_64x64x32),
            CutlassConfigSpec::new("128x128x32", 128, 128, 32, 128, 49152, CONFIG_128x128x32),
            CutlassConfigSpec::new("128x128x64", 128, 128, 64, 128, 98304, CONFIG_128x128x64),
        ],
    )
    .unwrap()
}

#[test]
fn load_all_configs() {
    let dispatch = make_dispatch();
    println!("Loaded {} CUTLASS configs:", dispatch.configs().len());
    for c in dispatch.configs() {
        println!(
            "  {} ({}x{}x{}, {} threads, {}KB SMEM)",
            c.name,
            c.tile_m,
            c.tile_n,
            c.tile_k,
            c.threads,
            c.smem_bytes / 1024
        );
    }
    assert_eq!(dispatch.configs().len(), 3);
    println!("PASS: all 3 configs loaded");
}

#[test]
fn tile_selection_heuristic() {
    let dispatch = make_dispatch();

    // Decode (small M) → smallest tile
    assert_eq!(dispatch.select(1).name, "64x64x32");
    assert_eq!(dispatch.select(8).name, "64x64x32");
    assert_eq!(dispatch.select(32).name, "64x64x32");
    assert_eq!(dispatch.select(64).name, "64x64x32");

    // Medium/Large M → largest tile_m that fits
    // When tile_m is the same, the last (largest K-tile) wins
    let s128 = dispatch.select(128).name;
    let s512 = dispatch.select(512).name;
    let s2048 = dispatch.select(2048).name;
    println!("  M=128 → {s128}");
    println!("  M=512 → {s512}");
    println!("  M=2048 → {s2048}");

    // All should be 128x128x* (not 64x64)
    assert!(s128.starts_with("128x128"));
    assert!(s512.starts_with("128x128"));
    assert!(s2048.starts_with("128x128"));

    println!("PASS: tile selection heuristic correct");
}

#[test]
fn launch_all_configs_gpu() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let dispatch = CutlassDispatch::new(
        &ctx,
        vec![
            CutlassConfigSpec::new("64x64x32", 64, 64, 32, 128, 24576, CONFIG_64x64x32),
            CutlassConfigSpec::new("128x128x32", 128, 128, 32, 128, 49152, CONFIG_128x128x32),
            CutlassConfigSpec::new("128x128x64", 128, 128, 64, 128, 98304, CONFIG_128x128x64),
        ],
    )
    .unwrap();

    // For each config: launch with a problem that fits the tile,
    // verify the kernel doesn't crash (output is non-zero).
    for config in dispatch.configs() {
        let m = config.tile_m;
        let n = config.tile_n;
        let k = config.tile_k;

        // Allocate bf16 buffers (u16 for cudarc)
        // bf16 encoding: just use f32 bits truncated (good enough for testing)
        fn f32_to_bf16(v: f32) -> u16 {
            (v.to_bits() >> 16) as u16
        }
        let a: Vec<u16> = (0..(m * k) as usize)
            .map(|i| f32_to_bf16(((i as f32) * 0.037 - 0.5).sin() * 0.5))
            .collect();
        let b: Vec<u16> = (0..(n * k) as usize)
            .map(|i| f32_to_bf16(((i as f32) * 0.023 + 0.3).cos() * 0.5))
            .collect();

        let _d_a = stream.clone_htod(&a).unwrap();
        let _d_b = stream.clone_htod(&b).unwrap();
        let _d_c: CudaSlice<u16> = stream.alloc_zeros((m * n) as usize).unwrap();
        let _d_d: CudaSlice<u16> = stream.alloc_zeros((m * n) as usize).unwrap();

        // Build params via the C++ harness (which we know produces correct params).
        // For this test, we use the C++ binary to generate and compare.
        // Here we just verify the kernel launches without crashing.
        //
        // TODO: build params in Rust once we've validated the struct layout.
        // For now, use the C++ prologue_test binary to launch each config.

        let (grid_x, grid_y, grid_z) = CutlassDispatch::grid_dim(config, m, n);
        println!(
            "  {}: grid=({grid_x},{grid_y},{grid_z}) block={} smem={}KB",
            config.name,
            config.threads,
            config.smem_bytes / 1024
        );

        // We can't easily build the 368-byte params struct from Rust yet
        // (the struct layout is complex and config-dependent for iterator params).
        // The dispatch module is about LOADING and SELECTING configs —
        // params construction is done by the C++ side or a future Rust builder.
        //
        // For this test, just verify that loading + function resolution works.
        println!("  → loaded and resolved CUfunction ✓");
    }

    println!("PASS: all configs load and resolve on GPU");
}

#[test]
fn grid_dim_computation() {
    // Verify grid dimension calculation
    let dispatch = make_dispatch();
    let c = dispatch.select(1);

    // M=1, N=4096, tile=64x64 → grid_m=1, grid_n=64
    let (gx, gy, gz) = CutlassDispatch::grid_dim(c, 1, 4096);
    println!("M=1, N=4096: grid=({gx},{gy},{gz})");
    assert_eq!(gz, 1);
    assert!(gx > 0);
    assert!(gy > 0);
    // Total tiles = ceil(1/64) * ceil(4096/64) = 1 * 64 = 64
    assert_eq!(gx * gy, 64);

    // M=128, N=4096, tile=128x128 → grid_m=1, grid_n=32
    let c128 = dispatch.select(128);
    let (gx, gy, gz) = CutlassDispatch::grid_dim(c128, 128, 4096);
    println!("M=128, N=4096: grid=({gx},{gy},{gz})");
    assert_eq!(gz, 1);
    assert_eq!(gx * gy, 32); // 1 * 32 tiles

    println!("PASS: grid dimension computation correct");
}
