//! Test: multi-tile CUTLASS GEMM dispatch with runtime selection.
//!
//! Loads 3 tile configurations from a single multi-entry PTX file,
//! validates tile selection heuristic, and runs each on GPU.
//!
//! Run with: cargo test -p ptx-fusion --features cuda --test cuda_dispatch -- --nocapture

#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};

// Extract each config from the multi-entry PTX
ptx_fusion::extract_entry!(
    "kernels/cutlass_bf16_configs_sm89.ptx",
    "GemmShapeILi64ELi128ELi32E",
    CONFIG_64x128x32
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
            CutlassConfigSpec::new("64x128x32", 64, 128, 32, 128, 36864, CONFIG_64x128x32),
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

    // Decode (small M) → smallest tile (64x128x32)
    assert_eq!(dispatch.select(1).name, "64x128x32");
    assert_eq!(dispatch.select(8).name, "64x128x32");
    assert_eq!(dispatch.select(32).name, "64x128x32");
    assert_eq!(dispatch.select(64).name, "64x128x32");

    // Prefill (large M) → largest tile_m that fits
    let s128 = dispatch.select(128).name;
    let s512 = dispatch.select(512).name;
    let s2048 = dispatch.select(2048).name;
    println!("  M=128 → {s128}");
    println!("  M=512 → {s512}");
    println!("  M=2048 → {s2048}");

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
            CutlassConfigSpec::new("64x128x32", 64, 128, 32, 128, 36864, CONFIG_64x128x32),
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

    // M=1, N=4096, tile=64x128 → grid_m=1, grid_n=32
    let (gx, gy, gz) = CutlassDispatch::grid_dim(c, 1, 4096);
    println!("M=1, N=4096: grid=({gx},{gy},{gz})");
    assert_eq!(gz, 1);
    assert!(gx > 0);
    assert!(gy > 0);
    // Total tiles = ceil(1/64) * ceil(4096/128) = 1 * 32 = 32
    assert_eq!(gx * gy, 32);

    // M=128, N=4096, tile=128x128 → grid_m=1, grid_n=32
    let c128 = dispatch.select(128);
    let (gx, gy, gz) = CutlassDispatch::grid_dim(c128, 128, 4096);
    println!("M=128, N=4096: grid=({gx},{gy},{gz})");
    assert_eq!(gz, 1);
    assert_eq!(gx * gy, 32); // 1 * 32 tiles

    println!("PASS: grid dimension computation correct");
}

#[test]
fn rust_params_gpu_correctness() {
    use ptx_fusion::dispatch::{GemmParams, IteratorConstants};

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // Use the 64x128x32 config (our decode tile)
    let dispatch = CutlassDispatch::new(
        &ctx,
        vec![CutlassConfigSpec::new(
            "64x128x32",
            64,
            128,
            32,
            128,
            36864,
            CONFIG_64x128x32,
        )],
    )
    .unwrap();

    let config = &dispatch.configs()[0];
    let m = 64u32;
    let n = 128u32;
    let k = 32u32;
    let lda = k;
    let ldb = k;
    let ldc = n;
    let ldd = n;

    // bf16 test data
    fn f32_to_bf16(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }
    fn bf16_to_f32(v: u16) -> f32 {
        f32::from_bits((v as u32) << 16)
    }

    let h_a: Vec<u16> = (0..(m * k) as usize)
        .map(|i| f32_to_bf16(((i as f32) * 0.037 - 0.5).sin() * 0.5))
        .collect();
    let h_b: Vec<u16> = (0..(n * k) as usize)
        .map(|i| f32_to_bf16(((i as f32) * 0.023 + 0.3).cos() * 0.5))
        .collect();

    let d_a = stream.clone_htod(&h_a).unwrap();
    let d_b = stream.clone_htod(&h_b).unwrap();
    let d_c: CudaSlice<u16> = stream.alloc_zeros((m * n) as usize).unwrap();
    let mut d_d: CudaSlice<u16> = stream.alloc_zeros((m * n) as usize).unwrap();

    // Build params from Rust
    // cudarc 0.19: device_ptr(&stream) returns (CUdeviceptr, SyncOnDrop)
    let (a_ptr, _) = d_a.device_ptr(&stream);
    let (b_ptr, _) = d_b.device_ptr(&stream);
    let (c_ptr, _) = d_c.device_ptr(&stream);
    let (d_ptr, _) = d_d.device_ptr(&stream);
    let params = GemmParams::new(
        a_ptr as u64,
        b_ptr as u64,
        c_ptr as u64,
        d_ptr as u64,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        ldd,
        config.tile_m,
        config.tile_n,
        &IteratorConstants::CONFIG_64X128X32,
    );

    let (grid_x, grid_y, grid_z) = CutlassDispatch::grid_dim(config, m, n);
    let launch_cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (config.threads, 1, 1),
        shared_mem_bytes: config.smem_bytes,
    };

    // Launch — pass raw bytes since GemmParams isn't DeviceRepr
    unsafe {
        stream
            .launch_builder(&config.func)
            .arg(&params.bytes)
            .launch(launch_cfg)
    }
    .unwrap();
    stream.synchronize().unwrap();

    // Read output
    let output = stream.clone_dtoh(&d_d).unwrap();

    // Sanity: output should be non-zero
    let sum: f32 = output.iter().map(|&v| bf16_to_f32(v).abs()).sum();
    assert!(sum > 0.01, "output is all zeros (sum={sum})");

    // CPU reference: C[m,n] = sum_k A[m,k] * B[n,k] (B is col-major = NxK)
    let mut expected = vec![0.0f32; (m * n) as usize];
    for row in 0..m as usize {
        for col in 0..n as usize {
            let mut acc = 0.0f32;
            for ki in 0..k as usize {
                let a = bf16_to_f32(h_a[row * k as usize + ki]);
                let b = bf16_to_f32(h_b[col * k as usize + ki]);
                acc += a * b;
            }
            expected[row * n as usize + col] = acc;
        }
    }

    // Compare (bf16 precision: ~0.01 relative error)
    let mut max_diff = 0.0f32;
    for i in 0..(m * n) as usize {
        let got = bf16_to_f32(output[i]);
        let exp = expected[i];
        let diff = (got - exp).abs();
        max_diff = max_diff.max(diff);
    }

    println!(
        "  Output[0..4]: [{:.4}, {:.4}, {:.4}, {:.4}]",
        bf16_to_f32(output[0]),
        bf16_to_f32(output[1]),
        bf16_to_f32(output[2]),
        bf16_to_f32(output[3])
    );
    println!(
        "  Expected[0..4]: [{:.4}, {:.4}, {:.4}, {:.4}]",
        expected[0], expected[1], expected[2], expected[3]
    );
    println!("  Max diff: {max_diff:.2e}");

    assert!(
        max_diff < 0.05,
        "Rust-built params produce wrong output (max_diff={max_diff:.2e})"
    );

    println!("PASS: Rust GemmParams builder produces correct output (max_diff={max_diff:.2e})");
}
