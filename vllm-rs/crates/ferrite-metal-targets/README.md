# ferrite-metal-targets

Device profiles and cost models for Apple Silicon GPUs (M1/M2/M3/M4).

## Overview

This crate provides:
- Hardware specifications for each Apple Silicon generation
- Cost tables mapping (kernel, M, N, K) → microseconds
- Device detection and profile selection at runtime

## Device Profiles

Each `MetalTargetProfile` contains:
- **Architecture generation**: M1, M2, M3, or M4
- **GPU cores**: Number of GPU cores (8-10 for base models)
- **Peak TFLOPS**: FP16 compute throughput
- **Memory bandwidth**: Unified memory bandwidth in GB/s
- **Threadgroup limits**: Max threads and shared memory per threadgroup

## Cost Model

Cost tables are populated by microbenchmarks (see `ferrite-cost-sweep`):
```rust
let profile = M1_8CORE;
let cost_us = profile.cost_us_for("rmsnorm_f16", 32, 4096, 0);
```

If no measured cost exists, implementations fall back to analytical models based on memory bandwidth and compute characteristics.

## Usage

```rust
use ferrite_metal_targets::{M1_8CORE, M2_10CORE, MetalTargetProfile};

// Use predefined profiles
let profile = M1_8CORE;
println!("Peak TFLOPS: {}", profile.peak_tflops_fp16);

// Or detect at runtime (see ferrite-metal-kernels)
```

## Adding New Devices

To add support for a new Apple Silicon generation:
1. Add variant to `AppleSiliconGen` enum
2. Create const profile with hardware specs
3. Update device detection in `ferrite-metal-kernels`
4. Run cost sweeps to populate cost tables
