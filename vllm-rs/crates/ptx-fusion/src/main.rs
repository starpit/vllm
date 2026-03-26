use ptx_fusion::{analyze_kernel, fuse_kernels, regfuse_kernels, rewrite_kernel};

// ── Step 1: Analyze both kernels ─────────────────────────────────────

// Easy kernel: RMSNorm
// Expected: reads input+weight from GMEM, writes output to GMEM, no SMEM, no MMA
analyze_kernel!("kernels/rms_norm.ptx");

// Hard kernel: Matvec
// Expected: reads matrix+vec_in from GMEM, writes vec_out to GMEM, uses SMEM, has barrier
analyze_kernel!("kernels/matvec.ptx");

// ── Step 2: Rewrite RMSNorm with register renames ────────────────────

// Rename some registers to prove we can transform PTX.
// If the rewritten kernel's protocol is structurally identical (same I/O pattern,
// same SMEM, same barriers) then the transform preserved semantics.
rewrite_kernel!("kernels/rms_norm.ptx", {
    "%f3" => "%f30",
    "%r3" => "%r30",
    "%rd3" => "%rd30"
});

// ── Step 3: Rewrite Matvec with register renames ─────────────────────

rewrite_kernel!("kernels/matvec.ptx", {
    "%f1" => "%f50",
    "%r5" => "%r50",
    "%rd3" => "%rd50"
});

// ── Step 4: Fuse rms_norm → matvec via SMEM handoff ─────────────────

fuse_kernels!(
    "kernels/rms_norm.ptx",
    "kernels/matvec.ptx",
    "fused_rms_norm_matvec",
    "output",
    "vec_in"
);

// ── Step 5: Register-level fusion (rms_norm → scale) ────────────────

regfuse_kernels!(
    "kernels/rms_norm.ptx",
    "kernels/scale.ptx",
    "regfused_rms_norm_scale",
    "output",
    "input"
);

fn main() {
    println!("=== PTX Fusion Protocol Extraction POC ===\n");

    println!("── EASY KERNEL: RMSNorm ──\n");
    RMS_NORM.display();

    println!("\n── HARD KERNEL: Matvec ──\n");
    MATVEC.display();

    println!("\n── REWRITTEN RMSNorm (registers renamed) ──\n");
    RMS_NORM_REWRITTEN_PROTOCOL.display();

    println!("\n── REWRITTEN Matvec (registers renamed) ──\n");
    MATVEC_REWRITTEN_PROTOCOL.display();

    // ── Validate that rewriting preserved the protocol ──────────────

    println!("\n=== VALIDATION ===\n");

    validate_protocols_match("rms_norm", &RMS_NORM, &RMS_NORM_REWRITTEN_PROTOCOL);
    validate_protocols_match("matvec", &MATVEC, &MATVEC_REWRITTEN_PROTOCOL);

    println!("\n── Rewritten RMSNorm PTX (first 20 lines) ──\n");
    for (i, line) in RMS_NORM_REWRITTEN.lines().take(20).enumerate() {
        println!("  {:3}: {}", i + 1, line);
    }

    println!("\n── Rewritten Matvec PTX (first 30 lines) ──\n");
    for (i, line) in MATVEC_REWRITTEN.lines().take(30).enumerate() {
        println!("  {:3}: {}", i + 1, line);
    }

    println!("\n\n=== FUSED KERNEL PTX ===\n");
    for (i, line) in FUSED_RMS_NORM_MATVEC.lines().enumerate() {
        println!("  {:3}: {}", i + 1, line);
    }

    // Write fused PTX to file for ptxas validation
    std::fs::write("/tmp/fused_rms_norm_matvec.ptx", FUSED_RMS_NORM_MATVEC).unwrap();
    println!("\n  Written to /tmp/fused_rms_norm_matvec.ptx");

    println!("\n\n=== REGISTER-FUSED KERNEL PTX ===\n");
    for (i, line) in REGFUSED_RMS_NORM_SCALE.lines().enumerate() {
        println!("  {:3}: {}", i + 1, line);
    }
    std::fs::write("/tmp/regfused_rms_norm_scale.ptx", REGFUSED_RMS_NORM_SCALE).unwrap();
    println!("\n  Written to /tmp/regfused_rms_norm_scale.ptx");
}

fn validate_protocols_match(
    name: &str,
    original: &ptx_fusion::KernelProtocol,
    rewritten: &ptx_fusion::KernelProtocol,
) {
    let mut pass = true;

    // Same number of global loads
    if original.global_loads.len() != rewritten.global_loads.len() {
        println!(
            "  FAIL [{name}]: global loads count mismatch ({} vs {})",
            original.global_loads.len(),
            rewritten.global_loads.len()
        );
        pass = false;
    }

    // Same number of global stores
    if original.global_stores.len() != rewritten.global_stores.len() {
        println!(
            "  FAIL [{name}]: global stores count mismatch ({} vs {})",
            original.global_stores.len(),
            rewritten.global_stores.len()
        );
        pass = false;
    }

    // Same params traced for loads
    for (i, (orig, rew)) in original
        .global_loads
        .iter()
        .zip(rewritten.global_loads.iter())
        .enumerate()
    {
        if orig.param_name != rew.param_name {
            println!(
                "  FAIL [{name}]: load[{i}] param mismatch: \"{}\" vs \"{}\"",
                orig.param_name, rew.param_name
            );
            pass = false;
        }
        if orig.data_type != rew.data_type {
            println!(
                "  FAIL [{name}]: load[{i}] type mismatch: {} vs {}",
                orig.data_type, rew.data_type
            );
            pass = false;
        }
    }

    // Same params traced for stores
    for (i, (orig, rew)) in original
        .global_stores
        .iter()
        .zip(rewritten.global_stores.iter())
        .enumerate()
    {
        if orig.param_name != rew.param_name {
            println!(
                "  FAIL [{name}]: store[{i}] param mismatch: \"{}\" vs \"{}\"",
                orig.param_name, rew.param_name
            );
            pass = false;
        }
    }

    // Same SMEM layout
    if original.total_smem_bytes != rewritten.total_smem_bytes {
        println!(
            "  FAIL [{name}]: SMEM bytes mismatch ({} vs {})",
            original.total_smem_bytes, rewritten.total_smem_bytes
        );
        pass = false;
    }

    // Same barriers
    if original.barriers != rewritten.barriers {
        println!(
            "  FAIL [{name}]: barriers mismatch ({:?} vs {:?})",
            original.barriers, rewritten.barriers
        );
        pass = false;
    }

    // Same MMA flag
    if original.has_mma != rewritten.has_mma {
        println!(
            "  FAIL [{name}]: MMA flag mismatch ({} vs {})",
            original.has_mma, rewritten.has_mma
        );
        pass = false;
    }

    if pass {
        println!("  PASS [{name}]: rewritten protocol matches original");
    }
}
