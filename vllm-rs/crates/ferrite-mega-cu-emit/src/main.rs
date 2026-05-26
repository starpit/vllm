// SPDX-License-Identifier: Apache-2.0
//! Phase C step 2 driver — walks the
//! `inventory::collect!(MegaCanonicalEmit)` registry that the
//! `#[forward]` proc-macro populates per canonical (gated on
//! `feature = "cuda"` + `FERRITE_MEGA=1` at proc-macro time) and
//! writes each `emit_fn()` output to
//! `<cudaforge cache>/megakernels/ferrite_<canonical>.cu`, where
//! `ferrite-cuda-builder/build.rs` picks them up for nvcc compile
//! into `libmegakernels.a`.
//!
//! Run before any `FERRITE_MEGA=1 cargo build --features cuda`:
//!
//! ```text
//! FERRITE_MODELS=llama-3.2-1b FERRITE_MEGA=1 \
//!   cargo run -p ferrite-mega-cu-emit --features cuda
//! ```
//!
//! The binary depends on `ferrite-cuda-builder` only so the
//! cudaforge `.a` libs (libvllm_kernels, libflashinfer_attn, ...)
//! exist when ferrite-kernels' externs need to link. We don't
//! actually use any CUDA APIs here — `emit_fn()` is pure Rust
//! string formatting.

// Force the linker to keep the per-arch model crates in the bin's
// link graph so their `inventory::submit!(MegaCanonicalEmit { ... })`
// blocks survive dead-code stripping. Mirrors the `extern crate
// ... as _keep_<arch>` lines in `ferrite-models/src/lib.rs` (those
// only keep symbols inside ferrite-models's own compilation; we
// repeat them here so the bin's link sees them too).
#[cfg(feature = "cuda")]
extern crate ferrite_model_llama as _keep_llama;

#[cfg(feature = "cuda")]
use ferrite_megakernel::cuda_emit::MegaCanonicalEmit;

#[cfg(feature = "cuda")]
fn main() {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("cudaforge/megakernels");
    std::fs::create_dir_all(&cache_dir)
        .unwrap_or_else(|e| panic!("create_dir_all({}): {e}", cache_dir.display()));

    // Track which .cu paths the current build's canonical inventory
    // owns. Anything else in the cache is from a prior build whose
    // canonical set has since drifted (different `FERRITE_MODELS`,
    // different solver pick, dropped variant, etc.) and would be
    // wired up by the proc-macro's `cu_path.exists()` check —
    // cascading stale broken kernels into the new build. Prune at
    // the end of this pass.
    let mut current_paths: std::collections::HashSet<std::path::PathBuf> =
        std::collections::HashSet::new();

    let mut count = 0u32;
    for entry in ferrite_megakernel::inventory::iter::<MegaCanonicalEmit>() {
        let cu = (entry.emit_fn)();
        let cu_path = cache_dir.join(format!("ferrite_{}.cu", entry.canonical));
        current_paths.insert(cu_path.clone());
        // A canonical with ANY render-time skipped op cannot launch
        // safely (the kernel reads from pages that the skipped op
        // would have populated, → CUDA launch failure). Drop the .cu
        // so the proc-macro's `cu_path.exists()` check fails on the
        // next build pass → wrapper not emitted → host fallback for
        // this bucket. Per `feedback_no_option_c`, we don't emit a
        // half-broken kernel; either the canonical fully renders or
        // it isn't a mega bucket.
        if !cu.skipped_variants.is_empty() {
            if cu_path.exists() {
                std::fs::remove_file(&cu_path)
                    .unwrap_or_else(|e| panic!("remove {}: {e}", cu_path.display()));
                println!(
                    "DROPPED {} (skipped variants: {:?})",
                    cu_path.display(),
                    cu.skipped_variants
                );
            } else {
                println!(
                    "SKIP {} (skipped variants: {:?})",
                    cu_path.display(),
                    cu.skipped_variants
                );
            }
            count += 1;
            continue;
        }
        // Idempotent write: skip if content unchanged so cudaforge's
        // content-hash cache doesn't trigger a recompile on every run.
        let needs_write = match std::fs::read_to_string(&cu_path) {
            Ok(existing) => existing != cu.source,
            Err(_) => true,
        };
        if needs_write {
            std::fs::write(&cu_path, &cu.source)
                .unwrap_or_else(|e| panic!("write {}: {e}", cu_path.display()));
            println!("WROTE {}", cu_path.display());
        } else {
            println!("UP-TO-DATE {}", cu_path.display());
        }
        count += 1;
    }
    // Prune stale .cu files: anything in the cache directory with a
    // `ferrite_*.cu` name but not in `current_paths`. Without this,
    // a previous build's canonical (e.g. `m_8_sk_512` from a prior
    // bucket pick) leaves its .cu in cache; the proc-macro sees it
    // exists, emits a `forward_mega_*` wrapper for it, and runtime
    // dispatch hits stale ABI / stale render code.
    let mut pruned = 0u32;
    if let Ok(rd) = std::fs::read_dir(&cache_dir) {
        for ent in rd.flatten() {
            let path = ent.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !name.starts_with("ferrite_") || !name.ends_with(".cu") {
                continue;
            }
            if current_paths.contains(&path) {
                continue;
            }
            std::fs::remove_file(&path)
                .unwrap_or_else(|e| panic!("prune {}: {e}", path.display()));
            println!("PRUNED-STALE {}", path.display());
            pruned += 1;
        }
    }

    println!(
        "\nferrite-mega-cu-emit: {count} canonical .cu file(s) processed, {pruned} stale pruned in {}",
        cache_dir.display()
    );
    if count == 0 {
        eprintln!(
            "WARNING: inventory empty — did you set FERRITE_MEGA=1 at proc-macro time? \
             The proc-macro gates `inventory::submit!` on `#[cfg(feature = \"cuda\")]` \
             AND only emits when `FERRITE_MEGA` is in the environment when the user \
             crate compiles."
        );
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!(
        "ferrite-mega-cu-emit requires --features cuda. The proc-macro emits \
         `inventory::submit!(MegaCanonicalEmit { ... })` only when the user \
         crate has `cuda` enabled."
    );
    std::process::exit(2);
}
