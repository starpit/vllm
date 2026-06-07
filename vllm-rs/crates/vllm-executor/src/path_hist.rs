// SPDX-License-Identifier: Apache-2.0
//! Per-`execute_model_inner`-step path histogram + invalidation log.
//!
//! Gated by env var `FERRITE_PATH_HISTOGRAM=1`. When off, every counter
//! call is a relaxed atomic increment (a few cycles); when on, the
//! per-step `Instant::now` pair adds ~30 ns. Output is printed at
//! worker shutdown via [`drain_and_print`].
//!
//! Quantifies which decode path each step takes during a batched bench
//! so we can localize where the rust-vs-python perf gap actually lives.
//! See `vllm-rs/PORT_PLAN.md` (decode-host-stall worktree) for context.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// One terminal path the executor's `execute_model_inner` can take.
///
/// `Init` is a sentinel for "no path tag set this step yet"; if a step
/// returns with `Init` it means the histogram missed a return point —
/// caught by the `untagged_step` counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PathTag {
    Init = 0,
    SuperFastHit,
    ColdDecodeGraphFastMeta,
    ColdDecodeGraphFullH2d,
    ColdDecodeGraphNoStaging,
    DecodeGraphNonGreedyOrFull,
    DecodeEagerNoGraph,
    MixedPrefillDecode,
    PrefillOnlyGraph,
    PrefillOnlyPiecewise,
    PrefillOnlyEager,
    PiecewiseDecode,
    PoolingPath,
    EmbeddingPath,
    Other,
}

const N_TAGS: usize = 16;

/// Reason a `graph_metadata_valid = false` flag flip was observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum InvalidationSite {
    BatchChanged5223 = 0,
    MixedPrefill6238,
    PiecewiseDecode6827,
    PrefillOrEagerElse6833,
    Close4992,
}

const N_INVAL: usize = 5;

/// Gate flag: read once at startup. If true, instrumentation is live.
static ENABLED: AtomicBool = AtomicBool::new(false);
static ENABLED_INIT: AtomicBool = AtomicBool::new(false);

/// Returns true if `FERRITE_PATH_HISTOGRAM=1`. Reads env exactly once.
#[inline]
pub fn enabled() -> bool {
    if ENABLED_INIT.load(Ordering::Relaxed) {
        return ENABLED.load(Ordering::Relaxed);
    }
    let v = std::env::var("FERRITE_PATH_HISTOGRAM")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    ENABLED.store(v, Ordering::Relaxed);
    ENABLED_INIT.store(true, Ordering::Relaxed);
    v
}

// --- counters ---------------------------------------------------------

/// Per-terminal-path step counter.
static PATH_COUNT: [AtomicU64; N_TAGS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Per-terminal-path total wall nanoseconds spent in
/// `execute_model_inner` (Instant::now at entry to Instant::now at
/// return). One Instant pair per step total, no per-arm cost.
static PATH_NANOS: [AtomicU64; N_TAGS] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

/// Super-fast-path gate-failure breakdown. Counts steps that *would*
/// have entered super-fast except for the named reason. Each step
/// increments at most one — the FIRST condition checked, so this is
/// "primary reason." Sum of these <= step_total; difference =
/// super_fast_hit count.
static GATE_FAIL_METADATA_INVALID: AtomicU64 = AtomicU64::new(0);
static GATE_FAIL_CHUNKED_PREFILL: AtomicU64 = AtomicU64::new(0);
static GATE_FAIL_NO_SK_BUCKET: AtomicU64 = AtomicU64::new(0);
static GATE_FAIL_NO_GRAPH_BS: AtomicU64 = AtomicU64::new(0);
static GATE_FAIL_NO_STAGING_OR_DEVICE: AtomicU64 = AtomicU64::new(0);
static GATE_FAIL_NOT_GREEDY: AtomicU64 = AtomicU64::new(0);
static GATE_FAIL_NEEDS_FULL: AtomicU64 = AtomicU64::new(0);

/// Sanity counters.
static STEP_TOTAL: AtomicU64 = AtomicU64::new(0);
static UNTAGGED_STEP: AtomicU64 = AtomicU64::new(0);
static DEFERRED_OUTPUTS: AtomicU64 = AtomicU64::new(0);
static SYNC_OUTPUTS: AtomicU64 = AtomicU64::new(0);
static BLOCKS_CHANGED_STEPS: AtomicU64 = AtomicU64::new(0);
static PENDING_RESOLVED_LATE: AtomicU64 = AtomicU64::new(0);

/// Phase-band timers: total ns spent in each named CPU phase across all
/// steps. Use `phase_record_ns(band, ns)` once per step.
static PHASE_PREPARE_INPUTS_NS: AtomicU64 = AtomicU64::new(0);

/// Per-site `graph_metadata_valid = false` invalidation count.
static INVALIDATION_COUNT: [AtomicU64; N_INVAL] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

// --- API --------------------------------------------------------------

/// Per-step state held on the worker stack. RAII: increments the
/// per-tag counter on Drop, so any early return / `?` propagation
/// automatically records. Set the tag with `set` before exiting; an
/// unset tag is recorded under `untagged_step`.
pub struct StepCtx {
    pub start: Option<Instant>,
    pub tag: PathTag,
}

impl StepCtx {
    /// Begin a step.
    #[inline]
    pub fn begin() -> Self {
        let start = if enabled() {
            Some(Instant::now())
        } else {
            None
        };
        Self {
            start,
            tag: PathTag::Init,
        }
    }

    /// Tag this step's terminal path. Last call wins; expected to be
    /// called exactly once per step before returning.
    #[inline]
    pub fn set(&mut self, tag: PathTag) {
        self.tag = tag;
    }
}

impl Drop for StepCtx {
    #[inline]
    fn drop(&mut self) {
        if !enabled() {
            return;
        }
        STEP_TOTAL.fetch_add(1, Ordering::Relaxed);
        let idx = self.tag as usize;
        if idx == 0 {
            UNTAGGED_STEP.fetch_add(1, Ordering::Relaxed);
            return;
        }
        PATH_COUNT[idx].fetch_add(1, Ordering::Relaxed);
        if let Some(t0) = self.start {
            let ns = t0.elapsed().as_nanos() as u64;
            PATH_NANOS[idx].fetch_add(ns, Ordering::Relaxed);
        }
    }
}

/// Record one super-fast-path gate failure with the FIRST failing
/// reason. Call ONCE per step where a gate fails (early-out semantics).
#[inline]
pub fn record_gate_fail(reason: GateFailReason) {
    if !enabled() {
        return;
    }
    match reason {
        GateFailReason::MetadataInvalid => &GATE_FAIL_METADATA_INVALID,
        GateFailReason::ChunkedPrefill => &GATE_FAIL_CHUNKED_PREFILL,
        GateFailReason::NoSkBucket => &GATE_FAIL_NO_SK_BUCKET,
        GateFailReason::NoGraphBs => &GATE_FAIL_NO_GRAPH_BS,
        GateFailReason::NoStagingOrDevice => &GATE_FAIL_NO_STAGING_OR_DEVICE,
        GateFailReason::NotGreedy => &GATE_FAIL_NOT_GREEDY,
        GateFailReason::NeedsFull => &GATE_FAIL_NEEDS_FULL,
    }
    .fetch_add(1, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug)]
pub enum GateFailReason {
    MetadataInvalid,
    ChunkedPrefill,
    NoSkBucket,
    NoGraphBs,
    NoStagingOrDevice,
    NotGreedy,
    NeedsFull,
}

/// Increment when an output is built via `ModelRunnerOutput::deferred`.
#[inline]
pub fn record_deferred() {
    if enabled() {
        DEFERRED_OUTPUTS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Increment when an output is built synchronously (sync D2H + commit).
#[inline]
pub fn record_sync_output() {
    if enabled() {
        SYNC_OUTPUTS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Increment when a step issued a block_table H2D update.
#[inline]
pub fn record_blocks_changed() {
    if enabled() {
        BLOCKS_CHANGED_STEPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Increment when a pending_commit is drained on the slow path
/// (post-fall-through), instead of in the super-fast-path's inline
/// drain.
#[inline]
pub fn record_pending_resolved_late() {
    if enabled() {
        PENDING_RESOLVED_LATE.fetch_add(1, Ordering::Relaxed);
    }
}

/// Phase bands for `phase_record_ns`. Names mirror the major
/// `execute_model_inner` regions for the bench's hot path.
#[derive(Clone, Copy, Debug)]
pub enum Phase {
    PrepareInputs,
}

#[inline]
pub fn phase_record_ns(phase: Phase, ns: u64) {
    if !enabled() {
        return;
    }
    let bucket = match phase {
        Phase::PrepareInputs => &PHASE_PREPARE_INPUTS_NS,
    };
    bucket.fetch_add(ns, Ordering::Relaxed);
}

/// Record an invalidation of `graph_metadata_valid`.
#[inline]
pub fn record_invalidation(site: InvalidationSite) {
    if !enabled() {
        return;
    }
    INVALIDATION_COUNT[site as usize].fetch_add(1, Ordering::Relaxed);
}

// --- printer ----------------------------------------------------------

/// Print the histogram to stderr. Idempotent — counters are not
/// reset, so a second call from a different worker would re-print
/// the same totals. Call from `Drop`.
pub fn drain_and_print() {
    if !enabled() {
        return;
    }
    let total = STEP_TOTAL.load(Ordering::Relaxed);
    if total == 0 {
        return;
    }

    let pct = |n: u64| -> f64 {
        if total == 0 {
            0.0
        } else {
            100.0 * n as f64 / total as f64
        }
    };

    let row = |label: &str, idx: usize| {
        let n = PATH_COUNT[idx].load(Ordering::Relaxed);
        if n == 0 {
            return;
        }
        let ns = PATH_NANOS[idx].load(Ordering::Relaxed);
        let mean_us = ns.checked_div(n).map(|m| m as f64 / 1000.0).unwrap_or(0.0);
        let total_us = ns / 1000;
        eprintln!(
            "  {:<40} count={:>8}  ({:>5.1}%)  total_us={:>10}  mean_us={:>7.2}",
            label,
            n,
            pct(n),
            total_us,
            mean_us
        );
    };

    eprintln!();
    eprintln!("FERRITE_PATH_HISTOGRAM total_steps={total}");
    row("super_fast_hit", PathTag::SuperFastHit as usize);
    row(
        "cold_decode_graph_greedy_fast_meta",
        PathTag::ColdDecodeGraphFastMeta as usize,
    );
    row(
        "cold_decode_graph_greedy_fullh2d",
        PathTag::ColdDecodeGraphFullH2d as usize,
    );
    row(
        "cold_decode_graph_greedy_no_staging",
        PathTag::ColdDecodeGraphNoStaging as usize,
    );
    row(
        "decode_graph_non_greedy_or_full",
        PathTag::DecodeGraphNonGreedyOrFull as usize,
    );
    row(
        "decode_eager_no_graph",
        PathTag::DecodeEagerNoGraph as usize,
    );
    row("mixed_prefill_decode", PathTag::MixedPrefillDecode as usize);
    row("prefill_only_graph", PathTag::PrefillOnlyGraph as usize);
    row(
        "prefill_only_piecewise",
        PathTag::PrefillOnlyPiecewise as usize,
    );
    row("prefill_only_eager", PathTag::PrefillOnlyEager as usize);
    row("piecewise_decode", PathTag::PiecewiseDecode as usize);
    row("pooling_path", PathTag::PoolingPath as usize);
    row("embedding_path", PathTag::EmbeddingPath as usize);
    row("other_terminal", PathTag::Other as usize);

    let untagged = UNTAGGED_STEP.load(Ordering::Relaxed);
    if untagged > 0 {
        eprintln!(
            "  {:<40} count={:>8}  ({:>5.1}%)  <-- BUG: instrumentation missed a return site",
            "UNTAGGED_STEP",
            untagged,
            pct(untagged),
        );
    }

    eprintln!();
    eprintln!("Super-fast gate-failure breakdown (primary reason per failing step):");
    let gate = |label: &str, c: &AtomicU64| {
        let n = c.load(Ordering::Relaxed);
        if n > 0 {
            eprintln!("  {:<40} count={:>8}  ({:>5.1}%)", label, n, pct(n));
        }
    };
    gate("gate_fail_metadata_invalid", &GATE_FAIL_METADATA_INVALID);
    gate("gate_fail_chunked_prefill", &GATE_FAIL_CHUNKED_PREFILL);
    gate("gate_fail_no_sk_bucket", &GATE_FAIL_NO_SK_BUCKET);
    gate("gate_fail_no_graph_bs", &GATE_FAIL_NO_GRAPH_BS);
    gate(
        "gate_fail_no_staging_or_device",
        &GATE_FAIL_NO_STAGING_OR_DEVICE,
    );
    gate("gate_fail_not_greedy", &GATE_FAIL_NOT_GREEDY);
    gate("gate_fail_needs_full", &GATE_FAIL_NEEDS_FULL);

    eprintln!();
    eprintln!("Output build:");
    let n_def = DEFERRED_OUTPUTS.load(Ordering::Relaxed);
    let n_sync = SYNC_OUTPUTS.load(Ordering::Relaxed);
    eprintln!(
        "  deferred_outputs                 count={:>8}  ({:>5.1}%)",
        n_def,
        pct(n_def)
    );
    eprintln!(
        "  sync_outputs                     count={:>8}  ({:>5.1}%)",
        n_sync,
        pct(n_sync)
    );

    let n_blocks = BLOCKS_CHANGED_STEPS.load(Ordering::Relaxed);
    let n_late = PENDING_RESOLVED_LATE.load(Ordering::Relaxed);
    eprintln!(
        "  blocks_changed_steps             count={:>8}  ({:>5.1}%)",
        n_blocks,
        pct(n_blocks)
    );
    eprintln!(
        "  pending_commit_resolved_late     count={:>8}  ({:>5.1}%)",
        n_late,
        pct(n_late)
    );

    eprintln!();
    eprintln!("Phase breakdown (total CPU ns across all steps; mean per step):");
    let phase = |label: &str, c: &AtomicU64| {
        let ns = c.load(Ordering::Relaxed);
        let mean_us = ns
            .checked_div(total)
            .map(|m| m as f64 / 1000.0)
            .unwrap_or(0.0);
        eprintln!(
            "  {:<40} total_ms={:>10.1}  mean_us={:>9.2}",
            label,
            ns as f64 / 1_000_000.0,
            mean_us
        );
    };
    phase("phase_prepare_inputs", &PHASE_PREPARE_INPUTS_NS);

    eprintln!();
    eprintln!("graph_metadata_valid invalidation sites:");
    let inval = |label: &str, idx: usize| {
        let n = INVALIDATION_COUNT[idx].load(Ordering::Relaxed);
        if n > 0 {
            eprintln!("  {:<40} count={:>8}  ({:>5.1}%)", label, n, pct(n));
        }
    };
    inval(
        "invalidation_5223_batch_changed",
        InvalidationSite::BatchChanged5223 as usize,
    );
    inval(
        "invalidation_6238_mixed_prefill",
        InvalidationSite::MixedPrefill6238 as usize,
    );
    inval(
        "invalidation_6827_piecewise_decode",
        InvalidationSite::PiecewiseDecode6827 as usize,
    );
    inval(
        "invalidation_6833_prefill_or_eager_else",
        InvalidationSite::PrefillOrEagerElse6833 as usize,
    );
    inval(
        "invalidation_4992_close",
        InvalidationSite::Close4992 as usize,
    );

    eprintln!();
}
