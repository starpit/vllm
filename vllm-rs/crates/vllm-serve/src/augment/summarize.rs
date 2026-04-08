//! Summarizer abstraction for RAG indexing.
//!
//! RAPTOR's tree build needs to call an LLM once per cluster per level.
//! The right backend depends on which path is hosting the indexer:
//!
//!   - **Server path (`AsyncEngine`)**: drives generation through the
//!     existing async server pipeline via `spans::execute_single_text`.
//!   - **Offline LLM path (`InprocClient`, sync `FnMut` closure)**: not yet
//!     implemented — see `OfflineSummarizer` below for the sketch and the
//!     thread-safety constraints that have to be resolved first.
//!
//! By stashing an `Arc<dyn Summarizer>` on `AugmentOptions`, the indexing
//! code stays decoupled from both backends and a future caller (e.g. a
//! standalone `vllm-rag` binary) can plug in a third impl without touching
//! `augment/raptor.rs`.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use spnl_core::optimizer::llo::llir::SingleGenerate;

use crate::server::AppState;

/// One-shot summarization backend used by RAPTOR phase-2 cluster summaries.
#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(&self, spec: &SingleGenerate) -> Result<String>;
}

/// Server-path summarizer: holds an `Arc<AppState>` and reuses
/// `spans::execute_single_text` so RAPTOR summaries go through the same
/// tokenization + scheduling pipeline as user-facing generates.
pub struct AppStateSummarizer {
    state: Arc<AppState>,
}

impl AppStateSummarizer {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl Summarizer for AppStateSummarizer {
    async fn summarize(&self, spec: &SingleGenerate) -> Result<String> {
        crate::spans::execute_single_text(&self.state, spec)
            .await
            .map_err(|e| anyhow!("server-path summarization failed: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Offline LLM path — scoped sync closure.
// ---------------------------------------------------------------------------
//
// The offline `LLM` struct in `llm.rs` drives generation through a
// `FnMut(&[Prompt], …) -> Vec<RequestOutput>` closure that captures
// `&mut self`, so it can never become `Send + Sync` long-term. Instead
// we install the closure into a scoped thread-local for the duration of
// one `augment::index` call, and use a unit-struct `Summarizer` impl
// that reads it back on the calling thread.
//
// Soundness rests on three invariants enforced by the API and by
// raptor's call shape:
//
//   1. `with_sync_summarizer` saves the previous slot value, installs
//      `func`, runs `body`, then restores. Re-entrancy is fine; the
//      slot is per-thread.
//
//   2. The tokio runtime that drives raptor's `cross_index` future MUST
//      be a `current_thread` runtime so every `.await` resumes on the
//      same OS thread that installed the slot. `execute_spnl_struct_sync`
//      uses exactly that. raptor's `spawn_blocking` calls (provider
//      `compute_embeddings`, builder `build_index`) never touch the
//      summarizer, so they're free to run on the blocking pool.
//
//   3. raptor's per-cluster summarizer calls run sequentially within
//      `buffer_unordered`'s polling loop — each `summarize().await`
//      runs the FnMut to completion before the next future is polled —
//      so there's never aliased mutable access to the closure.

use std::cell::Cell;

use spnl_core::optimizer::llo::llir::SingleGenerate as SingleGenerateLlir;

/// Type-erased sync summarizer FnMut. The lifetime parameter lets the
/// closure borrow caller-local state (tokenizer, chat template, the LLM's
/// own generate closure).
pub type SyncSummarizeFnMut<'a> = dyn FnMut(&SingleGenerateLlir) -> anyhow::Result<String> + 'a;

/// Internal slot type — a thin pointer to one of these is what we stash
/// in the thread-local. Boxing the fat trait-object pointer behind a
/// stack-local `SyncSlot` lets us store a single thin `*mut ()` instead
/// of having to split a fat pointer into two halves.
struct SyncSlot<'a> {
    func: &'a mut SyncSummarizeFnMut<'a>,
}

thread_local! {
    static SYNC_SLOT: Cell<*mut ()> = const { Cell::new(std::ptr::null_mut()) };
}

/// Install `func` into the thread-local sync-summarizer slot for the
/// duration of `body`, then restore the previous value.
///
/// Use from synchronous indexing call sites (e.g. the offline LLM path
/// in `execute_spnl_struct_sync`) that want RAPTOR to summarize via
/// their own sync generate closure. The closure must remain valid for
/// the entire scope of `body`.
pub fn with_sync_summarizer<F, R>(func: &mut SyncSummarizeFnMut<'_>, body: F) -> R
where
    F: FnOnce() -> R,
{
    // SAFETY: we extend the lifetime of `func` to that of `SyncSlot<'_>`
    // only so that the on-stack slot can hold it. The slot — and the
    // installed pointer — never escape this function, so the original
    // borrow's actual lifetime is honored at runtime.
    let mut slot: SyncSlot<'_> = SyncSlot {
        func: unsafe {
            std::mem::transmute::<&mut SyncSummarizeFnMut<'_>, &mut SyncSummarizeFnMut<'_>>(func)
        },
    };
    let raw = &mut slot as *mut SyncSlot<'_> as *mut ();
    let prev = SYNC_SLOT.with(|c| c.replace(raw));
    let result = body();
    SYNC_SLOT.with(|c| c.set(prev));
    result
}

/// Borrow the installed sync summarizer (if any) and call `f` with it.
/// Returns `None` if no slot is installed on the current thread.
fn with_installed<R>(f: impl FnOnce(&mut SyncSummarizeFnMut<'_>) -> R) -> Option<R> {
    let raw = SYNC_SLOT.with(|c| c.get());
    if raw.is_null() {
        return None;
    }
    // SAFETY: `raw` was installed by an enclosing `with_sync_summarizer`
    // call on this thread; the slot it points to is alive on the stack
    // until that call returns. raptor's per-cluster summary calls run
    // sequentially within a single `buffer_unordered` polling loop on
    // the calling thread, so we never construct two overlapping
    // mutable borrows of the inner closure.
    let slot: &mut SyncSlot<'_> = unsafe { &mut *(raw as *mut SyncSlot<'_>) };
    Some(f(slot.func))
}

/// Sync-closure summarizer: zero-sized; reads the thread-local slot at
/// call time and dispatches to the installed FnMut. Send + Sync because
/// it has no fields; safety relies on the contract that callers only
/// use it from the thread that installed the slot.
pub struct SyncClosureSummarizer;

#[async_trait]
impl Summarizer for SyncClosureSummarizer {
    async fn summarize(&self, spec: &SingleGenerate) -> Result<String> {
        with_installed(|f| f(spec)).ok_or_else(|| {
            anyhow!("SyncClosureSummarizer used without an installed sync summarizer slot")
        })?
    }
}
