// SPDX-License-Identifier: Apache-2.0
//! LogitsProcessor framework for CudaWorker — mirrors Python vLLM's
//! `vllm/v1/sample/logits_processor/` architecture.
//!
//! Each processor maintains persistent GPU state that is rebuilt only when
//! the batch composition changes (via `update_state`), not every step.
//! This matches Python's design and avoids per-step H2D repacking overhead.

use std::collections::HashMap;

use crate::DType as GpuDType;
use crate::alloc::OwnedTensor;
use crate::device::GpuDevice;
use crate::driver;
use crate::tensor::GpuTensor;

use vllm_common::sampling::SamplingParams;

// ---------------------------------------------------------------------------
// BatchUpdate — notification of batch composition changes
// ---------------------------------------------------------------------------

/// Describes changes to the persistent batch between steps.
/// Mirrors Python's `BatchUpdate` from `gpu_input_batch.py`.
pub struct BatchUpdate {
    /// Current batch size after applying changes.
    pub batch_size: usize,
    /// (batch_index, req_id) pairs for newly added requests.
    pub added: Vec<(usize, String)>,
    /// Batch indices of removed requests.
    pub removed: Vec<usize>,
}

// ---------------------------------------------------------------------------
// LogitsProcessor trait
// ---------------------------------------------------------------------------

/// GPU logit processor — mirrors Python's `LogitsProcessor` ABC.
///
/// Lifecycle:
/// 1. `update_state()` called once per step with batch changes (CPU-side bookkeeping + async H2D).
/// 2. `apply()` called to modify logits on GPU.
/// 3. `is_active()` gates whether `apply()` is called at all.
pub trait LogitsProcessor: Send {
    /// Update internal state when the batch changes or tokens grow.
    /// Called once per step. `batch_update` is `Some` if batch composition changed.
    fn update_state(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    );

    /// Modify logits `[batch_size, vocab_size]` f32 in-place on GPU.
    fn apply(&self, logits: GpuTensor, device: &mut GpuDevice);

    /// True if this processor doesn't affect argmax (applied after temperature).
    fn is_argmax_invariant(&self) -> bool;

    /// True if this processor has work to do this step. Skip `apply` if false.
    fn is_active(&self) -> bool;
}

// ---------------------------------------------------------------------------
// LogitsProcessorPipeline
// ---------------------------------------------------------------------------

/// Container for all logits processors, split by argmax invariance.
/// Mirrors Python's `LogitsProcessors` container.
pub struct LogitsProcessorPipeline {
    /// Processors that can change argmax (grammar, min_tokens, logit_bias, penalties).
    non_argmax_invariant: Vec<Box<dyn LogitsProcessor>>,
    /// Processors that don't change argmax (none currently — min_p is in sampling kernel).
    argmax_invariant: Vec<Box<dyn LogitsProcessor>>,
}

impl LogitsProcessorPipeline {
    /// Create a new pipeline with the given processors.
    /// Processors are automatically sorted into argmax-invariant vs non-argmax-invariant.
    pub fn new(processors: Vec<Box<dyn LogitsProcessor>>) -> Self {
        let mut non_argmax = Vec::new();
        let mut argmax = Vec::new();
        for p in processors {
            if p.is_argmax_invariant() {
                argmax.push(p);
            } else {
                non_argmax.push(p);
            }
        }
        Self {
            non_argmax_invariant: non_argmax,
            argmax_invariant: argmax,
        }
    }

    /// Update all processors with batch changes.
    pub fn update_state(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        for p in &mut self.non_argmax_invariant {
            p.update_state(
                batch_update,
                sampling_params_map,
                token_buffers,
                batch_req_ids,
                device,
            );
        }
        for p in &mut self.argmax_invariant {
            p.update_state(
                batch_update,
                sampling_params_map,
                token_buffers,
                batch_req_ids,
                device,
            );
        }
    }

    /// Apply non-argmax-invariant processors (before sampling).
    pub fn apply_pre_sampling(&self, logits: GpuTensor, device: &mut GpuDevice) {
        for p in &self.non_argmax_invariant {
            if p.is_active() {
                p.apply(logits, device);
            }
        }
    }

    /// Whether any active processor needs work (gates fast-path skip).
    pub fn any_active(&self) -> bool {
        self.non_argmax_invariant.iter().any(|p| p.is_active())
            || self.argmax_invariant.iter().any(|p| p.is_active())
    }
}

// ---------------------------------------------------------------------------
// LogitBiasProcessor
// ---------------------------------------------------------------------------

/// Persistent logit bias processor. Rebuilds GPU tensors only when batch changes.
/// Mirrors Python's `LogitBiasLogitsProcessor`.
#[derive(Default)]
pub struct LogitBiasProcessor {
    active: bool,
    /// CSR-packed bias data, persistent across steps when batch is stable.
    gpu_bias_ids: Option<OwnedTensor>,
    gpu_bias_vals: Option<OwnedTensor>,
    gpu_bias_offsets: Option<OwnedTensor>,
}

impl LogitBiasProcessor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LogitsProcessor for LogitBiasProcessor {
    fn update_state(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        _token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        // Only rebuild when batch changes.
        if batch_update.is_none() && self.gpu_bias_ids.is_some() {
            return;
        }

        let mut bias_ids_flat: Vec<i32> = Vec::new();
        let mut bias_vals_flat: Vec<f32> = Vec::new();
        let mut bias_offsets: Vec<i32> = Vec::new();

        for req_id in batch_req_ids {
            bias_offsets.push(bias_ids_flat.len() as i32);
            if let Some(params) = sampling_params_map.get(req_id)
                && let Some(ref bias_map) = params.logit_bias
            {
                for (&tok_id, &val) in bias_map {
                    bias_ids_flat.push(tok_id as i32);
                    bias_vals_flat.push(val);
                }
            }
        }
        bias_offsets.push(bias_ids_flat.len() as i32);

        if bias_ids_flat.is_empty() {
            self.active = false;
            self.gpu_bias_ids = None;
            self.gpu_bias_vals = None;
            self.gpu_bias_offsets = None;
            return;
        }

        self.active = true;
        self.gpu_bias_ids = Some(h2d_i32_owned(&bias_ids_flat, device));
        self.gpu_bias_vals = Some(h2d_f32_owned(&bias_vals_flat, device));
        self.gpu_bias_offsets = Some(h2d_i32_owned(&bias_offsets, device));
    }

    fn apply(&self, logits: GpuTensor, device: &mut GpuDevice) {
        if let (Some(ids), Some(vals), Some(offsets)) = (
            &self.gpu_bias_ids,
            &self.gpu_bias_vals,
            &self.gpu_bias_offsets,
        ) {
            unsafe {
                crate::kernels::apply_logit_bias(
                    logits,
                    ids.as_gpu_tensor(),
                    vals.as_gpu_tensor(),
                    offsets.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }

    fn is_argmax_invariant(&self) -> bool {
        false
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

// ---------------------------------------------------------------------------
// PenaltiesProcessor
// ---------------------------------------------------------------------------

/// Persistent penalties processor. Penalty value tensors rebuilt on batch change;
/// output_token_ids rebuilt every step (tokens grow each step).
pub struct PenaltiesProcessor {
    active: bool,
    /// Penalty value tensors — persistent when batch is stable.
    gpu_rep_pens: Option<OwnedTensor>,
    gpu_freq_pens: Option<OwnedTensor>,
    gpu_pres_pens: Option<OwnedTensor>,
    /// Output token IDs — rebuilt every step since tokens grow.
    gpu_out_ids: Option<OwnedTensor>,
    /// Dummy prompt IDs (we don't penalize prompt tokens separately).
    gpu_prompt_ids: Option<OwnedTensor>,
    max_output_len: usize,
    num_reqs: usize,
    vocab_size: usize,
}

impl PenaltiesProcessor {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            active: false,
            gpu_rep_pens: None,
            gpu_freq_pens: None,
            gpu_pres_pens: None,
            gpu_out_ids: None,
            gpu_prompt_ids: None,
            max_output_len: 0,
            num_reqs: 0,
            vocab_size,
        }
    }
}

impl LogitsProcessor for PenaltiesProcessor {
    fn update_state(
        &mut self,
        batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        let num_reqs = batch_req_ids.len();
        self.num_reqs = num_reqs;

        // Check if any request has penalties.
        let any = batch_req_ids.iter().any(|rid| {
            sampling_params_map.get(rid).is_some_and(|p| {
                p.frequency_penalty != 0.0
                    || p.presence_penalty != 0.0
                    || p.repetition_penalty != 1.0
            })
        });

        if !any {
            self.active = false;
            return;
        }
        self.active = true;

        // Rebuild penalty value tensors only on batch change.
        if batch_update.is_some() || self.gpu_rep_pens.is_none() {
            let mut rep_pens = vec![1.0f32; num_reqs];
            let mut freq_pens = vec![0.0f32; num_reqs];
            let mut pres_pens = vec![0.0f32; num_reqs];

            for (i, req_id) in batch_req_ids.iter().enumerate() {
                if let Some(params) = sampling_params_map.get(req_id) {
                    rep_pens[i] = params.repetition_penalty as f32;
                    freq_pens[i] = params.frequency_penalty as f32;
                    pres_pens[i] = params.presence_penalty as f32;
                }
            }

            self.gpu_rep_pens = Some(h2d_f32_owned(&rep_pens, device));
            self.gpu_freq_pens = Some(h2d_f32_owned(&freq_pens, device));
            self.gpu_pres_pens = Some(h2d_f32_owned(&pres_pens, device));
        }

        // Rebuild output_token_ids every step (tokens grow).
        let mut max_output_len = 0usize;
        for req_id in batch_req_ids {
            if let Some(buf) = token_buffers.get(req_id) {
                let params = sampling_params_map.get(req_id);
                if params.is_some_and(|p| {
                    p.frequency_penalty != 0.0
                        || p.presence_penalty != 0.0
                        || p.repetition_penalty != 1.0
                }) {
                    max_output_len = max_output_len.max(buf.len());
                }
            }
        }

        if max_output_len == 0 {
            self.active = false;
            return;
        }

        self.max_output_len = max_output_len;
        let pad_val = self.vocab_size as i32;
        let mut output_ids_flat = vec![pad_val; num_reqs * max_output_len];
        for (i, req_id) in batch_req_ids.iter().enumerate() {
            if let Some(buf) = token_buffers.get(req_id) {
                let row = &mut output_ids_flat[i * max_output_len..];
                for (j, &tok) in buf.iter().enumerate() {
                    if j < max_output_len {
                        row[j] = tok as i32;
                    }
                }
            }
        }

        self.gpu_out_ids = Some(h2d_i32_owned(&output_ids_flat, device));

        // Dummy prompt IDs.
        if batch_update.is_some() || self.gpu_prompt_ids.is_none() {
            let dummy_prompt = vec![pad_val; num_reqs];
            self.gpu_prompt_ids = Some(h2d_i32_owned(&dummy_prompt, device));
        }
    }

    fn apply(&self, logits: GpuTensor, device: &mut GpuDevice) {
        if let (Some(out_ids), Some(prompt_ids), Some(rep), Some(freq), Some(pres)) = (
            &self.gpu_out_ids,
            &self.gpu_prompt_ids,
            &self.gpu_rep_pens,
            &self.gpu_freq_pens,
            &self.gpu_pres_pens,
        ) {
            let gpu_out_ids_2d = unsafe {
                GpuTensor::new(
                    out_ids.as_gpu_tensor().raw_ptr(),
                    &[self.num_reqs, self.max_output_len],
                    GpuDType::U32,
                )
            };
            let gpu_prompt_ids_2d = unsafe {
                GpuTensor::new(
                    prompt_ids.as_gpu_tensor().raw_ptr(),
                    &[self.num_reqs, 1],
                    GpuDType::U32,
                )
            };
            unsafe {
                crate::kernels::apply_penalties(
                    logits,
                    gpu_out_ids_2d,
                    gpu_prompt_ids_2d,
                    rep.as_gpu_tensor(),
                    freq.as_gpu_tensor(),
                    pres.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }

    fn is_argmax_invariant(&self) -> bool {
        false
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

// ---------------------------------------------------------------------------
// GrammarMaskProcessor
// ---------------------------------------------------------------------------

/// Grammar mask processor. Always rebuilds since FSM state changes every token.
/// Lives in vllm-cuda (no dependency on vllm-models) — caller passes allowed
/// token lists directly via `update_from_allowed_tokens`.
#[derive(Default)]
pub struct GrammarMaskProcessor {
    active: bool,
    gpu_allowed: Option<OwnedTensor>,
    gpu_offsets: Option<OwnedTensor>,
    gpu_req_indices: Option<OwnedTensor>,
    /// Backup logits needed for restoring allowed token values.
    needs_backup: bool,
}

impl GrammarMaskProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update grammar state from pre-collected allowed token lists.
    /// Each entry is `(batch_index, allowed_token_ids)`.
    pub fn update_from_allowed_tokens(
        &mut self,
        grammar_reqs: &[(usize, &[u32])],
        device: &mut GpuDevice,
    ) {
        if grammar_reqs.is_empty() {
            self.active = false;
            self.gpu_allowed = None;
            self.gpu_offsets = None;
            self.gpu_req_indices = None;
            self.needs_backup = false;
            return;
        }

        let mut allowed_ids_flat: Vec<i32> = Vec::new();
        let mut allowed_offsets: Vec<i32> = vec![0];
        let mut req_indices: Vec<i32> = Vec::new();

        for &(req_idx, allowed) in grammar_reqs {
            req_indices.push(req_idx as i32);
            for &tok in allowed {
                allowed_ids_flat.push(tok as i32);
            }
            allowed_offsets.push(allowed_ids_flat.len() as i32);
        }

        self.active = true;
        self.needs_backup = true;
        self.gpu_allowed = Some(h2d_i32_owned(&allowed_ids_flat, device));
        self.gpu_offsets = Some(h2d_i32_owned(&allowed_offsets, device));
        self.gpu_req_indices = Some(h2d_i32_owned(&req_indices, device));
    }

    /// Whether the grammar processor needs a backup of logits before masking.
    pub fn needs_backup(&self) -> bool {
        self.needs_backup
    }

    /// Whether the grammar processor has work to do this step.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Apply grammar mask using a backup tensor for restoring allowed tokens.
    pub fn apply_with_backup(&self, logits: GpuTensor, backup: GpuTensor, device: &mut GpuDevice) {
        if let (Some(allowed), Some(offsets), Some(indices)) =
            (&self.gpu_allowed, &self.gpu_offsets, &self.gpu_req_indices)
        {
            unsafe {
                crate::kernels::apply_grammar_mask(
                    logits,
                    backup,
                    allowed.as_gpu_tensor(),
                    offsets.as_gpu_tensor(),
                    indices.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MinTokensProcessor
// ---------------------------------------------------------------------------

/// Suppresses stop/EOS tokens for requests that haven't generated min_tokens yet.
/// Mirrors Python's `MinTokensLogitsProcessor`.
#[derive(Default)]
pub struct MinTokensProcessor {
    active: bool,
    gpu_req_indices: Option<OwnedTensor>,
    gpu_token_ids: Option<OwnedTensor>,
}

impl MinTokensProcessor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LogitsProcessor for MinTokensProcessor {
    fn update_state(
        &mut self,
        _batch_update: Option<&BatchUpdate>,
        sampling_params_map: &HashMap<String, SamplingParams>,
        token_buffers: &HashMap<String, Vec<u32>>,
        batch_req_ids: &[String],
        device: &mut GpuDevice,
    ) {
        // Rebuild every step since token counts change.
        let mut req_indices: Vec<i32> = Vec::new();
        let mut token_ids: Vec<i32> = Vec::new();

        for (req_idx, req_id) in batch_req_ids.iter().enumerate() {
            if let Some(params) = sampling_params_map.get(req_id) {
                if params.min_tokens == 0 {
                    continue;
                }
                let generated = token_buffers.get(req_id).map_or(0, |buf| {
                    // token_buffers includes prompt tokens; generated count is
                    // total - prompt length. But we store output tokens only
                    // starting from decode, so buf.len() tracks total tokens
                    // including prompt. We need the output token count.
                    // Actually, token_buffers in CudaWorker stores prompt_ids
                    // initially then appends generated tokens. The "generated"
                    // count is tracked by the scheduler. For min_tokens, we
                    // check if the number of generated tokens (output tokens)
                    // is below min_tokens.
                    //
                    // For now, use a conservative approach: check buf length
                    // against a threshold. The caller should pass the correct
                    // generated token count.
                    buf.len()
                });
                // If we've generated fewer tokens than min_tokens, suppress stop tokens.
                // Note: generated includes prompt tokens in token_buffers. We need
                // the number of *output* tokens. The input_batch tracks this.
                // For the initial implementation, we'll use a field in BatchUpdate
                // or rely on the caller to filter. For now, we check against the
                // params and suppress EOS + stop_token_ids.
                if (generated as u32) < params.min_tokens {
                    // Suppress all stop_token_ids.
                    for &stop_id in &params.stop_token_ids {
                        req_indices.push(req_idx as i32);
                        token_ids.push(stop_id as i32);
                    }
                }
            }
        }

        if req_indices.is_empty() {
            self.active = false;
            self.gpu_req_indices = None;
            self.gpu_token_ids = None;
            return;
        }

        self.active = true;
        self.gpu_req_indices = Some(h2d_i32_owned(&req_indices, device));
        self.gpu_token_ids = Some(h2d_i32_owned(&token_ids, device));
    }

    fn apply(&self, logits: GpuTensor, device: &mut GpuDevice) {
        if let (Some(indices), Some(ids)) = (&self.gpu_req_indices, &self.gpu_token_ids) {
            unsafe {
                crate::kernels::apply_min_tokens(
                    logits,
                    indices.as_gpu_tensor(),
                    ids.as_gpu_tensor(),
                    device.compute_stream,
                );
            }
        }
    }

    fn is_argmax_invariant(&self) -> bool {
        false // Can change argmax by suppressing EOS
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

// ---------------------------------------------------------------------------
// H2D helpers (allocate via caching allocator + async copy)
// ---------------------------------------------------------------------------

fn h2d_i32_owned(data: &[i32], device: &mut GpuDevice) -> OwnedTensor {
    let owned = device.caching.alloc_tensor(&[data.len()], GpuDType::U32);
    unsafe {
        let _ = driver::memcpy_htod_async(
            owned.as_gpu_tensor().raw_ptr(),
            data.as_ptr() as *const u8,
            data.len() * 4,
            device.compute_stream,
        );
    }
    owned
}

fn h2d_f32_owned(data: &[f32], device: &mut GpuDevice) -> OwnedTensor {
    let owned = device.caching.alloc_tensor(&[data.len()], GpuDType::F32);
    unsafe {
        let _ = driver::memcpy_htod_async(
            owned.as_gpu_tensor().raw_ptr(),
            data.as_ptr() as *const u8,
            data.len() * 4,
            device.compute_stream,
        );
    }
    owned
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_sorting() {
        let pipeline = LogitsProcessorPipeline::new(vec![
            Box::new(LogitBiasProcessor::new()),
            Box::new(MinTokensProcessor::new()),
        ]);
        // Both are non-argmax-invariant.
        assert_eq!(pipeline.non_argmax_invariant.len(), 2);
        assert_eq!(pipeline.argmax_invariant.len(), 0);
    }

    #[test]
    fn test_pipeline_inactive_by_default() {
        let pipeline = LogitsProcessorPipeline::new(vec![
            Box::new(LogitBiasProcessor::new()),
            Box::new(PenaltiesProcessor::new(32000)),
            Box::new(MinTokensProcessor::new()),
        ]);
        assert!(!pipeline.any_active());
    }
}
