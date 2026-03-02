// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Worker trait defining the interface for device-level execution.
//!
//! A worker is responsible for a single device (GPU). It handles:
//! - Device initialization
//! - Model loading
//! - KV cache initialization
//! - Model forward pass execution
//! - Memory profiling
//!
//! Port of: `vllm/v1/worker/worker_base.py::WorkerBase`

use std::collections::HashMap;

use vllm_core::scheduler::output::SchedulerOutput;
use vllm_engine::executor::ModelRunnerOutput;

use crate::error::ExecutorResult;

// ---------------------------------------------------------------------------
// Worker trait
// ---------------------------------------------------------------------------

/// Configuration for initializing a worker.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Local device index (e.g., GPU 0, 1, 2, ...).
    pub local_rank: usize,
    /// Global rank in the distributed group.
    pub rank: usize,
    /// Whether this is the driver worker (rank 0 of TP group).
    pub is_driver_worker: bool,
    /// Distributed initialization method (e.g., "tcp://host:port").
    pub distributed_init_method: String,
}

/// Trait defining the worker interface.
///
/// Workers are the device-level execution units. Each worker manages one
/// device (GPU) and handles model execution, KV cache operations, and
/// memory management.
///
/// Port of: `vllm/v1/worker/worker_base.py::WorkerBase`
pub trait Worker: Send {
    /// Initialize the device (e.g., set CUDA device, init distributed).
    fn init_device(&mut self) -> ExecutorResult<()>;

    /// Load the model onto the device.
    fn load_model(&mut self) -> ExecutorResult<()>;

    /// Initialize KV cache from configuration.
    ///
    /// Called after memory profiling to set up the actual KV cache
    /// with the determined number of blocks.
    fn initialize_cache(
        &mut self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
    ) -> ExecutorResult<()>;

    /// Determine available memory for KV cache (in bytes).
    ///
    /// This typically involves a memory profiling step where a dummy
    /// forward pass is run and the remaining memory is measured.
    fn determine_available_memory(&mut self) -> ExecutorResult<usize>;

    /// Execute the model for one step.
    ///
    /// Takes the scheduler output describing which requests to process
    /// and returns the generated tokens and optional logprobs.
    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput>;

    /// Compile or warm up the model for inference.
    ///
    /// This may include CUDA graph capture, JIT compilation, etc.
    fn compile_or_warm_up_model(&mut self) -> ExecutorResult<()> {
        Ok(())
    }

    /// Check whether the worker is healthy.
    fn check_health(&self) -> ExecutorResult<()> {
        Ok(())
    }

    /// Put the worker to sleep, freeing device memory.
    ///
    /// `level` controls aggressiveness: 1 = free KV cache, 2 = free model weights.
    fn sleep(&mut self, _level: u32) -> ExecutorResult<()> {
        Ok(())
    }

    /// Wake the worker from sleep, restoring device memory.
    ///
    /// `tags` specifies which resources to restore. `None` = restore all.
    fn wake_up(&mut self, _tags: Option<&[String]>) -> ExecutorResult<()> {
        Ok(())
    }

    /// Compute embeddings for the given token ID sequences.
    ///
    /// Each inner slice is a single input to embed. Returns one embedding
    /// vector (as `Vec<f32>`) per input.
    ///
    /// Default implementation returns an error — override in workers that
    /// support embedding.
    fn embed(&mut self, _token_id_seqs: &[&[u32]]) -> ExecutorResult<Vec<Vec<f32>>> {
        Err(crate::error::ExecutorError::WorkerExecution(
            "embedding not supported".into(),
        ))
    }

    /// Take the pre-loaded tokenizer, if one was loaded during `load_model()`.
    ///
    /// Workers that support parallel tokenizer loading will parse `tokenizer.json`
    /// on a background thread during weight loading. This method retrieves (and
    /// consumes) that tokenizer so the caller can avoid a redundant load.
    fn take_preloaded_tokenizer(&mut self) -> Option<tokenizers::Tokenizer> {
        None
    }

    /// Return the resolved model architecture name (e.g. "LlamaForCausalLM").
    ///
    /// Available after `load_model()` has been called.
    fn architecture(&self) -> Option<String> {
        None
    }

    /// Shut down the worker and release all resources.
    fn shutdown(&mut self);

    /// The worker's global rank.
    fn rank(&self) -> usize;

    /// The worker's local device rank.
    fn local_rank(&self) -> usize;

    /// Whether this is the driver worker.
    fn is_driver_worker(&self) -> bool;
}

// ---------------------------------------------------------------------------
// NoopWorker -- a minimal worker for testing
// ---------------------------------------------------------------------------

/// A no-op worker for testing executor logic without a real GPU.
///
/// Generates sequential dummy token IDs and reports fake available memory.
pub struct NoopWorker {
    config: WorkerConfig,
    /// Fake available memory in bytes.
    available_memory: usize,
    /// Next token ID to generate (increments per request per step).
    next_token_id: u32,
    /// Whether the worker is initialized.
    initialized: bool,
    /// Whether the worker has been shut down.
    is_shutdown: bool,
}

impl NoopWorker {
    /// Create a new no-op worker.
    pub fn new(config: WorkerConfig, available_memory: usize) -> Self {
        Self {
            config,
            available_memory,
            next_token_id: 1000,
            initialized: false,
            is_shutdown: false,
        }
    }

    /// Create a no-op worker with default config (rank 0, driver).
    pub fn with_defaults(available_memory: usize) -> Self {
        Self::new(
            WorkerConfig {
                local_rank: 0,
                rank: 0,
                is_driver_worker: true,
                distributed_init_method: "tcp://localhost:0".to_string(),
            },
            available_memory,
        )
    }
}

impl Worker for NoopWorker {
    fn init_device(&mut self) -> ExecutorResult<()> {
        self.initialized = true;
        Ok(())
    }

    fn load_model(&mut self) -> ExecutorResult<()> {
        if !self.initialized {
            return Err(crate::error::ExecutorError::WorkerInit(
                "device not initialized".to_string(),
            ));
        }
        Ok(())
    }

    fn initialize_cache(
        &mut self,
        _num_gpu_blocks: usize,
        _num_cpu_blocks: usize,
    ) -> ExecutorResult<()> {
        Ok(())
    }

    fn determine_available_memory(&mut self) -> ExecutorResult<usize> {
        Ok(self.available_memory)
    }

    fn execute_model(
        &mut self,
        scheduler_output: &SchedulerOutput,
    ) -> ExecutorResult<ModelRunnerOutput> {
        let mut token_map = HashMap::new();

        for req_id in scheduler_output.num_scheduled_tokens.keys() {
            token_map.insert(req_id.clone(), vec![self.next_token_id]);
            self.next_token_id += 1;
        }

        Ok(ModelRunnerOutput::from_token_map(token_map))
    }

    fn shutdown(&mut self) {
        self.is_shutdown = true;
    }

    fn rank(&self) -> usize {
        self.config.rank
    }

    fn local_rank(&self) -> usize {
        self.config.local_rank
    }

    fn is_driver_worker(&self) -> bool {
        self.config.is_driver_worker
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config() {
        let config = WorkerConfig {
            local_rank: 0,
            rank: 0,
            is_driver_worker: true,
            distributed_init_method: "tcp://localhost:29500".to_string(),
        };
        assert_eq!(config.rank, 0);
        assert!(config.is_driver_worker);
    }

    #[test]
    fn test_noop_worker_lifecycle() {
        let mut worker = NoopWorker::with_defaults(1024 * 1024 * 1024);

        assert_eq!(worker.rank(), 0);
        assert_eq!(worker.local_rank(), 0);
        assert!(worker.is_driver_worker());

        // Init device.
        worker.init_device().unwrap();

        // Load model.
        worker.load_model().unwrap();

        // Profile memory.
        let memory = worker.determine_available_memory().unwrap();
        assert_eq!(memory, 1024 * 1024 * 1024);

        // Initialize cache.
        worker.initialize_cache(512, 0).unwrap();

        // Health check.
        worker.check_health().unwrap();

        // Shutdown.
        worker.shutdown();
        assert!(worker.is_shutdown);
    }

    #[test]
    fn test_noop_worker_load_before_init() {
        let mut worker = NoopWorker::with_defaults(1024);
        // Load model before init_device should fail.
        let result = worker.load_model();
        assert!(result.is_err());
    }

    #[test]
    fn test_noop_worker_execute() {
        let mut worker = NoopWorker::with_defaults(1024);
        worker.init_device().unwrap();

        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("req-1".to_string(), 50);
        num_scheduled.insert("req-2".to_string(), 30);

        let sched_output = SchedulerOutput {
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 80,
            ..SchedulerOutput::make_empty()
        };

        let output = worker.execute_model(&sched_output).unwrap();
        assert_eq!(output.num_requests(), 2);

        let t1 = output.get_tokens("req-1").unwrap();
        let t2 = output.get_tokens("req-2").unwrap();
        assert_eq!(t1.len(), 1);
        assert_eq!(t2.len(), 1);
        // Tokens should be different.
        assert_ne!(t1[0], t2[0]);
    }

    #[test]
    fn test_noop_worker_sequential_tokens() {
        let mut worker = NoopWorker::with_defaults(1024);
        worker.init_device().unwrap();

        let mut num_scheduled = HashMap::new();
        num_scheduled.insert("req-1".to_string(), 10);

        let sched_output = SchedulerOutput {
            num_scheduled_tokens: num_scheduled,
            total_num_scheduled_tokens: 10,
            ..SchedulerOutput::make_empty()
        };

        let out1 = worker.execute_model(&sched_output).unwrap();
        let out2 = worker.execute_model(&sched_output).unwrap();

        // Token IDs should increment.
        let t1 = out1.get_tokens("req-1").unwrap()[0];
        let t2 = out2.get_tokens("req-1").unwrap()[0];
        assert_eq!(t2, t1 + 1);
    }

    #[test]
    fn test_noop_worker_sleep_wake() {
        let mut worker = NoopWorker::with_defaults(1024);
        worker.sleep(1).unwrap();
        worker.wake_up(None).unwrap();
    }

    #[test]
    fn test_noop_worker_compile() {
        let mut worker = NoopWorker::with_defaults(1024);
        worker.compile_or_warm_up_model().unwrap();
    }
}
