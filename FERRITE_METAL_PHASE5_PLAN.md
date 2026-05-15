# Phase 5: Metal Executor Implementation Plan

## Current State (2026-05-06)

### ✅ Complete (Phase 5.1-5.5)
- **MetalExecutor**: Full ICB-based executor with 21 instruction recorders
- **Metal Dispatcher**: `MetalWeights` trait, `MetalWeightsWrapper`, `try_load_metal()`
- **Forward Pipeline**: 7-step execution (buffer allocation → ICB execution → sync)
- **Loop Unrolling**: Handles layer loops at init time
- **Weight Slot Mapping**: Dynamic weight buffer allocation

### 🔄 Next: Phase 5.6 - Test with Real Model (Use Existing Golden Infrastructure)

**Goal**: Wire Metal backend into existing `vllm-e2e` golden test framework

**Key Insight**: Don't reinvent testing infrastructure. Reuse:
- `vllm-e2e/tests/e_correctness.rs` - Golden test framework
- `ferrite-forward/src/cpu_golden.rs` - CPU reference implementations
- `testdata/golden/*.json` - Pre-generated golden references

---

## Step 5.6.1: Wire `#[forward]` Macro to Emit Metal Code

**Goal**: Make the `#[forward]` macro generate Metal executor code alongside CUDA code.

### Current State:
- `#[forward]` macro in `ferrite-forward-macro/src/lib.rs`
- Already has `codegen_metal.rs` module (partially implemented)
- Solver already emits backend-agnostic `Instruction<W>[]` tape

### Tasks:
1. **Study macro flow**:
   - Entry: `ferrite-forward-macro/src/lib.rs::forward_impl()`
   - Solver: `ferrite-forward-macro/src/solver.rs::solve()`
   - CUDA codegen: `ferrite-forward-macro/src/codegen.rs`
   - Metal codegen: `ferrite-forward-macro/src/codegen_metal.rs` (exists but incomplete)

2. **Enable Metal codegen path**:
   - In `lib.rs::forward_impl()`, add Metal codegen alongside CUDA
   - Emit `MetalWeightsWrapper<W>` for each model config
   - Emit `MetalArchRegistration` with `try_load_metal_<model>()` function
   - Use `#[cfg(feature = "metal")]` guards

3. **Implement weight loading in generated code**:
   - Generated `try_load_metal_<model>()` should:
     - Load safetensors via `MetalWeights::load_safetensors()`
     - Call solver to get instruction tape
     - Create `MetalExecutor::new(tape, weights, num_slots)`
     - Return `Box<dyn MetalWeights>`

4. **Test macro output**:
   - Run `cargo expand` on `ferrite-model-llama` with `--features metal`
   - Verify Metal code is generated alongside CUDA code
   - Check `try_load_metal_tinyllama_1_1b()` function exists

### Reference Files:
- `ferrite-forward-macro/src/lib.rs:1150-1250` - Arch dispatcher emission
- `ferrite-forward-macro/src/codegen.rs` - CUDA codegen pattern
- `vllm-executor/src/cuda_worker.rs:5297` - How CUDA uses `try_load()`

### Success Criteria:
- [ ] `#[forward]` macro emits Metal code when `--features metal` enabled
- [ ] Generated code compiles without errors
- [ ] `try_load_metal()` function exists and is callable
- [ ] TinyLlama registration is auto-generated

---

## Step 5.6.2: Implement `MetalWeights::load_safetensors()`

**Goal**: Load safetensors into Metal buffers (minimal implementation for TinyLlama).

### Tasks:
1. **Create `MetalWeights` struct**:
   - Location: `ferrite-metal-kernels/src/weights.rs` (new file)
   - Fields: `device`, `tensors: HashMap<String, metal::Buffer>`, `mmap`
   - Methods: `new()`, `load_safetensors()`, `get_tensor()`, `take_tensor()`

2. **Implement safetensors parser**:
   - Use `safetensors` crate to parse header
   - For each tensor: read shape, dtype, byte offset
   - Allocate Metal buffer and copy data from mmap
   - Store in HashMap with tensor name as key

3. **Handle fp16/bf16 dtypes**:
   - TinyLlama uses bf16
   - Metal supports both fp16 and bf16 natively
   - No conversion needed initially

4. **Test with TinyLlama checkpoint**:
   - Download TinyLlama-1.1B-Chat-v1.0 if not present
   - Load `model.safetensors` (or sharded files)
   - Verify all expected tensors loaded
   - Check buffer shapes match config

### Reference Files:
- `ferrite-cuda-core/src/weights.rs` - CUDA weight loading pattern
- `ferrite-forward/src/loaders.rs` - Weight loading helpers

### Success Criteria:
- [ ] `MetalWeights::load_safetensors()` compiles
- [ ] Can load TinyLlama checkpoint
- [ ] All tensors accessible via `get_tensor()`
- [ ] Buffer shapes validated against config

---

## Step 5.6.3: Add Metal Backend to `vllm-e2e` Tests

**Goal**: Extend existing golden test framework to support Metal backend.

### Tasks:
1. **Add Metal feature to `vllm-e2e`**:
   - Update `vllm-e2e/Cargo.toml` with `metal` feature
   - Add Metal-specific test models to `TestModels` struct
   - Example: `TINYLLAMA_METAL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"`

2. **Create Metal golden test**:
   - Location: `vllm-e2e/tests/e_correctness.rs`
   - Add `#[cfg(feature = "metal")]` test function
   - Example:
     ```rust
     #[tokio::test]
     #[ignore]
     #[cfg(feature = "metal")]
     async fn test_metal_correctness_tinyllama_1b() {
         run_correctness_test(TestModels::TINYLLAMA_METAL, "tinyllama_1b_metal").await;
     }
     ```

3. **Generate golden reference**:
   - Use existing `scripts/generate_golden_refs.py`
   - Run with TinyLlama on CUDA/CPU to generate golden
   - Save to `testdata/golden/tinyllama_1b_metal.json`

4. **Run Metal test**:
   - `cargo test -p vllm-e2e --features e2e,metal --test e_correctness -- test_metal_correctness_tinyllama_1b --ignored`
   - Compare Metal output against golden reference
   - Use existing `check_logprobs_close_with_threshold()` for validation

### Reference Files:
- `vllm-e2e/tests/e_correctness.rs` - Golden test framework
- `vllm-e2e/src/lib.rs` - TestModels definitions
- `scripts/generate_golden_refs.py` - Golden generation script

### Success Criteria:
- [ ] Metal test compiles and runs
- [ ] Golden reference generated for TinyLlama
- [ ] Metal output matches golden within threshold
- [ ] Test passes on M1 Max hardware

---

## Step 5.6.4: Debug and Validate

**Goal**: Ensure Metal forward pass produces correct outputs.

### Tasks:
1. **Compare against CPU golden**:
   - Use `ferrite-forward/src/cpu_golden.rs` reference implementations
   - Run same ops on CPU and Metal
   - Compare intermediate outputs (not just final logits)
   - Identify any numerical differences

2. **Add Metal-specific assertions**:
   - Check for NaN/Inf in output buffers
   - Verify logit range is reasonable (-100 to 100)
   - Check argmax per token gives valid token IDs
   - Validate output shape matches expected

3. **Profile execution**:
   - Use Metal's Instruments to profile
   - Check GPU utilization
   - Measure latency per token
   - Identify any bottlenecks

4. **Fix any issues**:
   - If outputs don't match: debug instruction recorders
   - If crashes: check buffer bindings and ICB recording
   - If slow: optimize buffer allocation or ICB execution

### Reference Files:
- `ferrite-forward/src/cpu_golden.rs` - CPU reference implementations
- `vllm-e2e/src/assertions.rs` - Assertion helpers

### Success Criteria:
- [ ] Metal outputs match CPU golden within tolerance
- [ ] No NaN/Inf in outputs
- [ ] Forward pass completes in <100ms for 5 tokens
- [ ] Test runs reliably without crashes

---

## Implementation Order

1. **Day 1**: Step 5.6.1 - Wire `#[forward]` macro (4-6 hours)
2. **Day 2**: Step 5.6.2 - Implement `MetalWeights::load_safetensors()` (4-6 hours)
3. **Day 3**: Step 5.6.3 - Add Metal to `vllm-e2e` tests (2-4 hours)
4. **Day 3**: Step 5.6.4 - Debug and validate (2-4 hours)

**Total: 2-3 days**

---

## Key Files to Modify

### Existing Files:
1. `ferrite-forward-macro/src/lib.rs` - Add Metal codegen path
2. `ferrite-forward-macro/src/codegen_metal.rs` - Complete Metal codegen
3. `vllm-e2e/Cargo.toml` - Add `metal` feature
4. `vllm-e2e/src/lib.rs` - Add Metal test models
5. `vllm-e2e/tests/e_correctness.rs` - Add Metal tests

### New Files:
1. `ferrite-metal-kernels/src/weights.rs` - Metal weight loading
2. `testdata/golden/tinyllama_1b_metal.json` - Golden reference

---

## Success Metrics

- [ ] `#[forward]` macro generates Metal code
- [ ] TinyLlama loads successfully on Metal
- [ ] Forward pass produces correct outputs
- [ ] Golden test passes on M1 Max
- [ ] Execution time <100ms for 5 tokens

---

## After Phase 5.6

- **Phase 5.7**: Verify numerical correctness against CUDA reference
- **Phase 5.8**: Profile and optimize hot paths
- **Phase 6**: Production readiness