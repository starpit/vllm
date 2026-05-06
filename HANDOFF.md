# Ferrite Metal Port - Current State

## Phase 5: Metal Interpreter-Based Executor 🔄 IN PROGRESS (Phase 5.1-5.2 Complete)

**Architecture:** Runtime interpreter that walks instruction tape at init time and records to ICB.

### What's Complete

**Phase 5.1-5.2: Skeleton Implementation:**
- `MetalExecutor` struct created in `ferrite-forward/src/interpreter/metal.rs`
- Instruction matching skeleton for all variants
- Integration with Phase 4.6's `RecordingContext`
- TODO markers for recording loop implementation

**Architecture (Parallel to CUDA):**
- **CUDA:** Walks instruction tape at runtime, calls `Instruction::eval()` on each
- **Metal:** Walks instruction tape at init time, calls `record_*()` methods to populate ICB, then executes ICB at runtime

**Key Insight:** Phase 5 USES Phase 4.6's ICB recording infrastructure. The interpreter walks the tape at init time and calls the existing `record_*()` methods to populate the ICB.

**Implementation Files:**
- `ferrite-forward/src/interpreter/metal.rs` (151 lines, skeleton)
- `ferrite-forward/src/interpreter/metal_tests.rs` (placeholder)
- `ferrite-forward/src/interpreter/mod.rs` (module declaration)

**Files from Phase 4.6 (Reused):**
- `ferrite-metal-kernels/src/instruction_executor/mod.rs` (RecordingContext)
- `ferrite-metal-kernels/src/instruction_executor/rmsnorm.rs`
- `ferrite-metal-kernels/src/instruction_executor/gemm.rs`
- `ferrite-metal-kernels/src/instruction_executor/attention.rs`
- `ferrite-metal-kernels/src/instruction_executor/fused.rs`

**Documentation:**
- `FERRITE_METAL_PHASE5_PLAN.md` - Phase 5.6+ roadmap
- `FERRITE_METAL_EXECUTION_OPTIONS.md` - Execution strategy analysis
- `FERRITE_METAL_ICB_ARCHITECTURE.md` - ICB architecture (historical)
- `FERRITE_METAL_ICB_INHERIT_BUFFERS.md` - ICB buffer inheritance (historical)
- `FERRITE_METAL_LOOP_HANDLING.md` - Loop handling strategies

### What's TODO (Phase 5.3-5.5)

**Phase 5.3: Implement instruction recording loop**
- Match each `Instruction` variant in `MetalExecutor::new()`
- Call corresponding `record_*()` method from Phase 4.6
- Handle weight slot mapping

**Phase 5.4: Implement `forward()` execution**
- Create tile table (Metal equivalent of Vec<Option<TileEntry>>)
- Bind runtime buffers (input_ids, positions)
- Execute ICB on compute encoder

**Phase 5.5: Add MetalExecutor tests**
- Test with synthetic instruction tape
- Verify ICB execution

### Next: Phase 5.6 - Test with Real Model

**Goal:** Wire Metal backend into existing `vllm-e2e` golden test framework.

**Steps:**
1. Complete Phase 5.3-5.5 (instruction recording loop + forward execution)
2. Wire `#[forward]` macro to emit Metal code
3. Implement `MetalWeights::load_safetensors()`
4. Add Metal backend to `vllm-e2e` tests
5. Debug and validate with TinyLlama-1.1B

**Estimated Time:** 2-3 days (after Phase 5.3-5.5 complete)

See `FERRITE_METAL_PHASE5_PLAN.md` for detailed breakdown.

---

## Historical Context: Phase 4.6 ICB Investigation

### 🎉 BREAKTHROUGH: Compute ICBs Work on Apple Silicon! (May 2026)

**The missing piece was `supportIndirectCommandBuffers=YES` on the pipeline.**

Metal debug layer revealed the issue:
```
-[MTLDebugComputeCommandEncoder executeCommandsInBuffer:withRange:]:1773: 
failed assertion `The indirect command buffer inherits pipelines 
( inheritPipelineState = YES) but the compute pipeline set on this 
encoder does not support indirect command buffers 
( supportIndirectCommandBuffers = NO )'
```

#### Complete Solution (Verified on M1 Max)

**1. Create pipeline with ICB support:**
```objective-c
MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
desc.computeFunction = function;
desc.supportIndirectCommandBuffers = YES;  // <-- THE KEY!

id<MTLComputePipelineState> pipeline = 
    [device newComputePipelineStateWithDescriptor:desc
                                          options:0
                                       reflection:nil
                                            error:&error];
```

**2. Create ICB with correct configuration:**
```objective-c
MTLIndirectCommandBufferDescriptor* icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
icbDesc.inheritPipelineState = YES;  // Pipeline comes from encoder
icbDesc.inheritBuffers = NO;
icbDesc.maxKernelBufferBindCount = 1;

id<MTLIndirectCommandBuffer> icb = 
    [device newIndirectCommandBufferWithDescriptor:icbDesc
                                   maxCommandCount:1
                                           options:MTLResourceStorageModeShared];  // SHARED, not Private!
```

**3. Encode ICB command (WITHOUT calling setComputePipelineState):**
```objective-c
id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
[cmd reset];

// DO NOT call setComputePipelineState - that causes segfault!
// Only set buffers and dispatch:
[cmd setKernelBuffer:buffer offset:0 atIndex:0];
[cmd concurrentDispatchThreadgroups:MTLSizeMake(1, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
```

**4. Execute ICB with pipeline on encoder:**
```objective-c
id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
[encoder setComputePipelineState:pipeline];  // ICB inherits this
[encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)];
[encoder endEncoding];
```

#### Test Results (M1 Max)

```
✅ Pipeline created with supportIndirectCommandBuffers=YES
✅ ICB created (inheritPipelineState=YES, StorageMode=SHARED)
✅ ICB command encoded (buffer + dispatch, no pipeline)
✅ ICB execution command encoded
✅ Command buffer completed successfully
Output: [42, 42, 42, 42]

🎉 SUCCESS! Compute ICBs work on Apple Silicon!
```

#### Investigation Files

- `PHASE4_ICB_BREAKTHROUGH.md` - Complete solution documentation
- `test_icb_with_icb_support.m` - Working test with correct configuration
- `test_icb_pipeline_support.m` - Alternative working test
- `test_icb_storage_mode_fix.m` - Storage mode investigation (still crashed)
- `test_icb_inherit_only.m` - Inheritance investigation (GPU hang)
- `test_icb_minimal.m` - Minimal crash reproduction
- `test_icb_minimal_validated.m` - Metal debug layer investigation

#### Lessons Learned

1. **Metal debug layer is essential** - It revealed the exact issue
2. **Read error messages carefully** - "supportIndirectCommandBuffers = NO" was the clue
3. **Don't give up too early** - The API works, we just needed the right configuration
4. **Apple's documentation is incomplete** - They don't mention `supportIndirectCommandBuffers` requirement for compute ICBs

**Phase 5 uses this ICB infrastructure** - The interpreter walks the tape at init time and calls the `record_*()` methods to populate the ICB.

---

## Progress Tracker

See `FERRITE_METAL_PROGRESS.md` for detailed phase-by-phase progress.

**Current Status:**
- Phase 1: Foundation ✅ COMPLETE
- Phase 2: Solver Integration ✅ COMPLETE
- Phase 3: Runtime Infrastructure ✅ COMPLETE
- Phase 4: Kernel Library ✅ COMPLETE (18 critical path ops + ICB infrastructure)
- Phase 5: Interpreter Executor 🔄 IN PROGRESS (5.1-5.2 complete, 5.3-5.5 TODO)
- Phase 5.6: Test with Real Model 🔜 NEXT
- Phase 6: Production Readiness 🔜 PLANNED
