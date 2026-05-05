# M3 Compute ICB Test Instructions

## Overview

These tests will determine if Metal Compute Indirect Command Buffers work on Apple M3 (Apple9 GPU family).

## Test Files

1. **test_m3_compute_icb.m** - Tests ICB with `inheritPipelineState=NO`
   - Pipeline state set on ICB command itself
   - This is the "standard" ICB approach

2. **test_m3_compute_icb_inherit.m** - Tests ICB with `inheritPipelineState=YES`
   - Pipeline state set on encoder, inherited by ICB commands
   - This is the approach we used in our Rust implementation

## How to Run

```bash
# Compile test 1
clang -framework Metal -framework Foundation test_m3_compute_icb.m -o test_m3_icb

# Run test 1
./test_m3_icb

# Compile test 2
clang -framework Metal -framework Foundation test_m3_compute_icb_inherit.m -o test_m3_icb_inherit

# Run test 2
./test_m3_icb_inherit
```

## Expected Results

### If Compute ICBs Work on M3:
```
Testing on: Apple M3
GPU Family: Apple9

=== Encoding ICB command from CPU ===
ICB command encoded:
  - Pipeline: 0x...
  - Buffer: 0x...
  - Dispatch: 1 threadgroup, 4 threads

=== Executing ICB ===

=== Results ===
Output buffer: [42, 42, 42, 42]

✅ SUCCESS: ICB executed correctly!
Compute ICBs (ConcurrentDispatch) ARE supported on this GPU.
```

### If Compute ICBs Don't Work on M3:
```
Testing on: Apple M3
GPU Family: Apple9

=== Results ===
Output buffer: [0, 0, 0, 0]

❌ FAILURE: ICB did not execute.
Compute ICBs (ConcurrentDispatch) are NOT supported on this GPU.

=== Testing direct dispatch (baseline) ===
Direct dispatch output: [42, 42, 42, 42]
✅ Direct dispatch works (GPU and shader are functional)
This confirms the issue is specifically with compute ICBs.
```

## What This Tells Us

### Scenario 1: Both tests PASS on M3
- Compute ICBs work on Apple9 (M3) but not Apple7 (M1 Max)
- **Conclusion:** Hardware limitation on older Apple Silicon
- **Action:** Document minimum GPU family requirement (Apple9+)

### Scenario 2: Test 1 FAILS, Test 2 PASSES on M3
- Only `inheritPipelineState=YES` works
- **Conclusion:** Our Rust implementation approach is correct
- **Action:** Investigate why it doesn't work on M1 Max

### Scenario 3: Both tests FAIL on M3
- Compute ICBs don't work on any Apple Silicon
- **Conclusion:** Metal limitation or CPU-side encoding not supported
- **Action:** Use direct encoder recording for Phase 4.6

### Scenario 4: Test 1 PASSES, Test 2 FAILS on M3
- Only `inheritPipelineState=NO` works
- **Conclusion:** Our Rust implementation needs to change approach
- **Action:** Update to set pipeline on ICB commands

## Current Status on M1 Max

Both configurations FAIL on M1 Max (Apple7):
- ❌ `inheritPipelineState=NO` - Crashes on `setComputePipelineState`
- ❌ `inheritPipelineState=YES` - No crash, but produces zeros (not executed)
- ✅ Direct dispatch works perfectly

## Next Steps After M3 Testing

Please run both tests and report:
1. GPU family detected (should be Apple9 for M3)
2. Output from test 1 (inheritPipelineState=NO)
3. Output from test 2 (inheritPipelineState=YES)

This will help us determine if:
- We should continue with ICB approach (if M3 works)
- We need to use direct encoder recording (if M3 also fails)
- There's a bug in our implementation (if M3 works but M1 Max doesn't)
