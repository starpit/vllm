# M3 ICB Test - Quick Start Guide

## TL;DR

Run this on your M3 Mac:
```bash
cd /Users/nickm/git/vllm/.claude/worktrees/ferrite-metal
./run_m3_icb_tests.sh
```

Share the output with the team.

## What This Tests

Metal Compute Indirect Command Buffers (ICB) don't work on M1 Max (Apple7 GPU). We need to know if they work on M3 (Apple9 GPU) to decide Phase 4.6 implementation approach.

## Possible Outcomes

### ✅ Both tests PASS
- Compute ICBs work on M3 but not M1 Max
- **Decision:** Continue with ICB, add hardware check for Apple9+

### ❌ Both tests FAIL  
- Compute ICBs don't work on any Apple Silicon
- **Decision:** Switch to direct encoder recording (simpler, works everywhere)

### ⚠️ Mixed results
- See `PHASE4_ICB_M3_TEST_HANDOFF.md` for detailed interpretation

## Why This Matters

Apple's official ICB sample only shows **render ICBs** (drawing), not **compute ICBs** (our use case). This suggests compute ICBs may not be fully supported.

## Next Steps After Testing

1. Share test output
2. Review `PHASE4_ICB_M3_TEST_HANDOFF.md` for decision tree
3. Implement chosen approach for Phase 4.6

## Files

- `run_m3_icb_tests.sh` - Run this script
- `PHASE4_ICB_M3_TEST_HANDOFF.md` - Complete context and decision tree
- `README_M3_ICB_TEST.md` - Detailed test documentation
- `test_m3_compute_icb.m` - Test source (standard approach)
- `test_m3_compute_icb_inherit.m` - Test source (our approach)
