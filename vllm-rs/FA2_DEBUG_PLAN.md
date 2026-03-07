# Paged FA2 Debug Plan — RESOLVED

**Fixed in commit `013cba535`.** See `HANDOFF.md` for full details.

## Root Cause

The FFI shim called the standard FA2 kernel (`compute_attn_1rowblock`) which has NO `block_table` support. Only the splitkv kernel (`compute_attn_1rowblock_splitkv`) handles paged KV. Fix: `force_split_kernel=true` when `block_table != nullptr`.

## How it was found

1. Wrote unit test comparing contiguous `[0,1,2]` vs shuffled `[1,2,0]` block tables → `max_diff=0.109`
2. Ran same test in Python vLLM → `max_diff=0.000` (proves kernel is fine, bug is in our shim)
3. Read upstream `csrc/flash_attn/flash_api.cpp` — discovered `seqlenq_ngroups_swapped` always routes paged GQA decode through splitkv
4. Inspected `flash_fwd_kernel.h` — confirmed `compute_attn_1rowblock` has zero `block_table` references, all paging logic is in `compute_attn_1rowblock_splitkv`
5. Added `force_split_kernel` flag to shim → `max_diff=0.000000` across all configs
