# Metal ICB Architecture: inheritBuffers Solution

## Problem
ICB's `set_kernel_buffer()` records actual buffer pointers, not slot indices. This means buffers must be known at ICB recording time, but ferrite-forward needs runtime buffers from `ForwardCtx`.

## Solution: inheritBuffers=true

Metal ICB supports `inheritBuffers` flag:
- When `true`: ICB commands inherit buffer bindings from the compute encoder
- When `false`: ICB commands record their own buffer bindings

### Current Implementation (Phase 4.6)
```rust
descriptor.set_inherit_buffers(false);  // Commands record own buffers
```

### Required Change
```rust
descriptor.set_inherit_buffers(true);   // Commands inherit from encoder
```

## New Architecture

### At Init Time (ICB Recording)
```rust
// Record dispatch parameters only - NO buffer bindings
for instr in instructions {
    match instr {
        Instruction::RmsNorm(out, inp, _, _) => {
            // Record threadgroup counts and kernel selection
            // Do NOT call command.set_kernel_buffer()
            recording_ctx.record_compute_dispatch_no_buffers(
                pipeline,
                threadgroups,
                threads_per_threadgroup,
            );
        }
    }
}
```

### At Forward Time (ICB Execution)
```rust
// 1. Allocate tile buffers
let mut tiles: Vec<Option<metal::Buffer>> = vec![None; num_slots];

// 2. Bind ALL buffers to encoder BEFORE executing ICB
for (slot, buffer) in tiles.iter().enumerate() {
    if let Some(buf) = buffer {
        encoder.setBuffer(buf, offset: 0, index: slot);
    }
}

// 3. Set pipeline state on encoder
encoder.setComputePipelineState(pipeline);

// 4. Execute ICB - commands inherit buffers from encoder
encoder.executeCommandsInBuffer(icb, range: 0..command_count);
```

## Implementation Changes Needed

1. **RecordingContext**: Add `record_compute_dispatch_no_buffers()` method
2. **Descriptor**: Change `set_inherit_buffers(false)` → `set_inherit_buffers(true)`
3. **Recording**: Don't call `command.set_kernel_buffer()` during recording
4. **Execution**: Bind all buffers to encoder before executing ICB

## Benefits
- ICB recorded once at init with dispatch parameters only
- Buffers bound dynamically at forward time
- No re-recording needed per forward pass
- Matches CUDA's tile table architecture exactly
