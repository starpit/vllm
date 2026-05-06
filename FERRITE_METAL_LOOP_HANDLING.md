# Metal ICB Loop Handling Strategy

## Problem
`Instruction::Loop(count, body_len)` re-runs the next `body_len` instructions `count` times with `ctx.layer_offset` set to the iteration index. ICB cannot encode control flow.

## Solution: Multi-ICB Execution

### At Init Time (Recording)
```rust
match instr {
    Instruction::Loop(count, body_len) => {
        // Mark the start of loop body in ICB
        let loop_start_cmd = recording_ctx.command_count();
        
        // Record body instructions once
        for body_instr in &instructions[i+1 .. i+1+body_len] {
            // Record each body instruction
            record_instruction(body_instr, &mut recording_ctx, ...);
        }
        
        let loop_end_cmd = recording_ctx.command_count();
        
        // Store loop metadata for forward time
        loop_regions.push(LoopRegion {
            start_cmd: loop_start_cmd,
            end_cmd: loop_end_cmd,
            count,
        });
        
        // Skip body instructions in main loop
        i += body_len;
    }
}
```

### At Forward Time (Execution)
```rust
// Execute pre-loop commands
encoder.executeCommandsInBuffer(icb, range: 0..loop_start);

// Execute loop body multiple times
for layer in 0..count {
    // Update layer_offset in weight buffers or via push constants
    update_layer_offset(layer);
    
    // Execute loop body commands
    encoder.executeCommandsInBuffer(icb, range: loop_start..loop_end);
}

// Execute post-loop commands
encoder.executeCommandsInBuffer(icb, range: loop_end..total_commands);
```

## Implementation Details

1. **Loop metadata storage**: `Vec<LoopRegion>` in `MetalExecutor`
2. **Layer offset**: Pass via push constants or update weight buffer bindings per iteration
3. **Weight access**: `wt_fn(&weights, layer_offset)` → bind different weight buffer per iteration

## Alternative: Flatten Loops at Init
Record loop body `count` times with different weight bindings. Simpler but uses more ICB space.
