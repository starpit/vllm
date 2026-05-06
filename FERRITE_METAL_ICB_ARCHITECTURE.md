# Metal ICB Architecture for Ferrite-Forward

## Key Insight: Buffer Slots vs Buffer Contents

Metal ICB records **buffer slot indices** (0-30), not buffer pointers. This enables dynamic buffer binding:

1. **At init time (ICB recording)**:
   - Walk instruction tape
   - Record dispatches with buffer slot indices
   - Example: `record_rmsnorm(slot_in=0, slot_out=1, slot_weight=2)`

2. **At forward time (ICB execution)**:
   - Create tile table (Metal buffers)
   - Bind tile buffers to encoder slots: `encoder.setBuffer(tiles[0], index: 0)`
   - Execute ICB: `encoder.executeCommandsInBuffer(icb)`
   - ICB reads from bound buffers

## Tile Table = Runtime Buffer Binding

```rust
// At forward time:
let mut tiles: Vec<Option<MetalBuffer>> = vec![None; num_slots];

// Bind all tile buffers to encoder
for (slot, buffer) in tiles.iter().enumerate() {
    if let Some(buf) = buffer {
        encoder.setBuffer(buf, offset: 0, index: slot);
    }
}

// Execute pre-recorded ICB
icb.execute_on_encoder(encoder, 0..icb.command_count());
```

## No Re-recording Needed

ICB is recorded ONCE at init with slot indices. Forward pass only:
1. Allocates/updates tile buffers
2. Binds buffers to encoder slots
3. Executes ICB

This is exactly parallel to CUDA's `run()`:
- CUDA: `tiles[slot]` → kernel reads/writes
- Metal: `encoder.setBuffer(tiles[slot], index: slot)` → ICB reads/writes

## Implementation Plan

1. Record ICB at init with slot indices (not buffer pointers)
2. At forward: allocate tile buffers, bind to encoder, execute ICB
3. Tile table is `Vec<Option<metal::Buffer>>` instead of `Vec<Option<TileEntry>>`
