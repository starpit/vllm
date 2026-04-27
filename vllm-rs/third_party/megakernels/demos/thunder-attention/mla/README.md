# MLA Decode Megakernel

This directory contains the ported MLA (Multi-head Latent Attention) decode kernel from ThunderKittens LCF to the modern Megakernels-Private interpreter framework.

## Architecture

The MLA decode operation is split into two main components:

1. **MLA Partial** (`mla_partial.cu`): Computes attention for chunks of the sequence
2. **MLA Reduction** (`mla_reduction.cu`): Performs tree reduction to combine partial results

## Key Features

- **Preserved Instruction Format**: Maintains the original ThunderKittens MLA instruction format for compatibility
- **Modern Interpreter Framework**: Uses the full megakernel interpreter with controller/loader/consumer/storer/launcher warps
- **Page-based Memory Management**: Leverages the sophisticated page allocation system
- **Pipeline Optimization**: Supports instruction pipelining with multiple stages

## File Structure

```
mla/
├── mla.cu              # Main kernel and Python bindings
├── mla.cuh             # Configuration and type definitions  
├── mla_partial.cu      # Partial attention computation
├── mla_reduction.cu    # Tree reduction logic
├── Makefile            # Build configuration
├── python/
│   └── test_mla.py     # Test and validation script
└── README.md           # This file
```

## Building

```bash
cd demos/thunder-attention/mla
make
```

This will create `mla_decode.cpython-*-linux-gnu.so` for Python integration.

## Usage

```python
import mla_decode

# Use the scheduler utility
quality = mla_decode.__get_quality__(next_times, num_processors, num_tokens, seq_length)

# Access 16-head and 8-head kernel variants
kernel_16 = mla_decode.mla_decode
kernel_8 = mla_decode.mla_decode_8_heads
```

## Testing

```bash
cd python
python test_mla.py
```

## Configuration

Key parameters in `mla.cuh`:
- `MLA_QKRot_D = 64`: Rotational dimension
- `MLA_QVO_D = 512`: Value/output dimension  
- `MLA_NUM_ROWS = 32`: Processing block size
- `MLA_PAGE_SIZE = 256`: KV cache page size

## Original vs Modern Framework

| Aspect | Original (LCF) | Modern (Megakernel) |
|--------|----------------|---------------------|
| Architecture | Producer/Consumer | Controller/Loader/Consumer/Storer |
| Memory | Manual shared memory | Page-based virtual memory |
| Sync | Custom barriers | Centralized semaphores |
| Pipeline | Template-specific | Global instruction pipeline |

## Performance

The ported version maintains the original algorithmic complexity while benefiting from:
- Better instruction pipelining
- Improved memory management
- Enhanced load balancing via global work queue
- Sophisticated synchronization primitives

## Limitations

- Currently supports 16 and 8 head variants
- Requires CUDA compute capability 9.0+ (Hopper)
- Page size must be compatible with sequence lengths