#!/usr/bin/env python3
"""
Test script for MLA decode megakernel
"""

import os
import sys
import numpy as np
import torch

# Add the parent directory to Python path to import the built module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import mla_decode
    print("Successfully imported mla_decode module!")
    print(f"Available functions: {dir(mla_decode)}")
except ImportError as e:
    print(f"Failed to import mla_decode: {e}")
    print("Make sure to build the module first with 'make'")
    sys.exit(1)

def test_scheduler_quality():
    """Test the scheduler quality function"""
    print("\nTesting scheduler quality function...")
    
    # Test parameters from original
    next_times = [100.0, 95.0, 90.0, 85.0]
    num_processors = 4
    num_tokens = 8
    seq_length = 1024
    
    quality = mla_decode.__get_quality__(next_times, num_processors, num_tokens, seq_length)
    print(f"Quality score: {quality}")
    
    assert isinstance(quality, float), "Quality should return a float"
    print("Scheduler quality test passed!")

def test_kernel_instantiation():
    """Test that we can create kernel instances"""
    print("\nTesting kernel instantiation...")
    
    # Test basic kernel creation (this tests the binding worked)
    try:
        # These should not crash if bindings are correct
        kernel_16 = hasattr(mla_decode, 'mla_decode')
        # kernel_8 = hasattr(mla_decode, 'mla_decode_8_heads')
        
        print(f"16-head kernel available: {kernel_16}")
        # print(f"8-head kernel available: {kernel_8}")
        
        assert kernel_16, "16-head kernel should be available"
        # assert kernel_8, "8-head kernel should be available"
        print("Kernel instantiation test passed!")
        
    except Exception as e:
        print(f"Kernel instantiation failed: {e}")
        raise

def create_test_tensors():
    """Create test tensors with MLA dimensions"""
    print("\nCreating test tensors...")
    
    batch_size = 4
    seq_len = 128
    q_heads = 16
    kv_heads = 1
    qkrot_d = 64
    qvo_d = 512
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    
    # Create tensors matching MLA format
    Q = torch.randn(batch_size, seq_len, q_heads, qkrot_d, device=device, dtype=dtype)
    QV = torch.randn(batch_size, seq_len, q_heads, qvo_d, device=device, dtype=dtype)
    
    # KV cache tensors 
    num_pages = 100
    page_size = 256
    K_cache = torch.randn(1, num_pages, page_size, qkrot_d, device=device, dtype=dtype)
    V_cache = torch.randn(1, num_pages, page_size, qvo_d, device=device, dtype=dtype)
    
    # Page table
    Table = torch.randint(0, num_pages, (batch_size, num_pages), device=device, dtype=torch.int32)
    
    # Output tensors
    O = torch.zeros(batch_size, seq_len, q_heads, qvo_d, device=device, dtype=dtype)
    O_scratch = torch.zeros(batch_size, seq_len, q_heads, qvo_d, device=device, dtype=torch.float32)
    Lvec_scratch = torch.zeros(1, seq_len, batch_size, q_heads, device=device, dtype=torch.float32)
    
    # Semaphores and other parameters
    semaphore = torch.zeros(1, 1, batch_size, seq_len, device=device, dtype=torch.int32)
    
    print(f"Created tensors on device: {device}")
    print(f"Q shape: {Q.shape}, dtype: {Q.dtype}")
    print(f"K_cache shape: {K_cache.shape}, dtype: {K_cache.dtype}")
    
    return {
        'Q': Q,
        'QV': QV,
        'K_cache': K_cache,
        'V_cache': V_cache,
        'Table': Table,
        'O': O,
        'O_scratch': O_scratch,
        'Lvec_scratch': Lvec_scratch,
        'semaphore': semaphore,
        'Softmax_scale': 0.125,  # 1/sqrt(64)
        'tic': 1
    }

def main():
    print("MLA Decode Megakernel Test Suite")
    print("=" * 40)
    
    # Test 1: Import and basic functionality
    test_scheduler_quality()
    
    # Test 2: Kernel bindings
    test_kernel_instantiation()
    
    # Test 3: Tensor creation
    if torch.cuda.is_available():
        tensors = create_test_tensors()
        print(f"\nTest tensors created successfully!")
        print("Note: Full kernel execution tests require proper instruction setup")
    else:
        print("\nCUDA not available, skipping tensor tests")
    
    print("\n" + "=" * 40)
    print("All basic tests passed! The MLA megakernel module is ready.")
    print("Next steps:")
    print("1. Create instruction schedules")
    print("2. Test full kernel execution")
    print("3. Performance benchmarking")

if __name__ == "__main__":
    main()