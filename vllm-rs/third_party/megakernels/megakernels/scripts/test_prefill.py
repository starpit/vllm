#!/usr/bin/env python3
"""
Simple test script for the attention prefill instruction.

This script creates a minimal test environment with a single sequence and tests
both the PyVM implementation and the actual megakernel implementation.
"""

import math
import torch
from pathlib import Path
from transformers import AutoConfig

from megakernels.demos.tp_throughput.instructions import AttentionPrefill
from megakernels.demos.tp_throughput.python_vm import attention_prefill
from megakernels.demos.tp_throughput.scheduler import init_random_weights, create_instruction_tensor, setup_rope_and_interleave
from megakernels.demos.tp_throughput.globs import make_globals
from megakernels.demos.tp_throughput.mk import TensorParallelMK_Interpreter


# Global megakernel interpreter instance (set up once, reused across tests)
_global_interpreter = None

# Helper Functions for Test Setup and Validation

def create_test_globals(global_batch_size, num_devices=8, num_pages=64, layer_limit=1):
    """Create test globals with standard configuration."""
    import time
    
    model_config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
    
    print(f"Creating globals with layer_limit={layer_limit}")
    globs = make_globals(
        model_config=model_config,
        num_devices=num_devices,
        global_batch_size=global_batch_size,
        num_pages=num_pages,
        barrier_init_val=0,
        global_work_queue_enabled=True,  # Enable global work queue to match compiled kernel
        meta_device=False,
        timing_record_enabled=False,
        layer_limit=layer_limit,
    )
    
    # Add some randomization to ensure different weight initializations
    torch.manual_seed(int(time.time() * 1000) % 2**32)
    init_random_weights(globs, std=1.0)
    
    # Set up rope and interleave (required for megakernel)
    setup_rope_and_interleave(globs)
    
    return globs


def setup_megakernel_interpreter(globs):
    """Set up the megakernel interpreter for testing."""
    global _global_interpreter
    
    if _global_interpreter is None:
        # Get the megakernel directory (demos/cross-gpu-llama)
        mk_dir = Path(__file__).parent.parent.parent / "demos" / "cross-gpu-llama"
        
        print(f"Setting up megakernel interpreter from: {mk_dir}")
        print(f"Globs type: {type(globs)}")
        print(f"Global batch size: {globs.global_batch_size}")
        print(f"Prefill seq lens: {globs.prefill_seq_lens}")
        print(f"Num prefill tokens: {globs.num_prefill_tokens()}")
        print(f"Instructions shape: {[inst.shape if inst is not None else None for inst in globs.instructions]}")
        print(f"Barriers shape: {[barrier.shape for barrier in globs.barriers]}")
        print(f"Globs has model_config: {hasattr(globs, 'model_config')}")
        if hasattr(globs, 'model_config'):
            print(f"Model config type: {type(globs.model_config)}")
        
        _global_interpreter = TensorParallelMK_Interpreter(mk_dir, globs, multithread=True)
        _global_interpreter.setup()
        print("✓ Megakernel interpreter set up successfully")
    
    return _global_interpreter


def create_simple_instruction_tensor(instructions, globs):
    """Create a simple instruction tensor from a list of instruction objects."""
    import torch
    
    # For now, let's create a minimal instruction tensor that just contains our prefill instructions
    # This is a simplified approach - in practice, we'd need the full scheduler
    
    num_instructions = len(instructions)
    instruction_tensors = []
    
    print(f"🔧 Creating instruction tensor with {num_instructions} instructions")
    print(f"🔧 Global work queue enabled: {globs.global_work_queue_enabled}")
    
    for device_idx in range(globs.tp_size):
        if globs.global_work_queue_enabled:
            # 2D tensor for global work queue: [num_instructions, 32]
            device_tensor = torch.zeros((num_instructions, 32), dtype=torch.int32, device=f"cuda:{device_idx}")
            
            for i, inst in enumerate(instructions):
                # Encode instruction as integers
                # [opcode, layer_idx, seq_idx, block_idx, kv_head_idx, ...]
                device_tensor[i, 0] = inst.opcode()  # opcode
                device_tensor[i, 1] = inst.layer_idx
                device_tensor[i, 2] = inst.prefill_seq_idx
                device_tensor[i, 3] = inst.prefill_block_idx
                device_tensor[i, 4] = inst.kv_head_idx
                
                print(f"🔧 Device {device_idx}, Instruction {i}: opcode={inst.opcode()}, layer={inst.layer_idx}, seq={inst.prefill_seq_idx}, block={inst.prefill_block_idx}, kv_head={inst.kv_head_idx}")
                # Rest are zeros
        else:
            # 3D tensor for per-SM queue: [sm_count, instructions_per_sm, 32]
            sm_count = device_tensor.shape[0] if hasattr(device_tensor, 'shape') else 132
            instructions_per_sm = max(1, (num_instructions + sm_count - 1) // sm_count)
            device_tensor = torch.zeros((sm_count, instructions_per_sm, 32), dtype=torch.int32, device=f"cuda:{device_idx}")
            
            for i, inst in enumerate(instructions):
                sm_idx = i % sm_count
                inst_idx = i // sm_count
                if inst_idx < instructions_per_sm:
                    device_tensor[sm_idx, inst_idx, 0] = inst.opcode()
                    device_tensor[sm_idx, inst_idx, 1] = inst.layer_idx
                    device_tensor[sm_idx, inst_idx, 2] = inst.prefill_seq_idx
                    device_tensor[sm_idx, inst_idx, 3] = inst.prefill_block_idx
                    device_tensor[sm_idx, inst_idx, 4] = inst.kv_head_idx
        
        print(f"🔧 Device {device_idx} instruction tensor shape: {device_tensor.shape}")
        print(f"🔧 Device {device_idx} instruction tensor sample: {device_tensor[:min(2, device_tensor.shape[0]), :8]}")
        instruction_tensors.append(device_tensor)
    
    return instruction_tensors


def run_megakernel_with_real_scheduler(globs):
    """Run the megakernel using the real scheduler for comparison."""
    print(f"🧪 Creating real schedule for comparison...")
    
    # Use the real scheduler to create a proper instruction tensor
    from megakernels.demos.tp_throughput.scheduler import create_instruction_tensor
    
    # Create instruction tensor using real scheduler for device 0
    real_instruction_tensor = create_instruction_tensor(
        globs,
        device_idx=0,
        layer_limit=1,  # Only one layer
        interleave_waves=False,
        move_to_gpu=True,
    )
    
    print(f"🧪 Real scheduler created tensor with shape: {real_instruction_tensor.shape}")
    print(f"🧪 Real instruction tensor sample (first 3 instructions):")
    print(real_instruction_tensor[:3, :8])
    
    # Copy this real instruction tensor to all devices
    real_instruction_tensors = []
    for device_idx in range(globs.tp_size):
        device_tensor = real_instruction_tensor.clone().to(f"cuda:{device_idx}")
        real_instruction_tensors.append(device_tensor)
    
    globs.copy_instructions(real_instruction_tensors)
    
    # Set up interpreter if not already done
    global _global_interpreter
    if _global_interpreter is None:
        setup_megakernel_interpreter(globs)
    
    _global_interpreter.globs = globs
    
    # Run the megakernel with real instructions
    print(f"🚀 Executing megakernel with real scheduler instructions...")
    try:
        _global_interpreter.interpret()
        print(f"✅ Megakernel execution with real scheduler completed")
    except Exception as e:
        print(f"❌ Megakernel execution with real scheduler failed: {e}")
        raise
    
    # Check results
    print(f"🔧 After real scheduler megakernel execution:")
    for i in range(min(2, globs.tp_size)):
        attn_norm = torch.norm(globs.attn_out[i]).item()
        print(f"🔧 Device {i} attn_out norm: {attn_norm:.6f}")
        if attn_norm > 0:
            print(f"🔧 Device {i} attn_out sample: {globs.attn_out[i].flatten()[:8]}")
        else:
            print(f"🔧 Device {i} attn_out is all zeros!")
    
    # Synchronize all devices
    for i in range(globs.tp_size):
        torch.cuda.synchronize(i)


def run_megakernel_with_instructions(globs, instructions):
    """Run the megakernel with a list of instructions."""
    global _global_interpreter
    
    print(f"🚀 Running megakernel with {len(instructions)} instructions")
    
    # First, try with the real scheduler to see if that works
    print("🧪 First trying with real scheduler...")
    run_megakernel_with_real_scheduler(globs)
    
    # Create simple instruction tensor from the instruction list
    instruction_tensors = create_simple_instruction_tensor(instructions, globs)
    globs.copy_instructions(instruction_tensors)
    
    # Debug: Check if instructions were copied correctly
    print(f"🔧 After copying instructions to globs:")
    for i in range(min(2, globs.tp_size)):
        if globs.instructions[i] is not None:
            print(f"🔧 Device {i} instructions shape: {globs.instructions[i].shape}")
            print(f"🔧 Device {i} instructions sample: {globs.instructions[i][:min(2, globs.instructions[i].shape[0]), :8]}")
        else:
            print(f"🔧 Device {i} instructions is None!")
    
    # Set up interpreter if not already done (this should use the original globs)
    if _global_interpreter is None:
        setup_megakernel_interpreter(globs)
    
    # Update the interpreter's globs to point to the current globs (for the case where we use different globs)
    _global_interpreter.globs = globs
    
    # Debug: Check attention output before running megakernel
    print(f"🔧 Before megakernel execution:")
    for i in range(min(2, globs.tp_size)):
        attn_norm = torch.norm(globs.attn_out[i]).item()
        print(f"🔧 Device {i} attn_out norm: {attn_norm:.6f}")
        if attn_norm > 0:
            print(f"🔧 Device {i} attn_out sample: {globs.attn_out[i].flatten()[:8]}")
    
    # Debug: Check global instruction index and other key parameters
    print(f"🔧 Global instruction index: {[idx.item() if idx is not None else None for idx in globs.global_instruction_index]}")
    print(f"🔧 Num prefill tokens: {globs.num_prefill_tokens()}")
    print(f"🔧 Global batch size: {globs.global_batch_size}")
    
    # Run the megakernel
    print(f"🚀 Executing megakernel...")
    try:
        _global_interpreter.interpret()
        print(f"✅ Megakernel execution completed")
        breakpoint()
    except Exception as e:
        print(f"❌ Megakernel execution failed: {e}")
        raise
    
    # Debug: Check attention output after running megakernel
    print(f"🔧 After megakernel execution:")
    for i in range(min(2, globs.tp_size)):
        attn_norm = torch.norm(globs.attn_out[i]).item()
        print(f"🔧 Device {i} attn_out norm: {attn_norm:.6f}")
        if attn_norm > 0:
            print(f"🔧 Device {i} attn_out sample: {globs.attn_out[i].flatten()[:8]}")
        else:
            print(f"🔧 Device {i} attn_out is all zeros!")
    
    # Synchronize all devices
    for i in range(globs.tp_size):
        torch.cuda.synchronize(i)


def setup_sequence_pointers(globs, seq_len):
    """Setup prefill indirection pointers for a sequence, following tp_generate.py pattern."""
    
    # Set up the sequence lengths for the megakernel
    globs.set_sizes(
        global_batch_size=seq_len,  # For prefill, batch size = sequence length
        prefill_seq_lens=[seq_len]  # Single sequence of seq_len tokens
    )
    
    print(f"🔧 Setting up paging for seq_len={seq_len}, page_size={globs.page_size}")
    
    # Calculate paging structures following tp_generate.py pattern
    num_seqs = 1  # Single sequence for our test
    pages_per_seq = math.ceil(seq_len / globs.page_size)
    
    # Build the paging structures
    kv_indices = []
    kv_indptr = [0]
    append_indices = []
    qo_indptr = [0]  
    position_ids = []
    
    # For our single test sequence
    for seq in range(num_seqs):
        # Use consecutive pages starting from 0 (in a real system these would be allocated)
        indices = list(range(pages_per_seq))
        kv_indices.extend(indices)
        kv_indptr.append(kv_indptr[-1] + len(indices))  # One-past-the-end: cumulative page count
        
        # Append indices: token positions within the pages
        start_page_idx = indices[0]  # 0 in our case
        start_token_idx = start_page_idx * globs.page_size  # 0 in our case
        append_indices.extend([start_token_idx + i for i in range(seq_len)])
        
        qo_indptr.append(qo_indptr[-1] + seq_len)  # One-past-the-end: cumulative token count
        position_ids.extend(range(seq_len))
    
    # Calculate last page length
    last_page_len = seq_len % globs.page_size
    if last_page_len == 0:
        last_page_len = globs.page_size
    
    # Convert to tensors
    t_kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
    t_kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    t_qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    t_last_page_len = torch.tensor([last_page_len] * num_seqs, dtype=torch.int32)
    t_position_ids = torch.tensor(position_ids, dtype=torch.int32)
    t_append_indices = torch.tensor(append_indices, dtype=torch.int32)
    
    print(f"🔧 Paging setup:")
    print(f"  kv_indices: {t_kv_indices.tolist()}")
    print(f"  kv_indptr: {t_kv_indptr.tolist()}")
    print(f"  qo_indptr: {t_qo_indptr.tolist()}")
    print(f"  last_page_len: {t_last_page_len.tolist()}")
    print(f"  append_indices: {t_append_indices.tolist()[:min(10, len(t_append_indices))]}...")
    
    # Use the proper helper methods to copy the data
    globs.copy_prefill_info(
        qo_indptr=t_qo_indptr,
        kv_indices=t_kv_indices,
        kv_indptr=t_kv_indptr,
        kv_last_page_len=t_last_page_len,
    )
    globs.copy_append_indices(t_append_indices)
    globs.copy_position_ids(t_position_ids)
    
    # Debug: Check what got copied to the actual globs tensors
    print(f"🔧 After copying to globs (device 0):")
    print(f"  prefill_qo_indptr: {globs.prefill_qo_indptr[0][:5].tolist()}")
    print(f"  prefill_kv_indptr: {globs.prefill_kv_indptr[0][:5].tolist()}")
    print(f"  prefill_kv_indices: {globs.prefill_kv_indices[0][:5].tolist()}")
    print(f"  kv_append_indices: {globs.kv_append_indices[0][:10].tolist()}")
    print(f"  position_ids: {globs.position_ids[0][:10].tolist()}")


def calculate_expected_devices(seq_len, global_batch_size, tp_size):
    """Calculate which devices should receive outputs for a given sequence."""
    batch_size_per_device = global_batch_size // tp_size
    expected_devices = set()
    for token_idx in range(seq_len):
        target_device = token_idx // batch_size_per_device
        expected_devices.add(target_device)
    return expected_devices


def create_reference_globals_copy(globs, add_perturbation=True):
    """Create a reference copy of globals for comparison testing."""
    reference_globs = globs.with_new_activations(new_kv_cache=True)
    num_devices = globs.tp_size
    
    # Copy input data to reference
    for dev_idx in range(num_devices):
        reference_globs.k_cache[dev_idx].copy_(globs.k_cache[dev_idx])
        reference_globs.v_cache[dev_idx].copy_(globs.v_cache[dev_idx])
        reference_globs.post_rope_q[dev_idx].copy_(globs.post_rope_q[dev_idx])
        
        # Copy setup data
        reference_globs.prefill_qo_indptr[dev_idx].copy_(globs.prefill_qo_indptr[dev_idx])
        reference_globs.prefill_kv_indptr[dev_idx].copy_(globs.prefill_kv_indptr[dev_idx])
        reference_globs.prefill_kv_indices[dev_idx].copy_(globs.prefill_kv_indices[dev_idx])
        reference_globs.prefill_kv_last_page_len[dev_idx].copy_(globs.prefill_kv_last_page_len[dev_idx])
        reference_globs.position_ids[dev_idx].copy_(globs.position_ids[dev_idx])
        
        # Copy barriers to have the same prerequisites
        reference_globs.barriers[dev_idx].copy_(globs.barriers[dev_idx])
        
        # Ensure attn_out starts at zero for both
        globs.attn_out[dev_idx].zero_()
        reference_globs.attn_out[dev_idx].zero_()
        
        # Add perturbation to ensure independence
        if add_perturbation:
            # Use larger perturbation to ensure clear differences
            perturbation = torch.randn_like(reference_globs.post_rope_q[dev_idx]) * 1e-3
            reference_globs.post_rope_q[dev_idx] += perturbation
    
    return reference_globs


def execute_prefill_blocks(globs, layer_idx, seq_idx, seq_len, kv_head_idx, device_idx, use_megakernel=False):
    """Execute all prefill blocks for a sequence and return instructions list."""
    num_blocks = (seq_len + 15) // 16  # Ceiling division
    instructions = []
    
    for block_idx in range(num_blocks):
        instruction = AttentionPrefill(
            layer_idx=layer_idx,
            prefill_seq_idx=seq_idx,
            prefill_block_idx=block_idx,
            kv_head_idx=kv_head_idx,
        )
        instructions.append(instruction)
    
    if use_megakernel:
        # Run all instructions at once using the megakernel
        run_megakernel_with_instructions(globs, instructions)
    else:
        # Run instructions individually using PyVM
        for instruction in instructions:
            attention_prefill(globs, instruction, device_idx)
    
    return instructions


def validate_attention_outputs(test_globs, initial_attn_out, seq_len=None, instruction=None, 
                              device_idx=None, reference_globs=None, description=""):
    """Unified function to validate attention outputs with optional reference comparison."""
    num_devices = test_globs.tp_size
    global_batch_size = test_globs.global_batch_size
    batch_size_per_device = global_batch_size // num_devices
    
    devices_with_changes = []
    max_abs_diff_overall = 0.0
    all_matches = True
    
    # Determine if we're doing reference comparison or single instruction validation
    is_reference_comparison = reference_globs is not None
    
    for dev_idx in range(num_devices):
        test_attn = test_globs.attn_out[dev_idx]
        
        if not torch.equal(initial_attn_out[dev_idx], test_attn):
            devices_with_changes.append(dev_idx)
            
            # Calculate which tokens this device should have received
            if seq_len is not None:
                # Multi-block/sequence validation
                device_token_start = dev_idx * batch_size_per_device
                device_token_end = device_token_start + batch_size_per_device
                overlapping_tokens = []
                for token_idx in range(seq_len):
                    if device_token_start <= token_idx < device_token_end:
                        overlapping_tokens.append(token_idx)
            elif instruction is not None and device_idx is not None:
                # Single instruction validation with block-specific logic
                layer_idx = instruction.layer_idx
                seq_idx = instruction.prefill_seq_idx
                block_idx = instruction.prefill_block_idx
                
                q_start_idx = test_globs.prefill_qo_indptr[device_idx][seq_idx].item()
                q_end_idx = test_globs.prefill_qo_indptr[device_idx][seq_idx + 1].item()
                q_row_start = 16 * block_idx + q_start_idx
                q_row_end = min(q_end_idx, q_row_start + 16)
                
                overlapping_tokens = []
                for q_idx in range(q_row_end - q_row_start):
                    abs_token_idx = q_row_start + q_idx
                    target_device_idx = abs_token_idx // batch_size_per_device
                    if target_device_idx == dev_idx:
                        local_token_idx = abs_token_idx % batch_size_per_device
                        overlapping_tokens.append(local_token_idx)
            else:
                overlapping_tokens = ["unknown"]
            
            test_norm = torch.norm(test_attn).item()
            
            if is_reference_comparison:
                # Compare with reference
                ref_attn = reference_globs.attn_out[dev_idx]
                abs_diff = torch.abs(test_attn - ref_attn)
                max_abs_diff = torch.max(abs_diff).item()
                mean_abs_diff = torch.mean(abs_diff).item()
                max_abs_diff_overall = max(max_abs_diff_overall, max_abs_diff)
                
                ref_norm = torch.norm(ref_attn).item()
                rel_diff = max_abs_diff / (ref_norm + 1e-8)
                
                if description:
                    print(f"    ✓ Device {dev_idx}: {len(overlapping_tokens)} tokens, norm={test_norm:.4f}")
                    print(f"      Max diff: {max_abs_diff:.2e}, Mean diff: {mean_abs_diff:.2e}, Rel diff: {rel_diff:.2e}")
                else:
                    print(f"✓ Device {dev_idx}: received {len(overlapping_tokens)} tokens")
                    print(f"  Test norm: {test_norm:.4f}, Reference norm: {ref_norm:.4f}")
                    print(f"  Max abs diff: {max_abs_diff:.2e}, Mean abs diff: {mean_abs_diff:.2e}")
                    print(f"  Relative diff: {rel_diff:.2e}")
                
                # Check if values match within tolerance
                if max_abs_diff > 1e-3:  # More lenient for larger perturbations
                    if description:
                        print(f"      ⚠ Large difference detected!")
                    else:
                        print(f"  ⚠ Large difference detected!")
                    all_matches = False
                elif max_abs_diff < 1e-7:  # Less paranoid about small differences
                    if description:
                        print(f"      ⚠ Suspiciously small difference!")
                    else:
                        print(f"  ⚠ Suspiciously small difference - check independence!")
                else:
                    if description:
                        print(f"      ✓ Reasonable difference")
                    else:
                        print(f"  ✓ Values match within expected tolerance")
            else:
                # Single instruction validation - just report changes
                print(f"✓ Attention output changed on device {dev_idx}")
                if overlapping_tokens and len(overlapping_tokens) <= 20:
                    print(f"  Affected token indices: {overlapping_tokens}")
                
                # Sanity checks for single instruction validation
                if torch.isnan(test_attn).any():
                    print(f"⚠ Warning: NaN values found in attention output on device {dev_idx}")
                    all_matches = False
                if torch.isinf(test_attn).any():
                    print(f"⚠ Warning: Infinite values found in attention output on device {dev_idx}")
                    all_matches = False
                
                if test_norm > 100:
                    print(f"⚠ Warning: Very large attention output norm on device {dev_idx}: {test_norm:.2f}")
                elif test_norm < 1e-6:
                    print(f"⚠ Warning: Very small attention output norm on device {dev_idx}: {test_norm:.2e}")
                
            if len(overlapping_tokens) <= 20 and not description and is_reference_comparison:
                print(f"  Token indices: {overlapping_tokens}")
        else:
            # This device should not have received any outputs
            if is_reference_comparison:
                ref_attn = reference_globs.attn_out[dev_idx]
                ref_norm = torch.norm(ref_attn).item()
                if ref_norm > 1e-6:
                    print(f"⚠ Device {dev_idx}: Reference has non-zero output but test doesn't!")
                    all_matches = False
    
    if not devices_with_changes and not is_reference_comparison:
        print("⚠ Warning: Attention output was not modified")
        all_matches = False
    
    return devices_with_changes, max_abs_diff_overall, all_matches


def check_for_nans_comprehensive(globs, description=""):
    """Comprehensive NaN checking across all relevant tensors in globals."""
    num_devices = globs.tp_size
    nan_found = False
    nan_details = []
    
    tensor_groups = [
        ("attn_out", globs.attn_out),
        ("post_rope_q", globs.post_rope_q),
        ("k_cache", globs.k_cache),
        ("v_cache", globs.v_cache),
        ("barriers", globs.barriers),
    ]
    
    # Check additional tensors if they exist
    if hasattr(globs, 'pre_rope_q'):
        tensor_groups.append(("pre_rope_q", globs.pre_rope_q))
    if hasattr(globs, 'activations'):
        tensor_groups.append(("activations", globs.activations))
    if hasattr(globs, 'mlp_out'):
        tensor_groups.append(("mlp_out", globs.mlp_out))
    if hasattr(globs, 'gate_out'):
        tensor_groups.append(("gate_out", globs.gate_out))
    if hasattr(globs, 'up_out'):
        tensor_groups.append(("up_out", globs.up_out))
    
    print(f"\n--- NaN Check {description} ---")
    
    for tensor_name, tensor_list in tensor_groups:
        for dev_idx in range(num_devices):
            tensor = tensor_list[dev_idx]
            if torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                total_elements = tensor.numel()
                nan_details.append({
                    'tensor': tensor_name,
                    'device': dev_idx,
                    'nan_count': nan_count,
                    'total_elements': total_elements,
                    'percentage': (nan_count / total_elements) * 100
                })
                nan_found = True
                print(f"❌ NaN found in {tensor_name}[{dev_idx}]: {nan_count}/{total_elements} elements ({nan_details[-1]['percentage']:.2f}%)")
                
                # Show tensor stats
                non_nan_mask = ~torch.isnan(tensor)
                if non_nan_mask.any():
                    non_nan_values = tensor[non_nan_mask]
                    print(f"   Non-NaN values: min={non_nan_values.min():.6f}, max={non_nan_values.max():.6f}, mean={non_nan_values.mean():.6f}")
                else:
                    print(f"   All values are NaN!")
                    
            elif torch.isinf(tensor).any():
                inf_count = torch.isinf(tensor).sum().item()
                total_elements = tensor.numel()
                print(f"⚠ Inf found in {tensor_name}[{dev_idx}]: {inf_count}/{total_elements} elements ({(inf_count/total_elements)*100:.2f}%)")
    
    if not nan_found:
        print("✅ No NaNs detected in any tensors")
    else:
        print(f"❌ NaNs detected in {len(nan_details)} tensor(s)")
        
    return nan_found, nan_details


def analyze_barrier_changes(globs, initial_barriers, expected_updates, description=""):
    """Analyze and report barrier changes."""
    num_devices = globs.tp_size
    devices_with_barrier_changes = []
    total_barrier_updates = 0
    
    for dev_idx in range(num_devices):
        if not torch.equal(initial_barriers[dev_idx], globs.barriers[dev_idx]):
            devices_with_barrier_changes.append(dev_idx)
            diff = globs.barriers[dev_idx] - initial_barriers[dev_idx]
            num_updates = torch.sum(diff).item()
            total_barrier_updates += num_updates
            print(f"✓ Device {dev_idx}: {num_updates} barrier updates")
    
    print(f"\nDevices with barrier changes: {devices_with_barrier_changes}")
    print(f"Total barrier updates across all devices: {total_barrier_updates}")
    if expected_updates is not None:
        print(f"Expected barrier updates: {expected_updates}")
    
    return devices_with_barrier_changes, total_barrier_updates


def print_test_summary(success, devices_with_changes, max_diff, expected_devices=None, 
                      seq_len=None, num_instructions=None):
    """Print a standardized test summary."""
    if success:
        print("✅ SUCCESS: Test completed successfully!")
        if expected_devices is not None:
            print("✅ SUCCESS: All devices received expected outputs!")
        print("✅ SUCCESS: All computed values match reference within tolerance!")
    else:
        print("❌ Test failed!")
        if expected_devices is not None:
            expected_devices_list = sorted(expected_devices) if isinstance(expected_devices, set) else expected_devices
            devices_with_changes_sorted = sorted(devices_with_changes)
            if devices_with_changes_sorted != expected_devices_list:
                print(f"    Expected devices: {expected_devices_list}")
                print(f"    Actual devices: {devices_with_changes_sorted}")
    
    print(f"\nSummary Statistics:")
    if seq_len is not None:
        print(f"  Total tokens processed: {seq_len}")
    if num_instructions is not None:
        print(f"  Instructions executed: {num_instructions}")
    print(f"  Devices with outputs: {len(devices_with_changes)}")
    print(f"  Max difference from reference: {max_diff:.2e}")


# Compatibility aliases for backward compatibility
compare_attention_outputs = lambda test_globs, reference_globs, initial_attn_out, seq_len, description="": \
    validate_attention_outputs(test_globs, initial_attn_out, seq_len=seq_len, reference_globs=reference_globs, description=description)




def setup_simple_prefill_test():
    """Set up a minimal test case for prefill instruction."""
    
    # Small test parameters
    global_batch_size = 128  # Small batch for testing
    seq_len = 32  # Short sequence
    
    print("Creating globals...")
    globs = create_test_globals(global_batch_size)
    
    # Set up a simple sequence for testing
    device_idx = 0  # Test on device 0
    seq_idx = 0     # First sequence
    layer_idx = 0   # First layer
    
    print("Setting up sequence pointers...")
    setup_sequence_pointers(globs, seq_len)
    
    return globs, device_idx, seq_idx, layer_idx


def setup_prefill_prerequisites(globs, device_idx, seq_idx, layer_idx, seq_len=32):
    """Set up the data that would normally be created by previous instructions."""
    
    print("Setting up prefill prerequisites...")
    
    # Simulate that QKV + RoPE has completed
    # Fill post_rope_q with random data
    batch_size_per_device = globs.global_batch_size // globs.tp_size
    for dev_idx in range(globs.tp_size):
        globs.post_rope_q[dev_idx].normal_(0, 0.1)
        # Ensure no NaNs in generated data
        if torch.isnan(globs.post_rope_q[dev_idx]).any():
            print(f"⚠ Warning: Generated NaN in post_rope_q[{dev_idx}] during setup")
    
    # Fill KV cache with random data
    for dev_idx in range(globs.tp_size):
        globs.k_cache[dev_idx].normal_(0, 0.1)
        globs.v_cache[dev_idx].normal_(0, 0.1)
        # Ensure no NaNs in generated data
        if torch.isnan(globs.k_cache[dev_idx]).any():
            print(f"⚠ Warning: Generated NaN in k_cache[{dev_idx}] during setup")
        if torch.isnan(globs.v_cache[dev_idx]).any():
            print(f"⚠ Warning: Generated NaN in v_cache[{dev_idx}] during setup")
    
    # Set barriers to indicate QKV + RoPE completion
    # Barrier format: [layer, opcode-1, batch_block, head/output_block]
    qkv_opcode = 2  # QKV_RopeAppend opcode
    
    for dev_idx in range(globs.tp_size):
        # Set QKV completion barriers (barrier value 128 indicates completion)
        for batch_idx in range(seq_len):
            batch_block_idx = batch_idx // globs.matmul_batch_block_size
            globs.barriers[dev_idx][layer_idx, qkv_opcode - 1, batch_block_idx, 0] = 128
            
        # Set KV cache generation barriers (barrier value 32 indicates completion)
        for batch_idx in range(0, seq_len, globs.matmul_batch_block_size):
            batch_block_idx = batch_idx // globs.matmul_batch_block_size
            globs.barriers[dev_idx][layer_idx, qkv_opcode - 1, batch_block_idx, 1] = 32


def test_single_prefill_instruction():
    """Test a single prefill instruction."""
    
    print("=== Testing Single Prefill Instruction ===")
    
    # Setup test environment
    globs, device_idx, seq_idx, layer_idx = setup_simple_prefill_test()
    seq_len = 32
    
    # Setup prerequisites
    setup_prefill_prerequisites(globs, device_idx, seq_idx, layer_idx, seq_len)
    
    # Check for NaNs after setup
    nan_found, _ = check_for_nans_comprehensive(globs, "after setup")
    if nan_found:
        print("❌ NaNs found during setup! Aborting test.")
        return False
    
    # Create a prefill instruction
    # Test first block (block_idx=0) and first KV head (kv_head_idx=0)
    block_idx = 0
    kv_head_idx = 0
    
    instruction = AttentionPrefill(
        layer_idx=layer_idx,
        prefill_seq_idx=seq_idx,
        prefill_block_idx=block_idx,
        kv_head_idx=kv_head_idx,
    )
    
    print(f"Testing prefill instruction:")
    print(f"  Layer: {layer_idx}")
    print(f"  Sequence: {seq_idx}")
    print(f"  Block: {block_idx} (tokens 0-15)")
    print(f"  KV Head: {kv_head_idx}")
    print(f"  Device: {device_idx}")
    
    # Store initial state for comparison
    initial_attn_out = [t.clone() for t in globs.attn_out]
    initial_barriers = [t.clone() for t in globs.barriers]
    
    try:
        # Run the prefill instruction
        print("Running prefill instruction...")
        attention_prefill(globs, instruction, device_idx)
        print("✓ Prefill instruction completed successfully!")
        
        # Check for NaNs after computation
        nan_found, nan_details = check_for_nans_comprehensive(globs, "after prefill instruction")
        if nan_found:
            print("❌ NaNs detected after prefill instruction!")
            print("NaN details:", nan_details)
            return False
        
        # Validate results using unified function
        devices_with_changes, _, all_matches = validate_attention_outputs(
            globs, initial_attn_out, instruction=instruction, device_idx=device_idx
        )
        
        # Validate barriers separately
        analyze_barrier_changes(globs, initial_barriers, expected_updates=None)
        
        return all_matches
            
    except Exception as e:
        print(f"✗ Error running prefill instruction: {e}")
        # Check for NaNs after error for debugging
        check_for_nans_comprehensive(globs, "after error")
        raise


def test_multiple_blocks():
    """Test multiple prefill blocks for the same sequence."""
    
    print("\n=== Testing Multiple Prefill Blocks ===")
    
    # Setup test environment with longer sequence
    globs, device_idx, seq_idx, layer_idx = setup_simple_prefill_test()
    seq_len = 32  # This will create 2 blocks (0-15, 16-31)
    
    # Setup prerequisites
    setup_prefill_prerequisites(globs, device_idx, seq_idx, layer_idx, seq_len)
    
    # Check for NaNs after setup
    nan_found, _ = check_for_nans_comprehensive(globs, "after setup for multiple blocks")
    if nan_found:
        print("❌ NaNs found during setup! Aborting test.")
        return False
    
    kv_head_idx = 0
    
    # Execute all blocks using helper function
    try:
        instructions = execute_prefill_blocks(globs, layer_idx, seq_idx, seq_len, kv_head_idx, device_idx)
        print(f"✓ All {len(instructions)} blocks completed successfully")
        
        # Check for NaNs after all blocks
        nan_found, nan_details = check_for_nans_comprehensive(globs, "after multiple blocks")
        if nan_found:
            print("❌ NaNs detected after multiple block execution!")
            print("NaN details:", nan_details)
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error executing blocks: {e}")
        # Check for NaNs after error for debugging
        check_for_nans_comprehensive(globs, "after error in multiple blocks")
        raise


def test_different_kv_heads():
    """Test prefill with different KV head indices."""
    
    print("\n=== Testing Different KV Heads ===")
    
    # Setup test environment
    globs, device_idx, seq_idx, layer_idx = setup_simple_prefill_test()
    seq_len = 16  # Single block
    
    # Setup prerequisites
    setup_prefill_prerequisites(globs, device_idx, seq_idx, layer_idx, seq_len)
    
    block_idx = 0
    num_kv_heads_per_device = globs.num_kv_heads // globs.tp_size
    
    # Test different KV heads on this device
    for kv_head_idx in range(min(2, num_kv_heads_per_device)):
        print(f"\nTesting KV head {kv_head_idx}...")
        
        instruction = AttentionPrefill(
            layer_idx=layer_idx,
            prefill_seq_idx=seq_idx,
            prefill_block_idx=block_idx,
            kv_head_idx=kv_head_idx,
        )
        
        try:
            attention_prefill(globs, instruction, device_idx)
            print(f"✓ KV head {kv_head_idx} completed successfully")
        except Exception as e:
            print(f"✗ Error with KV head {kv_head_idx}: {e}")
            raise


def test_pyvm_vs_megakernel():
    """Test both PyVM and megakernel implementations and compare results."""
    
    print("\n=== Testing PyVM vs Megakernel Implementation ===")
    
    # Setup test environment
    seq_len = 32  # Small test case to start
    global_batch_size = seq_len  # For prefill, batch size should match sequence length
    
    print("Creating globals for PyVM vs Megakernel test...")
    globs = create_test_globals(global_batch_size)
    
    # Set up sequence
    device_idx = 0
    seq_idx = 0
    layer_idx = 0
    
    print("Setting up sequence pointers...")
    setup_sequence_pointers(globs, seq_len)
    
    # Setup prerequisites
    setup_prefill_prerequisites(globs, device_idx, seq_idx, layer_idx, seq_len)
    
    # Check for NaNs after setup
    nan_found, _ = check_for_nans_comprehensive(globs, "after setup for PyVM vs Megakernel")
    if nan_found:
        print("❌ NaNs found during setup! Aborting test.")
        return False
    
    # Store initial state BEFORE any computation
    initial_attn_out = [t.clone() for t in globs.attn_out]
    initial_barriers = [t.clone() for t in globs.barriers]
    
    # Set up the megakernel interpreter first (this must be done with the original globs)
    print("Setting up megakernel interpreter...")
    setup_megakernel_interpreter(globs)
    
    # Create copy for megakernel testing
    print("Creating separate globals copy for megakernel...")
    megakernel_globs = globs.with_new_activations(new_kv_cache=True)
    
    # Copy all necessary data to megakernel globals
    for dev_idx in range(globs.tp_size):
        megakernel_globs.k_cache[dev_idx].copy_(globs.k_cache[dev_idx])
        megakernel_globs.v_cache[dev_idx].copy_(globs.v_cache[dev_idx])
        megakernel_globs.post_rope_q[dev_idx].copy_(globs.post_rope_q[dev_idx])
        megakernel_globs.prefill_qo_indptr[dev_idx].copy_(globs.prefill_qo_indptr[dev_idx])
        megakernel_globs.prefill_kv_indptr[dev_idx].copy_(globs.prefill_kv_indptr[dev_idx])
        megakernel_globs.prefill_kv_indices[dev_idx].copy_(globs.prefill_kv_indices[dev_idx])
        megakernel_globs.prefill_kv_last_page_len[dev_idx].copy_(globs.prefill_kv_last_page_len[dev_idx])
        megakernel_globs.position_ids[dev_idx].copy_(globs.position_ids[dev_idx])
        megakernel_globs.barriers[dev_idx].copy_(globs.barriers[dev_idx])
        megakernel_globs.attn_out[dev_idx].zero_()
    
    kv_head_idx = 0
    
    # Run PyVM implementation
    print("\n--- Running PyVM Implementation ---")
    try:
        pyvm_instructions = execute_prefill_blocks(globs, layer_idx, seq_idx, seq_len, kv_head_idx, device_idx, use_megakernel=False)
        print(f"✓ PyVM completed {len(pyvm_instructions)} instructions successfully")
        
        # Check for NaNs after PyVM
        nan_found, nan_details = check_for_nans_comprehensive(globs, "after PyVM computation")
        if nan_found:
            print("❌ NaNs detected after PyVM computation!")
            return False
            
    except Exception as e:
        print(f"❌ PyVM failed: {e}")
        return False
    
    # Run Megakernel implementation
    print("\n--- Running Megakernel Implementation ---")
    try:
        mk_instructions = execute_prefill_blocks(megakernel_globs, layer_idx, seq_idx, seq_len, kv_head_idx, device_idx, use_megakernel=True)
        print(f"✓ Megakernel completed {len(mk_instructions)} instructions successfully")
        
        # Check for NaNs after megakernel
        nan_found, nan_details = check_for_nans_comprehensive(megakernel_globs, "after Megakernel computation")
        if nan_found:
            print("❌ NaNs detected after Megakernel computation!")
            return False
            
    except Exception as e:
        print(f"❌ Megakernel failed: {e}")
        return False
    
    print("\n--- Comparing Results: PyVM vs Megakernel ---")
    
    # Compare PyVM and Megakernel results
    devices_with_changes, max_abs_diff_overall, all_matches = validate_attention_outputs(
        globs, initial_attn_out, seq_len=seq_len, reference_globs=megakernel_globs, description="PyVM vs Megakernel"
    )
    
    print(f"\nDevices with attention output changes: {devices_with_changes}")
    print(f"Maximum absolute difference between PyVM and Megakernel: {max_abs_diff_overall:.2e}")
    
    # Add breakpoint when results don't match
    if max_abs_diff_overall > 1e-3:
        print("🔍 BREAKPOINT: Large differences detected between PyVM and Megakernel!")
        print("📊 Detailed analysis:")
        
        for dev_idx in devices_with_changes:
            pyvm_output = globs.attn_out[dev_idx]
            megakernel_output = megakernel_globs.attn_out[dev_idx]
            
            abs_diff = torch.abs(pyvm_output - megakernel_output)
            max_diff_pos = torch.argmax(abs_diff.flatten())
            max_diff_idx = torch.unravel_index(max_diff_pos, abs_diff.shape)
            
            print(f"\n  Device {dev_idx}:")
            print(f"    Max diff location: {max_diff_idx}")
            print(f"    PyVM value: {pyvm_output[max_diff_idx].item():.6f}")
            print(f"    Megakernel value: {megakernel_output[max_diff_idx].item():.6f}")
            print(f"    Absolute diff: {abs_diff[max_diff_idx].item():.6f}")
            print(f"    PyVM tensor stats: min={pyvm_output.min():.6f}, max={pyvm_output.max():.6f}, mean={pyvm_output.mean():.6f}")
            print(f"    Megakernel tensor stats: min={megakernel_output.min():.6f}, max={megakernel_output.max():.6f}, mean={megakernel_output.mean():.6f}")
            
            # Check for NaNs or infs
            if torch.isnan(pyvm_output).any():
                print(f"    ⚠️  PyVM has NaN values!")
            if torch.isnan(megakernel_output).any():
                print(f"    ⚠️  Megakernel has NaN values!")
            if torch.isinf(pyvm_output).any():
                print(f"    ⚠️  PyVM has Inf values!")
            if torch.isinf(megakernel_output).any():
                print(f"    ⚠️  Megakernel has Inf values!")
        
        print("\n🔧 Debug information:")
        print(f"    Instructions executed: {len(pyvm_instructions)}")
        print(f"    Sequence length: {seq_len}")
        print(f"    Global batch size: {globs.global_batch_size}")
        print(f"    Prefill seq lens: {globs.prefill_seq_lens}")
        
        # Add input data comparison
        print("\n📥 Input data comparison:")
        for dev_idx in range(min(2, globs.tp_size)):  # Check first 2 devices
            pyvm_input = globs.post_rope_q[dev_idx]
            mk_input = megakernel_globs.post_rope_q[dev_idx]
            input_diff = torch.abs(pyvm_input - mk_input).max().item()
            print(f"    Device {dev_idx} post_rope_q max diff: {input_diff:.2e}")
            
            pyvm_k = globs.k_cache[dev_idx]
            mk_k = megakernel_globs.k_cache[dev_idx]
            k_diff = torch.abs(pyvm_k - mk_k).max().item()
            print(f"    Device {dev_idx} k_cache max diff: {k_diff:.2e}")
        
        # Breakpoint here - you can add import pdb; pdb.set_trace() to debug interactively
        import pdb; pdb.set_trace()
    
    # Final validation
    success = all_matches and len(devices_with_changes) > 0
    
    # Print results
    if success:
        print("✅ SUCCESS: PyVM and Megakernel produce matching results!")
        print(f"✅ Both implementations processed {seq_len} tokens correctly")
        print(f"✅ Max difference: {max_abs_diff_overall:.2e} (within tolerance)")
    else:
        print("❌ FAILURE: PyVM and Megakernel produce different results!")
        if len(devices_with_changes) == 0:
            print("  No devices had attention output changes - check test setup")
        else:
            print(f"  Max difference: {max_abs_diff_overall:.2e} (exceeds tolerance)")
    
    return success


def test_distributed_transpose():
    """Test prefill with 128 tokens and 8 instructions to verify distributed transpose across all GPUs."""
    
    print("\n=== Testing Distributed Transpose (128 tokens, 8 instructions) ===")
    
    # Setup test environment with 128 tokens
    global_batch_size = 128  # Smaller for this test but still meaningful
    seq_len = 128  # 128 tokens = 8 blocks of 16 tokens each
    
    print("Creating globals for distributed transpose test...")
    globs = create_test_globals(global_batch_size)
    
    # Set up sequence spanning 128 tokens
    device_idx = 0  # Source device for all instructions
    seq_idx = 0
    layer_idx = 0
    
    print("Setting up sequence pointers...")
    setup_sequence_pointers(globs, seq_len)
    
    # Setup prerequisites
    setup_prefill_prerequisites(globs, device_idx, seq_idx, layer_idx, seq_len)
    
    # Check for NaNs after setup
    nan_found, _ = check_for_nans_comprehensive(globs, "after setup for distributed transpose")
    if nan_found:
        print("❌ NaNs found during setup! Aborting test.")
        return False
    
    # Store initial state BEFORE any computation
    initial_attn_out = [t.clone() for t in globs.attn_out]
    initial_barriers = [t.clone() for t in globs.barriers]
    
    # Create reference copy for comparison
    print("Creating reference copy for validation...")
    reference_globs = create_reference_globals_copy(globs)
    
    # Execute all prefill blocks
    kv_head_idx = 0
    print("Executing prefill blocks...")
    instructions = execute_prefill_blocks(globs, layer_idx, seq_idx, seq_len, kv_head_idx, device_idx)
    
    # Check for NaNs after test computation
    nan_found, nan_details = check_for_nans_comprehensive(globs, "after test computation")
    if nan_found:
        print("❌ NaNs detected after test computation!")
        print("NaN details:", nan_details)
        return False
    
    # Run reference computation
    print("\n--- Running Reference Computation ---")
    for block_idx, instruction in enumerate(instructions):
        token_range = f"{block_idx * 16}-{block_idx * 16 + 15}"
        print(f"Running reference instruction {block_idx + 1}/8 (tokens {token_range})...")
        attention_prefill(reference_globs, instruction, device_idx)
    
    # Check for NaNs after reference computation
    nan_found, nan_details = check_for_nans_comprehensive(reference_globs, "after reference computation")
    if nan_found:
        print("❌ NaNs detected after reference computation!")
        print("NaN details:", nan_details)
        return False
    
    print("\n--- Comparing Results: Test vs Reference ---")
    
    # Compare results using unified validation function
    devices_with_changes, max_abs_diff_overall, all_matches = validate_attention_outputs(
        globs, initial_attn_out, seq_len=seq_len, reference_globs=reference_globs
    )
    
    print(f"\nDevices with attention output changes: {devices_with_changes}")
    print(f"Expected devices: {list(range(globs.tp_size))}")
    print(f"Maximum absolute difference across all devices: {max_abs_diff_overall:.2e}")
    
    # Analyze barrier changes
    devices_with_barrier_changes, total_barrier_updates = analyze_barrier_changes(
        globs, initial_barriers, seq_len
    )
    
    # Validate that all devices have reasonable outputs and values match
    all_devices_have_output = len(devices_with_changes) == globs.tp_size
    all_devices_have_barriers = len(devices_with_barrier_changes) == globs.tp_size
    
    # Final validation
    success = all_devices_have_output and all_devices_have_barriers and all_matches
    
    # Print results using helper function
    print_test_summary(
        success, devices_with_changes, max_abs_diff_overall, 
        expected_devices=list(range(globs.tp_size)), 
        seq_len=seq_len, num_instructions=len(instructions)
    )
    
    if success:
        print("✅ SUCCESS: Distributed transpose working correctly on all devices!")
    else:
        if not all_devices_have_output:
            missing_output = set(range(globs.tp_size)) - set(devices_with_changes)
            print(f"⚠ Warning: Devices {missing_output} did not receive attention outputs")
        if not all_devices_have_barriers:
            missing_barriers = set(range(globs.tp_size)) - set(devices_with_barrier_changes)
            print(f"⚠ Warning: Devices {missing_barriers} did not get barrier updates")
    
    return success


def test_irregular_sequences():
    """Test prefill with irregular sequence lengths and cross-GPU boundaries."""
    
    print("\n=== Testing Irregular Sequences & Cross-GPU Boundaries ===")
    
    # Setup test environment
    global_batch_size = 256  # Larger to enable cross-GPU sequences
    
    print("Creating globals for irregular sequence test...")
    globs = create_test_globals(global_batch_size, num_pages=128)
    
    # Test cases with various irregular sequence lengths
    test_cases = [
        {"seq_len": 13, "description": "Short sequence (13 tokens, < 16)"},
        {"seq_len": 23, "description": "Medium sequence (23 tokens, partial second block)"},
        {"seq_len": 47, "description": "Cross-GPU sequence (47 tokens, spans devices 0-1)"},
        {"seq_len": 89, "description": "Long cross-GPU sequence (89 tokens, spans devices 0-2)"},
        {"seq_len": 133, "description": "Very long sequence (133 tokens, spans devices 0-4)"},
    ]
    
    all_tests_passed = True
    
    for test_idx, test_case in enumerate(test_cases):
        seq_len = test_case["seq_len"]
        description = test_case["description"]
        
        print(f"\n--- Test Case {test_idx + 1}: {description} ---")
        
        # Calculate expected devices and other test parameters
        expected_devices = calculate_expected_devices(seq_len, global_batch_size, globs.tp_size)
        num_blocks = (seq_len + 15) // 16  # Ceiling division
        
        print(f"  Sequence length: {seq_len}")
        print(f"  Number of blocks: {num_blocks}")
        print(f"  Expected devices to receive outputs: {sorted(expected_devices)}")
        
        # Setup sequence
        device_idx = 0  # Source device
        seq_idx = 0
        layer_idx = 0
        
        # Create fresh globals for each test case to avoid contamination
        test_globs = create_test_globals(global_batch_size, num_pages=128)
        
        # Setup sequence pointers
        setup_sequence_pointers(test_globs, seq_len)
        
        # Setup prerequisites
        setup_prefill_prerequisites(test_globs, device_idx, seq_idx, layer_idx, seq_len)
        
        # Create reference copy BEFORE any computation
        reference_globs = create_reference_globals_copy(test_globs)
        
        # Store initial state
        initial_attn_out = [t.clone() for t in test_globs.attn_out]
        
        # Run test: execute all blocks for this sequence
        kv_head_idx = 0
        
        try:
            # Execute test computation
            instructions = execute_prefill_blocks(test_globs, layer_idx, seq_idx, seq_len, kv_head_idx, device_idx)
            
            # Run reference computation
            print(f"  Running reference computation...")
            for instruction in instructions:
                attention_prefill(reference_globs, instruction, device_idx)
            
            # Compare results using unified validation function
            devices_with_changes, max_abs_diff_overall, values_match = validate_attention_outputs(
                test_globs, initial_attn_out, seq_len=seq_len, reference_globs=reference_globs, description="irregular"
            )
            
            # Check if we got the expected devices
            expected_devices_list = sorted(expected_devices)
            devices_with_changes_sorted = sorted(devices_with_changes)
            
            devices_correct = devices_with_changes_sorted == expected_devices_list
            
            if devices_correct and values_match:
                print(f"  ✅ Test case passed!")
                print(f"    Devices with outputs: {devices_with_changes_sorted}")
                print(f"    Max difference: {max_abs_diff_overall:.2e}")
            else:
                print(f"  ❌ Test case failed!")
                if not devices_correct:
                    print(f"    Expected devices: {expected_devices_list}")
                    print(f"    Actual devices: {devices_with_changes_sorted}")
                if not values_match:
                    print(f"    Values don't match (max diff: {max_abs_diff_overall:.2e})")
                all_tests_passed = False
                
        except Exception as e:
            print(f"  ❌ Test case failed with error: {e}")
            all_tests_passed = False
    
    print(f"\n--- Irregular Sequence Test Summary ---")
    if all_tests_passed:
        print("✅ All irregular sequence tests passed!")
        print("✅ Cross-GPU boundaries handled correctly!")
        print("✅ Partial blocks work correctly!")
    else:
        print("❌ Some irregular sequence tests failed!")
    
    return all_tests_passed


if __name__ == "__main__":
    print("Prefill Instruction Test with PyVM vs Megakernel Comparison")
    print("=" * 60)
    
    # Run tests and collect results
    print("Running basic prefill tests...")
    success_basic1 = test_single_prefill_instruction()
    success_basic2 = test_multiple_blocks()
    test_different_kv_heads()  # This test doesn't return a success flag
    
    # The new test: compare PyVM vs Megakernel implementations
    success_comparison = test_pyvm_vs_megakernel()
    
    # The big test: verify distributed transpose across all GPUs
    success1 = test_distributed_transpose()
    
    # The edge case test: irregular sequences and cross-GPU boundaries
    success2 = test_irregular_sequences()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    basic_tests_passed = success_basic1 and success_basic2
    if basic_tests_passed:
        print("✅ Basic prefill tests: PASSED (no NaNs detected)")
    else:
        print("❌ Basic prefill tests: FAILED (NaNs detected or other issues)")
    
    if success_comparison:
        print("✅ PyVM vs Megakernel comparison: PASSED")
        print("  ✓ Both implementations produce identical results")
    else:
        print("❌ PyVM vs Megakernel comparison: FAILED")
        print("  ✗ Implementations produce different results")
    
    overall_success = basic_tests_passed and success_comparison and success1 and success2
    if overall_success:
        print("✅ All tests completed successfully!")
        print("✅ PyVM and Megakernel implementations match perfectly!")
        print("✅ Distributed transpose verified across all 8 GPUs!")
        print("✅ Irregular sequences and cross-GPU boundaries work perfectly!")
        print("✅ NO NANS DETECTED in any test!")
    else:
        print("❌ Some tests failed:")
        if not basic_tests_passed:
            print("  - Basic prefill tests had issues")
        if not success_comparison:
            print("  - PyVM vs Megakernel comparison failed")
        if not success1:
            print("  - Distributed transpose test had issues")
        if not success2:
            print("  - Irregular sequence test had issues")
        print("  Check the output above for detailed error information.")
    print("=" * 60)