print('hello -1')
import torch
print('hello 0')
import mla_decode
import math

print('hello 1')

# Test configuration
batch = 1
heads = 16
new_seq = 1  # Test with 1 new token
num_instructions = 1
num_partials = 2  # Number of partial results to reduce

print('hello 2')

# Initialize tensors similar to run_one_partial.py
instructions = torch.zeros((num_instructions, 32), dtype=torch.int32, device='cuda')

# Base tensors needed for the API (even though we won't use them for reduction)
Qv = torch.zeros((batch, new_seq, heads, 512), dtype=torch.bfloat16, device='cuda')
Qrot = torch.zeros((batch, new_seq, heads, 64), dtype=torch.bfloat16, device='cuda')
V = torch.zeros((256*batch, 256, 512), dtype=torch.bfloat16, device='cuda')
Krot = torch.zeros((256*batch, 256, 64), dtype=torch.bfloat16, device='cuda')
Table = torch.zeros((batch, 256), dtype=torch.int32, device='cuda')
O = torch.zeros((batch, new_seq, heads, 512), dtype=torch.bfloat16, device='cuda')

# The main tensors for reduction testing - these will contain our partial results
# Size them to accommodate the maximum UID we'll use (which is num_partials-1 = 2)
max_uid = num_partials - 1
O_scratch = torch.zeros((max_uid + 5, new_seq, heads, 512), dtype=torch.float32, device='cuda')
Lvec_scratch = torch.zeros((1, max_uid + 5, new_seq, heads), dtype=torch.float32, device='cuda')

# Completion flag to signal when partials are ready
completion_flag = torch.zeros((max_uid + 1, new_seq), dtype=torch.int32, device='cuda')

# Global instruction index for megakernel API
global_instruction_index = torch.zeros(1, 1, 1, 1, dtype=torch.int32, device='cuda')

# MLA parameters
softmax_scale = 1.0 / math.sqrt(512 + 64)  # 1/sqrt(D_Main + D_Rot)
tic = 1

# Timing tensor
timings = torch.zeros((num_instructions, 128), dtype=torch.int32, device='cuda')

print('hello 3 - setting up partial results')

# Set up mock partial results for testing
# We'll create realistic partial attention outputs with different max values and norms
torch.manual_seed(42)

# Store all partial lvec values for reference computation
partial_lvecs = torch.zeros((num_partials, heads), device='cuda', dtype=torch.float32)

# Create partial results with different attention patterns
for i in range(num_partials):
    # Create some realistic attention output values
    O_scratch[i, 0, :, :] = torch.randn(heads, 512, device='cuda', dtype=torch.float32) * 0.1
    
    # Each partial should have its own lvec value (log-space values)
    partial_lvecs[i, :] = torch.randn(heads, device='cuda', dtype=torch.float32) * 2.0
    
    # Store each partial's lvec in Lvec_scratch at the appropriate index
    Lvec_scratch[0, i, 0, :] = partial_lvecs[i, :]
    
    # Mark this partial result as ready
    completion_flag[i, 0] = tic

print('hello 4 - configuring reduction instruction')

# Configure the reduction instruction
# Format: [opcode, uid, num_iters, dst.batch_idx, dst.seq_idx, src_uid, load_uid0, load_uid1, load_uid2, ...]
instructions[0, :10] = torch.tensor([
    2,                # Opcode (OPCODE_MLA_Reduction)
    0,                # uid - unique identifier for this reduction
    num_partials - 1, # num_iters - number of additional iterations after the first one
    -4,               # dst.batch_idx - negative means write to scratch (scratch index 0)
    0,                # dst.seq_idx 
    0,                # src_uid - initial source (will be first partial, UID 0)
    1,                # load_uid for iter 1 (second partial)
    2,                # load_uid for iter 2 (third partial) 
    0,                # padding
    0,                # padding
], dtype=torch.int32)

print('Instructions configured:', instructions[0, :10])

def compute_reduction_reference(O_scratch_vals, partial_lvecs, num_partials):
    """
    Python reference implementation of the softmax reduction.
    This mirrors the logic in mla_reduction.cu consumer::run().
    """
    # Start with the first partial result
    o_acc = O_scratch_vals[0].clone()  # rt_fl<16, MLA_QVO_D / 8> o_acc
    lvec_acc = partial_lvecs[0].clone()  # col_vec<rt_fl<16, krot_tile::rows>> lvec_acc
    
    # Reduction loop over remaining partials
    for i in range(1, num_partials):
        o = O_scratch_vals[i]  # Current partial result
        lvec = partial_lvecs[i]  # Each partial has its own lvec value
        
        # Reduction computation (from lines 152-165 in mla_reduction.cu)
        max_lvec = torch.maximum(lvec_acc, lvec)  # warp::max(max_lvec, lvec_acc, lvec)
        lvec_acc = lvec_acc - max_lvec            # warp::sub(lvec_acc, lvec_acc, max_lvec)  
        lvec = lvec - max_lvec                    # warp::sub(lvec, lvec, max_lvec)
        lvec_acc = torch.exp2(lvec_acc)           # warp::exp2(lvec_acc, lvec_acc)
        lvec = torch.exp2(lvec)                   # warp::exp2(lvec, lvec)
        sum_lvec = lvec_acc + lvec                # warp::add(sum_lvec, lvec_acc, lvec)
        lvec_acc = lvec_acc / sum_lvec            # warp::div(lvec_acc, lvec_acc, sum_lvec)
        lvec = lvec / sum_lvec                    # warp::div(lvec, lvec, sum_lvec)
        
        # Apply weights to outputs and accumulate
        o_acc = o_acc * lvec_acc.unsqueeze(-1)    # warp::mul_row(o_acc, o_acc, lvec_acc)
        o = o * lvec.unsqueeze(-1)                # warp::mul_row(o, o, lvec)
        o_acc = o_acc + o                         # warp::add(o_acc, o_acc, o)
        
        # Update lvec_acc for next iteration
        sum_lvec = torch.log2(sum_lvec)           # warp::log2(sum_lvec, sum_lvec)
        lvec_acc = sum_lvec + max_lvec            # warp::add(lvec_acc, sum_lvec, max_lvec)
    
    return o_acc, lvec_acc

print('hello 5 - computing reference')

# Compute reference result
o_ref, lvec_ref = compute_reduction_reference(O_scratch[:, 0, :, :], partial_lvecs, num_partials)

print('hello 6 - running kernel')

# Run the kernel
try:
    mla_decode.mla_decode(
        instructions, 
        timings, 
        global_instruction_index,
        Qrot, Qv, Krot, V, Table, O, 
        O_scratch, Lvec_scratch, completion_flag, 
        softmax_scale, tic
    )
    torch.cuda.synchronize()
    print('Kernel completed successfully')
except Exception as e:
    print(f'Kernel execution failed: {e}')
    exit(1)

print('hello 7 - validating results')

# The result should be in O_scratch[0] (since dst.batch_idx = -1, meaning scratch index 0)  
kernel_output = O_scratch[3, 0, :, :]  # Shape: [heads, 512]
kernel_lvec = Lvec_scratch[0, 3, 0, :]  # Shape: [heads]

print(f'Reference output mean: {o_ref.abs().mean():.6f}')
print(f'Kernel output mean: {kernel_output.abs().mean():.6f}')
print(f'Reference lvec mean: {lvec_ref.abs().mean():.6f}') 
print(f'Kernel lvec mean: {kernel_lvec.abs().mean():.6f}')

# Compare outputs
output_diff = (kernel_output - o_ref).abs()
lvec_diff = (kernel_lvec - lvec_ref).abs()

print(f'Max absolute output difference: {output_diff.max():.6f}')
print(f'Average absolute output difference: {output_diff.mean():.6f}')
print(f'Max absolute lvec difference: {lvec_diff.max():.6f}')
print(f'Average absolute lvec difference: {lvec_diff.mean():.6f}')

# Validation thresholds
output_tolerance = 1e-2  # Allow for some numerical differences
lvec_tolerance = 1e-2

if output_diff.max() < output_tolerance and lvec_diff.max() < lvec_tolerance:
    print('✓ PASSED: Reduction instruction working correctly!')
else:
    print('✗ FAILED: Output differs significantly from reference')
    print('This might indicate an issue in the reduction implementation')

print('hello 8 - test completed')

# Optional: breakpoint for debugging
# breakpoint()