import math
from pathlib import Path
from time import time

import pydra
import torch
from tqdm import tqdm
from transformers import AutoConfig

from megakernels.demos.tp_throughput.globs import Globals, make_globals
from megakernels.instructions import Instruction
from megakernels.demos.tp_throughput.mk import TensorParallelMK_Interpreter
from megakernels.demos.tp_throughput.python_vm import INSTRUCTION_TO_SOLVER, attention_prefill
from megakernels.demos.tp_throughput.instructions import AttentionPrefill
from megakernels.demos.tp_throughput.scheduler import (
    create_instruction_tensor,
    init_random_weights,
    load_weights,
    schedule_model,
    schedule_to_tensor,
    setup_rope_and_interleave,
)
from megakernels.scheduler import create_timing_tensor
from megakernels.utils import get_sm_count


class Config(pydra.Config):
    model: str = "meta-llama/Llama-3.1-70B-Instruct"
    mk_dir = Path(__file__).parent.parent.parent / "demos" / "cross-gpu-llama"
    glob_bs: int = 1024  # should be at least 2048 for memory alignment
    num_pages: int = 32768
    seq_len: int = 32

    # Diff test specific options
    stop_after_op: str | None = None
    start_after_op: str | None = None
    layer_limit: int | None = 1  # Keep as layer_limit for compatibility but treat like num_layer_override
    skip_pyvm: bool = False
    skip_starting_ops: bool = False

    # Prefill-specific options
    test_mode: str = "decode"  # "decode", "prefill", or "both"
    prefill_seq_lens: list[int] | None = None  # If None, will use [seq_len]; use [1024] for 1 seq of 1024 tokens
    num_prefill_seqs: int = 32  # Number of prefill sequences to test (32 seqs * 32 tokens = 1024)

    # Must-not-change fields
    num_devices: int = 8

    skip_weight_load: bool = True
    use_random_weights: bool = True
    weight_std: float = 1.0
    bar_init_val: int = 0

    gwq: bool = True
    interleave_waves: bool = False
    interleave_buffer_factor: float = 1.0

    bp: bool = False


# Prefill testing helper functions (ported from test_prefill.py)

def setup_prefill_data(globs: Globals, config: Config):
    """Setup prefill sequences and indirection pointers."""
    import time
    
    # Determine prefill sequence lengths
    if config.prefill_seq_lens is None:
        prefill_seq_lens = [config.seq_len] * config.num_prefill_seqs
    else:
        prefill_seq_lens = config.prefill_seq_lens
        
    print(f"Setting up prefill data for sequences: {prefill_seq_lens}")
    
    # Set prefill sequence lengths in globals
    # Important: for prefill, global_batch_size should be total tokens, 
    # but we need to ensure it works with barrier calculations
    total_tokens = sum(prefill_seq_lens)
    print(f"Total prefill tokens: {total_tokens}")
    print(f"Matmul batch block size: {globs.matmul_batch_block_size}")
    
    globs.set_sizes(
        global_batch_size=total_tokens,
        prefill_seq_lens=prefill_seq_lens,
    )
    
    # Create prefill indirection pointers (same for all devices)
    # QO indirection pointers
    qo_indptr = torch.zeros(len(prefill_seq_lens) + 1, dtype=torch.int32, device="cuda:0")
    cumsum = 0
    for seq_idx, seq_len in enumerate(prefill_seq_lens):
        qo_indptr[seq_idx] = cumsum
        cumsum += seq_len
    qo_indptr[-1] = cumsum
    
    # KV indirection pointers and indices
    kv_indptr = torch.zeros(len(prefill_seq_lens) + 1, dtype=torch.int32, device="cuda:0")
    total_pages_needed = 0
    kv_indices_list = []
    kv_last_page_lens = []
    
    for seq_idx, seq_len in enumerate(prefill_seq_lens):
        pages_needed = math.ceil(seq_len / globs.page_size)
        kv_indptr[seq_idx] = total_pages_needed
        
        # Assign consecutive pages for this sequence
        seq_kv_indices = torch.arange(total_pages_needed, total_pages_needed + pages_needed, 
                                    dtype=torch.int32, device="cuda:0")
        kv_indices_list.append(seq_kv_indices)
        
        # Last page length for this sequence
        last_page_len = seq_len % globs.page_size or globs.page_size
        kv_last_page_lens.append(last_page_len)
        
        total_pages_needed += pages_needed
        
    kv_indptr[-1] = total_pages_needed
    kv_indices = torch.cat(kv_indices_list) if kv_indices_list else torch.empty(0, dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.tensor(kv_last_page_lens, dtype=torch.int32, device="cuda:0")
    
    # Copy prefill info to all devices
    globs.copy_prefill_info(
        qo_indptr=qo_indptr,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
    )
    
    # Setup per-device data
    for dev_idx in range(config.num_devices):
        # Position IDs for prefill sequences
        position_ids = torch.zeros(sum(prefill_seq_lens), dtype=torch.int32, device=f"cuda:{dev_idx}")
        offset = 0
        for seq_len in prefill_seq_lens:
            position_ids[offset:offset + seq_len] = torch.arange(seq_len, dtype=torch.int32, device=f"cuda:{dev_idx}")
            offset += seq_len
        
        # Set position IDs
        globs.position_ids[dev_idx][:len(position_ids)] = position_ids
        
        # Initialize random inputs for prefill
        torch.manual_seed(int(time.time() * 1000) % 2**32 + dev_idx)
        torch.nn.init.normal_(globs.hidden_states[dev_idx], std=0.1)
    
    # Debug: Print barrier state before IncBarrier
    print(f"Debug: Before IncBarrier - global_batch_size={globs.global_batch_size}")
    print(f"Debug: Barrier tensor shape: {globs.barriers[0].shape}")
    print(f"Debug: Initial barrier[0,0,0,0] = {globs.barriers[0][0,0,0,0].item()}")




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
                
                print(f"✓ Device {dev_idx}: norm={test_norm:.4f}")
                print(f"  Max abs diff: {max_abs_diff:.2e}, Mean abs diff: {mean_abs_diff:.2e}, Rel diff: {rel_diff:.2e}")
                
                # Check if values match within tolerance
                if max_abs_diff > 1e-3:  # More lenient for larger perturbations
                    print(f"  ⚠ Large difference detected!")
                    all_matches = False
                elif max_abs_diff < 1e-7:  # Less paranoid about small differences
                    print(f"  ⚠ Suspiciously small difference - check independence!")
                else:
                    print(f"  ✓ Values match within expected tolerance")
            else:
                # Single instruction validation - just report changes
                print(f"✓ Attention output changed on device {dev_idx}, norm={test_norm:.4f}")
                
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
    
    if not devices_with_changes and not is_reference_comparison:
        print("⚠ Warning: Attention output was not modified")
        all_matches = False
    
    return devices_with_changes, max_abs_diff_overall, all_matches


def diff_prefill_tensors(name, tensors_pyvm, tensors_mk, device_count, expected_devices=None):
    """Compare prefill-specific tensors with device filtering."""
    print(f"\n--- Comparing {name} (Prefill-focused) ---")
    
    devices_with_differences = []
    
    for dev_idx in range(device_count):
        t1 = tensors_pyvm[dev_idx]
        t2 = tensors_mk[dev_idx]

        if t1 is None or t2 is None:
            print(f"Device {dev_idx}: One or both tensors are None")
            continue

        # Check if this device should have changes
        has_changes = not torch.equal(t1, t2)
        
        if has_changes:
            breakpoint()
            devices_with_differences.append(dev_idx)
            
            # Compute differences
            abs_diff = torch.abs(t1 - t2)
            max_abs_diff = abs_diff.max().item()

            # Compute relative difference
            denominator = torch.abs(t1) + torch.abs(t2) + 1e-8
            rel_diff = 2 * abs_diff / denominator
            mean_rel_diff = rel_diff.mean().item()

            print(f"Device {dev_idx}: max_abs_diff={max_abs_diff:.6e}, mean_rel_diff={mean_rel_diff:.6e}")
        else:
            print(f"Device {dev_idx}: No differences detected")
    
    # Check against expected devices if provided
    if expected_devices is not None:
        expected_set = set(expected_devices)
        actual_set = set(devices_with_differences)
        
        if expected_set == actual_set:
            print(f"✓ Devices with differences match expected: {sorted(actual_set)}")
        else:
            print(f"⚠ Device mismatch - Expected: {sorted(expected_set)}, Actual: {sorted(actual_set)}")
            
    return devices_with_differences


def interpret_instructions_round_robin(
    globs: Globals, instruction_lists: list[list[Instruction]], device_count: int
):
    """
    Interpret instructions in round-robin fashion across all devices.
    This ensures proper data dependencies are maintained across GPUs.
    """
    # Find the maximum number of instructions across all devices
    max_instructions = max(len(inst_list) for inst_list in instruction_lists)

    print(
        f"Running PyVM interpreter in round-robin fashion across {device_count} devices"
    )
    print(f"Max instructions per device: {max_instructions}")

    instructions_executed = 0

    # Execute instructions in round-robin order
    for inst_idx in tqdm(range(max_instructions), desc="PyVM instructions"):
        for dev_idx in range(device_count):
            # Check if this device has an instruction at this index
            if inst_idx >= len(instruction_lists[dev_idx]):
                continue

            # Get the instruction for this device
            instruction = instruction_lists[dev_idx][inst_idx]

            # Execute the instruction using the appropriate solver
            solver = INSTRUCTION_TO_SOLVER[type(instruction)]
            solver(globs, instruction, dev_idx)

            instructions_executed += 1

    print(f"PyVM executed {instructions_executed} instructions total")


def diff_tensors(name, tensors_pyvm, tensors_mk, device_count):
    """Compare tensors across multiple devices between PyVM and MK runs."""
    print(f"\n--- Comparing {name} ---")

    for dev_idx in range(device_count):
        t1 = tensors_pyvm[dev_idx]
        t2 = tensors_mk[dev_idx]

        if t1 is None or t2 is None:
            print(f"Device {dev_idx}: One or both tensors are None")
            continue

        # Compute differences
        abs_diff = torch.abs(t1 - t2)
        max_abs_diff = abs_diff.max().item()

        # Compute relative difference
        denominator = torch.abs(t1) + torch.abs(t2) + 1e-8
        rel_diff = 2 * abs_diff / denominator
        mean_rel_diff = rel_diff.mean().item()

        print(
            f"Device {dev_idx}: max_abs_diff={max_abs_diff:.6e}, mean_rel_diff={mean_rel_diff:.6e}"
        )


@torch.inference_mode()
def main(config: Config):
    torch.manual_seed(10210)

    model_config = AutoConfig.from_pretrained(config.model)

    print("Making globals:")
    globs_pyvm = make_globals(
        model_config=model_config,
        global_batch_size=config.glob_bs,
        num_pages=config.num_pages,
        num_devices=config.num_devices,
        barrier_init_val=config.bar_init_val,
        global_work_queue_enabled=config.gwq,
        layer_limit=config.layer_limit,
    )

    # Initialize inputs based on test mode
    print(f"Initializing inputs for test mode: {config.test_mode}")
    
    if config.test_mode in ["decode", "both"]:
        # Setup decode inputs (original logic)
        print("Setting up decode inputs...")
        pages_per_seq = math.ceil(config.seq_len / globs_pyvm.page_size)
        total_indices = pages_per_seq * config.glob_bs

        assert total_indices <= globs_pyvm.num_pages, (
            f"Total indices {total_indices} must be less than or equal to num_pages "
            f"{globs_pyvm.num_pages}"
        )

        indices = torch.arange(total_indices, dtype=torch.int32)
        indptr = torch.arange(config.glob_bs + 1, dtype=torch.int32) * pages_per_seq

        last_page_per_seq = indptr[1:] - 1

        last_page_len = config.seq_len % globs_pyvm.page_size
        if last_page_len == 0:
            last_page_len = globs_pyvm.page_size

        append_indices = last_page_per_seq * globs_pyvm.page_size + (
            (config.seq_len - 1) % globs_pyvm.page_size
        )
        position_ids = torch.ones(globs_pyvm.global_batch_size, dtype=torch.int32) * (
            config.seq_len - 1
        )

        for dev_idx in range(config.num_devices):
            globs_pyvm.decode_kv_indices[dev_idx][:total_indices] = indices.to(f"cuda:{dev_idx}")
            globs_pyvm.decode_kv_indptr[dev_idx][: config.glob_bs + 1] = indptr.to(
                f"cuda:{dev_idx}"
            )
            globs_pyvm.decode_kv_last_page_len[dev_idx].fill_(last_page_len)
            globs_pyvm.kv_append_indices[dev_idx][:] = append_indices.to(f"cuda:{dev_idx}")
            globs_pyvm.position_ids[dev_idx][:] = position_ids.to(f"cuda:{dev_idx}")

            torch.manual_seed(dev_idx)
            torch.nn.init.normal_(globs_pyvm.hidden_states[dev_idx])
    
    if config.test_mode in ["prefill", "both"]:
        # Setup prefill inputs
        print("Setting up prefill inputs...")
        setup_prefill_data(globs_pyvm, config)
        
        # Barriers should be properly initialized by the actual operations, not pre-patched

    if not config.skip_weight_load:
        print("Loading model weights:")
        start_time = time()
        load_weights(config.model, globs_pyvm, layer_limit=config.layer_limit)
        end_time = time()
        print(f"Time taken to load weights: {end_time - start_time} seconds")
    elif config.use_random_weights:
        print("Initializing random weights:")
        init_random_weights(globs_pyvm, std=config.weight_std)
    else:
        print("Not initializing weights (probably using zero weights)")

    setup_rope_and_interleave(globs_pyvm)

    print(f"Scheduling instructions with interleave_waves={config.interleave_waves}:")

    if config.interleave_waves:
        sm_count = get_sm_count("cuda:0")
        overlap_buffer_size = round(sm_count * config.interleave_buffer_factor)
    else:
        overlap_buffer_size = None

    instruction_tensors = []
    timing_tensors = []
    pre_schedules = []
    schedules = []
    global_instruction_index_tensors = []

    def generate_schedule():
        for dev_idx in tqdm(range(config.num_devices), desc="Scheduling instructions"):
            insts = create_instruction_tensor(
                globs_pyvm,
                device_idx=dev_idx,
                layer_limit=config.layer_limit,
                interleave_waves=config.interleave_waves,
                interleave_buffer_size=overlap_buffer_size,
                stop_after_op=config.stop_after_op,
            )

            instruction_tensors.append(insts)
            print(f"Instruction tensor for device {dev_idx} has shape {insts.shape}")

        globs_pyvm.copy_instructions(instruction_tensors)
        
        # Breakpoint to inspect instruction tensors
        breakpoint()
        
        return [i.shape[0] for i in instruction_tensors]

    # Generate instruction objects for PyVM and instruction tensors for MK
    if config.start_after_op is not None:
        # Generate pre-schedule using schedule_model for PyVM
        for dev_idx in range(config.num_devices):
            pre_sched = schedule_model(
                globs_pyvm,
                device_idx=dev_idx,
                layer_limit=config.layer_limit,
                interleave_waves=config.interleave_waves,
                interleave_buffer_size=overlap_buffer_size,
                stop_after_op=config.start_after_op,
            )
            pre_schedules.append(pre_sched)
            
            # Also generate full schedule for remaining instructions
            full_sched = schedule_model(
                globs_pyvm,
                device_idx=dev_idx,
                layer_limit=config.layer_limit,
                interleave_waves=config.interleave_waves,
                interleave_buffer_size=overlap_buffer_size,
                stop_after_op=config.stop_after_op,
            )
            
            # Extract remaining schedule after pre_sched  
            remaining_sched = full_sched[len(pre_sched):]
            schedules.append(remaining_sched)
        
        # Generate instruction tensors for MK using create_instruction_tensor
        num_instructions = generate_schedule()
    else:
        # No starting ops - generate full schedule for PyVM
        for dev_idx in range(config.num_devices):
            full_sched = schedule_model(
                globs_pyvm,
                device_idx=dev_idx,
                layer_limit=config.layer_limit,
                interleave_waves=config.interleave_waves,
                interleave_buffer_size=overlap_buffer_size,
                stop_after_op=config.stop_after_op,
            )
            schedules.append(full_sched)
            pre_schedules.append(None)
        
        # Generate instruction tensors for MK
        num_instructions = generate_schedule()

    # Create timing tensors and global instruction index tensors
    for dev_idx in range(config.num_devices):
        times = create_timing_tensor(instruction_tensors[dev_idx])
        timing_tensors.append(times)
        
        global_instruction_index_tensors.append(
            torch.zeros(1, dtype=torch.int32, device=f"cuda:{dev_idx}")
        )

    print(
        f"Schedule contains {len(schedules[0])} instructions (instruction tensor shape {instruction_tensors[0].shape})"
    )

    globs_pyvm.instructions = instruction_tensors
    globs_pyvm.timings = timing_tensors
    globs_pyvm.global_instruction_index = global_instruction_index_tensors

    if config.start_after_op is not None and not config.skip_starting_ops:
        # Run starting instructions on both PyVM globals
        print("\n" + "=" * 80)
        print("Running starting instructions...")
        print("=" * 80)

        start_time = time()
        interpret_instructions_round_robin(
            globs_pyvm, pre_schedules, config.num_devices
        )

        for dev_idx in range(config.num_devices):
            torch.cuda.synchronize(dev_idx)

        starting_time = time() - start_time
        print(f"Starting instructions time: {starting_time:.3f}s")

    # Fork globals after running starting instructions
    print("\nForking globals for MK run...")
    globs_mk = globs_pyvm.with_new_activations()

    # Run PyVM interpreter on remaining instructions
    if not config.skip_pyvm:
        print("\n" + "=" * 80)
        print("Running PyVM...")
        print("=" * 80)

        start_time = time()
        interpret_instructions_round_robin(
            globs_pyvm,
            schedules,
            config.num_devices,
        )

        for dev_idx in range(config.num_devices):
            torch.cuda.synchronize(dev_idx)

        pyvm_time = time() - start_time
        print(f"PyVM time: {pyvm_time:.3f}s")

    # Run MK interpreter
    print("\n" + "=" * 80)
    print("Setting up and running MK interpreter...")
    print("=" * 80)

    for dev_idx in range(config.num_devices):
        torch.cuda.synchronize(dev_idx)

    # breakpoint()

    # For MK, we need to use the full instruction tensors
    interpreter = TensorParallelMK_Interpreter(config.mk_dir, globs_mk)
    interpreter.setup()

    for dev_idx in range(config.num_devices):
        torch.cuda.synchronize(dev_idx)

    start_time = time()

    interpreter.interpret()

    for dev_idx in range(config.num_devices):
        torch.cuda.synchronize(dev_idx)

    mk_time = time() - start_time
    print(f"MK time: {mk_time:.3f}s")

    # Compare results
    print("\n" + "=" * 80)
    print("Comparing PyVM and MK results...")
    print("=" * 80)

    # Compare key tensors based on test mode
    if config.test_mode == "prefill":
        # Prefill-focused comparisons
        print(f"\n=== PREFILL-SPECIFIC TENSOR COMPARISONS ===")
        
        # Determine expected devices with changes for prefill
        if config.prefill_seq_lens is None:
            prefill_seq_lens = [config.seq_len] * config.num_prefill_seqs
        else:
            prefill_seq_lens = config.prefill_seq_lens
            
        total_tokens = sum(prefill_seq_lens)
        batch_size_per_device = total_tokens // config.num_devices
        expected_devices = set()
        for token_idx in range(total_tokens):
            target_device = token_idx // batch_size_per_device
            if target_device < config.num_devices:
                expected_devices.add(target_device)
        
        print(f"Expected devices to have changes: {sorted(expected_devices)}")
        
        # Prefill-specific tensor comparisons
        prefill_tensors_to_compare = [
            ("attn_out", globs_pyvm.attn_out, globs_mk.attn_out),
            ("k_cache", globs_pyvm.k_cache, globs_mk.k_cache),
            ("v_cache", globs_pyvm.v_cache, globs_mk.v_cache),
            ("post_rope_q", globs_pyvm.post_rope_q, globs_mk.post_rope_q),
            ("barriers", globs_pyvm.barriers, globs_mk.barriers),
        ]
        
        for name, pyvm_tensors, mk_tensors in prefill_tensors_to_compare:
            diff_prefill_tensors(name, pyvm_tensors, mk_tensors, config.num_devices, expected_devices)
            
    elif config.test_mode == "decode":
        # Original decode comparisons
        print(f"\n=== DECODE-SPECIFIC TENSOR COMPARISONS ===")
        tensors_to_compare = [
            ("hidden_states", globs_pyvm.hidden_states, globs_mk.hidden_states),
            ("post_attn_norm", globs_pyvm.post_attn_norm, globs_mk.post_attn_norm),
            ("post_rope_q", globs_pyvm.post_rope_q, globs_mk.post_rope_q),
            ("k_cache", globs_pyvm.k_cache, globs_mk.k_cache),
            ("v_cache", globs_pyvm.v_cache, globs_mk.v_cache),
            ("attn_out", globs_pyvm.attn_out, globs_mk.attn_out),
            ("post_mlp_norm", globs_pyvm.post_mlp_norm, globs_mk.post_mlp_norm),
            ("mlp_intermediates", globs_pyvm.mlp_intermediates, globs_mk.mlp_intermediates),
            ("post_lm_head_norm", globs_pyvm.post_lm_head_norm, globs_mk.post_lm_head_norm),
            ("logits", globs_pyvm.logits, globs_mk.logits),
        ]

        for name, pyvm_tensors, mk_tensors in tensors_to_compare:
            diff_tensors(name, pyvm_tensors, mk_tensors, config.num_devices)
            
    else:  # "both"
        # Compare both decode and prefill tensors
        print(f"\n=== COMBINED (DECODE + PREFILL) TENSOR COMPARISONS ===")
        tensors_to_compare = [
            ("hidden_states", globs_pyvm.hidden_states, globs_mk.hidden_states),
            ("post_attn_norm", globs_pyvm.post_attn_norm, globs_mk.post_attn_norm),
            ("post_rope_q", globs_pyvm.post_rope_q, globs_mk.post_rope_q),
            ("k_cache", globs_pyvm.k_cache, globs_mk.k_cache),
            ("v_cache", globs_pyvm.v_cache, globs_mk.v_cache),
            ("attn_out", globs_pyvm.attn_out, globs_mk.attn_out),
            ("post_mlp_norm", globs_pyvm.post_mlp_norm, globs_mk.post_mlp_norm),
            ("mlp_intermediates", globs_pyvm.mlp_intermediates, globs_mk.mlp_intermediates),
            ("post_lm_head_norm", globs_pyvm.post_lm_head_norm, globs_mk.post_lm_head_norm),
            ("logits", globs_pyvm.logits, globs_mk.logits),
            ("barriers", globs_pyvm.barriers, globs_mk.barriers),
        ]

        for name, pyvm_tensors, mk_tensors in tensors_to_compare:
            diff_tensors(name, pyvm_tensors, mk_tensors, config.num_devices)

    print("\n" + "=" * 80)
    print("Diff test complete!")
    print("=" * 80)

    if config.bp:
        breakpoint()


if __name__ == "__main__":
    pydra.run(main)
