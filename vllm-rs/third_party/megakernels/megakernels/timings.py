#!/usr/bin/env python3
"""
Convert PyTorch timing tensors to efficient binary format for PIXI.js visualization.

Input: List of 3D tensors from multiple GPUs
Each tensor shape: (num_sms, num_instructions, 128)
- Variable number of SMs per GPU
- Variable number of instructions per SM  
- 128 timing events per instruction in nanoseconds (index 0 = instruction type/opcode)

Output binary format (.mkprof):
[magic_header]
"MKPROF1.2\n"

[json_config]
[padding]
0-3 bytes of padding to align to 4-byte boundary

[instruction_types] - Int32Array
int32[total_processors * max_instructions]: instruction type for each slot

[start_times] - Uint32Array  
uint32[total_processors * max_instructions]: start time in nanoseconds

[end_times] - Uint32Array
uint32[total_processors * max_instructions]: end time in nanoseconds

[events] - Dense event data
uint32[total_processors * max_instructions * 128]: all event times (indices 0-127)
"""

import struct
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch


def unflatten_tensors(timing_2d: torch.Tensor, instruction_2d: torch.Tensor, num_instructions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert flattened 2D timing and instruction tensors back to 3D format by grouping instructions by SM_ID.
    
    Args:
        timing_2d: 2D tensor of shape (total_instructions, 128)
                   Index 0 = opcode/instruction type
                   Index 1 = SM_ID for each instruction
                   Other indices = timing events in nanoseconds
        instruction_2d: 2D tensor of shape (total_instructions, 32)
                       Full instruction data (32 ints per instruction)
        num_instructions: Actual number of instructions to process (if None, uses tensor.shape[0])
    
    Returns:
        tuple of (timing_3d, instruction_3d): Both 3D tensors grouped by SM_ID
    """
    if timing_2d.dim() != 2 or instruction_2d.dim() != 2:
        raise ValueError(f"Expected 2D tensors, got timing: {timing_2d.dim()}D, instruction: {instruction_2d.dim()}D")
    
    if timing_2d.shape[1] != 128:
        raise ValueError(f"Expected 128 timing events per instruction, got {timing_2d.shape[1]}")
    
    if instruction_2d.shape[1] != 32:
        raise ValueError(f"Expected 32 ints per instruction, got {instruction_2d.shape[1]}")
    
    if timing_2d.shape[0] != instruction_2d.shape[0]:
        raise ValueError(f"Timing and instruction tensors must have same number of instructions: {timing_2d.shape[0]} vs {instruction_2d.shape[0]}")
    
    total_instructions, num_timing_events = timing_2d.shape
    _, num_instruction_ints = instruction_2d.shape
    
    # Only process the first num_instructions
    timing_subset = timing_2d[:num_instructions]
    instruction_subset = instruction_2d[:num_instructions]
    
    # Extract SM_IDs from timing data (index 1)
    sm_ids = timing_subset[:, 1].long()  # Convert to long for indexing
    
    # Find unique SM_IDs and count instructions per SM
    unique_sm_ids, counts = torch.unique(sm_ids, return_counts=True)
    max_sm_id = unique_sm_ids.max().item()
    max_instructions_per_sm = counts.max().item()
    num_sms = max_sm_id + 1  # SM_IDs are 0-indexed
    
    # Create 3D tensors: (num_sms, max_instructions_per_sm, events/ints)
    timing_3d = torch.zeros(num_sms, max_instructions_per_sm, num_timing_events, dtype=timing_2d.dtype, device=timing_2d.device)
    instruction_3d = torch.zeros(num_sms, max_instructions_per_sm, num_instruction_ints, dtype=instruction_2d.dtype, device=instruction_2d.device)
    
    # Group both timing and instruction data by SM_ID using the same logic
    instructions_per_sm = []
    for sm_id in unique_sm_ids:
        # Find all instructions for this SM
        sm_mask = (sm_ids == sm_id)
        sm_timing_data = timing_subset[sm_mask]  # Shape: (num_instructions_for_this_sm, 128)
        sm_instruction_data = instruction_subset[sm_mask]  # Shape: (num_instructions_for_this_sm, 32)
        
        # Place them in the 3D tensors at the same positions
        num_instructions_for_sm = sm_timing_data.shape[0]
        timing_3d[sm_id, :num_instructions_for_sm, :] = sm_timing_data
        instruction_3d[sm_id, :num_instructions_for_sm, :] = sm_instruction_data
        instructions_per_sm.append(num_instructions_for_sm)
    
    return timing_3d, instruction_3d, max(instructions_per_sm)


def unflatten_timing_and_instruction_tensors(timing_tensors_2d: List[torch.Tensor], instruction_tensors_2d: List[torch.Tensor], num_instructions: List[int]) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Convert lists of flattened 2D timing and instruction tensors back to 3D format.
    
    Args:
        timing_tensors_2d: List of 2D timing tensors, one per GPU
        instruction_tensors_2d: List of 2D instruction tensors, one per GPU
        num_instructions: List of actual number of instructions per GPU (if None, uses tensor shapes)
    
    Returns:
        tuple of (timing_tensors_3d, instruction_tensors_3d): Both converted to 3D format
    """
    timing_tensors_3d = []
    instruction_tensors_3d = []
    
    for gpu_id, (timing_2d, instruction_2d) in enumerate(zip(timing_tensors_2d, instruction_tensors_2d)):
        timing_3d, instruction_3d, num_instructions[gpu_id] = unflatten_tensors(timing_2d, instruction_2d, num_instructions[gpu_id])
        timing_tensors_3d.append(timing_3d)
        instruction_tensors_3d.append(instruction_3d)
    
    return timing_tensors_3d, instruction_tensors_3d


def extract_timing_data(timing_tensors: List[torch.Tensor], instruction_tensors: List[torch.Tensor], num_instructions: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract timing data keeping original nanosecond values.
    
    Args:
        timing_tensors: List of 3D tensors from each GPU (timing data in nanoseconds)
        instruction_tensors: List of instruction tensors with full instruction data (32 ints per instruction)
        num_instructions: List of actual number of instructions per GPU (if None, uses tensor.shape[1])
    
    Returns:
        instructions: int32 array of full instruction data [total_processors * max_instructions * 32]
        start_times: uint32 array of start times in nanoseconds (index 6)
        end_times: uint32 array of end times in nanoseconds (index 7) 
        all_events: uint32 array of all event times [total_processors * max_instructions * 128]
    """
    if not timing_tensors:
        raise ValueError("No timing tensors provided")
    
    # Find maximum instruction count across all GPUs
    max_instructions = max(num_instructions)
    total_processors = sum(tensor.shape[0] for tensor in timing_tensors)
    
    # Instruction data is always 32 ints per instruction
    instruction_size = 32
    
    # Preallocate arrays
    instructions = np.zeros(total_processors * max_instructions * instruction_size, dtype=np.int32)
    start_times = np.zeros(total_processors * max_instructions, dtype=np.uint32)
    end_times = np.zeros(total_processors * max_instructions, dtype=np.uint32)
    all_events = np.zeros(total_processors * max_instructions * 128, dtype=np.uint32)
    
    proc_offset = 0
    
    for gpu_id, tensor in enumerate(timing_tensors):
        num_sms, tensor_max_instrs, num_events = tensor.shape
        gpu_num_instrs = num_instructions[gpu_id]  # Actual number of instructions for this GPU
        
        # Keep timing data as uint32 (nanoseconds)
        timings_np = tensor.cpu().numpy().astype(np.uint32)
        
        # Get corresponding instruction tensor
        instruction_tensor = instruction_tensors[gpu_id].cpu().numpy().astype(np.int32)
        
        # Process each SM individually to handle padding correctly
        for sm_idx in range(num_sms):
            proc_idx = proc_offset + sm_idx
            
            # Calculate start index for this processor in the flat arrays
            flat_start = proc_idx * max_instructions
            
            # Extract data for this SM (only up to gpu_num_instrs)
            sm_data = timings_np[sm_idx, :gpu_num_instrs, :]  # Shape: (gpu_num_instrs, 128)
            
            # Extract full instruction data (32 ints per instruction)
            sm_instruction_data = instruction_tensor[sm_idx, :gpu_num_instrs, :]  # Shape: (gpu_num_instrs, 32)
            instr_start = flat_start * 32
            instr_end = instr_start + gpu_num_instrs * 32
            instructions[instr_start:instr_end] = sm_instruction_data.flatten()
            
            # Extract start times (index 6)
            start_times[flat_start:flat_start + gpu_num_instrs] = sm_data[:, 6]
            
            # Extract end times (index 7)
            end_times[flat_start:flat_start + gpu_num_instrs] = sm_data[:, 7]
            
            # Extract all events (0-127)
            event_start = flat_start * 128
            event_end = event_start + gpu_num_instrs * 128
            all_events[event_start:event_end] = sm_data.flatten()
        
        proc_offset += num_sms
    
    return instructions, start_times, end_times, all_events


def get_default_config():
    """Get default configuration for .mkprof files."""
    return {
        "format_version": "1.2",
        "instruction_types": {
            "0": {"name": "No Op", "color": "#808080", "params": {}},
            "1": {"name": "Attn Norm", "color": "#1f77b4", "params": {
                "1": "layer_idx", "2": "local_batch_indices_len", "3": "local_batch_indices[0]", "4": "local_batch_indices[1]", "5": "local_batch_indices[2]", "6": "local_batch_indices[3]"
            }},
            "2": {"name": "QKV Rope", "color": "#ff7f0e", "params": {
                "1": "layer_idx", "2": "local_batch_block_idx", "3": "local_output_block_idx", "4": "global_batch_block_idx", "5": "global_output_block_idx"
            }},
            "3": {"name": "GQA Prefill", "color": "#2ca02c", "params": {
                "1": "layer_idx", "2": "prefill_seq_idx", "3": "prefill_block_idx", "4": "kv_head_idx"
            }},
            "4": {"name": "GQA Decode", "color": "#2ca02c", "params": {
                "1": "layer_idx",
                "2": "(2)num_seq",
                "3": "seq_idx[0]",
                "4": "kv_head[0]",
                "5": "seq_idx[1]",
                "6": "kv_head[1]",
                "7": "seq_idx[2]",
                "8": "kv_head[2]",
                "9": "seq_idx[3]",
                "10": "kv_head[3]",
                "11": "seq_idx[4]",
                "12": "kv_head[4]",
                "13": "seq_idx[5]",
                "14": "kv_head[5]",
                "15": "seq_idx[6]",
                "16": "kv_head[6]",
                "17": "seq_idx[7]",
                "18": "kv_head[7]"
            }},
            "5": {"name": "O Proj", "color": "#d62728", "params": {
                "1": "layer_idx", "2": "local_batch_block_idx", "3": "local_output_block_idx", "4": "global_batch_block_idx", "5": "global_output_block_idx"
            }},
            "6": {"name": "MLP Norm", "color": "#9467bd", "params": {
                "1": "layer_idx", "2": "local_batch_indices_len", "3": "local_batch_indices[0]", "4": "local_batch_indices[1]", "5": "local_batch_indices[2]", "6": "local_batch_indices[3]"
            }},
            "7": {"name": "Gate SiLU", "color": "#8c564b", "params": {
                "1": "layer_idx", "2": "local_batch_block_idx", "3": "local_output_block_idx", "4": "global_batch_block_idx", "5": "global_output_block_idx"
            }},
            "8": {"name": "Up Matmul", "color": "#e377c2", "params": {
                "1": "layer_idx", "2": "local_batch_block_idx", "3": "local_output_block_idx", "4": "global_batch_block_idx", "5": "global_output_block_idx"
            }},
            "9": {"name": "Down Proj", "color": "#7f7f7f", "params": {
                "1": "layer_idx", "2": "local_batch_block_idx", "3": "local_output_block_idx", "4": "global_batch_block_idx", "5": "global_output_block_idx"
            }},
            "10": {"name": "LM Head Norm", "color": "#bcbd22", "params": {
                "1": "layer_idx", "2": "local_batch_indices_len", "3": "local_batch_indices[0]", "4": "local_batch_indices[1]", "5": "local_batch_indices[2]", "6": "local_batch_indices[3]"
            }},
            "11": {"name": "LM Head", "color": "#17becf", "params": {
                "1": "layer_idx", "2": "local_batch_block_idx", "3": "local_output_block_idx", "4": "global_batch_block_idx", "5": "global_output_block_idx"
            }},
            "12": {"name": "Inc Barriers", "color": "#ff1493", "params": {}}
        },
        "instruction_format": {
            "instruction_length": 32,
            "timing_length": 128
        },
        "functional_units": {
            "0": {
                "name": "Loader",
                "event_range": {"start": 16, "end": 47},
                "height_multiplier": 1.0,
                "special_events": {"start": 8, "end": 9}
            },
            "1": {
                "name": "Consumer", 
                "event_range": {"start": 48, "end": 79},
                "height_multiplier": 2.0,
                "special_events": {"start": 12, "end": 13}
            },
            "2": {
                "name": "Storer",
                "event_range": {"start": 112, "end": 127},
                "height_multiplier": 1.0,
                "special_events": {"start": 14, "end": 15}
            },
            "3": {
                "name": "Launcher",
                "event_range": {"start": 80, "end": 111},
                "height_multiplier": 1.0,
                "special_events": {"start": 10, "end": 11}
            }
        },
        "main_functional_unit": 1,
        "event_types": {
            "0": {"name": "LOAD_EVENT", "color": "#0000ff"},
            "1": {"name": "LOAD2_EVENT", "color": "#00ffff"},
            "2": {"name": "COMPUTE_EVENT", "color": "#aa00ff"},
            "3": {"name": "STORE_EVENT", "color": "#ffff00"},
            "4": {"name": "STORE2_EVENT", "color": "#ff8000"},
            "5": {"name": "WAIT_EVENT", "color": "#ff0000"},
            "6": {"name": "READY_EVENT", "color": "#00ff00"},
            "7": {"name": "ERROR_EVENT", "color": "#000000"}
        },
        "controller_events": {
            "5": {"name": "CONTROLLER_START", "color": "#FA8072"},
            "6": {"name": "CONTROLLER_READY", "color": "#32CD32"},
            "7": {"name": "CONTROLLER_CLEANUP", "color": "#FFFFFF"}
        },
        "num_gpus": 8,
        "total_processors": (8*132),
        "max_instructions": 1,
        "time_unit_flag": 1,
        "has_events_flag": 1
    }


def write_binary_format(output_file: str, 
                        instructions: np.ndarray,
                        start_times: np.ndarray, 
                        end_times: np.ndarray,
                        all_events: np.ndarray,
                        num_gpus: int,
                        total_processors: int,
                        max_instructions: int,
                        config: dict = None):
    """Write data in the binary format expected by PIXI.js."""
    
    if config is None:
        config = get_default_config()
    
    with open(output_file, 'wb') as f:
        # Write magic header
        magic_header = "MKPROF1.2\n"
        f.write(magic_header.encode('utf-8'))
        
        # Write JSON configuration (now includes binary header info)
        import json
        config["num_gpus"] = num_gpus
        config["total_processors"] = total_processors
        config["max_instructions"] = max_instructions
        config["time_unit_flag"] = 1
        config["has_events_flag"] = 1
        json_str = json.dumps(config, separators=(',', ':'))  # Compact JSON
        f.write(json_str.encode('utf-8'))
        
        # Add padding to align int32 arrays to 4-byte boundaries
        current_pos = f.tell()
        padding_needed = (4 - (current_pos % 4)) % 4
        if padding_needed > 0:
            f.write(b'\x00' * padding_needed)
        
        # Instructions (int32 array) - directly after JSON, no binary header
        f.write(instructions.tobytes())
        
        # Start times (uint32 array) 
        f.write(start_times.tobytes())
        
        # End times (uint32 array)
        f.write(end_times.tobytes())
        
        # All events (uint32 array) - dense format
        f.write(all_events.tobytes())
    
    file_size = Path(output_file).stat().st_size
    print(f"Binary file written: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")


def timings_to_mkprof(timing_tensors: List[torch.Tensor], instruction_tensors: List[torch.Tensor], num_instructions: List[int] | None, output_filename: str, config: dict = None):
    """
    Convert timing and instruction tensors to .mkprof binary format.
    
    Args:
        timing_tensors: List of timing tensors, one per GPU. Can be either:
                 - 2D tensors: Shape (total_instructions, 128) - will be automatically unflattened
                   Index 0 = opcode/instruction type
                   Index 1 = SM_ID for each instruction
                   Other indices = timing events in nanoseconds
                 - 3D tensors: Shape (num_sms, num_instructions, 128)
                   Index 0 = opcode/instruction type
                   Index 6 = start time in nanoseconds
                   Index 7 = end time in nanoseconds
                   Indices 8-127 = all event times in nanoseconds
        instruction_tensors: List of instruction tensors with full instruction data (shape: num_sms, num_instructions, 32)
        num_instructions: List of number of instructions per GPU. If None, will be inferred from instruction_tensors.
        output_filename: Target filename for .mkprof output
        config: Optional configuration dict. If None, uses default config.
    """
    output_filename = Path(output_filename).with_suffix(".mkprof")

    try:
        # Handle tensor format conversion
        timing_is_2d = timing_tensors and timing_tensors[0].dim() == 2
        instruction_is_2d = instruction_tensors and instruction_tensors[0].dim() == 2
        
        if timing_is_2d and instruction_is_2d:
            print("Detected 2D tensors (global work queue format) - unflattening both to 3D format...")
            if num_instructions is None:
                num_instructions = [tensor.shape[0] for tensor in instruction_tensors]
            timing_tensors, instruction_tensors = unflatten_timing_and_instruction_tensors(timing_tensors, instruction_tensors, num_instructions)
        elif timing_is_2d and not instruction_is_2d:
            raise ValueError("Timing tensors are 2D but instruction tensors are not - incompatible formats")
        elif not timing_is_2d and instruction_is_2d:
            raise ValueError("Instruction tensors are 2D but timing tensors are not - incompatible formats")
        elif timing_tensors and timing_tensors[0].dim() == 3 and instruction_tensors[0].dim() == 3:
            if num_instructions is None:
                num_instructions = [tensor.shape[1] for tensor in instruction_tensors]
            print("Using 3D tensors directly...")
        else:
            raise ValueError(f"Unsupported tensor dimensions. Expected 2D or 3D tensors, got timing: {timing_tensors[0].dim()}D, instruction: {instruction_tensors[0].dim()}D")
        
        # Extract timing data
        instructions, start_times, end_times, all_events = extract_timing_data(timing_tensors, instruction_tensors, num_instructions)
        
        # Calculate dimensions
        total_processors = sum(tensor.shape[0] for tensor in timing_tensors) 
        max_instructions = max(num_instructions)
        num_gpus = len(timing_tensors)
        
        # Write binary file
        write_binary_format(
            output_filename,
            instructions, 
            start_times,
            end_times, 
            all_events,
            num_gpus,
            total_processors,
            max_instructions,
            config
        )
        
        print(f"Success! .mkprof file written to '{output_filename}'")
        
    except Exception as e:
        print(f"Error converting tensors to mkprof: {e}")
        raise
