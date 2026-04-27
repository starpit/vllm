"""
Debug mode for reproducing stalls and other issues from pickle dumps.

This module contains debug functionality that was originally part of tp_generate.py
but has been extracted into a separate file for better organization.
"""

import math
import pickle
import sys
import types
from pathlib import Path
from time import time

import torch
from transformers import AutoConfig

from megakernels.demos.tp_throughput.cpp_scheduler import scheduler_cpp
from megakernels.demos.tp_throughput.globs import make_globals
from megakernels.demos.tp_throughput.mk import TensorParallelMK_Interpreter
from megakernels.demos.tp_throughput.scheduler import (
    # create_instruction_tensor,
    create_all_instruction_tensors,
    init_random_weights,
    load_weights,
    setup_rope_and_interleave,
)
from megakernels.utils import get_sm_count


def sync_all_devices(num_devices: int):
    """Synchronize all CUDA devices."""
    for dev_idx in range(num_devices):
        torch.cuda.synchronize(dev_idx)


def run_debug_mode(config):
    """
    Debug mode that loads metadata from a pickle file and repeatedly runs the kernel
    to reproduce stalls or other issues.
    """
    print(f"=== DEBUG MODE ===")
    print(f"Loading debug data from: {config.debug_pickle}")

    # Verify file exists
    if not config.debug_pickle.exists():
        print(f"ERROR: Debug pickle file not found: {config.debug_pickle}")
        return 1

    # Try to load the pickle file, handling potential module dependencies
    import sys
    import types

    # Create a more sophisticated dummy module to handle pickle imports
    class DummyModule(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__file__ = "<dummy>"
            self.__loader__ = None
            self.__package__ = name.rpartition('.')[0]
            self.__spec__ = None
            self.__path__ = []  # Important for package modules

        def __getattr__(self, name):
            # Return a dummy class that can be instantiated
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    # Store kwargs as attributes for potential later access
                    for k, v in kwargs.items():
                        setattr(self, k, v)

                def __getattr__(self, attr):
                    return None

                def __repr__(self):
                    return f"DummyClass({name})"
            return DummyClass

    # Temporarily add dummy modules to handle missing tokasaurus imports
    tokasaurus_modules = [
        'tokasaurus',
        'tokasaurus.model',
        'tokasaurus.model.types',
        'tokasaurus.model.debug_dump',
        'tokasaurus.common_types',
        'tokasaurus.utils',
        'megakernels.demos.tp_throughput.globs',  # In case the pickle references this
    ]

    original_modules = {}
    for module_name in tokasaurus_modules:
        if module_name not in sys.modules:
            dummy_mod = DummyModule(module_name)
            sys.modules[module_name] = dummy_mod
            original_modules[module_name] = None
            # Also set __path__ for package modules
            if '.' in module_name:
                parent_name = module_name.rsplit('.', 1)[0]
                if parent_name in sys.modules:
                    parent_mod = sys.modules[parent_name]
                    child_name = module_name.split('.')[-1]
                    setattr(parent_mod, child_name, dummy_mod)
        else:
            original_modules[module_name] = sys.modules[module_name]

    debug_data = None
    load_error = None

    # Try multiple loading strategies
    try:
        with open(config.debug_pickle, 'rb') as f:
            debug_data = pickle.load(f)
        print("Successfully loaded pickle file")
    except Exception as e:
        load_error = str(e)
        print(f"Standard load failed: {e}")

        # Try custom unpickler
        try:
            print("Trying custom unpickler...")

            class SafeUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Handle special cases
                    if module.startswith('tokasaurus'):
                        # Create a dummy class that preserves the data
                        class PreservingDummy:
                            def __init__(self, *args, **kwargs):
                                self.__dict__.update(kwargs)
                            def __reduce__(self):
                                return (dict, (self.__dict__,))
                        return PreservingDummy

                    try:
                        return super().find_class(module, name)
                    except (ImportError, AttributeError, ModuleNotFoundError):
                        # Return a dummy class for missing modules
                        class DummyClass:
                            def __init__(self, *args, **kwargs):
                                self.__dict__.update(kwargs)
                        return DummyClass

            with open(config.debug_pickle, 'rb') as f:
                debug_data = SafeUnpickler(f).load()
            print("Successfully loaded with custom unpickler")
        except Exception as e2:
            print(f"Custom unpickler also failed: {e2}")

            # Last resort: try to extract just the dictionary data
            try:
                print("Attempting minimal data extraction...")
                with open(config.debug_pickle, 'rb') as f:
                    # Read raw bytes and try to find the metadata dict
                    import re
                    content = f.read()
                    # This is a hack but might work for simple cases
                    # Try loading with protocol 2 for better compatibility
                    f.seek(0)
                    debug_data = pickle.load(f, encoding='bytes')
                print("Loaded with bytes encoding")
            except:
                pass

    # Clean up dummy modules
    for module_name, original in original_modules.items():
        if original is None and module_name in sys.modules:
            del sys.modules[module_name]
        elif original is not None:
            sys.modules[module_name] = original

    # Check if we successfully loaded data
    if debug_data is None:
        print(f"\nERROR: Failed to load pickle file after all attempts")
        print(f"Original error: {load_error}")
        print("\nTry regenerating the debug dump with the updated debug_dump.py")
        return 1

    # Validate that we have the expected structure
    if not isinstance(debug_data, dict):
        print(f"ERROR: Unexpected data structure. Expected dict, got {type(debug_data)}")
        return 1

    if "metadata" not in debug_data:
        print("ERROR: No 'metadata' key in debug data")
        print(f"Available keys: {list(debug_data.keys())}")
        return 1

    # Extract metadata
    metadata = debug_data["metadata"]
    if isinstance(metadata, dict):
        # Convert dict to object for easier access
        class MetadataHolder:
            def __init__(self, data):
                for k, v in data.items():
                    setattr(self, k, v)
        metadata = MetadataHolder(metadata)

    # Simple attribute getter
    def get_attr(obj, name, default=None):
        if hasattr(obj, name):
            return getattr(obj, name)
        elif isinstance(obj, dict):
            return obj.get(name, default)
        return default

    print(f"\nDebug dump info:")
    print(f"  Run ID: {get_attr(metadata, 'run_id', 'unknown')}")
    print(f"  Timestamp: {get_attr(metadata, 'timestamp', 'unknown')}")
    print(f"  Notes: {get_attr(metadata, 'notes', 'none')}")

    # Critical scheduling parameters
    print(f"\nScheduling parameters:")
    print(f"  Global batch size: {get_attr(metadata, 'global_batch_size', 'unknown')}")
    print(f"  Num pages: {get_attr(metadata, 'num_pages', 'unknown')}")
    print(f"  Global work queue: {get_attr(metadata, 'global_work_queue_enabled', 'unknown')}")
    print(f"  Interleave waves: {get_attr(metadata, 'interleave_waves', 'unknown')}")

    # Token information
    num_prefill_tokens = get_attr(metadata, 'num_prefill_tokens', 0)
    num_decode_tokens = get_attr(metadata, 'num_decode_tokens', 0)
    prefill_chunk_lens = get_attr(metadata, 'prefill_chunk_lens', [])
    prefill_extend_offsets = get_attr(metadata, 'prefill_extend_offsets', [])
    decode_seq_lens = get_attr(metadata, 'decode_sequence_lengths', [])

    print(f"\nToken info:")
    print(f"  Prefill tokens: {num_prefill_tokens}")
    print(f"  Decode tokens: {num_decode_tokens}")
    if prefill_chunk_lens:
        print(f"  Prefill chunk lengths: {prefill_chunk_lens}")
    if prefill_extend_offsets:
        print(f"  Prefill extend offsets: {prefill_extend_offsets}")
    if decode_seq_lens:
        print(f"  Decode sequence lengths: {decode_seq_lens}")

    # Verification data
    expected_counts = get_attr(metadata, 'instruction_counts_per_gpu', [])
    if expected_counts:
        print(f"\nVerification data:")
        print(f"  Instructions per GPU: {expected_counts}")

    # Override config with values from debug dump
    glob_bs = get_attr(metadata, 'global_batch_size', None)
    if not glob_bs:
        print(f"ERROR: Missing global_batch_size in debug dump!")
        return 1
    config.glob_bs = glob_bs

    num_pages = get_attr(metadata, 'num_pages', None)
    if not num_pages:
        print(f"ERROR: Missing num_pages in debug dump!")
        return 1
    config.num_pages = num_pages

    config.global_work_queue = get_attr(metadata, 'global_work_queue_enabled', True)
    config.interleave_waves = get_attr(metadata, 'interleave_waves', True)

    print(f"\nUsing config from debug dump:")
    print(f"  glob_bs: {config.glob_bs}")
    print(f"  num_pages: {config.num_pages}")
    print(f"  global_work_queue: {config.global_work_queue}")
    print(f"  interleave_waves: {config.interleave_waves}")

    # Set up model config
    model_config = AutoConfig.from_pretrained(config.model)

    print("\nMaking globals:")
    globs = make_globals(
        model_config=model_config,
        global_batch_size=config.glob_bs,
        num_pages=config.num_pages,
        num_devices=config.num_devices,
        barrier_init_val=config.bar_init_val,
        global_work_queue_enabled=config.global_work_queue,
        timing_record_enabled=config.force_enable_timing,
        layer_limit=config.num_layer_override,
    )

    # Set globs sizes using the captured data
    prefill_chunk_lens = get_attr(metadata, 'prefill_chunk_lens', [])
    prefill_extend_offsets = get_attr(metadata, 'prefill_extend_offsets', [])
    decode_seq_lens = get_attr(metadata, 'decode_sequence_lengths', [])

    print(f"\nSetting globs.set_sizes:")
    print(f"  global_batch_size: {config.glob_bs}")
    if prefill_chunk_lens:
        print(f"  prefill_chunk_lens: {prefill_chunk_lens}")
    if prefill_extend_offsets:
        print(f"  prefill_extend_offsets: {prefill_extend_offsets}")
    if decode_seq_lens:
        print(f"  decode_sequence_lengths: {decode_seq_lens}")

    # Call set_sizes with the captured prefill data
    set_sizes_params = {'global_batch_size': config.glob_bs}
    if prefill_chunk_lens:
        set_sizes_params['prefill_chunk_lens'] = prefill_chunk_lens
        set_sizes_params['prefill_extend_offsets'] = prefill_extend_offsets or [0] * len(prefill_chunk_lens)

    globs.set_sizes(**set_sizes_params)

    # Store decode sequence lengths for debugging
    if decode_seq_lens:
        globs._debug_decode_seq_lens = decode_seq_lens

    # Debug: show the final globs state that affects scheduling
    print(f"\nGlobs state after set_sizes:")
    print(f"  global_batch_size: {globs.global_batch_size}")
    print(f"  prefill_chunk_lens: {getattr(globs, 'prefill_chunk_lens', [])}")
    print(f"  prefill_extend_offsets: {getattr(globs, 'prefill_extend_offsets', [])}")

    # Show decode sequence lengths if we stored them
    if hasattr(globs, '_debug_decode_seq_lens'):
        print(f"  decode_sequence_lengths (from debug dump): {globs._debug_decode_seq_lens}")
    else:
        print(f"  decode_sequence_lengths: Not available in debug dump")

    # Show computed token counts
    try:
        if hasattr(globs, 'num_prefill_tokens'):
            print(f"  num_prefill_tokens(): {globs.num_prefill_tokens()}")
    except Exception as e:
        print(f"  num_prefill_tokens(): Error - {e}")

    try:
        if hasattr(globs, 'num_decode_seqs'):
            print(f"  num_decode_seqs(): {globs.num_decode_seqs()}")
    except Exception as e:
        print(f"  num_decode_seqs(): Error - {e}")

    print(f"page_size: {globs.page_size}")

    # Load weights
    if not config.skip_weight_load:
        print("Loading model weights:")
        start_time = time()
        load_weights(config.model, globs, layer_limit=config.num_layer_override)
        end_time = time()
        print(f"Time taken to load weights: {end_time - start_time} seconds")
    elif config.use_random_weights:
        print("Initializing random weights:")
        init_random_weights(globs)
    else:
        print("Not initializing weights (probably using zero weights)")

    setup_rope_and_interleave(globs)

    print("Setting up interpreter:")
    interpreter = TensorParallelMK_Interpreter(config.mk_dir, globs)
    interpreter.setup()

    sync_all_devices(config.num_devices)

    # Generate dummy schedule based on metadata
    print("\nGenerating schedule based on debug metadata...")

    if config.interleave_waves:
        sm_count = get_sm_count("cuda:0")
        overlap_buffer_size = round(sm_count * config.interleave_buffer_factor)
    else:
        overlap_buffer_size = None

    if config.cpp_sched:
        assert scheduler_cpp is not None
        sched_func = scheduler_cpp.create_all_instruction_tensors
    else:
        raise ValueError("Only cpp scheduler is supported for debug mode")
        # sched_func = create_all_instruction_tensors

    #  = []
    instruction_tensors = sched_func(
        globs,
        # device_idx=dev_idx,
        layer_limit=config.num_layer_override,
        interleave_waves=config.interleave_waves,
        interleave_buffer_size=overlap_buffer_size,
        stop_after_op=config.stop_after_op,
        zero_init=False,
        num_threads=64,
        disable_lm_head=True,
    )
    for dev_idx in range(config.num_devices):
            # all_insts_tensors = scheduler_cpp.create_all_instruction_tensors(
            #     self.globs,
            #     interleave_waves=self.config.megakernel_interleave_waves,
            #     interleave_buffer_size=self.sm_count,
            #     disable_lm_head=True,
            #     zero_init=False,
            #     num_threads=64,
            #     layer_limit=self.config.num_layer_override,
            #     stop_after_op=self.config.stop_after_op,
            #     # move_to_gpu=False,
            # )
        # instruction_tensors.append(insts)
        insts = instruction_tensors[dev_idx]
        print(f"Instruction tensor for device {dev_idx} has shape {insts.shape}")

        # Count opcodes for this GPU
        opcode_counts = {}
        for i in range(insts.shape[0]):
            opcode = insts[i, 0].item()  # First int of each instruction is the opcode
            opcode_counts[opcode] = opcode_counts.get(opcode, 0) + 1

        print(f"GPU {dev_idx} computed opcode counts: {dict(sorted(opcode_counts.items()))}")

        # Display opcode counts from the debug dump if available
        expected_opcode_counts = get_attr(metadata, 'opcode_counts_per_gpu', [])
        if expected_opcode_counts and dev_idx < len(expected_opcode_counts):
            print(f"GPU {dev_idx} debug dump opcode counts: {expected_opcode_counts[dev_idx]}")

            # Compare the counts
            if opcode_counts != expected_opcode_counts[dev_idx]:
                print(f"  WARNING: Opcode counts mismatch for GPU {dev_idx}")
                all_opcodes = set(opcode_counts.keys()) | set(expected_opcode_counts[dev_idx].keys())
                for opcode in sorted(all_opcodes):
                    computed = opcode_counts.get(opcode, 0)
                    expected = expected_opcode_counts[dev_idx].get(opcode, 0)
                    if computed != expected:
                        print(f"    Opcode {opcode}: computed={computed}, expected={expected}")

    # Verify instruction counts match the debug dump
    generated_counts = [t.shape[0] for t in instruction_tensors]
    expected_counts = get_attr(metadata, 'instruction_counts_per_gpu', [])

    if expected_counts:
        print(f"\nInstruction count verification:")
        print(f"Expected counts per GPU: {expected_counts}")
        print(f"Generated counts per GPU: {generated_counts}")

        matches = True
        for i, (expected, generated) in enumerate(zip(expected_counts, generated_counts)):
            if expected != generated:
                print(f"❌ GPU {i}: Expected {expected}, got {generated} (diff: {generated - expected})")
                matches = False
            else:
                print(f"✅ GPU {i}: {expected} instructions (match)")

        if not matches:
            print("\n⚠️  WARNING: Instruction counts don't match original dump!")
            print("This may indicate the schedule isn't being reproduced exactly.")

            # Ask user if they want to continue
            if config.debug_verbose:
                print("Continuing anyway for debugging purposes...")
        else:
            print("\n✅ All instruction counts match - schedule should be identical!")
    else:
        print("\nNo expected instruction counts in debug dump - unable to verify schedule reproduction")
        print(f"Generated counts per GPU: {generated_counts}")

    globs.copy_instructions(instruction_tensors)

    # breakpoint()

    # Set up inputs based on batch size
    print("\nSetting up inputs...")

    # Use the actual tensor sizes from the globals to avoid size mismatches
    actual_batch_size = globs.hidden_states[0].shape[0]
    actual_pos_size = globs.position_ids[0].shape[0]

    print(f"Globals expect batch_size={actual_batch_size}, position_size={actual_pos_size}")

    # Create position IDs with the right size
    position_ids = torch.arange(min(actual_pos_size, config.glob_bs), dtype=torch.long)
    for dev_idx in range(config.num_devices):
        globs.copy_position_ids(position_ids, dev_idx)

    # Create dummy hidden states with the right size
    for dev_idx in range(config.num_devices):
        dummy_hidden = torch.randn(
            actual_batch_size,  # Use the actual expected batch size
            model_config.hidden_size,
            dtype=torch.bfloat16,
            device=f"cuda:{dev_idx}"
        )
        globs.copy_hidden_states(dummy_hidden, dev_idx)

    # Check if we have actual KV paging data from the debug dump
    kv_paging_data = debug_data.get("kv_paging_data", {})

    if kv_paging_data and any(kv_paging_data.get(key) for key in ["decode_kv_indices", "prefill_kv_indices", "append_indices"]):
        print("\n✅ Using actual KV paging data from debug dump (broadcasting from single GPU to all)")

        # Use append indices from debug dump if available - broadcast from single GPU to all
        if kv_paging_data.get("append_indices"):
            append_indices = torch.tensor(kv_paging_data["append_indices"], dtype=torch.int32)
            for dev_idx in range(config.num_devices):
                globs.copy_append_indices(append_indices, dev_idx)
            print(f"  Broadcast {len(append_indices)} append indices to all devices")

        # Use decode KV paging data if available - broadcast from single GPU to all
        if kv_paging_data.get("decode_kv_indices"):
            raw_kv_indices = torch.tensor(kv_paging_data["decode_kv_indices"], dtype=torch.int32)
            raw_kv_indptr = torch.tensor(kv_paging_data["decode_kv_indptr"], dtype=torch.int32) if kv_paging_data.get("decode_kv_indptr") else None
            raw_kv_last_page_len = torch.tensor(kv_paging_data["decode_kv_last_page_len"], dtype=torch.int32) if kv_paging_data.get("decode_kv_last_page_len") else None

            if raw_kv_indptr is not None and raw_kv_last_page_len is not None:
                # Check if the data needs padding or truncation
                padded_batch = globs.global_batch_size
                max_pages_per_seq = 1024  # This matches the hardcoded value in globs.py
                expected_indices_size = padded_batch * max_pages_per_seq

                # Handle kv_indices - it might already be the full size or need padding
                if raw_kv_indices.shape[0] == expected_indices_size:
                    # Already the right size, just use it directly
                    kv_indices = raw_kv_indices
                elif raw_kv_indices.shape[0] < expected_indices_size:
                    # Need to pad
                    kv_indices = torch.zeros(expected_indices_size, dtype=torch.int32)
                    kv_indices[:raw_kv_indices.shape[0]] = raw_kv_indices
                else:
                    # Too large, truncate
                    kv_indices = raw_kv_indices[:expected_indices_size]

                # Handle kv_indptr - should be (batch_size + 1)
                expected_indptr_size = padded_batch + 1
                if raw_kv_indptr.shape[0] == expected_indptr_size:
                    kv_indptr = raw_kv_indptr
                elif raw_kv_indptr.shape[0] < expected_indptr_size:
                    kv_indptr = torch.zeros(expected_indptr_size, dtype=torch.int32)
                    kv_indptr[:raw_kv_indptr.shape[0]] = raw_kv_indptr
                    # Fill remaining entries with the last value
                    if raw_kv_indptr.shape[0] > 0:
                        kv_indptr[raw_kv_indptr.shape[0]:] = raw_kv_indptr[-1]
                else:
                    kv_indptr = raw_kv_indptr[:expected_indptr_size]

                # Handle kv_last_page_len - should be batch_size
                expected_page_len_size = padded_batch
                if raw_kv_last_page_len.shape[0] == expected_page_len_size:
                    kv_last_page_len = raw_kv_last_page_len
                elif raw_kv_last_page_len.shape[0] < expected_page_len_size:
                    kv_last_page_len = torch.zeros(expected_page_len_size, dtype=torch.int32)
                    kv_last_page_len[:raw_kv_last_page_len.shape[0]] = raw_kv_last_page_len
                else:
                    kv_last_page_len = raw_kv_last_page_len[:expected_page_len_size]

                for dev_idx in range(config.num_devices):
                    globs.copy_decode_info(
                        kv_indices=kv_indices,
                        kv_indptr=kv_indptr,
                        kv_last_page_len=kv_last_page_len,
                        device_idx=dev_idx,
                    )
                print(f"  Broadcast decode KV paging to all devices:")
                print(f"    kv_indices: {raw_kv_indices.shape[0]} -> {kv_indices.shape[0]} (expected: {expected_indices_size})")
                print(f"    kv_indptr: {raw_kv_indptr.shape[0]} -> {kv_indptr.shape[0]} (expected: {expected_indptr_size})")
                print(f"    kv_last_page_len: {raw_kv_last_page_len.shape[0]} -> {kv_last_page_len.shape[0]} (expected: {expected_page_len_size})")

        # Use prefill KV paging data if available - broadcast from single GPU to all
        if kv_paging_data.get("prefill_kv_indices"):
            raw_kv_indices = torch.tensor(kv_paging_data["prefill_kv_indices"], dtype=torch.int32)
            raw_kv_indptr = torch.tensor(kv_paging_data["prefill_kv_indptr"], dtype=torch.int32) if kv_paging_data.get("prefill_kv_indptr") else None
            raw_kv_last_page_len = torch.tensor(kv_paging_data["prefill_kv_last_page_len"], dtype=torch.int32) if kv_paging_data.get("prefill_kv_last_page_len") else None
            raw_qo_indptr = torch.tensor(kv_paging_data["prefill_qo_indptr"], dtype=torch.int32) if kv_paging_data.get("prefill_qo_indptr") else None

            if raw_kv_indptr is not None and raw_kv_last_page_len is not None and raw_qo_indptr is not None:
                # Check if the data needs padding or truncation
                padded_batch = globs.global_batch_size
                max_pages_per_seq = 1024  # This matches the hardcoded value in globs.py
                expected_indices_size = padded_batch * max_pages_per_seq

                # Handle kv_indices - it might already be the full size or need padding
                if raw_kv_indices.shape[0] == expected_indices_size:
                    kv_indices = raw_kv_indices
                elif raw_kv_indices.shape[0] < expected_indices_size:
                    kv_indices = torch.zeros(expected_indices_size, dtype=torch.int32)
                    kv_indices[:raw_kv_indices.shape[0]] = raw_kv_indices
                else:
                    kv_indices = raw_kv_indices[:expected_indices_size]

                # Handle kv_indptr - should be (batch_size + 1)
                expected_indptr_size = padded_batch + 1
                if raw_kv_indptr.shape[0] == expected_indptr_size:
                    kv_indptr = raw_kv_indptr
                elif raw_kv_indptr.shape[0] < expected_indptr_size:
                    kv_indptr = torch.zeros(expected_indptr_size, dtype=torch.int32)
                    kv_indptr[:raw_kv_indptr.shape[0]] = raw_kv_indptr
                    if raw_kv_indptr.shape[0] > 0:
                        kv_indptr[raw_kv_indptr.shape[0]:] = raw_kv_indptr[-1]
                else:
                    kv_indptr = raw_kv_indptr[:expected_indptr_size]

                # Handle kv_last_page_len - should be batch_size
                expected_page_len_size = padded_batch
                if raw_kv_last_page_len.shape[0] == expected_page_len_size:
                    kv_last_page_len = raw_kv_last_page_len
                elif raw_kv_last_page_len.shape[0] < expected_page_len_size:
                    kv_last_page_len = torch.zeros(expected_page_len_size, dtype=torch.int32)
                    kv_last_page_len[:raw_kv_last_page_len.shape[0]] = raw_kv_last_page_len
                else:
                    kv_last_page_len = raw_kv_last_page_len[:expected_page_len_size]

                # Handle qo_indptr - should be (batch_size + 1)
                if raw_qo_indptr.shape[0] == expected_indptr_size:
                    qo_indptr = raw_qo_indptr
                elif raw_qo_indptr.shape[0] < expected_indptr_size:
                    qo_indptr = torch.zeros(expected_indptr_size, dtype=torch.int32)
                    qo_indptr[:raw_qo_indptr.shape[0]] = raw_qo_indptr
                    if raw_qo_indptr.shape[0] > 0:
                        qo_indptr[raw_qo_indptr.shape[0]:] = raw_qo_indptr[-1]
                else:
                    qo_indptr = raw_qo_indptr[:expected_indptr_size]

                for dev_idx in range(config.num_devices):
                    globs.copy_prefill_info(
                        qo_indptr=qo_indptr,
                        kv_indices=kv_indices,
                        kv_indptr=kv_indptr,
                        kv_last_page_len=kv_last_page_len,
                        device_idx=dev_idx,
                    )
                print(f"  Broadcast prefill KV paging to all devices:")
                print(f"    kv_indices: {raw_kv_indices.shape[0]} -> {kv_indices.shape[0]} (expected: {expected_indices_size})")
                print(f"    kv_indptr: {raw_kv_indptr.shape[0]} -> {kv_indptr.shape[0]} (expected: {expected_indptr_size})")
                print(f"    kv_last_page_len: {raw_kv_last_page_len.shape[0]} -> {kv_last_page_len.shape[0]} (expected: {expected_page_len_size})")
                print(f"    qo_indptr: {raw_qo_indptr.shape[0]} -> {qo_indptr.shape[0]} (expected: {expected_indptr_size})")
    else:
        print("\n⚠️  No KV paging data in debug dump - using dummy values")
        print("This may cause different behavior than the original run!")

        # Fall back to dummy data
        # Set up dummy KV append indices - pointing to the last page to avoid overwrites
        dummy_append_indices = torch.full(
            (actual_pos_size,),
            (globs.num_pages * globs.page_size) - 1,  # Point to dummy location
            dtype=torch.int32
        )
        for dev_idx in range(config.num_devices):
            globs.copy_append_indices(dummy_append_indices, dev_idx)

        # Set up minimal decode info - all pointing to dummy pages
        dummy_kv_indices = torch.full((actual_pos_size,), globs.num_pages, dtype=torch.int32)
        dummy_kv_indptr = torch.arange(actual_pos_size + 1, dtype=torch.int32)
        dummy_kv_last_page_len = torch.ones(actual_pos_size, dtype=torch.int32)

        for dev_idx in range(config.num_devices):
            globs.copy_decode_info(
                kv_indices=dummy_kv_indices,
                kv_indptr=dummy_kv_indptr,
                kv_last_page_len=dummy_kv_last_page_len,
                device_idx=dev_idx,
            )

        # Set up minimal prefill info if we have any
        if actual_pos_size > 0:
            dummy_prefill_qo_indptr = torch.zeros(actual_pos_size + 1, dtype=torch.int32)
            for dev_idx in range(config.num_devices):
                globs.copy_prefill_info(
                    qo_indptr=dummy_prefill_qo_indptr,
                    kv_indices=dummy_kv_indices,
                    kv_indptr=dummy_kv_indptr,
                    kv_last_page_len=dummy_kv_last_page_len,
                    device_idx=dev_idx,
                )

    # Run the kernel repeatedly
    print(f"\n=== Starting debug iterations ({config.debug_iterations} iterations) ===")

    failures = 0
    successes = 0
    failure_types = {}
    timings = []
    first_barriers = None

    for iteration in range(config.debug_iterations):
        print(f"Iteration {iteration + 1}/{config.debug_iterations}...", end="", flush=True)

        print(f'Current streams: {[((i, torch.cuda.current_stream(i).cuda_stream)) for i in range(config.num_devices)]}')

        try:
            # Reset barriers - with error checking
            if hasattr(globs, 'barriers') and globs.barriers:
                for i, barrier in enumerate(globs.barriers):
                    if barrier is not None:
                        try:
                            barrier.fill_(config.bar_init_val)
                        except Exception as e:
                            print(f"\nWarning: Failed to reset barrier {i}: {e}")

            # Reset global instruction index if it exists
            if hasattr(globs, 'global_instruction_index') and globs.global_instruction_index:
                for gii in globs.global_instruction_index:
                    if gii is not None:
                        try:
                            gii.zero_()
                        except:
                            pass

            # Run the interpreter
            sync_all_devices(config.num_devices)
            start_time = time()
            interpreter.interpret()
            sync_all_devices(config.num_devices)
            end_time = time()

            elapsed = end_time - start_time
            timings.append(elapsed)
            # print(f" Success ({elapsed:.3f}s)")
            # if iteration == 0:
            #     first_barriers = [globs.barriers[i].detach().clone() for i in range(config.num_devices)]
            # else:
            #     for i in range(config.num_devices):
            #         if not torch.allclose(first_barriers[i], globs.barriers[i]):
            #             print(f"Barrier {i} mismatch at iteration {iteration}")
            #             breakpoint()
            # breakpoint()

            if iteration % 100 == 0 and iteration > 0:
                # Print summary every 100 iterations
                avg_time = sum(timings[-100:]) / len(timings[-100:])
                print(f" [Last 100 avg: {avg_time:.3f}s]")

            successes += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break

        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__

            print(f" FAILED: {error_type}: {error_str}")
            failures += 1

            # Track failure types
            if error_type not in failure_types:
                failure_types[error_type] = []
            failure_types[error_type].append((iteration, error_str[:100]))

            if config.debug_verbose:
                import traceback
                traceback.print_exc()

            # On CUDA errors, might need to reset
            if "CUDA" in error_str or "cuda" in error_str.lower():
                print("Attempting CUDA reset...")
                try:
                    torch.cuda.empty_cache()
                    sync_all_devices(config.num_devices)
                except:
                    pass

    # Print summary
    print(f"\n=== Debug run complete ===")
    print(f"Successes: {successes}/{iteration + 1}")
    print(f"Failures: {failures}/{iteration + 1}")

    if timings:
        avg_time = sum(timings) / len(timings)
        min_time = min(timings)
        max_time = max(timings)
        print(f"Timing stats: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")

    if failure_types:
        print("\nFailure breakdown:")
        for error_type, occurrences in failure_types.items():
            print(f"  {error_type}: {len(occurrences)} occurrences")
            if config.debug_verbose and occurrences:
                print(f"    First at iteration {occurrences[0][0]}: {occurrences[0][1]}")

    if failures > 0:
        print(f"\nERROR: {failures} failures detected!")
        return 1
    else:
        print("\nAll iterations completed successfully!")
        return 0


if __name__ == "__main__":
    import pydra
    from pathlib import Path

    class Config(pydra.Config):
        model: str = "meta-llama/Llama-3.1-70B-Instruct"
        mk_dir = Path(__file__).parent.parent.parent / "demos" / "cross-gpu-llama"

        # Must-not-change fields
        num_devices: int = 8

        skip_weight_load: bool = False
        use_random_weights: bool = True
        num_layer_override: int | None = None

        bar_init_val: int = 0

        global_work_queue: bool = True
        interleave_waves: bool = True
        interleave_buffer_factor: float = 1.0

        cpp_sched: bool = True

        stop_after_op: str | None = None

        force_enable_timing: bool = False

        # Scheduling limits
        max_oproj_instructions_per_gpu: int | None = None

        # Debug mode parameters - these are the main ones for standalone usage
        debug_pickle: Path | None = None
        debug_iterations: int = 100
        debug_verbose: bool = False

        # Default values for globals setup
        glob_bs: int = 4096
        num_pages: int = 16384 // 2

        def finalize(self):
            if self.debug_pickle is None:
                raise ValueError("debug_pickle must be specified when running debug_stall.py directly")

    def main(config: Config):
        # # Create custom streams at startup
        # custom_streams = []
        # for i in range(config.num_devices):
        #     torch.cuda.set_device(i)
        #     stream = torch.cuda.Stream(device=i)
        #     custom_streams.append(stream)
            
        #     # Set this as the current stream for this device
        #     torch.cuda.set_stream(stream)
        return run_debug_mode(config)

    pydra.run(main)