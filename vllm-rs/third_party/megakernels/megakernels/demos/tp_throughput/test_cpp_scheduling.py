import pytest
import torch
from transformers import AutoConfig

# Import the C++ extension (will be None if USE_CPP_SCHEDULER=0)
from megakernels.demos.tp_throughput.cpp_scheduler import scheduler_cpp
from megakernels.demos.tp_throughput.globs import make_globals
from megakernels.demos.tp_throughput.scheduler import (
    create_all_instruction_tensors,
    create_instruction_tensor,
)


@pytest.mark.parametrize("interleave_waves", [True, False])
@pytest.mark.parametrize("gwq", [True, False])
@pytest.mark.parametrize("layer_limit", [1, 5, None])
@pytest.mark.parametrize("num_threads", [1, 8])
@pytest.mark.parametrize("global_batch_size", [32, 1024, 4096])
@pytest.mark.parametrize("prefill_chunk_lens", [[], [256]])
@pytest.mark.parametrize("prefill_extend_offsets", [[], [128]])
def test_cpp_scheduling(
    interleave_waves: bool,
    gwq: bool,
    layer_limit: int | None,
    num_threads: int,
    global_batch_size: int,
    prefill_chunk_lens: list[int],
    prefill_extend_offsets: list[int],
):
    if len(prefill_chunk_lens) != len(prefill_extend_offsets):
        return

    move_to_gpu = False  # Default to CPU for testing
    assert scheduler_cpp is not None, "C++ scheduler extension not available"
    model = "meta-llama/Llama-3.1-70b-Instruct"
    model_config = AutoConfig.from_pretrained(model)

    num_devices = 8
    barrier_init_val = 0

    globs = make_globals(
        model_config=model_config,
        num_devices=num_devices,
        global_batch_size=global_batch_size,
        num_pages=1024,  # Default value for testing
        barrier_init_val=barrier_init_val,
        global_work_queue_enabled=gwq,
        meta_device=True,  # Use meta device to avoid actual GPU allocation
        layer_limit=layer_limit,
    )

    globs.set_sizes(
        global_batch_size=global_batch_size,
        prefill_chunk_lens=prefill_chunk_lens,
        prefill_extend_offsets=prefill_extend_offsets,
    )

    # Python implementation
    python_insts = create_instruction_tensor(
        globs,
        device_idx=0,
        layer_limit=layer_limit,
        interleave_waves=interleave_waves,
        interleave_buffer_size=globs.sm_count if interleave_waves else None,
        move_to_gpu=move_to_gpu,
    )

    # C++ implementation
    cpp_insts = scheduler_cpp.create_instruction_tensor(
        globs,
        device_idx=0,
        layer_limit=layer_limit,
        interleave_waves=interleave_waves,
        interleave_buffer_size=globs.sm_count if interleave_waves else None,
        move_to_gpu=move_to_gpu,
        num_threads=num_threads,
    )

    # Compare results
    layer_desc = "full model" if layer_limit is None else f"{layer_limit} layers"
    print(
        f"Testing {layer_desc}, interleave_waves={interleave_waves}, gwq={gwq}, num_threads={num_threads}"
    )
    print(f"Python instructions shape: {python_insts.shape}")
    print(f"C++ instructions shape: {cpp_insts.shape}")

    # Move to CPU for comparison
    python_insts_cpu = python_insts.cpu()
    cpp_insts_cpu = cpp_insts.cpu()

    # Check shapes match
    assert python_insts_cpu.shape == cpp_insts_cpu.shape, (
        f"Instruction shapes don't match: {python_insts_cpu.shape} vs {cpp_insts_cpu.shape}"
    )

    # Check instructions content match
    instructions_match = torch.equal(python_insts_cpu, cpp_insts_cpu)
    if not instructions_match:
        print("Instructions don't match exactly, checking elementwise...")
        diff = python_insts_cpu - cpp_insts_cpu
        non_zero_diff = torch.nonzero(diff)
        print(f"Number of differing elements: {non_zero_diff.shape[0]}")
        if non_zero_diff.shape[0] > 0:
            print(f"First few differences: {non_zero_diff[:10]}")
            for i in range(min(5, non_zero_diff.shape[0])):
                idx = tuple(non_zero_diff[i].tolist())
                print(
                    f"  At {idx}: Python={python_insts_cpu[idx].item()}, C++={cpp_insts_cpu[idx].item()}"
                )

    assert instructions_match, (
        "Instructions from Python and C++ implementations don't match"
    )

    print(
        f"✅ Test passed for {layer_desc}, interleave_waves={interleave_waves}, gwq={gwq}, num_threads={num_threads}"
    )


@pytest.mark.parametrize("interleave_waves", [True])
@pytest.mark.parametrize("gwq", [True])
@pytest.mark.parametrize("layer_limit", [1, None])
@pytest.mark.parametrize("num_threads", [8])
@pytest.mark.parametrize("global_batch_size", [32, 1024, 4096])
@pytest.mark.parametrize("prefill_chunk_lens", [[], [256]])
@pytest.mark.parametrize("prefill_extend_offsets", [[], [128]])
def test_cpp_create_all_instruction_tensors(
    interleave_waves: bool,
    gwq: bool,
    layer_limit: int | None,
    num_threads: int,
    global_batch_size: int,
    prefill_chunk_lens: list[int],
    prefill_extend_offsets: list[int],
):
    if len(prefill_chunk_lens) != len(prefill_extend_offsets):
        return

    move_to_gpu = False  # Default to CPU for testing
    assert scheduler_cpp is not None, "C++ scheduler extension not available"
    model = "meta-llama/Llama-3.1-70b-Instruct"
    model_config = AutoConfig.from_pretrained(model)

    num_devices = 8
    barrier_init_val = 0

    globs = make_globals(
        model_config=model_config,
        num_devices=num_devices,
        global_batch_size=global_batch_size,
        num_pages=1024,  # Default value for testing
        barrier_init_val=barrier_init_val,
        global_work_queue_enabled=gwq,
        meta_device=True,  # Use meta device to avoid actual GPU allocation
        layer_limit=layer_limit,
    )

    globs.set_sizes(
        global_batch_size=global_batch_size,
        prefill_chunk_lens=prefill_chunk_lens,
        prefill_extend_offsets=prefill_extend_offsets,
    )

    # Python implementation
    python_tensors = create_all_instruction_tensors(
        globs,
        layer_limit=layer_limit,
        interleave_waves=interleave_waves,
        interleave_buffer_size=globs.sm_count if interleave_waves else None,
        move_to_gpu=move_to_gpu,
    )

    # C++ implementation
    cpp_tensors = scheduler_cpp.create_all_instruction_tensors(
        globs,
        layer_limit=layer_limit,
        interleave_waves=interleave_waves,
        interleave_buffer_size=globs.sm_count if interleave_waves else None,
        move_to_gpu=move_to_gpu,
        num_threads=num_threads,
    )

    # Compare results
    layer_desc = "full model" if layer_limit is None else f"{layer_limit} layers"
    print(
        f"Testing create_all_instruction_tensors for {layer_desc}, interleave_waves={interleave_waves}, gwq={gwq}, num_threads={num_threads}"
    )

    # Check we have the right number of tensors
    assert len(python_tensors) == len(cpp_tensors) == num_devices, (
        f"Wrong number of tensors: Python={len(python_tensors)}, C++={len(cpp_tensors)}, expected={num_devices}"
    )

    # Check each tensor
    all_match = True
    for dev_idx in range(num_devices):
        python_tensor = python_tensors[dev_idx].cpu()
        cpp_tensor = cpp_tensors[dev_idx].cpu()

        # Check shapes match
        if python_tensor.shape != cpp_tensor.shape:
            print(
                f"Device {dev_idx}: Shape mismatch - Python {python_tensor.shape} vs C++ {cpp_tensor.shape}"
            )
            all_match = False
            continue

        # Check content matches
        if not torch.equal(python_tensor, cpp_tensor):
            print(f"Device {dev_idx}: Content mismatch")
            diff = python_tensor - cpp_tensor
            non_zero_diff = torch.nonzero(diff)
            print(f"  Number of differing elements: {non_zero_diff.shape[0]}")
            if non_zero_diff.shape[0] > 0:
                print(f"  First few differences: {non_zero_diff[:5]}")
            all_match = False

    assert all_match, "Tensors from Python and C++ implementations don't match"

    print(
        f"✅ create_all_instruction_tensors test passed for {layer_desc}, interleave_waves={interleave_waves}, gwq={gwq}, num_threads={num_threads}"
    )
