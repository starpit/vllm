import statistics
import time

import pydra
from tqdm import tqdm
from transformers import AutoConfig

# Import the C++ extension (will be None if USE_CPP_SCHEDULER=0)
from megakernels.demos.tp_throughput.cpp_scheduler import scheduler_cpp
from megakernels.demos.tp_throughput.globs import make_globals
from megakernels.demos.tp_throughput.scheduler import create_instruction_tensor, create_all_instruction_tensors


class Config(pydra.Config):
    model: str = "meta-llama/Llama-3.1-70b-Instruct"
    glob_bs: int = 4096
    num_devices: int = 8
    num_pages: int = 1024
    barrier_init_val: int = 0
    global_work_queue: bool = True
    interleave_waves: bool = True
    layer_limit: int | None = None
    move_to_gpu: bool = False
    skip_py: bool = False
    skip_cpp: bool = False
    zero_init: bool = True
    num_threads: int = 64
    all_gpu: bool = True

    num_warmups: int = 1
    num_iters: int = 3


def run_timing(func, config: Config):
    times = []
    for i in tqdm(range(config.num_warmups + config.num_iters)):
        start_time = time.time()
        func()
        elapsed = time.time() - start_time
        if i >= config.num_warmups:
            times.append(elapsed)

    return times


def main(config: Config):
    assert scheduler_cpp is not None, "C++ scheduler extension not available"
    
    print(f"Running benchmark in {'all_gpu' if config.all_gpu else 'single_gpu'} mode")
    if config.all_gpu:
        print(f"Creating instruction tensors for all {config.num_devices} devices")
    else:
        print("Creating instruction tensor for device 0 only")
    print()

    model_config = AutoConfig.from_pretrained(config.model)

    globs = make_globals(
        model_config=model_config,
        num_devices=config.num_devices,
        global_batch_size=config.glob_bs,
        num_pages=config.num_pages,
        barrier_init_val=config.barrier_init_val,
        global_work_queue_enabled=config.global_work_queue,
        meta_device=True,  # Use meta device to avoid GPU allocation for benchmarking
        layer_limit=config.layer_limit,
    )

    def go_python():
        if config.all_gpu:
            python_insts = create_all_instruction_tensors(
                globs,
                layer_limit=config.layer_limit,
                interleave_waves=config.interleave_waves,
                interleave_buffer_size=globs.sm_count if config.interleave_waves else None,
                move_to_gpu=config.move_to_gpu,
            )
        else:
            python_insts = create_instruction_tensor(
                globs,
                device_idx=0,
                layer_limit=config.layer_limit,
                interleave_waves=config.interleave_waves,
                interleave_buffer_size=globs.sm_count if config.interleave_waves else None,
                move_to_gpu=config.move_to_gpu,
            )
        return python_insts

    def go_cpp():
        if config.all_gpu:
            cpp_insts = scheduler_cpp.create_all_instruction_tensors(
                globs,
                layer_limit=config.layer_limit,
                interleave_waves=config.interleave_waves,
                interleave_buffer_size=globs.sm_count if config.interleave_waves else None,
                move_to_gpu=config.move_to_gpu,
                zero_init=config.zero_init,
                num_threads=config.num_threads,
            )
        else:
            cpp_insts = scheduler_cpp.create_instruction_tensor(
                globs,
                device_idx=0,
                layer_limit=config.layer_limit,
                interleave_waves=config.interleave_waves,
                interleave_buffer_size=globs.sm_count if config.interleave_waves else None,
                move_to_gpu=config.move_to_gpu,
                zero_init=config.zero_init,
                num_threads=config.num_threads,
            )
        return cpp_insts

    if not config.skip_py:
        print(
            f"Running Python implementation ({config.num_warmups} warmups, {config.num_iters} iterations)..."
        )
        python_times = run_timing(go_python, config)
        python_times_ms = [t * 1000 for t in python_times]  # Convert to milliseconds
        python_avg_ms = statistics.mean(python_times_ms)
        python_std_ms = (
            statistics.stdev(python_times_ms) if len(python_times_ms) > 1 else 0
        )

        print(f"Python implementation:")
        print(f"  Times (ms): {[f'{t:.2f}' for t in python_times_ms]}")
        print(f"  Mean: {python_avg_ms:.2f} ms")
        print(f"  Std Dev: {python_std_ms:.2f} ms")
        print(f"  Min: {min(python_times_ms):.2f} ms")
        print(f"  Max: {max(python_times_ms):.2f} ms")

    if not config.skip_cpp:
        print(
            f"Running C++ implementation ({config.num_warmups} warmups, {config.num_iters} iterations)..."
        )
        cpp_times = run_timing(go_cpp, config)
        cpp_times_ms = [t * 1000 for t in cpp_times]  # Convert to milliseconds
        cpp_avg_ms = statistics.mean(cpp_times_ms)
        cpp_std_ms = statistics.stdev(cpp_times_ms) if len(cpp_times_ms) > 1 else 0

        print(f"\nC++ implementation:")
        print(f"  Times (ms): {[f'{t:.2f}' for t in cpp_times_ms]}")
        print(f"  Mean: {cpp_avg_ms:.2f} ms")
        print(f"  Std Dev: {cpp_std_ms:.2f} ms")
        print(f"  Min: {min(cpp_times_ms):.2f} ms")
        print(f"  Max: {max(cpp_times_ms):.2f} ms")

        if cpp_avg_ms > 0 and not config.skip_py:
            speedup = python_avg_ms / cpp_avg_ms
            print(f"\nSpeedup: {speedup:.2f}x")
            if speedup > 1:
                print("✅ C++ implementation is faster!")
            else:
                print("⚠️  C++ implementation is slower than Python")

    # Verify correctness
    if not config.skip_py and not config.skip_cpp:
        print("\n" + "=" * 60)
        print("CORRECTNESS VERIFICATION")
        print("=" * 60)
        python_result = go_python()
        cpp_result = go_cpp()

        if config.all_gpu:
            # Check list of tensors
            if len(python_result) != len(cpp_result):
                print(f"❌ Different number of tensors: Python {len(python_result)} vs C++ {len(cpp_result)}")
            else:
                all_match = True
                for i in range(len(python_result)):
                    if python_result[i].shape != cpp_result[i].shape:
                        print(f"❌ Device {i} shape mismatch: Python {python_result[i].shape} vs C++ {cpp_result[i].shape}")
                        all_match = False
                    elif not (python_result[i].cpu() == cpp_result[i].cpu()).all():
                        print(f"❌ Device {i} results don't match")
                        all_match = False
                
                if all_match:
                    print(f"✅ All {len(python_result)} tensors match between Python and C++ implementations")
                else:
                    print("❌ Some tensors don't match between implementations")
        else:
            # Check single tensor
            if python_result.shape == cpp_result.shape:
                instructions_match = (python_result.cpu() == cpp_result.cpu()).all()

                if instructions_match:
                    print("✅ Results match between Python and C++ implementations")
                else:
                    print("❌ Results don't match between implementations")
            else:
                print(
                    f"❌ Shape mismatch: Python {python_result.shape} vs C++ {cpp_result.shape}"
                )

    print("=" * 60)


if __name__ == "__main__":
    pydra.run(main)
