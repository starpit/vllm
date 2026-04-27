import math
import random
import sys
from pathlib import Path
from time import time

import pydra
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from megakernels.demos.tp_throughput.cpp_scheduler import scheduler_cpp
from megakernels.demos.tp_throughput.globs import copy_into, make_globals
from megakernels.demos.tp_throughput.mk import TensorParallelMK_Interpreter
from megakernels.demos.tp_throughput.scheduler import (
    create_instruction_tensor,
    init_random_weights,
    load_weights,
    setup_rope_and_interleave,
)
from megakernels.demos.tp_throughput.timings_config import tp_llama_config
from megakernels.timings import timings_to_mkprof
from megakernels.utils import get_sm_count
from debug_stall import run_debug_mode


class Config(pydra.Config):
    model: str = "meta-llama/Llama-3.1-70B-Instruct"
    mk_dir = Path(__file__).parent.parent.parent / "demos" / "cross-gpu-llama"
    prompt: str = "tell me a funny joke about cookies"
    chat: bool = False
    ntok: int = 30

    glob_bs: int = 4096  # should be at least 2048 for memory alignment
    num_seqs: int = 1
    num_pages: int = 16384 // 2

    prefill_outfile: Path | None = None
    decode_outfile: Path | None = None

    # Benchmarking
    do_benchmark: bool = False
    num_warmups: int = 1
    num_iters: int = 5

    # Must-not-change fields
    num_devices: int = 8

    skip_weight_load: bool = False
    use_random_weights: bool = True
    num_layer_override: int | None = None

    bar_init_val: int = 0

    num_to_print: int = 16

    global_work_queue: bool = True
    interleave_waves: bool = True
    interleave_buffer_factor: float = 1.0

    cpp_sched: bool = True

    stop_after_op: str | None = None

    check_determinism: bool = False

    force_enable_timing: bool = False
    
    # Scheduling limits
    max_oproj_instructions_per_gpu: int | None = None
    
    # Debug mode parameters
    debug_pickle: Path | None = None
    debug_iterations: int = 100
    debug_verbose: bool = False

    def finalize(self):
        assert self.num_seqs <= self.glob_bs, (
            "num_seqs must be less than or equal to glob_bs"
        )


def sync_all_devices(config: Config):
    for dev_idx in range(config.num_devices):
        torch.cuda.synchronize(dev_idx)




@torch.inference_mode()
def main(config: Config):
    # Check if we're in debug mode
    if config.debug_pickle is not None:
        return run_debug_mode(config)
    
    model_config = AutoConfig.from_pretrained(config.model)

    print("Making globals:")
    globs = make_globals(
        model_config=model_config,
        global_batch_size=config.glob_bs,
        num_pages=config.num_pages,
        num_devices=config.num_devices,
        barrier_init_val=config.bar_init_val,
        global_work_queue_enabled=config.global_work_queue,
        timing_record_enabled=config.force_enable_timing
        or config.prefill_outfile is not None
        or config.decode_outfile is not None,
        layer_limit=config.num_layer_override,
    )
    print(f"page_size: {globs.page_size}")

    print(f"Scheduling instructions with interleave_waves={config.interleave_waves}:")

    if config.interleave_waves:
        sm_count = get_sm_count("cuda:0")
        overlap_buffer_size = round(sm_count * config.interleave_buffer_factor)
    else:
        overlap_buffer_size = None

    print(f"Using CPP scheduler: {config.cpp_sched}")

    if config.cpp_sched:
        assert scheduler_cpp is not None
        sched_func = scheduler_cpp.create_instruction_tensor
    else:
        sched_func = create_instruction_tensor

    def generate_schedule():
        instruction_tensors = []
        for dev_idx in tqdm(range(config.num_devices), desc="Scheduling instructions"):
            insts = sched_func(
                globs,
                device_idx=dev_idx,
                layer_limit=config.num_layer_override,
                interleave_waves=config.interleave_waves,
                interleave_buffer_size=overlap_buffer_size,
                stop_after_op=config.stop_after_op,
                max_oproj_instructions_per_gpu=config.max_oproj_instructions_per_gpu,
            )

            instruction_tensors.append(insts)
            print(f"Instruction tensor for device {dev_idx} has shape {insts.shape}")

        globs.copy_instructions(instruction_tensors)
        return [i.shape[0] for i in instruction_tensors]

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
    if interpreter.broadcast_lm_head_norm:
        print(
            "WARNING: Broadcasting LM head norm is designed for Toka, not tp_generate.py"
        )
        print("This will cause incorrect results")
        print("Exiting...")
        sys.exit(1)
    interpreter.setup()

    sync_all_devices(config)

    ########################################################
    # Generate input hidden states
    ########################################################

    print("Generating input hidden states:")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    # Generate input embeddings

    if config.chat:
        tok_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": config.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        tok_input = config.prompt

    input_ids = tokenizer(tok_input, return_tensors="pt", add_special_tokens=True)[
        "input_ids"
    ]
    input_seq_len = input_ids.shape[-1]

    print(f"tok_input: {tok_input}")
    print(f"input_ids shape: {input_ids.shape}")

    embeddings = [
        torch.nn.Embedding.from_pretrained(globs.embed_weights[dev_idx], freeze=True)
        for dev_idx in range(config.num_devices)
    ]
    single_seq_input_hidden_states: list[Tensor] = [
        embeddings[dev_idx](input_ids.to(dev_idx))
        for dev_idx in range(config.num_devices)
    ]

    ########################################################
    # Setup paging
    ########################################################

    max_len_per_seq = input_seq_len + config.ntok

    pages_per_seq = math.ceil(max_len_per_seq / globs.page_size)
    total_indices = pages_per_seq * config.num_seqs

    assert total_indices <= globs.num_pages, (
        f"Total indices {total_indices} must be less than or equal to num_pages "
        f"{globs.num_pages}"
    )

    indices_per_seq = []
    for seq in range(config.num_seqs):
        # THIS PLUS 1 IS REALLY, REALLY IMPORTANT.
        # PAGE 0 IS A DUMMY PAGE FOR DUMPING UNUSED PARTS OF THE QKV ROPE APPEND OUTPUTS
        indices_per_seq.append(
            list(range(1 + seq * pages_per_seq, 1 + (seq + 1) * pages_per_seq))
        )

    def setup_prefill_paging_inputs(start_seq_idx: int, end_seq_idx: int):
        num_seqs = end_seq_idx - start_seq_idx
        pages_per_seq = math.ceil(input_seq_len / globs.page_size)

        kv_indices = []
        kv_indptr = [0]
        append_indices = []
        qo_indptr = [0]

        position_ids = []

        for seq in range(start_seq_idx, end_seq_idx):
            indices = indices_per_seq[seq][:pages_per_seq]
            kv_indices.extend(indices)
            kv_indptr.append(kv_indptr[-1] + len(indices))
            start_page_idx = indices[0]
            assert indices == list(
                range(start_page_idx, start_page_idx + pages_per_seq)
            )
            start_token_idx = start_page_idx * globs.page_size
            append_indices.extend([start_token_idx + i for i in range(input_seq_len)])
            qo_indptr.append(qo_indptr[-1] + input_seq_len)
            position_ids.extend(range(input_seq_len))

        last_page_len = input_seq_len % globs.page_size
        if last_page_len == 0:
            last_page_len = globs.page_size

        t_kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
        t_kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
        t_qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
        t_last_page_len = torch.tensor(
            [last_page_len] * num_seqs * input_seq_len, dtype=torch.int32
        )
        t_position_ids = torch.tensor(position_ids, dtype=torch.int32)
        t_append_indices = torch.tensor(append_indices, dtype=torch.int32)

        globs.copy_prefill_info(
            qo_indptr=t_qo_indptr,
            kv_indices=t_kv_indices,
            kv_indptr=t_kv_indptr,
            kv_last_page_len=t_last_page_len,
        )
        globs.copy_append_indices(t_append_indices)
        globs.copy_position_ids(t_position_ids)

    def setup_decode_paging_inputs(cur_pos_id: int):
        seq_len = cur_pos_id + 1
        cur_pages_per_seq = math.ceil(seq_len / globs.page_size)

        last_page_len = seq_len % globs.page_size
        if last_page_len == 0:
            last_page_len = globs.page_size

        kv_indices = []
        kv_indptr = [0]
        append_indices = []

        for seq in range(config.num_seqs):
            indices = indices_per_seq[seq][:cur_pages_per_seq]
            kv_indices.extend(indices)
            kv_indptr.append(kv_indptr[-1] + len(indices))
            append_index = indices[-1] * globs.page_size + (
                cur_pos_id % globs.page_size
            )
            append_indices.append(append_index)

        t_kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
        t_kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
        t_append_indices = torch.tensor(append_indices, dtype=torch.int32)

        globs.copy_decode_info(
            t_kv_indices,
            t_kv_indptr,
            torch.tensor([last_page_len] * config.num_seqs, dtype=torch.int32),
        )
        globs.copy_append_indices(t_append_indices)
        globs.copy_position_ids(
            torch.tensor([cur_pos_id] * config.num_seqs, dtype=torch.int32)
        )

    ########################################################
    # Prefill the KV cache
    ########################################################

    # Zero out timings tensor
    for dev_idx in range(config.num_devices):
        globs.timings[dev_idx].zero_()

    print("Prefilling the KV cache:")
    sync_all_devices(config)

    seqs_per_batch = config.glob_bs // input_seq_len
    num_batches = math.ceil(config.num_seqs / seqs_per_batch)

    last_num_seqs = None

    for batch_idx in tqdm(range(num_batches), desc="Prefilling"):
        start_seq_idx = batch_idx * seqs_per_batch
        end_seq_idx = min(start_seq_idx + seqs_per_batch, config.num_seqs)
        batch_num_seqs = end_seq_idx - start_seq_idx

        setup_prefill_paging_inputs(start_seq_idx, end_seq_idx)

        if batch_num_seqs != last_num_seqs:
            globs.set_sizes(
                global_batch_size=batch_num_seqs * input_seq_len,
                prefill_chunk_lens=[input_seq_len] * batch_num_seqs,
            )
            print(f"Generating prefill schedule for {batch_num_seqs} sequences")
            num_instructions = generate_schedule()
            last_num_seqs = batch_num_seqs

        hiddens = (
            single_seq_input_hidden_states[0]
            .repeat(batch_num_seqs, 1, 1)
            .view(batch_num_seqs * input_seq_len, -1)
        )

        num_ids = hiddens.shape[0]

        pad_multiple = globs.matmul_batch_block_size * config.num_devices
        padded_num_ids = math.ceil(num_ids / pad_multiple) * pad_multiple

        padded_hiddens = torch.cat(
            [
                hiddens,
                torch.zeros(
                    padded_num_ids - num_ids, hiddens.shape[-1], device=hiddens.device
                ),
            ],
            dim=0,
        )

        viewed_as_blocks = padded_hiddens.view(
            -1, config.num_devices, globs.matmul_batch_block_size, 8192
        )

        for dev_idx in range(config.num_devices):
            copy_into(
                viewed_as_blocks[:, dev_idx, :, :].reshape(-1, hiddens.shape[-1]),
                globs.hidden_states[dev_idx],
            )

        # ⭐️ Launch the kernel ⭐️
        sync_all_devices(config)
        interpreter.interpret()
        sync_all_devices(config)

        for dev_idx in range(config.num_devices):
            if globs.global_instruction_index is not None:
                globs.global_instruction_index[dev_idx].zero_()
                globs.barriers[dev_idx].fill_(config.bar_init_val)

        sync_all_devices(config)

        # Save timings for the first prefill batch
        if batch_idx == 0 and config.prefill_outfile is not None:
            for dev_idx in range(config.num_devices):
                if globs.global_instruction_index is not None:
                    globs.global_instruction_index[dev_idx].zero_()
                    globs.barriers[dev_idx].fill_(config.bar_init_val)
            sync_all_devices(config)
            interpreter.interpret()
            sync_all_devices(config)
            timings_to_mkprof(
                globs.timings,
                globs.instructions,
                num_instructions,
                config.prefill_outfile,
                config=tp_llama_config,
            )

    ########################################################
    # Check determinism after prefill, if check_determinism is enabled
    ########################################################

    if config.check_determinism:
        # Check determinism of hidden states
        is_det = True
        for dev_idx in range(config.num_devices):
            for i in range(input_seq_len):
                diff = (
                    (
                        globs.hidden_states[dev_idx][i::input_seq_len]
                        - globs.hidden_states[dev_idx][i : i + 1]
                    )
                    .max()
                    .cpu()
                    .item()
                )
                if diff != 0:
                    print(
                        f"DEBUG: Prefill detected non-determinism in hidden states! Device {dev_idx}, Token {i}: {diff}"
                    )
                    is_det = False
        # Check determinism of k_cache and v_cache
        for dev_idx in range(config.num_devices):
            for i in range(len(globs.barriers[0])):
                k_diff = (
                    (
                        globs.k_cache[dev_idx][
                            1 + i * config.num_pages : 1
                            + i * config.num_pages
                            + config.num_seqs
                        ]
                        - globs.k_cache[dev_idx][
                            1 + i * config.num_pages : 2 + i * config.num_pages
                        ]
                    )
                    .max()
                    .cpu()
                    .item()
                )
                v_diff = (
                    (
                        globs.v_cache[dev_idx][
                            1 + i * config.num_pages : 1
                            + i * config.num_pages
                            + config.num_seqs
                        ]
                        - globs.v_cache[dev_idx][
                            1 + i * config.num_pages : 2 + i * config.num_pages
                        ]
                    )
                    .max()
                    .cpu()
                    .item()
                )
                if k_diff != 0 or v_diff != 0:
                    print(
                        f"DEBUG: Prefill detected non-determinism in k_cache or v_cache! Device {dev_idx}, Layer {i}: k {k_diff}, v {v_diff}"
                    )
                    is_det = False
        if is_det:
            print("DEBUG: Prefill is deterministic")
        else:
            print("DEBUG: Prefill is non-deterministic")
            breakpoint()

    ########################################################
    # Generate tokens
    ########################################################

    print("Generating decode schedule:")
    globs.set_sizes(
        global_batch_size=config.num_seqs,
    )
    num_instructions = generate_schedule()

    for t in globs.timings:
        t.zero_()

    print("Generating tokens:")

    for seq in range(config.num_seqs // input_seq_len):
        last_tok_in_seq_device = (
            (seq * input_seq_len + input_seq_len - 1)
            % (globs.matmul_batch_block_size * config.num_devices)
        ) // globs.matmul_batch_block_size
        last_tok_pos_on_device = (
            (seq * input_seq_len + input_seq_len - 1)
            // (globs.matmul_batch_block_size * config.num_devices)
        ) * globs.matmul_batch_block_size + (
            seq * input_seq_len + input_seq_len - 1
        ) % globs.matmul_batch_block_size
        next_token = torch.argmax(
            globs.logits[last_tok_in_seq_device][last_tok_pos_on_device]
        )
        print(f"next_token: {next_token}, decoded: {tokenizer.decode([next_token])}")

    # assert input_seq_len <= globs.matmul_batch_block_size
    # next_token = torch.argmax(globs.logits[0][input_seq_len-1])
    next_token_embeddings = [
        embeddings[dev_idx](torch.tensor([next_token], device=dev_idx))
        for dev_idx in range(config.num_devices)
    ]

    output_ids = [
        torch.empty(
            (globs.device_batch_size[dev_idx], config.ntok),
            dtype=torch.int32,
            device=dev_idx,
        )
        for dev_idx in range(config.num_devices)
    ]

    def setup_generation():
        # Generate the first token
        for dev_idx in range(config.num_devices):
            torch.cuda.synchronize(dev_idx)
            if globs.device_batch_size[dev_idx] > 0:
                output_ids[dev_idx][:, 0] = next_token
                globs.hidden_states[dev_idx][:] = next_token_embeddings[dev_idx]
                torch.cuda.synchronize(dev_idx)

        # Zero out timings tensor
        for dev_idx in range(config.num_devices):
            globs.timings[dev_idx].zero_()

    def generate_tokens(save_timings: bool = False):
        # Generate the rest of the tokens
        for i in tqdm(range(config.ntok - 1), desc="Generating tokens"):
            pos_id = input_seq_len + i
            t1 = time()
            setup_decode_paging_inputs(pos_id)
            sync_all_devices(config)
            t2 = time()

            # ⭐️ Launch the kernel ⭐️
            interpreter.interpret()

            sync_all_devices(config)

            if config.check_determinism:
                # Check determinism of hidden states after decode
                is_det = True
                for dev_idx in range(config.num_devices):
                    diff = (
                        (
                            globs.hidden_states[dev_idx]
                            - globs.hidden_states[dev_idx][:1]
                        )
                        .max()
                        .cpu()
                        .item()
                    )
                    if diff != 0:
                        print(
                            f"DEBUG: Decode detected non-determinism in hidden states! Generating token {i}, Device {dev_idx}: {diff}"
                        )
                        is_det = False
                # Check determinism of k_cache and v_cache
                for dev_idx in range(config.num_devices):
                    for jj in range(len(globs.barriers[0])):
                        k_diff = (
                            (
                                globs.k_cache[dev_idx][
                                    1 + jj * config.num_pages : 1
                                    + jj * config.num_pages
                                    + config.num_seqs
                                ]
                                - globs.k_cache[dev_idx][
                                    1 + jj * config.num_pages : 2
                                    + jj * config.num_pages
                                ]
                            )
                            .max()
                            .cpu()
                            .item()
                        )
                        v_diff = (
                            (
                                globs.v_cache[dev_idx][
                                    1 + jj * config.num_pages : 1
                                    + jj * config.num_pages
                                    + config.num_seqs
                                ]
                                - globs.v_cache[dev_idx][
                                    1 + jj * config.num_pages : 2
                                    + jj * config.num_pages
                                ]
                            )
                            .max()
                            .cpu()
                            .item()
                        )
                        if k_diff != 0 or v_diff != 0:
                            print(
                                f"DEBUG: Decode detected non-determinism in k_cache or v_cache! Generating token {i}, Device {dev_idx}, Layer {jj}: k {k_diff}, v {v_diff}"
                            )
                            is_det = False
                if is_det:
                    print(f"DEBUG: Decode generating token {i} is deterministic")
                else:
                    print(f"DEBUG: Decode generating token {i} is non-deterministic")
                    breakpoint()

            sync_all_devices(config)

            t3 = time()

            # Pick top-1 token
            for dev_idx in range(config.num_devices):
                num_tokens = globs.device_batch_size[dev_idx]
                output_ids[dev_idx][:num_tokens, i + 1] = torch.argmax(
                    globs.logits[dev_idx][:num_tokens], dim=-1
                )
                globs.hidden_states[dev_idx][:num_tokens, :] = embeddings[dev_idx](
                    output_ids[dev_idx][:num_tokens, i + 1]
                )
                # Reset global instruction index and barriers
                if globs.global_instruction_index is not None:
                    globs.global_instruction_index[dev_idx].zero_()
                    globs.barriers[dev_idx].fill_(config.bar_init_val)
            sync_all_devices(config)

            t4 = time()

            # print(f"Tok: {i} Setup: {1000*(t2 - t1)} ms, MEGA: {1000*(t3 - t2)} ms, Pick+Reset: {1000*(t4 - t3)} ms")

            if config.decode_outfile is not None and save_timings and i == 1:
                timings_to_mkprof(
                    globs.timings,
                    globs.instructions,
                    num_instructions,
                    config.decode_outfile,
                    config=tp_llama_config,
                )

    setup_generation()
    generate_tokens(save_timings=True)

    ########################################################
    # Print generated tokens
    ########################################################

    # Gather the output ids
    all_output_ids = torch.cat(
        [
            output_ids[dev_idx][: globs.device_batch_size[dev_idx]].to("cpu")
            for dev_idx in range(config.num_devices)
        ],
        dim=0,
    )

    all_decoded = tokenizer.batch_decode(all_output_ids)

    print("Llama outputs:")

    def print_outputs(vals):
        print(
            ("\n" + "-" * 80 + "\n").join(vals),
        )

    print(f"First {config.num_to_print} outputs:")
    print_outputs(all_decoded[: config.num_to_print])

    print(f"Last {config.num_to_print} outputs:")
    print_outputs(all_decoded[-config.num_to_print :])

    index_choices = random.sample(range(len(all_decoded)), k=config.num_to_print)
    choices = [all_decoded[i] for i in index_choices]
    print(f"Random {config.num_to_print} outputs (indices: {index_choices}):")
    print_outputs(choices)

    ########################################################
    # Benchmark
    ########################################################

    if config.do_benchmark:
        times = []
        for _ in tqdm(
            range(config.num_warmups + config.num_iters), desc="Benchmarking"
        ):
            start_time = time()
            generate_tokens(save_timings=False)
            end_time = time()
            times.append(end_time - start_time)

        non_warmup_times = times[config.num_warmups :]
        avg_time = sum(non_warmup_times) / len(non_warmup_times)
        print(f"Average time per iter: {avg_time * 1000:.2f} ms")
        print(f"Tokens per second: {config.num_seqs * (config.ntok - 1) / avg_time}")


if __name__ == "__main__":
    pydra.run(main)
