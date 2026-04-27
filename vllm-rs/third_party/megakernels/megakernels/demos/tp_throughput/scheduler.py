import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain
from multiprocessing import Pool, set_start_method
from pathlib import Path

import huggingface_hub
import torch
from safetensors import safe_open
from torch import Tensor
from tqdm import tqdm
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
)

from megakernels.demos.tp_throughput.globs import Globals
from megakernels.demos.tp_throughput.instructions import (
    AttentionDecode,
    AttentionPrefill,
    AttnNorm,
    Die,
    DownProjResidual,
    GateSilu,
    IncBarrier,
    Instruction,
    LM_Head,
    LM_Head_Norm,
    MatMulInstruction,
    MLP_Norm,
    NormInstruction,
    O_ProjResidual,
    QKV_RopeAppend,
    UpMatMul,
    AllDeviceBarrier,
)
from megakernels.scheduler import (
    convert_instruction_queues_to_tensor,
    round_robin_assign_to_sms,
    serialize_and_pad,
)
from megakernels.utils import assert_div, get_sm_count


@torch.inference_mode()
def load_for_device(
    dev_idx: int,
    filenames: list[str],
    snapshot_path: Path,
    globs: Globals,
    layer_limit: int | None,
):
    num_devices = globs.tp_size

    for filename in tqdm(
        filenames,
        desc=f"Loading safetensor files for each layer for device {dev_idx}",
    ):
        with safe_open(snapshot_path / filename, framework="pt", device=dev_idx) as f:
            for key in f.keys():
                tensor_slice = f.get_slice(key)
                match = re.search(r"layers\.(\d+)", key)
                if match is not None:
                    layer_idx = int(match.group(1))
                    if layer_limit is not None and layer_idx >= layer_limit:
                        continue
                else:
                    layer_idx = -1
                if key.endswith("input_layernorm.weight"):  # (8192,)
                    assert match is not None
                    assert tensor_slice.get_shape() == [
                        globs.hidden_size,
                    ]
                    globs.attn_norm_weights[dev_idx][layer_idx, :] = tensor_slice[:]
                elif key.endswith("self_attn.q_proj.weight"):  # (8192, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [
                        globs.hidden_size,
                        globs.hidden_size,
                    ]
                    elem_per_dev = (
                        globs.num_attention_heads * globs.head_dim // num_devices
                    )
                    globs.qkv_proj_weights[dev_idx][layer_idx, :elem_per_dev, :] = (
                        tensor_slice[
                            dev_idx * elem_per_dev : (dev_idx + 1) * elem_per_dev,
                            :,
                        ]
                    )
                elif key.endswith("self_attn.k_proj.weight"):  # (1024, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [
                        globs.num_kv_heads * globs.head_dim,
                        globs.hidden_size,
                    ]
                    num_q_elems = (
                        globs.num_attention_heads * globs.head_dim // num_devices
                    )
                    elem_per_dev = globs.num_kv_heads * globs.head_dim // num_devices
                    globs.qkv_proj_weights[dev_idx][
                        layer_idx, num_q_elems : num_q_elems + elem_per_dev, :
                    ] = tensor_slice[
                        dev_idx * elem_per_dev : (dev_idx + 1) * elem_per_dev, :
                    ]
                elif key.endswith("self_attn.v_proj.weight"):  # (1024, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [
                        globs.num_kv_heads * globs.head_dim,
                        globs.hidden_size,
                    ]
                    num_qk_elems = (
                        (globs.num_attention_heads + globs.num_kv_heads)
                        * globs.head_dim
                        // num_devices
                    )
                    elem_per_dev = globs.num_kv_heads * globs.head_dim // num_devices
                    globs.qkv_proj_weights[dev_idx][
                        layer_idx, num_qk_elems : num_qk_elems + elem_per_dev, :
                    ] = tensor_slice[
                        dev_idx * elem_per_dev : (dev_idx + 1) * elem_per_dev, :
                    ]
                elif key.endswith("self_attn.o_proj.weight"):  # (8192, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [8192, 8192]
                    elem_per_dev = globs.hidden_size // num_devices
                    globs.o_proj_weights[dev_idx][layer_idx, :, :] = tensor_slice[:, :]
                elif key.endswith("post_attention_layernorm.weight"):  # (8192,)
                    assert match is not None
                    assert tensor_slice.get_shape() == [
                        globs.hidden_size,
                    ]
                    globs.mlp_norm_weights[dev_idx][layer_idx, :] = tensor_slice[:]
                elif key.endswith("mlp.gate_proj.weight"):  # (28672, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [
                        globs.intermediate_size,
                        globs.hidden_size,
                    ]
                    elem_per_dev = globs.intermediate_size // num_devices
                    globs.gate_proj_weights[dev_idx][layer_idx, :, :] = tensor_slice[
                        dev_idx * elem_per_dev : (dev_idx + 1) * elem_per_dev, :
                    ]
                elif key.endswith("mlp.up_proj.weight"):  # (28672, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [
                        globs.intermediate_size,
                        globs.hidden_size,
                    ]
                    elem_per_dev = globs.intermediate_size // num_devices
                    globs.up_proj_weights[dev_idx][layer_idx, :, :] = tensor_slice[
                        dev_idx * elem_per_dev : (dev_idx + 1) * elem_per_dev, :
                    ]
                elif key.endswith("mlp.down_proj.weight"):  # (8192, 28672)
                    assert match is not None
                    assert tensor_slice.get_shape() == [
                        globs.hidden_size,
                        globs.intermediate_size,
                    ]
                    elem_per_dev = globs.intermediate_size // num_devices
                    globs.down_proj_weights[dev_idx][layer_idx, :, :] = tensor_slice[
                        :, dev_idx * elem_per_dev : (dev_idx + 1) * elem_per_dev
                    ]
                elif key == "model.norm.weight":  # (8192,)
                    assert tensor_slice.get_shape() == [
                        globs.hidden_size,
                    ]
                    globs.lm_head_norm_weights[dev_idx][:] = tensor_slice[:]
                elif key == "lm_head.weight":  # (128256, 8192)
                    assert tensor_slice.get_shape() == [
                        globs.vocab_size,
                        globs.hidden_size,
                    ]
                    globs.lm_head_weights[dev_idx][:, :] = tensor_slice[:, :]
                elif key == "model.embed_tokens.weight":  # (128256, 8192)
                    assert tensor_slice.get_shape() == [
                        globs.vocab_size,
                        globs.hidden_size,
                    ]
                    globs.embed_weights[dev_idx][:, :] = tensor_slice[:]
                else:
                    raise ValueError(f"Unknown key: {key}")


def load_weights(
    model_name: str,
    globs: Globals,
    layer_limit: int | None = None,
    parallel_load: bool = True,
    multiprocess: bool = False,
):
    num_devices = globs.tp_size

    # Download model repo

    snapshot_path_str = huggingface_hub.snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.json"],
    )
    snapshot_path = Path(snapshot_path_str)

    # Load safetensor files
    safetensors_index_path = snapshot_path / "model.safetensors.index.json"
    assert not (snapshot_path / "model.safetensors").exists()
    assert safetensors_index_path.exists()
    with open(safetensors_index_path, "r") as f:
        safetensors_index = json.load(f)

    # Load model weights
    filenames = set(safetensors_index["weight_map"].values())

    if parallel_load:
        func = partial(
            load_for_device,
            filenames=filenames,
            snapshot_path=snapshot_path,
            globs=globs,
            layer_limit=layer_limit,
        )

        if multiprocess:
            set_start_method("spawn", force=True)
            with Pool(num_devices) as p:
                p.map(
                    func,
                    range(num_devices),
                )
        else:
            with ThreadPoolExecutor(max_workers=num_devices) as executor:
                list(
                    executor.map(
                        func,
                        range(num_devices),
                    )
                )
    else:
        for dev_idx in tqdm(
            range(num_devices), desc="Loading safetensor files for each device"
        ):
            func(dev_idx)


def setup_rope_and_interleave(globs: Globals):
    num_devices = globs.tp_size

    ########################################################
    # Generate RoPE weights
    ########################################################

    # Generate RoPE weights
    dummy_float_input = torch.empty((1,), dtype=torch.float32, device="cpu")
    position_ids = torch.arange(globs.model_config.max_position_embeddings).unsqueeze(0)
    _rope_cos, _rope_sin = LlamaRotaryEmbedding(config=globs.model_config)(
        dummy_float_input, position_ids
    )
    _rope_cos = _rope_cos.squeeze(0)
    _rope_sin = _rope_sin.squeeze(0)
    assert _rope_cos.dtype == torch.float32
    assert _rope_sin.dtype == torch.float32

    ########################################################
    # Interleave RoPE and QKV weights
    ########################################################

    # Generate interleaved indices
    interleaved_indices = []
    for head_idx in range(globs.num_combined_heads // num_devices):
        for dim_idx in range(globs.head_dim // 2):
            if head_idx < globs.num_combined_heads // num_devices - 1:
                interleaved_indices.append(dim_idx + head_idx * globs.head_dim)
                interleaved_indices.append(
                    dim_idx + head_idx * globs.head_dim + globs.head_dim // 2
                )
            else:  # do not interleave V
                interleaved_indices.append(dim_idx * 2 + head_idx * globs.head_dim)
                interleaved_indices.append(dim_idx * 2 + head_idx * globs.head_dim + 1)

    # Interleave RoPE weights and load them into HBMs
    _rope_cos = _rope_cos[:, interleaved_indices[: globs.head_dim]]
    _rope_sin = _rope_sin[:, interleaved_indices[: globs.head_dim]]
    for dev_idx in range(num_devices):
        globs.rope_cos[dev_idx][:] = _rope_cos[
            : globs.model_config.max_position_embeddings, :
        ].to(dev_idx)
        globs.rope_sin[dev_idx][:] = _rope_sin[
            : globs.model_config.max_position_embeddings, :
        ].to(dev_idx)

    # Interleave QKV weights
    for dev_idx in range(num_devices):
        globs.qkv_proj_weights[dev_idx][:, :, :] = globs.qkv_proj_weights[dev_idx][
            :, interleaved_indices, :
        ]


def init_random_weights(globs: Globals, std: float = 1.0):
    init_fn = partial(torch.nn.init.normal_, std=std)

    for dev_idx in range(globs.tp_size):
        torch.manual_seed(dev_idx)
        init_fn(globs.attn_norm_weights[dev_idx])
        init_fn(globs.qkv_proj_weights[dev_idx])
        init_fn(globs.o_proj_weights[dev_idx])
        init_fn(globs.mlp_norm_weights[dev_idx])
        init_fn(globs.up_proj_weights[dev_idx])
        init_fn(globs.gate_proj_weights[dev_idx])
        init_fn(globs.down_proj_weights[dev_idx])
        init_fn(globs.lm_head_weights[dev_idx])
        init_fn(globs.lm_head_norm_weights[dev_idx])
        init_fn(globs.embed_weights[dev_idx])


def make_supergroup(iters1: list[int], iters2: list[int], group_size: int):
    num_iters1 = len(iters1)
    num_iters2 = len(iters2)

    vals = []
    num_groups = math.ceil(num_iters1 / group_size)
    for group in range(num_groups):
        start_in_group = group * group_size
        end_in_group = min(start_in_group + group_size, num_iters1)
        for j in range(num_iters2):
            for i in range(start_in_group, end_in_group):
                vals.append((iters1[i], iters2[j]))

    # raw_vals = []
    # for i in range(len(iters1)):
    #     for j in range(len(iters2)):
    #         raw_vals.append((iters1[i], iters2[j]))

    # assert set(raw_vals) == set(vals)

    return vals


# def make_supergroup(
#     iters1: list[int], iters2: list[int], group_size1: int, group_size2: int
# ):
#     num_iters1 = len(iters1)
#     num_iters2 = len(iters2)

#     vals = []

#     if group_size1 == -1:
#         group_size1 = num_iters1
#     if group_size2 == -1:
#         group_size2 = num_iters2

#     num_groups1 = math.ceil(num_iters1 / group_size1)
#     num_groups2 = math.ceil(num_iters2 / group_size2)
#     for g1 in range(num_groups1):
#         for g2 in range(num_groups2):
#             start1_in_group = g1 * group_size1
#             end1_in_group = min(start1_in_group + group_size1, num_iters1)
#             start2_in_group = g2 * group_size2
#             end2_in_group = min(start2_in_group + group_size2, num_iters2)

#             for i in range(start1_in_group, end1_in_group):
#                 for j in range(start2_in_group, end2_in_group):
#                     vals.append((iters1[i], iters2[j]))

#     # raw_vals = []
#     # for i in range(len(iters1)):
#     #     for j in range(len(iters2)):
#     #         raw_vals.append((iters1[i], iters2[j]))

#     # assert set(raw_vals) == set(vals)

#     return vals


def schedule_norm(
    func: type[NormInstruction],
    layer_idx: int,
    device_idx: int,
    globs: Globals,
    interleave_waves: bool,
):
    if interleave_waves:
        return schedule_norm_multi_wave(func, layer_idx, device_idx, globs)
    else:
        return schedule_norm_single_wave(func, layer_idx, device_idx, globs)


def schedule_norm_multi_wave(
    func: type[NormInstruction],
    layer_idx: int,
    device_idx: int,
    globs: Globals,
):
    instructions = []
    for batch_idx in range(0, globs.device_batch_size[device_idx]):
        ins = func(layer_idx=layer_idx, local_batch_indices=[batch_idx])
        instructions.append(ins)

    return instructions, globs.matmul_batch_block_size


def schedule_norm_single_wave(
    func: type[NormInstruction],
    layer_idx: int,
    device_idx: int,
    globs: Globals,
):
    # HACK
    sm_count = get_sm_count("cuda:0")

    round_robin_queues = [[] for _ in range(sm_count)]

    for bidx in range(0, globs.device_batch_size[device_idx]):
        sm_idx = bidx % sm_count
        round_robin_queues[sm_idx].append(bidx)

    max_vecs_per_inst = min(6 * 32768 // (globs.hidden_size * 2), 12)

    instructions = []
    for sm_idx in range(sm_count):
        queue = round_robin_queues[sm_idx]
        queue_size = len(queue)

        for start_idx in range(0, queue_size, max_vecs_per_inst):
            end_idx = min(start_idx + max_vecs_per_inst, queue_size)

            batch_indices = queue[start_idx:end_idx]

            ins = func(layer_idx=layer_idx, local_batch_indices=batch_indices)
            instructions.append(ins)

    return instructions, sm_count


# def get_gpu_aware_batch_block_order(
#     globs: Globals,
# ):
#     partitions = []
#     blocks_per_partition = assert_div(globs.num_batch_blocks(), globs.tp_size)
#     for i in range(globs.tp_size):
#         partitions.append(
#             list(range(i * blocks_per_partition, (i + 1) * blocks_per_partition))
#         )

#     concat = []
#     for i in range(blocks_per_partition):
#         for j in range(globs.tp_size):
#             concat.append(partitions[j][i])

#     return concat


# def get_local_batch_blocks_for_gpu(globs: Globals, device_idx: int):
#     num_blocks = globs.num_batch_blocks()
#     assert num_blocks % globs.tp_size == 0

#     return list(range(device_idx, num_blocks, globs.tp_size))


def schedule_matmul(
    func: type[MatMulInstruction],
    layer_idx: int,
    device_idx: int,
    globs: Globals,
    global_outdim: int,
    is_split_batch_dim: bool,
    is_split_output_dim: bool,
    supergroup_size: int = 8,
):
    instructions: list[Instruction] = []

    if is_split_batch_dim:
        num_local_batch_blocks = globs.local_batch_blocks(device_idx)
    else:
        num_local_batch_blocks = globs.num_batch_blocks()

    output_limit = global_outdim
    if is_split_output_dim:
        output_limit = output_limit // globs.tp_size

    num_local_output_blocks = assert_div(output_limit, globs.matmul_output_block_size)

    local_batch_idx_range = list(range(0, num_local_batch_blocks))
    local_output_idx_range = list(range(0, num_local_output_blocks))

    for local_batch_idx, local_output_idx in make_supergroup(
        iters1=local_batch_idx_range,
        iters2=local_output_idx_range,
        group_size=supergroup_size,
    ):
        if is_split_batch_dim:
            global_batch_block_idx = (
                num_local_batch_blocks * device_idx + local_batch_idx
            )
        else:
            global_batch_block_idx = local_batch_idx

        if is_split_output_dim:
            global_output_block_idx = (
                num_local_output_blocks * device_idx + local_output_idx
            )
        else:
            global_output_block_idx = local_output_idx

        inst = func(
            layer_idx=layer_idx,
            local_batch_block_idx=local_batch_idx,
            local_output_block_idx=local_output_idx,
            global_batch_block_idx=global_batch_block_idx,
            global_output_block_idx=global_output_block_idx,
        )
        instructions.append(inst)

    buffer_size = min(len(local_batch_idx_range), supergroup_size) * len(
        local_output_idx_range
    )

    return instructions, buffer_size


def schedule_attention_decode(
    globs: Globals,
    layer_idx: int,
):
    instructions: list[Instruction] = []

    local_kv_heads = globs.num_kv_heads // globs.tp_size

    data = []
    group_size = 8

    for seq_idx in range(globs.num_decode_seqs()):
        for kv_head_idx in range(local_kv_heads):
            data.append(seq_idx)
            data.append(kv_head_idx)

            if len(data) == group_size * 2:
                instructions.append(
                    AttentionDecode(
                        layer_idx=layer_idx,
                        data=data,
                    )
                )
                data = []

    # Handle remaining data that doesn't fill a complete group
    if len(data) > 0:
        instructions.append(
            AttentionDecode(
                layer_idx=layer_idx,
                data=data,
            )
        )

    # TODO need to update to account for prefill
    buffer_size = math.ceil(globs.matmul_batch_block_size * local_kv_heads / group_size)

    return instructions, buffer_size


def schedule_attention_prefill(
    globs: Globals,
    layer_idx: int,
    max_instructions: int | None = None,
):
    instructions: list[Instruction] = []

    local_kv_heads = globs.num_kv_heads // globs.tp_size

    buffer_size = 0
    cumsum_tokens_processed = 0

    prefill_block_size = globs.prefill_block_size

    for prefill_seq_idx, (prefill_chunk_len, prefill_extend_offset) in enumerate(
        zip(
            globs.prefill_chunk_lens,
            globs.prefill_extend_offsets,
        )
    ):
        num_prefill_blocks = math.ceil(prefill_chunk_len / prefill_block_size)

        # TODO is this the best nesting of these two loops? (shouldn't matter for
        # L70B on 8GPUs here since theres only 1 kv head)
        for prefill_block_idx in range(num_prefill_blocks):
            block_start_idx = prefill_block_idx * prefill_block_size
            block_end_idx = min(
                block_start_idx + prefill_block_size,
                prefill_chunk_len,
            )

            tokens_in_block = block_end_idx - block_start_idx

            for kv_head_idx in range(local_kv_heads):
                # Check if we've reached the maximum number of instructions
                if max_instructions is not None and len(instructions) >= max_instructions:
                    return instructions, buffer_size

                instructions.append(
                    AttentionPrefill(
                        layer_idx=layer_idx,
                        prefill_seq_idx=prefill_seq_idx,
                        prefill_block_idx=prefill_block_idx,
                        prefill_token_offset=prefill_extend_offset,
                        kv_head_idx=kv_head_idx,
                    )
                )

                if cumsum_tokens_processed < globs.matmul_batch_block_size:
                    buffer_size += 1

            cumsum_tokens_processed += tokens_in_block

    return instructions, buffer_size


def schedule_layer(
    globs: Globals,
    layer_idx: int,
    device_idx: int,
    interleave_waves: bool,
    stop_after_op: str | None = None,
    max_prefill_instructions_per_gpu: int | None = None,
):
    instructions_waves: list[list[Instruction]] = []
    wave_buffer_sizes = []

    def _register(ins: list[Instruction], buf_size: int):
        if len(ins) == 0:
            return

        instructions_waves.append(ins)
        wave_buffer_sizes.append(buf_size)

    def _sched_norm(func: type[NormInstruction]):
        ins, buf = schedule_norm(
            func=func,
            layer_idx=layer_idx,
            device_idx=device_idx,
            globs=globs,
            interleave_waves=interleave_waves,
            # interleave_waves=True,
        )
        _register(ins, buf)

    def _sched_matmul(
        func: type[MatMulInstruction],
        global_outdim: int,
        is_split_batch_dim: bool,
        is_split_output_dim: bool,
        supergroup_size: int = 8,
    ):
        ins, buf = schedule_matmul(
            func=func,
            layer_idx=layer_idx,
            device_idx=device_idx,
            globs=globs,
            global_outdim=global_outdim,
            is_split_batch_dim=is_split_batch_dim,
            is_split_output_dim=is_split_output_dim,
            supergroup_size=supergroup_size,
        )
        _register(ins, buf)

    _sched_norm(AttnNorm)
    if stop_after_op == "attn_norm":
        return instructions_waves, wave_buffer_sizes

    _sched_matmul(
        QKV_RopeAppend,
        global_outdim=(globs.num_attention_heads + 2 * globs.num_kv_heads)
        * globs.head_dim,
        is_split_batch_dim=False,
        is_split_output_dim=True,
    )
    if stop_after_op == "qkv":
        return instructions_waves, wave_buffer_sizes

    ins, buf = schedule_attention_prefill(globs, layer_idx, max_prefill_instructions_per_gpu)
    _register(ins, buf)
    if stop_after_op == "prefill":
        return instructions_waves, wave_buffer_sizes

    ins, buf = schedule_attention_decode(globs, layer_idx)
    _register(ins, buf)
    if stop_after_op == "decode":
        return instructions_waves, wave_buffer_sizes

    ins, buf = schedule_matmul(
        O_ProjResidual,
        layer_idx=layer_idx,
        device_idx=device_idx,
        globs=globs,
        global_outdim=globs.hidden_size,
        is_split_batch_dim=True,
        is_split_output_dim=False,
        supergroup_size=4,
    )

    _register(ins, buf)
    if stop_after_op == "oproj":
        return instructions_waves, wave_buffer_sizes

    _sched_norm(MLP_Norm)
    if stop_after_op == "mlp_norm":
        return instructions_waves, wave_buffer_sizes

    _sched_matmul(
        GateSilu,
        global_outdim=globs.intermediate_size,
        is_split_batch_dim=False,
        is_split_output_dim=True,
    )
    if stop_after_op == "gate":
        return instructions_waves, wave_buffer_sizes

    _sched_matmul(
        UpMatMul,
        global_outdim=globs.intermediate_size,
        is_split_batch_dim=False,
        is_split_output_dim=True,
    )
    if stop_after_op == "up":
        return instructions_waves, wave_buffer_sizes

    _sched_matmul(
        DownProjResidual,
        global_outdim=globs.hidden_size,
        is_split_batch_dim=False,
        is_split_output_dim=False,
    )

    return instructions_waves, wave_buffer_sizes


def schedule_die(
    globs: Globals,
):
    instructions: list[Instruction] = []
    for _ in range(globs.sm_count):
        instructions.append(Die())
    return instructions, len(instructions)


def schedule_all_device_barrier(
    globs: Globals,
    layer_idx: int,
    bar_idx: int,
):
    instructions: list[Instruction] = []
    for _ in range(globs.sm_count):
        instructions.append(AllDeviceBarrier(layer_idx=layer_idx, bar_idx=bar_idx))
    return instructions, len(instructions)

def schedule_model(
    globs: Globals,
    device_idx: int,
    layer_limit: int | None = None,
    interleave_waves: bool = False,
    interleave_buffer_size: int | None = None,
    stop_after_op: str | None = None,
    disable_lm_head: bool = False,
    add_final_sync: bool = True,
    max_prefill_instructions_per_gpu: int | None = None,
):
    assert stop_after_op is None or stop_after_op in [
        "attn_norm",
        "qkv",
        "decode",
        "prefill",
        "oproj",
        "mlp_norm",
        "gate",
        "up",
        "lm_head_norm",
    ]

    all_instructions_waves: list[list[Instruction]] = []
    all_wave_buffer_sizes = []

    # Add inc_barrier instruction at the start of model execution
    all_instructions_waves.append([IncBarrier()])
    all_wave_buffer_sizes.append(1)

    nlayers = layer_limit or globs.num_hidden_layers

    for layer_idx in range(nlayers):
        waves, sizes = schedule_layer(
            globs=globs,
            layer_idx=layer_idx,
            device_idx=device_idx,
            interleave_waves=interleave_waves,
            stop_after_op=stop_after_op,
            max_prefill_instructions_per_gpu=max_prefill_instructions_per_gpu,
        )
        all_instructions_waves.extend(waves)
        all_wave_buffer_sizes.extend(sizes)

    def _register(ins: list[Instruction], buf_size: int):
        if len(ins) == 0:
            return

        all_instructions_waves.append(ins)
        all_wave_buffer_sizes.append(buf_size)

    if nlayers == globs.num_hidden_layers:
        wave, size = schedule_norm(
            LM_Head_Norm,
            layer_idx=0,
            device_idx=device_idx,
            globs=globs,
            interleave_waves=interleave_waves,
        )
        _register(wave, size)

        if stop_after_op != "lm_head_norm" and not disable_lm_head:
            wave, size = schedule_matmul(
                LM_Head,
                layer_idx=0,
                device_idx=device_idx,
                globs=globs,
                global_outdim=globs.vocab_size,
                is_split_batch_dim=True,
                is_split_output_dim=False,
            )
            _register(wave, size)

    if add_final_sync:
        wave, size = schedule_all_device_barrier(globs, layer_idx=0, bar_idx=0)
        _register(wave, size)


    wave, size = schedule_die(globs)
    _register(wave, size)


    if interleave_waves:
        assert interleave_buffer_size is not None
        instructions = interleave_instruction_waves(
            all_instructions_waves,
            overlap_buffer_size=interleave_buffer_size,
            wave_buffer_sizes=all_wave_buffer_sizes,
        )
    else:
        instructions = list(chain(*all_instructions_waves))

    return instructions


def interleave(list_a: list, list_b: list):
    """
    Interleave, supporting variable lengths/ratios.
    """

    if len(list_a) < len(list_b):
        shorter_list = list_a
        longer_list = list_b
    else:
        shorter_list = list_b
        longer_list = list_a

    if len(shorter_list) == 0:
        return longer_list

    combined_list = []

    ratio = len(longer_list) / len(shorter_list)

    for i in range(len(shorter_list)):
        combined_list.append(shorter_list[i])

        longer_start = round(i * ratio)
        longer_end = round((i + 1) * ratio)

        combined_list.extend(longer_list[longer_start:longer_end])

    combined_list.extend(longer_list[round(len(shorter_list) * ratio) :])

    assert len(combined_list) == len(list_a) + len(list_b)

    return combined_list


def interleave_instruction_waves(
    instruction_waves: list[list[Instruction]],
    wave_buffer_sizes: list[int],
    overlap_buffer_size: int,
):
    for i, wave in enumerate(instruction_waves):
        assert len(wave) > 0, f"Empty wave at index {i}"

    wave_types = [ins[0].tags()["pool"] for ins in instruction_waves]
    wave_sizes = [len(ins) for ins in instruction_waves]

    num_waves = len(instruction_waves)

    def get_buffer_size(wave_idx: int):
        return wave_buffer_sizes[wave_idx] + overlap_buffer_size

    wave_partitions = []
    for wave_idx in range(num_waves):
        wave_type = wave_types[wave_idx]
        wave_size = wave_sizes[wave_idx]
        wave_buffer_size = get_buffer_size(wave_idx)

        max_overlappable = max(0, wave_size - wave_buffer_size)
        if max_overlappable == 0:
            wave_partitions.append((0, wave_size, 0))
            continue

        if wave_idx == 0:
            prev_wave_type = None
        else:
            prev_wave_type = wave_types[wave_idx - 1]

        if wave_idx == num_waves - 1:
            next_wave_type = None
        else:
            next_wave_type = wave_types[wave_idx + 1]

        prev_wave_diff = prev_wave_type is not None and prev_wave_type != wave_type
        next_wave_diff = next_wave_type is not None and next_wave_type != wave_type

        if prev_wave_diff:
            prev_size = wave_sizes[wave_idx - 1]
            prev_buffer_size = get_buffer_size(wave_idx - 1)
            usable_prev_size = max(prev_size - prev_buffer_size, 0)
        else:
            usable_prev_size = 0

        if next_wave_diff:
            next_size = wave_sizes[wave_idx + 1]
            next_buffer_size = get_buffer_size(wave_idx + 1)
            usable_next_size = max(next_size - next_buffer_size, 0)
        else:
            usable_next_size = 0

        denom = usable_prev_size + usable_next_size
        if denom == 0:
            denom = 1

        first_part = min(round(wave_size * usable_prev_size / denom), max_overlappable)
        last_part = min(round(wave_size * usable_next_size / denom), max_overlappable)

        middle_part = wave_size - first_part - last_part

        wave_partitions.append((first_part, middle_part, last_part))

    num_waves = len(instruction_waves)

    serialized_queue: list[Instruction] = []

    num_used_per_wave = [0] * num_waves

    def extract_instructions(wave_idx: int, num_instructions: int):
        num_used = num_used_per_wave[wave_idx]
        wave_size = wave_sizes[wave_idx]
        num_remaining = wave_size - num_used
        assert num_remaining >= num_instructions, (
            f"wave_idx: {wave_idx}, num_instructions: {num_instructions}, num_remaining: {num_remaining}"
        )
        extracted = instruction_waves[wave_idx][num_used : num_used + num_instructions]
        num_used_per_wave[wave_idx] += num_instructions
        return extracted

    def assign_instructions(wave_idx: int, num_instructions: int):
        extracted = extract_instructions(wave_idx, num_instructions)
        serialized_queue.extend(extracted)

    def assign_double(wave_idx1: int, size1: int, wave_idx2: int, size2: int):
        extracted1 = extract_instructions(wave_idx1, size1)
        extracted2 = extract_instructions(wave_idx2, size2)
        serialized_queue.extend(interleave(extracted1, extracted2))

    for wave_idx in range(num_waves):
        amount_pre, amount_middle, amount_post = wave_partitions[wave_idx]

        if wave_idx == 0:
            assert num_used_per_wave[wave_idx] == 0
            assign_instructions(wave_idx, amount_pre + amount_middle)
        else:
            assert num_used_per_wave[wave_idx] == amount_pre
            assign_instructions(wave_idx, amount_middle)

        if wave_idx < num_waves - 1:
            next_amount_pre = wave_partitions[wave_idx + 1][0]
            assign_double(wave_idx, amount_post, wave_idx + 1, next_amount_pre)
        else:
            assign_instructions(wave_idx, amount_post)

    num_og_instructions = sum([len(ins) for ins in instruction_waves])
    assert len(serialized_queue) == num_og_instructions

    return serialized_queue


def create_instruction_tensor(
    globs: Globals,
    device_idx: int,
    layer_limit: int | None = None,
    interleave_waves: bool = False,
    interleave_buffer_size: int | None = None,
    move_to_gpu: bool = True,
    stop_after_op: str | None = None,
    disable_lm_head: bool = False,
    max_prefill_instructions_per_gpu: int | None = None,
):
    if move_to_gpu:
        device = f"cuda:{device_idx}"
    else:
        device = "cpu"

    schedule = schedule_model(
        globs=globs,
        device_idx=device_idx,
        layer_limit=layer_limit,
        interleave_waves=interleave_waves,
        interleave_buffer_size=interleave_buffer_size,
        stop_after_op=stop_after_op,
        disable_lm_head=disable_lm_head,
        max_prefill_instructions_per_gpu=max_prefill_instructions_per_gpu,
    )

    # breakpoint()

    return schedule_to_tensor(schedule, globs, device)


def create_all_instruction_tensors(
    globs: Globals,
    layer_limit: int | None = None,
    interleave_waves: bool = False,
    interleave_buffer_size: int | None = None,
    move_to_gpu: bool = True,
    stop_after_op: str | None = None,
    disable_lm_head: bool = False,
    max_prefill_instructions_per_gpu: int | None = None,
):
    tensors: list[Tensor] = []

    for dev_idx in range(globs.tp_size):
        ins = create_instruction_tensor(
            globs=globs,
            device_idx=dev_idx,
            layer_limit=layer_limit,
            interleave_waves=interleave_waves,
            interleave_buffer_size=interleave_buffer_size,
            move_to_gpu=move_to_gpu,
            stop_after_op=stop_after_op,
            disable_lm_head=disable_lm_head,
            max_prefill_instructions_per_gpu=max_prefill_instructions_per_gpu,
        )
        tensors.append(ins)

    return tensors


def schedule_to_tensor(
    schedule: list[list[Instruction]],
    globs: Globals,
    device: str,
):
    if globs.global_work_queue_enabled:
        insts = torch.tensor(
            [serialize_and_pad(instruction) for instruction in schedule],
            dtype=torch.int32,
            device=device,
        )
        return insts

    else:
        divided_across_sms = round_robin_assign_to_sms(
            schedule, sm_count=globs.sm_count
        )

        insts = convert_instruction_queues_to_tensor(divided_across_sms, device)

        return insts
