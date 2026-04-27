import math
from dataclasses import dataclass, field, replace
from functools import partial

import torch
from einops import rearrange
from torch import Tensor
from transformers import LlamaConfig

from megakernels.llama import ExtraModelConfig, StackedParams
from megakernels.scheduler import (
    INTS_PER_INSTRUCTION,
    TIMING_SLOTS,
    create_timing_tensor,
)
from megakernels.utils import assert_div, ceil_div, get_sm_count

MAX_NUM_OPS = 20

MATMUL_BATCH_BLOCK_SIZE = 128
MATMUL_OUTPUT_BLOCK_SIZE = 256
PREFILL_BLOCK_SIZE = 16

PAGE_SIZE = 128


def local_to_global_batch_idx(
    globs: "Globals",
    local_batch_idx: int,
) -> int:
    local_batch_block_idx, pos_in_block = divmod(
        local_batch_idx, globs.matmul_batch_block_size
    )
    global_batch_block_idx = local_batch_block_idx * globs.tp_size + globs.tp_rank
    global_batch_idx = (
        global_batch_block_idx * globs.matmul_batch_block_size + pos_in_block
    )

    return global_batch_idx


def copy_into(
    src: Tensor,
    dst: Tensor,
    strict: bool = False,
    non_blocking: bool = False,
    fill_val: int | float | None = 0,
):
    if fill_val is not None:
        dst.fill_(fill_val)

    if strict:
        dst.copy_(src, non_blocking=non_blocking)
    else:
        src_len = src.shape[0]
        dst[:src_len].copy_(src, non_blocking=non_blocking)


def do_block_round_robin(x: Tensor, tp_size: int, batch_block_size: int):
    remaining_shape = x.shape[1:]

    numel = x.shape[0]

    pad_multiple = batch_block_size * tp_size

    padded_numel = math.ceil(numel / pad_multiple) * pad_multiple

    num_padding = padded_numel - numel

    padded = torch.cat(
        [x, torch.zeros(num_padding, *remaining_shape, dtype=x.dtype, device=x.device)],
        dim=0,
    )

    viewed_as_blocks = padded.view(-1, tp_size, batch_block_size, *remaining_shape)

    round_robin_blocks = rearrange(
        viewed_as_blocks,
        "blocks_per_rank tp_size batch_block ... -> tp_size (blocks_per_rank batch_block) ...",
    )

    return round_robin_blocks


@dataclass
class Globals:
    # model parameters, all layers stacked together in order
    qkv_proj_weights: list[Tensor]
    attn_norm_weights: list[Tensor]
    o_proj_weights: list[Tensor]
    mlp_norm_weights: list[Tensor]
    up_proj_weights: list[Tensor]
    gate_proj_weights: list[Tensor]
    down_proj_weights: list[Tensor]
    lm_head_norm_weights: list[Tensor]
    lm_head_weights: list[Tensor]
    embed_weights: list[Tensor]

    # KV cache in paged format
    # Shape: (num_layers, num_pages, page_size, num_kv_heads, head_dim)
    k_cache: list[Tensor]
    v_cache: list[Tensor]

    # not stacked for each layer
    rope_cos: list[Tensor]
    rope_sin: list[Tensor]

    # activations
    hidden_states: list[Tensor]
    post_attn_norm: list[Tensor]
    post_rope_q: list[Tensor]
    attn_out: list[Tensor]

    post_mlp_norm: list[Tensor]
    mlp_intermediates: list[Tensor]

    post_lm_head_norm: list[Tensor]
    logits: list[Tensor]

    # Paged attention indices
    decode_kv_indices: list[Tensor]
    decode_kv_indptr: list[Tensor]
    decode_kv_last_page_len: list[Tensor]

    prefill_qo_indptr: list[Tensor]
    prefill_kv_indices: list[Tensor]
    prefill_kv_indptr: list[Tensor]
    prefill_kv_last_page_len: list[Tensor]

    position_ids: list[Tensor]
    kv_append_indices: list[Tensor]

    model_config: LlamaConfig

    attn_scale: float
    rms_norm_eps: float

    padded_global_batch_size: int
    padded_local_batch_size: int

    # the current value
    global_batch_size: int

    matmul_batch_block_size: int
    matmul_output_block_size: int
    prefill_block_size: int

    tp_size: int
    tp_rank: int

    global_work_queue_enabled: bool
    sm_count: int

    # Paged KV cache parameters
    page_size: int
    num_pages: int
    timing_record_enabled: bool

    barriers: list[Tensor]
    global_instruction_index: list[Tensor]
    instructions: list[Tensor] | None = None
    timings: list[Tensor] | None = None

    # for each sequence being prefilled,
    # how many tokens are we handling for this seq
    prefill_chunk_lens: list[int] = field(default_factory=list)

    # for each prefill seq, how many tokens are already in the kv cache for that seq
    # (helpful for when we are doing seq extension, e.g. after a prefill cache hit).
    prefill_extend_offsets: list[int] = field(default_factory=list)

    # Other things helpful to know for each GPU
    device_batch_size: list[int] | None = None

    def __post_init__(self):
        # model constants
        self.num_hidden_layers = self.model_config.num_hidden_layers
        self.num_attention_heads = self.model_config.num_attention_heads
        self.num_kv_heads = self.model_config.num_key_value_heads
        self.head_dim = self.model_config.head_dim
        self.hidden_size = self.model_config.hidden_size
        self.intermediate_size = self.model_config.intermediate_size
        self.vocab_size = self.model_config.vocab_size

        self.num_combined_heads = self.num_attention_heads + self.num_kv_heads * 2

    def local_batch_blocks(self, rank: int) -> int:
        num_blocks = ceil_div(self.global_batch_size, self.matmul_batch_block_size)
        div, remainder = divmod(num_blocks, self.tp_size)

        num_blocks_for_this_rank = div + (1 if rank < remainder else 0)
        return num_blocks_for_this_rank

    def qkv_dim(self) -> int:
        return (self.num_attention_heads + self.num_kv_heads * 2) * self.head_dim

    def num_batch_blocks(self) -> int:
        return ceil_div(self.global_batch_size, self.matmul_batch_block_size)

    def num_output_blocks(self) -> int:
        return assert_div(self.hidden_size, self.matmul_output_block_size)

    def num_intermediate_blocks(self) -> int:
        return assert_div(self.intermediate_size, self.matmul_output_block_size)

    def num_vocab_blocks(self) -> int:
        return assert_div(self.vocab_size, self.matmul_output_block_size)

    def num_prefill_tokens(self) -> int:
        return sum(self.prefill_chunk_lens)

    def num_decode_seqs(self) -> int:
        return self.global_batch_size - self.num_prefill_tokens()

    def with_new_activations(self, new_kv_cache: bool = True):
        def clone_list(lst: list[Tensor]) -> list[Tensor]:
            return [t.clone() for t in lst]

        def maybe_clone_list(
            lst: list[Tensor] | None, enabled: bool = True
        ) -> list[Tensor] | None:
            if lst is None or not enabled:
                return None
            return clone_list(lst)

        new_globals = replace(
            self,
            hidden_states=clone_list(self.hidden_states),
            post_attn_norm=clone_list(self.post_attn_norm),
            post_rope_q=clone_list(self.post_rope_q),
            attn_out=clone_list(self.attn_out),
            post_mlp_norm=clone_list(self.post_mlp_norm),
            mlp_intermediates=clone_list(self.mlp_intermediates),
            post_lm_head_norm=clone_list(self.post_lm_head_norm),
            logits=clone_list(self.logits),
            k_cache=clone_list(self.k_cache),
            v_cache=clone_list(self.v_cache),
            instructions=maybe_clone_list(self.instructions),
            timings=maybe_clone_list(self.timings),
            global_instruction_index=maybe_clone_list(self.global_instruction_index),
            decode_kv_indices=maybe_clone_list(self.decode_kv_indices),
            decode_kv_indptr=maybe_clone_list(self.decode_kv_indptr),
            decode_kv_last_page_len=maybe_clone_list(self.decode_kv_last_page_len),
            prefill_qo_indptr=maybe_clone_list(self.prefill_qo_indptr),
            prefill_kv_indices=maybe_clone_list(self.prefill_kv_indices),
            prefill_kv_indptr=maybe_clone_list(self.prefill_kv_indptr),
            prefill_kv_last_page_len=maybe_clone_list(self.prefill_kv_last_page_len),
            position_ids=maybe_clone_list(self.position_ids),
            kv_append_indices=maybe_clone_list(self.kv_append_indices),
        )

        return new_globals

    def copy_instructions(self, instructions: list[Tensor]):
        for i in range(self.tp_size):
            insts = instructions[i]
            if self.global_work_queue_enabled:
                assert insts.ndim == 2
                num_instructions = insts.shape[0]
                self.instructions[i].fill_(0)
                self.instructions[i][:num_instructions] = insts
            else:
                assert insts.ndim == 3
                num_instructions = insts.shape[1]
                self.instructions[i].fill_(0)
                self.instructions[i][:, :num_instructions] = insts

    def set_sizes(
        self,
        global_batch_size: int,
        prefill_chunk_lens: list[int] | None = None,
        prefill_extend_offsets: list[int] | None = None,
    ):
        if prefill_chunk_lens is None:
            prefill_chunk_lens = []

        if prefill_extend_offsets is None:
            prefill_extend_offsets = [0] * len(prefill_chunk_lens)

        self.global_batch_size = global_batch_size
        self.prefill_chunk_lens = prefill_chunk_lens
        self.prefill_extend_offsets = prefill_extend_offsets

        assert len(self.prefill_chunk_lens) == len(self.prefill_extend_offsets)

        self.device_batch_size = make_device_batch_sizes(
            global_batch_size, self.tp_size, self.matmul_batch_block_size
        )

    def copy_prefill_info(
        self,
        qo_indptr: Tensor,
        kv_indices: Tensor,
        kv_indptr: Tensor,
        kv_last_page_len: Tensor,
        device_idx: int | None = None,
        non_blocking: bool = True,
    ):
        if device_idx is None:
            devices = range(self.tp_size)
        else:
            devices = [device_idx]

        local_copy_into = partial(copy_into, non_blocking=non_blocking)

        for didx in devices:
            local_copy_into(
                qo_indptr,
                self.prefill_qo_indptr[didx],
            )
            local_copy_into(
                kv_indices,
                self.prefill_kv_indices[didx],
            )
            local_copy_into(
                kv_indptr,
                self.prefill_kv_indptr[didx],
            )
            local_copy_into(
                kv_last_page_len,
                self.prefill_kv_last_page_len[didx],
            )

    def copy_decode_info(
        self,
        kv_indices: Tensor,
        kv_indptr: Tensor,
        kv_last_page_len: Tensor,
        device_idx: int | None = None,
        non_blocking: bool = True,
    ):
        if device_idx is None:
            devices = range(self.tp_size)
        else:
            devices = [device_idx]

        local_copy_into = partial(copy_into, non_blocking=non_blocking)

        for didx in devices:
            local_copy_into(kv_indices, self.decode_kv_indices[didx])
            local_copy_into(kv_indptr, self.decode_kv_indptr[didx])
            local_copy_into(kv_last_page_len, self.decode_kv_last_page_len[didx])

    def copy_position_ids(
        self,
        position_ids: Tensor,
        device_idx: int | None = None,
        non_blocking: bool = True,
    ):
        if device_idx is None:
            devices = range(self.tp_size)
        else:
            devices = [device_idx]

        for didx in devices:
            copy_into(position_ids, self.position_ids[didx], non_blocking=non_blocking)

    def copy_append_indices(
        self,
        append_indices: Tensor,
        device_idx: int | None = None,
        non_blocking: bool = True,
    ):
        if device_idx is None:
            devices = range(self.tp_size)
        else:
            devices = [device_idx]

        fill_val = (self.num_pages * self.page_size) - 1

        for didx in devices:
            copy_into(
                append_indices,
                self.kv_append_indices[didx],
                non_blocking=non_blocking,
                fill_val=fill_val,
            )

    def copy_hidden_states(
        self,
        hidden_states: Tensor,
        device_idx: int | None = None,
        non_blocking: bool = True,
    ):
        if device_idx is None:
            devices = range(self.tp_size)
        else:
            devices = [device_idx]

        for didx in devices:
            copy_into(
                hidden_states, self.hidden_states[didx], non_blocking=non_blocking
            )


def make_device_batch_sizes(
    global_batch_size: int, num_devices: int, matmul_batch_block_size: int
):
    common = global_batch_size // num_devices // matmul_batch_block_size
    remainder = global_batch_size % (num_devices * matmul_batch_block_size)
    num_gpus_with_full_remainder = remainder // matmul_batch_block_size
    partial_remainder = remainder % matmul_batch_block_size
    return [
        common * matmul_batch_block_size
        + (matmul_batch_block_size if i < num_gpus_with_full_remainder else 0)
        + (partial_remainder if i == num_gpus_with_full_remainder else 0)
        for i in range(num_devices)
    ]


def make_globals(
    model_config: LlamaConfig,
    num_devices: int,
    global_batch_size: int,
    num_pages: int,
    barrier_init_val: int,
    global_work_queue_enabled: bool = False,
    meta_device: bool = False,
    timing_record_enabled: bool = False,
    layer_limit: int | None = None,
):
    # Read model config
    NUM_LAYERS = layer_limit or model_config.num_hidden_layers
    HIDDEN_DIM = model_config.hidden_size
    INTERMEDIATE_DIM = model_config.intermediate_size
    HEAD_DIM = HIDDEN_DIM // model_config.num_attention_heads
    NUM_ATTN_HEADS = model_config.num_attention_heads
    NUM_KV_HEADS = model_config.num_key_value_heads
    NUM_HEADS = NUM_ATTN_HEADS + NUM_KV_HEADS * 2
    VOCAB_SIZE = model_config.vocab_size
    attn_scale = 1 / (HEAD_DIM**0.5)
    rms_norm_eps = model_config.rms_norm_eps

    def make_tensor_list(*shape, dtype=torch.bfloat16, fill_val=0, aligned=False):
        if aligned:

            def go(i):
                return make_aligned_tensor(
                    shape,
                    device=i if not meta_device else "meta",
                    dtype=dtype,
                    fill_val=fill_val,
                )
        else:

            def go(i):
                return torch.full(
                    shape,
                    fill_val,
                    dtype=dtype,
                    device=i if not meta_device else "meta",
                )

        return [go(i) for i in range(num_devices)]

    device_batch_size = make_device_batch_sizes(
        global_batch_size, num_devices, MATMUL_BATCH_BLOCK_SIZE
    )
    padded_global_batch_size = ceil_div(
        global_batch_size, 1, alignment=MATMUL_BATCH_BLOCK_SIZE
    )
    padded_local_batch_size = ceil_div(
        global_batch_size, num_devices, alignment=MATMUL_BATCH_BLOCK_SIZE
    )

    # These lists of tensors must be generated first for the memory alignment requirements by multicast
    # Now using PGL sizing since these correspond to PGL tensors in CUDA
    post_attn_norm = make_tensor_list(
        padded_global_batch_size, HIDDEN_DIM, aligned=True
    )
    post_mlp_norm = make_tensor_list(padded_global_batch_size, HIDDEN_DIM, aligned=True)
    post_lm_head_norm = make_tensor_list(
        padded_global_batch_size, HIDDEN_DIM, aligned=True
    )
    print(post_attn_norm[0].shape)

    hidden_states = make_tensor_list(padded_local_batch_size, HIDDEN_DIM)
    attn_out = make_tensor_list(padded_local_batch_size, HIDDEN_DIM)

    post_rope_q = make_tensor_list(global_batch_size, HIDDEN_DIM // num_devices)
    silu_out = make_tensor_list(global_batch_size, INTERMEDIATE_DIM // num_devices)
    logits = make_tensor_list(padded_local_batch_size, VOCAB_SIZE)
    rope_cos = make_tensor_list(
        model_config.max_position_embeddings, HEAD_DIM, dtype=torch.float32
    )
    rope_sin = make_tensor_list(
        model_config.max_position_embeddings, HEAD_DIM, dtype=torch.float32
    )

    # These don't use multicast
    barriers = make_tensor_list(
        NUM_LAYERS,
        MAX_NUM_OPS,
        global_batch_size,
        NUM_HEADS,
        dtype=torch.int32,
        fill_val=barrier_init_val,
    )

    qkv_weights = make_tensor_list(
        NUM_LAYERS,
        NUM_HEADS * HEAD_DIM // num_devices,
        HIDDEN_DIM,
    )
    attn_norm_weights = make_tensor_list(NUM_LAYERS, HIDDEN_DIM)
    o_weights = make_tensor_list(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM)
    mlp_norm_weights = make_tensor_list(NUM_LAYERS, HIDDEN_DIM)
    up_weights = make_tensor_list(
        NUM_LAYERS,
        INTERMEDIATE_DIM // num_devices,
        HIDDEN_DIM,
    )
    gate_weights = make_tensor_list(
        NUM_LAYERS,
        INTERMEDIATE_DIM // num_devices,
        HIDDEN_DIM,
    )
    down_weights = make_tensor_list(
        NUM_LAYERS,
        HIDDEN_DIM,
        INTERMEDIATE_DIM // num_devices,
    )
    lm_head_norm_weights = make_tensor_list(HIDDEN_DIM)
    lm_head_weights = make_tensor_list(VOCAB_SIZE, HIDDEN_DIM)

    # This is used before the kernel launch
    embed_weights = make_tensor_list(VOCAB_SIZE, HIDDEN_DIM)

    # Initialize paged KV cache tensors
    # k_cache and v_cache need to be reshaped for paged format
    k_cache_paged = make_tensor_list(
        NUM_LAYERS * num_pages + 1,
        PAGE_SIZE,
        NUM_KV_HEADS // num_devices,
        HEAD_DIM,
    )
    v_cache_paged = make_tensor_list(
        NUM_LAYERS * num_pages + 1,
        PAGE_SIZE,
        NUM_KV_HEADS // num_devices,
        HEAD_DIM,
    )

    global_instruction_index = make_tensor_list(1, dtype=torch.int32)

    # arbitrary
    max_pages_per_seq = min(num_pages, 1024)

    # Paged attention indices
    decode_kv_indices = make_tensor_list(
        max_pages_per_seq * global_batch_size, dtype=torch.int32
    )
    decode_kv_indptr = make_tensor_list(global_batch_size + 1, dtype=torch.int32)
    decode_kv_last_page_len = make_tensor_list(global_batch_size, dtype=torch.int32)

    prefill_qo_indptr = make_tensor_list(global_batch_size + 1, dtype=torch.int32)
    prefill_kv_indices = make_tensor_list(
        max_pages_per_seq * global_batch_size, dtype=torch.int32
    )
    prefill_kv_indptr = make_tensor_list(global_batch_size + 1, dtype=torch.int32)
    prefill_kv_last_page_len = make_tensor_list(global_batch_size, dtype=torch.int32)

    position_ids = make_tensor_list(global_batch_size, dtype=torch.int32)
    kv_append_indices = make_tensor_list(global_batch_size, dtype=torch.int32)

    max_instructions_per_token = (
        512
        * (layer_limit if layer_limit is not None else model_config.num_hidden_layers)
        // model_config.num_hidden_layers
    )
    num_instructions = max_instructions_per_token * max(global_batch_size, 128)

    if global_work_queue_enabled:
        instructions = [
            torch.zeros(
                (num_instructions, INTS_PER_INSTRUCTION),
                dtype=torch.int32,
                device=f"cuda:{i}",
            )
            for i in range(num_devices)
        ]
    else:
        sm_count = get_sm_count("cuda:0")
        instructions_per_sm = math.ceil(num_instructions / sm_count)
        instructions = [
            torch.zeros(
                (sm_count, instructions_per_sm, INTS_PER_INSTRUCTION),
                dtype=torch.int32,
                device=f"cuda:{i}",
            )
            for i in range(num_devices)
        ]

    if timing_record_enabled:
        timings = [create_timing_tensor(insts) for insts in instructions]
    else:
        # dummy tensors, we just need to not be None
        timings = [
            torch.zeros(
                (1, TIMING_SLOTS),
                dtype=torch.int32,
                device=f"cuda:{i}",
            )
            for i in range(num_devices)
        ]

    return Globals(
        qkv_proj_weights=qkv_weights,
        attn_norm_weights=attn_norm_weights,
        o_proj_weights=o_weights,
        mlp_norm_weights=mlp_norm_weights,
        up_proj_weights=up_weights,
        gate_proj_weights=gate_weights,
        down_proj_weights=down_weights,
        lm_head_norm_weights=lm_head_norm_weights,
        lm_head_weights=lm_head_weights,
        embed_weights=embed_weights,
        post_rope_q=post_rope_q,
        mlp_intermediates=silu_out,
        post_lm_head_norm=post_lm_head_norm,
        logits=logits,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
        decode_kv_indices=decode_kv_indices,
        decode_kv_indptr=decode_kv_indptr,
        decode_kv_last_page_len=decode_kv_last_page_len,
        prefill_qo_indptr=prefill_qo_indptr,
        prefill_kv_indices=prefill_kv_indices,
        prefill_kv_indptr=prefill_kv_indptr,
        prefill_kv_last_page_len=prefill_kv_last_page_len,
        position_ids=position_ids,
        kv_append_indices=kv_append_indices,
        barriers=barriers,
        attn_out=attn_out,
        hidden_states=hidden_states,
        post_attn_norm=post_attn_norm,
        post_mlp_norm=post_mlp_norm,
        attn_scale=attn_scale,
        rms_norm_eps=rms_norm_eps,
        tp_size=num_devices,
        tp_rank=0,
        global_batch_size=global_batch_size,
        padded_global_batch_size=padded_global_batch_size,
        padded_local_batch_size=padded_local_batch_size,
        matmul_batch_block_size=MATMUL_BATCH_BLOCK_SIZE,
        matmul_output_block_size=MATMUL_OUTPUT_BLOCK_SIZE,
        prefill_block_size=PREFILL_BLOCK_SIZE,
        model_config=model_config,
        global_work_queue_enabled=global_work_queue_enabled,
        sm_count=get_sm_count("cuda:0"),
        page_size=PAGE_SIZE,
        num_pages=num_pages,
        timing_record_enabled=timing_record_enabled,
        device_batch_size=device_batch_size,
        global_instruction_index=global_instruction_index,
        instructions=instructions,
        timings=timings,
    )


@dataclass
class SingleDeviceActivations:
    hidden_states: Tensor

    post_attn_norm: Tensor
    post_rope_q: Tensor
    attn_out: Tensor

    post_mlp_norm: Tensor
    mlp_intermediates: Tensor

    post_lm_head_norm: Tensor
    logits: Tensor

    barriers: Tensor
    instructions: Tensor
    global_instruction_index: Tensor

    decode_kv_indices: Tensor
    decode_kv_indptr: Tensor
    decode_kv_last_page_len: Tensor

    prefill_qo_indptr: Tensor
    prefill_kv_indices: Tensor
    prefill_kv_indptr: Tensor
    prefill_kv_last_page_len: Tensor

    position_ids: Tensor
    kv_append_indices: Tensor

    timings: Tensor | None = None

    def clone(self):
        return replace(
            self,
            hidden_states=self.hidden_states.clone(),
            post_attn_norm=self.post_attn_norm.clone(),
            post_rope_q=self.post_rope_q.clone(),
            attn_out=self.attn_out.clone(),
            post_mlp_norm=self.post_mlp_norm.clone(),
            mlp_intermediates=self.mlp_intermediates.clone(),
            post_lm_head_norm=self.post_lm_head_norm.clone(),
            logits=self.logits.clone(),
            barriers=self.barriers.clone(),
            instructions=self.instructions.clone(),
            global_instruction_index=self.global_instruction_index.clone(),
            decode_kv_indices=self.decode_kv_indices.clone(),
            decode_kv_indptr=self.decode_kv_indptr.clone(),
            decode_kv_last_page_len=self.decode_kv_last_page_len.clone(),
            prefill_qo_indptr=self.prefill_qo_indptr.clone(),
            prefill_kv_indices=self.prefill_kv_indices.clone(),
            prefill_kv_indptr=self.prefill_kv_indptr.clone(),
            prefill_kv_last_page_len=self.prefill_kv_last_page_len.clone(),
            position_ids=self.position_ids.clone(),
            timings=self.timings.clone() if self.timings is not None else None,
        )


@dataclass
class SingleDeviceGlobals:
    stacked_params: StackedParams
    activations: SingleDeviceActivations


def make_aligned_tensor(
    shape,
    device,
    alignment=2 * 1024 * 1024,
    dtype=torch.bfloat16,
    fill_val=0,
):
    num_bytes = math.prod(shape) * dtype.itemsize
    over_allocation = torch.full(
        (num_bytes + alignment,), fill_val, dtype=torch.uint8, device=device
    )

    raw_ptr = over_allocation.data_ptr()

    # Calculate the offset needed to align to the granularity
    offset = (alignment - (raw_ptr % alignment)) % alignment

    # Create a view of the over-allocated tensor starting at the aligned position
    aligned_bytes = over_allocation[offset : offset + num_bytes]

    # View as the desired dtype and reshape to the target shape
    aligned_tensor = aligned_bytes.view(dtype).view(shape)

    return aligned_tensor


def make_activations(
    model_config: LlamaConfig,
    tp_size: int,
    global_batch_size: int,
    device: str | torch.device,
    barrier_init_val: int,
    use_timings: bool = False,
    use_gwq: bool = False,
):
    def make_tensor(shape, dtype=torch.bfloat16, fill_val=0):
        return torch.full(shape, fill_val, dtype=dtype, device=device)

    make_aligned = partial(make_aligned_tensor, device=device)

    def make_int_buffer(shape):
        return torch.zeros(shape, device=device, dtype=torch.int32)

    # arbitrary, we just don't want this buffer to overflow
    max_pages_per_seq = 1024

    assert global_batch_size % tp_size == 0
    assert model_config.hidden_size % tp_size == 0
    assert model_config.intermediate_size % tp_size == 0

    sharded_hidden_size = model_config.hidden_size // tp_size
    sharded_intermediate_size = model_config.intermediate_size // tp_size

    padded_global_batch_size = ceil_div(global_batch_size, MATMUL_BATCH_BLOCK_SIZE)
    padded_local_batch_size = ceil_div(
        global_batch_size, tp_size * MATMUL_BATCH_BLOCK_SIZE
    )

    max_instructions_per_token = 512
    num_instructions = max_instructions_per_token * global_batch_size

    if use_gwq:
        instructions = make_tensor(
            (num_instructions, INTS_PER_INSTRUCTION), dtype=torch.int32
        )
    else:
        sm_count = get_sm_count(device)
        instructions_per_sm = math.ceil(num_instructions / sm_count)
        instructions = make_tensor(
            (sm_count, instructions_per_sm, INTS_PER_INSTRUCTION), dtype=torch.int32
        )

    if use_timings:
        timings = create_timing_tensor(instructions)
    else:
        timings = None

    return SingleDeviceActivations(
        # PGL tensors (need minimum sizing)
        hidden_states=make_tensor((padded_local_batch_size, model_config.hidden_size)),
        attn_out=make_tensor((padded_local_batch_size, model_config.hidden_size)),
        post_mlp_norm=make_aligned(
            (padded_global_batch_size, model_config.hidden_size)
        ),
        post_lm_head_norm=make_tensor(
            (padded_global_batch_size, model_config.hidden_size)
        ),
        # Regular tensors (can be exact size)
        logits=make_tensor((padded_local_batch_size, model_config.vocab_size)),
        # PGL tensors that need device-local sizing with minimum
        post_attn_norm=make_aligned(
            (padded_global_batch_size, model_config.hidden_size)
        ),
        post_rope_q=make_tensor((padded_global_batch_size, sharded_hidden_size)),
        mlp_intermediates=make_tensor(
            (padded_global_batch_size, sharded_intermediate_size)
        ),
        barriers=make_tensor(
            (MAX_NUM_OPS, padded_global_batch_size, model_config.num_attention_heads),
            dtype=torch.int32,
            fill_val=barrier_init_val,
        ),
        global_instruction_index=make_tensor((), dtype=torch.int32),
        instructions=instructions,
        decode_kv_indices=make_int_buffer((global_batch_size * max_pages_per_seq,)),
        decode_kv_indptr=make_int_buffer((global_batch_size + 1,)),
        decode_kv_last_page_len=make_int_buffer((global_batch_size,)),
        prefill_qo_indptr=make_int_buffer((global_batch_size + 1,)),
        prefill_kv_indices=make_int_buffer((global_batch_size * max_pages_per_seq,)),
        prefill_kv_indptr=make_int_buffer((global_batch_size + 1,)),
        prefill_kv_last_page_len=make_int_buffer((global_batch_size,)),
        kv_append_indices=make_int_buffer((global_batch_size,)),
        position_ids=make_int_buffer((global_batch_size,)),
        timings=timings,
    )


def make_single_device_globals(
    model_config: LlamaConfig,
    extra_config: ExtraModelConfig,
    stacked_params: StackedParams,
    global_batch_size: int,
    device: str | torch.device,
    use_gwq: bool,
    use_timings: bool,
):
    return SingleDeviceGlobals(
        stacked_params=stacked_params,
        activations=make_activations(
            model_config=model_config,
            tp_size=extra_config.tp_size,
            global_batch_size=global_batch_size,
            device=device,
            barrier_init_val=0,
            use_gwq=use_gwq,
            use_timings=use_timings,
        ),
    )
