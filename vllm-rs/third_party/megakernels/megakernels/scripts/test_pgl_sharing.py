from functools import partial
from pathlib import Path

import pydra
import torch
import torch.multiprocessing as mp
from megakernels.demos.tp_throughput.globs import (
    MATMUL_BATCH_BLOCK_SIZE,
    MATMUL_OUTPUT_BLOCK_SIZE,
    Globals,
    SingleDeviceGlobals,
    make_single_device_globals,
)
from megakernels.demos.tp_throughput.mk import TensorParallelMK_Interpreter
from megakernels.llama import ExtraModelConfig, LlamaConfig, StackedParams
from megakernels.utils import get_sm_count
from tqdm import tqdm


class ScriptConfig(pydra.Config):
    ngpu: int = 8
    num_pages: int = 1024
    page_size: int = 128
    global_batch_size: int = 1024
    use_timings: bool = False
    use_gwq: bool = True
    mk_dir: str = "/home/bfs/Megakernels-Private/demos/cross-gpu-llama/"
    multi_rank: bool = False


def assemble_globs(
    model_config: LlamaConfig,
    config: ScriptConfig,
    all_single_device_globs: list[SingleDeviceGlobals],
    rank: int,
):
    globs = Globals(
        qkv_proj_weights=[x.stacked_params.qkv_proj for x in all_single_device_globs],
        o_proj_weights=[x.stacked_params.o_proj for x in all_single_device_globs],
        attn_norm_weights=[
            x.stacked_params.attn_norm_weight for x in all_single_device_globs
        ],
        mlp_norm_weights=[
            x.stacked_params.mlp_norm_weight for x in all_single_device_globs
        ],
        up_proj_weights=[x.stacked_params.up_proj for x in all_single_device_globs],
        gate_proj_weights=[x.stacked_params.gate_proj for x in all_single_device_globs],
        down_proj_weights=[x.stacked_params.down_proj for x in all_single_device_globs],
        lm_head_norm_weights=[
            x.stacked_params.lm_head_norm_weight for x in all_single_device_globs
        ],
        lm_head_weights=[x.stacked_params.lm_head for x in all_single_device_globs],
        embed_weights=[
            x.stacked_params.embedding_table for x in all_single_device_globs
        ],
        rope_cos=[x.stacked_params.rope_cos for x in all_single_device_globs],
        rope_sin=[x.stacked_params.rope_sin for x in all_single_device_globs],
        k_cache=[x.stacked_params.k_cache for x in all_single_device_globs],
        v_cache=[x.stacked_params.v_cache for x in all_single_device_globs],
        hidden_states=[x.activations.hidden_states for x in all_single_device_globs],
        post_attn_norm=[x.activations.post_attn_norm for x in all_single_device_globs],
        post_rope_q=[x.activations.post_rope_q for x in all_single_device_globs],
        attn_out=[x.activations.attn_out for x in all_single_device_globs],
        post_mlp_norm=[x.activations.post_mlp_norm for x in all_single_device_globs],
        mlp_intermediates=[
            x.activations.mlp_intermediates for x in all_single_device_globs
        ],
        post_lm_head_norm=[
            x.activations.post_lm_head_norm for x in all_single_device_globs
        ],
        logits=[x.activations.logits for x in all_single_device_globs],
        barriers=[x.activations.barriers for x in all_single_device_globs],
        instructions=[x.activations.instructions for x in all_single_device_globs],
        timings=[x.activations.timings for x in all_single_device_globs],
        global_instruction_index=[
            x.activations.global_instruction_index for x in all_single_device_globs
        ],
        decode_kv_indices=[
            x.activations.decode_kv_indices for x in all_single_device_globs
        ],
        decode_kv_indptr=[
            x.activations.decode_kv_indptr for x in all_single_device_globs
        ],
        decode_kv_last_page_len=[
            x.activations.decode_kv_last_page_len for x in all_single_device_globs
        ],
        prefill_qo_indptr=[
            x.activations.prefill_qo_indptr for x in all_single_device_globs
        ],
        prefill_kv_indices=[
            x.activations.prefill_kv_indices for x in all_single_device_globs
        ],
        prefill_kv_indptr=[
            x.activations.prefill_kv_indptr for x in all_single_device_globs
        ],
        prefill_kv_last_page_len=[
            x.activations.prefill_kv_last_page_len for x in all_single_device_globs
        ],
        position_ids=[x.activations.position_ids for x in all_single_device_globs],
        kv_append_indices=[
            x.activations.kv_append_indices for x in all_single_device_globs
        ],
        model_config=model_config,
        attn_scale=1 / (model_config.head_dim**0.5),
        rms_norm_eps=model_config.rms_norm_eps,
        global_batch_size=config.global_batch_size,
        matmul_batch_block_size=MATMUL_BATCH_BLOCK_SIZE,
        matmul_output_block_size=MATMUL_OUTPUT_BLOCK_SIZE,
        tp_size=config.ngpu,
        tp_rank=rank,
        global_work_queue_enabled=config.use_gwq,
        sm_count=get_sm_count(f"cuda:{rank}"),
        page_size=config.page_size,
        num_pages=config.num_pages,
        timing_record_enabled=config.use_timings,
    )

    return globs


def make_interpreter_one_rank(config: ScriptConfig):
    model_config: LlamaConfig = LlamaConfig.from_pretrained(
        "meta-llama/Llama-3.1-70B-Instruct"
    )

    all_single_device_globs: list[SingleDeviceGlobals] = []

    for rank in tqdm(range(config.ngpu), desc="Creating single device globs"):
        device = f"cuda:{rank}"

        stacked_params = StackedParams.from_config(
            config=model_config,
            tp_size=config.ngpu,
            device=device,
            num_pages=config.num_pages,
            page_size=config.page_size,
        )

        single_device_globs = make_single_device_globals(
            model_config=model_config,
            extra_config=ExtraModelConfig(
                tp_size=config.ngpu,
                tp_rank=rank,
            ),
            stacked_params=stacked_params,
            global_batch_size=config.global_batch_size,
            device=device,
            use_gwq=config.use_gwq,
            use_timings=True,
        )

        all_single_device_globs.append(single_device_globs)

    globs = assemble_globs(model_config, config, all_single_device_globs, rank=rank)

    interpreter = TensorParallelMK_Interpreter(
        mk_dir=Path(config.mk_dir),
        globs=globs,
    )

    return interpreter


def exchange_through_queues(
    queues,
    x,
    tp_rank: int,
):
    num_devices = len(queues)

    for _ in range(num_devices - 1):
        queues[tp_rank].put(x)

    gathered = []
    for i in range(num_devices):
        if i == tp_rank:
            gathered.append(x)
        else:
            gathered.append(queues[i].get())

    return gathered


def make_interpreter_multi_rank(config: ScriptConfig, rank: int, queues):
    model_config: LlamaConfig = LlamaConfig.from_pretrained(
        "meta-llama/Llama-3.1-70B-Instruct"
    )

    device = f"cuda:{rank}"

    stacked_params = StackedParams.from_config(
        config=model_config,
        tp_size=config.ngpu,
        device=device,
        num_pages=config.num_pages,
        page_size=config.page_size,
    )

    single_device_globs = make_single_device_globals(
        model_config=model_config,
        extra_config=ExtraModelConfig(
            tp_size=config.ngpu,
            tp_rank=rank,
        ),
        stacked_params=stacked_params,
        global_batch_size=config.global_batch_size,
        device=device,
        use_gwq=config.use_gwq,
        use_timings=True,
    )

    if rank == 0:
        print("exchanging through queues start")

    all_single_device_globs: list[SingleDeviceGlobals] = exchange_through_queues(
        queues, single_device_globs, rank
    )

    if rank == 0:
        print("exchanging through queues end")

    globs = assemble_globs(model_config, config, all_single_device_globs, rank=rank)

    interpreter = TensorParallelMK_Interpreter(
        mk_dir=Path(config.mk_dir),
        globs=globs,
    )

    return interpreter


def go(rank, queues, config: ScriptConfig):
    if config.multi_rank:
        interpreter = make_interpreter_multi_rank(config, rank, queues)
    else:
        if rank == 0:
            interpreter = make_interpreter_one_rank(config)
        else:
            interpreter = None

    if rank == 0:
        print("created interpreter")
        interpreter.setup()
        breakpoint()
        print("setup interpreter")

        # TODO pickle, share across queues, etc.
    else:
        # TODO: receive and use PGL stuff.
        pass


@torch.inference_mode()
def main(config: ScriptConfig):
    mp.set_start_method("spawn")

    queues = [mp.Queue() for _ in range(config.ngpu)]

    func = partial(go, config=config, queues=queues)

    procs = []

    for rank in range(1, config.ngpu):
        proc = mp.Process(target=func, args=(rank,))
        procs.append(proc)
        proc.start()

    func(0)

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    pydra.run(main)