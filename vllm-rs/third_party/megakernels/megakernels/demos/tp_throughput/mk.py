import sys
from pathlib import Path

import torch

from megakernels.demos.tp_throughput.globs import Globals


class TensorParallelMK_Interpreter:
    def __init__(self, mk_dir: Path, globs: Globals, multithread: bool = True):
        self.multithread = multithread

        sys.path.append(str(mk_dir.expanduser().absolute()))
        import mk_llama_tp

        self.globs = globs
        self.KittensClub = mk_llama_tp.KittensClub
        self.enable_all_p2p_access = mk_llama_tp.enable_all_p2p_access
        self.broadcast_lm_head_norm = mk_llama_tp.broadcast_lm_head_norm()

        print(
            f"TIMING_RECORD_ENABLED: {self.globs.timing_record_enabled}, MULTITHREAD: {self.multithread}"
        )

        self.make_globals = (
            mk_llama_tp.make_globals_mk_llama_tp_timer
            if self.globs.timing_record_enabled
            else mk_llama_tp.make_globals_mk_llama_tp
        )

        if self.multithread:
            self.mk_llama_tp = (
                mk_llama_tp.mk_llama_tp_timer
                if self.globs.timing_record_enabled
                else mk_llama_tp.mk_llama_tp
            )
        else:
            self.mk_llama_tp = (
                mk_llama_tp.mk_llama_tp_timer_single_device
                if self.globs.timing_record_enabled
                else mk_llama_tp.mk_llama_tp_single_device
            )

    def setup(self):
        device_ids = [i for i in range(self.globs.tp_size)]
        self.enable_all_p2p_access(device_ids)
        globs = self.globs

        # deliberately not
        args = [
            globs.barriers,
            globs.instructions,
            globs.timings,
            globs.global_instruction_index,
            globs.qkv_proj_weights,
            globs.attn_norm_weights,
            globs.o_proj_weights,
            globs.mlp_norm_weights,
            globs.up_proj_weights,
            globs.gate_proj_weights,
            globs.down_proj_weights,
            globs.lm_head_norm_weights,
            globs.lm_head_weights,
            globs.k_cache,
            globs.v_cache,
            globs.rope_cos,
            globs.rope_sin,
            globs.hidden_states,
            globs.post_attn_norm,
            globs.post_mlp_norm,
            globs.post_rope_q,
            globs.attn_out,
            globs.mlp_intermediates,
            globs.post_lm_head_norm,
            globs.logits,
            globs.position_ids,
            globs.kv_append_indices,
            globs.prefill_qo_indptr,
            globs.prefill_kv_indptr,
            globs.prefill_kv_indices,
            globs.prefill_kv_last_page_len,
            globs.decode_kv_indptr,
            globs.decode_kv_indices,
            globs.decode_kv_last_page_len,
            globs.attn_scale,
            globs.rms_norm_eps,
            globs.num_pages,
            globs.global_batch_size,
            globs.num_prefill_tokens(),
        ]

        # deliberately slicing off global_batch_size and num_prefill_tokens
        # since they may change between kernel launches and it's error-prone to
        # mess with this list
        self.kernel_globals: list = self.make_globals(*args)[:-2]

        if self.multithread:
            print("INFO: Using multithreaded interpreter, with proper streams.")
            streams = [torch.cuda.current_stream(i) for i in range(self.globs.tp_size)]
            print(f"Streams: {[((i, stream)) for i, stream in enumerate(streams)]}")
            self.club = self.KittensClub(device_ids, streams)
        else:
            print("WARNING: Not using multithreaded interpreter, without proper streams.")
            self.club = None

    def interpret(self):
        if self.multithread:
            self.mk_llama_tp(
                self.club,
                *self.kernel_globals,
                self.globs.global_batch_size,
                self.globs.num_prefill_tokens(),
            )
        else:
            self.mk_llama_tp(
                *self.kernel_globals,
                self.globs.global_batch_size,
                self.globs.num_prefill_tokens(),
                self.globs.tp_rank,
                stream=torch.cuda.current_stream(),
            )
