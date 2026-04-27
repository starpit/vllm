tp_llama_config = {
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
        "3": {"name": "COMPUTE2_EVENT", "color": "#cc00aa"},
        "4": {"name": "COMPUTE3_EVENT", "color": "#cc44dd"},
        "5": {"name": "STORE_EVENT", "color": "#ffff00"},
        "6": {"name": "STORE2_EVENT", "color": "#ff8000"},
        "7": {"name": "WAIT_EVENT", "color": "#ff0000"},
        "8": {"name": "READY_EVENT", "color": "#00ff00"},
        "15": {"name": "ERROR_EVENT", "color": "#000000"}
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