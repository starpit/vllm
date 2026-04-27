thunder_mla_timings_config = {
    "format_version": "1.2",
    "instruction_types": {
        "0": {"name": "No Op", "color": "#808080", "params": {}},
        "1": {"name": "Attention Partial", "color": "#1F77B4", "params": {
            "1": "uid",
            "2": "dst.batch_idx",
            "3": "dst.seq_idx",
            "4": "q_batch_idx",
            "5": "q_seq_idx",
            "6": "start_pos",
            "7": "end_pos",
            "8": "length"
        }},
        "2": {"name": "Attention Reduction", "color": "#8CBC9A", "params": {
            "1": "uid",
            "2": "num_iters",
            "3": "dst.batch_idx",
            "4": "dst.seq_idx",
            "5": "load_uid[0]",
            "6": "load_uid[1]",
            "7": "load_uid[2]",
            "8": "load_uid[3]",
            "9": "load_uid[4]",
            "10": "load_uid[5]",
            "11": "load_uid[6]",
            "12": "load_uid[7]",
            "13": "load_uid[8]",
            "14": "load_uid[9]",
            "15": "load_uid[10]",
            "16": "load_uid[11]",
            "17": "load_uid[12]",
            "18": "load_uid[13]",
            "19": "load_uid[14]",
            "20": "load_uid[15]",
            "21": "load_uid[16]",
            "22": "load_uid[17]",
            "23": "load_uid[18]",
            "24": "load_uid[19]",
            "25": "load_uid[20]",
            "26": "load_uid[21]",
            "27": "load_uid[22]",
            "28": "load_uid[23]",
            "29": "load_uid[24]",
            "30": "load_uid[25]",
            "31": "load_uid[26]"
        }},
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
    "num_gpus": 1,
    "total_processors": 132,
    "max_instructions": 1,
    "time_unit_flag": 1,
    "has_events_flag": 1
}