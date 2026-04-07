// GENERATED: Fused prefill layer kernel ({{ mode_label }})

#define SM89_NUM_LAYERS             {{ nl }}
#define SM89_HIDDEN_DIM             {{ hd }}
#define SM89_INTERMEDIATE_DIM       {{ id }}
#define SM89_HEAD_DIM               {{ hdm }}
#define SM89_NUM_ATTENTION_HEADS    {{ nah }}
#define SM89_NUM_KV_HEADS           {{ nkh }}

#include "llama_sm89.cuh"
