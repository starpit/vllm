import huggingface_hub
import json
import re
import torch
import sys
from einops import einsum, rearrange
from pathlib import Path
from safetensors import safe_open
from transformers import AutoTokenizer, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from tqdm import tqdm

class Config:
    model: str = "meta-llama/Llama-3.1-70b-Instruct"
    cache_dir: str = "/data/ssul/llama-70b"
    mk_dir = Path(__file__).parent.parent.parent / "demos" / "cross-gpu-llama"
    batch_size: int = 2048
    max_seq_len: int = 128
    prompt: str = "tell me a funny joke about cookies"
    ntok: int = 20

    # Must-not-change fields
    num_devices: int = 8
    num_ops: int = 10
    interleave_rope: bool = True
    batch_block_size: int = 128
    matmul_block_size: int = 256

config = Config()
sys.path.append(str(config.mk_dir.expanduser().absolute()))
from mk_llama_tp import mk_llama_tp, KittensClub, make_globals, enable_all_p2p_access


########################################################
# Download model repository and load model config
########################################################


# Download model repo
snapshot_path_str = huggingface_hub.snapshot_download(
    config.model,
    allow_patterns=["*.safetensors", "*.json"],
    cache_dir=config.cache_dir
)
snapshot_path = Path(snapshot_path_str)

# Load model config
model_config = LlamaConfig.from_pretrained(config.model, cache_dir=config.cache_dir)

# Read model config
NUM_LAYERS = model_config.num_hidden_layers
HIDDEN_DIM = model_config.hidden_size
INTERMEDIATE_DIM = model_config.intermediate_size
HEAD_DIM = HIDDEN_DIM // model_config.num_attention_heads
NUM_ATTN_HEADS = model_config.num_attention_heads
NUM_KV_HEADS = model_config.num_key_value_heads
NUM_HEADS = NUM_ATTN_HEADS + NUM_KV_HEADS * 2
VOCAB_SIZE = model_config.vocab_size
vocab_blocks_per_dev = (VOCAB_SIZE // config.matmul_block_size - 1 + config.num_devices) // config.num_devices
attn_scale = 1 / (HEAD_DIM ** 0.5)
rms_norm_eps = model_config.rms_norm_eps
assert config.max_seq_len <= model_config.max_position_embeddings


########################################################
# Allocate tensors
########################################################


# No helper functions or utility classes for clarity
# These 3 lists of tensors must be generated first for the memory alignment requirements by multicast
hidden_states =             [torch.empty(config.batch_size, HIDDEN_DIM, dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
rms_rope_intermediates =    [torch.empty(config.batch_size, HIDDEN_DIM, dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
rms_gate_intermediates =    [torch.empty(config.batch_size, HIDDEN_DIM, dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]

# These don't use multicast
Bar =                       [torch.empty(NUM_LAYERS, config.num_ops, config.batch_size, NUM_HEADS,       dtype=torch.uint32, device=i) for i in range(config.num_devices)]
qkv_weights =               [torch.empty(NUM_LAYERS, NUM_HEADS*HEAD_DIM//config.num_devices, HIDDEN_DIM, dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
attn_norm_weights =         [torch.empty(NUM_LAYERS, HIDDEN_DIM,                                         dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
o_weights =                 [torch.empty(NUM_LAYERS, HIDDEN_DIM, HIDDEN_DIM//config.num_devices,         dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
mlp_norm_weights =          [torch.empty(NUM_LAYERS, HIDDEN_DIM,                                         dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
up_weights =                [torch.empty(NUM_LAYERS, INTERMEDIATE_DIM//config.num_devices, HIDDEN_DIM,   dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
gate_weights =              [torch.empty(NUM_LAYERS, INTERMEDIATE_DIM//config.num_devices, HIDDEN_DIM,   dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
down_weights =              [torch.empty(NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM//config.num_devices,   dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
lm_head_norm_weights =      [torch.empty(HIDDEN_DIM,                                                     dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
lm_head_weights =           [torch.empty(VOCAB_SIZE, HIDDEN_DIM,                                         dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
q_post_rope =               [torch.empty(config.batch_size, HIDDEN_DIM//config.num_devices,              dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
attn_out =                  [torch.empty(config.batch_size, HIDDEN_DIM//config.num_devices,              dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
silu_out =                  [torch.empty(config.batch_size, INTERMEDIATE_DIM//config.num_devices,        dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
rms_lm_head_intermediates = [torch.empty(config.batch_size//config.num_devices, HIDDEN_DIM,              dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
logits =                    [torch.empty(config.batch_size//config.num_devices, VOCAB_SIZE,              dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
rope_cos =                  [torch.empty(config.max_seq_len, HEAD_DIM,                                   dtype=torch.float32, device=i) for i in range(config.num_devices)]
rope_sin =                  [torch.empty(config.max_seq_len, HEAD_DIM,                                   dtype=torch.float32, device=i) for i in range(config.num_devices)]
k_cache =                   [torch.empty(NUM_LAYERS*config.batch_size, config.max_seq_len, NUM_KV_HEADS//config.num_devices, HEAD_DIM, dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]
v_cache =                   [torch.empty(NUM_LAYERS*config.batch_size, config.max_seq_len, NUM_KV_HEADS//config.num_devices, HEAD_DIM, dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]

# This is used before the kernel launch
embed_weights =             [torch.empty(VOCAB_SIZE, HIDDEN_DIM,                                         dtype=torch.bfloat16, device=i) for i in range(config.num_devices)]


########################################################
# Load model weights
########################################################


# Load safetensor files
safetensors_index_path = snapshot_path / "model.safetensors.index.json"
assert not (snapshot_path / "model.safetensors").exists()
assert safetensors_index_path.exists()
with open(safetensors_index_path, "r") as f:
    safetensors_index = json.load(f)

# Load model weights
filenames = set(safetensors_index['weight_map'].values())
for filename in tqdm(filenames, desc="Loading safetensor files"):
    for dev_idx in range(config.num_devices):
        with safe_open(snapshot_path / filename, framework="pt", device=dev_idx) as f:
            for key in f.keys():
                tensor_slice = f.get_slice(key)
                match = re.search(r"layers\.(\d+)", key)
                if match is not None:
                    layer_idx = int(match.group(1))
                else:
                    layer_idx = -1
                if key.endswith("input_layernorm.weight"): # (8192,)
                    assert match is not None
                    assert tensor_slice.get_shape() == [8192,]
                    attn_norm_weights[dev_idx][layer_idx, :] = tensor_slice[:]
                elif key.endswith("self_attn.q_proj.weight"): # (8192, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [8192, 8192]
                    elem_per_dev = NUM_ATTN_HEADS*HEAD_DIM//config.num_devices
                    qkv_weights[dev_idx][layer_idx, :elem_per_dev, :] = tensor_slice[dev_idx*elem_per_dev:(dev_idx+1)*elem_per_dev, :]
                elif key.endswith("self_attn.k_proj.weight"): # (1024, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [1024, 8192]
                    num_q_elems = NUM_ATTN_HEADS*HEAD_DIM//config.num_devices
                    elem_per_dev = NUM_KV_HEADS*HEAD_DIM//config.num_devices
                    qkv_weights[dev_idx][layer_idx, num_q_elems:num_q_elems+elem_per_dev, :] = tensor_slice[dev_idx*elem_per_dev:(dev_idx+1)*elem_per_dev, :]
                elif key.endswith("self_attn.v_proj.weight"): # (1024, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [1024, 8192]
                    num_qk_elems = (NUM_ATTN_HEADS+NUM_KV_HEADS)*HEAD_DIM//config.num_devices
                    elem_per_dev = NUM_KV_HEADS*HEAD_DIM//config.num_devices
                    qkv_weights[dev_idx][layer_idx, num_qk_elems:num_qk_elems+elem_per_dev, :] = tensor_slice[dev_idx*elem_per_dev:(dev_idx+1)*elem_per_dev, :]
                elif key.endswith("self_attn.o_proj.weight"): # (8192, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [8192, 8192]
                    elem_per_dev = HIDDEN_DIM//config.num_devices
                    o_weights[dev_idx][layer_idx, :, :] = tensor_slice[:, dev_idx*elem_per_dev:(dev_idx+1)*elem_per_dev]
                elif key.endswith("post_attention_layernorm.weight"): # (8192,)
                    assert match is not None
                    assert tensor_slice.get_shape() == [8192,]
                    mlp_norm_weights[dev_idx][layer_idx, :] = tensor_slice[:]
                elif key.endswith("mlp.gate_proj.weight"): # (28672, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [28672, 8192]
                    elem_per_dev = INTERMEDIATE_DIM//config.num_devices
                    gate_weights[dev_idx][layer_idx, :, :] = tensor_slice[dev_idx*elem_per_dev:(dev_idx+1)*elem_per_dev, :]
                elif key.endswith("mlp.up_proj.weight"): # (28672, 8192)
                    assert match is not None
                    assert tensor_slice.get_shape() == [28672, 8192]
                    elem_per_dev = INTERMEDIATE_DIM//config.num_devices
                    up_weights[dev_idx][layer_idx, :, :] = tensor_slice[dev_idx*elem_per_dev:(dev_idx+1)*elem_per_dev, :]
                elif key.endswith("mlp.down_proj.weight"): # (8192, 28672)
                    assert match is not None
                    assert tensor_slice.get_shape() == [8192, 28672]
                    elem_per_dev = INTERMEDIATE_DIM//config.num_devices
                    down_weights[dev_idx][layer_idx, :, :] = tensor_slice[:, dev_idx*elem_per_dev:(dev_idx+1)*elem_per_dev]
                elif key == "model.norm.weight": # (8192,)
                    assert tensor_slice.get_shape() == [8192,]
                    lm_head_norm_weights[dev_idx][:] = tensor_slice[:]
                elif key == "lm_head.weight": # (128256, 8192)
                    assert tensor_slice.get_shape() == [128256, 8192]
                    lm_head_weights[dev_idx][:, :] = tensor_slice[:, :]
                elif key == "model.embed_tokens.weight": # (128256, 8192)
                    assert tensor_slice.get_shape() == [128256, 8192]
                    embed_weights[dev_idx][:, :] = tensor_slice[:]
                else:
                    raise ValueError(f"Unknown key: {key}")


########################################################
# Generate RoPE weights
########################################################


# Generate RoPE weights
dummy_float_input = torch.empty((1,), dtype=torch.float32, device='cpu')
position_ids = torch.arange(model_config.max_position_embeddings).unsqueeze(0)
_rope_cos, _rope_sin = LlamaRotaryEmbedding(config=model_config)(dummy_float_input, position_ids)
_rope_cos = _rope_cos.squeeze(0)
_rope_sin = _rope_sin.squeeze(0)
assert _rope_cos.dtype == torch.float32
assert _rope_sin.dtype == torch.float32


########################################################
# Interleave RoPE and QKV weights
########################################################


# Generate interleaved indices
interleaved_indices = []
for head_idx in range(NUM_HEADS//config.num_devices):
    for dim_idx in range(HEAD_DIM // 2):
        if head_idx < NUM_HEADS//config.num_devices - 1:
            interleaved_indices.append(dim_idx + head_idx * HEAD_DIM)
            interleaved_indices.append(dim_idx + head_idx * HEAD_DIM + HEAD_DIM // 2)
        else: # do not interleave V
            interleaved_indices.append(dim_idx * 2 + head_idx * HEAD_DIM)
            interleaved_indices.append(dim_idx * 2 + head_idx * HEAD_DIM + 1)

# Interleave RoPE weights and load them into HBMs
_rope_cos = _rope_cos[:, interleaved_indices[:HEAD_DIM]]
_rope_sin = _rope_sin[:, interleaved_indices[:HEAD_DIM]]
for dev_idx in range(config.num_devices):
    rope_cos[dev_idx][:] = _rope_cos[:config.max_seq_len, :].to(dev_idx)
    rope_sin[dev_idx][:] = _rope_sin[:config.max_seq_len, :].to(dev_idx)

# Interleave QKV weights
for dev_idx in range(config.num_devices):
    qkv_weights[dev_idx][:, :, :] = qkv_weights[dev_idx][:, interleaved_indices, :]


########################################################
# Define PyVM decode function
########################################################


# Batches per device for data parallelism
batch_per_device = config.batch_size // config.num_devices

# Helper function for interleaved RoPE
def rotate_half_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    stacked = torch.stack((-x2, x1), dim=-1)
    return stacked.view_as(x)

# Helper function for RMS norm
def rms_norm(inp: torch.Tensor, weight: torch.Tensor, eps: float):
    input_dtype = inp.dtype
    inp = inp.to(torch.float32)
    variance = inp.pow(2).mean(-1, keepdim=True)
    inp = inp * torch.rsqrt(variance + eps)
    return weight * inp.to(input_dtype)

# The PyVM decode function
def pyvm_generate(pos_id: int):

    for layer_idx in range(NUM_LAYERS):

        # Attention RMS norm
        for dev_idx in range(config.num_devices):
            rms_norm_output = rms_norm(
                inp=hidden_states[dev_idx][dev_idx*batch_per_device:(dev_idx+1)*batch_per_device, :],
                weight=attn_norm_weights[dev_idx][layer_idx],
                eps=rms_norm_eps,
            )
            for peer_dev_idx in range(config.num_devices):
                rms_rope_intermediates[peer_dev_idx][dev_idx*batch_per_device:(dev_idx+1)*batch_per_device, :] = rms_norm_output.to(peer_dev_idx)
                torch.cuda.synchronize(peer_dev_idx)
            torch.cuda.synchronize(dev_idx)

        # QKV & RopE & KV cache append
        for dev_idx in range(config.num_devices):
            matmul_output = einsum(
                qkv_weights[dev_idx][layer_idx],
                rms_rope_intermediates[dev_idx],
                "o i, b i -> b o",
            )
            q = matmul_output[:, :NUM_ATTN_HEADS*HEAD_DIM//config.num_devices]
            k = matmul_output[:, NUM_ATTN_HEADS*HEAD_DIM//config.num_devices:NUM_ATTN_HEADS*HEAD_DIM//config.num_devices+NUM_KV_HEADS*HEAD_DIM//config.num_devices]
            v = matmul_output[:, NUM_ATTN_HEADS*HEAD_DIM//config.num_devices+NUM_KV_HEADS*HEAD_DIM//config.num_devices:]
            q = rearrange(q, "... (h d) -> ... h d", d=HEAD_DIM)
            k = rearrange(k, "... (h d) -> ... h d", d=HEAD_DIM)
            v = rearrange(v, "... (h d) -> ... h d", d=HEAD_DIM)
            cos = rope_cos[dev_idx][pos_id][torch.newaxis, torch.newaxis, :]
            sin = rope_sin[dev_idx][pos_id][torch.newaxis, torch.newaxis, :]
            q_with_rope = q * cos + rotate_half_interleaved(q) * sin
            k_with_rope = k * cos + rotate_half_interleaved(k) * sin
            q_with_rope = rearrange(q_with_rope, "... h d -> ... (h d)", d=HEAD_DIM)
            q_post_rope[dev_idx][:] = q_with_rope
            k_cache[dev_idx][layer_idx * config.batch_size:layer_idx * config.batch_size + config.batch_size, pos_id, 0, :] = k_with_rope[:, 0, :]
            v_cache[dev_idx][layer_idx * config.batch_size:layer_idx * config.batch_size + config.batch_size, pos_id, 0, :] = v[:, 0, :]
            torch.cuda.synchronize(dev_idx)

        # Attention decode
        for dev_idx in range(config.num_devices):
            q = rearrange(
                q_post_rope[dev_idx],
                "b (h d) -> b h d",
                b=config.batch_size,
                d=HEAD_DIM,
            ) # (batch_size, num_heads, head_dim)
            k = k_cache[dev_idx][layer_idx * config.batch_size:layer_idx * config.batch_size + config.batch_size, :pos_id+1, 0, :] # (batch_size, seq_len, head_dim)
            v = v_cache[dev_idx][layer_idx * config.batch_size:layer_idx * config.batch_size + config.batch_size, :pos_id+1, 0, :] # (batch_size, seq_len, head_dim)
            qk = torch.matmul(q.float(), k.float().transpose(-1, -2)) # (batch_size, num_heads, seq_len)
            scaled_qk = qk * attn_scale
            softmax = torch.softmax(scaled_qk, dim=-1)
            out = torch.matmul(softmax.float(), v.float()) # (batch_size, num_heads, head_dim)
            out = rearrange(out, "b h d -> b (h d)")
            attn_out[dev_idx][:] = out
            torch.cuda.synchronize(dev_idx)

        # Attention output projection + residual add
        for dev_idx in range(config.num_devices):
            matmul_output = einsum(attn_out[dev_idx], o_weights[dev_idx][layer_idx], "b i, o i -> b o")
            for peer_dev_idx in range(config.num_devices):
                hidden_states[peer_dev_idx][:] += matmul_output.to(peer_dev_idx)
                torch.cuda.synchronize(peer_dev_idx)
            torch.cuda.synchronize(dev_idx)

        # MLP RMS norm
        for dev_idx in range(config.num_devices):
            rms_norm_output = rms_norm(
                inp=hidden_states[dev_idx][dev_idx*batch_per_device:(dev_idx+1)*batch_per_device, :],
                weight=mlp_norm_weights[dev_idx][layer_idx],
                eps=rms_norm_eps,
            )
            for peer_dev_idx in range(config.num_devices):
                rms_gate_intermediates[peer_dev_idx][dev_idx*batch_per_device:(dev_idx+1)*batch_per_device, :] = rms_norm_output.to(peer_dev_idx)
                torch.cuda.synchronize(peer_dev_idx)
            torch.cuda.synchronize(dev_idx)

        # Gate + SiLU
        for dev_idx in range(config.num_devices):
            matmul_output = einsum(rms_gate_intermediates[dev_idx], gate_weights[dev_idx][layer_idx], "b i, o i -> b o")
            post_silu = torch.nn.functional.silu(matmul_output)
            silu_out[dev_idx][:] = post_silu
            torch.cuda.synchronize(dev_idx)

        # Up
        for dev_idx in range(config.num_devices):
            matmul_output = einsum(rms_gate_intermediates[dev_idx], up_weights[dev_idx][layer_idx], "b i, o i -> b o")
            gated = matmul_output * silu_out[dev_idx]
            silu_out[dev_idx][:] = gated
            torch.cuda.synchronize(dev_idx)

        # Down + residual add
        for dev_idx in range(config.num_devices):
            matmul_output = einsum(silu_out[dev_idx], down_weights[dev_idx][layer_idx], "b i, o i -> b o")
            for peer_dev_idx in range(config.num_devices):
                hidden_states[peer_dev_idx][:] += matmul_output.to(peer_dev_idx)
                torch.cuda.synchronize(peer_dev_idx)
            torch.cuda.synchronize(dev_idx)

    # LM head RMS norm
    for dev_idx in range(config.num_devices):
        rms_lm_head_intermediates[dev_idx][:, :] = rms_norm(
            inp=hidden_states[dev_idx][dev_idx*batch_per_device:(dev_idx+1)*batch_per_device, :],
            weight=lm_head_norm_weights[dev_idx],
            eps=rms_norm_eps,
        )
        torch.cuda.synchronize(dev_idx)

    # LM head
    for dev_idx in range(config.num_devices):
        logits[dev_idx][:, :] = einsum(
            rms_lm_head_intermediates[dev_idx],
            lm_head_weights[dev_idx],
            "b i, o i -> b o",
        )
        torch.cuda.synchronize(dev_idx)


########################################################
# Generate input hidden states
########################################################


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(snapshot_path_str)

# Generate input embeddings
input_ids = tokenizer(config.prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
embeddings = [torch.nn.Embedding.from_pretrained(embed_weights[dev_idx], freeze=True) for dev_idx in range(config.num_devices)]
input_hidden_states = [embeddings[dev_idx](input_ids.to(dev_idx).repeat(batch_per_device, 1)) for dev_idx in range(config.num_devices)] # (batch_size_per_device, seq_len, hidden_dim)


########################################################
# Prefill the KV cache
########################################################


# Hack: Just iteratively fill in using our decode kernel
input_seq_len = input_ids.shape[-1]
for pos_id in tqdm(range(input_seq_len), desc="Prefilling"):

    # Fill in the hidden_states
    for dev_idx in range(config.num_devices):
        hidden_states[dev_idx][dev_idx*batch_per_device:(dev_idx+1)*batch_per_device, :] = input_hidden_states[dev_idx][:, pos_id, :]

    # ⭐️ Launch the PyVM ⭐️
    pyvm_generate(pos_id)


########################################################
# Generate tokens
########################################################


# Accumulator tensor
output_ids = [torch.empty((batch_per_device, config.ntok), dtype=torch.int32, device=dev_idx) for dev_idx in range(config.num_devices)]

# Generate the first token
for dev_idx in range(config.num_devices):
    output_ids[dev_idx][:, 0] = torch.argmax(logits[dev_idx], dim=-1)
    hidden_states[dev_idx][dev_idx*batch_per_device:(dev_idx+1)*batch_per_device, :] = embeddings[dev_idx](output_ids[dev_idx][:, 0])

# Generate the rest of the tokens
for i in tqdm(range(config.ntok - 1), desc="Generating tokens"):

    # ⭐️ Launch the PyVM ⭐️
    pyvm_generate(input_seq_len + i)

    # Pick top-1 token
    for dev_idx in range(config.num_devices):
        output_ids[dev_idx][:, i + 1] = torch.argmax(logits[dev_idx], dim=-1)
        hidden_states[dev_idx][dev_idx*batch_per_device:(dev_idx+1)*batch_per_device, :] = embeddings[dev_idx](output_ids[dev_idx][:, i + 1])


########################################################
# Print results
########################################################


# Gather the output ids
output_ids = torch.cat([output_ids[dev_idx].to('cpu') for dev_idx in range(config.num_devices)], dim=0)

# We use identical sequences, so just print the first one
for batch_idx in range(config.batch_size):
    assert torch.equal(output_ids[0, :], output_ids[batch_idx, :])
print("Llama output:", tokenizer.decode(output_ids[0]))

