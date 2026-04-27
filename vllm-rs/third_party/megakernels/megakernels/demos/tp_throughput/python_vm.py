import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor

from megakernels.demos.tp_throughput.globs import Globals
from megakernels.demos.tp_throughput.instructions import (
    AttentionDecode,
    AttentionPrefill,
    AttnNorm,
    DownProjResidual,
    GateSilu,
    IncBarrier,
    LM_Head,
    LM_Head_Norm,
    MLP_Norm,
    O_ProjResidual,
    QKV_RopeAppend,
    UpMatMul,
)
from megakernels.llama import apply_rotary_pos_emb_interleaved
from megakernels.python_vm import get_start_end, rms_norm
from megakernels.utils import assert_div


def matmul(
    matA: Tensor,
    matB: Tensor,
):
    out = einsum(matA, matB, "a i, b i -> a b")
    return out


def matmul_with_residual(
    matA: Tensor,
    matB: Tensor,
    residual: Tensor,
):
    matmul_out = matmul(matA, matB)

    residual += matmul_out


def pre_attn_layer_norm(
    globals: Globals,
    instruction: AttnNorm,
    device_idx: int,
):
    layer_idx = instruction.layer_idx

    # Process each batch index in the batch
    for local_batch_idx in instruction.local_batch_indices:
        local_batch_block_idx = local_batch_idx // globals.matmul_batch_block_size

        if layer_idx > 0:
            assert (
                globals.barriers[device_idx][
                    layer_idx - 1,
                    instruction.prev_opcode() - 1,
                    local_batch_block_idx,
                    0,
                ]
                == globals.num_output_blocks()
            )

        # Apply RMS norm using proper device indexing
        if local_batch_idx >= globals.hidden_states[device_idx].shape[0]:
            print(f"Invalid local_batch_idx in AttnNorm:")
            print(f"  Instruction: {instruction}")
            print(f"  local_batch_idx: {local_batch_idx}")
            print(f"  hidden_states[{device_idx}].shape: {globals.hidden_states[device_idx].shape}")
            print(f"  expected local_batch_size for device {device_idx}: {globals.device_batch_size[device_idx]}")
            print(f"  global_batch_size: {globals.global_batch_size}")
            print(f"  all local_batch_indices in instruction: {instruction.local_batch_indices}")
            print(f"  device_idx: {device_idx}")
            breakpoint()
        
        pre_attn_ln = rms_norm(
            inp=globals.hidden_states[device_idx][local_batch_idx],
            weight=globals.attn_norm_weights[device_idx][layer_idx],
            eps=globals.rms_norm_eps,
        )

        # Broadcast to all devices
        # Convert local batch index to global batch index
        from megakernels.demos.tp_throughput.globs import local_to_global_batch_idx
        # Update globals with current device as tp_rank for this calculation
        old_tp_rank = globals.tp_rank
        globals.tp_rank = device_idx
        global_batch_idx = local_to_global_batch_idx(globals, local_batch_idx)
        globals.tp_rank = old_tp_rank
        global_batch_block_idx = global_batch_idx // globals.matmul_batch_block_size
        for peer_dev_idx in range(globals.tp_size):
            globals.post_attn_norm[peer_dev_idx][global_batch_idx] = pre_attn_ln.to(
                peer_dev_idx
            )
            # barrier update for each device
            globals.barriers[peer_dev_idx][
                layer_idx, instruction.opcode() - 1, global_batch_block_idx, 0
            ] += 1


def qkv_matmul_rope_append(
    globals: Globals,
    instruction: QKV_RopeAppend,
    device_idx: int,
):
    layer_idx = instruction.layer_idx
    tp_size = globals.tp_size

    batch_start_row = (
        instruction.global_batch_block_idx * globals.matmul_batch_block_size
    )
    batch_end_row = batch_start_row + globals.matmul_batch_block_size

    output_block_idx = instruction.local_output_block_idx
    output_start_col = output_block_idx * globals.matmul_output_block_size
    output_end_col = output_start_col + globals.matmul_output_block_size

    # Check barrier matches CUDA expectation: matmul_batch_block_size on index 0
    barrier_value = globals.barriers[device_idx][
        instruction.layer_idx,
        instruction.prev_opcode() - 1,
        instruction.global_batch_block_idx,
        0,
    ]
    expected_value = globals.matmul_batch_block_size
    if barrier_value != expected_value:
        print(f"QKV Barrier assertion failed:")
        print(f"  Layer: {instruction.layer_idx}")
        print(f"  Barrier index: {instruction.prev_opcode() - 1}")
        print(f"  Global batch block: {instruction.global_batch_block_idx}")
        print(f"  Barrier value: {barrier_value}")
        print(f"  Expected value: {expected_value}")
        print(f"  Device: {device_idx}")
        print(f"  Full barrier tensor shape: {globals.barriers[device_idx].shape}")
    assert barrier_value == expected_value

    # Activations are already broadcast by the norm operation
    matmul_output = einsum(
        globals.qkv_proj_weights[device_idx][
            layer_idx, output_start_col:output_end_col
        ],
        globals.post_attn_norm[device_idx][batch_start_row:batch_end_row],
        "o i, b i -> b o",
    )

    k_start = (globals.num_attention_heads // tp_size) * globals.head_dim
    v_start = k_start + (globals.num_kv_heads // tp_size) * globals.head_dim

    start, end = get_start_end(globals.matmul_output_block_size, output_block_idx)

    # Process output in chunks of head_dim (one head at a time)
    num_generated_heads = assert_div(globals.matmul_output_block_size, globals.head_dim)
    assert num_generated_heads == 2, "Expected 2 heads per instruction"

    # Rearrange matmul output to separate heads
    output_reshaped = rearrange(
        matmul_output, "b (h d) -> b h d", h=num_generated_heads, d=globals.head_dim
    )

    # Process each head
    for head_idx in range(num_generated_heads):
        head_output = output_reshaped[:, head_idx, :]  # Shape: (batch, head_dim)
        current_col_start = start + head_idx * globals.head_dim

        # Determine what type of head this is
        if current_col_start < k_start:
            head_type = "q"
        elif current_col_start < v_start:
            head_type = "k"
        else:
            head_type = "v"

        # Apply RoPE to Q and K heads only
        if head_type in "qk":
            # Apply RoPE for each batch element using position_ids

            pos_ids = globals.position_ids[device_idx][batch_start_row:batch_end_row]
            orig_head_output = head_output
            with_rope_batch, _ = apply_rotary_pos_emb_interleaved(
                q=orig_head_output.unsqueeze(-2),
                k=orig_head_output.unsqueeze(-2),
                cos=globals.rope_cos[device_idx][pos_ids],
                sin=globals.rope_sin[device_idx][pos_ids],
                unsqueeze_dim=-2,
            )
            head_output = with_rope_batch.squeeze(-2)

            # with_rope_list = []
            # for batch_idx in range(batch_start_row, batch_end_row):
            #     pos_id = globals.position_ids[device_idx][batch_idx]
            #     # Reshape for RoPE application
            #     head_single = head_output[
            #         batch_idx - batch_start_row : batch_idx - batch_start_row + 1
            #     ]
            #     head_single = head_single.unsqueeze(1)  # Add head dimension

            #     with_rope_single, _ = apply_rotary_pos_emb_interleaved(
            #         q=head_single,
            #         k=head_single,
            #         cos=globals.rope_cos[device_idx][pos_id],
            #         sin=globals.rope_sin[device_idx][pos_id],
            #         unsqueeze_dim=-2,
            #     )
            #     with_rope_list.append(
            #         with_rope_single.squeeze(1)
            #     )  # Remove head dimension

            # head_output = torch.cat(with_rope_list, dim=0)

        # Store the output based on head type
        if head_type == "q":
            globals.post_rope_q[device_idx][
                batch_start_row:batch_end_row,
                current_col_start : current_col_start + globals.head_dim,
            ] = head_output

        elif head_type == "k":
            k_head_idx = assert_div(current_col_start - k_start, globals.head_dim)

            append_indices = globals.kv_append_indices[device_idx][
                batch_start_row:batch_end_row
            ]

            flat_cache = rearrange(
                globals.k_cache[device_idx],
                "num_layers_and_pages page_size num_kv_heads head_dim -> (num_layers_and_pages page_size) num_kv_heads head_dim",
            )

            flat_cache[
                layer_idx * globals.num_pages * globals.page_size + append_indices,
                k_head_idx,
            ] = head_output.to(dtype=flat_cache.dtype)

        else:  # type == "v"
            v_head_idx = assert_div(current_col_start - v_start, globals.head_dim)

            flat_cache = rearrange(
                globals.v_cache[device_idx],
                "num_layers_and_pages page_size num_kv_heads head_dim -> (num_layers_and_pages page_size) num_kv_heads head_dim",
            )

            flat_cache[
                layer_idx * globals.num_pages * globals.page_size + append_indices,
                v_head_idx,
            ] = head_output.to(dtype=flat_cache.dtype)

    # Barrier update - increment by 32 each time (Q heads get 4x32=128, KV heads get 1x32=32)
    bar_idx = 0 if head_type == "q" else 1
    globals.barriers[device_idx][
        instruction.layer_idx,
        instruction.opcode() - 1,
        instruction.global_batch_block_idx,
        bar_idx,
    ] += 32


def attention_prefill(
    globals: Globals,
    instruction: AttentionPrefill,
    device_idx: int,
):
    """
    Attention prefill computation for variable-length sequences using paged KV cache.
    
    This function processes a block of queries (16 queries) for a specific sequence,
    computing attention against the full KV history for that sequence.
    """
    layer_idx = instruction.layer_idx
    
    # Extract instruction parameters from data field
    # Format: [seq_idx, block_idx, kv_head_idx]
    seq_idx = instruction.prefill_seq_idx
    block_idx = instruction.prefill_block_idx
    kv_head_idx = instruction.kv_head_idx
    
    num_local_attention_heads = globals.num_attention_heads // globals.tp_size
    num_local_kv_heads = globals.num_kv_heads // globals.tp_size
    gqa_ratio = num_local_attention_heads // num_local_kv_heads
    
    # Get query range for this block
    q_start_idx = globals.prefill_qo_indptr[device_idx][seq_idx].item()
    q_end_idx = globals.prefill_qo_indptr[device_idx][seq_idx + 1].item()
    q_row_start = 16 * block_idx + q_start_idx
    q_row_end = min(q_end_idx, q_row_start + 16)
    
    # Check barriers - wait for QKV to be computed
    batch_block_idx = q_row_start // globals.matmul_batch_block_size
    batch_block_idx_last = (q_row_end - 1) // globals.matmul_batch_block_size
    
    # Check QKV completion barriers
    assert globals.barriers[device_idx][layer_idx, instruction.prev_opcode() - 1, batch_block_idx, 0] == 128
    if batch_block_idx_last != batch_block_idx:
        assert globals.barriers[device_idx][layer_idx, instruction.prev_opcode() - 1, batch_block_idx_last, 0] == 128
    
    # Also check KV cache barriers - wait for current KV cache to be generated
    for check_row in range(q_start_idx, q_end_idx, globals.matmul_batch_block_size):
        check_batch_block = check_row // globals.matmul_batch_block_size
        assert globals.barriers[device_idx][layer_idx, instruction.prev_opcode() - 1, check_batch_block, 1] == 32
    
    # Get Q heads for this KV head (using GQA)
    q_head_start = kv_head_idx * gqa_ratio
    q_head_end = q_head_start + gqa_ratio
    start_q_dim = q_head_start * globals.head_dim
    end_q_dim = q_head_end * globals.head_dim
    
    # Load Q vectors for this block
    q = globals.post_rope_q[device_idx][q_row_start:q_row_end, start_q_dim:end_q_dim]
    actual_q_len = q_row_end - q_row_start
    q = rearrange(q, "q_len (h d) -> q_len h d", h=gqa_ratio, d=globals.head_dim)
    
    # Get KV cache data for this sequence
    kv_indptr_start = globals.prefill_kv_indptr[device_idx][seq_idx].item()
    kv_indptr_end = globals.prefill_kv_indptr[device_idx][seq_idx + 1].item()
    last_page_len = globals.prefill_kv_last_page_len[device_idx][seq_idx].item()
    
    kv_indices = globals.prefill_kv_indices[device_idx][kv_indptr_start:kv_indptr_end]
    
    # Load K and V from paged cache
    full_k = globals.k_cache[device_idx][
        layer_idx * globals.num_pages + kv_indices, :, kv_head_idx
    ].view(-1, globals.head_dim)
    
    full_v = globals.v_cache[device_idx][
        layer_idx * globals.num_pages + kv_indices, :, kv_head_idx  
    ].view(-1, globals.head_dim)
    
    # Handle partial last page
    total_seq_len = (len(kv_indices) - 1) * globals.page_size + last_page_len
    cutoff = globals.page_size - last_page_len
    if cutoff > 0:
        k = full_k[:-cutoff]
        v = full_v[:-cutoff]
    else:
        k = full_k
        v = full_v
    
    # Vectorized attention computation
    if actual_q_len > 0:
        # Compute Q @ K^T for all queries at once
        # q shape: (actual_q_len, gqa_ratio, head_dim)
        # k shape: (seq_len, head_dim)  
        qk = einsum(q.float(), k.float(), "q h d, s d -> q h s")  # (actual_q_len, gqa_ratio, seq_len)
        scaled_qk = qk * globals.attn_scale
        
        # Apply causal mask - each query can only attend to positions up to its own position
        # Following the CUDA implementation pattern
        seq_len = k.shape[0]
        remaining_sequence_length = total_seq_len - (q_end_idx - 1 - (q_row_start + actual_q_len - 1))
        
        # Create causal mask vectorized
        # Each query at position q_idx can attend to: remaining_sequence_length + q_idx - (actual_q_len - 1) positions
        q_indices = torch.arange(actual_q_len, device=scaled_qk.device)  # [0, 1, 2, ..., actual_q_len-1]
        s_indices = torch.arange(seq_len, device=scaled_qk.device)  # [0, 1, 2, ..., seq_len-1]
        
        # For each query, compute how many positions it can attend to
        valid_positions_per_query = remaining_sequence_length + q_indices - (actual_q_len - 1)  # (actual_q_len,)
        
        # Create mask: s_indices[None, :] >= valid_positions_per_query[:, None] 
        # This gives True where we should mask (invalid positions)
        causal_mask = s_indices[None, :] >= valid_positions_per_query[:, None]  # (actual_q_len, seq_len)
        
        # Expand to include head dimension and apply
        causal_mask = causal_mask[:, None, :].expand(-1, gqa_ratio, -1)  # (actual_q_len, gqa_ratio, seq_len)
        scaled_qk = scaled_qk.masked_fill(causal_mask, float('-inf'))
        
        # Compute softmax with causal mask applied
        softmax_weights = torch.softmax(scaled_qk, dim=-1).to(dtype=qk.dtype)
        
        # Compute attention output: softmax @ V
        # softmax_weights: (actual_q_len, gqa_ratio, seq_len)
        # v: (seq_len, head_dim)
        stacked_output = einsum(softmax_weights.float(), v.float(), "q h s, s d -> q h d")
        
        # Distributed transpose: route to the device that owns these tokens
        batch_size_per_device = globals.global_batch_size // globals.tp_size
        heads_per_device = globals.num_attention_heads // globals.tp_size
        col_start = device_idx * heads_per_device + q_head_start
        col_end = device_idx * heads_per_device + q_head_end
        
        for q_idx in range(actual_q_len):
            abs_token_idx = q_row_start + q_idx
            target_device_idx = abs_token_idx // batch_size_per_device  
            local_token_idx = abs_token_idx % batch_size_per_device
            
            # Store in target device's attn_out buffer
            attn_out_reshaped = rearrange(
                globals.attn_out[target_device_idx][local_token_idx],
                "(h d) -> h d", 
                d=globals.head_dim
            )
            attn_out_reshaped[col_start:col_end] = stacked_output[q_idx]
            
            # Update barrier on target device
            local_batch_block_idx = local_token_idx // globals.matmul_batch_block_size
            globals.barriers[target_device_idx][
                layer_idx, instruction.opcode() - 1, local_batch_block_idx, 0
            ] += 1


def attention_decode(
    globals: Globals,
    instruction: AttentionDecode,
    device_idx: int,
):
    # unpack info
    layer_idx = instruction.layer_idx

    num_local_attention_heads = globals.num_attention_heads // globals.tp_size
    num_local_kv_heads = globals.num_kv_heads // globals.tp_size
    gqa_ratio = num_local_attention_heads // num_local_kv_heads

    # Process each (batch_idx, kv_head_idx) pair in the instruction
    # data format: [batch_idx0, kv_head_idx0, batch_idx1, kv_head_idx1, ...]
    num_pairs = len(instruction.data) // 2

    for pair_idx in range(num_pairs):
        batch_idx = instruction.data[2 * pair_idx]
        kv_head_idx = instruction.data[2 * pair_idx + 1]

        batch_block_idx = batch_idx // globals.matmul_batch_block_size
        q_head_start = kv_head_idx * gqa_ratio
        q_head_end = q_head_start + gqa_ratio

        # barrier check - match CUDA expectations: 128 for Q heads, 32 for KV heads
        bars = globals.barriers[device_idx][
            layer_idx, instruction.prev_opcode() - 2, batch_block_idx
        ]
        # Check Q head barriers (expect 128)
        for i in range(gqa_ratio):
            assert bars[0] == 128
        # Check KV head barriers (expect 32) 
        assert bars[1] == 32

        # get Q for this batch and kv_head's corresponding Q heads
        start_q_dim = q_head_start * globals.head_dim
        end_q_dim = q_head_end * globals.head_dim
        q = globals.post_rope_q[device_idx][batch_idx, start_q_dim:end_q_dim]
        q = rearrange(q, "(h d) -> h d", h=gqa_ratio, d=globals.head_dim)

        # Get paged KV cache indices for this sequence
        indptr_start = globals.decode_kv_indptr[device_idx][batch_idx]
        indptr_end = globals.decode_kv_indptr[device_idx][batch_idx + 1]
        last_page_len = globals.decode_kv_last_page_len[device_idx][batch_idx].item()

        kv_indices = globals.decode_kv_indices[device_idx][indptr_start:indptr_end]
        full_k = globals.k_cache[device_idx][
            layer_idx * globals.num_pages + kv_indices, :, kv_head_idx
        ].view(-1, globals.head_dim)

        full_v = globals.v_cache[device_idx][
            layer_idx * globals.num_pages + kv_indices, :, kv_head_idx
        ].view(-1, globals.head_dim)

        cutoff = globals.page_size - last_page_len

        if cutoff > 0:
            k = full_k[:-cutoff]
            v = full_v[:-cutoff]
        else:
            k = full_k
            v = full_v

        # Standard attention computation
        # q shape: (gqa_ratio, head_dim)
        # k shape: (seq_len, head_dim)
        qk = einsum(q.float(), k.float(), "qheads d, seqlen d -> qheads seqlen")
        scaled_qk = qk * globals.attn_scale
        softmax = torch.softmax(scaled_qk, dim=-1).to(dtype=qk.dtype)
        attn_output = einsum(
            softmax.float(), v.float(), "qheads seqlen, seqlen d -> qheads d"
        )

        # Distributed transpose: route attention output to the GPU that owns this token
        batch_size_per_device = globals.global_batch_size // globals.tp_size
        target_device_idx = batch_idx // batch_size_per_device
        local_batch_idx = batch_idx % batch_size_per_device

        # Calculate which attention head columns this source device contributes
        heads_per_device = globals.num_attention_heads // globals.tp_size
        col_start = device_idx * heads_per_device + q_head_start
        col_end = device_idx * heads_per_device + q_head_end

        # Store result in the target device's attn_out
        # Each device receives all attention heads for its subset of tokens
        all_out = rearrange(
            globals.attn_out[target_device_idx][local_batch_idx],
            "(h d) -> h d",
            d=globals.head_dim,
        )
        all_out[col_start:col_end] = attn_output

        # Update barrier on the target device
        local_batch_block_idx = local_batch_idx // globals.matmul_batch_block_size
        globals.barriers[target_device_idx][
            layer_idx, instruction.opcode() - 1, local_batch_block_idx, 0
        ] += 1


def o_proj_residual(
    globals: Globals,
    instruction: O_ProjResidual,
    device_idx: int,
):
    layer_idx = instruction.layer_idx

    # Get batch indices based on instruction fields
    batch_start_row = (
        instruction.global_batch_block_idx * globals.matmul_batch_block_size
    )
    batch_end_row = batch_start_row + globals.matmul_batch_block_size

    output_start_col = (
        instruction.global_output_block_idx * globals.matmul_output_block_size
    )
    output_end_col = output_start_col + globals.matmul_output_block_size

    # Check barrier - should have received attention outputs from all devices
    # Each device contributes num_kv_heads attention outputs per batch
    expected_barrier_count = globals.tp_size * globals.matmul_batch_block_size
    if (
        globals.barriers[device_idx][
            layer_idx,
            instruction.prev_opcode() - 1,
            instruction.local_batch_block_idx,
            0,
        ]
        == expected_barrier_count
    ):
        breakpoint()
        raise Exception("Barrier check failed")

    # O-projection matmul (data parallel - full matrix on each device)
    # attn_out has the full attention output vector for this device's tokens
    matA = globals.attn_out[device_idx][
        batch_start_row:batch_end_row
    ]  # [batch_block, hidden_dim]
    matB = globals.o_proj_weights[device_idx][
        layer_idx, output_start_col:output_end_col
    ]  # [hidden_dim, output_block]

    output = matmul(matA, matB)

    # Add to residual stream
    globals.hidden_states[device_idx][
        batch_start_row:batch_end_row, output_start_col:output_end_col
    ] += output

    # Update barrier
    globals.barriers[device_idx][
        layer_idx, instruction.opcode() - 1, instruction.local_batch_block_idx, 0
    ] += 1


def pre_mlp_layer_norm(
    globals: Globals,
    instruction: MLP_Norm,
    device_idx: int,
):
    layer_idx = instruction.layer_idx

    # Process each batch index in the batch
    for local_batch_idx in instruction.local_batch_indices:
        batch_block_idx = local_batch_idx // globals.matmul_batch_block_size

        assert (
            globals.barriers[device_idx][
                layer_idx, instruction.prev_opcode() - 1, batch_block_idx, 0
            ]
            == globals.num_output_blocks()
        )

        # Apply RMS norm using proper device indexing
        post_mlp_ln = rms_norm(
            inp=globals.hidden_states[device_idx][local_batch_idx],
            weight=globals.mlp_norm_weights[device_idx][layer_idx],
            eps=globals.rms_norm_eps,
        )

        # Broadcast to all devices

        # Broadcast to all devices
        # Convert local batch index to global batch index
        from megakernels.demos.tp_throughput.globs import local_to_global_batch_idx
        # Update globals with current device as tp_rank for this calculation
        old_tp_rank = globals.tp_rank
        globals.tp_rank = device_idx
        global_batch_idx = local_to_global_batch_idx(globals, local_batch_idx)
        globals.tp_rank = old_tp_rank
        global_batch_block_idx = global_batch_idx // globals.matmul_batch_block_size
        for peer_dev_idx in range(globals.tp_size):
            globals.post_mlp_norm[peer_dev_idx][global_batch_idx] = post_mlp_ln.to(
                peer_dev_idx
            )
            # barrier update for each device
            globals.barriers[peer_dev_idx][
                layer_idx, instruction.opcode() - 1, global_batch_block_idx, 0
            ] += 1


def gate_silu(
    globals: Globals,
    instruction: GateSilu,
    device_idx: int,
):
    layer_idx = instruction.layer_idx

    batch_start_row = (
        instruction.global_batch_block_idx * globals.matmul_batch_block_size
    )
    batch_end_row = batch_start_row + globals.matmul_batch_block_size

    output_start_col = (
        instruction.global_output_block_idx * globals.matmul_output_block_size
    )
    output_end_col = output_start_col + globals.matmul_output_block_size

    assert (
        globals.barriers[device_idx][
            layer_idx,
            instruction.prev_opcode() - 1,
            instruction.global_batch_block_idx,
            0,
        ]
        == globals.matmul_batch_block_size
    )

    # Tensor-parallel matmul - activations are already broadcast by the norm
    # Each GPU computes a slice of the output columns
    matA = globals.post_mlp_norm[device_idx][
        batch_start_row:batch_end_row
    ]  # [batch_block, hidden_dim]
    matB = globals.gate_proj_weights[device_idx][
        layer_idx, output_start_col:output_end_col
    ]  # [intermediate_dim_slice, hidden_dim]

    # Apply SiLU activation: silu(x) = x * sigmoid(x)
    out = matmul(matA, matB)  # [batch_tile, output_tile]
    out = F.silu(out)

    # Save the output
    globals.mlp_intermediates[device_idx][
        batch_start_row:batch_end_row, output_start_col:output_end_col
    ] = out

    # Update barrier
    globals.barriers[device_idx][
        layer_idx,
        instruction.opcode() - 1,
        instruction.local_batch_block_idx,
        instruction.local_output_block_idx,
    ] += 1


def up_matmul(
    globals: Globals,
    instruction: UpMatMul,
    device_idx: int,
):
    layer_idx = instruction.layer_idx

    batch_start_row = (
        instruction.global_batch_block_idx * globals.matmul_batch_block_size
    )
    batch_end_row = batch_start_row + globals.matmul_batch_block_size

    output_start_col = (
        instruction.global_output_block_idx * globals.matmul_output_block_size
    )
    output_end_col = output_start_col + globals.matmul_output_block_size

    assert (
        globals.barriers[device_idx][
            layer_idx,
            instruction.prev_opcode() - 1,
            instruction.local_batch_block_idx,
            instruction.local_output_block_idx,
        ]
        == 1
    )

    # Tensor-parallel matmul - same as gate_silu
    matA = globals.post_mlp_norm[device_idx][
        batch_start_row:batch_end_row
    ]  # [batch_block, hidden_dim]
    matB = globals.up_proj_weights[device_idx][
        layer_idx, output_start_col:output_end_col
    ]  # [intermediate_dim_slice, hidden_dim]

    matmul_out = matmul(matA, matB)  # [batch_tile, output_tile]

    # Element-wise multiplication with the gate output from previous instruction
    gated = (
        matmul_out
        * globals.mlp_intermediates[device_idx][
            batch_start_row:batch_end_row, output_start_col:output_end_col
        ]
    )

    # Store result back in silu_out (reusing the buffer)
    globals.mlp_intermediates[device_idx][
        batch_start_row:batch_end_row, output_start_col:output_end_col
    ] = gated

    # Update barrier
    globals.barriers[device_idx][
        layer_idx, instruction.opcode() - 1, instruction.local_batch_block_idx, 0
    ] += 1


def down_proj_residual(
    globals: Globals,
    instruction: DownProjResidual,
    device_idx: int,
):
    layer_idx = instruction.layer_idx

    batch_start_row = (
        instruction.global_batch_block_idx * globals.matmul_batch_block_size
    )
    batch_end_row = batch_start_row + globals.matmul_batch_block_size

    output_start_col = (
        instruction.global_output_block_idx * globals.matmul_output_block_size
    )
    output_end_col = output_start_col + globals.matmul_output_block_size

    # Check barrier - should have all intermediate blocks from up_matmul
    assert (
        globals.barriers[device_idx][
            layer_idx,
            instruction.prev_opcode() - 1,
            instruction.local_batch_block_idx,
            0,
        ]
        == globals.num_intermediate_blocks() // globals.tp_size
    )

    # Tensor-parallel matmul
    # Input (silu_out) is tensor-parallel: each GPU has a slice of intermediate dim
    # Weight is also sliced along input dimension (intermediate dim)
    matA = globals.mlp_intermediates[device_idx][
        batch_start_row:batch_end_row
    ]  # [batch_block, intermediate_dim_slice]
    matB = globals.down_proj_weights[device_idx][
        layer_idx, output_start_col:output_end_col
    ]  # [hidden_dim, intermediate_dim_slice]

    # This produces a partial result that needs to be summed across GPUs
    partial_output = matmul(matA, matB)  # [batch_block, output_block]

    # Reduce-scatter: each GPU gets a slice of the batch and sums contributions from all GPUs
    # Determine which slice of the batch this device should receive
    batch_size_per_device = globals.global_batch_size // globals.tp_size

    # Each device contributes its partial result to all devices
    for target_dev_idx in range(globals.tp_size):
        # Calculate which batch rows the target device owns
        target_batch_start = target_dev_idx * batch_size_per_device
        target_batch_end = target_batch_start + batch_size_per_device

        # Find the intersection with our current batch block
        intersect_start = max(batch_start_row, target_batch_start)
        intersect_end = min(batch_end_row, target_batch_end)

        if intersect_start < intersect_end:
            # We have rows that belong to the target device
            local_start = intersect_start - batch_start_row
            local_end = intersect_end - batch_start_row
            target_local_start = intersect_start - target_batch_start
            target_local_end = intersect_end - target_batch_start

            # Add our contribution to the target device's hidden states
            globals.hidden_states[target_dev_idx][
                target_local_start:target_local_end, output_start_col:output_end_col
            ] += partial_output[local_start:local_end].to(target_dev_idx)

    # Update barriers on all devices
    for target_dev_idx in range(globals.tp_size):
        # Calculate local batch block index for the target device
        target_batch_block_idx = instruction.global_batch_block_idx % (
            batch_size_per_device // globals.matmul_batch_block_size
        )
        globals.barriers[target_dev_idx][
            layer_idx, instruction.opcode() - 1, target_batch_block_idx, 0
        ] += 1


def pre_lm_head_rms(
    globals: Globals,
    instruction: LM_Head_Norm,
    device_idx: int,
):
    # LM head norm happens after all layers, so we use num_hidden_layers - 1
    layer_idx = globals.num_hidden_layers - 1

    # Process each batch index in the batch
    for batch_idx in instruction.local_batch_indices:
        batch_block_idx = batch_idx // globals.matmul_batch_block_size

        assert (
            globals.barriers[device_idx][
                layer_idx,
                instruction.prev_opcode() - 1,
                batch_block_idx,
                0,
            ]
            == globals.num_output_blocks()
        )

        # Apply RMS norm - purely data parallel, no broadcasting
        post_ln = rms_norm(
            inp=globals.hidden_states[device_idx][batch_idx],
            weight=globals.lm_head_norm_weights[device_idx],
            eps=globals.rms_norm_eps,
        )

        # Store locally only - no broadcast to other devices
        globals.post_lm_head_norm[device_idx][batch_idx] = post_ln

        # Update local barrier only
        globals.barriers[device_idx][
            layer_idx,
            instruction.opcode() - 1,
            batch_block_idx,
            0,
        ] += 1


def lm_head(
    globals: Globals,
    instruction: LM_Head,
    device_idx: int,
):
    layer_idx = globals.num_hidden_layers - 1

    batch_start_row = (
        instruction.global_batch_block_idx * globals.matmul_batch_block_size
    )
    batch_end_row = batch_start_row + globals.matmul_batch_block_size

    output_start_col = (
        instruction.global_output_block_idx * globals.matmul_output_block_size
    )
    output_end_col = output_start_col + globals.matmul_output_block_size

    assert (
        globals.barriers[device_idx][
            layer_idx,
            instruction.prev_opcode() - 1,
            instruction.local_batch_block_idx,
            0,
        ]
        == globals.matmul_batch_block_size
    )

    # Simple data-parallel matmul - no all-gather needed
    # Each GPU has its own tokens and computes logits for those tokens
    matA = globals.post_lm_head_norm[device_idx][
        batch_start_row:batch_end_row
    ]  # [batch_block, hidden_dim]
    matB = globals.lm_head_weights[device_idx][
        output_start_col:output_end_col, :
    ]  # [vocab_block, hidden_dim]

    matmul_output = matmul(matA, matB)  # [batch_block, vocab_block]

    # Store logits
    globals.logits[device_idx][
        batch_start_row:batch_end_row, output_start_col:output_end_col
    ] = matmul_output


def global_batch_idx_to_local_idx_info(globals: Globals, global_batch_idx: int):
    """Convert global batch index to (device_idx, local_batch_idx) tuple."""
    global_block_idx = global_batch_idx // globals.matmul_batch_block_size
    pos_in_block = global_batch_idx % globals.matmul_batch_block_size
    
    device_idx = global_block_idx % globals.tp_size
    local_block_idx = global_block_idx // globals.tp_size
    local_batch_idx = local_block_idx * globals.matmul_batch_block_size + pos_in_block
    
    return (device_idx, local_batch_idx)


def inc_barrier(
    globals: Globals,
    instruction: IncBarrier,
    device_idx: int,
):
    """
    Increment barriers to handle batch size padding.
    
    This instruction increments barriers for various operations when the batch size
    is not evenly divisible by the matmul block size, ensuring proper synchronization
    for the extra padding tokens.
    """
    batch_size = globals.global_batch_size
    batch_size_wave_remainder = batch_size % (globals.tp_size * globals.matmul_batch_block_size)
    batch_size_gpu_remainder = batch_size_wave_remainder % globals.matmul_batch_block_size
    batch_size_ceil = ((batch_size + globals.matmul_batch_block_size - 1) // globals.matmul_batch_block_size) * globals.matmul_batch_block_size
    extra_batch_size = batch_size_ceil - batch_size
    
    if extra_batch_size > 0:
        global_batch_row = batch_size // globals.matmul_batch_block_size
        
        # Debug print matching the CUDA version
        print(f"GPU {device_idx} incrementing norm barriers by {extra_batch_size} and gqa barriers by {extra_batch_size*8}")
        
        # Increment barriers for all layers for AttnNorm (opcode 1) and MlpNorm (opcode 6)
        num_layers_in_barriers = globals.barriers[device_idx].shape[0]
        for layer_idx in range(num_layers_in_barriers):
            # AttnNorm barriers (OPCODE_AttnNorm-1 = 0)
            globals.barriers[device_idx][layer_idx, 0, global_batch_row, 0] += extra_batch_size
            # MlpNorm barriers (OPCODE_MlpNorm-1 = 5) 
            globals.barriers[device_idx][layer_idx, 5, global_batch_row, 0] += extra_batch_size
        
        # Handle attention decode and LM head norm barriers for the device that owns these tokens
        local_block_info = global_batch_idx_to_local_idx_info(globals, batch_size)
        
        if local_block_info[0] == device_idx:  # This device owns the last tokens
            local_batch_block_idx = local_block_info[1] // globals.matmul_batch_block_size
            
            for layer_idx in range(num_layers_in_barriers):
                # AttentionDecode barriers (OPCODE_GQA_AttentionDecode-1 = 3) 
                # increment by extra_batch_size * 8 (8 attention pairs per instruction)
                globals.barriers[device_idx][layer_idx, 3, local_batch_block_idx, 0] += extra_batch_size * 8
            
            # LM_HeadNorm barrier (OPCODE_LM_HeadNorm-1 = 9) for layer 0 only
            globals.barriers[device_idx][0, 9, local_batch_block_idx, 0] += extra_batch_size


INSTRUCTION_TO_SOLVER = {
    AttnNorm: pre_attn_layer_norm,
    QKV_RopeAppend: qkv_matmul_rope_append,
    AttentionPrefill: attention_prefill,
    AttentionDecode: attention_decode,
    O_ProjResidual: o_proj_residual,
    MLP_Norm: pre_mlp_layer_norm,
    GateSilu: gate_silu,
    UpMatMul: up_matmul,
    DownProjResidual: down_proj_residual,
    LM_Head_Norm: pre_lm_head_rms,
    LM_Head: lm_head,
    IncBarrier: inc_barrier,
}
