print('hello -1')
import torch
print('hello 0')
import mla_decode

print('hello 1')

batch = 1
heads = 16
new_seq = 4
num_instructions = 2

L = 512

instructions = torch.zeros((num_instructions,32), dtype=torch.int32, device='cuda')

Qv = torch.ones((batch, new_seq, heads, 512), dtype=torch.bfloat16, device='cuda')
Qrot = torch.ones((batch, new_seq, heads, 64), dtype=torch.bfloat16, device='cuda')

V = torch.randn((256*batch, 256, 512), dtype=torch.bfloat16, device='cuda')
Krot = torch.ones((256*batch, 256, 64), dtype=torch.bfloat16, device='cuda')

Table = torch.arange(256*batch, dtype=torch.int32, device='cuda').reshape((batch,256))

O = torch.zeros((batch, new_seq, heads, 512), dtype=torch.bfloat16, device='cuda')
O_scratch = torch.zeros((1, new_seq, heads, 512), dtype=torch.float32, device='cuda')
Lvec_scratch = torch.zeros((1, 1, new_seq, heads), dtype=torch.float32, device='cuda')

completion_flag = torch.zeros((1, new_seq), dtype=torch.int32, device='cuda')

# New required tensors for megakernel API
global_instruction_index = torch.zeros(1, 1, 1, 1, dtype=torch.int32, device='cuda')

Softmax_scale = (1/576)**.5
tic = 1

timings = torch.zeros((num_instructions, 128), dtype=torch.int32, device='cuda')

print('hello 2')

instructions[0,:9] = torch.tensor([
  1, # Opcode
  0, # Uid
  0, # dst.batch_idx
  0, # dst.seq_idx
  0, # q_batch_idx
  0, # q_seq_idx
  0, # start_pos
  L, # end_pos
  L, # length
], dtype=torch.int32, device='cuda')

def compute_ref(Qv, Qrot, Krot, V, L):
    q_cat = torch.cat((Qv, Qrot), dim=-1)
    k_cat = torch.cat((V, Krot), dim=-1).reshape((-1,576))[:L]
    # print('k_cat sum 0', k_cat[:,:64].sum(dim=-1))
    # print('k_cat sum ALL', k_cat.sum(dim=-1))
    logits = torch.einsum('bnhd,ld->bnhl', q_cat, k_cat).to(torch.float32)
    # print('LOGITS', logits)
    logits *= Softmax_scale
    # probs = torch.nn.functional.softmax(logits * Softmax_scale, dim=-1)
    logits -= logits.max(dim=-1, keepdim=True)[0]
    logits = torch.exp(logits)
    # print(logits.to(torch.bfloat16))
    probs = logits / logits.sum(dim=-1, keepdim=True)
    # print(probs)
    v   = V.reshape((-1,512))[:L]
    out = torch.einsum('bnhl,ld->bnhd', probs.to(torch.bfloat16), v)
    return out

print('instructions', instructions)

# New API: (instructions, timings, global_instruction_index, Qrot, Qv, Krot, V, Table, O, O_scratch, Lvec_scratch, completion_flag, Softmax_scale, tic)
mla_decode.mla_decode(instructions, timings, global_instruction_index, Qrot, Qv, Krot, V, Table, O, O_scratch, Lvec_scratch, completion_flag, Softmax_scale, tic)
torch.cuda.synchronize()
ref = compute_ref(Qv, Qrot, Krot, V, L)

print(f'O mean: {O.abs().mean()}, ref mean: {ref.abs().mean()}')
print(f'O max: {O.abs().max()}, ref max: {ref.abs().max()}')

print('avg diff:', (O-ref).abs().mean())


# save_gantt_chart(timings, instructions, save_all=True, name='single')

breakpoint()