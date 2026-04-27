import workstealing
import torch

instructions = []
for i in range(2):
    instructions.append([1, i] + [0] * 30)
# for i in range(132):
#     instructions.append([-1, i] + [0] * 30)
# for i in range(10000):
#     instructions.append([1, i] + [0] * 30)
instructions = torch.tensor(instructions, device=0, dtype=torch.int32)
timings = torch.zeros(instructions.shape[0], 128, device=0, dtype=torch.int32)
index = torch.zeros(1, device=0, dtype=torch.int32)

print(instructions.shape)
print(timings.shape)
print(index)

# breakpoint()

workstealing.workstealing(instructions, timings, index)

torch.cuda.synchronize()

# breakpoint()