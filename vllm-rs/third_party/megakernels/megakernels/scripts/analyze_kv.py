import torch

NUM_LAYERS=2

# Load saved k_cache and v_cache tensors from GPU 0
k_cache = torch.load("/data/bfs/l1_k_cache_gpu0.pt")[1:].reshape((NUM_LAYERS,-1,128,128)).to(0).to(float)[:,:,:8,:]
print(f"Loaded k_cache with shape: {k_cache.shape}")
v_cache = torch.load("/data/bfs/l1_v_cache_gpu0.pt")[1:].reshape((NUM_LAYERS,-1,128,128)).to(0).to(float)[:,:,:8,:]

k_cache = k_cache[:,:,:8,:]
v_cache = v_cache[:,:,:8,:]

print(f"Loaded k_cache with shape: {k_cache.shape}")
print(f"Loaded v_cache with shape: {v_cache.shape}")

k_diffs = (k_cache - k_cache[:,0:1])
v_diffs = (v_cache - v_cache[:,0:1])

# Compute relative differences
k_diffs_rel = 100 * 2*k_diffs.abs() / (k_cache.abs() + k_cache[:,0:1].abs() + 1e-16).abs()
v_diffs_rel = 100 * 2*v_diffs.abs() / (v_cache.abs() + v_cache[:,0:1].abs() + 1e-16).abs()

print(f"k_diffs: {k_diffs.shape}")
print(f"v_diffs: {v_diffs.shape}")

from tabulate import tabulate

# Create a table for K cache relative differences
# Shape: k_diffs_rel is (NUM_LAYERS, num_pages, 8, 128)
# We want 768 rows (num_pages) and NUM_LAYERS columns

num_pages = k_diffs_rel.shape[1]
k_table_data = []

# Create header with layer numbers
headers = ["Page"] + [f"Layer {i}" for i in range(NUM_LAYERS)]

# For each page, compute mean relative difference across heads and head_dim
for page_idx in range(num_pages):
    row = [page_idx]
    for layer_idx in range(NUM_LAYERS):
        # Mean across heads (dim 2) and head_dim (dim 3)
        mean_rel_diff = k_diffs_rel[layer_idx, page_idx].mean().cpu().item()
        row.append(f"{mean_rel_diff:.2f}")
    k_table_data.append(row)

print("K Cache Relative Differences (%) - Mean across heads and head dimensions:")
print(tabulate(k_table_data, headers=headers, tablefmt="grid", floatfmt=".2f"))



# print("  ------------ Mean relative differences per layer, up to 128: ------------")
# print(f" -- k_diffs: {[round(k_diffs_rel[i][:128].abs().mean().cpu().item(), 2) for i in range(NUM_LAYERS)]}")
# print(f" -- v_diffs: {[round(v_diffs_rel[i][:128].abs().mean().cpu().item(), 2) for i in range(NUM_LAYERS)]}")

# print("  ------------ Mean relative differences per layer, up to 512: ------------")
# print(f" -- k_diffs: {[round(k_diffs_rel[i][:512].abs().mean().cpu().item(), 2) for i in range(NUM_LAYERS)]}")
# print(f" -- v_diffs: {[round(v_diffs_rel[i][:512].abs().mean().cpu().item(), 2) for i in range(NUM_LAYERS)]}")

# print("  ------------ Mean relative differences per layer, after 512: ------------")
# print(f" -- k_diffs: {[round(k_diffs_rel[i][512:].abs().mean().cpu().item(), 2) for i in range(NUM_LAYERS)]}")
# print(f" -- v_diffs: {[round(v_diffs_rel[i][512:].abs().mean().cpu().item(), 2) for i in range(NUM_LAYERS)]}")

# print("  ------------ Mean relative differences per layer: ------------")
# print(f" -- k_diffs: {[round(k_diffs_rel[i].abs().mean().cpu().item(), 2) for i in range(NUM_LAYERS)]}")
# print(f" -- v_diffs: {[round(v_diffs_rel[i].abs().mean().cpu().item(), 2) for i in range(NUM_LAYERS)]}")





breakpoint()