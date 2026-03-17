import matplotlib.pyplot as plt
import numpy as np

n = [1, 2, 3, 4, 5, 6, 7, 8]

ollama_ttft = [137.72, 3246.46, 11050.95, 16212.32, 21059.40, 30552.80, 35780.98, 40989.36]
vllm_metal_ttft  = [879.57, 1327.52, 2385.84, 3175.04, 3999.88, 4801.04, 4801.04, 4801.04]
vllm_ttft   = [42.50, 522.09, 1947.21, 1997.62, 1963.31, 1965.42, 1962.97, 3359.16]
omlx_ttft = [184.42, 682.04, 2159.28, 2206.85, 4129.19, 5094.90, 6075.03, 6879.88]

ollama_itl     = [35.15, 50.85, 71.84, 86.17, 96.18, 120.43, 135.87, 147.58]
vllm_metal_itl = [12.05, 34.80, 52.23, 60.33, 74.10,  95.46, 0, 0]
vllm_itl       = [8.27, 11.73, 17.18, 23.12, 25.75, 40.68, 44.84, 47.34]
omlx_itl       = [7.93, 11.27, 15.68, 19.94, 24.57, 38.08, 40.00, 40.97]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("vllm-rs vs World — Apple Silicon (Llama3.2 3B Q4)", fontsize=13, fontweight='bold')

x = np.arange(len(n)) * 1
w = 0.2
OLLAMA = '#1f78b4'
METAL  = '#33a02c'
VLLM   = '#e31a1c'
OMLX   = '#ff7f00'

# --- TTFT ---
b1 = ax1.bar(x - 1.5*w, ollama_ttft,      w, label='Ollama',     color=OLLAMA)
b2 = ax1.bar(x - 0.5*w, vllm_metal_ttft,  w, label='vllm-metal', color=METAL)
b3 = ax1.bar(x + 0.5*w, omlx_ttft,        w, label='OMLX',       color=OMLX)
b4 = ax1.bar(x + 1.5*w, vllm_ttft,        w, label='vllm-rs',    color=VLLM)
ax1.set_title('TTFT (ms)  ↓ lower is better')
ax1.set_xlabel('Concurrent requests')
ax1.set_ylabel('Milliseconds (log scale)')
#ax1.set_yscale('log')
ax1.set_xticks(x)
ax1.set_xticklabels([f'n={i}' for i in n])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for bar, color in [(b, OLLAMA) for b in b1] + [(b, METAL) for b in b2] + [(b, VLLM) for b in b3] + [(b, OMLX) for b in b4]:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
             f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=6.5, color=color)

# --- ITL ---
b5 = ax2.bar(x - 1.5*w, ollama_itl,     w, label='Ollama',     color=OLLAMA)
b6 = ax2.bar(x - 0.5*w, vllm_metal_itl, w, label='vllm-metal', color=METAL)
b7 = ax2.bar(x + 0.5*w, omlx_itl,       w, label='OMLX',       color=OMLX)
b8 = ax2.bar(x + 1.5*w, vllm_itl,       w, label='vllm-rs',    color=VLLM)
ax2.set_title('ITL (ms)  ↓ lower is better')
ax2.set_xlabel('Concurrent requests')
ax2.set_ylabel('Milliseconds')
ax2.set_xticks(x)
ax2.set_xticklabels([f'n={i}' for i in n])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, color in [(b, OLLAMA) for b in b5] + [(b, METAL) for b in b6] + [(b, VLLM) for b in b7] + [(b, OMLX) for b in b8]:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=6.5, color=color)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/tmp/bench_plot.png', dpi=150, bbox_inches='tight')
print("saved to /tmp/bench_plot.png")