import matplotlib.pyplot as plt
import numpy as np

n = [1, 2, 3, 4, 5, 6]

ollama_ttft = [125.00, 184.72, 249.91, 16009.90, 22921.87, 28543.17]
vllm_metal_ttft  = [792.80, 1229.66, 2385.84, 3175.04, 3999.88, 4801.04]
vllm_ttft   = [42.00,  66.81,  108.38, 1763.49, 959.51, 899.15]

ollama_itl     = [10.40, 47.25, 83.90, 68.90, 94.08, 109.87]
vllm_metal_itl = [11.27, 33.80, 52.23, 60.33, 74.10,  95.46]
vllm_itl       = [8.33,  11.56, 15.79, 21.48, 26.03,  39.70]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("vllm-rs vs Ollama — Apple Silicon (Llama3.2 3B Q4)", fontsize=13, fontweight='bold')

x = np.arange(len(n))
w = 0.25
OLLAMA = '#e07b54'
METAL  = '#6dbf6d'
VLLM   = '#5b8dd9'

# --- TTFT ---
b1 = ax1.bar(x - w, ollama_ttft,      w, label='Ollama',        color=OLLAMA)
b2 = ax1.bar(x,     vllm_metal_ttft,  w, label='vllm-rs Metal', color=METAL)
b3 = ax1.bar(x + w, vllm_ttft,        w, label='vllm-rs',       color=VLLM)
ax1.set_title('TTFT (ms)  ↓ lower is better')
ax1.set_xlabel('Concurrent requests')
ax1.set_ylabel('Milliseconds (log scale)')
ax1.set_yscale('log')
ax1.set_xticks(x)
ax1.set_xticklabels([f'n={i}' for i in n])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for bar, color in [(b, OLLAMA) for b in b1] + [(b, METAL) for b in b2] + [(b, VLLM) for b in b3]:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
             f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=6.5, color=color)

# --- ITL ---
b4 = ax2.bar(x - w, ollama_itl,    w, label='Ollama',        color=OLLAMA)
b5 = ax2.bar(x,     vllm_metal_itl, w, label='vllm-rs Metal', color=METAL)
b6 = ax2.bar(x + w, vllm_itl,      w, label='vllm-rs',       color=VLLM)
ax2.set_title('ITL (ms)  ↓ lower is better')
ax2.set_xlabel('Concurrent requests')
ax2.set_ylabel('Milliseconds')
ax2.set_xticks(x)
ax2.set_xticklabels([f'n={i}' for i in n])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, color in [(b, OLLAMA) for b in b4] + [(b, METAL) for b in b5] + [(b, VLLM) for b in b6]:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
             f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=6.5, color=color)

plt.tight_layout()
plt.savefig('/tmp/bench_plot.png', dpi=150, bbox_inches='tight')
print("saved to /tmp/bench_plot.png")
