import json
import matplotlib.pyplot as plt
import os

# Define file paths
rank_files = {
    4096: '/root/autodl-fs/MSPLoRA/rank_alloc/lora_rank_42_alpaca_4096.json'
}

# Colors for each task
colors = {
    4096: 'purple'
}

# Function to parse JSON and extract ranks
def parse_ranks(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    ranks = {}
    for key, rank in data.items():
        parts = key.split('.')
        layer = int(parts[2])
        proj = parts[4]  # q_proj or v_proj
        if layer not in ranks:
            ranks[layer] = {}
        ranks[layer][proj[0]] = rank  # 'q' or 'v'
    return ranks

# Read all data
all_ranks = {}
for budget, file_path in rank_files.items():
    all_ranks[budget] = parse_ranks(file_path)

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted to be flatter for inline insertion

# Offset for bars to avoid overlap
offsets = {4096: 0}
height = 0.8

for budget, ranks in all_ranks.items():
    layers = sorted(ranks.keys())
    q_ranks = [ranks[layer]['q'] for layer in layers]
    v_ranks = [ranks[layer]['v'] for layer in layers]
    
    y_q = [layer + offsets[budget] for layer in layers]
    y_v = [layer + offsets[budget] for layer in layers]
    
    # Plot q ranks to the left
    ax.barh(y_q, q_ranks, left=[-q for q in q_ranks], height=height, color=colors[budget], alpha=0.7, label=f'q_proj {budget}', edgecolor='black', linewidth=0.5)
    # Plot v ranks to the right
    ax.barh(y_v, v_ranks, left=[0]*len(v_ranks), height=height, color=colors[budget], alpha=0.7, label=f'v_proj {budget}', hatch='//', edgecolor='black', linewidth=0.5)

# Add vertical line at rank 0
ax.axvline(0, color='black', linewidth=2, label='Rank 0 Axis')

# Set labels and title with larger fonts
ax.set_xlabel('Rank', fontsize=16)
ax.set_ylabel('Layer', fontsize=16)
ax.set_title('Rank Allocation for Budget 4096', fontsize=18)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, axis='x')

# Increase tick label sizes
ax.tick_params(axis='both', labelsize=14)

# Set xlim to symmetric
max_rank = max(ranks[layer]['q'] for layer in ranks for ranks in all_ranks.values()) + 10
ax.set_xlim(-max_rank, max_rank)

# Set custom xticks and labels to avoid negative numbers, with more ticks for clarity
ax.set_xticks([-max_rank, -max_rank//2, 0, max_rank//2, max_rank])
ax.set_xticklabels([str(max_rank), str(max_rank//2), '0', str(max_rank//2), str(max_rank)])

plt.tight_layout()
output_path = '/root/autodl-fs/MSPLoRA/fig/rank_plot_4096.png'
plt.savefig(output_path, dpi=300)  # Higher DPI for clarity
print(f'Plot saved to {output_path}')