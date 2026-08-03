import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Setup Figure with generous dimensions
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6), dpi=120)
fig.patch.set_facecolor('#1E1E1E')

# Adjust margins and spacing between subplots to eliminate overlaps
plt.subplots_adjust(wspace=0.35, top=0.82, bottom=0.20, left=0.05, right=0.95)

# Sample Attributes
state_attrs = ["LNWID", "RT_WIDTH", "AADT_TOTAL", "DESG_SPD"]
target_attrs = ["LANE_WIDTH", "LANE_WIDTH", "DAILY_TRAFFIC", "DESIGN_SPEED"]
N = len(state_attrs)

# Total animation frames
frames = N * N + 4

def update(frame):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#252526')
        ax.axis('off')

    # =========================================================================
    # PANEL 1: LEXICAL MATCHING
    # =========================================================================
    ax1.text(0.5, 1.06, "1. Lexical (String Overlap)", transform=ax1.transAxes,
             color='#F44747', fontsize=12, fontweight='bold', ha='center', va='bottom')
    ax1.text(0.5, 1.01, "Fails on Synonyms (52.4% Accuracy)", transform=ax1.transAxes,
             color='#CCCCCC', fontsize=9.5, ha='center', va='bottom')
    
    grid1 = np.zeros((N, N))
    grid1[0,0], grid1[1,1], grid1[2,2], grid1[3,3] = 0.1, 0.0, 0.4, 0.2
    ax1.imshow(grid1, cmap='Reds', vmin=0, vmax=1)
    
    ax1.text(0.5, -0.12, "Character Spelling Only!\n'LNWID' vs 'LANE_WIDTH' = 0% Overlap", 
             transform=ax1.transAxes, color='white', fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#333333', edgecolor='#F44747', linewidth=1))

    # =========================================================================
    # PANEL 2: STANDARD BERT CROSS-ATTENTION
    # =========================================================================
    curr_cell = min(frame, N * N - 1)
    row, col = curr_cell // N, curr_cell % N
    
    ax2.text(0.5, 1.06, "2. Standard BERT Cross-Attention", transform=ax2.transAxes,
             color='#DCDCAA', fontsize=12, fontweight='bold', ha='center', va='bottom')
    ax2.text(0.5, 1.01, f"O(N²) Bottleneck: Pass {curr_cell + 1}/{N*N}", transform=ax2.transAxes,
             color='#CCCCCC', fontsize=9.5, ha='center', va='bottom')
    
    grid2 = np.zeros((N, N))
    for r in range(N):
        for c in range(N):
            if r * N + c <= curr_cell:
                grid2[r, c] = 0.7
    grid2[row, col] = 1.0  # Highlight active pass
    
    ax2.imshow(grid2, cmap='YlOrRd', vmin=0, vmax=1)
    
    ax2.text(0.5, -0.12, f"Evaluating Pair ({row+1}, {col+1}): Full Forward Pass\nRequires {N*N} Expensive Neural Operations", 
             transform=ax2.transAxes, color='white', fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#333333', edgecolor='#DCDCAA', linewidth=1))

    # =========================================================================
    # PANEL 3: SBERT SIAMESE NETWORK (OUR APPROACH)
    # =========================================================================
    ax3.text(0.5, 1.06, "3. SBERT Siamese Vectors (Ours)", transform=ax3.transAxes,
             color='#4EC9B0', fontsize=12, fontweight='bold', ha='center', va='bottom')
    ax3.text(0.5, 1.01, "O(N) Encoding + Matrix Dot Product", transform=ax3.transAxes,
             color='#CCCCCC', fontsize=9.5, ha='center', va='bottom')
    
    cos_sim_matrix = np.array([
        [0.887, 0.887, 0.120, 0.050],
        [0.915, 0.915, 0.110, 0.040],
        [0.050, 0.040, 0.931, 0.080],
        [0.080, 0.070, 0.090, 0.764]
    ])
    
    if frame < 2:
        # Vector encoding phase
        blank_grid = np.zeros((N, N))
        ax3.imshow(blank_grid, cmap='Greys', vmin=0, vmax=1)
        ax3.text(0.5, 0.5, "Encoding N Prompts -> u ∈ ℝ³⁸⁴\nEncoding N Targets -> v ∈ ℝ³⁸⁴", 
                 transform=ax3.transAxes, color='#4EC9B0', fontsize=10, fontweight='bold', 
                 ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E1E1E'))
    else:
        # Instant matrix dot product
        ax3.imshow(cos_sim_matrix, cmap='GnBu', vmin=0, vmax=1)

    ax3.text(0.5, -0.12, "INSTANT MATRIX DOT PRODUCT!\nS = (u · v) / (||u|| ||v||)  |  1.76 ms / vector", 
             transform=ax3.transAxes, color='white', fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#007ACC', edgecolor='#4EC9B0', linewidth=1))

ani = animation.FuncAnimation(fig, update, frames=frames, interval=450, repeat=True)
# Save as a looping Animated GIF for PowerPoint/Google Slides
ani.save("schema_paradigm_simulation.gif", writer="pillow", fps=2)

# Save the final static snapshot as a high-res PNG image
plt.savefig("schema_paradigm_static.png", facecolor=fig.get_facecolor(), bbox_inches="tight")

print("\n[SUCCESS] Exported 'schema_paradigm_simulation.gif' and 'schema_paradigm_static.png'!")
plt.show()