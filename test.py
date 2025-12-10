import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_interleaver_concept():
    # Parameters 
    rows, cols = 8, 8
    total_bits = rows * cols
    
    # 1. Create a clean bitstream (all zeros)
    data = np.zeros(total_bits)
    
    # 2. Inject "Burst Error" (e.g., Cycle Slip)
    burst_start = 26
    burst_len = 8
    data[burst_start : burst_start + burst_len] = 1
    
    # 3. Reshape into Matrix (Row-by-Row Write)
    matrix_view = data.reshape(rows, cols)
    
    # 4. Transpose (Column-by-Column Read) -> The Interleaver
    interleaved_view = matrix_view.T
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    cmap = ListedColormap(['#e0e0e0', '#d62728']) # Grey=Clean, Red=Error
    
    # Plot A: The Physical Burst (Input)
    # FIXED: Removed 'edgecolor' argument
    axes[0].imshow(matrix_view, cmap=cmap, aspect='equal') 
    axes[0].set_title(f"1. Physical Burst Error\n(Contiguous Block of {burst_len})", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Write Direction $\\rightarrow$")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Add grid lines manually
    for i in range(rows):
        for j in range(cols):
            rect = plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, color='white', linewidth=1)
            axes[0].add_patch(rect)

    # Plot B: The Interleaver Action (Arrow)
    axes[1].axis('off')
    axes[1].text(0.5, 0.5, "Matrix Interleave\n(Read Columns)", ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1].arrow(0.2, 0.4, 0.6, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')

    # Plot C: The Logical Errors (Output)
    # FIXED: Removed 'edgecolor' argument
    axes[2].imshow(interleaved_view, cmap=cmap, aspect='equal')
    axes[2].set_title("2. Logical Error Pattern\n(Scattered Single Bit Errors)", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Read Direction $\\downarrow$")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    # Add grid lines for output
    for i in range(rows):
        for j in range(cols):
            rect = plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, color='white', linewidth=1)
            axes[2].add_patch(rect)

    plt.tight_layout()
    plt.savefig('interleaver_concept.png', dpi=300)
    print("Saved interleaver_concept.png")
    plt.show()

if __name__ == "__main__":
    plot_interleaver_concept()