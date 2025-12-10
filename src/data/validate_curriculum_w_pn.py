import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')

# Configuration
DATA_DIR = Path("data/curriculum_w_pn")
STAGES = ["stage_1", "stage_2", "stage_3", "stage_4", "stage_5"]

def load_sample(stage_name, idx=0):
    """Loads a single sequence from the generated .npy files."""
    # Construct path
    path = DATA_DIR / f"X_train_{stage_name}.npy"
    
    if not path.exists():
        print(f"❌ ERROR: File not found at: {path.absolute()}")
        return None

    try:
        # Load X (Complex data)
        X = np.load(path, mmap_mode='r') # mmap to avoid loading gigabytes
        
        # Extract one sequence: Shape [2, L] -> [L] complex
        sample = X[idx]
        complex_signal = sample[0] + 1j * sample[1]
        return complex_signal
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None

def analyze_phase_noise(signal):
    """
    Metric: Standard Deviation of the 4th Power Phase.
    For QPSK, signal^4 should collapse the 4 points to 1 point (conceptually).
    The spread of that point is a proxy for phase noise + CD noise.
    """
    # Normalize power
    sig = signal / (np.std(signal) + 1e-9)
    
    # 4th power removes QPSK modulation (mapped to 0 angle conceptually)
    sig_4th = sig ** 4
    
    # Measure the spread of the angles of the 4th power signal
    angles = np.angle(sig_4th)
    return np.std(angles)

def plot_stages():
    print(f"Checking data in: {DATA_DIR.absolute()}")
    
    fig, axes = plt.subplots(1, len(STAGES), figsize=(4 * len(STAGES), 4))
    if len(STAGES) == 1: axes = [axes]

    print(f"{'Stage':<10} | {'4th Pwr Phase Std':<25}")
    print("-" * 40)

    for i, stage in enumerate(STAGES):
        sig = load_sample(stage, idx=0) 
        
        if sig is None:
            # If data missing, just turn off the axis
            axes[i].axis('off')
            axes[i].set_title(f"{stage}\nMISSING")
            continue
            
        # 1. Calculate Metric
        pn_metric = analyze_phase_noise(sig)
        print(f"{stage:<10} | {pn_metric:.4f}")

        # 2. Plot Constellation
        ax = axes[i]
        # Plot first 2000 points
        ax.scatter(sig.real[:2000], sig.imag[:2000], s=1, alpha=0.5, c='blue')
        
        ax.set_title(f"{stage}\nMetric: {pn_metric:.2f}")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    save_path = Path("curriculum_validation.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Plot saved to: {save_path.absolute()}")

if __name__ == "__main__":
    plot_stages()