import sys
import logging
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.fftpack import fft, ifft, fftfreq

# --- PATH SETUP ---
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.bimamba import CoherentBiMamba

# --- CONFIGURATION ---
# Use Agg backend for headless
import matplotlib
matplotlib.use('Agg')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = project_root / "experiments/logs/bimamba_curriculum_pn/best_model.pt"
OUTPUT_DIR = project_root / "reports" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Physics: The "Standard" Case where we saw the floor
BAUD_RATE = 32e9
TEST_CD = 1200.0         # 1200 ps/nm
TEST_LW = 500e3          # 500 kHz
TEST_SNR = 20.0          # High SNR (to isolate tracking errors)
N_SYMBOLS = 4096

# --- REUSED PHYSICS (Must match exactly) ---
def apply_channel(signal, cd, lw, snr, rng):
    # 1. CD
    n = len(signal)
    freq = fftfreq(n, d=1/BAUD_RATE)
    const = (np.pi * (1550e-9)**2 * 100e3) / 299792458 # Based on 100km ref
    # Note: Using your training generator logic directly:
    # beta2 approx logic matching your generator's implementation
    lambda_0 = 1550e-9
    c = 3e8
    beta2_total = -(cd * 1e-12 * (lambda_0**2)) / (2 * np.pi * c)
    omega = 2 * np.pi * freq
    tf = np.exp(1j * 0.5 * beta2_total * (omega**2))
    sig = ifft(fft(signal) * tf)

    # 2. Phase Noise
    if lw > 0:
        ts = 1.0 / BAUD_RATE
        var = 2 * np.pi * lw * ts
        steps = rng.normal(0, np.sqrt(var), size=len(signal))
        sig = sig * np.exp(1j * np.cumsum(steps))

    # 3. AWGN
    pwr = np.mean(np.abs(sig)**2)
    noise_pwr = pwr / (10**(snr/10.0))
    noise = np.sqrt(noise_pwr/2) * (rng.standard_normal(len(sig)) + 1j * rng.standard_normal(len(sig)))
    sig = sig + noise
    return sig

def get_data():
    rng = np.random.default_rng(42)
    # Generate QPSK
    ints = rng.integers(0, 4, size=N_SYMBOLS)
    phase = (2 * ints + 1) * np.pi / 4
    tx = np.exp(1j * phase)
    
    # Apply Channel
    rx = apply_channel(tx, TEST_CD, TEST_LW, TEST_SNR, rng)
    
    # Normalize (Phase Preserving)
    rx_real = rx.real - rx.real.mean()
    rx_imag = rx.imag - rx.imag.mean()
    pwr = np.mean(rx_real**2 + rx_imag**2)
    rx = (rx_real + 1j*rx_imag) / (np.sqrt(pwr) + 1e-8)
    
    return tx, rx, ints

def plot_diagnosis(tx, rx, pred_complex, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Common settings
    bins = 100
    r = [[-2.5, 2.5], [-2.5, 2.5]]
    
    # 1. Input (Channel Output)
    axes[0].hist2d(rx.real, rx.imag, bins=bins, range=r, cmap="inferno", norm=LogNorm())
    axes[0].set_title(f"Model Input\n(CD={TEST_CD}, LW={TEST_LW/1e3}k)")
    
    # 2. Model Output (Equalized)
    axes[1].hist2d(pred_complex.real, pred_complex.imag, bins=bins, range=r, cmap="inferno", norm=LogNorm())
    axes[1].set_title("BiMamba Output\n(Equalized)")
    
    # 3. Ideal Reference (Ground Truth)
    # Just plot the 4 points for reference
    axes[2].scatter(tx.real, tx.imag, c='cyan', s=50, edgecolors='black', label="Tx Sent")
    # Overlay model output predictions (small dots)
    axes[2].scatter(pred_complex.real, pred_complex.imag, c='blue', s=1, alpha=0.1, label="Rx Recd")
    axes[2].set_title("Overlay vs. Ideal")
    axes[2].legend(loc="upper right")
    
    for ax in axes:
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)

    save_path = OUTPUT_DIR / f"diagnosis_constellation{title_suffix}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved diagnosis to: {save_path}")

def main():
    print("--- Running Visual Diagnostics ---")
    
    # Load Model
    model = CoherentBiMamba(num_classes=4, in_channels=2, d_model=64, num_layers=4).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # Get Data
    tx, rx, labels = get_data()
    
    # Prepare Tensor
    rx_tensor = torch.tensor(np.stack([rx.real, rx.imag], axis=0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logits = model(rx_tensor) # [1, 4, L]
        
        # We want the raw "soft" symbols before argmax to see the constellation.
        # However, this model outputs logits (4 classes). 
        # We can't easily invert the logits back to IQ unless we change the architecture to output Regression.
        # BUT, we can visualize the "Confidence" by plotting the Logits PCA or similar?
        # NO, simpler: The "Output" of a classification model is the probability distribution.
        
        # ACTUALLY: For constellation plots, usually models output [B, 2, L] (Regression).
        # Since yours is Classification, we can't plot the "Analog" output directly.
        # We can only plot the *Input* (Rx) and see if we can infer what happened,
        # OR we can modify the architecture to give us the internal state.
        
        # ALTERNATIVE: Plot the Input, but color-coded by whether the model got it RIGHT or WRONG.
        pass

    preds = logits.argmax(dim=1).squeeze().cpu().numpy()
    correct = (preds == labels)
    
    # Plot 1: Where are the errors occurring?
    plot_error_map(rx, correct)

def plot_error_map(rx, correct_mask):
    """
    Plots the RECEIVED signal, coloring correct points Blue and errors Red.
    This tells us if errors are at the edges (noise) or rotated (phase).
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot Correct (Blue, small, background)
    ax.scatter(rx.real[correct_mask], rx.imag[correct_mask], 
               c='blue', s=1, alpha=0.1, label="Correct")
    
    # Plot Errors (Red, medium, foreground)
    ax.scatter(rx.real[~correct_mask], rx.imag[~correct_mask], 
               c='red', s=10, marker='x', alpha=0.8, label="Error")
    
    ax.set_title(f"Error Distribution (Standard Case)\nRed = Misclassified")
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = OUTPUT_DIR / "error_distribution_map.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved Error Map to: {save_path}")

if __name__ == "__main__":
    main()