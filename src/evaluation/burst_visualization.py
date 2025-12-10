import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from scipy.fftpack import fft, ifft, fftfreq

# --- PATH SETUP ---
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.bimamba import CoherentBiMamba

# --- CONFIGURATION (Standard Case @ 18dB) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = project_root / "experiments/logs/bimamba_curriculum_pn/best_model.pt"
OUTPUT_DIR = project_root / "reports" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target: The "Standard" case where FEC is working hard
TEST_CD = 1200.0
TEST_LW = 500e3
TEST_SNR = 18.0
BAUD_RATE = 32e9
SYMBOL_MAP = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

# --- REUSED CLASSES ---
class MatrixInterleaver:
    def __init__(self, depth=14): self.depth = depth
    def interleave(self, bits):
        pad = (self.depth - (len(bits) % self.depth)) % self.depth
        bits_p = np.pad(bits, (0, pad))
        mat = bits_p.reshape(-1, self.depth)
        return mat.T.flatten(), len(bits)
    def deinterleave(self, bits, orig_len):
        rows = len(bits) // self.depth
        mat = bits.reshape(self.depth, rows).T
        return mat.flatten()[:orig_len]

G_sys = np.array([[1,0,0,0,1,1,0], [0,1,0,0,1,0,1], [0,0,1,0,0,1,1], [0,0,0,1,1,1,1]], dtype=int)
H_sys = np.array([[1,1,0,1,1,0,0], [1,0,1,1,0,1,0], [0,1,1,1,0,0,1]], dtype=int)

def encode_sys(data_bits):
    return np.dot(data_bits.reshape(-1, 4), G_sys).flatten() % 2

def decode_sys(rx_bits):
    blocks = rx_bits.reshape(-1, 7)
    syndromes = np.dot(blocks, H_sys.T) % 2
    s_int = syndromes.dot(np.array([4, 2, 1]))
    syn_to_col = {6:0, 5:1, 3:2, 7:3, 4:4, 2:5, 1:6}
    corrected = []
    for i, blk in enumerate(blocks):
        s = s_int[i]
        if s != 0 and s in syn_to_col: blk[syn_to_col[s]] ^= 1
        corrected.append(blk[:4])
    return np.concatenate(corrected)

# --- PHYSICS ---
def apply_channel(signal, cd, lw, snr, baud_rate, rng):
    # CD
    if cd != 0:
        freq = fftfreq(len(signal), d=1/baud_rate)
        beta2 = -(cd * 1e-12 * (1550e-9**2)) / (2 * np.pi * 3e8)
        tf = np.exp(1j * 0.5 * beta2 * (2 * np.pi * freq)**2)
        signal = ifft(fft(signal) * tf)
    # Phase Noise
    if lw > 0:
        var = 2 * np.pi * lw * (1/baud_rate)
        phase = np.cumsum(rng.normal(0, np.sqrt(var), size=len(signal)))
        signal = signal * np.exp(1j * phase)
    # AWGN
    pwr = np.mean(np.abs(signal)**2)
    noise_pwr = pwr / (10**(snr/10))
    noise = np.sqrt(noise_pwr/2) * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return signal + noise

def main():
    print(f"--- GENERATING BURST ERROR VISUALIZATION ---")
    
    # 1. Setup
    rng = np.random.default_rng(42) # Fixed seed for reproducibility
    N_BITS = 10000 
    # Use fewer bits for visualization so the "barcode" is readable
    # 10k bits is enough to see a burst but small enough to plot.
    
    # 2. Tx Chain
    tx_bits = rng.integers(0, 2, size=N_BITS - (N_BITS % 4))
    encoded_bits = encode_sys(tx_bits)
    
    interleaver = MatrixInterleaver(depth=42)
    tx_bits_intl, orig_len = interleaver.interleave(encoded_bits)
    
    # Symbol Map
    pairs = tx_bits_intl.reshape(-1, 2)
    indices = pairs[:, 0] * 2 + pairs[:, 1]
    tx_symbols = SYMBOL_MAP[indices]
    
    # 3. Channel
    # Pad for model batching
    rx_signal = apply_channel(tx_symbols, TEST_CD, TEST_LW, TEST_SNR, BAUD_RATE, rng)
    
    # Norm
    rx_signal = rx_signal - np.mean(rx_signal)
    rx_signal = rx_signal / (np.sqrt(np.mean(np.abs(rx_signal)**2)) + 1e-8)
    
    # 4. Neural Receiver
    model = CoherentBiMamba(num_classes=4, in_channels=2, d_model=64, num_layers=4).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    inputs = torch.tensor(np.stack([rx_signal.real, rx_signal.imag], axis=0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(inputs)
        preds = logits.argmax(dim=1).cpu().numpy().flatten()
        
    # 5. Rx Chain
    # Demap
    rx_bits_intl = []
    for s in preds: rx_bits_intl.extend([(s >> 1) & 1, s & 1])
    rx_bits_intl = np.array(rx_bits_intl)
    
    # ERROR VECTOR 1: Raw Channel Errors (Pre-Deinterleave)
    # Compare Rx Interleaved bits vs Tx Interleaved bits
    err_channel = (rx_bits_intl != tx_bits_intl).astype(int)
    
    # Deinterleave
    rx_bits_raw = interleaver.deinterleave(rx_bits_intl, orig_len)
    
    # ERROR VECTOR 2: Scrambled Errors (Post-Deinterleave)
    # Compare Rx Raw bits vs Encoded Tx bits
    err_scrambled = (rx_bits_raw != encoded_bits).astype(int)
    
    # Decode
    rx_data = decode_sys(rx_bits_raw)
    
    # ERROR VECTOR 3: Final Errors (Post-FEC)
    # Compare Final Data vs Original Data
    err_final = (rx_data != tx_bits).astype(int)
    
    print(f"Raw Channel Errors: {np.sum(err_channel)}")
    print(f"Final Errors:       {np.sum(err_final)}")
    
    # 6. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(15, 6), sharex=False) # sharex=False because lengths differ slightly due to coding
    
    # Helper to plot barcodes
    def plot_barcode(ax, data, title, color):
        ax.imshow(data[np.newaxis, :], aspect='auto', cmap='Greys', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.set_xlabel("Bit Index")
        
    # Plot 1: The Burst (Channel)
    plot_barcode(axes[0], err_channel[:2000], f"1. Raw Channel Errors (First 2000 bits) - Bursts Visible", 'black')
    
    # Plot 2: The Scatter (Interleaver)
    plot_barcode(axes[1], err_scrambled[:2000], f"2. De-Interleaved Errors (First 2000 bits) - Errors Scattered", 'black')
    
    # Plot 3: The Cleanup (FEC)
    plot_barcode(axes[2], err_final[:1142], f"3. Post-FEC Errors (First ~1000 bits) - Clean", 'black')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "error_barcode_visualization.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()