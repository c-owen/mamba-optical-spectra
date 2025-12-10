import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import torch
from scipy.fftpack import fft, ifft, fftfreq

# --- PATH SETUP ---
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.bimamba import CoherentBiMamba

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = project_root / "experiments/logs/bimamba_curriculum_pn/best_model.pt"
OUTPUT_DIR = project_root / "reports" / "sweeps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Physics Constants
BAUD_RATE = 32e9
LAMBDA_NM = 1550.0
C_MPS = 299792458

# QPSK MAPPING
SYMBOL_MAP = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])

# --- EXPERIMENT DEFINITION ---
EXPERIMENTS = {
    "Benign (Short Reach)":  {"cd": 400.0,  "lw": 100e3},  
    "Standard (Target)":     {"cd": 1200.0, "lw": 500e3},  
    "Stress (Extreme)":      {"cd": 1800.0, "lw": 1e6}     
}

# The SNR values to test
SNR_LEVELS = [12, 13, 14, 15, 16, 17, 18, 19, 20] 

# Bits per point
N_BITS = 100_000

# --- HAMMING FEC & INTERLEAVER SETUP ---

G_sys = np.array([
    [1,0,0,0, 1,1,0], [0,1,0,0, 1,0,1],
    [0,0,1,0, 0,1,1], [0,0,0,1, 1,1,1]
], dtype=int)

H_sys = np.array([
    [1,1,0,1, 1,0,0], [1,0,1,1, 0,1,0], [0,1,1,1, 0,0,1]
], dtype=int)

def encode_sys(data_bits):
    blocks = data_bits.reshape(-1, 4)
    return np.dot(blocks, G_sys).flatten() % 2

def decode_sys(rx_bits):
    blocks = rx_bits.reshape(-1, 7)
    syndromes = np.dot(blocks, H_sys.T) % 2
    s_int = syndromes.dot(np.array([4, 2, 1]))
    
    syn_to_col = {6:0, 5:1, 3:2, 7:3, 4:4, 2:5, 1:6}
    corrected = []
    for i, blk in enumerate(blocks):
        s = s_int[i]
        if s != 0 and s in syn_to_col:
            blk[syn_to_col[s]] ^= 1
        corrected.append(blk[:4])
    return np.concatenate(corrected)

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

# --- PHYSICS ENGINE ---

def apply_phase_noise(signal, linewidth, baud_rate, rng):
    if linewidth <= 0: return signal
    ts = 1.0 / baud_rate
    var = 2 * np.pi * linewidth * ts
    steps = rng.normal(0, np.sqrt(var), size=len(signal))
    phase = np.cumsum(steps)
    return signal * np.exp(1j * phase)

def apply_chromatic_dispersion(signal, dispersion_ps_nm, baud_rate):
    if dispersion_ps_nm == 0: return signal
    n_fft = len(signal)
    freq = fftfreq(n_fft, d=1/baud_rate)
    lambda_0 = 1550e-9
    c = 3e8
    beta2_total = -(dispersion_ps_nm * 1e-12 * (lambda_0**2)) / (2 * np.pi * c)
    omega = 2 * np.pi * freq
    transfer_function = np.exp(1j * 0.5 * beta2_total * (omega**2))
    spectrum = fft(signal)
    return ifft(spectrum * transfer_function)

def apply_awgn(signal, snr_db, rng):
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power/2) * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return signal + noise

# --- HELPER FUNCTIONS ---

def bits_to_symbols(bits):
    pairs = bits.reshape(-1, 2)
    indices = pairs[:, 0] * 2 + pairs[:, 1]
    return SYMBOL_MAP[indices]

def symbols_to_bits(indices):
    bits = []
    for s in indices:
        bits.extend([(s >> 1) & 1, s & 1])
    return np.array(bits, dtype=int)

def evaluate_point(model, cd, lw, snr):
    """Runs a single simulation point with FEC + Interleaving."""
    rng = np.random.default_rng(42)
    
    # 1. Generate & Encode
    # Ensure n_bits is multiple of 4 for Hamming(7,4)
    n_bits = N_BITS - (N_BITS % 4)
    tx_bits = rng.integers(0, 2, size=n_bits)
    
    # FEC Encode
    encoded_bits = encode_sys(tx_bits)
    
    # Interleave
    interleaver = MatrixInterleaver(depth=42)
    tx_bits_intl, orig_len = interleaver.interleave(encoded_bits)
    
    # Map to Symbols
    tx_symbols = bits_to_symbols(tx_bits_intl)
    
    # Pad for batching
    SEQ_LEN = 2048 
    n_syms = len(tx_symbols)
    n_batches = int(np.ceil(n_syms / SEQ_LEN))
    pad_len = n_batches * SEQ_LEN - n_syms
    tx_padded = np.pad(tx_symbols, (0, pad_len))
    
    # 2. Apply Channel
    rx_batch_list = []
    for i in range(n_batches):
        sig = tx_padded[i*SEQ_LEN : (i+1)*SEQ_LEN]
        sig = apply_chromatic_dispersion(sig, cd, BAUD_RATE)
        sig = apply_phase_noise(sig, lw, BAUD_RATE, rng)
        sig = apply_awgn(sig, snr, rng)
        
        # Norm
        sig_real = sig.real - np.mean(sig.real)
        sig_imag = sig.imag - np.mean(sig.imag)
        pwr = np.mean(sig_real**2 + sig_imag**2)
        scale = np.sqrt(pwr) + 1e-8
        rx_batch_list.append(np.stack([sig_real/scale, sig_imag/scale], axis=0))

    rx_tensor = torch.tensor(np.array(rx_batch_list), dtype=torch.float32)

    # 3. Model Inference
    rx_indices_all = []
    with torch.no_grad():
        dataset = torch.utils.data.TensorDataset(rx_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64)
        for batch in loader:
            inputs = batch[0].to(DEVICE)
            logits = model(inputs) 
            preds = logits.argmax(dim=1)
            rx_indices_all.append(preds.cpu().numpy().flatten())
            
    rx_indices_all = np.concatenate(rx_indices_all)
    rx_indices_valid = rx_indices_all[:n_syms]
    
    # 4. Decode & Measure
    rx_bits_intl = symbols_to_bits(rx_indices_valid)
    
    # De-Interleave
    rx_bits_raw = interleaver.deinterleave(rx_bits_intl, orig_len)
    
    # Pre-FEC BER (Compare Raw RX vs Encoded TX)
    pre_fec_errors = np.sum(rx_bits_raw != encoded_bits)
    pre_fec_ber = pre_fec_errors / len(encoded_bits)
    
    # FEC Decode
    rx_data = decode_sys(rx_bits_raw)
    
    # Post-FEC BER (Compare Decoded RX vs Original TX)
    post_fec_errors = np.sum(rx_data != tx_bits)
    post_fec_ber = post_fec_errors / len(tx_bits)
    
    return pre_fec_ber, post_fec_ber

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.info(f"--- STARTING PERFORMANCE SWEEP (With FEC/Interleaving) ---")
    
    # Load Model
    model = CoherentBiMamba(num_classes=4, in_channels=2, d_model=64, num_layers=4).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    results = []

    # Loop Experiments
    for condition_name, params in EXPERIMENTS.items():
        cd = params['cd']
        lw = params['lw']
        logging.info(f"Running Condition: {condition_name} (CD={cd}, LW={lw/1e3}k)")
        
        for snr in SNR_LEVELS:
            pre_ber, post_ber = evaluate_point(model, cd, lw, snr)
            
            logging.info(f"  > SNR {snr}dB: Pre-BER={pre_ber:.2e} | Post-BER={post_ber:.2e}")
            results.append({
                "Condition": condition_name,
                "SNR_dB": snr,
                "Pre_FEC_BER": pre_ber,
                "Post_FEC_BER": post_ber,
                "CD_ps_nm": cd,
                "Linewidth_Hz": lw
            })

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "ber_sweep_results_fec.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved results to {csv_path}")

    # Plot (Focusing on Pre-FEC Waterfall)
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=df, x="SNR_dB", y="Pre_FEC_BER", hue="Condition", marker="o", style="Condition")
    
    plt.yscale("log")
    plt.ylim(1e-5, 1.0) 
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.axhline(3.8e-3, color='r', linestyle='--', label="FEC Threshold (HD-FEC)")
    
    plt.title("Pre-FEC BER vs. SNR (Interleaved + Hamming Ready)")
    plt.ylabel("Pre-FEC Bit Error Rate")
    plt.xlabel("SNR (dB)")
    plt.legend()
    
    plot_path = OUTPUT_DIR / "ber_waterfall_plot_fec.png"
    plt.savefig(plot_path, dpi=300)
    logging.info(f"Saved plot to {plot_path}")
    print("\n✅ Sweep Complete!")

if __name__ == "__main__":
    main()