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
# We will run SNR sweeps for 3 different "Channel Conditions"
EXPERIMENTS = {
    "Benign (Short Reach)":  {"cd": 400.0,  "lw": 100e3},  # Easy
    "Standard (Target)":     {"cd": 1200.0, "lw": 500e3},  # Medium (Your training center)
    "Stress (Extreme)":      {"cd": 1800.0, "lw": 1e6}     # Hard (Outside training CD range)
}

# The SNR values to test for each experiment
SNR_LEVELS = [12, 13, 14, 15, 16, 17, 18, 19, 20] 

# Bits per point (Higher = smoother curves but slower)
N_BITS = 50_000

# --- PHYSICS ENGINE (REUSED) ---
# MUST MATCH run_fec_loop_w_pn.py EXACTLY

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
    """Runs a single simulation point."""
    rng = np.random.default_rng(42)
    
    # 1. Generate Bits
    n_bits = N_BITS - (N_BITS % 2)
    tx_bits = rng.integers(0, 2, size=n_bits)
    tx_symbols = bits_to_symbols(tx_bits)
    
    # Pad
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
    
    # 4. Measure BER (Pre-FEC only for speed/plotting)
    rx_bits = symbols_to_bits(rx_indices_valid)
    bit_errors = np.sum(rx_bits != tx_bits)
    ber = bit_errors / len(tx_bits)
    
    return ber

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logging.info(f"--- STARTING PERFORMANCE SWEEP ---")
    
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
            ber = evaluate_point(model, cd, lw, snr)
            
            logging.info(f"  > SNR {snr}dB: BER = {ber:.2e}")
            results.append({
                "Condition": condition_name,
                "SNR_dB": snr,
                "BER": ber,
                "CD_ps_nm": cd,
                "Linewidth_Hz": lw
            })

    # Save CSV
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "ber_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved results to {csv_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Use log scale for Y-axis (Standard for BER)
    sns.lineplot(data=df, x="SNR_dB", y="BER", hue="Condition", marker="o", style="Condition")
    
    plt.yscale("log")
    plt.ylim(1e-5, 1.0) # Standard BER window
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.axhline(3.8e-3, color='r', linestyle='--', label="FEC Threshold (HD-FEC)")
    
    plt.title("BER vs. SNR (Coherent BiMamba Performance)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.xlabel("SNR (dB)")
    
    plot_path = OUTPUT_DIR / "ber_waterfall_plot.png"
    plt.savefig(plot_path, dpi=300)
    logging.info(f"Saved plot to {plot_path}")
    print("\n✅ Sweep Complete!")

if __name__ == "__main__":
    main()