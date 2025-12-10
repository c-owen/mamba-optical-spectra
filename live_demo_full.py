import sys
import logging
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

# --- 1. SETUP & PATHS ---
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.bimamba import CoherentBiMamba

# --- 2. CONFIGURATION ---
BAUD_RATE = 32e9
LAMBDA_NM = 1550.0
FIBER_LENGTH_KM = 100.0
C_MPS = 299792458
SEQ_LEN = 1024 

# System Parameters
CD_VAL = 15.0
PN_RATE = 1e-5
SNR_DB = 23.0

SYMBOL_MAP = np.array([
    1 + 1j,   # 0
   -1 + 1j,   # 1
   -1 - 1j,   # 2
    1 - 1j    # 3
])

# --- 3. LOGGING & UTILS ---
class DemoLogger:
    """Custom logger for the live demo feel"""
    def header(self, text):
        print(f"\n\033[95m{'='*60}\033[0m")
        print(f"\033[1m   {text}\033[0m")
        print(f"\033[95m{'='*60}\033[0m")
        time.sleep(0.5)

    def info(self, text):
        timestamp = time.strftime("%H:%M:%S")
        print(f"\033[94m[{timestamp} INFO]\033[0m {text}")
        time.sleep(0.1) # Slight delay for readability during demo

    def success(self, text):
        timestamp = time.strftime("%H:%M:%S")
        print(f"\033[92m[{timestamp} OK]\033[0m   {text}")

    def warning(self, text):
        timestamp = time.strftime("%H:%M:%S")
        print(f"\033[93m[{timestamp} WARN]\033[0m {text}")

log = DemoLogger()

# --- 4. PHYSICS ENGINE ---
def apply_phase_noise(signal, linewidth_rate=1e-5, rng=None):
    if rng is None: rng = np.random.default_rng()
    phase_noise_var = 2 * np.pi * linewidth_rate
    phase_noise_std = np.sqrt(phase_noise_var)
    steps = rng.normal(0, phase_noise_std, size=len(signal))
    return signal * np.exp(1j * np.cumsum(steps))

def apply_chromatic_dispersion_local(signal, cd_value=20.0):
    n = len(signal)
    freqs = fftfreq(n, d=1/BAUD_RATE)
    constant_factor = (np.pi * (LAMBDA_NM * 1e-9)**2 * FIBER_LENGTH_KM * 1e3) / C_MPS
    # CRITICAL FIX: 1e-6 used here
    phase_factor = constant_factor * cd_value * 1e-6 
    transfer_func = np.exp(-1j * phase_factor * (freqs**2))
    return ifft(fft(signal) * transfer_func)

def apply_awgn_local(signal, snr_db=20.0, rng=None):
    if rng is None: rng = np.random.default_rng()
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power/2) * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return signal + noise

# --- 5. FEC CORE ---
G_sys = np.array([[1,0,0,0,1,1,0], [0,1,0,0,1,0,1], [0,0,1,0,0,1,1], [0,0,0,1,1,1,1]], dtype=int)
H_sys = np.array([[1,1,0,1,1,0,0], [1,0,1,1,0,1,0], [0,1,1,1,0,0,1]], dtype=int)

def encode_sys(data_bits):
    blocks = data_bits.reshape(-1, 4)
    return np.dot(blocks, G_sys).flatten() % 2

def decode_sys(rx_bits):
    blocks = rx_bits.reshape(-1, 7)
    syndromes = np.dot(blocks, H_sys.T) % 2
    s_int = syndromes.dot(np.array([4, 2, 1]))
    syn_to_col = {6:0, 5:1, 3:2, 7:3, 4:4, 2:5, 1:6}
    corrected = []
    for i, s in enumerate(s_int):
        blk = blocks[i].copy()
        if s != 0 and s in syn_to_col:
            blk[syn_to_col[s]] ^= 1
        corrected.append(blk[:4])
    return np.concatenate(corrected)

class MatrixInterleaver:
    def __init__(self, depth=42): self.depth = depth
    def interleave(self, bits):
        n = len(bits)
        pad_len = (self.depth - (n % self.depth)) % self.depth
        matrix = np.pad(bits, (0, pad_len)).reshape(-1, self.depth)
        return matrix.T.flatten(), n
    def deinterleave(self, bits, orig_len):
        rows = len(bits) // self.depth
        return bits.reshape(self.depth, rows).T.flatten()[:orig_len]

def bits_to_symbols(bits):
    pairs = bits.reshape(-1, 2)
    indices = pairs[:, 0] * 2 + pairs[:, 1]
    return SYMBOL_MAP[indices], indices

def symbols_to_bits(symbol_indices):
    bits = []
    for s in symbol_indices:
        bits.extend([(s >> 1) & 1, (s >> 0) & 1])
    return np.array(bits, dtype=int)

# --- 6. VISUALIZATION ---
def save_dashboard(tx_sym, rx_impaired, rx_nn_sym, rx_post_fec_sym, filename="demo_dashboard.png"):
    log.info(f"Rendering 4-Panel Analysis Dashboard...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#f4f4f4')
    
    kwargs = {'alpha': 0.05, 's': 1} 
    lim = 2.5
    
    # Titles and Colors
    configs = [
        ("1. Transmitted (Tx)\nIdeal QPSK", tx_sym, 'green', (-lim, lim)),
        ("2. Channel Input\n(CD + Phase Noise)", rx_impaired, 'red', (-4, 4)),
        ("3. BiMamba Recovery\n(Pre-FEC)", rx_nn_sym, 'blue', (-lim, lim)),
        ("4. Post-FEC Output\n(Corrected)", rx_post_fec_sym, 'purple', (-lim, lim))
    ]
    
    for ax, (title, data, color, limits) in zip(axes, configs):
        ax.scatter(data.real, data.imag, c=color, **kwargs)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(limits); ax.set_ylim(limits)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        
    plt.suptitle(f"Coherent Receiver - Live Demo | CD={CD_VAL} | SNR={SNR_DB}dB", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    log.success(f"Dashboard saved to: \033[4m{filename}\033[0m")

# --- 7. MAIN ENGINE ---
def run_verbose_demo():
    log.header("COHERENT RECEIVER DEMO")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Hardware Acceleration: {device}")
    
    # 1. Configuration
    n_data_bits = 50000
    log.info(f"Configuration: 32 GBaud | {FIBER_LENGTH_KM}km Fiber | {n_data_bits} Bits")
    
    # 2. Tx Generation
    log.info("Initializing Transmitter...")
    rng = np.random.default_rng(42)
    tx_bits = rng.integers(0, 2, size=n_data_bits)
    
    log.info("Applying Hamming(7,4) FEC Encoding...")
    encoded_bits = encode_sys(tx_bits)
    
    log.info(f"Interleaving (Depth=42) to mitigate burst errors...")
    interleaver = MatrixInterleaver(depth=42)
    tx_bits_interleaved, orig_len = interleaver.interleave(encoded_bits)
    tx_symbols, tx_indices = bits_to_symbols(tx_bits_interleaved)
    
    # 3. Channel Simulation
    log.header("SIMULATING OPTICAL CHANNEL")
    log.info(f" Injecting Chromatic Dispersion: {CD_VAL} ps/nm/km")
    log.info(f" Injecting Phase Noise (Linewidth: {PN_RATE})")
    log.info(f" Adding AWGN (SNR: {SNR_DB} dB)")
    
    # Pad to SEQ_LEN
    n_syms = len(tx_symbols)
    n_batches = int(np.ceil(n_syms / SEQ_LEN))
    pad_len = n_batches * SEQ_LEN - n_syms
    tx_padded = np.pad(tx_symbols, (0, pad_len))
    
    rx_impaired_list = []
    for i in range(n_batches):
        sig = tx_padded[i*SEQ_LEN : (i+1)*SEQ_LEN]
        sig = apply_chromatic_dispersion_local(sig, cd_value=CD_VAL)
        sig = apply_phase_noise(sig, linewidth_rate=PN_RATE, rng=rng)
        sig = apply_awgn_local(sig, snr_db=SNR_DB, rng=rng)
        rx_impaired_list.append(sig)
    
    rx_impaired_full = np.concatenate(rx_impaired_list)
    log.warning(f"Channel Simulation Complete. Signal Degraded.")

    # 4. Neural Receiver
    log.header("ACTIVATING NEURAL RECEIVER (BiMamba)")
    
    rx_batch_stack = np.stack([np.stack([x.real, x.imag]) for x in rx_impaired_list])
    rx_tensor = torch.tensor(rx_batch_stack, dtype=torch.float32)
    
    # Normalize
    means = rx_tensor.view(rx_tensor.size(0), -1).mean(dim=1).view(-1, 1, 1)
    stds = rx_tensor.view(rx_tensor.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-8
    rx_tensor = (rx_tensor - means) / stds
    
    # Load Model
    ckpt = project_root / "experiments/logs/bimamba_curriculum/model_epoch_14.pt"
    if not ckpt.exists(): 
        log.warning("Checkpoint not found!")
        return
        
    model = CoherentBiMamba(num_classes=4, in_channels=2, d_model=64, num_layers=4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    log.success(f"Model Loaded: {ckpt.name}")

    rx_indices_flat = []
    with torch.inference_mode():
        dataset = torch.utils.data.TensorDataset(rx_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        start_t = time.time()
        for i, batch in enumerate(loader):
            logits = model(batch[0].to(device))
            rx_indices_flat.append(logits.argmax(1).cpu().numpy().flatten())
            if i % 10 == 0:
                print(f"   Processing Batch {i}/{len(loader)}...", end='\r')
        end_t = time.time()
        
    print(" " * 40, end='\r') # Clear line
    log.success(f"Inference Complete ({end_t - start_t:.2f}s). Reconstructing Stream...")
    
    rx_indices = np.concatenate(rx_indices_flat)[:n_syms]
    rx_nn_symbols = SYMBOL_MAP[rx_indices]

    # 5. Demodulation & Stats
    log.header("ERROR ANALYSIS")
    
    rx_bits_scrambled = symbols_to_bits(rx_indices)
    rx_bits_raw = interleaver.deinterleave(rx_bits_scrambled, orig_len)
    
    bit_errors = np.sum(rx_bits_raw != encoded_bits)
    pre_fec_ber = bit_errors / len(encoded_bits)
    
    log.info(f"Raw Output (Pre-FEC): {bit_errors} Errors")
    log.info(f"Raw BER: \033[1m{pre_fec_ber:.5e}\033[0m")
    
    if pre_fec_ber > 0.05:
        log.warning("BER too high for FEC!")
    else:
        log.info("BER within FEC Threshold. Engaging Decoder...")

    # 6. FEC
    rx_data_bits = decode_sys(rx_bits_raw)
    post_fec_errors = np.sum(rx_data_bits != tx_bits)
    post_fec_ber = post_fec_errors / len(tx_bits)
    
    if post_fec_errors == 0:
        log.success(f"Post-FEC Errors: {post_fec_errors}")
        log.success(f"Post-FEC BER:    0.00000")
        log.header("STATUS: SIGNAL RECOVERED")
    else:
        log.warning(f"Post-FEC Errors: {post_fec_errors}")
        log.header("STATUS: SIGNAL DEGRADED")

    # 7. EXPORT DATA (Replaces the save_dashboard call)
    # Re-encode corrected bits to get the "Post-FEC" symbols
    corrected_encoded = encode_sys(rx_data_bits)
    corrected_interleaved, _ = interleaver.interleave(corrected_encoded)
    corrected_syms, _ = bits_to_symbols(corrected_interleaved)
    
    # Save raw arrays for offline analysis
    filename = "full_demo_data_dump.npz"
    log.info(f"Saving raw arrays to {filename}...")
    np.savez(
        filename,
        tx_sym=tx_symbols,
        rx_impaired=rx_impaired_full[:n_syms],
        rx_nn_sym=rx_nn_symbols,
        rx_post_fec_sym=corrected_syms
    )
    log.success("Data export complete.")
    
    save_dashboard(tx_symbols, rx_impaired_full[:n_syms], rx_nn_symbols, corrected_syms)

if __name__ == "__main__":
    run_verbose_demo()