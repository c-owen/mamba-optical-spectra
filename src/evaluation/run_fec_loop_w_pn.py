# No interleaver here
import sys
import logging
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

# Physics Constants
BAUD_RATE = 32e9
LAMBDA_NM = 1550.0
C_MPS = 299792458

# Test Parameters
N_DATA_BITS = 100_000   
TEST_TOTAL_DISPERSION = 1200.0  # Total Dispersion in ps/nm (Training range was 0-1600)
TEST_SNR = 18.0                 # dB (Training range was 14-25 in later stages)
TEST_LINEWIDTH = 500e3          # 500 kHz (Matches Stage 3)

# QPSK MAPPING
SYMBOL_MAP = np.array([
    1 + 1j,   # 0
   -1 + 1j,   # 1
   -1 - 1j,   # 2
    1 - 1j    # 3
])

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

# --- PHYSICS ENGINE (MATCHING GENERATOR) ---

def apply_phase_noise(signal, linewidth, baud_rate, rng=None):
    """
    Applies Wiener phase noise (Random Walk).
    Matches generate_curriculum_w_pn.py logic EXACTLY.
    """
    if rng is None: rng = np.random.default_rng()
    if linewidth <= 0: return signal
    
    ts = 1.0 / baud_rate
    var = 2 * np.pi * linewidth * ts
    
    # Generate random phase steps
    steps = rng.normal(0, np.sqrt(var), size=len(signal))
    phase = np.cumsum(steps)
    
    return signal * np.exp(1j * phase)

def apply_chromatic_dispersion(signal, dispersion_ps_nm, baud_rate):
    """
    Applies CD in Frequency Domain.
    Matches generate_curriculum_w_pn.py logic EXACTLY.
    """
    if dispersion_ps_nm == 0: return signal

    n_fft = len(signal)
    freq = fftfreq(n_fft, d=1/baud_rate)
    
    # Physics Constants (1550nm)
    lambda_0 = 1550e-9
    c = 3e8
    
    # Dispersion Transfer Function
    # Beta2 calculation matching generator
    beta2_total = -(dispersion_ps_nm * 1e-12 * (lambda_0**2)) / (2 * np.pi * c)
    omega = 2 * np.pi * freq
    transfer_function = np.exp(1j * 0.5 * beta2_total * (omega**2))
    
    # Apply
    spectrum = fft(signal)
    dispersed_spectrum = spectrum * transfer_function
    return ifft(dispersed_spectrum)

def apply_awgn(signal, snr_db, rng=None):
    if rng is None: rng = np.random.default_rng()
    
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_linear
    
    noise = np.sqrt(noise_power/2) * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return signal + noise

# --- HAMMING FEC & INTERLEAVER ---

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

# --- QPSK TOOLS ---

def bits_to_symbols(bits):
    pairs = bits.reshape(-1, 2)
    indices = pairs[:, 0] * 2 + pairs[:, 1]
    return SYMBOL_MAP[indices]

def symbols_to_bits(indices):
    bits = []
    for s in indices:
        bits.extend([(s >> 1) & 1, s & 1])
    return np.array(bits, dtype=int)

# --- MAIN LOOP ---

def run_simulation():
    setup_logging()
    logging.info(f"--- STARTING FEC TEST LOOP (Phase Noise Enabled) ---")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Model: {CHECKPOINT_PATH.name}")
    logging.info(f"Physics: TotalDisp={TEST_TOTAL_DISPERSION} ps/nm | SNR={TEST_SNR}dB | LW={TEST_LINEWIDTH/1e3}kHz")

    rng = np.random.default_rng(42)

    # 1. GENERATE & ENCODE
    n_bits = N_DATA_BITS - (N_DATA_BITS % 4)
    tx_bits = rng.integers(0, 2, size=n_bits)
    
    encoded_bits = encode_sys(tx_bits)
    interleaver = MatrixInterleaver(depth=42)
    tx_bits_intl, orig_len = interleaver.interleave(encoded_bits)
    
    tx_symbols = bits_to_symbols(tx_bits_intl)
    
    SEQ_LEN = 2048 
    n_syms = len(tx_symbols)
    n_batches = int(np.ceil(n_syms / SEQ_LEN))
    pad_len = n_batches * SEQ_LEN - n_syms
    
    tx_padded = np.pad(tx_symbols, (0, pad_len))
    logging.info(f"Transmitting {n_batches} blocks of {SEQ_LEN} symbols.")

    # 2. APPLY CHANNEL
    rx_batch_list = []
    for i in range(n_batches):
        sig = tx_padded[i*SEQ_LEN : (i+1)*SEQ_LEN]
        
        # A. CD (Using aligned Physics)
        sig = apply_chromatic_dispersion(sig, TEST_TOTAL_DISPERSION, BAUD_RATE)
        # B. Phase Noise (Using aligned Physics)
        sig = apply_phase_noise(sig, TEST_LINEWIDTH, BAUD_RATE, rng)
        # C. AWGN
        sig = apply_awgn(sig, TEST_SNR, rng)
        
        # D. Phase-Preserving Normalization
        # Center Independent
        sig_real = sig.real - np.mean(sig.real)
        sig_imag = sig.imag - np.mean(sig.imag)
        
        # Scale Global (RMS)
        pwr = np.mean(sig_real**2 + sig_imag**2)
        scale = np.sqrt(pwr) + 1e-8
        
        # Stack for Model [2, L]
        sample_iq = np.stack([sig_real/scale, sig_imag/scale], axis=0)
        rx_batch_list.append(sample_iq)

    rx_tensor = torch.tensor(np.array(rx_batch_list), dtype=torch.float32)

    # 3. NEURAL RECEIVER
    model = CoherentBiMamba(
        num_classes=4, in_channels=2, d_model=64, num_layers=4
    ).to(DEVICE)
    
    if not CHECKPOINT_PATH.exists():
        logging.error(f"Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # Use weights_only=True to silence warning
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    
    rx_indices_all = []
    
    with torch.no_grad():
        dataset = torch.utils.data.TensorDataset(rx_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        for batch in loader:
            inputs = batch[0].to(DEVICE)
            logits = model(inputs) # [B, 4, L]
            preds = logits.argmax(dim=1) # [B, L]
            rx_indices_all.append(preds.cpu().numpy().flatten())
            
    rx_indices_all = np.concatenate(rx_indices_all)
    rx_indices_valid = rx_indices_all[:n_syms]

    # 4. DECODE & MEASURE
    rx_bits_intl = symbols_to_bits(rx_indices_valid)
    rx_bits_raw = interleaver.deinterleave(rx_bits_intl, orig_len)
    
    # Pre-FEC BER
    bit_errors = np.sum(rx_bits_raw != encoded_bits)
    pre_fec_ber = bit_errors / len(encoded_bits)
    
    # Post-FEC BER
    rx_data = decode_sys(rx_bits_raw)
    post_errors = np.sum(rx_data != tx_bits)
    post_fec_ber = post_errors / len(tx_bits)
    
    logging.info("-" * 40)
    logging.info(f"RESULTS (CD={TEST_TOTAL_DISPERSION} ps/nm, LW={TEST_LINEWIDTH/1e3}kHz, SNR={TEST_SNR})")
    logging.info("-" * 40)
    logging.info(f"Pre-FEC BER:  {pre_fec_ber:.5e} ({bit_errors} errors)")
    logging.info(f"Post-FEC BER: {post_fec_ber:.5e} ({post_errors} errors)")
    logging.info("-" * 40)
    
    if post_fec_ber == 0:
        logging.info("🎉 PERFECT TRANSMISSION! Zero errors after FEC.")

if __name__ == "__main__":
    run_simulation()