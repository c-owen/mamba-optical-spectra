import sys
import logging
from pathlib import Path
import numpy as np
import torch
from scipy.fftpack import fft, ifft, fftfreq # Added for local physics

# Add project root
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.bimamba import CoherentBiMamba

# --- 1. HARDCODED CONFIGURATION (Must match generate_curriculum_data.py) ---
# Physics Constants
BAUD_RATE = 32e9
LAMBDA_NM = 1550.0
FIBER_LENGTH_KM = 100.0
C_MPS = 299792458

# EXACT SYMBOL MAPPING from generate_curriculum_data.py
# 0:(1,1), 1:(-1,1), 2:(-1,-1), 3:(1,-1)
SYMBOL_MAP = np.array([
    1 + 1j,   # Index 0 (Top Right)
   -1 + 1j,   # Index 1 (Top Left)
   -1 - 1j,   # Index 2 (Bottom Left)
    1 - 1j    # Index 3 (Bottom Right)
])

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def apply_phase_noise(signal, linewidth_rate=1e-4, rng=None):
    """
    Applies Phase Noise as a random walk (Wiener process).
    
    Args:
        signal (np.array): Complex signal
        linewidth_rate (float): Normalized linewidth (Delta_nu * Ts). 
                                Typical values: 1e-4 to 1e-3 for decent lasers.
                                0.1 (from your comment) is EXTREMELY high (catastrophic).
        rng: Numpy random generator
    """
    if rng is None: rng = np.random.default_rng()
    
    # Calculate variance of the phase step
    # sigma^2 = 2 * pi * (linewidth * Ts)
    phase_noise_var = 2 * np.pi * linewidth_rate
    phase_noise_std = np.sqrt(phase_noise_var)
    
    # Generate random steps
    steps = rng.normal(0, phase_noise_std, size=len(signal))
    
    # Integrate to get phase walk
    phase_walk = np.cumsum(steps)
    
    # Apply rotation
    return signal * np.exp(1j * phase_walk)

# --- 2. PHYSICS FUNCTIONS (Copied from Generator) ---
def apply_chromatic_dispersion_local(signal, cd_value=20.0):
    """
    Applies CD using the EXACT math from the training data generator.
    """
    n = len(signal)
    freqs = fftfreq(n, d=1/BAUD_RATE)
    
    # Physics Constant Calculation (Exact copy)
    constant_factor = (np.pi * (LAMBDA_NM * 1e-9)**2 * FIBER_LENGTH_KM * 1e3) / C_MPS
    
    # Phase Factor (Exact copy)
    # Note: Generator used cd_val * 1e-12. 
    phase_factor = constant_factor * cd_value * 1e-6
    
    transfer_func = np.exp(-1j * phase_factor * (freqs**2))
    
    rx_freq = fft(signal) * transfer_func
    rx_time = ifft(rx_freq)
    
    return rx_time

def apply_awgn_local(signal, snr_db=20.0, rng=None):
    """Adds White Gaussian Noise."""
    if rng is None: rng = np.random.default_rng()
    
    # Calculate signal power
    sig_power = np.mean(np.abs(signal)**2)
    
    # Calculate noise power based on SNR
    snr_linear = 10**(snr_db / 10)
    noise_power = sig_power / snr_linear
    
    # Generate complex noise
    noise = np.sqrt(noise_power/2) * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    return signal + noise

# --- 3. Hamming (7,4) Systematic Implementation ---
# G_sys = [I_4 | P]
G_sys = np.array([
    [1,0,0,0, 1,1,0],
    [0,1,0,0, 1,0,1],
    [0,0,1,0, 0,1,1],
    [0,0,0,1, 1,1,1]
], dtype=int)
# H_sys = [P^T | I_3]
H_sys = np.array([
    [1,1,0,1, 1,0,0],
    [1,0,1,1, 0,1,0],
    [0,1,1,1, 0,0,1]
], dtype=int)

def encode_sys(data_bits):
    blocks = data_bits.reshape(-1, 4)
    encoded = np.dot(blocks, G_sys) % 2
    return encoded.flatten()

def decode_sys(rx_bits):
    blocks = rx_bits.reshape(-1, 7)
    syndromes = np.dot(blocks, H_sys.T) % 2
    s_int = syndromes.dot(np.array([4, 2, 1]))
    
    # Map syndrome to column index to flip
    syn_to_col = {6:0, 5:1, 3:2, 7:3, 4:4, 2:5, 1:6}
    
    corrected_data = []
    for i in range(len(blocks)):
        blk = blocks[i]
        s = s_int[i]
        if s != 0:
            err_idx = syn_to_col.get(s)
            if err_idx is not None:
                blk[err_idx] ^= 1 
        corrected_data.append(blk[:4])
        
    return np.concatenate(corrected_data)

class MatrixInterleaver:
    """
    Block Interleaver to break up burst errors.
    Writes row-wise, reads column-wise.
    """
    def __init__(self, depth=14):
        # Depth 14 is chosen because it is a multiple of the Hamming block size (7)
        # This ensures that adjacent bits in a Hamming block are separated by 14 positions.
        self.depth = depth 
        
    def interleave(self, bits):
        n = len(bits)
        # Pad to ensure divisibility by depth
        pad_len = (self.depth - (n % self.depth)) % self.depth
        bits_padded = np.pad(bits, (0, pad_len), constant_values=0)
        
        # Reshape to Matrix (Rows, Depth)
        rows = len(bits_padded) // self.depth
        matrix = bits_padded.reshape(rows, self.depth)
        
        # Read out Column-wise (Transpose -> Flatten)
        interleaved = matrix.T.flatten()
        return interleaved, n # Return original length for receiver

    def deinterleave(self, bits, original_length):
        # Reconstruct Matrix from Column-wise stream
        # Input length must be divisible by depth (it will be, because we padded tx)
        rows = len(bits) // self.depth
        
        # To undo "Transpose -> Flatten", we reshape to (Depth, Rows) then Transpose
        matrix_T = bits.reshape(self.depth, rows)
        matrix = matrix_T.T
        
        # Flatten Row-wise to get original sequence
        deinterleaved = matrix.flatten()
        
        # Remove padding
        return deinterleaved[:original_length]

# --- 4. QPSK Mapper/Demapper ---
def bits_to_symbols(bits):
    # 00->0, 01->1, 10->2, 11->3
    pairs = bits.reshape(-1, 2)
    indices = pairs[:, 0] * 2 + pairs[:, 1]
    return SYMBOL_MAP[indices], indices

def symbols_to_bits(symbol_indices):
    bits = []
    for s in symbol_indices:
        b0 = (s >> 1) & 1
        b1 = (s >> 0) & 1
        bits.extend([b0, b1])
    return np.array(bits, dtype=int)

# --- 5. Main Loop ---
def run_simulation():
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running FEC Loop on {device}")

    # A. Setup (Align count with Script 2)
    n_data_bits = 50000 
    rng = np.random.default_rng(42)
    
    # B. Generate & Encode
    tx_bits = rng.integers(0, 2, size=n_data_bits)
    encoded_bits = encode_sys(tx_bits)
    logging.info(f"FEC Encoded: {len(tx_bits)} -> {len(encoded_bits)} bits.")

    interleaver = MatrixInterleaver(depth=42)
    tx_bits_interleaved, orig_len = interleaver.interleave(encoded_bits)
    
    # REMOVED: The redundant re-encoding line was here.
    
    # C. Modulate
    tx_symbols_c, tx_indices = bits_to_symbols(tx_bits_interleaved)
    
    # Pad to match Training Sequence Length (1024)
    seq_len = 1024 
    n_syms = len(tx_symbols_c)
    n_batches = int(np.ceil(n_syms / seq_len))
    pad_len = n_batches * seq_len - n_syms
    
    tx_symbols_padded = np.pad(tx_symbols_c, (0, pad_len), constant_values=0)
    
    # D. Apply Channel 
    logging.info("Applying Channel (CD=20.0, AWGN, Phase Noise=1e-5)...")
    
    rx_batch_list = []
    for i in range(n_batches):
        sig = tx_symbols_padded[i*seq_len : (i+1)*seq_len]
        
        # 1. Local Physics
        sig = apply_chromatic_dispersion_local(sig, cd_value=20.0)
        
        # 2. Phase Noise (UNCOMMENTED)
        # Using 1e-5 to match Script 2's LINEWIDTH constant
        sig = apply_phase_noise(sig, linewidth_rate=1e-5, rng=rng)
        
        # 3. AWGN
        sig = apply_awgn_local(sig, snr_db=20.0, rng=rng)
        
        sample_iq = np.stack([sig.real, sig.imag], axis=0)
        rx_batch_list.append(sample_iq)
        
    rx_tensor = torch.tensor(np.array(rx_batch_list), dtype=torch.float32) # [B, 2, L]

    # Normalize (Must match datasets_curriculum.py logic)
    # dataset uses mean/std of entire [2, L] array.
    # Here we do it per batch item.
    means = rx_tensor.view(rx_tensor.size(0), -1).mean(dim=1).view(-1, 1, 1)
    stds = rx_tensor.view(rx_tensor.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-8
    rx_tensor = (rx_tensor - means) / stds
    
    # E. Neural Receiver
    logging.info("Running BiMamba Equalizer...")
    model = CoherentBiMamba(
        num_classes=4, 
        in_channels=2, 
        d_model=64, 
        num_layers=4
    ).to(device)
    
    ckpt = project_root / "experiments/logs/bimamba_curriculum/model_epoch_14.pt"
    if ckpt.exists():
        logging.info(f"Loading weights from: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    else:
        raise FileNotFoundError("Checkpoint not found!")
        
    model.eval()
    
    rx_indices_flat = []
    
    with torch.inference_mode():
        dataset = torch.utils.data.TensorDataset(rx_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        
        for batch in loader:
            inputs = batch[0].to(device) 
            logits = model(inputs)       
            _, preds = logits.max(1)     
            rx_indices_flat.append(preds.cpu().numpy().flatten())
            
    rx_indices_flat = np.concatenate(rx_indices_flat)
    rx_indices_valid = rx_indices_flat[:n_syms]
    
    # F. Demodulate
    rx_bits_scrambled = symbols_to_bits(rx_indices_valid)

    rx_bits_raw = interleaver.deinterleave(rx_bits_scrambled, orig_len)
    
    bit_errors = np.sum(rx_bits_raw != encoded_bits)
    pre_fec_ber = bit_errors / len(encoded_bits)
    
    logging.info(f"--- RESULTS ---")
    logging.info(f"Pre-FEC Errors: {bit_errors} / {len(encoded_bits)}")
    logging.info(f"Pre-FEC BER:    {pre_fec_ber:.5e}")
    
    # G. FEC Decode
    rx_data_bits = decode_sys(rx_bits_raw)
    post_fec_errors = np.sum(rx_data_bits != tx_bits)
    post_fec_ber = post_fec_errors / len(tx_bits)
    
    logging.info(f"Post-FEC Errors:    {post_fec_errors}")
    logging.info(f"Post-FEC BER:       {post_fec_ber:.5e}")

if __name__ == "__main__":
    run_simulation()