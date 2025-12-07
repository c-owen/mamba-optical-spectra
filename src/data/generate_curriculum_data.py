import os
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from pathlib import Path
from tqdm import tqdm

# --- PATH SETUP ---
# Calculate project root: src/data/ -> src/ -> root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Output to mamba-optical-spectra/data/curriculum
OUTPUT_DIR = PROJECT_ROOT / "data" / "curriculum"

# --- Configuration ---
OUTPUT_DIR = Path("data/curriculum")
SEED = 42

# Data Volume
SAMPLES_PER_STAGE = {
    "train": 10000, 
    "val": 1000, 
    "test": 1000
}
SEQ_LEN = 1024  # Length of one sequence

# Physics Parameters (Matched to your successful DSP script)
BAUD_RATE = 32e9
LAMBDA_NM = 1550.0
FIBER_LENGTH_KM = 100.0
C_MPS = 299792458

# Curriculum Definitions (CD Ranges in ps/nm/km)
STAGES = {
    "stage_1": (0.0, 5.0),   # Boot Camp: Learn to pass signal
    "stage_2": (0.0, 15.0),  # Field Ops: Moderate dispersion
    "stage_3": (0.0, 25.0),  # Full Spec: Includes your target of 20.0
}

def generate_batch(size, min_cd, max_cd):
    """Generates a batch of QPSK data with random CD."""
    
    # 1. Generate QPSK Symbols (0, 1, 2, 3)
    # Map to Constellation: 0->(1+1j), 1->(-1+1j), 2->(-1-1j), 3->(1-1j)
    # Using gray coding or simple mapping? Let's stick to simple quadrant mapping for labels.
    
    # Random integers 0-3
    labels = np.random.randint(0, 4, (size, SEQ_LEN))
    
    # Map to complex QPSK symbols
    # 0: (+1, +1), 1: (-1, +1), 2: (-1, -1), 3: (+1, -1)
    tx_real = np.where(np.isin(labels, [1, 2]), -1, 1)
    tx_imag = np.where(np.isin(labels, [2, 3]), -1, 1)
    tx_signal = tx_real + 1j * tx_imag
    
    # 2. Prepare Frequency Domain Physics
    freqs = fftfreq(SEQ_LEN, d=1/BAUD_RATE)
    # Constant factor for dispersion phase shift
    constant_factor = (np.pi * (LAMBDA_NM * 1e-9)**2 * FIBER_LENGTH_KM * 1e3) / C_MPS
    
    rx_signals_real = []
    rx_signals_imag = []
    
    for i in range(size):
        # Sample random CD for this sequence
        cd_val = np.random.uniform(min_cd, max_cd)
        
        # Apply CD Transfer Function
        # Factor 1e-12 converts ps/nm/km to s/m/km context
        phase_factor = constant_factor * cd_val * 1e-12
        transfer_func = np.exp(-1j * phase_factor * (freqs**2))
        
        rx_freq = fft(tx_signal[i]) * transfer_func
        rx_time = ifft(rx_freq)
        
        # Add Noise (AWGN)
        noise_power = 0.01
        noise = np.sqrt(noise_power/2) * (np.random.randn(SEQ_LEN) + 1j * np.random.randn(SEQ_LEN))
        rx_noisy = rx_time + noise
        
        rx_signals_real.append(rx_noisy.real)
        rx_signals_imag.append(rx_noisy.imag)
        
    # Stack into [Batch, 2, Seq_Len]
    X = np.stack([np.array(rx_signals_real), np.array(rx_signals_imag)], axis=1)
    
    # Y is [Batch, Seq_Len] (The labels 0-3)
    y = labels
    
    return X, y

def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Generating Curriculum Data ---")
    print(f"Output: {OUTPUT_DIR}\n")
    
    for stage_name, (min_cd, max_cd) in STAGES.items():
        print(f"Processing {stage_name} [CD: {min_cd} - {max_cd}]")
        
        for split, num_samples in SAMPLES_PER_STAGE.items():
            print(f"  > Generating {split} ({num_samples} samples)...")
            
            X, y = generate_batch(num_samples, min_cd, max_cd)
            
            # Save files matching CurriculumDataset expectation
            # X_{split}_{stage}.npy
            x_name = OUTPUT_DIR / f"X_{split}_{stage_name}.npy"
            y_name = OUTPUT_DIR / f"y_{split}_{stage_name}.npy"
            
            np.save(x_name, X.astype(np.float32))
            np.save(y_name, y.astype(np.int64))
            
    print("\nGeneration Complete.")

if __name__ == "__main__":
    main()