import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Output Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "curriculum_w_pn"

# Seed for reproducibility
SEED = 42

# Data Volume (Matches your workflow)
SAMPLES_PER_STAGE = {
    "train": 10000, 
    "val": 2000,   # Now we generate val automatically
    "test": 1000
}

# Physics Parameters
SEQ_LEN = 2048       # Sequence Length
BAUD_RATE = 32e9     # 32 Gbaud
SAMPLE_RATE = 32e9   # 1 Sample Per Symbol (SPS)

# --- THE 5-STAGE STRATEGY ---
# CD: Random 0-1600 ps/nm (approx 100km fiber)
# PN: Ramping up Linewidth (Hz)
# SNR: Ramping down (making it harder)
STAGES = {
    "stage_1": {"cd_range": (0, 1600), "lw": 0.0,       "snr_range": (20, 25)}, # Clean
    "stage_2": {"cd_range": (0, 1600), "lw": 100e3,     "snr_range": (18, 22)}, # Drift (Wake up)
    "stage_3": {"cd_range": (0, 1600), "lw": 500e3,     "snr_range": (16, 20)}, # Tracking
    "stage_4": {"cd_range": (0, 1600), "lw": 1e6,       "snr_range": (14, 18)}, # High Speed
    "stage_5": {"cd_range": (0, 1600), "lw": "random",  "snr_range": (10, 25)}, # Boss Fight
}

# --- PHYSICS FUNCTIONS ---

def apply_phase_noise(signal: np.ndarray, linewidth: float, baud_rate: float) -> np.ndarray:
    """Applies Wiener phase noise (Random Walk)."""
    if linewidth <= 0:
        return signal
    
    ts = 1.0 / baud_rate
    var = 2 * np.pi * linewidth * ts
    
    # Generate random phase steps
    # Shape: [Batch, Time] or [Time]
    deltas = np.random.normal(loc=0, scale=np.sqrt(var), size=signal.shape)
    phase = np.cumsum(deltas, axis=-1)
    
    return signal * np.exp(1j * phase)

def apply_cd(signal: np.ndarray, dispersion_ps_nm: float, baud_rate: float) -> np.ndarray:
    """Applies Chromatic Dispersion in Frequency Domain."""
    if dispersion_ps_nm == 0:
        return signal

    n_fft = signal.shape[-1]
    freq = np.fft.fftfreq(n_fft, d=1/baud_rate)
    
    # Physics Constants (1550nm)
    lambda_0 = 1550e-9
    c = 3e8
    
    # Dispersion Transfer Function
    beta2_total = -(dispersion_ps_nm * 1e-12 * (lambda_0**2)) / (2 * np.pi * c)
    omega = 2 * np.pi * freq
    transfer_function = np.exp(1j * 0.5 * beta2_total * (omega**2))
    
    # Apply
    spectrum = np.fft.fft(signal)
    dispersed_spectrum = spectrum * transfer_function
    return np.fft.ifft(dispersed_spectrum)

def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Adds AWGN."""
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_linear
    
    noise = (np.random.normal(0, np.sqrt(noise_power/2), signal.shape) + 
             1j * np.random.normal(0, np.sqrt(noise_power/2), signal.shape))
    
    return signal + noise

def generate_batch(num_samples, params):
    """Generates a batch of data for a specific stage configuration."""
    
    # 1. Generate QPSK Symbols (0-3)
    ints = np.random.randint(0, 4, (num_samples, SEQ_LEN))
    # Map 0->1+j, 1->-1+j, 2->-1-j, 3->1-j (Standard QPSK)
    phase = (2 * ints + 1) * np.pi / 4
    clean_sig = np.exp(1j * phase)
    
    processed_sig = np.zeros_like(clean_sig, dtype=np.complex64)
    
    # 2. Apply Impairments per sample
    for i in range(num_samples):
        sig_i = clean_sig[i]
        
        # A. Chromatic Dispersion
        cd_min, cd_max = params["cd_range"]
        cd_val = np.random.uniform(cd_min, cd_max)
        sig_i = apply_cd(sig_i, cd_val, BAUD_RATE)
        
        # B. Phase Noise
        lw = params["lw"]
        if lw == "random":
            lw = np.random.uniform(0, 2e6) # Random up to 2MHz
        sig_i = apply_phase_noise(sig_i, float(lw), BAUD_RATE)
        
        # C. AWGN
        snr_min, snr_max = params["snr_range"]
        snr = np.random.uniform(snr_min, snr_max)
        sig_i = add_awgn(sig_i, snr)
        
        processed_sig[i] = sig_i
        
    # Stack Real/Imag: [N, 2, L]
    X = np.stack([processed_sig.real, processed_sig.imag], axis=1).astype(np.float32)
    y = ints.astype(np.int64)
    
    return X, y

def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Generating Curriculum Data (with Phase Noise) ---")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Iterate over Stages
    for stage_name, params in STAGES.items():
        print(f"Processing {stage_name} | LW: {params['lw']} | SNR: {params['snr_range']}")
        
        # Iterate over Splits (Train, Val, Test)
        for split, num_samples in SAMPLES_PER_STAGE.items():
            print(f"  > Generating {split} ({num_samples} samples)...")
            
            X, y = generate_batch(num_samples, params)
            
            # Save files: X_{split}_{stage}.npy
            np.save(OUTPUT_DIR / f"X_{split}_{stage_name}.npy", X)
            np.save(OUTPUT_DIR / f"y_{split}_{stage_name}.npy", y)
            
    print("\n✅ Generation Complete. All splits created.")

if __name__ == "__main__":
    main()