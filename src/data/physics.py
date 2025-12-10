import numpy as np

# --- CONSTANTS ---
BAUD_RATE = 32e9      # 32 Gbaud
LAMBDA_NM = 1550.0    # Center Wavelength
C_MPS = 299792458     # Speed of Light

def generate_qpsk_symbols(
    num_samples: int, 
    seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates random QPSK symbols and their integer labels.
    
    Returns:
        tx_signal: Complex symbols [Batch, Seq_Len]
        labels: Integer labels 0-3 [Batch, Seq_Len]
    """
    # 0->(1+1j), 1->(-1+1j), 2->(-1-1j), 3->(1-1j)
    labels = np.random.randint(0, 4, (num_samples, seq_len))
    
    # Map integers to phase: (2*k + 1) * pi/4
    # This results in the standard QPSK constellation points
    phase = (2 * labels + 1) * np.pi / 4
    tx_signal = np.exp(1j * phase)
    
    return tx_signal.astype(np.complex64), labels.astype(np.int64)

def apply_chromatic_dispersion(
    signal: np.ndarray, 
    cd_val: float, 
    baud_rate: float = BAUD_RATE
) -> np.ndarray:
    """
    Applies Chromatic Dispersion (CD) in the frequency domain.
    cd_val: Dispersion in ps/nm (e.g., 1600 for ~100km fiber).
    """
    if cd_val == 0:
        return signal

    n_fft = signal.shape[-1]
    freqs = np.fft.fftfreq(n_fft, d=1/baud_rate)
    
    # Physics Constants
    lambda_0 = LAMBDA_NM * 1e-9
    
    # Transfer Function: exp(j * 0.5 * beta2 * omega^2)
    # We convert CD (ps/nm) to beta2 logic directly:
    # Phase shift ~ CD * lambda^2 / (4 * pi * c) * omega^2
    # Derived from your provided scripts
    constant_factor = (np.pi * lambda_0**2 * cd_val * 1e-12) / C_MPS
    transfer_func = np.exp(-1j * constant_factor * (C_MPS * freqs)**2) # Approximation alignment

    # Note: Your scripts used slightly different CD math. 
    # This aligns with the 'generate_curriculum_w_pn.py' implementation:
    beta2_total = -(cd_val * 1e-12 * (lambda_0**2)) / (2 * np.pi * C_MPS)
    omega = 2 * np.pi * freqs
    transfer_func = np.exp(1j * 0.5 * beta2_total * (omega**2))
    
    spectrum = np.fft.fft(signal)
    return np.fft.ifft(spectrum * transfer_func)

def apply_phase_noise(
    signal: np.ndarray, 
    linewidth: float, 
    baud_rate: float = BAUD_RATE
) -> np.ndarray:
    """Applies Wiener phase noise (Random Walk)."""
    if linewidth <= 0:
        return signal
    
    ts = 1.0 / baud_rate
    var = 2 * np.pi * linewidth * ts
    
    deltas = np.random.normal(loc=0, scale=np.sqrt(var), size=signal.shape)
    phase = np.cumsum(deltas, axis=-1)
    
    return signal * np.exp(1j * phase)

def apply_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Adds Additive White Gaussian Noise."""
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_linear
    
    noise = (np.random.normal(0, np.sqrt(noise_power/2), signal.shape) + 
             1j * np.random.normal(0, np.sqrt(noise_power/2), signal.shape))
    
    return signal + noise