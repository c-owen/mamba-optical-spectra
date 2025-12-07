import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

def run_dsp_benchmark():
    # --- 1. System Parameters (Match these to your BiMamba config) ---
    NUM_SYMBOLS = 10000        # Number of symbols to simulate
    BAUD_RATE = 32e9           # 32 Gbaud (standard for optical)
    SAMPLE_RATE = BAUD_RATE    # 1 sample per symbol (simplest case)
    
    # Physics Parameters
    LAMBDA_NM = 1550.0         # Center wavelength (nm)
    CD_VALUE = 20.0            # Dispersion Parameter D (ps/nm/km) - MATCH YOUR LOGS
    FIBER_LENGTH_KM = 100.0    # Distance (km)
    
    # Calculated Physics Constants
    c_kms = 299792.458         # Speed of light in km/s
    center_freq = c_kms / (LAMBDA_NM * 1e-9)
    
    print(f"--- Running Classical DSP Benchmark ---")
    print(f"Target CD: {CD_VALUE} ps/nm/km | Length: {FIBER_LENGTH_KM} km")
    
    # --- 2. Generate Data (QPSK) ---
    # Random bits mapped to constellations: 00->1+j, 01->-1+j, etc.
    # Simple QPSK: Real and Imag parts are +/- 1
    tx_bits_i = 2 * np.random.randint(0, 2, NUM_SYMBOLS) - 1
    tx_bits_q = 2 * np.random.randint(0, 2, NUM_SYMBOLS) - 1
    tx_signal = tx_bits_i + 1j * tx_bits_q
    
    print(f"Generated {NUM_SYMBOLS} QPSK symbols.")

    # --- 3. Simulate The Channel (The "Problem") ---
    # Apply Chromatic Dispersion in Frequency Domain
    # Transfer Function H(w) = exp(-j * (D * lambda^2 * L * w^2) / (4 * pi * c))
    
    # Create frequency axis
    freqs = fftfreq(NUM_SYMBOLS, d=1/SAMPLE_RATE)
    omega = 2 * np.pi * freqs
    
    # Dispersion constant: beta2_approx = - (D * lambda^2) / (2 * pi * c)
    # Phase shift term
    D_si = CD_VALUE * 1e-6    # ps/nm/km -> s/m^2 approx conversion factor handling 
    # Simplified Phase shift for simulation (Standard CD Formula):
    # Phase = (D * lambda^2 * z * pi * f^2) / c
    # Note: Pay close attention to units here in your real generic simulator
    
    # Unit conversions for the phase equation:
    # We use a robust approximation often used in DSP sims:
    # dispersion_phase = 0.5 * beta2 * omega^2 * length
    # where beta2 = -D * lambda^2 / (2 * pi * c)
    
    # beta2 calculation (s^2/km)
    lambda_m = LAMBDA_NM * 1e-9
    c_mps = 299792458
    D_sm_km = CD_VALUE * 1e-6 # s/m/km ? No, commonly ps/nm/km ~ 17
    # Let's use the direct standard DSP formula for Phase Mask:
    # phi(f) = (pi * D * lambda^2 * L * f^2) / c
    
    phase_factor = (np.pi * CD_VALUE * 1e-12 * (LAMBDA_NM * 1e-9)**2 * FIBER_LENGTH_KM * 1e3) / c_mps
    # This factor needs to be multiplied by f^2 (Hz^2)
    
    channel_transfer_func = np.exp(-1j * phase_factor * (freqs**2))
    
    # Apply Channel
    rx_signal_freq = fft(tx_signal) * channel_transfer_func
    rx_signal_time = ifft(rx_signal_freq)
    
    # Add some AWGN Noise (Simulating OSNR)
    noise_power = 0.01  # Adjust to match your PN=0.1 logs
    noise = np.sqrt(noise_power/2) * (np.random.randn(NUM_SYMBOLS) + 1j * np.random.randn(NUM_SYMBOLS))
    rx_signal_noisy = rx_signal_time + noise

    print("Applied High Dispersion (Physics) + Noise.")

    # --- 4. The Classical DSP Solution (The "Fix") ---
    # We know the physics, so we simply invert the transfer function.
    # Inverse H(w) = Conjugate of Channel H(w)
    
    dsp_start_freq = fft(rx_signal_noisy)
    
    # The Equalizer: Simply flip the sign of the phase shift!
    compensator = np.exp(1j * phase_factor * (freqs**2))
    
    equalized_freq = dsp_start_freq * compensator
    equalized_time = ifft(equalized_freq)
    
    # --- 5. Analysis & BER ---
    # Slicing / Hard Decision
    rx_bits_i = np.sign(equalized_time.real)
    rx_bits_q = np.sign(equalized_time.imag)
    
    # Error Counting
    errors_i = np.sum(rx_bits_i != tx_bits_i)
    errors_q = np.sum(rx_bits_q != tx_bits_q)
    total_errors = errors_i + errors_q
    total_bits = NUM_SYMBOLS * 2
    
    ber = total_errors / total_bits
    
    print(f"\n--- RESULTS ---")
    print(f"Total Bits:     {total_bits}")
    print(f"Bit Errors:     {total_errors}")
    print(f"Pre-FEC BER:    {ber:.5e}")
    
    if ber < 3.8e-3:
        print(">> SUCCESS: Below HD-FEC Threshold (3.8e-3)")
    else:
        print(">> FAIL: Still too noisy")

    # --- 6. Visualization (Constellation) ---
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Raw RX Signal (CD={CD_VALUE})")
    plt.scatter(rx_signal_noisy.real[:1000], rx_signal_noisy.imag[:1000], alpha=0.3, s=1)
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.title(f"DSP Corrected (BER={ber:.2e})")
    plt.scatter(equalized_time.real[:1000], equalized_time.imag[:1000], alpha=0.3, s=1, c='r')
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    run_dsp_benchmark()