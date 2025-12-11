import os
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "curriculum"

SEED = 42
SAMPLES_PER_STAGE = {"train": 10000, "val": 1000, "test": 1000}
SEQ_LEN = 1024

BAUD_RATE = 32e9
LAMBDA_NM = 1550.0
FIBER_LENGTH_KM = 100.0
C_MPS = 299792458

# CD ranges in ps/nm/km
STAGES = {
    "stage_1": (0.0, 5.0),
    "stage_2": (0.0, 15.0),
    "stage_3": (0.0, 25.0),
}


def generate_batch(size, min_cd, max_cd):
    """Generate a batch of QPSK data with randomized chromatic dispersion."""
    labels = np.random.randint(0, 4, (size, SEQ_LEN))

    tx_real = np.where(np.isin(labels, [1, 2]), -1, 1)
    tx_imag = np.where(np.isin(labels, [2, 3]), -1, 1)
    tx_signal = tx_real + 1j * tx_imag

    freqs = fftfreq(SEQ_LEN, d=1/BAUD_RATE)
    dispersion_constant = (np.pi * (LAMBDA_NM * 1e-9)**2 * FIBER_LENGTH_KM * 1e3) / C_MPS

    rx_signals_real = []
    rx_signals_imag = []

    for i in range(size):
        cd_val = np.random.uniform(min_cd, max_cd)

        phase_factor = dispersion_constant * cd_val * 1e-6
        transfer_func = np.exp(-1j * phase_factor * (freqs**2))

        rx_freq = fft(tx_signal[i]) * transfer_func
        rx_time = ifft(rx_freq)

        noise_power = 0.01
        noise = np.sqrt(noise_power/2) * (np.random.randn(SEQ_LEN) + 1j * np.random.randn(SEQ_LEN))
        rx_noisy = rx_time + noise

        rx_signals_real.append(rx_noisy.real)
        rx_signals_imag.append(rx_noisy.imag)

    X = np.stack([np.array(rx_signals_real), np.array(rx_signals_imag)], axis=1)
    y = labels

    return X, y


def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating curriculum data to {OUTPUT_DIR}\n")

    for stage_name, (min_cd, max_cd) in STAGES.items():
        print(f"Processing {stage_name} [CD: {min_cd} - {max_cd}]")

        for split, num_samples in SAMPLES_PER_STAGE.items():
            print(f"  {split}: {num_samples} samples")

            X, y = generate_batch(num_samples, min_cd, max_cd)

            x_path = OUTPUT_DIR / f"X_{split}_{stage_name}.npy"
            y_path = OUTPUT_DIR / f"y_{split}_{stage_name}.npy"

            np.save(x_path, X.astype(np.float32))
            np.save(y_path, y.astype(np.int64))

    print("\nComplete.")


if __name__ == "__main__":
    main()