#generate_coherent.py
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Constants for QPSK
# Maps integers 0-3 to complex symbols
SYMBOL_MAP = np.array([
    1 + 1j,  # 0: Q1
    -1 + 1j, # 1: Q2
    -1 - 1j, # 2: Q3
    1 - 1j,  # 3: Q4
]) / np.sqrt(2) # Normalize power to 1


def setup_logging(verbosity: int = 1) -> None:
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def apply_phase_noise(signal: np.ndarray, linewidth_rate: float, rng: np.random.Generator) -> np.ndarray:
    """
    Applies random phase walk (Wiener process).
    linewidth_rate: strength of the phase noise (variance of step size).
    """
    if linewidth_rate <= 0:
        return signal
    
    # Generate random phase steps
    phase_steps = rng.normal(0, linewidth_rate, size=signal.shape)
    # Accumulate steps to create "random walk"
    phase_noise = np.cumsum(phase_steps)
    return signal * np.exp(1j * phase_noise)


def apply_chromatic_dispersion(signal: np.ndarray, cd_amount: float) -> np.ndarray:
    """
    Applies Chromatic Dispersion in the Frequency Domain.
    This acts as an all-pass filter with quadratic phase response.
    """
    if cd_amount == 0:
        return signal

    n = len(signal)
    freqs = np.fft.fftfreq(n)
    
    # Quadratic phase transfer function: exp(j * alpha * omega^2)
    # This smears the pulse in time
    transfer_function = np.exp(1j * cd_amount * (freqs ** 2))
    
    spectrum = np.fft.fft(signal)
    dispersed = np.fft.ifft(spectrum * transfer_function)
    return dispersed


def apply_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Additive White Gaussian Noise based on target SNR."""
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Complex noise
    noise = (rng.normal(0, 1, signal.shape) + 1j * rng.normal(0, 1, signal.shape)) 
    noise *= np.sqrt(noise_power / 2) # Divide by 2 because complex
    
    return signal + noise


def generate_batch(
    num_samples: int,
    seq_len: int,
    cd_range: Tuple[float, float],
    pn_range: Tuple[float, float],
    snr_range: Tuple[float, float],
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a batch of synthetic optical data.
    
    Returns:
        X: [Batch, 2, Seq_Len] (Channel 0 is Real/I, Channel 1 is Imag/Q)
        y: [Batch, Seq_Len] (Integer labels 0-3)
    """
    X_batch = []
    y_batch = []

    for i in range(num_samples):
        # 1. Generate random symbols (Labels)
        labels = rng.integers(0, 4, size=seq_len)
        y_batch.append(labels)
        
        # 2. Map to Complex QPSK
        tx_signal = SYMBOL_MAP[labels]

        # 3. Randomize Impairments per sample
        cd_val = rng.uniform(*cd_range)
        pn_val = rng.uniform(*pn_range)
        snr_val = rng.uniform(*snr_range)

        # 4. Apply Channel Physics
        # Order matters: Pulse Shaping (CD) -> Phase Noise -> AWGN
        rx_signal = apply_chromatic_dispersion(tx_signal, cd_val)
        rx_signal = apply_phase_noise(rx_signal, pn_val, rng)
        rx_signal = apply_awgn(rx_signal, snr_val, rng)

        # 5. Stack Real (I) and Imag (Q) channels
        # Shape: [2, Seq_Len]
        sample_iq = np.stack([rx_signal.real, rx_signal.imag], axis=0)
        X_batch.append(sample_iq)

        if (i + 1) % 500 == 0:
            logging.info(f"Generated {i+1}/{num_samples} samples")

    return np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Generate Coherent Optical QPSK Dataset")
    parser.add_argument("--output-dir", type=str, default="data/coherent")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seq-len", type=int, default=512)
    # Impairment settings
    parser.add_argument("--cd-max", type=float, default=20.0, help="Max Chromatic Dispersion factor")
    parser.add_argument("--pn-max", type=float, default=0.1, help="Max Phase Noise rate")
    parser.add_argument("--snr-min", type=float, default=10.0, help="Min SNR in dB")
    parser.add_argument("--snr-max", type=float, default=20.0, help="Max SNR in dB")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    # Split sizes
    n_train = int(0.7 * args.num_samples)
    n_val = int(0.15 * args.num_samples)
    n_test = args.num_samples - n_train - n_val

    # Generate Data
    logging.info("Generating Training Set...")
    X_train, y_train = generate_batch(n_train, args.seq_len, (0, args.cd_max), (0, args.pn_max), (args.snr_min, args.snr_max), rng)
    
    logging.info("Generating Validation Set...")
    X_val, y_val = generate_batch(n_val, args.seq_len, (0, args.cd_max), (0, args.pn_max), (args.snr_min, args.snr_max), rng)
    
    logging.info("Generating Test Set...")
    X_test, y_test = generate_batch(n_test, args.seq_len, (0, args.cd_max), (0, args.pn_max), (args.snr_min, args.snr_max), rng)

    # Save
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_val.npy", X_val)
    np.save(output_dir / "y_val.npy", y_val)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    # Save Metadata
    meta = {
        "num_classes": 4,
        "class_names": ["Q1", "Q2", "Q3", "Q4"],
        "impairments": {
            "cd_max": args.cd_max,
            "pn_max": args.pn_max,
            "snr_range": [args.snr_min, args.snr_max]
        }
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logging.info(f"Done. Saved to {output_dir}")

if __name__ == "__main__":
    main()