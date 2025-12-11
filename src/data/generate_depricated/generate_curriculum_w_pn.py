import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "curriculum_w_pn"

SEED = 42
SAMPLES_PER_STAGE = {"train": 10000, "val": 2000, "test": 1000}

SEQ_LEN = 2048
BAUD_RATE = 32e9
SAMPLE_RATE = 32e9

# CD: ps/nm (0-1600 ≈ 100km fiber), LW: Hz, SNR: dB
STAGES = {
    "stage_1": {"cd_range": (0, 1600), "lw": 0.0,      "snr_range": (20, 25)},
    "stage_2": {"cd_range": (0, 1600), "lw": 100e3,   "snr_range": (18, 22)},
    "stage_3": {"cd_range": (0, 1600), "lw": 500e3,   "snr_range": (16, 20)},
    "stage_4": {"cd_range": (0, 1600), "lw": 1e6,     "snr_range": (14, 18)},
    "stage_5": {"cd_range": (0, 1600), "lw": "random", "snr_range": (10, 25)},
}


def apply_phase_noise(signal: np.ndarray, linewidth: float, baud_rate: float) -> np.ndarray:
    """Apply Wiener phase noise via random walk."""
    if linewidth <= 0:
        return signal

    ts = 1.0 / baud_rate
    var = 2 * np.pi * linewidth * ts
    deltas = np.random.normal(loc=0, scale=np.sqrt(var), size=signal.shape)
    phase = np.cumsum(deltas, axis=-1)

    return signal * np.exp(1j * phase)


def apply_cd(signal: np.ndarray, dispersion_ps_nm: float, baud_rate: float) -> np.ndarray:
    """Apply chromatic dispersion in frequency domain."""
    if dispersion_ps_nm == 0:
        return signal

    n_fft = signal.shape[-1]
    freq = np.fft.fftfreq(n_fft, d=1/baud_rate)

    lambda_0 = 1550e-9
    c = 3e8

    beta2_total = -(dispersion_ps_nm * 1e-12 * (lambda_0**2)) / (2 * np.pi * c)
    omega = 2 * np.pi * freq
    transfer_function = np.exp(1j * 0.5 * beta2_total * (omega**2))

    spectrum = np.fft.fft(signal)
    dispersed_spectrum = spectrum * transfer_function
    return np.fft.ifft(dispersed_spectrum)


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add additive white Gaussian noise."""
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = sig_power / snr_linear

    noise = (np.random.normal(0, np.sqrt(noise_power/2), signal.shape) +
             1j * np.random.normal(0, np.sqrt(noise_power/2), signal.shape))

    return signal + noise


def generate_batch(num_samples, params):
    """Generate a batch of impaired QPSK data for a given stage configuration."""
    ints = np.random.randint(0, 4, (num_samples, SEQ_LEN))
    phase = (2 * ints + 1) * np.pi / 4
    clean_sig = np.exp(1j * phase)

    processed_sig = np.zeros_like(clean_sig, dtype=np.complex64)

    for i in range(num_samples):
        sig_i = clean_sig[i]

        cd_min, cd_max = params["cd_range"]
        cd_val = np.random.uniform(cd_min, cd_max)
        sig_i = apply_cd(sig_i, cd_val, BAUD_RATE)

        lw = params["lw"]
        if lw == "random":
            lw = np.random.uniform(0, 2e6)
        sig_i = apply_phase_noise(sig_i, float(lw), BAUD_RATE)

        snr_min, snr_max = params["snr_range"]
        snr = np.random.uniform(snr_min, snr_max)
        sig_i = add_awgn(sig_i, snr)

        processed_sig[i] = sig_i

    X = np.stack([processed_sig.real, processed_sig.imag], axis=1).astype(np.float32)
    y = ints.astype(np.int64)

    return X, y


def main():
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating curriculum data to {OUTPUT_DIR}\n")

    for stage_name, params in STAGES.items():
        print(f"Processing {stage_name} | LW: {params['lw']} | SNR: {params['snr_range']}")

        for split, num_samples in SAMPLES_PER_STAGE.items():
            print(f"  {split}: {num_samples} samples")

            X, y = generate_batch(num_samples, params)

            np.save(OUTPUT_DIR / f"X_{split}_{stage_name}.npy", X)
            np.save(OUTPUT_DIR / f"y_{split}_{stage_name}.npy", y)

    print("\nComplete.")


if __name__ == "__main__":
    main()