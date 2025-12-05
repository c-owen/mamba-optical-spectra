# cd mamba-optical-spectra
# source venv/bin/activate  # if not already active
# 
# python src/data/generate_spectra.py \
#   --output-dir data/processed \
#   --num-samples-per-class 2000 \
#   --num-points 512 \
#   --train-frac 0.7 \
#   --val-frac 0.15 \
#   --seed 42 \
#   -vv

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


CLASS_NAMES = [
    "stable_single_mode",
    "mode_hop",
    "clipped",
    "high_noise",
    "multi_mode",
]


def setup_logging(verbosity: int = 1) -> None:
    """Configure the logging module."""
    level = logging.INFO if verbosity <= 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.debug("Logging initialized with level %s", logging.getLevelName(level))


def gaussian(x: np.ndarray, mu: float, sigma: float, amplitude: float) -> np.ndarray:
    """Simple Gaussian function."""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_wavelength_axis(
    num_points: int, lam_min: float = 1550.0, lam_max: float = 1551.0
) -> np.ndarray:
    """Generate a wavelength axis in nanometers."""
    x = np.linspace(lam_min, lam_max, num_points, dtype=np.float32)
    logging.debug(
        "Generated wavelength axis from %.4f nm to %.4f nm with %d points",
        lam_min,
        lam_max,
        num_points,
    )
    return x


def generate_stable_single_mode(
    lam: np.ndarray,
    rng: np.random.Generator,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Single Gaussian peak with low noise."""
    center = rng.uniform(lam.min() + 0.2 * (np.ptp(lam)
), lam.max() - 0.2 * (np.ptp(lam)
))
    sigma = rng.uniform(0.02, 0.05) * np.ptp(lam)
    amplitude = rng.uniform(0.8, 1.0)

    signal = gaussian(lam, center, sigma, amplitude)
    noise = rng.normal(0.0, noise_level, size=lam.shape)
    spectrum = signal + noise
    logging.debug(
        "Generated stable_single_mode: center=%.4f, sigma=%.4f, amplitude=%.3f",
        center,
        sigma,
        amplitude,
    )
    return spectrum.astype(np.float32)


def generate_mode_hop(
    lam: np.ndarray,
    rng: np.random.Generator,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Spectrum exhibiting a mode hop between two peaks."""
    span = np.ptp(lam)
    center1 = rng.uniform(lam.min() + 0.1 * span, lam.min() + 0.4 * span)
    center2 = rng.uniform(lam.min() + 0.6 * span, lam.min() + 0.9 * span)
    sigma1 = rng.uniform(0.02, 0.05) * span
    sigma2 = rng.uniform(0.02, 0.05) * span
    amplitude1 = rng.uniform(0.7, 1.0)
    amplitude2 = rng.uniform(0.7, 1.0)

    midpoint_index = lam.size // 2
    signal1 = gaussian(lam, center1, sigma1, amplitude1)
    signal2 = gaussian(lam, center2, sigma2, amplitude2)

    # First half from mode 1, second half from mode 2
    signal = np.concatenate([signal1[:midpoint_index], signal2[midpoint_index:]])
    noise = rng.normal(0.0, noise_level, size=lam.shape)
    spectrum = signal + noise
    logging.debug(
        "Generated mode_hop: center1=%.4f, center2=%.4f, sigma1=%.4f, sigma2=%.4f",
        center1,
        center2,
        sigma1,
        sigma2,
    )
    return spectrum.astype(np.float32)


def generate_clipped(
    lam: np.ndarray,
    rng: np.random.Generator,
    noise_level: float = 0.01,
    clip_fraction: float = 0.8,
) -> np.ndarray:
    """Single peak with hard clipping at a fraction of its peak amplitude."""
    base = generate_stable_single_mode(lam, rng, noise_level=noise_level / 2.0)
    clip_level = clip_fraction * base.max()
    clipped = np.clip(base, None, clip_level)
    noise = rng.normal(0.0, noise_level, size=lam.shape)
    spectrum = clipped + noise
    logging.debug(
        "Generated clipped spectrum: clip_level=%.3f (%.0f%% of original max)",
        clip_level,
        clip_fraction * 100.0,
    )
    return spectrum.astype(np.float32)


def generate_high_noise(
    lam: np.ndarray,
    rng: np.random.Generator,
    base_noise_level: float = 0.01,
    extra_noise_level: float = 0.05,
) -> np.ndarray:
    """Single mode with significantly elevated noise floor."""
    base = generate_stable_single_mode(lam, rng, noise_level=base_noise_level)
    extra_noise = rng.normal(0.0, extra_noise_level, size=lam.shape)
    spectrum = base + extra_noise
    logging.debug(
        "Generated high_noise spectrum: base_noise=%.3f, extra_noise=%.3f",
        base_noise_level,
        extra_noise_level,
    )
    return spectrum.astype(np.float32)


def generate_multi_mode(
    lam: np.ndarray,
    rng: np.random.Generator,
    num_modes_range: Tuple[int, int] = (2, 4),
    noise_level: float = 0.01,
) -> np.ndarray:
    """Sum of multiple Gaussian modes plus noise."""
    span = np.ptp(lam)
    num_modes = rng.integers(num_modes_range[0], num_modes_range[1] + 1)

    signal = np.zeros_like(lam, dtype=np.float32)
    for i in range(num_modes):
        center = rng.uniform(lam.min() + 0.1 * span, lam.max() - 0.1 * span)
        sigma = rng.uniform(0.01, 0.04) * span
        amplitude = rng.uniform(0.2, 0.8)
        mode = gaussian(lam, center, sigma, amplitude)
        signal += mode.astype(np.float32)
        logging.debug(
            "Multi-mode component %d: center=%.4f, sigma=%.4f, amplitude=%.3f",
            i,
            center,
            sigma,
            amplitude,
        )

    noise = rng.normal(0.0, noise_level, size=lam.shape)
    spectrum = signal + noise
    logging.debug("Generated multi_mode spectrum with %d modes", num_modes)
    return spectrum.astype(np.float32)


def generate_samples_for_class(
    class_name: str,
    lam: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate spectra for a given class name."""
    logging.info(
        "Generating %d samples for class '%s'",
        n_samples,
        class_name,
    )

    generator_map = {
        "stable_single_mode": generate_stable_single_mode,
        "mode_hop": generate_mode_hop,
        "clipped": generate_clipped,
        "high_noise": generate_high_noise,
        "multi_mode": generate_multi_mode,
    }

    if class_name not in generator_map:
        raise ValueError(f"Unknown class name: {class_name}")

    spectra = []
    for i in range(n_samples):
        if i % max(1, n_samples // 5) == 0:
            logging.info(
                "Class '%s': generating sample %d / %d",
                class_name,
                i + 1,
                n_samples,
            )
        spectrum = generator_map[class_name](lam, rng)
        spectra.append(spectrum)

    return np.stack(spectra, axis=0)


def build_dataset(
    num_samples_per_class: int,
    num_points: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build full dataset X, y, and wavelength axis."""
    logging.info(
        "Building dataset with %d samples per class, %d points per spectrum, seed=%d",
        num_samples_per_class,
        num_points,
        seed,
    )
    rng = np.random.default_rng(seed)
    lam = generate_wavelength_axis(num_points=num_points)

    all_spectra = []
    all_labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_spectra = generate_samples_for_class(
            class_name=class_name,
            lam=lam,
            n_samples=num_samples_per_class,
            rng=rng,
        )
        labels = np.full(
            shape=(num_samples_per_class,),
            fill_value=class_idx,
            dtype=np.int64,
        )
        all_spectra.append(class_spectra)
        all_labels.append(labels)
        logging.info(
            "Finished generating class '%s' (index=%d)",
            class_name,
            class_idx,
        )

    X = np.concatenate(all_spectra, axis=0)
    y = np.concatenate(all_labels, axis=0)

    logging.info("Combined dataset shape: X=%s, y=%s", X.shape, y.shape)
    return X, y, lam


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Split dataset into train/val/test subsets."""
    logging.info(
        "Splitting dataset: train_frac=%.2f, val_frac=%.2f, seed=%d",
        train_frac,
        val_frac,
        seed,
    )

    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_end = int(train_frac * n_samples)
    val_end = train_end + int(val_frac * n_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    splits = {
        "train": {"X": X[train_idx], "y": y[train_idx]},
        "val": {"X": X[val_idx], "y": y[val_idx]},
        "test": {"X": X[test_idx], "y": y[test_idx]},
    }

    for split_name, split_data in splits.items():
        logging.info(
            "%s split: X=%s, y=%s",
            split_name,
            split_data["X"].shape,
            split_data["y"].shape,
        )

    return splits


def save_dataset(
    splits: Dict[str, Dict[str, np.ndarray]],
    lam: np.ndarray,
    output_dir: Path,
) -> None:
    """Save dataset splits and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving dataset to '%s'", output_dir)

    for split_name, split_data in splits.items():
        X_path = output_dir / f"X_{split_name}.npy"
        y_path = output_dir / f"y_{split_name}.npy"
        np.save(X_path, split_data["X"])
        np.save(y_path, split_data["y"])
        logging.info(
            "Saved %s split: X -> %s, y -> %s",
            split_name,
            X_path,
            y_path,
        )

    lam_path = output_dir / "wavelength_axis.npy"
    np.save(lam_path, lam)
    logging.info("Saved wavelength axis to %s", lam_path)

    class_mapping = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    meta = {
        "class_index_to_name": class_mapping,
        "num_classes": len(CLASS_NAMES),
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Saved metadata to %s", meta_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic optical spectra dataset for Mamba/CNN classification.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory where the dataset .npy files will be saved.",
    )
    parser.add_argument(
        "--num-samples-per-class",
        type=int,
        default=2000,
        help="Number of spectra to generate per class.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=512,
        help="Number of wavelength points per spectrum.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of data to use for training.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of data to use for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase verbosity (use -v for INFO, -vv for DEBUG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(verbosity=args.verbose)

    logging.info("Starting dataset generation.")
    logging.info("Configuration: %s", vars(args))

    output_dir = Path(args.output_dir)

    X, y, lam = build_dataset(
        num_samples_per_class=args.num_samples_per_class,
        num_points=args.num_points,
        seed=args.seed,
    )

    splits = train_val_test_split(
        X,
        y,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    save_dataset(
        splits=splits,
        lam=lam,
        output_dir=output_dir,
    )

    logging.info("Dataset generation completed successfully.")


if __name__ == "__main__":
    main()
