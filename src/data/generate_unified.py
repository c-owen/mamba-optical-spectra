import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

# Import our new physics module
import physics as phys

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIGURATIONS ---

# 1. Standard "Hard" Configuration (Like generate_coherent.py)
CONFIG_STANDARD = {
    "stages": {
        "standard": {
            "cd_range": (0, 1600),   # 0 to 1600 ps/nm
            "lw_range": (0, 100e3),  # 0 to 100 kHz
            "snr_range": (10, 20)    # 10 to 20 dB
        }
    }
}

# 2. Curriculum Configuration (Like generate_curriculum_w_pn.py)
CONFIG_CURRICULUM = {
    "stages": {
        "stage_1": {"cd_range": (0, 1600), "lw_range": (0, 0),       "snr_range": (20, 25)},
        "stage_2": {"cd_range": (0, 1600), "lw_range": (100e3, 100e3), "snr_range": (18, 22)},
        "stage_3": {"cd_range": (0, 1600), "lw_range": (500e3, 500e3), "snr_range": (16, 20)},
        "stage_4": {"cd_range": (0, 1600), "lw_range": (1e6, 1e6),     "snr_range": (14, 18)},
        "stage_5": {"cd_range": (0, 1600), "lw_range": (0, 2e6),       "snr_range": (10, 25)},
    }
}

class CoherentGenerator:
    """
    Unified Generator for Coherent Optical QPSK data.
    Can generate standard datasets OR curriculum-staged datasets based on config.
    """
    def __init__(self, output_dir: str, seq_len: int = 2048, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_batch(
        self, 
        num_samples: int, 
        params: Dict[str, Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a single batch of data based on impairment ranges.
        """
        # 1. Generate Clean QPSK
        tx_sig, labels = phys.generate_qpsk_symbols(num_samples, self.seq_len)
        rx_sig = np.zeros_like(tx_sig, dtype=np.complex64)

        # 2. Apply Physics Per Sample
        # (Vectorizing this is possible but complex due to varying params per sample)
        for i in range(num_samples):
            sig_i = tx_sig[i]

            # Sample impairments from ranges
            cd = self.rng.uniform(*params["cd_range"])
            
            # Handle Fixed vs Random Linewidth
            lw_min, lw_max = params["lw_range"]
            lw = self.rng.uniform(lw_min, lw_max)
            
            snr = self.rng.uniform(*params["snr_range"])

            # Apply Chain
            sig_i = phys.apply_chromatic_dispersion(sig_i, cd)
            sig_i = phys.apply_phase_noise(sig_i, lw)
            sig_i = phys.apply_awgn(sig_i, snr)
            
            rx_sig[i] = sig_i

        # 3. Format Output [Batch, 2, Seq_Len] (Real, Imag)
        X = np.stack([rx_sig.real, rx_sig.imag], axis=1).astype(np.float32)
        y = labels
        
        return X, y

    def run(self, config: Dict, samples_per_split: Dict[str, int]):
        """
        Main execution loop. Iterates through Stages -> Splits.
        """
        stages = config["stages"]
        
        for stage_name, params in stages.items():
            logger.info(f"--- Processing {stage_name} ---")
            logger.info(f"Params: {params}")

            for split, n_samples in samples_per_split.items():
                logger.info(f"Generating {split} set ({n_samples} samples)...")
                
                X, y = self.generate_batch(n_samples, params)
                
                # File Naming:
                # If "standard" stage (single stage), we omit the stage suffix for cleaner filenames
                # e.g., X_train.npy instead of X_train_standard.npy
                suffix = f"_{stage_name}" if len(stages) > 1 else ""
                
                x_path = self.output_dir / f"X_{split}{suffix}.npy"
                y_path = self.output_dir / f"y_{split}{suffix}.npy"
                
                np.save(x_path, X)
                np.save(y_path, y)
                logger.info(f"Saved to {x_path}")

def main():
    parser = argparse.ArgumentParser(description="Unified Coherent Optical Data Generator")
    parser.add_argument("--mode", choices=["standard", "curriculum"], default="standard", 
                        help="Generation mode: 'standard' (flat random) or 'curriculum' (staged)")
    parser.add_argument("--out", type=str, default="data/generated", help="Output directory")
    parser.add_argument("--n-train", type=int, default=10000, help="Train samples")
    parser.add_argument("--n-val", type=int, default=1000, help="Val samples")
    parser.add_argument("--n-test", type=int, default=1000, help="Test samples")
    
    args = parser.parse_args()

    # Select Configuration
    if args.mode == "curriculum":
        config = CONFIG_CURRICULUM
        out_dir = Path(args.out) / "curriculum"
    else:
        config = CONFIG_STANDARD
        out_dir = Path(args.out) / "standard"

    # Define Splits
    samples = {
        "train": args.n_train,
        "val": args.n_val,
        "test": args.n_test
    }

    # Run Generator
    gen = CoherentGenerator(output_dir=out_dir)
    gen.run(config, samples)

if __name__ == "__main__":
    main()