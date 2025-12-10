import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class CurriculumDataset(Dataset):
    """
    Dataset for Coherent Optical QPSK data with Curriculum Learning support.
    
    Loads data for specific curriculum stages (e.g., 'stage_1', 'stage_2').
    Expected file naming convention:
       X_{split}_{stage}.npy  (e.g., X_train_stage_1.npy)
       y_{split}_{stage}.npy
       
    Input (X): [2, Seq_Len] -> Channel 0: In-Phase (Real), Channel 1: Quadrature (Imag)
    Target (y): [Seq_Len]   -> Integer indices 0-3 corresponding to QPSK symbols
    """

    def __init__(
        self,
        split: str,
        stage: str,
        data_dir: Path | str = "data/curriculum_w_pn",
    ) -> None:
        super().__init__()
        self.split = split
        self.stage = stage
        self.data_dir = Path(data_dir)

        # Construct filename based on stage
        # e.g., "X_train_stage_1.npy"
        file_suffix = f"_{stage}" if stage else ""
        
        X_path = self.data_dir / f"X_{split}{file_suffix}.npy"
        y_path = self.data_dir / f"y_{split}{file_suffix}.npy"

        if not X_path.exists():
            raise FileNotFoundError(f"Missing {X_path}")

        logger.info("Loading curriculum split '%s' | Stage: '%s'", split, stage)
        # Use mmap_mode='r' if data is huge to avoid RAM spikes, 
        # but for <10GB datasets, standard load is faster.
        self.X = np.load(X_path) # [N, 2, L]
        self.y = np.load(y_path) # [N, L]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get raw complex IQ data
        x = self.X[idx].copy() # Shape [2, L] - Copy ensures we don't modify cached memory
        y = self.y[idx]        # Shape [L]

        # --- PHASE-PRESERVING NORMALIZATION ---
        
        # 1. Center Independent Channels (Remove DC Offset)
        # We must center I and Q separately to remove hardware/simulation DC bias.
        x[0] -= x[0].mean()
        x[1] -= x[1].mean()

        # 2. Scale Globally (Preserve Aspect Ratio)
        # We calculate total power of the complex signal to normalize.
        # If we scaled I and Q independently by their own std, we would turn 
        # a "Donut" into a "Square" and destroy phase information.
        power = np.mean(x[0]**2 + x[1]**2)
        x /= (np.sqrt(power) + 1e-8)

        return torch.from_numpy(x).float(), torch.from_numpy(y).long()

def get_stage_dataloader(
    stage: str,
    split: str = "train",
    data_dir: Path | str = "data/curriculum_w_pn",
    batch_size: int = 64,
    num_workers: int = 0
) -> DataLoader:
    """
    Helper to get a DataLoader for a specific curriculum stage.
    """
    ds = CurriculumDataset(split, stage, data_dir)
    
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )