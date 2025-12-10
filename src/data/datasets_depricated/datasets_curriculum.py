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
        data_dir: Path | str = "data/curriculum",
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
        self.X = np.load(X_path) # [N, 2, L]
        self.y = np.load(y_path) # [N, L]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get raw complex IQ data
        x = self.X[idx] # Shape [2, L]
        y = self.y[idx] # Shape [L]

        # Standardize (Simple Z-score normalization per sample)
        # We normalize Real and Imag parts together to preserve relative phase info
        mean = x.mean()
        std = x.std() + 1e-8
        x = (x - mean) / std

        return torch.from_numpy(x).float(), torch.from_numpy(y).long()

def get_stage_dataloader(
    stage: str,
    split: str = "train",
    data_dir: Path | str = "data/curriculum",
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