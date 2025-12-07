import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

class CoherentDataset(Dataset):
    """
    Dataset for Coherent Optical QPSK data.
    
    Input (X): [2, Seq_Len] -> Channel 0: In-Phase (Real), Channel 1: Quadrature (Imag)
    Target (y): [Seq_Len]   -> Integer indices 0-3 corresponding to QPSK symbols
    """

    def __init__(
        self,
        split: str,
        data_dir: Path | str = "data/coherent",
    ) -> None:
        super().__init__()
        self.split = split
        self.data_dir = Path(data_dir)

        X_path = self.data_dir / f"X_{split}.npy"
        y_path = self.data_dir / f"y_{split}.npy"

        if not X_path.exists():
            raise FileNotFoundError(f"Missing {X_path}")

        logger.info("Loading coherent split '%s'", split)
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

def get_coherent_dataloaders(
    data_dir: Path | str = "data/coherent",
    batch_size: int = 64,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = CoherentDataset(split, data_dir)
        loaders[split] = DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True
        )
        
    return loaders