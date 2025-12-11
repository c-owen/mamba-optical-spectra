import logging
from pathlib import Path
from typing import Optional, Tuple, Callable, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class OpticalDataset(Dataset):
    """PyTorch Dataset for optical communication data with optional curriculum staging."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        stage: Optional[str] = None,
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.stage = stage
        self.transform = transform
        self.normalize = normalize
        self.X, self.y = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        suffix = f"_{self.stage}" if self.stage else ""

        x_name = f"X_{self.split}{suffix}.npy"
        y_name = f"y_{self.split}{suffix}.npy"

        x_path = self.data_dir / x_name
        y_path = self.data_dir / y_name

        if not x_path.exists():
            raise FileNotFoundError(f"Data file not found: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Label file not found: {y_path}")

        logger.info(f"Loading {self.split} | Stage: {self.stage or 'standard'}")
        logger.debug(f"Source: {x_path}")

        X = np.load(x_path)
        y = np.load(y_path)

        return X, y

    def _normalize_iq(self, x: np.ndarray) -> np.ndarray:
        """Phase-preserving normalization: remove DC offset and scale by total power."""
        x_norm = x.copy()

        x_norm[0] -= x_norm[0].mean()
        x_norm[1] -= x_norm[1].mean()

        power = np.mean(x_norm[0]**2 + x_norm[1]**2)
        x_norm /= (np.sqrt(power) + 1e-8)

        return x_norm

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]

        if self.normalize:
            x = self._normalize_iq(x)

        if self.transform:
            x = self.transform(x)

        return torch.from_numpy(x).float(), torch.from_numpy(y).long()


def get_dataloader(
    data_dir: Union[str, Path],
    split: str = "train",
    stage: Optional[str] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    normalize: bool = True
) -> DataLoader:
    """Create a DataLoader for OpticalDataset."""
    dataset = OpticalDataset(
        data_dir=data_dir,
        split=split,
        stage=stage,
        normalize=normalize
    )

    if shuffle is None:
        shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )