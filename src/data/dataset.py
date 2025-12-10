import logging
from pathlib import Path
from typing import Optional, Tuple, Callable, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class OpticalDataset(Dataset):
    """
    Unified PyTorch Dataset for Optical Communication Data (Coherent/QPSK/Spectra).

    Handles loading of both standard datasets (e.g., 'X_train.npy') and 
    curriculum-based datasets (e.g., 'X_train_stage_1.npy').

    Attributes:
        data_dir (Path): Root directory containing the .npy files.
        split (str): Dataset split ('train', 'val', 'test').
        stage (Optional[str]): Curriculum stage name (e.g., 'stage_1'). 
                               If None, loads standard files.
        transform (Optional[Callable]): Optional transform to apply to samples.
        normalize (bool): If True, applies phase-preserving normalization to IQ data.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        stage: Optional[str] = None,
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ) -> None:
        """
        Initializes the OpticalDataset.

        Args:
            data_dir: Path to the directory containing data files.
            split: The subset to load ('train', 'val', 'test').
            stage: The curriculum stage to load. If provided, looks for files 
                   suffixed with _{stage}. Defaults to None.
            transform: A function/transform that takes in an input sample and returns 
                       a transformed version.
            normalize: Whether to apply standard IQ normalization (remove DC, scale power).
                       Defaults to True.
        
        Raises:
            FileNotFoundError: If the expected data files do not exist.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.stage = stage
        self.transform = transform
        self.normalize = normalize

        self.X, self.y = self._load_data()

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal helper to construct filenames and load .npy arrays.
        
        Returns:
            Tuple containing (X_data, y_data).
        """
        # Construct filename suffix: e.g., "_stage_1" or empty string
        suffix = f"_{self.stage}" if self.stage else ""
        
        x_name = f"X_{self.split}{suffix}.npy"
        y_name = f"y_{self.split}{suffix}.npy"
        
        x_path = self.data_dir / x_name
        y_path = self.data_dir / y_name

        if not x_path.exists():
            raise FileNotFoundError(f"Data file not found at: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Label file not found at: {y_path}")

        logger.info(f"Loading {self.split} set | Stage: {self.stage if self.stage else 'Standard'}")
        logger.debug(f"Source: {x_path}")

        # Load data (mmap_mode could be added here for large datasets)
        X = np.load(x_path)
        y = np.load(y_path)
        
        return X, y

    def _normalize_iq(self, x: np.ndarray) -> np.ndarray:
        """
        Applies phase-preserving normalization to complex IQ data.
        
        1. Centers I and Q channels independently (removes DC offset).
        2. Scales by the total average power to preserve the aspect ratio 
           (prevents distorting the constellation shape).

        Args:
            x: Input array of shape [2, Seq_Len].

        Returns:
            Normalized array of shape [2, Seq_Len].
        """
        # Copy to avoid modifying cached memory if using memmap
        x_norm = x.copy()

        # 1. Remove DC Offset (Center signals)
        x_norm[0] -= x_norm[0].mean()
        x_norm[1] -= x_norm[1].mean()

        # 2. Scale Global Power
        # We compute the power of the complex signal (I^2 + Q^2)
        power = np.mean(x_norm[0]**2 + x_norm[1]**2)
        
        # Avoid division by zero
        x_norm /= (np.sqrt(power) + 1e-8)
        
        return x_norm

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample and its label.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple (data_tensor, label_tensor).
        """
        x = self.X[idx]  # [2, L] for IQ data
        y = self.y[idx]  # [L] for labels

        if self.normalize:
            x = self._normalize_iq(x)

        # Apply external transforms (augmentations) if provided
        if self.transform:
            x = self.transform(x)

        # Convert to PyTorch tensors
        # Ensure float32 for data, long (int64) for class labels
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
    """
    Helper function to create a DataLoader for the OpticalDataset.

    Args:
        data_dir: Directory containing the data.
        split: 'train', 'val', or 'test'.
        stage: Curriculum stage name (optional).
        batch_size: Batch size.
        num_workers: Number of subprocesses for data loading.
        shuffle: Whether to shuffle data. If None, defaults to True for 'train', False otherwise.
        normalize: Whether to apply IQ normalization.

    Returns:
        PyTorch DataLoader instance.
    """
    dataset = OpticalDataset(
        data_dir=data_dir,
        split=split,
        stage=stage,
        normalize=normalize
    )

    # Default shuffle logic: Shuffle only on training set unless explicitly specified
    if shuffle is None:
        shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )