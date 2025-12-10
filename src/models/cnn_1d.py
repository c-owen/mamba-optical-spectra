# cnn_1d.py
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SpectraCNN(nn.Module):
    """
    1D CNN baseline for optical spectra classification.

    Input shape:  [batch_size, 1, num_points]
    Output shape: [batch_size, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(output_size=1),
        )

        self.classifier = nn.Linear(base_channels * 4, num_classes)

        logger.info(
            "Initialized SpectraCNN with base_channels=%d, num_classes=%d",
            base_channels,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [batch_size, in_channels, num_points]

        Returns:
            logits: tensor of shape [batch_size, num_classes]
        """
        x = self.features(x)          # [B, C, 1]
        x = x.squeeze(-1)             # [B, C]
        logits = self.classifier(x)   # [B, num_classes]
        return logits
