import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SimpleMambaBlock(nn.Module):
    """Lightweight Mamba-inspired block with depthwise conv, gating, and residual."""

    def __init__(self, d_model: int, kernel_size: int = 5) -> None:
        super().__init__()

        self.d_model = d_model

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )

        self.in_proj = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.out_proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        residual = x

        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv(x)

        u, v = torch.chunk(self.in_proj(x), chunks=2, dim=1)
        x = self.out_proj(torch.sigmoid(v) * u)

        return x + residual


class SpectraMamba(nn.Module):
    """Mamba-style model for 1D optical spectra classification."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        self.in_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)

        self.layers = nn.ModuleList([
            SimpleMambaBlock(d_model=d_model, kernel_size=kernel_size)
            for _ in range(num_layers)
        ])

        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Linear(d_model, num_classes)

        logger.info(
            "Initialized SpectraMamba with d_model=%d, num_layers=%d, num_classes=%d",
            d_model, num_layers, num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, L] -> logits: [B, num_classes]
        x = self.in_proj(x)

        for layer in self.layers:
            x = layer(x)

        return self.classifier(self.pool(x).squeeze(-1))