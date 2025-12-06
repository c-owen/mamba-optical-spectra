import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SimpleMambaBlock(nn.Module):
    """
    A lightweight, Mamba-inspired 1D sequence block for spectra.

    This is NOT a full reproduction of the original Mamba architecture, but
    captures some key ideas:
      - sequence mixing via depthwise Conv1d
      - gating
      - residual connection
      - normalization over channels

    Input:  [B, C, L]
    Output: [B, C, L]
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Depthwise conv for sequence mixing (per-channel)
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )

        # Channel mixing + gating
        self.in_proj = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.out_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Normalize over channels (LayerNorm expects [B, L, C])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]

        Returns:
            [B, C, L]
        """
        residual = x

        # Normalize across channels at each position
        x_ln = x.transpose(1, 2)          # [B, L, C]
        x_ln = self.norm(x_ln)
        x_ln = x_ln.transpose(1, 2)       # [B, C, L]

        # Depthwise conv mixes along sequence dimension
        x_conv = self.conv(x_ln)          # [B, C, L]

        # Gated channel mixing
        uv = self.in_proj(x_conv)         # [B, 2C, L]
        u, v = torch.chunk(uv, chunks=2, dim=1)
        x_gated = torch.sigmoid(v) * u    # [B, C, L]

        # Output projection and residual
        x_out = self.out_proj(x_gated)    # [B, C, L]
        return x_out + residual


class SpectraMamba(nn.Module):
    """
    Mamba-style model for 1D optical spectra classification.

    Input:  [B, 1, L]
    Output: [B, num_classes]
    """

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

        self.layers = nn.ModuleList(
            [
                SimpleMambaBlock(
                    d_model=d_model,
                    kernel_size=kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

        # Global pooling over wavelength axis
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.classifier = nn.Linear(d_model, num_classes)

        logger.info(
            "Initialized SpectraMamba with d_model=%d, num_layers=%d, num_classes=%d",
            d_model,
            num_layers,
            num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, L]

        Returns:
            logits: [B, num_classes]
        """
        x = self.in_proj(x)  # [B, d_model, L]

        for layer in self.layers:
            x = layer(x)      # [B, d_model, L]

        x = self.pool(x)      # [B, d_model, 1]
        x = x.squeeze(-1)     # [B, d_model]
        logits = self.classifier(x)
        return logits
