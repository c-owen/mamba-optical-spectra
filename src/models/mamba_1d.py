import logging

import torch
import torch.nn as nn
from mamba_ssm import Mamba

logger = logging.getLogger(__name__)


class MambaBlock(nn.Module):
    """Pre-norm Mamba block with residual connection."""
    
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False,
        )
        
    def forward(self, x):
        return x + self.mamba(self.norm(x))


class SpectraMamba(nn.Module):
    """Mamba model for optical spectra classification."""
    
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 2,
        d_state: int = 8,
        d_conv: int = 2,
        expand: int = 1,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model

        self.input_proj = nn.Linear(in_channels, d_model)

        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        logger.info(
            "Initialized SpectraMamba with d_model=%d, num_layers=%d, d_state=%d, d_conv=%d, expand=%d, num_classes=%d",
            d_model, num_layers, d_state, d_conv, expand, num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, L] -> logits: [B, num_classes]
        x = self.input_proj(x.transpose(1, 2))

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x).mean(dim=1)
        return self.head(x)