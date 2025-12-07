import logging
import torch
import torch.nn as nn
from mamba_ssm import Mamba

logger = logging.getLogger(__name__)

class MambaBlock(nn.Module):
    """
    Helper block that applies:
    1. LayerNorm
    2. Mamba Mixer
    3. Residual Connection
    """
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_fast_path=False, # Keep false if debugging, True for speed
            )
        
    def forward(self, x):
        # Save input for residual
        residual = x
        
        # 1. Normalize
        x = self.norm(x)
        
        # 2. Mamba Mixer
        x = self.mamba(x)
        
        # 3. Add Residual
        return x + residual

class SpectraMamba(nn.Module):
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

        # Project scalar intensity -> d_model feature space
        self.input_proj = nn.Linear(in_channels, d_model)

        # Stack of Mamba BLOCKS (now with residuals and norms)
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
                for _ in range(num_layers)
            ]
        )

        # Final Normalization (standard practice before the head)
        self.norm = nn.LayerNorm(d_model)

        # Final classifier
        self.head = nn.Linear(d_model, num_classes)

        logger.info(
            "Initialized REAL SpectraMamba (mamba-ssm) with "
            "d_model=%d, num_layers=%d, d_state=%d, d_conv=%d, expand=%d, num_classes=%d",
            d_model, num_layers, d_state, d_conv, expand, num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 1, L] -> [B, L, 1]
        x = x.transpose(1, 2)

        # [B, L, 1] -> [B, L, d_model]
        x = self.input_proj(x)

        # Pass through stacked Mamba Blocks
        for layer in self.layers:
            x = layer(x)   # Now includes norm and residual

        # Final Norm
        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classifier
        logits = self.head(x)
        return logits