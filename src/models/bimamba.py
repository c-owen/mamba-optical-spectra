import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BiMambaBlock(nn.Module):
    """Bi-directional Mamba block with residual connection."""
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        self.fwd_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False
        )
        
        self.bwd_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False
        )
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, d_model]
        out_fwd = self.fwd_mamba(x)
        
        x_rev = torch.flip(x, [1])
        out_bwd = torch.flip(self.bwd_mamba(x_rev), [1])
        
        return self.norm(out_fwd + out_bwd + x)


class CoherentBiMamba(nn.Module):
    """Bi-directional Mamba for high-dispersion optical decoding."""
    
    def __init__(self, num_classes: int = 4, in_channels: int = 2, d_model: int = 64, num_layers: int = 4):
        super().__init__()
        
        self.input_proj = nn.Linear(in_channels, d_model)
        
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, 2, L] -> logits: [B, num_classes, L]
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        return self.head(x).transpose(1, 2)