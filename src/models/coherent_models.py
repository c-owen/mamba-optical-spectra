import torch
import torch.nn as nn
from mamba_ssm import Mamba


class CoherentCNN(nn.Module):
    """1D CNN for sequence-to-sequence optical decoding (deep FIR filter)."""
    
    def __init__(self, num_classes: int = 4, in_channels: int = 2, hidden_dim: int = 64, kernel_size: int = 7, num_layers: int = 4):
        super().__init__()
        
        layers = []
        padding = kernel_size // 2
        
        layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=padding))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            
        layers.append(nn.Conv1d(hidden_dim, num_classes, kernel_size=1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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


class CoherentMamba(nn.Module):
    """Mamba model for sequence-to-sequence optical decoding."""
    
    def __init__(self, num_classes: int = 4, in_channels: int = 2, d_model: int = 64, num_layers: int = 4):
        super().__init__()
        
        self.input_proj = nn.Linear(in_channels, d_model)
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, 2, L] -> logits: [B, num_classes, L]
        x = self.input_proj(x.transpose(1, 2))
        
        for layer in self.layers:
            x = layer(x)
            
        return self.head(self.norm(x)).transpose(1, 2)