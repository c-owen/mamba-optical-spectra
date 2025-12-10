# coherent_models.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class CoherentCNN(nn.Module):
    """
    1D CNN for Sequence-to-Sequence Optical Decoding.
    Acts as a non-linear equalizer (Deep FIR Filter).
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 2, hidden_dim: int = 64, kernel_size: int = 7, num_layers: int = 4):
        super().__init__()
        
        layers = []
        padding = kernel_size // 2
        
        # 1. Input Layer
        layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=padding))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))
        
        # 2. Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            
        # 3. Output Projection
        layers.append(nn.Conv1d(hidden_dim, num_classes, kernel_size=1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MambaBlock(nn.Module):
    """
    Helper block for stability: Norm -> Mamba -> Residual
    """
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
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual


class CoherentMamba(nn.Module):
    """
    Mamba model for Sequence-to-Sequence Optical Decoding.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 2, d_model: int = 64, num_layers: int = 4):
        super().__init__()
        
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # STACK OF RESIDUAL BLOCKS (This was the fix)
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [Batch, 2, Seq_Len]
        
        # 1. Transpose for Mamba: [B, L, 2]
        x = x.transpose(1, 2)
        
        # 2. Project
        x = self.input_proj(x)
        
        # 3. Mamba Layers (Sequence Preserved)
        for layer in self.layers:
            x = layer(x)
            
        # 4. Norm
        x = self.norm(x)
        
        # 5. Classifier Head
        logits = self.head(x) # [B, L, 4]
        
        # 6. Transpose back to [B, 4, L] for CrossEntropyLoss
        return logits.transpose(1, 2)