import torch
import torch.nn as nn
from mamba_ssm import Mamba

class BiMambaBlock(nn.Module):
    """
    A Bi-Directional Mamba Block.
    
    It runs one SSM forward and one SSM backward (by flipping the input),
    then sums their outputs + the residual connection.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        # Branch 1: Forward
        self.fwd_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False
        )
        
        # Branch 2: Backward
        self.bwd_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=False
        )
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        
        # 1. Forward Pass
        out_fwd = self.fwd_mamba(x)
        
        # 2. Backward Pass
        # Flip sequence along dim 1, run Mamba, then flip back
        x_rev = torch.flip(x, [1])
        out_bwd = self.bwd_mamba(x_rev)
        out_bwd = torch.flip(out_bwd, [1])
        
        # 3. Combine (Additive + Residual)
        # We sum both directions and add the original input (residual)
        return self.norm(out_fwd + out_bwd + x)


class CoherentBiMamba(nn.Module):
    """
    Bi-Directional Mamba for High-Dispersion Optical Decoding.
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 2, d_model: int = 64, num_layers: int = 4):
        super().__init__()
        
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # Stack of Bi-Directional Blocks
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [Batch, 2, Seq_Len]
        
        # 1. Transpose: [B, L, 2]
        x = x.transpose(1, 2)
        
        # 2. Project
        x = self.input_proj(x)
        
        # 3. Bi-Mamba Layers
        for layer in self.layers:
            x = layer(x)
            
        # 4. Norm & Head
        x = self.norm(x)
        logits = self.head(x) # [B, L, 4]
        
        # 5. Output Format
        return logits.transpose(1, 2)