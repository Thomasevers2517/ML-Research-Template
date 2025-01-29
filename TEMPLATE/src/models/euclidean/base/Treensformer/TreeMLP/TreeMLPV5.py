import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from src.models.euclidean.base.Treensformer.avg_siblings import avg_siblings


class TreeMLPV5(nn.Module):
    def __init__(self, n_embd, n_levels, mlp_multiplier=4):
        super().__init__()
        self.n_embd = n_embd
        self.n_levels = n_levels
        # MLP: in_features = n_embd, out_features = n_embd//n_levels
        # Because each 'level' embedding has dimension R = (n_embd // n_levels)
        self.mlp = nn.Sequential(
            nn.Linear((n_embd // n_levels)*2, (n_embd // n_levels) * mlp_multiplier),
            NewGELU(),
            nn.Linear((n_embd // n_levels) * mlp_multiplier, (n_embd // n_levels)*2),
        )

    def forward(self, x):
        B, H, W, N_LEVELS, R = x.size()

        # We'll store updated embeddings for each level i in out_slices
        out_slices = [torch.zeros(B, H, W, R, device=x.device, dtype=torch.float32) for _ in range(N_LEVELS)]

        for i in range(N_LEVELS-1):

            
            merged = torch.cat([x[:, :, :, i, :] , x[:, :, :, i+1, :]], dim=3)
            
            new_slice = self.mlp(merged)
            new_slice = new_slice.unsqueeze(3)  # shape => (B,H,W,1,R*2)
            new_slice = new_slice.view(B, H, W, 2, R)  # shape => (B,H,W,2,R)
            out_slices[i] = out_slices[i] + new_slice[:, :, :, 0, :]
            
            out_slices[i+1] = out_slices[i+1] + avg_siblings(new_slice[:, :, :, 1, :], (i+1), h_summary_size=2, w_summary_size=2)
            if torch.isnan(new_slice).any():
                print(f"NaN detected in new_slice at level {i}")
            if torch.isnan(out_slices[i]).any():
                print(f"NaN detected in out_slices[{i}] before addition")
            if torch.isnan(out_slices[i+1]).any():
                print(f"NaN detected in out_slices[{i+1}] before addition")
                
        for i in range(N_LEVELS):
            out_slices[i] = out_slices[i].unsqueeze(3) # shape => (B,H,W,1,R)

        # Finally, cat along dim=3 => shape (B,H,W,N_LEVELS,R)
        out = torch.cat(out_slices, dim=3)
        return out

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
