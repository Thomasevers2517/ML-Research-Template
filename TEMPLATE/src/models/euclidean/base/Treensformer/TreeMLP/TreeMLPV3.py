import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from src.models.euclidean.base.Treensformer.avg_siblings import avg_siblings


class TreeMLPV3(nn.Module):
    def __init__(self, n_embd, n_levels):
        super().__init__()
        self.n_embd = n_embd
        self.n_levels = n_levels
        # MLP: in_features = n_embd, out_features = n_embd//n_levels
        # Because each 'level' embedding has dimension R = (n_embd // n_levels)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            NewGELU(),
            nn.Linear(n_embd * 4, n_embd // n_levels),
        )

    def forward(self, x):
        """
        x shape: (B, H, W, N_LEVELS, R)
          - Index=0 => 'children' level (lowest in the hierarchy).
          - Index=N_LEVELS-1 => 'root' level (highest).

        The idea:
          - For each level i in [0..N_LEVELS-1],
              * If i==0 => we are updating the "child" embedding
                           no averaging from 'below' because there's no lower level.
              * If i>0  => we gather the block of lower levels [0..i-1], average them
                           (since they are the true children). Then we combine them
                           with the block [i..end], flatten, and feed into MLP.
          - We'll build the updated embeddings for all levels in an out-of-place
            list, then cat them along dim=3 to produce shape (B, H, W, N_LEVELS, R).
        """
        B, H, W, N_LEVELS, R = x.size()

        # We'll store updated embeddings for each level i in out_slices
        out_slices = []

        for i in range(N_LEVELS):
            # 'i' is the level we're updating:
            #   i=0 => 'lowest' (children),
            #   i=N_LEVELS-1 => 'root'.

            # The "upper" block is x[:, :, :, i:, :] => shape (B,H,W, N_LEVELS-i, R).
            upper_block = x[:, :, :, i:, :]  # everything from 'i' up to root

            if i == 0:
                # The bottommost children: no "lower levels" exist.
                # So we just flatten (N_LEVELS * R) => pass MLP
                merged = upper_block.reshape(B, H, W, self.n_embd)
            else:
                # We have 'i' lower levels => index [0..i-1]
                # We'll average them because they are the "children" we need summarized
                lower_block = x[:, :, :, :i, :].clone()

                # For each of these i levels, do your "avg_siblings" or relevant averaging
                for child_idx in range(i):
                    lower_block[:, :, :, child_idx, :] = avg_siblings(
                        lower_block[:, :, :, child_idx, :],
                        sibling_order=i,  # or child_idx, depending on your usage
                        h_summary_size=2,
                        w_summary_size=2
                    )

                # Concat the lower block + upper block across the level dimension=3
                merged_full = torch.cat([lower_block, upper_block], dim=3)  # shape (B,H,W, N_LEVELS, R)
                merged = merged_full.reshape(B, H, W, self.n_embd)

            # Pass the flattened (N_LEVELS*R) data into MLP => shape (B,H,W,R)
            new_slice = self.mlp(merged)  # out => (B,H,W, R)

            # Store it (dim=3 => one slice for level i)
            new_slice = new_slice.unsqueeze(3)  # shape => (B,H,W,1,R)
            out_slices.append(new_slice)

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
