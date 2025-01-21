
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.euclidean.base.Tranformer.Attentions.DynamicAttention import MultiheadDynamicSelfAttention
from src.models.euclidean.base.Treensformer.TreeAttention import TreeAttention
from src.models.euclidean.base.Treensformer.avg_siblings import avg_siblings
import seaborn as sns
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


    
class TreensformerBlockV4(nn.Module):
    

    def __init__(self, block_size, n_embd, n_head, attn_pdrop, resid_pdrop, T_Threshold=0, n_levels=4):
        """ Constructor for the Block class 
        Args:
            """
        
        super().__init__()
        print("Building TreensformerBlock")
        self.n_head = n_head
        self.tree_attn = TreeAttention(block_size= block_size, n_embd=n_embd, n_head=n_head, 
                                           attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, T_Threshold=T_Threshold)
        self.ln_1 = nn.LayerNorm(n_embd//n_levels)
        self.ln_2 = nn.LayerNorm(n_embd//n_levels)

        self.tree_mlp = TreeMLPV3(n_embd, n_levels)
        
        self.reg_loss = torch.tensor(0.0, requires_grad=True)

    def forward(self, x):
        """ Forward pass for the Block class """
        B, N_PATCHES, N_LEVELS, R = x.size()
        H = int(math.sqrt(N_PATCHES))
        W = int(math.sqrt(N_PATCHES))
        
        C = R*N_LEVELS
        x = x.view(B, N_PATCHES, C)
        
        x = x.view(B, N_PATCHES, N_LEVELS, R)
        
        

        x = x + self.tree_attn(self.ln_1(x))
        x = x.view(B, H, W, N_LEVELS, R)
        
        x = self.equalize_parents(x)
        x = x.view(B, N_PATCHES, C)
        
        x = x.view(B, H, W, N_LEVELS, R)
        
        x = x + self.tree_mlp(self.ln_2(x))
        
        x = x.view(B, N_PATCHES, N_LEVELS, R)

        return x
    
    
    def equalize_parents(self, x):
        B, H, W, N_LEVELS, R = x.size()
        h_summary_size  = 2
        w_summary_size = 2
        for i in range(N_LEVELS):
            x[:, :, :, i, :] = avg_siblings(x[:, :, :, i, :], sibling_order=i, h_summary_size=h_summary_size, w_summary_size=w_summary_size)
        return x
    


    
class TreeMLP(nn.Module):
    def __init__(self, n_embd, n_levels):
        super().__init__()
        self.n_embd = n_embd
        self.n_levels = n_levels
        
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            NewGELU(),
            nn.Linear(n_embd * 4, n_embd//n_levels),
        )
        
    def forward(self, x):
        B, N_PATCHES, N_LEVELS, R = x.size()
        
        for i in range(N_LEVELS):
            
            input = x[:, :, i:, :].view(B, N_PATCHES, (N_LEVELS-i)*R)
            zeros = torch.zeros(B, N_PATCHES, i*R, device=x.device)
            input = torch.concatenate([zeros, input], dim=2)
            x[:, :, i, :] = self.mlp(input)

        return x
    
class TreeMLPV2(nn.Module):
    def __init__(self, n_embd, n_levels):
        super().__init__()
        self.n_embd = n_embd
        self.n_levels = n_levels
        
        self.mlp_list = nn.ModuleList()
        for i in range(n_levels):
            self.mlp_list.append(nn.Sequential(
                nn.Linear(n_embd*(n_levels-i)//n_levels, n_embd//n_levels * 4),
                NewGELU(),
                nn.Linear(n_embd//n_levels * 4, n_embd//n_levels),
            ))

    def forward(self, x):
        B, N_PATCHES, N_LEVELS, R = x.size()
        outputs = []
        
        for i in range(N_LEVELS):
            input = x[:, :, i:, :].view(B, N_PATCHES, (N_LEVELS - i) * R)
            outputs.append(self.mlp_list[i](input))
        
        # Stack outputs along the level dimension
        x = torch.stack(outputs, dim=2)  # Shape: [B, N_PATCHES, N_LEVELS, R]
        return x
        
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

