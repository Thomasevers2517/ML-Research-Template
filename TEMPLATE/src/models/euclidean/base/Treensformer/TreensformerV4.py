
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.euclidean.base.Tranformer.Attentions.DynamicAttention import MultiheadDynamicSelfAttention
from src.models.euclidean.base.Treensformer.TreeAttention import TreeAttention
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
        
        x = x.view(B, N_PATCHES, N_LEVELS, R)
        
        
        x = x + self.tree_mlp(self.ln_2(x))
        

        return x
    
    
    def equalize_parents(self, x):
        B, H, W, N_LEVELS, R = x.size()
        h_summary_size  = 2
        w_summary_size = 2
        for i in range(N_LEVELS):
            h_num_sum = h_summary_size**i
            w_num_sum = w_summary_size**i
            h_n_splits = H//h_num_sum
            w_n_splits = W//w_num_sum
            assert isinstance(h_n_splits, int), "h_n_splits must be an integer"
            assert isinstance(w_n_splits, int), "w_n_splits must be an integer"
            h_n_splits = int(h_n_splits)
            w_n_splits = int(w_n_splits)
            
            x_temp =  x[:, :, :, i, :].reshape(B, h_n_splits,h_num_sum, w_n_splits, w_num_sum, R)
            x_temp =  x_temp.mean(dim=[2, 4]) # B, h_n_splits, w_n_splits, R
            x_temp = torch.repeat_interleave(x_temp, repeats=h_num_sum, dim=1)
            x[:, :, :, i, :] = torch.repeat_interleave(x_temp, repeats=w_num_sum, dim=2)
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
        
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            NewGELU(),
            nn.Linear(n_embd * 4, n_embd//n_levels),
        )
        
    def forward(self, x):
        B, N_PATCHES, N_LEVELS, R = x.size()
        
        for i in range(N_LEVELS):
            
            input = x[:, :, i:, :].view(B, N_PATCHES, (N_LEVELS-i)*R)
            avgd = x[:, :, :i, :].view(B, N_PATCHES, i*R)
            avgd = avgd.mean(dim=1, keepdim=True)
            avgd = avgd.repeat(1, N_PATCHES, 1)
            input = torch.concatenate([avgd, input], dim=2)
            x[:, :, i, :] = self.mlp(input)

        return x