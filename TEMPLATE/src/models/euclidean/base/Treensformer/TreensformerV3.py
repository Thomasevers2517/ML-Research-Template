
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.euclidean.base.Tranformer.Attentions.DynamicAttention import MultiheadDynamicSelfAttention
import seaborn as sns
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


    
class TreensformerBlockV3(nn.Module):
    

    def __init__(self, block_size, n_embd, n_head, attn_pdrop, resid_pdrop, T_Threshold=0):
        """ Constructor for the Block class 
        Args:
            """
        
        super().__init__()
        print("Building TreensformerBlock")

        self.attn = MultiheadDynamicSelfAttention(block_size= block_size, n_embd=n_embd, n_head=n_head, 
                                           attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, T_Threshold=T_Threshold)
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

        self.mlpf = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            NewGELU(),
            nn.Linear(n_embd * 4, n_embd),
        )
        
        self.reg_loss = torch.tensor(0.0, requires_grad=True)

    def forward(self, x):
        """ Forward pass for the Block class """
        B, N_PATCHES, N_LEVELS, R = x.size()
        C = R*N_LEVELS
        x = x.view(B, N_PATCHES, C)
        x = x + self.attn(self.ln_1(x))
        x = x.view(B, H, W, N_LEVELS, R)
        x = self.equalize_parents(x, H, W)
        x = x.view(B, N_PATCHES, C)

        x = x + self.mlpf(self.ln_2(x))
        x = x.view(B, N_PATCHES, N_LEVELS, R)
        H = int(math.sqrt(N_PATCHES))
        W = int(math.sqrt(N_PATCHES))
        
        x = x.view(B, H, W, N_LEVELS, R)
        
        x = self.equalize_parents(x, H, W)
            
        x = x.view(B, N_PATCHES, N_LEVELS, R)
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