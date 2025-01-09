
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.euclidean.base.Tranformer.Attentions.DynamicAttention import MultiheadDynamicSelfAttention
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class TreensformerBlock(nn.Module):
    

    def __init__(self, block_size, n_embd, n_head, attn_pdrop, resid_pdrop, T_Threshold=0, tree_mask=None):
        """ Constructor for the Block class """
        
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiheadDynamicSelfAttention(block_size= block_size, n_embd=n_embd, n_head=n_head, 
                                           attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, T_Threshold=T_Threshold, tree_mask=tree_mask)
        self.ln_2 = nn.LayerNorm(n_embd)

        self.mlpf = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            NewGELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop))

    def forward(self, x):
        """ Forward pass for the Block class """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
