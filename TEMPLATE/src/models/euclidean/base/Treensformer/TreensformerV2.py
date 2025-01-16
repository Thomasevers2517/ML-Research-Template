
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


    
class TreensformerBlockV2(nn.Module):
    

    def __init__(self, block_size, n_embd, n_head, attn_pdrop, resid_pdrop, T_Threshold=0):
        """ Constructor for the Block class 
        Args:
            tree_structure: A tuple  (parent_map, children_map, sibling_map, n_levels) representing the tree structure
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
        
        self.reg_loss = 0

    def forward(self, x):
        """ Forward pass for the Block class """
        B, N_PATCHES, N_LEVELS, R = x.size()
        C = R*N_LEVELS
        x = x.view(B, N_PATCHES, C)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        x = x.view(B, N_PATCHES, N_LEVELS, R)
        self.reg_loss += self.hierarchy_regularisation(x, alpha=0.1)

        return x
    
    def regularisations(self, x, alpha):
        loss = self.hierarchy_regularisation(x, alpha)
        return loss 
    
    def hierarchy_regularisation(self, x, alpha):
        loss = alpha * torch.var(x[:,:,0,:], dim=1).sum() # 
        loss += alpha * torch.var(x[:,:,1,:], dim=1).sum() #
        return loss
        