import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from src.models.euclidean.base.Treensformer.avg_siblings import avg_siblings
class TreeAttentionV3(nn.Module):
 

    def __init__(self, block_size, n_embd, n_head, attn_pdrop, resid_pdrop, T_Threshold=0, n_levels=4):
        """ V3 takes only the lose embedding and not the parent embeddings. 
        But the attention consist out of a multiplication of the attentions between the parents and the children. This way a low parent attention already dooms the child
        . Also the parent attention can be regularized to be equal to be similar to the avg attention of the children"""
        super().__init__()
        assert n_embd % n_head == 0
        self.n_levels = n_levels
        self.n_head = n_head
        self.n_embd = n_embd
        self.T_Threshold = T_Threshold
        self.R = n_embd//n_levels
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.R, 3 * self.R)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # just so that the attention map is logged
        self.att_map = AttentionMap()
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.T_Threshold = T_Threshold

    def forward(self, x):
        B, N_PATCHES, N_LEVELS, R = x.size()
        level_attention = torch.zeros(B, N_PATCHES, N_PATCHES)
        for i in range(N_LEVELS, 0, -1):
            # select a single sample of each parent at level i
            x_temp =  x[:, :, i, :]
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k ,v  = self.c_attn(x).split(self.R, dim=2) # (B, N_PATCHES, N_LEVELS, R) -> (B, N_PATCHES, N_LEVELS, R)
            
            
            k = k.view(B, self.n_head, N_PATCHES, R // self.n_head).transpose(1, 2)
            q = q.view(B, self.n_head, N_PATCHES, R // self.n_head).transpose(1, 2)
            v = v.view(B, self.n_head, N_PATCHES, R // self.n_head).transpose(1, 2)
            
            # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = F.relu(att - self.T_Threshold)
            att = self.att_map(att)
            att = self.attn_dropout(att)
            
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        y = y.view(B, N_PATCHES, N_LEVELS, R)
        return y

    
class AttentionMap(nn.Module):
    """
    AttentionMap class
    """
    def __init__(self):
        super().__init__()

    def forward(self, att):
        return att
