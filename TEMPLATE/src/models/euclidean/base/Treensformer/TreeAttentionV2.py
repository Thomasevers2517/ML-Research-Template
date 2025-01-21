import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class TreeAttentionV2(nn.Module):
 

    def __init__(self, block_size, n_embd, n_head, attn_pdrop, resid_pdrop, T_Threshold=0):
        
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
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
        N_NODES = 0
        for i in range(N_LEVELS):
            N_NODES += 4**i
        
        x_nodes = torch.zeros(B, N_NODES, N_LEVELS, R)
        
        for i in range(N_LEVELS):
            h_num_sum = 2**i
            w_num_sum = 2**i
            h_n_splits = N_PATCHES//h_num_sum
            w_n_splits = N_PATCHES//w_num_sum
            for j in range(h_num_sum):
                for k in range(w_num_sum):
                    local_h = int(j*h_n_splits)
                    local_w = int(k*w_n_splits)
                    x_temp =  x[:, :, :, i, :].reshape(B, h_n_splits,h_num_sum, w_n_splits, w_num_sum, R)
                    x_temp =  x_temp.mean(dim=[2, 4]) # B, h_n_splits, w_n_splits, R
                    
        
        q = torch.zeros(B, self.n_head, N_NODES, N_LEVELS * R/self.n_head)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
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
