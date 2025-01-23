# tree_attention.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_node_id_map(H, W, n_levels):
    """
    Creates a node_id_map of shape (H, W, n_levels) that assigns each
    (h, w, level) a unique ID in [0..M-1], so that nodes repeated
    (i.e. the same 'parent') share the same ID.

    Example logic:
      - For level=0, each patch is unique => node_id = h*W + w
      - For higher levels, each 2^level block is one node
      - For top level, everything is node_id=0 (the root).
    Modify as needed for your actual duplication structure. 
    """
    node_id_map = torch.zeros(H, W, n_levels, dtype=torch.long)
    for lvl in range(n_levels):
        block_size = 2 ** lvl
        for h in range(H):
            for w in range(W):
                bh = h // block_size
                bw = w // block_size
                # flatten
                if lvl == 0:
                    node_id_map[h, w, lvl] = bh * (H // block_size) + bw 
                else:
                    node_id_map[h, w, lvl] = bh * (H // block_size) + bw  + node_id_map[-1,-1,lvl-1] + 1

    return node_id_map


def unify_nodes(x, node_id_map, M):
    """
    x: (B, H, W, n_levels, R)
    node_id_map: (H, W, n_levels), same for entire batch
    M: total unique nodes
    
    - Each (h,w,lvl) that refers to the same physical node has the same ID.

    Returns:
      unique_nodes: (B, M, R) after averaging duplicates
    """
    B, H, W, L, R = x.shape
    device = x.device

    # Flatten (H,W,L) => N
    N = H * W * L
    x_flat = x.view(B, N, R)

    node_id_flat = node_id_map.view(N)
    

    sum_buffer = torch.zeros(B, M, R, device=device)
    count_buffer = torch.zeros(B, M, device=device)

    # For each sample in the batch, we scatter_add
    for b in range(B):
        local_x = x_flat[b]             # (N, R)
        idx_expanded = node_id_flat.unsqueeze(1).expand(N, R)  # (N, R)
        sum_buffer[b] = sum_buffer[b].scatter_add(0, idx_expanded, local_x)

        ones = torch.ones(N, device=device)
        count_buffer[b] = count_buffer[b].scatter_add(0, node_id_flat, ones)

    # avoid /0
    count_buffer = torch.clamp(count_buffer, min=1e-6)
    unique_nodes = sum_buffer / count_buffer.unsqueeze(2)  # (B, M, R)
    return unique_nodes


# def scatter_back(x, updated_nodes, node_id_map, M):
#     """
#     x: (B, H, W, L, R) [only for shape references]
#     updated_nodes: (B, M, R)
#     node_id_map: (H, W, L)
#     M: total node IDs

#     Returns:
#       new_x => (B, H, W, L, R)
#         Each position picks updated_nodes[b, node_id_map[h,w,level], :]
#     """
#     B, H, W, L, R = x.shape
#     N = H * W * L
#     device = x.device

#     node_id_flat = node_id_map.view(N)
#     # We'll build new_x_flat => shape (B, N, R)
#     new_x_flat = torch.zeros_like(x.view(B, N, R))

#     for i in range(N):
#         # For each node, we pick the updated node
#         new_x_flat[:, i, :] = updated_nodes[:, node_id_flat[i], :]
    
#     new_x = new_x_flat.view(B, H, W, L, R)
#     return new_x

def scatter_back(x, updated_nodes, node_id_map, M):
    """
    x: (B, H, W, L, R) [only for shape references]
    updated_nodes: (B, M, R)
    node_id_map: (H, W, L)
    M: total node IDs

    Returns:
      new_x => (B, H, W, L, R)
        Each position picks updated_nodes[b, node_id_map[h,w,level], :]
    """
    B, H, W, L, R = x.shape

    # Flatten node_id_map to (H * W * L,)
    node_id_flat = node_id_map.view(-1)  # Shape: (H * W * L)

    # Repeat `node_id_flat` for each batch => shape (B, H * W * L)
    node_id_broadcast = node_id_flat.unsqueeze(0).expand(B, -1)

    # Gather from updated_nodes => shape (B, H * W * L, R)
    new_x_flat = torch.gather(updated_nodes, 1, node_id_broadcast.unsqueeze(-1).expand(-1, -1, R))

    # Reshape back to (B, H, W, L, R)
    new_x = new_x_flat.view(B, H, W, L, R)

    return new_x



class SimpleMHA(nn.Module):
    """
    A minimal multi-head attention for shape (B, N, R) -> (B, N, R).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, R)
        B, N, R = x.shape
        assert R == self.embed_dim

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, N, hd)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, nh, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, nh, N, hd)
        out = out.transpose(1, 2).contiguous().view(B, N, R)
        out = self.out_proj(out)
        return out
