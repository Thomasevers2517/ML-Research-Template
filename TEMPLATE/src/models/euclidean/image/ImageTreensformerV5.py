
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from src.models.euclidean.base.Treensformer.TreensformerV5 import TreensformerBlockV5

class ImageTreensformerV5(nn.Module):
    """
    End-to-end: extracts patches, builds hierarchical tokens,
    then runs multiple TreensformerBlockV4 layers, finally does classification.
    """

    def __init__(
        self,
        input_shape,      # (C, H, W)
        output_shape, 
        num_layers,
        embedding_size,
        num_heads,
        patch_size,
        dropout=0.4
    ):
        super().__init__()
        # Basic
        self.C, self.H, self.W = input_shape
        self.output_size = output_shape[0]  # e.g. number of classes
        self.patch_size = patch_size
        self.n_embd = embedding_size
        self.num_levels = int(math.log2(self.H)) + 1
        self.inner_dim = self.n_embd // self.num_levels

        # a simple linear embedding
        self.flat_dim = patch_size*patch_size*self.C
        self.embedding = nn.Linear(self.flat_dim*self.num_levels, self.n_embd)
        
        # Build transformer layers
        self.layers = nn.Sequential(
            *[
                TreensformerBlockV5(
                    n_embd=embedding_size,
                    num_heads=num_heads,
                    n_levels=self.num_levels,
                    attn_pdrop=dropout,
                    resid_pdrop=dropout
                )
                for _ in range(num_layers)
            ]
        )
        # classification head
        self.mlp_cls = nn.Linear(self.inner_dim, self.output_size)

    def build_token_tree(self, x):
        """
        Convert from (B,H,W,flat_dim) to (B,H,W,n_levels,flat_dim).
        We'll do a top-down approach, storing each level by averaging sub-blocks.
        """
        B, H_, W_, fd = x.shape
        # in your code you do log2, etc.
        # We'll replicate that approach:
        # For each level L, block size = 2^L
        # This is the same logic you used in your 'build_token_tree' snippet
        n_levels = int(math.log2(H_)) + 1
        x_tree = torch.zeros(B, H_, W_, n_levels, fd, device=x.device)
        for level in range(n_levels):
            block_size = 2**level
            for i in range(H_):
                for j in range(W_):
                    local_i = (i//block_size)*block_size
                    local_j = (j//block_size)*block_size
                    # average sub-block
                    x_tree[:, i, j, level, :] = x[:, local_i:local_i+block_size,
                                                     local_j:local_j+block_size, :].mean(dim=(1,2))
        return x_tree

    def forward(self, x):
        """
        x shape => (B, C, H, W).
        1) extract patches
        2) build token tree
        3) embed + run layers
        4) classify
        """
        B, C, H, W = x.shape
        # basic patch extraction
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        # => (B, C, H//patch_size, W//patch_size, patch_size, patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  
        # => (B, h, w, C, p, p)
        # flatten
        bh, bw = x.shape[1], x.shape[2]
        x = x.view(B, bh, bw, C*self.patch_size*self.patch_size)

        # build token tree => (B, bh, bw, n_levels, flat_dim)
        x_tree = self.build_token_tree(x)  # shape => (B, H, W, n_levels, flat_dim)

        # now flatten the last dim => we want [flat_dim * n_levels] => we do it differently:
        # Actually we do it in the embedding step:
        B_, H_, W_, L_, fd_ = x_tree.shape
        x_tree_reshaped = x_tree.view(B_, H_, W_, L_*fd_)  # => (B,H,W, n_levels*flat_dim)
        x_embed = self.embedding(x_tree_reshaped)  # => (B,H,W, n_embd)
        
        # reshape => (B,H,W,n_levels, n_embd//n_levels)
        x_embed = x_embed.view(B_, H_, W_, self.num_levels, self.inner_dim)

        # pass through layers
        out = self.layers(x_embed)  # => (B,H,W,n_levels,inner_dim)

        # for classification, maybe we pick the root node => out[:,:,:, -1, :]
        # or do mean of level=0 => depends on your approach
        root_nodes = out[:, :, :, self.num_levels-1, :]  # shape (B,H,W,R)
        # average them => (B,R)
        root_avg = root_nodes.mean(dim=(1,2))
        logits = self.mlp_cls(root_avg)
        return logits