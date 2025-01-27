import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.euclidean.base.Treensformer.TreeMLP.TreeMLPV3 import TreeMLPV3

from src.models.euclidean.base.Treensformer.TreensformerV6 import TreensformerBlockV6

class ImageTreensformerV6(nn.Module):
    """
    End-to-end: extracts patches, builds hierarchical tokens,
    applies multiple TreensformerBlockV5 layers, then does classification.
    Includes positional embeddings of shape (1,bh,bw,n_levels,inner_dim).
    """

    def __init__(
        self,
        input_shape,      # (C, H, W)
        output_shape, 
        num_layers,
        embedding_size,
        num_heads,
        patch_size,
        dropout=0.4,
        mask=None,
        mlp = {"HIDDEN_MULTIPLIER": 4, "DROPOUT": 0.3},
    ):
        super().__init__()
        # Basic
        self.C, self.H, self.W = input_shape
        self.output_size = output_shape[0]  # e.g. number of classes
        self.patch_size = patch_size
        self.n_embd = embedding_size

        # Number of levels = log2(H/patch_size) + 1 if you want the full hierarchy
        self.num_levels = int(math.log2(self.H/patch_size)) + 1
        self.inner_dim = self.n_embd // self.num_levels
        
        
        self.flat_dim = patch_size * patch_size * self.C
                
        # We'll define pos_embed of shape (1, bh, bw, n_levels, self.flat_dim).
        # where bh = H//patch_size, bw = W//patch_size
        bh = self.H // self.patch_size
        bw = self.W // self.patch_size
        
        self.pos_embed = torch.nn.ParameterList()
        
        for level in range(self.num_levels):
            self.pos_embed.append(nn.Parameter(torch.zeros((1, bh//2**(level), bw//2**(level) ,self.inner_dim))))

        # A simple linear embedding for each patch-level cell
        self.embedding = nn.Linear(self.flat_dim, self.inner_dim)

        #for trying mlp with shared weights among transformer layers
        
        # shared_mlp = TreeMLPV3(
        #     n_embd=self.inner_dim,
        #     n_levels=self.num_levels,
        #     hidden_multiplier=mlp["HIDDEN_MULTIPLIER"],
        # )
        
        # Build transformer layers
        self.layers = nn.Sequential(
            *[
                TreensformerBlockV6(
                    n_embd=embedding_size,
                    num_heads=num_heads,
                    n_levels=self.num_levels,
                    attn_pdrop=dropout,
                    resid_pdrop=dropout,
                    mask=mask,
                    mlp=mlp
                )
                for _ in range(num_layers)
            ]
        )
        # Classification head
        self.mlp_cls = nn.Linear(self.inner_dim, self.output_size)

    def build_token_tree(self, x):
        """
        Convert from (B, h, w, flat_dim) to (B, h, w, n_levels, flat_dim).
        We'll do a top-down approach, storing each level by averaging sub-blocks.
        """
        B, h, w, fd = x.shape
        n_levels = int(math.log2(h)) + 1

        x_tree = torch.zeros(B, h, w, n_levels, fd, device=x.device, dtype=x.dtype)
        for level in range(n_levels):
            block_size = 2**level
            for i in range(h):
                for j in range(w):
                    bi = (i // block_size) * block_size
                    bj = (j // block_size) * block_size
                    # average sub-block
                    x_tree[:, i, j, level, :] = x[:, bi:bi+block_size, bj:bj+block_size, :].mean(dim=(1,2))
        return x_tree

    def forward(self, x):
        """
        x shape => (B, C, H, W).
        1) extract patches => (B,h,w, flat_dim)
        2) build token tree => (B,h,w,n_levels, flat_dim)
        3) embed => (B,h,w,n_levels, inner_dim)
        4) add pos_embed => (B,h,w,n_levels, inner_dim)
        5) pass through TreensformerBlockV5 layers
        6) root node => classification
        """
        B, C, H, W = x.shape
        # 1) Patch extraction
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        # => shape (B, C, h, w, patch_size, patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  
        # => (B, h, w, C, patch_size, patch_size)
        bh, bw = x.shape[1], x.shape[2]
        x = x.view(B, bh, bw, C * (self.patch_size**2))


        # 2) Build token tree => shape (B,bh,bw,n_levels,flat_dim)
        x_tree = self.build_token_tree(x)
        
        # 3) Embedding => shape (B,bh,bw,n_levels,flat_dim) => (B,bh,bw,n_levels,inner_dim)
        B_, h_, w_, L_, fd_ = x_tree.shape
        x_embed = self.embedding(x_tree)

        # 4) Add positional embeddings => 
        for level in range(self.num_levels):
            pos_interleaved = self.pos_embed[level].repeat_interleave(2**level, dim=1).repeat_interleave(2**level, dim=2)
            pos_broadcast = pos_interleaved.expand(B, -1, -1, -1)
            x_embed[:, :, :, level, :] = x_embed[:, :, :, level, :] + pos_broadcast
            

        # 5) Pass through layers => (B,h_,w_,L_,inner_dim)
        out = self.layers(x_embed)

        # 6) root node => out[:,:,:,L_-1,:], average => (B,inner_dim)
        root_nodes = out[:, :, :, L_-1, :]  # => (B,h_,w_,inner_dim)
        root_avg = root_nodes.mean(dim=(1,2))  # => (B,inner_dim)

        logits = self.mlp_cls(root_avg)
        return logits
