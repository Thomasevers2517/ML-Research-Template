import torch
import torch.nn as nn
import math
from collections import defaultdict

from src.models.euclidean.base.BaseModule import BaseModule
from src.models.euclidean.base.Treensformer.TreensformerV4 import TreensformerBlockV4

# Add these imports for visualization
import networkx as nx
import matplotlib.pyplot as plt

class ImageTreensformerV4(BaseModule):
    def __init__(
        self,
        input_shape: tuple,      # (C, H, W)
        output_shape: tuple, 
        num_layers: int,
        embedding_size: int,
        num_heads: int,
        patch_size: int,
        dropout: float = 0.4,
        T_Threshold=0,
        h_reg = 0
    ):
        super(ImageTreensformerV4, self).__init__(input_shape=input_shape, output_shape=output_shape)
        print("Building ImageTreensformer")
        
        # Initialize basic parameters
        self.patch_size = patch_size
        self.H = input_shape[1] // patch_size
        self.W = input_shape[2] // patch_size
        
        self.n_patches = self.H * self.W
        self.num_heads = num_heads
        self.num_children_h = 2  # Quad-tree structure
        self.num_children_w = 2
        
        n_h_levels = math.log(self.H, self.num_children_h) + 1
        n_w_levels = math.log(self.W, self.num_children_w) + 1
        
        assert n_h_levels.is_integer(), "Image size must be compatible with number of children"
        assert n_w_levels.is_integer(), "Image size must be compatible with number of children"
        assert n_h_levels == n_w_levels, "Image size must be compatible with number of children"
        
        self.N = self.H * self.W
        self.n_levels = int(n_w_levels)
        

        
        self.n_emb = embedding_size
        assert self.n_emb % self.n_levels * self.num_heads == 0, "Embedding size must be divisible by the number of levels times the number of heads, for the tree structure"

        self.flat_emb = patch_size * patch_size * input_shape[0]
        assert self.n_emb % self.n_levels == 0, "Embedding size must be divisible by the number of levels"
        # Define layers
        print(f"Number of patches: {self.n_patches}, Number of levels: {self.n_levels}, Embedding size: {self.n_emb}")
        self.embedding = nn.Linear(self.flat_emb * self.n_levels, self.n_emb)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, self.N, self.n_levels, self.n_emb//self.n_levels))

        self.treensformer = nn.Sequential(
            *[
                TreensformerBlockV4(
                    self.N,
                    self.n_emb,
                    num_heads,
                    dropout,
                    dropout,
                    T_Threshold=T_Threshold,
                    n_levels=self.n_levels
         
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp_cls = nn.Linear(self.n_emb//self.n_levels, self.output_size)  # Root token is the classification token

    def forward(self, x):
        """
        Forward pass: extract patches, embed them, and apply the transformer.
        """
        B, C, H, W = x.shape

        # Extract patches and reshape
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        B_, h, w, C_, ps1, ps2 = x.shape
        x = x.view(B_, h, w, C_ * ps1 * ps2)  # (B, h, w, flat_emb)

        # Build token tree
        x = self.build_token_tree(x)  # (B, H*W, n_levels * flat_emb) == (B, n_patches, n_emb)
        
        # Forward pass through the network
        x = self.embedding(x)
        x = x.view(B, self.n_patches, self.n_levels, self.n_emb // self.n_levels) 
        expanded_positional_embeddings = self.positional_embeddings.expand(B, self.N, self.n_levels, self.n_emb//self.n_levels)
        x = x + self.positional_embeddings
        x = self.treensformer(x)  
        x = self.mlp_cls( torch.mean(x[:, :, self.n_levels-1, :], dim=1) ) # Average pooling of the root token of all tokens
        return x
    

    def build_token_tree(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build a hierarchical token tree from the input tensor.
        """
        B, H, W, flat_emb = x.shape  
        self.n_levels = int(math.log(H, 2)) + 1
        
        x_tree = torch.zeros(B, H, W, self.n_levels, flat_emb, dtype=x.dtype, device=x.device)
        
        for level in range(self.n_levels):
            summary_size_h = 2**level
            summary_size_w = 2**level
            for i in range(H):
                for j in range(W):
                    local_h = int(i // summary_size_h)
                    local_w = int(j // summary_size_w)
                    x_tree[:, i, j, level, :] = x[:, local_h*summary_size_h:(local_h+1)*summary_size_h, local_w*summary_size_w:(local_w+1)*summary_size_w, :].mean(dim=(1, 2))
        
        x_tree = x_tree.view(B, H*W, self.n_levels * flat_emb)
        
        return x_tree