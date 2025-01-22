import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# If you have them in separate modules:
from src.models.euclidean.base.Treensformer.avg_siblings import avg_siblings
from src.models.euclidean.base.Treensformer.TreeAttentionV3 import (
    build_node_id_map, unify_nodes, scatter_back, SimpleMHA
)
from src.models.euclidean.base.Treensformer.TreeMLP.TreeMLPV3 import TreeMLPV3


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class TreensformerBlockV5(nn.Module):
    """
    A hierarchical 'tree' block that:

      1) Normalizes input at each node (LN).
      2) Unifies duplicates across (H,W,level) => single node embedding per unique ID.
      3) Runs multi-head attention on (B,M,R).
      4) Scatters back to (B,H,W,level,R).
      5) Residual sum w/ dropout.
      6) LN again -> TreeMLPV3
      7) Residual sum
    """

    def __init__(
        self,
        n_embd: int,     # total embed dim (e.g. 256)
        num_heads: int,  # number of attention heads
        n_levels: int,   # number of hierarchical levels
        attn_pdrop: float = 0.4,
        resid_pdrop: float = 0.4
    ):
        super().__init__()
        print("Building TreensformerBlock V5")

        self.n_embd = n_embd
        self.n_levels = n_levels
        self.num_heads = num_heads
        self.inner_dim = n_embd // n_levels  # e.g. R

        # Verify inner_dim is divisible by num_heads
        assert (self.inner_dim % self.num_heads) == 0, \
            f"inner_dim={self.inner_dim} must be multiple of num_heads={self.num_heads}"

        self.node_id_map = None  # Will build once we know H,W,L

        # Layers
        self.ln_1 = nn.LayerNorm(self.inner_dim)
        self.ln_2 = nn.LayerNorm(self.inner_dim)

        # A multi-head attention that consumes (B, N, R)
        self.attn = SimpleMHA(self.inner_dim, self.num_heads, dropout=attn_pdrop)
        self.resid_pdrop = nn.Dropout(resid_pdrop)

        # Use TreeMLPV3 for the second part
        self.tree_mlp = TreeMLPV3(n_embd, n_levels)
        
        self.M = 0  # 
        for i in range(n_levels):
            self.M += 4**i

    def forward(self, x):
        """
        x shape => (B, H, W, n_levels, R=(n_embd//n_levels))

        Steps:
          (1) LN
          (2) unify => (B,M,R)
          (3) attention => (B,M,R)
          (4) scatter => (B,H,W,L,R)
          (5) residual
          (6) LN => tree_mlp
          (7) residual
        """
        from src.models.euclidean.base.Treensformer.TreeAttentionV3 import (
            build_node_id_map, unify_nodes, scatter_back
        )  # or keep them at top

        B, H, W, L, R = x.shape
        assert L == self.n_levels, f"Expected n_levels={self.n_levels}, got L={L}"
        assert R == self.inner_dim, f"Expected R={self.inner_dim}, got {R}"

        # If no node_id_map built yet, do so. Then store for reuse
        if self.node_id_map is None:
            self.node_id_map = build_node_id_map(H, W, L).to(x.device)

        # 1) LN
        x_ln = self.ln_1(x)  # (B,H,W,L,R)

        # 2) unify => (B,M,R)
        unique_nodes = unify_nodes(x_ln, self.node_id_map, self.M)

        # 3) attn => (B,M,R)
        attn_out = self.attn(unique_nodes)

        # 4) scatter => shape (B,H,W,L,R)
        x_attn = scatter_back(x_ln, attn_out, self.node_id_map, self.M)

        # 5) residual sum w/ dropout
        x_res = x + self.resid_pdrop(x_attn)

        # 6) LN => TreeMLPV3 => shape (B,H,W,L,R)
        x_ln2 = self.ln_2(x_res)
        mlp_out = self.tree_mlp(x_ln2)  # => (B,H,W,L,R)

        # 7) final residual
        x_final = x_res + mlp_out
        return x_final
