
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.euclidean.base.Tranformer.Attentions.DynamicAttention import MultiheadDynamicSelfAttention
from src.models.euclidean.base.Treensformer.TreeAttentionV3 import TreeAttentionV3
from src.models.euclidean.base.Treensformer.avg_siblings import avg_siblings
import seaborn as sns
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# treensformer.py

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from TreeAttentionV3 import (
    build_node_id_map,
    unify_nodes,
    scatter_back,
    SimpleMHA,
)

class TreensformerBlockV4(nn.Module):
    def __init__(
        self,
        n_embd,     # total embed dim, e.g. 256
        num_heads,  # number of attention heads
        n_levels,   # number of hierarchical levels
        attn_pdrop=0.4,
        resid_pdrop=0.4
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_levels = n_levels
        self.num_heads = num_heads
        self.inner_dim = n_embd // n_levels  # e.g. R
        # Node id map for each (H,W,level) we might store or build on the fly
        self.node_id_map = None

        self.ln_1 = nn.LayerNorm(self.inner_dim)
        self.ln_2 = nn.LayerNorm(self.inner_dim)

        self.attn = SimpleMHA(self.inner_dim, num_heads, dropout=attn_pdrop)
        self.attn_dropout = nn.Dropout(attn_pdrop)

        self.mlp = nn.Sequential(
            nn.Linear(self.inner_dim, 4*self.inner_dim),
            nn.GELU(),
            nn.Linear(4*self.inner_dim, self.inner_dim),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        """
        x shape => (B, H, W, n_levels, R=(n_embd//n_levels))
        We'll do:
         (1) LN
         (2) unify -> (B,M,R)
         (3) attn
         (4) scatter -> (B,H,W,L,R)
         (5) residual
         (6) LN + MLP
         (7) residual
        """

        B, H, W, L, R = x.shape
        assert L == self.n_levels
        assert R == self.inner_dim

        # If we haven't built node_id_map or it depends on (H,W,L), do it here
        # or else store it as a buffer. For demonstration:
        if self.node_id_map is None:
            self.node_id_map = build_node_id_map(H, W, L).to(x.device)

        # Step 1) LN
        x_ln = self.ln_1(x)  # shape (B,H,W,L,R)

        # Step 2) unify
        unique_nodes, M = unify_nodes(x_ln, self.node_id_map)  # => (B,M,R)

        # Step 3) attn => shape (B,M,R)
        attn_out = self.attn(unique_nodes)

        # Step 4) scatter => shape (B,H,W,L,R)
        x_attn = scatter_back(x_ln, attn_out, self.node_id_map, M)

        # Step 5) residual
        x_res = x + self.attn_dropout(x_attn)

        # Step 6) LN + MLP
        x_ln2 = self.ln_2(x_res)
        # Flatten for MLP => shape (B,H*W*L,R)
        B_, N, R_ = x_ln2.view(B, -1, R).shape
        mlp_out = self.mlp(x_ln2.view(B_, N, R_))  # => (B_,N,R_)
        mlp_out = mlp_out.view(B,H,W,L,R)

        # Step 7) final residual
        x_final = x_res + mlp_out
        return x_final