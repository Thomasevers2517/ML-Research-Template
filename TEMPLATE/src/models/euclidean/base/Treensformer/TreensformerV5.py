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
            
        self.att_mask = None

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
        if self.att_mask is None:
            self.att_mask = create_mask(self.node_id_map, self.M)

        # 3) attn => (B,M,R)
        attn_out = self.attn(unique_nodes, mask=self.att_mask)

        # 4) scatter => shape (B,H,W,L,R)
        x_attn = scatter_back(x_ln, attn_out, self.node_id_map, self.M)

        # 5) residual sum w/ dropout
        # x_res = x + self.resid_pdrop(x_attn)
        x_res = x + x_attn

        # 6) LN => TreeMLPV3 => shape (B,H,W,L,R)
        x_ln2 = self.ln_2(x_res)
        mlp_out = self.tree_mlp(x_ln2)  # => (B,H,W,L,R)

        # 7) final residual
        x_final = x_res + mlp_out
        return x_final
    
def create_mask(node_id_map, M):
    """
    node_id_map: (H, W, L) => int nodeID in [0..M-1].
    M: total # of unique nodes.

    Returns:
      mask: (M, M) bool or float. 
        mask[i,j] = True => node i attends to node j.
    
    We'll identify:
      1) child->parent connections
      2) parent->child connections
      3) siblings
      4) grandparents, grandchildren if wanted
      ...
    We'll let lines be easily commentable so you can remove certain edges.
    """

    # We'll build a list of coords for each node ID
    # node_positions[i] = list of (h,w,ell)
    H, W, L = node_id_map.shape
    device = node_id_map.device
    mask = torch.zeros(M, M, dtype=torch.bool, device=device)
    
    for w in range(W):  
        for h in range(H):
            for ell in range(L):
                # mask[receivers(q), senders(k)]

                node_id = node_id_map[h,w,ell].item()
                if ell == L-1:
                    continue
                
                parent_ids = [None] * ((L-1) - ell)
                for i in range((L-1)-ell):
                    parent_ids[i] = node_id_map[h, w, ell+1+i].item()
                    
                sibling_ids = []
                for i, parent in enumerate(parent_ids):
                    level_siblings = set()
                    for w_ in range(W):
                        for h_ in range(H):
                            if node_id_map[h_, w_, ell+1+i].item() == parent:
                                sibling_id = node_id_map[h_, w_, ell]
                                if sibling_id.item() != node_id:
                                    level_siblings.add(sibling_id.item())
                    sibling_ids.append(level_siblings)
                print(f"Node {node_id} has parents {parent_ids} and siblings {sibling_ids}")      
                # # NODE attents to PARENT
                # mask[node_id, parent_ids[0]] = True
                
                # PARENT attents to NODE
                mask[parent_ids[0], node_id] = True
                
                # #NODE attends to ALL PARENT
                # for parent in parent_ids:
                #     mask[node_id, parent] = True
                
                # ALL PARENTS attend to NODE    
                for parent in parent_ids:
                    mask[parent, node_id] = True
                
                    

                # NODE attends to direct SIBLINGS
                for sibling in sibling_ids[0]:
                    mask[node_id, sibling] = True
                    
                # DIRECT SIBLINGS attend to NODE
                for sibling in sibling_ids[0]:
                    mask[sibling, node_id] = True
                    
                # NODE attends to SIBLINGS of any order (as in cousins and such)
                for level_siblings in sibling_ids:
                    for sibling in level_siblings:
                        mask[node_id, sibling] = True
    print(mask)
    return mask
                
                
                
                            
                    
                    

                
                
                
                
            
                
            # # GRANDPARENT or further, if you want to comment it out easily
            

            # # SIBLINGS
            

    return mask
