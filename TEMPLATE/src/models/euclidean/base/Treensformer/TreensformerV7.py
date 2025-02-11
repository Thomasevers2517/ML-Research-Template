import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# If you have them in separate modules:
from src.models.euclidean.base.Treensformer.avg_siblings import avg_siblings
from src.models.euclidean.base.Treensformer.TreeAttentionV3 import (
    build_node_id_map, unify_nodes, scatter_back, SimpleMHA
)



class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class TreensformerBlockV7(nn.Module):
    """
    A hierarchical 'tree' block that:

      1) Normalizes input at each node (LN).
      2) Unifies duplicates across (H,W,level) => single node embedding per unique ID.
      3) Runs multi-head attention on (B,M,R).
      4) Scatters back to (B,H,W,level,R).
      5) Residual sum w/ dropout.
      6) LN again -> TreeMLPV4
      7) Residual sum
    """

    def __init__(
        self,
        n_embd: int,     # total embed dim (e.g. 256)
        num_heads: int,  # number of attention heads
        n_levels: int,   # number of hierarchical levels
        attn_pdrop: float = 0.4,
        resid_pdrop: float = 0.4,
        mask =None,
        mlp = None,
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

        self.mlp = torch.nn.Sequential(
            nn.Linear(self.n_embd, self.n_embd * mlp["HIDDEN_MULTIPLIER"]),
            NewGELU(),
            nn.Linear(self.n_embd * mlp["HIDDEN_MULTIPLIER"], self.n_embd),
        )
        
        for i in range(n_levels):
            self.M += 4**i
            
        H, W = 2**(n_levels-1), 2**(n_levels-1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.node_id_map = build_node_id_map(H, W, self.n_levels).to(device=device)
        
        self.register_buffer("att_mask", create_mask(self.node_id_map, self.M, mask_dict=mask))
        
        

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

        B, H, W, L, R = x.shape
        assert L == self.n_levels, f"Expected n_levels={self.n_levels}, got L={L}"
        assert R == self.inner_dim, f"Expected R={self.inner_dim}, got {R}"

        # 1) LN
        x_ln = self.ln_1(x)  # (B,H,W,L,R)

        # 2) unify => (B,M,R)
        unique_nodes = unify_nodes(x_ln, self.node_id_map, self.M)


        # 3) attn => (B,M,R)
        attn_out = self.attn(unique_nodes, mask=self.att_mask)

        # 5) scatter => shape (B,H,W,L,R)
        x_attn = scatter_back(x_ln, attn_out, self.node_id_map, self.M)

        # 6) residual sum
        x_res = x + x_attn

        # 7) LN => TreeMLPV3 => shape (B,H,W,L,R)
        x_ln2 = self.ln_2(x_res)
        x_ln2 = x_ln2.view(B, H, W, L*R)
        
        # 8 ) MLP on all branches and avg the parents
        mlp_out = self.mlp(x_ln2)  # => (B,H,W,L,R)
        mlp_out = mlp_out.view(B, H, W, L, R)
        for i in range(L):
            mlp_out[:, :, :, i, :] = avg_siblings(mlp_out[:, :, :, i, :], sibling_order=i, h_summary_size=H, w_summary_size=W)
        

        # 8) residual sum MLP-ed (Must be unified because the dropout needs to occor at the same nodes not different nodes at different branches)
        x_unify = unify_nodes(mlp_out, self.node_id_map, self.M)
        dropout_x_unify = self.resid_pdrop(x_unify)
        dropout_x = scatter_back(x_ln2, dropout_x_unify, self.node_id_map, self.M)
        
        assert torch.all(dropout_x[0, :, :, -1, 0] == dropout_x[0, 0, 0, -1, 0]), \
            f"Dropout should be the same for all nodes at the same level, but got {dropout_x[0, :, :, -1, 0]}"
            
        x_final = x_res + dropout_x
        
        return x_final
    
def create_mask(node_id_map, M, mask_dict=None):
    """
    Create a mask for the attention mechanism.
    The mask is a (M, M) matrix where M is the total number of nodes in the tree.
    The mask is a boolean matrix where mask[i, j] = True means that node i can attend to node j.
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
                    if mask_dict["SELF"]:
                        # NODE attends to itself
                        mask[node_id, node_id] = True
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
                
                
                    
                if mask_dict["CHILD_PARENT"]:
                    # NODE attents to PARENT
                    mask[node_id, parent_ids[0]] = True
                    
                if mask_dict["PARENT_CHILD"]:
                    # PARENT attents to NODE
                    mask[parent_ids[0], node_id] = True
                
                if mask_dict["GRANDPARENT_GRANDCHILD"]:
                    for parent in parent_ids:
                        mask[node_id, parent] = True
                if mask_dict["GRANDCHILD_GRANDPARENT"]:
                    # ALL PARENTS attend to NODE    
                    for parent in parent_ids:
                        mask[parent, node_id] = True
                if mask_dict["SIBLING"]:
                    # NODE attends to direct SIBLINGS
                    for sibling in sibling_ids[0]:
                        mask[node_id, sibling] = True
                if mask_dict["COUSIN"]:
                    # NODE attends to SIBLINGS of any order (as in cousins and such)
                    for level_siblings in sibling_ids:
                        for sibling in level_siblings:
                            mask[node_id, sibling] = True
    return mask
                
                
                

