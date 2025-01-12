
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.models.euclidean.base.Tranformer.Attentions.DynamicAttention import MultiheadDynamicSelfAttention
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class TreensformerBlock(nn.Module):
    

    def __init__(self, block_size, n_embd, n_head, attn_pdrop, resid_pdrop, T_Threshold=0, tree_mask=None):
        """ Constructor for the Block class """
        
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiheadDynamicSelfAttention(block_size= block_size, n_embd=n_embd, n_head=n_head, 
                                           attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, T_Threshold=T_Threshold, tree_mask=tree_mask)
        self.ln_2 = nn.LayerNorm(n_embd)

        self.mlpf = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            NewGELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop))

    def forward(self, x):
        """ Forward pass for the Block class """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
    
class TreensformerBlock_BranchMLP(nn.Module):
    

    def __init__(self, block_size, n_embd, n_head, attn_pdrop, resid_pdrop, T_Threshold=0, tree_mask=None, tree_structure=None):
        """ Constructor for the Block class 
        Args:
            tree_structure: A tuple  (parent_map, children_map, sibling_map, n_levels) representing the tree structure
            """
        assert tree_structure is not None, "tree_structure must be provided "
        
        super().__init__()
        
        self.parent_map, self.children_map, self.sibling_map, self.n_levels = tree_structure
        
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiheadDynamicSelfAttention(block_size= block_size, n_embd=n_embd, n_head=n_head, 
                                           attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, T_Threshold=T_Threshold, tree_mask=tree_mask)
        self.ln_2 = nn.LayerNorm(n_embd)

        self.mlpf = FullBranchMLP(n_embd, n_levels=self.n_levels, parent_map=self.parent_map, children_map=self.children_map, block_size=block_size)

    def forward(self, x):
        """ Forward pass for the Block class """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
    
class FullBranchMLP(nn.Module):
    """ An MLP which takes a leaf node and all its parents to predict the output """

    def __init__(self, n_embd, n_levels, parent_map, children_map, block_size):
        super().__init__()
        self.n_nodes = block_size
        self.n_levels = n_levels
        self.n_leafs = 4**n_levels
        
        self.B1 = nn.Parameter(torch.zeros(n_embd*4, self.n_nodes, n_embd))
        self.W1 = torch.zeros(n_embd* 4, self.n_nodes, self.n_nodes, n_embd)
        self.linear2 = nn.Linear(n_embd*4, n_embd)
        
        self.ancestor_map = build_ancestor_map(parent_map)
        
        
        self.W1_parts = torch.nn.ParameterList([None] * n_levels, [None] * n_levels)
        for i in range(n_levels):
            for j in range(n_levels):
                # W1_parts[i][j] is the weight matrix for the MLP from level i to level j
                if i > j:
                    #child doesnt effect the mlp of the parent
                    continue
                if i == j:
                    # On the same level only the node itself effects its mlp
                    self.W1_parts[f"ii"] = nn.parameter.Parameter(torch.zeros(4*n_embd, n_embd), requires_grad=True)
                    
                if i < j:
                    # For level i parent to level j child
                    self.W1_parts[f"ij"] = nn.parameter.Parameter(torch.zeros(4*n_embd, n_embd), requires_grad=True)
        
        for child, parents in self.ancestor_map.items():
            child_level = len(parents)
            #The part of the mlp where the child effects itself
            self.W1[:, child, child, :] = self.W1_parts[child_level, child_level]
            
            #The part of the mlp where the parents effect the child
            for i, parent in enumerate(parents):
                self.W1[:, child, parent, :] = self.W1_parts[child_level][child_level+i]
                    
        
    def forward(self, x):
        """ Forward pass for the FullBranchMLP class """
        #x: [B, num_nodes, dim]
        batch_size = x.size(0)
        
        W1_expanded = self.W1.unsqueeze(0).expand(batch_size, -1, -1, -1, -1) # [B, 4*dim, num_nodes, num_nodes, dim]
        B1_expanded = self.B1.unsqueeze(0).expand(batch_size, -1, -1) # [B, num_nodes, dim*4]
        x = torch.einsum('bni,bjmni -> bmj', x, W1_expanded) + B1_expanded
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
def build_ancestor_map(parent_map, all_nodes=None):
    """
    Given a dict `parent_map` of {child_node: parent_node, ...},
    build a dict `ancestor_map` of:
      node -> [parent, grandparent, ..., root]
    If all_nodes is None, we infer it from parent_map keys and values.
    """
    if all_nodes is None:
        # Infer all distinct node IDs from parent_map
        # by collecting keys and values
        unique_keys = set(parent_map.keys())
        unique_vals = set(parent_map.values())
        all_nodes = unique_keys.union(unique_vals)
    else:
        all_nodes = set(all_nodes)

    ancestor_map = {}
    for node in all_nodes:
        ancestors = []
        current = node
        while current in parent_map:  # climb until no more parent
            p = parent_map[current]
            ancestors.append(p)
            current = p
        ancestor_map[node] = ancestors

    return ancestor_map