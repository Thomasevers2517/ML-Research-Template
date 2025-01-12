
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

        self.mlpf = FullBranchMLP(n_embd, n_levels=self.n_levels, parent_map=self.parent_map, children_map=self.children_map)

    def forward(self, x):
        """ Forward pass for the Block class """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
class FullBranchMLP(nn.Module):
    """ An MLP which takes a leaf node and all its parents to predict the output """

    def __init__(self, n_embd, n_levels, parent_map, children_map):
        super().__init__()
        self.n_levels = n_levels
        self.level_idx = [None] * n_levels
        for i in range(n_levels):
            self.level_idx[i] = list()
            
        #populate the level_idx to know the levels of the nodes
        for parent, child_list in children_map.items():
            print(f"Parent: {parent}, i: {i}")
            print(f"Child_list: {child_list}")
            if parent not in parent_map.keys():
                # This is a root node
                self.level_idx[0].append(parent) 
            for i in range(n_levels):
                if parent in self.level_idx[i]:
                    for child in child_list:
                        self.level_idx[i + 1].append(child)
                        
        assert len(self.level_idx[0]) == 1, "There should be only one leaf node"
        
        self.A = torch.nn.ParameterList([None] * n_levels)
        self.num_leafs = len(self.level_idx[-1])
        for i in range(n_levels):
            self.A[i] = nn.parameter.Parameter(torch.zeros(n_embd, 4*n_embd), requires_grad=True)
            

    def forward(self, x):
        """ Forward pass for the FullBranchMLP class """
        batch_size = x.size(0)
        
        A_expanded = self.A[-1].unsqueeze(0).expand(batch_size, -1, -1)
        x[:, self.level_idx[-1], :] = nn.functional.linear(x[:, self.level_idx[-1], :], A_expanded)
        raise NotImplementedError("Must implement more. \
                                  In a way where all data can be used and forgotten. \
                                      Create a big complex matrix structure which all of x can just be trhwon into\
                                          most complexity should be at init")
        
        return x