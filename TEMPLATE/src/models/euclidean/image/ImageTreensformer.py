import torch
import torch.nn as nn
import math
from collections import defaultdict

from src.models.euclidean.base.BaseModule import BaseModule
from src.models.euclidean.base.Treensformer.Treensformer import TreensformerBlock

# Add these imports for visualization
import networkx as nx
import matplotlib.pyplot as plt

class ImageTreensformerV2(BaseModule):
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
    ):
        super(ImageTreensformerV2, self).__init__(input_shape=input_shape, output_shape=output_shape)
        print("Building ImageTreensformer")
        
        # Initialize basic parameters
        self.patch_size = patch_size
        self.H = input_shape[1] // patch_size
        self.W = input_shape[2] // patch_size
        
        
        
        self.num_children_h = 2  # Quad-tree structure
        self.num_children_w = 2
        
        n_h_levels = math.log(self.H, self.num_children_h) + 1
        n_w_levels = math.log(self.W, self.num_children_w) + 1
        
        assert n_h_levels.is_integer(), "Image size must be compatible with number of children"
        assert n_w_levels.is_integer(), "Image size must be compatible with number of children"
        assert n_h_levels == n_w_levels, "Image size must be compatible with number of children"
        
        self.num_patches = self.H * self.W
        self.num_levels = n_w_levels
        
        assert self.num_levels.is_integer(), "Image size must be compatible with number of children"
        self.num_levels = int(self.num_levels)
        
        self.embed_dim = embedding_size
        self.flatten_dim = patch_size * patch_size * input_shape[0]
     
        # Define layers
        self.embedding = nn.Linear(self.flatten_dim, self.embed_dim)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, self.n_nodes, self.embed_dim))

        self.transformer = nn.Sequential(
            *[
                TreensformerBlock(
                    self.n_nodes,
                    self.embed_dim,
                    num_heads,
                    dropout,
                    dropout,
                    T_Threshold=T_Threshold,
                    tree_mask=self.tree_mask,
                    tree_structure=(self.parent_map, self.children_map, self.sibling_map, self.num_levels),
                    mlp_type="full_branch",
                    attn_type="per_node"
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp_cls = nn.Linear(self.embed_dim * 1, self.output_size)  # Root token is the classification token

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
        x = x.view(B_, h, w, C_ * ps1 * ps2)  # (B, h, w, flatten_dim)

        x = self.build_token_tree(x)  # Build token tree
        # self.visualize_tree_mask()
        
        # Forward pass through the network
        x = self.embedding(x)  # (1, n_nodes, embed_dim)
        x = x + self.positional_embeddings
        x = self.transformer(x)  # (B, n_nodes, embed_dim)
        x = self.mlp_cls(x[:, 0, :])  # (B, output_size)
        return x
    

    # ----------------------------------------------------------------
        

    # ----------------------------------------------------------------
    # Building the Tree 
    # ----------------------------------------------------------------
    def build_token_tree(self, x: torch.Tensor) -> torch.Tensor:
        """
        x has shape (1, h, w, flatten_dim).
        We'll do a DFS-based approach that creates (1, n_nodes, flatten_dim).
        And populate self.parent_map, self.children_map, self.sibling_map.
        """
        self.parent_map.clear()
        self.children_map.clear()
        self.sibling_map.clear()

        B, h, w, dim = x.shape  
        x_tree = torch.zeros(B, self.n_nodes, dim, dtype=x.dtype, device=x.device)

        # Start DFS
        self._add_parent(x_tree, idx=0, parent_idx=None, x=x)

        # Once done => build sibling_map from children_map
        for p, child_list in self.children_map.items():
            for c in child_list:
                # Siblings = all children except c
                sibs = [ch for ch in child_list if ch != c]
                self.sibling_map[c] = sibs
        
        return x_tree

    def _add_parent(self, x_tree, idx: int, parent_idx: int, x: torch.Tensor):
        """
        DFS step:
          1) x_tree[:, idx, :] = average embedding of the subregion
          2) record parent->child, child->parent
          3) subdiv if not leaf
        """
        B, h, w, dim = x.shape
        current_idx = idx
        # average embed:
        x_tree[:, current_idx, :] = x.mean(dim=(1, 2))

        # parent/child relations
        if parent_idx is not None:
            self.parent_map[current_idx] = parent_idx
            self.children_map[parent_idx].append(current_idx)

        idx += 1

        # If h == 1 or w == 1, we can't subdiv => leaf
        if h == 1 or w == 1:
            return idx

        # subdiv => top-left, top-right, bottom-left, bottom-right
        hl, wl = h // 2, w // 2
        # top-left
        idx = self._add_parent(x_tree, idx, current_idx, x[:, :hl, :wl, :])
        # top-right
        idx = self._add_parent(x_tree, idx, current_idx, x[:, :hl, wl:, :])
        # bottom-left
        idx = self._add_parent(x_tree, idx, current_idx, x[:, hl:, :wl, :])
        # bottom-right
        idx = self._add_parent(x_tree, idx, current_idx, x[:, hl:, wl:, :])
        return idx

    # ----------------------------------------------------------------
    # Building the Mask in __init__
    # ----------------------------------------------------------------
    def _create_mask(self, x_tree: torch.Tensor) -> torch.Tensor:
        """
        Create a boolean mask of shape (n_nodes, n_nodes) that encodes
        your desired connectivity (e.g. children<->parent, leaves<->leaves, siblings<->siblings).

        x_tree shape => (1, n_nodes, dim).
        We'll use self.parent_map, self.children_map, self.sibling_map.
        """
        _, n_nodes, _ = x_tree.shape
        mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool, device=x_tree.device)

        # 1) Mark parent <-> child
        for child, parent in self.parent_map.items():
            mask[child, parent] = True
            mask[parent, child] = True

        # 2) Mark siblings <-> siblings
        for node, sibs in self.sibling_map.items():
            for s in sibs:
                mask[node, s] = True
                mask[s, node] = True

        # 3) Mark leaf <-> leaf (optional)
        #    leaves are those not in self.children_map
        all_nodes = set(range(n_nodes))
        parents_with_children = set(self.children_map.keys())
        leaves = list(all_nodes - parents_with_children)
        for i in leaves:
            for j in leaves:
                mask[i, j] = True
                mask[j, i] = True
                
        # 4) Allow parent to communicate to all children
        for parent, children in self.children_map.items():
            for child in children:
                mask[parent, child] = True
                mask[child, parent] = True

        # 5) Optionally allow self-attention
        mask.fill_diagonal_(True)

        return mask
    
    def visualize_tree_mask(self):
        """
        Visualize the attention mask 
        """
        # Plot the adjacency matrix
        plt.figure(figsize=(6, 6))
        plt.imshow(self.tree_mask.cpu().detach().numpy(), cmap="viridis")
        plt.title("Tree Mask")
        plt.show()
        