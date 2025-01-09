import torch
import torch.nn as nn
import math
from collections import defaultdict

from src.models.euclidean.base.BaseModule import BaseModule
from src.models.euclidean.base.Treensformer.Treensformer import TreensformerBlock

# Add these imports for visualization
import networkx as nx
import matplotlib.pyplot as plt



class ImageTreensformer(BaseModule):
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
        num_cls=1
    ):
        super(ImageTreensformer, self).__init__(input_shape=input_shape, output_shape=output_shape)
        
        # -------------------------------------------------
        # Basic definitions
        # -------------------------------------------------
        self.patch_size = patch_size
        self.num_patches = (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
        self.num_children = 4 #will only work for quad-tree for now
        # For a perfect quad-tree: log(...) etc. 
        self.num_levels = math.log(math.sqrt(self.num_patches), math.sqrt(self.num_children)) + 1
        assert self.num_levels.is_integer(), "Image size must be compatible with number of children"
        self.num_levels = int(self.num_levels)
        
        self.n_nodes = sum(self.num_children**i for i in range(self.num_levels))

        
        self.embed_dim = embedding_size
        self.flatten_dim = patch_size * patch_size * input_shape[0]
        self.num_cls = num_cls
        
        assert input_shape[1] == input_shape[2], "Input image must be square (H==W)."



        # -------------------------------------------------
        # Build the tree structure and mask ONCE at init
        # -------------------------------------------------
        # 1) Build a dummy patch-grid that has shape (1, h, w, flatten_dim).
        h = input_shape[1] // patch_size
        w = input_shape[2] // patch_size
        # Make a dummy tensor; the actual values don't matter, just shape
        dummy_x = torch.zeros(1, h, w, self.flatten_dim)  
        # We'll store parent_map, children_map, sibling_map as normal dicts
        self.parent_map = {}
        self.children_map = defaultdict(list)
        self.sibling_map = defaultdict(list)

        # 2) Build the DFS-based tree => returns (B=1, n_nodes, flatten_dim)
        x_tree = self.build_token_tree(dummy_x)  # shape => (1, n_nodes, flatten_dim)

        # 3) Create the mask (n_nodes, n_nodes) or (1, n_nodes, n_nodes)
        #    as you see fit. Then store as a buffer so it moves with .to(device)
        tree_mask = self._create_mask(x_tree)
        tree_mask = torch.cat([torch.ones(1, tree_mask.size(1), 
                                          dtype=tree_mask.dtype, device=tree_mask.device), tree_mask], dim=0)
        tree_mask = torch.cat([torch.ones(tree_mask.size(0), 1, 
                                          dtype=tree_mask.dtype, device=tree_mask.device), tree_mask], dim=1)
        self.register_buffer("tree_mask", tree_mask)
        
        # -------------------------------------------------
        # Layers
        # -------------------------------------------------
        self.embedding = nn.Linear(self.flatten_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
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
                )
                for _ in range(num_layers)
            ]
        )
        self.mlp_cls = nn.Linear(self.embed_dim, self.output_size)

    def forward(self, x):
        """
        In forward, you can do normal patch extraction + embedding,
        but the mask (self.tree_mask) is already built and ready for use.
        """
        B, C, H, W = x.shape

        # Basic patch extraction for real data:
        #  (B, C, H//ps, W//ps, ps, ps) => rearr => (B, h, w, flatten_dim)
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        # Move patch dimension forward => (B, h, w, C, ps, ps)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        B_, h, w, C_, ps1, ps2 = x.shape
        x = x.view(B_, h, w, C_ * ps1 * ps2)  # (B, h, w, flatten_dim)

        x = self.build_token_tree(x)  # (1, n_nodes, flatten_dim)
        # self.visualize_tree_mask()
        #---------------------------------------------------
        # Forward pass
        x = self.embedding(x)  # (1, n_nodes, embed_dim)
        x = x + self.positional_embeddings
        
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # shape: (B, num_cls, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.transformer(x) # (B, n_nodes, embed_dim)
        x = self.mlp_cls(x[:, 0])  # (B, num_cls)   
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

        # 4) Optionally allow self-attention
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
        