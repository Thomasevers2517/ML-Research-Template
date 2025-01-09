import torch
import torch.nn as nn
from src.models.euclidean.base.BaseModule import BaseModule
from src.models.euclidean.base.Tranformer.Transformer import Block
class VIT(BaseModule):
    def __init__(self, input_shape: tuple, output_shape: tuple, num_layers: int, embedding_size: int, 
                 num_heads: int, patch_size: int, dropout: float = 0.4, T_Threshold=0, num_cls_tkn=1):
        super(VIT, self).__init__(input_shape=input_shape, output_shape=output_shape)
        
        self.patch_size = patch_size
        self.num_patches = (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
        self.embed_dim = embedding_size
        self.flatten_dim = patch_size * patch_size * input_shape[0]
        self.num_cls_tkn = num_cls_tkn
        # Linear projection of flattened patches
        self.embedding = nn.Linear(self.flatten_dim, self.embed_dim)

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_cls_tkn, self.embed_dim))
        self.positional_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + self.num_cls_tkn, self.embed_dim))

        # Transformer encoder
        self.transformer = nn.Sequential(*[Block(self.num_patches+self.num_cls_tkn, self.embed_dim, num_heads, 
                                                 dropout, dropout, T_Threshold=T_Threshold) for _ in range(num_layers)])

        # MLP head for classification
        self.mlp_cls = nn.Linear(self.embed_dim*num_cls_tkn, self.output_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # Divide images into patches
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.num_patches, -1)

        # Linear embedding
        x = self.embedding(x)

        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.positional_embeddings

        # Transformer encoding
        x = self.transformer(x)

        # Classification head
        x = self.mlp_cls( x[:, :self.num_cls_tkn].reshape(B, -1))        

        return x