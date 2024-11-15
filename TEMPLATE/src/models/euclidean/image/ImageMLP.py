import torch
from src.models.euclidean.base.MLP.MLP import MLP
import torch.nn as nn

class ImageMLP(MLP):
    def __init__(self, input_shape: tuple, output_shape: tuple, num_layers: int, hidden_size: int):
        super(ImageMLP, self).__init__(input_shape=input_shape, output_shape=output_shape, num_layers=num_layers, hidden_size=hidden_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = super().forward(x)
        x = nn.functional.softmax(x, dim=1)
        return x
