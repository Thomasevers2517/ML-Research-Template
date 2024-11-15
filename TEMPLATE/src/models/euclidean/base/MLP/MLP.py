import torch
from src.models.euclidean.base.BaseModule import BaseModule
import torch.nn as nn

class MLP(BaseModule):
    def __init__(self, input_shape: tuple, output_shape: tuple, num_layers: int, hidden_size: int):
        super(MLP, self).__init__(input_shape=input_shape, output_shape=output_shape)
        self.input_size = torch.prod(torch.tensor(self.input_shape)).item()
        self.output_size = torch.prod(torch.tensor(self.output_shape)).item()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.MLP = self.create_MLP()
        
    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten the input tensor
        x = self.MLP(x)
        return x
    
    def create_MLP(self):
        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU())
        for _ in range(self.num_layers - 1 ):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_size, self.output_size))
        MLP = nn.Sequential(*layers)
        return MLP
    