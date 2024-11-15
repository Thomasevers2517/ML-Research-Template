import torch
class BaseModule(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple):
        super(BaseModule, self).__init__()
        
        self.input_shape = input_shape    
        self.output_shape = output_shape

        
    
    def forward(self, x):
        raise NotImplementedError
        return x
    
    def calculate_parameters(self):
        raise Warning("calculate_parameters not implemented")
        return 0