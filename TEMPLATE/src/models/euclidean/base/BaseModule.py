import torch
import wandb

class BaseModule(torch.nn.Module):
    def __init__(self, input_shape: tuple, output_shape: tuple):
        super(BaseModule, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activations = {}
        self.hook_handles = {}

    def forward(self, x):
        raise NotImplementedError

    def register_hooks(self):
        for name, module in self.named_modules():
            # if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            self.hook_handles[name] = module.register_forward_hook(self.get_activation(name))
    def remove_hooks(self):
        for handle in self.hook_handles.values():
            handle.remove()
            
    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook