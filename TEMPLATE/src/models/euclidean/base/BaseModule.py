import torch
import wandb

class BaseModule(torch.nn.Module):
    """ Base class for all modules """
    def __init__(self, input_shape: tuple, output_shape: tuple):
        super(BaseModule, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_size = torch.prod(torch.tensor(self.input_shape)).item()
        self.output_size = torch.prod(torch.tensor(self.output_shape)).item()
        self.activations = {}
        self.hook_handles = {}

    def forward(self, x):
        raise NotImplementedError

    def register_hooks(self):
        """ Register forward hooks for all modules in the model """
        for name, module in self.named_modules():
            #CAN BE MODIFIED TO ONLY REGISTER HOOKS FOR SPECIFIC MODULES
            self.hook_handles[name] = module.register_forward_hook(self.get_activation(name))
    def remove_hooks(self):
        """ Remove all hooks """
        for handle in self.hook_handles.values():
            handle.remove()
            
    def get_activation(self, name):
        """ Return a hook function that saves the activation of a module """
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                # Detach each tensor within the tuple or list
                detached = []
                for o in output:
                    if isinstance(o, torch.Tensor):
                        detached.append(o.detach())
                    else:
                        detached.append(o)  # Keep non-tensor elements as is
                self.activations[name] = detached if isinstance(output, list) else tuple(detached)
            else:
                raise Warning(f"Unhandled output type: {type(output)}")
                self.activations[name] = output  # Store as is or handle accordingly

        return hook
