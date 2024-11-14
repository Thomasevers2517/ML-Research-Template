import torch
class Linear_MNIST(torch.nn.Module):
    def __init__(self):
        super(Linear_MNIST, self).__init__()
        self.linear = torch.nn.Linear(784, 100)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(100, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x