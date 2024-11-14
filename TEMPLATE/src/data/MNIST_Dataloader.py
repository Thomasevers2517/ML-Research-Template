import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from typing import Callable

class MNIST_Data(Dataset):
    def __init__(self, data_dir: str, train: bool = True, transform: Callable = None):
        """
        Initializes the MNIST_Data class with the given data directory, train flag, and transform.

        Args:
            data_dir (str): The directory where the MNIST data is stored.
            train (bool): Whether to load the training data or test data.
            transform (Callable): The transform to apply to the data.
        """
        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        self.dataset = datasets.MNIST(root=self.data_dir, train=self.train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def get_dataloaders(data_dir: str, batch_size: int, val_split: float = 0.1):
    """
    Returns the train, validation, and test dataloaders.

    Args:
        data_dir (str): The directory where the MNIST data is stored.
        batch_size (int): The batch size for the dataloaders.
        val_split (float): The fraction of the training data to use for validation.

    Returns:
        tuple: A tuple containing the train, validation, and test dataloaders.
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST_Data(data_dir, train=True, transform=transform)
    test_dataset = MNIST_Data(data_dir, train=False, transform=transform)

    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader