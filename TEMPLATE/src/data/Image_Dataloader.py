import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from typing import Callable

class Image_Data(Dataset):
    def __init__(self, dataset_name:str, data_dir: str, train: bool = True, transform: Callable = None):
        """
        Initializes the MNIST_Data class with the given data directory, train flag, and transform.

        Args:
            dataset_name (str): The name of the dataset to load. Supported datasets are 'MNIST', 'CIFAR10', 'CIFAR100', 'FashionMNIST', and 'ImageNet'.
            data_dir (str): The directory where the MNIST data is stored.
            train (bool): If True, returns the training data, otherwise returns the test data.
            transform (Callable): The transform to apply to the data.
        """
        self.dataset_name = dataset_name    
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        
        self.full_data_dir = os.path.join(self.data_dir, self.dataset_name)

        if dataset_name == 'MNIST':
            self.dataset = datasets.MNIST(root=self.full_data_dir, train=self.train, download=True, transform=self.transform)
        elif dataset_name == 'CIFAR10':
            self.dataset = datasets.CIFAR10(root=self.full_data_dir, train=self.train, download=True, transform=self.transform)
        elif dataset_name == 'CIFAR100':
            self.dataset = datasets.CIFAR100(root=self.full_data_dir, train=self.train, download=True, transform=self.transform)
        elif dataset_name == 'FashionMNIST':
            self.dataset = datasets.FashionMNIST(root=self.full_data_dir, train=self.train, download=True, transform=self.transform)
        elif dataset_name == 'ImageNet':
            self.dataset = datasets.ImageNet(root="/space2/thomasevers/data/imagenet", split='train' if self.train else 'val', transform=self.transform)
            raise NotImplementedError("ImageNet not yet supported. Images differ in size and require different models")
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")
        self.num_classes =len(self.dataset.classes)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def get_dataloaders(dataset_name:str, data_dir: str, batch_size: int, val_split: float = 0.1, num_workers: int = 1, train_transform: Callable = transforms.ToTensor()):


    train_dataset = Image_Data(dataset_name= dataset_name, data_dir = data_dir, train=True, transform=train_transform)
    test_dataset = Image_Data(dataset_name= dataset_name, data_dir = data_dir, train=False, transform=train_transform)

    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True, num_workers=num_workers, persistent_workers=True)

    return train_loader, val_loader, test_loader