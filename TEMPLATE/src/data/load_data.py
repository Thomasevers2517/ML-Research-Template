import os
import torchvision
import torchvision.transforms as transforms

def load_data(raw_data_dir: str) -> None:
    """
    Loads the MNIST dataset and saves it into the specified raw data directory.

    Args:
        raw_data_dir (str): The directory where the raw data will be saved.
    """
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(root=raw_data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=raw_data_dir, train=False, download=True, transform=transform)

    print(f"MNIST dataset downloaded and saved to {raw_data_dir}")

# Example usage
load_data('TEMPLATE/data/Wikipedia/raw/MNIST')