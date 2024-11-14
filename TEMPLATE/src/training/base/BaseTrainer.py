import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable
import wandb
from tqdm import tqdm


class BaseTrainer:
    def __init__(self, model: Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, loss_fn: Callable, device: torch.device):
        """
        Initializes the BaseTrainer with the given model, data loaders, optimizer, loss function, and device.

        Args:
            model (Module): The neural network model to be trained.
            train_loader (DataLoader): The data loader for loading training data.
            val_loader (DataLoader): The data loader for loading validation data.
            optimizer (Optimizer): The optimizer for updating model parameters.
            loss_fn (Callable): The loss function to calculate the loss.
            device (torch.device): The device to run the training on (CPU or GPU).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    
    def train(self, epochs: int) -> None:
        """
        Trains the model for a specified number of epochs.
    
        Args:
            epochs (int): The number of epochs to train the model.
        """
        epoch_pbar = tqdm(range(epochs), desc='Epochs')
        for epoch in epoch_pbar:
            average_loss, val_loss = self.train_epoch()
            
            epoch_pbar.set_postfix({
            'train_loss': f'{average_loss:.4f}',
            'val_loss': f'{val_loss:.4f}'
            })
            wandb.log({
                "train_loss": average_loss,
                "val_loss": val_loss,
                "epoch": epoch
            })
            
        epoch_pbar.close()
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        for [inputs, targets] in self.train_loader:
            loss = self.train_step(inputs, targets)
            total_loss += loss
            
            # Update batch progress bar
            
        average_loss = total_loss / len(self.train_loader)
        val_loss = self.validate()
        
        # Update epoch progress bar
        
        return average_loss, val_loss
    def on_epoch_end(self, epoch: int) -> None:
        """
        Callback function that is called at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
        """
        pass
    
    def train_batch(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Performs a single training step on the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data for training.
            targets (torch.Tensor): The target data for training.

        Returns:
            float: The loss value for the training step.
        """
        self.optimizer.zero_grad()
        loss = self.batch_loss(inputs, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    def on_batch_end(self, batch: int) -> None:
        """
        Callback function that is called at the end of each batch.

        Args:
            batch (int): The current batch number.
        """
        pass
    
    def batch_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculates the loss for a single batch of inputs and targets.

        Args:
            inputs (torch.Tensor): The input data for the batch.
            targets (torch.Tensor): The target data for the batch.

        Returns:
            float: The loss value for the batch.
        """
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        
        return loss
    
    def data_loss(self, dataloader: DataLoader) -> float:
        """
        Calculates the loss for the entire dataset.

        Args:
            dataloader (DataLoader): The data loader for the dataset.

        Returns:
            float: The loss value for the dataset.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for [inputs, targets] in dataloader:
                loss = self.batch_loss(inputs, targets)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validates the model on the validation dataset.

        Returns:
            float: The average loss on the validation dataset.
        """
        return self.data_loss(val_loader)
    def test (self, test_loader) -> float:
        """
        Tests the model on the test dataset.

        Returns:
            float: The average loss on the test dataset.
        """
        return self.data_loss(test_loader)