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
        
        self.first_batch = True
    
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
        self.on_epoch_start()
        self.model.train()
        total_loss = 0
        
        for [inputs, targets] in self.train_loader:
            
            loss = self.train_batch(inputs, targets)
            total_loss += loss
            
            # Update batch progress bar
            
        average_loss = total_loss / len(self.train_loader)
        val_loss = self.validate(val_loader=self.val_loader)
        
        # Update epoch progress bar
        self.on_epoch_end()
        return average_loss, val_loss
    def on_epoch_start(self) -> None:
        """
        Callback function that is called at the start of each epoch.

        Args:
            epoch (int): The current epoch number.
        """
        self.first_batch = True
        self.model.register_hooks()
        pass
    
    def on_epoch_end(self) -> None:
        """
        Callback function that is called at the end of each epoch.

        """
        
        pass
    
    def on_batch_start(self) -> None:
        """
        Callback function that is called at the start of each batch.
        """
        pass
    

    def on_batch_end(self) -> None:
        """
        Callback function that is called at the end of each batch.
        """
        if self.first_batch:
            for name, activation in self.model.activations.items():
                print(f'activations/{name}')
                print(activation.cpu().numpy().shape)
                wandb.log({f'activations/sample1/{name}': wandb.Histogram(activation.cpu().numpy()[0])})
            self.first_batch = False    
            self.model.activations.clear()
            self.model.remove_hooks()
        pass
        
    
    def train_batch(self, inputs: torch.Tensor, targets: torch.Tensor, log_level : int = None) -> float:
        """
        Performs a single training step on the given inputs and targets.

        Args:
            inputs (torch.Tensor): The input data for training.
            targets (torch.Tensor): The target data for training.
            log_level (int): The level of logging to perform.

        Returns:
            float: The loss value for the training step.
        """
        self.on_batch_start()
        self.optimizer.zero_grad()
        loss = self.calculate_batch_loss(inputs, targets)
        loss.backward()
        self.optimizer.step()
        
        self.on_batch_end()
        return loss.item()
    
    def calculate_batch_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
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
    
    def calculate_data_loss(self, dataloader: DataLoader) -> float:
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
                loss = self.calculate_batch_loss(inputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validates the model on the validation dataset.

        Returns:
            float: The average loss on the validation dataset.
        """
        return self.calculate_data_loss(val_loader)
    def test (self, test_loader: DataLoader) -> float:
        """
        Tests the model on the test dataset.

        Returns:
            float: The average loss on the test dataset.
        """
        return self.calculate_data_loss(test_loader)