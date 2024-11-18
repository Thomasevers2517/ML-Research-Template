import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module
from typing import Callable
import wandb
from tqdm import tqdm


class BaseTrainer:
    def __init__(self, model: Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, loss_fn: Callable, device: torch.device, 
                 val_interval: int = 1, log_interval: int = 1):
        """
        Initializes the BaseTrainer with the given model, data loaders, optimizer, loss function, and device.

        Args:
            model (Module): The neural network model to be trained.
            train_loader (DataLoader): The data loader for loading training data.
            val_loader (DataLoader): The data loader for loading validation data.
            optimizer (Optimizer): The optimizer for updating model parameters.
            loss_fn (Callable): The loss function to calculate the loss.
            device (torch.device): The device to run the training on (CPU or GPU).
            val_interval (int): The number of epochs between each validation step.
            log_interval (int): The number of epochs between each log step.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.val_interval = val_interval
        self.log_interval = log_interval
        
        # Flag to check if it is the first batch of a validation step
        self.first_batch = False
        
    def train(self, epochs: int) -> None:
        """
        Trains the model for a specified number of epochs.
    
        Args:
            epochs (int): The number of epochs to train the model.
        """
        epoch_pbar = tqdm(range(epochs), desc='Epochs')
        for epoch in epoch_pbar:
            avg_train_loss = self.train_epoch(epoch=epoch)
            
            if epoch % self.val_interval == 0:
                val_loss = self.validate()
            
            epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}'
            })

            
        epoch_pbar.close()
        self.on_training_end()
    def on_training_end(self) -> None:
        """
        Callback function that is called at the end of the training process.
        """
        pass
        
    def train_epoch(self, epoch:int) -> float:
        self.on_epoch_start()
        
        self.model.train()
        
        total_loss = 0
        batch_pbar = tqdm(self.train_loader, desc='Batches', leave=False, mininterval=1)
        for inputs, targets in batch_pbar:
            loss = self.train_batch(inputs, targets)
            total_loss += loss
            
        batch_pbar.close()   
            # Update batch progress bar
        avg_train_loss = total_loss / len(self.train_loader)

        
        
        # Update epoch progress bar
        self.on_epoch_end(epoch, avg_train_loss)
        return avg_train_loss
    
    def on_epoch_start(self) -> None:
        """
        Callback function that is called at the start of each epoch.

        Args:
            epoch (int): The current epoch number.
        """

        pass
    
    def on_epoch_end(self, epoch, avg_train_loss) -> None:
        """
        Callback function that is called at the end of each epoch.

        """
        wandb.log({
                "train_loss": avg_train_loss,
                "epoch": epoch
            })
        
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
            self.log_hooks()
            self.first_batch = False    
            
        pass
    def log_hooks(self) -> None:
        self.log_activations(self.model.activations)
        self.model.remove_hooks()

        
    def log_activations(self, activations: dict, n_samples_to_log:int=1) -> None:
        """
        Logs the activations of the model.

        Args:
            activations (dict): A dictionary containing the activations of the model.
        """
        if n_samples_to_log > len(activations[list(activations.keys())[0]]):
            n_samples_to_log = len(activations[list(activations.keys())[0]])
            raise Warning("num_batches is greater than the number of samples in the batch")

        for name, activation in activations.items():
            for sample in range(n_samples_to_log):
                wandb.log({f'activations/{sample}/{name}': wandb.Histogram(activation.cpu().numpy()[sample])})
        self.model.activations.clear()
        
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
    
    def validate(self) -> float:
        self.on_validation_start()
        """
        Validates the model on the validation dataset.

        Returns:
            float: The average loss on the validation dataset.
        """
        val_loss = self.calculate_data_loss(self.val_loader) 
        
        self.on_validation_end(val_loss=val_loss)
        return val_loss
    def on_validation_start(self) -> None:
        """
        Callback function that is called at the start of the validation step.
        """
        self.first_batch = True
        self.model.register_hooks()
    
    def on_validation_end(self, val_loss: float) -> None:
        
        """
        Callback function that is called at the end of the validation step.
        """
        wandb.log({
            "val_loss": val_loss
        })
        
    
    
    def test (self, test_loader: DataLoader) -> float:
        """
        Tests the model on the test dataset.

        Returns:
            float: The average loss on the test dataset.
        """
        return self.calculate_data_loss(test_loader)