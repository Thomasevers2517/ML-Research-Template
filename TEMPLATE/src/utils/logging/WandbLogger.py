import wandb
import numpy as np
import re
import torch

class WandbLogger:
    """
    A logger class for logging model activations and attention maps to Weights & Biases (wandb).

    Attributes:
        model (torch.nn.Module): The model whose activations and attention maps are to be logged.
        model_name (str): A sanitized and shortened version of the model's class name.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Initializes the WandbLogger with the given model.

        Args:
            model (torch.nn.Module): The model to be logged.
        """
        self.model = model
        self.model_name = re.sub(r'\W+', '', str(model.__class__.__name__))
        self.model_name = self.model_name.lower()
        self.model_name = self.model_name[:10]
        
    def log_activations(self, activations: dict[str, torch.Tensor], n_samples_to_log: int = 1) -> None:
        """
        Logs the activations of the model.

        Args:
            activations (dict[str, torch.Tensor]): A dictionary containing the activations of the model.
            n_samples_to_log (int): The number of samples to log from the activations.
        """
        if n_samples_to_log > len(activations[list(activations.keys())[0]]):
            n_samples_to_log = len(activations[list(activations.keys())[0]])
            raise Warning("num_samples_to_log is greater than the number of samples in the batch")
        
        for sample_idx in range(n_samples_to_log):
            for name, activation in activations.items():
                activation = activation.cpu().numpy()
                if re.match(r"transformer\.\d+\.attn\.att_map", name):
                    self.log_all_attentions(name=name, sample_idx=sample_idx, att_map=activation[sample_idx])
                else:
                    wandb.log({f"{name}, sample_idx-{sample_idx}": activation[sample_idx]})
                     
        self.model.activations.clear()

    def log_all_attentions(self, name: str, sample_idx: int, att_map: np.ndarray) -> None:
        """
        Logs all attention maps for a given sample.

        Args:
            name (str): The name of the attention map.
            sample_idx (int): The index of the sample.
            att_map (np.ndarray): The attention map to be logged. Shape: (n_heads, seq_len, seq_len) or (seq_len, seq_len)
        """
        if len(att_map.shape) == 2:
            nH = 1
        else:
            nH = att_map.shape[0]  # Number of heads
        layer_idx = name.split('.')[1]

        for head in range(nH):
            attention_map_description = f"sample_idx-{sample_idx}, layer-{layer_idx}, head-{head}"
        
            self.log_cls_to_patch_attmap(att_map[head], attention_map_description)
            self.log_patch_to_cls__attmap(att_map[head], attention_map_description)
    
    def log_patch_to_cls__attmap(self, att_map: np.ndarray, attention_map_description: str) -> None:
        """
        Logs the patch-to-class attention map.

        Args:
            att_map (np.ndarray): The attention map to be logged. Shape: (seq_len, seq_len)
            attention_map_description (str): A description of the attention map.
        """
        cls_to_patch_attentions = att_map[1:, 0]
        cls_to_patch_attentions = np.reshape(cls_to_patch_attentions, (
                                        int(self.model.input_shape[1] / self.model.patch_size), 
                                        int(self.model.input_shape[2] / self.model.patch_size)))
        log_name = f"patch_to_cls_att, {attention_map_description}"
        self.log_image(cls_to_patch_attentions, log_name)                 

    def log_cls_to_patch_attmap(self, att_map: np.ndarray, attention_map_description: str) -> None:
        """
        Logs the class-to-patch attention map.

        Args:
            att_map (np.ndarray): The attention map to be logged. Shape: (seq_len, seq_len)
            attention_map_description (str): A description of the attention map.
        """
        cls_to_patch_attentions = att_map[0, 1:]
        cls_to_patch_attentions = np.reshape(cls_to_patch_attentions, (
                                        int(self.model.input_shape[1] / self.model.patch_size), 
                                        int(self.model.input_shape[2] / self.model.patch_size)))
        log_name = f"cls_to_patch_att, {attention_map_description}"
        self.log_image(cls_to_patch_attentions, log_name)    

    def log_image(self, image: np.ndarray, log_name: str) -> None:
        """
        Logs a normalized attention map as an image to wandb.

        Args:
            image (np.ndarray): The attention map to be logged. Shape: (height, width)
            log_name (str): The name under which the attention map will be logged.
        """
        max_value = np.max(image[:, :])
        normalized_attention_map = image[:, :] / max_value

        wandb.log({log_name:
            wandb.Image(normalized_attention_map, caption=f"max_value- {max_value}")})