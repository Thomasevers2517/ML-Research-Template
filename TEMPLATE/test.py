import src.data.Image_Dataloader as Image_Dataloader
from src.training.base.BaseTrainer import BaseTrainer
import src.models.euclidean.image.ImageMLP as ImageMLP
import src.models.euclidean.image.VIT as VIT
import torch
import wandb

if __name__ == '__main__':
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.device_count() > 1:
            device = torch.device('cuda:3')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    # Initialize wandb
    wandb.init(project="template_test", dir="TEMPLATE/log")

    train_loader, val_loader, test_loader = Image_Dataloader.get_dataloaders(dataset_name='CIFAR10', data_dir='data', batch_size=32)
    
    inputs, targets = next(iter(train_loader))

    input_shape = inputs.shape[1:]  # Exclude batch size
    targets_shape = (train_loader.dataset.dataset.num_classes,)
    print(f"Input shape: {input_shape}, Output shape: {targets_shape}")
   
    
    # model = ImageMLP.ImageMLP(input_shape=input_shape, output_shape=targets_shape, 
    # num_layers=1, hidden_size=50).to(device)
    
    model = VIT.VIT(input_shape=input_shape, output_shape=targets_shape, 
                    num_layers=8, embedding_size=128, num_heads=4, 
                    patch_size=1, T_Threshold=0.01).to(device)
    wandb.watch(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = BaseTrainer(model, train_loader, val_loader, optimizer, loss_fn, device, log_interval=4)
    
    trainer.train(epochs=1500)
    wandb.log({"test_loss": trainer.test(test_loader=test_loader)})
    
    
    inputs, targets = next(iter(test_loader))
    predictions = model(inputs.to(device)).argmax(dim=1)
    
    wandb.log({"test_image": [wandb.Image(inputs[i], caption=f"Prediction: {predictions[i]}, Target: {targets[i]}") for i in range(8)]})