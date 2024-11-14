import src.data.MNIST_Dataloader as MNIST_Dataloader
from src.training.base.BaseTrainer import BaseTrainer
import src.models.MNIST.Simple_MNIST_Model as Simple_MNIST_Model
import torch
import wandb

if __name__ == '__main__':
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Initialize wandb
    wandb.init(project="template_test", dir="TEMPLATE/log")

    train_loader, val_loader, test_loader = MNIST_Dataloader.get_dataloaders('TEMPLATE/data/MNIST/raw', 64)
    
    model = Simple_MNIST_Model.Linear_MNIST().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    trainer = BaseTrainer(model, train_loader, val_loader, optimizer, loss_fn, device)
    trainer.train(epochs=5)
    wandb.log({"test_loss": trainer.test()})