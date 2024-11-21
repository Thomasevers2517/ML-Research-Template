import src.data.Image_Dataloader as Image_Dataloader
from src.training.base.BaseTrainer import BaseTrainer
import src.models.euclidean.image.ImageMLP as ImageMLP
import src.models.euclidean.image.VIT as VIT
import torch
import wandb
from configs.training.single_run_config import TRAIN_CONFIG

if __name__ == '__main__':
    print("CONFIGURATION \n", TRAIN_CONFIG)
    
    torch.manual_seed(0)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if len(TRAIN_CONFIG['TRAINER_PARAMS']['DEVICES']) > 1:
            DATA_PARALLEL = True
            device = torch.device(f"cuda:{TRAIN_CONFIG['TRAINER_PARAMS']['DEVICES'][0]}")
        elif len(TRAIN_CONFIG['TRAINER_PARAMS']['DEVICES']) == 1:
            DATA_PARALLEL = False
            device = torch.device(f"cuda:{TRAIN_CONFIG['TRAINER_PARAMS']['DEVICES'][0]}")
    else:
        DATA_PARALLEL = False
        device = torch.device('cpu')
        
    print(f"Using device: {device} and DATA_PARALLEL: {DATA_PARALLEL}")
    
    # Initialize wandb
    wandb.init(project=TRAIN_CONFIG["WANDB_LOGGING_PARAMS"]["PROJECT"], dir=TRAIN_CONFIG["WANDB_LOGGING_PARAMS"]["DIR"])

    if TRAIN_CONFIG['DATA_LOADER'] == 'Image_Dataloader':
        train_loader, val_loader, test_loader = Image_Dataloader.get_dataloaders(dataset_name=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["DATASET_NAME"], 
                                                                                data_dir=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["DATA_DIR"], 
                                                                                batch_size=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["BATCH_SIZE"])
    
    inputs, targets = next(iter(train_loader))

    input_shape = inputs.shape[1:]  # Exclude batch size
    targets_shape = (train_loader.dataset.dataset.num_classes,)
    print(f"Input shape: {input_shape}, Output shape: {targets_shape}")
   
    
    # model = ImageMLP.ImageMLP(input_shape=input_shape, output_shape=targets_shape, 
    # num_layers=1, hidden_size=50).to(device)
    
    model = VIT.VIT(input_shape=input_shape, output_shape=targets_shape, 
                    num_layers=TRAIN_CONFIG['MODEL_PARAMS']['NUM_LAYERS'],
                    embedding_size=TRAIN_CONFIG['MODEL_PARAMS']['EMBEDDING_SIZE'],
                    num_heads=TRAIN_CONFIG['MODEL_PARAMS']['NUM_HEADS'],
                    patch_size=TRAIN_CONFIG['MODEL_PARAMS']['PATCH_SIZE'],
                    T_Threshold=TRAIN_CONFIG['MODEL_PARAMS']['T_THRESHOLD']).to(device)
    
    if DATA_PARALLEL:
        model = torch.nn.DATA_PARALLEL(model, device_ids=TRAIN_CONFIG['TRAINER_PARAMS']['DEVICES'])
    model = model.to(device)
    wandb.watch(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = BaseTrainer(model, train_loader, val_loader, optimizer, loss_fn, device, data_parallel=DATA_PARALLEL, log_interval=4)
    
    trainer.train(epochs=1500)
    wandb.log({"test_loss": trainer.test(test_loader=test_loader)})
    
    inputs, targets = next(iter(test_loader))
    predictions = model(inputs.to(device)).argmax(dim=1)
    
    wandb.log({"test_image": [wandb.Image(inputs[i], caption=f"Prediction: {predictions[i]}, Target: {targets[i]}") for i in range(8)]})