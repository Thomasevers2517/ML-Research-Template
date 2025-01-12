import src.data.Image_Dataloader as Image_Dataloader
from src.training.base.BaseTrainer import BaseTrainer
import src.models.euclidean.image.ImageMLP as ImageMLP
import src.models.euclidean.image.VIT as VIT
import src.models.euclidean.image.ImageTreensformer as ImageTreensformer
import torch
import wandb
import yaml
from src.utils.EarlyStopping.BaseEarlyStopping import BaseEarlyStopping
from ptflops import get_model_complexity_info


if __name__ == '__main__':
    with open("TEMPLATE/configs/training/default_run_config.yaml", 'r') as f:
        DF_TRAIN_CONFIG = yaml.safe_load(f)["TRAIN_CONFIG"]
        
    wandb.init(project=DF_TRAIN_CONFIG["WANDB_LOGGING_PARAMS"]["PROJECT"], 
               dir=DF_TRAIN_CONFIG["WANDB_LOGGING_PARAMS"]["DIR"], config=DF_TRAIN_CONFIG,
               notes=DF_TRAIN_CONFIG["WANDB_LOGGING_PARAMS"]["NOTES"],)
    TRAIN_CONFIG = wandb.config
    
    print("CONFIGURATION \n", TRAIN_CONFIG)
    
    torch.manual_seed(0)
    
    if torch.cuda.is_available():
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('high')
        if TRAIN_CONFIG['TRAINER_PARAMS']['DATA_PARALLEL']:
            device = torch.device(f"cuda:0")
        elif len(TRAIN_CONFIG['TRAINER_PARAMS']['DEVICES']) == 1:
            device = torch.device(f"cuda:0")
    else:
        device = torch.device('cpu')
        
    print(f"Using devices: {device} and DATA_PARALLEL: {TRAIN_CONFIG['TRAINER_PARAMS']['DATA_PARALLEL']}")
    
    if TRAIN_CONFIG['DATA_LOADER_PARAMS']['AUGMENTATION']:
        import torchvision.transforms as transforms
        if TRAIN_CONFIG['DATA_LOADER_PARAMS']['DATASET_NAME'] == 'CIFAR10':
            MEAN = (0.4914, 0.4822, 0.4465)
            STD = (0.2023, 0.1994, 0.2010)
            SIZE = 32
        elif TRAIN_CONFIG['DATA_LOADER_PARAMS']['DATASET_NAME'] == 'CIFAR100':
            MEAN = (0.5071, 0.4867, 0.4408)
            STD = (0.2675, 0.2565, 0.2761)
            SIZE = 32
        else:
            raise NotImplementedError(f"Dataset {TRAIN_CONFIG['DATA_LOADER_PARAMS']['DATASET_NAME']} not implemented")

        # Define training transformations
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),                # Randomly flip images horizontally
            transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),  # Randomly resize and crop images
            transforms.RandomRotation(15),                   # Random rotation within ±15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
            transforms.ToTensor(),                            # Convert images to PyTorch tensors
            transforms.Normalize(MEAN, STD) # Normalize using dataset mean and std
    ])
        
    if TRAIN_CONFIG['DATA_LOADER'] == 'Image_Dataloader':
        train_loader, val_loader, test_loader = Image_Dataloader.get_dataloaders(dataset_name=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["DATASET_NAME"], 
                                                                                data_dir=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["DATA_DIR"], 
                                                                                batch_size=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["BATCH_SIZE"],
                                                                                num_workers=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["NUM_WORKERS"],
                                                                                train_transform=train_transform)
    
    inputs, targets = next(iter(train_loader))

    input_shape = inputs.shape[1:]  # Exclude batch size
    targets_shape = (train_loader.dataset.dataset.num_classes,)
    print(f"Input shape: {input_shape}, Output shape: {targets_shape}")
   
    
    # model = ImageMLP.ImageMLP(input_shape=input_shape, output_shape=targets_shape, 
    # num_layers=1, hidden_size=50).to(device)
    if TRAIN_CONFIG['MODEL'] == 'VIT':
        model = VIT.VIT(input_shape=input_shape, output_shape=targets_shape, 
                        num_layers=TRAIN_CONFIG['MODEL_PARAMS']['NUM_LAYERS'],
                        embedding_size=TRAIN_CONFIG['MODEL_PARAMS']['EMBEDDING_SIZE'],
                        num_heads=TRAIN_CONFIG['MODEL_PARAMS']['NUM_HEADS'],
                        patch_size=TRAIN_CONFIG['MODEL_PARAMS']['PATCH_SIZE'],
                        T_Threshold=TRAIN_CONFIG['MODEL_PARAMS']['T_THRESHOLD'],
                        num_cls_tkn = TRAIN_CONFIG['MODEL_PARAMS']['NUM_CLS'],
                        ).to(device)
    elif TRAIN_CONFIG['MODEL'] == 'ImageTreensformer':
        model = ImageTreensformer.ImageTreensformer(input_shape=input_shape, output_shape=targets_shape,
                                                    num_layers=TRAIN_CONFIG['MODEL_PARAMS']['NUM_LAYERS'],
                                                    embedding_size=TRAIN_CONFIG['MODEL_PARAMS']['EMBEDDING_SIZE'],
                                                    num_heads=TRAIN_CONFIG['MODEL_PARAMS']['NUM_HEADS'],
                                                    patch_size=TRAIN_CONFIG['MODEL_PARAMS']['PATCH_SIZE'],
                                                    T_Threshold=TRAIN_CONFIG['MODEL_PARAMS']['T_THRESHOLD'],
                                                    num_cls_tkn=TRAIN_CONFIG['MODEL_PARAMS']['NUM_CLS']).to(device)
    

    macs, params = get_model_complexity_info(
        model, tuple(input_shape), verbose=False, as_strings=False
    )
    print(f"FLOPs: {macs}")
    print(f"Parameters: {params}")
    wandb.log({"FLOPs": macs, "Parameters": params})
    
    if TRAIN_CONFIG['TRAINER_PARAMS']['DATA_PARALLEL']:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    if TRAIN_CONFIG['TEST_RUN']== False:
        print("Compiling model")
        model = torch.compile(model)
        print("Model compiled")
    # wandb.watch(model)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['OPTIMIZER_PARAMS']['LR'])
    if TRAIN_CONFIG['OPTIMIZER_PARAMS']['LR_SCHEDULER'] == 'StepLR':
        raise NotImplementedError("StepLR not implemented")
    elif TRAIN_CONFIG['OPTIMIZER_PARAMS']['LR_SCHEDULER'] == 'ReduceLROnPlateau':
        raise NotImplementedError("ReduceLROnPlateau not implemented")
    elif TRAIN_CONFIG['OPTIMIZER_PARAMS']['LR_SCHEDULER'] == 'CosineAnnealingLR':
        print("Using CosineAnnealingLR")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG['OPTIMIZER_PARAMS']['NUM_EPOCHS'])
    elif TRAIN_CONFIG['OPTIMIZER_PARAMS']['LR_SCHEDULER'] == 'None':     
        scheduler = None
    if TRAIN_CONFIG['LOSS_FN'] == 'CrossEntropyLoss':
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss function {TRAIN_CONFIG['LOSS_FN']} not implemented")
    
    early_stopper = BaseEarlyStopping(patience=TRAIN_CONFIG['EARLY_STOPPING_PARAMS']['PATIENCE'], 
                                      verbose=TRAIN_CONFIG['EARLY_STOPPING_PARAMS']['VERBOSE'], 
                                      delta=TRAIN_CONFIG['EARLY_STOPPING_PARAMS']['DELTA'])
    
    trainer = BaseTrainer(model, train_loader, val_loader, optimizer, loss_fn, device, data_parallel=TRAIN_CONFIG['TRAINER_PARAMS']['DATA_PARALLEL'], 
                          log_interval=TRAIN_CONFIG['TRAINER_PARAMS']['LOG_INTERVAL'], EarlyStopper=early_stopper, scheduler=scheduler)
    
    test_loss, test_accuracy = trainer.train(epochs=TRAIN_CONFIG['OPTIMIZER_PARAMS']['NUM_EPOCHS'])
