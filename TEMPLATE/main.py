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
               dir=DF_TRAIN_CONFIG["WANDB_LOGGING_PARAMS"]["DIR"], config=DF_TRAIN_CONFIG)
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
    
    # Initialize wandb

    if TRAIN_CONFIG['DATA_LOADER'] == 'Image_Dataloader':
        train_loader, val_loader, test_loader = Image_Dataloader.get_dataloaders(dataset_name=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["DATASET_NAME"], 
                                                                                data_dir=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["DATA_DIR"], 
                                                                                batch_size=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["BATCH_SIZE"],
                                                                                num_workers=TRAIN_CONFIG["DATA_LOADER_PARAMS"]["NUM_WORKERS"],)
    
    inputs, targets = next(iter(train_loader))

    input_shape = inputs.shape[1:]  # Exclude batch size
    targets_shape = (train_loader.dataset.dataset.num_classes,)
    print(f"Input shape: {input_shape}, Output shape: {targets_shape}")
   
    
    # model = ImageMLP.ImageMLP(input_shape=input_shape, output_shape=targets_shape, 
    # num_layers=1, hidden_size=50).to(device)
    
    # model = VIT.VIT(input_shape=input_shape, output_shape=targets_shape, 
    #                 num_layers=TRAIN_CONFIG['MODEL_PARAMS']['NUM_LAYERS'],
    #                 embedding_size=TRAIN_CONFIG['MODEL_PARAMS']['EMBEDDING_SIZE'],
    #                 num_heads=TRAIN_CONFIG['MODEL_PARAMS']['NUM_HEADS'],
    #                 patch_size=TRAIN_CONFIG['MODEL_PARAMS']['PATCH_SIZE'],
    #                 T_Threshold=TRAIN_CONFIG['MODEL_PARAMS']['T_THRESHOLD'],
    #                 num_cls = TRAIN_CONFIG['MODEL_PARAMS']['NUM_CLS'],
    #                 ).to(device)
    model = ImageTreensformer.ImageTreensformer(input_shape=input_shape, output_shape=targets_shape,
                                                num_layers=TRAIN_CONFIG['MODEL_PARAMS']['NUM_LAYERS'],
                                                embedding_size=TRAIN_CONFIG['MODEL_PARAMS']['EMBEDDING_SIZE'],
                                                num_heads=TRAIN_CONFIG['MODEL_PARAMS']['NUM_HEADS'],
                                                patch_size=TRAIN_CONFIG['MODEL_PARAMS']['PATCH_SIZE'],
                                                T_Threshold=TRAIN_CONFIG['MODEL_PARAMS']['T_THRESHOLD'],
                                                num_cls=TRAIN_CONFIG['MODEL_PARAMS']['NUM_CLS']).to(device)
    

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
    
    trainer.train(epochs=TRAIN_CONFIG['OPTIMIZER_PARAMS']['NUM_EPOCHS'])
    wandb.log({"test_loss": trainer.test(test_loader=test_loader)})
    
    inputs, targets = next(iter(test_loader))
    predictions = model(inputs.to(device)).argmax(dim=1)
    targets = targets.to(device)
    accuracy = (predictions == targets).float().mean()
    
    wandb.log({"test_accuracy": accuracy})
    wandb.log({"test_image": [wandb.Image(inputs[i], caption=f"Prediction: {predictions[i]}, Target: {targets[i]}") for i in range(8)]})