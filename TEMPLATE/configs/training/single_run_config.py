

TRAIN_CONFIG = { 
    "MODEL": "VIT",
    "MODEL_PARAMS": {
        "NUM_LAYERS": 4,
        "EMBEDDING_SIZE": 64,
        "NUM_HEADS": 4,
        "PATCH_SIZE": 1,
        "T_THRESHOLD": 0.001
    },
    "OPTIMIZER": "AdamW",
    "OPTIMIZER_PARAMS": {
        "LR": 0.0002,
        "NUM_EPOCHS": 1500,
        "LR_SCHEDULER": "None",
    },
    "LOSS": "CrossEntropyLoss",
    "TRAINER": "BaseTrainer",
    "TRAINER_PARAMS": {
        "LOG_INTERVAL": 4,
        "VAL_INTERVAL": 1,
        "DEVICES": [3, 4],
    },
    "DATA_LOADER": "Image_Dataloader",
    "DATA_LOADER_PARAMS": {
        "DATASET_NAME": "CIFAR10",
        "DATA_DIR": "data",
        "BATCH_SIZE": 32
    },
    "WANDB_LOGGING_PARAMS": {
        "PROJECT": "template_test",
        "DIR": "TEMPLATE/log"
    }
}
