import os
import wandb # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]

from torch import nn # pyright: ignore[reportMissingImports]
from torchvision import models # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]

from src.models.cnn_simple import CNN_simple
from src.loader.tooth_data import ToothData
from src.train.train_classification import Trainer
from src.train.validation_classification import metrics_caries_icdas
from src.utils.load_data_main_cbct import make_folds, create_cross_val

def train_simple(batch_size, epochs, folds=5):
    
    device = os.getenv('gpu')    
    api_key = os.getenv('WANDB_API_KEY')
    
    wandb.login(key=api_key)
    
    wandb.init(
        project="caries_cnn_simple",
        notes="first_experimental",
    )
    
    wandb.config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate_init": 0.001, 
    }
    
    data_folds = make_folds(total_folds=folds)
    data_train, data_val = create_cross_val(data_folds, fold=1)

    dataset_train = ToothData(data_train)
    dataset_val = ToothData(data_val)
    
    train_data = DataLoader(dataset_train, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    val_data = DataLoader(dataset_val, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    
    cnn = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    cnn.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model = CNN_simple(cnn, num_classes=5).to('cuda:'+str(device))
    
    loss_function = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_cnn = Trainer(
        loss_fn=loss_function, 
        optimizer=optimizer, 
        model=model, 
        get_metrics=metrics_caries_icdas, 
        scheduler=scheduler, 
        device=device
    )
    
    train_cnn.train_epochs(train_data=train_data, val_data=val_data, epochs=epochs)
    
    model = train_cnn.model
    
    torch.save(model.state_dict(), './src/models/cnn_'+str(1)+'.pth')
    
    del model
    
    return