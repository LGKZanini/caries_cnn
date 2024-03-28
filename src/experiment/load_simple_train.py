import os
import wandb # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]

from torch import nn # pyright: ignore[reportMissingImports]
from torchvision import models # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]

from src.models.cnn_simple import create_model
from src.loader.tooth_data import ToothData
from src.train.train_classification import Trainer
from src.train.validation_classification import metrics_caries_icdas
from src.utils.load_data_main_cbct import create_train_test

def train_simple(batch_size, epochs, folds=5, classify_type=None, backbone='resnet18', backbone_arch=None):

    device = os.getenv('gpu')    
    api_key = os.getenv('WANDB_API_KEY')
    
    wandb.login(key=api_key)
    

    run = wandb.init(
        project="caries_cnn_simple",
        entity='luizzanini',
        name='classify_'+backbone+'_'+str(classify_type),
        config = { 
            "folds": folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate_init": 0.01,
            "backbone":backbone,
        }
    )
    
    data_train, data_test = create_train_test()

    dataset_train = ToothData(data_train)
    dataset_val = ToothData(data_test)
    
    train_data = DataLoader(dataset_train, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
    val_data = DataLoader(dataset_val, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
    
    if classify_type is None :

        model = create_model(backbone, device)
        
    else:

        model = create_model(backbone, device, backbone_arch=backbone_arch)


    loss_function = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_cnn = Trainer(
        get_metrics=metrics_caries_icdas, 
        loss_fn=loss_function,
        scheduler=scheduler, 
        optimizer=optimizer,
        device=device, 
        model=model,  
    )
    
    train_cnn.train_epochs(train_data=train_data, val_data=val_data, epochs=epochs)
    
    model = train_cnn.model
    
    torch.save(model.state_dict(), './src/models/cnn_ssl_'+str(classify_type)+'_'+str(backbone)+'.pth')

    artifact = wandb.Artifact('classify_'+str(classify_type)+'_'+str(backbone), type='model')
    artifact.add_file('./src/models/cnn_ssl_'+str(classify_type)+'_'+str(backbone)+'.pth')
    
    run.log_artifact(artifact)
    run.finish()
    
    return