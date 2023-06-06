import os
import torch # pyright: ignore[reportMissingImports]
import wandb# pyright: ignore[reportMissingImports]


from torch import nn # pyright: ignore[reportMissingImports]
from torchvision import models # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]
 
from src.models.cnn_simple import CNN_simple
from src.loader.tooth_data import ToothDataRotate
from src.train.train_classification import Trainer
from src.train.validation_classification import metrics_caries_ssl_rotate
from src.utils.load_data_main_cbct import make_path_ssl


def train_ssl(batch_size, epochs):
    
    data_folds = make_path_ssl()
    idx_total = len(data_folds)

    data_train = data_folds[ : int(0.8*idx_total)]
    data_val = data_folds[int(0.8*idx_total) : idx_total]
    
    device = os.getenv('gpu')
    api_key = os.getenv('WANDB_API_KEY')    

    wandb.login(key=api_key)
    
    run = wandb.init(
        project="caries_cnn_simple",
        notes="first_experimental",
        name='SSL_Rotate'
    )
    
    wandb.config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate_init": 0.001, 
    }

    dataset_train = ToothDataRotate(data_train) 
    dataset_val = ToothDataRotate(data_val)
    
    train_data = DataLoader(dataset_train, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    val_data = DataLoader(dataset_val, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    
    cnn = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    cnn.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model = CNN_simple(cnn, num_classes=4).to('cuda:'+str(device))
    
    loss_function = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_cnn = Trainer(
        get_metrics=metrics_caries_ssl_rotate,
        loss_fn=loss_function, 
        optimizer=optimizer, 
        scheduler=scheduler,
        device=device,
        model=model,   
    )
    
    train_cnn.train_epochs(train_data=train_data, val_data=val_data, epochs=epochs)
    
    model = train_cnn.model
    
    torch.save(model.state_dict(), './src/models/cnn_ssl_'+str(1)+'.pth')

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file('./src/models/cnn_ssl_'+str(1)+'.pth')
    run.log_artifact(artifact)
    run.finish()
    
    return