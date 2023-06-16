import os
import torch # pyright: ignore[reportMissingImports]
import wandb# pyright: ignore[reportMissingImports]


from torch import nn # pyright: ignore[reportMissingImports]
from torchvision import models # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]
 
from src.models.cnn_simple import CNN_simple
from src.loader.tooth_data import ToothDataRotate, ToothDataJigsaw
from src.train.train_classification import Trainer
from src.train.validation_classification import metrics_caries_rotate, metrics_caries_jigsaw
from src.utils.load_data_main_cbct import make_path_ssl

def make_data(load_data, batch_size):
    
    data_folds = make_path_ssl()
    idx_total = len(data_folds)

    data_train = data_folds[ : int(0.8*idx_total)]
    data_val = data_folds[int(0.8*idx_total) : idx_total]

    dataset_train = load_data(data_train) 
    dataset_val = load_data(data_val)

    train_data = DataLoader(dataset_train, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    val_data = DataLoader(dataset_val, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)

    return train_data, val_data

def configure_setup(epochs, batch_size, name):

    api_key = os.getenv('WANDB_API_KEY')    

    wandb.login(key=api_key)
    
    return wandb.init(
        project="caries_cnn_simple",
        notes="first_experimental",
        name=name,
        config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate_init": 0.001, 
        }
    )

def make_data_loader(dataset_train, dataset_val, batch_size):

    train_data = DataLoader(dataset_train, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    val_data = DataLoader(dataset_val, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)

    return train_data, val_data

def get_model_raw():

    cnn = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    cnn.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    return cnn


def get_model(metrics, cnn, num_classes, learning_rate, device):

    model = CNN_simple(cnn, num_classes=num_classes).to('cuda:'+str(device))
    
    loss_function = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    return Trainer(
        get_metrics=metrics,
        loss_fn=loss_function, 
        optimizer=optimizer, 
        scheduler=scheduler,
        device=device,
        model=model,   
    )

def train_model(data_train, data_val, train_cnn, epochs, run, type_ssl):

    train_cnn.train_epochs(train_data=data_train, val_data=data_val, epochs=epochs)
    
    model = train_cnn.model
    
    torch.save(model.state_dict(), './src/models/cnn_ssl_'+str(1)+'.pth')

    artifact = wandb.Artifact(type_ssl, type='model')
    artifact.add_file('./src/models/cnn_ssl_'+str(1)+'.pth')
    run.log_artifact(artifact)
    run.finish()


def train_ssl(batch_size, epochs, type_ssl='rotate'):
    
    run = configure_setup(epochs, batch_size, type_ssl)
    device = os.getenv('gpu')
    cnn = get_model_raw()


    if type_ssl == 'rotate':

        data_train, data_val = make_data(ToothDataRotate, batch_size)
        trainer = get_model(metrics_caries_rotate, cnn, num_classes=4, learning_rate=0.001, device=device)
        train_model(data_train, data_val, trainer, epochs, run, type_ssl)

    if type_ssl == 'jigsaw':

        data_train, data_val = make_data(ToothDataJigsaw, batch_size)
        trainer = get_model(metrics_caries_jigsaw, cnn, num_classes=9, learning_rate=0.001, device=device)
        train_model(data_train, data_val, trainer, epochs, run, type_ssl)


    return