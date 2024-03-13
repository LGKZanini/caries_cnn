import os
import wandb # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]

from torch import nn # pyright: ignore[reportMissingImports]
from torchvision import models # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]

from src.models.cnn_simple import CNN_simple, create_model
from src.loader.tooth_data import ToothData
from src.train.train_classification import Trainer
from src.train.validation_classification import metrics_caries_icdas
from src.utils.load_data_main_cbct import make_folds, create_train_test


def model_ssl(classify_type, cnn, run, device):

    # preciso mudar dps
    if classify_type == 'rotate':
        
        artifact = run.use_artifact('luizzanini/caries_cnn_simple/rotate:v0', type='model')
        artifact_dir = artifact.download()
        
        #mudar aqui
        model = CNN_simple(cnn, num_classes=4)
        model.load_state_dict(torch.load(artifact_dir+'/cnn_ssl_'+str(1)+'.pth'))

    if classify_type == 'simclr':
        
        artifact = run.use_artifact('luizzanini/caries_cnn_simple/simclr:v0', type='model')
        artifact_dir = artifact.download()

        #mudar aqui 
        model = CNN_simple(cnn, num_classes=4)
        model.load_state_dict(torch.load(artifact_dir+'/cnn_ssl_'+str(1)+'.pth'))


    if classify_type == 'byol':
        
        artifact = run.use_artifact('luizzanini/caries_cnn_simple/byol:v0', type='model')
        artifact_dir = artifact.download()

        #mudar aqui
        model = CNN_simple(cnn, num_classes=4)
        model.load_state_dict(torch.load(artifact_dir+'/cnn_ssl_'+str(1)+'.pth'))

    if classify_type == 'VICReg':
        
        artifact = run.use_artifact('luizzanini/caries_cnn_simple/VICReg:v0', type='model')
        artifact_dir = artifact.download()

        #mudar aqui
        model = CNN_simple(cnn, num_classes=4)
        model.load_state_dict(torch.load(artifact_dir+'/cnn_ssl_'+str(1)+'.pth'))

    
    else :

        artifact = run.use_artifact('luizzanini/caries_cnn_simple/jigsaw:v0', type='model')
        artifact_dir = artifact.download()

        model = CNN_simple(cnn, num_classes=9)
        model.load_state_dict(torch.load(artifact_dir+'/cnn_ssl_'+str(1)+'.pth'))


    model.linear2 = nn.Linear(2000, 5)

    return model.to('cuda:'+str(device))

def train_simple(batch_size, epochs, folds=5, classify_type=None, backbone='resnet18'):

    device = os.getenv('gpu')    
    api_key = os.getenv('WANDB_API_KEY')
    
    wandb.login(key=api_key)
    

    run = wandb.init(
        project="caries_cnn_simple",
        notes="first_experimental",
        name='classify'+str(classify_type),
        config = { 
            "folds": folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate_init": 0.001,
        }
    )
    
    data_train, data_test = create_train_test()

    dataset_train = ToothData(data_train)
    dataset_val = ToothData(data_test)
    
    train_data = DataLoader(dataset_train, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    val_data = DataLoader(dataset_val, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)
    
    if classify_type is None :

        model = create_model(backbone, device)
        
    else:

        #preicos mudar dps
        cnn = models.efficientnet_b0()
        cnn.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        model = model_ssl(classify_type, cnn, run, device)


    loss_function = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
    
    torch.save(model.state_dict(), './src/models/cnn_ssl_'+str(1)+'.pth')

    artifact = wandb.Artifact('classify_'+str(classify_type), type='model')
    artifact.add_file('./src/models/cnn_ssl_'+str(1)+'.pth')
    run.log_artifact(artifact)
    run.finish()
    
    return