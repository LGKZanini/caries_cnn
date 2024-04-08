import os
import torch # pyright: ignore[reportMissingImports]
import wandb# pyright: ignore[reportMissingImports]


from torch import nn # pyright: ignore[reportMissingImports]
from torchvision import models # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]

from lightly.loss import NegativeCosineSimilarity, NTXentLoss, VICRegLoss # pyright: ignore[reportMissingImports]
from lightly.transforms.simclr_transform import SimCLRTransform # pyright: ignore[reportMissingImports]
from lightly.transforms.vicreg_transform import VICRegTransform # pyright: ignore[reportMissingImports]
from lightly.data import LightlyDataset, ImageCollateFunction # pyright: ignore[reportMissingImports]

from lightly.models.utils import  update_momentum # pyright: ignore[reportMissingImports]
from lightly.transforms.byol_transform import ( # pyright: ignore[reportMissingImports]
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
) # pyright: ignore[reportMissingImports]

from torchvision import transforms # pyright: ignore[reportMissingImports]

from lightly.utils.scheduler import cosine_schedule # pyright: ignore[reportMissingImports]

 
from src.models.cnn_simple import CNN_simple
from src.models.lighty import SimCLR, VICReg, BYOL

from src.loader.tooth_data import ToothDataRotate, ToothDataSSL
from src.train.train_classification import Trainer
from src.train.validation_classification import metrics_caries_rotate
from src.utils.load_data_main_cbct import make_path_ssl

from src.experiment.load_simple_train import train_simple

def make_data(path_data, batch_size):

    batch_size = int(os.getenv('BATCH_SIZE_SSL', '128'))

    paths = make_path_ssl(path_data) 

    dataset_train = ToothDataRotate(paths)
    
    train_data = DataLoader(dataset_train, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)

    return train_data


def configure_setup(epochs, batch_size, name, path_data):

    api_key = os.getenv('WANDB_API_KEY')    

    wandb.login(key=api_key)
    
    return wandb.init(
        project="caries_cnn_simple",
        notes="first_experimental",
        name=name,
        config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate_init": 0.01, 
            "path_data": path_data,
        }
    )

def make_data_loader(dataset_train, batch_size, path_data):

    train_data = DataLoader(dataset_train, batch_size=batch_size, num_workers=2, shuffle=True, pin_memory=True)

    return train_data

def get_model_raw():

    cnn = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
    cnn.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    return cnn


def get_model(metrics, learning_rate, device):

    resnet = models.resnet18()
    backbone_nn = nn.Sequential(*list(resnet.children())[:-1])

    model =  CNN_simple(cnn=backbone_nn, input_nn=2048, num_classes=5).to('cuda:'+str(device))
    
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

custom_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=112, scale=(0.6, 1.0)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.4),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
])

def custom_collate_fn(batch):

    # `batch` é uma lista de pares (imagem, label)
    imgs, labels, outro_campo = zip(*batch)
    
    # Aplica as transformações duas vezes em cada imagem para criar duas vistas
    views_1 = torch.stack([custom_transforms(img) for img in imgs])
    views_2 = torch.stack([custom_transforms(img) for img in imgs])
    
    return views_1, views_2, labels


def train_model_lighty(backbone, type_ssl, device, run, path_data):

    batch_size = int(os.getenv('BATCH_SIZE_SSL', '128'))
    epochs = int(os.getenv('EPOCHS_SSL', '500'))

    resnet18 = models.resnet18()
    resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    backbone_nn = nn.Sequential(*list(resnet18.children())[:-1]).to('cuda:'+str(device))

    if type_ssl == 'byol':

        model = BYOL(backbone_nn).to('cuda:'+str(device))
        criterion = NegativeCosineSimilarity()
        
    elif type_ssl == 'vicreg':

        model = VICReg(backbone_nn).to('cuda:'+str(device))
        criterion = VICRegLoss()

    else:

        model = SimCLR(backbone_nn).to('cuda:'+str(device))
        criterion = NTXentLoss()

    dataset = LightlyDataset(input_dir=path_data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    if type_ssl == 'byol':

        for epoch in range(epochs):

            total_loss = 0
            momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)

            for batch in dataloader:

                x0 = batch[0]
                x1 = batch[1]

                x0 = x0[:, :1, :, :].to('cuda:'+str(device))
                x1 = x1[:, :1, :, :].to('cuda:'+str(device))

                update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)
                update_momentum(model.projection_head, model.projection_head_momentum, m=momentum_val)

                p0 = model(x0)
                z0 = model.forward_momentum(x0)
                p1 = model(x1)
                z1 = model.forward_momentum(x1)
                loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
                total_loss += loss.detach()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(dataloader)
            print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    else:

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=0)
            
        for epoch in range(epochs):

            total_loss = 0

            for batch in dataloader:

                x0 = batch[0]
                x1 = batch[1]

                x0 = x0[:, :1, :, :].to('cuda:'+str(device))
                x1 = x1[:, :1, :, :].to('cuda:'+str(device))

                z0 = model(x0)
                z1 = model(x1)

                loss = criterion(z0, z1)
                total_loss += loss.detach()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(dataloader)

            print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

            scheduler.step()

    torch.save(model.state_dict(), './src/models/cnn_ssl_'+type_ssl+'_'+backbone+'_.pth')
    artifact = wandb.Artifact(type_ssl, type='model')
    artifact.add_file('./src/models/cnn_ssl_'+type_ssl+'_'+backbone+'_.pth')
    run.log_artifact(artifact)
    run.finish()

    return model.backbone


def train_model(data_train, data_val, train_cnn, epochs, run, type_ssl):

    train_cnn.train_epochs(train_data=data_train, val_data=data_val, epochs=epochs)
    
    model = train_cnn.model
    
    torch.save(model.state_dict(), './src/models/cnn_ssl_'+str(1)+'.pth')

    artifact = wandb.Artifact(type_ssl, type='model')
    artifact.add_file('./src/models/cnn_ssl_'+str(1)+'.pth')
    run.log_artifact(artifact)
    run.finish()


def train_model_rotate(backbone, type_ssl, learning_rate, device, run, dataloader):

    epochs = int(os.getenv('EPOCHS_SSL', '500'))

    resnet50 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    backbone_nn = nn.Sequential(*list(resnet50.children())[:-1]).to('cuda:'+str(device))

    model = CNN_simple(backbone_nn, input_nn=512, num_classes=4).to('cuda:'+str(device))
    
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=25, gamma=0.5)

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        total_loss = 0

        for X_train, y_train in dataloader:
        
            X_train = X_train[:, :1, :, :].to('cuda:'+str(device))
            y_train = y_train.to('cuda:'+str(device))

            y_predicted = model(X_train)

            loss = loss_function(y_predicted, y_train)  

            total_loss += loss.detach()
            
            loss.backward()

            total_loss += loss.detach()
            
            optimizer_adam.step()
            optimizer_adam.zero_grad()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), './src/models/cnn_ssl_'+type_ssl+'_'+backbone+'_.pth')
    artifact = wandb.Artifact(type_ssl, type='model')
    artifact.add_file('./src/models/cnn_ssl_'+type_ssl+'_'+backbone+'_.pth')
    run.log_artifact(artifact)
    run.finish()

    return model.cnn


def train_ssl(batch_size, epochs, type_ssl, backbone, path_data=None):
    
    run = configure_setup(epochs, batch_size, type_ssl, path_data)
    device = os.getenv('gpu')
    learning_rate = 0.001

    if type_ssl == 'rotate':

        data_train = make_data(path_data, batch_size)
        
        backbone_arch= train_model_rotate(backbone, type_ssl, learning_rate, device, run, data_train)

    elif type_ssl == 'simclr':

        backbone_arch = train_model_lighty(backbone, type_ssl, device, run, path_data)

    elif type_ssl == 'byol':

        backbone_arch = train_model_lighty(backbone, type_ssl, device, run, path_data)

    else:

        backbone_arch = train_model_lighty(backbone, type_ssl, device, run, path_data)


    train_simple(epochs=epochs, batch_size=batch_size, folds=4, classify_type=type_ssl, backbone=backbone, backbone_arch=backbone_arch, fold_ssl=path_data)

    return