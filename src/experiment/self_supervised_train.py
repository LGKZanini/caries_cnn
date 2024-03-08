import os
import torch # pyright: ignore[reportMissingImports]
import wandb# pyright: ignore[reportMissingImports]


from torch import nn # pyright: ignore[reportMissingImports]
from torchvision import models # pyright: ignore[reportMissingImports]
from torch.utils.data import DataLoader # pyright: ignore[reportMissingImports]

from lightly.loss import NegativeCosineSimilarity, NTXentLoss, VICRegLoss # pyright: ignore[reportMissingImports]
from lightly.data.dataset import LightlyDataset # pyright: ignore[reportMissingImports]

from lightly.transforms.simclr_transform import SimCLRTransform # pyright: ignore[reportMissingImports]


from lightly.transforms.vicreg_transform import VICRegTransform # pyright: ignore[reportMissingImports]

from lightly.models.utils import  update_momentum # pyright: ignore[reportMissingImports]
from lightly.transforms.byol_transform import ( # pyright: ignore[reportMissingImports]
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
) # pyright: ignore[reportMissingImports]

from lightly.utils.scheduler import cosine_schedule # pyright: ignore[reportMissingImports]

 
from src.models.cnn_simple import CNN_simple
from src.models.lighty import SimCLR, VICReg, BYOL

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


def train_model_lighty(backbone, type_ssl, learning_rate, device, run, epochs):
    
    backbone_nn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to('cuda:'+str(device))

    if type_ssl == 'byol':

        model = BYOL(backbone_nn)
        criterion = NegativeCosineSimilarity()

        transform = BYOLTransform(
            view_1_transform=BYOLView1Transform(input_size=224, gaussian_blur=0.0, normalize={'mean': [86.01, 86.01, 86.01], 'std': [79.11, 79.11, 79.11]} ),
            view_2_transform=BYOLView2Transform(input_size=224, gaussian_blur=0.0, normalize={'mean': [86.01, 86.01, 86.01], 'std': [79.11, 79.11, 79.11]} ),
        )
        

    elif type_ssl == 'vicreg':

        model = VICReg(backbone_nn)
        criterion = VICRegLoss()
        transform = VICRegTransform(input_size=224, normalize={'mean': [86.01, 86.01, 86.01], 'std': [79.11, 79.11, 79.11]})

    else:

        model = SimCLR(backbone_nn)
        criterion = NTXentLoss()
        transform = SimCLRTransform(input_size=224, gaussian_blur=0.0, normalize={'mean': [86.01, 86.01, 86.01], 'std': [79.11, 79.11, 79.11]})


    folder = './data/'

    dataset = LightlyDataset(folder, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=10, gamma=0.5)

    if type_ssl == 'byol':

        for epoch in range(epochs):

            total_loss = 0
            momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)

            for batch in dataloader:

                x0, x1 = batch[0]

                print(x0.shape)

                update_momentum(model.backbone, model.backbone_momentum, m=momentum_val)

                update_momentum(
                    model.projection_head, model.projection_head_momentum, m=momentum_val
                )
                x0 = x0.to('cuda:'+str(device))
                x1 = x1.to('cuda:'+str(device))

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
            
        for epoch in range(epochs):

            total_loss = 0

            for batch in dataloader:

                x0, x1 = batch[0]
                x0 = x0.to(device)
                x1 = x1.to(device)

                z0 = model(x0)
                z1 = model(x1)

                loss = criterion(z0, z1)
                total_loss += loss.detach()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(dataloader)

            print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")


    torch.save(model.state_dict(), './src/models/cnn_ssl_'+type_ssl+'_'+backbone+'_.pth')

    artifact = wandb.Artifact(type_ssl, type='model')
    artifact.add_file('./src/models/cnn_ssl_'+str(1)+'.pth')
    run.log_artifact(artifact)
    run.finish()

    return 


def train_model(data_train, data_val, train_cnn, epochs, run, type_ssl):

    train_cnn.train_epochs(train_data=data_train, val_data=data_val, epochs=epochs)
    
    model = train_cnn.model
    
    torch.save(model.state_dict(), './src/models/cnn_ssl_'+str(1)+'.pth')

    artifact = wandb.Artifact(type_ssl, type='model')
    artifact.add_file('./src/models/cnn_ssl_'+str(1)+'.pth')
    run.log_artifact(artifact)
    run.finish()


def train_ssl(batch_size, epochs, type_ssl, backbone=None):
    
    run = configure_setup(epochs, batch_size, type_ssl)
    device = os.getenv('gpu')
    cnn = get_model_raw()
    learning_rate = 0.001


    if type_ssl == 'rotate':

        data_train, data_val = make_data(ToothDataRotate, batch_size)
        trainer = get_model(metrics_caries_rotate, cnn, num_classes=4, learning_rate=0.001, device=device)
        train_model(data_train, data_val, trainer, epochs, run, type_ssl)

    if type_ssl == 'jigsaw':

        data_train, data_val = make_data(ToothDataJigsaw, batch_size)
        trainer = get_model(metrics_caries_jigsaw, cnn, num_classes=9, learning_rate=0.001, device=device)
        train_model(data_train, data_val, trainer, epochs, run, type_ssl)

    if type_ssl == 'simclr':

        train_model_lighty(backbone, type_ssl, learning_rate, device, run, epochs)

    if type_ssl == 'byol':

        train_model_lighty(backbone, type_ssl, learning_rate, device, run, epochs)

    if type_ssl == 'vicreg':

        train_model_lighty(backbone, type_ssl, learning_rate, device, run, epochs)


    return