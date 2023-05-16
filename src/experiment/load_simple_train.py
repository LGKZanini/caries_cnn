import os
import torch

from torch import nn
from torchvision import models
from torch.utils.data import DataLoader

from src.models.cnn_simple import CNN_simple
from src.loader.tooth_data import ToothData
from src.train.train_classification import Trainer
from src.utils.load_data_main_cbct import make_folds, create_cross_val

def train_simple(batch_size, epochs, folds=5):
    
    data_folds = make_folds(total_folds=folds)
    device = os.getenv('gpu')
    
    for f in range(folds):
        
        data_train, data_val = create_cross_val(data_folds, fold=f)

        dataset_train = ToothData(data_train)
        dataset_val = ToothData(data_val)
        
        train_data = DataLoader(dataset_train, batch_size=batch_size, num_workers=2, shuffle=True)
        val_data = DataLoader(dataset_val, batch_size=batch_size, num_workers=2, shuffle=True)
        
        cnn = models.efficientnet_b0(weights='EfficientNet_B0_Weights.DEFAULT')
        cnn.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = CNN_simple(cnn, num_classes=5).to('cuda:'+str(device))
        
        loss_function = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        train_cnn = Trainer(loss_fn=loss_function, optimizer=optimizer, model=model, scheduler=scheduler, device=device)
        
        train_cnn.train_epochs(train_data=train_data, val_data=val_data, epochs=epochs)