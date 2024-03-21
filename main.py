import os
import sys
import numpy as np # pyright: ignore[reportMissingImports]

from src.experiment.load_simple_train import train_simple

from src.experiment.self_supervised_train import train_ssl

if __name__ == '__main__' :

    epochs = int(os.getenv('NUM_EPOCHS', '48')) 
    batch_size = int(os.getenv('BATCH_SIZE', '200'))
    device = str(os.getenv('GPU_ID', '0'))
    experiment = os.getenv('MODEL_TYPE', 'cnn')
    type_train = os.getenv('TRAIN_TYPE', 'simple')
    backbone = os.getenv('ARCH', 'resnet18')
    path_data = os.getenv('SSL_SIZE', './data/ssl_seg_100')

    os.environ['gpu'] = device
    
    if experiment == 'cnn':

        if type_train == 'simple':

            train_simple(epochs=epochs, batch_size=batch_size, folds=4, backbone=backbone)

        else:
            
            train_simple(epochs=epochs, batch_size=batch_size, folds=4, classify_type=type_train)

    else:

        train_ssl(epochs=epochs, batch_size=batch_size, type_ssl=experiment, path_data=path_data, backbone=backbone)
    