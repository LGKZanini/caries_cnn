import os
import sys
import numpy as np # pyright: ignore[reportMissingImports]

from src.experiment.load_simple_train import train_simple

from src.experiment.self_supervised_train import train_ssl

if __name__ == '__main__' :
    
    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    device = sys.argv[3]
    experiment = sys.argv[4]
    type_train = sys.argv[5]
    backbone = sys.argv[5]
    
    os.environ['gpu'] = device
    
    if experiment == 'cnn':

        if type_train == 'simple':

            train_simple(epochs=epochs, batch_size=batch_size, folds=4)

        else:
            
            train_simple(epochs=epochs, batch_size=batch_size, folds=4, classify_type=type_train)

    else:

        train_ssl(epochs=epochs, batch_size=batch_size, type_ssl=experiment, backbone=backbone)
    