import os
import sys
import numpy as np 
import torch # pyright: ignore[reportMissingImports]

from src.experiment.load_simple_train import train_simple

from src.experiment.self_supervised_train import train_ssl

if __name__ == '__main__' :
    
    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    device = sys.argv[3]
    experiment = sys.argv[3]
    
    os.environ['gpu'] = device
    
    if experiment == 'cnn':
        train_simple(epochs=epochs, batch_size=batch_size, folds=4)
    
    if experiment == 'ssl':
        train_ssl(epochs=epochs, batch_size=batch_size)
    