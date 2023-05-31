import os
import sys
import numpy as np 
import torch # pyright: ignore[reportMissingImports]

from src.experiment.load_simple_train import train_simple

if __name__ == '__main__' :
    
    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    device = sys.argv[3]
    
    os.environ['gpu'] = device
    
    train_simple(epochs=epochs, batch_size=batch_size, folds=3)
    
    