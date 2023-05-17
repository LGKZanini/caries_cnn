import os
import sys
import numpy as np 
import torch

from src.experiment.load_simple_train import train_simple

if __name__ == '__main__' :
    
    device = sys.argv[2]
    batch_size = int(sys.argv[1])
    
    os.environ['gpu'] = device
    
    torch.cuda.set_device(int(device))
    
    train_simple(epochs=50, batch_size=batch_size, folds=5)
    
    