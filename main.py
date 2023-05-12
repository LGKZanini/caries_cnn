import os
import sys
import numpy as np 

from src.experiment.load_simple_train import train_simple

if __name__ == '__main__' :
    
    batch_size = sys.argv[2]
    
    os.environ['gpu'] = '0'
    
    train_simple(epochs=50, batch_size=batch_size, folds=5)
    
    