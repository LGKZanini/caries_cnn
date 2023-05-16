import torch
import torchvision.transforms as T
import numpy as np

from torch.utils.data import Dataset

class ToothData(Dataset):
    
    def __init__(self, path, shape_size=(224,224)):
        
        super(Dataset, self).__init__()
        
        self.data = path
        self.shape_size = shape_size
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(shape_size),
            lambda x: x/4096
        ])
            
    def __len__(self):
        
        return len(self.data)
    
    
    def torch_y(self, value):
        
        if value == 0:
            return torch.tensor([1.0,0.0,0.0,0.0,0.0])
            
        if value == 1:
            return torch.tensor([0.0,1.0,0.0,0.0,0.0])
            
        if value == 2:
            return torch.tensor([0.0,0.0,1.0,0.0,0.0])
            
        if value == 3:
            return torch.tensor([0.0,0.0,0.0,1.0,0.0])
            
        if value == 4:
            return torch.tensor([0.0,0.0,0.0,0.0,1.0])
        
    
    def __getitem__(self, index):
        
        x = np.load(self.data[index][0])
        y = self.data[index][1]
        
        return self.transform(x), self.torch_y(y)