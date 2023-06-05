import random
import torch # pyright: ignore[reportMissingImports]
import torchvision.transforms as T # pyright: ignore[reportMissingImports]
import numpy as np

from torchvision.io import read_image # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset # pyright: ignore[reportMissingImports]
from scipy.ndimage import rotate # pyright: ignore[reportMissingImports]

class ToothData(Dataset):
    
    def __init__(self, path, shape_size=(224,224)):
        
        super(Dataset, self).__init__()
        
        self.data = path
        self.shape_size = shape_size
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(shape_size, T.InterpolationMode.BILINEAR, antialias=True),
            lambda x: x/4096
        ])
            
    def __len__(self):
        
        return len(self.data)
    
    
    def torch_y(self, value):
        
        if value == 0:
            return torch.tensor([1.0,0.0,0.0,0.0,0.0], requires_grad=False)
            
        if value == 1:
            return torch.tensor([0.0,1.0,0.0,0.0,0.0], requires_grad=False)
            
        if value == 2:
            return torch.tensor([0.0,0.0,1.0,0.0,0.0], requires_grad=False)
            
        if value == 3:
            return torch.tensor([0.0,0.0,0.0,1.0,0.0], requires_grad=False)
            
        if value == 4:
            return torch.tensor([0.0,0.0,0.0,0.0,1.0], requires_grad=False)
        
    
    def __getitem__(self, index):
        
        x = np.load(self.data[index][0])
        y = self.data[index][1]
        
        return self.transform(x), self.torch_y(y)
    
    
class ToothDataRotate(Dataset):
    
    def __init__(self, path, shape_size=(224,224)):
        
        super(Dataset, self).__init__()
        
        self.data = path
        self.shape_size = shape_size
        self.angle = [0, 90, 180, 270]
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Resize(shape_size, T.InterpolationMode.BILINEAR, antialias=True),
            lambda x: x/255
        ])
            
    def __len__(self):
        
        return len(self.data)
    
    
    def torch_y(self, value):
        
        if value == 0:
            return torch.tensor([1.0,0.0,0.0,0.0], requires_grad=False)
            
        if value == 1:
            return torch.tensor([0.0,1.0,0.0,0.0], requires_grad=False)
            
        if value == 2:
            return torch.tensor([0.0,0.0,1.0,0.0], requires_grad=False)
            
        if value == 3:
            return torch.tensor([0.0,0.0,0.0,1.0], requires_grad=False)
        
    
    def __getitem__(self, index):
        
        choose_angle = random.randint(0, 3)
        
        x = read_image(self.data[index])
        x = rotate(x, angle=self.angle[choose_angle])

        return self.transform(x), self.torch_y(choose_angle)