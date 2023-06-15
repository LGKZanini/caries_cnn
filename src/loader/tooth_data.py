import random
import torch # pyright: ignore[reportMissingImports]
import torchvision.transforms as T # pyright: ignore[reportMissingImports]
import numpy as np

from torchvision.io import read_image # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset # pyright: ignore[reportMissingImports]
from scipy.ndimage import rotate # pyright: ignore[reportMissingImports]
from PIL import Image

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
        
        img = Image.open(self.data[index])
        x = rotate(img, angle=self.angle[choose_angle])

        return self.transform(x), self.torch_y(choose_angle)
class ToothDataJigsaw(Dataset):
    
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
            return torch.tensor([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], requires_grad=False)
            
        if value == 1:
            return torch.tensor([0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], requires_grad=False)
            
        if value == 2:
            return torch.tensor([0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0], requires_grad=False)
            
        if value == 3:
            return torch.tensor([0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0], requires_grad=False)

        if value == 4:
            return torch.tensor([0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0], requires_grad=False)
            
        if value == 5:
            return torch.tensor([0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0], requires_grad=False)
            
        if value == 6:
            return torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0], requires_grad=False)
            
        if value == 7:
            return torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0], requires_grad=False)

        if value == 8:
            return torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0], requires_grad=False)


    def get_idx(self, resize_to=224 ):
        
        idx_y = [0,
                (resize_to // 3),
                (int(2 * resize_to) // 3),
                resize_to]
        idx_x = [0,
                (resize_to // 3),
                (int(2 * resize_to) // 3),
                resize_to]

        return [
            [idx_y[0], idx_y[1], idx_x[0], idx_x[1]], # upper left
            [idx_y[1], idx_y[2], idx_x[0], idx_x[1]], # center left
            [idx_y[2], idx_y[3], idx_x[0], idx_x[1]], # lower left

            [idx_y[0], idx_y[1], idx_x[1], idx_x[2]], # upper center
            [idx_y[1], idx_y[2], idx_x[1], idx_x[2]], # center
            [idx_y[2], idx_y[3], idx_x[1], idx_x[2]], # lower center

            [idx_y[0], idx_y[1], idx_x[2], idx_x[3]], # upper right
            [idx_y[1], idx_y[2], idx_x[2], idx_x[3]], # center right
            [idx_y[2], idx_y[3], idx_x[2], idx_x[3]], # lower right
        ]
    
    def __getitem__(self, index):
        
        choose_jig = random.randint(0, 8)
        
        img = Image.open(self.data[index])
        idx = self.get_idx()[choose_jig]

        img_jig = img[ idx[0]:idx[1], idx[2]:idx[3], :]

        return self.transform(img_jig), self.torch_y(choose_jig)