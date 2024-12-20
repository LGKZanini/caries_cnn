import random
import torch # pyright: ignore[reportMissingImports]
import torchvision.transforms as T # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]

from torchvision.io import read_image # pyright: ignore[reportMissingImports]
from torch.utils.data import Dataset # pyright: ignore[reportMissingImports]
from scipy.ndimage import rotate # pyright: ignore[reportMissingImports]
from PIL import Image

class ToothData(Dataset):
    
    def __init__(self, path):
        
        super(Dataset, self).__init__()
        
        self.data = path
        
        self.transform = T.Compose([
            T.ToTensor(),
        ])
            
    def __len__(self):
        
        return len(self.data)
    
    
    def torch_y(self, value):

        array = [ 0.0 for i in range(5)]
        array[value] = 1.0

        return torch.tensor(array, requires_grad=False)
        
    
    def __getitem__(self, index):
        
        x = np.load(self.data[index][0])
        y = int(self.data[index][1])
        
        return self.transform(x).float(), self.torch_y(y).float()
    
    
class ToothDataRotate(Dataset):
    
    def __init__(self, path):
        
        super(Dataset, self).__init__()
        
        self.data = path
        self.angle = [0, 90, 180, 270]
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
        ])
            
    def __len__(self):
        
        return len(self.data)
    
    
    def torch_y(self, value):
        
        array = [ 0.0 for i in range(4)]
        array[value] = 1.0

        return torch.tensor(array, requires_grad=False)
        
    
    def __getitem__(self, index):
        
        choose_angle = random.randint(0, 3)
        
        img = Image.open(self.data[index])
        x = rotate(img, angle=self.angle[choose_angle])

        return self.transform(x), self.torch_y(choose_angle)


class ToothDataSSL(Dataset):
    
    def __init__(self, path, transform):
        
        super(Dataset, self).__init__()
        
        self.data = np.load(path+'/save.npy')
        self.transform = transform
            
    def __len__(self):
        
        return len(self.data)
        
    
    def __getitem__(self, index):
        
        x = np.load(self.data[index])
        x = torch.from_numpy(x)

        return self.transform(x)

class ToothDataJigsaw(Dataset):
    
    def __init__(self, path, shape_size=(224,224)):
        
        super(Dataset, self).__init__()
        
        self.data = path
        self.angle = [0, 90, 180, 270]
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize(shape_size, T.InterpolationMode.BILINEAR, antialias=True),
        ])
            
    def __len__(self):
        
        return len(self.data)
    
    
    def torch_y(self, value):
        
        array = [ 0.0 for i in range(9)]
        array[value] = 1.0

        return torch.tensor(array, requires_grad=False)


    def get_idx(self, resize_y=224, resize_x=224):
        
        idx_y = [0,
                (resize_y // 3),
                (int(2 * resize_y) // 3),
                resize_y]
        idx_x = [0,
                (resize_x // 3),
                (int(2 * resize_x) // 3),
                resize_x]

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
        
        img = np.array(Image.open(self.data[index]))
        y, x = img.shape

        idx = self.get_idx(y, x)[choose_jig]

        img_jig = img[ idx[0]:idx[1], idx[2]:idx[3]]

        return self.transform(img_jig), self.torch_y(choose_jig)