from torch import nn # pyright: ignore[reportMissingImports]
from torch.nn import functional as F # pyright: ignore[reportMissingImports]
import torchvision.models as models # pyright: ignore[reportMissingImports]

 
class CNN_simple(nn.Module):

    def __init__(self, cnn, input_nn, num_classes=1, dp=0.3):

        super(CNN_simple, self).__init__()
        self.cnn = cnn
        self.dp = nn.Dropout(p=dp)
        
        self.linear1 = nn.Linear(input_nn, input_nn // 2,  bias=True)
        self.bn1 = nn.BatchNorm1d(num_features= input_nn // 2)  

        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

        self.linear2 = nn.Linear(in_features=input_nn // 2, out_features=input_nn // 4, bias=True)  # Adicionando mais uma camada intermediária
        self.bn2 = nn.BatchNorm1d(num_features=input_nn // 4)  # Batch Normalization após a segunda camada linear

        self.linear3 = nn.Linear(input_nn // 4, num_classes,  bias=True)
        
    def forward(self, x):

        x = self.cnn(x).flatten(start_dim=1)
        
        x = self.dp(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.lrelu(x) 
        x = self.dp(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.lrelu(x)  
        x = self.linear3(x)

        return x

class InceptionGrayScaleAdapter(nn.Module):
                
    def __init__(self, original_model):
        super(InceptionGrayScaleAdapter, self).__init__()
        self.gray_to_rgb = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)  # Adapta de 1 para 3 canais
        self.original_model = original_model
    
    def forward(self, x):
        x = self.gray_to_rgb(x)  # Converte escala de cinza para "RGB"
        x = self.original_model(x)  # Passa pela InceptionV3 original
        return x 

def create_model(backbone, device, backbone_arch=None):

    if backbone == 'resnet18':

        if backbone_arch is None:

            resnet18 = models.resnet18(pretrained=True)
            resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            resnet18_conv = nn.Sequential(*list(resnet18.children())[:-1]).to('cuda:'+str(device))

            return CNN_simple(cnn=resnet18_conv, input_nn=512 , num_classes=5).to('cuda:'+str(device))

        else:

            return CNN_simple(cnn=backbone_arch, input_nn=512 , num_classes=5).to('cuda:'+str(device))
            
    
    elif backbone == 'resnet50':
        
        if backbone_arch is None:

            resnet50 = models.resnet50(pretrained=True)
            resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            resnet50_conv = nn.Sequential(*list(resnet50.children())[:-1]).to('cuda:'+str(device))

            return CNN_simple(cnn=resnet50_conv, input_nn=2048 , num_classes=5).to('cuda:'+str(device))

        else:

            return CNN_simple(cnn=backbone_arch, input_nn=2048 , num_classes=5).to('cuda:'+str(device))

    elif backbone == 'densenet121':

        if backbone_arch is None:
            
            densenet121 = models.densenet121(pretrained=True)
            densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            densenet121_conv = densenet121.features
            densenet121_conv.add_module('global_average_pooling', nn.AdaptiveAvgPool2d((1, 1)))
            
            return CNN_simple(cnn=densenet121_conv, input_nn=1024 , num_classes=5).to('cuda:'+str(device))

        else:

            return CNN_simple(cnn=backbone_arch, input_nn=1024 , num_classes=5).to('cuda:'+str(device))
    
    else:

        if backbone_arch is None:
            
            inception = models.inception_v3(aux_logits=False)
            inception.fc = nn.Identity()  # Remove a última camada FC
            
            adapted_inception = InceptionGrayScaleAdapter(inception)

            return CNN_simple(cnn=adapted_inception, input_nn=2048, num_classes=5).to('cuda:'+str(device))

        else:

            return CNN_simple(cnn=backbone_arch, input_nn=2048, num_classes=5).to('cuda:'+str(device))

    
    # Implementar dps if backbone == 'vgg16':




