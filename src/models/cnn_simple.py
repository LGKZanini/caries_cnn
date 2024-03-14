from torch import nn # pyright: ignore[reportMissingImports]
from torch.nn import functional as F # pyright: ignore[reportMissingImports]
import torchvision.models as models # pyright: ignore[reportMissingImports]
 
class CNN_simple(nn.Module):

    def __init__(self, cnn, input_nn, num_classes=1, dp=0.3):

        super(CNN_simple, self).__init__()
        self.cnn = cnn
        self.dp = nn.Dropout(p=dp)
        
        self.linear1 = nn.Linear(input_nn, input_nn // 2)
        self.relu = nn.ReLU()        
        self.linear2 = nn.Linear(input_nn // 2, num_classes)
        
    def forward(self, x):

        output_cnn = self.cnn(x).flatten(start_dim=1)
        
        dp_fc = self.dp(output_cnn)
        output_fc1 = self.relu(self.linear1(dp_fc))
        output_fc = self.linear2(output_fc1)

        return output_fc


def create_model(backbone, device):

    if backbone == 'resnet18':

        resnet18 = models.resnet18(pretrained=True)
        resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet18_conv = nn.Sequential(*list(resnet18.children())[:-1]).to('cuda:'+str(device))

        return CNN_simple(cnn=resnet18_conv, input_nn=512 , num_classes=5).to('cuda:'+str(device))

    
    elif backbone == 'resnet50':
        
        resnet50 = models.resnet50(pretrained=True)
        resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet50_conv = nn.Sequential(*list(resnet50.children())[:-1]).to('cuda:'+str(device))

        return CNN_simple(cnn=resnet50_conv, input_nn=2048 , num_classes=5).to('cuda:'+str(device))

    elif backbone == 'densenet121':

        densenet121 = models.densenet121(pretrained=True)
        densenet121.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        densenet121_conv = densenet121.features
        densenet121_conv.add_module('global_average_pooling', nn.AdaptiveAvgPool2d((1, 1)))
        
        return CNN_simple(cnn=densenet121_conv,input_nn=1024 , num_classes=5).to('cuda:'+str(device))
    
    else:

        vgg19 = models.vgg19(pretrained=True)
        vgg19.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        vgg19_backbone = vgg19.features
        vgg19_backbone.add_module('global_average_pooling', nn.AdaptiveAvgPool2d((1, 1)))

        return CNN_simple(cnn=vgg19_backbone, input_nn=512, num_classes=5).to('cuda:'+str(device))

    
    # Implementar dps if backbone == 'VGG19':




