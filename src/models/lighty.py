
import copy

import torch # pyright: ignore[reportMissingImports]
import torchvision # pyright: ignore[reportMissingImports]
from torch import nn # pyright: ignore[reportMissingImports]

from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead # pyright: ignore[reportMissingImports]
from lightly.models.modules.heads import VICRegProjectionHead # pyright: ignore[reportMissingImports]
from lightly.models.modules import SimCLRProjectionHead # pyright: ignore[reportMissingImports] 
from lightly.models.utils import deactivate_requires_grad # pyright: ignore[reportMissingImports]
 

class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


class VICReg(nn.Module):
    def __init__(self, backbone):

        super().__init__()

        self.backbone = backbone

        self.projection_head = VICRegProjectionHead(
            input_dim=512,
            hidden_dim=2048,
            output_dim=2048,
            num_layers=2,
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

        self.projection_head = BYOLProjectionHead(2048, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)

        return p

    def forward_momentum(self, x):

        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()

        return z