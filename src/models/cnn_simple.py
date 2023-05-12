from torch import nn
from torch.nn import functional as F

class CNN_simple(nn.Module):

    def __init__(self, cnn, num_classes=1, dp=0.3):

        super(CNN_simple, self).__init__()
        self.cnn = cnn
        self.dp = nn.Dropout(p=dp)
        self.linear = nn.Linear(1000, num_classes)
        
    def forward(self, x):

        output_cnn = self.cnn(x)
        dp_fc = self.dp(output_cnn)
        output_fc = self.linear(dp_fc)

        return output_fc