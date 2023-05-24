from torch import nn # pyright: ignore[reportMissingImports]
from torch.nn import functional as F # pyright: ignore[reportMissingImports]

class CNN_simple(nn.Module):

    def __init__(self, cnn, num_classes=1, dp=0.3):

        super(CNN_simple, self).__init__()
        self.cnn = cnn
        self.dp = nn.Dropout(p=dp)
        
        self.linear1 = nn.Linear(1000, 2000)
        self.relu = nn.ReLU()        
        self.linear2 = nn.Linear(2000, num_classes)
        
    def forward(self, x):

        output_cnn = self.cnn(x)
        
        dp_fc = self.dp(output_cnn)
        output_fc1 = self.relu(self.linear1(dp_fc))
        output_fc = self.linear2(output_fc1)

        return output_fc