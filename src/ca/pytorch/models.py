import torch.nn as nn

'''Minimalist neural network architectures for mimicing the Game of Life update rule.'''

class GameOfLifeModel(nn.Module):
    def __init__(self):
        super(GameOfLifeModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1)
        self.prelu = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.prelu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


class GameOfLifeModel2(nn.Module):
    def __init__(self):
        super(GameOfLifeModel2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=(1, 1))
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):   
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
