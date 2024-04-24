import torch
import torch.nn as nn
import torch.nn.functional as F


class MandelbrotModel(nn.Module):
    '''Kernel to predict the next image in the Mandelbrot zoom sequence.'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=9, padding=4)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=9, padding=4)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=9, padding=4)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        return x