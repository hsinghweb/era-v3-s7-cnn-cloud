import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

dropout_value = 0.02

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block with dilated convolution
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=2, dilation=2, bias=False),
            nn.LeakyReLU(0.15),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        )
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=22, kernel_size=(3, 3), padding=1, bias=False),
            nn.LeakyReLU(0.15),
            nn.BatchNorm2d(22),
            nn.Dropout(dropout_value)
        )

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=22, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.15)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)

        # CONVOLUTION BLOCK 2 with grouped convolution
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=22, kernel_size=(3, 3), padding=1, groups=2, bias=False),
            nn.LeakyReLU(0.15),            
            nn.BatchNorm2d(22),
            nn.Dropout(dropout_value)
        )

        self.pool2 = nn.MaxPool2d(2, 2)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=1, padding=1)

        # Depthwise Separable Convolution
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=22, out_channels=22, kernel_size=(3, 3), padding=1, groups=22, bias=False),
            nn.Conv2d(in_channels=22, out_channels=16, kernel_size=(1, 1), bias=False),
            nn.LeakyReLU(0.15),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )

        self.pool3 = nn.MaxPool2d(2, 2)

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),
            nn.LeakyReLU(0.15),            
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        )

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.LeakyReLU(0.15),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        )
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool0(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.avgpool1(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.avgpool2(x)
        x = self.convblock5(x)
        x = self.pool3(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
