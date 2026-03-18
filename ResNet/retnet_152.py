import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import numpy as np
from PIL import Image, ImageOps


class Residual(torch.nn.Module):
    def __init__(self, input_channels, num_channels, u_1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1,
                                     stride=strides, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1,
                                     padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)
        self.conv3 = torch.nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=1, stride=1,
                                     padding=0)
        self.bn3 = torch.nn.BatchNorm2d(num_channels)
        if u_1conv:
            self.conv4 = torch.nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1,
                                         stride=strides, padding=0)
        else:
            self.conv4 = None

    def forward(self, a):
        x = self.conv1(a)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.conv4:
            y = self.conv4(a)
        else:
            y = a
        x = x + y
        return torch.nn.functional.relu(x)


class My(torch.nn.Module):
    def __init__(self):
        super(My,self).__init__()
        b1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                                 torch.nn.BatchNorm2d(64),
                                 torch.nn.ReLU(),
                                 torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        def Resnet_block(input_channels,num_channels,num_residuals,one = False):
            blk = []
            for i in range(num_residuals):
                if i==0 and not one:
                    blk.append(Residual(input_channels,num_channels,strides=2))
                else:
                    blk.append(Residual(num_channels,num_channels))
            return blk

        b2 = torch.nn.Sequential(*Resnet_block(64,256,3,True))
        b3 = torch.nn.Sequential(*Resnet_block(256, 512, 8))
        b4 = torch.nn.Sequential(*Resnet_block(512, 1024, 36))
        b5 = torch.nn.Sequential(*Resnet_block(1024, 2048, 3))

        self.net = torch.nn.Sequential(b1,b2,b3,b4,b5,
                                  torch.nn.AdaptiveAvgPool2d((1, 1)),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(2048,10))

    def forward(self, x):
        x = self.net(x)
        return x