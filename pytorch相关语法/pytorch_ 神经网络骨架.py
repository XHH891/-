from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import torch
from PIL import Image
import os

dataset = torchvision.datasets.CIFAR10(root = './dataset',train = False,download = True,
                                       transform = torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size = 64)

class my(torch.nn.Module):
    def __init__(self):
        super(my,self).__init__()
        ''' 一个卷积层 '''
        self.conv1 = torch.nn.Conv2d(in_channels = 3,out_channels = 6,kernel_size = 3,stride = 1,padding = 0)
        ''' 一个池化层 '''
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size = 3,ceil_mode = True)#最大池化层
        ''' 一个ReLU '''
        self.relu1 = torch.nn.ReLU()
        ''' 一个sigmoid '''
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)
        x = self.sigmoid(x)
        return x

a = my()

for data in dataloader:
    img,targets = data
    output = a(img)
    print(output.shape)