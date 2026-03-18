import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import numpy as np
from PIL import Image, ImageOps

class Residual(torch.nn.Module):
    def __init__(self,input_channels,num_channels,u_1conv=False,strides = 1):
        super(Residual, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, stride=strides, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3,stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)
        if u_1conv:
            self.conv3 = torch.nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides, padding=0)
        else:
            self.conv3 = None

    def forward(self,a):
        x = self.conv1(a)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.conv3:
            y = self.conv3(a)
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

        b2 = torch.nn.Sequential(*Resnet_block(64,64,2,True))
        b3 = torch.nn.Sequential(*Resnet_block(64, 128, 2))
        b4 = torch.nn.Sequential(*Resnet_block(128, 256, 2))
        b5 = torch.nn.Sequential(*Resnet_block(256, 512, 2))

        self.net = torch.nn.Sequential(b1,b2,b3,b4,b5,
                                  torch.nn.AdaptiveAvgPool2d((1, 1)),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(512,10))

    def forward(self, x):
        x = self.net(x)
        return x



def cs(test_dataset,my):
    a = 0
    b = 0
    my.eval()
    with torch.no_grad():
        for img, labels in test_dataset:
            img = img.cuda()
            labels = labels.cuda()
            outputs = my.forward(img)
            _, predicted = torch.max(outputs, 1)
            a += (predicted == labels).sum().item()
            b += labels.size(0)
    return a / b

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像调整为 224x224 大小
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True,transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    my = My()
    my = my.cuda()

    loss = torch.nn.CrossEntropyLoss()  # 损失函数
    loss = loss.cuda()
    optim = torch.optim.SGD(my.parameters(), lr=0.01,weight_decay=1e-4)

    for i in range(100):
        my.train()
        for j,(img,labels) in enumerate(train_loader):
            img = img.cuda()
            labels = labels.cuda()
            optim.zero_grad()
            outputs = my(img)
            dataloss = loss(outputs,labels)
            dataloss.backward()
            optim.step()
        print("第",i+1,"轮正确率为:",cs(test_loader,my))

if __name__ == "__main__":
    main()