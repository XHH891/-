import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import numpy as np
from PIL import Image, ImageOps

class InceptionModule(torch.nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
        super(InceptionModule,self).__init__(**kwargs)
        self.p1_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1, stride=1, padding=0)

        self.p2_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1, stride=1, padding=0)
        self.p2_2 = torch.nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, stride=1, padding=1)

        self.p3_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1, stride=1, padding=0)
        self.p3_2 = torch.nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, stride=1, padding=2)

        self.p4_1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = torch.nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1, stride=1, padding=0)

    def forward(self,x):
        p1 = torch.nn.functional.relu(self.p1_1(x))
        p2 = torch.nn.functional.relu(self.p2_2(torch.nn.functional.relu(self.p2_1(x))))
        p3 = torch.nn.functional.relu(self.p3_2(torch.nn.functional.relu(self.p3_1(x))))
        p4 = torch.nn.functional.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim = 1)

class My(torch.nn.Module):
    def __init__(self):
        super(My,self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inceptionmodule1_1 = InceptionModule(192,64,(96,128),(16,32),32)
        self.inceptionmodule1_2 = InceptionModule(256,128,(128,192),(32,96),64)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inceptionmodule2_1 = InceptionModule(480,192,(96,208),(16,48),64)
        self.inceptionmodule2_2 = InceptionModule(512,160,(112,224),(24,64),64)
        self.inceptionmodule2_3 = InceptionModule(512,128,(128,256),(24,64),64)
        self.inceptionmodule2_4 = InceptionModule(512,112,(144,288),(32,64),64)
        self.inceptionmodule2_5 = InceptionModule(528,256,(160,320),(32,128),128)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inceptionmodule3_1 = InceptionModule(832,256,(160,320),(32,128),128)
        self.inceptionmodule3_2 = InceptionModule(832,384,(192,384),(48,128),128)
        self.avgpool1 = torch.nn.AdaptiveAvgPool2d((1,1))

        self.fc = torch.nn.Linear(1024,10)
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3 (x))
        x = self.maxpool2(x)
        x = self.inceptionmodule1_1 (x)
        x = self.inceptionmodule1_2 (x)
        x = self.maxpool3 (x)
        x = self.inceptionmodule2_1 (x)
        x = self.inceptionmodule2_2 (x)
        x = self.inceptionmodule2_3 (x)
        x = self.inceptionmodule2_4(x)
        x = self.inceptionmodule2_5 (x)
        x = self.maxpool4(x)
        x = self.inceptionmodule3_1 (x)
        x = self.inceptionmodule3_2(x)
        x = self.avgpool1 (x)
        x = torch.flatten(x,start_dim=1)
        x =self.fc(x)
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
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(32, padding=4),
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
