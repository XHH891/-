import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
import numpy as np
from PIL import Image, ImageOps

class My(torch.nn.Module):
    def __init__(self):
        super(My,self).__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)
        self.conv2_1 = torch.nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3,stride=2, ceil_mode=True)
        self.conv3_1 = torch.nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv3_2= torch.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3,stride=2, ceil_mode=True)
        self.fc1 = torch.nn.Linear(256 * 6 * 6, 4096)#4096
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.fc3 = torch.nn.Linear(4096, 10)
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1_1(x))
        x = self.maxpool1(x)
        x = torch.nn.functional.relu(self.conv2_1(x))
        x = self.maxpool2(x)
        x = torch.nn.functional.relu(self.conv3_1(x))
        x = torch.nn.functional.relu(self.conv3_2(x))
        x = torch.nn.functional.relu(self.conv3_3(x))
        x = self.maxpool3(x)
        x = torch.flatten(x,start_dim=1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
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
            # for i, output in enumerate(outputs):
            #     if torch.argmax(output) == labels[i]:
            #         a += 1
            #     b += 1
    return a / b

def main():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616])
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
