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
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=2,stride = 2, ceil_mode=True)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,stride = 1,padding = 0)
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=2,stride = 2, ceil_mode=True)
        self.fc1 = torch.nn.Linear(16*5*5,120)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.fc2 = torch.nn.Linear(120, 84)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.fc3 = torch.nn.Linear(84,10)
    def forward(self, x):
        x = torch.nn.functional.sigmoid(self.conv1(x))
        x = self.avgpool1(x)
        x = torch.nn.functional.sigmoid(self.conv2(x))
        x = self.avgpool2(x)
        x = torch.flatten(x,start_dim=1)
        x = torch.nn.functional.sigmoid(self.fc1(x))
        #x = self.dropout1(x)
        x = torch.nn.functional.sigmoid(self.fc2(x))
        #x = self.dropout2(x)
        x = self.fc3(x)
        return x

def cs(test_dataset,my):
    a = 0
    b = 0
    my.eval()
    with torch.no_grad():
        for img,labels in test_dataset:
            '''加入GPU'''
            img = img.cuda()
            labels = labels.cuda()
            outputs = my.forward(img)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == labels[i]:
                    a += 1
                b += 1
    return a/b

def p(a):#图片处理
    img = Image.open(a).convert("L")
    img = ImageOps.invert(img)
    img_array = np.array(img)
    non_zero_cols = np.where(img_array.max(axis=0) > 0)[0]
    non_zero_rows = np.where(img_array.max(axis=1) > 0)[0]
    left, right = non_zero_cols[0], non_zero_cols[-1]
    top, bottom = non_zero_rows[0], non_zero_rows[-1]
    cropped = img.crop((left, top, right + 1, bottom + 1))
    size = max(cropped.width, cropped.height) + 10
    squared = Image.new("L", (size, size), 0)
    offset_x = (size - cropped.width) // 2
    offset_y = (size - cropped.height) // 2
    squared.paste(cropped, (offset_x, offset_y))
    resized = squared.resize((28, 28), Image.LANCZOS)
    transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    tensor_img = transform(resized)
    return tensor_img

def main():
    transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)#训练
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)#测试
    train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False)

    my = My()
    '''加入GPU'''
    my = my.cuda()

    loss = torch.nn.CrossEntropyLoss()#损失函数
    '''加入GPU'''
    loss = loss.cuda()
    optim = torch.optim.SGD(my.parameters(),lr = 0.09,weight_decay=1e-4)#优化器   随机梯度下降
    my.train()
    for i in range(5):
        for j,(img,labels) in enumerate(train_loader):
            '''加入GPU'''
            img = img.cuda()
            labels = labels.cuda()
            optim.zero_grad()
            outputs = my(img)
            dataloss = loss(outputs,labels)
            dataloss.backward()
            optim.step()
        print("第",i+1,"轮正确率为:",cs(test_loader,my))
    while 1:
        dz = input("请输入图片保存地址:")
        data = p(dz)
        data = data.cuda()
        my.eval()
        with torch.no_grad():
            output = my(data.unsqueeze(0))
            predict = torch.argmax(output).item()
        print(f"预测结果: {predict}")
        a = input("是否继续?(y/n")
        if a == 'n' or a =='N':
            break


if __name__ == "__main__":
    main()


