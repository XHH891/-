import torch
from torch import nn
import dataset
import model_vit

def try_gpu():
    if torch.cuda.is_available():
        print("使用GPU")
        return torch.device('cuda:0')  # 使用第一个GPU
    print("使用cpu")
    return torch.device('cpu')

def cs(test_dataset,net,devices):
    a = 0
    b = 0
    net.eval()
    with torch.no_grad():
        for img, labels in test_dataset:
            img = img.to(devices)
            labels = labels.to(devices)
            outputs = net.forward(img)
            _, predicted = torch.max(outputs, 1)
            a += (predicted == labels).sum().item()
            b += labels.size(0)
    return a / b

def tarin(tdata,vdata,net,devices,loss):
    net = net.to(devices)
    optim = torch.optim.SGD(net.parameters(), lr=0.005, weight_decay=1e-4)
    for i in range(10):
        net.train()
        for j, (img, labels) in enumerate(tdata):
            img = img.to(devices)
            labels = labels.to(devices)
            optim.zero_grad()
            outputs = net(img)
            dataloss = loss(outputs, labels)
            dataloss.backward()
            optim.step()
        print("第", i + 1, "轮正确率为:", cs(vdata,net,devices))


tdata,vdata = dataset.data(r"D:\python程序\机器学习\神经网络\香蕉成熟度\香蕉\classifier\train",
                           r"D:\python程序\机器学习\神经网络\香蕉成熟度\香蕉\classifier\val")
vitm = model_vit.vitmodle(key_size=768, query_size=768, value_size=768, num_hiddens=768,
                 norm_shape=768, ffn_num_input=768, ffn_num_hiddens=3072, num_heads=12,
                 dropout=0.1, num_layers=8,class_number=100,use_bias=False, patch=16,
                 d=768, in_chans=3, H=224,dimension=1024)
devices =try_gpu()
loss = nn.CrossEntropyLoss()
tarin(tdata,vdata,vitm,devices,loss)
torch.save(vitm.state_dict(), 'model_params.pth')