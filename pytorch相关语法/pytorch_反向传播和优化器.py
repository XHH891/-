import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
from PIL import Image
import os

inputs = torch.tensor([1,2,3],dtype = torch.float32)
targets = torch.tensor([1,2,5],dtype = torch.float32)

inputs = torch.reshape(inputs,(1,1,1,3))
targets = torch.reshape(targets,(1,1,1,3))

'''平均绝对误差损失函数'''
loss = torch.nn.L1Loss(reduction = 'sum')
result = loss(inputs,targets)

'''均方误差损失函数'''
loss_1 = torch.nn.MSELoss()
result_mse = loss_1(inputs,targets)

'''交叉熵损失函数'''
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
'''CrossEntropyLoss中已包含softmax函数'''
loss_2 = torch.nn.CrossEntropyLoss()
result_cross = loss_2(x,y)

optim = torch.optim.SGD(my.parameters(),lr = 0.01,)#优化器

print(result,result_mse,result_cross)