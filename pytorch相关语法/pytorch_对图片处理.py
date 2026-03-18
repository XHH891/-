import torch
from torchvision import transforms
from PIL import Image

'''
ToTensor 将图像转换为计算机能处理的张量形式
Normalize 对张量进行归一化处理
resize 改变图像的尺寸大小
Compose  多个图像变换操作组合成一个序列,一次性对图像执行多个预处理步骤
RandomCrop  对输入的图像进行随机裁剪
...
'''

# 定义一个转换操作，将图像转换为张量并进行归一化
transform = transforms.Compose([
    transforms.RandomCrop((224, 224)),
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 打开一个示例图像
image = Image.open('D:/python程序/机器学习/神经网络/one/01x1.png ')

# 应用转换操作
tensor_image = transform(image)#表示对图片分步进行处理

img,i = tensor_image
img.show()


