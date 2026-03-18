from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision
from PIL import Image
import os


''' 用于导入数据集 '''
class mydata (Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

a = "D:/python程序/机器学习/神经网络/数据集/a~b/Img"
b = "0"
c = "1"

zero = mydata(a,b)
one = mydata(a,c)


'''用于导入pytorch所提供的数据集'''
train_set = torchvision.datasets.CIFAR10(root = './dataset',train = True,download = True)
test_set = torchvision.datasets.CIFAR10(root = './dataset',train = False,download = True)

test_loader = DataLoader(dataset = test_set,batch_size = 4,shuffle = True,num_workers = 0,drop_last = False)

img,target = test_set[0]
img.show()
print(target)