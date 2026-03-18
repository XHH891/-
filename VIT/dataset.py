import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class mydata(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

        for class_name in class_names:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

def data(t,v):
    train_transform = transforms.Compose([
        transforms.CenterCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    train_dataset = mydata(root_dir=t,transform=train_transform)
    val_dataset = mydata(root_dir=v,transform=train_transform)
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
    valdataloader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=32,shuffle=False)
    return dataloader,valdataloader


