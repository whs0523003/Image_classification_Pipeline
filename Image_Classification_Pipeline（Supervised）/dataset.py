import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2


# 读取训练集
class HandDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_info = self.get_train_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，改变尺寸，转为tensor等等
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_train_img_info(data_dir):
        data_info = []
        # root即原本的data_dir路径，dirs即['1', '100']，files即文件夹1和100里的文件（这里不用files，所以填_)
        for root, dirs, _ in os.walk(data_dir):  # 输出在文件夹中的文件名
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.JPG'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = sub_dir
                    # 最终读取的数据为格式为[(path_img1, label), (path_img2, label) ...]
                    # 这里的path_img其实就是单纯的图片路径加名字，之后的操作交给Dataloader来做
                    data_info.append((path_img, int(label)))

        return data_info


# 读取测试集
class HandTestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_info = self.get_test_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，改变尺寸，转为tensor等等
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_test_img_info(data_dir):
        data_info = []
        for img in os.listdir(data_dir):
            path_img = os.path.join(data_dir, img)
            label = img[-5]
            data_info.append((path_img, int(label)))

        return data_info


# 读取待测数据
class HandPredictDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_info = self.get_test_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，改变尺寸，转为tensor等等
        return img

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_test_img_info(data_dir):
        data_info = []
        for img in os.listdir(data_dir):
            path_img = os.path.join(data_dir, img)
            data_info.append(path_img)

        return data_info

# CIFAR-10和MNIST可以通过pytorch自带的函数torchvision.datasets.CIFAR10和torchvision.datasets.MNIST读取




