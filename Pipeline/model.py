import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from collections import OrderedDict



class DNNNet(nn.Module):
    def __init__(self, classes):
        super(DNNNet, self).__init__()
        # 全连接层1
        self.fc1 = nn.Linear(3 * 32 * 32, 512)
        # 全连接层2
        self.fc2 = nn.Linear(512, 512)
        # 全连接层3
        self.fc3 = nn.Linear(512, classes)

    def forward(self, x):
        '''
        这里x.view()是把图像拉平
        x.size(0)代表第一个维度，即Batch size，
        -1代表后边的维度，即C H W
        最后拉成的维度为(B, C*H*W)，这里不用不直接用x.view(-1)是因为不能把batch也给拉平进维度
        '''
        x = x.view(x.size(0), -1)        # B 3 32 32  -> out.view   -> B 3072

        # 全连接层1
        x = F.relu(self.fc1(x))          # B 3072     -> fc1        -> B 512
        # 全连接层2
        x = F.relu(self.fc2(x))          # B 512      -> fc2        -> B 512
        # 全连接层3
        x = F.relu(self.fc3(x))          # B 512      -> fc3        -> B 10
        return x


class CNNNet(nn.Module):
    def __init__(self, classes):  # 修改这里的classes就可以改变最后分类的种类
        super(CNNNet, self).__init__()
        # 输入通道数3，输出的通道数6，卷积核的大小5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 输入通道数6，输出通道数16，卷积核的大小5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 这里的16是通道，但5*5是图片尺寸，是经过(w+2p-f)/s + 1)这个公式一层层计算得到的，和卷积核大小相同只是一个巧合
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))        # B 3 32 32  -> nn.Conv2d(3, 6, 5)  -> B 6  28 28
        x = F.max_pool2d(x, 2)           # B 6 28 28  -> max_pooling         -> B 6  14 14
        x = F.relu(self.conv2(x))        # B 6 14 14  -> nn.Conv2d(6, 16, 5) -> B 16 10 10
        x = F.max_pool2d(x, 2)           # B 16 10 10 -> max_pooling         -> B 16 5 5

        # 拉平进入全连接层
        x = x.view(x.size(0), -1)        # B 16 5 5   -> out.view            -> B 400

        x = self.fc1(x)                  # B 400      -> fc1                 -> B 120
        x = self.fc2(x)                  # B 120      -> fc2                 -> B 84
        x = self.fc3(x)                  # B 84       -> fc3                 -> B classes
        return x

    # 不适用随机初始化，而是使用特定的初始化，控制方差，防止网络梯度消失和爆炸
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


class AlexNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU())

        self.max_pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, 1),
            nn.ReLU(True))

        self.max_pool2 = nn.MaxPool2d(3, 2)

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 384),
            nn.ReLU(True))

        self.fc2 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(True))

        self.fc3 = nn.Linear(192, classes)

    def forward(self, x):
        x = self.conv1(x)                # B 3 32 32  -> nn.Conv2d(3, 6, 5)  -> B 6 28 28
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)

        # 将图片矩阵拉平
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class LeNet(nn.Module):
    def __init__(self, classes):
        super(LeNet, self).__init__()
        # 输入通道数3，输出通道数6，卷积核大小5，输出大小根据公式得：(32+2*0-5)/1+1 = 28
        # 这里的图片大小跟卷积层设计没有关系，只要看通道数即可，算下一层输出大小才会用到图片大小
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 输入通道数6，输出通道数16，卷积核大小5，输出大小根据公式得：(14+2*0-5)/1+1 = 10（这里的14是经过最大池化后的）
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))      # B 3 32 32  -> nn.Conv2d(3, 6, 5)  -> B 6 28 28
        out = F.max_pool2d(out, 2)       # B 6 28 28  -> max_pooling         -> B 6 14 14
        out = F.relu(self.conv2(out))    # B 6 14 14  -> nn.Conv2d(6, 16, 5) -> B 16 10 10
        out = F.max_pool2d(out, 2)       # B 16 10 10 -> max_pooling         -> B 16 5 5
        out = out.view(out.size(0), -1)  # B 16 5 5   -> out.view            -> B 400
        out = F.relu(self.fc1(out))      # B 400      -> fc1                 -> B 120
        out = F.relu(self.fc2(out))      # B 120      -> fc2                 -> B 84
        out = self.fc3(out)              # B 84       -> fc3                 -> B classes
        return out


# 这个网络相当大
class MT_CNN(nn.Module):
    """
    CNN from Mean Teacher paper
    """

    def __init__(self, classes, isL2 = False, dropRatio = 0.0):
        super(MT_CNN, self).__init__()

        self.isL2 = isL2

        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.drop1  = nn.Dropout(0.5)
        # self.drop1  = nn.Dropout(dropRatio)
        self.drop  = nn.Dropout(dropRatio)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.drop2  = nn.Dropout(0.5)
        # self.drop2  = nn.Dropout(dropRatio)

        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 =  weight_norm(nn.Linear(128, classes))
        self.fc2 =  weight_norm(nn.Linear(128, classes))

    def forward(self, x, debug=False):
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop(x)

        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop(x)

        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)

        x = x.view(-1, 128)
        if self.isL2:
            x = F.normalize(x)
        # return self.fc1(x), self.fc2(x), x
        return self.fc1(x)  #, self.fc2(x), x