import random
import numpy as np
import math

import torch
import torch
import torch.nn as nn


# ===================================== 设置随机种子 =====================================
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 当随机种子为0时候启用deterministic mode，速度稍慢，但是结果更易复现
    # 相当于锁死参数，不去寻找更优解
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 其他情况则不启用deterministic mode，速度稍快，但更不易复现
    # 这个设置可以让内置的cudnn的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    # 但是由于随机性，每次网络前馈的结果略有差异
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ===================================== Mixup数据增强 =====================================
# pipeline中并没有应用Mixup,使用方法见：https://github.com/facebookresearch/mixup-cifar10
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===================================== Attention机制 =====================================
# 3个注意力机制模块，SENet，CBAM和ECANet

# SEnet，考虑不同通道的权重，是通道注意力机制的典型实现，可以用于嵌入任何网络中
class SENet(nn.Module):
    def __init__(self, channel, reduction=16):  # channel为输入通道数，reduction为缩放比例
        super(SENet, self).__init__()  # 初始化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 对宽高进行池化将其都变为1，得到特征长条，条数为输入通道数

        # 全连接层，共2层
        self.fc = nn.Sequential(
            # channel为输入通道数，channel // reduction为输出通道数，不使用偏置
            # 相当于通过缩放比例减少神经元个数
            nn.Linear(channel, channel // reduction, bias=False),
            # 激活函数
            nn.ReLU(inplace=True),
            # channel // reduction为输入通道数，channel为输出通道数，不使用偏置
            nn.Linear(channel // reduction, channel, bias=False),
            # 激活函数，把值固定到0-1之间
            nn.Sigmoid()
        )

    # 将SEnet模块作用到原来的输入x里
    def forward(self, x):
        b, c, _, _ = x.size()  # b c h w
        avg = self.avg_pool(x).view(b, c)  # 首先对输入的特征层在宽高上进行全局平均池化，压缩空间深度，再reshape
        fc = self.fc(avg).view(b, c, 1, 1)  # 经过全连接层，得到每个通道权重，再reshape
        # 和输入的特征x进行点乘，应用通道注意力机制
        return x * fc.expand_as(x)  # expand_as()函数把一个tensor变成和函数括号内的tensor相同形状


# ECANet，是SENet的改进版。采用1d卷积而不是SENet中的全连接进行特征提取
class ECANet(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECANet, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))  # 根据输入通道数自适应的计算卷积核大小
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)  # 1d卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # 先进行全局平均池化
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # 调整维度并进行1d卷积
        y = self.sigmoid(y)
        return x * y.expand_as(x)  # 和输入的特征x进行点乘，应用通道注意力机制


# CBAM，是通道注意力机制和空间注意力机制的结合，可以用于嵌入任何网络中
# 首先定义通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):  # in_planes为输入通道数，reduction为缩放比例
        super(ChannelAttention, self).__init__()
        # 对宽高进行池化将其都变为1，得到特征长条，条数为输入通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)  # 第一次全连接，神经元较少
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)  # 第二次全连接

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 平均池化
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 最大池化
        out = avg_out + max_out  # 两个结果相加
        return self.sigmoid(out)


# 再定义空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):  # 卷积核大小默认为3或7，也必须为3或7
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1  # 根据卷积核大小填充

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)  # keepdim=True保存通道数以维持维度bchw
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)  # 把两个结果堆叠
        x = self.conv(x)  # 卷积
        return self.sigmoid(x)


# 将CBAM的通道注意力机制模块和空间注意力机制模块结合
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(channel, reduction=reduction)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x) * x  # 和原本的特征x进行点乘，应用通道注意力机制。（广播机制）
        x = x * self.spatialattention(x) * x  # 和经过通道注意力机制的特征x进行点乘，应用空间注意力机制。（广播机制）
        return x