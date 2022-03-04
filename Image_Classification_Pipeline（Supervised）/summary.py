import torch
from torchsummary import summary

from models import *

# ====================================
# 该部分代码只用于看网络结构
# ====================================

if __name__ == "__main__":
    # 判断使用gpu还是cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 实例化一个网络
    model = resnet.resnet34(classes=10).to(device)
    # 查看网络结构
    summary(model, (3, 224, 224))