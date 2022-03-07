import torch.nn as nn
import torch.nn.functional as F

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

    # 训练技巧，不使用随机初始化，而是使用特定的初始化，控制方差，防止网络梯度消失和爆炸
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