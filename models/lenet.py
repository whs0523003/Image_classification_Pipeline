import torch.nn as nn
import torch.nn.functional as F

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