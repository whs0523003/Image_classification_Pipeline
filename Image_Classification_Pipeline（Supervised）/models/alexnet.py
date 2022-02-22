import torch.nn as nn

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