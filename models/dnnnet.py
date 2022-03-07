import torch.nn as nn
import torch.nn.functional as F

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