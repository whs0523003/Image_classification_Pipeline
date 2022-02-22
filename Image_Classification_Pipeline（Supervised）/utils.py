import random
import numpy as np
import torch


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
# ========================================================================================