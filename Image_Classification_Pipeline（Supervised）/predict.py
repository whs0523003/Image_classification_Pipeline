import argparse
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import HandPredictDataset
from models import dnnnet, cnnnet, lenet, alexnet, mtnet, googlenet, resnet
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=1, help='随机种子，设为0则开启deterministic mode')
    parser.add_argument('--dataset', type=str, default='handgesture', help='数据集')
    parser.add_argument('--test_dir', type=str, default='./inputs', help='测试数据文件夹')
    parser.add_argument('--test_batch_size', type=int, default=10, help='测试集batch_size')
    parser.add_argument('--path_state_dict', type=str, default='./checkpoints', help='保存模型的路径')
    parser.add_argument('--network', type=str, default='CNN', help='主干网络')
    parser.add_argument('--num_classes', type=int, default=10, help='预测类别数')

    opt = parser.parse_args()
    return opt

def train():
    # ===================================== 可选步骤 0/1 配置 =====================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CPU或GPU
    utils.set_seed(seed=opt.random_seed)  # 随机种子设置

    # ===================================== 核心步骤 1/3 数据 =====================================
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # 构建HandDataset实例
    test_data = HandPredictDataset(data_dir=opt.test_dir, transform=test_transform)

    # 构建DataLoder
    test_loader = DataLoader(dataset=test_data, batch_size=opt.test_batch_size, shuffle=False)

    # ===================================== 核心步骤 2/3 模型 =====================================
    # 实例化网络并把网络放到合适的设备上
    if opt.network == 'dnnnet':
        print('Loading DNN_Net...')
        # 对网络进行操作时是inplace，直接把网络放到指定device上就可以生效
        net = dnnnet.DNNNet(classes=opt.num_classes).to(device)

    elif opt.network == 'cnnnet':
        print('Loading CNN_Net...')
        net = cnnnet.CNNNet(classes=opt.num_classes).to(device)

    elif opt.network == 'lenet':
        print('Loading Le_Net...')
        net = lenet.LeNet(classes=opt.num_classes).to(device)

    elif opt.network == 'alexnet':
        print('Loading Alex_Net...')
        net = alexnet.AlexNet(classes=opt.num_classes).to(device)

    elif opt.network == 'mtnet':
        print('Loading MT_Net...')
        net = mtnet.MT_CNN(classes=opt.num_classes).to(device)

    # 需要设置初始参数
    elif opt.network == 'googlenet':
        print('Loading Google_Net...')
        net = googlenet.GoogLeNet(classes=opt.num_classes).to(device)

    # 需要设置初始参数
    elif opt.network == 'resnet':
        print('Loading Res_Net...')
        net = resnet.ResNeXt(classes=opt.num_classes).to(device)

    # 找到保存的模型文件
    model_file = os.listdir(opt.path_state_dict)
    for file in model_file:
        if file.startswith(opt.dataset + '_' + opt.network + '_Model_' + 'BestEpoch'):
            model_file = file

    # 往实例化的网络中加载训练好的参数
    state_dict_load = torch.load(os.path.join(opt.path_state_dict, model_file), map_location=device)
    net.load_state_dict(state_dict_load)

    # ===================================== 核心步骤 3/3 预测 =====================================
    # 模型预测
    net.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs = data
            inputs = inputs.to(device)  # 数据放到指定device上
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

        print("Predict label:\n{}".format(predicted))


if __name__ == '__main__':
    opt = parse_args()
    train()