import argparse
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import HandPredictDataset
from models.model import DNNNet, CNNNet, LeNet, AlexNet
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
    parser.add_argument('--dataset', type=str, default='handgesture', help='Training dataset')
    parser.add_argument('--test_batch_size', type=int, default=10, help='Number of images in each mini-batch')
    parser.add_argument('--test_dir', type=str, default='./inputs', help='Root for test data_raw')
    parser.add_argument('--path_state_dict', type=str, default='./checkpoints', help='Root for saved model')
    parser.add_argument('--network', type=str, default='CNN', help='The backbone of the network')
    parser.add_argument('--num_classes', type=int, default=10, help='Classes of prediction')

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
    if opt.network == 'DNN':
        print('Loading DNN_Net...')
        net = DNNNet(classes=opt.num_classes).to(device)

    elif opt.network == 'CNN':
        print('Loading CNN_Net...')
        net = CNNNet(classes=opt.num_classes).to(device)

    elif opt.network == 'LeNet':
        print('Loading Le_Net...')
        net = LeNet(classes=opt.num_classes).to(device)

    elif opt.network == 'AlexNet':
        print('Loading Alex_Net...')
        net = AlexNet(classes=opt.num_classes).to(device)

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