import argparse
import os
import torch
import time

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2
from PIL import Image

from dataset import HandPredictDataset
from models import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    # ==================================== 配置 ====================================
    parser.add_argument('--random_seed', type=int, default=1, help='随机种子，设为0则开启deterministic mode')
    parser.add_argument('--test_batch_size', type=int, default=1, help='一次测试多少张图片')
    parser.add_argument('--num_worker', type=int, default=0, help='和cpu/gpu几核有关，设的越大越吃显存')
    # ==================================== 数据集 ====================================
    parser.add_argument('--dataset', type=str, default='cifar10', help='数据集，为了找到保存文件路径')
    parser.add_argument('--test_dir', type=str, default='./inputs', help='测试数据文件夹')
    # ==================================== 网络 ====================================
    parser.add_argument('--path_state_dict', type=str, default='./checkpoints', help='保存模型的路径')
    parser.add_argument('--network', type=str, default='resnet34', help='主干网络')
    parser.add_argument('--num_classes', type=int, default=6, help='预测类别数')
    # ==================================== 优化器 ====================================
    parser.add_argument('--optimizer', type=str, default='sgd', help='优化器选择，可选sgd或adam，为了找到保存文件路径')
    # ==================================== 载入预训练模型 ====================================
    parser.add_argument('--load_model', type=int, default=2, help='选择载入的预训练模型，1为下载的，2为之前训练好的')
    parser.add_argument('--weight', type=str, default='cifar10_resnet34_sgd_Model_BestEpoch_5_TestLoss_5.5191_TestAcc_0.32.pth', help='载入预训练权重的文件名')

    opt = parser.parse_args()
    return opt

def train():
    # ===================================== 可选步骤 0/1 配置 =====================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CPU或GPU
    set_seed(seed=opt.random_seed)  # 随机种子设置

    # 标签和名称一一对应
    img_label = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

    # ===================================== 核心步骤 1/3 数据 =====================================
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    # 构建HandDataset实例
    test_data = HandPredictDataset(data_dir=opt.test_dir, transform=test_transform)

    # 构建DataLoder
    test_loader = DataLoader(dataset=test_data, batch_size=opt.test_batch_size,
                             shuffle=False, num_workers=opt.num_worker)

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

    # 需要更改输入图片尺寸
    elif opt.network == 'mobilenetv1':
        print('Loading Mobile_Net...')
        net = mobilenetv1.MobileNetV1(classes=opt.num_classes).to(device)

    # 需要设置初始参数
    elif opt.network == 'googlenet':
        print('Loading Google_Net...')
        net = googlenet.GoogLeNet(classes=opt.num_classes).to(device)

    # 需要设置初始参数
    elif opt.network == 'resnet34':
        print('Loading Res_Net34...')
        net = resnet.resnet34(classes=opt.num_classes).to(device)

    if opt.load_model == 1:  # 1为使用下载的权重
        # 找到checkpoints里所有保存的模型文件
        model_file = os.listdir(opt.path_state_dict)
        for file in model_file:
            # 找到满足以下格式的那个
            if file.startswith(opt.weight):
                model_file = file

        # 读取权重文件
        state_dict_load = torch.load(os.path.join(opt.path_state_dict, model_file), map_location=device)
        # 建立映射关系
        model_dict = net.state_dict()
        # 重新制作预训练的权重
        pretrained_dict = {k: v for k, v in state_dict_load.items() if (k in model_dict and 'fc' not in k)}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    elif opt.load_model == 2:  # 2为之前训练的权重
        # 找到checkpoints里保存的之前训练的测试集上正确率最高的epoch的权重
        model_file = os.listdir(opt.path_state_dict)
        for file in model_file:
            # 找到满足以下格式的那个
            if file.startswith(opt.dataset + '_' + opt.network + '_' + opt.optimizer + '_Model_' + 'BestEpoch'):
                model_file = file

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
            predicted = img_label[predicted.numpy().item()]  # tensor转int，并根据img_label一一对应
            print("Predict label:\n{}".format(predicted))

if __name__ == '__main__':
    opt = parse_args()
    train()