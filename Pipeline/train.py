import argparse
import sys
import os
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.datasets
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import HandDataset, HandTestDataset
from model import DNNNet, CNNNet, LeNet, AlexNet, MT_CNN

import utils
import cifar10_parse # 仅CIFAR-10自定义读取时使用


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
    parser.add_argument('--dataset', type=str, default='handgesture', help='Training dataset')
    parser.add_argument('--train_dir', type=str, default='../datasets/HandGesture/data/train', help='Root for train data')
    parser.add_argument('--test_dir', type=str, default='../datasets/HandGesture/data/test', help='Root for test data')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--train_batch_size', type=int, default=10, help='Number of images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=5, help='Number of images in each mini-batch')
    parser.add_argument('--network', type=str, default='CNN', help='The backbone of the network')
    parser.add_argument('--net_initialize', type=bool, default=False, help='Special method to initialize the parameter of network')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=10, help='Classes of prediction')
    parser.add_argument('--path_state_dict', type=str, default='./checkpoints', help='Root for saved model')
    parser.add_argument('--save_model', type=bool, default=False, help='Save model or not')
    parser.add_argument('--save_metrics', type=bool, default=False, help='Save metrics or not')
    parser.add_argument('--plot', type=bool, default=False, help='Plot or not')

    opt = parser.parse_args()
    return opt


def train():
    # ===================================== 可选步骤 0/3 配置 =====================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CPU或GPU
    utils.set_seed(seed=opt.random_seed)  # 随机种子设置

    # ===================================== 核心步骤 1/5 数据 =====================================
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    if opt.dataset == 'handgesture':
        # 构建HandDataset实例
        train_data = HandDataset(data_dir=opt.train_dir, transform=train_transform)
        test_data = HandTestDataset(data_dir=opt.test_dir, transform=test_transform)

        # 构建DataLoder
        train_loader = DataLoader(dataset=train_data, batch_size=opt.train_batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_data, batch_size=opt.test_batch_size, shuffle=False, num_workers=0)

    elif opt.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root='../datasets/CIFAR-10', train=True, download=False, transform=train_transform)
        test_data = torchvision.datasets.CIFAR10(root='../datasets/CIFAR-10', train=False, download=False, transform=test_transform)

        # 自定义读取方法，可以是随机读一部分，也可以是读其中几个类
        train_data = cifar10_parse.custom_cifar10(train_data, 100)
        test_data = cifar10_parse.custom_cifar10(test_data, 10)

        train_loader = DataLoader(dataset=train_data, batch_size=opt.train_batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_data, batch_size=opt.test_batch_size, shuffle=False, num_workers=0)

    elif opt.dataset == 'mnist':  # MNIST是单通道网络，需要修改网络通道数，只需把3改成1即可
        train_data = torchvision.datasets.MNIST(root='../datasets/MNIST', train=True, download=False, transform=train_transform)
        test_data = torchvision.datasets.MNIST(root='../datasets/MNIST', train=False, download=False, transform=test_transform)

        train_loader = DataLoader(dataset=train_data, batch_size=opt.train_batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(dataset=test_data, batch_size=opt.test_batch_size, shuffle=False, num_workers=0)

    # ===================================== 核心步骤 2/5 模型 =====================================
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

    # 网络权重初始化，initialize_weights定义在网络里，加上可以缓解网络梯度消失和爆炸，一定程度上提升网络性能。不是必要操作
    if opt.net_initialize:
        net.initialize_weights()

    # ===================================== 核心步骤 3/5 损失函数 =====================================
    criterion = nn.CrossEntropyLoss()  # 选择损失函数

    # ===================================== 核心步骤 4/5 优化器 =====================================
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)  # 选择优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

    # ===================================== 核心步骤 5/5 训练 =====================================
    # 存储每个iter的结果，过渡计算使用
    train_loss_iter = []
    train_acc_iter = []
    test_loss_iter = []
    test_acc_iter = []

    # 存储最终每个epoch的结果
    train_loss_epoch = []
    train_acc_epoch = []
    test_loss_epoch = []
    test_acc_epoch = []

    count = 0  # 用来保存最优模型

    for epoch in range(opt.epochs):

        loss_train = 0.
        correct_train = 0.
        total_train = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # 前向传播，需要手写，即网络架构
            inputs, labels = data
            outputs = net(inputs)

            # 反向传播，交给框架
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新优化器，每个iter更新一次
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).squeeze().sum().numpy()

            acc_train = correct_train / total_train  # 每个iter的acc
            loss_train += loss.item()  # 每个iter的loss

            # 把每个iter的acc和loss存到epoch列表里，1个epoch里有200多个iter
            train_acc_iter.append(acc_train)
            train_loss_iter.append(loss_train)

            loss_train = 0.

        # 用于打印每个epoch信息
        print("Train：Epoch[{:0>3}/{:0>3}]  Loss：{:.4f}  Acc：{:.2%}  Ratio：[{:.0f}/{:.0f}]".format(
            epoch, opt.epochs, np.mean(train_loss_iter), np.mean(train_acc_iter), correct_train, total_train))

        # 每个epoch的结果
        train_loss_epoch.append(np.mean(train_loss_iter))
        train_acc_epoch.append(np.mean(train_acc_iter))

        # 清空每个iter的结果
        train_acc_iter.clear()
        train_loss_iter.clear()

        # 更新学习率，每个epoch更新一次
        scheduler.step()

        # 验证模型
        loss_test = 0.
        correct_test = 0.
        total_test = 0.

        net.eval()
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).squeeze().sum().numpy()

                acc_test = correct_test / total_test  # 每个iter的acc
                loss_test += loss.item()  # 每个iter的loss

                # 把每个iter的acc和loss存到列表里
                test_acc_iter.append(acc_test)
                test_loss_iter.append(loss_test)

                loss_test = 0.

            print("Valid：Epoch[{:0>3}/{:0>3}]  Loss：{:.4f}  Acc：{:.2%}  Ratio：[{:.0f}/{:.0f}]".format(
                epoch, opt.epochs, np.mean(test_loss_iter), np.mean(test_acc_iter), correct_test, total_test))

            print('=' * 60)

            # 每个epoch的结果
            test_loss_epoch.append(np.mean(test_loss_iter))
            test_acc_epoch.append(np.mean(test_acc_iter))

            # 清空每个iter的结果
            test_acc_iter.clear()
            test_loss_iter.clear()

    # ===================================== 可选步骤 1/3 保存模型 =====================================
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')

        # 1.保存性能最好的模型
        # 先保存第一个epoch的模型和优化器
        if epoch == 0:
            if opt.save_model:
                best_loss_test = test_loss_epoch[0]
                best_acc_test = test_acc_epoch[0]
                snapBest = 'BestEpoch_%d_TestLoss_%.4f_TestAcc_%.2f' % (epoch, best_loss_test, best_acc_test)

                torch.save(net.state_dict(),
                           os.path.join(opt.path_state_dict,
                                        opt.dataset + '_' + opt.network + '_Model_' + snapBest + '.pth'))  # 保存网络中的参数

                torch.save(optimizer.state_dict(),
                           os.path.join(opt.path_state_dict,
                                        opt.dataset + '_' + opt.network + '_Optimizer_' + snapBest + '.pth'))  # 保存优化器中的参数

        # 然后判断模型的TestAcc是否比第一个epoch的好，
        # 是则删除掉前1个epoch保存的模型和优化器，并保存当前epoch的模型和优化器，不是则跳过
        else:
            if opt.save_model:
                if test_acc_epoch[epoch] > best_acc_test:
                    best_acc_test = test_acc_epoch[epoch]

                    # 删掉多余的保存文件
                    if count > 0:
                        try:
                            os.remove(os.path.join(opt.path_state_dict,
                                                   opt.dataset + '_' + opt.network + '_Model_' + snapBest + '.pth'))
                            os.remove(os.path.join(opt.path_state_dict,
                                                   opt.dataset + '_' + opt.network + '_Optimizer_' + snapBest + '.pth'))

                        except OSError:
                            pass

                    snapBest = 'BestEpoch_%d_TestLoss_%.4f_TestAcc_%.2f' % (epoch, best_loss_test, best_acc_test)

                    torch.save(net.state_dict(),
                               os.path.join(opt.path_state_dict,
                                            opt.dataset + '_' + opt.network + '_Model_' + snapBest + '.pth'))  # 保存网络中的参数

                    torch.save(optimizer.state_dict(),
                               os.path.join(opt.path_state_dict,
                                            opt.dataset + '_' + opt.network + '_Optimizer_' + snapBest + '.pth'))  # 保存优化器中的参数

        # 用于计数
        count += 1

        # 2.保存训练结束时的模型和优化器
        if epoch == opt.epochs - 1:
            if opt.save_model:
                snapLast = 'LastEpoch_%d_TestLoss_%.4f_TestAcc_%.2f' % (epoch, test_loss_epoch[-1], test_acc_epoch[-1])

                torch.save(net.state_dict(),
                           os.path.join(opt.path_state_dict,
                                        opt.dataset + '_' + opt.network + '_Model_' + snapLast + '.pth'))  # 保存网络中的参数

                torch.save(optimizer.state_dict(),
                           os.path.join(opt.path_state_dict,
                                        opt.dataset + '_' + opt.network + '_Optimizer_' + snapLast + '.pth'))  # 保存优化器中的参数

    if opt.save_model:
        print('Model saved!')

    print('Training finished!')

    # ===================================== 可选步骤 2/3 保存指标 =====================================
    # 先判断文件夹存不存在，np.save找不到路径会报错
    if not os.path.exists('./metrics'):
        os.mkdir('./metrics')

    if opt.save_metrics:
        # 保存loss
        np.save('./metrics/loss_train.npy', np.asarray(train_loss_epoch))
        np.save('./metrics/loss_test.npy', np.asarray(test_loss_epoch))

        # 保存acc
        np.save('./metrics/accuracy_train.npy', np.asarray(train_acc_epoch))
        np.save('./metrics/accuracy_test.npy', np.asarray(test_acc_epoch))

        print('Metrics saved!')

    # ===================================== 可选步骤 3/3 画图 =====================================
    # loss
    if opt.plot:
        train_loss_x = range(len(train_loss_epoch))
        train_loss_y = train_loss_epoch
        train_acc_x = range(len(train_acc_epoch))
        train_acc_y = train_acc_epoch

        # acc
        valid_loss_x = range(len(test_loss_epoch))
        valid_loss_y = test_loss_epoch
        valid_acc_x = range(len(test_acc_epoch))
        valid_acc_y = test_acc_epoch

        plt.subplot(1, 2, 1)  # 第一个对象
        plt.plot(train_loss_x, train_loss_y, label='Train_Loss')
        plt.plot(valid_loss_x, valid_loss_y, label='Valid_Loss')
        plt.title(opt.network + '_Loss')  # 标题
        plt.xlabel('Epoch')  # x轴
        plt.ylabel('Loss')  # y轴
        plt.legend()

        plt.subplot(1, 2, 2)  # 第二个对象
        plt.plot(train_acc_x, train_acc_y, label='Train_Acc')
        plt.plot(valid_acc_x, valid_acc_y, label='Valid_Acc')
        plt.title(opt.network + '_Accuracy')  # 标题
        plt.xlabel('Epoch')  # x轴
        plt.ylabel('Acc')  # y轴
        plt.legend()

        plt.show()


if __name__ == '__main__':
    opt = parse_args()
    train()



