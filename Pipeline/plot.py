import random
import numpy as np
import matplotlib.pyplot as plt


def plot():
    train_loss_x = range(len(loss_train))
    train_loss_y = loss_train
    train_acc_x = range(len(accuracy_train))
    train_acc_y = accuracy_train

    # acc
    valid_loss_x = range(len(loss_test))
    valid_loss_y = loss_test
    valid_acc_x = range(len(accuracy_test))
    valid_acc_y = accuracy_test

    plt.subplot(1, 2, 1)  # 第一个对象
    plt.plot(train_loss_x, train_loss_y, label='Train_Loss')
    plt.plot(valid_loss_x, valid_loss_y, label='Valid_Loss')
    plt.title('Loss')  # 标题
    plt.xlabel('Epoch')  # x轴
    plt.ylabel('Loss')  # y轴
    plt.legend()

    plt.subplot(1, 2, 2)  # 第二个对象
    plt.plot(train_acc_x, train_acc_y, label='Train_Acc')
    plt.plot(valid_acc_x, valid_acc_y, label='Valid_Acc')
    plt.title('Accuracy')  # 标题
    plt.xlabel('Epoch')  # x轴
    plt.ylabel('Acc')  # y轴
    plt.legend()

    plt.show()


if __name__ == '__main__':
    accuracy_train = np.load('./metrics/accuracy_train.npy')
    accuracy_test = np.load('./metrics/accuracy_test.npy')

    loss_train = np.load('./metrics/loss_train.npy')
    loss_test = np.load('./metrics/loss_test.npy')

    plot()